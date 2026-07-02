"""Structure dataclass + readers / writers for XYZ / PDB / PySCF / ASE.

The :class:`Structure` is the lingua franca between builders (peptide,
nucleic) and consumers (file writers, downstream analysis).  Every
builder returns one of these; every output format is just a method on
it.  Adding a new format means adding one method here, not touching the
builders.

Loading external geometry into the package goes through the inverse
``from_xyz`` / ``from_pdb`` classmethods (or the top-level
``molbuilder.load`` convenience function), which means an XYZ or PDB
exported by a different tool can be fed straight into the SIESTA
pipeline without re-building it from scratch.

Transport-relevant attributes (see the three-stage contract in
docs/design.md):

  frozen_atoms : List[int]
      0-based indices of atoms whose coordinates the geometry
      optimiser must NOT move.  Loaded from a structure sidecar
      JSON by /modify; consumed by Spectra (cfg.frozen_indices)
      and by the Build SIESTA / PySCF emitters (warn-only today
      pending the design.md "fully respected" rollout).

  regions : Dict[str, List[int]]
      Named groups of atom indices for transport-style partition
      (keys are user-facing labels like ``"L-electrode"`` /
      ``"R-electrode"`` / ``"bridge"``; values are 0-based atom
      indices).  Validated as pairwise-disjoint at __post_init__.
      (Docstring used to say List[List[int]]; corrected 2026-06-09
      after siesta/input.py rebuilt regions as a list comprehension,
      crashed on any non-empty dict — fixed in commit 7d9cd54.)

These two attributes are the load-bearing carriers for the
boundary-conditions axis of the three-stage contract.  Any emitter
that drops them silently (rather than warning) violates the
contract.  ``Structure.copy()`` / ``.translated()`` MUST carry
them through (see the methods + their tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# --------------------------------------------------------------------- #
#  Source-resolver helper                                               #
# --------------------------------------------------------------------- #


def _resolve_source(source: Union[str, Path]) -> str:
    """Return the textual content for a file path or for raw text.

    Accepts:
      * a :class:`pathlib.Path` -> always read from disk.
      * a string that names an existing file -> read from disk.
      * any other string (multi-line content, or a non-existent name)
        -> treated as the file content directly.

    The "treat as text" fallback is what lets callers pass a string
    they pulled out of an HTTP request body or a blob storage object.
    """
    if isinstance(source, Path):
        # ``utf-8-sig`` accepts an optional UTF-8 BOM (some Windows
        # editors emit one) — without an explicit encoding Python
        # falls back to the platform locale (cp1252 / latin-1 on
        # some Windows / older Linux installs), which mojibakes any
        # non-ASCII in the XYZ comment line or PDB residue names.
        # Same hardening molstruct_json / spectra_json / transport_json
        # carry.
        return source.read_text(encoding="utf-8-sig")
    if isinstance(source, str):
        # A real file path won't contain newlines and will exist on disk.
        # Anything else is text.  We deliberately don't accept paths
        # with newlines -- ambiguous and not a real filesystem path.
        if "\n" not in source and os.path.isfile(source):
            with open(source, "r", encoding="utf-8-sig") as fh:
                return fh.read()
        return source
    raise TypeError(
        f"source must be str or Path, got {type(source).__name__}"
    )


# ---------------------------------------------------------------------- #
#  Atomic-mass table (used only when callers ask for an ASE Atoms        #
#  object; ase has its own tables but we don't want a hard dep here).   #
# ---------------------------------------------------------------------- #


# ---------------------------------------------------------------------- #
#  Per-atom annotation channels (atom-annotations.md)                    #
# ---------------------------------------------------------------------- #

_CHANNEL_KINDS = ("tag", "flag", "value")


@dataclass
class AtomChannel:
    """One named per-atom metadata channel (atom-annotations.md § 2).

    ``kind``:
      * ``"tag"``  / ``"flag"`` — a *subset* of atoms; ``data`` is a
        sorted ``List[int]`` of member indices.  (Both share this shape;
        ``tag`` = a named region-like set, ``flag`` = a boolean property.)
      * ``"value"`` — a per-atom scalar; ``data`` is ``Dict[int, Any]``
        mapping atom index -> value (sparse; absent atoms have no value).

    ``color`` / ``fdf`` are optional hints: a presentation color and the
    id of the fdf emit-strategy this channel maps to (a channel with no
    strategy is carried but not emitted -- § 4).
    """
    kind: str
    data: Any = None
    color: Optional[str] = None
    fdf: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _CHANNEL_KINDS:
            raise ValueError(
                f"AtomChannel.kind must be one of {_CHANNEL_KINDS}; "
                f"got {self.kind!r}")
        if self.data is None:
            self.data = {} if self.kind == "value" else []

    def remapped(self, old_to_new: "dict[int, int]") -> "AtomChannel":
        """Return a copy with atom indices translated through
        ``old_to_new`` (survivors only) -- for structure edits (§ 2.1)."""
        if self.kind == "value":
            data: Any = {old_to_new[i]: v for i, v in self.data.items()
                         if i in old_to_new}
        else:
            data = sorted(old_to_new[i] for i in self.data if i in old_to_new)
        return AtomChannel(self.kind, data, self.color, self.fdf)

    def copy(self) -> "AtomChannel":
        data = dict(self.data) if self.kind == "value" else list(self.data)
        return AtomChannel(self.kind, data, self.color, self.fdf)


def copy_annotations(ann: "dict[str, AtomChannel]") -> "dict[str, AtomChannel]":
    """Deep-copy an annotations map (channels carried verbatim, § 2.1)."""
    return {name: ch.copy() for name, ch in ann.items()}


def remap_annotations(ann: "dict[str, AtomChannel]",
                      old_to_new: "dict[int, int]") -> "dict[str, AtomChannel]":
    """Remap every channel's atom indices through ``old_to_new`` (the
    all-channel generalization of ``modify.remap_frozen_and_regions``,
    atom-annotations.md § 2.1).  Channels that end up empty are dropped."""
    out: "dict[str, AtomChannel]" = {}
    for name, ch in ann.items():
        remapped = ch.remapped(old_to_new)
        if remapped.data:                      # drop channels emptied by the edit
            out[name] = remapped
    return out


@dataclass
class Structure:
    """All-atom 3D structure of a (poly)molecule.

    The arrays are 1:1 by atom index:
        elements[i]    chemical symbol (e.g. "C", "N", "P", "Au")
        positions[i]   xyz in Angstrom
        atom_names[i]  PDB-style atom name ("CA", "N1", "OP1", ...)
        residue_ids[i] residue number this atom belongs to (1-based)
        residue_names[i]   3-letter residue name ("ALA", "DA",  "SEP", ...)
        chain_ids[i]   single-character chain id ("A" by default)

    None of the per-atom optional fields are required to write XYZ --
    they only matter for PDB (which uses them) and the various viewers
    / loaders that consume PDB.

    Two transport-oriented attributes (added 2026-05-20) carry
    information about which atoms are which in a molecular junction:

        regions       atom-index lists keyed by region label
                      (e.g. ``{"L-electrode": [0..11],
                              "R-electrode": [30..41],
                              "bridge":      [12..29]}``).
                      Region membership is NOT mutually exclusive --
                      an atom may carry multiple labels at once
                      (e.g. ``"L-electrode"`` + ``"interface"``).
                      Engines that need a disjoint partition (e.g.
                      TranSIESTA 2-terminal) enforce that as a
                      separate preflight at engine-load time.
                      Empty by default; populated by the modify-tab
                      "Mark region" workflow + by builders that
                      assemble junctions with explicit electrode
                      regions.

        frozen_atoms  0-based indices of atoms whose positions stay
                      fixed during downstream relaxations and
                      Hessian builds.  ("Frozen" is molbuilder's
                      canonical term; some QC contexts call this
                      "fixed atoms".  The two names are
                      synonymous; we standardise on "frozen" to
                      match the spectroscopy literature.)  Carried
                      through from build / modify time; consumed
                      by SpectraConfig (relax + Hessian) and
                      TransportConfig (NEGF lead-fixing).  Sorted +
                      deduped on validation.  Empty default.

    Both fields are pure metadata -- nothing in this module reads
    them.  Downstream consumers (spectra, transport) decide what to
    do with them.
    """

    elements: List[str]
    positions: np.ndarray                  # (N, 3), Angstrom
    atom_names:    Optional[List[str]] = None
    residue_ids:   Optional[List[int]] = None
    residue_names: Optional[List[str]] = None
    chain_ids:     Optional[List[str]] = None
    title:         str = ""
    # Transport-oriented metadata (2026-05-20).  Defaults keep every
    # existing call site working without change.
    regions:       Dict[str, List[int]] = field(default_factory=dict)
    frozen_atoms:  List[int]            = field(default_factory=list)
    # Periodic lattice (2026-06-27).  ``cell`` is the (3, 3) matrix whose
    # ROWS are the lattice vectors in Angstrom (ASE convention), or None
    # for a non-periodic molecule.  ``pbc`` is per-axis periodicity:
    # True = periodic (the structure tiles, no vacuum), False = vacuum
    # along that axis.  This is the SOURCE OF TRUTH for the cell — the
    # transport/SIESTA emitters preserve it verbatim instead of
    # fabricating an orthorhombic vacuum box from atom extents.  Both
    # default to "no lattice" so every existing call site is unchanged.
    cell:          Optional[np.ndarray]            = None
    pbc:           Optional[Tuple[bool, bool, bool]] = None
    # Extensible per-atom annotations (atom-annotations.md).  Holds
    # channels BEYOND the two built-ins (regions -> tag channels,
    # frozen_atoms -> the "frozen" flag channel), e.g. future per-atom
    # value channels (charge / spin / basis-override).  The unified read
    # API is ``channels()`` / ``get_channel()`` / ``atom_annotations()``,
    # which present regions + frozen + these together.  Empty default so
    # every existing call site is unchanged.
    annotations:   Dict[str, AtomChannel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        n = len(self.positions)
        if len(self.elements) != n:
            raise ValueError(
                f"elements ({len(self.elements)}) does not match positions ({n})"
            )
        # Default-fill optional metadata so PDB writer never has to special-case
        if self.atom_names    is None: self.atom_names    = list(self.elements)
        if self.residue_ids   is None: self.residue_ids   = [1] * n
        if self.residue_names is None: self.residue_names = ["MOL"] * n
        if self.chain_ids     is None: self.chain_ids     = ["A"] * n
        for name, arr in (
            ("atom_names",    self.atom_names),
            ("residue_ids",   self.residue_ids),
            ("residue_names", self.residue_names),
            ("chain_ids",     self.chain_ids),
        ):
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")

        # Normalise the periodic lattice.  A provided cell must be a
        # 3x3 of finite floats; pbc defaults to "fully periodic" when a
        # cell is present (a lattice implies periodicity) and "no
        # periodicity" when it is absent.
        if self.cell is not None:
            cell = np.asarray(self.cell, dtype=float)
            if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
                raise ValueError(
                    f"Structure.cell must be a 3x3 matrix of finite "
                    f"floats (lattice vectors as rows, Angstrom); got "
                    f"shape {cell.shape}"
                )
            # Reject a singular/degenerate lattice (zero volume, or two
            # parallel/duplicated vectors): it would blow up later in
            # reciprocal-space / k-grid math (1/det, inv(cell)) with an
            # opaque LinAlgError instead of a clear message here.
            if abs(float(np.linalg.det(cell))) < 1e-8:
                raise ValueError(
                    "Structure.cell is singular/degenerate (near-zero "
                    "volume); the three lattice vectors must be linearly "
                    "independent."
                )
            self.cell = cell
        if self.pbc is None:
            self.pbc = ((True, True, True) if self.cell is not None
                        else (False, False, False))
        else:
            pbc = tuple(bool(b) for b in self.pbc)
            if len(pbc) != 3:
                raise ValueError(
                    f"Structure.pbc must have exactly 3 entries "
                    f"(one per axis); got {len(pbc)}"
                )
            self.pbc = pbc

        # Validate transport metadata.  Both fields default to empty,
        # so a caller that doesn't care about regions / frozen atoms
        # sees no behaviour change.
        self._validate_regions(n)
        self._validate_frozen_atoms(n)
        self._validate_annotations(n)

    def _validate_regions(self, n: int) -> None:
        """Per-atom index in [0, n); region names are non-empty
        strings.  Indices within each region are sorted + deduped
        in place for stable equality + serialisation.

        Region MEMBERSHIP is NOT mutually exclusive: an atom may
        appear in multiple regions (e.g. ``"L-electrode"`` +
        ``"interface"``).  This is freeform user labelling.

        Engines that need disjoint regions for physics reasons
        (e.g. TranSIESTA 2-terminal: L-electrode / R-electrode /
        bridge must partition the device atoms) enforce that as a
        separate preflight check at engine-load time -- the data
        model itself doesn't constrain it.
        """
        if not self.regions:
            return
        normalised: Dict[str, List[int]] = {}
        for region_name, idxs in self.regions.items():
            if not isinstance(region_name, str) or not region_name:
                raise ValueError(
                    f"Structure.regions: region label must be a "
                    f"non-empty string; got {region_name!r}"
                )
            unique: set = set()
            for raw in idxs:
                idx = int(raw)
                if not 0 <= idx < n:
                    raise ValueError(
                        f"Structure.regions[{region_name!r}]: atom "
                        f"index {idx} out of range [0, {n})"
                    )
                unique.add(idx)
            normalised[region_name] = sorted(unique)
        self.regions = normalised

    def _validate_frozen_atoms(self, n: int) -> None:
        """0-based indices in [0, n); sorted + deduped in place."""
        if not self.frozen_atoms:
            return
        unique: set = set()
        for raw in self.frozen_atoms:
            idx = int(raw)
            if not 0 <= idx < n:
                raise ValueError(
                    f"Structure.frozen_atoms: atom index {idx} out of "
                    f"range [0, {n})"
                )
            unique.add(idx)
        self.frozen_atoms = sorted(unique)

    def _validate_annotations(self, n: int) -> None:
        """Extra channels: names must not collide with a built-in
        (a region label or ``"frozen"``); atom indices must be in
        [0, n).  Normalises tag/flag data to sorted-unique in place."""
        if not self.annotations:
            return
        reserved = set(self.regions) | {"frozen"}
        for name, ch in self.annotations.items():
            if name in reserved:
                raise ValueError(
                    f"Structure.annotations[{name!r}] collides with a "
                    f"built-in channel (a region label or 'frozen'); "
                    f"edit .regions / .frozen_atoms instead.")
            if not isinstance(ch, AtomChannel):
                raise ValueError(
                    f"Structure.annotations[{name!r}] must be an "
                    f"AtomChannel; got {type(ch).__name__}")
            idxs = ch.data.keys() if ch.kind == "value" else ch.data
            for idx in idxs:
                if not 0 <= int(idx) < n:
                    raise ValueError(
                        f"Structure.annotations[{name!r}]: atom index "
                        f"{idx} out of range [0, {n})")
            if ch.kind != "value":
                ch.data = sorted({int(i) for i in ch.data})

    # ------------------------------------------------------------------ #
    #  Unified annotation channels (atom-annotations.md § 2)             #
    # ------------------------------------------------------------------ #

    def channels(self) -> Dict[str, AtomChannel]:
        """The unified per-atom channel registry: each region label as a
        ``tag`` channel, ``frozen`` as a ``flag`` channel (when non-empty),
        plus every extensible channel in ``self.annotations``.  This is
        the one place to read ALL per-atom metadata uniformly."""
        out: Dict[str, AtomChannel] = {}
        for label, idxs in self.regions.items():
            out[label] = AtomChannel("tag", list(idxs))
        if self.frozen_atoms:
            out["frozen"] = AtomChannel("flag", list(self.frozen_atoms))
        for name, ch in self.annotations.items():
            out[name] = ch
        return out

    def get_channel(self, name: str) -> Optional[AtomChannel]:
        """One channel by name (built-in or extensible), or ``None``."""
        return self.channels().get(name)

    def atom_annotations(self, index: int) -> Dict[str, Any]:
        """Everything on atom ``index``: ``{channel_name: value}`` where a
        tag/flag contributes ``True`` and a value channel its scalar.
        The per-atom view the selection filter / UI reads."""
        out: Dict[str, Any] = {}
        for name, ch in self.channels().items():
            if ch.kind == "value":
                if index in ch.data:
                    out[name] = ch.data[index]
            elif index in ch.data:
                out[name] = True
        return out

    def set_channel(self, name: str, channel: AtomChannel) -> None:
        """Set an EXTENSIBLE channel (stored in ``annotations``).  Reject
        built-in names -- edit ``.regions`` / ``.frozen_atoms`` for those.
        Re-validates against the current atom count."""
        if name in self.regions or name == "frozen":
            raise ValueError(
                f"{name!r} is a built-in channel; edit .regions / "
                f".frozen_atoms instead.")
        self.annotations[name] = channel
        self._validate_annotations(len(self.positions))

    # ------------------------------------------------------------------ #
    #  Convenience accessors                                              #
    # ------------------------------------------------------------------ #

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def n_residues(self) -> int:
        return len(set(self.residue_ids)) if self.residue_ids else 0

    def summary(self) -> str:
        from collections import Counter
        formula = Counter(self.elements)
        formula_str = "".join(
            f"{el}{n}" if n > 1 else el
            for el, n in sorted(formula.items())
        )
        return (
            f"<Structure {self.title!r}: "
            f"{self.n_atoms} atoms, {self.n_residues} residues, "
            f"formula {formula_str}>"
        )

    def __repr__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------ #
    #  Input: XYZ                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_xyz(cls, source: Union[str, Path], *,
                 title: Optional[str] = None) -> "Structure":
        """Load a Structure from an XYZ file path or XYZ text content.

        Standard xmol layout is expected::

            N
            <comment / title>
            <El>  x  y  z
            ...   (N atom lines)

        Extra trailing whitespace and blank lines after the N atoms
        are ignored.  XYZ stores no atom names / residues, so all
        atoms come back tagged as residue 1 ("MOL", chain "A") and
        atom names default to the element symbol.
        """
        text = _resolve_source(source)
        lines = text.splitlines()
        if len(lines) < 2:
            raise ValueError("XYZ too short: need header + comment + atoms")
        try:
            n = int(lines[0].strip())
        except ValueError as e:
            raise ValueError(
                f"first line of XYZ must be an integer atom count; got "
                f"{lines[0]!r}"
            ) from e
        if n < 0:
            raise ValueError(f"negative atom count in XYZ: {n}")
        elements: List[str] = []
        positions: List[List[float]] = []
        for raw in lines[2:2 + n]:
            parts = raw.split()
            if len(parts) < 4:
                raise ValueError(
                    f"malformed XYZ atom line (need 'El x y z'): {raw!r}"
                )
            # Same case-canonicalisation as from_pdb: an XYZ produced
            # by an external tool might emit ``FE`` / ``ZN``; downstream
            # consumers (siesta/input._detect_species, ase.data) key on
            # the ``Fe``/``Zn`` form.
            elements.append(parts[0].capitalize())
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(elements) != n:
            raise ValueError(
                f"XYZ header says {n} atoms but only {len(elements)} found"
            )
        comment = lines[1].strip() if len(lines) >= 2 else ""
        return cls(
            elements=elements,
            positions=np.asarray(positions, dtype=float),
            title=(title if title is not None else comment),
        )

    # ------------------------------------------------------------------ #
    #  Input: PDB                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pdb(cls, source: Union[str, Path], *,
                 title: Optional[str] = None) -> "Structure":
        """Load a Structure from a PDB file path or PDB text content.

        Reads ATOM and HETATM records.  Other record types (HEADER,
        REMARK, CONECT, etc.) are ignored.  Multi-MODEL files: only
        the first MODEL is read; this is what most viewers show by
        default for a relaxation trajectory.

        TER records are honoured as polymer-chain boundaries.  Two
        common situations a naive parser gets wrong:

          * a homemade PDB exporter that omits the chain-id column
            entirely (col 22 blank) and relies on TER alone to mark
            chain boundaries;
          * a file that reuses the same chain-id letter across TERs
            (e.g. all 'A') for what are logically separate polymers.

        We track a segment counter (incremented on each TER) and tag
        every atom with `(chain_letter, segment)`.  After the parse:
          - a chain letter unique to one segment passes through as-is,
            preserving back-compat for well-formed PDBs;
          - a chain letter spanning multiple segments is disambiguated
            by appending the segment index, so the resulting chain ids
            are unique;
          - a blank chain-id column ('_' internally) becomes 'A' when
            unambiguous, '_<n>' when it spans multiple TER segments.
        """
        text = _resolve_source(source)

        elements: List[str] = []
        positions: List[List[float]] = []
        atom_names: List[str] = []
        residue_ids: List[int] = []
        residue_names: List[str] = []
        raw_chain_letters: List[str] = []
        atom_segments: List[int] = []

        seen_model = False
        pdb_title = ""
        segment_index = 0
        # Track (chain, residue_id, atom_name) keys we've already
        # emitted so altLoc dedup keeps the FIRST conformation and
        # skips subsequent alternates (see the altLoc comment in
        # the ATOM/HETATM branch below).
        _seen_altloc_keys: set = set()

        for line in text.splitlines():
            rec = line[:6]
            if rec.startswith("TITLE"):
                # PDB TITLE records use cols 11-80 for the actual title
                pdb_title += line[10:].strip() + " "
                continue
            if rec.startswith("MODEL"):
                if seen_model:
                    break          # only first MODEL block
                seen_model = True
                continue
            if rec.startswith("ENDMDL"):
                break
            if rec.startswith("TER"):
                # Polymer-chain boundary: bump segment so reused chain
                # letters across TERs end up in distinct logical chains.
                # Multiple consecutive TERs are harmless -- each just
                # bumps the counter without affecting an empty segment.
                segment_index += 1
                continue
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            # altLoc (column 17, 1-indexed = line[16]): indicates one
            # of several crystallographically-resolved conformations
            # for the same atom.  PySCF / SIESTA expect a single
            # well-defined geometry; loading EVERY conformation puts
            # near-coincident atoms in the same Mol, which makes
            # forces explode (1/r² Coulomb at sub-Å distances).
            #
            # Standard practice (ASE, MDAnalysis, PyMOL "PyMOL default
            # group state"): keep the FIRST conformation per
            # (chain, residue_id, atom_name).  We treat blank altLoc
            # as the default ('A' is fine too); only filter when an
            # alternate (B, C, ...) duplicates a key we've already
            # seen.
            altloc = line[16:17] if len(line) > 16 else " "
            res_name  = line[17:20].strip() or "MOL"
            # '_' is our internal placeholder for "chain-id column was
            # blank in the file"; it never appears in well-formed PDBs.
            chain_letter = line[21:22].strip() or "_"
            try:
                res_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            altloc_key = (chain_letter, res_id, atom_name)
            if altloc.strip():
                # Has an altLoc indicator -- only accept if we
                # haven't seen this (chain, residue, atom_name) yet.
                if altloc_key in _seen_altloc_keys:
                    continue
                _seen_altloc_keys.add(altloc_key)
            else:
                # Blank altLoc: a "single conformation" line.  Record
                # the key so a LATER alternate doesn't sneak in
                # (unusual file order but possible).
                _seen_altloc_keys.add(altloc_key)
            # PDB cols 77-78 hold the element symbol right-justified but
            # with NO case convention -- many PDBs (incl. PDB Bank's
            # canonical) emit ``FE``/``CL``/``NA`` upper-cased.
            # Canonicalise to upper-then-lower (``Fe``/``Cl``/``Na``)
            # so downstream consumers (siesta/input._detect_species,
            # ase.data.atomic_numbers, the chemistry hint table) all see
            # the form their tables key on.  Without this the SIESTA
            # emitter crashes with ``KeyError: 'FE'`` for any PDB with
            # a transition metal (caught 2026-05-25 against the
            # hemeC-dithiol structure).
            element = line[76:78].strip().capitalize()
            if not element:
                # Element column 77-78 empty -- fall back to PDB-format
                # column rules + a known-symbols check:
                #
                #  * cols 13-14 hold the element symbol when the first
                #    character is non-blank (typical for two-letter
                #    elements like Zn, Fe, Cl);
                #  * cols 14-15 hold the element when col 13 is blank
                #    (one-letter elements like C, N, O on protein
                #    backbones; "CA" = alpha carbon, not calcium).
                #
                # We try the two-letter form first against the known
                # symbol set (ase.data.atomic_numbers); fall through to
                # the single-letter form if not matched.  This fixes
                # the previous bug where "FE", "ZN", "MG", "NA", ...
                # silently degraded to "F", "Z", "M", "N", which are
                # the wrong elements (or invalid symbols).
                try:
                    from ase.data import atomic_numbers as _ase_atomic_numbers
                    _known = _ase_atomic_numbers
                except Exception:
                    _known = None
                raw = (line[12:14] if len(line) >= 14 else "").strip()
                cand2 = raw[:2].capitalize() if len(raw) >= 2 else ""
                cand1 = raw[:1].upper() if raw else ""
                if cand2 and _known and cand2 in _known:
                    element = cand2
                elif cand1:
                    element = cand1
                else:
                    # Last resort: leading alphabetic chars of atom_name.
                    element = "".join(c for c in atom_name if c.isalpha())[:1].upper()
            elements.append(element)
            positions.append([x, y, z])
            atom_names.append(atom_name)
            residue_ids.append(res_id)
            residue_names.append(res_name)
            raw_chain_letters.append(chain_letter)
            atom_segments.append(segment_index)

        if not elements:
            raise ValueError("no ATOM/HETATM records found in PDB input")

        # Disambiguation pass.  A chain letter that appears in only one
        # segment passes through unchanged (preserves back-compat with
        # well-formed PDBs); a letter that spans multiple segments has
        # the segment index appended so the resulting ids are unique.
        # Empty chain-id columns ('_' placeholder) map to 'A' in the
        # unambiguous case (matches the previous parser's behaviour).
        letter_segments: dict = {}
        for letter, seg in zip(raw_chain_letters, atom_segments):
            letter_segments.setdefault(letter, set()).add(seg)
        needs_disambig = {l for l, segs in letter_segments.items()
                          if len(segs) > 1}

        chain_ids: List[str] = []
        for letter, seg in zip(raw_chain_letters, atom_segments):
            if letter in needs_disambig:
                # e.g. "A0", "A1", or "_0", "_1" for blank columns
                chain_ids.append(f"{letter}{seg}")
            else:
                chain_ids.append("A" if letter == "_" else letter)

        return cls(
            elements=elements,
            positions=np.asarray(positions, dtype=float),
            atom_names=atom_names,
            residue_ids=residue_ids,
            residue_names=residue_names,
            chain_ids=chain_ids,
            title=(title if title is not None else pdb_title.strip()),
        )

    # ------------------------------------------------------------------ #
    #  Output: XYZ                                                        #
    # ------------------------------------------------------------------ #

    def to_xyz(self, path: Optional[str] = None, *, comment: str = "") -> str:
        """Return XMol .xyz text; if *path* is given, also write to it.

        The result drops directly into a SIESTA
        ``%block AtomicCoordinatesAndAtomicSpecies`` once you map symbols
        to species indices, or into any other code that reads .xyz.
        """
        buf = StringIO()
        buf.write(f"{self.n_atoms}\n")
        buf.write((comment or self.title or "Built by molbuilder").strip() + "\n")
        for el, (x, y, z) in zip(self.elements, self.positions):
            buf.write(f"{el:<3s} {x: 12.6f} {y: 12.6f} {z: 12.6f}\n")
        text = buf.getvalue()
        if path:
            # ``encoding="utf-8"`` is REQUIRED: without it Python falls
            # back to the platform locale, which silently corrupts non-
            # ASCII residue names / title comments on cp1252 / latin-1
            # systems (and disagrees with the encoding-utf-8-sig read
            # path in ``_resolve_source``).
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    # ------------------------------------------------------------------ #
    #  Output: PDB                                                        #
    # ------------------------------------------------------------------ #

    def to_pdb(self, path: Optional[str] = None) -> str:
        """Standard PDB. Hydrogens included, single MODEL, no CONECT."""
        buf = StringIO()
        if self.title:
            buf.write(f"TITLE     {self.title:<70s}\n")
        for i in range(self.n_atoms):
            el   = self.elements[i]
            name = self.atom_names[i]
            res  = self.residue_names[i]
            chn  = (self.chain_ids[i] or "A")[:1]    # PDB chain id is 1 char
            rid  = self.residue_ids[i]
            x, y, z = self.positions[i]
            # PDB ATOM record: cols are fixed-width.  Atom-name field has
            # the quirk that 1- and 2-letter element symbols start in
            # column 14, while 3-4-letter names start in column 13.
            atname = name if len(name) >= 4 else f" {name:<3s}"
            # PDB serial column is 5 chars (cols 7-11).  Per spec, beyond
            # 99999 we wrap to "*****" rather than overflow the field.
            serial = i + 1
            serial_str = f"{serial:5d}" if serial <= 99999 else "*****"
            # Residue id is 4 chars (cols 23-26) -- same wrap rule.
            rid_str = f"{rid:4d}" if rid <= 9999 else "****"
            buf.write(
                f"ATOM  {serial_str} {atname:<4s} {res:>3s} {chn}{rid_str}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n"
            )
        buf.write("END\n")
        text = buf.getvalue()
        if path:
            # ``encoding="utf-8"`` parity with ``to_xyz`` + the
            # encoding-utf-8-sig read path in ``_resolve_source``.
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    # ------------------------------------------------------------------ #
    #  Output: PySCF                                                      #
    # ------------------------------------------------------------------ #

    def to_pyscf(self, *, as_string: bool = False
                 ) -> Union[List[Sequence], str]:
        """Return the molecule in the form ``pyscf.gto.M`` accepts.

        Default is a list of ``(symbol, (x, y, z))`` tuples, which you
        can drop straight into::

            mol = pyscf.gto.M(atom=struct.to_pyscf(), basis="6-31g*")

        Pass ``as_string=True`` to get a multiline string instead, in
        the format PySCF also accepts (one atom per line:
        ``"C  0.0  0.0  0.0"``).
        """
        if as_string:
            return "\n".join(
                f"{el} {x: .8f} {y: .8f} {z: .8f}"
                for el, (x, y, z) in zip(self.elements, self.positions)
            )
        return [
            (el, (float(x), float(y), float(z)))
            for el, (x, y, z) in zip(self.elements, self.positions)
        ]

    # ------------------------------------------------------------------ #
    #  Output: ASE                                                        #
    # ------------------------------------------------------------------ #

    def to_ase(self):
        """Return an :class:`ase.Atoms` instance.

        Raises ImportError if ASE isn't installed -- this is the only
        method with an optional dep.
        """
        try:
            from ase import Atoms
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_ase() needs the 'ase' package; install with "
                "`pip install ase`"
            ) from exc
        return Atoms(symbols=self.elements, positions=self.positions)

    # ------------------------------------------------------------------ #
    #  Combine / translate / center -- handy small utilities              #
    # ------------------------------------------------------------------ #

    def copy(self) -> "Structure":
        """Return a deep-ish copy: all metadata lists are duplicated;
        ``positions`` is copied so the new Structure can be mutated
        without affecting the original.  Used by op helpers that
        return the input unchanged (e.g. ``add_electrode_slab`` with
        ``n_layers <= 0`` short-circuits to ``struct.copy()`` rather
        than open-coding the field-by-field rebuild three times).
        """
        return Structure(
            elements      = list(self.elements),
            positions     = self.positions.copy(),
            atom_names    = list(self.atom_names),
            residue_ids   = list(self.residue_ids),
            residue_names = list(self.residue_names),
            chain_ids     = list(self.chain_ids),
            title         = self.title,
            regions       = {k: list(v) for k, v in self.regions.items()},
            frozen_atoms  = list(self.frozen_atoms),
            cell          = (self.cell.copy() if self.cell is not None
                             else None),
            pbc           = self.pbc,
            annotations   = copy_annotations(self.annotations),
        )

    def translated(self, vec: Sequence[float]) -> "Structure":
        v = np.asarray(vec, dtype=float).reshape(3)
        return Structure(
            elements      = list(self.elements),
            positions     = self.positions + v,
            atom_names    = list(self.atom_names),
            residue_ids   = list(self.residue_ids),
            residue_names = list(self.residue_names),
            chain_ids     = list(self.chain_ids),
            title         = self.title,
            regions       = {k: list(v) for k, v in self.regions.items()},
            frozen_atoms  = list(self.frozen_atoms),
            # A rigid translation leaves the lattice unchanged.
            cell          = (self.cell.copy() if self.cell is not None
                             else None),
            pbc           = self.pbc,
            annotations   = copy_annotations(self.annotations),
        )

    def centered(self) -> "Structure":
        """Translate so the **atom-coordinate mean** lands at the
        world origin.

        Note the choice of centring: this is the unweighted mean of
        atomic positions, NOT the bounding-box centre and NOT the
        centre of mass.  For asymmetric molecules with a long
        substituent (alkyl chain off a benzenedithiol, etc.) the
        atom-mean will shift toward the heavier side.  When you
        need the **anchor-pair midpoint** at the origin (the typical
        transport-junction convention), use
        ``orient_along_axis(struct, anchors, center='midpoint')``
        instead -- it explicitly anchors on a user-chosen atom pair.
        """
        return self.translated(-self.positions.mean(axis=0))

    @classmethod
    def concat(cls, structures: Sequence["Structure"], *,
               renumber_residues: bool = True,
               title: str = "") -> "Structure":
        """Concatenate several structures into one.

        With ``renumber_residues=True`` (default) residue IDs are made
        globally unique by offsetting each structure's IDs to start
        right after the previous one.
        """
        if not structures:
            return cls(elements=[], positions=np.zeros((0, 3)))
        elements: List[str] = []
        atom_names: List[str] = []
        residue_ids: List[int] = []
        residue_names: List[str] = []
        chain_ids: List[str] = []
        positions = []
        # Transport metadata (frozen_atoms + regions) must be re-indexed
        # per-input because each structure's atom indices are 0-based and
        # the concatenation shifts the i-th structure's atoms by the sum
        # of n_atoms across all earlier structures.  Regions with the
        # same label across inputs merge into one combined index list.
        frozen_atoms: List[int] = []
        regions: Dict[str, List[int]] = {}
        atom_offset = 0
        offset = 0
        for s in structures:
            elements.extend(s.elements)
            atom_names.extend(s.atom_names)
            residue_names.extend(s.residue_names)
            chain_ids.extend(s.chain_ids)
            positions.append(s.positions)
            ids = s.residue_ids
            if renumber_residues and ids:
                this_offset = offset - (min(ids) - 1)
                residue_ids.extend(i + this_offset for i in ids)
                offset = max(residue_ids)
            else:
                residue_ids.extend(ids)
            frozen_atoms.extend(i + atom_offset for i in s.frozen_atoms)
            for label, idxs in s.regions.items():
                regions.setdefault(label, []).extend(
                    i + atom_offset for i in idxs
                )
            atom_offset += s.n_atoms
        # Carry the FIRST input's lattice (the conventional base when
        # concatenating, e.g. add_electrode_slab builds onto a base
        # structure).  The caller owns making the cell big enough for
        # the merged atoms — concat can't infer a new lattice.
        base_cell = next((s.cell for s in structures if s.cell is not None),
                         None)
        base_pbc = next((s.pbc for s in structures if s.cell is not None),
                        None)
        return cls(
            elements      = elements,
            positions     = np.vstack(positions),
            atom_names    = atom_names,
            residue_ids   = residue_ids,
            residue_names = residue_names,
            chain_ids     = chain_ids,
            title         = title,
            regions       = regions,
            frozen_atoms  = frozen_atoms,
            cell          = (base_cell.copy() if base_cell is not None
                             else None),
            pbc           = base_pbc,
        )
