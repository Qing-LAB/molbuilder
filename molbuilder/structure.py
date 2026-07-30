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

#: Containment tolerance in FRACTIONAL units (§ 6.1): loose enough to forgive a
#: round-tripped float, tight enough that "half the molecule outside the box"
#: can never pass.  Shared with periodicity_gate, which delegates containment
#: to :meth:`Structure.cell_contains_atoms`.
_CONTAIN_EPS = 1e-6

#: Minimum-thickness floor for a DERIVED box (§ 6.1, decided 2026-07-29).  An
#: isolated axis whose derived length would fall below _MIN_DERIVED_CELL_LENGTH
#: (Angstrom) gets its per-side vacuum raised to _MIN_ISOLATED_VACUUM, so a flat
#: or linear molecule can never yield a zero-thickness box (a zero determinant,
#: which used to surface as "degenerate cell" from the emitter and blocked a
#: reset-to-derived in the gate) and there is always a real gap between periodic
#: images.  STRUCTURAL minimum only -- physical adequacy is a separate, larger
#: number the validator asks for (>= 8 A per side neutral; see
#: validation/siesta.py:_check_siesta_vacuum_adequacy).
_MIN_DERIVED_CELL_LENGTH = 3.0
_MIN_ISOLATED_VACUUM = 3.0


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

    def union(self, other: "AtomChannel") -> "AtomChannel":
        """Merge another channel of the SAME kind, assuming DISJOINT atom
        indices (as when concatenating structures -- § 2.1).  ``self``'s
        colour/fdf win."""
        if self.kind != other.kind:
            raise ValueError(
                f"cannot union a {self.kind!r} channel with a "
                f"{other.kind!r} channel")
        if self.kind == "value":
            data: Any = {**self.data, **other.data}
        else:
            data = sorted(set(self.data) | set(other.data))
        return AtomChannel(self.kind, data, self.color, self.fdf)

    def copy(self) -> "AtomChannel":
        data = dict(self.data) if self.kind == "value" else list(self.data)
        return AtomChannel(self.kind, data, self.color, self.fdf)

    def to_json(self) -> dict:
        """JSON-friendly form (for the .molstruct.json sidecar, § 3).
        ``value`` data keys become strings (JSON object keys)."""
        if self.kind == "value":
            data: Any = {str(k): v for k, v in self.data.items()}
        else:
            data = list(self.data)
        out = {"kind": self.kind, "data": data}
        if self.color is not None:
            out["color"] = self.color
        if self.fdf is not None:
            out["fdf"] = self.fdf
        return out

    @classmethod
    def from_json(cls, obj: dict) -> "AtomChannel":
        """Inverse of :meth:`to_json` (value keys back to ints)."""
        kind = obj["kind"]
        raw = obj.get("data")
        if kind == "value":
            data: Any = {int(k): v for k, v in (raw or {}).items()}
        else:
            data = [int(i) for i in (raw or [])]
        return cls(kind, data, obj.get("color"), obj.get("fdf"))


def annotations_to_json(ann: "dict[str, AtomChannel]") -> dict:
    """Serialize an annotations map for the sidecar (§ 3)."""
    return {name: ch.to_json() for name, ch in ann.items()}


def annotations_from_json(obj: Optional[dict]) -> "dict[str, AtomChannel]":
    """Deserialize an annotations map from the sidecar (§ 3).  STRICT: each
    value must be a JSON channel dict.  ``AtomChannel`` objects live only
    in-memory on a Structure; the metadata dict that crosses the API boundary
    (metadata_to_dict / apply_metadata_dict / to_dict) is always JSON."""
    return {name: AtomChannel.from_json(v) for name, v in (obj or {}).items()}


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


def merge_annotations(base: "dict[str, AtomChannel]",
                      add: "dict[str, AtomChannel]") -> "dict[str, AtomChannel]":
    """Union two annotation maps -- channels sharing a name are merged
    (assuming DISJOINT atom indices, as when concatenating structures,
    § 2.1).  Used by ``Structure.concat``."""
    out = {name: ch.copy() for name, ch in base.items()}
    for name, ch in add.items():
        out[name] = out[name].union(ch) if name in out else ch.copy()
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
    # Per-axis periodicity KIND (structure-periodicity.md) -- the authoritative
    # periodicity field.  Values: "periodic" (k-sampled / tileable lattice),
    # "isolated" (vacuum box), "transport" (electrode-matched, semi-infinite).
    # ``pbc`` above is the DERIVED ASE view (periodic|transport -> True,
    # isolated -> False).  None -> derived from ``pbc``/``cell`` in __post_init__.
    axis_kind:     Optional[Tuple[str, str, str]] = None
    # Isolation padding (Å) on isolated axes -- the PER-SIDE vacuum gap.
    # (k-grid is NOT here: it's a reciprocal-space SAMPLING knob, a CALCULATION
    # parameter that lives on SiestaConfig / TransportConfig, not the geometry.
    # structure-periodicity.md.)  Default 0 keeps existing call sites unchanged.
    vacuum:        Tuple[float, float, float]  = (0.0, 0.0, 0.0)
    # World-space LOW CORNER an EXPLICIT ``cell`` emanates from (Angstrom), or None
    # = (0,0,0) (structure-periodicity.md § 3c).  Editing convenience: an op that
    # builds a cell AROUND off-origin atoms (e.g. add_electrode_slab, whose slabs
    # straddle the origin) sets this to the structure's low corner so the cell WRAPS
    # the atoms WITHOUT moving them -- the molecule/selection stays pinned where the
    # user placed it.  SIESTA correctness is restored at generation: render_fdf
    # translates atoms by -resolve_cell_origin() so they sit in [0, cell); the
    # ``calibrate`` op bakes that same shift into the stored coords (cell_origin ->
    # 0).  ONLY meaningful with an explicit ``cell``; None for a derived cell (its
    # origin is computed from atom extents) and for an imported crystal (atoms are
    # already in [0, cell), so the cell sits at the world origin).
    cell_origin:   Optional[np.ndarray]            = None
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

        # Reconcile axis_kind <-> pbc (structure-periodicity.md).  axis_kind is
        # authoritative; pbc is its derived ASE view.  "transport" can't be
        # recovered from a boolean, so legacy pbc-only callers get
        # periodic/isolated; a builder sets transport explicitly.
        _KINDS = ("periodic", "isolated", "transport")
        if self.axis_kind is None:
            self.axis_kind = tuple("periodic" if b else "isolated"
                                   for b in self.pbc)
        else:
            ak = tuple(str(k) for k in self.axis_kind)
            if len(ak) != 3 or any(k not in _KINDS for k in ak):
                raise ValueError(
                    f"Structure.axis_kind must be exactly 3 of {_KINDS}; "
                    f"got {self.axis_kind!r}"
                )
            self.axis_kind = ak
            self.pbc = tuple(k != "isolated" for k in ak)   # pbc DERIVED
        # cell_origin: the low corner an EXPLICIT cell emanates from (§ 3c).  Only
        # meaningful WITH an explicit cell -- a derived cell computes its origin from
        # atom extents (resolve_cell_origin), so a stray cell_origin without a cell is
        # dropped to keep the field a faithful "explicit-cell offset from (0,0,0)".
        if self.cell_origin is not None:
            co = np.asarray(self.cell_origin, dtype=float).reshape(3)
            if not np.all(np.isfinite(co)):
                raise ValueError(
                    "Structure.cell_origin must be 3 finite floats (Angstrom)")
            self.cell_origin = co if self.cell is not None else None
        # Shape/clamp vacuum (per-side gap).
        self.vacuum = tuple(float(v) for v in self.vacuum)
        if len(self.vacuum) != 3:
            raise ValueError("Structure.vacuum must have exactly 3 entries")

        # Validate transport metadata.  Both fields default to empty,
        # so a caller that doesn't care about regions / frozen atoms
        # sees no behaviour change.
        self._validate_regions(n)
        self._validate_frozen_atoms(n)
        self._validate_annotations(n)

    def resolve_cell(self) -> Optional[np.ndarray]:
        """The 3x3 lattice for this structure (structure-periodicity.md § 3).

        An explicit ``self.cell`` (imported / captured at construction / user
        override) wins verbatim.  Otherwise derive per ``axis_kind``:

          * ``isolated``  -> bbox + 2*vacuum (a box; vacuum >= 0)
          * ``transport`` -> bbox (matched device length; vacuum ignored)
          * ``periodic``  -> ERROR -- a periodic axis needs a commensurate
            lattice from construction/import, never a bounding box.

        ``vacuum[i]`` is the **per-side gap** (Angstrom): the box gets ``vacuum``
        of empty space on EACH face of an isolated axis, so the cell length is
        ``bbox[i] + 2*vacuum[i]`` and the molecule sits centred with ``vacuum`` of
        clearance on both sides.  This matches the SIESTA FDF vacuum box
        (``render_fdf``: ``extent + 2*cell_padding``, centred) so the displayed
        cell reflects what the calculation actually uses.  See
        ``resolve_cell_origin`` for the box's low corner.

        Assumes a block-orthogonal cell (per-axis diagonal); a general triclinic
        cell must arrive explicit.  Returns None for an empty structure.
        """
        if self.cell is not None:
            return self.cell
        if len(self.positions) == 0:
            return None
        extent = self.positions.max(axis=0) - self.positions.min(axis=0)
        out = np.zeros((3, 3), dtype=float)
        for i, kind in enumerate(self.axis_kind):
            if kind == "periodic":
                raise ValueError(
                    f"axis {i} is 'periodic' but Structure.cell is None; a "
                    f"periodic axis needs a commensurate lattice from "
                    f"construction/import (never a bounding box)."
                )
            # vacuum is the PER-SIDE gap -> 2*vacuum total padding.  The
            # EFFECTIVE vacuum applies the § 6.1 floor so a flat or linear
            # molecule can never produce a zero-thickness box.
            pad = 2.0 * self.effective_vacuum()[i] if kind == "isolated" else 0.0
            out[i, i] = float(extent[i]) + pad
        return out

    def resolve_cell_origin(self) -> Optional[np.ndarray]:
        """The low corner (Angstrom) the resolved cell emanates from (§ 3c).

        The consumer contract: the viewer draws the cell wireframe FROM this corner
        (so the box wraps the structure), and ``render_fdf`` translates atoms by
        ``-resolve_cell_origin()`` so SIESTA receives them inside ``[0, cell)`` with
        the cell at ``(0,0,0)``.  Three cases (structure-periodicity.md § 3c):

          * EXPLICIT cell + ``cell_origin`` set (an electrode junction: the cell was
            built AROUND off-origin atoms) -> ``cell_origin``.  The box wraps the
            atoms where they are; generation shifts them into the cell.
          * EXPLICIT cell, NO ``cell_origin`` -> the corner is **derived**, never
            assumed to be the world origin (decided 2026-07-29, § 6.1): ``None``
            (= world origin, no shift) only when the box AT the world origin
            already contains every atom along the non-periodic axes (an imported
            crystal), otherwise the wrapping corner so the box encloses the
            structure instead of jumping to ``(0,0,0)``.
          * DERIVED (bbox) cell -> ``bbox_min - vacuum`` (isolated) / ``bbox_min``
            (transport), so the molecule is centred with ``vacuum`` clearance/side.

        "No explicit origin" therefore means "derive the corner" at EVERY seam,
        and a derived corner is never materialised back into the truth (§ 6.1
        clause 1).  The frame-contract gate validates this state; it does not
        rewrite it.

        Returns ``None`` for an empty structure (nothing to anchor)."""
        if len(self.positions) == 0:
            return None
        if self.cell is None:
            return self.expected_cell_corner()
        if self.cell_origin is not None:
            return self.cell_origin.astype(float)
        return self._derived_corner_under_explicit_cell()

    # -- the ONE definition of the derived corner + containment (§ 6.1) -- #
    # periodicity_gate.expected_corner / contains_atoms delegate here so the
    # rule cannot fork between the view and the gate.

    def effective_vacuum(self) -> Tuple[float, float, float]:
        """The per-side vacuum the DERIVED box actually uses (§ 6.1).

        Identical to ``self.vacuum`` except for the **minimum-thickness floor**:
        on an ISOLATED axis whose derived length would come out below
        ``_MIN_DERIVED_CELL_LENGTH``, the vacuum is raised to
        ``_MIN_ISOLATED_VACUUM``.  That guarantees two things the rest of the
        stack relied on and never had: the derived cell is always genuinely
        three-dimensional, and there is always a real gap between periodic
        images.  Without it a FLAT molecule (water, benzene — zero extent along
        one axis) with no vacuum produced a zero-thickness box: a zero
        determinant that surfaced as "degenerate cell" errors from the emitter
        and blocked a reset-to-derived in the gate.

        This is a RESOLVED value, never written back: ``self.vacuum`` keeps
        exactly what the user typed (§ 6.1 clause 1), and the gate emits a
        notice when the floor is in effect so the box is never silently
        different from the number on screen.

        The floor is a STRUCTURAL minimum, not a claim of physical adequacy: 3 Å
        keeps the cell well-formed, while a converged isolated-molecule
        calculation wants far more (the SIESTA validator still asks for ≥ 8 Å
        per side, ≥ 25 Å charged — ``cell.vacuum_thin``).  Vacuum is meaningless
        on a periodic axis (the lattice sets the length) and on a transport axis
        (the device length is matched), so the floor applies to neither."""
        vac = [float(v) for v in self.vacuum]
        if len(self.positions) == 0:
            return (vac[0], vac[1], vac[2])
        extent = self.positions.max(axis=0) - self.positions.min(axis=0)
        for i, kind in enumerate(self.axis_kind):
            if kind != "isolated":
                continue
            if float(extent[i]) + 2.0 * vac[i] < _MIN_DERIVED_CELL_LENGTH:
                vac[i] = max(vac[i], _MIN_ISOLATED_VACUUM)
        return (vac[0], vac[1], vac[2])

    def vacuum_floor_axes(self) -> List[int]:
        """Isolated axes where :meth:`effective_vacuum` raised the stored value
        (the § 6.1 minimum-thickness floor).  Empty in the normal case; the gate
        turns a non-empty list into a user notice."""
        eff = self.effective_vacuum()
        return [i for i in range(3)
                if abs(eff[i] - float(self.vacuum[i])) > 1e-9]

    def expected_cell_corner(self) -> np.ndarray:
        """The low corner that wraps the structure honouring the per-direction
        vacuum: ``bbox_min - vacuum`` on an isolated axis, ``bbox_min`` on a
        transport axis, ``0`` on a periodic axis (the phase convention).

        Uses the EFFECTIVE vacuum (§ 6.1 floor) so the corner and the cell
        length agree — otherwise a floored axis would grow the box on both faces
        while the corner stayed put, and the molecule would sit off-centre."""
        out = np.zeros(3, dtype=float)
        if len(self.positions) == 0:
            return out
        lo = self.positions.min(axis=0).astype(float)
        eff = self.effective_vacuum()
        for i, kind in enumerate(self.axis_kind):
            if kind == "isolated":
                out[i] = lo[i] - eff[i]
            elif kind == "transport":
                out[i] = lo[i]
        return out

    def _frac_coords(self, origin) -> np.ndarray:
        """Fractional coordinates relative to ``(origin, cell)``.  Triclinic-
        safe: solves ``cell.T @ frac = pos - origin``."""
        rel = (self.positions.astype(float)
               - np.asarray(origin, dtype=float).reshape(1, 3))
        return np.linalg.solve(np.asarray(self.cell, dtype=float).T, rel.T).T

    def cell_contains_atoms(self, origin=None) -> bool:
        """True when every atom sits inside ``[origin, origin + cell)`` along
        every NON-PERIODIC axis.  Along a periodic axis atoms outside the cell
        are legitimate periodic images (the engine wraps them), so containment
        is never required there.  ``origin=None`` = the world origin; trivially
        True with no explicit cell or no atoms."""
        if self.cell is None or len(self.positions) == 0:
            return True
        o = (np.zeros(3) if origin is None
             else np.asarray(origin, dtype=float).reshape(3))
        frac = self._frac_coords(o)
        for i, kind in enumerate(self.axis_kind):
            if kind == "periodic":
                continue
            if not (np.all(frac[:, i] >= -_CONTAIN_EPS)
                    and np.all(frac[:, i] <= 1.0 + _CONTAIN_EPS)):
                return False
        return True

    def _derived_corner_under_explicit_cell(self) -> Optional[np.ndarray]:
        """The corner for an explicit cell that stores no origin: the world
        origin when the atoms are already inside it (imported crystal), else
        the wrapping corner, else -- when the cell fits the structure but not
        structure + vacuum -- the structure centred in the box."""
        if self.cell_contains_atoms(None):
            return None
        corner = self.expected_cell_corner()
        if self.cell_contains_atoms(corner):
            return corner
        frac = self._frac_coords(np.zeros(3))
        lens = np.linalg.norm(np.asarray(self.cell, dtype=float), axis=1)
        lo = self.positions.min(axis=0).astype(float)
        for i, kind in enumerate(self.axis_kind):
            if kind == "periodic":
                continue
            ext = float(frac[:, i].max() - frac[:, i].min()) * lens[i]
            corner[i] = lo[i] - max(0.0, lens[i] - ext) / 2.0
        return corner

    # ------------------------------------------------------------------ #
    #  Sidecar-metadata contract -- the ONE get/set (data-vocabulary.md)  #
    # ------------------------------------------------------------------ #
    # The persisted ``.molstruct.json`` sidecar IS the serialization of this
    # dataclass's metadata.  These TWO methods are the SINGLE place the
    # metadata field set is enumerated: ``metadata_to_dict`` (struct -> dict)
    # and ``apply_metadata_dict`` (dict -> struct).  The sidecar read + write
    # modules and the workspace codec ALL route through them, so the write and
    # read paths physically cannot drift a field -- the exact class of bug that
    # silently dropped ``cell_origin`` on reload.  Add a metadata field = add it
    # to the dataclass + these two methods, and nowhere else.
    #
    # SCOPE = the dataclass's OWN metadata (periodicity + selection tags +
    # annotations).  ``selection_rules`` is NOT a Structure field (a sidecar-only
    # pass-through) and the JSON envelope (schema_version / n_atoms_total /
    # structure_hash / created_by / created_at) belongs to the sidecar layer;
    # both sit AROUND this contract, not inside it.

    def metadata_to_dict(self) -> dict:
        """Serialize this structure's metadata to a JSON-friendly dict (the
        sidecar field set).  The fields are already validated by
        ``__post_init__``; this is a pure conversion to JSON types."""
        return {
            "regions":      {k: list(v)
                             for k, v in (self.regions or {}).items()},
            "frozen_atoms": list(self.frozen_atoms or []),
            "cell":         self.cell.tolist() if self.cell is not None else None,
            "cell_origin":  (self.cell_origin.tolist()
                             if self.cell_origin is not None else None),
            "pbc":          ([bool(x) for x in self.pbc]
                             if self.pbc is not None else None),
            "axis_kind":    (list(self.axis_kind)
                             if self.axis_kind is not None else None),
            "vacuum":       [float(x) for x in self.vacuum],
            "annotations":  annotations_to_json(self.annotations),
        }

    def apply_metadata_dict(self, data: Optional[dict]) -> None:
        """Apply a sidecar metadata dict onto this structure IN PLACE, then
        re-run the dataclass's own reconciliation + validation
        (``__post_init__``) so there is ONE validator.

        Full-REPLACE semantics: an absent key resets that field to its default
        (absent cell -> non-periodic; absent regions -> none), matching a v3
        back-read.  Raises ``ValueError`` on any invalid field (bad cell,
        out-of-range index, ...), sourced from the same invariants a freshly
        constructed Structure enforces."""
        data = data or {}
        self.regions      = dict(data.get("regions") or {})
        self.frozen_atoms = list(data.get("frozen_atoms") or [])
        self.cell         = (np.asarray(data["cell"], dtype=float)
                             if data.get("cell") is not None else None)
        self.cell_origin  = (np.asarray(data["cell_origin"], dtype=float)
                             if data.get("cell_origin") is not None else None)
        self.pbc          = (tuple(bool(x) for x in data["pbc"])
                             if data.get("pbc") is not None else None)
        self.axis_kind    = (tuple(str(k) for k in data["axis_kind"])
                             if data.get("axis_kind") is not None else None)
        self.vacuum       = (tuple(float(x) for x in data["vacuum"])
                             if data.get("vacuum") is not None else (0.0, 0.0, 0.0))
        self.annotations  = annotations_from_json(data.get("annotations"))
        # Re-run the dataclass invariants ONCE: cell 3x3 + non-singular, the
        # axis_kind<->pbc reconciliation (axis_kind authoritative), cell_origin
        # only-with-a-cell, and region/frozen/annotation indices in range.
        self.__post_init__()

    # ------------------------------------------------------------------ #
    #  Whole-structure codec -- the ONE (de)serialiser (structure-        #
    #  authority.md § 3.1).  ``to_dict``/``from_dict`` carry coordinates  #
    #  + per-atom columns + the full metadata block (delegated to         #
    #  ``metadata_to_dict``/``apply_metadata_dict``) as a single          #
    #  round-trippable dict: ``Structure.from_dict(s.to_dict())`` == s.   #
    #  Outside this class NOBODY assembles or picks apart a structure     #
    #  dict -- that is what let ``cell_origin`` drift on every hand-rolled #
    #  repack.  Add a field = add it here + the two metadata methods,     #
    #  nowhere else.                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """The ONE canonical serialiser: everything needed to reconstruct this
        Structure -- coordinates, per-atom identity columns, AND the full
        metadata field set (nested under ``metadata`` via
        :meth:`metadata_to_dict`).  Loss-free + filesystem-free; the round-trip
        unit the persistence + sidecar + CLI layers store.  Inverse:
        :meth:`from_dict`."""
        return {
            "title":         self.title or "",
            "elements":      list(self.elements),
            "positions":     self.positions.tolist(),
            "atom_names":    list(self.atom_names)    if self.atom_names    else [],
            "residue_ids":   list(self.residue_ids)   if self.residue_ids   else [],
            "residue_names": list(self.residue_names) if self.residue_names else [],
            "chain_ids":     list(self.chain_ids)     if self.chain_ids     else [],
            "metadata":      self.metadata_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Structure":
        """The ONE canonical deserialiser -- inverse of :meth:`to_dict`.
        Constructs a Structure from the canonical dict, then applies +
        validates the metadata through the SAME single authority
        (:meth:`apply_metadata_dict` -> ``__post_init__``) a freshly built
        Structure runs.  Outside this class NOBODY picks coordinate / metadata
        keys out of a structure dict."""
        if data is None:
            raise ValueError("Structure.from_dict: data is None")
        s = cls(
            elements=list(data["elements"]),
            positions=np.asarray(data["positions"], dtype=float),
            atom_names=list(data.get("atom_names")    or []) or None,
            residue_ids=list(data.get("residue_ids")   or []) or None,
            residue_names=list(data.get("residue_names") or []) or None,
            chain_ids=list(data.get("chain_ids")     or []) or None,
            title=data.get("title", "") or "",
        )
        # Full-replace + revalidate the metadata block through the ONE codec.
        s.apply_metadata_dict(data.get("metadata"))
        return s

    def to_wire(self) -> dict:
        """The read-only server->client view (structure-authority.md § 3.2):
        the metadata-bearing portion of the wire response, assembled by
        Structure so no blueprint enumerates a field.  It is the identity
        columns + the FULL ``periodicity`` block (the raw cell/origin PLUS the
        server-resolved ``resolved_cell`` / ``resolved_cell_origin``, computed
        HERE via the one resolver so they can never drift or drop) +
        ``annotations``.

        The web layer composes this with its own render/validation concerns
        (the flat ``atoms`` list, ``issues``, ``text``, ``extra``); it must NOT
        re-list any periodicity / metadata field.  Not round-tripped by
        :meth:`from_dict` -- ``resolved_*`` are derived, read-only fields."""
        # The ONE resolver (structure-periodicity.md § 3a/3c), run once, here.
        # A periodic axis without a lattice raises -> no resolved box (None),
        # matching the previous hand-rolled behaviour in _shared.
        try:
            _rc = self.resolve_cell()
            resolved_cell = _rc.tolist() if _rc is not None else None
        except Exception:  # noqa: BLE001
            resolved_cell = None
        try:
            _ro = self.resolve_cell_origin()
            resolved_origin = _ro.tolist() if _ro is not None else None
        except Exception:  # noqa: BLE001
            resolved_origin = None
        return {
            "title":         self.title or "",
            "elements":      list(self.elements),
            "atom_names":    list(self.atom_names)    if self.atom_names    else [],
            "residue_ids":   list(self.residue_ids)   if self.residue_ids   else [],
            "residue_names": list(self.residue_names) if self.residue_names else [],
            "chain_ids":     list(self.chain_ids)     if self.chain_ids     else [],
            "n_residues":    self.n_residues,
            "periodicity": {
                "cell":                 self.cell.tolist() if self.cell is not None else None,
                "cell_origin":          (self.cell_origin.tolist()
                                         if self.cell_origin is not None else None),
                "resolved_cell":        resolved_cell,
                "resolved_cell_origin": resolved_origin,
                "axis_kind":            (list(self.axis_kind)
                                         if self.axis_kind is not None else None),
                "vacuum":               [float(x) for x in self.vacuum],
                # The vacuum the derived box ACTUALLY uses: identical to
                # ``vacuum`` unless the § 6.1 minimum-thickness floor raised it
                # on a flat/linear axis.  Sent so the Cell page can show the
                # effective number -- a box thicker than the vacuum on screen
                # must never be a surprise (it is a VIEW, like resolved_cell:
                # the stored vacuum keeps what the user typed).
                "resolved_vacuum":      [float(x) for x in
                                         self.effective_vacuum()],
                # (No "kgrid": a sampling knob on the config, not geometry --
                # structure-periodicity.md.)
            },
            "annotations":   annotations_to_json(self.annotations),
        }

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

    def _carry_periodicity(self) -> dict:
        """Periodicity fields (cell / pbc / axis_kind / vacuum) for
        reconstructing a Structure that EDITS ATOMS but keeps the lattice.

        None of these are per-atom, so an add / delete / rigid transform
        carries them verbatim.  Dropping any of them silently reverts a
        periodic or transport cell to isolated defaults (axis_kind -> derived
        from pbc, vacuum -> 0) -- e.g. deleting a stray atom would wipe a
        transport cell, and the emitted SIESTA FDF would omit
        ``LatticeVectors``.  Every op-helper that rebuilds a Structure spreads
        this so the lattice survives the edit.
        """
        return dict(
            cell        = (self.cell.copy() if self.cell is not None else None),
            cell_origin = (self.cell_origin.copy()
                           if self.cell_origin is not None else None),
            pbc         = self.pbc,
            axis_kind   = self.axis_kind,
            vacuum      = self.vacuum,
        )

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
            annotations   = copy_annotations(self.annotations),
            **self._carry_periodicity(),
        )

    def affine(self, linear: Sequence[Sequence[float]],
               translation: Sequence[float]) -> "Structure":
        """Apply one rigid/affine map ``x -> x @ linearᵀ + translation`` to the
        WHOLE structure -- atoms AND the unit-cell box -- so an explicit box keeps
        wrapping the atoms after a rigid transform (§ 3c).  THE single transform
        primitive ``translated`` / ``rotate_around_axis`` route through.

        * atoms:        ``positions @ linearᵀ + translation``
        * lattice VECTORS (``cell`` rows): ``cell @ linearᵀ`` -- the vectors rotate/
          shear with the linear part; the translation cancels (they are differences
          of points).
        * cell ORIGIN (``cell_origin``, the world-space corner): the FULL affine
          ``cell_origin @ linearᵀ + translation`` -- it is a point and follows the
          atoms.

        A DERIVED cell (``cell`` None) / unset origin (``cell_origin`` None) needs no
        update: ``resolve_cell`` / ``resolve_cell_origin`` recompute it from the new
        atom extents.  Index-preserving, so regions / frozen / annotations / axis_kind
        / vacuum / pbc carry verbatim.  For a pure rotation ``linear`` is orthogonal
        (det +1), so the cell stays non-singular."""
        L = np.asarray(linear, dtype=float).reshape(3, 3)
        t = np.asarray(translation, dtype=float).reshape(3)
        per = self._carry_periodicity()
        if per.get("cell") is not None:
            per["cell"] = per["cell"] @ L.T
        if per.get("cell_origin") is not None:
            per["cell_origin"] = per["cell_origin"] @ L.T + t
        return Structure(
            elements      = list(self.elements),
            positions     = self.positions @ L.T + t,
            atom_names    = list(self.atom_names),
            residue_ids   = list(self.residue_ids),
            residue_names = list(self.residue_names),
            chain_ids     = list(self.chain_ids),
            title         = self.title,
            regions       = {k: list(v) for k, v in self.regions.items()},
            frozen_atoms  = list(self.frozen_atoms),
            annotations   = copy_annotations(self.annotations),
            **per,
        )

    def translated(self, vec: Sequence[float]) -> "Structure":
        # A rigid translation: linear part = identity (lattice vectors unchanged),
        # the cell's world-space corner moves WITH the atoms (§ 3c).  Routed through
        # the ONE affine primitive so atoms + box stay consistent.
        return self.affine(np.eye(3), np.asarray(vec, dtype=float).reshape(3))

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
        annotations: Dict[str, AtomChannel] = {}
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
            # Extensible annotation channels re-index the same way (§ 2.1):
            # offset this input's atom indices, then union by channel name.
            if s.annotations:
                off = {i: i + atom_offset for i in range(s.n_atoms)}
                annotations = merge_annotations(
                    annotations, remap_annotations(s.annotations, off))
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
            annotations   = annotations,
            cell          = (base_cell.copy() if base_cell is not None
                             else None),
            pbc           = base_pbc,
        )
