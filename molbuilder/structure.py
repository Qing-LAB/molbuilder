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

#: THE spelling of the reserved "held still during relaxation" label
#: (atom-annotations.md § 2, molview.md § 6.6).  It is an ORDINARY label: same
#: store (``Structure.regions``), same validation, same serialisation, same
#: filtering, same panel row as ``L-electrode`` or anything a user types.  The
#: only thing that makes it reserved is that something downstream ACTS on it --
#: the SIESTA ``%block Geometry.Constraints`` emitter, the PySCF freeze list --
#: and for that there is exactly one designated read, :attr:`Structure.frozen_atoms`.
#:
#: One constant because the name is the whole cost of a reserved meaning.  It is
#: the name the label has ALWAYS had on the wire, on disk and in the browser;
#: what a second storage bought was a SECOND spelling for the same fact -- the
#: ``frozen`` flag channel this module used to synthesise beside the label --
#: and an alias between them at every boundary that touched both.
FROZEN_LABEL = "frozen_atoms"

#: THE metadata field set -- what :meth:`Structure.metadata_to_dict` writes and
#: :meth:`Structure.apply_metadata_dict` accepts, named once so the two cannot
#: enumerate different sets.  A dict carrying anything else is REFUSED rather
#: than partly applied: a key this does not know is a fact the caller believes
#: it stored, and silently dropping it is how a structure reaches a calculation
#: missing labels nobody noticed were gone.
METADATA_FIELDS = ("regions", "cell", "cell_origin", "pbc", "axis_kind",
                   "vacuum", "annotations")

#: Containment tolerance in FRACTIONAL units (§ 6.1): loose enough to forgive a
#: round-tripped float, tight enough that "half the molecule outside the box"
#: can never pass.  Shared with periodicity_gate, which delegates containment
#: to :meth:`Structure.cell_contains_atoms`.
_CONTAIN_EPS = 1e-6

#: The per-side gap a DERIVED box uses on an isolated axis when the user set no
#: vacuum at all (§ 6.1; the rule was rewritten 2026-08-03 — cell-plan.md § 3b).
#:
#: IT IS A DEFAULT GAP, NOT A MINIMUM BOX LENGTH.  3 Å of empty space is 3 Å
#: whether the molecule is 2 Å across or 200, so every isolated axis gets it
#: when nothing was said.  The previous rule raised the vacuum only when the
#: resulting BOX came out under 3 Å, which meant a large molecule got no gap at
#: all and a typed 1.0 Å got overridden to 3.0.
#:
#: It keeps a cell well-formed; it is NOT a claim of physical adequacy.  A
#: converged isolated-molecule run wants far more, and the validator says so
#: (``cell.vacuum_thin``: ≥ 8 Å per side neutral, ≥ 25 Å charged).
_DEFAULT_ISOLATED_VACUUM = 3.0


def _vacuum_from_stored(raw) -> Optional[Tuple[float, float, float]]:
    """Read a stored vacuum, treating ``[0, 0, 0]`` as UNSET.

    Decided 2026-08-03 (docs/model/cell-plan.md § 5).  Every
    ``.molstruct.json`` written before that date says ``vacuum: [0,0,0]``,
    because the field had no unset state and defaulted to zeros.  Reading those
    as "the user explicitly chose zero" would silently change what an existing
    flat-molecule structure does on its next load -- it would stop getting the
    default gap, and its box would lose its volume -- which is exactly the class
    of silent change this work exists to prevent.

    So an all-zero stored vacuum reads as *nothing was said*.  It is slightly
    dishonest for exactly one value, and it is the price of not rewriting files
    nobody asked us to touch.  A user who genuinely wants a zero gap says so,
    and once the writer emits ``null`` for unset the two are distinguishable in
    every new file.

    DO NOT "FIX" THE ASYMMETRY without reading § 5: it is what protects
    structures already on disk.
    """
    if raw is None:
        return None
    values = tuple(float(x) for x in raw)
    if len(values) != 3:
        raise ValueError("Structure.vacuum must have exactly 3 entries")
    if not any(values):
        return None                     # a stored all-zero reads as UNSET
    return values



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

    Some labels are RESERVED -- something downstream acts on the
    name.  ``frozen`` (:data:`FROZEN_LABEL`) marks atoms whose
    positions stay fixed during relaxations and Hessian builds;
    it is consumed by SpectraConfig (relax + Hessian) and
    TransportConfig (NEGF lead-fixing).  A reserved label is stored,
    validated, filtered and serialised exactly like any other; the
    only thing it gets of its own is one designated read, the
    :attr:`frozen_atoms` accessor.

    ``regions`` is pure metadata -- nothing in this module reads it.
    Downstream consumers (spectra, transport) decide what the names
    mean.
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
    # THE label store -- every label, including the reserved ones (FROZEN_LABEL).
    # There is no second store: a reserved meaning costs a NAME and one
    # designated read (`frozen_atoms` below), and nothing else.  `frozen_atoms`
    # was a field here until 2026-07-31, which bought two of everything --
    # two validators, two remaps on every atom-count change, two keys in the
    # saved file, two spellings on the wire -- and a live inconsistency: the
    # selection panel saw frozen as a label on `/api/selection/eval` and as a
    # flag on `/api/selection/atoms`, so it double-rendered until a route-
    # conditional patch hid it.
    regions:       Dict[str, List[int]] = field(default_factory=dict)
    # NOT A FIELD -- a constructor door onto the reserved label, replaced below
    # the class by the `frozen_atoms` property.  Declared here so `Structure(...,
    # frozen_atoms=[...])` still reaches the ONE place that spells the name,
    # instead of every construction site writing `regions={FROZEN_LABEL: ...}`
    # for itself.  `regions` is declared FIRST on purpose: the dataclass assigns
    # in declaration order, so the store exists when the setter writes into it.
    # Default None means "say nothing about it" -- a caller passing only
    # `regions` (with the label already in it) must not have it cleared.
    frozen_atoms:  Optional[List[int]] = None
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
    vacuum:        Optional[Tuple[float, float, float]] = None
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
            # THE shared threshold (cell.ZERO_VOLUME_TOL).  It was 1e-8 here
            # and 1e-6 in the emitter until 2026-08-03 -- two numbers for one
            # question.  Imported lazily: ``cell`` imports this module.
            from .cell import ZERO_VOLUME_TOL
            if abs(float(np.linalg.det(cell))) < ZERO_VOLUME_TOL:
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
        # Shape vacuum (per-side gap).  ``None`` MEANS THE STRUCTURE SAYS
        # NOTHING -- the same "unset" its three siblings (cell, cell_origin,
        # axis_kind) have always had, and the state the whole regime model
        # needs in order to tell "I want no gap" from "I never said" (see
        # docs/model/cell-plan.md § 3a).  Until 2026-08-03 this field defaulted
        # to (0, 0, 0) and those two were one value, so no rule could branch on
        # the difference.
        if self.vacuum is not None:
            self.vacuum = tuple(float(v) for v in self.vacuum)
            if len(self.vacuum) != 3:
                raise ValueError("Structure.vacuum must have exactly 3 entries")

        # Validate transport metadata.  Both fields default to empty,
        # so a caller that doesn't care about regions / frozen atoms
        # sees no behaviour change.
        self._validate_regions(n)
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
            # vacuum is the PER-SIDE gap -> 2*vacuum total padding, which is
            # also the distance between the molecule and its periodic image.
            # The EFFECTIVE vacuum supplies the § 6.1 default where the user
            # set nothing, so a flat or linear molecule can never produce a
            # zero-thickness box.
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

        TWO STATES, AND ONLY TWO (decided 2026-08-03 — cell-plan.md § 3b).

          * **The user set a vacuum** → it is used, verbatim, on every axis.
            However small.  They dictate what they want; a thin gap is warned
            about (``cell.vacuum_thin``) and never overridden.
          * **The user set nothing** → each ISOLATED axis gets
            ``_DEFAULT_ISOLATED_VACUUM`` per side.  It is a default GAP, not a
            floor on the box length: 3 Å of empty space is 3 Å whether the
            molecule is 2 Å across or 200 Å, so a large molecule gets it too.

        Vacuum is meaningless on a periodic axis (the lattice sets the length)
        and on a transport axis (the device length is matched), so neither gets
        a default.

        The default is a STARTING gap, not a claim of physical adequacy: it
        keeps a derived cell three-dimensional even for a FLAT molecule (water,
        benzene — zero extent along one axis), while a converged
        isolated-molecule calculation wants far more.  The SIESTA validator
        still asks for ≥ 8 Å per side, ≥ 25 Å charged (``cell.vacuum_thin``),
        and says so about the default too.

        This is a RESOLVED value, never written back: ``self.vacuum`` keeps
        exactly what the user typed, or ``None`` when they typed nothing
        (§ 6.1 clause 1).  The gate announces the default on every hand-over
        (``cell.check`` → ``cell.vacuum_defaulted``) so the box is never
        silently different from the number on screen.

        UNTIL 2026-08-03 THIS WAS A FLOOR ON THE BOX, ``extent + 2·vacuum <
        3``, which asked about the box rather than about what the user wanted.
        It raised a typed 1.0 Å to 3.0 — overriding a stated value — and it
        left a large molecule with NO gap at all, because its box was already
        over 3 Å.  Both are the same confusion: a minimum box length is not a
        vacuum."""
        if self.vacuum is not None:
            return tuple(float(v) for v in self.vacuum)
        kinds = self.axis_kind or ("isolated", "isolated", "isolated")
        vac = [(_DEFAULT_ISOLATED_VACUUM if k == "isolated" else 0.0)
               for k in kinds]
        return (vac[0], vac[1], vac[2])

    def defaulted_vacuum_axes(self) -> List[int]:
        """Axes whose gap is the DEFAULT, because the user set no vacuum.

        The gate turns a non-empty list into a user notice: a number nobody
        chose is about to size the box a calculation runs in, and that must
        never be a surprise.

        Empty whenever a vacuum IS set — whatever was typed is what is used, on
        every axis — and empty for periodic / transport axes, which get no
        default because vacuum does not apply to them.

        Named ``vacuum_floor_axes`` until 2026-08-03, when the rule stopped
        being a floor.  Nothing is raised any more: there was no stored value
        to raise, only an absent one to fill in."""
        if self.vacuum is not None:
            return []
        eff = self.effective_vacuum()
        return [i for i in range(3) if eff[i] > 0.0]

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
            # Every label, reserved ones included -- ONE key, because there is
            # one store.  A `frozen_atoms` key beside this one is what schema 6
            # wrote; `apply_metadata_dict` still reads it, nothing writes it.
            "regions":      {k: list(v)
                             for k, v in (self.regions or {}).items()},
            "cell":         self.cell.tolist() if self.cell is not None else None,
            "cell_origin":  (self.cell_origin.tolist()
                             if self.cell_origin is not None else None),
            "pbc":          ([bool(x) for x in self.pbc]
                             if self.pbc is not None else None),
            "axis_kind":    (list(self.axis_kind)
                             if self.axis_kind is not None else None),
            "vacuum":       ([float(x) for x in self.vacuum]
                             if self.vacuum is not None else None),
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
        unknown = [k for k in data if k not in METADATA_FIELDS]
        if unknown:
            raise ValueError(
                f"Structure.apply_metadata_dict: unknown metadata "
                f"{sorted(unknown)!r}; known fields are "
                f"{list(METADATA_FIELDS)!r}.  A key this does not know is a "
                f"fact you believe you stored -- it is refused rather than "
                f"dropped.  (The reserved `frozen_atoms` label lives in "
                f"`regions` with every other label.)")
        self.regions      = dict(data.get("regions") or {})
        self.cell         = (np.asarray(data["cell"], dtype=float)
                             if data.get("cell") is not None else None)
        self.cell_origin  = (np.asarray(data["cell_origin"], dtype=float)
                             if data.get("cell_origin") is not None else None)
        self.pbc          = (tuple(bool(x) for x in data["pbc"])
                             if data.get("pbc") is not None else None)
        self.axis_kind    = (tuple(str(k) for k in data["axis_kind"])
                             if data.get("axis_kind") is not None else None)
        self.vacuum       = _vacuum_from_stored(data.get("vacuum"))
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
                "vacuum":               ([float(x) for x in self.vacuum]
                                         if self.vacuum is not None else None),
                # The vacuum the derived box ACTUALLY uses: identical to
                # ``vacuum`` unless it is UNSET, when the § 6.1 default gap
                # supplies one.  Sent so the Cell page can show the
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
        if not isinstance(self.regions, dict):
            raise ValueError(
                f"Structure.regions must be a dict of label -> atom indices; "
                f"got {type(self.regions).__name__}")
        if not self.regions:
            return
        normalised: Dict[str, List[int]] = {}
        for region_name, idxs in self.regions.items():
            if not isinstance(region_name, str) or not region_name:
                raise ValueError(
                    f"Structure.regions: region label must be a "
                    f"non-empty string; got {region_name!r}"
                )
            # A LIST of indices, checked to its depth.  A str is iterable and a
            # dict iterates its keys, so both would "work" here and produce
            # nonsense -- `{"x": "012"}` would become atoms 0, 1, 2.
            if not isinstance(idxs, (list, tuple)):
                raise ValueError(
                    f"Structure.regions[{region_name!r}] must be a list of "
                    f"atom indices; got {type(idxs).__name__}")
            unique: set = set()
            for raw in idxs:
                # A REAL int.  `int(raw)` would accept "3" (a string index from
                # a JSON round-trip that lost its typing), truncate 3.7 to 3
                # without telling anyone, and take True as atom 1.
                if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
                    raise ValueError(
                        f"Structure.regions[{region_name!r}]: atom index must "
                        f"be an int; got {type(raw).__name__} ({raw!r})")
                idx = int(raw)
                if not 0 <= idx < n:
                    raise ValueError(
                        f"Structure.regions[{region_name!r}]: atom "
                        f"index {idx} out of range [0, {n})"
                    )
                unique.add(idx)
            normalised[region_name] = sorted(unique)
        self.regions = normalised

    def _validate_annotations(self, n: int) -> None:
        """Extra channels: names must not collide with a label;
        atom indices must be in
        [0, n).  Normalises tag/flag data to sorted-unique in place."""
        if not self.annotations:
            return
        for name, ch in self.annotations.items():
            if name in self.regions:
                raise ValueError(
                    f"Structure.annotations[{name!r}]: {name!r} is already a "
                    f"label; edit .regions instead.")
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
        """The unified per-atom channel registry: every label as a ``tag``
        channel -- reserved ones included, on identical footing -- plus every
        extensible channel in ``self.annotations``.  The one place to read ALL
        per-atom metadata uniformly."""
        out: Dict[str, AtomChannel] = {}
        for label, idxs in self.regions.items():
            out[label] = AtomChannel("tag", list(idxs))
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
        """Set an EXTENSIBLE channel (stored in ``annotations``).  A name that
        is already a label belongs to the label store -- edit ``.regions``.
        Re-validates against the current atom count."""
        if name in self.regions:
            raise ValueError(
                f"{name!r} is already a label; edit .regions instead.")
        self.annotations[name] = channel
        self._validate_annotations(len(self.positions))

    # ------------------------------------------------------------------ #
    #  The reserved-label read (molview.md § 6.6)                        #
    # ------------------------------------------------------------------ #

    @property
    def _frozen_atoms(self) -> List[int]:
        """The atoms carrying the reserved :data:`FROZEN_LABEL` label.

        THE one way to ask.  A reserved label is an ordinary label -- it is in
        ``regions`` with everything else and is stored, validated, filtered and
        displayed identically -- but because something downstream ACTS on this
        one (SIESTA's ``%block Geometry.Constraints``, PySCF's freeze list), it
        gets a designated read so that "which atoms are held still" is answered
        in one place instead of at every point of use.

        A cut of the label store, never a second home for the fact.  Callers use
        this rather than reaching into ``regions`` for the name themselves:
        every caller that spells the name is another place it can be spelled
        differently, which is the same defect as a separate field reached from
        the other side.
        """
        return list(self.regions.get(FROZEN_LABEL, ()))

    @_frozen_atoms.setter
    def _frozen_atoms(self, indices) -> None:
        """Write the reserved label -- an ordinary label write, normalised
        (sorted + deduped) the way ``_validate_regions`` normalises every other.
        An empty set REMOVES the label rather than storing an empty one, so
        "carries no label" and "carries an empty label" cannot both exist.
        ``None`` says nothing about it and leaves the store untouched, which is
        what an omitted constructor argument means."""
        if indices is None:
            return
        kept = sorted({int(i) for i in indices})
        if kept:
            self.regions[FROZEN_LABEL] = kept
        else:
            self.regions.pop(FROZEN_LABEL, None)

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
                 title: Optional[str] = None,
                 frames_out: Optional[List] = None) -> "Structure":
        """Load a Structure from an XYZ file path or XYZ text content.

        THE PARSE IS ASE'S, NOT OURS (``ase.io.read(..., format="extxyz")``).
        ASE's extended-XYZ reader is a superset reader: it handles the plain
        xmol layout AND the ``Lattice="…" Properties=… pbc="…"`` comment line,
        it canonicalises an external tool's ``FE`` / ``ZN`` to ``Fe`` / ``Zn``,
        and it reads every frame of a multi-frame file.  ASE is a declared
        dependency of this project **for exactly this** (``pyproject.toml``:
        *"XYZ I/O + atomic-number table"*).

        WHY THIS IS NOT HAND-ROLLED ANY MORE.  It was, and the hand-rolled
        parser read the atoms out of the first block and nothing else -- so a
        file this class had itself written with ``to_extxyz`` came back with
        **no cell, no pbc and one frame**.  The project already knew: a second
        reader had been built at ``siesta/input.py`` whose comment said *"Use
        ASE for XYZ -- it understands extended-XYZ headers and gives us the
        lattice when present, which our hand-rolled parser doesn't."*  Two
        readers of one format is two answers to "what is in this file", and the
        one that lost data was the one every other caller used.

        XYZ stores no atom names / residues, so all atoms come back tagged as
        residue 1 ("MOL", chain "A") and atom names default to the element
        symbol.

        :param title: overrides the comment line.
        :param frames_out: when given, EVERY frame's positions are appended to
            it, in file order -- the read-side inverse of ``to_extxyz(frames=)``.
            The Structure itself holds one geometry (frames live with the caller
            that needs them), so a multi-frame file is otherwise read as its
            first frame and this is how the rest is recovered.
        """
        text = _resolve_source(source)
        # THE TITLE IS OURS, and it is the one thing read here rather than
        # parsed by ASE.  ASE's extended-XYZ reader treats the comment line as
        # `key=value` pairs, so a human comment -- "water molecule" -- comes
        # back as ``{'water': True, 'molecule': True}`` and the sentence is
        # gone.  The line is taken verbatim for this one field, which is
        # metadata this class owns, not part of reading the structure.
        lines = text.splitlines()
        comment = lines[1].strip() if len(lines) >= 2 else ""

        from ase.io import read as _ase_read
        try:
            images = _ase_read(StringIO(text), index=":", format="extxyz")
        except StopIteration as exc:                    # an empty document
            raise ValueError(
                "XYZ is empty: need an atom count, a comment line and the atoms"
            ) from exc
        except Exception as exc:                        # ASE's own diagnosis
            raise ValueError(f"could not read XYZ: {exc}") from exc
        if not images:
            raise ValueError("XYZ holds no frames")
        first = images[0]

        if frames_out is not None:
            frames_out.extend(np.asarray(im.get_positions(), dtype=float)
                              for im in images)

        # THE CELL, ONLY WHERE IT MEANS ONE.  A `Lattice=` is adopted as this
        # structure's explicit cell only when some axis is actually periodic.
        # Our own `to_extxyz` writes the RESOLVED box for an isolated molecule
        # too -- its bounding box plus vacuum, with `pbc="F F F"` beside it --
        # and adopting that would promote a DERIVED value into a stored one, so
        # the box would stop tracking the vacuum it came from
        # (structure-periodicity.md's raw-vs-resolved line).  A `.xyz` that
        # travels with its `.molstruct.json` gets the real cell from the
        # sidecar anyway, applied after this parse.
        cell = np.asarray(first.cell, dtype=float)
        periodic = tuple(bool(b) for b in first.pbc)
        carries_cell = bool(cell.any()) and any(periodic)

        return cls(
            elements=list(first.get_chemical_symbols()),
            positions=np.asarray(first.get_positions(), dtype=float),
            title=(title if title is not None else comment),
            cell=(cell.tolist() if carries_cell else None),
            pbc=(periodic if carries_cell else None),
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
    #  Output: extended XYZ (one frame, or a whole trajectory)            #
    # ------------------------------------------------------------------ #

    def to_extxyz(
        self,
        path: Optional[str] = None,
        *,
        frames: Optional[Sequence[Any]] = None,
        comment: str = "",
    ) -> str:
        """Return extended-XYZ text for this structure, or for *frames* of it.

        Extended XYZ is plain XYZ with the per-frame comment line carrying
        key=value metadata -- the convention ASE reads and writes, and what
        every trajectory tool expects.  Two keys go out:

        ``Lattice``
            The cell **as it will actually be used** (:meth:`resolve_cell`),
            row-major, in Angstrom.  This is the same box MolView draws and the
            Cell page reports, so a file and the viewer it came from cannot
            describe different systems.
        ``pbc``
            Which axes are periodic (``T``/``F``), from :attr:`pbc`.  It is what
            keeps the Lattice honest: an isolated molecule still HAS a resolved
            box -- its bounding box plus vacuum -- and writing that without
            ``pbc="F F F"`` would tell the reader the system repeats when it
            does not.

        WHY THIS EXISTS BESIDE ``to_xyz`` AND NOT INSTEAD OF IT.  A plain
        ``.xyz`` has nowhere to put a cell, so a periodic structure written that
        way loses its box -- and a *trajectory* written that way loses it on
        every frame.  ``to_xyz`` stays for the single-frame, cell-less case that
        every code reads; this is for the cases it cannot carry.

        Parameters
        ----------
        frames
            Optional sequence of coordinate arrays, each shaped like
            :attr:`positions` -- one block is written per frame, **in order**.
            Every frame must carry this structure's atom count: the elements and
            the cell are written from ``self`` and are the same for all of them,
            which is what makes it one trajectory rather than a pile of
            structures (the same-atoms rule the frame model rests on).  Omitted,
            one block is written from :attr:`positions`.
        """
        blocks = [self.positions] if frames is None else list(frames)
        if not blocks:
            raise ValueError("to_extxyz: needs at least one frame")

        n = self.n_atoms
        for i, frame in enumerate(blocks):
            got = len(frame)
            if got != n:
                raise ValueError(
                    f"to_extxyz: frame {i} has {got} atoms, but the structure "
                    f"has {n}; every frame of a trajectory carries the same "
                    f"atoms")

        # The box every frame shares.  ``resolve_cell`` can refuse on a
        # structure whose state is contradictory -- the same degradation
        # ``to_wire`` performs, rather than failing the write.
        try:
            cell = self.resolve_cell()
        except ValueError:
            cell = None
        lattice = ""
        if cell is not None:
            flat = " ".join(f"{v:.6f}" for row in np.asarray(cell) for v in row)
            lattice = f'Lattice="{flat}" '
        flags = " ".join("T" if p else "F" for p in self.pbc)
        head = (f'{lattice}Properties=species:S:1:pos:R:3 pbc="{flags}"')
        title = (comment or self.title or "Built by molbuilder").strip()

        buf = StringIO()
        for frame in blocks:
            buf.write(f"{n}\n")
            # The title rides in front of the key=value pairs, where a reader
            # that only wants the metadata still finds it and a human still
            # sees which structure this is.
            buf.write(f"{title} {head}\n" if title else f"{head}\n")
            for el, (x, y, z) in zip(self.elements, frame):
                buf.write(f"{el:<3s} {x: 12.6f} {y: 12.6f} {z: 12.6f}\n")
        text = buf.getvalue()
        if path:
            # Same rule as ``to_xyz``: explicit utf-8, never the platform
            # locale, or a non-ASCII title is silently corrupted on cp1252.
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
        # Labels must be re-indexed per-input because each structure's atom
        # indices are 0-based and the concatenation shifts the i-th structure's
        # atoms by the sum of n_atoms across all earlier structures.  The same
        # label across inputs merges into one combined index list -- reserved
        # labels included, by the same rule, because they are the same thing.
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
            annotations   = annotations,
            cell          = (base_cell.copy() if base_cell is not None
                             else None),
            pbc           = base_pbc,
        )


# The reserved label's accessor, installed under its real name AFTER ``@dataclass``
# has read the class body.  Defining it as ``frozen_atoms`` inside the body would
# make the property object the field's default; defining it here means the
# generated ``__init__`` executes ``self.frozen_atoms = <arg>`` straight into the
# setter, so construction and later assignment go through the same one door.
Structure.frozen_atoms = Structure._frozen_atoms
del Structure._frozen_atoms
