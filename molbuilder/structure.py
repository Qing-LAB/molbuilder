"""Structure dataclass + readers / writers for XYZ / PDB / PySCF / ASE.

The :class:`Structure` is the lingua franca between builders (peptide,
nucleic) and consumers (file writers, downstream analysis).  Every
builder returns one of these; every output format is just a method on
it.  Adding a new format means adding one method here, not touching the
builders.

Loading a stored structure goes through ``StructureCodec().load(path)``,
which reads the pair; ``from_xyz`` / ``from_pdb`` take a document's text, so
an XYZ or PDB exported by a different tool can be fed straight in without
re-building it from scratch.

TRAVELS beside every PySCF script, in ``mb_pyscf.pyz``
(``runwrap.PYSCF_COMPANIONS``, ``engines/pyscf.md`` § 3): the script saves its
geometries through the codec (``to_xyz``) and a continuing run reads its last
one back (``from_xyz``).  So it imports only the standard library and numpy
at load; a method that reaches further (``to_wire``'s ``cell``, ASE in
``from_xyz``/``from_pdb``/``to_ase``) imports inside itself, and of those the
script calls only ``from_xyz`` -- ASE, which the PySCF env carries.

Transport-relevant attributes (see the three-stage contract in
docs/design.md):

  frozen_atoms : List[int]
      0-based indices of atoms the geometry optimiser must NOT
      move.  Not a field of its own: a designated read of
      ``regions[FROZEN_LABEL]``, so there is one store and one
      spelling.  Consumed by Spectra (``cfg.frozen_indices``) and
      EMITTED as a real constraint -- SIESTA's ``%block
      Geometry.Constraints``, PySCF's ``$freeze``.

  regions : Dict[str, List[int]]
      Named groups of atom indices for transport-style partition
      (keys are user-facing labels like ``"L-electrode"`` /
      ``"R-electrode"`` / ``"bridge"``; values are 0-based atom
      indices).  A DICT, not a list of lists.  Validated as
      pairwise-disjoint at ``__post_init__``.

These two attributes are the load-bearing carriers for the
boundary-conditions axis of the three-stage contract.  Any emitter
that drops them silently (rather than warning) violates the
contract.  ``Structure.copy()`` / ``.translated()`` MUST carry
them through (see the methods + their tests).
"""

from __future__ import annotations

import copy as _copy
import json as _json
import re as _re

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union  # noqa: F401 -- `Any` annotates AtomChannel.data

import numpy as np


# --------------------------------------------------------------------- #
#  Source-resolver helper                                               #
# --------------------------------------------------------------------- #


def _require_text(source: object, fmt: str) -> str:
    """The readers take a DOCUMENT, never a path.

    `model/parse.md` § 7 states the rule these readers follow: a reader
    takes a path or it takes text, never both, and a caller holding a path
    reads the file itself.  The one door for a STORED structure is
    ``StructureCodec().load(path)`` -- which reads the pair.
    """
    not_a_path = (
        f"Structure.from_{fmt}() takes {fmt.upper()} text, not a path. "
        f"To read a file use StructureCodec().load(path), which reads the "
        f".molstruct.json sidecar beside it too.")
    if isinstance(source, Path):
        raise TypeError(not_a_path)
    if not isinstance(source, str):
        raise TypeError(
            f"Structure.from_{fmt}() takes {fmt.upper()} text as str, got "
            f"{type(source).__name__}")
    # A `str` PATH is the same mistake in a different type.  A document has
    # line breaks; a one-liner ending in a structure suffix is a filename.
    #
    # Judged by the SHAPE OF THE ARGUMENT, never by touching the disk, so the
    # same string means the same thing here whether or not the file happens
    # to exist.
    if "\n" not in source and source.strip().lower().endswith((".xyz", ".pdb")):
        raise TypeError(not_a_path)
    return source


# ---------------------------------------------------------------------- #
#  Per-atom annotation channels (model/structure-annotations.md)       #
# ---------------------------------------------------------------------- #

_CHANNEL_KINDS = ("tag", "flag", "value")

#: THE spelling of the reserved "held still during relaxation" label
#: (``model/structure-annotations.md`` § 2, ``web/molview.md`` § 6.6).  It is an ORDINARY label: same
#: store (``Structure.regions``), same validation, same serialisation, same
#: filtering, same panel row as ``L-electrode`` or anything a user types.  The
#: only thing that makes it reserved is that something downstream ACTS on it --
#: the SIESTA ``%block Geometry.Constraints`` emitter, the PySCF freeze list --
#: and for that there is exactly one designated read, :attr:`Structure.frozen_atoms`.
#:
#: One constant because the name is the whole cost of a reserved meaning.  It
#: is the name the label carries on the wire, on disk and in the browser.  A
#: second storage -- a synthesised ``frozen`` flag channel beside the label --
#: would be a SECOND spelling for one fact, and would need an alias at every
#: boundary that touches both.
FROZEN_LABEL = "frozen_atoms"

#: THE metadata field set -- what :meth:`Structure.metadata_to_dict` writes and
#: :meth:`Structure.apply_metadata_dict` accepts, named once so the two cannot
#: enumerate different sets.  A dict carrying anything else is REFUSED rather
#: than partly applied: a key this does not know is a fact the caller believes
#: it stored, and silently dropping it is how a structure reaches a calculation
#: missing labels nobody noticed were gone.
METADATA_FIELDS = ("regions", "cell", "engine_offset", "axis_kind",
                   "vacuum", "annotations", "customized")

#: What every door that would edit a frame set answers (``model/structure.md``
#: § 2.2e).  A frame set is a data set made by scripts through the frame doors
#: -- ``with_frames``, ``take``, ``set_customized`` -- and never edited: moving
#: atoms in it, adding or removing them, would make its frames disagree.  One
#: sentence, so every door says the same thing.
FRAME_SET_NOT_EDITED = (
    "a frame set is not edited -- take one frame with frame_at(i), or build "
    "the set again with with_frames (model/structure.md § 2.2e)")

#: The keys of one ``customized`` row (``model/structure.md`` § 2.2d).
_ROW_KEYS = ("name", "value", "unit", "note")


def _customized_row(raw: Any, where: str) -> Dict[str, Any]:
    """One row, validated and in its one form: ``name`` a non-empty string,
    ``value`` a number, text or true/false, ``unit`` and ``note`` text and
    present only when they say something."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"{where}: a row is an object {{name, value, unit?, note?}}; got "
            f"{type(raw).__name__}")
    stray = sorted(k for k in raw if k not in _ROW_KEYS)
    if stray:
        raise ValueError(
            f"{where}: a row carries {stray!r}; its keys are "
            f"{list(_ROW_KEYS)!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: a row's name is a non-empty string; got "
                         f"{name!r}")
    value = raw.get("value")
    if isinstance(value, (bool, np.bool_)):
        value = bool(value)
    elif isinstance(value, (int, np.integer)):
        value = int(value)
    elif isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{where} ({name!r}): a value must be finite; "
                             f"got {value!r}")
    elif not isinstance(value, str):
        raise ValueError(
            f"{where} ({name!r}): a value is a number, text or true/false; "
            f"got {type(value).__name__}")
    out: Dict[str, Any] = {"name": name, "value": value}
    for key in ("unit", "note"):
        said = raw.get(key)
        if said is None:
            continue
        if not isinstance(said, str):
            raise ValueError(f"{where} ({name!r}): {key} is text; got "
                             f"{type(said).__name__}")
        if said:
            out[key] = said
    return out


def _unique_names(rows: List[Dict[str, Any]], where: str) -> None:
    seen: set = set()
    for row in rows:
        if row["name"] in seen:
            raise ValueError(f"{where}: {row['name']!r} names two rows; a name "
                             f"is unique within one set")
        seen.add(row["name"])


def normalise_customized(raw: Any, n_frames: int,
                         where: str = "Structure.customized"
                         ) -> Optional[Dict[str, Any]]:
    """The ``customized`` section in its one form (``model/structure.md``
    § 2.2d): ``None`` when it holds no row at all, otherwise ``{"rows": [...],
    "frames": [...]}`` with EXACTLY ``n_frames`` frame row sets -- every
    per-frame list of a structure has one entry per frame.

    ``frames`` may be left out of ``raw`` (no frame has a row); given, it
    must have one entry per frame.  Names are unique within a set, and a name
    the structure's rows hold may not also be a frame's."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{where} is an object {{rows, frames}}; got "
                         f"{type(raw).__name__}")
    stray = sorted(k for k in raw if k not in ("rows", "frames"))
    if stray:
        raise ValueError(f"{where} carries {stray!r}; its keys are "
                         f"['rows', 'frames']")
    rows_raw = raw.get("rows")
    rows_raw = [] if rows_raw is None else rows_raw
    if not isinstance(rows_raw, list):
        raise ValueError(f"{where}.rows is a list of rows; got "
                         f"{type(rows_raw).__name__}")
    frames_raw = raw.get("frames")
    if frames_raw is None:
        frames_raw = [[] for _ in range(n_frames)]
    if not isinstance(frames_raw, list) or len(frames_raw) != n_frames:
        got = (len(frames_raw) if isinstance(frames_raw, list)
               else type(frames_raw).__name__)
        raise ValueError(
            f"{where}.frames holds {got} frame row set(s); this structure has "
            f"{n_frames} frame(s), and there is one set per frame")
    rows = [_customized_row(r, f"{where}.rows[{k}]")
            for k, r in enumerate(rows_raw)]
    _unique_names(rows, f"{where}.rows")
    frames: List[List[Dict[str, Any]]] = []
    for f, frame_raw in enumerate(frames_raw):
        if not isinstance(frame_raw, list):
            raise ValueError(f"{where}.frames[{f}] is a list of rows; got "
                             f"{type(frame_raw).__name__}")
        frame_rows = [_customized_row(r, f"{where}.frames[{f}][{k}]")
                      for k, r in enumerate(frame_raw)]
        _unique_names(frame_rows, f"{where}.frames[{f}]")
        frames.append(frame_rows)
    both = sorted({r["name"] for frame_rows in frames for r in frame_rows}
                  & {r["name"] for r in rows})
    if both:
        raise ValueError(
            f"{where}: {both!r} is both a row of the structure and a row of a "
            f"frame; a name is the structure's or the frames', never both")
    if not rows and not any(frames):
        return None
    return {"rows": rows, "frames": frames}

#: Keys a sidecar ON DISK may carry that this build no longer stores.
#: ACCEPTED AND IGNORED wherever a stored payload is read -- never refused.
#:
#: An UNKNOWN key and a RETIRED one are different states, and the difference
#: is the user's file.  Unknown is a fact they believe they stored and this
#: build cannot honour, so it is named and refused.  Retired is a key this
#: project wrote and has stopped writing: the file is not wrong, it is older,
#: and refusing it would make a pair that opened before unopenable now for no
#: gain.
#:
#: RETIREMENT IS ONLY SAFE WHEN NOTHING IS LOST BY IGNORING THE KEY, and that
#: is shown, not assumed.  `pbc` qualifies: the boolean was always recomputed
#: from `axis_kind`, every sidecar at a readable schema version carries a real
#: `axis_kind`, and `Structure.pbc()` reproduces it on demand.  A key whose
#: fact lives nowhere else cannot be retired this way -- it needs a schema
#: bump and a reader that migrates it.
#:
#: SHARED, because THREE gates ask this question and they must give one
#: answer:
#:
#:   1. `parse.sidecars.molstruct.load_text` -- refuses stray keys while the
#:      payload is still whole, deliberately upstream of anything that
#:      applies it;
#:   2. `sidecars.molstruct.apply_to_structure` -- refuses them again before
#:      applying;
#:   3. `apply_metadata_dict` -- refuses them on the way onto a Structure.
#:
#: Gate 2 is not reachable from `StructureCodec.load`, because gate 1 answers
#: first -- but it takes a payload DIRECTLY, so a caller that builds one in
#: code reaches it.  A retirement written into some gates and not others
#: leaves the file readable by one door and refused by another, which is the
#: failure this shared constant exists to make impossible.
#:
#: ``cell_origin`` is the second, and it qualifies BY DECISION rather than by
#: derivation (user, 2026-09-25: "your option (a) is fine, and when files are
#: saved, make sure no old retired key is written again").  It stored a box
#: corner; placement is now the engine offset, the rule's unless the structure
#: states one (`model/structure-periodicity.md` § 6.0).  Ignoring it can lose
#: a corner a person typed by hand, who assigns it again on the Cell page --
#: chosen over migrating it because v9 cannot tell a typed corner from the
#: electrode builder's flush one, the placement TranSIESTA refused.
RETIRED_METADATA_KEYS = ("pbc", "cell_origin")

#: Identity keys a sidecar ON DISK may carry that this build no longer writes.
#: Same three gates as ``RETIRED_METADATA_KEYS``, for the same reason.
#:
#: ``title`` qualifies under the rule above because its fact lives in the
#: PAIRED GEOMETRY FILE -- it is the `.xyz` comment line and the PDB TITLE
#: record (``model/structure.md`` § 2.2c).  It remains a ``Structure`` field
#: and still rides ``to_dict`` / ``to_wire`` / ``replace``; what it is not is
#: a column of the sidecar.
RETIRED_IDENTITY_KEYS = ("title",)

#: The structural keys an extended-XYZ comment line carries.  The title is the
#: free text BEFORE the first of them (:meth:`Structure.from_xyz`).
_EXTXYZ_KEY = _re.compile(r"\b(?:Lattice|Properties|pbc)\s*=")

#: The per-atom IDENTITY columns -- the canonical-dict spellings
#: (``to_dict`` / ``from_dict`` carry them at the TOP level, beside
#: ``metadata``).  Persisted by the sidecar, and only when a column is REAL
#: (see ``identity_to_dict``): an xyz-born structure's synthesized
#: placeholders stay out of the file.
#:
#: ``title`` is NOT one of these.  The geometry file owns it
#: (``model/structure.md`` § 2.2c); it is in ``RETIRED_IDENTITY_KEYS``.
IDENTITY_FIELDS = ("atom_names", "residue_ids", "residue_names",
                   "chain_ids")


#: The per-side gap a DERIVED box uses on an isolated axis when the user set no
#: vacuum at all (``model/structure-periodicity.md`` § 6.1).
#:
#: IT IS A DEFAULT GAP, NOT A MINIMUM BOX LENGTH.  3 Å of empty space is 3 Å
#: whether the molecule is 2 Å across or 200, so every isolated axis gets it
#: when nothing was said.  Keying it off the resulting BOX length instead is
#: wrong twice over: a large molecule then gets no gap at all, and a typed
#: 1.0 Å gets overridden.
#:
#: It keeps a cell well-formed; it is NOT a claim of physical adequacy.  A
#: converged isolated-molecule run wants far more, and the validator says so
#: (``cell.vacuum_thin``: ≥ 8 Å per side neutral, ≥ 25 Å charged).
_DEFAULT_ISOLATED_VACUUM = 3.0


def _vacuum_from_stored(raw) -> Optional[Tuple[float, float, float]]:
    """Read a stored vacuum.  ``None`` means nobody chose one; ``[0,0,0]``
    means somebody chose no gap.  Those are different, and both are honoured.

    THERE IS NO LEGACY BRANCH, deliberately.  Reading an all-zero triple as
    UNSET would cost the ability to express a deliberate zero at all -- the
    third state this function exists to preserve.
    """
    if raw is None:
        return None
    values = tuple(float(x) for x in raw)
    if len(values) != 3:
        raise ValueError("Structure.vacuum must have exactly 3 entries")
    return values


@dataclass
class AtomChannel:
    """One named per-atom metadata channel (``model/structure-annotations.md`` § 2).

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
    """Remap every channel's atom indices through ``old_to_new``
    (``model/structure-annotations.md`` § 2.1).  Channels that end up empty are dropped."""
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

    Two transport-oriented attributes carry
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
    name.  ``frozen_atoms`` (:data:`FROZEN_LABEL`) marks atoms whose
    positions stay fixed during relaxations and Hessian builds;
    it is consumed by the vibration deck (relax + Hessian), the
    relaxation decks, and the transport lead gate (a lead's atoms must
    be held -- `transport/wizard.py`).  A reserved label is stored,
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
    # THE label store -- every label, including the reserved ones
    # (FROZEN_LABEL).  There is no second store: a reserved meaning costs a
    # NAME and one designated read (`frozen_atoms` below), and nothing else.
    # Giving a reserved label its own field instead buys two of everything --
    # two validators, two remaps on every atom-count change, two keys in the
    # saved file, two spellings on the wire -- and lets the two disagree.
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
    # ``cell`` is the (3, 3) matrix whose ROWS are the lattice vectors in
    # Angstrom (ASE convention), or None for a non-periodic molecule.
    # Per-axis periodicity is ``axis_kind`` below; the boolean view is
    # :meth:`pbc`, computed on demand and stored nowhere.
    #
    # THE SOURCE OF TRUTH FOR THE BOX: the transport/SIESTA emitters preserve
    # it verbatim rather than fabricating an orthorhombic vacuum box from atom
    # extents, which would put the atoms and the box in different frames.
    cell:          Optional[np.ndarray]            = None
    # Per-axis periodicity KIND (``model/structure-periodicity.md``) -- THE
    # periodicity field, and the only one.  Values: "periodic" (k-sampled
    # / tileable lattice), "isolated" (vacuum box), "transport"
    # (electrode-matched, semi-infinite).  None -> "periodic" on every axis
    # when a cell is stated, "isolated" otherwise; `transport` is never
    # guessed, a builder states it.
    #
    # The boolean view is :meth:`pbc`, an accessor for the two outside
    # formats that need one.
    axis_kind:     Optional[Tuple[str, str, str]] = None
    # Isolation padding (Å) on isolated axes -- the PER-SIDE vacuum gap.
    # (k-grid is NOT here: it's a reciprocal-space SAMPLING knob, a CALCULATION
    # parameter that lives on SiestaConfig, not the geometry.
    # structure-periodicity.md.)
    vacuum:        Optional[Tuple[float, float, float]] = None
    # THE OFFSET THIS STRUCTURE STATES (Angstrom), or None -- then the rule
    # computes it (`model/structure-periodicity.md` § 6.0, *A stated
    # offset*).  Two things state one: an origin the person assigns, P, stored
    # as -P; and coordinates that came from an engine, which state 0 -- their
    # origin is the engine's, set together with them.  ``cell.to_engine``
    # applies it as it stands, and nothing derives it; the sidecar carries it
    # (``metadata_to_dict``), and ``None`` there means the rule.
    engine_offset: Optional[np.ndarray]            = None
    # Extensible per-atom annotations (model/structure-annotations.md).  Holds
    # channels BEYOND the labels (regions -> tag channels, reserved ones
    # included), e.g. future per-atom value channels (charge / spin /
    # basis-override).  The unified read API is ``channels()`` /
    # ``get_channel()`` / ``atom_annotations()``, which present the labels
    # and these together.
    annotations:   Dict[str, AtomChannel] = field(default_factory=dict)
    #: Named values the structure carries (``model/structure.md`` § 2.2d):
    #: the structure's rows, and one row set per frame -- what a frame IS
    #: (a displacement's mode, its weight in an average).  STRUCTURAL, in
    #: METADATA_FIELDS: the sidecar carries it and an edit of it is an edit.
    #: ``None`` when it holds no row, else ``{"rows": [...], "frames":
    #: [...]}`` with one set per frame (:func:`normalise_customized`).
    #: Written through :meth:`set_customized` / :meth:`remove_customized`
    #: and read through :meth:`customized_value` / :meth:`customized_rows`;
    #: nothing else names a key of it.
    customized:    Optional[Dict[str, Any]] = None
    #: METADATA (model/structure.md § 2.2a) -- what the MolView Metadata
    #: pane shows -- that is NOT part of the structure: no emitter reads
    #: it, it never enters `structure_hash`, and the read-only gate does
    #: not apply to it.  Those three are why it sits outside
    #: METADATA_FIELDS, which is the strictly-enumerated STRUCTURAL block
    #: (hash input, gate-controlled, unknown keys refused); `info` is the
    #: open store beside it, with to_dict/from_dict as its door, and it
    #: is in the sidecar from schema 9.
    #:
    #: IT TRAVELS AND A STRIP IS EXPLICIT.  Every seam that derives one
    #: Structure from another carries it -- `_carry_nonatom`, `replace`,
    #: `copy`, `concat`, the ops, the codecs.  A rebuild that simply did
    #: not list the field is a defect: a vanished contract cannot be told
    #: apart from one that was never recorded.
    #:
    #: **IT IS A NAMESPACE OF CLUSTERS, ONE PER SUBSYSTEM.**  A top-level key is a cluster name and its value is
    #: that subsystem's own metadata; nothing else writes inside someone
    #: else's cluster.  `calculation` is the one this project ships -- the
    #: recorded contract a finished run leaves on the pair -- and any
    #: further non-structural metadata goes beside it under its own name
    #: rather than being flattened in with it.
    #:
    #: That shape is the browser's too, where `molview.data.info` offers
    #: `set(key, value)` / `remove(key)`; :meth:`set_info`,
    #: :meth:`drop_info` and :meth:`apply_info_dict` are the Python half.
    info:          Dict[str, Any] = field(default_factory=dict)
    #: THE FRAMES of a frame set (``model/structure.md`` § 2.2e): shape
    #: ``(F, N, 3)`` when ``F > 1``, and ``positions`` IS ``frames[0]`` -- one
    #: coordinate store, never two copies.  ``None`` for one frame.  Read-only:
    #: a frame set is a data set and is never edited in place; the frame
    #: doors (:meth:`with_frames`, :meth:`take`) make a new one.
    frames:        Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        n = len(self.positions)
        if len(self.elements) != n:
            raise ValueError(
                f"elements ({len(self.elements)}) does not match positions ({n})"
            )
        if self.frames is not None:
            frames = np.asarray(self.frames, dtype=float)
            if frames.ndim != 3 or len(frames) == 0 \
                    or frames.shape[1:] != (n, 3):
                raise ValueError(
                    f"Structure.frames is (F, {n}, 3) -- every frame carries "
                    f"the structure's {n} atoms; got shape {frames.shape}")
            if not np.array_equal(frames[0], self.positions):
                raise ValueError(
                    "Structure.positions is frame 0 of Structure.frames -- "
                    "one coordinate store (model/structure.md § 2.2e)")
            if len(frames) == 1:
                self.frames = None
            else:
                if frames.flags.writeable:
                    frames = frames.copy()
                    frames.flags.writeable = False
                self.frames = frames
                self.positions = frames[0]
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
        # 3x3 of finite floats.  What a missing ``axis_kind`` defaults to is
        # decided below, off the cell's presence -- "a lattice implies
        # periodicity", isolated without one.
        if self.cell is not None:
            cell = np.asarray(self.cell, dtype=float)
            if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
                raise ValueError(
                    f"Structure.cell must be a 3x3 matrix of finite "
                    f"floats (lattice vectors as rows, Angstrom); got "
                    f"shape {cell.shape}"
                )
            # READING DOES NOT JUDGE (``model/structure-periodicity.md``
            # § 8.2).  A singular or left-handed box LOADS.  Refusing it here
            # would make a pair whose sidecar holds one impossible to open,
            # and the Cell page is the one place a box can be corrected -- so
            # the only ways out would be hand-editing the `.molstruct.json`
            # outside molbuilder, or deleting it and losing the labels with
            # it.  § 6.1a puts `cell.no_volume` at WARNING on load and error
            # only at generate, and the doors that ACT enforce that already:
            # `periodicity_gate._refuse_on_error` rejects the value you type
            # and `validation.report` refuses to emit.  Every reader that
            # inverts the cell answers `None` rather than raising
            # (`cell._fractional`).
            self.cell = cell
        # ONE PERIODICITY FIELD.  ``axis_kind`` is it, and there is nothing to
        # reconcile: a second boolean field would be redundant by construction,
        # because ``pbc`` is ``axis_kind`` with `transport` and
        # `periodic` both flattened to True, so it never holds a fact
        # ``axis_kind`` does not.
        #
        # The boolean view is :meth:`pbc`, which is an INTEROP
        # accessor and nothing more -- ASE takes booleans and the extxyz
        # header writes `pbc="T T F"`.  Nothing inside molbuilder reads it.
        _KINDS = ("periodic", "isolated", "transport")
        if self.axis_kind is None:
            # A stated cell means a lattice; no cell means a vacuum box.
            # `transport` is never guessed -- a builder says it.
            self.axis_kind = (("periodic",) * 3 if self.cell is not None
                              else ("isolated",) * 3)
        else:
            ak = tuple(str(k) for k in self.axis_kind)
            if len(ak) != 3 or any(k not in _KINDS for k in ak):
                raise ValueError(
                    f"Structure.axis_kind must be exactly 3 of {_KINDS}; "
                    f"got {self.axis_kind!r}"
                )
            self.axis_kind = ak
        # A stated offset is kept whether or not the structure types a cell:
        # an engine's output states 0 against the cell the ENGINE used, which
        # the next deck is handed as its box.
        if self.engine_offset is not None:
            eo = np.asarray(self.engine_offset, dtype=float).reshape(3)
            if not np.all(np.isfinite(eo)):
                raise ValueError(
                    "Structure.engine_offset must be 3 finite floats (Angstrom)")
            self.engine_offset = eo
        # Shape vacuum (per-side gap).  ``None`` MEANS THE STRUCTURE SAYS
        # NOTHING -- the same "unset" its siblings (cell, engine_offset,
        # axis_kind) have always had, and the state the whole regime model
        # needs in order to tell "I want no gap" from "I never said" (see
        # ``model/structure-periodicity.md`` § 6.1).  Defaulting this field to
        # (0, 0, 0) collapses the two states into one value, and no rule can
        # then branch on the difference.
        if self.vacuum is not None:
            self.vacuum = tuple(float(v) for v in self.vacuum)
            if len(self.vacuum) != 3:
                raise ValueError("Structure.vacuum must have exactly 3 entries")

        # Validate the labels and the extra channels.
        self._validate_regions(n)
        self._validate_annotations(n)
        # The named values, one row set per frame.
        self.customized = normalise_customized(self.customized, self.n_frames)

    def pbc(self) -> Tuple[bool, bool, bool]:
        """Per-axis periodicity as BOOLEANS — an interop accessor, not state.

        ``periodic`` and ``transport`` are both True; ``isolated`` is False.

        **Nothing inside molbuilder should call this.**  The mapping is onto,
        not one-to-one, so the answer cannot tell a device axis from a bulk
        one — ask :attr:`axis_kind`, which says which it is.  Reading the
        boolean instead is a measured defect, not a hypothetical:
        ``transiesta._lattice_block`` labelled every **transport** axis
        ``periodic`` in the deck it wrote, and its "the transport axis is not
        periodic" warning could never fire, both because a boolean cannot
        express the distinction it was branching on.

        It exists for the two formats outside this project that require the
        boolean form and have no richer one:

          * ASE — ``Atoms(pbc=…)`` (:meth:`to_ase`);
          * extended XYZ — the ``pbc="T T F"`` header (:meth:`to_extxyz`).

        Not a stored field: ``axis_kind`` contains this and more, so storing
        the boolean view as well would be two spellings of one fact.
        """
        return tuple(k != "isolated" for k in
                     (self.axis_kind or ("isolated",) * 3))

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
        clearance on both sides.  This is the box the
        SIESTA deck states, so the displayed cell is the one the calculation uses.  Where the box sits
        is the engine offset's (``cell.engine_offset``, § 6.0).

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

    def effective_vacuum(self) -> Tuple[float, float, float]:
        """The per-side vacuum the DERIVED box actually uses (§ 6.1).

        TWO STATES, AND ONLY TWO.

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

        IT IS NOT A FLOOR ON THE BOX.  A rule of the form ``extent + 2·vacuum
        < 3`` asks about the box rather than about what the user wanted: it
        raises a typed 1.0 Å to 3.0 — overriding a stated value — and it
        leaves a large molecule with NO gap at all, because its box is already
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

        Nothing is raised here: there is no stored value to raise, only an
        absent one to fill in."""
        if self.vacuum is not None:
            return []
        eff = self.effective_vacuum()
        return [i for i in range(3) if eff[i] > 0.0]

    # ------------------------------------------------------------------ #
    #  Sidecar-metadata contract -- the ONE get/set                       #
    #  (`model/structure.md` § 2.2)                                     #
    # ------------------------------------------------------------------ #
    # The persisted ``.molstruct.json`` sidecar IS the serialization of this
    # dataclass's metadata.  These TWO methods are the SINGLE place the
    # metadata field set is enumerated: ``metadata_to_dict`` (struct -> dict)
    # and ``apply_metadata_dict`` (dict -> struct).  The sidecar read + write
    # modules and the workspace codec ALL route through them, so the write and
    # read paths physically cannot drift a field.  Add a metadata field = add
    # it to the dataclass + these two methods, and nowhere else.
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
            # one store.
            "regions":      {k: list(v)
                             for k, v in (self.regions or {}).items()},
            "cell":         self.cell.tolist() if self.cell is not None else None,
            # The STATED offset (§ 6.0), None meaning the rule's.  Never
            # `cell_origin`: that key is retired and no writer emits it again.
            "engine_offset": (self.engine_offset.tolist()
                              if self.engine_offset is not None else None),
            "axis_kind":    (list(self.axis_kind)
                             if self.axis_kind is not None else None),
            "vacuum":       ([float(x) for x in self.vacuum]
                             if self.vacuum is not None else None),
            "annotations":  annotations_to_json(self.annotations),
            # None when no row, else every row and one set per frame (§ 2.2d).
            "customized":   _copy.deepcopy(self.customized),
        }

    def identity_to_dict(self) -> dict:
        """The identity columns that are REAL -- {} when everything is the
        placeholder ``__post_init__`` synthesizes (names = elements, resid 1,
        residue ``MOL``, chain ``A``, empty title).  The default test lives
        HERE, beside the synthesis it mirrors, so the two cannot drift: a
        writer that persisted the placeholders would make every xyz-born
        sidecar claim an identity nobody stated.  Column-granular -- a format
        that names residues but not chains persists exactly what it said."""
        n = len(self.elements)
        out: dict = {}
        # NO ``title`` HERE.  It belongs to the geometry file -- it IS the
        # `.xyz` comment line and the PDB TITLE record
        # (``model/structure.md`` § 2.2c).  Persisting it here gives one fact
        # two homes with no authority between them: a hand-edited comment line
        # is then silently overridden by this copy, and a non-empty title
        # alone makes ``keep_sidecar`` true, so a `.molstruct.json` exists
        # to hold one string already in the file beside it.
        if self.atom_names is not None \
                and list(self.atom_names) != list(self.elements):
            out["atom_names"] = list(self.atom_names)
        if self.residue_ids is not None \
                and list(self.residue_ids) != [1] * n:
            out["residue_ids"] = [int(v) for v in self.residue_ids]
        if self.residue_names is not None \
                and list(self.residue_names) != ["MOL"] * n:
            out["residue_names"] = list(self.residue_names)
        if self.chain_ids is not None \
                and list(self.chain_ids) != ["A"] * n:
            out["chain_ids"] = list(self.chain_ids)
        return out

    def apply_metadata_dict(self, data: Optional[dict]) -> None:
        """Apply a sidecar metadata dict onto this structure IN PLACE, then
        re-run the dataclass's own reconciliation + validation
        (``__post_init__``) so there is ONE validator.

        Full-REPLACE semantics: an absent key resets that field to its default
        (absent cell -> non-periodic; absent regions -> none), matching a v3
        back-read.  Raises ``ValueError`` on any invalid field (bad cell,
        out-of-range index, ...), sourced from the same invariants a freshly
        constructed Structure enforces.

        **An unknown key is REFUSED, a RETIRED one is ignored.**  Those are
        different states and the difference is the user's file.  An unknown
        key is a fact they believe they stored and this build cannot honour,
        so saying nothing would be a silent loss -- it is named and refused.
        A retired key is one THIS project used to write and has since stopped:
        the file is not wrong, it is older, and refusing it would make a pair
        that opened yesterday unopenable today for no gain.
        ``RETIRED_METADATA_KEYS`` (module scope, because two more gates ask
        the same question upstream of this one and must agree) is that list.

        Retiring a key is only safe when nothing is lost by ignoring it, and
        that has to be shown rather than assumed.  For ``pbc`` it is: the
        boolean was always recomputed from ``axis_kind``, every sidecar at a
        readable schema version carries a real ``axis_kind`` (``__post_init__``
        has always set one), and :meth:`pbc` reproduces the value on demand.
        A key whose fact lives nowhere else cannot be retired this way -- it
        needs a schema bump and a reader that migrates it."""
        data = data or {}
        unknown = [k for k in data
                   if k not in METADATA_FIELDS
                   and k not in RETIRED_METADATA_KEYS]
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
        self.engine_offset = (np.asarray(data["engine_offset"], dtype=float)
                              if data.get("engine_offset") is not None
                              else None)
        self.axis_kind    = (tuple(str(k) for k in data["axis_kind"])
                             if data.get("axis_kind") is not None else None)
        self.vacuum       = _vacuum_from_stored(data.get("vacuum"))
        self.annotations  = annotations_from_json(data.get("annotations"))
        self.customized   = data.get("customized")
        # Re-run the dataclass invariants ONCE: cell 3x3 of finite floats, the
        # ``axis_kind`` default and value check (nothing to reconcile: there
        # is one periodicity field), a stated offset of three finite
        # floats, region/frozen/annotation indices in range, and one
        # ``customized`` row set per frame this structure holds.
        self.__post_init__()

    # ------------------------------------------------------------------ #
    #  Whole-structure codec -- the ONE (de)serialiser (structure-        #
    #  authority.md § 3.1).  ``to_dict``/``from_dict`` carry coordinates  #
    #  + per-atom columns + the full metadata block (delegated to         #
    #  ``metadata_to_dict``/``apply_metadata_dict``) as a single          #
    #  round-trippable dict: ``Structure.from_dict(s.to_dict())`` == s.   #
    #  Outside this class NOBODY assembles or picks apart a structure     #
    #  dict, so no hand-rolled repack can drop a field.  Add a field =    #
    #  add it here + the two metadata methods, nowhere else.              #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """The ONE canonical serialiser: everything needed to reconstruct this
        Structure -- coordinates, per-atom identity columns, AND the full
        metadata field set (nested under ``metadata`` via
        :meth:`metadata_to_dict`).  Loss-free + filesystem-free; the round-trip
        unit the persistence + sidecar + CLI layers store.  Inverse:
        :meth:`from_dict`.

        A frame set carries ``frames`` IN PLACE OF ``positions`` -- one key or
        the other, never both, because ``positions`` is frame 0
        (``model/structure.md`` § 2.1, § 2.2e)."""
        coordinates = ({"frames": self.frames.tolist()} if self.n_frames > 1
                       else {"positions": self.positions.tolist()})
        return {
            "title":         self.title or "",
            "elements":      list(self.elements),
            **coordinates,
            "atom_names":    list(self.atom_names)    if self.atom_names    else [],
            "residue_ids":   list(self.residue_ids)   if self.residue_ids   else [],
            "residue_names": list(self.residue_names) if self.residue_names else [],
            "chain_ids":     list(self.chain_ids)     if self.chain_ids     else [],
            "metadata":      self.metadata_to_dict(),
            "info":          dict(self.info),
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
        has_frames = data.get("frames") is not None
        if has_frames and data.get("positions") is not None:
            raise ValueError(
                "Structure.from_dict: the dict carries both 'positions' and "
                "'frames'; a frame set states 'frames' alone, its frame 0 "
                "being the positions (model/structure.md § 2.1)")
        frames = (np.asarray(data["frames"], dtype=float) if has_frames
                  else None)
        if frames is not None and (frames.ndim != 3 or len(frames) == 0):
            raise ValueError(
                f"Structure.from_dict: 'frames' is a list of frames, each a "
                f"list of [x, y, z]; got shape {frames.shape}")
        s = cls(
            elements=list(data["elements"]),
            positions=(frames[0] if frames is not None
                       else np.asarray(data["positions"], dtype=float)),
            frames=frames,
            atom_names=list(data.get("atom_names")    or []) or None,
            residue_ids=list(data.get("residue_ids")   or []) or None,
            residue_names=list(data.get("residue_names") or []) or None,
            chain_ids=list(data.get("chain_ids")     or []) or None,
            title=data.get("title", "") or "",
        )
        # Full-replace + revalidate the metadata block through the ONE codec.
        s.apply_metadata_dict(data.get("metadata"))
        # THROUGH THE SAME DOOR AS EVERY OTHER WRITER: the browser reaches
        # this one -- `_shared.struct_from_body` builds every edited structure
        # through here -- so `info: {"": ...}` posted to any modify route is
        # refused here.  One door, one set of rules, whichever direction the
        # store arrives from.
        if data.get("info") is not None:
            s.apply_info_dict(data.get("info"))
        return s

    def to_wire(self) -> dict:
        """The read-only server->client view (``model/structure.md`` § 2.1 + § 4):
        the metadata-bearing portion of the wire response, assembled by
        Structure so no blueprint enumerates a field.  It is the identity
        columns + the FULL ``periodicity`` block (the raw cell and stated
        offset PLUS the server-resolved ``resolved_cell`` and ``box_corner``,
        computed HERE through the one resolver and the one placement rule, so
        they can never drift or drop) + ``annotations``.

        The web layer composes this with its own render/validation concerns
        (the flat ``atoms`` list, ``issues``, ``text``, ``extra``); it must NOT
        re-list any periodicity / metadata field.  Not round-tripped by
        :meth:`from_dict` -- ``resolved_*`` are derived, read-only fields."""
        # The ONE resolver (structure-periodicity.md §§ 4, 6), run once, here.
        # A periodic axis without a lattice raises -> no resolved box (None).
        try:
            _rc = self.resolve_cell()
            resolved_cell = _rc.tolist() if _rc is not None else None
        except Exception:  # noqa: BLE001
            resolved_cell = None
        # WHERE THE BOX IS DRAWN: at -engine_offset of these coordinates
        # (`model/structure-periodicity.md` § 6.0) -- the offset the structure
        # states, or the rule's.  Worked out here and never in the browser,
        # which draws the coordinates it holds and moves no atom.
        # A STATED offset is the corner whatever the atoms: an origin the
        # person assigned stays theirs with every atom deleted, and sending no
        # corner would show it as Automatic -- and clear it on the next Apply.
        box_corner = None
        if self.engine_offset is not None:
            box_corner = (-np.asarray(self.engine_offset, dtype=float)).tolist()
        elif resolved_cell is not None and self.n_atoms:
            try:
                from .cell import engine_offset as _engine_offset
                box_corner = (-_engine_offset(self)).tolist()
            except Exception:  # noqa: BLE001
                box_corner = None
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
                # The offset the structure STATES, as stored -- None means the
                # rule places it.  Raw, so a client echoes it back unchanged.
                "engine_offset":        (self.engine_offset.tolist()
                                         if self.engine_offset is not None else None),
                "resolved_cell":        resolved_cell,
                "box_corner":           box_corner,
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
            # The `info` store (``model/structure.md`` § 2.2a): sent whole
            # so the viewer's Metadata pane shows exactly what the pair
            # will carry.
            "info":          dict(self.info),
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
    #  Unified annotation channels (model/structure-annotations.md § 2)  #
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

    def replace(self, **changes) -> "Structure":
        """THE door for deriving a modified copy — use instead of
        ``dataclasses.replace``.

        Why this exists.  ``frozen_atoms`` is both an ``__init__`` field and a
        derived READ of ``regions[FROZEN_LABEL]``, which is what makes
        ``Structure(..., frozen_atoms=[...])`` reach the one place that spells
        the label.  ``dataclasses.replace`` re-passes every field by reading it
        off the instance — and reading ``frozen_atoms`` goes through the
        property, which returns a list and never ``None``.  So the setter's
        "``None`` says nothing about it" contract is unreachable from
        ``replace``, and an explicit new ``regions`` silently gets the OLD
        frozen set stamped back into it:

            replace(s, regions={"electrode_L": [1]})
            -> {"electrode_L": [1], "frozen_atoms": [0]}      # not asked for

        Here, a caller who states ``regions`` states the WHOLE store, so the
        frozen door is not re-passed. A caller who states ``frozen_atoms`` (with
        or without ``regions``) gets exactly what they asked for.

        **``dataclasses.replace`` NEVER routes here, on any version.**
        `dataclasses.replace` ends `return obj.__class__(**changes)` and the
        word ``__replace__`` does not appear in the function at all.  What
        3.13 added is `copy.replace`, a DIFFERENT helper, and that one does
        dispatch through the hook -- which is the only reason the alias below
        is worth keeping.  So `dataclasses.replace(struct, ...)` carries the
        trap on every interpreter, and this method is the only correct door,
        with no upgrade that changes it.

        IT IS ``copy()`` PLUS THE CHANGES, and that is the second reason to
        use it.  ``dataclasses.replace`` re-passes the mutable fields BY
        REFERENCE: the derived structure shared ``positions``, ``cell`` and
        the ``info`` dict with its source, so writing to one wrote to the
        other — measured, including through ``info.calculation`` -- and
        enumerating fields to copy them is how fields come to be forgotten
        (§ 2.2a).  Deriving through here copies, so a caller states
        only what CHANGES and nothing it did not name can alias or vanish.

        ``frozen_atoms`` is never re-passed at all: a copied ``regions`` is the
        whole label store and already carries the reserved label, so the trap
        above is gone by construction rather than by a special case.

        ON A FRAME SET the coordinates and the atoms change only together with
        the frames (``model/structure.md`` § 2.2e): ``positions`` or
        ``elements`` without ``frames`` is refused, :data:`FRAME_SET_NOT_EDITED`.
        The frame doors -- :meth:`frame_at`, :meth:`with_frames`, :meth:`take`
        -- pass ``frames``, and ``positions`` follows them as their frame 0.
        """
        if self.n_frames > 1 and "frames" not in changes \
                and ("positions" in changes or "elements" in changes):
            raise ValueError(FRAME_SET_NOT_EDITED)
        kw = {
            "elements":      list(self.elements),
            "positions":     self.positions.copy(),
            "atom_names":    list(self.atom_names),
            "residue_ids":   list(self.residue_ids),
            "residue_names": list(self.residue_names),
            "chain_ids":     list(self.chain_ids),
            "title":         self.title,
            "regions":       {k: list(v) for k, v in self.regions.items()},
            "annotations":   copy_annotations(self.annotations),
            # Read-only, so shared rather than copied: nothing writes into it.
            "frames":        self.frames,
            **self._nonatom(),
        }
        if changes.get("frames") is not None and "positions" not in changes:
            kw["positions"] = np.asarray(changes["frames"], dtype=float)[0]
        kw.update(changes)
        return type(self)(**kw)

    #: `copy.replace` (3.13+) dispatches here; `dataclasses.replace` does
    #: NOT, on any version -- it calls `obj.__class__(**changes)` directly.
    #: Inert on 3.12, which has no `copy.replace`.
    __replace__ = replace

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

    # ------------------------------------------------------------------ #
    #  Frames (model/structure.md § 2.2e) -- a frame set is a data set    #
    # ------------------------------------------------------------------ #

    @property
    def n_frames(self) -> int:
        """How many frames this structure holds -- 1 unless it is a set."""
        return 1 if self.frames is None else len(self.frames)

    def _frame_index(self, frame: Any) -> Optional[int]:
        """``None`` (the structure's own rows) or a frame index in range,
        refused naming how many frames there are."""
        if frame is None:
            return None
        if isinstance(frame, (bool, np.bool_)) \
                or not isinstance(frame, (int, np.integer)):
            raise TypeError(f"a frame is an index, 0-based; got "
                            f"{type(frame).__name__}")
        f = int(frame)
        if not 0 <= f < self.n_frames:
            raise ValueError(
                f"frame {f} is outside this structure's {self.n_frames} "
                f"frame(s): 0 … {self.n_frames - 1}")
        return f

    def frame_at(self, i: int) -> "Structure":
        """Frame ``i`` as a one-frame structure: the shared facts, ``info``,
        the structure's ``customized`` rows, and frame ``i``'s rows as its own
        frame 0 (``model/structure.md`` § 2.2e).

        ONE ORIGIN FOR THE SET (``structure-periodicity.md`` § 6.0, *A frame
        set gets one offset*): frame 0 is as it is, and a later frame states
        frame 0's offset -- the stated one, or the rule's computed on frame 0
        -- so every frame reaches the engine through ``cell.to_engine`` with
        the same one and no electrode atom moves between frames.  Where the
        rule cannot place frame 0 -- no cell to place it in -- there is no
        offset to state, and the door that acts refuses the cell itself."""
        f = self._frame_index(i)
        if self.n_frames == 1:
            return self.copy()
        customized = None
        if self.customized is not None:
            customized = {"rows": _copy.deepcopy(self.customized["rows"]),
                          "frames": [_copy.deepcopy(
                              self.customized["frames"][f])]}
        offset = (None if self.engine_offset is None
                  else np.asarray(self.engine_offset, dtype=float).copy())
        if f > 0 and offset is None:
            from .cell import engine_offset as _engine_offset
            try:
                # `positions` IS frame 0, so the rule placing this structure
                # places frame 0.
                offset = _engine_offset(self)
            except ValueError:
                offset = None
        return self.replace(frames=None, positions=self.frames[f].copy(),
                            customized=customized, engine_offset=offset)

    def with_frames(self, coordinates: Any,
                    frame_rows: Optional[Sequence[Sequence[dict]]] = None
                    ) -> "Structure":
        """A frame set of this structure's atoms at ``coordinates`` -- shape
        ``(F, N, 3)``, frame 0 the new ``positions`` -- carrying every shared
        fact, ``info`` and the structure's ``customized`` rows; each frame's
        rows are ``frame_rows[f]``, none when it is left out.  THE door a
        frame set is built through (``model/structure.md`` § 2.2e); one frame
        makes a one-frame structure."""
        frames = np.asarray(coordinates, dtype=float)
        if frames.ndim != 3 or len(frames) == 0 \
                or frames.shape[1:] != (self.n_atoms, 3):
            raise ValueError(
                f"with_frames: the frames are (F, {self.n_atoms}, 3) -- every "
                f"frame this structure's {self.n_atoms} atoms in its order; "
                f"got shape {frames.shape}")
        F = len(frames)
        sets = ([[] for _ in range(F)] if frame_rows is None
                else [list(rows) for rows in frame_rows])
        if len(sets) != F:
            raise ValueError(
                f"with_frames: {len(sets)} frame row set(s) for {F} frame(s); "
                f"there is one set per frame")
        customized = {"rows": self.customized_rows(), "frames": sets}
        return self.replace(frames=(frames if F > 1 else None),
                            positions=frames[0].copy(),
                            customized=customized)

    def take(self, order: Sequence[int]) -> "Structure":
        """The atoms reordered: ``order[j]`` is the index of the atom that
        sits at ``j``.  Every per-atom field moves with its atom -- the
        identity columns, the labels, the channels -- and so does every
        frame's coordinates (``model/structure.md`` § 2.2e).  A reorder is
        not an edit; the transport sort goes through here."""
        n = self.n_atoms
        idx = [int(i) for i in order]
        if sorted(idx) != list(range(n)):
            raise ValueError(
                f"take: the order names each of the {n} atoms once; got "
                f"{len(idx)} indices")
        old_to_new = {old: new for new, old in enumerate(idx)}
        changes: Dict[str, Any] = dict(
            elements=[self.elements[i] for i in idx],
            atom_names=[self.atom_names[i] for i in idx],
            residue_ids=[self.residue_ids[i] for i in idx],
            residue_names=[self.residue_names[i] for i in idx],
            chain_ids=[self.chain_ids[i] for i in idx],
            regions={label: sorted(old_to_new[i] for i in members)
                     for label, members in self.regions.items()},
            annotations=remap_annotations(self.annotations, old_to_new),
        )
        if self.n_frames > 1:
            changes["frames"] = self.frames[:, idx, :]
        else:
            changes["positions"] = self.positions[idx].copy()
        return self.replace(**changes)

    # ------------------------------------------------------------------ #
    #  The `customized` section (model/structure.md § 2.2d)              #
    # ------------------------------------------------------------------ #

    def customized_rows(self, frame: Optional[int] = None
                        ) -> List[Dict[str, Any]]:
        """A copy of the structure's rows (``frame`` omitted) or of frame
        ``frame``'s -- ``[]`` when it has none."""
        f = self._frame_index(frame)
        if self.customized is None:
            return []
        rows = (self.customized["rows"] if f is None
                else self.customized["frames"][f])
        return _copy.deepcopy(rows)

    def customized_value(self, name: str, frame: Optional[int] = None) -> Any:
        """The value of row ``name`` in that one set -- the structure's, or
        frame ``frame``'s -- or ``None`` when the set has no such row.  Never
        a fall-back from one set to the other."""
        for row in self.customized_rows(frame):
            if row["name"] == name:
                return row["value"]
        return None

    def set_customized(self, name: str, value: Any, unit: Optional[str] = None,
                       note: Optional[str] = None,
                       frame: Optional[int] = None) -> None:
        """Write one row, in place: of the structure (``frame`` omitted) or of
        frame ``frame``.  A row of that name in that set is replaced where it
        stands; otherwise the row is added at the end."""
        f = self._frame_index(frame)
        row = _customized_row({"name": name, "value": value, "unit": unit,
                               "note": note}, "Structure.set_customized")
        section = (_copy.deepcopy(self.customized)
                   or {"rows": [], "frames": [[] for _ in range(self.n_frames)]})
        rows = section["rows"] if f is None else section["frames"][f]
        for k, old in enumerate(rows):
            if old["name"] == name:
                rows[k] = row
                break
        else:
            rows.append(row)
        self.customized = normalise_customized(section, self.n_frames)

    def remove_customized(self, name: str,
                          frame: Optional[int] = None) -> bool:
        """Remove one row, in place.  ``True`` if it was there."""
        f = self._frame_index(frame)
        if self.customized is None:
            return False
        section = _copy.deepcopy(self.customized)
        rows = section["rows"] if f is None else section["frames"][f]
        kept = [row for row in rows if row["name"] != name]
        if len(kept) == len(rows):
            return False
        rows[:] = kept
        self.customized = normalise_customized(section, self.n_frames)
        return True

    def geometry_lines(self) -> List[str]:
        """One canonical line per atom, in THIS structure's order: the
        element and the position rounded to a millionth of an ångström,
        negative zero folded into zero (`model/parse.md` § 5b.1).

        The rounding is what lets the run's parser (SIESTA prints eight
        decimals) and a pair written by the codec (six) agree on every line;
        the sign fold is what lets a coordinate that crossed the browser's
        JSON, where ``-0`` becomes ``0``, agree too.  A record about atoms
        names them by these lines (`held_atom_keys`), never by an index,
        because a deck's copy may reorder them.
        """
        pos = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        return [f"{str(el)} " + " ".join(f"{round(float(v), 6) + 0.0:.6f}"
                                         for v in xyz)
                for el, xyz in zip(self.elements, pos)]

    def geometry_fingerprint(self) -> str:
        """``sha256:`` over the SORTED :meth:`geometry_lines` -- what a
        record about THESE coordinates pins itself to (`model/parse.md`
        § 5b.1), the same whatever order the atoms are listed in: the
        vibration kind's deck is written from a copy sorted held-first, and
        the gate judges that copy.

        Not :attr:`structure_hash` (the sidecar's pin over the written
        document's bytes, which a formatting choice changes) and not
        `sidecars.spectra.structure_hash_text` (the artifact's pin over the
        input order with the job's label in it): this is a fact about the
        geometry alone.  Labels, cell and metadata are left out on purpose;
        a rigid shift is NOT folded in: a record is judged against the
        coordinates the person holds (``validate``'s ``design``), never the
        placed copy a deck writes, and a structure saved from a run states
        its engine's origin, so its deck moves nothing (`model/
        structure-periodicity.md` § 6.0).
        """
        import hashlib as _hashlib
        return "sha256:" + _hashlib.sha256(
            "\n".join(sorted(self.geometry_lines())).encode("utf-8")).hexdigest()

    @property
    def formula(self) -> str:
        """The element counts as one string — ``C6H4S2``, elements sorted.

        A **witness**, in `engines/stages.md § 6.3`'s sense: ``task.json``
        records it beside the structure's path so a description opened against
        a structure that has since changed can *say so* rather than silently
        building a different calculation under the same id.  It is also half of
        the run id (`run-identity.md § 2.0a`).

        Sorted alphabetically rather than by Hill convention: this is compared
        for equality and normalised into an identifier, never read as chemistry.
        One property, so the description and ``summary()`` cannot disagree
        about what this structure is.
        """
        from collections import Counter
        return "".join(f"{el}{n}" if n > 1 else el
                       for el, n in sorted(Counter(self.elements).items()))

    def summary(self) -> str:
        return (
            f"<Structure {self.title!r}: "
            f"{self.n_atoms} atoms, {self.n_residues} residues, "
            f"formula {self.formula}>"
        )

    def __repr__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------ #
    #  Input: XYZ                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_xyz(cls, text: str, *,
                 title: Optional[str] = None) -> "Structure":
        """Parse a Structure from XYZ TEXT.  Not a path -- see
        :func:`_require_text`; ``StructureCodec().load(path)`` reads files.

        THE PARSE IS ASE'S, NOT OURS (``ase.io.read(..., format="extxyz")``).
        ASE's extended-XYZ reader is a superset reader: it handles the plain
        xmol layout AND the ``Lattice="…" Properties=… pbc="…"`` comment line,
        it canonicalises an external tool's ``FE`` / ``ZN`` to ``Fe`` / ``Zn``,
        and it reads every frame of a multi-frame file.  ASE is a declared
        dependency of this project **for exactly this** (``pyproject.toml``:
        *"XYZ I/O + atomic-number table"*).

        XYZ stores no atom names / residues, so all atoms come back tagged as
        residue 1 ("MOL", chain "A") and atom names default to the element
        symbol.

        EVERY FRAME, AS ONE FRAME SET (``model/structure.md`` § 2.2e, § 2.3):
        a document of several frames is one structure holding them, and its
        frames must BE one -- a frame whose atom count, species, atom order,
        ``Lattice=`` or ``pbc=`` differs from frame 0's is refused, naming it.
        Which frame a caller gets is the codec's choice (``load``), never the
        reader's.

        :param title: overrides the comment line.
        """
        text = _require_text(text, "xyz")
        # THE TITLE IS OURS, and it is the one thing read here rather than
        # parsed by ASE.  ASE's extended-XYZ reader treats the comment line as
        # `key=value` pairs, so a human comment -- "water molecule" -- comes
        # back as ``{'water': True, 'molecule': True}`` and the sentence is
        # gone.  The line is taken verbatim for this one field, which is
        # metadata this class owns, not part of reading the structure.
        lines = text.splitlines()
        comment = lines[1].strip() if len(lines) >= 2 else ""
        # ...BUT THE MACHINE HALF OF THAT LINE IS NOT A NAME.  An extended-XYZ
        # comment is `<free text> Lattice="..." Properties=... pbc="..."`.
        # The header states the cell, which is parsed into `cell`; calling
        # it the title would file the same fact twice, once mislabelled, in
        # the one line of the file another tool reads as the name
        # (``model/structure.md`` § 2.2c).
        #
        # Only the three STRUCTURAL keys are cut, and only from the first one
        # on.  A sentence is what the paragraph above exists to protect, so
        # "anneal at T=300K" keeps its `=`; `Lattice`/`Properties`/`pbc` are
        # what `to_extxyz` writes and what ASE emits, and nothing else is
        # guessed at.
        _mk = _EXTXYZ_KEY.search(comment)
        if _mk is not None:
            comment = comment[:_mk.start()].strip()

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
        symbols = list(first.get_chemical_symbols())
        first_cell = np.asarray(first.cell, dtype=float)
        first_pbc = tuple(bool(b) for b in first.pbc)
        # ONE FRAME SET: the frames share everything but their coordinates.
        # Counted from 1 here, because a person reads it.
        for k, im in enumerate(images[1:], start=1):
            differs = (
                "atom count" if len(im) != len(first) else
                "species or atom order" if list(im.get_chemical_symbols())
                != symbols else
                "Lattice=" if not np.array_equal(
                    np.asarray(im.cell, dtype=float), first_cell) else
                "pbc=" if tuple(bool(b) for b in im.pbc) != first_pbc else
                None)
            if differs:
                raise ValueError(
                    f"frame {k + 1} of {len(images)} differs from frame 1 in "
                    f"its {differs}; the frames of one document are one frame "
                    f"set -- the same atoms in the same order, in the same "
                    f"cell (model/structure.md § 2.3)")
        coordinates = np.asarray([im.get_positions() for im in images],
                                 dtype=float)

        # THE CELL, ONLY WHERE IT MEANS ONE.  A `Lattice=` is adopted as this
        # structure's explicit cell only when some axis is actually periodic.
        # Our own `to_extxyz` writes the RESOLVED box for an isolated molecule
        # too -- its bounding box plus vacuum, with `pbc="F F F"` beside it --
        # and adopting that would promote a DERIVED value into a stored one, so
        # the box would stop tracking the vacuum it came from
        # (structure-periodicity.md's raw-vs-resolved line).  A `.xyz` that
        # travels with its `.molstruct.json` gets the real cell from the
        # sidecar anyway, applied after this parse.
        cell = first_cell
        periodic = first_pbc
        carries_cell = bool(cell.any()) and any(periodic)

        return cls(
            elements=symbols,
            positions=coordinates[0],
            frames=(coordinates if len(coordinates) > 1 else None),
            title=(title if title is not None else comment),
            cell=(cell.tolist() if carries_cell else None),
            # THE BOOLEANS BECOME KINDS AT THE DOOR.  extxyz carries only
            # `pbc="T T F"`, so this is the one place in the project that
            # legitimately starts from booleans -- and it converts here
            # rather than storing a second periodicity field.  `transport`
            # cannot be recovered from a boolean and is not guessed; a pair's
            # sidecar restores it.
            axis_kind=(tuple("periodic" if b else "isolated"
                             for b in periodic) if carries_cell else None),
        )

    # ------------------------------------------------------------------ #
    #  Input: PDB                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pdb(cls, text: str, *,
                 title: Optional[str] = None) -> "Structure":
        """Parse a Structure from PDB TEXT.  Not a path -- see
        :func:`_require_text`; ``StructureCodec().load(path)`` reads files.

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
          - a chain letter unique to one segment passes through as-is;
          - a chain letter spanning multiple segments is disambiguated
            by appending the segment index, so the resulting chain ids
            are unique;
          - a blank chain-id column ('_' internally) becomes 'A' when
            unambiguous, '_<n>' when it spans multiple TER segments.
        """
        text = _require_text(text, "pdb")

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
            # DECODING THE FORMAT, not correcting the user.  Uppercase IS
            # the convention for cols 77-78 -- every two-letter element in
            # this repository's PDB corpus is written that way (MG, NA, CL,
            # MN, CD), and PDB Bank's canonical files agree.  So mapping to
            # symbol case here is reading the field, exactly as reading the
            # element out of cols 13-14 below is.
            #
            # This is the boundary rule: a FILE is
            # authoritative and we decode what it says; a label a PERSON
            # types is gated at create/add/modify and never case-corrected
            # (``chemistry.resolve_element``, which does no folding at all).
            # Without the fold, an undecoded ``FE`` reaches the SIESTA
            # emitter as a species that is not iron.
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
                # the single-letter form if not matched, so "FE", "ZN",
                # "MG", "NA", ... do not degrade to "F", "Z", "M", "N",
                # which are the wrong elements (or invalid symbols).
                #
                # NO FALLBACK: without the table "FE" reads as "F",
                # silently, so a missing ASE (a declared
                # dependency) raises here.  It is the table
                # `chemistry.atomic_number` reads too; this module sits
                # below `chemistry` and asks the table itself.
                from ase.data import atomic_numbers as _known
                raw = (line[12:14] if len(line) >= 14 else "").strip()
                cand2 = raw[:2].capitalize() if len(raw) >= 2 else ""
                cand1 = raw[:1].upper() if raw else ""
                if cand2 and cand2 in _known:
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
        # segment passes through unchanged; a letter that spans multiple
        # segments has the segment index appended so the resulting ids are
        # unique.  Empty chain-id columns ('_' placeholder) map to 'A' in the
        # unambiguous case.
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

    def to_xyz(self, *, comment: str = "") -> str:
        """Return XMol .xyz text.  TEXT, not a file -- see
        :func:`_require_text` for the read side of the same rule;
        ``StructureCodec().write(struct, path)`` writes files.

        The result drops directly into a SIESTA
        ``%block AtomicCoordinatesAndAtomicSpecies`` once you map symbols
        to species indices, or into any other code that reads .xyz.

        WHY THERE IS NO ``path`` ARGUMENT.  ``from_xyz`` refuses a path and
        points at the codec, because the one door for a STORED structure
        reads the ``.molstruct.json`` beside it; a lone ``.xyz`` written here
        would drop the frozen atoms, the region labels and the explicit cell.
        `model/structure.md` § 2.4 states the rule both halves keep -- *every
        structure-to-bytes translation goes through this codec*.
        """
        buf = StringIO()
        buf.write(f"{self.n_atoms}\n")
        buf.write((comment or self.title or "Built by molbuilder").strip() + "\n")
        for el, (x, y, z) in zip(self.elements, self.positions):
            buf.write(f"{el:<3s} {x: 12.6f} {y: 12.6f} {z: 12.6f}\n")
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    #  Output: extended XYZ (one frame, or a whole trajectory)            #
    # ------------------------------------------------------------------ #

    def to_extxyz(self, *, comment: str = "") -> str:
        """Return extended-XYZ TEXT for this structure -- one block per frame
        it holds (``model/structure.md`` § 2.2e).  Not a file:
        ``StructureCodec().write(struct, path)`` writes one, and
        :meth:`to_xyz` records why there is no ``path`` argument.

        Extended XYZ is plain XYZ with the per-frame comment line carrying
        key=value metadata -- the convention ASE reads and writes, and what
        every trajectory tool expects.  Two keys go out:

        ``Lattice``
            The cell **as it will actually be used** (:meth:`resolve_cell`),
            row-major, in Angstrom.  This is the same box MolView draws and the
            Cell page reports, so a file and the viewer it came from cannot
            describe different systems.
        ``pbc``
            Which axes are periodic (``T``/``F``), from :meth:`pbc`.  It is what
            keeps the Lattice honest: an isolated molecule still HAS a resolved
            box -- its bounding box plus vacuum -- and writing that without
            ``pbc="F F F"`` would tell the reader the system repeats when it
            does not.

        WHY THIS EXISTS BESIDE ``to_xyz`` AND NOT INSTEAD OF IT.  A plain
        ``.xyz`` has nowhere to put a cell, so a periodic structure written that
        way loses its box -- and a *trajectory* written that way loses it on
        every frame.  ``to_xyz`` stays for the single-frame, cell-less case that
        every code reads; this is for the cases it cannot carry.

        The elements and the cell are written from ``self`` and are the same
        for every block, which is what makes the document one frame set rather
        than a pile of structures -- and what ``from_xyz`` checks on the way
        back in.
        """
        blocks = list(self.frames) if self.n_frames > 1 else [self.positions]
        n = self.n_atoms

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
        flags = " ".join("T" if p else "F" for p in self.pbc())
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
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    #  Output: PDB                                                        #
    # ------------------------------------------------------------------ #

    def to_pdb(self) -> str:
        """Standard PDB TEXT. Hydrogens included, single MODEL, no CONECT.
        Not a file: ``StructureCodec().write(struct, path)`` writes one, and
        :meth:`to_xyz` records why there is no ``path`` argument."""
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
        return buf.getvalue()

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

        Raises ImportError if ASE isn't installed.
        """
        try:
            from ase import Atoms
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_ase() needs the 'ase' package; install with "
                "`pip install ase`"
            ) from exc
        # WITH THE BOX, because `pbc` exists precisely as the ASE-interop view
        # of `axis_kind` (§ 1) and this is the one ASE door: atoms alone
        # would describe a crystal as a gas-phase cluster.  `resolve_cell()` rather than
        # the raw field, so a derived box travels too; `None` stays unset.
        resolved = self.resolve_cell()
        return Atoms(symbols=self.elements, positions=self.positions,
                     cell=(None if resolved is None
                           else np.asarray(resolved, dtype=float)),
                     pbc=self.pbc())

    # ------------------------------------------------------------------ #
    #  Combine / translate / center -- handy small utilities              #
    # ------------------------------------------------------------------ #

    def _carry_nonatom(self) -> dict:
        """The NON-PER-ATOM facts an atom edit carries: the lattice, and
        the free store.

        None of these are per-atom, so an add / delete / rigid transform
        carries them verbatim.  Dropping any of them silently reverts a
        periodic or transport cell to isolated defaults -- ``axis_kind`` falls
        back to ``isolated`` on every axis when no cell is carried either, and
        ``vacuum`` to unset.  Deleting a stray atom would then wipe a transport
        cell, and the emitted SIESTA FDF would omit ``LatticeVectors``.

        ``pbc`` is not carried: it is :meth:`pbc`, computed from
        ``axis_kind`` on demand.

        Every op-helper that rebuilds a Structure spreads this so those facts
        survive the edit.

        ``info`` RIDES HERE TOO, and the reason is the opposite of what
        dropping it looks like.  An edit is meant to OUTDATE the recorded
        contract, not erase it: `molview.md` § 8.4a splits
        ``structure_modified`` from ``labels_modified`` so a later reader
        is told WHICH kind of edit happened.  Rebuilding without ``info``
        deletes the record the flag marks -- so the flag has nothing to
        mark.  Voiding a calculation is a MARK on the record, and a record
        that is gone cannot carry one.  ``customized`` rides for the same
        reason (``model/structure.md`` § 2.2d).

        A FRAME SET IS REFUSED HERE (:data:`FRAME_SET_NOT_EDITED`): every
        atom edit that rebuilds a Structure takes its non-atom facts from
        this door, so this is where an edit of a frame set is stopped -- one
        place, before a rebuild could keep frame 0 and drop the rest.
        """
        if self.n_frames > 1:
            raise ValueError(FRAME_SET_NOT_EDITED)
        return self._nonatom()

    def _nonatom(self) -> dict:
        """The non-per-atom facts, copied -- what :meth:`replace` derives with
        and :meth:`_carry_nonatom` hands an atom edit."""
        return dict(
            cell        = (self.cell.copy() if self.cell is not None else None),
            engine_offset = (self.engine_offset.copy()
                             if self.engine_offset is not None else None),
            axis_kind   = self.axis_kind,
            vacuum      = self.vacuum,
            info        = _copy.deepcopy(self.info) if self.info else {},
            customized  = _copy.deepcopy(self.customized),
        )

    # ------------------------------------------------------------------ #
    #  The `info` namespace -- one cluster per subsystem (§ 2.2a)         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _json_safe(value, where: str):
        """A cluster must survive the sidecar, so it must be JSON.

        Checked on the way IN, where the caller and the offending value are
        both in hand -- not at save time, several steps away, on a structure
        that has already been edited.  Mirrors the browser's own door, which
        does the same round-trip before accepting a cluster.
        """
        try:
            return _json.loads(_json.dumps(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{where}: the value must be JSON-serialisable, because "
                f"`info` is written to the .molstruct.json sidecar -- "
                f"{exc}") from exc

    def set_info(self, key: str, value: Any) -> None:
        """Write ONE cluster of :attr:`info`, in place, leaving the rest alone.

        ``info`` is a namespace: a top-level key names a subsystem and owns
        everything under it (§ 2.2a).  This writes one of them, which is the
        difference that matters -- assigning ``struct.info`` replaces the
        WHOLE store, so a caller recording its own metadata would take the
        recorded calculation contract with it unless it happened to copy that
        across too.

        The browser's half is ``molview.data.info.set``; this is the Python
        half.
        """
        if not isinstance(key, str) or not key:
            raise ValueError(
                "Structure.set_info: the cluster name must be a non-empty "
                "string (it is a top-level key in `info`)")
        safe = self._json_safe(value, f"Structure.set_info({key!r})")
        self.info = dict(self.info or {})
        self.info[key] = safe

    def drop_info(self, key: str) -> bool:
        """Remove ONE cluster.  ``True`` if it was there.

        A STRIP IS EXPLICIT (§ 2.2a), and this is how one is said for a
        single cluster -- as against `replace(info={})`, which says it for
        the whole store.  Removing a cluster that is not there is not an
        error: the caller asked for it to be gone and it is.
        """
        if not self.info or key not in self.info:
            return False
        self.info = {k: v for k, v in self.info.items() if k != key}
        return True

    def apply_info_dict(self, data: Optional[dict]) -> None:
        """Replace the WHOLE ``info`` store, in place.

        The in-place sibling of ``replace(info=...)``, and the door the three
        whole-store writers needed: a sidecar load, a caller-stated block,
        and the Results tab's run record each adopt an entire store rather
        than one cluster.

        ``None`` or ``{}`` clears it -- which is what "this pair records
        nothing" means, and is why absence and emptiness are the same here.
        """
        if data is None:
            self.info = {}
            return
        if not isinstance(data, dict):
            raise ValueError(
                f"Structure.apply_info_dict: `info` is an object of "
                f"cluster-name -> value (§ 2.2a); got {type(data).__name__}")
        bad = [k for k in data if not isinstance(k, str) or not k]
        if bad:
            raise ValueError(
                f"Structure.apply_info_dict: cluster names must be non-empty "
                f"strings; got {bad!r}")
        self.info = self._json_safe(dict(data), "Structure.apply_info_dict")

    def mark_contract_outdated(self, what: str = "structure") -> None:
        """An edit OUTDATES the recorded contract; it does not erase it.

        The Python half of `molview/model.js`'s ``markContractOutdated``,
        with the same three rules so the two languages cannot disagree
        about one record (`model/structure.md` § 2.2a):

        * **Never invented.** No ``info.calculation`` means nothing was
          recorded, so there is nothing to outdate and no block is created.
        * **Never cleared.** Un-editing is what a retract/restore is for;
          both flags ride the pair like everything else in the store.
        * **Two flags, because they void different things.**
          ``labels_modified`` says the electrode/device partition was
          renamed — the settings still stand, but what they were sorted on
          may not.  ``structure_modified`` says the geometry or the cell
          moved, so the mesh cutoff and transverse k-mesh were converged
          for something that is no longer there.  `transport/compose.py`
          reads both and warns in those words.

        Called at the places that ARE edits, not inside `replace()`:
        `transport.sort.categorical_sort` reorders atoms through the same
        door and is not an edit, so marking there would warn about every
        composed junction.
        """
        block = self.info.get("calculation") if self.info else None
        if not isinstance(block, dict):
            return
        key = "labels_modified" if what == "labels" else "structure_modified"
        if not block.get(key):
            block[key] = True

    def copy(self) -> "Structure":
        """Return a deep-ish copy: all metadata lists are duplicated;
        ``positions`` is copied so the new Structure can be mutated
        without affecting the original.  Used by op helpers that
        return the input unchanged (e.g. ``add_slab`` with
        ``n_layers <= 0`` short-circuits to ``struct.copy()`` rather
        than open-coding the field-by-field rebuild three times).

        ONE implementation, in :meth:`replace`: this is that door with
        nothing changed.  Two copies of the field list is how a field comes
        to be duplicated in one and missing from the other.
        """
        return self.replace()

    def affine(self, linear: Sequence[Sequence[float]],
               translation: Sequence[float]) -> "Structure":
        """Apply one rigid/affine map ``x -> x @ linearᵀ + translation`` to the
        ATOMS, and to nothing else (user, 2026-09-25: *"leave the cell alone,
        moving atoms only moves atoms"*).  THE single transform primitive
        ``translated`` / ``rotate_around_axis`` / ``orient_along_axis`` route
        through.

        The cell's vectors and a stated ``engine_offset`` stay exactly as they
        were -- under Automatic the rule re-centres the box on the moved atoms
        -- because moving atoms is an edit of where they are, not a change of
        frame.  A person who wants the box elsewhere sets it on the
        Cell page, and an atom the move leaves outside the box is named there
        and at the deck (`model/structure-periodicity.md` § 6.0).
        Index-preserving, so every label and the rest of the metadata carry
        verbatim."""
        L = np.asarray(linear, dtype=float).reshape(3, 3)
        t = np.asarray(translation, dtype=float).reshape(3)
        out = self.replace(positions=self.positions @ L.T + t)
        # A RIGID TRANSFORM IS AN EDIT (`molview/model.js` marks every
        # `applyOp` the same way): the atoms the contract was converged on
        # have moved, so the record is outdated -- not erased.
        out.mark_contract_outdated()
        return out

    def translated(self, vec: Sequence[float]) -> "Structure":
        # A rigid translation of the atoms, through the ONE affine primitive;
        # the box stays where it is.
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
        # A MERGE JOINS ONE-FRAME STRUCTURES (`model/structure.md` § 2.2b):
        # the frames of a set have no counterpart in the other input.
        if any(s.n_frames > 1 for s in structures):
            raise ValueError(FRAME_SET_NOT_EDITED)
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
        # WHO SUPPLIES EACH NON-ATOM FIELD -- `model/structure.md` § 2.2b.
        # The cell and a stated offset come from whoever STATES a cell, because a
        # lattice is the one thing a fragment can supply that a cell-less
        # canvas lacks.  `axis_kind`, `vacuum` and `info` come from the first
        # input either way: they are facts OF the canvas, equally true of a
        # structure that states no lattice, so a fragment arriving with a box
        # must not restate them.  Both halves matter and they point opposite
        # ways -- taking the whole block from the cell-carrier replaced a
        # typed 8 A vacuum with (0,0,0) and turned isolated axes crystalline;
        # taking none of it threw away the only lattice in play.
        #
        # The merged box is not made to fit: `concat` cannot infer a lattice,
        # so atoms outside the adopted cell are left for `cell.check` to
        # report as `cell.unfittable`.  That is the stated outcome (§ 2.2b),
        # not a loss to paper over.
        base = next((s for s in structures if s.cell is not None), None)
        first = structures[0]
        lattice = first._carry_nonatom()
        if base is not None and base is not first:
            _box = base._carry_nonatom()
            lattice["cell"] = _box["cell"]
            lattice["engine_offset"] = _box["engine_offset"]
        # `info` IS NOT THE LATTICE'S.  The recorded contract belongs to the
        # one being appended TO, which is the first, cell or no cell.
        lattice["info"] = (_copy.deepcopy(first.info) if first.info else {})
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
            **lattice,
        )


# The reserved label's accessor, installed under its real name AFTER ``@dataclass``
# has read the class body.  Defining it as ``frozen_atoms`` inside the body would
# make the property object the field's default; defining it here means the
# generated ``__init__`` executes ``self.frozen_atoms = <arg>`` straight into the
# setter, so construction and later assignment go through the same one door.
Structure.frozen_atoms = Structure._frozen_atoms
del Structure._frozen_atoms
