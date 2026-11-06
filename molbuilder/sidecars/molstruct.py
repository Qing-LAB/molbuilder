"""``.molstruct.json`` sidecar — its codec: the builder, the writer, the
read door and the consumer helpers.

The ONE DOOR for this sidecar: the write side and the read door
(:func:`load`) live here, so a caller needs one import for both; the text
reader the read door asks is ``load_text`` in
:mod:`molbuilder.parse.sidecars.molstruct` (``model/parse.md`` § 4).

TRAVELS beside every PySCF script, in ``mb_pyscf.pyz``
(``runwrap.PYSCF_COMPANIONS``): the codec's ``write_moved`` saves a run's
geometries through :func:`renewed`, :func:`geometry_hash`, :func:`dumps`
and :func:`sidecar_path_for`.  So it imports only the standard library at
load; the functions that read through the package (``to_dict``,
``apply_to_structure``, ``load``) import it inside themselves -- of them,
a PySCF vibration's script asks ``apply_to_structure``
(`spectra.pyscf_vibration.equilibrium_structure`), imported two ways.

Public surface here
-------------------

* :data:`SCHEMA_VERSION`            — current on-disk schema.  The number
  lives ONLY on the constant below.
* :exc:`MolstructJsonError`         — raised on malformed input or
  invariant violations; named in `structure_pair`, the pair's refusals.
* :func:`sidecar_path_for`          — canonical ``<stem>.molstruct
  .json`` derivation.
* :func:`geometry_hash`             — the pin: a geometry's atom lines and
  the cell the sidecar states for them (``structure-molstruct.md`` § 3).
* :func:`to_dict`                   — THE ONE BUILDER of a payload, the
  envelope and the metadata fields spread in, for a payload written and
  one read alike.  Normalises + validates.
* :func:`renewed`                   — a payload's envelope renewed for a
  geometry whose atoms moved (``StructureCodec.write_moved``).
* :func:`dumps`                     — the one serialisation.
* :func:`load`                      — the read door: a sidecar file back
  as its normalised payload.
* :func:`apply_to_structure`        — copy loaded sidecar payload
  onto a Structure (validates atom-count match).
"""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import json as _json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# THE MAP'S KEY, named once (`atom_permutation.MAP_KEY`) -- this module
# travels beside a job with the map's module.
try:                                        # inside molbuilder
    from ..atom_permutation import MAP_KEY
    from ..persist import json_text
except ImportError:                         # beside a job, in a bundle
    from atom_permutation import MAP_KEY  # type: ignore[no-redef]
    from persist import json_text  # type: ignore[no-redef]


#: The on-disk sidecar schema, READ STRICTLY (:data:`READABLE_VERSIONS`).
#:
#: ``frozen_atoms`` is an ORDINARY LABEL in ``regions``, like every other one.
#: What makes it reserved is not storage: it is a special INTERPRETATION where the context calls
#: for it (SIESTA's ``Geometry.Constraints``) plus one designated accessor to pull
#: that group out (:func:`frozen_atoms` here, ``getFrozen()`` in the browser).  One
#: store, one spelling of the name, interpreted at the end.
#:
#: A version gate that admits a version the code cannot honour is worse than
#: no gate: it turns a loud failure into a quiet one.
SCHEMA_VERSION = 17  # 17: `selection_rules` RETIRED -- never written; a
#                      7-16 file's is read past (RETIRED_ENVELOPE_KEYS).
#                      Every older file reads whole.
#                   16: + the OPTIONAL `species_labels`
#                      (structure.SPECIES_LABELS), each atom's species
#                      label where one is not the name its geometry file
#                      holds (model/chemistry.md § 3) -- additive: a 7-15
#                      file's symbols are its labels.
#                   15: + `atom_index_map`, which index the atoms are
#                   listed by and each atom's indices (model/overview.md
#                   § 2.2b), an envelope key the codec writes from the door
#                   that writes the pair.
#                   14: + `atom_order`, the order record of a structure
#                      in an engine's order, null in the user order
#                      (model/overview.md § 2.2a) -- additive: a 7-13 file
#                      reads as the user order.
#                   13: the pin is `geometry_hash` -- the atom lines as
#                      written and the cell, never the comment line
#                      (structure-molstruct.md § 3) -- and `structure_hash`,
#                      the file's bytes, is RETIRED: read and never
#                      compared, never written again.  Additive: a 7-12
#                      file opens unchecked.
#                   12: + the OPTIONAL `offset_by` -- whose the
#                      stated `engine_offset` is, the person's or the
#                      engine's (model/structure-periodicity.md § 6.0) --
#                      additive: a 7-11 file's stated offset reads as the
#                      person's, kept as stated.
#                   11: + the OPTIONAL `customized` block (the
#                      structure's named values, one row set per frame;
#                      model/structure.md § 2.2d) and the envelope's
#                      `n_frames_total` -- additive: a 7-10 file holds
#                      nothing indexed by frame.
#                   10: `cell_origin` RETIRED -- read and
#                      ignored, never written again (structure.
#                      RETIRED_METADATA_KEYS; the user's decision, plan § 5q
#                      D2) -- and the OPTIONAL stated `engine_offset` added
#                      (null = the rule places the atoms;
#                      model/structure-periodicity.md § 6.0).
#                    9: + the OPTIONAL `info` block (the free-form
#                      NON-structural store -- additive; absent
#                      means "nothing recorded", and it never enters
#                      the pin).
#                    8: + the OPTIONAL identity columns
#                      (structure.IDENTITY_FIELDS, canonical-dict spellings,
#                      written only when real -- user ruling: additive
#                      "extra", never conflicting).  A 7 file simply lacks
#                      them and reads as before; an OLDER reader meeting an
#                      8 file that carries them refuses with the stray-key
#                      message, which names exactly what it does not know.

#: The versions THIS build reads whole.  v8 only ADDED the
#: optional identity columns, so a v7 file loses nothing read under v8 rules
#: (absent identity = the synthesized defaults, which is what v7 meant).
#: Everything older stores the same facts in DIFFERENT places (v3's top-level
#: ``frozen_atoms``), and partial reads of those are the silent-loss case the
#: strict gate exists for -- so the set widens only when an addition is
#: provably additive.
#:
#: **AND ONLY WHEN A READER HAS BEEN RUN AGAINST A FILE AT EACH VERSION
#: CLAIMED.**  That is the check, not the reasoning: ``load_text`` reads the
#: keys it NAMES, so a key an older version kept elsewhere is never looked
#: for.
#:
#: Each version in the set below is read by a test against a real file at
#: that version (``tests/test_molstruct_json.py::TestSchemaVersioning``, and
#: v9 in ``TestARetiredKeyPassesEveryGate``).
#:
#: v10 removed a key and added an optional one: a v7-v9 file's
#: `cell_origin` is retired -- read and ignored, by decision (D2) -- and a
#: missing `engine_offset` means the rule, which is what those files meant for
#: every structure whose corner nobody typed.
#:
#: v11 added two: `customized`, absent meaning no row, and `n_frames_total`,
#: which a 7-10 file does not state and needs no more than it needs rows --
#: nothing in it is indexed by frame.
#:
#: v12 added `offset_by`, absent meaning an offset whose kind nobody recorded
#: -- read as the person's, so the file is placed exactly as before.
#:
#: v13 put `geometry_hash` in place of `structure_hash`: a 7-12 file has no
#: geometry hash, and its pair opens unchecked -- metadata is never required.
#:
#: v14 added `atom_order`, absent meaning the user order -- which every pair a
#: 7-13 writer made for a person is in.
#: v15 added `atom_index_map`; a 7-14 file states none, and its map is made
#: from its `atom_order` -- a 7-13 file read as listed by user index, warned
#: (`model/overview.md` § 2.2b, F4).
#: v16 added `species_labels`; a 7-15 file states none, and its geometry
#: file's symbols are its labels.
READABLE_VERSIONS = frozenset({7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17})

#: The sidecar LAYER's own keys -- everything in a payload that is not a
#: Structure metadata field.  Named so :func:`apply_to_structure` can hand the
#: gate exactly the fields it owns and REFUSE anything that is neither, instead
#: of passing the whole payload and letting unknown keys fall on the floor.
ENVELOPE_KEYS = ("schema_version", "n_atoms_total", "n_frames_total",
                 "geometry_hash", "created_by", "created_at", MAP_KEY)

#: An envelope key the schema once wrote and no longer does -- tolerated on
#: read, never applied, dropped on rewrite, as `structure.RETIRED_METADATA_KEYS`
#: (`structure-molstruct.md` § 2, *Retired keys*): `structure_hash`, the
#: sha256 of the geometry file's bytes, until v13; `selection_rules`, how
#: each region was selected, which nothing filled and nothing read (the
#: user, 2026-11-03: "retire selection_rules").
RETIRED_ENVELOPE_KEYS = ("structure_hash", "selection_rules")


def known_keys() -> frozenset:
    """EVERY KEY A SIDECAR MAY CARRY -- the structure's metadata fields and
    per-atom columns (`structure.ATOM_COLUMNS`), the envelope, the free-form ``info`` block, and the
    retired keys an older sidecar still opens with (accepted, not applied):
    the one list each gate refuses a stray key against, the read door's
    (`parse.sidecars.molstruct.load_text`) and :func:`apply_to_structure`'s
    alike.  A key nobody reads is metadata the writer thinks it saved."""
    try:                                        # inside molbuilder
        from molbuilder.structure import (ATOM_COLUMNS, FILE_FIELDS,
                                          RETIRED_IDENTITY_KEYS,
                                          RETIRED_METADATA_KEYS)
    except ImportError:                         # beside a job, in a bundle
        from structure import (ATOM_COLUMNS, FILE_FIELDS,
                               RETIRED_IDENTITY_KEYS, RETIRED_METADATA_KEYS)
    return frozenset(set(FILE_FIELDS) | set(ATOM_COLUMNS)
                     | set(ENVELOPE_KEYS) | {"info"}
                     | set(RETIRED_METADATA_KEYS) | set(RETIRED_IDENTITY_KEYS)
                     | set(RETIRED_ENVELOPE_KEYS))

# Canonical sidecar suffix.  ``<job>.xyz`` -> ``<job>.molstruct.json``.
#: PUBLIC: a suffix with a composer (:func:`sidecar_path_for`) has a public
#: name (`project-layout.md` § 4.5).
SUFFIX = ".molstruct.json"

#: THE GEOMETRY FILES A SIDECAR PAIRS WITH -- the two formats the one read
#: door opens (`StructureCodec.load`, `model/structure.md` § 2.4).  Asked
#: here, beside the pairing rule, by everything that tells a structure file
#: from another file.
GEOMETRY_SUFFIXES = (".xyz", ".pdb")


# THE PAIR'S REFUSALS are the saved pair's own (`structure_pair`, L1), so the
# structure raises the same pairing refusal for a deck's block.
try:                                        # inside molbuilder
    from ..structure_pair import MolstructJsonError, MolstructPairingError
except ImportError:                         # beside a job, in a bundle
    from structure_pair import MolstructJsonError, MolstructPairingError


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def sidecar_path_for(structure_path: Union[str, Path]) -> Path:
    """Return the canonical sidecar path for ``structure_path``.

    Strips the LAST suffix and appends ``.molstruct.json``::

        relaxed.xyz           -> relaxed.molstruct.json
        chain.pdb             -> chain.molstruct.json
        job.spectra.xyz       -> job.spectra.molstruct.json
        bridge                -> bridge.molstruct.json    (no suffix)

    THE SUFFIX IS NOT READ, and the `.pdb` row is there to say so: a
    structure molbuilder opens is `.xyz` or `.pdb` (`StructureCodec`
    accepts exactly those two), and both carry their regions, frozen
    atoms and cell in the same sister file.

    ``Path.with_suffix`` can't be chained for compound suffixes
    (it'd replace ``.molstruct`` with ``.json`` on the second call),
    so build the name explicitly from the stem.
    """
    p = Path(structure_path)
    return p.with_name(p.stem + SUFFIX)


def is_geometry(path: Union[str, Path]) -> bool:
    """Is this a structure file a sidecar may pair with, whatever the case of
    its suffix?"""
    return Path(path).suffix.lower() in GEOMETRY_SUFFIXES


def is_sidecar(path: Union[str, Path]) -> bool:
    """Is this a sidecar?  Asked of the one module that names the suffix."""
    return Path(path).name.endswith(SUFFIX)


def geometry_for(sidecar: Union[str, Path]) -> Optional[Path]:
    """The geometry file a sidecar sits beside -- the pairing rule read
    backwards: ``<stem>.xyz`` or ``<stem>.pdb`` beside
    ``<stem>.molstruct.json``, whichever is there; ``None`` when neither is,
    or when ``sidecar`` is not one."""
    p = Path(sidecar)
    if not is_sidecar(p):
        return None
    stem = p.name[:-len(SUFFIX)]
    for ext in GEOMETRY_SUFFIXES:
        if (p.parent / f"{stem}{ext}").is_file():
            return p.parent / f"{stem}{ext}"
    return None


def _atom_lines(document: str, fmt: str) -> "List[List[str]]":
    """The atom lines of a geometry document, frame by frame, as written:
    an XYZ's every frame block, its count line and comment line left out;
    a PDB's ATOM and HETATM records, one frame."""
    lines = document.splitlines()
    if fmt == "pdb":
        return [[ln for ln in lines
                 if ln.startswith("ATOM  ") or ln.startswith("HETATM")]]
    frames: List[List[str]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        frames.append(lines[i + 2:i + 2 + n])
        i += 2 + n
    return frames


def geometry_hash(document: str, fmt: str, cell, axis_kind) -> str:
    """THE PIN (`structure-molstruct.md` § 3): the sha256 (hex) of a
    geometry's atom lines -- each one's fields as written, one space between,
    after a ``frame`` line per frame -- then the cell and the axis kinds the
    sidecar states for them, ``none`` for one it states none of.  Never the
    comment line, the offset, the labels or the times.  ``fmt`` is the
    document's format, ``xyz`` or ``pdb``.  Made by every writer of a pair
    over the document it writes, and by the read door over the file it
    reads, to compare."""
    rows: List[str] = []
    for frame in _atom_lines(document, fmt):
        rows.append("frame")
        rows.extend(" ".join(line.split()) for line in frame)
    rows.append("cell " + ("none" if cell is None else _json.dumps(
        [[float(v) for v in row] for row in cell])))
    rows.append("axis_kind " + ("none" if axis_kind is None else _json.dumps(
        [str(k) for k in axis_kind])))
    return _hashlib.sha256(
        "".join(r + "\n" for r in rows).encode("utf-8")).hexdigest()


def is_geometry_hash(value) -> bool:
    """A geometry hash's form: 64 lowercase hexadecimal characters."""
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def stamp() -> str:
    """THE MOMENT A SIDECAR IS WRITTEN -- current UTC time as ISO-8601 with
    an explicit ``Z``, its writer's ``created_at`` (``StructureCodec.pair``,
    :func:`renewed`).
    ``datetime.utcnow`` is deprecated in 3.12+; use
    ``datetime.now(timezone.utc)`` and format manually for ``Z``
    semantics."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------- #
#  Serialise                                                            #
# --------------------------------------------------------------------- #


# The metadata fields (cell / axis_kind / vacuum / engine_offset) are validated
# in exactly ONE place, ``Structure.__post_init__`` (reached via
# ``structure_fields_via_dataclass`` -> ``apply_metadata_dict``), so the write
# and read paths cannot validate a field differently.


def structure_fields_via_dataclass(
    n_atoms_total: int, raw: Dict[str, Any],
    n_frames_total: int) -> Dict[str, Any]:
    """Validate + canonicalise the Structure metadata fields through the ONE
    dataclass authority (`model/structure.md` § 2.2): apply ``raw`` to a scratch
    :class:`~molbuilder.structure.Structure` of ``n_atoms_total`` atoms and
    ``n_frames_total`` frames -- which validates every field exactly as a live
    structure does, the frame rows against the frames -- then read it back
    normalised via ``metadata_to_dict``.  Asked by :func:`to_dict`, the one
    builder of a payload written and of one read, so the two can never
    enumerate a different field set.
    Raises :class:`MolstructJsonError` on any invalid field.
    """
    try:                                        # inside molbuilder
        from molbuilder.structure import FILE_FIELDS, Structure
    except ImportError:                         # beside a job, in a bundle
        from structure import FILE_FIELDS, Structure
    import numpy as _np
    n = int(n_atoms_total)
    frames = int(n_frames_total)
    scratch = Structure(elements=["C"] * n, positions=_np.zeros((n, 3)),
                        frames=(_np.zeros((frames, n, 3)) if frames > 1
                                else None))
    try:
        scratch.apply_metadata_dict(raw)
    except (ValueError, TypeError) as exc:
        raise MolstructJsonError(str(exc)) from exc
    # WHAT A FILE STATES: every field but the view's (`structure.FILE_FIELDS`)
    # -- the view's mark is stated only as the pair's atom index map.
    return {k: v for k, v in scratch.metadata_to_dict().items()
            if k in FILE_FIELDS}


def atom_columns_via_dataclass(n_atoms_total: int,
                               columns: Optional[Dict[str, Any]]
                               ) -> Dict[str, Any]:
    """Validate + canonicalise the per-atom columns a sidecar carries beside
    the metadata fields (`structure.ATOM_COLUMNS`: the identity columns and
    the species labels) through the ONE dataclass authority, as
    :func:`structure_fields_via_dataclass` does the fields: a scratch
    Structure of ``n_atoms_total`` atoms carrying them -- the labels as its
    ``elements`` -- re-runs ``__post_init__``'s own length and type checks,
    and what comes back is exactly what a reader will reconstruct.  Only
    the columns handed in come back.  A retired key is not stray (an older
    sidecar carrying ``title`` still opens, `structure.RETIRED_IDENTITY_KEYS`
    says why): it is dropped, which is the third gate a retired key gets.
    Asked by :func:`to_dict`, the one builder of a payload written and of
    one read.
    Raises :class:`MolstructJsonError` on a stray key or an invalid column.
    """
    try:                                        # inside molbuilder
        from molbuilder.structure import (ATOM_COLUMNS, RETIRED_IDENTITY_KEYS,
                                          SPECIES_LABELS, Structure)
    except ImportError:                         # beside a job, in a bundle
        from structure import (ATOM_COLUMNS, RETIRED_IDENTITY_KEYS,
                               SPECIES_LABELS, Structure)
    columns = dict(columns or {})
    stray = [k for k in columns
             if k not in ATOM_COLUMNS and k not in RETIRED_IDENTITY_KEYS]
    if stray:
        raise MolstructJsonError(
            f"identity carries {sorted(stray)!r}; the per-atom columns are "
            f"{list(ATOM_COLUMNS)!r}")
    n = int(n_atoms_total)
    labels = columns.get(SPECIES_LABELS)
    if labels is not None and not (
            isinstance(labels, list) and len(labels) == n
            and all(isinstance(lb, str) and lb.strip() for lb in labels)):
        raise MolstructJsonError(
            f"{SPECIES_LABELS} is one species label per atom, {n} of them; "
            f"got {labels!r}")
    try:
        scratch = Structure(
            elements=(labels if labels is not None else ["X"] * n),
            positions=[[0.0, 0.0, 0.0]] * n,
            atom_names=columns.get("atom_names"),
            residue_ids=columns.get("residue_ids"),
            residue_names=columns.get("residue_names"),
            chain_ids=columns.get("chain_ids"),
        )
    except (ValueError, TypeError) as exc:
        raise MolstructJsonError(str(exc)) from exc
    return {k: v for k, v in {
        "atom_names":    list(scratch.atom_names),
        "residue_ids":   [int(v) for v in scratch.residue_ids],
        "residue_names": list(scratch.residue_names),
        "chain_ids":     list(scratch.chain_ids),
        SPECIES_LABELS:  list(scratch.elements),
    }.items() if k in columns}


def frozen_atoms(payload: Optional[Dict[str, Any]]) -> List[int]:
    """The atoms carrying the reserved ``frozen`` label in a sidecar payload.

    THE way to ask a sidecar dict, the same way :attr:`Structure.frozen_atoms`
    is the way to ask a Structure -- one place spells the name, so no caller
    has to.  It reads the label store, because that is the only place the fact
    lives.
    """
    try:                                        # inside molbuilder
        from molbuilder.structure import FROZEN_LABEL   # lazy: stdlib-only at load
    except ImportError:                         # beside a job, in a bundle
        from structure import FROZEN_LABEL
    if not isinstance(payload, dict):
        return []
    regions = payload.get("regions")
    if not isinstance(regions, dict):
        return []
    return sorted({int(i) for i in (regions.get(FROZEN_LABEL) or ())
                   if isinstance(i, int) and not isinstance(i, bool)})


def to_dict(
    fields: Optional[Dict[str, Any]] = None,
    *,
    identity: Optional[Dict[str, Any]] = None,
    n_atoms_total: int,
    n_frames_total: Optional[int] = 1,
    geometry_hash: Optional[str],
    info: Optional[Dict[str, Any]] = None,
    created_by: str = "molbuilder",
    created_at: Optional[str] = None,
    atom_index_map: Optional[Dict[str, Any]] = None,
    schema_version: int = SCHEMA_VERSION,
) -> Dict[str, Any]:
    """THE ONE BUILDER of a sidecar payload -- the envelope, and the
    metadata fields, the per-atom columns and ``info`` spread in, each
    checked here -- for a payload written (``StructureCodec.pair``) and for
    one read (``parse.sidecars.molstruct.load_text``) alike, so the two can
    never check or assemble it differently.

    ``fields`` is the metadata field dict -- the SAME shape
    :meth:`Structure.metadata_to_dict` produces (``METADATA_FIELDS``).
    STRICT type: ``annotations`` are JSON channel dicts, NOT
    ``AtomChannel`` objects -- serialise a live map with
    :func:`molbuilder.structure.annotations_to_json` first.  A subset is fine
    (an absent key -> the Structure default).  This ONE dict-shaped parameter
    matches ``apply_metadata_dict`` / ``apply_to_structure`` -- the whole
    metadata API set speaks the same format.

    The envelope (``schema_version`` / ``n_atoms_total`` / ``n_frames_total``
    / ``geometry_hash`` / ``created_*``) is layered on.  ``schema_version``
    is the version the envelope states: this build's for a payload written,
    the file's own for one read, never this build's in its place
    (`model/overview.md` § 2.2b, F4).  A version states its envelope whole
    -- the frame count from v11, the geometry hash from v13
    (`structure-molstruct.md` § 2); a file of an older version states
    neither, ``None`` here, and its fields are checked against one frame.
    ``created_at`` is the moment the payload was made, as given: its
    writer's stamp for a payload written (``StructureCodec.pair``), the
    file's own for one read -- ``None`` where a file states none, never the
    moment it is read.
    ``atom_index_map`` -- which index the atoms are listed by, and each
    atom's indices (`model/overview.md` § 2.2b) -- the codec's, from the door
    that writes the pair, or the file's; read by the map's one reader.
    Doesn't write anywhere; ``StructureCodec.write`` writes the pair.
    """
    if not isinstance(n_atoms_total, int) or n_atoms_total < 0:
        raise MolstructJsonError(
            f"n_atoms_total must be a non-negative int; got "
            f"{n_atoms_total!r}"
        )
    if not (geometry_hash is None and schema_version < 13) \
            and not is_geometry_hash(geometry_hash):
        raise MolstructJsonError(
            f"geometry_hash must be a sha256 hex string (got "
            f"{geometry_hash!r})")
    if not (n_frames_total is None and schema_version < 11) and (
            isinstance(n_frames_total, bool)
            or not isinstance(n_frames_total, int) or n_frames_total < 1):
        raise MolstructJsonError(
            f"n_frames_total must be a positive int; got {n_frames_total!r}")

    # The Structure FIELDS -> validated + canonicalised through the ONE dataclass
    # authority (a scratch N-atom Structure IS the schema), for a payload
    # written and one read alike, so no field can drift between them.
    fields = structure_fields_via_dataclass(n_atoms_total, fields or {},
                                            n_frames_total or 1)
    # The OPTIONAL per-atom columns (identity, schema 8; species labels,
    # v16): validated the same way -- through the ONE dataclass authority.
    # Only REAL columns are handed in (Structure.identity_to_dict already
    # skipped the synthesized defaults); `title` is NOT rebuilt -- the
    # geometry file owns it (``model/structure.md`` § 2.2c).
    identity = atom_columns_via_dataclass(n_atoms_total, identity)

    out = {
        # Envelope -- the sidecar LAYER's own keys (not Structure fields).
        "schema_version":  int(schema_version),
        "n_atoms_total":   n_atoms_total,
        # The frame count (v11) and THE PIN (v13), where the version states
        # them: a v7-v12 file has no pin, and its pair opens unchecked
        # (`structure-molstruct.md` § 3).
        **({"n_frames_total": n_frames_total}
           if n_frames_total is not None else {}),
        **({"geometry_hash": geometry_hash}
           if geometry_hash is not None else {}),
        # The Structure metadata block, VERBATIM from the ONE codec
        # (metadata_to_dict, via structure_fields_via_dataclass): every field
        # of METADATA_FIELDS -- and never a retired key.  Spread -- NOT re-listed -- so a field added to the
        # dataclass rides onto the sidecar automatically and this layer can no
        # longer drop or drift one (`model/structure.md` § 2.2: the ONE
        # serialization authority; add a key there and nowhere else).
        **fields,
        # The per-atom columns (identity, schema 8; species labels, v16) --
        # absent entirely when nothing is real, so an xyz-born sidecar
        # carries none.
        **identity,
        "created_by":      str(created_by),
        "created_at":      created_at,
    }
    if atom_index_map is not None:
        try:                                    # inside molbuilder
            from ..atom_permutation import AtomIndexMap, AtomIndexError
        except ImportError:                     # beside a job, in a bundle
            from atom_permutation import (  # type: ignore
                AtomIndexMap, AtomIndexError)
        try:
            out[MAP_KEY] = AtomIndexMap.from_dict(
                atom_index_map, n_atoms=n_atoms_total).to_dict()
        except AtomIndexError as exc:
            raise MolstructJsonError(str(exc)) from exc
    # The `info` block (schema 9): the free-form NON-structural store,
    # written only when something is recorded -- an empty store writes
    # no `info` key.
    # JSON-checked here so an unserialisable value fails at WRITE time
    # with a clear owner, never at a later save of unrelated work.
    if info:
        if not isinstance(info, dict):
            raise MolstructJsonError(
                f"info must be a dict of key -> JSON value, got "
                f"{type(info).__name__}")
        try:
            _json.dumps(info)
        except (TypeError, ValueError) as exc:
            raise MolstructJsonError(
                f"info holds a value JSON cannot carry: {exc}")
        out["info"] = info
    return out


def renewed(sidecar: Dict[str, Any], *, geometry_hash: str,
            atom_index_map: Dict[str, Any]) -> Dict[str, Any]:
    """THE ENVELOPE OF A GEOMETRY WHOSE ATOMS MOVED (`model/structure.md`
    § 2.4; ``StructureCodec.write_moved``): ``sidecar`` -- the payload
    :func:`to_dict` made for the structure before it moved -- with its
    envelope renewed for the document written now: ``geometry_hash`` made
    over that document, ``created_at`` the moment it is written, and the
    pair's own ``atom_index_map``.  Every other key -- the labels, the cell,
    the periodicity -- is the payload's, never derived a second time.  The
    standard library alone: a PySCF run asks it beside the job."""
    return dict(sidecar, geometry_hash=geometry_hash,
                created_at=stamp(), **{MAP_KEY: atom_index_map})


def dumps(payload: Dict[str, Any]) -> str:
    """The canonical on-disk TEXT for a sidecar payload -- the ONE serialisation.

    ``StructureCodec.write`` writes exactly this, and anything that needs the same bytes
    without writing them (a download, a comparison, a preview) asks for it here.
    The text is the byte shape of every JSON file molbuilder writes
    (`persist.json_text`): a non-ASCII region label ("α-helix") a literal,
    never an escape.  A number JSON cannot hold -- a NaN recorded in
    ``info`` -- is refused with :class:`ValueError` before any text is made,
    where ``json_text`` would write it null: the sidecar its serialiser
    refuses stops the pair's write with the pair on disk as it was
    (`model/structure.md` § 2.4).
    """
    return json_text(payload, refuse_non_finite=True)


# --------------------------------------------------------------------- #
#  Apply to a Structure                                                 #
# --------------------------------------------------------------------- #


def apply_to_structure(struct, sidecar_data: Dict[str, Any]) -> None:
    """Apply a loaded sidecar payload's metadata onto ``struct`` IN PLACE.

    Delegates the whole field set (``METADATA_FIELDS``) to
    :meth:`molbuilder.structure.Structure.apply_metadata_dict` -- the SINGLE
    dict->struct authority (`model/structure.md` § 2.2).  Because the writer
    (``Structure.metadata_to_dict``) and this reader share that one method, they
    cannot drift a field.

    Validates that the sidecar's ``n_atoms_total`` matches the structure's atom
    count -- a mismatch usually means the XYZ was edited separately and the
    sidecar's indices no longer point at the right atoms.  That one raises
    :class:`MolstructPairingError` rather than the base error, so a lenient
    surface can keep this guard while forgiving the version failures below
    (see that class for why the two must not share an answer).  The sidecar's
    ``geometry_hash`` is NOT verified here: the read door compares it with the
    file it read (`StructureCodec.load`).  A key that is neither a
    structure metadata field nor an envelope key is REFUSED here rather than
    ignored -- a key nobody reads is metadata the writer thinks it saved.
    """
    sidecar_n = sidecar_data.get("n_atoms_total")
    struct_n = len(struct.elements)
    if sidecar_n != struct_n:
        raise MolstructPairingError(
            f"sidecar n_atoms_total={sidecar_n} but structure has "
            f"{struct_n} atoms.  The sidecar's region / frozen-atom "
            f"indices no longer point at the right atoms; re-export "
            f"the sidecar from /modify after structural edits."
        )
    # THE FRAME COUNT, beside the atom count and for the same reason: each
    # frame's `customized` rows are indexed by frame (`structure-molstruct.md`
    # § 3).  A v7-v10 payload states none and holds nothing indexed by frame.
    sidecar_f = sidecar_data.get("n_frames_total")
    if sidecar_f is not None and sidecar_f != struct.n_frames:
        raise MolstructPairingError(
            f"sidecar n_frames_total={sidecar_f} but the structure holds "
            f"{struct.n_frames} frame(s).  The sidecar's frame rows no longer "
            f"belong to these frames; write the pair again from the frame set "
            f"it describes.")
    try:                                        # inside molbuilder
        from molbuilder.structure import (FILE_FIELDS, IDENTITY_FIELDS,
                                          SPECIES_LABELS, split_species_label)
    except ImportError:                         # beside a job, in a bundle
        from structure import (FILE_FIELDS, IDENTITY_FIELDS, SPECIES_LABELS,
                               split_species_label)
    # A RETIRED KEY IS NOT A STRAY ONE -- the gates ask that question of
    # one list (:func:`known_keys`), so they give the same answer
    # (`structure.RETIRED_METADATA_KEYS` says which and why).
    #
    # `StructureCodec.load` never reaches here with a stray key, because
    # `parse.sidecars.molstruct.load_text` answers first.  This gate takes a
    # payload DIRECTLY, so a caller that builds one in code does reach it.
    stray = [k for k in sidecar_data if k not in known_keys()]
    if stray:
        raise MolstructJsonError(
            f"sidecar carries {sorted(stray)!r}, which is neither a structure "
            f"metadata field {list(FILE_FIELDS)!r} nor an envelope key "
            f"{list(ENVELOPE_KEYS)!r}.  Refused rather than ignored: a key "
            f"nobody reads is metadata the writer thinks it saved.")
    # Identity first (schema 8), FULL-REPLACE -- the same semantics
    # apply_metadata_dict documents for the metadata block: an absent key
    # resets the field to its default (post_init refills the placeholders).
    # This is what makes the sidecar the identity AUTHORITY for the four
    # columns it carries: an absent column means the synthesized default, so
    # a placeholder the reader invented cannot round-trip into the next
    # sidecar as something the person stated.  Plain attribute sets, so the apply_metadata_dict call below re-runs
    # __post_init__ over them -- one validator, one message.
    # `title` IS NOT IN THIS LOOP.  The geometry file owns it
    # (``model/structure.md`` § 2.2c), so this door leaves it alone.  Were it
    # here, the `else` branch would reset it to "" -- erasing the .xyz comment
    # line of any pair whose sidecar states no title -- and the `if` branch
    # would silently override a comment line the user had edited by hand.
    for k in IDENTITY_FIELDS:
        if k in sidecar_data:
            setattr(struct, k, sidecar_data[k])
        else:
            setattr(struct, k, None)
    # THE SPECIES LABELS (v16) back over the names the geometry file holds
    # (`model/chemistry.md` § 3) -- each label's name the atom's, or the pair
    # is refused naming the atom.  Before the apply below, so the atom names
    # it synthesizes are the labels, as they were when the pair was written.
    labels = sidecar_data.get(SPECIES_LABELS)
    if labels is not None:
        held = struct.chemical_symbols()
        named = [split_species_label(lb)[0] for lb in labels]
        wrong = next((i for i, (a, b) in enumerate(zip(named, held))
                      if a != b), None)
        if wrong is not None:
            raise MolstructPairingError(
                f"the sidecar's species label {labels[wrong]!r} for atom "
                f"{wrong} names {named[wrong]}, but the geometry file holds "
                f"{held[wrong]} there.  The labels no longer belong to these "
                f"atoms; write the pair again from the structure it "
                f"describes.")
        struct.elements = list(labels)
    # The info block (schema 9), FULL-REPLACE like everything else here:
    # absent means "nothing recorded", and a stale store must not survive
    # a pair that no longer carries one.
    raw_info = sidecar_data.get("info")
    # THROUGH THE DOOR.  A sidecar states the whole store, so this is the
    # whole-store door -- and it is a door rather than an assignment so the
    # shape is checked here, at the file that supplied it, instead of at
    # the writer several steps later (§ 2.2a).
    struct.apply_info_dict(raw_info if isinstance(raw_info, dict) else None)
    try:
        struct.apply_metadata_dict(
            {k: v for k, v in sidecar_data.items() if k in FILE_FIELDS})
    except (ValueError, TypeError) as exc:
        # Surface field-validation failures as the sidecar-layer error type,
        # preserving the clear per-field message from the dataclass invariants.
        raise MolstructJsonError(str(exc)) from exc
    # A V7-V11 FILE STATES AN OFFSET AND NOT WHOSE: it is the person's
    # (`model/structure-molstruct.md` v12), stated so -- a re-save then
    # writes `person`, never `null` beside an offset.
    if (struct.engine_offset is not None and struct.offset_by is None
            and int(sidecar_data.get("schema_version") or SCHEMA_VERSION) < 12):
        struct.offset_by = "person"
    # THE ATOM INDEX MAP AGAINST THE ORDER RECORD (`model/overview.md`
    # § 2.2b): one statement of one order -- a pair whose map says another
    # order than its record is refused by name.
    if MAP_KEY in sidecar_data:
        _map_agrees(sidecar_data[MAP_KEY], struct)


def _map_agrees(block, struct) -> None:
    """THE MAP A SIDECAR STATES, read by the map's one reader and checked
    against the structure's order record through the map's own makers
    (`model/overview.md` § 2.2b): with a record, listed by engine index
    with that record's columns; with none, listed by user index, ``not
    stated`` (F4), or by engine index with both columns the identity -- an
    engine record whose sort moved nothing.  Refused by name otherwise: one
    pair, one order."""
    try:                                        # inside molbuilder
        from ..atom_permutation import (AtomIndexError, AtomIndexMap,
                                        LISTED_BY_ENGINE)
    except ImportError:                         # beside a job, in a bundle
        from atom_permutation import (  # type: ignore
            AtomIndexError, AtomIndexMap, LISTED_BY_ENGINE)
    try:
        atoms = AtomIndexMap.from_dict(block, n_atoms=struct.n_atoms)
    except AtomIndexError as exc:
        raise MolstructJsonError(str(exc)) from exc
    record = struct.atom_order
    if record is not None or atoms.listed_by == LISTED_BY_ENGINE:
        want = AtomIndexMap.of_engine_record(record, struct.n_atoms)
        agrees = (atoms.listed_by == LISTED_BY_ENGINE
                  and (atoms.user, atoms.engine) == (want.user, want.engine))
    else:
        agrees = True
    if not agrees:
        raise MolstructJsonError(
            "the sidecar's atom_index_map states another order than its "
            "atom_order record -- one pair, one order")


def load(sidecar_path: Union[str, Path], *,
         retired_out: "Dict[str, Any] | None" = None,
         stated_out: "set | None" = None) -> Dict[str, Any]:
    """THE READ DOOR of a sidecar file: its text read, then its one reader
    (``parse.sidecars.molstruct.load_text`` -- the version gate, the
    stray-key gate, then :func:`to_dict`), the normalised payload back, its
    version the file's.  An unreadable file raises
    :class:`MolstructJsonError` naming it.  ``retired_out`` is the
    reader's (a retired key that carried a value), and ``stated_out``
    (every metadata field the file states, a null one among them).  The
    reader is imported here, not at load: it imports this module.
    """
    from molbuilder.parse.sidecars.molstruct import load_text
    sidecar_path = Path(sidecar_path)
    try:
        # ``encoding="utf-8-sig"`` accepts an optional BOM (some
        # Windows editors insert one) and decodes as UTF-8.
        text = sidecar_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise MolstructJsonError(
            f"failed to read sidecar {sidecar_path}: {exc}"
        ) from exc
    return load_text(text, source=str(sidecar_path), retired_out=retired_out,
                     stated_out=stated_out)


__all__ = [
    "SCHEMA_VERSION",
    "READABLE_VERSIONS",
    "frozen_atoms",
    "dumps",
    "geometry_hash",
    "is_geometry_hash",
    "GEOMETRY_SUFFIXES",
    "geometry_for",
    "is_sidecar",
    "apply_to_structure",
    "load",
    "renewed",
    "stamp",
    "sidecar_path_for",
    "to_dict",
]
