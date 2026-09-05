"""``.molstruct.json`` sidecar — write-side + consumer helpers.

The ONE DOOR for this sidecar: the write side lives here, and the read
side is re-exported from :mod:`molbuilder.parse.sidecars.molstruct`, so
a caller needs one import for both. Absorbed from the legacy
``molbuilder.parsers.molstruct_json`` (deleted 2026-06-21).  The split
is what ``model/parse.md`` § 4 requires (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

Public surface here
-------------------

* :data:`SCHEMA_VERSION`            — current on-disk schema.  The number
  lives ONLY on the constant below — typed here too, it went stale at
  "(6)" while the constant said 7 (redo 2026-08-12).
* :exc:`MolstructJsonError`         — raised on malformed input or
  invariant violations.  Canonical home; the read-side re-imports.
* :func:`sidecar_path_for`          — canonical ``<stem>.molstruct
  .json`` derivation.
* :func:`sha256_of_file`            — content hash for the
  ``structure_hash`` field.
* :func:`to_dict`                   — build the canonical sidecar
  dict from the metadata FIELDS dict + envelope.  Normalises + validates.
* :func:`with_lock`                 — POSIX advisory lock context-
  manager for read-modify-write cycles.  No-op on Windows.
* :func:`save`                      — atomic write of a canonical
  dict via tempfile + ``os.replace`` + ``fsync``.
* :func:`apply_to_structure`        — copy loaded sidecar payload
  onto a Structure (validates atom-count match).

Concurrency contract
--------------------

Two concurrent ``/api/selection/save`` calls otherwise hit a lost-
update race: A reads X, B reads X, A writes X+a, B writes X+b →
A's update is silently lost.  Wrap the entire read-modify-write
cycle in :func:`with_lock`:

    with sidecars.molstruct.with_lock(sidecar_path):
        existing = sidecars.molstruct.load(sidecar_path) \\
                   if sidecar_path.exists() else {}
        new = {**existing, ...}
        sidecars.molstruct.save(sidecar_path, new)

The lock target is a sibling file ``<sidecar>.lock`` (NOT the
sidecar itself) because :func:`save` swaps inodes via
``os.replace``; a lock held on the old sidecar's fd does NOT
carry to the new file.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib as _hashlib
import json as _json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

# POSIX-only flock.  When fcntl is unavailable (Windows), the
# ``with_lock`` context manager degrades to a no-op so the rest of
# the module still works -- correctness of concurrent saves is only
# guaranteed where ``fcntl`` is present.  Web-server deployments are
# POSIX in practice; the fallback path exists so unit tests / dev
# tools running on Windows don't fail to import.
try:
    import fcntl as _fcntl
    _HAVE_FLOCK = True
except ImportError:                  # pragma: no cover - Windows branch
    _fcntl = None
    _HAVE_FLOCK = False


#: The on-disk sidecar schema.  BUMPED TO 7 (2026-07-31) with the one-label-store
#: change, and READ STRICTLY: this build reads 7 and refuses everything else.
#:
#: WHAT CHANGED IN 7.  ``frozen_atoms`` stopped being a field of its own and
#: became an ORDINARY LABEL in ``regions``, like every other one.  What makes it
#: reserved is not storage: it is a special INTERPRETATION where the context calls
#: for it (SIESTA's ``Geometry.Constraints``) plus one designated accessor to pull
#: that group out (:func:`frozen_atoms` here, ``getFrozen()`` in the browser).  One
#: store, one spelling of the name, interpreted at the end.
#:
#: WHY THE CLEAN BREAK.  Versions 3-6 were left in the readable set, so a v3 file
#: was ACCEPTED and then read by a loader that no longer looks at the old top-level
#: key -- it came back with its frozen atoms silently gone, and the generated
#: SIESTA input carried no ``Geometry.Constraints`` block.  A junction's electrodes
#: were free to relax; the run converged and was wrong.  A version gate that admits
#: a version the code cannot honour is worse than no gate: it turns a loud failure
#: into a quiet one.
SCHEMA_VERSION = 9   # 9 (2026-08-29): + the OPTIONAL `info` block (the
#                      free-form NON-structural store,
#                      archive/2026-09-01-structure-info-plan.md -- additive; absent
#                      means "nothing recorded", and it never enters
#                      structure_hash).
#                    8 (2026-08-20): + the OPTIONAL identity columns
#                      (structure.IDENTITY_FIELDS, canonical-dict spellings,
#                      written only when real -- user ruling: additive
#                      "extra", never conflicting).  A 7 file simply lacks
#                      them and reads as before; an OLDER reader meeting an
#                      8 file that carries them refuses with the stray-key
#                      message, which names exactly what it does not know.

#: The versions THIS build reads whole.  v8 (2026-08-20) only ADDED the
#: optional identity columns, so a v7 file loses nothing read under v8 rules
#: (absent identity = the synthesized defaults, which is what v7 meant).
#: Everything older stores the same facts in DIFFERENT places (v3's top-level
#: ``frozen_atoms``), and partial reads of those are the silent-loss case the
#: strict gate exists for -- so the set widens only when an addition is
#: provably additive.
#:
#: **AND ONLY WHEN A READER HAS BEEN RUN AGAINST A FILE AT EACH VERSION
#: CLAIMED.**  That is the check, not the reasoning: this list ran
#: ``(3, 4, 5, 6)`` on the argument that every bump was additive, and the
#: argument stopped being true at 7 -- which moved ``frozen_atoms`` out of
#: its own top-level key and into ``regions``.  A v3 file passed the gate,
#: and then ``load_text`` -- which reads the keys it NAMES -- never named
#: the old one.  The atoms did not fail to load; **they were never looked
#: for.**  A junction came back with its 50 frozen electrode atoms gone,
#: ``Geometry.Constraints`` vanished from the generated SIESTA input, and
#: the run converged on a structure nobody asked for (2026-07-31).
#:
#: Each version in the set below is read by a test against a real file at
#: that version (``tests/test_molstruct_json.py::TestSchemaVersioning``);
#: adding a fourth without one is how the above happens again.
READABLE_VERSIONS = frozenset({7, 8, 9})

#: The sidecar LAYER's own keys -- everything in a payload that is not a
#: Structure metadata field.  Named so :func:`apply_to_structure` can hand the
#: gate exactly the fields it owns and REFUSE anything that is neither, instead
#: of passing the whole payload and letting unknown keys fall on the floor.
ENVELOPE_KEYS = ("schema_version", "n_atoms_total", "structure_hash",
                 "selection_rules", "created_by", "created_at")

# Canonical sidecar suffix.  ``<job>.xyz`` -> ``<job>.molstruct.json``.
_SIDECAR_SUFFIX = ".molstruct.json"


class MolstructJsonError(ValueError):
    """Sidecar JSON is malformed, refers to a different structure, or
    fails an invariant check.  Distinct exception type so callers can
    differentiate "user-error sidecar" from "I/O failure"."""


class MolstructPairingError(MolstructJsonError):
    """THIS METADATA IS NOT FOR THIS STRUCTURE -- the one condition above
    that no surface may answer leniently.

    `structure-molstruct.md` § 3 states the guards as **"two independent
    guards, deliberately kept separate"**: the atom count, checked here on
    apply, and `structure_hash`, checked by the caller against the geometry
    it loaded.  Both answer the same question -- does this metadata describe
    *these* atoms? -- and the answer is binary, because the metadata is
    indexed by atom position and a near-miss mis-assigns every label after
    the first inserted atom.  So § 2 says a mismatch "is refused, never
    mis-applied", and `apply_to_structure` is documented as never partially
    applying one.

    Separate TYPE, not a separate message, because the surfaces above differ
    on everything else.  A block whose *form* this build cannot read (a
    pre-v7 run's top-level `frozen_atoms`) is a version fact: the in-script
    block's surface answers it by loading without the labels and saying why,
    since refusing there would leave a finished run unopenable and so
    unfixable.  A block for a *different structure* is not a version fact and
    gets no such leniency anywhere.  Until 2026-09-05 the two shared one
    exception type, so a caller that wanted to be lenient about the first had
    no way to stay strict about the second -- and one that tried swallowed
    the count guard whole.
    """


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def sidecar_path_for(xyz_path: Union[str, Path]) -> Path:
    """Return the canonical sidecar path for ``xyz_path``.

    Strips the LAST suffix and appends ``.molstruct.json``::

        relaxed.xyz           -> relaxed.molstruct.json
        job.spectra.xyz       -> job.spectra.molstruct.json
        bridge                -> bridge.molstruct.json    (no suffix)

    ``Path.with_suffix`` can't be chained for compound suffixes
    (it'd replace ``.molstruct`` with ``.json`` on the second call),
    so build the name explicitly from the stem.
    """
    p = Path(xyz_path)
    return p.with_name(p.stem + _SIDECAR_SUFFIX)


def sha256_of_file(path: Union[str, Path]) -> str:
    """Hex SHA-256 of ``path``'s bytes.  Used as the
    ``structure_hash`` invariant pin.  Stable across platforms (we
    hash file CONTENT, not metadata)."""
    h = _hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso_z() -> str:
    """Current UTC time as ISO-8601 with explicit Z suffix.
    ``datetime.utcnow`` is deprecated in 3.12+; use
    ``datetime.now(timezone.utc)`` and format manually for ``Z``
    semantics."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------- #
#  Serialise                                                            #
# --------------------------------------------------------------------- #


# NOTE: the standalone ``normalise_cell_pbc`` / ``normalise_cell_origin`` field
# validators were REMOVED once the metadata contract landed -- cell / pbc /
# axis_kind / vacuum / cell_origin are now validated in exactly ONE place,
# ``Structure.__post_init__`` (reached via ``structure_fields_via_dataclass`` ->
# ``apply_metadata_dict``).  Keeping a second copy here is what let cell_origin
# drift between the write + read paths; there is no second copy now.


def structure_fields_via_dataclass(
    n_atoms_total: int, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate + canonicalise the Structure metadata fields through the ONE
    dataclass authority (`model/structure.md` § 2.2): apply ``raw`` to a scratch
    N-atom :class:`~molbuilder.structure.Structure` -- which validates every
    field exactly as a live structure does -- then read it back normalised via
    ``metadata_to_dict``.  Shared by the write validator (:func:`to_dict`) and
    the read validator (``parse/sidecars/molstruct._normalised_dict``) so the
    two can never enumerate a different field set (the cell_origin drift).
    Raises :class:`MolstructJsonError` on any invalid field.
    """
    from molbuilder.structure import Structure
    import numpy as _np
    n = int(n_atoms_total)
    scratch = Structure(elements=["C"] * n, positions=_np.zeros((n, 3)))
    try:
        scratch.apply_metadata_dict(raw)
    except (ValueError, TypeError) as exc:
        raise MolstructJsonError(str(exc)) from exc
    return scratch.metadata_to_dict()


def frozen_atoms(payload: Optional[Dict[str, Any]]) -> List[int]:
    """The atoms carrying the reserved ``frozen`` label in a sidecar payload.

    THE way to ask a sidecar dict, the same way :attr:`Structure.frozen_atoms`
    is the way to ask a Structure -- one place spells the name, so no caller
    has to.  It reads the label store, because that is the only place the fact
    lives.
    """
    from molbuilder.structure import FROZEN_LABEL   # lazy: same reason as :152
    if not isinstance(payload, dict):
        return []
    regions = payload.get("regions")
    if not isinstance(regions, dict):
        return []
    return sorted({int(i) for i in (regions.get(FROZEN_LABEL) or ())
                   if isinstance(i, int) and not isinstance(i, bool)})


def normalise_selection_rules(
    selection_rules: Optional[Dict[str, Any]],
    valid_regions,
) -> Dict[str, Any]:
    """Validate the sidecar-only ``selection_rules`` map (NOT a Structure field):
    each target must name a normalised label,
    and each value is re-parsed through :mod:`molbuilder.selection` so a
    malformed recipe fails at sidecar-build time.  Shared by the write + read
    validators so the rule schema is enforced in ONE place.
    """
    normed: Dict[str, Any] = {}
    if not selection_rules:
        return normed
    from molbuilder.selection import from_json as _rule_from_json
    from molbuilder.selection import to_json as _rule_to_json
    from molbuilder.selection import SelectionError
    valid_targets = set(valid_regions)
    for target, rule_payload in selection_rules.items():
        if not isinstance(target, str) or not target:
            raise MolstructJsonError(
                f"selection_rules: target label must be non-empty string; "
                f"got {target!r}")
        if target not in valid_targets:
            raise MolstructJsonError(
                f"selection_rules: target {target!r} doesn't match any label "
                f"(known: {sorted(valid_targets)!r})")
        try:
            rule = _rule_from_json(rule_payload)
        except SelectionError as exc:
            raise MolstructJsonError(
                f"selection_rules[{target!r}]: invalid rule: {exc}") from exc
        normed[target] = _rule_to_json(rule)   # re-serialise -> normalised
    return normed


def to_dict(
    fields: Optional[Dict[str, Any]] = None,
    *,
    identity: Optional[Dict[str, Any]] = None,
    n_atoms_total: int,
    structure_hash: str,
    selection_rules: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None,
    created_by: str = "molbuilder",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical sidecar dict from the metadata FIELDS dict + envelope.

    ``fields`` is the metadata field dict -- the SAME shape
    :meth:`Structure.metadata_to_dict` produces (``regions`` / ``frozen_atoms``
    / ``cell`` / ``cell_origin`` / ``pbc`` / ``axis_kind`` / ``vacuum`` /
    ``annotations``).  STRICT type: ``annotations`` are JSON channel dicts, NOT
    ``AtomChannel`` objects -- serialise a live map with
    :func:`molbuilder.structure.annotations_to_json` first.  A subset is fine
    (an absent key -> the Structure default).  This ONE dict-shaped parameter
    matches ``apply_metadata_dict`` / ``apply_to_structure`` -- the whole
    metadata API set speaks the same format.

    The envelope (``schema_version`` / ``n_atoms_total`` / ``structure_hash`` /
    ``created_*``) and the sidecar-only ``selection_rules`` (keyed by region
    label or the literal ``"frozen_atoms"``, each a JSON rule tree validated by
    re-parsing through :func:`molbuilder.selection.from_json`) are layered on.
    Doesn't write anywhere; use :func:`save` for that.
    """
    if not isinstance(n_atoms_total, int) or n_atoms_total < 0:
        raise MolstructJsonError(
            f"n_atoms_total must be a non-negative int; got "
            f"{n_atoms_total!r}"
        )
    if not isinstance(structure_hash, str) or len(structure_hash) < 16:
        raise MolstructJsonError(
            f"structure_hash must be a hex string (got "
            f"{structure_hash!r})"
        )

    # The Structure FIELDS -> validated + canonicalised through the ONE dataclass
    # authority (a scratch N-atom Structure IS the schema).  Shared verbatim with
    # the read side + apply_to_structure, so no field can drift between them.
    fields = structure_fields_via_dataclass(n_atoms_total, fields or {})
    # The OPTIONAL identity columns (schema 8): validated the same way --
    # through the ONE dataclass authority.  A scratch Structure carrying them
    # re-runs __post_init__'s own length/type checks, and what comes back is
    # exactly what a reader will reconstruct; only REAL columns are handed in
    # (Structure.identity_to_dict already skipped the synthesized defaults).
    identity = dict(identity or {})
    if identity:
        from molbuilder.structure import IDENTITY_FIELDS, Structure
        stray_id = [k for k in identity if k not in IDENTITY_FIELDS]
        if stray_id:
            raise MolstructJsonError(
                f"identity carries {sorted(stray_id)!r}; the identity "
                f"columns are {list(IDENTITY_FIELDS)!r}")
        try:
            scratch = Structure(
                elements=["X"] * n_atoms_total,
                positions=[[0.0, 0.0, 0.0]] * n_atoms_total,
                title=identity.get("title", ""),
                atom_names=identity.get("atom_names"),
                residue_ids=identity.get("residue_ids"),
                residue_names=identity.get("residue_names"),
                chain_ids=identity.get("chain_ids"),
            )
        except (ValueError, TypeError) as exc:
            raise MolstructJsonError(str(exc)) from exc
        identity = {k: v for k, v in {
            "title":         scratch.title or "",
            "atom_names":    list(scratch.atom_names),
            "residue_ids":   [int(v) for v in scratch.residue_ids],
            "residue_names": list(scratch.residue_names),
            "chain_ids":     list(scratch.chain_ids),
        }.items() if k in identity and (k != "title" or v)}
    # selection_rules -- a sidecar-only pass-through (not a Structure field),
    # validated against the normalised region set (one shared validator).
    normed_rules = normalise_selection_rules(
        selection_rules, set(fields["regions"]))

    out = {
        # Envelope -- the sidecar LAYER's own keys (not Structure fields).
        "schema_version":  SCHEMA_VERSION,
        "n_atoms_total":   n_atoms_total,
        "structure_hash":  structure_hash,
        # The Structure metadata block, VERBATIM from the ONE codec
        # (metadata_to_dict, via structure_fields_via_dataclass): regions /
        # frozen_atoms / cell / cell_origin / pbc / axis_kind / vacuum /
        # annotations.  Spread -- NOT re-listed -- so a field added to the
        # dataclass rides onto the sidecar automatically and this layer can no
        # longer drop or drift one (`model/structure.md` § 2.2: the ONE
        # serialization authority; add a key there and nowhere else).
        **fields,
        # The identity columns (schema 8) -- absent entirely when nothing is
        # real, so an xyz-born sidecar is byte-identical to a schema-7 one
        # apart from the version stamp.
        **identity,
        # selection_rules -- a sidecar-only pass-through (not a Structure field).
        "selection_rules": normed_rules,
        "created_by":      str(created_by),
        "created_at":      created_at or _now_iso_z(),
    }
    # The `info` block (schema 9): the free-form NON-structural store,
    # written only when something is recorded -- an empty store leaves
    # the file byte-identical to a schema-8 one apart from the stamp.
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


# --------------------------------------------------------------------- #
#  Concurrency                                                          #
# --------------------------------------------------------------------- #


@contextlib.contextmanager
def with_lock(sidecar_path: Union[str, Path]) -> "Iterator[None]":
    """Serialise concurrent read-modify-write cycles on a sidecar.

    Implementation notes:

    * The lock target is a sibling file ``<sidecar>.lock`` (NOT the
      sidecar itself).  :func:`save` atomic-replaces the sidecar via
      ``os.replace``, which swaps inodes — a lock held on the OLD
      sidecar's fd does NOT carry to the new file, so subsequent
      readers would re-race.  Locking a separate file that nobody
      replaces keeps the serialisation honest across saves.
    * ``flock`` is process-wide and advisory; both processes must
      voluntarily acquire it.  Every writer in the codebase is
      responsible for entering this CM; other readers / writers that
      don't participate can race.
    * The lock file is created on demand + left behind after the
      block exits.  No cleanup: another save is likely to follow
      and need the same file.  An empty 0-byte ``.lock`` file in
      the sidecar dir is the visible cost.
    * On platforms without ``fcntl`` (Windows), the CM is a no-op
      and concurrent saves remain racy.
    """
    sidecar_path = Path(sidecar_path).resolve()
    if not _HAVE_FLOCK:
        yield
        return
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = sidecar_path.with_suffix(sidecar_path.suffix + ".lock")
    # ``O_RDWR | O_CREAT`` so the file exists when we lock it; nothing
    # is written.  ``O_CLOEXEC`` so a forked child doesn't inherit
    # the lock (which would let a runaway subprocess hold up the
    # parent's next save indefinitely).
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
    finally:
        os.close(fd)


# --------------------------------------------------------------------- #
#  Save                                                                 #
# --------------------------------------------------------------------- #


def dumps(payload: Dict[str, Any]) -> str:
    """The canonical on-disk TEXT for a sidecar payload -- the ONE serialisation.

    :func:`save` writes exactly this, and anything that needs the same bytes
    without writing them (a download, a comparison, a preview) asks for it here.
    Two serialisers is two answers to "what does this sidecar look like": the
    settings below are not cosmetic -- ``ensure_ascii=False`` is what keeps a
    non-ASCII region label ("α-helix") a literal instead of an escape, so a
    second writer without it produces a different file for the same structure.
    """
    return _json.dumps(payload, indent=2, sort_keys=False,
                       ensure_ascii=False, allow_nan=False) + "\n"


def save(
    sidecar_path: Union[str, Path],
    payload: Dict[str, Any],
) -> Path:
    """Write a canonical sidecar to disk.  Returns the resolved path.

    Atomic-write via tempfile + ``os.replace`` so a crash mid-write
    doesn't leave a partial JSON the next reader chokes on.

    NB: this function does NOT take the sidecar lock — if you're
    doing a read-modify-write cycle, wrap the entire cycle in
    :func:`with_lock`.  Calling :func:`save` in isolation (one
    writer, no concurrent updates) is safe without the lock.
    """
    sidecar_path = Path(sidecar_path).resolve()
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    try:
        # ``encoding="utf-8"`` is REQUIRED: without it Python defaults
        # to the platform locale's encoding, which corrupts non-ASCII
        # region labels (e.g. "α-helix") on cp1252 / latin-1 systems.
        # ``allow_nan=False`` prevents the writer from emitting NaN/Inf
        # tokens (a divergent SCF could write one) — same safety net
        # as spectra + transport sidecars.
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(dumps(payload))
            fh.flush()
            # fsync before os.replace so a crash between fclose() and
            # the rename can't leave the OS write buffer holding the
            # only copy of the new bytes.  Matches spectra + transport.
            try:
                os.fsync(fh.fileno())
            except OSError:
                # Some filesystems (tmpfs on certain kernels) reject
                # fsync — the data is still in the OS buffer + will
                # land before the replace.  Don't let a quirky FS
                # block the write.
                pass
        os.replace(tmp, sidecar_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return sidecar_path


# --------------------------------------------------------------------- #
#  Apply to a Structure                                                 #
# --------------------------------------------------------------------- #


def apply_to_structure(struct, sidecar_data: Dict[str, Any]) -> None:
    """Apply a loaded sidecar payload's metadata onto ``struct`` IN PLACE.

    Delegates the whole field set (regions / frozen_atoms / cell / cell_origin /
    pbc / axis_kind / vacuum / annotations) to
    :meth:`molbuilder.structure.Structure.apply_metadata_dict` -- the SINGLE
    dict->struct authority (`model/structure.md` § 2.2).  Because the writer
    (``Structure.metadata_to_dict``) and this reader share that one method, they
    can no longer drift a field (the class of bug that dropped ``cell_origin`` on
    reload).  ``selection_rules`` is a sidecar-only pass-through (not a Structure
    field) and is intentionally not applied here.

    Validates that the sidecar's ``n_atoms_total`` matches the structure's atom
    count -- a mismatch usually means the XYZ was edited separately and the
    sidecar's indices no longer point at the right atoms.  That one raises
    :class:`MolstructPairingError` rather than the base error, so a lenient
    surface can keep this guard while forgiving the version failures below
    (see that class for why the two must not share an answer).  The sidecar's
    ``structure_hash`` is NOT verified here (the caller compares it against the
    on-disk XYZ's hash with a path it knows about).  A key that is neither a
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
    from molbuilder.structure import IDENTITY_FIELDS, METADATA_FIELDS
    stray = [k for k in sidecar_data
             if k not in METADATA_FIELDS and k not in ENVELOPE_KEYS
             and k not in IDENTITY_FIELDS and k != "info"]
    if stray:
        raise MolstructJsonError(
            f"sidecar carries {sorted(stray)!r}, which is neither a structure "
            f"metadata field {list(METADATA_FIELDS)!r} nor an envelope key "
            f"{list(ENVELOPE_KEYS)!r}.  Refused rather than ignored: a key "
            f"nobody reads is metadata the writer thinks it saved.")
    # Identity first (schema 8), FULL-REPLACE -- the same semantics
    # apply_metadata_dict documents for the metadata block: an absent key
    # resets the field to its default (post_init refills the placeholders).
    # This is what makes the sidecar the identity AUTHORITY for a pair: the
    # xyz half's comment line is provenance, and without the reset the
    # loader's comment-as-title lift would round-trip a "Built by
    # molbuilder" the user never stated into the next sidecar.  Plain
    # attribute sets, so the apply_metadata_dict call below re-runs
    # __post_init__ over them -- one validator, one message.
    for k in IDENTITY_FIELDS:
        if k in sidecar_data:
            setattr(struct, k, sidecar_data[k])
        else:
            setattr(struct, k, "" if k == "title" else None)
    # The info block (schema 9), FULL-REPLACE like everything else here:
    # absent means "nothing recorded", and a stale store must not survive
    # a pair that no longer carries one.
    raw_info = sidecar_data.get("info")
    struct.info = dict(raw_info) if isinstance(raw_info, dict) else {}
    try:
        struct.apply_metadata_dict(
            {k: v for k, v in sidecar_data.items() if k in METADATA_FIELDS})
    except (ValueError, TypeError) as exc:
        # Surface field-validation failures as the sidecar-layer error type,
        # preserving the clear per-field message from the dataclass invariants.
        raise MolstructJsonError(str(exc)) from exc


def load(sidecar_path):
    """Read-side convenience re-export: delegates to
    :func:`molbuilder.parse.sidecars.molstruct._load` so consumers
    have a single ``molbuilder.sidecars.molstruct`` namespace that
    carries the full sidecar surface (read + write).  Local import
    avoids a module-load-time cycle: parse.sidecars.molstruct
    imports MolstructJsonError + SCHEMA_VERSION from THIS module.
    """
    from molbuilder.parse.sidecars.molstruct import _load
    return _load(sidecar_path)


def load_text(text, *, source="<sidecar>"):
    """Read-side convenience re-export: parse + validate sidecar JSON from a
    STRING (not a path).  Delegates to
    :func:`molbuilder.parse.sidecars.molstruct.load_text` so consumers have a
    single ``molbuilder.sidecars.molstruct`` namespace (read + write).  The
    ``/api/build/load`` seam uses this to apply an in-body sidecar whose bytes
    the browser read through the projects file package.
    """
    from molbuilder.parse.sidecars.molstruct import load_text as _load_text
    return _load_text(text, source=source)


__all__ = [
    "SCHEMA_VERSION",
    "READABLE_VERSIONS",
    "frozen_atoms",
    "dumps",
    "MolstructJsonError",
    "MolstructPairingError",
    "apply_to_structure",
    "load",
    "load_text",
    "save",
    "sha256_of_file",
    "sidecar_path_for",
    "to_dict",
    "with_lock",
]
