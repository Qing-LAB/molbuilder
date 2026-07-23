"""``.molstruct.json`` sidecar — write-side + consumer helpers.

H2 of parse-module.md migration: absorbed from the legacy
:mod:`molbuilder.parsers.molstruct_json` module's write-side
surface.  The read-side (``load`` + ``_normalised_dict``) lives in
:mod:`molbuilder.parse.sidecars.molstruct` per the parse-module
contract.

Public surface here
-------------------

* :data:`SCHEMA_VERSION`            — current on-disk schema (6).
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


SCHEMA_VERSION = 6

# Canonical sidecar suffix.  ``<job>.xyz`` -> ``<job>.molstruct.json``.
_SIDECAR_SUFFIX = ".molstruct.json"


class MolstructJsonError(ValueError):
    """Sidecar JSON is malformed, refers to a different structure, or
    fails an invariant check.  Distinct exception type so callers can
    differentiate "user-error sidecar" from "I/O failure"."""


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
    dataclass authority (data-vocabulary.md): apply ``raw`` to a scratch
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


def normalise_selection_rules(
    selection_rules: Optional[Dict[str, Any]],
    valid_regions,
) -> Dict[str, Any]:
    """Validate the sidecar-only ``selection_rules`` map (NOT a Structure field):
    each target must name a normalised region or the literal ``"frozen_atoms"``,
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
    valid_targets = set(valid_regions) | {"frozen_atoms"}
    for target, rule_payload in selection_rules.items():
        if not isinstance(target, str) or not target:
            raise MolstructJsonError(
                f"selection_rules: target label must be non-empty string; "
                f"got {target!r}")
        if target not in valid_targets:
            raise MolstructJsonError(
                f"selection_rules: target {target!r} doesn't match any region "
                f"or 'frozen_atoms' (known: {sorted(valid_targets)!r})")
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
    n_atoms_total: int,
    structure_hash: str,
    selection_rules: Optional[Dict[str, Any]] = None,
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
    # selection_rules -- a sidecar-only pass-through (not a Structure field),
    # validated against the normalised region set (one shared validator).
    normed_rules = normalise_selection_rules(
        selection_rules, set(fields["regions"]))

    return {
        # Envelope -- the sidecar LAYER's own keys (not Structure fields).
        "schema_version":  SCHEMA_VERSION,
        "n_atoms_total":   n_atoms_total,
        "structure_hash":  structure_hash,
        # The Structure metadata block, VERBATIM from the ONE codec
        # (metadata_to_dict, via structure_fields_via_dataclass): regions /
        # frozen_atoms / cell / cell_origin / pbc / axis_kind / vacuum /
        # annotations.  Spread -- NOT re-listed -- so a field added to the
        # dataclass rides onto the sidecar automatically and this layer can no
        # longer drop or drift one (structure-authority.md § 3.4).
        **fields,
        # selection_rules -- a sidecar-only pass-through (not a Structure field).
        "selection_rules": normed_rules,
        "created_by":      str(created_by),
        "created_at":      created_at or _now_iso_z(),
    }


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
            _json.dump(payload, fh, indent=2, sort_keys=False,
                       ensure_ascii=False, allow_nan=False)
            fh.write("\n")
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
    dict->struct authority (data-vocabulary.md).  Because the writer
    (``Structure.metadata_to_dict``) and this reader share that one method, they
    can no longer drift a field (the class of bug that dropped ``cell_origin`` on
    reload).  ``selection_rules`` is a sidecar-only pass-through (not a Structure
    field) and is intentionally not applied here.

    Validates that the sidecar's ``n_atoms_total`` matches the structure's atom
    count -- a mismatch usually means the XYZ was edited separately and the
    sidecar's indices no longer point at the right atoms.  The sidecar's
    ``structure_hash`` is NOT verified here (the caller compares it against the
    on-disk XYZ's hash with a path it knows about).  A pre-v5 ``kgrid`` key is
    ignored (k-grid is a sampling knob on the config, not a geometry field).
    """
    sidecar_n = sidecar_data.get("n_atoms_total")
    struct_n = len(struct.elements)
    if sidecar_n != struct_n:
        raise MolstructJsonError(
            f"sidecar n_atoms_total={sidecar_n} but structure has "
            f"{struct_n} atoms.  The sidecar's region / frozen-atom "
            f"indices no longer point at the right atoms; re-export "
            f"the sidecar from /modify after structural edits."
        )
    try:
        struct.apply_metadata_dict(sidecar_data)
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
    "MolstructJsonError",
    "apply_to_structure",
    "load",
    "load_text",
    "save",
    "sha256_of_file",
    "sidecar_path_for",
    "to_dict",
    "with_lock",
]
