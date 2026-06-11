"""Sidecar ``.molstruct.json`` reader / writer.

``Structure`` carries two transport-oriented attributes (``regions``
and ``frozen_atoms``) that are NOT representable in plain XYZ.  The
sidecar JSON files defined here travel alongside a ``.xyz`` and carry
those attributes so the whole picture round-trips cleanly between
the modify tab, the transport script generator, and any external
tools the user invokes between them.

Schema (v3, 2026-05-21) -- renamed the ``fixed_atoms`` key to
``frozen_atoms`` for terminology consistency with the rest of
molbuilder ("frozen" is the canonical term across UI, fields, and
engines; matches the spectroscopy literature).  No backward read of
v1/v2 sidecars -- old files with ``"fixed_atoms"`` will fail to
load (raises :class:`MolstructJsonError`).  Re-export from /modify
to produce a v3 sidecar::

    {
      "schema_version": 3,
      "n_atoms_total":  42,
      "structure_hash": "sha256-of-the-xyz-file-bytes",  # hex
      "regions": {
        "L-electrode": [0, 1, 2, ...],
        "R-electrode": [30, 31, ...],
        "bridge":      [12, 13, ...]
      },
      "frozen_atoms":  [0, 1, 2, ...],     # 0-based, sorted, unique
      "selection_rules": {                  # optional
        "L-electrode": {"op": "first_n",
                        "rule": {"op": "by_element", "elements": ["Au"]},
                        "n": 12},
        "R-electrode": {"op": "first_n", ...},
        "bridge":      {"op": "minus", "a": {"op": "all"},
                        "b": {"op": "or", "operands": [...]}},
        "frozen_atoms": {"op": "by_element", "elements": ["Au"]}
      },
      "created_by":    "molbuilder modify" | "user-edited" | "import",
      "created_at":    "2026-05-20T14:23:00Z"
    }

The ``structure_hash`` field pins which XYZ this sidecar was written
against.  When the transport script generator (or any consumer)
loads both files, it verifies the hash matches the on-disk XYZ.  A
mismatch surfaces as a clear "regions are stale -- re-export from
modify" error rather than the silent drift of "labels point at
atoms that aren't there any more".

The ``regions`` and ``frozen_atoms`` fields are OPTIONAL on disk
(empty / missing means "no labels were assigned").  Every other
field is required.

This module does NOT depend on :mod:`molbuilder.structure` directly --
it just reads / writes the JSON.  The :func:`apply_to_structure`
helper is the one place we touch ``Structure``, and it accepts any
object with mutable ``regions`` / ``frozen_atoms`` attributes (i.e.,
a duck-typed protocol).
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
# ``with_lock`` context manager below degrades to a no-op so the rest
# of the module still works -- correctness of concurrent saves is
# only guaranteed where ``fcntl`` is present.  Web-server deployments
# are POSIX in practice (Linux / macOS via Werkzeug or gunicorn); the
# fallback path exists so unit tests / dev tools running on Windows
# don't fail to import.
try:
    import fcntl as _fcntl
    _HAVE_FLOCK = True
except ImportError:                  # pragma: no cover - Windows-only branch
    _fcntl = None
    _HAVE_FLOCK = False

SCHEMA_VERSION = 3

# Only the current schema version loads.  v1/v2 sidecars (which used
# the older ``fixed_atoms`` key name) must be re-exported from /modify
# to produce a v3 file.
_READABLE_SCHEMA_VERSIONS = (3,)

# Canonical sidecar suffix.  ``<job>.xyz`` -> ``<job>.molstruct.json``.
_SIDECAR_SUFFIX = ".molstruct.json"


class MolstructJsonError(ValueError):
    """Sidecar JSON is malformed, refers to a different structure, or
    fails an invariant check.  Distinct exception type so callers
    can differentiate "user-error sidecar" from "I/O failure"."""


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def sidecar_path_for(xyz_path: Union[str, Path]) -> Path:
    """Return the canonical sidecar path for ``xyz_path``.

    Strips the LAST suffix and appends ``.molstruct.json``.  Examples::

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
    """Current UTC time as ISO-8601 with explicit Z suffix.  ``utcnow``
    is deprecated in Python 3.12+; use ``datetime.now(timezone.utc)``
    and format manually for ``Z`` semantics."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------- #
#  Serialise                                                            #
# --------------------------------------------------------------------- #


def to_dict(
    *,
    n_atoms_total: int,
    structure_hash: str,
    regions: Optional[Dict[str, List[int]]] = None,
    frozen_atoms: Optional[List[int]] = None,
    selection_rules: Optional[Dict[str, Any]] = None,
    created_by: str = "molbuilder",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the sidecar dict in canonical form.  Doesn't write
    anywhere; use :func:`save` for that.

    Indices in ``regions`` are sorted + deduped per region (mirrors
    Structure's own normalisation).  ``frozen_atoms`` is sorted +
    deduped globally.  ``selection_rules`` is an optional dict
    keyed by region label (or the literal ``"frozen_atoms"``) whose
    values are JSON rule trees from :mod:`molbuilder.selection`;
    the rule tree is validated by re-parsing it through
    :func:`molbuilder.selection.from_json` so a malformed recipe is
    caught at sidecar-build time, not at engine-load time.
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

    # Region membership is NOT mutually exclusive (an atom may carry
    # multiple labels).  Engines that need a disjoint partition
    # (e.g. 2-terminal transport: L-electrode / R-electrode / bridge)
    # check that separately at engine-load time.
    normed_regions: Dict[str, List[int]] = {}
    if regions:
        for name, idxs in regions.items():
            if not isinstance(name, str) or not name:
                raise MolstructJsonError(
                    f"region label must be non-empty string; got "
                    f"{name!r}"
                )
            unique = sorted({int(i) for i in idxs})
            for idx in unique:
                if not 0 <= idx < n_atoms_total:
                    raise MolstructJsonError(
                        f"regions[{name!r}]: atom index {idx} out of "
                        f"range [0, {n_atoms_total})"
                    )
            normed_regions[name] = unique

    normed_frozen: List[int] = []
    if frozen_atoms:
        unique = sorted({int(i) for i in frozen_atoms})
        for idx in unique:
            if not 0 <= idx < n_atoms_total:
                raise MolstructJsonError(
                    f"frozen_atoms: atom index {idx} out of range "
                    f"[0, {n_atoms_total})"
                )
        normed_frozen = unique

    normed_rules: Dict[str, Any] = {}
    if selection_rules:
        # Import locally to keep the parser module's import graph
        # narrow -- selection.py is a leaf utility module and we
        # only need it here if the caller is using selection
        # rules at all.
        from molbuilder.selection import from_json as _rule_from_json
        from molbuilder.selection import to_json as _rule_to_json
        from molbuilder.selection import SelectionError
        valid_targets = set(normed_regions) | {"frozen_atoms"}
        for target, rule_payload in selection_rules.items():
            if not isinstance(target, str) or not target:
                raise MolstructJsonError(
                    f"selection_rules: target label must be non-empty "
                    f"string; got {target!r}"
                )
            if target not in valid_targets:
                raise MolstructJsonError(
                    f"selection_rules: target {target!r} doesn't match "
                    f"any region or 'frozen_atoms' (known: "
                    f"{sorted(valid_targets)!r})"
                )
            try:
                rule = _rule_from_json(rule_payload)
            except SelectionError as exc:
                raise MolstructJsonError(
                    f"selection_rules[{target!r}]: invalid rule: {exc}"
                ) from exc
            # Re-serialise so the stored form is normalised (in case
            # the caller built it by hand with stray keys).
            normed_rules[target] = _rule_to_json(rule)

    return {
        "schema_version":  SCHEMA_VERSION,
        "n_atoms_total":   n_atoms_total,
        "structure_hash":  structure_hash,
        "regions":         normed_regions,
        "frozen_atoms":    normed_frozen,
        "selection_rules": normed_rules,
        "created_by":      str(created_by),
        "created_at":      created_at or _now_iso_z(),
    }


@contextlib.contextmanager
def with_lock(sidecar_path: Union[str, Path]) -> "Iterator[None]":
    """Serialise concurrent read-modify-write cycles on a sidecar.

    The caller's pattern is:

        with molstruct_json.with_lock(sidecar_path):
            existing = (molstruct_json.load(sidecar_path)
                        if sidecar_path.exists() else {})
            new = {**existing, ...}                   # mutate
            molstruct_json.save(sidecar_path, new)

    Two concurrent ``/api/selection/save`` calls otherwise hit a
    classic lost-update race:

      * call A reads the sidecar (state X),
      * call B reads the sidecar (also state X),
      * A mutates to X+a in memory + writes,
      * B mutates to X+b in memory + writes.

    Call B's write clobbers call A's: only ``X+b`` lands on disk;
    A's update is silently lost.  Holding an exclusive flock across
    the whole read-modify-write window forces B to wait until A's
    write commits + the lock releases, so B reads ``X+a`` and writes
    ``X+a+b`` instead.

    Implementation notes
    --------------------

    * The lock target is a sibling file ``<sidecar>.lock`` (NOT the
      sidecar itself).  ``save()`` atomic-replaces the sidecar via
      ``os.replace``, which swaps inodes -- a lock held on the OLD
      sidecar's fd does NOT carry to the new file, so subsequent
      readers would re-race.  Locking a separate file that nobody
      replaces keeps the serialisation honest across saves.
    * ``flock`` is process-wide and advisory; both processes must
      voluntarily acquire it.  Every writer in the codebase (just
      this endpoint today) is responsible for entering this CM.
      Other readers / writers that don't participate can race.
    * The lock file is created on demand + left behind after the
      block exits.  No cleanup: another save is likely to follow
      and need the same file.  An empty 0-byte ``.lock`` file in
      the sidecar dir is the visible cost.
    * On platforms without ``fcntl`` (Windows), the CM is a no-op
      and concurrent saves remain racy.  The intended deployment
      target is POSIX (Linux containers, macOS dev); the no-op
      fallback exists so the module still imports.
    """
    sidecar_path = Path(sidecar_path).resolve()
    if not _HAVE_FLOCK:
        yield
        return
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = sidecar_path.with_suffix(sidecar_path.suffix + ".lock")
    # ``O_RDWR | O_CREAT`` so the file exists when we lock it; nothing
    # is written.  ``O_CLOEXEC`` so a forked child doesn't inherit the
    # lock (which would let a runaway subprocess hold up the parent's
    # next save indefinitely).
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
    finally:
        os.close(fd)


def save(
    sidecar_path: Union[str, Path],
    payload: Dict[str, Any],
) -> Path:
    """Write a canonical sidecar to disk.  Returns the resolved path.

    Atomic-write via tempfile + os.replace so a crash mid-write
    doesn't leave a partial JSON the next reader chokes on.

    NB: this function does NOT take the sidecar lock -- if you're
    doing a read-modify-write cycle, wrap the entire cycle in
    :func:`with_lock`.  Calling ``save`` in isolation (one writer,
    no concurrent updates) is safe without the lock.
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
        # as spectra_json + transport_json.
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2, sort_keys=False,
                       ensure_ascii=False, allow_nan=False)
            fh.write("\n")
            fh.flush()
            # fsync before os.replace so a crash between fclose() and
            # the rename can't leave the OS write buffer holding the
            # only copy of the new bytes (orphan .tmp would also block
            # the next save).  Matches spectra_json + transport_json.
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
#  Deserialise                                                          #
# --------------------------------------------------------------------- #


def load(sidecar_path: Union[str, Path]) -> Dict[str, Any]:
    """Read + validate a sidecar JSON.  Returns the parsed dict.

    Validation is strict: missing required fields, wrong types, out-
    of-range indices, and unknown schema version all raise
    :class:`MolstructJsonError` with a message that points at the
    problem.

    Region overlap (one atom in two regions) is INTENTIONALLY
    permitted -- the data model is multi-label freeform tagging.
    Engines that need a disjoint partition validate that themselves
    at engine-load time; see ``Structure._validate_regions``.

    The on-disk schema is permissive about ``regions`` and
    ``frozen_atoms`` (missing == empty); everything else is required.
    """
    sidecar_path = Path(sidecar_path)
    try:
        # ``encoding="utf-8-sig"`` accepts an optional BOM (some
        # Windows editors insert one) and decodes as UTF-8 — without
        # the explicit encoding, Python defaults to the platform
        # locale's encoding, which mojibakes non-ASCII region labels.
        text = sidecar_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise MolstructJsonError(
            f"failed to read sidecar {sidecar_path}: {exc}"
        ) from exc
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError as exc:
        raise MolstructJsonError(
            f"sidecar {sidecar_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: top-level value must be an "
            f"object; got {type(data).__name__}"
        )

    sv = data.get("schema_version")
    if sv not in _READABLE_SCHEMA_VERSIONS:
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: schema_version is {sv!r}; "
            f"this molbuilder build reads versions "
            f"{list(_READABLE_SCHEMA_VERSIONS)!r}"
        )

    for key in ("n_atoms_total", "structure_hash"):
        if key not in data:
            raise MolstructJsonError(
                f"sidecar {sidecar_path} missing required field "
                f"{key!r}"
            )

    n = data["n_atoms_total"]
    if not isinstance(n, int) or n < 0:
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: n_atoms_total must be a non-"
            f"negative int; got {n!r}"
        )

    sh = data["structure_hash"]
    if not isinstance(sh, str) or len(sh) < 16:
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: structure_hash must be a hex "
            f"string of >= 16 chars; got {sh!r}"
        )

    # Re-validate regions + frozen_atoms via to_dict()'s validators.
    # This catches malformed user-edited JSON BEFORE any consumer
    # tries to apply the data to a Structure (where the same checks
    # would run in __post_init__, but with a less specific error
    # message that doesn't mention which file).
    try:
        return to_dict(
            n_atoms_total   = n,
            structure_hash  = sh,
            regions         = data.get("regions"),
            frozen_atoms    = data.get("frozen_atoms"),
            selection_rules = data.get("selection_rules"),
            created_by      = data.get("created_by", "unknown"),
            created_at      = data.get("created_at"),
        )
    except MolstructJsonError as exc:
        # Re-raise with the source path so the user knows which
        # sidecar is at fault when they have several open.
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: {exc}"
        ) from exc


# --------------------------------------------------------------------- #
#  Apply to a Structure                                                 #
# --------------------------------------------------------------------- #


def apply_to_structure(struct, sidecar_data: Dict[str, Any]) -> None:
    """Copy ``sidecar_data``'s ``regions`` + ``frozen_atoms`` onto
    ``struct``.  Validates that the sidecar's ``n_atoms_total``
    matches the structure's atom count -- a mismatch usually means
    the user edited the XYZ separately and the sidecar's indices no
    longer point at the right atoms.

    The sidecar's ``structure_hash`` is NOT verified here -- the
    caller (typically the transport script generator) compares it
    against the on-disk XYZ's hash with a path it knows about.
    Doing the hash check here would couple this module to the XYZ
    file location.
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
    # Assigning via setattr respects Structure's __post_init__-style
    # validation if the caller decides to invoke it; here we trust
    # to_dict()'s normalisation already ran during load().
    struct.regions      = dict(sidecar_data.get("regions") or {})
    struct.frozen_atoms = list(sidecar_data.get("frozen_atoms") or [])


__all__ = [
    "SCHEMA_VERSION",
    "MolstructJsonError",
    "apply_to_structure",
    "load",
    "save",
    "sha256_of_file",
    "sidecar_path_for",
    "to_dict",
]
