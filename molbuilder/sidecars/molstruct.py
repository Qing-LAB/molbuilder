"""``.molstruct.json`` sidecar — write-side + consumer helpers.

H2 of parse-module.md migration: absorbed from the legacy
:mod:`molbuilder.parsers.molstruct_json` module's write-side
surface.  The read-side (``load`` + ``_normalised_dict``) lives in
:mod:`molbuilder.parse.sidecars.molstruct` per the parse-module
contract.

Public surface here
-------------------

* :data:`SCHEMA_VERSION`            — current on-disk schema (3).
* :exc:`MolstructJsonError`         — raised on malformed input or
  invariant violations.  Canonical home; the read-side re-imports.
* :func:`sidecar_path_for`          — canonical ``<stem>.molstruct
  .json`` derivation.
* :func:`sha256_of_file`            — content hash for the
  ``structure_hash`` field.
* :func:`to_dict`                   — build the canonical sidecar
  dict from raw fields.  Normalises + validates.
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
import math as _math
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


SCHEMA_VERSION = 4

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


def normalise_cell_pbc(
    cell: Optional[Any], pbc: Optional[Any],
) -> "tuple[Optional[List[List[float]]], List[bool]]":
    """Validate + canonicalise the optional periodic lattice.

    ``cell`` is a 3x3 of lattice vectors (rows, Angstrom) or None;
    ``pbc`` is a length-3 of per-axis periodicity bools or None.
    Returns ``(cell_out, pbc_out)`` where ``cell_out`` is a list of 3
    lists of 3 floats (JSON-friendly) or None, and ``pbc_out`` is a
    list of 3 bools (defaults to all-periodic when a cell is present,
    all-False otherwise).  Additive-optional in schema v3 — absence is
    a valid (non-periodic) structure.
    """
    cell_out: Optional[List[List[float]]] = None
    if cell is not None:
        try:
            rows = [[float(x) for x in row] for row in cell]
        except (TypeError, ValueError) as exc:
            raise MolstructJsonError(
                f"cell must be a 3x3 numeric matrix; got {cell!r} ({exc})"
            ) from exc
        if len(rows) != 3 or any(len(r) != 3 for r in rows):
            raise MolstructJsonError(
                f"cell must be 3x3 (3 lattice-vector rows); got "
                f"{len(rows)} row(s)"
            )
        if any(not _math.isfinite(x) for r in rows for x in r):
            raise MolstructJsonError("cell entries must all be finite")
        # Reject a singular/degenerate lattice (zero volume / parallel
        # vectors) -- it breaks downstream reciprocal-space math.  3x3
        # determinant via the rule of Sarrus (avoid a numpy dependency
        # in this stdlib-light sidecar module).
        a, b, c = rows
        det = (a[0] * (b[1] * c[2] - b[2] * c[1])
               - a[1] * (b[0] * c[2] - b[2] * c[0])
               + a[2] * (b[0] * c[1] - b[1] * c[0]))
        if abs(det) < 1e-8:
            raise MolstructJsonError(
                "cell is singular/degenerate (near-zero volume); the "
                "three lattice vectors must be linearly independent")
        cell_out = rows
    if pbc is None:
        pbc_out = [cell_out is not None] * 3
    else:
        try:
            pbc_out = [bool(b) for b in pbc]
        except TypeError as exc:
            raise MolstructJsonError(
                f"pbc must be a length-3 list of bools; got {pbc!r}"
            ) from exc
        if len(pbc_out) != 3:
            raise MolstructJsonError(
                f"pbc must have exactly 3 entries (one per axis); got "
                f"{len(pbc_out)}"
            )
    return cell_out, pbc_out


def to_dict(
    *,
    n_atoms_total: int,
    structure_hash: str,
    regions: Optional[Dict[str, List[int]]] = None,
    frozen_atoms: Optional[List[int]] = None,
    selection_rules: Optional[Dict[str, Any]] = None,
    cell: Optional[Any] = None,
    pbc: Optional[Any] = None,
    annotations: Optional[Dict[str, Any]] = None,
    created_by: str = "molbuilder",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the sidecar dict in canonical form.  Doesn't write
    anywhere; use :func:`save` for that.

    Indices in ``regions`` are sorted + deduped per region (mirrors
    Structure's own normalisation).  ``frozen_atoms`` is sorted +
    deduped globally.  ``selection_rules`` is an optional dict keyed
    by region label (or the literal ``"frozen_atoms"``) whose values
    are JSON rule trees from :mod:`molbuilder.selection`; the tree
    is validated by re-parsing it through :func:`from_json` so a
    malformed recipe is caught at sidecar-build time, not at
    engine-load time.
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
    # multiple labels).  Engines that need a disjoint partition check
    # separately at engine-load time.
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
        # Import locally to keep the import graph narrow -- selection
        # is a leaf utility we only need when the caller is using
        # selection rules at all.
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

    normed_cell, normed_pbc = normalise_cell_pbc(cell, pbc)

    # Extensible per-atom annotation channels (schema v4; atom-annotations.md
    # § 3).  Additive alongside regions/frozen_atoms (which are still written
    # -- "dual-write" so v3 readers keep working).  Accepts a mapping of
    # {name: AtomChannel} (serialised) or already-JSON channel dicts.
    normed_annotations: Dict[str, Any] = {}
    if annotations:
        from molbuilder.structure import AtomChannel as _AtomChannel
        for name, ch in annotations.items():
            if isinstance(ch, _AtomChannel):
                normed_annotations[name] = ch.to_json()
            elif isinstance(ch, dict) and "kind" in ch:
                # round-trip a raw channel dict through the type so a
                # hand-built one is normalised + validated.
                normed_annotations[name] = _AtomChannel.from_json(ch).to_json()
            else:
                raise MolstructJsonError(
                    f"annotations[{name!r}] must be an AtomChannel or a "
                    f"channel dict with a 'kind'; got {type(ch).__name__}")

    return {
        "schema_version":  SCHEMA_VERSION,
        "n_atoms_total":   n_atoms_total,
        "structure_hash":  structure_hash,
        "regions":         normed_regions,
        "frozen_atoms":    normed_frozen,
        "selection_rules": normed_rules,
        "cell":            normed_cell,
        "pbc":             normed_pbc,
        "annotations":     normed_annotations,
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
    """Copy ``sidecar_data``'s ``regions`` + ``frozen_atoms`` onto
    ``struct``.  Validates that the sidecar's ``n_atoms_total``
    matches the structure's atom count — a mismatch usually means
    the user edited the XYZ separately and the sidecar's indices no
    longer point at the right atoms.

    The sidecar's ``structure_hash`` is NOT verified here — the
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
    struct.regions      = dict(sidecar_data.get("regions") or {})
    struct.frozen_atoms = list(sidecar_data.get("frozen_atoms") or [])

    # Extensible annotation channels (schema v4; atom-annotations.md § 3).
    # Absent on a v3 sidecar -> leave struct.annotations empty (regions +
    # frozen above already carry the built-ins).
    ann_raw = sidecar_data.get("annotations")
    if ann_raw:
        from molbuilder.structure import annotations_from_json
        struct.annotations = annotations_from_json(ann_raw)

    # Periodic lattice (additive-optional, v3).  Absent / null cell ->
    # leave the structure non-periodic.  Re-validated through the same
    # normaliser the writer uses, so a hand-edited sidecar fails here
    # with a clear message rather than deep in an emitter.
    cell_raw = sidecar_data.get("cell")
    pbc_raw  = sidecar_data.get("pbc")
    norm_cell, norm_pbc = normalise_cell_pbc(cell_raw, pbc_raw)
    if norm_cell is not None:
        import numpy as _np
        struct.cell = _np.asarray(norm_cell, dtype=float)
        struct.pbc  = tuple(norm_pbc)
    elif pbc_raw is not None:
        # pbc without a cell is unusual but legal (records intent).
        struct.pbc = tuple(norm_pbc)


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


__all__ = [
    "SCHEMA_VERSION",
    "MolstructJsonError",
    "apply_to_structure",
    "load",
    "save",
    "sha256_of_file",
    "sidecar_path_for",
    "to_dict",
    "with_lock",
]
