"""Versioned-document helpers — the shared ``molbuilder/<name>@<major>``
schema convention + JSON IO (`execution/job-contracts.md` § 6).

The one place the major-version check lives, for every persisted artifact.

L1: pure stdlib, no molbuilder deps -- any layer may use it.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def schema_major(schema: str) -> str:
    """The major token of a ``molbuilder/<name>@<major>`` string, or ``""``
    when there is no ``@`` (so a bare/garbage value never matches a real
    major).

    A dotted version keeps only what precedes the first dot, so ``@1.4``
    has major ``1``.  ``job-contracts.md`` § 6.1 states the rule the whole
    convention rests on -- *"checked major-only, tolerating same-major minor
    bumps, rejecting a different major"*.
    """
    s = str(schema or "")
    if "@" not in s:
        return ""
    return s.rsplit("@", 1)[-1].split(".", 1)[0]


def schema_name(schema: str) -> str:
    """The name half of ``molbuilder/<name>@<major>`` -- everything before
    the last ``@``, or ``""`` when there is no ``@`` (so a bare/garbage
    value never matches a real name)."""
    s = str(schema or "")
    if "@" not in s:
        return ""
    return s.rsplit("@", 1)[0]


def check_schema(got: str, want: str, *, label: str = "") -> None:
    """Raise :class:`ValueError` unless ``got`` and ``want`` name the SAME
    artifact at the same major (tolerate same-major minor bumps, reject a
    different major).  ``label`` prefixes the message so the artifact is
    obvious.

    The single enforcement point for the schema convention.  The NAME half
    is what says which artifact this is: comparing majors alone, any ``@1``
    artifact would parse as any other, and ``task.json`` handed to the
    Environment reader would sail through the gate and fail later."""
    if (schema_name(got) != schema_name(want)
            or schema_major(got) != schema_major(want)):
        prefix = f"{label} " if label else ""
        raise ValueError(
            f"{prefix}schema mismatch: got {got!r}, need "
            f"{schema_name(want)}@{schema_major(want)} "
            f"(same name, same major; minors tolerated).")


def read_json(path) -> Any:
    """Read + parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stage_bytes(target, data: bytes, *, mode: Optional[int] = None,
                sync: bool = False) -> Path:
    """``data`` written to a UNIQUE temp beside ``target`` -- :func:`write_bytes`'
    first half, and the whole of a writer that puts several files in place
    together (a structure's pair, `workingcopy_structure`): each staged, then
    each renamed.  Its mode is ``mode``, else the target's own, else 0644;
    ``sync`` flushes it to the disk before it is returned -- a filesystem
    that refuses (tmpfs on some kernels) still has the bytes before the
    rename.  The caller renames it into place or removes it."""
    target = Path(target)
    if mode is None:
        try:
            mode = target.stat().st_mode & 0o777
        except OSError:
            mode = 0o644
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent),
                                    prefix=target.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            if sync:
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
        os.chmod(tmp, mode)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return tmp


def write_bytes(target, data: bytes, *, mode: Optional[int] = None,
                exclusive: bool = False, sync: bool = False) -> Path:
    """Write via a **unique** temp + ``os.replace`` — the one writer
    (`checkpoint._atomic_write_bytes` delegates here).

    Two properties: a reader never sees a partial file, and **two writers
    never collide**.  A DERIVED temp name (``<target>.tmp``) is the trap `checkpointing.md` § 6
    names: two concurrent writers agree on one temp path, one renames it into
    place, and the other's ``os.replace`` either fails on a file that is no
    longer there or — worse — installs the other writer's half-written bytes.
    A unique name reduces the shared moment to the rename itself, which is
    atomic and where last-writer-wins is the right answer.

    **The temp is staged beside its target, always**, so the one rename
    stays inside one directory: a rename across directories is refused by
    some filesystems molbuilder runs on (Sol's `/scratch` refused the one
    molbuilder made, 2026-10-20).  A crash between write and rename leaves
    the unique temp behind as inert litter beside its target (a derived name
    would also poison the NEXT writer), and a checkpoint saves it as any
    file in the folder (`checkpointing.md` S1).

    ``mkstemp`` creates 0600, which is not what a shared artifact should end
    up as, so BY DEFAULT the mode is the target's own if it already exists and
    0644 if it does not — what an ordinary create under a normal umask gives.

    **``mode`` is how a CREDENTIAL is written** (`configuration.md` § 2.3).
    Pass ``0o600`` and the widening above is skipped: ``mkstemp`` already made
    the file owner-only, so there is no moment at any other mode at all, the
    target is never opened for writing — a crash cannot leave a truncated
    secret — and a symlink planted at the path is REPLACED rather than
    followed.

    **``exclusive`` is for a file that must never be REPLACED** -- it creates
    the name or raises ``FileExistsError``, and the caller reads back what is
    already there.  Last-writer-wins is right for an artifact a person is
    re-saving; it is wrong for the session key, where replacing the file logs
    out everyone signed in.  A parameter rather than a second writer, because a
    second writer is what § 2.3 was written to stop.  The rename becomes
    ``os.link``, which is equally atomic, does not follow a symlink planted at
    the target, and fails rather than clobbers.  It is a LINK, which cannot
    cross a filesystem -- and never has to, the temp being beside its
    target.

    ``sync`` flushes the bytes to the disk before the rename
    (:func:`stage_bytes`) -- a structure's files, which a crash must not
    leave holding nothing.
    """
    target = Path(target)
    tmp = stage_bytes(target, data, mode=mode, sync=sync)
    try:
        if exclusive:
            # CREATES the name or fails; never replaces.  `os.replace` below
            # would silently destroy the file this call was told not to touch.
            os.link(str(tmp), str(target))
        else:
            os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    if exclusive:
        # `os.link` leaves the temp behind under its own name -- it is a
        # second link to the same inode, not a rename.  Unlinking it after the
        # target exists cannot lose the data.
        try:
            tmp.unlink()
        except OSError:
            pass
    return target


def json_safe(obj: Any) -> Any:
    """``obj`` with every number JSON cannot hold -- NaN, an infinity --
    as ``None``: *not stated*.  A reader decodes a Fortran field the engine
    overflowed as NaN so the rest of its row survives
    (`siesta_grammar.fortran_float`); JSON has no NaN, and a strict reader
    -- a browser's ``JSON.parse`` -- refuses the token, so on the way out
    it is the null every JSON reader takes, and a plot draws it as a gap.
    THE ONE RULE: both JSON doors ask it (:func:`json_text` for a file, the
    app's JSON provider for a route), the trajectory's wire form asks it,
    and so does a reader that reasons on an engine's numbers before they
    leave (the transport record).  Numpy's scalars and arrays are numbers
    too (``float32`` is no ``float``), walked without importing numpy."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, numbers.Real) and not isinstance(obj, (bool, int)):
        try:
            return obj if math.isfinite(float(obj)) else None
        except (TypeError, ValueError, OverflowError):
            return obj
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        try:
            return json_safe(obj.tolist())
        except (TypeError, ValueError):
            return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(json_safe(v) for v in obj)
    return obj


def json_text(obj: Any) -> str:
    """THE byte shape of every JSON artifact this package writes —
    pretty, 2-space indent, trailing newline, and JSON a browser reads
    (:func:`json_safe`).  One spelling:
    :func:`write_json` writes it to disk, and a surface that hands the
    text to another writer reads it from here, so the two roads cannot
    produce different bytes."""
    # The walk only when it is needed: almost every artifact holds no
    # number JSON cannot hold, and `allow_nan=False` says when one does.
    try:
        return json.dumps(obj, indent=2, allow_nan=False) + "\n"
    except ValueError:
        return json.dumps(json_safe(obj), indent=2, allow_nan=False) + "\n"


def write_json(path, obj: Any, *, mode: Optional[int] = None) -> Path:
    """Write ``obj`` as pretty JSON (:func:`json_text`), atomically and
    collision-safely via :func:`write_bytes`.  Serialisation happens
    BEFORE the temp file exists, so an unserialisable object fails
    clean — no temp created, the target untouched.  Returns the path.

    ``mode`` is :func:`write_bytes`'s own: how a JSON file that carries a
    credential is written ``0600`` from its first byte (`configuration.md`
    § 2.3, *"privacy is a PARAMETER of the one writer"*).  A parameter the
    rule names has to exist on every door.
    """
    data = json_text(obj).encode("utf-8")
    return write_bytes(path, data, mode=mode)


def sha256_of(path) -> str:
    """The sha256 of the file at ``path``, hex -- the one digest a record
    states of a file it names (a run in phases' index: each call's deck
    and coordinates, `engines/vibration.md` § 5.11 § 7)."""
    from pathlib import Path
    return sha256_of_bytes(Path(path).read_bytes())


def sha256_of_bytes(data: bytes) -> str:
    """:func:`sha256_of` of bytes already read -- the digest of exactly
    what a reader holds."""
    import hashlib
    return hashlib.sha256(data).hexdigest()


__all__ = ["schema_major", "schema_name", "check_schema", "read_json",
           "json_text", "write_json", "stage_bytes", "write_bytes",
           "sha256_of", "sha256_of_bytes"]
