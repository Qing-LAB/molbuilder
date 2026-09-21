"""Versioned-document helpers — the shared ``molbuilder/<name>@<major>``
schema convention + JSON IO (`execution/job-contracts.md` § 6).

The major-version check was hand-rolled identically in three persisted
artifacts (``scheduler/record.py``, ``bench/result.py``, ``jobset/model.py``),
with a subtle inconsistency in how a missing ``@`` was handled.  This is the
one place that logic lives now.

L1: pure stdlib, no molbuilder deps -- any layer may use it.
"""

from __future__ import annotations

import json
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
    bumps, rejecting a different major"* -- and until 2026-08-07 this
    function did not implement its first half: it compared the entire token,
    so ``@1.4`` was rejected against ``@1`` with a message saying the major
    differed when it did not.  Nothing shipped had a minor, so nothing had
    exercised it; ``task@1`` is the first artifact whose reader tests the
    claim.  The change only ever widens acceptance -- no string that passed
    before can fail now.
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

    The single enforcement point for the schema convention -- adopted by
    ``environment@1`` / ``bench-result@1`` / ``job-set@1`` (was duplicated).

    *Named ``check_schema_major`` until U9 (2026-08-12), and the name was
    the bug's alibi: it compared ONLY the majors, so any ``@1`` artifact
    parsed as any other -- ``task.json`` handed to the Environment reader
    sailed through the schema gate and failed later, somewhere the message
    named the wrong thing.  The NAME half is what says which artifact this
    is; a version check that ignores it checks nothing worth the word.*"""
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


def write_bytes(target, data: bytes, *, tmp_dir: Optional[Path] = None,
                mode: Optional[int] = None,
                exclusive: bool = False) -> Path:
    """Write via a **unique** temp + ``os.replace`` — the checkpoint's shape
    (`checkpoint._atomic_write_bytes` carried it first and now delegates
    here), adopted package-wide at U8 (2026-08-12).

    Two properties, and ``write_json`` used to have only the first: a reader
    never sees a partial file, and **two writers never collide**.  The old
    DERIVED temp name (``<target>.tmp``) is the trap `checkpointing.md` § 6
    names: two concurrent writers agree on one temp path, one renames it into
    place, and the other's ``os.replace`` either fails on a file that is no
    longer there or — worse — installs the other writer's half-written bytes.
    A unique name reduces the shared moment to the rename itself, which is
    atomic and where last-writer-wins is the right answer.

    A crash between write and rename leaves the unique temp behind as inert
    litter; with the derived name it also poisoned the NEXT writer.  For a
    target inside a folder a checkpoint save will store, pass ``tmp_dir``
    pointing somewhere never stored (the checkpoint passes ``.git``), so the
    litter cannot be committed into history.

    ``mkstemp`` creates 0600, which is not what a shared artifact should end
    up as, so BY DEFAULT the mode is the target's own if it already exists and
    0644 if it does not — what an ordinary create under a normal umask gives.

    **``mode`` is how a CREDENTIAL is written** (`configuration.md` § 2.3).
    Pass ``0o600`` and the widening above is skipped: ``mkstemp`` already made
    the file owner-only, so there is no moment at any other mode at all, the
    target is never opened for writing — a crash cannot leave a truncated
    secret — and a symlink planted at the path is REPLACED rather than
    followed.  Until 2026-09-12 this package had an atomic writer that widened
    and a private writer (``auth_setup.write_secret_file``) that truncated the
    target in place, and every secret went through the second one; § 2.3 has
    the measurement.

    **``exclusive`` is for a file that must never be REPLACED** -- it creates
    the name or raises ``FileExistsError``, and the caller reads back what is
    already there.  Last-writer-wins is right for an artifact a person is
    re-saving; it is wrong for the session key, where replacing the file logs
    out everyone signed in.  A parameter rather than a second writer, because a
    second writer is what § 2.3 was written to stop.  The rename becomes
    ``os.link``, which is equally atomic, does not follow a symlink planted at
    the target, and fails rather than clobbers.  It is a LINK, so it cannot
    cross a filesystem:
    combining ``exclusive`` with a ``tmp_dir`` on another mount raises
    ``OSError(EXDEV)``.  The default staging is the target's own directory, so
    no present caller can reach that, and the two options are for opposite
    situations anyway -- ``tmp_dir`` keeps litter out of a checkpointed folder,
    and a credential is not in one.
    """
    target = Path(target)
    parent = Path(tmp_dir) if tmp_dir is not None else target.parent
    if not parent.is_dir():
        parent = target.parent
    if mode is None:
        try:
            mode = target.stat().st_mode & 0o777
        except OSError:
            mode = 0o644
    fd, tmp_name = tempfile.mkstemp(dir=str(parent),
                                    prefix=target.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, mode)
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


def json_text(obj: Any) -> str:
    """THE byte shape of every JSON artifact this package writes —
    pretty, 2-space indent, trailing newline.  One spelling:
    :func:`write_json` writes it to disk, and a surface that hands the
    text to another writer (the transport tab's describe, whose browser
    writes through the file layer) reads it from here, so the two roads
    cannot produce different bytes."""
    return json.dumps(obj, indent=2) + "\n"


def write_json(path, obj: Any, *, tmp_dir: Optional[Path] = None,
               mode: Optional[int] = None) -> Path:
    """Write ``obj`` as pretty JSON (:func:`json_text`), atomically and
    collision-safely via :func:`write_bytes`.  Serialisation happens
    BEFORE the temp file exists, so an unserialisable object fails
    clean — no temp created, the target untouched.  Returns the path.

    ``mode`` is :func:`write_bytes`'s own: how a JSON file that carries a
    credential is written ``0600`` from its first byte (`configuration.md`
    § 2.3, *"privacy is a PARAMETER of the one writer"*).  This function did
    not take it until 2026-09-13, and the one caller that needed it -- the
    seeded ``molbuilder.json`` -- worked around the gap by ``touch``-ing the
    target ``0600`` first so the preserve-the-mode branch would do the job
    sideways.  A parameter the rule names has to exist on every door.
    """
    data = json_text(obj).encode("utf-8")
    return write_bytes(path, data, tmp_dir=tmp_dir, mode=mode)


__all__ = ["schema_major", "schema_name", "check_schema", "read_json",
           "json_text", "write_json", "write_bytes"]
