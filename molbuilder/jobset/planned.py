"""What a prep will write, decided before any of it is.

`job-system.md` § 5.0, rule 3: prep makes its whole plan -- every check,
every value, the text of every file, every folder and what goes into it --
with nothing written; a refusal can only come from there, and writes nothing
but its line in the ledger, so there is nothing to put back.  Then the
folder's state is saved, and then the plan is carried out, which decides
nothing and refuses nothing.

A :class:`Plan` is that list: each file's text or bytes, each copy, each
move and each removal, in the order the steps decided them.  A step that
reads what an earlier step will write reads it through the plan -- the
wrapper the deck's text, the attempt the stage's files, the screening the
pseudopotentials -- and the plan answers with what it holds, the disk with
the rest (:meth:`Plan.is_file`, :meth:`Plan.read_bytes`, :meth:`Plan.glob`).
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..deck_record import without_stamps
from ..persist import write_bytes


def _at(path) -> Path:
    """One spelling of a path, for the plan's own bookkeeping: absolute and
    normalised, never resolved -- a link is a file of its own."""
    return Path(os.path.normpath(os.path.abspath(str(path))))


@dataclass(frozen=True)
class _Put:
    path: Path
    data: bytes
    mode: Optional[int]


@dataclass(frozen=True)
class _Copy:
    src: Path
    path: Path
    #: The file on disk its bytes come from, by size and time (:func:`found`),
    #: as it was when the copy was planned -- ``None`` when they are the
    #: plan's own.
    stamp: Optional[str] = None


@dataclass(frozen=True)
class _Move:
    src: Path
    path: Path
    stamp: Optional[str] = None


@dataclass(frozen=True)
class _Remove:
    path: Path


@dataclass(frozen=True)
class _Folder:
    path: Path


#: What a path holds in the plan: bytes, a file on disk it copies, or
#: ``None`` -- removed.
_Held = Union[bytes, Path, None]


def found(path) -> str:
    """A file as it is now, in words -- ``1234 bytes, written 2026-10-05
    10:00:01.123456789``, or ``absent``.  Two readings agree exactly when
    nothing wrote the file between them: what a plan's copy is stamped with,
    and what `launch` reads in place and compares at its send
    (`job-system.md` § 6.0, step 4)."""
    try:
        st = Path(path).stat()
    except FileNotFoundError:
        return "absent"
    secs, ns = divmod(st.st_mtime_ns, 10 ** 9)
    when = datetime.fromtimestamp(secs).strftime("%Y-%m-%d %H:%M:%S")
    return f"{st.st_size} bytes, written {when}.{ns:09d}"


def _stamp(held: _Held) -> Optional[str]:
    """What a copy or a move takes, when it is a file on disk: its size and
    time (:func:`found`), read when the step is planned -- the moment the
    plan describes, whatever a later step of the same plan does to the
    file."""
    if not isinstance(held, Path):
        return None
    return found(held)


class Plan:
    """Every file a prep will write, in order, held until :meth:`carry_out`."""

    def __init__(self) -> None:
        self._ops: list = []
        self._held: Dict[Path, _Held] = {}

    # -- deciding -----------------------------------------------------------

    def text(self, path, text: str, *, mode: Optional[int] = None) -> Path:
        """``path`` will hold ``text``."""
        return self.bytes(path, text.encode("utf-8"), mode=mode)

    def bytes(self, path, data: bytes, *, mode: Optional[int] = None) -> Path:
        """``path`` will hold ``data``."""
        p = _at(path)
        self._ops.append(_Put(p, bytes(data), mode))
        self._held[p] = bytes(data)
        return p

    def copy(self, src, path) -> Path:
        """``path`` will hold a copy of ``src`` as it is now -- planned or on
        disk."""
        s, p = _at(src), _at(path)
        held = self._held.get(s, s)
        if held is None or (isinstance(held, Path) and not held.is_file()):
            raise FileNotFoundError(f"nothing to copy at {src}")
        self._ops.append(_Copy(s, p, _stamp(held)))
        self._held[p] = held
        return p

    def move(self, src, path) -> Path:
        """``src`` will be moved to ``path``."""
        s, p = _at(src), _at(path)
        held = self._held.get(s, s)
        if held is None or (isinstance(held, Path) and not held.is_file()):
            raise FileNotFoundError(f"nothing to move at {src}")
        self._ops.append(_Move(s, p, _stamp(held)))
        self._held[p] = held
        self._held[s] = None
        return p

    def remove(self, path) -> None:
        """``path`` will be gone."""
        p = _at(path)
        self._ops.append(_Remove(p))
        self._held[p] = None

    def folder(self, path) -> Path:
        """``path`` will be a folder, empty if nothing is put in it."""
        p = _at(path)
        self._ops.append(_Folder(p))
        return p

    # -- the folder as it will be -------------------------------------------

    def is_file(self, path) -> bool:
        p = _at(path)
        if p in self._held:
            return self._held[p] is not None
        return p.is_file()

    def read_bytes(self, path) -> bytes:
        p = _at(path)
        held = self._held.get(p, p)
        if held is None:
            raise FileNotFoundError(f"{path} is removed by this plan")
        return held if isinstance(held, bytes) else held.read_bytes()

    def read_text(self, path) -> str:
        return self.read_bytes(path).decode("utf-8")

    def source_of(self, path) -> Optional[Path]:
        """The file on disk that holds ``path``'s bytes now -- the path
        itself, or what it copies or moves -- or ``None`` when its bytes are
        the plan's own (or it is removed)."""
        p = _at(path)
        held = self._held.get(p, p)
        return held if isinstance(held, Path) else None

    def glob(self, folder, pattern: str) -> List[Path]:
        """The files directly in ``folder`` matching ``pattern``, as the
        plan will leave them."""
        f = _at(folder)
        found = {p for p in (f.glob(pattern) if f.is_dir() else ())
                 if p.is_file()}
        for p, held in self._held.items():
            if p.parent == f and fnmatch.fnmatchcase(p.name, pattern):
                if held is None:
                    found.discard(p)
                else:
                    found.add(p)
        return sorted(found)

    def described(self) -> List[str]:
        """Every operation in words, in order: a folder made; a file written,
        by a digest of its bytes with their stamps masked (when and by which
        build it was written -- `deck_record.without_stamps`, the rule two
        decks are compared by) and its mode; a file copied or moved, from
        where, and that file's size and time as they were when the step was
        planned; a file removed.  Two plans of one folder describe the same
        lines exactly when they would write the same thing -- what
        :meth:`identity` names, and what `launch`'s send compares line by
        line (`job-system.md` § 6.0, step 4)."""
        out: List[str] = []
        for op in self._ops:
            if isinstance(op, _Folder):
                out.append(f"makes the folder {op.path}")
            elif isinstance(op, _Remove):
                out.append(f"removes {op.path}")
            elif isinstance(op, _Put):
                digest = hashlib.sha256(without_stamps(op.data)).hexdigest()
                out.append(f"writes {op.path} (its text {digest[:16]}"
                           + ("" if op.mode is None else f", mode {op.mode:o}")
                           + ")")
            else:
                out.append(("copies " if isinstance(op, _Copy) else "moves ")
                           + f"{op.src} to {op.path} "
                           + (f"({op.stamp})" if op.stamp
                              else "(as this plan holds it)"))
        return out

    def identity(self) -> str:
        """What this plan IS, less the moment it was made -- its
        :meth:`described` lines, hashed: two plans of one folder agree
        exactly when they would write the same thing.  The preview's plan is
        named by it, and Prep refuses a plan that differs (`job-system.md`
        § 5.0)."""
        return hashlib.sha256(
            "\n".join(self.described()).encode("utf-8")).hexdigest()

    def writes(self) -> List[Path]:
        """Every path this plan leaves holding a file, in the order it was
        first planned, once each -- a file a later step moves on is listed
        where it lands."""
        out: Dict[Path, None] = {}
        for op in self._ops:
            if (isinstance(op, (_Put, _Copy, _Move))
                    and self._held.get(op.path) is not None):
                out[op.path] = None
        return list(out)

    # -- carrying it out ----------------------------------------------------

    def carry_out(self) -> None:
        """Write the plan, in its order.  Nothing here decides or refuses:
        a write that fails (a full disk) is the only error, and the state
        saved before it is the way back (`job-system.md` § 5.0, 6)."""
        for op in self._ops:
            if isinstance(op, _Folder):
                op.path.mkdir(parents=True, exist_ok=True)
                continue
            if isinstance(op, _Remove):
                if op.path.is_symlink() or op.path.is_file():
                    op.path.unlink()
                continue
            op.path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(op, _Put):
                # THE PACKAGE'S ONE WRITER: a unique temp, renamed into
                # place -- a reader never sees half a file, and a link at
                # the path is replaced, never written through
                # (`persist.write_bytes`).
                write_bytes(op.path, op.data, mode=op.mode)
                continue
            # A LINK IS REPLACED, never written through: the file a link
            # points at is someone else's.
            if op.path.is_symlink():
                op.path.unlink()
            if isinstance(op, _Copy):
                shutil.copy2(op.src, op.path)
            else:
                os.replace(op.src, op.path)
        self._ops.clear()


__all__ = ["Plan", "found"]
