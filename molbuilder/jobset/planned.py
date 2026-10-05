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

*(Until 2026-10-05 prep wrote as it went and put three files back when it
refused -- the plan, `STAGE-PLAN.md` and the calculation's copy of its
machine's record -- while the decks, wrappers, data files and the attempt it
had opened stayed behind; W55 D16.)*
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

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


@dataclass(frozen=True)
class _Move:
    src: Path
    path: Path


@dataclass(frozen=True)
class _Remove:
    path: Path


@dataclass(frozen=True)
class _Folder:
    path: Path


#: What a path holds in the plan: bytes, a file on disk it copies, or
#: ``None`` -- removed.
_Held = Union[bytes, Path, None]

#: THE CLOCK READINGS a planned file carries -- an ISO date-time stamp (a
#: deck's and a run script's ``generated-at``, a record's ``created_at``,
#: the pipeline log's banner) and the progress seed's ``wall_time`` -- which
#: a plan made a moment later reads differently while deciding nothing
#: differently (measured on two plans of one stage, 2026-10-05: every other
#: byte the same).
_CLOCK = re.compile(
    rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    rb"|wall_time: [0-9.]+")


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
        self._ops.append(_Copy(s, p))
        self._held[p] = held
        return p

    def move(self, src, path) -> Path:
        """``src`` will be moved to ``path``."""
        s, p = _at(src), _at(path)
        held = self._held.get(s, s)
        if held is None or (isinstance(held, Path) and not held.is_file()):
            raise FileNotFoundError(f"nothing to move at {src}")
        self._ops.append(_Move(s, p))
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

    def identity(self) -> str:
        """What this plan IS, less the moment it was made: every operation
        in order, each file's bytes with its clock readings masked, and what
        a copy or a move takes by its source's path, size and time -- so
        two plans of one folder agree exactly when they would write the same
        thing.  The preview's plan is named by it, and Prep refuses a plan
        that differs (`job-system.md` § 5.0)."""
        h = hashlib.sha256()
        for op in self._ops:
            h.update(type(op).__name__.encode() + b"\0")
            h.update(str(op.path).encode() + b"\0")
            if isinstance(op, _Put):
                h.update(_CLOCK.sub(b"<clock>", op.data))
                h.update(repr(op.mode).encode())
            elif isinstance(op, (_Copy, _Move)):
                held = self._source_on_disk(op.src)
                h.update(str(op.src).encode() + b"\0")
                if held is not None:
                    st = held.stat()
                    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
            h.update(b"\n")
        return h.hexdigest()

    def _source_on_disk(self, src: Path) -> Optional[Path]:
        """The file a copy reads, when it is on disk rather than planned."""
        return src if src not in self._held and src.is_file() else None

    def writes(self) -> List[Path]:
        """Every path this plan puts something in, in order, once each."""
        out: Dict[Path, None] = {}
        for op in self._ops:
            if isinstance(op, (_Put, _Copy, _Move)):
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


__all__ = ["Plan"]
