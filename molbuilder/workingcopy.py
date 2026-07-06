"""Working copy of an artifact you're editing in the browser.

Dead simple: **load → edit → save (overwrite or save-as)**.  A draft copy is
kept on disk so an unsaved edit survives a reload/crash.  There is NO gate, no
hashing, no "changed underneath" check — you own the data you loaded and you own
the save.  The artifact format is opaque; an application supplies a Codec.

Layer: L1 — depends only on ``persist`` + stdlib.
"""
from __future__ import annotations

import os
import uuid
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Tuple

from . import persist

#: Per-project draft directory.  Gitignore-able; created lazily.
SCRATCH_DIR = ".molbuilder_workspace"
_ENVELOPE_SCHEMA = "molbuilder/workingcopy-draft@2"


class Codec(Protocol):
    """The only format-specific part.  The core treats ``data`` as opaque."""
    def load(self, source_path: Path) -> Any: ...
    def files(self, data: Any, target: Path) -> Sequence[Tuple[Path, bytes]]: ...
    def scratch_blob(self, data: Any) -> Any: ...
    def from_scratch(self, blob: Any) -> Any: ...


@dataclass
class ScratchRecord:
    path: Path
    source: Optional[str]
    session: str
    ts: float
    blob: Any


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _scratch_dir(project_dir: Path) -> Path:
    return Path(project_dir) / SCRATCH_DIR


def _scratch_path(project_dir: Path, stem: str, session: str) -> Path:
    return _scratch_dir(project_dir) / f"{stem}.{session}.wc.json"


@dataclass
class WorkingCopy:
    """The artifact you're editing.  ``data`` is the working state; ``source``
    is where it was loaded from (or where it was last saved)."""
    codec: Codec
    session: str
    project_dir: Path
    data: Any
    source: Optional[Path]
    dirty: bool = False
    new_id: Optional[str] = None

    # ---- open / new / recover ---------------------------------------- #
    @classmethod
    def open(cls, source_path, codec: Codec, *, session: str,
             project_dir) -> "WorkingCopy":
        """Load the artifact from a file."""
        src = Path(source_path).resolve()
        return cls(codec=codec, session=session, project_dir=Path(project_dir),
                   data=codec.load(src), source=src, dirty=False)

    @classmethod
    def new(cls, codec: Codec, *, session: str, project_dir,
            data: Any = None) -> "WorkingCopy":
        """A fresh artifact with no source yet; first save is a save-as."""
        return cls(codec=codec, session=session, project_dir=Path(project_dir),
                   data=data, source=None, dirty=False,
                   new_id=uuid.uuid4().hex[:12])

    @classmethod
    def recover(cls, record: ScratchRecord, codec: Codec, *,
                project_dir) -> "WorkingCopy":
        """Adopt a draft left behind by a crashed session."""
        return cls(codec=codec, session=record.session,
                   project_dir=Path(project_dir),
                   data=codec.from_scratch(record.blob),
                   source=Path(record.source).resolve() if record.source else None,
                   dirty=True)

    # ---- edit -> draft ----------------------------------------------- #
    def is_dirty(self) -> bool:
        return self.dirty

    def _stem(self) -> str:
        return self.source.stem if self.source else f"new-{self.new_id or self.session}"

    def _scratch_file(self) -> Path:
        return _scratch_path(self.project_dir, self._stem(), self.session)

    def update(self, data: Any) -> None:
        """Change the working data + keep a draft so a reload/crash won't lose it."""
        self.data = data
        self.dirty = True
        env = {"schema": _ENVELOPE_SCHEMA, "source": str(self.source) if self.source else None,
               "session": self.session, "ts": time.time(),
               "blob": self.codec.scratch_blob(self.data)}
        scratch = self._scratch_file()
        scratch.parent.mkdir(parents=True, exist_ok=True)
        persist.write_json(scratch, env)

    # ---- save (overwrite or save-as) --------------------------------- #
    def save(self, target_path) -> Path:
        """Write the working copy to ``target_path`` — same path overwrites, a
        new path is a save-as.  Writes all of the artifact's files, then drops
        the draft.  No gate: this overwrites whatever is on disk.

        The multi-file commit writes ALL temp files first, then does ALL the
        ``os.replace`` renames back-to-back — so the window in which the
        ``.xyz`` and ``.molstruct.json`` could be mismatched (a crash between the
        two writes) shrinks to just the two renames, with no I/O between
        (review finding b2).  True cross-file atomicity isn't available on the
        filesystem; this makes a half-written pair near-impossible."""
        target = Path(target_path).resolve()
        old_scratch = self._scratch_file()
        staged: List[Tuple[Path, Path]] = []
        for path, blob in self.codec.files(self.data, target):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_bytes(blob)
            staged.append((tmp, p))
        for tmp, dest in staged:                 # renames only -- no I/O between
            os.replace(tmp, dest)
        self.source = target
        self.dirty = False
        if old_scratch.exists():
            old_scratch.unlink()
        return target

    def discard(self) -> None:
        """Drop the draft without writing anything."""
        p = self._scratch_file()
        if p.exists():
            p.unlink()


# ---- crash recovery / housekeeping ---------------------------------- #
def list_orphans(project_dir, live_sessions: Sequence[str]) -> List[ScratchRecord]:
    """Drafts whose session is no longer live (a crash/restart left them)."""
    d = _scratch_dir(Path(project_dir))
    live = set(live_sessions)
    out: List[ScratchRecord] = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.wc.json")):
        try:
            env = persist.read_json(p)
        except Exception:
            continue
        if env.get("session") not in live:
            out.append(ScratchRecord(path=p, source=env.get("source"),
                                     session=env.get("session"),
                                     ts=env.get("ts"), blob=env.get("blob")))
    return out


def discard_orphan(record: ScratchRecord) -> None:
    if record.path.exists():
        record.path.unlink()


def clean_all(project_dir) -> int:
    d = _scratch_dir(Path(project_dir))
    n = 0
    if d.exists():
        for p in d.glob("*.wc.json"):
            p.unlink()
            n += 1
    return n
