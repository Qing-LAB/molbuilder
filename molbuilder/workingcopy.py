"""Working-copy persistence core — the format-agnostic transient-data
foundation.  See ``docs/protocols/working-copy-persistence.md`` for the full
contract (goal/boundary/API/invariants/risks).

Holds a user-edited working copy of ONE artifact: loaded from a source, mirrored
to a scratch record so it survives reloads/crashes, and written to a durable
target ONLY on explicit ``commit`` — gated by the source hash so a changed
source can never be silently overwritten.  The artifact format is opaque; an
application supplies a :class:`Codec` (§5).  Single-user (no concurrency
arbitration).

Layer: L1 — depends only on ``persist`` + stdlib; knows nothing about atoms,
structures, or any engine.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Tuple

from . import persist

#: Per-project scratch directory (§3).  Gitignore-able; created lazily.
SCRATCH_DIR = ".molbuilder_workspace"
_ENVELOPE_SCHEMA = "molbuilder/workingcopy-scratch@1"


class WorkingCopyError(Exception):
    """Base for working-copy errors."""


class Codec(Protocol):
    """The only format-specific seam (§5).  The core treats ``data`` as opaque.

    ``files()`` MUST return the durable files **in write order with the
    identity/source file (the one ``hash_source`` hashes) LAST** (§9.3), so a
    partial failure leaves the source unchanged and a retry gate still passes.
    """

    def load(self, source_path: Path) -> Any: ...
    def hash_source(self, source_path: Path) -> str: ...
    def files(self, data: Any, target: Path) -> Sequence[Tuple[Path, bytes]]: ...
    def scratch_blob(self, data: Any) -> Any: ...
    def from_scratch(self, blob: Any) -> Any: ...


@dataclass
class ScratchRecord:
    """A scratch record on disk (§8/§10)."""
    path: Path
    source: Optional[str]
    source_hash: Optional[str]
    session: str
    ts: float
    blob: Any


@dataclass
class CommitResult:
    """Outcome of :meth:`WorkingCopy.commit`."""
    ok: bool
    target: Optional[Path] = None
    reason: str = ""                      # "committed" | "mismatch"
    expected_hash: Optional[str] = None   # on mismatch: the source hash at load
    actual_hash: Optional[str] = None     # on mismatch: the current on-disk hash


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """temp-file + os.replace (per-file atomic on the same filesystem)."""
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
    """The live, transient working copy of one artifact.  Owned by the core;
    an application reads ``.data`` / ``.is_dirty()`` from it and never keeps a
    parallel copy that can drift (§7)."""

    codec: Codec
    session: str
    project_dir: Path
    data: Any
    source: Optional[Path]
    source_hash: Optional[str]
    dirty: bool = False
    new_id: Optional[str] = None   # unique scratch key for a sourceless copy

    # ---- constructors (§6) ------------------------------------------- #
    @classmethod
    def open(cls, source_path, codec: Codec, *, session: str,
             project_dir) -> "WorkingCopy":
        """Load from a source; record its hash.  No scratch write yet (§6).

        The source path is resolved (absolute, symlinks collapsed) so the
        commit gate compares like with like — a later commit passing a
        relative/other form of the SAME file must not slip past the gate as if
        it were a save-as.
        """
        src = Path(source_path).resolve()
        return cls(codec=codec, session=session, project_dir=Path(project_dir),
                   data=codec.load(src), source=src,
                   source_hash=codec.hash_source(src), dirty=False)

    @classmethod
    def new(cls, codec: Codec, *, session: str, project_dir,
            data: Any = None) -> "WorkingCopy":
        """A freshly-built artifact with no source (S6); first commit = save-as.

        Gets a unique ``new_id`` so two sourceless copies in one session don't
        collide on the same scratch file.
        """
        return cls(codec=codec, session=session, project_dir=Path(project_dir),
                   data=data, source=None, source_hash=None, dirty=False,
                   new_id=uuid.uuid4().hex[:12])

    @classmethod
    def recover(cls, record: ScratchRecord, codec: Codec, *,
                project_dir) -> "WorkingCopy":
        """Adopt a scratch record back into a working session (§10)."""
        return cls(codec=codec, session=record.session,
                   project_dir=Path(project_dir),
                   data=codec.from_scratch(record.blob),
                   source=Path(record.source).resolve() if record.source else None,
                   source_hash=record.source_hash, dirty=True)

    # ---- state ------------------------------------------------------- #
    def is_dirty(self) -> bool:
        return self.dirty

    def _stem(self) -> str:
        return self.source.stem if self.source else f"new-{self.new_id or self.session}"

    def _scratch_file(self) -> Path:
        return _scratch_path(self.project_dir, self._stem(), self.session)

    # ---- edit -> mirror to scratch (§6) ------------------------------ #
    def update(self, data: Any) -> None:
        """Set the working data + mirror to scratch (the caller debounces)."""
        self.data = data
        self.dirty = True
        env = {
            "schema":      _ENVELOPE_SCHEMA,
            "source":      str(self.source) if self.source else None,
            "source_hash": self.source_hash,
            "session":     self.session,
            "ts":          time.time(),
            "blob":        self.codec.scratch_blob(self.data),
        }
        scratch = self._scratch_file()
        scratch.parent.mkdir(parents=True, exist_ok=True)
        persist.write_json(scratch, env)

    # ---- commit (the ONLY durable write, §6/§9) ---------------------- #
    def commit(self, target_path, on_mismatch: str = "refuse") -> CommitResult:
        """Gate → write (identity file last) → re-anchor → clear scratch.

        The gate fires only when writing back to the loaded source
        (``target == source``): a save-as to a *different* target does not
        overwrite the source, so target-existence is the application's concern
        (§1.2 boundary).  ``on_mismatch`` is ``"refuse"`` (return a mismatch
        result) or ``"force"`` (proceed anyway — the app made an explicit
        choice).
        """
        target = Path(target_path).resolve()         # normalize so the gate
        old_scratch = self._scratch_file()           # compares like with like

        # § 9.1 gate: source changed on disk since load?
        if (self.source is not None and self.source_hash is not None
                and target == self.source and self.source.exists()):
            current = self.codec.hash_source(self.source)
            if current != self.source_hash and on_mismatch != "force":
                return CommitResult(ok=False, reason="mismatch",
                                    expected_hash=self.source_hash,
                                    actual_hash=current)

        # Write each durable file atomically, in codec order (identity last).
        for path, blob in self.codec.files(self.data, target):
            _atomic_write_bytes(Path(path), blob)

        # § 9.4 re-anchor to the committed target so a second save works.
        self.source = target
        self.source_hash = self.codec.hash_source(target)
        self.dirty = False

        # Clear scratch only after all files landed (§9.3).
        if old_scratch.exists():
            old_scratch.unlink()
        return CommitResult(ok=True, target=target, reason="committed")

    def discard(self) -> None:
        """Drop the working copy's scratch without writing anything durable."""
        p = self._scratch_file()
        if p.exists():
            p.unlink()


# ---- crash recovery / housekeeping (project-scoped, §10) ------------- #
def list_orphans(project_dir, live_sessions: Sequence[str]) -> List[ScratchRecord]:
    """Scratch records whose session is not in ``live_sessions`` — the orphans a
    crash (or, for no-auth localhost, a server restart) left behind (§10)."""
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
            out.append(ScratchRecord(
                path=p, source=env.get("source"),
                source_hash=env.get("source_hash"), session=env.get("session"),
                ts=env.get("ts"), blob=env.get("blob")))
    return out


def discard_orphan(record: ScratchRecord) -> None:
    if record.path.exists():
        record.path.unlink()


def clean_all(project_dir) -> int:
    """Remove every scratch record for a project.  Returns the count removed."""
    d = _scratch_dir(Path(project_dir))
    n = 0
    if d.exists():
        for p in d.glob("*.wc.json"):
            p.unlink()
            n += 1
    return n
