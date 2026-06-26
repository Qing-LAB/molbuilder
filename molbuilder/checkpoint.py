"""Run-checkpoints — git-based working-dir state management.

See docs/protocols/run-checkpoints.md for the full design contract.
This module is the L1 + L2 implementation (data model + ``Repo``
orchestration).  CLI lives in :mod:`molbuilder.cli`.

Single user, lowest-directory scope, no auto-commit (user-driven
checkpoints only).  The wrapper does the first-run bootstrap via a
shell prologue; this module owns the user-facing operations.

Three primitives:
  * Phase 1 init         -- ``Repo.init()``
  * Phase 2 checkpoint   -- ``Repo.checkpoint(message)``
  * Phase 5 restore      -- ``Repo.restore(ref)``

Plus listing + tagging.  Big binaries (``.DM``, ``.HSX``, ``.TSHS``,
``.TBT.AVTRANS_*``) are archived by SHA in ``.binsnapshots/<sha>/``
keyed to their corresponding commit, NOT committed to the git
object database.  Restore copies them back from the archive.

Every git invocation goes through :func:`_run_git` -- argv list,
``cwd`` pinned to the working dir, no shell strings.  Errors raise
:class:`CheckpointError` subclasses with the git stderr verbatim.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# Big binary patterns (gitignored, archived by SHA on each checkpoint).
# Mirrors the .gitignore policy in docs/protocols/run-checkpoints.md § 9.1.
# Glob patterns; matched non-recursively against the working dir top level.
_BIG_BINARY_GLOBS = (
    "*.DM",
    "*.HSX",
    "*.TSHS",
    "*.TBT.AVTRANS_*",
    "*.TBT.CC",
    "*.TBT.DOS",
)

# Default .gitignore contents written by ``init`` when the file doesn't
# already exist.  See docs/protocols/run-checkpoints.md § 9 for the
# rationale per pattern.
_DEFAULT_GITIGNORE = """\
# molbuilder run-checkpoints contract: this dir is managed by git.
# Large binary state files are archived separately in .binsnapshots/<sha>/.
# See docs/protocols/run-checkpoints.md § 9 for rationale per pattern.

# Big binary state (archived by SHA, not committed)
*.DM
*.HSX
*.TSHS
*.TBT.AVTRANS_*
*.TBT.CC
*.TBT.DOS

# Pseudopotential cache files (deterministic from .psml)
*.ion.nc
*.ion.xml

# SIESTA scratch / rotating logs
fdf.*.log
WORK_*
INPUT_TMP.*

# Binary archive directory
.binsnapshots/

# Other SIESTA per-run scratch / large MD trajectories
*.MD
*.MD_CAR

# IDE / editor noise
*.swp
*~
"""

# File markers indicating a directory is itself a working dir (must not
# be subsumed by a parent's checkpoint repo per P5 lowest-directory rule).
_NESTED_WORKING_DIR_MARKERS = (".fdf", ".py", ".run.sh")


# --------------------------------------------------------------------- #
#  Exceptions                                                           #
# --------------------------------------------------------------------- #


class CheckpointError(Exception):
    """Base class for run-checkpoints errors."""


class GitNotInstalledError(CheckpointError):
    """``git`` is not on PATH."""


class DirtyWorkingTreeError(CheckpointError):
    """Operation refuses to proceed on a dirty working tree."""


class NoSuchRefError(CheckpointError):
    """The named ref / tag / branch / SHA does not resolve."""


class NestedRepoRefusedError(CheckpointError):
    """Init refused because the directory contains nested working dirs
    (per P5 lowest-directory rule)."""


# --------------------------------------------------------------------- #
#  Data model                                                           #
# --------------------------------------------------------------------- #


@dataclass
class Checkpoint:
    sha:           str
    short_sha:     str
    summary:       str
    author_at:     str        # ISO timestamp
    refs:          List[str]  # tags + branches pointing here
    has_archive:   bool
    archive_bytes: Optional[int] = None


@dataclass
class RepoState:
    path:           str
    initialized:    bool
    head:           Optional[str] = None
    current_branch: Optional[str] = None
    dirty:          bool          = False
    untracked:      int           = 0
    archive_total_bytes: int      = 0


# --------------------------------------------------------------------- #
#  Internals                                                            #
# --------------------------------------------------------------------- #


def _run_git(argv: List[str], cwd: str, *,
             check: bool = True) -> subprocess.CompletedProcess:
    """Run ``git argv`` in ``cwd``; return CompletedProcess.

    Sets the molbuilder identity locally per § 11 decision 1.
    """
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME",     "molbuilder")
    env.setdefault("GIT_AUTHOR_EMAIL",
                   f"molbuilder@{os.uname().nodename}")
    env.setdefault("GIT_COMMITTER_NAME",  env["GIT_AUTHOR_NAME"])
    env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    try:
        return subprocess.run(
            ["git", *argv],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=check,
        )
    except FileNotFoundError as e:
        raise GitNotInstalledError(
            "git is not on PATH.  molbuilder requires git ≥ 2.20; "
            "install via your distro's package manager OR activate the "
            "molbuilder conda env (`conda activate molbuilder-host` -- "
            "every molbuilder env ships git as a conda_packages entry)."
        ) from e
    except subprocess.CalledProcessError as e:
        # Re-raise with stderr surfaced so callers can wrap.
        msg = (e.stderr or e.stdout or str(e)).strip()
        raise CheckpointError(
            f"git {' '.join(argv[:3])}... failed: {msg}"
        ) from e


def _check_nested_working_dirs(path: Path) -> List[str]:
    """Walk ``path``'s subdirectories; return relative paths of any
    sub-dir that contains a working-dir marker (``.fdf``, ``.py``,
    ``.run.sh``).  Empty list means safe to init here per P5."""
    nested: List[str] = []
    for sub in path.rglob("*"):
        if not sub.is_dir():
            continue
        if sub == path:
            continue
        # Don't walk into our own .binsnapshots or .git.
        if any(p.name in (".git", ".binsnapshots")
               for p in sub.relative_to(path).parents):
            continue
        if sub.name in (".git", ".binsnapshots"):
            continue
        # Does this subdir contain a working-dir marker?
        for entry in sub.iterdir():
            if entry.is_file() and any(
                    entry.name.endswith(m)
                    for m in _NESTED_WORKING_DIR_MARKERS):
                nested.append(str(sub.relative_to(path)))
                break
    return nested


def _list_big_binaries(path: Path) -> List[Path]:
    """Top-level big binaries (per ``_BIG_BINARY_GLOBS``)."""
    found: List[Path] = []
    for pat in _BIG_BINARY_GLOBS:
        found.extend(p for p in path.glob(pat) if p.is_file())
    return found


def _archive_dir(path: Path, sha: str) -> Path:
    return path / ".binsnapshots" / sha


def _archive_binaries(path: Path, sha: str) -> int:
    """Copy big binaries from working dir into ``.binsnapshots/<sha>/``;
    write MANIFEST.  Returns total bytes archived (0 if no binaries)."""
    binaries = _list_big_binaries(path)
    if not binaries:
        return 0
    target = _archive_dir(path, sha)
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    manifest_lines: List[str] = []
    for src in binaries:
        dst = target / src.name
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        sha256 = hashlib.sha256(dst.read_bytes()).hexdigest()
        manifest_lines.append(f"{sha256}  {size}  {src.name}")
        total += size
    (target / "MANIFEST").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8")
    return total


def _restore_archived_binaries(path: Path, sha: str) -> List[str]:
    """Copy archived binaries for ``sha`` back into ``path``.  Verifies
    each file's sha256 against MANIFEST before copying; raises if any
    mismatch.  Returns list of restored file names."""
    arch = _archive_dir(path, sha)
    manifest = arch / "MANIFEST"
    if not arch.is_dir():
        return []
    if not manifest.is_file():
        # Archive exists but no manifest -- treat as if no binaries.
        return []
    restored: List[str] = []
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        sha256, _bytes, name = parts
        expected[name] = sha256
    for name, want_sha256 in expected.items():
        src = arch / name
        if not src.is_file():
            raise CheckpointError(
                f"archive manifest references {name!r} but the file is "
                f"missing from {arch}; refusing to restore.")
        actual = hashlib.sha256(src.read_bytes()).hexdigest()
        if actual != want_sha256:
            raise CheckpointError(
                f"archive integrity check failed for {name!r} "
                f"(expected sha256 {want_sha256!r}, got {actual!r}); "
                f"refusing to restore.")
        shutil.copy2(src, path / name)
        restored.append(name)
    return restored


# --------------------------------------------------------------------- #
#  Public surface                                                       #
# --------------------------------------------------------------------- #


class Repo:
    """A single working directory tracked by a checkpoint repository.

    Single-user.  Lowest-directory scope (the directory containing the
    ``.fdf`` / ``.py`` / ``.run.sh``; not its parent).  No auto-commit
    -- every commit is the result of an explicit user call to
    :meth:`checkpoint`.
    """

    def __init__(self, path: str) -> None:
        self.path = str(Path(path).resolve())

    # -- predicates ----------------------------------------------- #

    @property
    def initialized(self) -> bool:
        return (Path(self.path) / ".git").is_dir()

    def _require_init(self) -> None:
        if not self.initialized:
            raise CheckpointError(
                f"{self.path}: not a checkpoint repository.  Run "
                f"`molbuilder snapshot init` first.")

    # -- Phase 1: init -------------------------------------------- #

    def init(self) -> None:
        """Initialise the working dir as a git repository.  Idempotent
        (no-op if already initialised).  Refuses if the directory
        contains nested working dirs (P5)."""
        p = Path(self.path)
        if self.initialized:
            return
        nested = _check_nested_working_dirs(p)
        if nested:
            raise NestedRepoRefusedError(
                f"{self.path}: cannot init -- nested working dirs "
                f"present: {nested}.  Each lowest-directory must be "
                f"its own checkpoint repo (run-checkpoints.md § P5).")

        _run_git(["init", "-q"], cwd=self.path)
        host = os.uname().nodename
        _run_git(["config", "user.email", f"molbuilder@{host}"],
                 cwd=self.path)
        _run_git(["config", "user.name", "molbuilder"], cwd=self.path)
        _run_git(["config", "commit.gpgsign", "false"], cwd=self.path)

        gi = p / ".gitignore"
        if not gi.exists():
            gi.write_text(_DEFAULT_GITIGNORE, encoding="utf-8")
        snaps = p / ".binsnapshots"
        snaps.mkdir(exist_ok=True)
        (snaps / ".gitkeep").touch()

        _run_git(["add", "."], cwd=self.path)
        # If the dir is empty (no files at all) git complains; tolerate.
        st = _run_git(["status", "--porcelain"], cwd=self.path,
                      check=False)
        if not st.stdout.strip():
            # Nothing to commit (truly empty dir + .gitignore + .gitkeep
            # already added means there's at least the gitignore).
            # Force-create an empty initial commit so HEAD exists.
            _run_git(
                ["commit", "--allow-empty", "-q",
                 "-m", f"molbuilder: initialised empty checkpoint repo "
                       f"({Path(self.path).name})"],
                cwd=self.path,
            )
        else:
            _run_git(
                ["commit", "-q",
                 "-m", f"molbuilder: initial state of "
                       f"{Path(self.path).name}"],
                cwd=self.path,
            )

        # Archive big binaries to the new HEAD's SHA.
        head_sha = self._head_sha()
        _archive_binaries(p, head_sha)

    # -- Phase 2: checkpoint -------------------------------------- #

    def checkpoint(self, message: Optional[str] = None
                   ) -> Optional[Checkpoint]:
        """Stage everything in the working tree and create a new commit.

        Big binaries are archived to ``.binsnapshots/<new_sha>/`` after
        the commit lands.  Returns the new :class:`Checkpoint`, or
        ``None`` if the working tree was clean (nothing to commit).
        """
        self._require_init()
        _run_git(["add", "."], cwd=self.path)
        st = _run_git(["status", "--porcelain"], cwd=self.path,
                      check=False)
        if not st.stdout.strip():
            # Check unstaged big binaries separately -- they are
            # gitignored so the status above is clean, but they might
            # be new files that should appear in the archive.
            head_sha = self._head_sha()
            arch = _archive_dir(Path(self.path), head_sha)
            if not arch.is_dir() and _list_big_binaries(Path(self.path)):
                # Archive missing for current HEAD but binaries exist;
                # create the archive so the user can restore.
                _archive_binaries(Path(self.path), head_sha)
                return self._checkpoint_from_sha(head_sha)
            return None
        if message is None:
            message = (
                f"checkpoint "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )
        _run_git(["commit", "-q", "-m", message], cwd=self.path)
        new_sha = self._head_sha()
        _archive_binaries(Path(self.path), new_sha)
        return self._checkpoint_from_sha(new_sha)

    # -- Phase 3: tag --------------------------------------------- #

    def tag(self, label: str, message: str, at: str = "HEAD") -> str:
        """Create an annotated tag.  Always annotated (carries message)
        per § 11 decision 3 (lightweight tags not used)."""
        self._require_init()
        if not message:
            raise CheckpointError("tag message must be non-empty")
        _run_git(["tag", "-a", label, at, "-m", message],
                 cwd=self.path)
        return label

    # -- Phase 5: restore ----------------------------------------- #

    def restore(self, ref: str, *,
                include_binaries: bool = True) -> List[str]:
        """Restore the working tree to the state at ``ref``.  Refuses on
        a dirty working tree.  Returns the list of restored binary
        files (empty if no archive at the ref's SHA).
        """
        self._require_init()
        st = _run_git(["status", "--porcelain"], cwd=self.path,
                      check=False)
        if st.stdout.strip():
            raise DirtyWorkingTreeError(
                "working tree has uncommitted changes; "
                "checkpoint or discard them before restoring.")
        # Resolve ref to a SHA.
        sha = self._resolve_ref(ref)
        # git restore: rewinds the working tree but keeps HEAD on
        # the current branch (per § 11 decision 3).
        _run_git(["restore", "--source", ref, "--worktree",
                  "--staged", "."],
                 cwd=self.path)
        if not include_binaries:
            return []
        return _restore_archived_binaries(Path(self.path), sha)

    # -- introspection -------------------------------------------- #

    def state(self) -> RepoState:
        p = Path(self.path)
        if not self.initialized:
            return RepoState(path=self.path, initialized=False)
        head = self._head_sha()
        branch_r = _run_git(["rev-parse", "--abbrev-ref", "HEAD"],
                            cwd=self.path)
        branch = branch_r.stdout.strip() or None
        if branch == "HEAD":
            branch = None  # detached
        st = _run_git(["status", "--porcelain"], cwd=self.path)
        lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
        dirty = bool(lines)
        untracked = sum(1 for ln in lines if ln.startswith("?"))
        snaps = p / ".binsnapshots"
        total = 0
        if snaps.is_dir():
            for sub in snaps.rglob("*"):
                if sub.is_file():
                    total += sub.stat().st_size
        return RepoState(
            path=self.path, initialized=True,
            head=head, current_branch=branch,
            dirty=dirty, untracked=untracked,
            archive_total_bytes=total,
        )

    def list_checkpoints(self, limit: int = 50) -> List[Checkpoint]:
        self._require_init()
        # Format: SHA|short|author_iso|subject|refnames (split by |||)
        fmt = "%H|||%h|||%aI|||%s|||%D"
        r = _run_git(["log", f"-n{int(limit)}", f"--pretty=format:{fmt}",
                      "--all"], cwd=self.path)
        out: List[Checkpoint] = []
        for line in r.stdout.splitlines():
            if not line:
                continue
            parts = line.split("|||")
            if len(parts) != 5:
                continue
            sha, short, iso, subject, refs_raw = parts
            refs = [r.strip()
                    for r in refs_raw.split(",")
                    if r.strip()]
            arch = _archive_dir(Path(self.path), sha)
            has_arch = arch.is_dir() and any(
                p.name != "MANIFEST" and p.name != ".gitkeep"
                for p in arch.iterdir())
            arch_bytes = (
                sum(p.stat().st_size for p in arch.iterdir()
                    if p.is_file() and p.name not in (
                        "MANIFEST", ".gitkeep"))
                if has_arch else None
            )
            out.append(Checkpoint(
                sha=sha, short_sha=short, summary=subject,
                author_at=iso, refs=refs,
                has_archive=has_arch, archive_bytes=arch_bytes,
            ))
        return out

    # -- helpers -------------------------------------------------- #

    def _head_sha(self) -> str:
        r = _run_git(["rev-parse", "HEAD"], cwd=self.path)
        return r.stdout.strip()

    def _resolve_ref(self, ref: str) -> str:
        """Resolve a ref to the COMMIT SHA it points at.

        ``ref^{commit}`` peels through annotated-tag objects to the
        underlying commit -- without this, `git rev-parse my-tag`
        returns the tag-object SHA (not the commit) for annotated
        tags, and the binary-archive lookup misses.
        """
        try:
            r = _run_git(["rev-parse", f"{ref}^{{commit}}"],
                         cwd=self.path)
        except CheckpointError as e:
            raise NoSuchRefError(
                f"no such ref: {ref!r}.  Use `molbuilder snapshot list` "
                f"to see available tags / branches / checkpoints."
            ) from e
        return r.stdout.strip()

    def _checkpoint_from_sha(self, sha: str) -> Checkpoint:
        # Single-row lookup; cheap.
        r = _run_git(["log", "-1", sha,
                      "--pretty=format:%H|||%h|||%aI|||%s|||%D"],
                     cwd=self.path)
        parts = r.stdout.split("|||")
        refs = [t.strip() for t in parts[4].split(",") if t.strip()] \
            if len(parts) >= 5 else []
        arch = _archive_dir(Path(self.path), sha)
        has_arch = arch.is_dir() and any(
            p.name != "MANIFEST" and p.name != ".gitkeep"
            for p in arch.iterdir())
        arch_bytes = (
            sum(p.stat().st_size for p in arch.iterdir()
                if p.is_file() and p.name not in ("MANIFEST", ".gitkeep"))
            if has_arch else None
        )
        return Checkpoint(
            sha=parts[0], short_sha=parts[1],
            summary=parts[3] if len(parts) >= 4 else "",
            author_at=parts[2] if len(parts) >= 3 else "",
            refs=refs, has_archive=has_arch,
            archive_bytes=arch_bytes,
        )


__all__ = [
    "Repo", "Checkpoint", "RepoState",
    "CheckpointError", "GitNotInstalledError",
    "DirtyWorkingTreeError", "NoSuchRefError",
    "NestedRepoRefusedError",
]
