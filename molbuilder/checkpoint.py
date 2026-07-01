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
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


# --------------------------------------------------------------------- #
#  MANIFEST format -- canonical lockdown per § 10 of                    #
#  docs/protocols/run-checkpoints.md.                                   #
#                                                                       #
#  Exactly one format.  Strict parser raises on any deviation -- no     #
#  silent skip, no fallback, no field-count tolerance.  Legacy 2-col    #
#  sha256sum-style MANIFESTs are migrated via                           #
#  ``Repo.migrate_manifest(ref)`` -- never transparently accepted.      #
# --------------------------------------------------------------------- #

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SIZE_INT_RE   = re.compile(r"^(0|[1-9][0-9]*)$")
_FILENAME_RE   = re.compile(r"^[!-~]+$")    # ASCII printable, no spaces
_MANIFEST_NAME = "MANIFEST"
_MANIFEST_TMP  = "MANIFEST.tmp"
_SHA_DIR_RE    = re.compile(r"^[0-9a-f]{40}$")


def _format_canonical_manifest(entries: List[Tuple[str, int, str]]) -> str:
    """Render entries as canonical-format text.

    Entries are tuples ``(sha256_hex, size_bytes, filename)``.
    Sorted by filename; one entry per line; ``\\n`` terminator; final
    newline; no header / comments / blank lines.
    """
    sorted_entries = sorted(entries, key=lambda e: e[2])
    return "".join(
        f"{sha256}  {int(size)}  {name}\n"
        for sha256, size, name in sorted_entries
    )


def _parse_canonical_manifest(raw: bytes,
                              where: str) -> Dict[str, Tuple[str, int]]:
    """Strict canonical MANIFEST parser.

    Returns a dict ``{filename: (sha256_hex, size_bytes)}``.  Raises
    :class:`CheckpointError` with a specific reason on ANY deviation
    from § 10.2 -- no field-count fallback, no header skip, no comment
    skip, no BOM tolerance.

    The 2-column legacy ``sha256sum > MANIFEST`` shape is detected
    explicitly and raises with a pointer to the migrate-manifest CLI.

    ``where`` is included in error messages so the user knows which
    archive directory to look at.
    """
    # BOM rejection (before utf-8 decode so the message names the bytes).
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CheckpointError(
            f"malformed MANIFEST in {where}: file starts with a UTF-8 "
            f"BOM; canonical MANIFEST must be plain ASCII (§ 10.2).")
    # CRLF rejection -- the design pins LF-only.
    if b"\r" in raw:
        raise CheckpointError(
            f"malformed MANIFEST in {where}: contains CR bytes; "
            f"canonical MANIFEST uses LF line terminators only "
            f"(§ 10.2).")
    # Final newline required.
    if not raw.endswith(b"\n"):
        raise CheckpointError(
            f"malformed MANIFEST in {where}: missing final newline "
            f"(§ 10.2 requires a trailing LF).")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise CheckpointError(
            f"malformed MANIFEST in {where}: non-ASCII byte at "
            f"offset {e.start} (§ 10.2 requires ASCII-only)."
        ) from e
    lines = text.split("\n")
    # split("\n") on a trailing-newline text gives a "" final element.
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise CheckpointError(
            f"malformed MANIFEST in {where}: file is empty "
            f"(§ 10.2 requires at least one entry per archive).")
    out: Dict[str, Tuple[str, int]] = {}
    seen_order: List[str] = []
    for idx, line in enumerate(lines, start=1):
        if line == "":
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} is blank "
                f"(§ 10.2 forbids blank lines).")
        # Legacy 2-column sha256sum default form -- detect explicitly so
        # the user gets a useful error rather than "wrong field count".
        # Pattern: ``<64-hex-sha>  <name>`` (exactly two-space separator
        # is sha256sum's default).
        if (len(line) >= 66
                and line[0:64].count(" ") == 0
                and _SHA256_HEX_RE.match(line[:64])
                and line[64:66] == "  "
                and " " not in line[66:]
                and line[66:]):
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} is the "
                f"legacy 2-column sha256sum format (sha + name).  "
                f"Canonical MANIFEST is 3-column "
                f"(sha + bytes + name; § 10.2).  Run "
                f"`molbuilder snapshot migrate-manifest <ref>` to "
                f"convert this archive in-place (§ 10.4).")
        # Canonical 3-column shape: <sha>__<size>__<name>, exactly two
        # ASCII spaces between fields.  Strict split on the literal
        # double-space separator.  Embedded whitespace in filename is
        # forbidden so "  " is unambiguous as the field separator.
        parts = line.split("  ")
        if len(parts) != 3:
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} has "
                f"{len(parts)} fields (separated by '  '); canonical "
                f"requires exactly 3 (§ 10.2).")
        sha256, size_s, name = parts
        if not _SHA256_HEX_RE.match(sha256):
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} sha256 "
                f"field {sha256!r} is not 64 lowercase hex chars "
                f"(§ 10.2).")
        if not _SIZE_INT_RE.match(size_s):
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} size "
                f"field {size_s!r} is not a non-negative decimal "
                f"integer without leading zeros (§ 10.2).")
        if not _FILENAME_RE.match(name):
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} filename "
                f"{name!r} contains non-printable or whitespace "
                f"characters (§ 10.2 requires ASCII printable, no "
                f"spaces).")
        if "/" in name or name == "..":
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} filename "
                f"{name!r} contains a path separator or parent "
                f"reference (§ 10.2 requires bare basename only).")
        if name.startswith("."):
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} filename "
                f"{name!r} starts with a dot; canonical archive does "
                f"not contain dotfiles (§ 10.1).")
        if name == _MANIFEST_NAME:
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} lists the "
                f"MANIFEST file itself; canonical MANIFEST must not "
                f"self-reference (§ 10.2).")
        if name in out:
            raise CheckpointError(
                f"malformed MANIFEST in {where}: filename {name!r} "
                f"appears more than once (§ 10.2).")
        out[name] = (sha256, int(size_s))
        seen_order.append(name)
    # Sort-order check (§ 10.2: lines are alphabetical by filename).
    if seen_order != sorted(seen_order):
        raise CheckpointError(
            f"malformed MANIFEST in {where}: entries are not sorted "
            f"alphabetically by filename (§ 10.2).")
    return out


def _atomic_write_text(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` atomically via .tmp + os.replace.

    The body is fully flushed + fsync'd before the rename, so any
    crash mid-write leaves either the prior file intact or the new
    file fully written -- never a half-MANIFEST.
    """
    tmp = target.parent / (target.name + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass        # tmpfs / some FS lack fsync; replace will still atom-swap
    os.replace(tmp, target)


def _archive_binaries(path: Path, sha: str) -> int:
    """Copy big binaries from working dir into ``.binsnapshots/<sha>/``;
    write MANIFEST in canonical format (§ 10.2).  Returns total bytes
    archived (0 if no binaries)."""
    binaries = _list_big_binaries(path)
    if not binaries:
        return 0
    if not _SHA_DIR_RE.match(sha):
        raise CheckpointError(
            f"_archive_binaries: SHA dir name {sha!r} is not 40 "
            f"lowercase hex chars (§ 10.1).")
    target = _archive_dir(path, sha)
    target.mkdir(parents=True, exist_ok=True)
    entries: List[Tuple[str, int, str]] = []
    total = 0
    for src in binaries:
        dst = target / src.name
        # Hash the SOURCE, copy, then re-hash the ARCHIVED copy and require
        # they match.  Deriving the MANIFEST sha from the copy alone would
        # make a silent copy corruption (disk error) self-consistent -- the
        # bad bytes would later "verify" against their own bad sha and be
        # restored as truth.  Verify FIDELITY (source == archive) at save
        # time so a corrupt copy fails loudly here, not silently on restore.
        src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copy2(src, dst)
        dst_sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        if dst_sha != src_sha:
            raise CheckpointError(
                f"archive copy of {src.name!r} is corrupt: source sha256 "
                f"{src_sha!r} != archived {dst_sha!r} (disk error?); the "
                f"checkpoint was NOT safely archived.")
        size = dst.stat().st_size
        entries.append((src_sha, size, src.name))
        total += size
    _atomic_write_text(target / _MANIFEST_NAME,
                       _format_canonical_manifest(entries))
    return total


def _verify_archived_binaries(path: Path, sha: str
                              ) -> Dict[str, Tuple[str, int]]:
    """Verify the archived binaries for ``sha`` against their MANIFEST
    (existence + size + sha256), touching NOTHING.  Raises
    :class:`CheckpointError` on ANY mismatch.  Returns the expected
    ``{name: (sha256, size)}`` map, or ``{}`` when there is no archive (a
    binary-free checkpoint is legal, § 4.6).

    Split out from the copy so callers (``restore``) can verify the archive
    is intact BEFORE they mutate the working tree at all -- a corrupt archive
    must abort the WHOLE restore, not leave a half-restored tree (§ 10.3)."""
    arch = _archive_dir(path, sha)
    manifest = arch / _MANIFEST_NAME
    if not arch.is_dir() or not manifest.is_file():
        return {}
    expected = _parse_canonical_manifest(manifest.read_bytes(), where=str(arch))
    if not expected:
        raise CheckpointError(
            f"archive at {arch}: canonical MANIFEST is present but "
            f"contained zero entries (§ 10.2).")
    for name, (want_sha256, want_size) in expected.items():
        src = arch / name
        if not src.is_file():
            raise CheckpointError(
                f"archive at {arch}: MANIFEST lists {name!r} but the "
                f"file is missing; refusing to restore (§ 10.3).")
        actual_size = src.stat().st_size
        if actual_size != want_size:
            raise CheckpointError(
                f"archive at {arch}: integrity check failed for "
                f"{name!r} -- expected {want_size} bytes, got "
                f"{actual_size} (§ 10.3).")
        actual_sha256 = hashlib.sha256(src.read_bytes()).hexdigest()
        if actual_sha256 != want_sha256:
            raise CheckpointError(
                f"archive at {arch}: integrity check failed for "
                f"{name!r} -- expected sha256 {want_sha256!r}, got "
                f"{actual_sha256!r}; refusing to restore (§ 10.3).")
    return expected


def _copy_archived_binaries(path: Path, sha: str,
                            expected: Dict[str, Tuple[str, int]]
                            ) -> List[str]:
    """Copy the (already-verified) archived binaries into the working tree.
    Sorted order = deterministic restore log.  Call ONLY after
    :func:`_verify_archived_binaries` has passed."""
    arch = _archive_dir(path, sha)
    restored: List[str] = []
    for name in sorted(expected.keys()):
        shutil.copy2(arch / name, path / name)
        restored.append(name)
    return restored


def _restore_archived_binaries(path: Path, sha: str) -> List[str]:
    """Verify then copy the archived binaries for ``sha`` (verification
    aborts BEFORE any byte hits the working tree, § 10.3)."""
    expected = _verify_archived_binaries(path, sha)
    return _copy_archived_binaries(path, sha, expected)


def _migrate_legacy_manifest(arch: Path) -> Dict[str, Tuple[str, int]]:
    """Convert a 2-column ``sha256sum`` MANIFEST in ``arch`` to the
    canonical 3-column form (§ 10.4).

    Behaviour:
      1. Reads existing MANIFEST.  If it already parses as canonical,
         no-ops and returns the parsed contents.
      2. If 2-column legacy: parses sha + name, re-hashes each file
         against the recorded sha, stat()s for the size column, writes
         canonical MANIFEST atomically.
      3. Any other shape: raises -- no auto-fix.

    Verification step (3) is load-bearing: if a recorded sha doesn't
    match the file on disk, the migration aborts before writing
    anything, leaving the legacy MANIFEST untouched.

    Returns the parsed canonical contents.
    """
    manifest = arch / _MANIFEST_NAME
    if not manifest.is_file():
        raise CheckpointError(
            f"migrate-manifest: {arch}: no MANIFEST file present.")
    raw = manifest.read_bytes()
    # Try canonical first -- short-circuits already-migrated archives.
    try:
        canon = _parse_canonical_manifest(raw, where=str(arch))
        return canon
    except CheckpointError as canon_err:
        canon_msg = str(canon_err)
    # Try legacy 2-column shape.  This must not silently accept anything
    # the canonical parser rejected for reasons OTHER than format.
    if not raw.endswith(b"\n"):
        raise CheckpointError(
            f"migrate-manifest: {arch}: MANIFEST missing final "
            f"newline; refusing to guess at the format.")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise CheckpointError(
            f"migrate-manifest: {arch}: MANIFEST has non-ASCII bytes "
            f"at offset {e.start}; refusing."
        ) from e
    parsed: Dict[str, Tuple[str, int]] = {}
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for idx, line in enumerate(lines, start=1):
        # sha256sum default output: "<64-hex-sha>  <name>" (two spaces).
        if not (len(line) >= 66 and _SHA256_HEX_RE.match(line[:64])
                and line[64:66] == "  "):
            raise CheckpointError(
                f"migrate-manifest: {arch}: line {idx} is neither "
                f"canonical 3-column nor legacy 2-column "
                f"sha256sum.  Original parser error was: "
                f"{canon_msg}.  Refusing to guess.")
        sha256_hex = line[:64]
        name = line[66:]
        if not _FILENAME_RE.match(name):
            raise CheckpointError(
                f"migrate-manifest: {arch}: line {idx} filename "
                f"{name!r} contains non-printable or whitespace "
                f"characters; refusing.")
        if "/" in name or name == ".." or name.startswith("."):
            raise CheckpointError(
                f"migrate-manifest: {arch}: line {idx} filename "
                f"{name!r} is not a bare basename; refusing.")
        if name == _MANIFEST_NAME:
            # sha256sum * includes MANIFEST itself; skip it -- the
            # canonical form must not self-reference.
            continue
        if name in parsed:
            raise CheckpointError(
                f"migrate-manifest: {arch}: filename {name!r} "
                f"appears more than once; refusing.")
        parsed[name] = (sha256_hex, -1)   # size filled in below
    if not parsed:
        raise CheckpointError(
            f"migrate-manifest: {arch}: legacy MANIFEST has no "
            f"payload entries (only MANIFEST self-reference?); "
            f"refusing.")
    # Re-hash + size every file BEFORE writing anything.  Any mismatch
    # aborts and leaves the original MANIFEST untouched (§ 10.4).
    verified: Dict[str, Tuple[str, int]] = {}
    for name, (want_sha256, _) in parsed.items():
        src = arch / name
        if not src.is_file():
            raise CheckpointError(
                f"migrate-manifest: {arch}: MANIFEST references "
                f"{name!r} but the file is missing.")
        actual_sha256 = hashlib.sha256(src.read_bytes()).hexdigest()
        if actual_sha256 != want_sha256:
            raise CheckpointError(
                f"migrate-manifest: {arch}: integrity check failed "
                f"for {name!r} -- expected sha256 {want_sha256!r}, "
                f"got {actual_sha256!r}.  Refusing to migrate; "
                f"original MANIFEST left untouched.")
        verified[name] = (actual_sha256, src.stat().st_size)
    # All entries verified; write canonical MANIFEST atomically.
    entries = [(sha, size, name) for name, (sha, size) in verified.items()]
    _atomic_write_text(arch / _MANIFEST_NAME,
                       _format_canonical_manifest(entries))
    return verified


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

    # -- Phase 4: experimental branching -------------------------- #

    def branch(self, name: str) -> str:
        """Create a new branch and switch to it (§ 4.5, Phase 4) -- the
        user's subsequent checkpoints land on it.  Equivalent to
        ``git checkout -b <name>``; carries any uncommitted changes onto the
        new branch (git's default) so a user can branch mid-edit before an
        experiment.  Raises :class:`CheckpointError` (with git's message) if
        the branch already exists or the name is invalid."""
        self._require_init()
        if not name.strip():
            raise CheckpointError("branch name must be non-empty")
        _run_git(["checkout", "-q", "-b", name], cwd=self.path)
        return name

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
        # ATOMICITY (§ 10.3): verify the binary archive BEFORE mutating the
        # working tree, so a corrupt/incomplete archive aborts the WHOLE
        # restore -- text AND binaries -- rather than leaving a half-restored
        # tree (text rewound, binaries stale).  git restore touches text; the
        # binary integrity check must gate it, not follow it.
        expected: Dict[str, Tuple[str, int]] = {}
        if include_binaries:
            expected = _verify_archived_binaries(Path(self.path), sha)
        # git restore: rewinds the working tree but keeps HEAD on
        # the current branch (per § 11 decision 3).
        _run_git(["restore", "--source", ref, "--worktree",
                  "--staged", "."],
                 cwd=self.path)
        if not include_binaries:
            return []
        # Copy the already-verified binaries (verification passed above).
        return _copy_archived_binaries(Path(self.path), sha, expected)

    # -- § 10.4 migrate-manifest ---------------------------------- #

    def migrate_manifest(self, ref: str) -> Dict[str, Tuple[str, int]]:
        """Convert a legacy 2-column ``sha256sum`` MANIFEST in the
        archive for ``ref`` to canonical 3-column form (§ 10.4).
        No-op (returns the parsed contents) if already canonical.

        Raises :class:`CheckpointError` with a specific reason on any
        other shape (or on hash mismatch); the original MANIFEST is
        left untouched in that case.
        """
        self._require_init()
        sha = self._resolve_ref(ref)
        arch = _archive_dir(Path(self.path), sha)
        if not arch.is_dir():
            raise CheckpointError(
                f"migrate-manifest: no archive directory at {arch}.")
        return _migrate_legacy_manifest(arch)

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
        # NOTE: archive_total_bytes is deliberately left at its default
        # (0). This snapshot is the refresh-path read (run-checkpoints.md
        # § 5.2, § 6.2) hit on every directory-enter; walking
        # .binsnapshots/ here would charge an O(archive) stat sweep on
        # each enter for a number the sensor never displays. Archive size
        # is computed only by the list/detail surfaces that show it
        # (list_checkpoints()).
        return RepoState(
            path=self.path, initialized=True,
            head=head, current_branch=branch,
            dirty=dirty, untracked=untracked,
        )

    def archive_total_bytes(self) -> int:
        """Total size of all archived binaries under ``.binsnapshots/``.

        Walks the archive -- an O(archived files) stat sweep -- so this
        is deliberately NOT part of ``state()`` (the refresh-path read,
        § 6.2).  Call it only from one-shot surfaces that actually
        display archive size: the CLI ``snapshot init`` confirmation and
        the list/detail route.
        """
        snaps = Path(self.path) / ".binsnapshots"
        if not snaps.is_dir():
            return 0
        total = 0
        for sub in snaps.rglob("*"):
            if sub.is_file() and sub.name not in ("MANIFEST", ".gitkeep"):
                total += sub.stat().st_size
        return total

    # -- public ref + diff surface (callers shouldn't touch _-prefixed) #

    def resolve_ref(self, ref: str) -> str:
        """Resolve ``ref`` to a commit SHA.  Raises
        :class:`NoSuchRefError` when the ref doesn't exist.  Use this
        from blueprints / external consumers instead of reaching at
        ``_resolve_ref``."""
        self._require_init()
        return self._resolve_ref(ref)

    def diff(self, ref_a: str, ref_b: str,
             pathspec: Optional[List[str]] = None) -> str:
        """Unified-diff text between two refs, optionally restricted to
        ``pathspec`` (a list of git pathspec globs).  Both refs are
        validated up-front so an unknown ref surfaces as
        :class:`NoSuchRefError` rather than a generic git-failure
        string."""
        self._require_init()
        self._resolve_ref(ref_a)
        self._resolve_ref(ref_b)
        argv = ["diff", f"{ref_a}..{ref_b}"]
        if pathspec:
            argv.append("--")
            argv.extend(pathspec)
        return _run_git(argv, cwd=self.path).stdout

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
