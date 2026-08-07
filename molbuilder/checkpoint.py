"""Run-checkpoints — git-based working-dir state management.

See docs/execution/running-a-job.md § 6 for the full design contract.
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
# Mirrors the .gitignore policy in docs/execution/running-a-job.md § 6
# Glob patterns; matched non-recursively against the working dir top level.
# Big-binary classification is ENGINE-SPECIFIC and PERSISTED per repo
# (``.mbcheckpoint.json``, git-tracked), so restore uses the SAME
# classification the checkpoint used and the user can edit it.  These are the
# built-in defaults seeded at ``init`` from the EXPLICIT engine (the web/CLI
# caller already knows it -- a UI<->API contract); the persisted config is the
# source of truth thereafter.  See docs/execution/running-a-job.md § 6
_ENGINE_BIG_BINARY_GLOBS = {
    "siesta": ("*.DM", "*.HSX", "*.TSHS",
               "*.TBT.AVTRANS_*", "*.TBT.CC", "*.TBT.DOS"),
    "pyscf":  ("*.chk", "*.cube"),
}
# Safe default when the engine is unspecified: the UNION of every engine's
# patterns.  Over-archiving is harmless; under-archiving silently loses data.
_DEFAULT_ARCHIVE_GLOBS = tuple(sorted({
    g for globs in _ENGINE_BIG_BINARY_GLOBS.values() for g in globs}))

_CHECKPOINT_CONFIG = ".mbcheckpoint.json"
_CHECKPOINT_CONFIG_SCHEMA = "molbuilder/checkpoint-config@1"

# Engine-independent .gitignore tail: scratch, caches, editor noise, and the
# archive dir itself.  The big-binary section is GENERATED from the repo's
# archive globs (below) so git-ignore and sha-archive never drift apart.
_GITIGNORE_FIXED_TAIL = """\
# Pseudopotential cache files (deterministic from .psml)
*.ion.nc
*.ion.xml

# Scratch / rotating logs
fdf.*.log
WORK_*
INPUT_TMP.*

# Binary archive directory
.binsnapshots/

# Large MD trajectories
*.MD
*.MD_CAR

# IDE / editor noise
*.swp
*~
"""


def _render_gitignore(archive_globs) -> str:
    """Render ``.gitignore`` with the big-binary section generated from
    ``archive_globs`` so the git-ignore set and the sha-archive set stay
    consistent (run-checkpoints.md § 9)."""
    head = ("# molbuilder run-checkpoints contract: this dir is managed by "
            "git.\n"
            "# Large binary state files are archived separately in "
            ".binsnapshots/<sha>/.\n"
            "# See docs/execution/running-a-job.md § 6 for rationale.\n\n"
            "# Big binary state (archived by SHA, not committed) -- "
            "engine-specific, editable\n")
    return head + "\n".join(archive_globs) + "\n\n" + _GITIGNORE_FIXED_TAIL


#: Marks molbuilder's own region of a .gitignore.  Everything between the two
#: lines is regenerated from ``archive_globs``; everything outside is the
#: user's and is preserved byte-for-byte.
#: The first line of a pre-marker generated block -- how one is recognised so
#: it can be excised rather than left in force beside a new section.
_GITIGNORE_LEGACY_HEAD = "# molbuilder run-checkpoints contract:"
_GITIGNORE_BEGIN = "# === molbuilder checkpoint BEGIN ==="
_GITIGNORE_END   = "# === molbuilder checkpoint END ==="


def _write_gitignore_section(gi: Path, archive_globs) -> None:
    """Write molbuilder's marked section into ``gi``, preserving the rest.

    S1 ("a regular file is tracked XOR archived") rests on the ignore set and
    the archive set agreeing, and S1a makes that safe by having ONE writer
    derive both from one resolved list.  This is that writer: a directory that
    already has a ``.gitignore`` -- which every benchmark bundle and every
    worked-in directory does -- must still end up ignoring exactly what the
    archive claims, or the two drift and a big binary lands in git as a blob.

    Three cases, in order:

    1. **An UNMARKED block from before the markers existed is excised first.**
       Repos initialised earlier have one, and appending a new section beside it
       would leave the OLD patterns in force.  Narrow the globs and a file would
       then be ignored by the stale block but no longer archived -- in NO
       snapshot at all, which is S1's data-losing branch.  The legacy block is
       deterministic (a fixed header, a fixed tail), so it can be cut exactly.
    2. **A marked section is replaced in place**, leaving everything outside it
       byte-for-byte.
    3. **Otherwise the section is appended.**
    """
    body = (_GITIGNORE_BEGIN + "\n"
            + _render_gitignore(archive_globs).rstrip("\n") + "\n"
            + _GITIGNORE_END + "\n")
    if not gi.exists():
        gi.write_text(body, encoding="utf-8")
        return
    text = gi.read_text(encoding="utf-8")

    # 1. excise a pre-marker molbuilder block, if one is there.
    if _GITIGNORE_LEGACY_HEAD in text and _GITIGNORE_FIXED_TAIL in text:
        s = text.index(_GITIGNORE_LEGACY_HEAD)
        e = text.index(_GITIGNORE_FIXED_TAIL, s) + len(_GITIGNORE_FIXED_TAIL)
        if e > s:
            text = text[:s] + text[e:]

    # 2. replace a marked section -- ``end > begin`` so a hand-edit that
    #    reversed or duplicated the markers cannot splice head and tail into
    #    each other.
    b = text.find(_GITIGNORE_BEGIN)
    e = text.find(_GITIGNORE_END)
    if b != -1 and e > b:
        text = (text[:b] + body
                + text[e + len(_GITIGNORE_END):].lstrip("\n"))
    # 3. otherwise append.
    elif text.strip():
        text = text.rstrip("\n") + "\n\n" + body
    else:
        text = body
    gi.write_text(text, encoding="utf-8")


def _resolve_archive_globs(engine, archive_globs) -> tuple:
    """Init-time resolution: explicit ``archive_globs`` win; else the
    ``engine``'s built-in defaults; else (engine unspecified) the safe
    union.  Raises for an unknown engine (better a loud error than a silently
    wrong archive set)."""
    if archive_globs:
        return tuple(archive_globs)
    if engine:
        globs = _ENGINE_BIG_BINARY_GLOBS.get(engine)
        if globs is None:
            raise CheckpointError(
                f"unknown engine {engine!r} for checkpoint archive globs; "
                f"known: {sorted(_ENGINE_BIG_BINARY_GLOBS)}.  Pass explicit "
                f"archive_globs=[...] to override.")
        return globs
    return _DEFAULT_ARCHIVE_GLOBS


def _read_archive_globs(path) -> tuple:
    """The repo's persisted big-binary classification (``.mbcheckpoint.json``);
    falls back to the safe union for repos created before this config existed
    or if the config is unreadable (robust -- never crash a restore over it)."""
    cfg = Path(path) / _CHECKPOINT_CONFIG
    if cfg.is_file():
        try:
            from . import persist
            globs = persist.read_json(cfg).get("archive_globs")
            if globs:
                return tuple(globs)
        except Exception:
            pass
    return _DEFAULT_ARCHIVE_GLOBS

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


#: A directory holding one of these declares itself **one unit of work whose
#: subdirectories are its own parts** -- a staged calculation's stages, a
#: job-set's job directories, a benchmark's trials.  Each is already an entry in
#: the persisted-artifact registry (job-contracts.md § 6.1), so this reuses the
#: system's existing self-description instead of inventing a marker file.
_BUNDLE_DESCRIPTORS = ("stages.json", "job-set.json", "bench-manifest.json")


def _is_bundle_root(path: Path) -> bool:
    """Does ``path`` declare itself the root of one multi-directory unit of
    work?  See :data:`_BUNDLE_DESCRIPTORS`."""
    return any((path / name).is_file() for name in _BUNDLE_DESCRIPTORS)


def _scan_subtree(path: Path) -> Tuple[List[str], List[str]]:
    """One pruned walk returning ``(working_dirs, git_repos)`` beneath
    ``path``, both as relative POSIX-ish paths.

    * ``working_dirs`` -- subdirectories holding a working-dir marker
      (``.fdf`` / ``.py`` / ``.run.sh``).  Symlinked decks count, because a
      prepped job directory links its deck in rather than copying it.
    * ``git_repos`` -- subdirectories that are themselves repositories.  The
      walk does **not** descend into one: what is inside another repository is
      that repository's business.

    Dot-directories are skipped entirely.  Before, only ``.git`` and
    ``.binsnapshots`` were -- so a ``.venv/`` beside a run (full of ``.py``)
    read as a nested working directory and blocked ``init`` for a reason that
    had nothing to do with calculations.
    """
    working: List[str] = []
    repos: List[str] = []
    stack: List[Path] = [path]
    while stack:
        try:
            entries = sorted(stack.pop().iterdir())
        except OSError:                       # unreadable dir: not our problem
            continue
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir():
                continue
            if entry.name.startswith("."):    # .git, .binsnapshots, .venv, …
                continue
            rel = str(entry.relative_to(path))
            if (entry / ".git").is_dir():
                repos.append(rel)
                continue                      # never descend into another repo
            if any(f.is_file() and f.name.endswith(_NESTED_WORKING_DIR_MARKERS)
                   for f in entry.iterdir()):
                working.append(rel)
            stack.append(entry)
    return sorted(working), sorted(repos)


#: Directories the archive walk never descends into.  ``.binsnapshots`` would
#: make the archive archive itself; ``.git`` holds packfiles that are git's own
#: business.  Any other dot-directory is tooling scratch, not run state.
_ARCHIVE_SKIP_DIRS = frozenset({".git", ".binsnapshots"})


def _archive_key(root: Path, p: Path) -> str:
    """The archive/MANIFEST key for a working-tree file: its path relative to
    the repo root, POSIX separators.  A file at the top level keys as its bare
    basename, which is what every archive written before nested run folders
    existed contains -- so the key space WIDENED and old archives still read."""
    return p.relative_to(root).as_posix()


def _list_big_binaries(path: Path) -> List[Path]:
    """Big binaries anywhere in the run tree, per the repo's PERSISTED
    classification (``.mbcheckpoint.json`` via ``_read_archive_globs`` --
    engine-specific, user-editable).

    RECURSIVE, and it has to be.  ``.gitignore`` receives the raw globs
    (``*.DM``), and a gitignore pattern with no slash matches at EVERY level --
    so a nested ``coarse/job.DM`` is ignored by git.  A top-level-only archive
    walk would then leave it ignored AND unarchived: in no snapshot at all,
    silently absent after a restore.  Both sides of the classification resolve
    depth the same way or the two disagree, and the disagreement loses data
    (docs/execution/checkpointing.md, L2).

    SYMLINKS ARE SKIPPED.  A carried restart file is a link to the stage that
    produced it until localize-on-run replaces it with a real copy; the
    producer's file is archived once, under its own key, and archiving the link
    as a second copy would both duplicate content and restore a regular file
    where a link belongs."""
    root = Path(path)
    # Dedupe by archive key: OVERLAPPING globs (e.g. "*.DM" and "*.D*") would
    # otherwise list the same file twice -> duplicate MANIFEST entries, which
    # the strict parser REJECTS on restore (trapping the checkpoint).
    found: List[Path] = []
    seen: set = set()
    for pat in _read_archive_globs(root):
        for p in sorted(root.rglob(pat)):
            rel = p.relative_to(root)
            # EVERY component, basename included.  Filtering only the parents
            # let a file like `.hidden.DM` through -- pathlib's ``*`` matches a
            # leading dot -- and the MANIFEST parser rejects a dot-prefixed
            # component anywhere in a key.  The archive would then write a
            # MANIFEST its own parser refuses, and that checkpoint could never
            # be verified or restored again.  Writer and reader agree on the
            # same rule or the archive traps itself.
            if any(part in _ARCHIVE_SKIP_DIRS or part.startswith(".")
                   for part in rel.parts):
                continue
            if p.is_symlink() or not p.is_file():
                continue
            key = rel.as_posix()
            if key not in seen:
                seen.add(key)
                found.append(p)
    return found


def _archive_files(arch: Path) -> List[Path]:
    """Every archived payload file under ``arch``, at any depth.

    RECURSIVE.  Keys became repo-relative paths on 2026-08-06, so an archive
    holding ``01_coarse/job.DM`` has nothing but a DIRECTORY at its top level --
    a top-level-only scan reports such an archive as empty and its size as
    zero, which is what ``list_checkpoints`` and ``_checkpoint_from_sha`` did
    while ``archive_total_bytes`` (already ``rglob``) disagreed."""
    if not arch.is_dir():
        return []
    return [p for p in sorted(arch.rglob("*"))
            if p.is_file() and p.name not in (_MANIFEST_NAME, ".gitkeep")]


def _archive_dir(path: Path, sha: str) -> Path:
    return path / ".binsnapshots" / sha


# --------------------------------------------------------------------- #
#  MANIFEST format -- canonical lockdown per § 10 of                    #
#  docs/execution/running-a-job.md § 6.                                   #
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
    # A ZERO-BYTE MANIFEST is the canonical "this commit archived nothing".
    # Checked first: the trailing-LF rule below is about SEPARATING entries, and
    # an empty file has none to separate.  Every commit gets an archive
    # directory, so a missing one means the archive was LOST -- which is the
    # ambiguity this shape exists to remove.
    if raw == b"":
        return {}
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
        # A ZERO-ENTRY MANIFEST IS LEGAL and means "this checkpoint archived
        # nothing", which is different from "this checkpoint has no archive".
        # Every commit gets an archive directory (``Repo.checkpoint``), so a
        # MISSING directory is evidence of a lost archive rather than of a
        # binary-free run -- the ambiguity ``missing_archive_warning`` has to
        # guess around today.
        return {}
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
        # A key is a REPO-RELATIVE POSIX PATH.  A bare basename is one, which
        # is why every archive written before nested run folders existed still
        # parses.  Each component is validated separately: a traversal or a
        # dot-directory in the middle would let a MANIFEST direct a restore
        # outside the working tree or into .git/.binsnapshots.
        if name.startswith("/") or "\\" in name:
            raise CheckpointError(
                f"malformed MANIFEST in {where}: line {idx} filename "
                f"{name!r} is absolute or uses a backslash; keys are "
                f"repo-relative POSIX paths (§ 10.2).")
        parts = name.split("/")
        for part in parts:
            if part == "" or part == "." or part == "..":
                raise CheckpointError(
                    f"malformed MANIFEST in {where}: line {idx} filename "
                    f"{name!r} has an empty, current- or parent-directory "
                    f"component; a restore must not be able to escape the "
                    f"run directory (§ 10.2).")
            if part.startswith("."):
                raise CheckpointError(
                    f"malformed MANIFEST in {where}: line {idx} filename "
                    f"{name!r} has a dot-prefixed component {part!r}; the "
                    f"canonical archive contains no dotfiles or "
                    f"dot-directories (§ 10.1).")
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


def _published_archive_index(path: Path) -> Dict[str, Path]:
    """``{sha256: an existing archived file with that content}``, read from the
    MANIFESTs of every published archive.

    Only real published archives are indexed -- a directory name must be 40 hex
    chars, so the transient ``<sha>.tmp`` and ``<sha>.old`` of an in-flight
    publish can never be linked against.

    The MANIFEST is trusted for *which* sha a file claims; the claim is verified
    before anything links to it (see ``_link_or_copy``).  Reading the MANIFESTs
    is cheap: they are small text files, one per checkpoint.
    """
    index: Dict[str, Path] = {}
    root = path / ".binsnapshots"
    if not root.is_dir():
        return index
    for arch in sorted(root.iterdir()):
        if not arch.is_dir() or not _SHA_DIR_RE.match(arch.name):
            continue
        manifest = arch / _MANIFEST_NAME
        if not manifest.is_file():
            continue
        try:
            entries = _parse_canonical_manifest(manifest.read_bytes(),
                                                where=str(arch))
        except CheckpointError:
            continue                      # a damaged MANIFEST indexes nothing
        for key, (entry_sha, _size) in entries.items():
            if entry_sha in index:
                continue
            cand = arch / key
            if cand.is_file():
                index[entry_sha] = cand
    return index


def _link_or_copy(src: Path, dst: Path, src_sha: str,
                  index: Dict[str, Path], verified: Dict[str, bool]) -> bool:
    """Put ``src``'s content at ``dst``, reusing an identical archived copy when
    one exists.  Returns True if it was reused (hard link), False if copied.

    **This is L5.** An archived file is never rewritten (I1), so a file whose
    content is already in the archive does not need storing twice -- a hard link
    is a directory entry, and the second checkpoint of an unchanged binary costs
    no disk. Everything downstream is untouched: the archive still has a real
    file at ``<sha>/<key>``, so restore, verify and the MANIFEST format do not
    know the difference.

    **It serves both directory shapes with one rule, which is why it is by
    content rather than by "the attempt did not change".** In the hierarchical
    shape an attempt is immutable, so its files link forever after the first
    save. In the flat shape one ``<id>.DM`` is overwritten every stage, so its
    content differs and it is copied -- correct rather than wasteful
    (``project-layout.md § 6.2``), and nothing has to special-case which shape
    it is in.

    **The candidate is verified before it is trusted.** The index knows only
    what a MANIFEST *claims*; linking to a file that had rotted would record a
    sha the bytes do not have, turning a cheap save into a corrupt one. So the
    candidate is hashed once per checkpoint (memoised in ``verified``) and
    rejected if it does not match. That is one read where a copy would have
    done a read *and* a write, so the I/O is lower too.
    """
    cand = index.get(src_sha)
    if cand is not None and cand.is_file():
        ok = verified.get(src_sha)
        if ok is None:
            try:
                ok = hashlib.sha256(cand.read_bytes()).hexdigest() == src_sha
            except OSError:
                ok = False
            verified[src_sha] = ok
        if ok:
            try:
                os.link(cand, dst)
                return True
            except OSError:
                pass                      # cross-device, or links unsupported
    shutil.copy2(src, dst)
    return False


def _archive_binaries(path: Path, sha: str) -> int:
    """Copy big binaries from working dir into ``.binsnapshots/<sha>/``;
    write MANIFEST in canonical format (§ 10.2).  Returns total bytes
    archived (0 if no binaries).

    ALWAYS writes the directory and a MANIFEST, even with nothing to archive.
    An empty MANIFEST says "this commit archived nothing"; a MISSING directory
    then says "this commit's archive is lost".  Before, absence meant both, and
    ``missing_archive_warning`` had to guess between them from whether OTHER
    commits had archives (its docstring: "a lost archive cannot be proven")."""
    binaries = _list_big_binaries(path)
    if not _SHA_DIR_RE.match(sha):
        raise CheckpointError(
            f"_archive_binaries: SHA dir name {sha!r} is not 40 "
            f"lowercase hex chars (§ 10.1).")
    final = _archive_dir(path, sha)
    # ATOMIC PUBLISH: build the archive in a sibling ``.tmp`` dir and rename it
    # into place only after every binary is copied AND the MANIFEST is written.
    # A crash mid-copy then leaves only the throwaway .tmp -- never a PARTIAL
    # archive at the real path that restore would mistake for complete (silent
    # loss) or the parser would choke on (§ 10.3).
    tmp = final.parent / (sha + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    entries: List[Tuple[str, int, str]] = []
    total = 0
    # L5: what is already archived is not archived again.  Built once per
    # checkpoint, BEFORE the publish, so it never sees this archive's own .tmp.
    index = _published_archive_index(path)
    verified: Dict[str, bool] = {}
    try:
        for src in binaries:
            key = _archive_key(path, src)
            dst = tmp / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Hash the SOURCE, copy, then re-hash the ARCHIVED copy and require
            # they match.  Deriving the MANIFEST sha from the copy alone would
            # make a silent copy corruption (disk error) self-consistent -- the
            # bad bytes would later "verify" against their own bad sha and be
            # restored as truth.  Verify FIDELITY (source == archive) at save
            # time so a corrupt copy fails loudly here, not silently on restore.
            src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
            reused = _link_or_copy(src, dst, src_sha, index, verified)
            if not reused:
                # Only a real copy can be corrupted in transit.  A reused entry
                # was hashed before it was linked, and a hard link is the same
                # inode -- re-reading it would cost a full read to confirm what
                # was just confirmed, which is exactly the cost L5 exists to
                # remove.
                dst_sha = hashlib.sha256(dst.read_bytes()).hexdigest()
                if dst_sha != src_sha:
                    raise CheckpointError(
                        f"archive copy of {key!r} is corrupt: source sha256 "
                        f"{src_sha!r} != archived {dst_sha!r} (disk error?); "
                        f"the checkpoint was NOT safely archived.")
            size = dst.stat().st_size
            entries.append((src_sha, size, key))
            total += size
        _atomic_write_text(tmp / _MANIFEST_NAME,
                           _format_canonical_manifest(entries))
        # RENAME ASIDE, PUBLISH, THEN DELETE.  `rmtree(final)` followed by
        # `os.replace` leaves a window in which NEITHER archive exists: a crash
        # there destroys the archive that was already there and publishes
        # nothing, which is the one outcome "complete archive or nothing" must
        # not include.  Moving it aside first means the worst case is a stray
        # `.old` directory beside a complete archive.
        old = None
        if final.exists():                     # idempotent re-archive
            old = final.parent / (sha + ".old")
            if old.exists():
                shutil.rmtree(old)
            os.replace(final, old)
        try:
            os.replace(tmp, final)             # atomic publish
        except BaseException:
            if old is not None and not final.exists():
                os.replace(old, final)         # put the previous one back
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)  # never leave a stray .tmp
        raise
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
    # A zero-entry MANIFEST is a legitimate "this commit had no big binaries",
    # written so that a MISSING archive is unambiguous.  Nothing to verify.
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
        try:
            dst = path / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(arch / name, dst)
        except OSError as e:
            # Verified moments ago, so this is a TOCTOU/IO fault mid-copy.
            # Surface it clearly (the working tree may be partially restored)
            # rather than let a raw OSError escape.
            raise CheckpointError(
                f"restore: archive verified but copying {name!r} failed "
                f"({e}); the working tree may be partially restored -- fix "
                f"the disk/archive and re-run restore.")
        restored.append(name)
    return restored


def _restore_archived_binaries(path: Path, sha: str) -> List[str]:
    """Verify then copy the archived binaries for ``sha`` (verification
    aborts BEFORE any byte hits the working tree, § 10.3)."""
    expected = _verify_archived_binaries(path, sha)
    return _copy_archived_binaries(path, sha, expected)


def _working_binaries_dirty(path: Path, head_sha: str) -> List[str]:
    """Big-binary files in the working dir that DIFFER from what was archived
    at ``head_sha`` -- i.e. uncommitted binary changes a restore would
    overwrite.  Big binaries are gitignored, so ``git status`` cannot see
    them; restore checks this separately to honor P3 (the user decides; the
    system never silently discards binary work).  Returns sorted names."""
    expected: Dict[str, Tuple[str, int]] = {}
    arch = _archive_dir(path, head_sha)
    manifest = arch / _MANIFEST_NAME
    if arch.is_dir() and manifest.is_file():
        try:
            expected = _parse_canonical_manifest(manifest.read_bytes(),
                                                 where=str(arch))
        except CheckpointError as e:
            # A corrupt HEAD MANIFEST is NOT "you have local changes" -- say so
            # plainly instead of the misleading dirty-binary message below.
            raise CheckpointError(
                f"cannot check for uncommitted binary changes: HEAD's archive "
                f"MANIFEST is unreadable ({e}).  The archive at {arch} is "
                f"corrupt; restore is unsafe until it is repaired.")
    dirty: List[str] = []
    for wb in _list_big_binaries(path):
        key = _archive_key(path, wb)
        want = expected.get(key)
        actual = hashlib.sha256(wb.read_bytes()).hexdigest()
        if want is None or want[0] != actual:
            dirty.append(key)
    return sorted(dirty)


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

    def init(self, engine: Optional[str] = None,
             archive_globs: Optional[List[str]] = None) -> None:
        """Initialise the working dir as a git repository.  Idempotent
        (no-op if already initialised).

        **Scope.** A repository covers one calculation, in either directory
        shape (``execution/project-layout.md`` § 1):

        * **flat** -- one directory, no subdirectories to worry about;
        * **hierarchical** -- a calculation root whose subdirectories are its
          own stages, attempts and benchmark.  Permitted because the root
          carries a description saying they are one unit of work
          (:data:`_BUNDLE_DESCRIPTORS`).

        Two things are refused, for reasons that differ:

        * **a subdirectory that is already a repository** -- a history inside a
          history has no consistent restore;
        * **nested working dirs with nothing declaring them one calculation** --
          that is several independent calculations, and one history over them
          would rewind all of them together.

        ``engine`` (``"siesta"`` / ``"pyscf"``) selects the built-in
        big-binary classification seeded into the persisted config
        (``.mbcheckpoint.json``); the web/CLI caller already knows it at task
        setup (UI<->API contract, run-checkpoints.md § 9).  ``archive_globs``
        overrides with an explicit set.  When neither is given, the safe
        union of all engines' patterns is used.  Both the persisted config
        AND the ``.gitignore`` big-binary section are derived from the same
        resolved globs, so they never drift."""
        p = Path(self.path)
        if self.initialized:
            return

        working, inner_repos = _scan_subtree(p)

        # A repository inside a repository has no consistent restore -- the
        # outer one cannot rewind files the inner one owns.  Refused in EITHER
        # shape, bundle root or not.
        if inner_repos:
            raise NestedRepoRefusedError(
                f"{self.path}: cannot init -- these subdirectories are "
                f"already checkpoint repositories: {inner_repos}.  A history "
                f"inside a history cannot be restored consistently; "
                f"checkpoint them where they are, or move them aside first.")

        # Nested working dirs are fine WHEN THEY ARE THIS CALCULATION'S OWN.
        # A bundle root says so by holding its description; anything else is
        # several independent calculations, and one history over them would
        # rewind all of them together (execution/project-layout.md § 6).
        if working and not _is_bundle_root(p):
            raise NestedRepoRefusedError(
                f"{self.path}: cannot init -- nested working dirs present: "
                f"{working}, and nothing here says they belong to one "
                f"calculation.  Initialising would put several independent "
                f"calculations in one history, so a restore would rewind all "
                f"of them.  Run `snapshot init` inside each instead -- or, if "
                f"these really are one calculation's stages, its root should "
                f"carry the description that says so "
                f"({', '.join(_BUNDLE_DESCRIPTORS)}).")

        globs = _resolve_archive_globs(engine, archive_globs)

        _run_git(["init", "-q"], cwd=self.path)
        host = os.uname().nodename
        _run_git(["config", "user.email", f"molbuilder@{host}"],
                 cwd=self.path)
        _run_git(["config", "user.name", "molbuilder"], cwd=self.path)
        _run_git(["config", "commit.gpgsign", "false"], cwd=self.path)

        # The big-binary section is REGENERATED even when a .gitignore already
        # exists.  Leaving an existing file alone was the one hole in "one list,
        # one writer": archive_globs would say *.DM while git happily tracked
        # it, so the file was archived AND committed -- a large blob in history
        # forever, S1's "never both".  Any directory a user has worked in, and
        # every benchmark bundle, arrives with a .gitignore.
        #
        # Lines the user added are PRESERVED: only molbuilder's own section is
        # rewritten, identified by the marker below.
        _write_gitignore_section(p / ".gitignore", globs)
        # Persist the classification (git-tracked, written BEFORE the commit
        # so it lands in the initial commit and is present for the archive
        # step below).  The UNIFIED accessor for it is archive_globs() /
        # set_archive_globs().
        from . import persist
        persist.write_json(p / _CHECKPOINT_CONFIG, {
            "schema": _CHECKPOINT_CONFIG_SCHEMA,
            "engine": engine or "unspecified",
            "archive_globs": list(globs),
        })
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

    # -- Big-binary classification: the UNIFIED accessor ---------- #

    def archive_globs(self) -> List[str]:
        """The big-binary patterns this repo archives -- the persisted,
        engine-specific, user-editable classification (run-checkpoints.md
        § 9).  THE single read API; CLI, web, and internals all go through
        it (falls back to the safe union for pre-config repos)."""
        return list(_read_archive_globs(Path(self.path)))

    def set_archive_globs(self, globs: List[str]) -> List[str]:
        """Update this repo's big-binary classification (the user-customizable
        table) and regenerate the ``.gitignore`` big-binary section to match,
        so git-ignore and sha-archive stay consistent.  Persisted in
        ``.mbcheckpoint.json``; the change is a normal edit the user then
        checkpoints.  THE single write API (CLI + web share it).  Raises on an
        empty set."""
        self._require_init()
        cleaned = [str(g).strip() for g in globs if str(g).strip()]
        if not cleaned:
            raise CheckpointError(
                "archive_globs cannot be empty -- restore would archive "
                "nothing and silently lose binary state.")
        from . import persist
        p = Path(self.path)
        cfg = p / _CHECKPOINT_CONFIG
        data: Dict[str, Any] = {}
        if cfg.is_file():
            try:
                data = persist.read_json(cfg)
            except Exception:
                data = {}
        data["schema"] = _CHECKPOINT_CONFIG_SCHEMA
        data["archive_globs"] = cleaned
        persist.write_json(cfg, data)
        # keep .gitignore's big-binary section consistent with the archive set,
        # through the same single writer init uses -- and preserving whatever
        # the user put outside molbuilder's section.
        _write_gitignore_section(p / ".gitignore", cleaned)
        return cleaned

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
        text_changed = bool(st.stdout.strip())

        # BIG BINARIES ARE GITIGNORED, so `git status` above is blind to them.
        # A run that rewrote only a .DM leaves the status clean, and a
        # checkpoint that trusted it would report "nothing to commit" while the
        # new density matrix went into no snapshot at all -- surfacing much
        # later as a REFUSED restore ("uncommitted binary changes"), about work
        # the user believed they had saved.  So ask the binaries directly, with
        # the same comparison restore uses.  Only when git already has
        # something to commit do we skip the hashing pass, since the archive is
        # rewritten either way.
        bins_changed = (not text_changed
                        and bool(_working_binaries_dirty(Path(self.path),
                                                         self._head_sha())))
        if not text_changed and not bins_changed:
            return None

        if message is None:
            message = (
                f"checkpoint "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )
        commit_argv = ["commit", "-q", "-m", message]
        if not text_changed:
            # Nothing git can see changed, but the binary state did.  An empty
            # commit gives that state its own sha to be archived under.  The
            # alternative -- rewriting HEAD's archive in place -- would make a
            # restore of HEAD return bytes it never held, which is exactly what
            # an immutable checkpoint must not do.
            commit_argv.insert(1, "--allow-empty")
        _run_git(commit_argv, cwd=self.path)
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
        # Big binaries are gitignored, so the git-status check above cannot
        # see uncommitted changes to them.  Restore would overlay the ref's
        # archived binaries and silently destroy that work -- refuse instead
        # (P3: the user decides; asymmetry with text would be a data-loss
        # trap).  Skipped when include_binaries is False (binaries untouched).
        if include_binaries:
            dirty_bins = _working_binaries_dirty(Path(self.path),
                                                 self._head_sha())
            if dirty_bins:
                raise DirtyWorkingTreeError(
                    "uncommitted binary changes would be overwritten by "
                    f"restore: {', '.join(dirty_bins)}.  Checkpoint or move "
                    "them aside first (big binaries are gitignored, so "
                    "'git status' does not show them).")
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

    def missing_archive_warning(self, ref: str) -> Optional[str]:
        """Return a warning string if restoring ``ref`` would restore NO
        binaries in a project that clearly USES big binaries (the working dir
        has some, or other checkpoints have archives) -- a sign ``ref``'s
        archive is missing/incomplete (e.g. an interrupted checkpoint in the
        commit->archive window), NOT that the checkpoint was legitimately
        binary-free.  Returns ``None`` when there is no reason to warn.

        This is the honest bound on § 10.3: because big binaries are
        gitignored, git records nothing about what a commit "should" have, so
        a lost archive cannot be proven -- but it CAN be flagged loudly so a
        restore never silently returns text-only for a binary project (#1)."""
        self._require_init()
        sha = self._resolve_ref(ref)
        base = Path(self.path)
        if (_archive_dir(base, sha) / _MANIFEST_NAME).is_file():
            return None                          # ref has an archive -> fine
        snaps = base / ".binsnapshots"
        others = [d for d in snaps.iterdir()
                  if d.is_dir() and (d / _MANIFEST_NAME).is_file()] \
            if snaps.is_dir() else []
        if not (_list_big_binaries(base) or others):
            return None                          # binary-free project -> normal
        return (
            f"checkpoint {ref!r} has NO binary archive, but this project uses "
            f"big binaries -- if {ref!r} had .DM/.HSX/.TSHS files they were "
            f"NOT restored (the archive may be incomplete, e.g. a checkpoint "
            f"interrupted between commit and archive).  Verify the result; "
            f"re-checkpoint to heal the archive.")

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
        return sum(f.stat().st_size for f in _archive_files(snaps))

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
            files = _archive_files(_archive_dir(Path(self.path), sha))
            has_arch = bool(files)
            arch_bytes = (sum(f.stat().st_size for f in files)
                          if has_arch else None)
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
        files = _archive_files(_archive_dir(Path(self.path), sha))
        has_arch = bool(files)
        arch_bytes = (sum(f.stat().st_size for f in files)
                      if has_arch else None)
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
