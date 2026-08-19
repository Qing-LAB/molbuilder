"""Checkpointing — saving a calculation folder so its state can be brought back.

**Module:** L1 + L2 (the data model and ``Repo``, the class every surface goes
through).  **Callers:** the ``molbuilder checkpoint`` CLI group
(:mod:`molbuilder.cli`) and the HTTP routes in
``molbuilder/web/blueprints/checkpoint.py``; nothing else in the codebase
constructs a ``Repo``, which is what keeps *"nothing is ever saved without you
saying so"* true.

**Contract:** [`execution/checkpointing.md`](?doc=execution/checkpointing.md) —
the goal, the rules (S/I/A/L) and their status.  The **MANIFEST and archive
formats** are [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
§ 6.1; the **guide** (what to type, what the buttons do) is
[`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 6.

Single user, one repository per calculation, no auto-commit.  **No git ever
runs on a compute node** — a generated wrapper contains none, and initialising
is a CLI/UI act (I4; the "wrapper bootstraps git" path was dropped).

The whole surface is ``Repo``: ``init``, ``save``, ``restore``, ``status``,
``states``, ``tag``.  A **state** is a saved snapshot of the folder, a **tag**
is a name you gave one, and the folder always **stands at** exactly one state.
There is no branch verb -- going back to a state and saving from it is how you
fork (§ 7.1).

Large files do not go into git.  Which files those are is decided by
**measuring** them against a limit in molbuilder's own config (§ 4, S1b), and
they are copied whole into ``.binsnapshots/<digest>/``, where the directory's
name is the sha256 of its own MANIFEST -- so an archive is named by what it
holds rather than by the state that refers to it (§ 3).

Every git invocation goes through :func:`_run_git` -- argv list, ``cwd`` pinned
to the working dir, no shell strings.  Errors raise :class:`CheckpointError`
subclasses with the git stderr verbatim.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------- #
#  Classification -- which store a file goes to                         #
#  Contract: checkpointing.md § 3, S1, S1a, S1b, S1c                    #
# --------------------------------------------------------------------- #

#: Where the archive lives, one directory per distinct content (§ 3).
ARCHIVE_DIR = ".binsnapshots"

#: The only two paths that are never stored (S1).  Not a policy about worth --
#: a store cannot contain itself, and that is the entire list.  Anything else
#: excluded here would be a file in no snapshot, which is S1's losing branch.
#:
#: Derived from :data:`ARCHIVE_DIR` rather than spelling it again: two literals
#: for one directory are two things that can disagree, and the one that would
#: go stale is the exclusion -- leaving the archive walkable by the save that
#: is writing it.
NEVER_STORED = (".git", ARCHIVE_DIR)

_GITIGNORE_BEGIN = "# === molbuilder checkpoint BEGIN ==="
_GITIGNORE_END = "# === molbuilder checkpoint END ==="


def classification_for(engine: Optional[str] = None) -> Dict[str, object]:
    """The size limit and always-large families for this calculation (§ 4).

    Read from molbuilder's own config, never from the folder being saved
    (S1c) -- a per-folder copy is a file somebody can edit between a save and a
    restore, which is exactly the hazard I2c is about.  There is **no**
    directory argument, and that absence is the rule: a function that accepted
    one would read a scope beside the folder being saved, which is the trap.
    """
    from .runtime_config import get_checkpoint
    return get_checkpoint(engine)


def is_big(path: Path, size_limit: int, always_large=()) -> bool:
    """Does *path* belong in the archive rather than in git?

    **The size decides (S1b).**  ``always_large`` names families that are
    always over the limit, so those skip the ``stat`` -- a hint that can make a
    save faster and can never make it store less (§ 3).  A file matching no
    hint is measured, which is why an unknown engine is merely slower.
    """
    if any(fnmatch.fnmatch(path.name, pat) for pat in always_large):
        return True
    try:
        return path.stat().st_size > size_limit
    except OSError:
        # Unreadable is not "small".  Calling it big sends it to the archive,
        # where :func:`publish_archive` refuses the save and names the file --
        # rather than to git, which would commit whatever it could read.
        return True


def render_gitignore(always_large=()) -> str:
    """The generated ignore block -- ONE source, and it is the classification.

    S1a: what git skips must be exactly what the archive takes, or a file falls
    between them and is in no snapshot at all.  So this renders **nothing but
    archive patterns** plus the archive directory itself; there is no fixed
    tail, because every line in one would be a file excluded by code nobody
    agreed to (S1).

    The size gate cannot be expressed as a gitignore pattern, so a file that is
    big *by measurement* is kept out of git by the save's pathspec instead of
    by a line here (see :meth:`Repo.save`).
    """
    lines = [
        _GITIGNORE_BEGIN,
        "# Generated by molbuilder -- do not edit.",
        "# Rewritten from the classification on every save; edits are lost.",
        "# See docs/execution/checkpointing.md S1a.",
        "",
        f"{ARCHIVE_DIR}/",
    ]
    lines += sorted(always_large)
    lines += [_GITIGNORE_END, ""]
    return "\n".join(lines)


def write_gitignore(root: Path, always_large=()) -> None:
    """Write the generated block, preserving anything outside the markers.

    A user's own entries above or below the markers are theirs and are left
    byte-for-byte; everything between them is regenerated, which is what makes
    a hand edit inside detectable by recomputing rather than by a digest
    (I2b).

    **Written atomically**, like the MANIFEST and for the same reason (S9).
    ``write_text`` truncates and then writes, so a second saver -- or the
    ``git add`` of the save already running -- can read the file in the window
    between the two and find it empty, which is an ignore list that excludes
    nothing. Two savers write identical bytes here, so the race has no wrong
    *content* to reach; it had a wrong *moment*.
    """
    gi = root / ".gitignore"
    body = render_gitignore(always_large)
    if not gi.exists():
        _atomic_write_bytes(gi, body.encode("utf-8"), tmp_dir=root / ".git")
        return
    text = gi.read_text(encoding="utf-8")
    b, e = text.find(_GITIGNORE_BEGIN), text.find(_GITIGNORE_END)
    if b != -1 and e > b:
        text = text[:b] + body + text[e + len(_GITIGNORE_END):].lstrip("\n")
    elif text.strip():
        text = text.rstrip("\n") + "\n\n" + body
    else:
        text = body
    _atomic_write_bytes(gi, text.encode("utf-8"), tmp_dir=root / ".git")


def gitignore_is_current(root: Path, always_large=()) -> bool:
    """Does the marked block match what regenerating it would produce (I2b)?

    `.gitignore` records a *derivation*, so the derivation is the check: a
    stored digest could not catch an edit that rode into a save, because the
    save would hash whatever it found and bless it.
    """
    gi = root / ".gitignore"
    if not gi.is_file():
        return False
    text = gi.read_text(encoding="utf-8")
    b, e = text.find(_GITIGNORE_BEGIN), text.find(_GITIGNORE_END)
    if b == -1 or e <= b:
        return False
    return text[b:e + len(_GITIGNORE_END)] == render_gitignore(
        always_large).rstrip("\n")


#: A subdirectory holding one of these is somebody's working directory.  One
#: repository covers one calculation (L1), so a folder full of these is only
#: acceptable when the folder itself says they are one calculation's parts.
#:
#: **`.py` is here for PySCF, whose deck IS a `.py`**, and it is the entry with
#: a known false positive: an ordinary `analysis/` or `scripts/` subdirectory
#: holding any Python file reads as a working directory too.  Narrowing it is
#: not free -- dropping `.py` would let a folder of PySCF runs be initialised
#: as one history with nothing declaring them one calculation, which is the
#: failure L1 exists for, and no extension can tell `<id>.py` from `plot.py`.
#:
#: So the breadth is deliberate and the remedy is one file: the refusal names
#: the directories it found and says to put a description at the root, which is
#: what a real staged calculation has anyway.  Dot-directories are skipped
#: before this is consulted, which is what stopped `.venv/` tripping it.
_NESTED_WORKING_DIR_MARKERS = (".fdf", ".py", ".run.sh")


# --------------------------------------------------------------------- #
#  Exceptions                                                           #
# --------------------------------------------------------------------- #


class CheckpointError(Exception):
    """Base class for every refusal this module raises."""


class GitNotInstalledError(CheckpointError):
    """``git`` is not on PATH."""


class DirtyWorkingTreeError(CheckpointError):
    """Operation refuses to proceed on a dirty working tree."""


class NoSuchRefError(CheckpointError):
    """The given name is neither a state id nor a tag (§ 5)."""


class NestedRepoRefusedError(CheckpointError):
    """Init refused: this folder holds several independent calculations, and
    one history over them would rewind all of them together (L1)."""


class CalculationNameError(CheckpointError):
    """The calculation's name would have to be repaired to be used (L3).

    Its own class because it is the one refusal here a **user** resolves, by
    choosing a name or renaming the folder.  Every other failure in this module
    is a fault -- git missing, an archive damaged, a copy corrupt -- and a
    surface that cannot tell them apart reports "please fix your input" for a
    broken disk.
    """


# --------------------------------------------------------------------- #
#  Data model                                                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class State:
    """One saved snapshot of the whole folder (§ 5).

    ``id`` is git's own commit hash: permanent, never reused, and nothing
    assigns or stores it.  ``archive`` is the sha256 of that state's MANIFEST —
    a *name* and a *proof* in one value (I2b), present on every state including
    those that archived nothing.
    """
    id:          str
    note:        str
    parent:      Optional[str]      # the state this one came from
    at:          str                # ISO-8601, with the offset git recorded
    archive:     Optional[str] = None   # Manifest-SHA256; None if damaged
    calculation: Optional[str] = None   # which calculation this belongs to
    tags:        Tuple[str, ...] = ()

    @property
    def short(self) -> str:
        return self.id[:7]


@dataclass(frozen=True)
class Tag:
    """A name you gave a state so you could find it again (§ 5, L4).

    Nothing creates one on your behalf; the namespace is yours alone.
    """
    name:  str
    state: str
    note:  str


@dataclass(frozen=True)
class FolderStatus:
    """Where the folder stands, and what is unsaved (§ 5, A5).

    ``standing_at`` is the one state the folder is currently at, and everything
    else is measured **against that state and never against the newest one** —
    which is why going back does not make the whole folder read as modified.
    It is the **State**, not its id: answering "what is unsaved" requires
    reading it anyway, so returning only the id made every caller that wanted
    the note or the parent ask git for the same thing a second time.

    The three lists are A5's three shapes, and all three are lost when a
    restore makes the folder equal its target.
    """
    path:        str
    initialized: bool
    standing_at: Optional["State"] = None
    changed:      Tuple[str, ...] = ()
    added:        Tuple[str, ...] = ()
    deleted:      Tuple[str, ...] = ()
    ignore_edited: bool = False     # the generated block was hand-edited (I2b)

    @property
    def clean(self) -> bool:
        return not (self.changed or self.added or self.deleted)

    def unsaved(self) -> Tuple[str, ...]:
        """Everything at risk, in one sorted list — what A5's warning names."""
        return tuple(sorted(set(self.changed) | set(self.added)
                            | set(self.deleted)))


# --------------------------------------------------------------------- #
#  Internals                                                            #
# --------------------------------------------------------------------- #


def _run_git(argv: List[str], cwd: str, *,
             check: bool = True,
             stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run ``git argv`` in ``cwd``; return CompletedProcess.

    Sets a molbuilder git identity in the environment so a checkpoint works on
    a machine where the user never ran ``git config --global user.email``.
    ``setdefault``, so a real identity already present is left alone.
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
            input=stdin,
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
# ``bench-manifest.json`` was the third entry until U19 (2026-08-12): its
#: producer (``bench generate``, the shipped-bundle lifecycle) died in step
#: 6 u5, and a descriptor nothing writes declares nothing.
_BUNDLE_DESCRIPTORS = ("task.json", "job-set.json")


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
            try:
                marked = any(
                    f.is_file() and f.name.endswith(_NESTED_WORKING_DIR_MARKERS)
                    for f in entry.iterdir())
            except OSError:
                marked = False            # unreadable: not our problem, as above
            if marked:
                working.append(rel)
            stack.append(entry)
    return sorted(working), sorted(repos)


def archive_key(root: Path, path: Path) -> str:
    """A file's key inside an archive: its path relative to the folder root.

    Repo-relative rather than a bare basename, so one archive can hold a `.DM`
    from every stage without them colliding (L2).
    """
    return path.relative_to(root).as_posix()


def walk_entries(root: Path):
    """Every entry outside the two stores — regular files **and symlinks**.

    The one traversal the two walks below share, and the one a **restore**
    sweeps with: deciding what is a leftover is a question about everything in
    the folder, not only about the things a store holds.

    **Pruned, not filtered.**  ``rglob`` descends into every directory and lets
    the caller discard what it collected, which means walking the whole archive
    -- every big file, once per state that ever held it -- to answer a question
    about the working tree.  That cost grows with the length of the history and
    is paid on every directory-enter, so the stores are never entered at all.
    """
    stack: List[Path] = [root]
    while stack:
        try:
            entries = sorted(stack.pop().iterdir())
        except OSError:                       # unreadable dir: not our problem
            continue
        for entry in entries:
            # Any component, not just the top one: a stray `sub/.git` is
            # somebody else's repository and is no more ours to store than our
            # own.  The walk and the key check must exclude the same set, or a
            # file the walk collects is a file the MANIFEST refuses and the
            # save dies on.
            if entry.name in NEVER_STORED:
                continue
            if entry.is_symlink():
                # Yielded, never descended through: a link to an ancestor would
                # make the walk infinite, and what is inside the target is the
                # target's business.  Checked BEFORE `is_dir`/`is_file`, which
                # follow the link and would classify a dangling one as neither.
                yield entry
            elif entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                yield entry


def walk_files(root: Path):
    """Every regular file that belongs in a store (S1).

    Skips only :data:`NEVER_STORED` — the two stores themselves — and symlinks.
    **No other exclusion**: "no category of file is exempt", and a walk with a
    private skip list is a walk that agrees with itself about files nobody
    stores.

    Symlinks are excluded because a link has no content of its own: the real
    file is stored once wherever it lives, and following the link would archive
    that content a *second* time under the link's path.  The link itself is
    still saved -- to git, by :meth:`Repo.save`, which is what makes S1's
    carve-out a carve-out from the **archive** rather than from the snapshot.
    """
    for entry in walk_entries(root):
        if not entry.is_symlink():
            yield entry


def walk_symlinks(root: Path):
    """Every symlink outside the two stores.

    A save needs these by name: an always-large hint matches on a *name*, and a
    name says nothing about a link.  A carry link called ``job.DM`` is twenty
    bytes of path text that `.gitignore` would send to the store that does not
    take links -- so it would land in neither, and a restore would neither
    bring it back nor remove it (S1).
    """
    for entry in walk_entries(root):
        if entry.is_symlink():
            yield entry


def big_files(root: Path, size_limit: int, always_large=()) -> List[Path]:
    """The files that go to the archive rather than to git (S1, S1b).

    Only one side is ever wanted: git is told about everything else by a
    pathspec that names *these*, so building a second list of the small ones
    was work nobody consumed -- on every save and every directory-enter, over
    every file in the folder.
    """
    return [path for path in walk_files(root)
            if is_big(path, size_limit, always_large)]


# --------------------------------------------------------------------- #
#  MANIFEST + the content-addressed archive                             #
#  Format: job-contracts.md § 6.1.  Naming: checkpointing.md § 3.       #
# --------------------------------------------------------------------- #

#: I2b: a record named so nobody mistakes it for a setting.  Chosen once, here;
#: nothing has to read an older name.
MANIFEST_NAME = "MANIFEST.do_not_edit"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIZE_RE   = re.compile(r"^(0|[1-9][0-9]*)$")
_SEP       = "\t"

#: sha256 of b"" -- the archive every state with no big files points at.
EMPTY_MANIFEST_DIGEST = hashlib.sha256(b"").hexdigest()


def _bad(where: str, why: str) -> "CheckpointError":
    return CheckpointError(f"malformed MANIFEST in {where}: {why} "
                           f"(job-contracts.md § 6.1).")


def format_manifest(entries: Sequence[Tuple[str, int, str]]) -> bytes:
    """Render entries as the canonical MANIFEST (job-contracts.md § 6.1).

    One set of files has exactly **one** possible MANIFEST, byte for byte,
    because its sha256 is the archive's directory name (§ 3).  Sorting and the
    single spelling of every field are what make that true, so this refuses to
    emit anything it could not read back.
    """
    seen = set()
    lines = []
    for sha256, size, key in sorted(entries, key=lambda e: e[2]):
        if not _SHA256_RE.match(sha256):
            raise CheckpointError(
                f"cannot write MANIFEST: {sha256!r} is not 64 lowercase hex "
                f"characters (job-contracts.md § 6.1).")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CheckpointError(
                f"cannot write MANIFEST: size {size!r} for {key!r} is not a "
                f"non-negative integer (job-contracts.md § 6.1).")
        _check_key(key, "cannot write MANIFEST")
        if key in seen:
            raise CheckpointError(
                f"cannot write MANIFEST: {key!r} appears more than once; a key "
                f"names one file (job-contracts.md § 6.1).")
        seen.add(key)
        lines.append(f"{sha256}{_SEP}{size}{_SEP}{key}")
    return ("".join(ln + "\n" for ln in lines)).encode("ascii")


def _check_key(key: str, prefix: str) -> None:
    """A key must never steer a restore out of the folder (job-contracts § 6.1)."""
    if not key:
        raise CheckpointError(f"{prefix}: an empty key is not a path.")
    if any(ord(c) < 0x20 or ord(c) > 0x7E for c in key):
        raise CheckpointError(
            f"{prefix}: {key!r} contains non-printable or non-ASCII "
            f"characters (job-contracts.md § 6.1).")
    if key.startswith("/") or "\\" in key:
        raise CheckpointError(
            f"{prefix}: {key!r} must be a repo-relative POSIX path "
            f"(job-contracts.md § 6.1).")
    for part in key.split("/"):
        if part in ("", ".", ".."):
            raise CheckpointError(
                f"{prefix}: {key!r} has an empty or dot component, which "
                f"could steer a restore outside the folder "
                f"(job-contracts.md § 6.1).")
        if part in NEVER_STORED:
            raise CheckpointError(
                f"{prefix}: {key!r} names a store ({part}), and a restore "
                f"writing there would corrupt the history it is restoring "
                f"from (job-contracts.md § 6.1).")


def parse_manifest(raw: bytes, where: str) -> Dict[str, Tuple[str, int]]:
    """Strict reader.  Returns ``{key: (sha256, size)}``.

    Refuses everything that is not exactly the canonical form — no field-count
    fallback, no header, no comments, no BOM tolerance.  A reader that guesses
    is a reader that restores the wrong bytes, and under content addressing a
    reader that tolerates two spellings cannot agree with the writer about the
    archive's name.
    """
    if raw == b"":
        return {}                       # legal: this state archived nothing
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _bad(where, "starts with a UTF-8 BOM; the format is plain ASCII")
    if b"\r" in raw:
        raise _bad(where, "contains CR bytes; line endings are LF only")
    if not raw.endswith(b"\n"):
        raise _bad(where, "missing the final newline")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise _bad(where, f"non-ASCII byte at offset {e.start}") from e

    out: Dict[str, Tuple[str, int]] = {}
    keys: List[str] = []
    for idx, line in enumerate(text.split("\n")[:-1], start=1):
        if not line:
            raise _bad(where, f"line {idx} is blank")
        parts = line.split(_SEP)
        if len(parts) != 3:
            raise _bad(where, f"line {idx} has {len(parts)} tab-separated "
                              f"field(s); the format is exactly three")
        sha256, size_s, key = parts
        if not _SHA256_RE.match(sha256):
            raise _bad(where, f"line {idx}: {sha256!r} is not 64 lowercase "
                              f"hex characters")
        if not _SIZE_RE.match(size_s):
            raise _bad(where, f"line {idx}: {size_s!r} is not a decimal "
                              f"integer without leading zeros")
        _check_key(key, f"malformed MANIFEST in {where}, line {idx}")
        if key in out:
            raise _bad(where, f"line {idx}: {key!r} appears more than once")
        out[key] = (sha256, int(size_s))
        keys.append(key)
    if keys != sorted(keys):
        raise _bad(where, "lines are not sorted by key; two writers would "
                          "otherwise produce two different archives")
    return out


def manifest_digest(raw: bytes) -> str:
    """The archive's name: the sha256 of its own MANIFEST (§ 3).

    One value that locates the archive, proves it, and makes it impossible to
    modify without becoming a different archive.
    """
    return hashlib.sha256(raw).hexdigest()


def archive_dir(root: Path, digest: str) -> Path:
    """``<root>/.binsnapshots/<digest>/`` — named by content, not by state."""
    if not _SHA256_RE.match(digest):
        raise CheckpointError(
            f"archive name {digest!r} is not a sha256 digest; an archive is "
            f"named by its MANIFEST's content (checkpointing.md § 3).")
    return root / ARCHIVE_DIR / digest


def _atomic_write_bytes(target: Path, data: bytes,
                        tmp_dir: Optional[Path] = None) -> None:
    """Write via a **unique** temp + ``os.replace``.

    The shape was built HERE (two properties, and the second was missing
    package-wide: a reader never sees a partial file, and two writers never
    collide — the derived-name trap § 6 names, plus the ``tmp_dir``
    never-stored escape for targets inside the folder being saved).  At U8
    (2026-08-12) it moved to :func:`molbuilder.persist.write_bytes` so every
    persisted artifact writes the same way; this name stays because it is
    this module's seam — the states tests inject failure through it.
    """
    from .persist import write_bytes
    write_bytes(target, data, tmp_dir=tmp_dir)


#: Big enough that the syscall overhead disappears, small enough that a 2 GB
#: density matrix never becomes a 2 GB allocation.
_HASH_CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    """The sha256 of a file, read in chunks.

    Every hash in this module goes through here.  ``read_bytes()`` would load
    the whole file first, and the files this system exists for are the large
    ones: a folder of density matrices would need as much memory as it needs
    disk, on a save, on a verify, and on every exact status.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refuse_unstorable(root: Path, path: Path) -> str:
    """The archive key for *path*, or a refusal a person can act on.

    ``_check_key`` guards the MANIFEST against a key that could steer a restore
    out of the folder, and its message is written for whoever is reading the
    record.  Reaching it from a save means an ordinary file the user just
    created cannot be archived -- most often a name outside ASCII, which the
    format does not carry -- so the refusal has to say which file and what to
    do about it, rather than cite a format section.
    """
    key = archive_key(root, path)
    try:
        _check_key(key, "cannot archive")
        return key
    except CheckpointError as exc:
        raise CheckpointError(
            f"{path} cannot go into the archive: {exc}\n\n"
            f"This file is stored in `.binsnapshots/` rather than in git, and "
            f"the archive's record (MANIFEST) is plain ASCII.  Either rename "
            f"the file, or adjust `checkpoint.size_limit_bytes` / "
            f"`checkpoint.engines` in molbuilder.json so it goes to git "
            f"instead -- git carries the name unchanged.  Nothing was saved."
        ) from exc


# --------------------------------------------------------------------- #
#  Publishing an archive (A1, I1, § 6)                                  #
# --------------------------------------------------------------------- #


def _existing_by_sha(root: Path, wanted: Sequence[str]) -> Dict[str, Path]:
    """Where each of *wanted* already sits in the archive, if it does.

    Serves the *Disk cost* property (§ 12): identical content is stored once, so
    a second save of an unchanged 2 GB binary costs a directory entry.

    **Bounded, not exhaustive.**  It takes the hashes it is looking for and
    stops as soon as it has them all, because the alternative is parsing every
    MANIFEST of every archive on every save -- work that grows with the length
    of the history rather than with what is being saved, and that is paid even
    when the answer was found in the first directory.
    """
    index: Dict[str, Path] = {}
    outstanding = set(wanted)
    if not outstanding:
        return index
    base = root / ARCHIVE_DIR
    if not base.is_dir():
        return index
    # Newest first: a file being saved again is likeliest to sit in a recent
    # archive, so the common case reads one MANIFEST rather than all of them.
    for adir in sorted(base.iterdir(), reverse=True):
        if not outstanding:
            break
        man = adir / MANIFEST_NAME
        # A staging directory is not an archive until it is renamed into place;
        # indexing one would offer a half-written file for reuse.
        if not (adir.is_dir() and man.is_file()) or not _SHA256_RE.match(adir.name):
            continue
        try:
            entries = parse_manifest(man.read_bytes(), str(adir))
        except CheckpointError:
            continue                     # a damaged archive indexes nothing
        for key, (sha256, _size) in entries.items():
            if sha256 in outstanding:
                index[sha256] = adir / key
                outstanding.discard(sha256)
    return index


def publish_archive(root: Path, big: Sequence[Path]) -> str:
    """Write the archive for *big* and return its digest (§ 6).

    Hash the source, copy into a private staging directory, **re-hash the
    copy** and compare, write the MANIFEST, then rename into place at the
    digest — *create if absent, never overwrite* (A1).  An archive at a given
    name always holds the same content, so there is nothing to replace and
    nothing to move aside.

    Re-hashing the copy is not belt-and-braces: if the MANIFEST's checksum came
    from the copy alone, a copy corrupted on the way to disk would be
    *self-consistent* and would verify against its own bad checksum forever.

    **New content is therefore read twice and written once, and that is the
    floor.** The source is hashed (its digest names the archive), copied, and
    the copy read back *from disk* -- reading back is the whole point, so
    hashing the buffer on the way past would prove nothing.

    **Content already in the archive is hard-linked instead of copied, and it
    is checked exactly the same way.**  A link avoids the *write*, never the
    read.  This paragraph used to end *"...and no re-check, because a hard link
    is the same inode that was verified when it was first written"* -- which is
    what the body did until the check moved below the branch, and it survived
    here describing behaviour that is gone.  Verified *when written* is true;
    unchanged *since* is the assumption I2b exists because it fails, so a
    damaged inode would otherwise be linked into a brand-new archive whose
    MANIFEST disagrees with its own bytes from the moment it is published.
    """
    entries: List[Tuple[str, int, str]] = []
    for src in big:
        # The key is checked BEFORE anything is hashed or copied: a name the
        # record cannot carry should cost a message, not a sweep over every
        # large file in the folder first.
        key = _refuse_unstorable(root, src)
        try:
            entries.append((sha256_of(src), src.stat().st_size, key))
        except OSError as exc:
            # S1 exempts nothing, so a file that cannot be read STOPS the save
            # rather than being quietly left out of it -- a snapshot of most of
            # a folder is not a snapshot of the folder.
            raise CheckpointError(
                f"cannot read {src} in order to archive it: {exc}.  Every file "
                f"in the folder is stored, so this stops the save rather than "
                f"producing a snapshot missing one file (checkpointing.md S1). "
                f"Fix the permissions, or move the file out of the calculation "
                f"folder.") from exc
    raw = format_manifest(entries)
    digest = manifest_digest(raw)
    final = archive_dir(root, digest)
    if (final / MANIFEST_NAME).is_file():
        # ALREADY PUBLISHED -- BUT THE NAME PROVES WHAT IT SHOULD HOLD, NOT
        # THAT IT STILL DOES.
        #
        # This used to `return digest` on the strength of the name alone, and
        # that made a save adopt a damaged archive and report success: the new
        # state carried a digest whose archive no longer verified, so a state
        # the user was told they could return to was not restorable.  § 1's
        # promise, broken by the one operation that makes it.
        #
        # I2b allows two outcomes and not three -- "it matches, or it is
        # refused" -- and skipping the question entirely was the third.
        #
        # SHALLOW, and that is the considered answer rather than a shortcut.
        # Git commits over a corrupt object store without looking at it:
        # verification lives on the READ, where zlib gives it away for free,
        # and in `fsck`.  Our archive is raw, so the read cannot be free -- but
        # putting a full re-hash HERE would double the commonest save there is
        # (a tweak to an input, gigabytes of density matrices untouched) to
        # catch, one save earlier, damage that `restore` refuses anyway.  What
        # the cheap check still buys is the moment: right now the files that
        # could rebuild this archive are on disk.
        #
        # The way out is cheap and safe *because* the archive is content-
        # addressed: delete the damaged directory and save again, and it is
        # rebuilt byte-identically from the files still in the folder -- which
        # are the very files whose content matched, or this branch would not
        # have been reached.  Repairing it here instead would be molbuilder
        # quietly writing over an archive, which I1 does not allow it to do.
        try:
            verify_archive(root, digest, deep=False)
        except CheckpointError as exc:
            raise CheckpointError(
                f"the archive this save would reuse is damaged, so the state "
                f"would not be restorable: {exc}\n\n"
                f"Nothing was saved.  The archive is named by its content, so "
                f"removing it and saving again rebuilds it from the files in "
                f"this folder:\n"
                f"    rm -rf {archive_dir(root, digest)}\n"
                f"    molbuilder checkpoint save -m \"…\"\n"
                f"(checkpointing.md I2b, § 1)."
            ) from exc
        return digest

    # A UNIQUE staging directory per publisher, not `<digest>.tmp`.
    #
    # Content addressing makes the FINAL name safe to race for -- same content,
    # same digest, same bytes -- but it makes the temporary name *collide*,
    # because two publishers of the same content agree about that too.  Sharing
    # it, one `rmtree`s the directory the other is still writing into: the
    # loser sees a vanished file, and the winner can publish a half-copied
    # archive under a name that promises the opposite.
    #
    # So the shared point is reduced to the single `os.replace` below, which is
    # atomic and already handles "somebody else got there first".
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=str(final.parent), prefix=digest + "."))
    index = _existing_by_sha(root, [sha for sha, _size, _key in entries])
    try:
        for sha256, _size, key in entries:
            src = root / key
            dst = tmp / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            reuse = index.get(sha256)
            linked = False
            if reuse is not None and reuse.is_file():
                try:
                    os.link(reuse, dst)
                    linked = True         # no copy: the same inode, reused
                except OSError:
                    pass                  # cross-device or unsupported: copy
            if not linked:
                shutil.copy2(src, dst)
            # CHECKED THE SAME WAY WHETHER IT WAS COPIED OR LINKED.
            #
            # This used to `continue` past the check after a link, reasoning
            # that a hard link is "the same inode that was verified when it was
            # first written".  Verified *when written* is true; unchanged
            # *since* is the assumption, and I2b exists precisely because that
            # assumption fails -- bit rot, or somebody tidying a directory.
            # Reusing a damaged inode would build a new archive whose MANIFEST
            # disagrees with its own bytes from the moment it is published.
            #
            # A link still avoids the write, which is the expensive half; what
            # it no longer avoids is the read.
            if sha256_of(dst) != sha256:
                raise CheckpointError(
                    f"archive copy of {key!r} is corrupt: it does not match "
                    f"the source it was copied from.  Refusing to publish an "
                    f"archive that would verify against its own bad checksum "
                    f"(checkpointing.md § 6).")
            # AND THIS ARCHIVE'S OWN COPIES ARE OFFERED FOR REUSE TOO.
            #
            # *Identical content is stored once* (§ 12) was only ever true
            # ACROSS saves: the index is built from already-PUBLISHED archives,
            # so two paths holding the same bytes in the SAME save were each
            # copied in full.  A stage that carries its predecessor's 2 GB
            # density matrix forward -- by copy, or by a hard link in the
            # working tree, which the walk sees as two ordinary files -- cost
            # 4 GB of archive, on every save.
            #
            # Added only after the verify, so nothing unchecked is ever offered
            # as a link target.  The MANIFEST is untouched by this: same shas,
            # same sizes, same keys, so the digest is identical and every
            # archive already on disk stays exactly as valid as it was.
            index.setdefault(sha256, dst)
        _atomic_write_bytes(tmp / MANIFEST_NAME, raw)
        try:
            os.replace(tmp, final)
        except OSError:
            if (final / MANIFEST_NAME).is_file():
                shutil.rmtree(tmp, ignore_errors=True)   # someone else won
            else:
                raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return digest


def verify_archive(root: Path, digest: str, *,
                   deep: bool = True) -> Dict[str, Tuple[str, int]]:
    """Check an archive against its MANIFEST, touching nothing (I2).

    Returns the expected map, so a caller can act on what it just verified
    rather than reading the record twice.

    **Two depths, and it is § 7.2's split pointed at the archive** — who pays
    for exactness, and what being wrong costs.

    ``deep=True`` reads every file and compares its sha256.  This is what runs
    before a **restore**, and there it is not optional: unlike git's own
    objects, these are stored raw.  Git objects are zlib-compressed, so reading
    one validates it as a side effect — corrupt one and `git checkout` answers
    *"inflate: data stream error"* and restores nothing.  Ours would hand the
    bytes over, so the check has to be explicit.  This *is* our inflate check,
    not extra caution.

    ``deep=False`` reads the MANIFEST — one small file, whose digest is the
    archive's own name — and then checks existence and size.  It catches a lost
    archive, a truncated file, a deleted entry: the damage that actually
    happens, for a stat each.

    **A save uses the shallow one, and that is deliberate.** Git commits over a
    corrupt object store without looking at it; verification belongs on the
    read and in `fsck`, not on every write.  Re-hashing here would double the
    cost of the commonest save there is — a tweak to an input, with gigabytes
    of density matrices untouched — to catch, one save earlier, damage the
    restore refuses anyway.  What the shallow check still buys is the moment:
    at save time the files that could rebuild the archive are still on disk.
    """
    adir = archive_dir(root, digest)
    man = adir / MANIFEST_NAME
    if not man.is_file():
        raise CheckpointError(
            f"archive {digest[:12]}… is missing: no {MANIFEST_NAME} at {adir}. "
            f"The state that names it recorded this archive, so it was lost "
            f"(checkpointing.md I2b).")
    raw = man.read_bytes()
    if manifest_digest(raw) != digest:
        raise CheckpointError(
            f"archive {digest[:12]}…: its {MANIFEST_NAME} does not hash to the "
            f"name it is stored under.  The record was modified "
            f"(checkpointing.md I2b).")
    expected = parse_manifest(raw, str(adir))
    for key, (sha256, size) in expected.items():
        src = adir / key
        if not src.is_file():
            raise CheckpointError(
                f"archive {digest[:12]}…: {MANIFEST_NAME} lists {key!r} but "
                f"the file is missing; refusing to restore (I2).")
        if src.stat().st_size != size:
            raise CheckpointError(
                f"archive {digest[:12]}…: {key!r} is {src.stat().st_size} "
                f"bytes, the record says {size}; refusing to restore (I2).")
        if deep and sha256_of(src) != sha256:
            raise CheckpointError(
                f"archive {digest[:12]}…: {key!r} does not match its recorded "
                f"sha256; refusing to restore (I2).")
    return expected


# --------------------------------------------------------------------- #
#  The trailer -- one value that is both pointer and proof (I2b)        #
# --------------------------------------------------------------------- #

TRAILER = "Manifest-SHA256"
CALC_TRAILER = "Calculation"
_TRAILERS = (TRAILER, CALC_TRAILER)

#: L3: used verbatim in every state's message and never rewritten, so a name
#: needing repair is refused rather than quietly fixed -- silently normalising
#: an id would decouple the history's name from the folder's.
_CALC_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def check_calculation_name(name: str) -> str:
    """L3: refuse a name that would have to be repaired to be used."""
    if not _CALC_NAME_RE.match(name or ""):
        raise CalculationNameError(
            f"calculation name {name!r} is not [A-Za-z0-9_-]+.  It is written "
            f"verbatim into every state and never rewritten, so a name that "
            f"needs repair is refused rather than quietly changed -- fixing it "
            f"silently would decouple this history's name from the folder's "
            f"(checkpointing.md L3).")
    return name


def message_with_trailers(note: str, digest: str, calculation: str) -> str:
    """A state's message: your note, then what the state is *of* and *holds*.

    Both trailers are appended by the SAVE, never by a note helper: a save may
    carry any note at all, and hanging them off a helper would let every save
    that skipped it ship unanchored and unnamed (I2b, L3).
    """
    return (f"{note.rstrip()}\n\n"
            f"{CALC_TRAILER}: {calculation}\n"
            f"{TRAILER}: {digest}\n")


def _trailer_block(message: str) -> Tuple[List[str], List[str]]:
    """Split a message into ``(note_lines, trailer_lines)``.

    Trailers are the **last** block of ``Key: value`` lines, which is git's own
    convention and not a preference.  Scanning the whole message instead lets a
    note that happens to begin *"Calculation: rerun with a tighter mesh"* be
    eaten as a trailer -- vanishing from the note AND becoming the calculation's
    name.  A note is free text a person wrote; only the tail is ours.
    """
    lines = message.splitlines()
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1                                  # ignore trailing blank lines
    start = end
    while start and any(lines[start - 1].startswith(k + ":") for k in _TRAILERS):
        start -= 1
    if start == end:
        return lines, []
    return lines[:start], lines[start:end]


def _trailer_value(message: str, key: str) -> Optional[str]:
    _note, trailers = _trailer_block(message)
    for line in trailers:
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip() or None
    return None


def trailer_of(message: str) -> Optional[str]:
    """The archive digest a state names, or None if it carries no anchor."""
    value = _trailer_value(message, TRAILER)
    return value if value and _SHA256_RE.match(value) else None


def calculation_of(message: str) -> Optional[str]:
    """Which calculation this state belongs to (L3).

    Why a state carries it at all: a folder can be copied to a cluster or
    opened a year later, and a history whose states say only "stage 2
    converged" cannot say which calculation that was.
    """
    return _trailer_value(message, CALC_TRAILER)


def note_of(message: str) -> str:
    """The note, without the trailers -- what a person reads in a list."""
    note_lines, _trailers = _trailer_block(message)
    return "\n".join(note_lines).strip()


# --------------------------------------------------------------------- #
#  Repo -- the class every surface goes through (§ 15)                  #
# --------------------------------------------------------------------- #

#: One ref per state, so **every state stays reachable forever** whatever else
#: is restored afterwards (A6).  HEAD still says where the folder *stands*, but
#: reachability no longer rides on it -- which is what lets a restore move the
#: folder without leaving the state it moved away from unreferenced.
_STATE_REF = "refs/molbuilder/state/"


class Repo:
    """A calculation folder under checkpointing.

    **Contract:** `execution/checkpointing.md`.  The vocabulary is the
    contract's: a **state** is a saved snapshot of the whole folder, a **tag**
    is a name you gave one, and the folder always **stands at** exactly one
    state (§ 5).  There are no branches and no lines; a fork is what happens
    when you save from a restored state (§ 7.1).
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.root = Path(path)

    # -- setup ---------------------------------------------------- #

    @property
    def initialized(self) -> bool:
        return (self.root / ".git").is_dir()

    def _require_init(self) -> None:
        if not self.initialized:
            raise CheckpointError(
                f"{self.path} is not a checkpoint folder; run "
                f"`molbuilder checkpoint init` first.")

    def init(self, engine: Optional[str] = None,
             note: str = "set up",
             calculation: Optional[str] = None) -> Optional["State"]:
        """Make this folder a checkpoint folder and save its first state.

        One repository per calculation (L1).  A folder whose subdirectories are
        working dirs is accepted only when this folder declares them one
        calculation by carrying its description; without one, several
        independent calculations would share a history and rewind together.
        """
        if self.initialized:
            # Idempotent, and honestly typed: a folder somebody ran `git init`
            # in by hand is "initialised" with no state to stand at, so this is
            # Optional rather than a State that might not be there.
            #
            # THAT FOLDER IS ALSO THE ONE L3 WAS ESCAPING THROUGH.  § 2.0 says
            # people run bare git in these directories, and a folder they had
            # already `git init`-ed never reached the name check below: `save`
            # then wrote the raw folder name into every state's trailer, so
            # `Calculation: has spaces!` shipped in a history that `init` would
            # have refused outright.  Setting it here makes `init` the verb
            # that repairs such a folder -- and `--calculation` the way to give
            # it a name its directory cannot spell.
            if not self._configured_calculation():
                _run_git(["config", "molbuilder.calculation",
                          check_calculation_name(calculation or self.root.name)],
                         cwd=self.path)
            return self.standing_at()
        working, inner = _scan_subtree(self.root)
        if inner:
            raise NestedRepoRefusedError(
                f"{self.path}: cannot init -- these subdirectories are already "
                f"checkpoint folders: {inner}.  A history inside a history "
                f"cannot be restored consistently (L1).")
        if working and not _is_bundle_root(self.root):
            raise NestedRepoRefusedError(
                f"{self.path}: cannot init -- nested working dirs present: "
                f"{working}, and nothing here says they belong to one "
                f"calculation.  Its root should carry the description that "
                f"says so ({', '.join(_BUNDLE_DESCRIPTORS)}) (L1).")
        name = check_calculation_name(calculation or self.root.name)
        _run_git(["init", "-q"], cwd=self.path)
        _run_git(["config", "molbuilder.calculation", name], cwd=self.path)
        if engine:
            _run_git(["config", "molbuilder.engine", engine], cwd=self.path)
        # `.gitignore` is not written here: `save` regenerates it as its first
        # act (I2b), so writing it twice only creates a second place the
        # derivation happens.
        return self.save(note)

    def _engine(self) -> Optional[str]:
        """Which config entry this folder's saves use (§ 4).

        Kept in the repository's own git config -- inside ``.git/``, which is
        never stored (S1) and is not a file anybody edits by hand.  It is
        **not** a second classification: it names which entry to read, while
        the patterns and the size limit stay in molbuilder.json (S1c).

        Losing it is safe by design -- an unnamed engine resolves to
        ``generic``, which measures every file and is "always correct and
        merely measures more".  It is persisted so that property costs a stat
        sweep only when nobody said what the engine was, rather than always.
        """
        if not self.initialized:
            return None
        out = _run_git(["config", "--get", "molbuilder.engine"],
                       cwd=self.path, check=False).stdout.strip()
        return out or None

    def _configured_calculation(self) -> Optional[str]:
        """The name this folder was given, or None if it was never set."""
        out = _run_git(["config", "--get", "molbuilder.calculation"],
                       cwd=self.path, check=False).stdout.strip()
        return out or None

    def calculation(self) -> str:
        """Which calculation this folder's history belongs to (L3).

        Defaults to the folder's own name, which is what a person recognises,
        and is fixed at init: it is written verbatim into every state, so
        changing it later would make one history claim two names.

        **Read-only and unvalidated on purpose.**  ``checkpoint config`` and the
        panel show this on folders that may be in any condition, and a reader
        that raises turns "your name needs fixing" into "this folder cannot be
        looked at".  The refusal L3 asks for belongs where the name is
        *written into a state*, which is :meth:`save`.
        """
        self._require_init()
        return self._configured_calculation() or self.root.name

    def classification(self) -> Dict[str, object]:
        """Which files this folder's saves send to the archive (§ 4, S1c).

        Public: two surfaces print it — ``checkpoint config`` and
        ``GET /api/checkpoint/config`` — and a private name they both reach
        past is not a seam, it is a seam being ignored.

        Resolves through :func:`classification_for`, whose name differs from
        this one on purpose: a module function and a method sharing one name
        read like a recursive call to anyone skimming the body.
        """
        return classification_for(self._engine())

    def _manifest_of(self, state: "State") -> Dict[str, Tuple[str, int]]:
        """What *state* held, read off its own record (I2b, I2c).

        **Damage is named, never absorbed.**  A state with no digest, or a
        digest whose archive is gone, used to leave this empty — and an empty
        record makes every archived file look newly added, so the panel said
        "12 unsaved" about files that were saved and are now unreachable.
        I2b's whole point is that those are two different observations:
        verification matches or it is refused, and there is no third answer.
        """
        if not state.archive:
            raise CheckpointError(
                f"state {state.short} carries no archive digest, so what it "
                f"held cannot be read.  Every state carries one from the first "
                f"onwards — including states that archived nothing — so this "
                f"is damage rather than a state without big files "
                f"(checkpointing.md I2b).")
        adir = archive_dir(self.root, state.archive)
        man = adir / MANIFEST_NAME
        if not man.is_file():
            raise CheckpointError(
                f"the archive for state {state.short} is missing: no "
                f"{MANIFEST_NAME} at {adir}.  The state records that archive, "
                f"so it was lost rather than never written, and nothing here "
                f"can be called saved or unsaved until that is resolved.\n\n"
                f"Two ways on, and neither needs git:\n"
                f"    molbuilder checkpoint save -m \"…\"\n"
                f"        records what is on disk now -- and rebuilds THIS "
                f"archive byte-identically if those large files are "
                f"unchanged, since an archive is named by its content;\n"
                f"    molbuilder checkpoint restore <other-state> --force\n"
                f"        leaves this state, accepting whatever is here.\n"
                f"(checkpointing.md I2b, § 2.0.)")
        raw = man.read_bytes()
        if manifest_digest(raw) != state.archive:
            raise CheckpointError(
                f"state {state.short}: its {MANIFEST_NAME} does not hash to "
                f"the name it is stored under.  The record was modified "
                f"(checkpointing.md I2b).")
        return parse_manifest(raw, str(man))

    # -- states ---------------------------------------------------- #

    #: One record per state, one call for any number of them.  ``%D`` carries
    #: the tags pointing at each commit, which is why nothing has to ask git a
    #: second time per state -- listing fifty states used to cost a hundred and
    #: one subprocesses.
    _FORMAT = "%H%x1f%P%x1f%aI%x1f%D%x1f%B%x1e"

    @staticmethod
    def _tags_from_decoration(decoration: str) -> Tuple[str, ...]:
        """The tag names in a ``%D`` decoration.

        ``%D`` also lists HEAD and the per-state ref; neither is a tag, and only
        tags are a user's names for a state (L4).
        """
        names = []
        for part in decoration.split(","):
            part = part.strip()
            if part.startswith("tag:"):
                names.append(part[4:].strip())
        return tuple(sorted(names))

    @classmethod
    def _parse_states(cls, out: str) -> List["State"]:
        states = []
        for record in out.split("\x1e"):
            if not record.strip():
                continue
            sha, parents, at, decoration, message = (
                record.split("\x1f", 4) + [""] * 5)[:5]
            states.append(State(
                id=sha.strip(),
                note=note_of(message),
                parent=parents.split()[0] if parents.strip() else None,
                at=at.strip(),
                archive=trailer_of(message),
                calculation=calculation_of(message),
                tags=cls._tags_from_decoration(decoration)))
        return states

    def _state_from(self, sha: str) -> "State":
        out = _run_git(["show", "-s", "--format=" + self._FORMAT, sha],
                       cwd=self.path).stdout
        found = self._parse_states(out)
        if not found:
            raise NoSuchRefError(f"{sha!r} does not name a state in {self.path}.")
        return found[0]

    def standing_at(self) -> Optional["State"]:
        """The one state the folder is currently at (§ 5).

        Set by `init`, by `save`, and by `restore`.  Everything `status` reports
        is measured against it and never against the newest state, which is why
        going back does not make the whole folder read as modified.
        """
        self._require_init()
        r = _run_git(["rev-parse", "HEAD"], cwd=self.path, check=False)
        sha = r.stdout.strip()
        return self._state_from(sha) if r.returncode == 0 and sha else None

    def states(self, limit: Optional[int] = None) -> List["State"]:
        """Every state, newest first.  Nothing is ever removed (A6).

        Topological order, not date order.  Several states can share a second --
        a scripted sweep saves faster than the clock ticks -- and a tie there
        would print a child above its own parent.  Topology cannot tie: a state
        always follows the state it came from.
        """
        self._require_init()
        argv = ["log", "--topo-order", "--format=" + self._FORMAT,
                "--glob=" + _STATE_REF + "*"]
        if limit is not None:
            argv.insert(1, f"-n{int(limit)}")
        out = _run_git(argv, cwd=self.path, check=False).stdout
        return self._parse_states(out)

    def resolve(self, name: str) -> str:
        """A state id or a tag -> the state's id.  One kind of argument (§ 5)."""
        self._require_init()
        r = _run_git(["rev-parse", "--verify", "--quiet", f"{name}^{{commit}}"],
                     cwd=self.path, check=False)
        sha = r.stdout.strip()
        if r.returncode != 0 or not sha:
            raise NoSuchRefError(
                f"{name!r} does not name a state or a tag in {self.path}.")
        return sha

    # -- what is unsaved ------------------------------------------- #

    def status(self, deep: bool = False) -> "FolderStatus":
        """Where the folder stands, and what differs from it (§ 5, A5).

        Three shapes -- changed, added, deleted -- covering text and big files
        alike.  **git is not asked about big files at all**: the always-large
        families are gitignored and the rest are kept out of the index by the
        save's pathspec, so git either cannot see them or sees them as
        untracked, and neither answer is the truth.  They are compared against
        the standing state's MANIFEST, which is the record of what that state
        held (I2c).

        **Two depths, and the difference is who pays for exactness.**

        ``deep=False`` (the default) is the *display*: size and timestamp only,
        never content.  It answers the sidebar on every directory-enter and the
        CLI's ``list``, where reading a 2 GB density matrix to draw a badge is a
        cost the answer does not earn.

        **Its blind spot is a same-size file whose mtime is not more than a
        second past the standing state's timestamp** -- which is wider than
        "rewritten inside the same second", the way this used to be written.
        The comparison is ``>``, and it has to be: a restore writes archived
        files with ``copy2``, so a legitimately restored file carries an mtime
        far OLDER than the state, and anything stricter would call every
        restored folder unsaved.  The cost of that is that an mtime-preserving
        arrival -- ``cp -p``, ``tar -x``, or the ``rsync`` § 2 recommends for
        moving a folder between machines -- lands under the threshold whatever
        its content.  Being wrong here still costs nothing, because **nothing
        is moving**: it is a sentence on a screen, and every operation that can
        lose something asks with ``deep=True``.

        ``deep=True`` hashes.  It is what runs before an operation that changes
        the folder, where being wrong costs data rather than a sentence, and it
        is what a Refresh control asks for when somebody wants certainty now.
        """
        if not self.initialized:
            return FolderStatus(path=self.path, initialized=False)
        here = self.standing_at()
        cls = self.classification()           # read once; it hits the filesystem
        edited = not gitignore_is_current(self.root, cls["always_large"])
        changed, added, deleted = set(), set(), set()

        big = big_files(self.root, int(cls["size_limit_bytes"]),
                        cls["always_large"])
        on_disk = {archive_key(self.root, p): p for p in big}

        expected: Dict[str, Tuple[str, int]] = {}
        if here is not None:
            expected = self._manifest_of(here)

        def _archives(name: str) -> bool:
            """Is this git talking about a file the archive owns?

            git's view of a big file is noise and must be dropped.  A file that
            is big by *size* rather than by *name* cannot be named in
            `.gitignore`, so git sees it as untracked and would report it as
            added on every status -- including immediately after it was saved.
            The MANIFEST comparison below is the only thing entitled to speak
            about these (I2c).
            """
            return name in on_disk or name in expected

        # `-z`, and that matters for correctness rather than parsing taste.
        # In its default form git C-escapes non-ASCII names and wraps anything
        # containing a space in double quotes, so `01 coarse/job.DM` arrives as
        # `"01 coarse/job.DM"`: a string that matches no key, slips past the
        # guard above, and makes a saved big file read unsaved forever.  With
        # -z the records are NUL-separated and never quoted.
        git_says: Dict[str, str] = {}
        fields = _run_git(["status", "--porcelain", "-z", "-uall"],
                          cwd=self.path).stdout.split("\0")
        idx = 0
        while idx < len(fields):
            record = fields[idx]
            idx += 1
            if len(record) < 4:               # the trailing empty field
                continue
            code, name = record[:2], record[3:]
            if code[0] in ("R", "C"):
                # A rename or copy carries its SOURCE in the next field.  Read
                # as one record it becomes the path `old -> new`, which names
                # nothing on disk and cannot be matched, saved or restored.
                source = fields[idx] if idx < len(fields) else ""
                idx += 1
                if source:
                    git_says[source] = "D "   # the pair is a delete and an add
                git_says[name] = "A "
                continue
            git_says[name] = code

        # Which files the standing state held **in git**.  A5's three shapes ask
        # whether *the state held a file at this path*, and a state holds files
        # in BOTH stores -- so answering from the MANIFEST alone calls a file
        # that crossed the size limit "added", when the state had it all along.
        tracked = {name for name in
                   _run_git(["ls-files", "-z"], cwd=self.path).stdout.split("\0")
                   if name}

        for name, code in git_says.items():
            if _archives(name):
                continue                      # the archive's business, below
            if code == "??" or code[0] == "A":
                added.add(name)
            elif "D" in code:
                deleted.add(name)
            else:
                changed.add(name)

        # I2c: what is unsaved is measured against the standing state's
        # MANIFEST -- the record of what that state HELD -- unioned with what
        # the classification says is big NOW, which is the only way a
        # brand-new big file is noticed.  Iterating the classification alone
        # would report a file as *deleted* the moment it stopped matching an
        # always-large pattern, while it sat untouched on disk.
        # Cheap first, exact only when it has to be.  This runs on every
        # directory-enter in the sidebar (docs/web/projects.md), and a folder
        # holding a 2 GB density matrix must not be read end-to-end to answer
        # "is anything unsaved".  A different SIZE is already an answer; only a
        # same-size file that was touched after the state was saved needs
        # hashing, and an untouched one cannot have changed.
        saved_at = _epoch_of(here.at) if here is not None else None
        for key in set(expected) | set(on_disk):
            path = self.root / key
            if not path.is_file():
                deleted.add(key)
                continue
            want = expected.get(key)
            if want is None:
                # Not in the standing state's archive.  That is only "added" if
                # the state had NO file at this path -- and it may well have had
                # one in git, which is what happens when a file crosses the size
                # limit.  A5's shapes are about the file, not about which store
                # held it.
                if key in tracked:
                    # git tracked it, so the state held it.  Whether it differs
                    # is git's own answer: a modification shows in the porcelain
                    # above, and silence there means the bytes still match what
                    # the state holds.  A file that merely got RECLASSIFIED --
                    # the size limit moved, nothing was touched -- is therefore
                    # not unsaved, and saying it was would be the false alarm
                    # § 7.2 warns trains people to ignore the real one.
                    if key in git_says:
                        changed.add(key)
                else:
                    added.add(key)
                continue
            stat = path.stat()
            if stat.st_size != want[1]:
                changed.add(key)              # a different size is an answer
                continue
            if not deep and saved_at is not None:
                # ONE SECOND of tolerance, and it is resolution rather than
                # slack.  A state's timestamp is whole seconds; a file's is
                # not.  Save a file at 12:00:00.3 and the state records
                # 12:00:00, so a bare `>` calls the file newer than the state
                # that just saved it -- the folder would read unsaved the
                # instant after a save, which is the normal flow and not a
                # corner case.
                #
                # What this cannot see is a same-size file whose mtime is not
                # more than a second past the state -- which includes an mtime
                # OLDER than the state, not merely a rewrite inside the same
                # second.  `>` is deliberate and cannot be tightened: `restore`
                # copies with `copy2`, so a correctly restored file carries the
                # mtime it had when it was archived, and `!=` would report an
                # entire folder unsaved the moment you went back to it.  So an
                # mtime-preserving arrival (`cp -p`, `tar -x`, `rsync -a`)
                # passes here whatever it contains.
                #
                # AND DO NOT COMPARE AGAINST THE ARCHIVE'S OWN COPY INSTEAD.
                # It looks exact and free -- one `copy2` makes both, so the two
                # mtimes agree by construction -- but identical content is
                # stored once (§ 12), so an archive's copy carries the mtime of
                # the FIRST time those bytes were archived, whether the whole
                # directory was reused or the one file hard-linked.  A rerun
                # writing byte-identical output would then read unsaved
                # PERMANENTLY.  § 7.2 records that door and the test is
                # `test_a_rerun_that_writes_identical_bytes_still_reads_clean`.
                #
                # All of it accepted deliberately: nothing moves on a status
                # call, so being wrong costs a sentence and not a byte, and
                # every operation that CAN lose something checks content.
                if stat.st_mtime > saved_at + 1:
                    changed.add(key)
                continue
            # Falls through to the exact comparison when asked for it -- and
            # also when the state's timestamp is unreadable, because "cannot
            # rule it out cheaply" must mean *check*, not *assume changed*.
            if sha256_of(path) != want[0]:
                changed.add(key)

        return FolderStatus(
            path=self.path, initialized=True,
            standing_at=here,
            changed=tuple(sorted(changed - added - deleted)),
            added=tuple(sorted(added)), deleted=tuple(sorted(deleted)),
            ignore_edited=edited)

    # -- save ------------------------------------------------------ #

    def _ignored(self, keys: Sequence[str]) -> set:
        """Which of *keys* git has already been told to skip.

        Asked of git rather than derived, because there are two sources and
        only one of them is ours: the generated block (the always-large
        families) and whatever the user wrote above or below the markers, which
        S1a leaves alone deliberately.  Guessing would mean naming an ignored
        path in a pathspec, which `git add` refuses outright.

        **DO NOT ADD ``--no-index`` HERE.**  It reads like the more careful
        flag and it would reinstate the blob leak.  `check-ignore` consults the
        index by default, so a **tracked** file is reported as *not* ignored --
        which is exactly right, because the question this asks is not "do the
        ignore rules match it" but "would ``git add`` otherwise take it".  git
        stages a tracked file regardless of any pattern.  With ``--no-index`` a
        file that a config change just reclassified would be called ignored,
        dropped from the exclusion list, and re-staged by `add` -- writing its
        blob into `.git/objects`, which is what § 3 forbids.
        """
        if not keys:
            return set()
        result = _run_git(["check-ignore", "-z", "--stdin"], cwd=self.path,
                          check=False, stdin="\0".join(keys))
        return {name for name in result.stdout.split("\0") if name}

    def save(self, note: str) -> Optional["State"]:
        """Save the folder as a new state.  The note is required (L3).

        The new state's parent is **where the folder stood**, then the folder
        stands at the new state.  That is the whole of the branching mechanism
        (§ 7.1): you never declare a fork, you save from where you are.

        Returns ``None`` when nothing changed.
        """
        self._require_init()
        if not (note or "").strip():
            raise CheckpointError(
                "a state needs a note saying what happened and what you were "
                "about to do -- it is the only thing that answers the question "
                "you bring to a history a month later, and it is not "
                "defaulted (checkpointing.md L3).")
        # L3's other half, checked HERE because this is where the name is
        # written verbatim into a state.  `init` refuses a name needing repair,
        # but a folder somebody had already `git init`-ed skipped that gate
        # entirely (§ 2.0 says they do), so the raw directory name reached the
        # trailer.  `calculation()` stays unvalidated so read-only surfaces
        # still work on such a folder; the refusal lands on the write.
        try:
            check_calculation_name(self.calculation())
        except CalculationNameError as exc:
            raise CalculationNameError(
                f"{exc}\n\nThis folder is a git repository that molbuilder did "
                f"not name -- most likely `git init` was run here by hand. "
                f"Give it one, and nothing else changes:\n"
                f"    molbuilder checkpoint init --calculation <name>\n"
                f"Nothing was saved.") from exc
        cls = self.classification()
        always = cls["always_large"]
        # I2b: `.gitignore` records a DERIVATION, so the derivation is the
        # check -- regenerating it every save is what makes a hand edit
        # harmless rather than a file silently dropped from every store.
        write_gitignore(self.root, always)
        big = big_files(self.root, int(cls["size_limit_bytes"]), always)
        keys = [archive_key(self.root, path) for path in big]

        # THE BIG FILES ARE NEVER SHOWN TO `git add`, and that is not an
        # optimisation.  `add` hashes, compresses and WRITES every file it is
        # given: a big file reaching it lands in `.git/objects` and stays
        # there, because the `rm --cached` below only drops the index entry.
        # The blob is then on disk forever, re-created on every save, for a
        # file § 3 says never goes into git.  "Every file is in exactly one
        # store" has to be true of the object database, not just of the index.
        #
        # `:(exclude,literal)` is exact -- no glob, so a path holding `[`, `*`
        # or a space is excluded as itself and nothing else.
        #
        # Only the files git would OTHERWISE TAKE are named.  `git add` refuses
        # a pathspec that names an ignored path -- even to exclude it -- and
        # the always-large families are ignored already, by the block this save
        # just regenerated.  So the exclusions are exactly the files that are
        # big by MEASUREMENT, which is the set that has no other way to be kept
        # out of git (the size gate cannot be a gitignore pattern).
        ignored = self._ignored(keys)          # one call, not one per key
        excludes = [f":(exclude,literal){key}"
                    for key in keys if key not in ignored]
        _run_git(["add", "-A", "--", ".", *excludes], cwd=self.path)

        # A SYMLINK IS NEVER LARGE, so an always-large hint must not reach it.
        #
        # S1b lets a name skip a *measurement* for a family that is always big.
        # A link has no size worth measuring -- it is twenty bytes of path text
        # -- so for a link the hint is simply wrong, and `.gitignore` was
        # sending `02_tight/job.DM -> ../01_coarse/job.DM` to the store that does
        # not take links.  It was then in NEITHER store: `add` skipped it as
        # ignored, the archive skipped it as a link, and a restore neither
        # brought it back (git never had it) nor removed it (`git clean`
        # without `-x` leaves ignored paths alone).  § 3's "exactly one store"
        # quietly did not hold, for exactly the links `jobset/materialize.py`
        # lays between stages.
        #
        # `-f` ONLY for links, and that limit is the whole safety of it: forcing
        # a big *file* past the ignore rules is S1's losing branch, a blob in
        # `.git/objects` on every save.
        links = [archive_key(self.root, path)
                 for path in walk_symlinks(self.root)]
        swallowed = sorted(self._ignored(links))
        if swallowed:
            _run_git(["add", "-f", "--", *swallowed], cwd=self.path)
        # S7: a file that was small last save and is big now is still in the
        # index, and the pathspec above cannot remove it -- excluded means
        # untouched.  This is the half that makes a category change complete:
        # a file leaves the store it came from.  (The reverse direction needs
        # nothing: a file that shrank is no longer excluded, so `add` takes it.)
        for key in keys:
            _run_git(["rm", "--cached", "-q", "--ignore-unmatch", "--", key],
                     cwd=self.path, check=False)

        digest = publish_archive(self.root, big)
        here = self.standing_at()
        staged = _run_git(["diff", "--cached", "--name-only"],
                          cwd=self.path, check=False).stdout.strip()
        if not staged and here is not None and here.archive == digest:
            return None                   # nothing changed, in either store

        _run_git(["commit", "-q", "--allow-empty", "-m",
                  message_with_trailers(note, digest, self.calculation())],
                 cwd=self.path)
        sha = _run_git(["rev-parse", "HEAD"], cwd=self.path).stdout.strip()
        _run_git(["update-ref", _STATE_REF + sha, sha], cwd=self.path)
        return self._state_from(sha)


    # -- restore ---------------------------------------------------- #

    def restore(self, state: str, force: bool = False) -> "State":
        """Make the folder equal *state* exactly (A5), or refuse (A2, A4).

        ``state`` is a state id or a tag -- the one kind of argument there is.

        Order is the rule.  The two refusals come first because they are about
        the **target** -- an unknown state or an archive that does not verify
        means the operation cannot happen at all, and nobody should be asked to
        accept a loss for an operation that then fails for an unrelated reason.
        The question about unsaved work comes last, immediately before the
        first byte moves.

        A restore is whole or it does not happen (A4): text and big files are
        one state, and returning half of one save and half of another produces
        a folder no save ever held.
        """
        self._require_init()
        sha = self.resolve(state)                     # refusal 1: unknown
        target = self._state_from(sha)
        if not target.archive:
            raise CheckpointError(
                f"state {target.short} carries no archive digest, so what it "
                f"held cannot be checked.  Refusing to restore from a record "
                f"that cannot be verified (I2b).")
        expected = verify_archive(self.root, target.archive)   # refusal 2

        # THE QUESTION, LAST -- AND ONLY WHEN THERE IS A QUESTION TO ASK.
        #
        # `force` IS the answer (§ 5: "--force ... answers yes, for a script"),
        # so asking is pure cost once it is set.  This used to compute the
        # answer unconditionally and then discard it, which cost twice:
        #
        #   * every large file in the folder was hashed for a message nobody
        #     would see -- on a real calculation, the whole density-matrix set,
        #     read end to end and thrown away.  § 6 refuses to double a SAVE for
        #     less than this;
        #   * and it made a forced restore fail for a reason about the state
        #     you are LEAVING.  `status` reads the standing state's MANIFEST, so
        #     a damaged archive over there refused a restore whose target was
        #     perfectly intact -- and since `status` and `list` fail the same
        #     way, there was no verb left that could move the folder anywhere.
        #     § 2.0 promises the verbs cover the work.
        #
        # DEEP when it is asked, because this is the moment the folder is about
        # to change: "what will be lost" is answered by content, never by a
        # timestamp.  The cheap read exists for drawing a badge (§ 7.2).
        if not force:
            here = self.status(deep=True)
            if not here.clean:
                raise DirtyWorkingTreeError(
                    "this folder has work that is not saved, and restoring "
                    "will lose it:\n"
                    + _describe(here)
                    + "\n\nSave it first with `molbuilder checkpoint save "
                      "-m \"…\"`, or pass --force to accept the loss.")

        # --- from here on the folder is being changed --------------- #
        _run_git(["checkout", "--force", "--detach", sha], cwd=self.path)
        _run_git(["clean", "-fdq"], cwd=self.path)

        # Everything the target did not hold is removed -- not a loss, and not
        # warned about (A5): those files are still in the state that holds
        # them, and leaving a stage 2 .DM in a folder claiming to be stage 1 is
        # exactly the file a later run picks up unasked.
        #
        # What belongs here is decided by TWO RECORDS AND NO CONFIGURATION
        # (I2a): git says which text the target held, the MANIFEST says which
        # big files it held, and anything else is a leftover.  Deriving the set
        # from the classification instead would make a restore mean different
        # things before and after a config edit -- and `git clean` alone cannot
        # do it, because a file matching an ignore pattern is invisible to it.
        #
        # `-z` and a NUL split, NOT `.split()`.  Whitespace is not a separator
        # here: `01 coarse/job.fdf` split into `01` and `coarse/job.fdf`, so the
        # real key matched nothing, the file counted as a leftover, and the loop
        # below DELETED a tracked file the target holds -- with nothing to put
        # it back, since only archived files are copied afterwards.  `-z` also
        # stops git quoting a non-ASCII name, which fails the same way.
        tracked = {name for name in
                   _run_git(["ls-files", "-z"], cwd=self.path).stdout.split("\0")
                   if name}
        # `walk_entries`, not `walk_files`: SYMLINKS ARE LEFTOVERS TOO.
        #
        # `git clean` above removes an untracked link, but only when no ignore
        # pattern matches its name -- so a stray `job.DM -> ../01_coarse/job.DM`
        # survived a restore of a state that never held it, pointing a later
        # run at the wrong stage's output.  A link the target did not hold is a
        # leftover exactly like a file, and A5 removes leftovers without asking
        # because they are not a loss.  `unlink` on a link removes the LINK; the
        # file it pointed at is somebody else's entry in this same walk.
        for path in walk_entries(self.root):
            key = archive_key(self.root, path)
            if key not in tracked and key not in expected:
                path.unlink()

        adir = archive_dir(self.root, target.archive)
        for key in sorted(expected):
            dst = self.root / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            # REMOVE FIRST -- NEVER WRITE THROUGH WHAT IS ALREADY THERE.
            #
            # `copy2` opens the destination and truncates it *in place*, so it
            # writes through both kinds of link, and each fails differently:
            #
            #   * a HARD LINK shares its inode with another path, so the copy
            #     lands in both.  Restore a state that held `a.DM` and `b.DM`
            #     with different content, over a folder where somebody linked
            #     them together, and the second copy overwrites the first --
            #     one path ends up holding the other's bytes and the restore
            #     reports SUCCESS.  A5 says the folder equals the target
            #     exactly, and it did not.
            #   * a SYMLINK is followed, so the bytes land on whatever it
            #     points at -- possibly outside the folder entirely -- while
            #     the link itself stays where a real file belongs.
            #
            # Unlinking makes every restored file a fresh inode, which is what
            # the target state actually describes: independent paths.  It also
            # turns a directory sitting at an archived path into a loud error
            # rather than `copy2` quietly writing `job.DM/job.DM`.
            #
            # `is_symlink()` first, and `or` rather than `and`: `exists()`
            # follows the link, so a DANGLING symlink reports False and would
            # otherwise be left in the way.
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            shutil.copy2(adir / key, dst)
        return target

    # -- tags ------------------------------------------------------ #

    def tag(self, name: str, note: str, at: Optional[str] = None) -> "Tag":
        """Give a state a name so you can find it again (§ 5, L4)."""
        self._require_init()
        if not (name or "").strip():
            raise CheckpointError("a tag needs a name.")
        if not (note or "").strip():
            raise CheckpointError(
                "a tag needs a note saying why this state is worth returning "
                "to (L3).")
        sha = self.resolve(at) if at else self.resolve("HEAD")
        _run_git(["tag", "-a", name, sha, "-m", note], cwd=self.path)
        return Tag(name=name, state=sha, note=note)

    def tags(self) -> List["Tag"]:
        self._require_init()
        # ``*objectname`` is the commit an annotated tag points AT; plain
        # ``objectname`` would be the tag object itself, which is not a state.
        #
        # Space-separated with the free text LAST, not %x00-separated:
        # `for-each-ref` does not interpret %xNN escapes, so a NUL format
        # emits the four literal characters and the whole line parses as one
        # field.  A tag name and a hash cannot contain spaces; a note can, so
        # it goes last and the split is bounded.
        out = _run_git(["for-each-ref",
                        "--format=%(refname:short) %(*objectname) "
                        "%(contents:subject)", "refs/tags"],
                       cwd=self.path, check=False).stdout
        found = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = (line.split(" ", 2) + ["", ""])[:3]
            name, state, subject = parts
            found.append(Tag(name=name, state=state or self.resolve(name),
                             note=subject))
        return sorted(found, key=lambda t: t.name)


def _epoch_of(iso: str) -> Optional[float]:
    """An ISO-8601 timestamp as epoch seconds, or None if unreadable.

    Unreadable means "cannot rule the file out cheaply", and the caller then
    hashes -- slower, never wrong.
    """
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return None


def _describe(status: "FolderStatus") -> str:
    """A5's three shapes, named -- never a count.

    "3 unsaved files" tells nobody anything; `01_coarse/job.DM` tells them the
    density matrix is what they are about to walk away from.
    """
    parts = []
    for label, names in (("changed", status.changed),
                         ("added", status.added),
                         ("deleted", status.deleted)):
        for name in names:
            parts.append(f"  {label:>7}  {name}")
    return "\n".join(parts)
