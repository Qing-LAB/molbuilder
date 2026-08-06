"""L2 — the archive reaches into per-stage subdirectories.

The acceptance test for the depth mismatch described in
``docs/execution/checkpointing.md`` (S1, S1a, L2).

THE DEFECT THESE PIN.  ``.gitignore`` receives the raw archive globs
(``*.DM``), and a gitignore pattern with no slash matches at EVERY level -- so
``coarse/job.DM`` is ignored by git.  The archive walk used ``path.glob``,
which matches only the TOP level.  A nested big binary was therefore gitignored
AND unarchived: in no snapshot at all, and silently absent after a restore --
S1's "never neither" branch, whose failure mode is losing data rather than
wasting disk.

Real filesystem, real git.  A mocked version of these would have passed against
the broken code, because the bug lived in the disagreement between two real
subsystems' idea of depth.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from molbuilder.checkpoint import Repo, CheckpointError, _list_big_binaries


def _seed_nested(root: Path) -> Path:
    """A run directory whose outputs land one level down.

    NOT the staged folder of docs/engines/stages.md § 7.1 -- ``Repo.init``
    REFUSES a directory holding sub-dirs with working-dir markers
    (``.fdf`` / ``.py`` / ``.run.sh``): the shipped lowest-directory rule, which
    L1 in docs/execution/checkpointing.md proposes to revisit and this test does
    not prejudge.  What it pins is the depth gap that is reachable TODAY: a
    marker-free subdirectory holding big binaries."""
    (root / "job.fdf").write_text("SystemLabel job\n")
    (root / "Au.psml").write_text("<pseudo/>\n")
    for sub, fill in (("coarse", b"\x01"), ("tight", b"\x02")):
        d = root / sub
        d.mkdir()
        (d / "job.DM").write_bytes(fill * 4096)      # the big binary
        (d / "job.XV").write_text(f"{sub} coords\n")     # text: git-tracked
    return root


def _git_tracked(root: Path) -> set:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                         text=True, check=True).stdout
    return {line for line in out.splitlines() if line}


def _archived_keys(root: Path, sha: str) -> set:
    manifest = root / ".binsnapshots" / sha / "MANIFEST"
    if not manifest.is_file():
        return set()
    return {line.split("  ", 2)[2]
            for line in manifest.read_text().splitlines() if line}


# ------------------------------------------------------------------ #
#  L2 — the walk reaches into subdirectories                          #
# ------------------------------------------------------------------ #


def test_nested_big_binary_is_listed_for_archiving(tmp_path):
    """The walk finds a stage's .DM, not only a top-level one."""
    _seed_nested(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")

    keys = {p.relative_to(tmp_path).as_posix() for p in _list_big_binaries(tmp_path)}
    assert keys == {"coarse/job.DM", "tight/job.DM"}, (
        "the archive walk must reach per-stage subdirectories; a top-level-only "
        f"walk yields {keys!r} and loses every nested binary")


def test_two_stages_may_share_a_basename(tmp_path):
    """Both subdirectories hold a `job.DM`.  Keys are repo-relative paths, so
    both are archived -- a basename-keyed archive would silently keep one."""
    _seed_nested(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.checkpoint(message="two stages")
    sha = repo.resolve_ref("HEAD")

    assert _archived_keys(tmp_path, sha) == {"coarse/job.DM", "tight/job.DM"}


# ------------------------------------------------------------------ #
#  S1 — tracked XOR archived, over the whole tree                     #
# ------------------------------------------------------------------ #


def test_every_file_is_tracked_or_archived_never_both_never_neither(tmp_path):
    """S1 over a produced two-stage folder.  This is the assertion the defect
    failed: `coarse/job.DM` was in neither set."""
    _seed_nested(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.checkpoint(message="stage folder")
    sha = repo.resolve_ref("HEAD")

    tracked = _git_tracked(tmp_path)
    archived = _archived_keys(tmp_path, sha)

    on_disk = {
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*")
        if p.is_file() and not p.is_symlink()
        and not any(part.startswith(".") for part in p.relative_to(tmp_path).parts[:-1])
        and p.name != ".gitignore"
    }
    # .mbcheckpoint.json is tracked config, not run state; keep it in the set.
    for rel in sorted(on_disk):
        in_git = rel in tracked
        in_archive = rel in archived
        assert in_git != in_archive, (
            f"S1 violated for {rel!r}: tracked={in_git} archived={in_archive}. "
            "Both means a large blob in git history forever; neither means the "
            "file is in no snapshot and a restore loses it silently.")

    assert "coarse/job.DM" in archived and "coarse/job.DM" not in tracked
    assert "coarse/job.XV" in tracked and "coarse/job.XV" not in archived


# ------------------------------------------------------------------ #
#  The consequence: a restore actually brings the stage state back    #
# ------------------------------------------------------------------ #


def test_restore_returns_a_nested_binary(tmp_path):
    """The failure a user would have met: restore a checkpoint and get the
    geometry with no density matrix beside it."""
    _seed_nested(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.checkpoint(message="converged")
    tag = repo.tag("job/coarse/20260806T000000Z", message="coarse converged")

    original = (tmp_path / "coarse" / "job.DM").read_bytes()
    (tmp_path / "coarse" / "job.DM").write_bytes(b"\xff" * 16)   # clobber it
    # A big binary is gitignored, so a binary-only change leaves `git status`
    # clean and `checkpoint()` finds nothing to commit.  Touch a tracked file so
    # the commit -- and with it the new archive -- actually lands.
    (tmp_path / "coarse" / "job.XV").write_text("clobbered coords\n")
    repo.checkpoint(message="clobbered")

    repo.restore(tag)

    assert (tmp_path / "coarse" / "job.DM").read_bytes() == original, (
        "a restore must bring back the stage's binary state; before the L2 fix "
        "the file was in no snapshot and the restore left the clobbered bytes")


def test_symlinked_carry_is_not_archived_as_a_second_copy(tmp_path):
    """A carried restart file is a link to its producer until localize-on-run
    replaces it.  Archiving the link would duplicate content and restore a
    regular file where a link belongs (S3)."""
    _seed_nested(tmp_path)
    (tmp_path / "tight" / "job.DM").unlink()
    (tmp_path / "tight" / "job.DM").symlink_to(Path("..") / "coarse" / "job.DM")

    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.checkpoint(message="tight not yet run")
    sha = repo.resolve_ref("HEAD")

    assert _archived_keys(tmp_path, sha) == {"coarse/job.DM"}


# ------------------------------------------------------------------ #
#  The walk must not archive git's own state, or the archive itself   #
# ------------------------------------------------------------------ #


def test_walk_skips_dot_directories(tmp_path):
    """`.git` and `.binsnapshots` are never archive candidates -- a recursive
    walk that forgot them would archive the archive."""
    _seed_nested(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.checkpoint(message="first")

    # A file matching an archive glob, planted inside each excluded directory.
    (tmp_path / ".git" / "decoy.DM").write_bytes(b"\x00" * 8)
    binsnap = next(d for d in (tmp_path / ".binsnapshots").iterdir()
                   if d.is_dir())          # skip the .gitkeep beside the archives
    (binsnap / "decoy.DM").write_bytes(b"\x00" * 8)

    keys = {p.relative_to(tmp_path).as_posix() for p in _list_big_binaries(tmp_path)}
    assert not any(k.startswith(".") for k in keys), (
        f"the walk descended into a dot-directory: {sorted(keys)!r}")


# ------------------------------------------------------------------ #
#  Old archives still read: a bare basename is a relative path        #
# ------------------------------------------------------------------ #


def test_flat_archive_written_before_the_change_still_restores(tmp_path):
    """The key space WIDENED rather than moved.  An archive whose MANIFEST
    holds bare basenames -- every archive written before nested run folders
    existed -- parses and restores unchanged."""
    (tmp_path / "job.fdf").write_text("SystemLabel job\n")
    (tmp_path / "job.DM").write_bytes(b"\x07" * 1024)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.checkpoint(message="flat")
    sha = repo.resolve_ref("HEAD")

    assert _archived_keys(tmp_path, sha) == {"job.DM"}   # bare basename

    original = (tmp_path / "job.DM").read_bytes()
    (tmp_path / "job.DM").write_bytes(b"\x00")
    (tmp_path / "job.fdf").write_text("SystemLabel job  # touched\n")
    repo.checkpoint(message="clobbered")
    repo.restore(sha)
    assert (tmp_path / "job.DM").read_bytes() == original
