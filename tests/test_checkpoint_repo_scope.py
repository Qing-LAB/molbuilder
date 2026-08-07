"""Repository scope: which directory a checkpoint repo may cover.

`docs/execution/project-layout.md` § 1 defines two directory shapes and
`docs/execution/checkpointing.md` § 5 requires the checkpoint to serve **both**:

  * **flat** -- one directory, stages told apart by filename suffix;
  * **hierarchical** -- a calculation root with a directory per stage and per
    attempt.

The guard that made the second impossible refused *any* directory whose
subdirectories held a `.fdf` / `.py` / `.run.sh`.  That is right when those
subdirectories are **other people's calculations** -- one history over several
would rewind all of them together -- and wrong when they are **this
calculation's own stages**.  A root that carries its description says which it
is, and these tests pin both halves.

They assert **outcomes on disk**: a real `git init`, a real checkpoint, a real
restore.  A test that only checked `init` did not raise would pass against a
repository that could never restore anything.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from molbuilder.checkpoint import Repo, NestedRepoRefusedError


def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(not _have_git(), reason="git not on PATH")


# ------------------------------------------------------------------ #
#  Builders — the two shapes, as the tools actually produce them      #
# ------------------------------------------------------------------ #

def _flat(root: Path, label: str = "job") -> Path:
    """The shipped flat shape: decks suffixed per stage, warm files shared."""
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{label}_stage1.fdf").write_text(f"SystemLabel {label}\n")
    (root / f"{label}_stage2.fdf").write_text(f"SystemLabel {label}\n")
    (root / f"{label}_stage1.run.sh").write_text("#!/bin/bash\n")
    (root / f"{label}.DM").write_bytes(b"density-v1")
    return root


def _jobset_bundle(root: Path, label: str = "job") -> Path:
    """The shipped `jobset prep` shape: a bundle root plus `point-<name>/`
    directories with the deck **symlinked** in."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "job-set.json").write_text('{"schema": "molbuilder/job-set@1"}\n')
    for stage in ("stage1", "stage2"):
        (root / f"{label}_{stage}.fdf").write_text(f"SystemLabel {label}\n")
        d = root / f"point-{stage}"
        d.mkdir()
        (d / f"{label}_{stage}.fdf").symlink_to(f"../{label}_{stage}.fdf")
    return root


def _staged_calculation(root: Path, label: str = "bdt_au") -> Path:
    """The proposed hierarchical shape: a description at the root, a directory
    per stage, a directory per attempt."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "stages.json").write_text('{"schema": "molbuilder/stages@1"}\n')
    (root / f"{label}.template.fdf").write_text(f"SystemLabel {label}\n")
    for seq, name in ((1, "coarse"), (2, "tight")):
        stage = root / f"0{seq}_{name}"
        (stage / "run-0").mkdir(parents=True)
        (stage / f"{label}.fdf").write_text(f"SystemLabel {label}\n")
        (stage / f"{label}.run.sh").write_text("#!/bin/bash\n")
        (stage / "run-0" / f"{label}.DM").write_bytes(
            f"density-{name}".encode())
        (stage / "run-0" / f"{label}.out").write_text("Job completed\n")
    return root


# ------------------------------------------------------------------ #
#  Both shapes can be initialised                                     #
# ------------------------------------------------------------------ #

def test_flat_directory_initialises(tmp_path):
    """Unchanged: a flat run directory has no subdirectories to argue about."""
    Repo(str(_flat(tmp_path / "flat"))).init(engine="siesta")
    assert (tmp_path / "flat" / ".git").is_dir()


def test_shipped_jobset_bundle_initialises(tmp_path):
    """THE REGRESSION THIS FIXES.  `jobset prep` has always produced
    `point-<name>/` directories with the deck linked in, and the old guard
    refused every one of them -- so a staged bundle could not be checkpointed
    at all.  Verified against the real prep shape, symlinks included."""
    root = _jobset_bundle(tmp_path / "bundle")
    Repo(str(root)).init(engine="siesta")
    assert (root / ".git").is_dir()


def test_staged_calculation_initialises(tmp_path):
    """The hierarchical shape: three levels, decks and wrappers inside stage
    directories, big binaries two levels down."""
    root = _staged_calculation(tmp_path / "calc")
    Repo(str(root)).init(engine="siesta")
    assert (root / ".git").is_dir()


# ------------------------------------------------------------------ #
#  What the guard still protects                                      #
# ------------------------------------------------------------------ #

def test_refuses_a_parent_holding_independent_calculations(tmp_path):
    """The protection the guard exists for, and it must survive the fix: a
    topic directory holding two unrelated calculations has no description
    saying they are one, so one history over both would rewind both."""
    topic = tmp_path / "optimization"
    _flat(topic / "calc-a", label="alpha")
    _flat(topic / "calc-b", label="beta")

    with pytest.raises(NestedRepoRefusedError) as e:
        Repo(str(topic)).init(engine="siesta")

    msg = str(e.value)
    assert "calc-a" in msg and "calc-b" in msg
    assert "rewind" in msg, "the message must say what would go wrong"
    assert not (topic / ".git").exists(), "it must refuse before writing"


def test_refuses_when_a_subdirectory_is_already_a_repository(tmp_path):
    """A history inside a history cannot be restored consistently -- so this is
    refused even in a bundle root, where nested working dirs are fine."""
    root = _jobset_bundle(tmp_path / "bundle")
    Repo(str(root / "point-stage1")).init(engine="siesta")

    with pytest.raises(NestedRepoRefusedError, match="already checkpoint"):
        Repo(str(root)).init(engine="siesta")

    assert not (root / ".git").exists()


def test_a_dot_directory_of_python_files_does_not_block_init(tmp_path):
    """`.venv/` beside a run is full of `.py` and read as a nested working
    directory, blocking init for a reason with nothing to do with
    calculations.  Dot-directories are skipped."""
    root = _flat(tmp_path / "flat")
    venv = root / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "something.py").write_text("x = 1\n")

    Repo(str(root)).init(engine="siesta")
    assert (root / ".git").is_dir()


# ------------------------------------------------------------------ #
#  …and a hierarchical folder actually round-trips                    #
# ------------------------------------------------------------------ #

def test_hierarchical_folder_checkpoints_and_restores(tmp_path):
    """The real proof that the shape is supported: init is only the door.

    A stage's big binary two levels down must be archived, survive a later
    change, and come back on restore -- otherwise `init` succeeding would just
    be a repository that silently loses results."""
    root = _staged_calculation(tmp_path / "calc")
    repo = Repo(str(root))
    repo.init(engine="siesta")           # init commits the folder as it stands

    first = repo.list_checkpoints()[0]   # that commit is the point to come back to
    dm = root / "01_coarse" / "run-0" / "bdt_au.DM"
    assert dm.read_bytes() == b"density-coarse"

    # A later stage runs: its big binary changes, and so does a text file so
    # that both halves of the restore are exercised.
    (root / "02_tight" / "run-0" / "bdt_au.DM").write_bytes(b"density-v2")
    (root / "02_tight" / "run-0" / "bdt_au.out").write_text("Job completed 2\n")
    assert repo.checkpoint("tight ran") is not None, "the change must commit"

    repo.restore(first.sha)

    assert dm.read_bytes() == b"density-coarse", (
        "the coarse attempt's density matrix must come back")
    assert (root / "02_tight" / "run-0" / "bdt_au.DM").read_bytes() == \
        b"density-tight", "restore rewinds every stage, not just one"


def test_big_binaries_at_depth_are_archived_not_committed(tmp_path):
    """S1 read against the hierarchical shape: a `.DM` two levels down is the
    archive's, never git's.  This is the pairing that was broken until the
    archive walk became recursive -- gitignored AND unarchived is *gone*."""
    root = _staged_calculation(tmp_path / "calc")
    repo = Repo(str(root))
    repo.init(engine="siesta")
    repo.checkpoint("first")

    tracked = subprocess.run(["git", "ls-files"], cwd=str(root),
                             capture_output=True, text=True).stdout.split()
    assert not [f for f in tracked if f.endswith(".DM")], \
        "a big binary must not be in git"

    archived = list((root / ".binsnapshots").rglob("*.DM"))
    names = {str(p.relative_to(root)) for p in archived}
    assert any("01_coarse/run-0" in n for n in names), \
        f"the coarse attempt's .DM must be archived; got {names}"
    assert any("02_tight/run-0" in n for n in names), \
        f"the tight attempt's .DM must be archived; got {names}"
