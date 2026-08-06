"""One run directory per attempt, immutable once written.

The wrapper half of ``docs/execution/project-layout.md`` § 1.2.  Opt-in via
``attempt_dirs=True``; a flat run directory keeps today's behaviour untouched,
which is the first thing asserted here.

REAL EXECUTION, not rendered text.  A fake ``siesta`` on PATH writes nothing, so
any warm file that appears in a later attempt can only have been carried -- a
test that read the generated bash would have passed against a carry block that
never ran.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from molbuilder.runwrap import render_run_wrapper, write_run_wrapper


def _fake_engine(tmp_path: Path, body: str = "") -> dict:
    """A ``siesta`` on PATH that consumes stdin and writes only what ``body``
    says.  Returns the env for running the wrapper."""
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    (b / "siesta").write_text("#!/bin/bash\ncat > /dev/null\n" + body)
    (b / "siesta").chmod(0o755)
    return dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", MB_MONITOR="0")


def _wrapper(tmp_path: Path, **kw) -> Path:
    (tmp_path / "job.fdf").write_text("SystemLabel job\nNumberOfAtoms 4\n")
    return write_run_wrapper(tmp_path / "job.fdf", env=None, mpi_np=1,
                             emit_sbatch=False, **kw)


def _run(tmp_path: Path, w: Path, env: dict, *args):
    return subprocess.run(["bash", w.name, *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


# ------------------------------------------------------------------ #
#  The flat case is untouched                                         #
# ------------------------------------------------------------------ #


def test_flat_wrapper_is_unchanged_by_default(tmp_path):
    """A plain run directory IS a run (project-layout.md § 1.1), so it keeps
    filename indexing and every flag it has today."""
    text = render_run_wrapper(tmp_path / "job.fdf", mpi_np=1)
    assert "-run${_run_n}.out" in text, "filename indexing still resolves"
    assert "run-*/" not in text, "no attempt directories unless asked for"
    assert "--force" in text, "--force survives in the flat shape"


# ------------------------------------------------------------------ #
#  One directory per attempt                                          #
# ------------------------------------------------------------------ #


def test_each_invocation_gets_its_own_directory(tmp_path):
    env = _fake_engine(tmp_path)
    w = _wrapper(tmp_path, attempt_dirs=True)

    for expected in ("run-0", "run-1", "run-2"):
        r = _run(tmp_path, w, env)
        assert r.returncode == 0, r.stderr
        assert (tmp_path / expected).is_dir(), f"{expected} not created"


def test_inputs_are_linked_in_not_copied(tmp_path):
    """The deck and the shared package are linked, so five attempts do not
    make five copies of a pseudopotential."""
    (tmp_path / "Au.psml").write_text("<pseudo/>\n")
    env = _fake_engine(tmp_path)
    w = _wrapper(tmp_path, attempt_dirs=True)
    _run(tmp_path, w, env)

    assert (tmp_path / "run-0" / "job.fdf").is_symlink()
    assert (tmp_path / "run-0" / "Au.psml").is_symlink()


def test_outputs_land_inside_the_attempt(tmp_path):
    """Invariant 8: a run writes only inside itself."""
    env = _fake_engine(tmp_path, 'echo out > job.out\necho geom > job.XV\n')
    w = _wrapper(tmp_path, attempt_dirs=True)
    _run(tmp_path, w, env)

    assert (tmp_path / "run-0" / "job.XV").is_file()
    assert not (tmp_path / "job.XV").exists(), (
        "the container must not accumulate run output")


# ------------------------------------------------------------------ #
#  Carrying, and why it is a copy                                     #
# ------------------------------------------------------------------ #


def test_a_later_attempt_carries_the_previous_one_s_warm_state(tmp_path):
    """The fake engine writes nothing, so these can only have been carried."""
    env = _fake_engine(tmp_path)
    w = _wrapper(tmp_path, attempt_dirs=True)
    _run(tmp_path, w, env)
    (tmp_path / "run-0" / "job.DM").write_bytes(b"density")
    (tmp_path / "run-0" / "job.XV").write_text("coords\n")

    _run(tmp_path, w, env)

    assert (tmp_path / "run-1" / "job.DM").read_bytes() == b"density"
    assert (tmp_path / "run-1" / "job.XV").is_file()


def test_carried_state_is_a_copy_so_the_previous_attempt_stays_immutable(
        tmp_path):
    """A LINK would let the engine write through it into the attempt that
    produced the file -- the trap `jobset` closes with localize-on-run, and the
    thing that makes an attempt immutable at all."""
    env = _fake_engine(tmp_path, 'echo overwritten > job.DM\n')
    w = _wrapper(tmp_path, attempt_dirs=True)
    _run(tmp_path, w, env)
    (tmp_path / "run-0" / "job.DM").write_bytes(b"density")

    _run(tmp_path, w, env)

    assert not (tmp_path / "run-1" / "job.DM").is_symlink()
    assert (tmp_path / "run-0" / "job.DM").read_bytes() == b"density", (
        "run-0 must be byte-identical after a later attempt ran")


def test_cold_skips_the_carry_and_leaves_the_previous_attempt_alone(tmp_path):
    """`--cold` needs no move-aside here: the previous attempt keeps its own
    state in its own directory, because nothing is written back into it."""
    env = _fake_engine(tmp_path)
    w = _wrapper(tmp_path, attempt_dirs=True)
    _run(tmp_path, w, env)
    (tmp_path / "run-0" / "job.DM").write_bytes(b"density")

    _run(tmp_path, w, env, "--cold")

    assert not (tmp_path / "run-1" / "job.DM").exists()
    assert (tmp_path / "run-0" / "job.DM").read_bytes() == b"density"


# ------------------------------------------------------------------ #
#  --force is retired, and says so                                    #
# ------------------------------------------------------------------ #


def test_force_is_refused_rather_than_ignored(tmp_path):
    """Its whole purpose is to reset the sequence and clobber a result.  With a
    directory per attempt there is nothing to clobber -- and silently doing
    something else would leave a user who asked for a reset believing they got
    one."""
    env = _fake_engine(tmp_path)
    w = _wrapper(tmp_path, attempt_dirs=True)

    r = _run(tmp_path, w, env, "--force")

    assert r.returncode == 2
    assert "--force is not available" in r.stderr
    assert not (tmp_path / "run-0").exists(), "it must refuse before writing"


# ------------------------------------------------------------------ #
#  The session log follows its attempt                                #
# ------------------------------------------------------------------ #


def test_the_session_log_ends_up_inside_the_attempt(tmp_path):
    """It OPENS in the container -- it records the setup that builds the
    attempt -- and is moved in at exit, so the container does not accumulate
    one log per invocation and the finished attempt owns every byte."""
    env = _fake_engine(tmp_path)
    w = _wrapper(tmp_path, attempt_dirs=True)
    _run(tmp_path, w, env)
    _run(tmp_path, w, env)

    assert list(tmp_path.glob("run-*/job.runwrap-*.log")), "moved in"
    assert not list(tmp_path.glob("job.runwrap-*.log")), (
        "the container must not keep them")


def test_failure_hints_can_still_read_the_session_log_after_the_cd(tmp_path):
    """The regression this nearly shipped: `$_runwrap_log` was relative, so
    after the cd the propor/ERROR greps -- which read `"$_out_file"
    "$_runwrap_log"` together -- resolved it against the attempt directory,
    where it is not, and silently searched half the evidence."""
    text = render_run_wrapper(tmp_path / "job.fdf", mpi_np=1, attempt_dirs=True)
    assert '_runwrap_log="$PWD/' in text, (
        "must be absolute, because the wrapper cd's after opening it")
