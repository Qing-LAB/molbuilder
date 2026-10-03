"""SIESTA warm-retry (``continue_retries``) — wrapper contract + ground truth.

The retry feature re-execs the wrapper with ``--continue`` when a SIESTA run
failed in a RETRIABLE way.  Two detection points, each grounded in the
project's frozen real-output fixtures (``tests/watch/fixtures/siesta_frozen/``):

  * SCF abort — with ``SCF.MustConverge`` (SIESTA's default, and molbuilder
    emits no override) an unconverged SCF stops the run NON-zero with
    ``SCF_NOT_CONV:`` in the ``.out`` (then ABNORMAL_TERMINATION + MPI
    abort).  So the SCF retry lives in the wrapper's non-zero-exit branch,
    and warm ``--continue`` resumes from the banked ``.DM`` with a fresh
    iteration budget.
  * Geometry cap — a relaxation that exhausts its MD step budget unconverged
    exits 0 and prints ``outcoor: Final (unrelaxed) atomic coordinates``
    (a converged relax prints ``Relaxed atomic coordinates``).  So the
    geometry retry lives in the zero-exit path.

2026-07-29: replaces the first-cut implementation whose zero-exit-only check
could never fire (SCF aborts exit non-zero; its geometry marker string does
not occur in SIESTA output), and whose ``exec "$0"`` PATH-searched a bare
relative name — exit 127 under the canonical ``bash <base>.run.sh``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.jobset.model import Resources
from molbuilder.runwrap import render_run_wrapper

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "tests" / "watch" / "fixtures" / "siesta_frozen"

# The exact idioms the wrapper renders (tested both as source text and,
# below, behaviourally).  If these change in runwrap.py, change them here
# in the same commit.
SELF_LINE = '_mb_self="$(readlink -f -- "$0" 2>/dev/null || echo "$0")"'
EXEC_LINE = 'exec bash "$_mb_self" --continue'
#: What the retries ASK -- the door's questions (`_run_ending.QUESTIONS`),
#: the SCF's cause rendered from the grammar's own marker.  The wrapper
#: grepped strings of its own until 2026-09-26.
from molbuilder.parse.engines.siesta_grammar import SCF_NOT_CONV_MARKER
ASK_SCF = f'_mb_ending stopped-by "{SCF_NOT_CONV_MARKER}"'
ASK_GEOM = "_mb_ending relaxation-capped"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """An isolated config root and synthetic caps (mirrors test_runwrap_v2);
    the activation the generator needs is the record's, `_MACHINE`."""
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
    # working-directory step, which is gone (configuration.md § 2.1a) --
    # without naming the directory the write lands in a file nothing
    # opens, and the test passes having configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    set_capabilities(Capabilities(runtime_config={},
                                  conda_binary="/usr/bin/conda"))
    yield tmp_path
    set_capabilities(None)


def _machine():
    """This machine's record: how a shell enters an environment here --
    the activation the generator reads (`configuration.md` § 4)."""
    from molbuilder.scheduler import Environment, Topology
    return Environment(scheduler="workstation", topology=Topology(),
                       env_init={"preamble": "module load mamba",
                                          "activation": "source activate"})


def _render(**kw) -> str:
    """The wrapper's text for a four-rank, one-thread SIESTA job.

    ``kw`` are allocation fields: the door takes the object whole (A8), so
    they are set on it rather than passed beside it."""
    return render_run_wrapper(Path("/x/JOB.fdf"), machine_record=_machine(),
                              resources=Resources(mpi_np=4, cpus_per_task=1,
                                                  **kw))


# --------------------------------------------------------------------- #
#  Rendered-wrapper contract                                            #
# --------------------------------------------------------------------- #


class TestRenderedContract:

    def test_no_retry_machinery_without_continue_retries(self, sandbox):
        text = _render()
        assert "_mb_warm_retry" not in text
        assert ASK_SCF not in text
        assert ASK_GEOM not in text

    def test_scf_retry_lives_in_the_nonzero_exit_branch(self, sandbox):
        """The retriable SCF failure exits non-zero (SCF.MustConverge
        default) — the question must be asked BEFORE ``exit
        "$_siesta_exit"``."""
        text = _render(continue_retries=2)
        assert "_mb_warm_retry() {" in text
        i_scf = text.index(ASK_SCF)
        i_exit = text.index('exit "$_siesta_exit"')
        assert i_scf < i_exit, (
            "SCF_NOT_CONV retry must precede the non-zero-exit exit — "
            "SIESTA aborts non-zero on SCF non-convergence")

    def test_geometry_retry_is_asked_on_the_zero_exit_path(self, sandbox):
        """The zero-exit retry asks the door whether the relaxation ran out
        of moves -- the door reads the marker SIESTA actually prints
        (`TestTheDoorOnRealOutput`), not an invented phrase."""
        text = _render(continue_retries=2)
        assert ASK_GEOM in text
        assert "Geometry step did NOT converge" not in text
        # ... and it sits AFTER the non-zero-exit branch closes.
        assert text.index('exit "$_siesta_exit"') < text.index(ASK_GEOM)

    def test_reexec_is_bash_on_an_absolute_self_path(self, sandbox):
        """``exec "$0"`` PATH-searches a bare name under ``bash x.run.sh``
        (exit 127); the wrapper must exec via bash + captured abs path."""
        text = _render(continue_retries=2)
        assert SELF_LINE in text
        assert EXEC_LINE in text
        # No bare self-exec anywhere outside the explanatory comment.
        for line in text.splitlines():
            if 'exec "$0"' in line:
                assert line.lstrip().startswith("#")

    def test_retry_preserves_original_args_minus_continuation_flags(
            self, sandbox):
        text = _render(continue_retries=2)
        assert '_mb_orig_args=(${@:+"$@"})' in text
        # --force / --cold are filtered: they would reset the run-index /
        # move aside the warm-start files the retry depends on.
        assert "--continue|-c|--force|-f|--cold|--from-scratch) ;;" in text

    def test_monitor_is_stopped_as_a_retry_before_reexec(self, sandbox):
        """exec skips the EXIT trap; without the stop each retry would
        stack another mb_monitor.py appending to the same util CSV -- and
        it is stopped as a RETRY (SIGUSR1), because the job goes on and
        "it ended" would be false (`run-reports.md` § 2)."""
        text = _render(continue_retries=2)
        fn = text[text.index("_mb_warm_retry() {"):text.index(EXEC_LINE)]
        assert "_mb_stop_monitor USR1" in fn


# --------------------------------------------------------------------- #
#  Marker ground truth — frozen real SIESTA output                       #
# --------------------------------------------------------------------- #


class TestTheDoorOnRealOutput:
    """What the retries ask, answered by `_run_ending` on frozen real SIESTA
    output -- if SIESTA's wording ever drifts, these fail first.  They pinned
    the wrapper's own grep strings until 2026-09-26, when the wrapper began
    asking the door."""

    def test_the_scf_aborts_were_stopped_by_the_scf(self):
        from molbuilder.parse.engines._run_ending import QUESTIONS, ending_of
        for name in ("hemeC-stage1-scf_not_conv-5fr.out",
                     "hemeC-stage3-scf_not_conv-1fr.out"):
            end = ending_of(FROZEN / name)
            # SIESTA stated the SCF fatal -- "(required)" -- and died: the
            # retriable case, on the non-zero branch
            assert QUESTIONS["stopped-by"](end, SCF_NOT_CONV_MARKER), name
            assert not QUESTIONS["relaxation-capped"](end), name

    def test_the_geometry_cap_is_capped_and_no_scf_abort(self):
        from molbuilder.parse.engines._run_ending import QUESTIONS, ending_of
        end = ending_of(FROZEN / "hemeC-stage2-run3-finished-42fr.out")
        assert QUESTIONS["relaxation-capped"](end)
        # ...and is NOT an SCF abort (exit 0 path).
        assert not QUESTIONS["stopped-by"](end, SCF_NOT_CONV_MARKER)


# --------------------------------------------------------------------- #
#  Behavioural regression — the exit-127 re-exec bug                     #
# --------------------------------------------------------------------- #


class TestReexecMechanics:

    def test_self_reexec_survives_bare_bash_invocation(self, tmp_path):
        """Run the EXACT self-path + exec idiom the wrapper renders, as a
        standalone script invoked the canonical way (``bash x.run.sh``
        with a bare relative name, from the script's dir AND from a
        parent dir).  The first-cut ``exec "$0"`` died here with 127."""
        sub = tmp_path / "sub"
        sub.mkdir()
        scr = sub / "t.run.sh"
        scr.write_text(
            "#!/bin/bash\nset -u\n"
            + SELF_LINE + "\n"
            + 'if [ "${MB_RETRY_N:-0}" -ge 1 ]; then\n'
            + '    echo "SECOND_RUN_OK argv=$*"\n    exit 0\nfi\n'
            + "export MB_RETRY_N=1\n"
            + EXEC_LINE + "\n")
        # From the script's own directory (bare name — the 127 trap):
        r1 = subprocess.run(["bash", "t.run.sh"], cwd=sub,
                            capture_output=True, text=True, timeout=30)
        assert r1.returncode == 0, r1.stderr
        assert "SECOND_RUN_OK" in r1.stdout
        assert "--continue" in r1.stdout      # the retry flag arrived
        # From a parent dir via a relative path (the sbatch shape):
        r2 = subprocess.run(["bash", "sub/t.run.sh"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
        assert r2.returncode == 0, r2.stderr
        assert "SECOND_RUN_OK" in r2.stdout


# (TestInstallWrapperEndpointValidation retired 2026-08-21 with its
#  subject: /api/run/install-wrapper had zero browser callers -- `prep`
#  writes the wrapper beside every deck it renders, and the wrapper's
#  own validation is pinned above through render_run_wrapper directly.)
