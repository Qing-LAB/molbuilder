"""SIESTA warm-retry (``continue_retries``) — wrapper contract + ground truth.

The retry feature re-execs the wrapper with ``--continue`` when a SIESTA run
failed in a RETRIABLE way.  Two detection points, each answered on a run made
on the road with the real SIESTA (`tests/test_siesta_stopped_run_e2e.py`,
`tests/test_siesta_flat_run_e2e.py`):

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
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.jobset.model import Resources
from molbuilder.runwrap import render_run_wrapper

REPO = Path(__file__).resolve().parents[1]
# What the retries ask, answered on real output, is the e2e tier's: an SCF
# stop on the stopped run made on the road
# (`tests/test_siesta_stopped_run_e2e.py`), a relaxation out of moves on one
# allowed a single move (`tests/test_siesta_flat_run_e2e.py`).

# The exact idioms the wrapper renders.  If these change in runwrap.py,
# change them here in the same commit.
SELF_LINE = '_mb_self="$(readlink -f -- "$0" 2>/dev/null || echo "$0")"'
EXEC_LINE = 'exec bash "$_mb_self" --continue'
#: What the retries ASK -- the door's questions (`_run_ending.QUESTIONS`),
#: the SCF's cause rendered from the grammar's own marker.
from molbuilder.parse.engines.siesta_grammar import SCF_NOT_CONV_MARKER
from molbuilder.runfiles import RunNames
ASK_SCF = f'_mb_ending stopped-by "{SCF_NOT_CONV_MARKER}"'
ASK_GEOM = "_mb_ending relaxation-capped"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """An isolated config root and synthetic caps; the activation the generator needs is the record's, `_MACHINE`."""
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT (configuration.md § 2.1a).
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
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
    names = RunNames.of("JOB", "01_coarse", "flat")
    return render_run_wrapper(Path("/x") / names.name(".fdf"),
                              machine_record=_machine(), names=names,
                              resources=Resources(mpi_np=4, cpus_per_task=1,
                                                  **kw))


# --------------------------------------------------------------------- #
#  Rendered-wrapper contract                                            #
# --------------------------------------------------------------------- #


class TestRenderedContract:


    def test_geometry_retry_is_asked_on_the_zero_exit_path(self, sandbox):
        """The zero-exit retry asks the door whether the relaxation ran out
        of moves -- the door reads the marker SIESTA actually prints, not
        an invented phrase."""
        text = _render(continue_retries=2)
        assert ASK_GEOM in text
        # ... and it sits AFTER the non-zero-exit branch closes.
        assert text.index('exit "$_siesta_exit"') < text.index(ASK_GEOM)


    def test_monitor_is_stopped_as_a_retry_before_reexec(self, sandbox):
        """exec skips the EXIT trap; without the stop each retry would
        stack another mb_monitor.py appending to the same util CSV -- and
        it is stopped as a RETRY (SIGUSR1), because the job goes on and
        "it ended" would be false (`run-reports.md` § 2)."""
        text = _render(continue_retries=2)
        fn = text[text.index("_mb_warm_retry() {"):text.index(EXEC_LINE)]
        assert "_mb_stop_monitor USR1" in fn
