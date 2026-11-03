"""A monitor says, in the run's session log, that it loaded
(`execution/run-reports.md` § 2.6): ``monitor: starting ...`` before it
imports the monitor and ``monitor: mb_monitor.pyz started`` once it has, in
the session log's own line -- and only then opens its own `.monitor.log`.
The session log carries the engine's wall too (`running-a-job.md` § 4.1).

Read on the pass's H2 PySCF run, as the road prepared and launched it
(`support.real_runs.pyscf`): ``molbuilder describe`` -> ``prep task`` ->
``launch task --mode direct``.  A monitor that dies loading is not made here:
no state the road writes yields a bundle the job's python cannot load, and a
bundle edited by hand after prep is not one molbuilder writes
(`process/testing.md` § 6).
"""
from __future__ import annotations

import pytest

from _road import conda_hook, env_available

CONDA_SH = conda_hook()

pytestmark = pytest.mark.engine

_needs = {env: pytest.mark.skipif(
    not (CONDA_SH.is_file() and env_available(env)),
    reason=f"needs the {env} env + a detectable conda hook")
    for env in ("molbuilder-pySCF",)}


def _session_log(ran_in) -> str:
    """The session log of the run in ``ran_in`` -- its one start's
    (`runs.Run.session_log`) -- or ``""`` where it holds none."""
    from molbuilder.runs import run_of
    log = run_of(ran_in).session_log
    return log.read_text(errors="replace") if log is not None else ""


@_needs["molbuilder-pySCF"]
def test_a_monitor_that_loads_says_so(real_h2_pyscf):
    """``starting`` and then ``started``, and the monitor went on to open its
    own log -- on the pass's PySCF run, its bundle as prepared
    (`support.real_runs.pyscf`).  And the session log carries the engine's
    wall, as it does for SIESTA (`running-a-job.md` § 4.1, *both* engines):
    the run record reads it, and the run's answer carries it once the run
    has ended (`web/trajectory.md` § 4, the badge's *total*).

    MUTATIONS THIS MUST FAIL AGAINST: the bundle's entry importing the
    monitor with nothing said (`runwrap._BUNDLE_MAIN` before 2026-09-27); a
    PySCF run script writing no wall line, so a finished PySCF run's badge
    said no total (2.56.2's first revision).
    """
    from molbuilder.jobset.runstatus import run_answer
    from molbuilder.runs import folder_answer, run_of
    from molbuilder.runwrap import MONITOR_BUNDLE
    ran_in = real_h2_pyscf.attempt("coarse")
    said = _session_log(ran_in)
    starting = said.find(f"[INFO ] monitor: starting {MONITOR_BUNDLE} on python ")
    started = said.find(f"[INFO ] monitor: {MONITOR_BUNDLE} started")
    assert starting != -1 and started != -1, said[-3000:]
    assert starting < started, "started was said before starting"
    assert run_of(ran_in).file(".monitor.log") is not None, (
        "the monitor said it started and never opened its own log")
    wall = (folder_answer(ran_in)["record"]["computation"].get("time", {})
            .get("engine_elapsed_s"))
    assert wall is not None and wall > 0, said[-3000:]
    assert run_answer(ran_in)["engine_elapsed_s"] == wall
