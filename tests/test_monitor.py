"""The background monitor -- what is its OWN: when it samples, when it logs,
when it stops, and that it reads a run through the framework's readers
(`execution/run-reports.md` § 2.3).

**Every output here is a real run's, replayed.**  The monitor has no reader of
its own since 2026-09-26, so what these tests feed it is what a job leaves:
the measured H2 relaxation under `tests/fixtures/siesta_relax` (SIESTA 5.4.2)
written out as it grew, and the diverging TranSIESTA device's output run
through the timing tee the wrapper renders.  They are API-level on measured
fixtures -- a loop driven by an injected clock is the only way to hold its
timing rules -- and the road itself is held by the SIESTA and PySCF runs in
`test_siesta_vibration_e2e.py` and `test_vibration_e2e.py`.  *(Until then these
tests wrote their own `scf:` lines, and pinned a private reader, `parse_status`,
that the framework's readers replaced.)*

Deterministic: ``run_monitor`` takes injectable ``sleep``/``clock`` and a
``max_ticks`` bound, so no real time passes and no real process is spawned.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from molbuilder import monitor

#: The measured H2 relaxation: label H2, stage 01_relax, run 0.
_H2 = (Path(__file__).parent / "fixtures" / "siesta_relax" / "01_relax"
       / "run-0")
_H2_OUT = "H2_01_relax-run0.out"


@pytest.fixture(autouse=True)
def _clean_notifiers():
    monitor.clear_notifiers()
    yield
    monitor.clear_notifiers()


def _fake_clock(values):
    """Clock returning successive ``values``; holds the last when drained
    (so a short list never raises StopIteration mid-loop)."""
    it = iter(values)
    box = {"last": values[0] if values else 0.0}

    def _c():
        try:
            box["last"] = next(it)
        except StopIteration:
            pass
        return box["last"]
    return _c


def _replay(tmp_path, upto: int):
    """The H2 run's directory with its ``.out`` written up to line ``upto``
    -- the run as a monitor meets it part-way -- and a ``grow(to)`` that
    writes on to line ``to``, as SIESTA would."""
    run = tmp_path / "run-0"
    shutil.copytree(_H2, run, ignore=shutil.ignore_patterns(_H2_OUT))
    lines = (_H2 / _H2_OUT).read_text().splitlines(keepends=True)
    out = run / _H2_OUT
    out.write_text("".join(lines[:upto]))
    at = {"n": upto}

    def grow(to: int) -> None:
        with out.open("a") as fh:
            fh.write("".join(lines[at["n"]:to]))
        at["n"] = to
    return run, monitor.WatchedRun(label="H2", stage="01_relax", run=0,
                                   directory=run), grow


def _statuses(run: Path):
    return [ln for ln in (run / "H2_01_relax-run0.monitor.log")
            .read_text().splitlines() if "[STATUS]" in ln]


# --------------------------------------------------------------------- #
#  The PID says WHEN; the framework says HOW                             #
# --------------------------------------------------------------------- #

def test_a_completion_marker_does_not_stop_the_sampling(tmp_path):
    """The loop keeps going while the watched PID lives, whatever the output
    says: the whole run is on disk, `>> End of run` included, and this
    process -- certainly alive -- is the watched PID.

    `job-contracts.md`: the monitor *"follows the launcher's PID -- so it
    knows authoritatively when the run ended, rather than guessing from
    output markers"*.  `siesta: Final energy` prints before a run is over,
    and a monitor that stopped on a marker stopped sampling while the job
    still held its CPUs and GPUs (2026-08-26).
    """
    run, watched, _grow = _replay(tmp_path, upto=10_000)
    slept = []
    final = monitor.run_monitor(
        watched, interval=1, watch_pid=os.getpid(), max_ticks=3,
        sleep=lambda s: slept.append(s),
        clock=_fake_clock([1000.0, 1001.0, 1002.0, 1003.0, 1004.0]))
    assert len(slept) == 3, (
        f"the loop ran {len(slept)} time(s) -- a completion marker ended it")
    assert final.state == "running", "the watched PID is alive"


def test_when_the_pid_goes_the_verdict_is_run_status_s(tmp_path):
    """Asked once the PID has gone, HOW it ended is the Results tab's own
    reading -- `run_status` over the rung's files -- with each phase's
    convergence and the process's goodbye (`run-reports.md` § 2.3).  The
    monitor's own word for it, ``gone``, stood here until 2026-09-26, so every
    finish report read the same whatever the run did.  And a relaxation says
    whether its GEOMETRY converged, as its output states it -- the H2 run
    printed its relaxed coordinates."""
    from molbuilder.parse.dirs import run_status
    run, watched, _grow = _replay(tmp_path, upto=10_000)
    final = monitor.run_monitor(
        watched, interval=1, watch_pid=999_999_999,
        sleep=lambda s: None, clock=_fake_clock([0.0, 0.0, 1.0]))
    rs = run_status(run, "H2_01_relax")
    assert (final.state, final.detail) == (rs.state, rs.detail) == (
        "finished", "job_completed")
    assert final.converged == {"periodic": True}
    assert final.exit and final.exit.startswith("rc=0")
    closing = _statuses(run)[-1]
    assert "finished (job_completed)" in closing and "converged: periodic yes" in closing
    assert final.relaxed is True and "geometry relaxed" in closing, closing
    assert "job ended" in (run / "H2_01_relax-run0.monitor.log").read_text()


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 4 tests here watched a measured output cut short (its marker
# deleted) or the trimmed TranSIESTA device output under a new name (`process/testing.md` § 6).


# --------------------------------------------------------------------- #
#  Looking often, logging on change                                      #
# --------------------------------------------------------------------- #

def test_each_advance_is_a_status_line_stating_where_the_run_is(tmp_path):
    """A PROGRESSING run logs a [STATUS] line on each advance, and the line
    says what is going on: the phase and iteration, the energy, each residual
    beside the criterion the run states for it, and the step SIESTA began --
    in its own words and with its own number.

    Who gets TOLD is policy (`test_monitor_notify_policy.py`); the log is the
    record and stays dense."""
    run, watched, grow = _replay(tmp_path, upto=386)     # move 0, row 1
    chunks = iter([392, 478, 551])                       # rows 7; move 1; move 2

    def _grow(_):                                        # injected sleep
        grow(next(chunks))
    monitor.run_monitor(watched, interval=1, watch_pid=0, sleep=_grow,
                        max_ticks=3,
                        clock=_fake_clock([0.0, 0.0, 1.0, 2.0, 3.0]))
    lines = _statuses(run)
    assert len(lines) == 3, lines
    assert "periodic SCF iteration 7" in lines[0], lines[0]
    assert "dDmax" in lines[0] and "(tol 1e-05)" in lines[0], lines[0]
    assert "dHmax" in lines[0] and "(tol 0.001)" in lines[0], lines[0]
    assert "Broyden opt. move 2" in lines[-1], lines[-1]
    assert "max force 0.2546 eV/Ang (tol 0.01)" in lines[-1], lines[-1]


def test_a_run_that_does_not_advance_writes_and_sends_nothing(tmp_path):
    """A live run that does not advance -- for over an hour here, as a heavy
    SCF step does -- adds no status line and tells no channel anything: the
    monitor judges no stall (`run-reports.md` § 2)."""
    run, watched, _grow = _replay(tmp_path, upto=400)
    seen = []
    monitor.register_notifier(lambda st, ev: seen.append(ev))
    monitor.run_monitor(watched, interval=1, watch_pid=0,
                        sleep=lambda s: None, max_ticks=4,
                        clock=_fake_clock([0.0, 0.0, 0.0, 1200.0, 2400.0,
                                           3600.0, 4800.0]))
    text = (run / "H2_01_relax-run0.monitor.log").read_text()
    assert "[STATUS]" not in text, text
    assert [e for e in seen if e not in ("start", "finish")] == [], seen


def test_util_sampling_is_change_gated_and_summarised(tmp_path):
    """The run's `.util.csv` gets a row only when a metric moves >= 10 % (or a
    keepalive passes), and a [UTIL-SUMMARY] verdict lands at the end.  The
    samples are scripted -- the SAMPLER is the monitor's own, and what it
    reads is the machine, not a file."""
    run, watched, _grow = _replay(tmp_path, upto=10_000)
    seq = [
        monitor.UtilSample(0.0, 50.0, 100.0, [(0, 95.0, 40.0, 20.0)]),
        monitor.UtilSample(0.0, 51.0, 100.0, [(0, 94.0, 40.0, 20.0)]),  # flat
        monitor.UtilSample(0.0, 52.0, 100.0, [(0, 60.0, 40.0, 20.0)]),  # drop
        monitor.UtilSample(0.0, 52.0, 100.0, [(0, 60.0, 40.0, 20.0)]),
    ]
    it = iter(seq)
    last = {"s": seq[0]}

    def _sampler():
        try:
            last["s"] = next(it)
        except StopIteration:
            pass
        return last["s"]

    monitor.run_monitor(watched, interval=1, watch_pid=999_999_999,
                        sleep=lambda s: None, max_ticks=4, util=True,
                        util_keepalive_s=1e9, sampler=_sampler,
                        clock=_fake_clock([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]))
    rows = (run / "H2_01_relax-run0.util.csv").read_text().splitlines()
    assert rows[0].startswith("epoch,iso,cpu_pct,mem_gb,gpu0_sm")
    assert len(rows) >= 2
    log = (run / "H2_01_relax-run0.monitor.log").read_text()
    assert "[UTIL-SUMMARY]" in log and "gpu0 sm mean=" in log


def test_a_failing_notifier_does_not_break_the_loop(tmp_path):
    """Ends on the WATCHED PID, the only stop signal -- a test must not
    borrow its exit from behaviour it is not testing."""
    run, watched, _grow = _replay(tmp_path, upto=10_000)

    def _boom(st, ev):
        raise RuntimeError("notifier blew up")

    seen = []
    monitor.register_notifier(_boom)
    monitor.register_notifier(lambda st, ev: seen.append(ev))
    final = monitor.run_monitor(watched, interval=1, watch_pid=999_999_999,
                                sleep=lambda s: None,
                                clock=_fake_clock([0.0, 0.0, 1.0]))
    assert final.state == "finished"
    assert "finish" in seen, "the second notifier ran despite the first"


def test_pid_alive_self():
    assert monitor._pid_alive(os.getpid()) is True
    assert monitor._pid_alive(0) is True             # 0 => not watching
