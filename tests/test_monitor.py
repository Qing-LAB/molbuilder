"""Tests for the background job-monitor + notifier hooks
(``molbuilder.monitor``) -- the PoC front end of the job-monitor/watcher
+ notifier surface (execution/running-a-job.md § 4.1, item F).

Deterministic: ``run_monitor`` takes injectable ``sleep``/``clock`` and a
``max_ticks`` bound, so no real time passes and no real process is spawned.
"""
from __future__ import annotations

from pathlib import Path

import os

import pytest

from molbuilder import monitor


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


# --------------------------------------------------------------------- #
#  parse_status                                                        #
# --------------------------------------------------------------------- #


def test_parse_status_counts_iters_and_energy(tmp_path):
    out = tmp_path / "j.out"
    out.write_text("siesta: start\nscf:   1   -100.5   -100.5  1.0\n"
                   "scf:   2   -100.6   -100.6  0.5\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n200.0 2 scf: 2 -100.6\n")
    st = monitor.parse_status(out, timing, start_epoch=1000.0,
                              now_epoch=1010.0)
    assert st.n_iters == 2
    assert st.scf_iter == "2"
    assert st.elapsed_s == pytest.approx(10.0)
    assert st.per_iter_s == pytest.approx(5.0)       # elapsed / n_iters
    assert st.energy == "-100.6"                      # last scf line, field 3
    assert st.state == "running"


def test_parse_status_never_calls_the_run_over(tmp_path):
    """`job-contracts.md` gives this module one stop signal: the watched
    PID.  Reading the artifacts must not produce a second one.

    Until 2026-08-26 a tail scan for completion markers set
    ``state = "done"`` right here, and ``run_monitor`` returned on it.  One
    of those markers was ``siesta: Final energy``, which SIESTA prints
    BEFORE the run is over -- so the monitor could stop sampling while the
    job still held its CPUs and GPUs, losing exactly the utilisation data
    it exists to collect.
    """
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n"
                   "siesta: Final energy = -1\n"
                   ">> End of run: completed\nJob completed\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    st = monitor.parse_status(out, timing, 1000.0, 1001.0)
    assert st.state != "done", (
        "the artifacts must not decide the run is over -- only the PID does")
    assert not hasattr(st, "done_marker")


def test_parse_status_missing_files_safe(tmp_path):
    st = monitor.parse_status(tmp_path / "nope.out",
                              tmp_path / "nope.log", 0.0, 5.0)
    assert st.n_iters == 0
    assert st.state == "starting"
    assert st.elapsed_s == pytest.approx(5.0)


# --------------------------------------------------------------------- #
#  run_monitor loop + notifier hooks                                   #
# --------------------------------------------------------------------- #


def test_a_completion_marker_does_not_stop_the_sampling(tmp_path):
    """The loop keeps going while the watched PID lives, whatever the
    ``.out`` says.

    Bounded by ``max_ticks`` rather than by the marker: if the marker were
    still terminal this would return after one tick, and the assertion on
    the tick count is what catches it.  ``os.getpid()`` is a PID that is
    certainly alive -- this process.
    """
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n"
                   "siesta: Final energy = -1\n>> End of run:\nJob completed\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"

    # Count LOOP ITERATIONS, not notifier ticks: the tick hook fires only
    # while the job is progressing (quiet-when-stalled), and this fixture
    # is deliberately static.  One sleep per iteration is the direct
    # measure of "did the loop keep going".
    slept = []
    final = monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=os.getpid(),
        max_ticks=3, sleep=lambda s: slept.append(s),
        clock=_fake_clock([1000.0, 1001.0, 1002.0, 1003.0, 1004.0]),
    )

    assert final.state != "gone", "the watched pid is this process; it is alive"
    assert len(slept) >= 2, (
        f"the loop ran {len(slept)} time(s) -- a completion marker in the "
        ".out ended it early instead of the watched pid")


def test_run_monitor_stops_when_watch_pid_gone(tmp_path):
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")          # no done marker
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"
    # PID 999999999 almost certainly does not exist -> "gone" on tick 1.
    final = monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=999_999_999,
        sleep=lambda s: None, clock=_fake_clock([0.0, 0.0, 1.0]),
    )
    assert final.state in ("gone", "done")
    assert "job ended" in log.read_text()


def test_run_monitor_logs_each_progress_tick(tmp_path):
    # A PROGRESSING job (the timing log grows by one iter every wake)
    # logs a [STATUS] line on each advance, with the per-iter average
    # shown.
    #
    # It used to also assert a `tick` NOTIFICATION per advance.  That
    # coupling WAS the defect: notifying on every advance is notifying on
    # every wake, so a webhook fired every few seconds for the length of a
    # run (2026-08-26).  The log stays dense -- it is the record -- while
    # who gets TOLD is policy, owned by test_monitor_notify_policy.py.
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")          # no done marker
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"
    step = {"n": 1}

    def _grow(_):                       # injected sleep: advance the run
        step["n"] += 1
        with timing.open("a", encoding="utf-8") as fh:
            fh.write(f"{100.0 + step['n']} {step['n']} scf: {step['n']}\n")

    monitor.clear_notifiers()
    monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=0,   # 0 => never "gone"
        sleep=_grow, max_ticks=3,
        clock=_fake_clock([0.0, 0.0, 1.0, 2.0, 3.0]),
    )
    text = log.read_text()
    assert text.count("[STATUS]") == 3
    assert "avg_per_iter=" in text                  # progressing -> shown


def test_run_monitor_quiet_when_stalled(tmp_path):
    # A STALLED live job (no new iters, no energy change) must NOT flush a
    # [STATUS]/timing line on every wake, and (heartbeat window not yet
    # reached) emits nothing at all.
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")    # frozen forever
    log = tmp_path / "j.monitor.log"
    # Notifications are policy now.  With none set, a stalled job inside
    # its heartbeat window must produce no PROGRESS event -- start and
    # finish bracket every run and are not what "quiet" is about.
    seen = []
    monitor.clear_notifiers()
    monitor.register_notifier(lambda st, ev: seen.append(ev))
    monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=0,
        sleep=lambda s: None, max_ticks=4,
        stall_heartbeat_s=1000.0,                   # never reached here
        clock=_fake_clock([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    text = log.read_text()
    assert "[STATUS]" not in text                   # no spam
    assert "[STALL]" not in text                    # window not reached
    assert [e for e in seen if e not in ("start", "finish")] == [], (
        f"a stalled job inside its window reported progress: {seen}")


def test_run_monitor_stall_heartbeat_is_throttled(tmp_path):
    # A long stall emits at most one [STALL] heartbeat per window, with NO
    # per-iteration estimate in it.
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"
    monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=0,
        sleep=lambda s: None, max_ticks=3,
        stall_heartbeat_s=5.0,
        # start=0 (last_emit=0); ticks at now=3 (<5), 6 (>=5 -> STALL),
        # 9 (6+3<5+6 -> no).  Exactly one heartbeat.
        clock=_fake_clock([0.0, 0.0, 0.0, 3.0, 6.0, 9.0]),
    )
    text = log.read_text()
    stall_lines = [ln for ln in text.splitlines() if "[STALL]" in ln]
    assert len(stall_lines) == 1
    assert "avg_per_iter=" not in stall_lines[0]    # no timing in the ping
    assert "[STATUS]" not in text


def test_run_monitor_stall_heartbeat_zero_is_silent(tmp_path):
    # --stall-heartbeat 0 -> no [STALL] line ever; a stalled live job is
    # completely quiet until it progresses or ends.
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"
    monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=0,
        sleep=lambda s: None, max_ticks=5,
        stall_heartbeat_s=0.0,
        clock=_fake_clock([0.0, 0.0, 0.0, 100.0, 200.0, 300.0, 400.0, 500.0]),
    )
    text = log.read_text()
    assert "[STALL]" not in text
    assert "[STATUS]" not in text                   # nothing changed, ever


def test_util_sampling_change_gated_and_summary(tmp_path):
    # The util CSV is change-gated: a row is written only when a metric
    # moves >=10% from the last logged row (+ a keepalive), and a
    # [UTIL-SUMMARY] verdict lands at finish.
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"
    util = tmp_path / "j.util.csv"

    # Scripted samples: high GPU sm (saturated), one flat repeat (should be
    # skipped), then a >10% drop (should log).
    seq = [
        monitor.UtilSample(0.0, 50.0, 100.0, [(0, 95.0, 40.0, 20.0)]),
        monitor.UtilSample(0.0, 51.0, 100.0, [(0, 94.0, 40.0, 20.0)]),  # flat
        monitor.UtilSample(0.0, 52.0, 100.0, [(0, 60.0, 40.0, 20.0)]),  # sm drop
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

    monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=999_999_999,
        sleep=lambda s: None, max_ticks=4, util_path=util,
        util_keepalive_s=1e9, sampler=_sampler,
        clock=_fake_clock([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    rows = util.read_text().splitlines()
    assert rows[0].startswith("epoch,iso,cpu_pct,mem_gb,gpu0_sm")
    # header + first sample + the sm-drop sample (the flat one is skipped);
    # the watched PID is dead so the loop ends on tick 1 -> final forced row.
    assert len(rows) >= 2
    assert "[UTIL-SUMMARY]" in log.read_text()
    assert "gpu0 sm mean=" in log.read_text()


def test_parse_status_geometry_move(tmp_path):
    # The geometry-relaxation move number is parsed from the .out tail
    # (None for single-point runs); the highest move wins.
    out = tmp_path / "j.out"
    out.write_text("Begin CG move = 1\nscf:   3   -100.7\n"
                   "Begin CG move = 3\nscf:   5   -100.9\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 5 scf: 5 -100.9\n")
    st = monitor.parse_status(out, timing, 0.0, 10.0)
    assert st.geom_step == 3


# --------------------------------------------------------------------- #
#  notifier robustness                                                  #
# --------------------------------------------------------------------- #


def test_failing_notifier_does_not_break_loop(tmp_path):
    # Ends on the WATCHED PID, which is the only stop signal (2026-08-26).
    # This wrote "Job completed" and watched pid 0 -- it was terminated by
    # the completion marker, and when that path went the test looped
    # forever instead of failing.  A test must not borrow its exit from
    # behaviour it is not testing.
    out = tmp_path / "j.out"; out.write_text("scf:   1   -100.5\n")
    timing = tmp_path / "j.scf-timing.log"; timing.write_text("100.0 1 scf:1\n")
    log = tmp_path / "j.monitor.log"

    def _boom(st, ev):
        raise RuntimeError("notifier blew up")

    seen = []
    monitor.clear_notifiers()
    monitor.register_notifier(_boom)
    monitor.register_notifier(lambda st, ev: seen.append(ev))
    try:
        final = monitor.run_monitor(out, timing, log, interval=1,
                                    watch_pid=999_999_999,
                                    sleep=lambda s: None,
                                    clock=_fake_clock([0.0, 0.0, 1.0]))
    finally:
        monitor.clear_notifiers()
    # The second notifier still ran despite the first raising.
    assert final.state == "gone"
    assert seen  # at least the start/finish fired




def test_pid_alive_self():
    import os
    assert monitor._pid_alive(os.getpid()) is True
    assert monitor._pid_alive(0) is True             # 0 => not watching


# --------------------------------------------------------------------- #
#  A TranSIESTA device's NEGF loop, and the closing summary              #
#  (`model/parse.md` § 5d.2, § 5d.5; plan W35 P1)                        #
# --------------------------------------------------------------------- #

_TS_FIXTURE = (Path(__file__).parent / "parse" / "fixtures" / "transiesta"
               / "device-diverging.out")


def _rendered_tee(tmp_path):
    """The timing tee as the wrapper renders it -- extracted from a real
    wrapper, never re-typed."""
    from molbuilder.jobset.model import Resources
    from molbuilder.runwrap import render_run_wrapper
    import json
    deck = tmp_path / "JOB.fdf"
    deck.write_text("SystemLabel JOB\n")
    (tmp_path / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    text = render_run_wrapper(deck, resources=Resources(mpi_np=1), env="e")
    start = text.index("_mb_scf_tee() {")
    return text[start:text.index("\n}\n", start) + 3]


def test_the_rendered_timing_tee_stamps_every_negf_iteration(tmp_path):
    """The wrapper's instrument counts TranSIESTA's ``ts-scf:`` rows as well
    as SIESTA's ``scf:`` rows -- it counted 7 of a device's 1007 iterations
    and timed them at "4049.94 s/iter" (2026-09-25).  Runs the tee the
    wrapper actually writes over a real device output."""
    import subprocess
    tee = tmp_path / "tee.sh"
    tee.write_text(_rendered_tee(tmp_path))
    out, log = tmp_path / "copy.out", tmp_path / "timing.log"
    subprocess.run(["bash", "-c", f'source "{tee}"; _mb_scf_tee "{out}" "{log}"'],
                   stdin=_TS_FIXTURE.open("rb"), check=True)
    prefixes = [line.split()[2] for line in log.read_text().splitlines()]
    assert prefixes.count("scf:") == 7
    assert prefixes.count("ts-scf:") == 8
    assert out.read_bytes() == _TS_FIXTURE.read_bytes()


def test_the_monitor_follows_a_negf_loop(tmp_path):
    """The monitor reads the tee's log and the output's last SCF row of
    either phase -- it reported "no SCF progress" for 7.6 hours of NEGF
    iterations."""
    import subprocess
    tee = tmp_path / "tee.sh"
    tee.write_text(_rendered_tee(tmp_path))
    out, log = tmp_path / "copy.out", tmp_path / "timing.log"
    subprocess.run(["bash", "-c", f'source "{tee}"; _mb_scf_tee "{out}" "{log}"'],
                   stdin=_TS_FIXTURE.open("rb"), check=True)
    st = monitor.parse_status(out, log, start_epoch=0.0, now_epoch=150.0)
    assert st.n_iters == 15
    assert st.scf_iter == "1000"
    # E_KS, the energy the parser gives this run -- not the Eharris column
    # beside it, which the monitor reported until 2026-09-26.
    assert st.energy == "-205444.335258"


def _rendered_monitor_flags(tmp_path):
    """The grammar flags the wrapper passes the shipped monitor -- read out of
    a rendered wrapper, never re-typed."""
    import re as _re
    import shlex
    from molbuilder.jobset.model import Resources
    from molbuilder.runwrap import render_run_wrapper
    import json
    deck = tmp_path / "JOB.fdf"
    deck.write_text("SystemLabel JOB\n")
    (tmp_path / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    text = render_run_wrapper(deck, resources=Resources(mpi_np=1), env="e")
    m = _re.search(r'(--scf-row ".*?" --scf-energy-field \d+ '
                   r'--geom-row ".*?") ', text)
    assert m, "the wrapper passes the monitor no grammar"
    return shlex.split(m.group(1))


@pytest.mark.parametrize("sig,ended", [("SIGTERM", True), ("SIGUSR1", False)],
                         ids=["the-job-ended", "a-retry-in-place"])
def test_the_shipped_monitor_closes_on_the_wrapper_s_signal(tmp_path, sig,
                                                            ended):
    """The monitor as a compute node runs it -- the shipped copy, molbuilder
    unimportable, the grammar only in the flags the wrapper renders --
    follows a real device's NEGF rows and, when the wrapper stops it, writes
    its closing lines at once however long its interval.  SIGTERM is the
    job's end and sends "it ended"; SIGUSR1 is a warm retry in the same
    process, and sends nothing (`run-reports.md` § 2).  Until 2026-09-26 the
    monitor wrote no closing lines at all (0 of 9 logs), then for a while
    sent "it ended" on every retry."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from molbuilder.runwrap import _config_dir_source, _monitor_source
    run = tmp_path / "run"
    run.mkdir()
    (run / "mb_monitor.py").write_text(_monitor_source())
    (run / "config_dir.py").write_text(_config_dir_source())
    flags = _rendered_monitor_flags(tmp_path)
    tee = tmp_path / "tee.sh"
    tee.write_text(_rendered_tee(tmp_path))
    subprocess.run(["bash", "-c", f'source "{tee}"; _mb_scf_tee '
                                  f'"{run}/dev.out" "{run}/dev.timing.log"'],
                   stdin=_TS_FIXTURE.open("rb"), check=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    assert subprocess.run([sys.executable, "-c", "import molbuilder"],
                          cwd=run, env=env).returncode != 0, (
        "molbuilder is importable here, so this is not a compute node")
    log = run / "dev.monitor.log"
    watched, mon = subprocess.Popen(["sleep", "120"]), None
    try:
        mon = subprocess.Popen(
            [sys.executable, "mb_monitor.py", "--out", "dev.out",
             "--timing", "dev.timing.log", "--log", log.name,
             "--util", "dev.util.csv", "--interval", "60",
             "--watch-pid", str(watched.pid), "--nice", "0", *flags],
            cwd=run, env=env)
        for _ in range(300):
            if log.exists() and "[MONITOR] start" in log.read_text():
                break
            time.sleep(0.1)
        mon.send_signal(getattr(signal, sig))
        assert mon.wait(timeout=10) == 0, "the stop waited out the interval"
    finally:
        watched.kill()
        if mon is not None and mon.poll() is None:
            mon.kill()
    text = log.read_text()
    closing = [line for line in text.splitlines() if "[STATUS]" in line][-1]
    assert "energy=-205444.335258" in closing and "last_iter=1000" in closing
    assert "[UTIL-SUMMARY]" in text
    assert ("(stopped by SIGTERM)" in text) is ended
    assert ("retrying this attempt in place" in text) is not ended
    assert ("finish:" in text) is ended
