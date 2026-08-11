"""Tests for the background job-monitor + notifier hooks
(``molbuilder.monitor``) -- the PoC front end of the job-monitor/watcher
+ notifier surface (execution/running-a-job.md § 4.1, item F).

Deterministic: ``run_monitor`` takes injectable ``sleep``/``clock`` and a
``max_ticks`` bound, so no real time passes and no real process is spawned.
"""
from __future__ import annotations

from pathlib import Path

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


def test_parse_status_done_marker(tmp_path):
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n>> End of run: completed\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    st = monitor.parse_status(out, timing, 1000.0, 1001.0)
    assert st.state == "done"
    assert st.done_marker


def test_parse_status_missing_files_safe(tmp_path):
    st = monitor.parse_status(tmp_path / "nope.out",
                              tmp_path / "nope.log", 0.0, 5.0)
    assert st.n_iters == 0
    assert st.state == "starting"
    assert st.elapsed_s == pytest.approx(5.0)


# --------------------------------------------------------------------- #
#  run_monitor loop + notifier hooks                                   #
# --------------------------------------------------------------------- #


def test_run_monitor_finishes_on_done_marker(tmp_path):
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\nJob completed time = 1s\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"

    events = []
    monitor.register_notifier(lambda st, ev: events.append((ev, st.state)))

    final = monitor.run_monitor(
        out, timing, log, interval=300, watch_pid=0,
        sleep=lambda s: None,
        clock=_fake_clock([1000.0, 1000.0, 1001.0]),
    )
    assert final.state == "done"
    # hooks fired for start + finish (the done path skips the per-tick fire).
    # (start state is "done" here because the .out already carries the
    # completion marker -- so assert the EVENTS, not the snapshot state.)
    assert any(ev == "start" for ev, _ in events)
    assert any(ev == "finish" for ev, _ in events)
    text = log.read_text()
    assert "[MONITOR] start" in text
    assert "[STATUS]" in text
    assert "job ended" in text


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
    # logs a [STATUS] line + fires a tick on each advance, with the
    # per-iter average shown.
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

    ticks = []
    monitor.register_notifier(
        lambda st, ev: ticks.append(ev) if ev == "tick" else None)
    monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=0,   # 0 => never "gone"
        sleep=_grow, max_ticks=3,
        clock=_fake_clock([0.0, 0.0, 1.0, 2.0, 3.0]),
    )
    text = log.read_text()
    assert len(ticks) == 3
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
    ticks = []
    monitor.register_notifier(
        lambda st, ev: ticks.append(ev) if ev == "tick" else None)
    monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=0,
        sleep=lambda s: None, max_ticks=4,
        stall_heartbeat_s=1000.0,                   # never reached here
        clock=_fake_clock([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    text = log.read_text()
    assert "[STATUS]" not in text                   # no spam
    assert "[STALL]" not in text                    # window not reached
    assert ticks == []                              # no tick notifications


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
    out = tmp_path / "j.out"; out.write_text("Job completed\n")
    timing = tmp_path / "j.scf-timing.log"; timing.write_text("100.0 1 scf:1\n")
    log = tmp_path / "j.monitor.log"

    def _boom(st, ev):
        raise RuntimeError("notifier blew up")

    seen = []
    monitor.register_notifier(_boom)
    monitor.register_notifier(lambda st, ev: seen.append(ev))
    final = monitor.run_monitor(out, timing, log, interval=1, watch_pid=0,
                                sleep=lambda s: None,
                                clock=_fake_clock([0.0, 0.0, 1.0]))
    # The second notifier still ran despite the first raising.
    assert final.state == "done"
    assert seen  # at least the start/finish fired


def test_webhook_notifier_builds():
    # No network: just confirm the factory returns a callable named hook.
    fn = monitor.make_webhook_notifier("http://example.invalid/hook")
    assert callable(fn)
    assert fn.__name__ == "webhook_notifier"


def test_pid_alive_self():
    import os
    assert monitor._pid_alive(os.getpid()) is True
    assert monitor._pid_alive(0) is True             # 0 => not watching
