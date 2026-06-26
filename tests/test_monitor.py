"""Tests for the background job-monitor + notifier hooks
(``molbuilder.monitor``) -- the PoC front end of the job-monitor/watcher
+ notifier surface (slurm-integration.md § 11.0b, item F).

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


def test_run_monitor_ticks_until_max(tmp_path):
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")          # no done marker
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"
    ticks = []
    monitor.register_notifier(
        lambda st, ev: ticks.append(ev) if ev == "tick" else None)
    monitor.run_monitor(
        out, timing, log, interval=1, watch_pid=0,   # 0 => never "gone"
        sleep=lambda s: None, max_ticks=3,
        clock=_fake_clock([0.0, 0.0, 1.0, 2.0, 3.0]),
    )
    assert len(ticks) == 3
    assert log.read_text().count("[STATUS]") == 3


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
