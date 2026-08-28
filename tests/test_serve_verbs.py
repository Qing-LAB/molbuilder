"""``molbuilder serve`` as verbs — daemon, pidfile, log roll, recovery.

Contract: `docs/ops/deployment.md` § 1.0a–1.0d (user-approved design,
2026-08-28).  The properties under guard, each named for its failure:

* the log CANNOT grow without bound — cap, gzip, keep N, oldest deleted;
* a pid is verified before it is signalled — a stale file whose pid was
  recycled is *reported*, never signalled;
* a child killed by a signal comes BACK (the 2026-08-28 hung-child
  repair), and a flapping child is given up on rather than looped;
* the old bare ``molbuilder serve`` spelling is DEAD (rename = delete).

The integration tests drive a REAL detmon-less supervisor subprocess with
a stub child — the signal handling and respawn logic live in process
mechanics no unit test can see.
"""
from __future__ import annotations

import gzip
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from molbuilder import serve_daemon as sd


# ------------------------------------------------------------- the roll

def test_the_log_cannot_grow_without_bound(tmp_path):
    log = tmp_path / "serve-1.log"
    roll = sd.LogRoll(log, max_bytes=1000, keep=2)
    for i in range(12):
        roll.write(b"x" * 400)
    roll.close()
    assert log.stat().st_size <= 1000, "the live file blew past its cap"
    archives = sorted(p.name for p in tmp_path.glob("serve-1.log.*.gz"))
    assert archives == ["serve-1.log.1.gz", "serve-1.log.2.gz"], (
        f"keep=2 must keep exactly two archives, oldest deleted: {archives}")
    with gzip.open(tmp_path / "serve-1.log.1.gz", "rb") as fh:
        assert fh.read(4) == b"xxxx", "the archive must hold the real bytes"


def test_rotation_shifts_archives_up_and_drops_the_oldest(tmp_path):
    log = tmp_path / "s.log"
    roll = sd.LogRoll(log, max_bytes=10, keep=2)
    roll.write(b"FIRST-11bytes")      # rotate 1 -> .1.gz holds FIRST
    roll.write(b"SECOND-12bytes")     # rotate 2 -> .1.gz SECOND, .2.gz FIRST
    roll.write(b"THIRD-11bytes")      # rotate 3 -> FIRST falls off the end
    roll.close()
    with gzip.open(tmp_path / "s.log.1.gz", "rb") as fh:
        assert b"THIRD" in fh.read()
    with gzip.open(tmp_path / "s.log.2.gz", "rb") as fh:
        assert b"SECOND" in fh.read()
    assert not (tmp_path / "s.log.3.gz").exists(), (
        "keep=2 kept a third archive -- the cap is not a cap")


# ------------------------------------------------- verify before signal

def test_a_dead_pid_reads_dead_and_a_live_foreign_one_foreign(tmp_path):
    assert sd.pid_state(None) == "dead"
    assert sd.pid_state(999_999_999) == "dead"
    sleeper = subprocess.Popen([sys.executable, "-c",
                                "import time; time.sleep(30)"])
    try:
        assert sd.pid_state(sleeper.pid) == "foreign", (
            "a live process of ours that is NOT a molbuilder serve must "
            "read foreign -- a recycled pid is exactly this shape, and "
            "signalling it would kill an innocent")
    finally:
        sleeper.kill()
        sleeper.wait()


def test_signalling_a_stale_pidfile_reports_and_cleans_never_signals(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sd.run_dir().mkdir(parents=True)
    sd.pid_path(7777).write_text("999999999\n")
    sent = []
    monkeypatch.setattr(os, "kill", lambda *a: sent.append(a))
    ok, msg = sd.signal_supervisor(7777, signal.SIGTERM)
    assert not ok and "stale" in msg
    assert sent == [], "a stale pid was SIGNALLED"
    assert not sd.pid_path(7777).exists(), "the stale file must be cleaned"


# ---------------------------------------------------- the respawn policy

def test_the_exit_policy_is_the_2026_08_28_repair():
    assert sd.child_exit_action(sd.RELOAD_EXIT_CODE) == "respawn"
    assert sd.child_exit_action(-signal.SIGTERM) == "respawn", (
        "a hand-killed hung child must come back -- this is the repair")
    assert sd.child_exit_action(-signal.SIGKILL) == "respawn"
    assert sd.child_exit_action(1) == "exit", (
        "a clean nonzero is a server that cannot start; respawning it "
        "is a tight loop wearing a recovery's clothes")
    assert sd.child_exit_action(0) == "exit"


def test_two_crashes_in_the_window_is_flapping():
    now = 1000.0
    assert not sd.flapping([now - 40.0], now)
    assert not sd.flapping([now - 40.0, now - 35.0], now)
    assert sd.flapping([now - 10.0, now - 1.0], now)


# ------------------------------------------------- the real supervisor

def _spawn_supervisor(tmp_path, port=9321, child_sleep=60):
    """A REAL supervisor subprocess with HOME in tmp and a stub child."""
    code = (
        "import sys\n"
        "from molbuilder.serve_daemon import supervise\n"
        f"supervise({port}, [sys.executable, '-u', '-c', "
        f"'import time,sys; print(\"child up\", flush=True); "
        f"time.sleep({child_sleep})'], "
        "log_max_bytes=1_000_000, log_keep=2)\n")
    env = dict(os.environ, HOME=str(tmp_path))
    return subprocess.Popen([sys.executable, "-c", code], env=env,
                            cwd="/home/qqing/molbuilder")


def _child_pids(sup_pid):
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(sup_pid)],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def _wait_for(cond, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.2)
    return None


def test_the_daemon_lifecycle_end_to_end(tmp_path):
    """pidfile written · SIGHUP recycles the child · a KILLED child comes
    back (the repair) · SIGTERM stops cleanly and removes the pidfile."""
    sup = _spawn_supervisor(tmp_path)
    try:
        pidfile = tmp_path / ".molbuilder" / "run" / "serve-9321.pid"
        assert _wait_for(lambda: pidfile.exists()), "no pidfile appeared"
        assert int(pidfile.read_text()) == sup.pid

        first = _wait_for(lambda: (_child_pids(sup.pid) or [None])[0])
        assert first, "no child appeared"

        # restart: SIGHUP -> a DIFFERENT child
        os.kill(sup.pid, signal.SIGHUP)
        second = _wait_for(
            lambda: (lambda c: c[0] if c and c[0] != first else None)(
                _child_pids(sup.pid)))
        assert second, "SIGHUP did not produce a fresh child"

        # the repair: kill the CHILD directly -> it comes back
        os.kill(second, signal.SIGKILL)
        third = _wait_for(
            lambda: (lambda c: c[0] if c and c[0] != second else None)(
                _child_pids(sup.pid)))
        assert third, (
            "a child killed by a signal did not come back -- the "
            "2026-08-28 hung-child repair is not holding")

        # stop: SIGTERM -> clean exit, pidfile gone
        os.kill(sup.pid, signal.SIGTERM)
        assert sup.wait(timeout=15) == 0
        assert not pidfile.exists(), "stop must remove the pidfile"

        log = tmp_path / ".molbuilder" / "logs" / "serve-9321.log"
        text = log.read_text()
        assert "child up" in text, "the child's output must reach the roll"
        assert "restart requested" in text
        assert "died by signal" in text
        assert "stopped on request" in text
    finally:
        if sup.poll() is None:
            sup.kill()
            sup.wait()


def test_a_flapping_child_is_given_up_on(tmp_path):
    """Two immediate signal-deaths and the supervisor exits 1 -- a server
    that dies on arrival is a config problem, not a thing to loop."""
    code = (
        "import sys\n"
        "from molbuilder.serve_daemon import supervise\n"
        "raise SystemExit(supervise(9322, "
        "[sys.executable, '-c', "
        "'import os,signal; os.kill(os.getpid(), signal.SIGKILL)'], "
        "log_max_bytes=1_000_000, log_keep=1))\n")
    env = dict(os.environ, HOME=str(tmp_path))
    sup = subprocess.Popen([sys.executable, "-c", code], env=env,
                           cwd="/home/qqing/molbuilder")
    try:
        assert sup.wait(timeout=30) == 1
        log = (tmp_path / ".molbuilder" / "logs" / "serve-9322.log")
        assert "giving up" in log.read_text()
    finally:
        if sup.poll() is None:
            sup.kill()
            sup.wait()


# ----------------------------------------------------- the CLI surface

def test_the_bare_serve_spelling_is_dead():
    """Rename = delete the old everywhere: `molbuilder serve --port N`
    must refuse, not quietly run a foreground server."""
    from click.testing import CliRunner
    from molbuilder.cli import cli
    r = CliRunner().invoke(cli, ["serve", "--port", "8123"])
    assert r.exit_code != 0
    assert "no such option" in r.output.lower() or "Usage:" in r.output


def test_start_refuses_when_already_running(monkeypatch):
    from click.testing import CliRunner
    from molbuilder.cli import cli
    monkeypatch.setattr(sd, "read_pid", lambda port: 4242)
    monkeypatch.setattr(sd, "pid_state", lambda pid: "ours")
    r = CliRunner().invoke(cli, ["serve", "start", "--port", "8123"])
    assert r.exit_code != 0
    assert "already running" in r.output
