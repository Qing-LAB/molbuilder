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
from molbuilder.cli import _DEFAULT_SERVE_HOST

#: The clone root -- the supervisor subprocesses must import
#: molbuilder from THIS tree, wherever it is checked out.
_REPO = str(Path(__file__).resolve().parents[1])


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


@pytest.mark.parametrize("which, what", [("serve", "molbuilder serve"),
                                         ("notebook", "molbuilder notebook")])
@pytest.mark.parametrize("found, said, kept", [
    ("stale", "stale pidfile: pid {pid} is gone (removed {path})", False),
    ("names-no-pid", "not running ({path} names no pid)", True),
    ("recycled", "refusing: pid {pid} is not your {what} -- the pidfile is "
                 "stale and the pid was recycled", True),
])
def test_both_pidfiles_are_verified_by_one_step_in_one_set_of_words(
        tmp_path, monkeypatch, which, what, found, said, kept):
    """Every verb verifies before it acts (`deployment.md` § 1.0a; the
    notebook's, `web/jupyter.md` § 3.3): the supervisor's pidfile and the
    notebook shepherd's by ONE step, in one set of words -- a stale file
    removed and said by its path; a file that names no pid said as such,
    never as no file at all; a pid that is alive and not ours refused, the
    file kept.  THE SILENT FAILURE: a recycled pid signalled -- an innocent
    process killed.  The pidfile lives under a temporary RUNTIME root."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    from molbuilder.config_dir import jupyter_pidfile, serve_pidfile
    path = (serve_pidfile if which == "serve" else jupyter_pidfile)(7777)
    path.parent.mkdir(parents=True, exist_ok=True)
    sleeper = subprocess.Popen([sys.executable, "-c",
                                "import time; time.sleep(30)"])
    pid = {"stale": 999_999_999, "recycled": sleeper.pid}.get(found)
    path.write_text(f"{pid}\n" if pid is not None else "not a pid\n")
    sent, real_kill = [], os.kill
    monkeypatch.setattr(os, "kill", lambda *a: sent.append(a))
    try:
        if which == "serve":
            ok, msg = sd.signal_supervisor(7777, signal.SIGTERM)
        else:
            ok, msg = sd.stop_by_pidfile(path, marker=sd.SHEPHERD_MARKER,
                                         what=what)
    finally:
        # THE TEST'S OWN PROCESS, stopped by the real call, not counted.
        real_kill(sleeper.pid, signal.SIGKILL)
        sleeper.wait()
    assert not ok and sent == [], "a pid that is not ours was SIGNALLED"
    assert msg == said.format(pid=pid, path=path, what=what), msg
    assert path.exists() is kept


def test_status_says_a_pidfile_that_names_no_pid_in_the_same_words(
        tmp_path, monkeypatch):
    """`serve status` answers "is the process up" by the pidfile checks
    `stop` makes (`deployment.md` § 1.0b), so a pidfile that is there and
    names no pid is said as such -- never as no pidfile at all."""
    from click.testing import CliRunner
    from molbuilder.cli import cli
    from molbuilder.config_dir import serve_pidfile
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    path = serve_pidfile(7778)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a pid\n")
    r = CliRunner().invoke(cli, ["serve", "status", "--port", "7778"])
    assert f"not running ({path} names no pid)" in r.output, r.output


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

def _spawn_supervisor(tmp_path, port=9321, child_sleep=60, *,
                      host=_DEFAULT_SERVE_HOST, child=None):
    """A REAL supervisor subprocess with HOME in tmp and a stub child --
    or ``child``, the child's own Python -- handed ``host`` as `serve
    start` hands the supervisor its ``--host``: always, the default bind
    unless told."""
    child = child or (f'import time,sys; print("child up", flush=True); '
                      f'time.sleep({child_sleep})')
    code = (
        "import sys\n"
        "from molbuilder.serve_daemon import supervise\n"
        f"supervise({port}, [sys.executable, '-u', '-c', {child!r}], "
        f"host={host!r}, log_max_bytes=1_000_000, log_keep=2)\n")
    # The supervisor writes its pidfile under the RUNTIME directory and its
    # log under the STATE directory, not under HOME (`configuration.md`
    # § 2.1d).  These are CHILD PROCESSES, so the roots are handed over
    # explicitly rather than monkeypatched.
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(tmp_path),
               XDG_STATE_HOME=str(tmp_path / "state"),
               XDG_RUNTIME_DIR=str(tmp_path / "runtime"))
    return subprocess.Popen([sys.executable, "-c", code], env=env,
                            cwd=_REPO)


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
        pidfile = tmp_path / "runtime" / "molbuilder" / "serve-9321.pid"
        assert _wait_for(lambda: pidfile.exists()), "no pidfile appeared"
        assert sd.read_pidfile_lines(pidfile) == (sup.pid, _DEFAULT_SERVE_HOST)

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

        log = tmp_path / "state" / "molbuilder" / "logs" / "serve-9321.log"
        text = log.read_text()
        assert "child up" in text, "the child's output must reach the roll"
        assert "restart requested" in text
        assert "died by signal" in text
        assert "stopped on request" in text
        # the log is the RECORD (deployment.md 1.0c, user ruling
        # 2026-08-28): every daemon event line carries a timestamp --
        # an event without a time is half a record.
        import re as _re
        stamped = _re.findall(r"\[serve-daemon\] \d{4}-\d{2}-\d{2}T"
                              r"\d{2}:\d{2}:\d{2}", text)
        events = [l for l in text.splitlines() if "[serve-daemon]" in l]
        assert events and len(stamped) == len(events), (
            "every [serve-daemon] event line must be stamped: "
            f"{len(stamped)}/{len(events)}")
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
        f"host={_DEFAULT_SERVE_HOST!r}, log_max_bytes=1_000_000, "
        "log_keep=1))\n")
    # The supervisor writes its pidfile under the RUNTIME directory and its
    # log under the STATE directory, not under HOME (`configuration.md`
    # § 2.1d).  These are CHILD PROCESSES, so the roots are handed over
    # explicitly rather than monkeypatched.
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(tmp_path),
               XDG_STATE_HOME=str(tmp_path / "state"),
               XDG_RUNTIME_DIR=str(tmp_path / "runtime"))
    sup = subprocess.Popen([sys.executable, "-c", code], env=env,
                           cwd=_REPO)
    try:
        assert sup.wait(timeout=30) == 1
        log = (tmp_path / "state" / "molbuilder" / "logs" / "serve-9322.log")
        assert "giving up" in log.read_text()
    finally:
        if sup.poll() is None:
            sup.kill()
            sup.wait()


# ----------------------------------------------------- the CLI surface

#: A server's own /api/health, answering on the one address it is given --
#: the child `serve start --host <address>` runs, as far as status can see.
#: A STAND-IN, for a named need: status asks the child that route and
#: nothing else, and the real child (`serve foreground`) is the whole web
#: application -- its config, sign-in and notebook -- which this test of
#: where status asks has no use for (`testing.md` § 3b).
_ANSWERS_ON = (
    "import http.server, socket\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')\n"
    "    def log_message(self, *a): pass\n"
    "class S(http.server.HTTPServer):\n"
    "    address_family = socket.AF_INET6 if ':' in HOST else socket.AF_INET\n"
    "S((HOST, PORT), H).serve_forever()\n")

#: (how status is typed, its arguments): the survey and the one-port detail
#: answer one question one way (`deployment.md` § 1.0b, § 1.0c).
STATUS_SPELLINGS = [("one port", True), ("the survey", False)]
#: (the address `serve start --host` names): one that is not the default
#: bind, and the IPv6 loopback, asked bracketed.
ADDRESSES = [("one loopback address", "127.0.0.2"),
             ("the IPv6 loopback", "::1")]


@pytest.mark.parametrize("where, address", ADDRESSES,
                         ids=[r[0] for r in ADDRESSES])
@pytest.mark.parametrize("spelling, names_the_port", STATUS_SPELLINGS,
                         ids=[r[0] for r in STATUS_SPELLINGS])
def test_a_server_started_on_one_address_is_asked_there(
        tmp_path, spelling, names_the_port, where, address):
    """`deployment.md` § 1.0b: *is it answering* is asked "at the address
    the server was started on" -- its pidfile's second line.

    THE SILENT FAILURE: a healthy server started with `--host 127.0.0.2`
    (or one interface) called "answering: NO", exit 4, a false DETECTED
    line in its log, and the advice to restart it."""
    import socket
    try:
        with socket.socket(socket.AF_INET6 if ":" in address
                           else socket.AF_INET) as s:
            s.bind((address, 0))
            port = s.getsockname()[1]
    except OSError:
        pytest.skip(f"this machine cannot bind {address}")
    sup = _spawn_supervisor(
        tmp_path, port, host=address,
        child=_ANSWERS_ON.replace("HOST", repr(address))
                         .replace("PORT", str(port)))
    try:
        def up():
            try:
                with socket.create_connection((address, port), 1):
                    return True
            except OSError:
                return False
        assert _wait_for(up), "the child never listened"
        env = dict(os.environ, HOME=str(tmp_path),
                   XDG_STATE_HOME=str(tmp_path / "state"),
                   XDG_RUNTIME_DIR=str(tmp_path / "runtime"))
        r = subprocess.run(
            [sys.executable, "-m", "molbuilder", "serve", "status",
             *(["--port", str(port)] if names_the_port else [])],
            env=env, cwd=_REPO, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "answering: yes" in r.stdout, r.stdout
        log = tmp_path / "state" / "molbuilder" / "logs" / f"serve-{port}.log"
        assert "DETECTED" not in (log.read_text() if log.exists() else "")
    finally:
        sup.terminate()
        sup.wait(timeout=30)


def test_start_on_the_ipv6_loopback_is_not_refused(tmp_path, monkeypatch):
    """`::1` is a loopback bind the server makes (`serve foreground --host
    ::1` serves on it), so `serve start`'s bind check binds it as the
    server will -- IPv6 -- and goes on to detach.

    THE SILENT FAILURE: "cannot bind ::1 ... the address itself is the
    problem", for an address this machine has."""
    import socket
    try:
        with socket.socket(socket.AF_INET6) as s:
            s.bind(("::1", 0))
            port = s.getsockname()[1]
    except OSError:
        pytest.skip("this machine has no IPv6 loopback to bind")

    # THE DETACH IS FAKED, the one fake here, for a named need: `serve start`
    # would fork this test's own process into a supervisor.  The bind check
    # is the outcome, and the detach comes after it (`testing.md` § 3b).
    class Detached(Exception):
        pass

    def detach():
        raise Detached
    from click.testing import CliRunner
    from molbuilder.cli import cli
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(sd, "daemonize", detach)
    r = CliRunner().invoke(cli, ["serve", "start", "--host", "::1",
                                 "--port", str(port)])
    assert isinstance(r.exception, Detached), r.output


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
    monkeypatch.setattr(sd, "pid_state",
                        # `**kw` because the real signature takes a
                        # `marker` -- `serve` for the supervisor,
                        # `_shepherd` for the notebook -- and a stub
                        # narrower than the contract breaks the day a
                        # caller uses the rest of it.
                        lambda pid, **kw: "ours")
    r = CliRunner().invoke(cli, ["serve", "start", "--port", "8123"])
    assert r.exit_code != 0
    assert "already running" in r.output


@pytest.mark.skipif(os.geteuid() == 0, reason="root makes any directory")
def test_start_refuses_a_pidfile_directory_it_cannot_make(tmp_path,
                                                          monkeypatch):
    """`configuration.md` § 2.1d: an ``XDG_RUNTIME_DIR`` naming a directory
    that cannot be made here -- on a node reached through a scheduler, the
    login node's session directory -- is refused by `serve start` before it
    detaches, naming the variable and the way out.  Without it the terminal
    said *starting in the background* and the detached supervisor died on
    its pidfile, unseen (measured on Sol, 2026-10-17).

    MUTATION THIS MUST FAIL AGAINST: the directory left to the supervisor,
    made after ``daemonize()``."""
    import socket
    from click.testing import CliRunner
    from molbuilder.cli import cli
    shut = tmp_path / "run-user"
    shut.mkdir(mode=0o500)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(shut / "961166"))
    monkeypatch.setattr(sd, "daemonize",
                        lambda: pytest.fail("serve start detached"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = CliRunner().invoke(cli, ["serve", "start", "--port", str(port)])
    assert r.exit_code == 1, r.output
    assert f"XDG_RUNTIME_DIR names {shut / '961166'}" in r.output, r.output
    assert "`unset XDG_RUNTIME_DIR`" in r.output, r.output


def test_status_writes_its_detection_into_the_log(tmp_path, monkeypatch):
    """The log is the record (deployment.md 1.0c): a `serve status` that
    finds the server up-but-not-answering must append that detection to
    the daemon's log, not only print it to whoever asked."""
    from click.testing import CliRunner
    from molbuilder.cli import cli
    # In-process, so the roots are monkeypatched rather than passed in a child
    # environment.  The state directory decides where the log goes
    # (`configuration.md` § 2.1d).
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(sd, "read_pid", lambda port: os.getpid())
    monkeypatch.setattr(sd, "pid_state",
                        # `**kw` because the real signature takes a
                        # `marker` -- `serve` for the supervisor,
                        # `_shepherd` for the notebook -- and a stub
                        # narrower than the contract breaks the day a
                        # caller uses the rest of it.
                        lambda pid, **kw: "ours")
    # port 9329: nothing listens -- the health probe fails fast
    r = CliRunner().invoke(cli, ["serve", "status", "--port", "9329"])
    assert r.exit_code == 4
    log = tmp_path / "state" / "molbuilder" / "logs" / "serve-9329.log"
    assert log.exists(), "the detection never reached the log"
    text = log.read_text()
    assert "[serve-status]" in text and "DETECTED" in text
    assert "not" in text or "gave nothing" in text
