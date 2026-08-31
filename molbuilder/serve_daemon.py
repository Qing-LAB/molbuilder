"""The background half of ``molbuilder serve`` — daemon, pidfile, log roll.

Contract: `docs/ops/deployment.md` § 1.0a–1.0c.

**Imports nothing of the application**, for the same reason
`reload_protocol.py` doesn't: this code supervises the server, and a
supervisor that imports the code it restarts dies with it.  Stdlib only.

Three design facts, each user-stated (2026-08-28):

* **the log is capped and rotated** — gzip the full one, keep at most N
  archives, delete the oldest; a long-lived server cannot fill the disk;
* **per-user by construction** — pidfile and logs live under the caller's
  own home, and every verb VERIFIES the pid (alive, ours, actually a
  molbuilder serve) before signalling, so a stale file whose pid was
  recycled is reported stale, never signalled.  Cross-user signalling is
  already impossible at the kernel (EPERM); the checks make the refusal
  honest rather than mysterious;
* **a question verb answers two questions** — `status` reports *process
  up* and *answering /api/health* separately, because the 2026-08-28
  wedge was exactly a server that was up and not answering.
"""
from __future__ import annotations

import gzip
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .reload_protocol import RELOAD_EXIT_CODE, SUPERVISED_ENV


# --------------------------------------------------------------------- #
#  paths — functions, never module constants (the F4 lesson: a path      #
#  that depends on the environment is a question, asked when asked)      #
# --------------------------------------------------------------------- #

def run_dir() -> Path:
    """Where the supervisor's pidfile goes.

    Asked of ``runtime_config`` rather than computed, so the ``paths`` block
    and the XDG runtime directory both reach it (`configuration.md` § 2.1d).
    Imported inside the function to keep this module's import list what it is
    -- a supervisor that pulls the config reader in at import time pays for it
    on every start, and this is the one path it needs before doing anything.
    """
    from .runtime_config import run_dir as _resolved
    return _resolved()


def log_dir() -> Path:
    from .runtime_config import logs_dir as _resolved
    return _resolved()


def pid_path(port: int) -> Path:
    return run_dir() / f"serve-{port}.pid"


def log_path(port: int) -> Path:
    return log_dir() / f"serve-{port}.log"


def stacks_path(port: int) -> Path:
    return log_dir() / f"serve-{port}.stacks.log"


# --------------------------------------------------------------------- #
#  the log roll                                                          #
# --------------------------------------------------------------------- #

class LogRoll:
    """An append-only log with a size cap: on overflow the current file is
    gzipped to ``<name>.1.gz`` (older archives shift up) and at most
    ``keep`` archives survive — the oldest is deleted.

    Bytes in, because it swallows a *process's* output verbatim; it never
    parses, reorders or drops a byte that fits.
    """

    def __init__(self, path: Path, *, max_bytes: int, keep: int) -> None:
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.keep = int(keep)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "ab")

    def write(self, data: bytes) -> None:
        if not data:
            return
        self._fh.write(data)
        self._fh.flush()
        if self._fh.tell() >= self.max_bytes:
            self._rotate()

    def _rotate(self) -> None:
        self._fh.close()
        # shift .{i}.gz -> .{i+1}.gz from the oldest down, dropping past keep
        for i in range(self.keep, 0, -1):
            src = self.path.with_name(self.path.name + f".{i}.gz")
            if not src.exists():
                continue
            if i >= self.keep:
                src.unlink()
            else:
                src.rename(self.path.with_name(self.path.name + f".{i+1}.gz"))
        if self.keep > 0:
            dst = self.path.with_name(self.path.name + ".1.gz")
            with open(self.path, "rb") as fin, gzip.open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        self._fh = open(self.path, "wb")        # truncate and continue

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


# --------------------------------------------------------------------- #
#  pid verification — never signal what you have not identified          #
# --------------------------------------------------------------------- #

def read_pid(port: int) -> Optional[int]:
    try:
        return int(pid_path(port).read_text().strip())
    except (OSError, ValueError):
        return None


def pid_state(pid: Optional[int]) -> str:
    """``"ours"`` | ``"foreign"`` | ``"dead"`` — what the pid actually is.

    ``foreign`` covers both *someone else's process* and *a recycled pid
    now running something that is not a molbuilder serve* — either way it
    is nothing this module may signal.
    """
    if pid is None:
        return "dead"
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return "dead"
    try:
        if proc.stat().st_uid != os.getuid():
            return "foreign"
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return "foreign"
    if b"molbuilder" not in cmdline or b"serve" not in cmdline:
        return "foreign"
    return "ours"


# --------------------------------------------------------------------- #
#  respawn policy — pure, so the 2026-08-28 repairs are testable         #
# --------------------------------------------------------------------- #

#: Two crashes inside this window and the supervisor stops trying: a
#: server that dies on arrival is a config problem, and respawning it is
#: a tight loop wearing a recovery's clothes.
FLAP_WINDOW_S = 30.0


def child_exit_action(code: int) -> str:
    """What the supervisor does when the child exits with ``code``.

    * the reload sentinel — **respawn**: the Reload button / `restart`;
    * killed by a signal (negative) — **respawn**: the O2 repair.  A hung
      child that somebody killed by hand must come back; before this, the
      supervisor read the kill as *"not a reload"* and quit, taking the
      site down exactly when recovery was needed (2026-08-28);
    * anything else — **exit**: a clean nonzero is a server that cannot
      start, and Ctrl-C's 130 is a person saying stop.
    """
    if code == RELOAD_EXIT_CODE:
        return "respawn"
    if code < 0:
        return "respawn"
    return "exit"


def flapping(crash_times: List[float], now: float) -> bool:
    recent = [t for t in crash_times if now - t <= FLAP_WINDOW_S]
    return len(recent) >= 2


# --------------------------------------------------------------------- #
#  the daemon supervisor                                                 #
# --------------------------------------------------------------------- #

def daemonize() -> None:
    """Classic double fork + setsid.  **The working directory is kept** —
    the projects tree resolves from it, and a daemon that silently
    chdir'd to ``/`` would serve an empty sidebar."""
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)


def _note(roll: "LogRoll", msg: str) -> None:
    """A daemon EVENT line, stamped: the log is the record of concerns,
    detections and respawns (user ruling 2026-08-28) -- and an event
    without a time is half a record.  Child output is pumped verbatim
    elsewhere; only the daemon's own lines come through here."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    roll.write(f"[serve-daemon] {stamp} {msg}\n".encode())


def supervise(port: int, child_argv: List[str], *,
              log_max_bytes: int, log_keep: int,
              max_restarts: Optional[int] = None) -> int:
    """The daemon's main loop: run the child, pump its output through the
    roll, apply :func:`child_exit_action` when it exits.

    ``SIGHUP`` recycles the child (the `restart` verb); ``SIGTERM`` takes
    the child down and exits cleanly, removing the pidfile.
    ``max_restarts`` exists for bounded tests; production passes None.
    """
    roll = LogRoll(log_path(port), max_bytes=log_max_bytes, keep=log_keep)
    run_dir().mkdir(parents=True, exist_ok=True)
    pid_path(port).write_text(f"{os.getpid()}\n")

    state = {"child": None, "hup": False, "term": False}

    def _on_hup(signum, frame):
        state["hup"] = True
        if state["child"] is not None:
            state["child"].terminate()

    def _on_term(signum, frame):
        state["term"] = True
        if state["child"] is not None:
            state["child"].terminate()

    signal.signal(signal.SIGHUP, _on_hup)
    signal.signal(signal.SIGTERM, _on_term)

    env = dict(os.environ)
    env[SUPERVISED_ENV] = "1"
    crashes: List[float] = []
    restarts = 0
    code = 0
    try:
        while True:
            _note(roll, f"starting child: {' '.join(child_argv)}")
            child = subprocess.Popen(child_argv, env=env,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
            state["child"] = child
            # the pump: the child's every byte, through the roll.  Runs in
            # THIS thread -- the daemon has nothing else to do -- and ends
            # when the child closes its output, i.e. exits.
            assert child.stdout is not None
            for chunk in iter(lambda: child.stdout.read(8192), b""):
                roll.write(chunk)
            code = child.wait()
            state["child"] = None

            if state["term"]:
                _note(roll, "stopped on request")
                return 0
            if state["hup"]:
                state["hup"] = False
                _note(roll, "restart requested -- starting a fresh server")
            else:
                action = child_exit_action(code)
                if action == "exit":
                    _note(roll, f"child exited {code}; not a case to respawn")
                    return code
                if code < 0:
                    now = time.monotonic()
                    crashes.append(now)
                    if flapping(crashes, now):
                        _note(roll, "two crashes within 30s -- giving up "
                                    "rather than flapping")
                        return 1
                    _note(roll, f"child died by signal {-code}; "
                                f"respawning (the hung-child repair)")
                else:
                    _note(roll, "reload requested -- starting a fresh server")
            restarts += 1
            if max_restarts is not None and restarts > max_restarts:
                return code
    finally:
        try:
            pid_path(port).unlink()
        except OSError:
            pass
        roll.close()


# --------------------------------------------------------------------- #
#  the acting verbs' shared halves                                       #
# --------------------------------------------------------------------- #

def signal_supervisor(port: int, sig: int) -> Tuple[bool, str]:
    """Verify, then signal.  The verification IS the per-user honesty:
    the kernel would refuse a foreign pid anyway (EPERM), but *"that pid
    is not yours / not a molbuilder serve"* beats *"permission denied"*.
    """
    pid = read_pid(port)
    state = pid_state(pid)
    if state == "dead":
        if pid is not None:
            try:
                pid_path(port).unlink()      # stale file: say so, clean up
            except OSError:
                pass
            return False, (f"stale pidfile: pid {pid} is gone "
                           f"(removed {pid_path(port)})")
        return False, f"not running (no pidfile at {pid_path(port)})"
    if state == "foreign":
        return False, (f"refusing: pid {pid} is not your molbuilder serve "
                       f"-- the pidfile is stale and the pid was recycled")
    os.kill(pid, sig)                        # state == "ours"
    return True, f"signalled pid {pid}"
