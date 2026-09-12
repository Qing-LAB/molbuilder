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
    """Where the supervisor's pidfile goes -- ``config_dir.runtime_dir()``.

    **The XDG default, NOT `molbuilder.json`'s `paths` block**, and the
    layering says why: this module is L1 and `runtime_config` is L2, so
    reading the config here would invert the dependency.  That constraint and
    the bootstrap rule are the same fact seen twice.

    The supervisor must be able to write its log BEFORE any config is read --
    including when reading it is what failed.  A supervisor that consulted
    `paths.logs` to find out where to report a malformed `molbuilder.json`
    would have nowhere to report it, which is the one log nobody can afford to
    lose (`configuration.md` § 2.1d).

    So `paths.logs` moves molbuilder's application logs; the supervisor's own
    follow the state directory.  Both move together when the person moves
    `$XDG_STATE_HOME`, which is the case the split is for.
    """
    from .config_dir import runtime_dir
    return runtime_dir()


def log_dir() -> Path:
    """The supervisor's log directory -- ``config_dir.state_dir()/logs``.

    Same reasoning as :func:`run_dir`: L1, and writable before config.
    """
    from .config_dir import logs_dir
    return logs_dir()


def pid_path(port: int) -> Path:
    from .config_dir import serve_pidfile
    return serve_pidfile(port)


def log_path(port: int) -> Path:
    from .config_dir import serve_log
    return serve_log(port)


def stacks_path(port: int) -> Path:
    from .config_dir import serve_stacks_log
    return serve_stacks_log(port)


# --------------------------------------------------------------------- #
#  the log roll                                                          #
# --------------------------------------------------------------------- #

def _mkdir_private(d: Path) -> None:
    """``mkdir -p`` at 0700, and tighten an existing directory.

    ``mode=`` covers the directories this call CREATES; an existing one keeps
    whatever it had, which is how a 0775 logs directory survived.  So the mode
    is asserted afterwards too -- best-effort, because a directory somebody
    else owns is not ours to re-mode and is not a reason to refuse to log.
    """
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        if (d.stat().st_mode & 0o777) != 0o700:
            os.chmod(d, 0o700)
    except OSError:
        pass


def _open_private(path: Path, mode: str):
    """``open(path, mode)`` for a file that must be 0600 from its first byte.

    The mode rides the descriptor via ``os.open``, so there is no window where
    the file exists readable with content in it -- the same discipline
    `auth_setup.write_secret_file` uses, for the same reason.
    """
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if "a" in mode else os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    try:
        if (os.fstat(fd).st_mode & 0o777) != 0o600:
            os.fchmod(fd, 0o600)          # pre-existing loose file
    except OSError:
        pass
    return os.fdopen(fd, "wb")



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
        # 0700 ON THE DIRECTORY, 0600 ON THE FILE -- this log is a SECRET
        # SINK, measured 2026-09-12 at 0664 in a 0775 directory.
        #
        # It swallows the child's stderr verbatim, and what the child writes
        # there on an auth failure is chosen ON PURPOSE to be the thing too
        # sensitive for the browser: `web/auth_providers/oauth.py` routes a
        # provider error to `logging.exception` precisely because the `params`
        # dict "on certain misbehaving providers can include the
        # client_secret", and says "don't risk leaking that into the
        # user-visible response".  The response was protected and this file
        # was not.  A CAS ticket (`cas.py`), an OAuth code, and the `--cert` /
        # `--key` paths in the child argv land here too.
        #
        # Which is the inversion this project already named and fixed one
        # level over, for the run-report records: "the KEY file was always
        # 0600 and the DATA it protects was not, which is the wrong way round
        # on a shared server" (`run-reports.md`).
        #
        # Mode on the descriptor at CREATE time, not a chmod afterwards: a
        # chmod races the first write, and `configuration.md` 2.1b requires
        # the mode to be right "before there is anything to read".
        _mkdir_private(self.path.parent)
        self._fh = _open_private(self.path, "ab")

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
            # The ARCHIVE holds the same bytes, so it gets the same mode.
            with open(self.path, "rb") as fin, \
                    gzip.GzipFile(fileobj=_open_private(dst, "wb"),
                                  mode="wb") as fout:
                shutil.copyfileobj(fin, fout)
        self._fh = _open_private(self.path, "wb")   # truncate and continue

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
