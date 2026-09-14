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
import time
from pathlib import Path
from typing import List, Optional, Tuple

# `config_dir` is the bootstrap that owns these paths and the one creator of a
# private directory; it is stdlib-only and answers before any config is read,
# which is what lets the supervisor -- L1, importing nothing of the application
# it restarts -- reach it.  Four one-line pass-throughs (`run_dir`, `pid_path`,
# `log_path`, `stacks_path`) and a private `_mkdir_private` stood between this
# module and those doors until 2026-09-13 (K-D3, I8's shape).
from .config_dir import (ensure_private_dir, jupyter_log, jupyter_pidfile,
                          runtime_dir, serve_log, serve_pidfile)
from .reload_protocol import RELOAD_EXIT_CODE, SUPERVISED_ENV


# --------------------------------------------------------------------- #
#  paths — functions, never module constants (the F4 lesson: a path      #
#  that depends on the environment is a question, asked when asked)      #
# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
#  the log roll                                                          #
# --------------------------------------------------------------------- #

def open_private(path: Path, mode: str):
    """``open(path, mode)`` for a file that must be 0600 from its first byte.

    PUBLIC, because `configuration.md` § 2.3's writer table already names it as
    the door for an appended log -- the one shape temp-and-rename cannot serve.
    It was private while two other surfaces appended to the same logs with a
    bare `open(..., "a")`, landing 0664 on a file that carries a provider's
    `client_secret` (A2, I5): a door nobody outside the module can reach is a
    door the neighbours route around.

    The mode rides the descriptor via ``os.open``, so there is no window where
    the file exists readable with content in it -- the same discipline
    `auth_setup.write_secret_file` uses, for the same reason.

    ``mode`` is `open`'s: ``"a"`` / ``"w"`` give a text file (UTF-8),
    ``"ab"`` / ``"wb"`` bytes.  It returned bytes whatever was asked until
    2026-09-13, so a caller writing text had to know to say ``"a"`` and
    then ``.encode()`` -- and a `logging` handler, which writes text, could
    not use it at all.
    """
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if "a" in mode else os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    try:
        if (os.fstat(fd).st_mode & 0o777) != 0o600:
            os.fchmod(fd, 0o600)          # pre-existing loose file
    except OSError:
        pass
    if "b" in mode:
        return os.fdopen(fd, mode)
    return os.fdopen(fd, mode, encoding="utf-8")



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
        # Mode on the descriptor at CREATE time, not a chmod afterwards: a
        # chmod races the first write, and `configuration.md` 2.1b requires
        # the mode to be right "before there is anything to read".
        # ``tighten=True`` here and nowhere in the seeding path: the log
        # directory is ours, we are about to write a log into it that carries a
        # provider's ``client_secret``, and it was measured at 0775.  A
        # directory the OPERATOR made -- the config root on a cluster, often
        # pointed at scratch -- is not ours to re-mode; `envs doctor` reports
        # that one instead.
        ensure_private_dir(self.path.parent, tighten=True)
        self._fh = open_private(self.path, "ab")

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
            # WE CLOSE WHAT WE OPENED.  `gzip.GzipFile` borrows a `fileobj`
            # and documents that it never closes one -- it closes only a file
            # it opened itself.  Handing it ours and walking away left the
            # handle with no owner, waiting on the garbage collector, which is
            # CPython's refcounting rather than a promise.
            # (A1 also claimed the archive stayed unflushed until then.  That
            # part is not reproducible on CPython 3.14 -- but the ownership was
            # wrong either way, which is why this changed.)
            with open(self.path, "rb") as fin, \
                    open_private(dst, "wb") as raw, \
                    gzip.GzipFile(fileobj=raw, mode="wb") as fout:
                shutil.copyfileobj(fin, fout)
        self._fh = open_private(self.path, "wb")   # truncate and continue

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
        return int(serve_pidfile(port).read_text().strip())
    except (OSError, ValueError):
        return None


def pid_state(pid: Optional[int], *, marker: bytes = b"serve") -> str:
    """``"ours"`` | ``"foreign"`` | ``"dead"`` — what the pid actually is.

    ``foreign`` covers both *someone else's process* and *a recycled pid
    now running something that is not ours* — either way it is nothing this
    module may signal.

    ``marker`` is the second word the command line must carry, beside
    ``molbuilder``: ``serve`` for the supervisor, ``_shepherd`` for the
    notebook server it holds (`molbuilder.jupyter`).  One verification, two
    pidfiles -- a second copy of "alive, yours, actually ours" is how one of
    them comes to be laxer than the other.
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
    if b"molbuilder" not in cmdline or marker not in cmdline:
        return "foreign"
    return "ours"


def stop_by_pidfile(path: Path, *, marker: bytes, grace_s: float = 5.0,
                    ) -> Tuple[bool, str]:
    """Verify, SIGTERM, wait, and take the group if it did not go.

    The general form of what `signal_supervisor` does for the serve pidfile,
    used for the notebook's (`jupyter.md` § 3.2).  The group matters there:
    the process named by that file leads its own session, so its group is the
    notebook server AND every `ipykernel` under it -- which are what hold
    memory and GPUs, and what stopping the server alone routinely leaves.

    A stale file is cleaned up and REPORTED; a recycled pid is never
    signalled.
    """
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        pid = None
    state = pid_state(pid, marker=marker)
    if state == "dead":
        if pid is not None:
            try:
                path.unlink()
            except OSError:
                pass
            return False, f"stale pidfile: pid {pid} is gone (removed)"
        return False, "not running"
    if state == "foreign":
        return False, (f"refusing: pid {pid} is not ours -- the pidfile is "
                       f"stale and the pid was recycled")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if pid_state(pid, marker=marker) != "ours":
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass
        return True, f"stopped (pid {pid}, forced)"
    try:
        path.unlink()
    except OSError:
        pass
    return True, f"stopped (pid {pid})"


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
              jupyter_argv: Optional[List[str]] = None,
              max_restarts: Optional[int] = None) -> int:
    """The daemon's main loop: run the child, pump its output through the
    roll, apply :func:`child_exit_action` when it exits.

    ``SIGHUP`` recycles the child (the `restart` verb); ``SIGTERM`` takes
    the child down and exits cleanly, removing the pidfile.
    ``max_restarts`` exists for bounded tests; production passes None.

    **``SIGUSR1`` / ``SIGUSR2`` start and stop the NOTEBOOK server**
    (`docs/web/jupyter.md` § 3.4).  It is parented HERE, one level above the
    server child, and the reason is the reload: this loop respawns that child
    on `RELOAD_EXIT_CODE`, so a Jupyter parented to it would be killed by
    every unrelated code change -- destroying notebook state for a reason that
    had nothing to do with notebooks.  Parented here it survives a reload and
    goes when the daemon goes.

    That is also why the web tab cannot start it directly: the tab runs IN the
    server child.  It signals this process instead, and so does
    ``molbuilder jupyter start``.

    ``jupyter_argv`` is a list of strings, handed over exactly as
    ``child_argv`` is -- **this module imports nothing of the application**,
    which is the property that lets a child failing to import leave the parent
    alive to be fixed.  ``None`` means this molbuilder was started without a
    notebook command line, and the signals are then no-ops that say so.
    """
    roll = LogRoll(serve_log(port), max_bytes=log_max_bytes, keep=log_keep)
    # THE XDG DEFAULTS, NOT `molbuilder.json`'s `paths` block -- and there is
    # no config key for these, deliberately.  The supervisor must be able to
    # write its log BEFORE any config is read, including when reading it is
    # what failed: a supervisor that had to read `molbuilder.json` to find out
    # where to report a malformed `molbuilder.json` would have nowhere to
    # report it, which is the one log nobody can afford to lose
    # (`configuration.md` § 2.1d).  `$XDG_STATE_HOME` and `$XDG_RUNTIME_DIR`
    # move these directories and answer before any config is read; a
    # `paths.logs` / `paths.run` in `molbuilder.json` is REFUSED by
    # `runtime_config._read_paths`.
    #
    # 0700, like every other directory this program makes: when
    # $XDG_RUNTIME_DIR is absent this falls back INSIDE the state root, where
    # the default umask gives 0775.
    ensure_private_dir(runtime_dir(), tighten=True)
    serve_pidfile(port).write_text(f"{os.getpid()}\n")

    # LAYER 3 (docs/web/jupyter.md § 3.3): a notebook server that outlived the
    # molbuilder that started it -- a machine crash, or a survivor re-parented
    # before the signal landed -- is stopped now, before this one runs.
    #
    # DONE HERE WITH THIS MODULE'S OWN HELPERS, not by calling
    # `molbuilder.jupyter`: that module reaches the env door and the recipe
    # registry, and this one imports nothing of the application -- which is
    # the property that lets a child failing to import leave the supervisor
    # alive.  All reconciliation needs is a pidfile path (`config_dir`, the
    # stdlib-only bootstrap) and "verify, then signal", which is the job this
    # module already does for its own pidfile.
    if jupyter_argv is not None:
        nb_pidfile = jupyter_pidfile(port)
        if nb_pidfile.exists():
            ok, said = stop_by_pidfile(nb_pidfile, marker=b"_shepherd")
            _note(roll, f"notebook left from a previous run: {said}")

    state: dict = {"child": None, "hup": False, "term": False,
                   "jupyter": None}

    def _start_jupyter() -> None:
        """Launch the shepherd, in ITS OWN SESSION.

        The new session makes the shepherd a process-group leader, so
        everything the door launches below it -- the manager's ``run``,
        ``jupyter-server``, every ``ipykernel`` -- lands in one group the
        shepherd can take down by signalling itself.  It stays a CHILD of
        this process, which is what `PR_SET_PDEATHSIG` keys on.
        """
        if jupyter_argv is None:
            _note(roll, "notebook: not configured for this server")
            return
        live = state["jupyter"]
        if live is not None and live.poll() is None:
            _note(roll, f"notebook: already running (pid {live.pid})")
            return
        # ITS OUTPUT GOES TO ITS OWN LOG, opened here.  The shepherd's
        # stderr is otherwise this process's, and this process is daemonised
        # -- so a notebook that refuses to start (no env manager, env not
        # installed) said why into /dev/null.  Measured on the dev server
        # 2026-09-14: the supervisor logged "started", the shepherd exited,
        # and there was nothing anywhere to say what was wrong.
        #
        # Opened per start and handed over, so the shepherd owns it after the
        # fork and this loop does no pumping -- the log is the record, not a
        # stream this process has to service.
        try:
            log = jupyter_log(port)
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "ab", buffering=0)
        except OSError as exc:
            _note(roll, f"notebook: could not open its log -- {exc}")
            return
        try:
            state["jupyter"] = subprocess.Popen(
                jupyter_argv, start_new_session=True,
                stdout=fh, stderr=subprocess.STDOUT)
            _note(roll, f"notebook: started (pid {state['jupyter'].pid}); "
                        f"it logs to {log}")
        except OSError as exc:
            _note(roll, f"notebook: could not start -- {exc}")
        finally:
            fh.close()

    def _stop_jupyter() -> None:
        """SIGTERM the shepherd; its own handler takes the group with it."""
        live = state["jupyter"]
        if live is None or live.poll() is not None:
            state["jupyter"] = None
            return
        try:
            live.terminate()
            live.wait(timeout=10)
            _note(roll, f"notebook: stopped (pid {live.pid})")
        except subprocess.TimeoutExpired:
            # The shepherd did not go.  Its GROUP does -- and since it leads
            # its own session, that is the notebook and every kernel.
            try:
                os.killpg(os.getpgid(live.pid), signal.SIGKILL)
            except OSError:
                pass
            _note(roll, f"notebook: stopped (pid {live.pid}, forced)")
        except OSError as exc:
            _note(roll, f"notebook: could not stop -- {exc}")
        state["jupyter"] = None

    def _on_usr1(signum, frame):
        _start_jupyter()

    def _on_usr2(signum, frame):
        _stop_jupyter()

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
    signal.signal(signal.SIGUSR1, _on_usr1)
    signal.signal(signal.SIGUSR2, _on_usr2)

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
        # THE NOTEBOOK GOES WITH THE DAEMON.  Layers 1 and 2 would get it
        # anyway (the shepherd's PDEATHSIG fires when this process exits, and
        # takes its group), but asking politely first lets a kernel mid-cell
        # unwind rather than being killed by the kernel.
        _stop_jupyter()
        try:
            serve_pidfile(port).unlink()
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
                serve_pidfile(port).unlink()      # stale file: say so, clean up
            except OSError:
                pass
            return False, (f"stale pidfile: pid {pid} is gone "
                           f"(removed {serve_pidfile(port)})")
        return False, f"not running (no pidfile at {serve_pidfile(port)})"
    if state == "foreign":
        return False, (f"refusing: pid {pid} is not your molbuilder serve "
                       f"-- the pidfile is stale and the pid was recycled")
    os.kill(pid, sig)                        # state == "ours"
    return True, f"signalled pid {pid}"
