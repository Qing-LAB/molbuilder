"""Background job monitor + notifier hooks (proof-of-concept).

The front end of the job-monitor/watcher + notifier surface
(docs/protocols/slurm-integration.md § 11.0b, item F).  A lightweight,
periodically-waking process that **parses** a running job's artifacts
(the SIESTA ``.out`` + per-run ``.scf-timing.log``), appends a structured
status line to ``<basename>.monitor.log``, and invokes any registered
**notifier hooks** on each tick -- today a log stub that records what
*would* be sent; tomorrow a webhook / email / molwatch push.

Why a Python module (not a bash script): parsing + a clean hook interface
belong in Python.  Run it inside the job's activated env so ``molbuilder``
is importable.  The run-wrapper backgrounds it at low OS priority
(``nice``) so it never competes with the heavy compute ranks on the same
node -- it sleeps ~99.99% of the time and does a few ms of tail-reads per
wake (default 5 min), far below any benchmark's measurement noise.

CLI (added to ``molbuilder`` as ``molbuilder monitor``)::

    nice -n 19 python -m molbuilder monitor \\
        --out job-run0.out --timing job-run0.scf-timing.log \\
        --log job.monitor.log --interval 60 --watch-pid $$ &

Connect a real notifier programmatically::

    from molbuilder import monitor
    monitor.register_notifier(lambda st, ev: my_push(st.as_text()))

or via env ``MB_NOTIFY_URL`` (auto-registers a stdlib webhook POST).

Design notes:
- **stdlib-only hot path** -- no heavy imports in the loop.
- Each tick reads only the **tail** of the ``.out`` (cheap on a large
  file) and the whole (tiny) timing log for the iteration COUNT.
- The authoritative stop signal is ``--watch-pid`` going away (the job
  wrapper's PID); ``.out`` completion markers are a best-effort hint.
- The reliable live per-iteration estimate is ``elapsed / n_iters``
  (running average) -- consistent with the benchmark's ``total/N`` metric
  (§ 11.0); SIESTA's own per-scf time is not trusted.  It is reported
  ONLY while the job is progressing: a stalled job's ``elapsed /
  frozen_n_iters`` keeps inflating with wall time and is meaningless, so
  it is suppressed (§ 11.0c).
- **Quiet when stalled** -- the loop wakes often (default 60 s) but logs
  a ``[STATUS]`` line only when the SCF iteration count / geometry move /
  energy / state actually changed.  A persistent stall emits at most one
  throttled ``[STALL]`` heartbeat per ``--stall-heartbeat`` window
  instead of flushing a (misleading) timing line every wake.
"""

from __future__ import annotations

import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


# --------------------------------------------------------------------- #
#  Parsed status                                                        #
# --------------------------------------------------------------------- #


@dataclass
class JobStatus:
    """One snapshot of a running (or finished) SIESTA job.

    ``per_iter_s`` is the **running average** ``elapsed / n_iters`` -- the
    live, reliable per-iteration estimate (§ 11.0); ``None`` until the
    first SCF iteration appears.
    """
    state: str = "starting"          # starting | running | done | gone
    elapsed_s: float = 0.0
    n_iters: int = 0
    scf_iter: Optional[str] = None   # last iteration number (as printed)
    per_iter_s: Optional[float] = None
    energy: Optional[str] = None     # last reported total energy (as printed)
    done_marker: Optional[str] = None  # the .out completion marker, if seen
    geom_step: Optional[int] = None  # geometry-relaxation move # (CG/FIRE/...)
    progressing: bool = True         # did the iteration count advance since the
                                     # previous tick?  Set by run_monitor, not
                                     # parse_status (which is stateless).  When
                                     # False the per-iter estimate is suppressed
                                     # -- ``elapsed / frozen_n_iters`` only
                                     # inflates and means nothing (§ 11.0c).

    def as_text(self) -> str:
        """One-line human/notifier summary.

        ``avg_per_iter`` is shown ONLY while the job is progressing: a
        stalled job's ``elapsed / n_iters`` keeps growing with wall time
        even though no iteration completed, so reporting it is actively
        misleading (§ 11.0c).
        """
        bits = [f"state={self.state}",
                f"elapsed={self.elapsed_s:.0f}s",
                f"scf_iters={self.n_iters}"]
        if self.geom_step is not None:
            bits.append(f"geom_move={self.geom_step}")
        if self.scf_iter is not None:
            bits.append(f"last_iter={self.scf_iter}")
        if self.per_iter_s is not None and self.progressing:
            bits.append(f"avg_per_iter={self.per_iter_s:.2f}s")
        if self.energy is not None:
            bits.append(f"energy={self.energy}")
        return " ".join(bits)


# Markers that mean "the SIESTA run finished" (best-effort hint; the
# authoritative signal is the watched PID disappearing).
_DONE_MARKERS = (
    "Job completed",
    "End of run",
    "siesta: Final energy",
    ">> End of run:",
)
_SCF_LINE = re.compile(r"^[ \t]*scf:[ \t]*[0-9]")
# Geometry-relaxation move marker.  SIESTA prints e.g.
# ``Begin CG move = 3`` / ``Begin FIRE move = 12`` / ``Begin Broyden
# move = 5`` / ``Begin Z-matrix opt. move = 1`` at each ionic step.  We
# take the highest move number seen in the .out tail as the current
# geometry step (None for single-point runs, which never print it).
_GEOM_LINE = re.compile(r"Begin\b.*\bmove\b\D*([0-9]+)", re.IGNORECASE)


def _tail_bytes(path: Path, nbytes: int = 16384) -> str:
    """Return the last ``nbytes`` of ``path`` as text ('' if absent)."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > nbytes:
                fh.seek(-nbytes, os.SEEK_END)
            data = fh.read()
        return data.decode("utf-8", "replace")
    except OSError:
        return ""


def parse_status(out_path: Path, timing_path: Path,
                 start_epoch: float, now_epoch: float) -> JobStatus:
    """Parse a :class:`JobStatus` from the job artifacts.

    ``n_iters`` (the reliable count) comes from the per-run timing log;
    ``scf_iter`` / ``energy`` from the last ``scf:`` line; ``done_marker``
    from a tail scan of the ``.out``.  Pure reads -- never raises on a
    missing/locked file.
    """
    st = JobStatus(elapsed_s=max(0.0, now_epoch - start_epoch))

    # N (count) + last iteration number from the timing log.
    timing_path = Path(timing_path)
    last_line = ""
    try:
        if timing_path.is_file():
            with timing_path.open("r", encoding="utf-8", errors="replace") as fh:
                n = 0
                for line in fh:
                    if line.strip():
                        n += 1
                        last_line = line
                st.n_iters = n
    except OSError:
        pass
    if last_line:
        parts = last_line.split()
        # timing line: <epoch.ns> <iter#> <scf: ...>
        if len(parts) >= 2:
            st.scf_iter = parts[1]
    if st.n_iters > 0:
        st.per_iter_s = st.elapsed_s / st.n_iters
        st.state = "running"

    # Last energy + geometry step + done marker from the .out tail.
    tail = _tail_bytes(Path(out_path))
    if tail:
        lines = tail.splitlines()
        for line in reversed(lines):
            if _SCF_LINE.match(line):
                f = line.split()
                if len(f) >= 3:
                    st.energy = f[2]
                break
        # Highest geometry-move number in the tail (None if not relaxing).
        for line in reversed(lines):
            gm = _GEOM_LINE.search(line)
            if gm:
                st.geom_step = int(gm.group(1))
                break
        for m in _DONE_MARKERS:
            if m in tail:
                st.done_marker = m
                st.state = "done"
                break
    return st


# --------------------------------------------------------------------- #
#  Notifier hooks                                                       #
# --------------------------------------------------------------------- #

# A notifier is ``fn(status, event)`` where ``event`` is one of
# "start" | "tick" | "finish".  Register as many as you like; they are
# called in registration order and individually guarded (one failing
# hook never breaks the loop or the other hooks).
Notifier = Callable[[JobStatus, str], None]

_NOTIFIERS: List[Notifier] = []


def register_notifier(fn: Notifier) -> None:
    """Add a notifier hook.  This is the connection point for a real
    push (webhook/email/molwatch)."""
    _NOTIFIERS.append(fn)


def clear_notifiers() -> None:
    """Drop all registered notifiers (used by tests)."""
    _NOTIFIERS.clear()


def _fire(status: JobStatus, event: str) -> None:
    for fn in list(_NOTIFIERS):
        try:
            fn(status, event)
        except Exception as exc:                       # noqa: BLE001
            # A notifier must never break monitoring.
            print(f"[monitor] notifier {getattr(fn, '__name__', fn)!r} "
                  f"raised: {exc!r}", flush=True)


def make_webhook_notifier(url: str) -> Notifier:
    """A stdlib webhook notifier: POSTs ``event`` + the status summary to
    ``url`` (form-encoded).  Best-effort, short timeout, never raises out."""
    def _hook(status: JobStatus, event: str) -> None:
        data = urllib.parse.urlencode(
            {"event": event, "text": status.as_text()}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        try:
            urllib.request.urlopen(req, timeout=5).close()
        except Exception:                              # noqa: BLE001
            pass
    _hook.__name__ = "webhook_notifier"
    return _hook


def _install_env_notifiers() -> None:
    """Auto-register notifiers from the environment (``MB_NOTIFY_URL``)."""
    url = os.environ.get("MB_NOTIFY_URL")
    if url:
        register_notifier(make_webhook_notifier(url))


# --------------------------------------------------------------------- #
#  The monitor loop                                                     #
# --------------------------------------------------------------------- #


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True   # not watching a pid
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, not ours
    except OSError:
        return True


def _append(log_path: Path, line: str) -> None:
    try:
        with Path(log_path).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _progressed(curr: JobStatus, prev: JobStatus) -> bool:
    """Did real work advance between two ticks?  True iff the SCF
    iteration count or the geometry-move number went up."""
    if curr.n_iters > prev.n_iters:
        return True
    return (curr.geom_step or 0) > (prev.geom_step or 0)


def run_monitor(out: Path, timing: Path, log: Path, *,
                interval: float = 60.0,
                watch_pid: int = 0,
                start_epoch: Optional[float] = None,
                max_ticks: Optional[int] = None,
                stall_heartbeat_s: float = 600.0,
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.time) -> JobStatus:
    """Periodically parse + log + notify until the watched job ends.

    Returns the final :class:`JobStatus`.  ``max_ticks`` bounds the loop
    (tests pass a small value); ``sleep``/``clock`` are injectable for
    deterministic testing.

    Wakes every ``interval`` seconds (short, so progress is reported
    promptly) but is QUIET when nothing changed (§ 11.0c): a ``[STATUS]``
    line + ``tick`` notification fire only when the job actually advanced
    (SCF iteration or geometry move) or its energy/state changed.  A
    long stall emits at most one throttled ``[STALL]`` heartbeat every
    ``stall_heartbeat_s`` seconds -- with NO per-iteration estimate,
    because ``elapsed / frozen_n_iters`` only inflates and is meaningless
    when no iteration has completed.  This keeps a stalled job from
    flushing a misleading timing line on every wake.  Set
    ``stall_heartbeat_s <= 0`` to silence the stall heartbeat entirely
    (the log then goes quiet until the job next progresses or ends).
    """
    out, timing, log = Path(out), Path(timing), Path(log)
    start = clock() if start_epoch is None else start_epoch
    _install_env_notifiers()

    st0 = parse_status(out, timing, start, clock())
    _append(log, f"[{_iso(clock())}] [MONITOR] start "
                 f"(interval={interval:.0f}s watch_pid={watch_pid}) "
                 f"{st0.as_text()}")
    _fire(st0, "start")

    prev = st0
    last_emit = start          # wall time of the last [STATUS]/[STALL] line
    ticks = 0
    while True:
        sleep(interval)
        ticks += 1
        now = clock()
        alive = _pid_alive(watch_pid)
        st = parse_status(out, timing, start, now)
        if not alive:
            st.state = "gone" if st.state != "done" else "done"

        st.progressing = _progressed(st, prev)

        if not alive or st.state == "done":
            # Terminal: the cumulative ``elapsed / n_iters`` over the whole
            # run IS a valid final average, so report it (force-show even
            # though this last tick added no new iteration).
            st.progressing = True
            _append(log, f"[{_iso(now)}] [STATUS] {st.as_text()}")
            _append(log, f"[{_iso(now)}] [MONITOR] job ended "
                         f"(watch_pid alive={alive}, marker="
                         f"{st.done_marker or '-'}); final notify + exit")
            _fire(st, "finish")
            return st

        if not st.progressing:
            # (1) Never report the wrong estimate while LIVE + stalled:
            # elapsed / frozen-iters only inflates with wall time.
            st.per_iter_s = None

        changed = (st.progressing
                   or st.energy != prev.energy
                   or st.state != prev.state)
        if changed:
            _append(log, f"[{_iso(now)}] [STATUS] {st.as_text()}")
            _fire(st, "tick")
            last_emit = now
        elif stall_heartbeat_s > 0 and now - last_emit >= stall_heartbeat_s:
            # (2) Throttled liveness ping only -- no iteration-time message.
            _append(log, f"[{_iso(now)}] [STALL] no SCF/geometry progress "
                         f"for {now - last_emit:.0f}s; state={st.state} "
                         f"scf_iters={st.n_iters} "
                         f"(alive={alive})")
            _fire(st, "tick")
            last_emit = now

        prev = st
        if max_ticks is not None and ticks >= max_ticks:
            return st


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch))


# --------------------------------------------------------------------- #
#  Default log notifier (the PoC stub)                                  #
# --------------------------------------------------------------------- #


def make_log_notifier(log: Path) -> Notifier:
    """The PoC notifier: records into the monitor log what *would* be
    pushed.  Swap/extend with a real channel via :func:`register_notifier`."""
    def _hook(status: JobStatus, event: str) -> None:
        if event in ("start", "finish"):
            _append(Path(log),
                    f"[{_iso(time.time())}] [NOTIFY] (stub) {event}: "
                    f"{status.as_text()}")
    _hook.__name__ = "log_notifier"
    return _hook


# --------------------------------------------------------------------- #
#  Standalone entry (stdlib only -- runs WITHOUT the molbuilder package) #
# --------------------------------------------------------------------- #
#
# CRITICAL: this module imports ONLY the stdlib (os/re/time/urllib/
# dataclasses/pathlib/typing) -- no molbuilder, no numpy.  That is what
# lets the run-wrapper SHIP this file as ``mb_monitor.py`` next to the
# job and run it with the JOB's own python (e.g. the minimal
# ``molbuilder-siesta-gpu`` env, which has no numpy/molbuilder), from the
# working directory, with no install and no repo on PATH.  Keep it
# stdlib-only.


def main(argv=None) -> int:
    """argparse entry for the SHIPPED standalone ``mb_monitor.py``.

    Mirrors the ``molbuilder monitor`` click command but with zero
    third-party deps so it runs in any python.  Self-lowers priority via
    ``os.nice`` and installs the default PoC log notifier.
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="mb_monitor",
        description="molbuilder background job-monitor + notifier hooks "
                    "(self-contained; § 11.0b)")
    p.add_argument("--out", required=True, help="SIESTA .out to watch")
    p.add_argument("--timing", required=True,
                   help="per-run .scf-timing.log (iteration COUNT)")
    p.add_argument("--log", required=True,
                   help="append status lines here (<basename>.monitor.log)")
    p.add_argument("--interval", type=float, default=60.0,
                   help="seconds between wakes (default 60); a stalled job "
                        "stays quiet -- see --stall-heartbeat")
    p.add_argument("--stall-heartbeat", type=float, default=600.0,
                   dest="stall_heartbeat_s",
                   help="when the job is making no SCF/geometry progress, "
                        "emit at most one liveness ping this often "
                        "(seconds, default 600); no per-iter timing is "
                        "printed while stalled.  Use 0 to silence the "
                        "stall heartbeat entirely")
    p.add_argument("--watch-pid", type=int, default=0, dest="watch_pid",
                   help="stop when this PID disappears; 0 = until done")
    p.add_argument("--nice", type=int, default=19, dest="nice_level",
                   help="self-lower OS priority by this much (default 19)")
    a = p.parse_args(argv)
    try:
        os.nice(max(0, a.nice_level))
    except (OSError, AttributeError):
        pass
    register_notifier(make_log_notifier(a.log))
    run_monitor(a.out, a.timing, a.log,
                interval=a.interval, watch_pid=a.watch_pid,
                stall_heartbeat_s=a.stall_heartbeat_s)
    return 0


__all__ = [
    "JobStatus",
    "parse_status",
    "run_monitor",
    "register_notifier",
    "clear_notifiers",
    "make_webhook_notifier",
    "make_log_notifier",
    "Notifier",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
