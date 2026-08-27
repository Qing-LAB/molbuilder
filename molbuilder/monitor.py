"""Background job monitor + notifier hooks.

The front end of the job-monitor/watcher + notifier surface
(`execution/running-a-job.md` § 4.1, `execution/job-contracts.md` — the
monitor's section).  A lightweight, periodically-waking process that
**parses** a running job's artifacts (the SIESTA ``.out`` + per-run
``.scf-timing.log``), appends a structured status line to
``<basename>.monitor.log``, samples utilisation into
``<basename>.util.csv``, and notifies -- rarely, and only when the
calculation's own policy says to.

**It never runs inside molbuilder.**  A verbatim copy of this file ships
beside the job as ``mb_monitor.py`` and runs with the JOB's own python from
the working directory: molbuilder is not installed on a compute node, and
the `molbuilder-siesta` env has no python of its own at all.  That is why
this module is stdlib-only, and it is a constraint rather than a
preference.  *(This paragraph said "run it inside the job's activated env
so molbuilder is importable" until 2026-08-26 -- the exact opposite of the
arrangement the wrapper has always used.)*

The run-wrapper backgrounds it at low OS priority (``nice -n 19``) so it
never competes with the compute ranks on the same node: it sleeps almost
all of the time and does a few ms of tail-reads per wake (default 10 s),
far below any benchmark's measurement noise.

**Looking and telling are separate.**  It looks often, because
``util.csv`` is the diagnostic record.  It tells rarely, because a message
per wake is a message every ten seconds for the length of a run -- which
is what a notifier registered here received until 2026-08-26.  When to
tell is the calculation's, carried from `task.json`'s ``notify`` block:
``--notify-on-scf``, ``--notify-every-hours``, and a run ending, always.

CLI (also available as ``molbuilder monitor``)::

    nice -n 19 python mb_monitor.py \\
        --out job-run0.out --timing job-run0.scf-timing.log \\
        --log job.monitor.log --util job.util.csv \\
        --notify-every-hours 6 --watch-pid $$ &

**Where** to send it is never here and never in the description: it is the
user's own file, :func:`default_notify_path`, mode 0600 on the machine
that runs the job.  ``MB_NOTIFY_URL`` overrides it for a one-off.  A notifier can
also be registered programmatically::

    from molbuilder import monitor
    monitor.register_notifier(lambda st, ev: my_push(st.as_text()))

Design notes:
- **stdlib-only hot path** -- no heavy imports in the loop.
- Each tick reads only the **tail** of the ``.out`` (cheap on a large
  file) and the whole (tiny) timing log for the iteration COUNT.
- The stop signal is ``--watch-pid`` going away (the job wrapper's PID),
  and it is the ONLY one.  Output markers are not consulted: they can
  appear before a run is actually over, which would end the sampling
  early (`job-contracts.md`, the monitor's section).
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

import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


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


# NO COMPLETION MARKERS HERE.  `job-contracts.md` states the rule this
# module follows: the monitor "follows the launcher's PID -- so it knows
# authoritatively when the run ended, rather than guessing from output
# markers".  A private marker tuple lived here until 2026-08-26 and did
# exactly the guessing the contract forbids.
#
# It was not harmless.  `siesta: Final energy` prints BEFORE a run is over,
# so the loop could return while the job was still holding CPUs and GPUs --
# ending the utilisation sampling that is the whole reason this process
# exists, and skipping the ticks a notifier hook would have fired.  The
# watched PID cannot be early: the wrapper outlives the engine it launched.
_SCF_LINE = re.compile(r"^[ \t]*scf:[ \t]*[0-9]")
# Geometry-relaxation move marker.  SIESTA prints e.g.
# ``Begin CG move = 3`` / ``Begin FIRE move = 12`` / ``Begin Broyden
# move = 5`` / ``Begin Z-matrix opt. move = 1`` at each ionic step.  We
# take the highest move number seen in the .out tail as the current
# geometry step (None for single-point runs, which never print it).
# The literal ``move = <N>`` (and case-sensitive ``Begin``) is required
# so narrative text like "begin to move 8 atoms" can't false-positive
# (audit 2026-06-27 B-5).
_GEOM_LINE = re.compile(r"Begin\b.*\bmove\b\s*=\s*([0-9]+)")


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
    ``scf_iter`` / ``energy`` from the last ``scf:`` line.  It reports what
    the artifacts SAY and never judges whether the run is over -- that is
    the watched PID's answer alone.  Pure reads -- never raises on a
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
    return st


# --------------------------------------------------------------------- #
#  Utilization sampling (cpu% / mem / GPU sm% / VRAM)                    #
# --------------------------------------------------------------------- #
#
# The SAME monitor loop that watches SCF progress also samples machine
# utilization, so a post-run plot answers "were we GPU-bound or host/CPU-
# bound?" (sustained GPU sm% high => GPU-bound; low while cpu% pegged =>
# host-bound).  To keep the file small it is CHANGE-GATED like the status
# log: a row is written only when some metric moved >= ``change_frac``
# from the last logged row (or a keepalive elapsed).  With timestamps the
# sparse series still plots cleanly.  Stdlib only: ``/proc`` + nvidia-smi.


def _cgroup_paths() -> Dict[str, str]:
    """``{controller: path}`` from ``/proc/self/cgroup``, both generations.

    **The path must come from here.**  Reading ``/sys/fs/cgroup/cpu.stat``
    directly lands on the ROOT cgroup and silently answers for the whole
    node -- the very defect this reader exists to end, in a new spelling.

    v2 writes one ``0::/path`` line, registered under the key ``""``.  v1
    writes one line per hierarchy, ``id:controllers:/path``, controllers
    comma-separated -- so ``cpu,cpuacct`` is registered under its joined
    spelling (which is also the mount directory) AND under each name.
    """
    out: Dict[str, str] = {}
    try:
        with open(_PROC_CGROUP, encoding="ascii") as fh:
            for line in fh:
                bits = line.rstrip("\n").split(":", 2)
                if len(bits) != 3:
                    continue
                ctrls, path = bits[1], bits[2]
                if not ctrls:                      # v2: "0::/path"
                    out[""] = path
                    continue
                out[ctrls] = path                  # the mount-dir spelling
                for c in ctrls.split(","):
                    out.setdefault(c, path)
    except OSError:
        pass
    return out


#: ``memory.limit_in_bytes`` reads ``2**63 - 4096`` when nothing is
#: enforced.  MEASURED on ASU Sol 2026-08-26: the *task* cgroup carries
#: exactly that while the *job* cgroup one level up carries the real ask.
#: A reader that takes the sentinel for a limit reports 0% of an
#: astronomical number, so it is recognised as NO LIMIT STATED.
_NO_LIMIT = 1 << 62

#: cgroup v1's mount layout is ``/sys/fs/cgroup/<controller>/<path>``; v2's
#: is ``/sys/fs/cgroup/<path>``.  Both spellings of the cpu hierarchy's
#: directory are tried because sites mount it either way.
#:
#: These two are NAMED rather than inlined so a test can point them at a
#: fixture.  Every machine this code will ever run on has exactly one
#: answer for each, so a test that cannot supply its own is a test that can
#: only check the machine it happens to be running on -- and the whole
#: point here is reading a layout (SLURM cgroup v1) that this workstation
#: does not have.
_CG = "/sys/fs/cgroup"
_PROC_CGROUP = "/proc/self/cgroup"


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, encoding="ascii") as fh:
            return int(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _job_cgroup(path: str) -> str:
    """The JOB cgroup for a step/task path.

    SLURM nests ``<job>/step_N/task_M`` and enforces memory on the JOB.
    Measured on Sol: the task level answers with the no-limit sentinel
    while the job level answers ``8589934592`` for ``--mem=8G``.
    """
    cut = path.find("/step_")
    return path[:cut] if cut > 0 else path


def _read_cpu_used_ns() -> Optional[Tuple[int, str]]:
    """``(cpu-nanoseconds this job has consumed, which rung answered)``.

    THE NUMERATOR.  ``/proc/stat``'s aggregate line counts every process on
    the node, including other people's jobs, so it is the last resort and
    labels itself ``node`` -- a caller must be able to see when the number
    it is showing is not the job's.
    """
    cg = _cgroup_paths()
    p = cg.get("cpuacct") or cg.get("cpu")
    if p:                                          # cgroup v1
        for d in ("cpu,cpuacct", "cpuacct"):
            ns = _read_int(_CG + "/" + d + p + "/cpuacct.usage")
            if ns is not None:
                return ns, "cgroup-v1"
    v2 = cg.get("")
    if v2 is not None:                             # cgroup v2
        try:
            with open(_CG + v2 + "/cpu.stat", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("usage_usec"):
                        return int(line.split()[1]) * 1000, "cgroup-v2"
        except (OSError, ValueError, IndexError):
            pass
    node = _read_node_busy_ns()
    if node is not None:
        return node, "node"                        # NOT this job's
    return None


def _read_node_busy_ns() -> Optional[int]:
    """Node-wide busy time in nanoseconds, from ``/proc/stat``.

    The fallback rung, kept honest: converted to the same unit as the
    cgroup readings so one subtraction serves all three, and always
    reported under the label ``node`` so nobody mistakes it for the job.
    """
    try:
        with open("/proc/stat", encoding="ascii") as fh:
            parts = fh.readline().split()
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)   # idle + iowait
        hz = os.sysconf("SC_CLK_TCK") or 100
        return int((sum(vals) - idle) * (1000000000 // hz))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _alloc_cores() -> Tuple[int, str]:
    """``(cores this job may use, which rung answered)``.

    THE DENOMINATOR, and the whole of the rule: *a run reports how well it
    used WHAT IT WAS GIVEN.*  Cores it did not ask for are unpredictable and
    are not its business -- and a fraction taken over them makes a job that
    is saturating its own allocation look starved, which argues for a bigger
    machine, which is the queue this practice exists to stay out of.

    The affinity mask answers first: measured on Sol, a ``-c 4`` job reports
    exactly 4 through it, it needs no cgroup path, and it works on both
    generations.
    """
    try:
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n, "affinity"
    except (AttributeError, OSError):
        pass
    for var in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        try:
            n = int(os.environ.get(var, ""))
            if n > 0:
                return n, var
        except ValueError:
            pass
    return (os.cpu_count() or 1), "node"


def _read_mem_used_gb() -> Optional[Tuple[float, str]]:
    """``(GB this job's cgroup holds, which rung)``, else the node's.

    ``MemTotal - MemAvailable`` -- the previous reading -- is every process
    on the machine, so on a shared node it was measuring other people's
    jobs as much as this one's.
    """
    cg = _cgroup_paths()
    p = cg.get("memory")
    if p:
        b = _read_int(_CG + "/memory" + p + "/memory.usage_in_bytes")
        if b is not None:
            return round(b / 1073741824.0, 2), "cgroup-v1"
    v2 = cg.get("")
    if v2 is not None:
        b = _read_int(_CG + v2 + "/memory.current")
        if b is not None:
            return round(b / 1073741824.0, 2), "cgroup-v2"
    try:
        info: Dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                info[k] = int(rest.split()[0])           # kB
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return round((info.get("MemTotal", 0) - avail) / 1048576.0, 2), "node"
    except (OSError, ValueError, IndexError):
        return None


def _read_mem_peak_gb() -> Optional[float]:
    """The kernel's OWN running peak, when it keeps one.

    v1's ``memory.max_usage_in_bytes`` is a counter the kernel maintains, so
    it is a true peak rather than the largest value a 10-second sampler
    happened to catch.  Measured on Sol it read ABOVE ``usage_in_bytes``,
    which is what proves it is not a copy of current.
    """
    cg = _cgroup_paths()
    p = cg.get("memory")
    if p:
        b = _read_int(_CG + "/memory" + p + "/memory.max_usage_in_bytes")
        if b is not None:
            return round(b / 1073741824.0, 2)
    v2 = cg.get("")
    if v2 is not None:                             # newer v2 kernels only
        b = _read_int(_CG + v2 + "/memory.peak")
        if b is not None:
            return round(b / 1073741824.0, 2)
    return None


def _read_mem_limit_gb() -> Optional[float]:
    """The limit the kernel ENFORCES, or ``None`` when nothing is stated.

    Read from the JOB cgroup, never the task one.  ``None`` for the no-limit
    sentinel: `scheduler.md` R3 -- *an unstated limit never bars* -- and a
    sentinel is an absence wearing a number.
    """
    cg = _cgroup_paths()
    p = cg.get("memory")
    if p:
        b = _read_int(_CG + "/memory" + _job_cgroup(p)
                      + "/memory.limit_in_bytes")
        if b is not None:
            return None if b >= _NO_LIMIT else round(b / 1073741824.0, 2)
    v2 = cg.get("")
    if v2 is not None:
        try:
            with open(_CG + _job_cgroup(v2) + "/memory.max",
                      encoding="ascii") as fh:
                raw = fh.read().strip()
            return None if raw == "max" else round(int(raw) / 1073741824.0, 2)
        except (OSError, ValueError):
            pass
    return None


_GPU_QUERY = ["nvidia-smi",
              "--query-gpu=index,utilization.gpu,utilization.memory,"
              "memory.used",
              "--format=csv,noheader,nounits"]


def _gpu_present() -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                           timeout=5)
        return r.returncode == 0 and b"GPU " in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def _sample_gpus() -> List[Tuple[int, float, float, float]]:
    """Per-GPU ``(index, sm%, mem_util%, vram_gb)`` via nvidia-smi
    (empty list on any failure -- best-effort)."""
    try:
        r = subprocess.run(_GPU_QUERY, capture_output=True, text=True,
                          timeout=10)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 4:
                out.append((int(f[0]), float(f[1]), float(f[2]),
                            round(float(f[3]) / 1024.0, 2)))   # MiB -> GB
        return out
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def _metric_moved(new, old, frac: float) -> bool:
    """True if ``new`` differs from ``old`` by >= ``frac`` (relative, with
    a floor of 1 so near-zero values still register a real jump)."""
    if old is None or new is None:
        return new is not None
    return abs(new - old) >= max(abs(old), 1.0) * frac


@dataclass
class UtilSample:
    epoch: float
    cpu_pct: Optional[float]
    mem_gb: Optional[float]
    gpus: List[Tuple[int, float, float, float]] = field(default_factory=list)

    def changed_from(self, other: "UtilSample", frac: float) -> bool:
        if other is None:
            return True
        if _metric_moved(self.cpu_pct, other.cpu_pct, frac):
            return True
        if _metric_moved(self.mem_gb, other.mem_gb, frac):
            return True
        og = {g[0]: g for g in other.gpus}
        for idx, sm, mu, vram in self.gpus:
            o = og.get(idx)
            if o is None or _metric_moved(sm, o[1], frac) \
                    or _metric_moved(vram, o[3], frac):
                return True
        return False


@dataclass
class UtilAccum:
    """Running cpu% + per-GPU sm% stats for the end-of-run verdict.
    Flat memory (sum/min/max/count), so it is safe for arbitrarily long
    runs -- no per-sample list."""
    cpu_sum: float = 0.0
    cpu_n: int = 0
    cpu_min: float = 1e9
    cpu_max: float = -1e9
    gpu_sum: Dict[int, float] = field(default_factory=dict)
    gpu_n: Dict[int, int] = field(default_factory=dict)
    gpu_min: Dict[int, float] = field(default_factory=dict)
    gpu_max: Dict[int, float] = field(default_factory=dict)

    def add(self, s: UtilSample) -> None:
        if s.cpu_pct is not None:
            self.cpu_sum += s.cpu_pct
            self.cpu_n += 1
            self.cpu_min = min(self.cpu_min, s.cpu_pct)
            self.cpu_max = max(self.cpu_max, s.cpu_pct)
        for idx, sm, _mu, _vram in s.gpus:
            self.gpu_sum[idx] = self.gpu_sum.get(idx, 0.0) + sm
            self.gpu_n[idx] = self.gpu_n.get(idx, 0) + 1
            self.gpu_min[idx] = min(self.gpu_min.get(idx, 1e9), sm)
            self.gpu_max[idx] = max(self.gpu_max.get(idx, -1e9), sm)

    def summary(self) -> str:
        bits = []
        if self.cpu_n:
            bits.append(f"cpu mean={self.cpu_sum / self.cpu_n:.0f}% "
                        f"({self.cpu_min:.0f}-{self.cpu_max:.0f})")
        gpu_means = []
        for idx in sorted(self.gpu_n):
            m = self.gpu_sum[idx] / self.gpu_n[idx]
            gpu_means.append(m)
            bits.append(f"gpu{idx} sm mean={m:.0f}% "
                        f"({self.gpu_min[idx]:.0f}-{self.gpu_max[idx]:.0f})")
        verdict = ""
        if gpu_means:
            mx = max(gpu_means)
            if mx >= 85:
                verdict = " -> GPU-bound (GPU saturated)"
            elif mx <= 60:
                verdict = " -> host/CPU-bound (GPU starved)"
            else:
                verdict = " -> mixed (GPU not saturated)"
        return "; ".join(bits) + verdict


def _util_csv_header(ngpu: int) -> str:
    cols = ["epoch", "iso", "cpu_pct", "mem_gb"]
    for i in range(ngpu):
        cols += [f"gpu{i}_sm", f"gpu{i}_memutil", f"gpu{i}_vram_gb"]
    return ",".join(cols)


def _util_csv_row(s: UtilSample, ngpu: int) -> str:
    def _f(v):
        return "" if v is None else f"{v:g}"
    cells = [f"{s.epoch:.0f}", _iso(s.epoch), _f(s.cpu_pct), _f(s.mem_gb)]
    by_idx = {g[0]: g for g in s.gpus}
    for i in range(ngpu):
        g = by_idx.get(i)
        if g is None:
            cells += ["", "", ""]
        else:
            cells += [_f(g[1]), _f(g[2]), _f(g[3])]
    return ",".join(cells)


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


#: How long a POST may take before the monitor gives up on it.  Short on
#: purpose: this runs beside compute ranks, the server may be down, and a
#: notification is never worth costing the run anything.  Each hook is also
#: individually guarded, so a dead endpoint cannot break the loop either.
NOTIFY_TIMEOUT_S = 2.0

#: The destination file's name inside molbuilder's config directory.  NOT in
#: the calculation's description: `task.json` travels -- to a cluster, into a
#: handoff bundle, to whoever is handed the calculation -- and a token must
#: not travel with it.  The policy is in the description; the secret is here.
NOTIFY_FILENAME = "notify"


def _config_dir():
    """molbuilder's config directory, from the one module that defines it.

    Imported two ways because this file runs two ways: inside the package on
    a login node, and as a standalone `mb_monitor.py` on a compute node with
    no molbuilder installed.  The wrapper ships `config_dir.py` beside it
    (`runwrap._config_dir_source`), so the SAME EIGHT LINES answer on both
    machines.

    Restating the rule here instead would be the fourth copy of it, which
    `tests/test_config_dir_has_one_home.py` exists to prevent -- three
    modules once computed it independently, and two of them said so in prose:
    *"a comment is not a mechanism"*.
    """
    try:
        from .config_dir import config_dir      # inside the package
    except ImportError:                          # shipped beside the job
        from config_dir import config_dir
    return config_dir()


def default_notify_path() -> Path:
    """Where the destination file lives: ``<config dir>/notify``.

    Honouring ``XDG_CONFIG_HOME`` -- which :func:`_config_dir` does, because
    `config_dir.py` does -- is load-bearing HERE in particular.  On an HPC
    login node ``$HOME`` is NFS-mounted and often snapshotted, and
    ``XDG_CONFIG_HOME=/scratch/$USER`` is how a person keeps a token off it.
    This file is read on a compute node; a path hardcoded to ``$HOME`` would
    give them no way to.
    """
    return _config_dir() / NOTIFY_FILENAME


@dataclass(frozen=True)
class NotifyPolicy:
    """WHEN to speak -- the two settable occasions, as one value.

    They are one thing everywhere else: born together in `task.Notify`,
    carried together on `jobset.Resources`, consumed together here.  Passing
    them as two loose arguments is the shape `architecture.md` § 3.1's rule
    A8 forbids, and for a measured reason -- a caller re-assembling an
    object the callee should have been handed is how a third field gets
    forgotten.  It has cost this codebase two fields already.

    Not imported from `task.Notify`: this module ships to a compute node as
    a standalone file with no molbuilder importable (see the module
    docstring).  The wire between them is the CLI flag pair, and
    `tests/test_wrapper_notify_flags.py` pins that the wrapper only ever
    emits flags this file accepts.
    """
    on_scf: bool = False
    every_hours: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.on_scf or self.every_hours > 0)


def load_destination(path: Optional[str] = None,
                     log: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The webhook destination, or ``None`` when the user has not set one.

    A JSON object: ``url``, and optional ``headers``.  Two shapes of
    destination, one mechanism -- Slack and Discord put the credential IN
    the url, while a private endpoint takes a plain url and a token in a
    header.  Either way the file is the user's own, mode 0600, and nothing
    here is created on anybody's behalf.

    **Absent is not an error.**  No file means no notifier, and the run
    proceeds exactly as it does with the feature switched off.  A malformed
    file is not an error either: this is a monitor, and refusing to watch a
    job because a notification could not be configured would be the tail
    wagging the dog.  It says so and carries on.

    **It says so in the LOG**, not on stdout.  The wrapper backgrounds this
    process as ``... >/dev/null 2>&1 &``, so anything printed goes nowhere:
    a misconfigured destination would produce no notifications and no
    explanation, which is the worst of both.  ``log`` is the monitor log the
    user actually reads.  Printing is the fallback for a caller that has no
    log yet -- an interactive `molbuilder monitor`, or a test.
    """
    def _say(msg: str) -> None:
        line = f"[{_iso(time.time())}] [NOTIFY] {msg}"
        if log is not None:
            _append(Path(log), line)
        else:
            print(f"[monitor] {msg}", flush=True)

    p = Path(os.path.expanduser(path)) if path else default_notify_path()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        _say(f"{p}: not valid JSON ({exc}); not notifying")
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("url"), str) \
            or not obj["url"]:
        _say(f"{p}: needs an object with a 'url' string; not notifying")
        return None
    headers = obj.get("headers") or {}
    if not isinstance(headers, dict):
        _say(f"{p}: 'headers' must be an object; ignoring them")
        headers = {}
    return {"url": obj["url"],
            "headers": {str(k): str(v) for k, v in headers.items()}}


def make_webhook_notifier(url: str,
                          headers: Optional[Dict[str, str]] = None) -> Notifier:
    """A stdlib webhook notifier: POSTs ``event`` + the status summary to
    ``url`` as JSON.  Best-effort, short timeout, never raises out.

    JSON rather than form encoding because both destinations want it: Slack
    and Discord read a JSON body, and a private endpoint that appends to a
    record log wants structure rather than one flattened string.
    """
    def _hook(status: JobStatus, event: str) -> None:
        body = json.dumps({
            "event":      event,
            "text":       status.as_text(),
            # Slack and Discord both render a bare "text" field, so the
            # same body is readable in a channel and parseable by us.
            "state":      status.state,
            "elapsed_s":  round(status.elapsed_s, 1),
            "n_iters":    status.n_iters,
            "energy":     status.energy,
            "geom_step":  status.geom_step,
            "per_iter_s": status.per_iter_s,
        }).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", **(headers or {})})
        try:
            urllib.request.urlopen(req, timeout=NOTIFY_TIMEOUT_S).close()
        except Exception:                              # noqa: BLE001
            pass
    _hook.__name__ = "webhook_notifier"
    return _hook


def _install_env_notifiers(log: Optional[Path] = None) -> None:
    """Register the user's notifier, if they configured one.

    ``MB_NOTIFY_URL`` wins when set -- an explicit environment override is
    how you test a destination once without editing a file.  Otherwise the
    standing configuration at :func:`default_notify_path` is used.  Neither present
    means no notifier at all.

    **Registers at most one.**  :data:`_NOTIFIERS` is module state and
    ``run_monitor`` calls this every time, so without the guard a second
    call in one process adds a second copy of the same webhook and every
    event is POSTed twice.  A shipped `mb_monitor.py` runs one job per
    process and would never have shown it; anything embedding this would.
    """
    if any(getattr(fn, "__name__", "") == "webhook_notifier"
           for fn in _NOTIFIERS):
        return
    url = os.environ.get("MB_NOTIFY_URL")
    if url:
        register_notifier(make_webhook_notifier(url))
        return
    dest = load_destination(log=log)
    if dest:
        register_notifier(make_webhook_notifier(dest["url"], dest["headers"]))


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
                interval: float = 10.0,
                watch_pid: int = 0,
                start_epoch: Optional[float] = None,
                max_ticks: Optional[int] = None,
                stall_heartbeat_s: float = 600.0,
                util_path: Optional[Path] = None,
                util_change_frac: float = 0.10,
                util_keepalive_s: float = 300.0,
                sampler: Optional[Callable[[], "UtilSample"]] = None,
                notify: "NotifyPolicy" = NotifyPolicy(),
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.time) -> JobStatus:
    """Periodically parse + log + notify until the watched job ends.

    Returns the final :class:`JobStatus`.  ``max_ticks`` bounds the loop
    (tests pass a small value); ``sleep``/``clock`` are injectable for
    deterministic testing.

    **How often it LOOKS and how often it TELLS you are different numbers.**
    It wakes every ``interval`` seconds and writes a ``[STATUS]`` line
    whenever the job advanced -- that is the record, and it stays dense.
    Notifying is separate and rare, set by ``notify``
    (`execution/run-reports.md` § 2).  Until
    2026-08-26 they were the same thing: a webhook configured against this
    fired on every changed sample, which for a running job is every wake.

    Wakes every ``interval`` seconds (short, so progress is reported
    promptly) but is QUIET when nothing changed (§ 11.0c): a ``[STATUS]``
    line fires only when the job actually advanced
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
    _install_env_notifiers(log)

    st0 = parse_status(out, timing, start, clock())
    _append(log, f"[{_iso(clock())}] [MONITOR] start "
                 f"(interval={interval:.0f}s watch_pid={watch_pid}) "
                 f"{st0.as_text()}")
    _fire(st0, "start")
    # Policy state (§ 2.9).  ``last_notify`` starts at the job's start, so
    # the first periodic message lands one full period in -- not immediately,
    # which would make "every 6 hours" mean "now, then every 6 hours".
    last_notify = start
    notify_period_s = max(0.0, notify.every_hours) * 3600.0

    # --- utilization sampling setup (same loop, separate change-gated
    # output file; § 11.0e).  ``sampler`` is injectable for tests. ---
    _sample = sampler if sampler is not None else _make_default_sampler(clock)
    util_accum = UtilAccum()
    util_prev: Optional[UtilSample] = None
    util_ngpu = 0
    util_last_log = start
    if util_path is not None:
        first = _sample()
        util_ngpu = len(first.gpus)
        try:
            Path(util_path).write_text(
                _util_csv_header(util_ngpu) + "\n", encoding="utf-8")
        except OSError:
            util_path = None
        if util_path is not None:
            _append(util_path, _util_csv_row(first, util_ngpu))
            util_accum.add(first)
            util_prev = first

    def _util_tick(now: float, *, force: bool = False) -> None:
        nonlocal util_prev, util_last_log
        if util_path is None:
            return
        s = _sample()
        util_accum.add(s)
        if (force or util_prev is None
                or s.changed_from(util_prev, util_change_frac)
                or now - util_last_log >= util_keepalive_s):
            _append(util_path, _util_csv_row(s, util_ngpu))
            util_prev = s
            util_last_log = now

    prev = st0
    last_emit = start          # wall time of the last [STATUS]/[STALL] line
    ticks = 0
    while True:
        sleep(interval)
        ticks += 1
        now = clock()
        _util_tick(now)
        alive = _pid_alive(watch_pid)
        st = parse_status(out, timing, start, now)
        if not alive:
            st.state = "gone"

        st.progressing = _progressed(st, prev)

        if not alive:
            # Terminal: the cumulative ``elapsed / n_iters`` over the whole
            # run IS a valid final average, so report it (force-show even
            # though this last tick added no new iteration).
            st.progressing = True
            _util_tick(now, force=True)        # anchor the series end
            _append(log, f"[{_iso(now)}] [STATUS] {st.as_text()}")
            if util_path is not None:
                _append(log, f"[{_iso(now)}] [UTIL-SUMMARY] "
                             f"{util_accum.summary()}")
                _append(log, f"[{_iso(now)}] [UTIL-BASIS] "
                             f"{measurement_provenance()}")
            _append(log, f"[{_iso(now)}] [MONITOR] job ended "
                         f"(watched pid {watch_pid} gone); "
                         f"final notify + exit")
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
            last_emit = now
        elif stall_heartbeat_s > 0 and now - last_emit >= stall_heartbeat_s:
            # (2) Throttled liveness ping only -- no iteration-time message.
            _append(log, f"[{_iso(now)}] [STALL] no SCF/geometry progress "
                         f"for {now - last_emit:.0f}s; state={st.state} "
                         f"scf_iters={st.n_iters} "
                         f"(alive={alive})")
            # A stall IS worth telling someone about, whatever the policy
            # says: it is the "something special" case -- a job that has
            # stopped moving but not stopped running.  Already throttled to
            # one per `stall_heartbeat_s`, so it cannot become noise.
            _fire(st, "stall")
            last_notify = now
            last_emit = now

        # --- the two settable triggers (§ 2.9) ---------------------------
        #
        # A GEOMETRY STEP ADVANCING means the previous SCF cycle reached its
        # criterion -- SIESTA prints `Begin CG move = N` when it starts the
        # next one.  Read that way rather than by scanning for a convergence
        # phrase, because this file no longer keeps a marker table: the one
        # it used to keep decided the run was over and was wrong about it.
        # A single point has no move lines, so nothing fires and the finish
        # message is the whole report -- which is what it should be.
        if (notify.on_scf and st.geom_step is not None
                and prev.geom_step is not None
                and st.geom_step > prev.geom_step):
            _fire(st, "scf_converged")
            last_notify = now
        elif notify_period_s > 0 and now - last_notify >= notify_period_s:
            # `elif`: a step and a period landing on the same wake is one
            # thing worth saying, not two.  The step is the more informative
            # of them, so it wins and resets the clock.
            _fire(st, "periodic")
            last_notify = now

        prev = st
        if max_ticks is not None and ticks >= max_ticks:
            return st


def measurement_provenance() -> str:
    """One line naming what every percentage in this run is a fraction OF.

    **A percentage whose denominator is invisible is how the Au-BDT-Au sweep
    went wrong**: 48 ranks on a 128-core node capped the node-wide reading at
    37.5%, and 32.2% read as idleness rather than as a job at 86% of its own
    allocation.  With two cgroup generations in play a number that does not
    say where it came from cannot be checked at all, so the rung that
    answered is part of the measurement, not a debug aid.
    """
    cores, csrc = _alloc_cores()
    cpu = _read_cpu_used_ns()
    mem = _read_mem_used_gb()
    bits = [f"cpu% of {cores} core(s) [{csrc}]",
            f"cpu time [{cpu[1] if cpu else 'unavailable'}]",
            f"mem [{mem[1] if mem else 'unavailable'}]"]
    peak = _read_mem_peak_gb()
    if peak is not None:
        bits.append(f"peak {peak:g} GB (kernel counter)")
    lim = _read_mem_limit_gb()
    bits.append(f"limit {lim:g} GB" if lim is not None
                else "limit not stated")
    return "; ".join(bits)


def _make_default_sampler(clock: Callable[[], float]
                          ) -> Callable[[], "UtilSample"]:
    """A stateful sampler closure.

    cpu% is ``delta cpu-time / (delta wall-time x cores allocated)``, so it
    holds the previous cumulative reading and the wall clock that went with
    it.  **The denominator is the allocation**, resolved once up front: it
    cannot change during a run, and re-reading it per tick would let a
    percentage silently change meaning mid-series.  GPU presence is probed
    once too (no per-tick ``nvidia-smi -L``).
    """
    cores, _ = _alloc_cores()
    first = _read_cpu_used_ns()
    state = {"ns": first[0] if first else None, "t": clock()}
    gpu_on = _gpu_present()

    def _s() -> "UtilSample":
        now = clock()
        cur = _read_cpu_used_ns()
        cpu_pct = None
        if cur is not None and state["ns"] is not None:
            d_ns = cur[0] - state["ns"]
            d_t = now - state["t"]
            if d_t > 0 and cores > 0 and d_ns >= 0:
                cpu_pct = round(100.0 * d_ns / (d_t * 1e9 * cores), 1)
        if cur is not None:
            state["ns"] = cur[0]
        state["t"] = now
        mem = _read_mem_used_gb()
        return UtilSample(epoch=now, cpu_pct=cpu_pct,
                          mem_gb=mem[0] if mem is not None else None,
                          gpus=_sample_gpus() if gpu_on else [])
    return _s


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch))


# --------------------------------------------------------------------- #
#  Default log notifier (the PoC stub)                                  #
# --------------------------------------------------------------------- #


def make_log_notifier(log: Path) -> Notifier:
    """The PoC notifier: records into the monitor log what *would* be
    pushed.  Swap/extend with a real channel via :func:`register_notifier`."""
    def _hook(status: JobStatus, event: str) -> None:
        # Every event a notifier can see.  `start` is here because the log
        # is also the record of what the monitor did; a real destination
        # gets the same set, and it is the POLICY upstream -- not this
        # list -- that decides which of them ever occur.
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
    p.add_argument("--interval", type=float, default=10.0,
                   help="seconds between wakes (default 5; this is the "
                        "utilization sample rate -- status lines stay "
                        "change-gated, so a fast rate does not spam)")
    p.add_argument("--stall-heartbeat", type=float, default=600.0,
                   dest="stall_heartbeat_s",
                   help="when the job is making no SCF/geometry progress, "
                        "emit at most one liveness ping this often "
                        "(seconds, default 600); no per-iter timing is "
                        "printed while stalled.  Use 0 to silence the "
                        "stall heartbeat entirely")
    p.add_argument("--util", default=None, dest="util_path",
                   help="append change-gated cpu%%/mem/GPU-sm%%/VRAM samples "
                        "to this CSV (e.g. <basename>.util.csv); omit to "
                        "disable utilization sampling")
    p.add_argument("--util-keepalive", type=float, default=300.0,
                   dest="util_keepalive_s",
                   help="even with no >10%% change, write a util row at "
                        "least this often (seconds, default 300) so the "
                        "plotted series has anchor points")
    p.add_argument("--watch-pid", type=int, default=0, dest="watch_pid",
                   help="stop when this PID disappears; 0 = until done")
    p.add_argument("--nice", type=int, default=19, dest="nice_level",
                   help="self-lower OS priority by this much (default 19)")
    # WHEN to tell someone -- the calculation's own policy, carried here from
    # `task.json`'s `notify` block by the wrapper.  Neither flag says WHERE:
    # the destination is the user's file on this machine (NOTIFY_FILE).
    p.add_argument("--notify-on-scf", action="store_true",
                   dest="notify_on_scf",
                   help="notify when a geometry step completes (one SCF "
                        "cycle reached its criterion)")
    p.add_argument("--notify-every-hours", type=float, default=0.0,
                   dest="notify_every_hours",
                   help="notify every N hours; 0 = never (default)")
    a = p.parse_args(argv)
    try:
        os.nice(max(0, a.nice_level))
    except (OSError, AttributeError):
        pass
    register_notifier(make_log_notifier(a.log))
    run_monitor(a.out, a.timing, a.log,
                interval=a.interval, watch_pid=a.watch_pid,
                stall_heartbeat_s=a.stall_heartbeat_s,
                util_path=a.util_path,
                util_keepalive_s=a.util_keepalive_s,
                notify=NotifyPolicy(on_scf=a.notify_on_scf,
                                    every_hours=a.notify_every_hours))
    return 0


__all__ = [
    "JobStatus",
    "parse_status",
    "run_monitor",
    "register_notifier",
    "clear_notifiers",
    "make_webhook_notifier",
    "make_log_notifier",
    "load_destination",
    "NotifyPolicy",
    "NOTIFY_FILENAME",
    "default_notify_path",
    "NOTIFY_TIMEOUT_S",
    "Notifier",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
