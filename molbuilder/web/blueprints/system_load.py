"""System-load API endpoint -- powers the Results-tab load monitor widget.

The widget polls ``/api/system/load`` every few seconds while the
Results tab is active (paused when the document is hidden) and renders
four bars: CPU %, RAM %, GPU util %, GPU mem %.  Cheap to call (~1 ms
on a Linux box with NVIDIA driver loaded).

Single dependency rationale:

* ``psutil`` (required, ~150 KB): cross-platform CPU + RAM stats.  Its
  ``cpu_percent(interval=None)`` handles the two-sample delta
  internally -- no manual ``/proc/stat`` bookkeeping.  First call
  returns 0.0 (no prior sample); every subsequent call returns the
  percent between the prior call and this one.
* GPU stats come from ``nvidia-smi`` run as a SUBPROCESS with a hard
  timeout, cached for a few seconds -- **never an in-process NVML
  call**.  This module used ``pynvml`` until 2026-08-28, when a
  request thread entered the driver during another user's GPU work
  and never came back: NVML has no timeout anywhere in its API, a
  thread blocked in a driver ioctl cannot be cancelled from Python,
  and the whole web server froze for eight hours behind one page
  widget.  The process boundary is the only real fence: a stuck
  child is abandoned at the timeout and the server keeps serving
  (user rule: *a temporary failure of the inquiry, never a failure
  of the system*).  The monitor on compute nodes has always sampled
  this way (`monitor._sample_gpus`), so this is the server catching
  up to its own convention, not a new one.  NVML itself is built for
  many concurrent readers -- nvidia-smi, DCGM, several molbuilder
  instances -- so the subprocess costs nothing in correctness; the
  cache makes its ~50 ms cost irrelevant at widget cadence.

Degrading gracefully is right; degrading SILENTLY is not.  When the
GPU is missing because something on the host is broken rather than
because there is no GPU, the reason travels with the snapshot as
``gpu_error`` and the widget prints it where the cells would have
been -- a query that TIMES OUT is exactly such a reason.

Concurrency: the only shared state is psutil's internal CPU-tick
snapshot (held in a process-global by psutil itself).  Concurrent
requests interleave their reads of that snapshot; the worst case is
one request sees a slightly stale delta.  Acceptable for a
human-driven widget polling at multi-second cadence.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify

import psutil


bp = Blueprint("system_load", __name__)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  GPU stats -- a subprocess with a hard timeout, behind a small cache
# ---------------------------------------------------------------------------
#
# THE PROCESS BOUNDARY IS THE FENCE (module docstring): the query runs in
# a child that can be abandoned, so a wedged driver costs one stale tick,
# never a request thread.  The cache keeps widget polling cheap and means
# several open pages share one query per TTL rather than one each.

import threading as _threading
import time as _time

#: One row per GPU, comma-separated, no units -- the same invocation
#: shape the compute-node monitor has always used.
_SMI_QUERY = ("nvidia-smi",
              "--query-gpu=name,utilization.gpu,utilization.memory,"
              "memory.used,memory.total,power.draw,temperature.gpu,"
              "clocks.sm,clocks.mem",
              "--format=csv,noheader,nounits")

#: The no-block property, as a number: a query that has not answered in
#: this long is abandoned and REPORTED, and the server keeps serving.
_SMI_TIMEOUT_S = 2.0

#: How long one answer serves every open page.  Widget cadence is a few
#: seconds; sampling faster than this buys nothing.
_SMI_TTL_S = 3.0

_smi_lock = _threading.Lock()
_smi_cache: Dict[str, Any] = {"t": 0.0, "gpus": [], "error": None}


def _num(tok: str) -> Optional[float]:
    """One csv token -> float, or ``None`` for the fields a card does not
    expose (nvidia-smi prints ``[N/A]``, ``[Not Supported]``...)."""
    try:
        return float(tok.strip())
    except ValueError:
        return None


def _parse_smi(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(text.strip().splitlines()):
        f = [c.strip() for c in line.split(",")]
        if len(f) < 9:
            continue
        used, total = _num(f[3]), _num(f[4])
        pct = (100.0 * used / total) if used is not None and total else None
        util, util_m = _num(f[1]), _num(f[2])
        power, temp = _num(f[5]), _num(f[6])
        sm, mem = _num(f[7]), _num(f[8])
        out.append({
            "index":         i,
            "name":          f[0] or "<unknown>",
            "util_pct":      util,
            "util_mem_pct":  util_m,
            "mem_used_mb":   None if used  is None else round(used, 1),
            "mem_total_mb":  None if total is None else round(total, 1),
            "mem_pct":       None if pct   is None else round(pct, 1),
            "power_w":       None if power is None else round(power, 1),
            "temp_c":        None if temp  is None else int(temp),
            "sm_clock_mhz":  None if sm    is None else int(sm),
            "mem_clock_mhz": None if mem   is None else int(mem),
        })
    return out


def _gpu_snapshot() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """``(gpus, gpu_error)`` -- cached, subprocess-backed, never blocking
    a request longer than ``_SMI_TIMEOUT_S``.

    Three honest answers:

      * a CPU-only box (no ``nvidia-smi`` on PATH) -- ``([], None)``,
        quietly: nothing is wrong, the widget drops its GPU cells;
      * the tool answered -- the parsed rows, error ``None``;
      * the tool failed or TIMED OUT -- ``([], reason)``: the box is
        meant to have a GPU and cannot currently be asked about it,
        which is a fact the widget prints where the cells would be.
        A timeout in particular is the driver being held by someone's
        compute -- a temporary failure of the inquiry, and the next
        TTL tick simply asks again.
    """
    now = _time.monotonic()
    with _smi_lock:
        if now - _smi_cache["t"] < _SMI_TTL_S:
            return list(_smi_cache["gpus"]), _smi_cache["error"]
        # Claim the slot BEFORE the query so concurrent requests serve
        # the previous answer instead of piling onto one stuck child.
        _smi_cache["t"] = now
    gpus: List[Dict[str, Any]] = []
    err: Optional[str] = None
    try:
        cp = subprocess.run(_SMI_QUERY, capture_output=True, text=True,
                            timeout=_SMI_TIMEOUT_S)
        if cp.returncode == 0:
            gpus = _parse_smi(cp.stdout)
        else:
            err = (f"nvidia-smi exited {cp.returncode}: "
                   f"{(cp.stderr or '').strip()[:200] or 'no message'}")
    except FileNotFoundError:
        err = None                        # CPU-only box: a choice, not a fault
    except subprocess.TimeoutExpired:
        err = (f"GPU query timed out after {_SMI_TIMEOUT_S:g}s -- the "
               f"driver is busy or held (someone's compute job?); "
               f"will retry")
    except OSError as exc:
        err = f"{type(exc).__name__}: {exc}"
    if err:
        _log.warning("GPU stats unavailable this tick: %s", err)
    with _smi_lock:
        _smi_cache["gpus"] = gpus
        _smi_cache["error"] = err
        _smi_cache["t"] = _time.monotonic()
    return list(gpus), err


# --------------------------------------------------------------------- #
#  Per-socket CPU topology + utilisation                                 #
# --------------------------------------------------------------------- #
#
# Why per-socket matters: on multi-socket boxes running the SIESTA-GPU
# wrapper with NUMA pinning, only one socket is active (ranks pinned
# to the GPU-proximate socket; the other socket sits idle).  Aggregate
# CPU% then reads "~10/20 cores busy", which a casual observer reads
# as half-saturated.  Per-socket lets the tooltip say "socket 0:
# 9/10 busy, socket 1: idle" so the diagnosis is "NUMA pin healthy,
# GPU socket saturated", not "machine under-used".
#
# Topology is one-shot at module import (sockets don't move at run
# time); per-CPU utilisation runs per request via psutil.


def _read_cpu_to_socket_map() -> Dict[int, int]:
    """Map ``logical_cpu_id -> socket_id`` from ``lscpu -p=CPU,Core,Socket,Node``.

    Returns ``{}`` when lscpu is unavailable or the parse fails -- the
    snapshot then skips ``per_socket_pct`` (front-end falls back to the
    aggregate view, no breakage).
    """
    try:
        cp = subprocess.run(
            ["lscpu", "-p=CPU,Core,Socket,Node"],
            capture_output=True, text=True, timeout=2.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}
    if cp.returncode != 0:
        return {}
    mapping: Dict[int, int] = {}
    for line in cp.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            cpu_id = int(parts[0])
            socket_id = int(parts[2])
        except ValueError:
            continue
        mapping[cpu_id] = socket_id
    return mapping


_CPU_TO_SOCKET: Dict[int, int] = _read_cpu_to_socket_map()


def _per_socket_pct() -> List[Dict[str, Any]]:
    """Aggregate psutil per-CPU% by socket.

    Returns one entry per socket, sorted by socket id:
      ``[{"socket": 0, "pct": 87.3, "cpu_count": 20}, ...]``

    Returns ``[]`` when topology couldn't be read (single-socket
    box, lscpu missing, etc.) -- caller treats absence as "skip the
    per-socket breakdown, show only the aggregate".

    Note: psutil's per-CPU list is indexed by LOGICAL CPU id (HT
    siblings are separate entries).  Two HT-siblings on the same
    physical core land in the same socket bucket, which is what we
    want -- the saturation reading is "how busy is socket 0", not
    "how busy is core 0".
    """
    if not _CPU_TO_SOCKET:
        return []
    try:
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
    except Exception:  # noqa: BLE001 -- defensive; psutil rarely fails here
        return []
    if not per_cpu:
        return []
    # Aggregate.
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for cpu_idx, pct in enumerate(per_cpu):
        sock = _CPU_TO_SOCKET.get(cpu_idx)
        if sock is None:
            continue
        sums[sock]    = sums.get(sock, 0.0) + float(pct)
        counts[sock]  = counts.get(sock, 0) + 1
    out: List[Dict[str, Any]] = []
    for sock in sorted(sums.keys()):
        n = counts[sock] or 1
        out.append({
            "socket":    sock,
            "pct":       round(sums[sock] / n, 1),
            "cpu_count": counts[sock],
        })
    return out


def snapshot() -> Dict[str, Any]:
    """Single-call system-load snapshot.

    Public so unit tests + a future CLI ``molbuilder system load``
    subcommand can both use this without going through Flask.

    Why we emit BOTH ``cpu_pct`` and the absolute load fields:
    ``50%`` on a 20-physical / 40-logical box means "the equivalent
    of 10 physical cores fully busy" or "20 logical threads busy" --
    the bare percentage hides which.  ``cpu_count_physical`` /
    ``cpu_count_logical`` let the UI report ``~10/20 cores`` so the
    user reads the actual saturation.  ``loadavg_*`` (the classic
    Unix 1/5/15 min queue-depth average) tells the truth about
    OVER-subscription: load 25 on a 20-core box means the run queue
    is queueing -- which a 100% cpu_pct can hide.

    ``per_socket_pct`` (added 2026-06-16) emits one entry per CPU
    socket so the tooltip can distinguish "GPU socket saturated +
    other socket idle" (NUMA-pin healthy) from "both sockets half-
    busy" (rank spread, paying UPI penalty).  Empty list on
    single-socket / lscpu-less hosts -- the front-end falls back to
    the aggregate-only view.
    """
    vm = psutil.virtual_memory()
    # ``getloadavg`` isn't on Windows; psutil emulates it but may
    # raise ``OSError`` on platforms where it can't.  Default to
    # (None, None, None) so the wire shape stays stable.
    try:
        la1, la5, la15 = psutil.getloadavg()
    except (OSError, AttributeError):  # pragma: no cover -- non-POSIX hosts
        la1 = la5 = la15 = None
    _gpus, _gpu_err = _gpu_snapshot()
    return {
        "cpu_pct":             psutil.cpu_percent(interval=None),
        "cpu_count_physical":  psutil.cpu_count(logical=False),
        "cpu_count_logical":   psutil.cpu_count(logical=True),
        "per_socket_pct":      _per_socket_pct(),
        "loadavg_1m":          (None if la1  is None else round(la1,  2)),
        "loadavg_5m":          (None if la5  is None else round(la5,  2)),
        "loadavg_15m":         (None if la15 is None else round(la15, 2)),
        "ram_pct":             vm.percent,
        "ram_used_gb":         round(vm.used  / (1 << 30), 2),
        "ram_total_gb":        round(vm.total / (1 << 30), 2),
        "gpus":                _gpus,
        # ``None`` unless a GPU was expected and could not be asked about
        # this tick (the tool failed, or the query TIMED OUT under a held
        # driver).  An empty ``gpus`` alone cannot tell the widget whether
        # to stay quiet or speak up, because a CPU-only box produces the
        # same empty list.
        "gpu_error":           _gpu_err,
    }


@bp.get("/api/system/load")
def api_system_load():
    """Return the current load snapshot.

    Response envelope (web-api.md § 1):

        { "ok": true, "data": { ...snapshot... } }

    Failure mode is "snapshot raised" -- psutil never raises on a
    healthy box; if it does, log and return ok=false with the
    exception message.  The widget then shows "—" everywhere and
    keeps polling.
    """
    try:
        data = snapshot()
    except Exception as exc:  # noqa: BLE001 -- defensive top-level
        _log.exception("system load snapshot failed")
        return jsonify({
            "ok":    False,
            "error": f"{type(exc).__name__}: {exc}",
        }), 500
    return jsonify({"ok": True, "data": data})


__all__ = ["bp", "snapshot"]
