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
* ``pynvml`` (optional, ~80 KB): official ``nvidia-ml-py`` binding to
  NVML, the same library ``nvidia-smi`` uses under the hood.  ~1 ms
  per call vs ~50-100 ms for a subprocess fork.  Import-guarded so a
  CPU-only host or a missing NVML library degrades gracefully to
  ``gpus: []`` -- the JS widget then hides its GPU bars.

Concurrency: the only shared state is psutil's internal CPU-tick
snapshot (held in a process-global by psutil itself).  Concurrent
requests interleave their reads of that snapshot; the worst case is
one request sees a slightly stale delta.  Acceptable for a
human-driven widget polling at multi-second cadence.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, List

from flask import Blueprint, jsonify

import psutil


bp = Blueprint("system_load", __name__)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  NVML init -- one-shot at import, guarded so a missing driver doesn't
#  break the rest of the web app.
# ---------------------------------------------------------------------------
#
# We do NOT call nvmlInit() on every request -- that's the same
# expensive driver handshake nvidia-smi pays per invocation.  Init
# once at import; the handle objects we cache below stay valid for
# the process lifetime.  Driver unload mid-process would invalidate
# them, but that requires root + isn't a real failure mode.

_NVML_OK: bool = False
_GPU_HANDLES: List[Any] = []

try:
    import pynvml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover -- CPU-only test environments
    pynvml = None  # type: ignore[assignment]
    _log.info("pynvml not importable; GPU stats disabled "
              "(install molbuilder[gpu] to enable)")
else:
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        _GPU_HANDLES = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        _NVML_OK = True
        _log.info("NVML initialised; %d GPU(s) discovered", n)
    except Exception as exc:  # noqa: BLE001 -- NVML throws many exception types
        # No NVIDIA driver loaded, no GPU on the box, MIG partitioning
        # turned on without permission, kernel mismatch, etc.  All
        # legitimate "no GPU stats today" causes -- log once at startup
        # and proceed.
        _log.info("NVML init failed (%s: %s); GPU stats disabled",
                  type(exc).__name__, exc)


def _gpu_name(handle) -> str:
    """Return the GPU's product name, falling back to '<unknown>' on
    error.  NVML's ``nvmlDeviceGetName`` returns bytes in some pynvml
    versions and str in others -- normalise to str."""
    try:
        n = pynvml.nvmlDeviceGetName(handle)
    except Exception:  # noqa: BLE001
        return "<unknown>"
    if isinstance(n, bytes):
        try:
            return n.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return "<unknown>"
    return str(n)


def _try(fn, *args):
    """NVML field probe wrapper.  Returns ``fn(*args)`` or ``None`` if
    NVML raises -- some fields aren't supported on every chip
    (consumer cards lack power-cap, etc.) so the snapshot stays
    partial-fail rather than all-or-nothing."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 -- NVML throws many distinct types
        return None


def _gpu_snapshot() -> List[Dict[str, Any]]:
    """One entry per GPU; empty list when NVML is unavailable.

    Fields emitted per GPU (all best-effort -- consumer cards may not
    expose every probe, in which case the value is ``None`` and the
    widget shows ``—`` for that cell):

      * ``util_pct``       -- SM compute %% (kernels running)
      * ``util_mem_pct``   -- memory controller %% (VRAM bandwidth
                              utilisation -- distinguishes compute-bound
                              from memory-bandwidth-bound kernels;
                              same struct as util_pct, no extra call)
      * ``mem_used_mb`` / ``mem_total_mb`` / ``mem_pct``
                             -- VRAM occupancy (capacity, not bw)
      * ``power_w``        -- current power draw (W); drops mid-run
                              signal thermal/power throttle
      * ``temp_c``         -- GPU package temperature (°C);
                              >83 °C on consumer cards triggers throttle
      * ``sm_clock_mhz``   -- graphics clock (boost behaviour visible
                              here when correlated against util_pct)
      * ``mem_clock_mhz``  -- memory clock (drops when not under load)
    """
    if not _NVML_OK or pynvml is None:
        return []
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(_GPU_HANDLES):
        try:
            util  = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem_used_mb  = mem.used  / (1 << 20)
            mem_total_mb = mem.total / (1 << 20)
            mem_pct = (100.0 * mem.used / mem.total) if mem.total else 0.0
            # Remaining fields are best-effort: every consumer card
            # exposes them, but data-center variants or virtualised
            # environments sometimes don't.
            power_mw = _try(pynvml.nvmlDeviceGetPowerUsage, h)
            temp_c   = _try(pynvml.nvmlDeviceGetTemperature, h,
                            pynvml.NVML_TEMPERATURE_GPU)
            sm_mhz   = _try(pynvml.nvmlDeviceGetClockInfo, h,
                            pynvml.NVML_CLOCK_GRAPHICS)
            mem_mhz  = _try(pynvml.nvmlDeviceGetClockInfo, h,
                            pynvml.NVML_CLOCK_MEM)
            out.append({
                "index":         i,
                "name":          _gpu_name(h),
                "util_pct":      float(util.gpu),
                "util_mem_pct":  float(util.memory),
                "mem_used_mb":   round(mem_used_mb, 1),
                "mem_total_mb": round(mem_total_mb, 1),
                "mem_pct":      round(mem_pct, 1),
                "power_w":      (None if power_mw is None
                                  else round(power_mw / 1000.0, 1)),
                "temp_c":       (None if temp_c   is None else int(temp_c)),
                "sm_clock_mhz":  (None if sm_mhz   is None else int(sm_mhz)),
                "mem_clock_mhz": (None if mem_mhz  is None else int(mem_mhz)),
            })
        except Exception as exc:  # noqa: BLE001
            # Don't let one flaky GPU kill the whole snapshot --
            # include an explicit entry with the failure so the UI
            # can flag it instead of silently dropping it.
            out.append({
                "index": i,
                "name":  "<probe failed>",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return out


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
        "gpus":                _gpu_snapshot(),
    }


@bp.get("/api/system/load")
def api_system_load():
    """Return the current load snapshot.

    Response envelope (web-api.md § 1.1.1):

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
