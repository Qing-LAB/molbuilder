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


def _gpu_snapshot() -> List[Dict[str, Any]]:
    """One entry per GPU; empty list when NVML is unavailable."""
    if not _NVML_OK or pynvml is None:
        return []
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(_GPU_HANDLES):
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem_used_mb  = mem.used  / (1 << 20)
            mem_total_mb = mem.total / (1 << 20)
            mem_pct = (100.0 * mem.used / mem.total) if mem.total else 0.0
            out.append({
                "index":         i,
                "name":          _gpu_name(h),
                "util_pct":      float(util),
                "mem_used_mb":   round(mem_used_mb, 1),
                "mem_total_mb": round(mem_total_mb, 1),
                "mem_pct":      round(mem_pct, 1),
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


def snapshot() -> Dict[str, Any]:
    """Single-call system-load snapshot.

    Public so unit tests + a future CLI ``molbuilder system load``
    subcommand can both use this without going through Flask.
    """
    vm = psutil.virtual_memory()
    return {
        "cpu_pct":      psutil.cpu_percent(interval=None),
        "cpu_count":    psutil.cpu_count(logical=False),
        "ram_pct":      vm.percent,
        "ram_used_gb":  round(vm.used  / (1 << 30), 2),
        "ram_total_gb": round(vm.total / (1 << 30), 2),
        "gpus":         _gpu_snapshot(),
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
