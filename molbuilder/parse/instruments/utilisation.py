"""Which utilisation figure to believe — the sibling upgrade, resolved.

**The monitor's own means where it stated them, the samples where it did
not** *(user ruling, 2026-09-03)*.  Two files describe one fact and each
is better at something:

| | basis | availability |
|---|---|---|
| `.monitor.log`'s `[UTIL-SUMMARY]` | **every tick** — the accumulator runs unconditionally | only if the monitor reached its terminal branch |
| `.util.csv` | a **change-gated** subset — a row lands when a metric moves ≥ 10% or a 300 s keepalive elapses | always |

So the summary is the right number and the csv is the one that is always
there.  A trial the scheduler KILLS has a csv and no summary — and that
is the trial a benchmark most needs to read, which is why the csv path
cannot simply be dropped.

**Resolution is not accuracy.**  The monitor prints its means at whole
percent while the csv reconstruction carries one decimal — but a
change-gated subset can be biased by several percent, where rounding
costs at most half of one.  The better basis wins.

**One resolver, not two readers.**  The choice lives HERE rather than at
each call site, so there is exactly one thing to be wrong.  Neither parser
reads the other's file (`parse.md` § 5a).
"""
from __future__ import annotations

from typing import Any, Dict


def utilisation(monitor: Dict[str, Any], csv: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a `monitor-log` result's metrics over a `util-csv` result's.

    Keys are the csv's, plus ``util_basis`` naming where the MEANS came
    from (``"monitor-summary"`` | ``"util-csv"`` | ``"mixed"``) so a
    reader can tell an exact figure from a reconstruction, and the job's
    memory peak, ``mem_peak_gb``, with ``mem_peak_from`` saying which it
    is -- the kernel's counter, or the largest sample.  The sampled window
    (``monitored_elapsed_s``) and peak VRAM come from the csv either way:
    the summary does not carry them.

    **``"mixed"`` exists because ONE label cannot describe TWO means.**
    The monitor's ``summary()`` emits a ``cpu mean=`` bit and a ``gpuN sm
    mean=`` bit independently, so a ``[UTIL-SUMMARY]`` truncated
    mid-write -- a partial flush when a trial is killed, and a killed
    trial is the one a benchmark most needs to read -- states the CPU
    and not the GPU, and ``"monitor-summary"`` over a GPU figure taken
    from the change-gated csv is the mistake the field exists to
    prevent.
    """
    out = dict(csv or {})
    mon = monitor or {}
    # THE JOB'S MEMORY PEAK, one figure (`model/parse.md` § 5c.1): the
    # kernel's own counter where the job has a cgroup of its own -- the
    # monitor states it on `[UTIL-BASIS]` -- else the largest sample, which
    # a run started directly, measured on its process tree, has alone.
    sampled = out.pop("mem_peak_sampled_gb", None)
    kernel = mon.get("mem_peak_kernel_gb")
    if kernel is not None:
        out["mem_peak_gb"], out["mem_peak_from"] = kernel, "kernel counter"
    elif sampled is not None:
        out["mem_peak_gb"], out["mem_peak_from"] = sampled, "largest sample"
    cpu = mon.get("stated_cpu_mean_pct")
    gpu = mon.get("stated_gpu_sm_mean_pct")
    if cpu is None and gpu is None:
        if out:
            out["util_basis"] = "util-csv"
        return out
    # Which means are PRESENT, and which of those the monitor stated.
    # A mean the csv never carried is not a disagreement -- a CPU-only
    # node has no GPU figure from either source, and that is one basis.
    exact, reconstructed = [], []
    for key, stated in (("cpu_mean_pct", cpu), ("gpu_sm_mean_pct", gpu)):
        if stated is not None:
            out[key] = stated
            exact.append(key)
        elif key in out:
            reconstructed.append(key)
    out["util_basis"] = "mixed" if (exact and reconstructed) else "monitor-summary"
    return out
