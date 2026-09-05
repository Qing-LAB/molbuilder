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

**One resolver, not two readers.**  The pair was collapsed into the csv
alone on 2026-08-19 after two readers of one fact diverged.  That lesson
holds; the answer is that the choice lives HERE rather than at each call
site, so there is still exactly one thing to be wrong.  Neither parser
reads the other's file (`parse.md` § 5a).
"""
from __future__ import annotations

from typing import Any, Dict


def utilisation(monitor: Dict[str, Any], csv: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a `monitor-log` result's metrics over a `util-csv` result's.

    Keys are the csv's, plus ``util_basis`` naming where the MEANS came
    from (``"monitor-summary"`` | ``"util-csv"`` | ``"mixed"``) so a
    reader can tell an exact figure from a reconstruction.  Peak RSS, the
    sampled wall window and peak VRAM come from the csv either way: the
    summary does not carry them.

    **``"mixed"`` exists because ONE label cannot describe TWO means.**
    The monitor's ``summary()`` emits a ``cpu mean=`` bit and a ``gpuN sm
    mean=`` bit independently, so a ``[UTIL-SUMMARY]`` truncated
    mid-write -- a partial flush when a trial is killed, and a killed
    trial is the one a benchmark most needs to read -- states the CPU
    and not the GPU.  This stamped ``"monitor-summary"`` over a GPU
    figure it had taken from the change-gated csv, which is precisely
    the mistake the field exists to prevent (`plan` § E2: "a
    reconstruction is never mistaken for an exact figure").
    """
    out = dict(csv or {})
    mon = monitor or {}
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
