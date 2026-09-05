"""``<base>.util.csv`` — the monitor's raw utilisation samples.

Peak RSS, the sampled wall window, mean CPU%, per-GPU mean SM% (max
across GPUs) and peak VRAM.

**Change-gated, not every tick.** A row lands only when a metric moves
by ≥ 10% or a 300 s keepalive elapses, so these means are a
reconstruction from a biased subset. The `.monitor.log`'s
`[UTIL-SUMMARY]` states means over EVERY tick and is the better basis
where it exists — `parse.md` § 5a's sibling upgrade, resolved by the
caller, not by either file reading the other. The csv is the one that is
always there: a trial the scheduler killed has a csv and no summary, and
that is the trial a benchmark most needs to read.

*(Moved from `bench/result.py` on 2026-09-04 -- `parse.md` § 5c.)*
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import InstrumentResult

from ._helpers import build_instrument_result


#: util.csv columns -> metric keys.  ``gpu<N>_sm`` / ``gpu<N>_vram_gb``
#: are matched per GPU index; the metric takes the max across GPUs
#: (mean per GPU first for sm%, peak anywhere for VRAM).
_CSV_GPU_SM = re.compile(r"^gpu\d+_sm$")

_CSV_GPU_VRAM = re.compile(r"^gpu\d+_vram_gb$")

def _time_weighted(series: List[Tuple[float, float]]) -> float:
    """The mean of a ``(epoch, value)`` series over TIME, not over samples.

    Takes PAIRS, not two parallel lists, and that is the point: a value and
    the instant it was read cannot be separated, so a column with a hole
    cannot borrow its neighbour's clock.  Two parallel lists made that
    silent -- a single ``[N/A]`` from ``nvidia-smi`` (MIG instances and
    some drivers report it) shortened one column, shifted every later
    reading onto the wrong interval, and returned a plausible number.

    ``util.csv`` is CHANGE-GATED: the monitor writes a row only when a
    metric moves >= its threshold, or when a 300 s keepalive elapses.  The
    rows are therefore deliberately NOT uniformly spaced, and an ordinary
    ``sum/len`` over them weights a one-second startup transient exactly as
    heavily as five minutes of steady state.

    Measured on the Au-BDT-Au sweep (2026-08-25): a 316 s CPU trial logged
    five rows, one of which covered 300 s of it.  The unweighted mean read
    **31.5%**; the true time-weighted figure is **40.3%** -- a 28% relative
    error, in the direction that makes a healthy run look idle.  Dense
    trials (30+ rows) were within ~1 point, which is exactly why this went
    unnoticed: it is worst on the shortest runs.

    Each sample is held to weigh the interval until the NEXT sample; the
    last has no successor and so contributes nothing, which is correct --
    it marks the end of the window rather than a span within it.  Falls
    back to the plain mean when there is no usable time base.
    """
    if len(series) < 2:
        return series[0][1] if series else 0.0
    span = series[-1][0] - series[0][0]
    if span <= 0:
        return sum(v for _, v in series) / len(series)
    total = 0.0
    for (t0, v0), (t1, _) in zip(series, series[1:]):
        total += v0 * (t1 - t0)
    return total / span


def util_csv_metrics(csv_text: str) -> Dict[str, float]:
    """One reader for the monitor's raw samples (``util.csv``): peak
    RSS, sampled wall window, mean CPU%, per-GPU mean SM% (max across
    GPUs) and peak VRAM.

    Returns only the keys it could derive — an empty dict for an empty
    or headerless file — so a caller can fold the result straight into a
    point's metrics.  Keys: ``peak_rss_gb``, ``wall_s``, ``cpu_mean_pct``,
    ``gpu_sm_mean_pct``, ``gpu_vram_peak_gb``.

    **``wall_s`` is the MONITORED WINDOW, not the job's wall time**
    (corrected 2026-09-03; this said "the monitor runs for the life of
    the job, so this is the job's wall time to sampling resolution").
    It is last written row − first, and both ends are anchored only when
    the monitor reaches its terminal branch: the first sample is always
    written, and the last is force-written there.  A trial the scheduler
    KILLS never reaches it, so the series ends at the last CHANGED row —
    up to a keepalive (300 s) short, and further if the metrics had gone
    flat.  **The tell is free**: that same branch writes
    ``[UTIL-SUMMARY]``, so a monitor log without one is a log whose
    ``wall_s`` is a lower bound.

    **Not the door for a point's metrics — :func:`parse_utilisation` is.**
    This reads the samples; that one decides between these numbers and the
    monitor's own, which are better where they exist (user ruling,
    2026-09-03).  Calling this directly for a metric is how the summary stops
    being read again.

    **The means are TIME-WEIGHTED because the series is change-gated.**
    ``util.csv`` holds a row only when a metric moved ≥ 10% or a
    keepalive elapsed, so a row stands for the interval until the next
    one and a plain average over rows would weight a long steady stretch
    the same as a brief spike.  That also makes these means a
    RECONSTRUCTION: the monitor's own ``[UTIL-SUMMARY]`` averages every
    tick and is exact, but is absent whenever a trial did not reach the
    monitor's terminal branch (see :func:`parse_util_bound`).
    """
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    header = [h.strip() for h in lines[0].split(",")]
    # ROW BY ROW, so a value never loses its timestamp.  Each row becomes
    # {header: float} holding only the cells that parse; a blank or
    # non-numeric cell is simply ABSENT from that row rather than dropped
    # into a shorter list where it would silently take a later row's place.
    rows: List[Dict[str, float]] = []
    for row in lines[1:]:
        cells = row.split(",")
        rec: Dict[str, float] = {}
        for i, h in enumerate(header):
            if i < len(cells) and cells[i].strip():
                try:
                    v = float(cells[i])
                except ValueError:
                    continue
                if math.isfinite(v):
                    rec[h] = v
        if rec:
            rows.append(rec)

    def _values(h: str) -> List[float]:
        return [r[h] for r in rows if h in r]

    def _series(h: str) -> List[Tuple[float, float]]:
        """The column paired with its own epochs, holes excluded."""
        return [(r["epoch"], r[h]) for r in rows if h in r and "epoch" in r]

    cols: Dict[str, List[float]] = {h: _values(h) for h in header}
    out: Dict[str, float] = {}
    epochs = cols.get("epoch") or []
    if cols.get("mem_gb"):
        # The job's OWN memory since 2026-08-26: the monitor reads its
        # cgroup (`monitor._read_mem_used_gb`), so this is the calculation
        # rather than the machine.  It used to be MemTotal - MemAvailable,
        # every process on the node -- correct only when the job held the
        # whole node, and on a shared one it was measuring other people's
        # jobs as much as this one's.
        #
        # Still the max of a sampled series, so it is bounded below by the
        # true peak.  `monitor._read_mem_peak_gb` reads the kernel's own
        # counter and is exact; folding that in is `plan` § 2.6's job,
        # because it needs somewhere in the record to put it.
        out["peak_rss_gb"] = max(cols["mem_gb"])
    if len(epochs) >= 2 and epochs[-1] > epochs[0]:
        out["wall_s"] = round(epochs[-1] - epochs[0], 1)
    if cols.get("cpu_pct"):
        out["cpu_mean_pct"] = round(_time_weighted(_series("cpu_pct")), 1)
    sm_means = [_time_weighted(_series(h)) for h, v in cols.items()
                if _CSV_GPU_SM.match(h) and v]
    if sm_means:
        out["gpu_sm_mean_pct"] = round(max(sm_means), 1)
    vram_peaks = [max(v) for h, v in cols.items()
                  if _CSV_GPU_VRAM.match(h) and v]
    if vram_peaks:
        out["gpu_vram_peak_gb"] = round(max(vram_peaks), 2)
    return out

class UtilCsvFileParser(FileParser):
    """The wrapper monitor's sample table."""

    name   = "util-csv"
    label  = "wrapper utilisation samples (.util.csv)"
    hint   = "files ending in .util.csv"
    output = InstrumentResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".util.csv") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> InstrumentResult:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return build_instrument_result(metrics=util_csv_metrics(text),
                                       parser_name=cls.name, source=path)
