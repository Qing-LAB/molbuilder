"""Benchmark results -> a portable ``bench-result`` record.

The decoupling point between the *bench* stage and the *run* stage
(docs/protocols/benchmark-workflow.md § 5.3): summarize reads each
measured point's artifacts (timing log, utilization, peak memory) and
writes ``bench-result@1`` -- the only input prep-run needs.  Its ``choice``
block is the **portable** decision (engine + ranks-per-GPU K); the
concrete knobs are re-resolved per machine by the adapter (§ 5.4).

**Stdlib-only** (parsers + json + dataclasses): ships to the target with
the rest of the prep layer.  The pure ``parse_*`` functions take text and
are unit-tested; ``build_bench_result`` assembles them.

NOTE (output isolation): a point is identified by its ``label``; the
caller hands each point its own artifacts.  The sweep must therefore give
every (G, K) point a DISTINCT output set (its own SystemLabel/subdir) --
otherwise all GPU points share the ``job-gpu`` basename and clobber each
other.  Wiring that into ``format_bench`` is a tracked follow-up; this
module is ready to consume properly-isolated points.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

SCHEMA = "molbuilder/bench-result@1"


# --------------------------------------------------------------------- #
#  Pure parsers (text -> values; unit-tested)                           #
# --------------------------------------------------------------------- #


def parse_scf_timing(text: str) -> Dict[str, Optional[float]]:
    """Steady-state seconds/iteration from a ``.scf-timing.log``.

    Each line starts with an epoch timestamp; consecutive deltas are the
    per-iteration durations.  We drop the first delta (iter 1->2, which can
    still carry warm-up) when >=3 iterations are present, matching the
    "mean of iters 3-5" rule (slurm-integration.md § 11.0).  Returns
    ``{"s_per_iter": float|None, "iters_measured": int}``.
    """
    epochs: List[float] = []
    for line in text.splitlines():
        tok = line.split(None, 1)[0] if line.strip() else ""
        try:
            v = float(tok)
        except ValueError:
            continue
        if math.isfinite(v):                         # reject nan/inf tokens
            epochs.append(v)
    if len(epochs) < 2:
        return {"s_per_iter": None, "iters_measured": 0}
    # Only forward (positive) deltas are real iteration durations.
    deltas = [d for d in (epochs[i] - epochs[i - 1]
                          for i in range(1, len(epochs))) if d > 0]
    if not deltas:
        return {"s_per_iter": None, "iters_measured": 0}
    measured = deltas[1:] if len(deltas) >= 2 else deltas
    return {"s_per_iter": round(sum(measured) / len(measured), 1),
            "iters_measured": len(measured)}


_CPU_MEAN = re.compile(r"cpu mean=([0-9.]+)%")
_GPU_MEAN = re.compile(r"gpu\d+ sm mean=([0-9.]+)%")
_VERDICT = re.compile(r"->\s*(GPU-bound|host/CPU-bound|mixed)")
_BOUND_MAP = {"GPU-bound": "gpu", "host/CPU-bound": "host", "mixed": "mixed"}


def parse_util_summary(monitor_log: str) -> Dict[str, Optional[object]]:
    """Read the ``[UTIL-SUMMARY]`` line from a ``.monitor.log``: mean
    cpu%, the highest per-GPU mean sm%, and the bound verdict
    (``gpu`` | ``host`` | ``mixed``).  Missing pieces -> ``None``."""
    line = ""
    for ln in monitor_log.splitlines():
        if "[UTIL-SUMMARY]" in ln:
            line = ln                                # last one wins
    out: Dict[str, Optional[object]] = {
        "cpu_mean_pct": None, "gpu_sm_mean_pct": None, "bound": None}
    if not line:
        return out
    m = _CPU_MEAN.search(line)
    if m:
        out["cpu_mean_pct"] = float(m.group(1))
    gpus = [float(x) for x in _GPU_MEAN.findall(line)]
    if gpus:
        out["gpu_sm_mean_pct"] = max(gpus)
    v = _VERDICT.search(line)
    if v:
        out["bound"] = _BOUND_MAP.get(v.group(1))
    return out


def parse_util_csv_peak_mem(csv_text: str) -> Optional[float]:
    """Peak ``mem_gb`` over a ``util.csv`` (workstation fallback for peak
    RSS when there is no scheduler accounting)."""
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    header = [h.strip() for h in lines[0].split(",")]
    try:
        idx = header.index("mem_gb")
    except ValueError:
        return None
    peak = None
    for row in lines[1:]:
        cells = row.split(",")
        if idx < len(cells) and cells[idx].strip():
            try:
                v = float(cells[idx])
            except ValueError:
                continue
            peak = v if peak is None else max(peak, v)
    return peak


_SACCT_MEM = re.compile(r"\bmem=([0-9.]+)([KMGT])", re.IGNORECASE)
_SACCT_UNIT = {"K": 1 / 1048576, "M": 1 / 1024, "G": 1.0, "T": 1024.0}


def parse_sacct_mem(sacct_text: str) -> Optional[float]:
    """Peak memory in GB from ``sacct`` output -- the ``mem=<n><unit>`` in
    a ``TRESUsageInMax`` field (the most robust place; Sol leaves the bare
    ``MaxRSS`` column blank for some jobs).  Takes the max seen.

    CONTRACT: the caller's ``sacct -o`` format MUST include
    ``TRESUsageInMax`` (not only ``MaxRSS``) -- a bare ``MaxRSS`` column
    has no ``mem=`` token and is deliberately NOT scanned (a generic
    ``<n><unit>`` scan would also match ``ReqMem``/``MaxVMSize`` and
    over-report).  When nothing matches this returns ``None`` and the
    caller's recommendation simply omits ``mem_gb`` (visibly absent, not
    silently wrong)."""
    peak = None
    for val, unit in _SACCT_MEM.findall(sacct_text):
        try:
            gb = float(val) * _SACCT_UNIT[unit.upper()]
        except (ValueError, KeyError):
            continue
        peak = gb if peak is None else max(peak, gb)
    return None if peak is None else round(peak, 1)


# --------------------------------------------------------------------- #
#  Data model (§ 5.3)                                                   #
# --------------------------------------------------------------------- #


@dataclass
class BenchPoint:
    # Defaults so a malformed/partial point in loaded JSON degrades to an
    # empty-label entry rather than raising TypeError (F5).
    label:   str = ""
    engine:  str = ""                                # "cpu" | "gpu"
    knobs:   Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)      # s_per_iter, sm%, rss...
    bound:   Optional[str] = None                    # gpu | host | mixed
    state:   str = "unknown"                         # completed|timeout|...

    def s_per_iter(self) -> Optional[float]:
        v = self.metrics.get("s_per_iter")
        return v if isinstance(v, (int, float)) else None


@dataclass
class BenchResult:
    environment: Dict = field(default_factory=dict)
    system:      Dict = field(default_factory=dict)
    points:      List[BenchPoint] = field(default_factory=list)
    choice:      Dict = field(default_factory=dict)
    recommend:   Dict = field(default_factory=dict)
    generated_at: Optional[str] = None
    tool:        str = "bench-summarize@1"

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "generated_at": self.generated_at,
            "environment": self.environment,
            "system": self.system,
            "points": [asdict(p) for p in self.points],
            "choice": self.choice,
            "recommend": self.recommend,
            "tool": self.tool,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchResult":
        got = str(d.get("schema", ""))
        want_major = SCHEMA.rsplit("@", 1)[-1]
        got_major = got.rsplit("@", 1)[-1] if "@" in got else ""
        if got_major != want_major:
            raise ValueError(
                f"bench-result schema mismatch: got {got!r}, need major "
                f"{want_major} ({SCHEMA}).")
        pt_fields = {f for f in BenchPoint.__dataclass_fields__}
        points = [BenchPoint(**{k: v for k, v in p.items() if k in pt_fields})
                  for p in (d.get("points") or [])]
        return cls(
            environment=d.get("environment") or {},
            system=d.get("system") or {},
            points=points,
            choice=d.get("choice") or {},
            recommend=d.get("recommend") or {},
            generated_at=d.get("generated_at"),
            tool=str(d.get("tool", "bench-summarize@1")),
        )


# --------------------------------------------------------------------- #
#  Winner selection + recommendation                                    #
# --------------------------------------------------------------------- #


def choose_winner(points: List[BenchPoint]) -> Dict:
    """The portable ``choice`` (§ 5.4): the fastest COMPLETED point by
    steady-state s/iter.  Returns ``{}`` if no point produced a time."""
    ranked = [p for p in points
              if p.state == "completed" and p.s_per_iter() is not None]
    if not ranked:
        return {}
    win = min(ranked, key=lambda p: p.s_per_iter())
    others = sorted((p for p in ranked if p is not win),
                    key=lambda p: p.s_per_iter())
    bits = [f"{win.label} fastest ({win.s_per_iter():g} s/iter)"]
    if win.bound:
        bits.append(f"{win.bound}-bound")
    if others:
        nxt = others[0]
        bits.append(f"vs {nxt.label} {nxt.s_per_iter():g} s/iter")
    return {"engine": win.engine, "knobs": dict(win.knobs),
            "rationale": "; ".join(bits)}


def recommend_resources(points: List[BenchPoint], choice: Dict, *,
                        mem_safety: float = 1.15,
                        prod_iters: int = 200,
                        time_safety: float = 1.5
                        ) -> Dict:
    """Production sizing from the MEASURED winner: ``mem_gb`` from its peak
    RSS x safety, and a suggested ``time`` from its s/iter x an assumed
    production iteration budget x safety (best-effort -- the real iteration
    count depends on the production SCF/relaxation, so it is a starting
    point, not a guarantee)."""
    if not choice:
        return {}
    win = next((p for p in points
                if p.engine == choice.get("engine")
                and p.knobs == choice.get("knobs")), None)
    rec: Dict = {}
    if win is not None:
        rss = win.metrics.get("peak_rss_gb")
        if isinstance(rss, (int, float)) and rss > 0:
            rec["mem_gb"] = math.ceil(rss * mem_safety)
        spi = win.s_per_iter()
        if spi:
            secs = int(spi * prod_iters * time_safety)
            d, rem = divmod(secs, 86400)
            h, rem = divmod(rem, 3600)
            m, s = divmod(rem, 60)
            rec["time"] = f"{d}-{h:02d}:{m:02d}:{s:02d}"
            rec["time_basis"] = (f"{spi:g}s/iter x {prod_iters} iters x "
                                 f"{time_safety} (adjust to your run)")
    return rec


def build_bench_result(points: List[BenchPoint], *,
                       environment: Optional[dict] = None,
                       system: Optional[dict] = None,
                       now_iso: Optional[str] = None) -> BenchResult:
    """Assemble the ``bench-result`` record from measured points."""
    choice = choose_winner(points)
    return BenchResult(
        environment=environment or {},
        system=system or {},
        points=list(points),
        choice=choice,
        recommend=recommend_resources(points, choice),
        generated_at=now_iso,
    )


__all__ = [
    "SCHEMA", "BenchPoint", "BenchResult",
    "parse_scf_timing", "parse_util_summary", "parse_util_csv_peak_mem",
    "parse_sacct_mem", "choose_winner", "recommend_resources",
    "build_bench_result",
]
