"""``bench summarize`` -- read a run sweep's artifacts -> ``bench-result``.

Step 3 of the target side (benchmark-workflow.md § 7.4): walk the
per-point ``point-G<g>K<k>/`` directories the sweep produced (§ 7.3
isolation), parse each point's timing / utilization / state with the pure
parsers in :mod:`molbuilder.bench.result`, and write ``bench-result.json``
(§ 5.3) -- the only input ``prep-run`` needs.

**Stdlib-only** (imports the stdlib ``result`` model + parsers): ships to
the target.  ``summarize_bundle`` is the testable core; ``main`` is the
standalone entry; the ``molbuilder bench summarize`` CLI calls the core.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .result import (
    BenchPoint, BenchResult, build_bench_result, parse_mpi_ranks,
    parse_scf_timing, parse_util_csv_peak_mem, parse_util_summary,
)

# "the run finished" markers in a SIESTA .out (best-effort; a capped bench
# with SCF.MustConverge .false. still exits cleanly and prints these).
_DONE_MARKERS = ("Job completed", "End of run", "siesta: Final energy",
                 ">> End of run:")
_POINT_RE = re.compile(r"^point-G(\d+)K(\d+)$")
_RUN_IDX = re.compile(r"-run(\d+)\.")


def _read(path: Path, *, tail: Optional[int] = None) -> str:
    try:
        if tail is None:
            return path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > tail:
                fh.seek(-tail, 2)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _latest_run_file(d: Path, basename: str, suffix: str) -> Optional[Path]:
    """The ``<basename>-runN.<suffix>`` with the highest run index N."""
    cands = list(d.glob(f"{basename}-run*.{suffix}"))
    if not cands:
        return None

    def _idx(p: Path) -> int:
        m = _RUN_IDX.search(p.name)
        return int(m.group(1)) if m else -1
    return max(cands, key=_idx)


def parse_point(label: str, d: Path, basename: str, engine: str,
                knobs: Dict) -> BenchPoint:
    """Parse one point's artifacts (in directory ``d``, output basename
    ``basename``) into a :class:`BenchPoint`."""
    metrics: Dict = {}
    knobs = dict(knobs)

    timing = _latest_run_file(d, basename, "scf-timing.log")
    if timing is not None:
        metrics.update(parse_scf_timing(_read(timing)))

    bound = None
    mon = d / f"{basename}.monitor.log"
    if mon.is_file():
        u = parse_util_summary(_read(mon))
        bound = u.get("bound")
        for key in ("gpu_sm_mean_pct", "cpu_mean_pct"):
            if u.get(key) is not None:
                metrics[key] = u[key]

    util = d / f"{basename}.util.csv"
    if util.is_file():
        peak = parse_util_csv_peak_mem(_read(util))
        if peak is not None:
            metrics["peak_rss_gb"] = peak

    # Read the .out once: end-of-run state + (CPU) the np from the
    # "Running on N nodes" header, which no filename records.
    out = _latest_run_file(d, basename, "out")
    out_tail = _read(out, tail=16384) if out is not None else ""
    if out is None:
        state = "unknown"
    elif any(m in out_tail for m in _DONE_MARKERS):
        state = "completed"
    else:
        state = "incomplete"
    if engine == "cpu" and "ranks" not in knobs:
        n = parse_mpi_ranks(out_tail)
        if n is not None:
            knobs["ranks"] = n

    return BenchPoint(label=label, engine=engine, knobs=knobs,
                      metrics=metrics, bound=bound, state=state)


def discover_points(bundle) -> List[BenchPoint]:
    """Find + parse every sweep point: the GPU ``point-G<g>K<k>/`` dirs,
    plus a single CPU run (``job-cpu-*`` in the bundle root) if present."""
    bundle = Path(bundle)
    pts: List[BenchPoint] = []
    for d in sorted(p for p in bundle.glob("point-G*K*") if p.is_dir()):
        m = _POINT_RE.match(d.name)
        if not m:
            continue
        pts.append(parse_point(
            d.name, d, "job-gpu", "gpu",
            {"gpus": int(m.group(1)), "ranks_per_gpu": int(m.group(2))}))
    if list(bundle.glob("job-cpu-run*.out")):
        # the CPU bench is a single root-level run; np isn't recorded in
        # the filenames (set via sbatch -n), so knobs stay empty.
        pts.append(parse_point("cpu", bundle, "job-cpu", "cpu", {}))
    return pts


def _read_environment(bundle: Path) -> Dict:
    p = Path(bundle) / "environment.json"
    if p.is_file():
        try:
            return json.loads(_read(p))
        except ValueError:
            return {}
    return {}


def _norm(key: str) -> str:
    return key.lower().replace(".", "").replace("-", "").replace("_", "")


def _read_system(bundle: Path) -> Dict:
    """Minimal system descriptor from the fdf (engine + NumberOfAtoms)."""
    sysd: Dict = {"engine": "siesta"}
    for name in ("job-gpu.fdf", "job-cpu.fdf"):
        fdf = Path(bundle) / name
        if not fdf.is_file():
            continue
        for line in _read(fdf).splitlines():
            toks = line.split("#", 1)[0].split()
            if len(toks) >= 2 and _norm(toks[0]) == "numberofatoms":
                try:
                    sysd["n_atoms"] = int(float(toks[1]))
                except ValueError:
                    pass
                break
        break
    return sysd


def summarize_bundle(bundle, *, now_iso: Optional[str] = None) -> BenchResult:
    """Assemble the ``bench-result`` for a run bundle directory."""
    return build_bench_result(
        discover_points(bundle),
        environment=_read_environment(bundle),
        system=_read_system(bundle),
        now_iso=now_iso)


def run_summarize(bundle, *, out=None, now_iso: Optional[str] = None):
    """Summarize ``bundle`` and write ``bench-result.json``; returns
    ``(BenchResult, out_path)``."""
    res = summarize_bundle(bundle, now_iso=now_iso)
    out_path = Path(out) if out else Path(bundle) / "bench-result.json"
    out_path.write_text(res.to_json() + "\n", encoding="utf-8")
    return res, out_path


def summary_text(res: BenchResult, out_path: Path) -> str:
    lines = ["bench-summarize: ranked points (fastest first)"]
    ranked = sorted(
        res.points,
        key=lambda p: (p.s_per_iter() if p.s_per_iter() is not None
                       else float("inf")))
    for p in ranked:
        spi = p.s_per_iter()
        spi_s = f"{spi:g}s/iter" if spi is not None else "n/a"
        lines.append(f"  {p.label:<12} {p.engine:<4} {spi_s:<12} "
                     f"{p.state:<11} bound={p.bound}")
    if res.choice:
        lines.append(f"  winner: {res.choice.get('rationale')}")
    if res.recommend:
        lines.append(f"  recommend: {res.recommend}")
    lines.append(f"  wrote: {out_path}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="bench-summarize",
        description="Read a benchmark sweep's outputs (point-G*K*/ dirs) "
                    "and write bench-result.json.")
    p.add_argument("--bundle", default=".", help="bundle dir (default: .)")
    p.add_argument("--out", default=None,
                   help="output path (default: <bundle>/bench-result.json)")
    a = p.parse_args(argv)
    from .prep import utc_now_iso
    res, out_path = run_summarize(a.bundle, out=a.out, now_iso=utc_now_iso())
    print(summary_text(res, out_path))
    return 0


__all__ = [
    "parse_point", "discover_points", "summarize_bundle", "run_summarize",
    "summary_text", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
