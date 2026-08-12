"""The sweep's reader — trials' artifacts → ``bench-result.json``.

Serves ``molbuilder jobset summarize bench`` (step 6 u4): discovery keyed
by ``job-set.json``'s own data, each trial parsed with the pure parsers in
:mod:`molbuilder.bench.result`, the verdict written as a recommendation.

The OLD half — ``discover_points``' directory-name regex,
``summarize_bundle``/``run_summarize`` over the shipped bundle format —
was DELETED 2026-08-12 (u5) with that lifecycle: the token is an
identifier, never a parser target (`job-contracts.md` § 6.3).
"""

from __future__ import annotations

import datetime
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
_RUN_IDX = re.compile(r"-run(\d+)\.")


def _read(path: Path, *, tail: Optional[int] = None,
          head: Optional[int] = None) -> str:
    """Whole file, or its first ``head`` / last ``tail`` bytes.  The split
    matters: a SIESTA ``.out`` announces its launch (the rank count) in
    the first KB and its fate (the end-of-run markers) in the last —
    reading one window for both answers one of them wrong."""
    try:
        if head is not None:
            with path.open("rb") as fh:
                return fh.read(head).decode("utf-8", "replace")
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

    # The .out is read TWICE, one window each, because its two answers
    # live at opposite ends: the end-of-run markers in the tail, and the
    # "Running on N nodes" launch HEADER in the first KB.  Until U11
    # (2026-08-12) the ranks were searched in the tail window, so any run
    # whose .out outgrew 16 KB -- i.e. any real run -- silently lost its
    # rank count, and the verdict's CPU half had no np.
    out = _latest_run_file(d, basename, "out")
    out_tail = _read(out, tail=16384) if out is not None else ""
    if out is None:
        state = "unknown"
    elif any(m in out_tail for m in _DONE_MARKERS):
        state = "completed"
    else:
        state = "incomplete"
    if engine == "cpu" and "ranks" not in knobs:
        n = (parse_mpi_ranks(_read(out, head=16384))
             if out is not None else None)
        if n is not None:
            knobs["ranks"] = n

    return BenchPoint(label=label, engine=engine, knobs=knobs,
                      metrics=metrics, bound=bound, state=state)



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


def discover_points_from_jobset(bundle, jobset,
                                stage: Optional[str] = None
                                ) -> List[BenchPoint]:
    """Discovery keyed by the DATA floor 3 wrote — the fold's reader
    (plan step 6 u4).

    Each trial's directory comes from the naming authority
    (``job_dir_names``) and its knobs from the job's own ``resources`` —
    never by parsing a directory name back (`job-contracts.md` § 6.3: the
    token is an identifier, not a parser target).  ``stage`` filters to the
    trials of one stage, read off each deck's own token.

    Its regex-keyed predecessor ``discover_points`` died with the OLD
    bundle format (u5).
    """
    from ..jobset.materialize import (_trial_stage_token, job_dir_names,
                                      shape_of)
    bundle = Path(bundle)
    dirs = job_dir_names(jobset, shape_of(jobset, bundle))
    pts: List[BenchPoint] = []
    for j in jobset.jobs:
        if stage is not None:
            tok = _trial_stage_token(jobset, j)
            if not tok or not tok.endswith(f"_{stage}"):
                continue
        knobs: Dict = {}
        if j.resources.mpi_np:
            knobs["ranks"] = j.resources.mpi_np
        if j.resources.cpus_per_task:
            knobs["cores_per_rank"] = j.resources.cpus_per_task
        if j.resources.gres:
            knobs["gres"] = j.resources.gres
        pts.append(parse_point(
            j.name, bundle / dirs[j.name], Path(j.script).stem,
            "gpu" if j.resources.gres else "cpu", knobs))
    return pts


def run_summarize_jobset(jobset, bundle, *, stage: Optional[str] = None,
                         out=None, now_iso: Optional[str] = None):
    """Summarize a described sweep through the data-keyed reader and write
    ``bench-result.json``; returns ``(BenchResult, out_path)``."""
    res = build_bench_result(
        discover_points_from_jobset(bundle, jobset, stage=stage),
        environment=_read_environment(Path(bundle)),
        system=_read_system(Path(bundle)),
        now_iso=now_iso)
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


def utc_now_iso() -> str:
    """UTC timestamp ``YYYY-MM-DDThh:mm:ssZ`` (moved from the deleted
    ``bench/prep.py`` at u5 -- its one surviving caller is the summarize
    verb's stamp)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "parse_point", "discover_points_from_jobset", "run_summarize_jobset",
    "summary_text", "utc_now_iso",
]
