"""Benchmark results -> a portable ``bench-result`` record.

The decoupling point between the *bench* stage and the *run* stage
(docs/execution/job-system.md): summarize reads each
measured point's artifacts (timing log, utilization, peak memory) and
writes ``bench-result@1``.  Its ``choice`` block is the decision as DATA
(U13, 2026-08-12): the winner's ``label``, its ``knobs`` in the job-set's
own exchange vocabulary (mpi_np / cpus_per_task / gres), and its
``mechanism`` read from the winning trial's deck -- consumed by
`jobset prep run`'s verdict offer, which ASKS before applying
(§ 2.3.2).  The adapter re-resolution this paragraph used to describe
died with `prep-run` (step 6 u5).

**Stdlib-only** (parsers + json + dataclasses): ships to the target with
the rest of the prep layer.  The pure ``parse_*`` functions take text and
are unit-tested; ``build_bench_result`` assembles them.

NOTE (output isolation): a point is identified by its ``label``; the
caller hands each point its own artifacts.  Each trial runs in its own
``bench-<point>/`` directory inside the stage's ``bench/`` container
(job-contracts § 6.3), so points never clobber a shared basename --
`summarize` maps directories back to points through the job-set's own
data, never by parsing names (the ``adapters.format_bench`` sweep this
note used to cite died in step 6 u5).
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
    "mean of iters 3-5" rule (job-system.md § 7).  Returns
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


_MPI_RANKS = re.compile(r"Running on\s+(\d+)\s+nodes", re.IGNORECASE)


def parse_mpi_ranks(out_text: str) -> Optional[int]:
    """The MPI rank count from a SIESTA ``.out`` header line
    ``* Running on <N> nodes in parallel.`` (``None`` if absent).  Used to
    recover a CPU point's ``np`` -- it is set via ``sbatch -n`` and so is
    not in any filename."""
    m = _MPI_RANKS.search(out_text)
    return int(m.group(1)) if m else None


# What the run ACTUALLY used, printed by SIESTA itself and by the wrapper.
# Formats verified against real frozen output in
# tests/watch/fixtures/siesta_frozen/ and against the writers in SIESTA's
# Src/runinfo_m.F90:60, Src/initparallel.F:256 and Src/diag_option.F90:385ff.
_PROCY_BS = re.compile(r"^\s*\*\s*ProcessorY,\s*Blocksize:\s*(\d+)\s+(\d+)",
                       re.MULTILINE)
_DIAG_ALG = re.compile(r"^\s*diag:\s*Algorithm\s+=\s*(\S+)", re.MULTILINE)
_ELPA_GPU = re.compile(r"^\s*diag:\s*ELPA GPU string key\s+=\s*(\S+)",
                       re.MULTILINE)
#: The wrapper's own record of the launch it resolved (``runwrap.py``'s
#: ``_log INFO "ranks / omp     : $_mpi_np ranks x $_omp_threads OMP threads"``).
#: The thread count exists NOWHERE in SIESTA's output, so this is its only
#: witness.
_WRAP_RANKS_OMP = re.compile(
    r"ranks\s*/\s*omp\s*:\s*(\d+)\s+ranks\s+x\s+(\d+)\s+OMP threads")


def parse_effective_run(out_text: str = "", wrapper_log: str = "") -> Dict:
    """What a trial REALLY ran with, read back from its own artifacts.

    A benchmark that records the settings it *asked for* measures a table
    of labels, not of runs: SIESTA falls back silently (an unavailable
    ELPA/GPU build drops to the CPU solver and says so only in its
    output), a launcher may hand back fewer ranks than requested, and
    ``OMP_NUM_THREADS`` set in the environment overrides the scheduler's
    ``-c``.  Any of those makes a row describe a run that did not happen.

    Returns only the keys it could actually find, so a caller can tell
    *"checked and matched"* from *"could not check"*:

    ``mpi_np``          MPI ranks -- **SIESTA's own** ``* Running on N
                        nodes in parallel.``, and nowhere else.  The
                        wrapper logs a rank count too, but it writes that
                        line BEFORE it launches, so it records what the
                        wrapper INTENDED; MPI can hand back fewer.  Using
                        it as a fallback would let an intention be
                        recorded as an observation -- the exact
                        conflation this function exists to prevent -- and
                        it would agree with the request by construction.
    ``omp_threads``     OMP threads per rank -- the wrapper's log only,
                        since no SIESTA output states it.  This one IS
                        the wrapper's resolved value, but the wrapper
                        exports it in the same script that runs the
                        engine, so it is what the process actually got.
    ``blocksize``       the ScaLAPACK/ELPA block size SIESTA settled on.
    ``diag_algorithm``  the eigensolver actually used (``ELPA-2stage``,
                        ``D&C``, ...) -- the fallback witness.
    ``elpa_gpu``        ELPA's GPU string key, printed only by an
                        ELPA-enabled build that reached the ELPA path.

    Pure text in, values out -- both texts are optional, and an empty or
    unparsable one contributes nothing rather than raising.
    """
    eff: Dict = {}

    ranks = parse_mpi_ranks(out_text)
    if ranks is not None:
        eff["mpi_np"] = ranks
    m = _WRAP_RANKS_OMP.search(wrapper_log)
    if m:
        # Only the thread count is taken from the wrapper.  Its rank
        # count (group 1) is deliberately ignored -- see the docstring.
        eff["omp_threads"] = int(m.group(2))

    m = _PROCY_BS.search(out_text)
    if m:
        # The line prints ProcessorY first, then Blocksize.
        eff["blocksize"] = int(m.group(2))
    m = _DIAG_ALG.search(out_text)
    if m:
        eff["diag_algorithm"] = m.group(1)
    m = _ELPA_GPU.search(out_text)
    if m:
        eff["elpa_gpu"] = m.group(1)
    return eff


def compare_asked_to_ran(asked: Dict, effective: Dict) -> Dict:
    """Where a trial's run disagrees with what it was asked to do.

    ``{knob: {"asked": <requested>, "ran": <observed>}}`` for every knob
    present on BOTH sides and unequal; ``{}`` when everything comparable
    agreed, or when nothing could be compared.  A knob only one side
    knows about is not a disagreement -- it is an unanswered question,
    and silence is the honest answer.

    ``cpus_per_task`` is the scheduler's cores-per-rank and
    ``omp_threads`` is what the wrapper set from it (``runwrap.py``
    resolves ``OMP_NUM_THREADS`` → ``SLURM_CPUS_PER_TASK``), so they are
    the same question asked of two layers and are compared as one.
    Eigensolver names are compared case-blind because the deck shouts
    (``ELPA-1STAGE``) where SIESTA prints mixed case (``ELPA-1stage``).
    """
    # ``blocksize`` is deliberately NOT here.  It is read back and kept
    # in ``effective`` -- it is real measured data and belongs in the
    # record -- but SIESTA ADJUSTING it is normal, not a fault:
    # ``initparallel.F`` shrinks the requested block so every rank
    # receives one, and ELPA's GPU path rounds it up to a power of two.
    # Both depend on the rank count, which is the axis a sweep varies.
    # Comparing it would therefore mark most trials of a small system as
    # "ran something other than asked" and bar them from winning, which
    # would leave a legitimate benchmark with no winner at all.  A
    # mismatch here must mean the trial's LABEL is a lie, not that the
    # engine adapted a tuning parameter the way it documents.
    pairs = (("mpi_np", "mpi_np"),
             ("cpus_per_task", "omp_threads"),
             ("diag_algorithm", "diag_algorithm"))
    out: Dict = {}
    for asked_key, ran_key in pairs:
        a, r = asked.get(asked_key), effective.get(ran_key)
        if a is None or r is None:
            continue
        if isinstance(a, str) and isinstance(r, str):
            same = a.strip().lower() == r.strip().lower()
        else:
            same = a == r
        if not same:
            out[ran_key] = {"asked": a, "ran": r}
    return out


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
    #: What the run ACTUALLY used, read back from its own artifacts
    #: (:func:`parse_effective_run`).  Empty when nothing could be read.
    effective: Dict = field(default_factory=dict)
    #: Where ``effective`` disagrees with ``knobs`` --
    #: ``{knob: {"asked": x, "ran": y}}``.  Empty means everything
    #: comparable agreed, OR that nothing was comparable; ``effective``
    #: is what tells those two apart.
    mismatch: Dict = field(default_factory=dict)

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
        from ..persist import check_schema
        check_schema(str(d.get("schema", "")), SCHEMA,
                           label="bench-result")
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


def mismatch_phrase(mismatch: Dict) -> str:
    """``{"mpi_np": {"asked": 8, "ran": 4}}`` -> ``"mpi_np asked 8, ran 4"``
    -- one readable clause per disagreeing knob."""
    return "; ".join(f"{k} asked {v.get('asked')}, ran {v.get('ran')}"
                     for k, v in sorted(mismatch.items()))


def choose_winner(points: List[BenchPoint]) -> Dict:
    """The portable ``choice`` (§ 5.4): the fastest COMPLETED point by
    steady-state s/iter.  Returns ``{}`` if no point produced a time.

    **A trial that did not run what it was asked to run cannot win.**  Its
    time is real, but it measures a different configuration than its label
    claims -- a point asking for the GPU eigensolver that silently fell
    back to the CPU one would otherwise be compared against the GPU points
    as though it were one, and the recommendation drawn from that table
    would be advice about a machine nobody used.  Such points stay in the
    record (with their ``mismatch`` on their face) and are named in the
    rationale; they are only barred from winning.  If every timed point
    disagrees with its request, there is no winner -- ``{}`` rather than
    the least-wrong of them."""
    timed = [p for p in points
             if p.state == "completed" and p.s_per_iter() is not None]
    ranked = [p for p in timed if not p.mismatch]
    excluded = [p for p in timed if p.mismatch]
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
    if excluded:
        bits.append("excluded (ran something other than asked): "
                    + ", ".join(f"{p.label} [{mismatch_phrase(p.mismatch)}]"
                                for p in excluded))
    # ``label`` is DATA (U13, 2026-08-12): it identified the winner only
    # inside the rationale prose, so anything needing the winning trial
    # back -- the mechanism read, a human's cross-check -- had to parse a
    # sentence.  The id is the id.
    return {"label": win.label, "engine": win.engine,
            "knobs": dict(win.knobs), "rationale": "; ".join(bits)}


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
    "parse_sacct_mem", "parse_mpi_ranks",
    "parse_effective_run", "compare_asked_to_ran",
    "mismatch_phrase", "choose_winner",
    "recommend_resources", "build_bench_result",
]
