"""Benchmark results -> a portable ``bench-result`` record.

The decoupling point between the *bench* stage and the *run* stage
(docs/execution/job-system.md): summarize reads each
measured point's artifacts (timing log, utilization, peak memory) and
writes ``bench-result@1``.  Its ``choice`` block is the decision as DATA
(U13, 2026-08-12): the winner's ``label``, its ``knobs`` in the job-set's
own exchange vocabulary (mpi_np / cpus_per_task / gres), and its
``mechanism`` read from the winning trial's deck -- materialised by
`summarize` into the report ``bench-recommendation.txt`` that a PERSON
applies (§ 2.3.2; interactive ask until 2026-08-19).  The adapter re-resolution this paragraph used to describe
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
from typing import Dict, List, Optional, Tuple

SCHEMA = "molbuilder/bench-result@1"


# --------------------------------------------------------------------- #
#  Pure parsers (text -> values; unit-tested)                           #
# --------------------------------------------------------------------- #


def parse_scf_timing(text: str) -> Dict[str, Optional[float]]:
    """Steady-state seconds/iteration from a ``.scf-timing.log``.

    Each line starts with an epoch timestamp; consecutive deltas are the
    per-iteration durations.  We drop the first delta (iter 1->2, which can
    still carry warm-up) when >=2 deltas are present and average the rest
    (job-system.md § 7).  Under the capped 3-iteration trial (2026-08-21)
    that is ONE clean sample -- iteration 3 -- which the user's measured
    experience says reads the scaling story as well as the older
    5-iteration three-sample mean; records of either shape parse the same
    way.  Returns ``{"s_per_iter": float|None, "iters_measured": int}``.
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


_VERDICT = re.compile(r"->\s*(GPU-bound|host/CPU-bound|mixed)")
_BOUND_MAP = {"GPU-bound": "gpu", "host/CPU-bound": "host", "mixed": "mixed"}


def parse_util_bound(monitor_log: str) -> Optional[str]:
    """The bound VERDICT (``gpu`` | ``host`` | ``mixed``) from the
    ``[UTIL-SUMMARY]`` line of a ``.monitor.log``, or ``None``.

    The verdict is the monitor's own call — it encodes the monitor's
    thresholds, so it is read from the monitor and nowhere else.  The
    utilisation NUMBERS on the same line are deliberately NOT read here
    (2026-08-19), for two reasons that still hold: the summary line can
    be **missing entirely** — it is written only in the monitor's
    terminal branch, so a trial killed by the scheduler leaves a csv and
    no summary — and two readers of one fact had already diverged once.

    **The two are not the same number, and this used to claim they
    were** (corrected 2026-09-03).  It said the summary is *"a digest of
    the same samples util.csv records raw"*.  It is not: the monitor
    accumulates EVERY tick (``util_accum.add`` runs unconditionally)
    while the csv is **change-gated** — a row lands only when a metric
    moved ≥ 10% or a 300 s keepalive elapsed (``monitor.py``, the
    util-sampling header).  So the summary's mean is exact over every
    sample and :func:`parse_util_csv`'s is a reconstruction over a
    subset.  Preferring the summary when it exists and falling back to
    the csv would get both exactness and coverage — that is a design
    call against the two-readers lesson above, not a cleanup, and it is
    `plan.md` **E2**.
    """
    line = ""
    for ln in monitor_log.splitlines():
        if "[UTIL-SUMMARY]" in ln:
            line = ln                                # last one wins
    v = _VERDICT.search(line)
    return _BOUND_MAP.get(v.group(1)) if v else None


_MACHINE = re.compile(
    r"\[MACHINE\]\s+node=(\S+)\s+cores=(\S+)\s+mem_gb=(\S+)\s+gpu=(.+)$")


def parse_machine(monitor_log: str) -> Dict[str, str]:
    """The ``[MACHINE]`` line of a ``.monitor.log`` — what kind of node was
    under this run (`scheduler.md` R12) — or ``{}`` when the log predates it.

    The monitor writes it once, first line, at start; the FIRST match wins
    so a corrupted or concatenated log still reads the start record.  The
    ``gpu`` field is everything after ``gpu=`` because device models
    contain spaces (the monitor's ``machine_line`` puts it last for
    exactly this reason).
    """
    for ln in monitor_log.splitlines():
        m = _MACHINE.search(ln)
        if m:
            return {"node": m.group(1), "cores": m.group(2),
                    "mem_gb": m.group(3), "gpu": m.group(4).strip()}
    return {}


def machine_kind(machine: Dict) -> Optional[Tuple[str, str, str]]:
    """What makes two nodes THE SAME MACHINE for comparison — R11's kind.

    ``(cores, memory to the nearest 10 GB, device models)``, and **never
    the node name**: SLURM spreads a sweep over whatever boxes are free,
    so comparing names would report "different machines" for every sweep
    ever run (trap T1).

    Memory is rounded because identical boxes genuinely jitter: Sol's
    128-core standard nodes report MemTotal as 503.2, 503.4 and 503.5 GB
    (BIOS reservations differ), while the tiers that really differ are
    hundreds of GB apart.  ``None`` when the record predates the
    ``[MACHINE]`` line — *cannot tell* is not a kind, and a reader must
    not compare it against one (`scheduler.md` R3).
    """
    if not machine:
        return None
    try:
        mem = f"{round(float(machine.get('mem_gb', '?')) / 10) * 10:g}"
    except ValueError:
        mem = "?"
    return (str(machine.get("cores", "?")), mem,
            str(machine.get("gpu", "?")))


def machine_brief(machine: Dict) -> str:
    """One short human spelling of a machine's KIND -- ``48c 500G A100`` --
    used by every surface that names one, so the table and the web page
    cannot drift into two spellings of one node.

    Built from :func:`machine_kind` (same rounding, same fields) with the
    vendor prefix and the form-factor tail trimmed from the device --
    ``NVIDIA A100-SXM4-80GB`` and ``NVIDIA A100-PCIE-40GB`` are different
    THINGS but the same reading effort, and the full model stays in the
    record for anyone who needs the distinction.  ``""`` when the record
    predates the ``[MACHINE]`` line: absent is absent, never ``"?c ?G"``.
    """
    kind = machine_kind(machine)
    if kind is None:
        return ""
    cores, mem, gpu = kind
    if gpu in ("none", "?"):
        dev = "no gpu"
    else:
        names = []
        for g in gpu.split(", "):
            g = re.sub(r"^(NVIDIA|AMD|Intel)\s+", "", g)
            names.append(g.split("-", 1)[0] or g)
        dev = "+".join(names)
    return f"{cores}c {mem}G {dev}"


def machine_census(points) -> List[Tuple[str, int]]:
    """``[(brief, how many trials)]``, one entry per KIND, sorted stably.

    THE one grouping of a sweep's machines -- the terminal statement and
    the web payload both read this, because the census was briefly grouped
    twice (once per surface) and two spellings of "which machines are in
    play" is the exact drift `machine_brief` exists to prevent one level
    down.  Points whose record predates the ``[MACHINE]`` line are not
    counted: absent is absent, never a kind called "?".
    """
    kinds: Dict[Tuple, List] = {}
    for p in points:
        k = machine_kind(getattr(p, "machine", None) or {})
        if k is not None:
            kinds.setdefault(k, [0, machine_brief(p.machine)])[0] += 1
    return [(brief, n) for _, (n, brief) in sorted(kinds.items())]


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


def parse_util_csv(csv_text: str) -> Dict[str, float]:
    """One reader for the monitor's raw samples (``util.csv``): peak
    RSS, sampled wall window, mean CPU%, per-GPU mean SM% (max across
    GPUs) and peak VRAM.

    Returns only the keys it could derive — an empty dict for an empty
    or headerless file — so a caller can fold the result straight into a
    point's metrics.  Keys: ``peak_rss_gb``, ``wall_s`` (last sample −
    first sample; the monitor runs for the life of the job, so this is
    the job's wall time to sampling resolution), ``cpu_mean_pct``,
    ``gpu_sm_mean_pct``, ``gpu_vram_peak_gb``.

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


#: The wrapper's own measurement of the NODE it landed on, logged before it
#: launches: ``molbuilder: detected phys_cores=48, n_sockets=2,
#: cores_per_socket=24``.  `lscpu -p=Core,Socket` ignores the affinity mask
#: (verified), so these are the node's PHYSICAL cores, not the allocation's.
_WRAP_NODE = re.compile(
    r"detected\s+phys_cores=(\d+),\s*n_sockets=(\d+),\s*"
    r"cores_per_socket=(\d+)")

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
    ``node_phys_cores`` the cores of the NODE THIS TRIAL RAN ON, measured
                        there by the wrapper.  It is here because the
                        probed ``environment.json`` cannot answer it: that
                        record describes ONE node of a partition
                        (``scontrol show node <picked>``), faithfully, and
                        a heterogeneous partition has no single shape.  A
                        sweep whose trials landed on different node types
                        can now say so instead of quoting a number none of
                        them used (Au-BDT-Au ran on a 2x24 node while the
                        record said 2x32).

    Pure text in, values out -- both texts are optional, and an empty or
    unparsable one contributes nothing rather than raising.
    """
    eff: Dict = {}

    node = _WRAP_NODE.search(wrapper_log or "")
    if node:
        eff["node_phys_cores"] = int(node.group(1))
        eff["node_sockets"] = int(node.group(2))
        eff["node_cores_per_socket"] = int(node.group(3))

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
    #: The trial's sweep coordinate, read from `job-set.json`'s per-trial
    #: ``point`` (data, never parsed from the label -- job-contracts.md
    #: § 6.3).  Carries the VALUE coordinates a 2β sweep declared
    #: (`generator.md` § 4.3a); ``{}`` for pre-2β records.
    point:   Dict = field(default_factory=dict)
    #: What kind of node was under the run -- the monitor's ``[MACHINE]``
    #: line (:func:`parse_machine`, `scheduler.md` R12).  Part of the
    #: measurement, not metadata about it: on a queue holding many machine
    #: types, two trials of one sweep can land on different hardware, and
    #: a summary that hides which is comparing silently
    #: (`generator.md` § 4.4b).  ``{}`` when the log predates the line.
    machine: Dict = field(default_factory=dict)

    def s_per_iter(self) -> Optional[float]:
        v = self.metrics.get("s_per_iter")
        return v if isinstance(v, (int, float)) else None


@dataclass
class BenchResult:
    environment: Dict = field(default_factory=dict)
    system:      Dict = field(default_factory=dict)
    points:      List[BenchPoint] = field(default_factory=list)
    choice:      Dict = field(default_factory=dict)
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
            generated_at=d.get("generated_at"),
            tool=str(d.get("tool", "bench-summarize@1")),
        )


# --------------------------------------------------------------------- #
#  Winner selection                                                     #
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
    # sentence.  The id is the id.  ``point`` rides for the same reason
    # (§ 4.3a): the winner's VALUE coordinates are reported
    # pins, and they must come from the record, not from its name.
    return {"label": win.label, "engine": win.engine,
            "knobs": dict(win.knobs), "point": dict(win.point),
            "rationale": "; ".join(bits)}


# `recommend_resources` was DELETED 2026-08-24 (user).  It produced
# ``mem_gb = peak RSS x 1.15`` and ``time = s/iter x prod_iters x 1.5`` --
# a safety factor and a production iteration count NOBODY CHOSE, the second
# a default sitting in its own signature.  `summarize` wrote both into
# `run-config.toml`, `prep` folded them in when no flag said otherwise, and
# they reached `sbatch`: the estimation purge's own target, surviving in the
# one path it did not sweep.  What a sweep recommends is what it MEASURED --
# the winning configuration and its knobs; the wall and the memory are the
# person's to state (`execution/submission.md` S1, S2).


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
        generated_at=now_iso,
    )


__all__ = [
    "SCHEMA", "BenchPoint", "BenchResult",
    "parse_scf_timing", "parse_util_bound", "parse_util_csv",
    "parse_sacct_mem", "parse_mpi_ranks",
    "parse_effective_run", "compare_asked_to_ran",
    "mismatch_phrase", "choose_winner",
    "build_bench_result",
]
