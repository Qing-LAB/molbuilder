"""Benchmark results -> a portable ``bench-result`` record.

The decoupling point between the *bench* stage and the *run* stage
(docs/execution/job-system.md): summarize reads each
measured point's artifacts (timing log, utilization, peak memory) and
writes ``bench-result@1``.  Its ``choice`` block is the decision as DATA
(U13, 2026-08-12): the winner's ``label``, its ``knobs`` in the job-set's
own exchange vocabulary (mpi_np / cpus_per_task / gres), and its
``mechanism`` read from the winning trial's deck -- materialised by
`summarize` into the report it PRINTS for a PERSON
to apply (§ 2.3.2).

The pure ``parse_*`` functions take text; ``build_bench_result``
assembles the record.

NOTE (output isolation): a point is identified by its ``label``; the
caller hands each point its own artifacts.  Each trial runs in its own
``bench-<point>/`` directory inside the stage's ``bench/`` container
(job-contracts § 6.3), so points never clobber a shared basename --
`summarize` maps directories back to points through the job-set's own
data, never by parsing names.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

SCHEMA = "molbuilder/bench-result@1"


# --------------------------------------------------------------------- #
#  Pure parsers (text -> values; unit-tested)                           #
# --------------------------------------------------------------------- #



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
    the web payload both read this.  Points whose record predates the ``[MACHINE]`` line are not
    counted: absent is absent, never a kind called "?".
    """
    kinds: Dict[Tuple, List] = {}
    for p in points:
        k = machine_kind(getattr(p, "machine", None) or {})
        if k is not None:
            kinds.setdefault(k, [0, machine_brief(p.machine)])[0] += 1
    return [(brief, n) for _, (n, brief) in sorted(kinds.items())]


# What the run ACTUALLY used, printed by SIESTA itself (read through
# `parse/engines/siesta_grammar.py`) and by the wrapper (read through
# `wrapper_log.read_wrapper_log`, the lines' one reader).


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

    from molbuilder.wrapper_log import read_wrapper_log
    sections = read_wrapper_log(wrapper_log or "")
    wrap = sections[0] if sections else {}
    for key in ("node_phys_cores", "node_sockets", "node_cores_per_socket"):
        if key in wrap:
            eff[key] = wrap[key]

    # SIESTA's own launch and solver lines, through the family's one reader
    # of each (`parse/engines/siesta_grammar.py`: serial mode is one rank;
    # the block size is the orbital distribution's, `Src/initparallel.F`).
    from molbuilder.parse.engines import siesta_grammar as _G
    launch: Dict = {}
    solver: Dict = {}
    for line in (out_text or "").splitlines():
        _G.read_launch_line(line, launch) or _G.read_diag_line(line, solver)
    if "n_mpi_processes" in launch:
        eff["mpi_np"] = launch["n_mpi_processes"]
    if "threads" in wrap:
        # Only the thread count is taken from the wrapper.  Its rank count is
        # deliberately ignored -- see the docstring.
        eff["omp_threads"] = wrap["threads"]

    # The bench's own names for them (`bench-result@1`).
    for ours, key, facts in (("blocksize", "blocksize", launch),
                             ("diag_algorithm", "algorithm", solver),
                             ("elpa_gpu", "elpa_gpu", solver)):
        if key in facts:
            eff[ours] = facts[key]
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
    # receives one, which depends on the rank count -- the axis a sweep
    # varies.
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
    #: line (`parse/instruments/monitor.py`, `scheduler.md` R12).  Part of the
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
    steady-state s/iter -- or, when no point can win, ``{"none": <why>}``,
    the one sentence the terminal, the page and the ledger all say.

    **A trial that did not run what it was asked to run cannot win.**  Its
    time is real, but it measures a different configuration than its label
    claims -- a point asking for the GPU eigensolver that silently fell
    back to the CPU one would otherwise be compared against the GPU points
    as though it were one, and the recommendation drawn from that table
    would be advice about a machine nobody used.  Such points stay in the
    record (with their ``mismatch`` on their face) and are named in the
    rationale; they are only barred from winning.  If every timed point
    disagrees with its request, there is no winner -- the reason, rather
    than the least-wrong of them."""
    timed = [p for p in points
             if p.state == "completed" and p.s_per_iter() is not None]
    ranked = [p for p in timed if not p.mismatch]
    excluded = [p for p in timed if p.mismatch]
    if not timed:
        by_state: Dict[str, int] = {}
        for p in points:
            by_state[p.state] = by_state.get(p.state, 0) + 1
        census = ", ".join(f"{n} {s}" for s, n in sorted(by_state.items()))
        return {"none": f"no completed, timed trial to rank ({census}) -- "
                        f"launch the trials and summarize again"}
    if not ranked:
        return {"none": "every timed trial ran something other than it was "
                        "asked to -- "
                        + ", ".join(f"{p.label} [{mismatch_phrase(p.mismatch)}]"
                                    for p in excluded)
                        + ".  The times are real, but they do not measure "
                          "the settings on their labels: fix the cause and "
                          "re-run before trusting a choice"}
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
    # ``label`` is DATA, so anything needing the winning trial back -- the
    # mechanism read, a human's cross-check -- never parses the rationale
    # prose.  ``point`` rides for the same reason
    # (§ 4.3a): the winner's VALUE coordinates are reported
    # pins, and they must come from the record, not from its name.
    return {"label": win.label, "engine": win.engine,
            "knobs": dict(win.knobs), "point": dict(win.point),
            "rationale": "; ".join(bits)}


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
    "parse_sacct_mem",
    "parse_effective_run", "compare_asked_to_ran",
    "mismatch_phrase", "choose_winner",
    "build_bench_result",
]
