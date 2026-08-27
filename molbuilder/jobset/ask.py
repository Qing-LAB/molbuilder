"""What this job needs — asked once, answered once, shown once.

**One question, one answer, one interface, one output** (user, 2026-08-23).

A scientist knows what their calculation needs better than any rule this
framework can write.  So it asks, rather than deriving: *how much time, how
much memory.*  Everything else follows from the answer, and nothing has to be
explained afterwards because nothing was invented.

**What this replaces.**  The numbers used to arrive by themselves.  A job
asked for 128 GB because SLURM grants 2 GB a core and it had 64 of them; it
asked for 38 minutes because a per-trial default nobody set was multiplied by
a trial count nobody saw.  Both were arithmetic on inputs the person had never
been offered.  The first attempt at a fix was a provenance system — five
categories, an announcement rule, a display — machinery whose entire purpose
was to cope with numbers nobody chose.  **Asking removes the problem instead
of labelling it.**

Everything that lives here, and it lives nowhere else:

    Ask              the question, and the answer to it
    queue_table      the queues this machine offers, and which can take the job
    gpu_share_notes  what a GPU-sharing request means, stated once
    confirm          the one interface — approve, change, or skip

*This list said "four things" and named three of them until 2026-08-24.*  An
inventory that does not match the module is how a resident goes unnoticed:
`parse_duration` and `parse_memory` lived here for months without appearing
on it, and being in a job-submission package is what stopped `task.py` --
which only wanted to canonicalise a time string -- from reaching them
without importing the whole of `jobset`.  They are now in
`scheduler/quantities.py`, with every other dialect of the same two
quantities (`docs/design.md`, "Architecture": an L1 module is named for the
object it owns).

    The one output is the PLAN the launch door prints -- the exact sbatch
    command of every job, from the same code that submits it.  A `render`
    summary lived here until 2026-08-24 and described a submission that
    never happens (it ignored the per-shelf split); with it went
    `bench_bound`/`bench_total` and their slack/startup constants -- TIME
    IS NEVER DERIVED (user dictation, 2026-08-24): the user states it, or
    the target queue's own ceiling stands.

A `fits(ask, rows)` sat here until it was reviewed and found to be **a third
implementation of "does this fit"** — `admits` is the one check, placement
uses it, and `queue_table` renders it per queue.  It was documented as *"the
whole point"* and called by nobody.  A check designed and not wired is the
defect this file exists to remove, one layer up.

The CLI and the browser call the same four.  Two surfaces asking one question
two ways is how they come to disagree about what was asked.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# The two quantities a job asks for -- a wall and an amount of memory -- and
# every dialect each is written in, live in `scheduler/quantities.py`
# (2026-08-24).  They sat HERE until then, in a module whose own docstring
# lists what lives in it and never mentioned them: put where they were first
# needed rather than where they belong, which is what forced `task.py` to
# import this whole job-submission package to canonicalise a time string.
# A codec on a basic unit is L1; this module is L2 (`docs/design.md`).
from ..scheduler.quantities import human_wall

#: MPS's own hard ceiling -- more than this many processes cannot even
#: ATTACH to one device (`engines/tuning.md` § 2.12, citing
#: `references.bib: NvidiaMPS`, the A100-generation figure).  A site FACT
#: cited from the vendor, not an estimate -- the estimation purge of
#: 2026-08-24 deliberately kept it.
MPS_MAX_CLIENTS_PER_DEVICE = 48

#: This stack's tuned point, ~4 ranks/GPU without NCCL (`engines/tuning.md`
#: § 2.12) -- a MEASURED literature figure, also kept: both feed
#: `gpu_share_notes`, which INFORMS and never decides.
GPU_TUNED_RANKS_PER_DEVICE = 4


@dataclass(frozen=True)
class Ask:
    """What one job needs, as the person said it.

    ``None`` means *not answered* — never zero, never a default wearing a
    number's clothes.  A caller that must have a value asks; a caller that can
    proceed without one says so.
    """
    time_s: Optional[int] = None
    mem_gb: Optional[float] = None

    def __bool__(self) -> bool:
        return self.time_s is not None or self.mem_gb is not None


def queue_table(rows: Sequence, ask: Ask, *, cores: Optional[int] = None,
                gpu: bool = False) -> str:
    """**Every queue this machine offers, and which of them can take this job.**

    The framework does not choose (user, 2026-08-23: *"just don't guess
    shit"*).  It shows what exists, marks what fits, and the person picks with
    ``--domain``.  Which queue to spend a day of wall-clock in is a judgement
    about priority, contention and what else is running — none of which is on
    this machine's record, and all of which the person has.

    A queue that cannot take the job is still LISTED, with the reason.
    Hiding it would answer *"why is my queue not an option?"* with silence,
    and that question has a real answer worth reading.

    **Each row is up to four lines**, and the last three appear only when
    they have something to say::

        2  htc   htc/public   4h   48-128   -   a100 x4, a100.20gb x16
             - 128 cores  x134 node(s)                    <- the machines,
             - 64 cores  x3 node(s)  a100.20gb x16           when there is
             - 48 cores  x51 node(s)  a100 x4  (too small)   more than one
               64 cores fits 137 of 188 nodes here (72%)  <- given an ask
           -> needs 240 min but debug allows 00:15:00      <- why not

    ``cores`` is the **maximum core range** (:func:`core_range`) -- one
    number when every machine is the same size.  The fit line
    (:func:`fits_how_many`) is there because the range alone misleads: on
    ``htc``, 128 looks like the rare extreme and is 134 of 188 nodes.
    """
    from ..scheduler import domain_ceiling_s
    if not rows:
        return "this machine states no queues -- the job runs directly."
    head = (f"  {'':2} {'name':<12} {'partition/qos':<22} {'max time':>9} "
            f"{'cores':>8} {'memory':>9}  gpu")
    lines = ["this machine offers:", head]
    for i, d in enumerate(rows, 1):
        secs = domain_ceiling_s(d)
        wall = human_wall(secs)
        mem = f"{float(d.max_mem_gb):g} GB" if d.max_mem_gb else "-"
        dev = ", ".join(f"{x.type} x{x.per_node}" for x in d.devices) or "-"
        why = _why_not(d, ask, cores=cores, gpu=gpu)
        mark = "  " if not why else "! "
        lines.append(
            f"{mark}{i:<2} {d.name:<12} {d.partition + '/' + d.qos:<22} "
            f"{wall:>9} {core_range(d):>8} {mem:>9}  {dev}")
        for shape in _machine_lines(d, cores=cores):
            lines.append(f"       {shape}")
        fit = fits_how_many(d, cores)
        if fit:
            lines.append(f"         {fit}")
        if why:
            lines.append(f"     -> {'; '.join(why)}")
    lines.append("")
    lines.append("  choose one with --domain <name>.  Nothing is submitted "
                 "until you do.")
    return "\n".join(lines)


def core_range(row) -> str:
    """This domain's **maximum core range**, for the table's cores column.

    The arithmetic is `scheduler.quantities.core_range` -- ONE spelling, so
    the queue table here and the browser's machine card cannot disagree
    about how a range is written.  This wrapper adds only the fallback: a
    record with no ``node_types`` (every record written before 2026-08-27)
    still has ``max_cores``, and R3 says an unstated fact never bars.
    """
    from ..scheduler.quantities import core_range as _range
    shapes = [t.get("cores") for t in (getattr(row, "node_types", None) or [])]
    return _range(shapes) or str(row.max_cores or "-")


def fits_how_many(row, cores: Optional[int]) -> str:
    """How much of this queue an ask of ``cores`` could actually land on.

    **The range alone misleads, and the count is what makes it a
    decision.**  Reading ``48-128`` you would take 128 for the rare
    extreme; on Sol's ``htc`` it is 134 of 188 nodes -- the COMMON machine,
    with the 48-core GPU nodes in the minority.  So a large CPU ask there
    costs almost nothing in scheduling, which is the opposite of what the
    range implies on its own.

    Nothing is printed without an ask to measure, or when every machine
    fits: a line saying *all of them* on every row is noise.
    """
    shapes = [t for t in (getattr(row, "node_types", None) or [])
              if t.get("cores") and t.get("nodes")]
    if not cores or not shapes:
        return ""
    total = sum(t["nodes"] for t in shapes)
    fit = sum(t["nodes"] for t in shapes if t["cores"] >= cores)
    if fit == total:
        return ""
    # NO LEADING ARROW.  `->` already means *this queue is refused* one
    # indent out, and two arrows at two indents saying different things is
    # a table you have to decode.  This is a SUMMARY of the machines listed
    # directly above it, so it sits with them and reads as their total.
    if not fit:
        return f"{cores} cores fits none of this queue's {total} nodes"
    return (f"{cores} cores fits {fit} of {total} nodes here "
            f"({100 * fit // total}%)")


def _machine_lines(row, *, cores: Optional[int] = None) -> List[str]:
    """The machines a domain holds, **one line per SIZE**.

    **A partition is a queue, not a machine type.**  Sol's ``htc`` is 48-,
    64-, 96- and 128-core nodes under one name.  The ``cores`` column above
    summarises them as a range (:func:`core_range`); these lines are the
    machines themselves.

    **Grouped by size, and that is not cosmetic.**  `sinfo` reports one row
    per *gres group*, and on the real Sol record `htc` has FOURTEEN -- nine
    of them 48-core rows differing only in which card they carry.  Printed
    one-per-group the menu ran to 68 lines and stopped being readable,
    which is the opposite of showing what exists.  Sizes are what a person
    picking a machine is choosing between; the cards available AT each size
    ride along, because that is the pairing no single figure could state:
    on ``htc`` you can have 128 cores or you can have an A100, never both.

    Nothing is printed for a domain that holds ONE size: the ``cores``
    column already said it, and a line repeating it is noise on every row
    with nothing to disclose.
    """
    shapes = getattr(row, "node_types", None) or []
    by_size: Dict[int, Dict[str, Any]] = {}
    for t in shapes:
        c = t.get("cores")
        if not c:
            continue
        g = by_size.setdefault(int(c), {"nodes": 0, "dev": set(), "mem": []})
        g["nodes"] += int(t.get("nodes") or 0)
        g["dev"].update((t.get("gpu") or {}).keys())
        if t.get("mem_gb"):
            g["mem"].append(float(t["mem_gb"]))
    if len(by_size) < 2:
        return []
    out = []
    for c in sorted(by_size, reverse=True):
        g = by_size[c]
        bits = [f"{c:>4} cores", f"x{g['nodes']} node(s)"]
        if g["mem"]:
            lo, hi = min(g["mem"]), max(g["mem"])
            bits.append(f"{lo:g} GB" if lo == hi else f"{lo:g}-{hi:g} GB")
        if g["dev"]:
            bits.append(", ".join(sorted(g["dev"])))
        # A size too small for THIS ask is marked, not hidden: it is why
        # the queue is slower than its node count suggests.
        fits = "" if (cores is None or c >= cores) else "   (too small)"
        out.append("- " + "  ".join(bits) + fits)
    return out


def _why_not(row, ask: Ask, *, cores=None, gpu: bool = False) -> List[str]:
    """Why this queue cannot take this job — empty when it can.

    Reuses the scheduler's own admission so the listing and the submission
    cannot disagree about what fits: a table that says yes where the check
    says no is worse than no table.
    """
    from ..scheduler.admit import Request, admits
    return list(admits(row, Request(ranks=cores, walltime_s=ask.time_s,
                                    mem_gb=ask.mem_gb,
                                    gpus=1 if gpu else None)))


def gpu_share_notes(gpu_count: Optional[int], mpi_np: Optional[int], *,
                    cores_per_rank: Optional[int] = None,
                    node_cores: Optional[int] = None) -> List[str]:
    """What a GPU-sharing request means, stated once so the bench-grid
    enumeration and the submission display cannot disagree about it --
    and so neither drifts from what `runwrap.py` actually runs.

    **The four things this states, always in this order:**

    1. ALWAYS -- how many ranks land on one device.  `runwrap.py`'s own
       load-balance line is this exact arithmetic (``_ranks_per_gpu =
       mpi_np / ngpu``, `running-a-job.md` § 3.3); stated here so the
       person choosing --domain sees it BEFORE a day in the queue, not in
       stderr after the job has already started.
    2. a WARNING past :data:`MPS_MAX_CLIENTS_PER_DEVICE` -- ranks past the
       ceiling do not run slowly, they fail to attach.
    3. a bare NOTE past :data:`GPU_TUNED_RANKS_PER_DEVICE` -- this stack's
       measured sweet spot.  Past it is a real choice a person may have good
       reason to make, so this states both numbers and renders no verdict
       (`docs/execution/submission.md` -- ask, do not derive).
    4. a node-fit check, ``K * C <= cores / G`` (`engines/tuning.md` § 2.12's
       own arithmetic, rearranged: ``G*K*C <= node_cores``) -- only when the
       caller has both ``cores_per_rank`` and ``node_cores`` to check it
       with.  The bench-grid enumeration already enforces this one as a HARD
       drop (`jobset/_cli.py`'s per-family core cap); it is repeated here so
       a caller with a single, non-swept request gets the same protection.

    ``gpu_count`` falsy/``None`` returns no lines -- a CPU-family request,
    nothing here applies.  ``mpi_np`` falsy/``None`` the same -- there is no
    rank count yet to say anything about.
    """
    if not gpu_count or not mpi_np:
        return []
    ranks_per_gpu = mpi_np // gpu_count
    remainder = mpi_np % gpu_count
    lines = [
        f"  gpu share  {mpi_np} rank(s) / {gpu_count} GPU(s) = "
        f"{ranks_per_gpu} rank(s)/GPU"
        + (f"  (uneven: {remainder} device(s) carry one extra rank)"
           if remainder else "")
    ]
    if ranks_per_gpu > MPS_MAX_CLIENTS_PER_DEVICE:
        lines.append(
            f"  WARNING {ranks_per_gpu} ranks/GPU exceeds MPS's own "
            f"ceiling of {MPS_MAX_CLIENTS_PER_DEVICE} clients/device -- "
            f"ranks past the {MPS_MAX_CLIENTS_PER_DEVICE}th on one device "
            f"will FAIL TO ATTACH, not just run slowly "
            f"(engines/tuning.md § 2.12).")
    elif ranks_per_gpu > GPU_TUNED_RANKS_PER_DEVICE:
        lines.append(
            f"  NOTE {ranks_per_gpu} ranks/GPU; this stack's tuned point "
            f"(no NCCL) is ~{GPU_TUNED_RANKS_PER_DEVICE} "
            f"(engines/tuning.md § 2.12).")
    if cores_per_rank and node_cores:
        need = gpu_count * ranks_per_gpu * cores_per_rank
        if need > node_cores:
            lines.append(
                f"  WARNING {gpu_count} GPU(s) x {ranks_per_gpu} "
                f"rank(s)/GPU x {cores_per_rank} core(s)/rank = {need} "
                f"cores -- more than this node's {node_cores} "
                f"(engines/tuning.md § 2.12: K x C <= cores / G).")
    return lines


def confirm(text: str, *, auto_yes: bool = False, echo=None,
            prompt=None) -> bool:
    """**The one interface** — show it, then act on the answer.

    ``auto_yes`` is how a person says *I have decided to trust this*; its
    absence is not permission.  ``echo``/``prompt`` are injected so the same
    function serves a terminal, a test, and anything else that can show a
    string and read a yes — the browser included.
    """
    import sys
    import click
    echo = echo or click.echo
    echo(text)
    if auto_yes:
        echo("  (--yes)")
        return True
    if prompt is None and not sys.stdin.isatty():
        # NO TERMINAL TO ASK.  S4 says the absence of `--yes` is not
        # permission, so this declines -- but it declines by SAYING WHY and
        # naming the flag, because a scripted run that aborts with no
        # explanation is a worse failure than the one the gate prevents.
        echo("  not a terminal, so there is nobody to ask -- pass --yes to "
             "submit what is printed above without confirming.")
        return False
    prompt = prompt or (lambda: click.confirm("  submit this?",
                                              default=True))
    return bool(prompt())


# --------------------------------------------------------------------- #
#  When would this actually start?  (`launch --mode ask`)                #
# --------------------------------------------------------------------- #
#
# Everything above answers from the RECORD and works anywhere: what each
# queue allows, which of them could take this job.  Nothing above knows
# what the cluster is busy with.
#
# **There is no prediction without the cluster** (user, 2026-08-27).  Node
# counts are a proxy for wait and a poor one -- 134 wide nodes are no help
# if all 134 are busy for two days -- so the only honest answer comes from
# asking the scheduler, on the machine that has one.
#
# `sbatch --test-only` validates a request and reports when it WOULD be
# scheduled.  **It creates no job.**  That is what makes it safe to run in
# a loop while you tune: change the domain, change the cores, ask again,
# and either take the better wait or decide you can live with this one.
#
# **It is a MODE of launch, not a verb of its own** (user, 2026-08-27:
# *instead of submit, we can just say ask -- we don't have to reinvent
# something*).  `--mode ask` walks the identical path `--mode submit`
# walks and inserts one flag, so the line asked about IS the line that
# would be sent.  A separate verb would have had to re-render the flags,
# and two renderings of one fact are two things that can disagree.
#
# Only the PARSING lives here: it is pure, so it can be tested with no
# cluster present -- and this workstation has none.

#: THREE INDEPENDENT READS, not one regex with optional tails.
#:
#: Measured on ASU Sol 2026-08-27, and this is why:
#:
#:     sbatch: Job 62266174 to start at 2026-08-27T11:22:03 a using 4
#:             processors on nodes sc078 in partition htc
#:                                        ^
#: There is a token between the timestamp and ``using`` -- what it is, is
#: SLURM's business and not ours.  A single pattern that chained the three
#: facts required them to be adjacent, so that one stray character cost the
#: processor count AND the node name while the time still parsed.  Read
#: separately, anything SLURM puts between them is simply ignored, and one
#: field going missing cannot take the others with it.
_WHEN_RE = re.compile(r"to start at (\S+)", re.I)
_PROCS_RE = re.compile(r"using\s+(\d+)\s+processors?", re.I)
_NODES_RE = re.compile(r"on nodes?\s+(\S+)", re.I)


@dataclass(frozen=True)
class Prediction:
    """What the scheduler said about one request.

    ``start`` is ``None`` whenever SLURM did not give a time -- it declined
    to predict, or it refused the request outright.  **That is reported as
    unknown, never as soon**: a missing prediction is the absence of an
    answer, and dressing it as a good one is how a person ends up waiting a
    day for a queue that looked instant.
    """
    #: What was asked about -- the JOB's name.  Not the queue: `ask` sends
    #: one request, and which queue it named is on the command line printed
    #: beneath the table.  It was called ``domain`` for one revision and the
    #: column header said "queue" while the value was the job, which is the
    #: kind of label that quietly teaches the wrong thing.
    label:   str = ""
    start:   Optional[str] = None
    nodes:   Optional[str] = None
    procs:   Optional[int] = None
    refused: Optional[str] = None
    #: There was no scheduler to ask.  A separate flag rather than a
    #: sniffable string, because the reply is a different SENTENCE -- not
    #: "the queue could not say", but "there is no queue".  A workstation
    #: is not a cluster having a bad day.
    no_scheduler: bool = False


def parse_test_only(text: str) -> Prediction:
    """SLURM's answer -> a :class:`Prediction`.  Pure, so it is testable
    without a scheduler -- and it had to be, since this workstation has
    none.

    Anything without a recognisable *to start at* leaves ``start`` as
    ``None`` and keeps the raw text in ``refused``.  **Keeping the text
    rather than matching a known prefix is what makes this work**: the
    refusal was written against an invented ``sbatch: error: ...`` line,
    and Sol actually says ``allocation failure: Requested node
    configuration is not available``.  A parser that recognised prefixes
    would have thrown away the one sentence worth reading.

    The processor count and node name are read independently of the time
    and of each other, so a field SLURM omits -- or a token it inserts --
    costs only itself.
    """
    blob = (text or "").strip()
    m = _WHEN_RE.search(blob)
    if not m:
        first = next((ln.strip() for ln in blob.splitlines() if ln.strip()),
                     "")
        return Prediction(refused=first or "no answer")
    pm = _PROCS_RE.search(blob)
    nm = _NODES_RE.search(blob)
    return Prediction(start=m.group(1),
                      procs=int(pm.group(1)) if pm else None,
                      nodes=nm.group(1) if nm else None)


def prediction_table(preds: Sequence[Prediction]) -> str:
    """What the scheduler said, one line per job asked about.

    Ordered as it was asked, not by which looks fastest.  Sorting would be
    a recommendation, and the wait is only one of the things a person is
    weighing -- the others (what else is running, whose allocation, how
    long the job really needs) are not on this machine.
    """
    if not preds:
        return "nothing to ask about."
    # NO SCHEDULER AT ALL -- a different answer, not an empty table.  The
    # header would say "asked the scheduler" when none was asked, and the
    # footer would offer to change --domain, which means nothing here.
    if all(p.no_scheduler for p in preds):
        # The FACT only.  What to do about it is the caller's closing line,
        # or the two say `--mode direct` one after the other.
        return ("there is no scheduler on this machine, so there is nothing "
                "to wait for -- a job here starts immediately.")
    head = f"  {'job':<16} {'would start':<22} {'on':<16} procs"
    lines = ["asked the scheduler (nothing was submitted):", head]
    for p in preds:
        if p.start:
            lines.append(f"  {p.label:<16} {p.start:<22} "
                         f"{(p.nodes or '-'):<16} {p.procs or '-'}")
        else:
            lines.append(f"  {p.label:<16} {'no prediction':<22} "
                         f"-> {p.refused or 'the scheduler did not say'}")
    lines.append("")
    lines.append("  a time is an ESTIMATE from the queue as it is right now; "
                 "it moves.")
    lines.append("  change --domain or --cores and ask again, or launch when "
                 "you are happy.")
    return "\n".join(lines)
