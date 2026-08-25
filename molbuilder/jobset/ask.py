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

from dataclasses import dataclass
from typing import List, Optional, Sequence

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
    """
    from ..scheduler import domain_ceiling_s, domain_serves_gpu
    if not rows:
        return "this machine states no queues -- the job runs directly."
    head = (f"  {'':2} {'name':<12} {'partition/qos':<22} {'max time':>9} "
            f"{'cores':>6} {'memory':>9}  gpu")
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
            f"{wall:>9} {str(d.max_cores or '-'):>6} {mem:>9}  {dev}")
        if why:
            lines.append(f"     -> {'; '.join(why)}")
    lines.append("")
    lines.append("  choose one with --domain <name>.  Nothing is submitted "
                 "until you do.")
    return "\n".join(lines)


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
