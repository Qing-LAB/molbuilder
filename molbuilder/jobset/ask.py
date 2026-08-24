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

Four things live here and nowhere else:

    Ask          the question, and the answer to it
    queue_table  the queues this machine offers, and which can take the job
    render       the one output — what is about to be requested
    confirm      the one interface — approve, change, or skip

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
from typing import List, Optional, Sequence, Tuple

#: The benchmark launcher's own shape, stated once so the total shown is the
#: total requested.  Two spellings of one formula is how a displayed number
#: comes to differ from the one that reaches the scheduler.
GROUP_SLACK = 1.1
GROUP_STARTUP_S = 300

#: MPS's own hard ceiling -- more than this many processes cannot even
#: ATTACH to one device (`engines/tuning.md` § 2.12, citing
#: `references.bib: NvidiaMPS`, the A100-generation figure).  Past it a
#: submission is not slow, it is broken: the ceiling-plus-first rank's MPS
#: client fails to connect.
MPS_MAX_CLIENTS_PER_DEVICE = 48

#: This stack's own tuned point -- ~4 ranks/GPU, because this ELPA build has
#: no NCCL (`engines/tuning.md` § 2.12: "the wrapper's default lands near
#: ~4 ranks per GPU with MPS -- the tuned point for this kind of build").
#: Past it, extra ranks queue for compute rather than running concurrently --
#: not wrong, just past where more sharing helps.
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


def parse_duration(text) -> Optional[int]:
    """``"4h"`` / ``"90m"`` / ``"45"`` -> seconds.  Bare numbers are minutes.

    Raises :class:`ValueError` with the forms it accepts, because a refusal
    that does not show the shape is a refusal you have to guess past.
    """
    if text is None or str(text).strip() == "":
        return None
    t = str(text).strip().lower()
    mult = 60
    for suffix, m in (("h", 3600), ("m", 60), ("s", 1)):
        if t.endswith(suffix):
            mult, t = m, t[:-1]
            break
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"{text!r} is not a duration -- write 4h, 90m, "
                         f"or 45 (bare numbers are minutes)")
    if v <= 0:
        raise ValueError("a duration must be positive")
    return int(v * mult)


def parse_memory(text) -> Optional[float]:
    """``"128G"`` / ``"0.5T"`` / ``"128"`` -> gigabytes.  Bare numbers are GB.

    SLURM's own spelling, so what a person types here is what they would have
    typed into ``--mem``.
    """
    if text is None or str(text).strip() == "":
        return None
    t = str(text).strip().upper()
    mult = 1.0
    for suffix, m in (("T", 1024.0), ("G", 1.0), ("M", 1 / 1024.0)):
        if t.endswith(suffix):
            mult, t = m, t[:-1]
            break
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"{text!r} is not an amount of memory -- write "
                         f"128G, 0.5T, or 128 (bare numbers are GB)")
    if v <= 0:
        raise ValueError("memory must be positive")
    return v * mult


def bench_bound(total_s: int, n_trials: int) -> int:
    """A benchmark's TOTAL -> the per-trial bound that fits inside it.

    The person states the total — how long they are willing to wait — and the
    per-trial bound is arithmetic on top.  It ran the other way round: a
    per-trial default produced a total nobody saw.

    A bound below a minute measures nothing, so it floors there and the total
    is exceeded instead.  :func:`render` says so when that happens; honouring
    a budget into uselessness would be the worse answer.
    """
    n = max(1, int(n_trials))
    usable = max(0, int(total_s) - GROUP_STARTUP_S)
    return max(60, int(usable / (n * GROUP_SLACK)))


def bench_total(bound_s: int, n_trials: int) -> int:
    """The per-trial bound -> the total the allocation asks for.  The
    launcher's own formula, which :func:`bench_bound` inverts."""
    return int(max(1, int(n_trials)) * int(bound_s) * GROUP_SLACK) \
        + GROUP_STARTUP_S


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
        wall = f"{secs // 3600}h" if secs and secs >= 3600 else (
            f"{secs // 60}m" if secs else "-")
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


def render(ask: Ask, *, placement=None, n_trials: Optional[int] = None,
           bound_s: Optional[int] = None, extra: Sequence[str] = ()) -> str:
    """**The one output** — what is about to be requested, in full.

    Every line is a number that will reach the scheduler, and a line the
    person can change while changing it is still free.  Shown by the CLI and
    by the browser from this one function, because two renderings of one
    request is how a surface comes to show something the scheduler was not
    told.
    """
    lines: List[str] = ["about to request:"]
    if ask.time_s is not None:
        lines.append(f"  time     {ask.time_s // 3600}h "
                     f"{(ask.time_s % 3600) // 60:02d}m")
    if ask.mem_gb is not None:
        lines.append(f"  memory   {ask.mem_gb:g} GB")
    if n_trials is not None and bound_s is not None:
        total = bench_total(bound_s, n_trials)
        lines.append(f"  {n_trials} trial(s), {bound_s // 60} min each "
                     f"-> {total // 60} min total")
        if ask.time_s is not None and total > ask.time_s:
            lines.append(
                f"  NOTE {n_trials} trials do not fit in "
                f"{ask.time_s // 60} min -- a bound under a minute measures "
                f"nothing, so the total is {total // 60} min.  Fewer trials, "
                f"or more time.")
    if placement is not None:
        lines.append(f"  queue    {placement.partition} / {placement.qos}")
    lines.extend(f"  {e}" for e in extra)
    return "\n".join(lines)


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
