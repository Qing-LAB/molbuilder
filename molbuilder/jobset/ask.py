"""What this job needs — asked once, answered once, shown once.

**One question, one answer, one interface, one output** (user, 2026-08-23).

A scientist knows what their calculation needs better than any rule this
framework can write.  So it asks, rather than deriving: *how much time, how
much memory.*  Everything else follows from the answer, and nothing has to be
explained afterwards because nothing was invented.

Everything that lives here, and it lives nowhere else:

    queue_table      the queues the target's record lists, their machines,
                     and the job's fit on each
    gpu_share_notes  what a GPU-sharing request means, stated once
    confirm          the one interface — approve, change, or skip

    The one output is the PLAN the launch door prints -- the exact sbatch
    command of every job and the header it sends, from the same code that
    submits it.  TIME IS NEVER DERIVED (user dictation, 2026-08-24): the
    user states it, or prep refuses (`architecture.md` § 5.2).

`scheduler.fit` is the one reading of "how does this fit": placement prints
it, and `queue_table` renders it per queue (`scheduler.md` R6).

The command line asks through these.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..scheduler.fit import Request

# The two quantities a job asks for -- a wall and an amount of memory -- and
# every dialect each is written in, live in `scheduler/quantities.py`:
# a codec on a basic unit is L1; this module is L2 (`docs/design.md`).
from ..scheduler.quantities import human_wall, mem_range

#: MPS's own hard ceiling -- more than this many processes cannot even
#: ATTACH to one device (`engines/tuning.md` § 2.12, citing
#: `references.bib: NvidiaMPS`, the A100-generation figure).  A site FACT
#: cited from the vendor, not an estimate.
MPS_MAX_CLIENTS_PER_DEVICE = 48

#: This stack's tuned point, ~4 ranks/GPU without NCCL (`engines/tuning.md`
#: § 2.12) -- a MEASURED literature figure: both feed
#: `gpu_share_notes`, which INFORMS and never decides.
GPU_TUNED_RANKS_PER_DEVICE = 4


def queue_table(rows: Sequence, request: "Request") -> str:
    """**Every queue the target's record lists, its machines, and this job's
    fit on each** (`scheduler.md` R4, R6).

    The framework does not choose (user, 2026-08-23: *"just don't guess
    shit"*).  It shows what exists and how the job sits on each, and the
    person names one -- prep shows this when a run names none
    (`placement.launch_refusal`).  Which queue to spend a day of wall-clock
    in is a judgement about priority, contention and what else is running --
    none of which is on the target's record, and all of which the person
    has.

    **Each queue is up to three kinds of line**::

        1  htc   htc/public   4h   48-128   501-1511.2 GB   a100 x4, h100 x8, …
             - 128 cores  x137 node(s)  503.2-1511.2 GB  a30, h200   <- the
             - 48 cores  x69 node(s)  501-503.5 GB  a100, l40          machines,
           64 cores · 900 GB fits 3 of 212 nodes of htc (1%): …     when several
                                                                     <- the fit

    The cores and memory columns are the **ranges** across the queue's
    machines (`Domain.core_range`, `Domain.mem_range`), one number when
    every machine is the same.
    """
    from ..scheduler.fit import domain_ceiling_s, fit
    head = (f"  {'':2} {'name':<12} {'partition/qos':<22} {'max time':>9} "
            f"{'cores':>8} {'memory':>14}  gpu")
    lines = ["the target's record lists:", head]
    for i, d in enumerate(rows, 1):
        wall = human_wall(domain_ceiling_s(d))
        dev = ", ".join(x.type + (f" x{x.per_node}" if x.per_node else "")
                        for x in d.devices) or "-"
        lines.append(
            f"  {i:<2} {d.name:<12} {d.partition + '/' + d.qos:<22} "
            f"{wall:>9} {d.core_range:>8} {d.mem_range:>14}  {dev}")
        for shape in _machine_lines(d):
            lines.append(f"       {shape}")
        lines.append(f"     {fit(d, request).line}")
    return "\n".join(lines)


def _machine_lines(row) -> List[str]:
    """The machines a domain holds, **one line per SIZE**.

    **A partition is a queue, not a machine type.**  Sol's ``htc`` is 48-,
    64-, 96- and 128-core nodes under one name.  The ``cores`` column above
    summarises them as a range; these lines are the machines themselves.

    **Grouped by size, and that is not cosmetic.**  `sinfo` reports one row
    per *gres group*, and on Sol `htc` has FOURTEEN -- nine of them 48-core
    rows differing only in which card they carry.  Sizes are what a person
    picking a machine is choosing between; the cards available AT each size
    ride along, because that is the pairing no single figure could state:
    on ``htc`` you can have 128 cores or you can have an A100, never both.

    Nothing is printed for a domain that holds ONE size: the ``cores``
    column already said it.
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
            bits.append(mem_range(g["mem"]))
        if g["dev"]:
            bits.append(", ".join(sorted(g["dev"])))
        out.append("- " + "  ".join(bits))
    return out


def gpu_share_notes(gpu_count: Optional[int], mpi_np: Optional[int], *,
                    one_process: bool = False) -> List[str]:
    """What a GPU-sharing request means, said at a run's prep -- by the two
    thresholds the bench grid's note reads too (`prep_inputs.bench_inputs`),
    so the two cannot disagree -- and so neither drifts from what
    `runwrap.py` actually runs.

    **The three things this states, always in this order:**

    1. ALWAYS -- how many ranks land on one device.  `runwrap.py`'s own
       load-balance line is this exact arithmetic (``_ranks_per_gpu =
       mpi_np / ngpu``, `running-a-job.md` § 3.3); stated here so the
       person sees it at prep, BEFORE a day in the queue, not in
       stderr after the job has already started.
    2. a WARNING past :data:`MPS_MAX_CLIENTS_PER_DEVICE` -- ranks past the
       ceiling do not run slowly, they fail to attach.
    3. a bare NOTE past :data:`GPU_TUNED_RANKS_PER_DEVICE` -- this stack's
       measured sweet spot.  Past it is a real choice a person may have good
       reason to make, so this states both numbers and renders no verdict
       (`docs/execution/submission.md` -- ask, do not derive).

    Whether the cores fit a node is not said here: it is each cell's fit on
    the benchmark's queue (`jobset/prep_inputs._cells_on_the_queue`).

    ``gpu_count`` falsy/``None`` returns no lines -- a CPU-family request,
    nothing here applies.  ``mpi_np`` falsy/``None`` the same -- there is no
    rank count yet to say anything about.  An engine that runs ONE process
    (``one_process``, PySCF) shares no device among ranks, and fewer ranks
    than GPUs share none either: each is said as it is.
    """
    if not gpu_count or not mpi_np:
        return []
    if one_process:
        return [f"  gpu share  one process on {gpu_count} GPU(s)"]
    if mpi_np < gpu_count:
        return [f"  gpu share  {mpi_np} rank(s) / {gpu_count} GPU(s): a GPU "
                f"each, and {gpu_count - mpi_np} GPU(s) with no rank"]
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
    return lines


@dataclass(frozen=True)
class Said:
    """A question's answer: whether to go, and the words the ledger keeps --
    each question and its answer is written down, a *no* too (`job-system.md`
    § 5.0, agreement 6).  True when the answer is yes.  ``asked`` is false
    when there was nobody to ask -- no terminal and no ``--yes`` -- which
    the launch refuses (`submission.md` S4)."""
    yes: bool
    words: str
    asked: bool = True
    #: The question's third answer was given -- its ``other`` (:func:`confirm`):
    #: yes, the other way.
    other: bool = False

    def __bool__(self) -> bool:
        return self.yes


def confirm(text: str, *, auto_yes: bool = False, echo=None,
            prompt=None, question: str = "submit this?",
            err: bool = False, other=None, default: bool = True) -> Said:
    """**The one interface** — show it, then act on the answer.

    ``auto_yes`` is how a person says *I have decided to trust this*; its
    absence is not permission.  ``echo``/``prompt`` are injected so the same
    function serves a terminal, a test, and anything else that can show a
    string and read a yes — the browser included.  ``question`` is what is asked: a run here is asked as a submission is
    (`submission.md` S4).  ``err`` asks on stderr, for a command whose stdout
    is its output (`molbuilder modify ... -`).  ``other`` -- ``(letter,
    word)`` -- is a third answer, yes the other way (a run here: in the
    foreground, or in the background, `job-system.md` § 6.0): asked
    ``[Y/n/<word>(<LETTER>)]``, Enter yes; given, the answer is
    :attr:`Said.other` and its words ``word``.  ``prompt`` then returns what
    was typed.  ``default`` is what Enter answers: yes, but for a question
    whose yes cannot be taken back (`molbuilder jobs stop`, ``[y/N]``).
    """
    import click
    # THROUGH THE GUARDED DOOR: a bare `sys.stdin.isatty()` raises on exactly
    # the input this branch exists to handle (`ValueError` when stdin is
    # closed, `AttributeError` when it is None).  `hints` is floor 1 with no
    # dependencies, so asking it is a downward import (A7).
    from ..envs.hints import stdin_can_answer
    echo = echo or (lambda s: click.echo(s, err=err))
    echo(text)
    if auto_yes:
        echo("  (--yes)")
        return Said(True, "yes (--yes)")
    if prompt is None and not stdin_can_answer():
        # NO TERMINAL TO ASK.  S4 says the absence of `--yes` is not
        # permission: nothing goes, and the launch REFUSES -- naming the
        # flag -- so a script reading the exit code is not told it went
        # (`submission.md` S4; `job-system.md` § 6.0, step 3).
        return Said(False, "no answer (not a terminal): nothing sent",
                    asked=False)
    if other is None:
        prompt = prompt or (lambda: click.confirm(f"  {question}",
                                                  default=default, err=err))
        yes = bool(prompt())
        return Said(yes, "yes" if yes else "no")
    letter, word = other
    prompt = prompt or (lambda: click.prompt(
        f"  {question} [Y/n/{word}({letter.upper()})]", default="y",
        type=click.Choice(["y", "yes", "n", "no", letter, word],
                          case_sensitive=False),
        show_choices=False, show_default=False, err=err))
    got = str(prompt()).strip().lower()
    if got in (letter, word):
        return Said(True, word, other=True)
    yes = got in ("y", "yes")
    return Said(yes, "yes" if yes else "no")


@dataclass(frozen=True)
class Picked:
    """A choice's answer: what was picked, and the words the ledger keeps.
    ``asked`` is false when there was nobody to ask -- no terminal -- which
    the caller refuses, naming the flag that answers it (`job-system.md`,
    *The task*)."""
    picks: Tuple[str, ...]
    words: str
    asked: bool = True


def choose(text: str, preselected, *, echo=None, prompt=None,
           question: str = "which?") -> Picked:
    """**The one choice** -- show ``text``, then ask which, Enter taking
    ``preselected``: names, comma-separated, as the person types them
    (resolved by the caller, through the one stage grammar).  ``echo`` /
    ``prompt`` are injected as :func:`confirm`'s are; with no terminal and
    no ``prompt``, nobody is asked."""
    import click
    from ..envs.hints import stdin_can_answer
    echo = echo or click.echo
    echo(text)
    if prompt is None and not stdin_can_answer():
        return Picked((), "no answer (not a terminal)", asked=False)
    default = ",".join(preselected)
    prompt = prompt or (lambda: click.prompt(f"  {question}", default=default,
                                             show_default=bool(default)))
    raw = str(prompt() or "").strip() or default
    picks = tuple(w.strip() for w in raw.split(",") if w.strip())
    return Picked(picks, "picked " + (", ".join(picks) or "nothing"))


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
# a loop while you tune: change the wall and ask again, or prepare again on
# another queue or with other resources, and either take the better wait or
# decide you can live with this one.
#
# **It is a MODE of launch, not a verb of its own** (user, 2026-08-27:
# *instead of submit, we can just say ask -- we don't have to reinvent
# something*).  `--mode ask` walks the identical path `--mode submit`
# walks and inserts one flag, so the line asked about -- the header its prep
# wrote -- IS the line that would be sent.
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
    #: beneath the table.
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
    rather than matching a known prefix is what makes this work**: Sol says
    ``allocation failure: Requested node configuration is not available``,
    and a parser that recognised prefixes would throw away the one sentence
    worth reading.

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


#: DISCOVERY, A READER PER KIND (`scheduler.md` § 6a): each kind with a queue
#: -> the reader of its start-time answer, which no format states.
_ANSWER_READERS = {"slurm": parse_test_only}


def answer_of(kind, text: str) -> Prediction:
    """The scheduler's answer to an ask (`kinds.Kind.asking`), read by its
    kind's reader (:data:`_ANSWER_READERS`)."""
    return _ANSWER_READERS[kind.name](text)


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
    # footer would offer to change a wall nothing waits on.
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
    # WHAT A LAUNCH MAY RESTATE (`commands.LAUNCH_FLAGS`): the wall; the
    # queue, ranks, cores, GPUs and memory are prep's.
    lines.append("  change the wall (--time) and ask again, or launch when you "
                 "are happy -- the queue, ranks, cores, GPUs and memory are "
                 "prep's.")
    return "\n".join(lines)
