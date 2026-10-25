"""The launch door's engine -- the one entry `launch` runs: its plan, shown
and asked by the verb, then sent as it was shown (`docs/execution/
job-system.md` § 6.0; § 5.3, § 7; the wrapper contract is
`job-contracts.md` § 2).

``prep`` derives the work and lays out the tree; THIS module sends it, in the
order § 6.0 states:

1. **the plan** (:func:`plan_launch`), nothing written -- the work
   (:func:`_the_group`) and its SUBMISSIONS, each one scheduler job or one
   process here walking one or more MEMBERS, each planned by the one member
   planner (:func:`_plan_member`; a benchmark's trial, :func:`_plan_trial`):
   the run it runs in, what it takes and how that run ended; the gates; the
   queue; the exact line and every script the send will write.  The
   submission builders, one plan (:class:`LaunchPlan`) each:

   * **one job** (:func:`_plan_stage`) -- a stage, or a benchmark's named
     trial; a stage whose run is a run of points sent as the points' walk
     (:func:`_points_walk`).  A stage launched before is launched again,
     however it ended -- warm, from its own latest run, or cold (user,
     2026-10-07: "run continue warm or cold is user's decision, and error or
     not, that's user's responsibility");
   * **a group's job** (:func:`_plan_group`) -- its members walked in the
     ladder's order, under the header its prep wrote;
   * **a benchmark's walk** -- ONE job per resource shelf of a sweep sent
     to a queue, or its unlaunched trials run here, the trials in sequence
     (`generator.md` § 4.3a, `_walk_script`);

2. **shown** (:meth:`LaunchPlan.shown`) and 3. **asked**, by the verb --
   ``ask`` mode asks the scheduler about each submission instead
   (:func:`ask_launch`: its kind's ask on the same line), and a dry
   run stops there; neither writes anything;
4. **the send** (:func:`send_launch`), deciding nothing: the folder checked
   against the one the plan was made from, the plan's writes carried out,
   each submission out through ONE function (:func:`_go`), each member's
   ``run.json`` written by the one writer;
5. **the record** -- each submission handed to the verb's ledger the moment
   it goes.

Every door builds its scheduler line through ONE request
(:func:`_sbatch_request`): the header its prep wrote, sent as it is, and the
one value a launch may restate, a submitted job's wall.  The scheduler is
handed one job per invocation -- a grouped bench one per shelf.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional,
                    Sequence, Tuple)

if TYPE_CHECKING:
    from ..scheduler.record import Environment

# The record's one spelling for a wall, written in ONE place.  A second
# formatter here would be a second answer to "what does a walltime look
# like on disk?".
from ..scheduler.quantities import slurm_time as _slurm_time

# NOTE: `job_dir_names` is NOT imported here.  It is the naming
# authority (materialize.py), and the places THIS module needs it import
# it locally beside the siblings they use.
from .agreement import TrialNotCold, check_trial_starts_cold
from .model import KIND_LADDER, KIND_SWEEP, Job, JobSet, Resources
from .continuation import Continuation, relaunch
from .planned import Plan, found
from ..paths import attempt_dir
from ..runfiles import FIRST_ATTEMPT, LAUNCH_DIR, GroupNames, RunNames
from .materialize import (CONTINUED, GATHERED, HandOver, ever_launched,
                          hand_over)
from .naming import open_container
from ..runrecord import LaunchRecordError, RunRef
from ..scheduler.kinds import LAUNCH_CLAIM, LAUNCH_CLAIM_VAR
from .commands import block, command as _cmd, launch_lines, rollback


class SubmitError(Exception):
    """A JobSet could not be launched -- a mode, a member or a prepared
    file it lacks, or the scheduler's refusal -- in the sentence a person
    reads."""


#: The codes a :class:`JobResult` can carry -- a CLOSED set, enforced on
#: construction, so a reader matches a code and never a sentence's words.
JOB_RESULTS: Tuple[str, ...] = (
    "submitted", "ran", "failed", "unread", "started", "planned", "asked",
    "not asked", "scheduler refused", "stays pending", "rides", "already run",
    "skipped", "takes")


@dataclass
class JobResult:
    """What happened to one job.  ``command`` is the exact line that ran or
    would run -- empty on a line that only reports.  In ``submit`` mode
    ``job_id`` is the scheduler's id; in ``direct`` mode ``returncode`` is the
    process exit status.  ``status`` is one of :data:`JOB_RESULTS` -- a
    code, which every reader matches -- and ``said`` the sentence shown
    beside it, when there is one:

    * ``submitted`` / ``ran`` -- it went; ``ran``, here and waited on, its
      process ended; ``failed`` -- it ran here and a member did not finish,
      by the status door (the launch verb's reading, ``detail`` its words);
      ``unread`` -- it ran here, and how it ended could not be read
      (``detail``: why);
      ``started`` -- it went here, in the background;
    * ``planned`` (dry-run) / ``asked`` (``--mode ask``: nothing was sent,
      and ``prediction`` carries what the scheduler said; ``command`` is the
      line that would be SENT) / ``not asked`` (past the query cap);
    * ``scheduler refused`` (this shelf was rejected; the rest still went) /
      ``stays pending`` (its shelf was refused, so it was never sent);
    * ``rides`` -- a stage of a group, a benchmark's trial or a run's point,
      launched by its group's, its shelf's or its sweep's one job (``said``:
      which);
    * ``already run`` (ask) / ``skipped`` (direct) -- a trial measured
      before, passed over by name;
    * ``takes`` -- a member, and what it takes (``said``: ``WOULD launch it
      ...`` when planned or asked, ``launched ...`` when sent)."""
    name:       str
    command:    List[str]
    status:     str
    job_id:     Optional[str] = None
    returncode: Optional[int] = None
    #: What is known about this shelf's fate.  After a ``scheduler refused``,
    #: the scheduler's own words -- carried rather than raised, because ONE
    #: refused shelf must not cancel the shelves behind it, so the failure
    #: travels back as data (2026-08-30).
    detail:     Optional[str] = None
    #: Only in ``ask`` mode.  ``None`` everywhere else, and a ``Prediction``
    #: whose ``start`` is ``None`` when the scheduler declined to predict -- which
    #: is reported as *unknown*, never as *soon*.
    prediction: Optional[object] = None
    #: WHICH QUEUE prep placed it on, by name (2026-08-30): where several
    #: queues share one partition, ``-p htc`` reads as *htc* to anyone
    #: scanning it -- Sol's `debug` is (htc, debug) and its `htc` is (htc,
    #: public).  The name is the fact; the header's lines its rendering.
    domain:     Optional[str] = None
    #: THE REQUEST THE LINE SENDS, as the header its prep wrote carries it
    #: -- its directive lines, read back by the header's reader
    #: (`Directives.lines_of`), the wall the line restates in its place:
    #: what the question and a dry run show (`submission.md` S4).  ``()``
    #: for a run here.
    request:    Tuple[str, ...] = ()
    #: The sentence shown beside the status, when it has one -- what a
    #: member takes, what a member rides, how a trial was passed over.
    said:       Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in JOB_RESULTS:
            raise ValueError(f"JobResult.status must be one of "
                             f"{JOB_RESULTS}; got {self.status!r}")

    @property
    def shown(self) -> str:
        """The result as a person reads it: its sentence, else its code."""
        return self.said or self.status


@dataclass(frozen=True)
class _Placed:
    """Where prep placed the work -- the queue its jobs record
    (`job-set.json`'s ``placement``, `job-system.md` § 6.0): its name, and
    the partition and qos prep bound it to on the target's record."""
    name: str
    partition: str
    qos: str


def _placed_by_prep(jobs) -> "Optional[_Placed]":
    """The one queue the work's prep recorded -- its members' placement,
    which a group and a benchmark share by construction (prep refuses a group
    whose members name two, `group.envelope`; a benchmark measures on one,
    `prep_inputs.bench_inputs`) -- or ``None`` where it placed none: a
    machine with no queues, a prep that wrote no header.  Launch reads it and
    decides nothing (`scheduler.md` R9)."""
    p = jobs[0].placement or {}
    return (_Placed(p["domain"], p["partition"], p["qos"])
            if p.get("domain") else None)


@dataclass(frozen=True)
class _Sent:
    """THE REQUEST A LINE SENDS, read off what is sent (`job-system.md`
    § 6.0: launch places nothing again) -- a queue's: the header its prep
    wrote, its wall and memory as the header carries them
    (`kinds.Kind.directive_value`), the wall the launch's ``--time`` when it
    restates it (``restated``); a run here's: the wall its own card states
    (`running-a-job.md` § 3.6).  What ``run.json``'s ``placed_on`` records
    beside the queue and the job is told (``MB_WALL_S``)."""
    time: Optional[str] = None
    mem: Optional[str] = None
    #: The wall is the launch's ``--time``, restated on the line.
    restated: bool = False


def _wall_s(r: Resources) -> Optional[int]:
    """The wall a run's own card states, in seconds -- what a run here is
    told (`running-a-job.md` § 3.6) -- or ``None``: it runs until done."""
    from ..scheduler.quantities import parse_walltime
    return parse_walltime(r.time) if r.time else None


def _counts_of(r: Resources) -> List[str]:
    """A run script's counts, as its line passes them -- ``-np N``, ``-omp
    M`` -- the ones stated."""
    return ((["-np", str(r.mpi_np)] if r.mpi_np else [])
            + (["-omp", str(r.cpus_per_task)] if r.cpus_per_task else []))


def _run_sh_args(r: Resources, run: int, *, cold: bool = False
                 ) -> List[str]:
    """What every ``.run.sh`` is started with: its run's number, which
    launch decides and the script refuses to start without (``--run N``,
    `project-layout.md` § 1.6.1), its job's ``-np`` / ``-omp``
    (runwrap.py § arg-parsing), and ``--cold --force`` for a flat stage
    launched again cold -- the person said so at launch, and was shown the
    line (`job-system.md` § 5.4)."""
    args: List[str] = ["--run", str(run)]
    if cold:
        args += ["--cold", "--force"]
    return args + _counts_of(r)


# --------------------------------------------------------------------- #
#  the two execution paths                                              #
# --------------------------------------------------------------------- #

def _machine_of(base) -> "Environment":
    """THE MACHINE THE CALCULATION IS SET TO -- its own copy of that
    machine's record, prep's file and no other (`configuration.md` M-3;
    `scheduler.md` R9), read through its door
    (`record.calculation_environment`), whose scheduler kind spells the line
    (§ 6a).  Read to spell, never to decide: what is sent is prep's."""
    from ..scheduler.record import UnknownTarget, calculation_environment
    try:
        rec = calculation_environment(base)
    except UnknownTarget as exc:
        raise SubmitError(str(exc)) from None
    if rec is None:
        raise SubmitError(
            f"{base} holds no record of its machine -- a calculation is "
            f"launched after its prep, which copies it (configuration.md "
            f"M-3)")
    return rec


def _job_id_of(kind, answer: str) -> str:
    """The id the scheduler answered a send with (its kind's ``job_id``)."""
    jid = kind.job_id_of(answer)
    if jid is None:
        raise SubmitError(f"could not read the job's id from {kind.what}'s "
                          f"answer: {answer!r}")
    return jid


def _own_walls(mode: str, inside: Optional["Inside"]) -> bool:
    """Whether each run is told its own card's wall (`running-a-job.md`
    § 3.6): a run here, or a walk here -- unless it rides a sequence's job
    sent to a queue, whose every run counts the job's wall from the job's
    start, handed down unchanged."""
    return mode == "direct" and not (inside is not None
                                     and inside.in_a_queue_job)


def _needs_header(mode: str, machine) -> bool:
    """Whether a submission's header must be there: it is sent
    (``submit``), or asked about where the machine has a scheduler to ask
    -- its record's kind, never this computer's PATH (`scheduler.md`
    § 6a)."""
    return mode == "submit" or machine.kind.queue


def _scheduler_job_name(jobset: JobSet, name: str) -> str:
    """What the scheduler lists: the calculation, then the job within it --
    ``bdt_au/coarse``, ``bdt_au/G1K2C4``, ``bdt_au/bench-group-cpu``
    (`job-system.md` § 6).

    The id first, because that is the thing you are trying to tell apart
    when several calculations are queued at once, and the job second
    because within one calculation that is the question.  EVERY door names
    its job here.
    """
    return f"{jobset.name}/{name}"


#: How many questions to the scheduler one `--mode ask` will make.
#:
#: Politeness, not a rule about queues: asking enqueues nothing, so the
#: one-at-a-time submission rule does not reach it.  A benchmark grid is
#: the case that matters and is usually a handful of shelves; past this the
#: rest are named as unasked rather than silently dropped.
ASK_MAX_QUERIES = 24

def _no_sbatch(what: str, name: str, *, base, told, machine,
               redo: Optional[str] = None) -> str:
    """The one answer to *there is no scheduler header*, whichever door
    finds it missing (`job-system.md` § 6.1): prep withholds the
    ``.sbatch`` where the machine it prepared for names no queue, or where
    it was told to (``prep --no-sbatch``) -- each said with its own way
    back.  A machine with a queue is another machine, and a
    calculation is set to the machine of its first prep (`configuration.md`
    M-3): the way there is the state saved before that prep.

    THE RUN HERE AS A COMMAND -- the launch ``told`` names, with its flags
    as typed that a run here reads (`commands.launch_with`).  ``redo`` is
    how the work is prepared again (`_redo`; a benchmark's, the state saved
    before its prep).  ``machine`` is the calculation's copy of its
    machine's record (:func:`_machine_of`)."""
    from .commands import block, launch_with
    here = block([launch_with(told["kind"], told["stage"], told["trial"],
                              base=base, mode="direct",
                              typed=told["flags"],
                              groups=(told["group"]
                                      if isinstance(told.get("group"), list)
                                      else [told["group"]]
                                      if told.get("group") is not None
                                      else ()))])
    if machine.kind.queue:
        # THE MACHINE NAMES A QUEUE, so prep was told to write no header
        # (`prep --no-sbatch`, `job-system.md` § 6.1's second answer).
        return (f"{what}: there is no scheduler header ({name}) -- it was "
                f"prepared with --no-sbatch.  To run it here:\n{here}\n"
                f"  To send it to a queue, prepare it again without "
                f"--no-sbatch: "
                + (redo or rollback("the benchmark's prep", base=base)))
    return (f"{what}: there is no scheduler header ({name}) -- the "
            f"machine it was prepared for names no queue (its record's "
            f"scheduler has none, job-system.md § 6.1).  To run it here:\n"
            f"{here}\n  For a machine with a queue, prep it for that "
            f"machine (--target, its record's name) -- a calculation is "
            f"set to the machine of its first prep, so "
            + rollback("its first prep", base=base))


def _sbatch_request(*, header: Path, placement: "Optional[_Placed]",
                    time_s: Optional[int], job_name: str, script: str,
                    run_args: Sequence[str], machine
                    ) -> Tuple[Optional[_Sent], object, List[str],
                               Optional[str]]:
    """THE ONE LINE, for every door that hands work to the scheduler -- a
    stage, a group, a grouped bench's shelf, a run of points
    (`job-system.md` § 6.1): the header prep wrote -- for a walk, the one
    written from the queue prep recorded and its members' requests -- sent
    as it is.  **Launch decides nothing** (`job-system.md` § 6.0;
    `scheduler.md` R9): the line names the job, carries what the launch
    restates (`commands.LAUNCH_FLAGS` -- a submitted job's wall,
    ``time_s``), the launch claim and the run's arguments, in the words of
    the scheduler kind of ``machine`` -- the calculation's copy of its
    machine's record -- from where its commands are there
    (`kinds.Kind.line`, `scheduler.md` § 6a).

    Returns ``(sent, placement, command, outputs_in)``: the request as
    sent, read off ``header`` -- the one prep wrote, where it lies -- its
    wall the one restated (:class:`_Sent`), which ``run.json`` records beside
    the queue (:func:`_placed_on`); the placement prep recorded; the exact
    line; and the folder, from where the line runs, the scheduler writes the
    job's own output files in: the header's own, as it asks
    (`runwrap.render_sbatch`'s ``where``) -- ``None``, and no line, with no
    queue.

    ``run_args`` are the run script's arguments (`_run_sh_args`: ``--run
    N``, ``-np``, ``-omp``), set after the ``.sbatch`` name, which forwards
    them (``bash <base>.run.sh "$@"``).  A submission whose script walks
    several members hands each member its own inside that script, and passes
    none here.
    """
    from ..scheduler.emit import Directives
    from .commands import launch_flag
    kind = machine.kind
    # A MACHINE WITH NO QUEUE HAS NO LINE: asked about, it answers that
    # there is no scheduler (:func:`_ask`); sent, its missing header is
    # refused first (:func:`_needs_header`).
    if not kind.queue:
        return None, placement, [], None
    lines = Directives.lines_of(header.read_text(errors="replace"), kind)
    row = launch_flag("--time")
    sent = _Sent(time=(_slurm_time(time_s) if time_s is not None
                       else kind.directive_value("wall", lines)),
                 mem=kind.directive_value("memory", lines),
                 restated=time_s is not None)
    if sent.time is None:
        # A HEADER THAT STATES NO WALL is not one prep wrote: prep states
        # every job's wall or refuses (R-P13).  Refused here, the header
        # named, as a prepared stage that is not what prep writes is
        # (R-L15, R-L16).
        raise SubmitError(
            f"{header} states no wall ({kind.name}'s `wall` directive), so "
            f"the launch cannot say what it sends -- prep writes one into "
            f"every header, so this one was edited since.  Prepare it again "
            f"where nothing of it was launched, or restore the state saved "
            f"before it was edited (`molbuilder checkpoint list`); --time "
            f"states the wall for this launch alone (job-system.md § 6.0, "
            f"R-L44).")
    restated = (kind.restated(row.says, sent.time) if sent.restated
                else [])
    # The launch-door claim, EXPLICIT on the line: environment inheritance
    # alone is fragile (a site may not pass a plain variable on), and the
    # line wins over site defaults, so the claim reaches the job wherever it
    # runs (job-contracts.md § 2.6).
    # THE WALL IT IS SENT WITH, told to the job (`running-a-job.md`
    # § 3.6): the one restated, else the header's -- the value `run.json`
    # records as `placed_on.time` -- in seconds.
    from ..scheduler.quantities import parse_walltime
    cmd = kind.line(machine.scheduler.commands(), job=job_name,
                    restated=restated, launched_by=LAUNCH_CLAIM,
                    header=script, args=run_args,
                    wall_s=parse_walltime(sent.time))
    return sent, placement, cmd, os.path.dirname(script)


def _ask(name: str, cmd: List[str], cwd: Path, *, kind,
         domain: Optional[str] = None) -> JobResult:
    """The line that WOULD be sent, asked about -- its kind's ``ask`` after
    the command (`kinds.Kind.asking`): the scheduler answers instead of
    enqueueing.  Writes and records nothing.

    The result carries the line that would be SENT; the ask goes in only
    in what is run, so it cannot be shadowed and is never shown as the line
    to send.  The header asked about is the one the line sends, its prep's.
    No scheduler is an answer, not an error: on a machine whose record's
    kind has no queue there is nothing to queue behind.
    """
    from .ask import Prediction, answer_of
    if not kind.queue:
        return JobResult(name, cmd, "asked", domain=domain,
                         prediction=Prediction(no_scheduler=True,
                                               refused="no scheduler here"))
    try:
        cp = subprocess.run(kind.asking(cmd), cwd=str(cwd),
                            capture_output=True, text=True)
    except OSError as exc:
        # "I could not reach the scheduler" is what the person asked about.
        return JobResult(name, cmd, "asked", domain=domain,
                         prediction=Prediction(
                             refused=f"could not run {cmd[0]!r}: {exc}"))
    # A non-zero return is an answer too -- "this queue cannot take it" is
    # often the one worth reading.
    return JobResult(name, cmd, "asked", domain=domain,
                     prediction=answer_of(kind, (cp.stdout or "")
                                          + (cp.stderr or "")))


def _rel(path, base) -> str:
    """``path`` from the calculation's folder when it lies in it -- how the
    plan names what it holds."""
    try:
        return str(Path(path).relative_to(base))
    except (TypeError, ValueError):
        return str(path)


def _as_found(path, base) -> str:
    """A file the plan reads where it lies, as it is now (`planned.found`)."""
    return f"{_rel(path, base)}: {found(path)}"


def _gathered_from(took: List[dict]) -> str:
    """What a gather takes, each file by the run it comes from --
    ``T.DM from 01_seed/run-0; T.TS.HSX from 04_device/run-1`` -- a point's
    file said by its run (`RunRef.run_folder`), its points the walk's to say
    (`job-system.md` § 6.0, step 1: what each member takes, shown)."""
    by: Dict[str, List[str]] = {}
    for t in took:
        files = by.setdefault(t["from"].run_folder(), [])
        if t["file"] not in files:
            files.append(t["file"])
    return "; ".join(f"{', '.join(fs)} from {run}" for run, fs in by.items())


@dataclass
class _Member:
    """One prepared attempt a submission runs, decided BEFORE anything is
    written (`job-system.md` § 6.0, step 1) -- a run is one member, a
    benchmark's shelf its pending trials, a bias sweep its points: where it
    runs, where its deck and wrappers are read from now, and what it
    follows."""
    job: Job
    #: The job's own directory: the stage's, the trial's or the point's.
    container: Path
    #: Where it runs: its attempt -- for a continuation, the one the send
    #: opens -- or, with no attempt layer, the container.
    run_dir: Path
    #: Whether the run has an attempt of its own (else the container is it).
    has_attempt: bool
    #: Where its deck and wrappers are read before it is sent -- the files
    #: its run runs from: the stage's own folder its run is filled from
    #: (`materialize.prepare_attempt`) -- on the flat layout the folder it
    #: runs in; a run of points, its first walked point's; a benchmark's
    #: trial, its run.
    read_from: Path
    #: The names of its stage's files (`naming.run_names`): what its
    #: launch record, and what it continues from, are named by.
    names: "RunNames"
    #: Its run's number, decided here (`runrecord.next_run`) and handed to
    #: its run script (``--run N``; `project-layout.md` § 1.6.1).
    run: int
    #: What the results and the ledger call it when the job's name does not
    #: -- a bias point's ``<stage>@<point>``.
    label: Optional[str] = None
    #: It is a stage launched before, launched again (`job-system.md` § 5.4).
    again: bool = False
    #: Its stage's FIRST launch, taking what it builds on (`job-system.md`
    #: § 6.0, step 1).
    first: bool = False
    #: Cold: it takes nothing from a run -- launched again, nothing of its
    #: own; at a first launch, nothing of the stage before it.
    cold: bool = False
    #: What it continues from -- at a first launch, what it builds on (the
    #: stage before it, or a force-constant stage's `relax` run its prep
    #: took); launched again warm, its own latest run, or a run named by
    #: ``--from`` -- one `Continuation` from the one door, or ``None``.
    continuation: Optional[Continuation] = None
    #: WHAT ITS RUN TAKES FROM OTHER RUNS, each one hand-over
    #: (`materialize.HandOver`, `job-system.md` § 5.4): the run it continues
    #: from and what it copies -- a stage's restart files, nothing on the
    #: flat layout, where they lie in the folder; a run of points continuing
    #: from the run its points are taken over from -- and a transport rung's
    #: inputs, gathered at its first launch, anew with ``--gather``, or
    #: launched again those its last run was given.  A point's own
    #: hand-overs are its walk's (:func:`_points_walk`).
    hands: Tuple[HandOver, ...] = ()
    #: The gather as the ledger records it: each file, the run it came
    #: from, the point it was gathered at -- a gather made at this launch.
    took: List[dict] = field(default_factory=list)
    #: A RUN OF POINTS: each point's folder in its run and the point
    #: (`transport.stages.points_in`), as its planner opened them -- the
    #: points' walk sends them (:func:`_points_walk`) -- or ``None`` for a
    #: run of one.
    points: Optional[list] = None
    #: Launched again: the stage's newest launched run, which it continues
    #: (:func:`_newest_launched`) -- ``None`` at a first launch, and on the
    #: flat layout.
    ran: Optional[Path] = None
    #: A run of points' walk, said whole (:func:`_points_walk`): which
    #: points it takes over from the newest launched run and which it runs,
    #: each from what start -- ``None`` for a run of one.
    walk: Optional[str] = None
    #: A RUN OF POINTS' FIRST LAUNCH in a calculation that reuses a finished
    #: one: the points it takes over from that calculation
    #: (`jobset.reuse.Take`; `engines/transport.md` § 2a.16).
    copied: tuple = ()
    #: The calculation's folder, so what the plan says names it.
    base: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.label or self.job.name

    @property
    def continued(self) -> Optional[HandOver]:
        """Its run's continued hand-over -- the run it continues from --
        or ``None``."""
        return next((h for h in self.hands if h.record == CONTINUED), None)

    @property
    def carries(self) -> List[str]:
        """The restart files it copies from the run it continues from."""
        h = self.continued
        return h.names_taken if h is not None else []

    @property
    def warned(self) -> Dict[str, str]:
        """The files it takes that were written as their run ended
        (`running-a-job.md` § 3.6): a hand-over's, from the folders it
        copies from; on the flat layout, the warm files the run reads in
        place, against the run they were written by."""
        from .materialize import written_as_ended
        h = self.continued
        if h is not None and h.files:
            return h.written_as_ended(self.base)
        # ON THE FLAT LAYOUT the hand-over copies nothing (its record alone
        # is written): the run reads the warm files in the folder, against
        # the run they were written by.
        c = self.continuation
        if c is None or not c.run:
            return {}
        present = [w.name for w in self.job.warm
                   if (self.run_dir / w.name).is_file()]
        return written_as_ended(self.run_dir, self.names.label, present,
                                run=c.written_by or c.run)

    @property
    def gathered(self) -> List[str]:
        """A transport rung's inputs, copied with their record."""
        return [n for h in self.hands if h.record == GATHERED
                for n in h.names_taken]

    @property
    def follows(self) -> bool:
        """It takes from a run, or says it takes nothing -- a stage launched
        again, or a stage's first launch."""
        return self.again or self.first

    def _where(self) -> str:
        """Where it runs: its attempt, or its folder, as the run its number
        names."""
        return (f"into {self.run_dir.name}" if self.has_attempt
                else f"in the same folder, as its run {self.run}, where its "
                     f"files are")

    def _copied(self) -> List[str]:
        # EVERY FILE IT TAKES: what it carries from the run it continues,
        # then what was gathered for it -- never one name twice: warm, a
        # name it carries itself is not gathered (a device's `.DM`).
        return list(dict.fromkeys([*self.carries, *self.gathered]))

    def said(self) -> str:
        """The member as the plan holds it, in one line: where it runs, what
        it follows, how that run ended and what is copied from it."""
        line = f"{self.name}: runs in {_rel(self.run_dir, self.base)}"
        if self.walk is not None:
            line += "; " + self.walk
        elif self.again:
            line += "; it " + self._takes(would=True)
        return line

    def _takes(self, *, would: bool) -> str:
        """What it takes, warm or cold."""
        if self.continuation is not None:
            return self.continuation.line(self._copied(), would=would,
                                          warned=self.warned)
        # A SWEEP'S WALK SAYS WHICH POINTS RUN AND FROM WHAT -- what its
        # first launch gathers, or warm or cold what it takes over
        # (`engines/transport.md` § 2a.11: "Launch shows which points it
        # takes over and which it runs"); cold, it is every point.
        if self.walk is not None:
            return self.walk
        start = "would start" if would else "starts"
        if self.first:
            if self.took:
                return (f"{'would gather' if would else 'gathered'} "
                        + _gathered_from(self.took))
            return (f"{start} from its deck"
                    + (" -- cold, " if self.cold else ", ")
                    + "nothing taken from another run")
        gathered = (f", with the inputs its kind gathered "
                    f"({', '.join(self.gathered)})" if self.gathered else "")
        if self.cold:
            return (f"{start} cold -- from its deck alone, nothing taken "
                    f"from a run of its own{gathered}")
        return (f"{start} over -- nothing is handed on from a run of its "
                f"own{gathered}")

    def would(self) -> str:
        """What it follows, said before anything is sent."""
        return (f"WOULD launch it{' again' if self.again else ''} "
                f"{self._where()}: it " + self._takes(would=True))

    def note(self) -> Optional[str]:
        """The line that says what this launch follows, once it is sent, or
        ``None``."""
        if not self.follows:
            return None
        # THE WAY TO START OVER, beside a launch again that took from a run
        # -- a run continued, a run of points' done points taken over; a
        # cold one, or one that took nothing, already did.
        warm = not self.cold and (self.continuation is not None
                                  or self.walk is not None)
        return (f"launched{' again' if self.again else ''} {self._where()}: "
                f"it " + self._takes(would=False)
                + (".  To start it over, launch it again with --cold "
                   "(job-system.md § 5.4)" if self.again and warm else ""))


@dataclass
class Submission:
    """One submission of a launch -- one scheduler job, or one process here
    -- walking its members, its exact line and every script it needs decided
    in the plan (`job-system.md` § 6.0, step 1)."""
    name: str
    #: The exact line -- what is shown, asked about and sent.
    command: List[str]
    #: Where the line is run.
    cwd: Path
    #: Run here (``bash``, left in the background or waited for,
    #: `LaunchPlan.background`), or handed to the scheduler.
    direct: bool
    members: List[_Member]
    #: The queue prep placed it on (:func:`_placed_by_prep`) -- ``None`` for
    #: a run here, or on a machine with no queues.
    placement: "Optional[_Placed]" = None
    #: The request as sent (:class:`_Sent`) -- a queue's read off its
    #: header, its wall the launch's ``--time`` when restated
    #: (:func:`_sbatch_request`); a run here's, the wall its card states:
    #: what its ``run.json`` records beside the queue, and the job is told
    #: (`job-system.md` § 6.0).  ``None`` for a walk run here, whose members
    #: are each told their own (:func:`_walk_script`).
    sent: Optional[_Sent] = None
    #: What each member says in the results when the submission walks
    #: several (``rides the group``, ``rides the sweep``); ``None`` for one
    #: run, which is the submission itself.
    rides: Optional[str] = None
    #: Where `ask` asks about it, when not where the line is run.
    ask_in: Optional[Path] = None
    #: The header the line sends, where it lies -- its prep's: what the
    #: question and a dry run show the request as (:meth:`LaunchPlan.shown`).
    header: Optional[Path] = None
    #: What a refusal by the scheduler adds to the scheduler's own words.
    refusal_hint: str = ""
    #: Where a walk's members write their output -- a benchmark's trials, a
    #: group's stages, a sweep's points: the one path its script writes and
    #: the plan shows (:func:`_walk_script`).
    log: Optional[Path] = None
    #: The calculation's copy of its machine's record, whose scheduler kind
    #: spelled the line (:func:`_machine_of`) -- ``None`` for a run here.
    machine: Optional[object] = None
    #: The folder, from ``cwd``, the scheduler writes the job's own output
    #: files in -- the header's own (:func:`_sbatch_request`): ``""``, the
    #: run's folder, or a walk's ``launch/`` -- ``None`` for a run here.
    outputs_in: Optional[str] = None
    #: A SEQUENCE OF GROUPS (`job-system.md` § 6.0): ``{"name", "groups":
    #: [{"number", "stages"}], "record"}`` -- its name, its groups in order
    #: and its launch record's path -- its job walking the groups' launches,
    #: no member of its own; ``None`` for every other submission.
    sequence: Optional[dict] = None
    #: What the plan shows before this submission's own line -- a
    #: sequence's: its first group's plan, and each later group said
    #: (:func:`_plan_sequence`).
    preview: List["JobResult"] = field(default_factory=list)

    def scheduler_said(self, job_id: Optional[str],
                       run_dir: Path) -> Dict[str, Optional[str]]:
        """What a member's launch record says of the scheduler the job was
        sent to (`runrecord.write_launch`'s ``scheduler``): its kind, and
        where it writes the job's error file from the member's own folder
        ``run_dir`` -- the file the run door reads for how it ended the job
        (`scheduler.md` R16); the file unnamed when the job's id was not
        read."""
        from ..runfiles import SCHEDULER_STDERR
        kind = self.machine.kind
        stderr = None
        if job_id is not None and self.outputs_in is not None:
            err = (Path(self.cwd) / self.outputs_in
                   / kind.output_file(SCHEDULER_STDERR, job_id=job_id))
            stderr = os.path.relpath(err, run_dir)
        return {"kind": kind.name, "stderr": stderr}

    @property
    def domain(self) -> Optional[str]:
        """The NAME of the queue prep placed it on."""
        return self.placement.name if self.placement else None

    def request(self) -> Tuple[str, ...]:
        """The request the line sends -- its header's directive lines, as
        its prep wrote them (`Directives.lines_of`), with the wall the line
        restates in place of the header's, said so: the scheduler takes the
        line's over the header's, so that is the wall sent (`scheduler.md`
        R9); ``()`` for a run here, or a header not written."""
        from ..scheduler.emit import Directives
        if self.direct or self.header is None or not self.header.is_file():
            return ()
        kind = self.machine.kind
        lines = Directives.lines_of(self.header.read_text(errors="replace"),
                                    kind)
        if self.sent is not None and self.sent.restated:
            said = kind.directive("wall", wall=self.sent.time)[0]
            for i, line in enumerate(lines):
                had = kind.directive_value("wall", [line])
                if had is not None and had != self.sent.time:
                    lines[i] = (f"{said} (this launch's --time; the header "
                                f"says {had})")
        return tuple(lines)


@dataclass(frozen=True)
class Inside:
    """A GROUP'S LAUNCH INSIDE A SEQUENCE'S JOB (`job-system.md` § 6.0, *A
    sequence of groups*): the sequence's name, its launch record, and the
    job's address -- a queue job's id, or a walk run here's
    ``<host>-<pid>`` -- as its walk types them (``--in-sequence``,
    ``--in-job``)."""
    name: str
    record: object
    job: str
    #: The scheduler kind of the calculation's machine, which names the
    #: job's error file -- set by the plan, which reads the machine.
    kind: object = None

    @classmethod
    def of(cls, base, name: str, job: str) -> "Inside":
        """Read the sequence's launch record from the calculation's
        `launch/` (`sequence.read_record`); a record not there or not
        reading refused (R-L46)."""
        from ..runfiles import GroupNames
        from .sequence import SequenceError, read_record, record_path
        try:
            rec = read_record(record_path(base, GroupNames(name)))
        except (SequenceError, ValueError) as exc:
            raise SubmitError(f"--in-sequence {name}: {exc} -- a sequence's "
                              f"own launch types it (job-system.md R-L46).")
        return cls(name=name, record=rec, job=str(job))

    @property
    def in_a_queue_job(self) -> bool:
        """Whether the job it rides was sent to a queue: then every run
        inside it is told the JOB's wall, counted from the job's start, as
        the members of any walk sent to a queue are -- never its own card's
        (`running-a-job.md` § 3.6)."""
        return self.record.mode == "submit"

    def run_said(self, m) -> Dict[str, object]:
        """What member ``m``'s launch record says of the job it rides: the
        sequence's record's (`sequence.Record.run_said`), with the
        sequence's name."""
        said = self.record.run_said(
            job_id=self.job if self.record.mode == "submit" else None,
            run_dir=m.run_dir, base=m.base, kind=self.kind)
        return {**said, "sequence": self.name}


@dataclass
class LaunchPlan:
    """One launch, whole, decided before anything is sent
    (`job-system.md` § 6.0, step 1): its submissions, the trials passed
    over, what the send writes and the files read where they lie -- shown,
    asked, and then sent as it is (:func:`send_launch`)."""
    base: Path
    mode: str
    submissions: List[Submission]
    #: Trials passed over by name -- measured before -- by a walk, here or
    #: a queue's shelves, and by a question to the scheduler.
    skipped: List[JobResult] = field(default_factory=list)
    #: What the send writes before anything goes: an attempt a re-launch
    #: opens and what is copied into it (`materialize.prepare_attempt`, the
    #: one opener), a shelf's or a chain's scripts.
    writes: Plan = field(default_factory=Plan)
    #: Each file read where it lies, as it was found -- every member's deck,
    #: run script and header (`planned.found`).
    reads: List[str] = field(default_factory=list)
    #: A submission the scheduler refuses leaves the rest to go -- a
    #: benchmark's shelves, each measuring on its own -- or ends the launch.
    tolerant: bool = False
    #: The same plan, made again from the folder as it is now -- the send's
    #: check (§ 6.0, step 4).
    remake: Optional[Callable[[], "LaunchPlan"]] = None
    #: What the verb was told -- the kind, the stage, the trial, the mode and
    #: where it came from, the queue's source, which config files answered
    #: -- written on each of this launch's lines in the ledger.
    told: Dict[str, object] = field(default_factory=dict)
    #: Whether this launch writes its decisions down: a dry run writes
    #: nothing, the ledger included (§ 6.0, step 3), and neither does the
    #: plan the send makes again to compare.
    records: bool = True
    #: The queue the work goes to and where that came from -- the queue its
    #: prep recorded (:func:`_the_queue`); ``(None, None)`` for a run here
    #: or a machine with no queues.
    queue: Tuple[Optional[str], Optional[str]] = (None, None)
    #: A run here started and left to run -- a run here's default; waited
    #: on under ``launch --foreground``, and the question's third answer
    #: takes the other way (`job-system.md` § 6.0): the send returns once
    #: the process exists; its output is in the files its scripts write
    #: (the run's runwrap log, a walk's log), and `status` follows it.
    background: bool = False
    #: A GROUP'S FIRST LAUNCH saves the folder's state at the send, before
    #: anything is written (`job-system.md` § 6.0, step 4): what the save's
    #: note says it is before (``launch task group 2``) -- ``None`` for a
    #: launch again.
    save_before: Optional[str] = None
    #: The state that save made or stood at, said (`checkpoint.Kept.said`).
    saved: Optional[str] = None
    #: The calculation's engine, which the save reads its ignores by.
    engine: Optional[str] = None
    #: A GROUP'S LAUNCH INSIDE A SEQUENCE'S JOB (:class:`Inside`): each run
    #: it sends records that job, and it adds its runs to that job's
    #: address instead of writing its own (`job-system.md` § 6.0).
    inside: Optional["Inside"] = None
    #: QUESTIONS ASKED BEFORE THE PLAN -- which group to launch -- each a
    #: ledger line's facts, written with the send's own question: after the
    #: save, so the state saved holds no line of this launch, as it holds
    #: none of a prep (§ 5.0, checkpoint 7).
    held: List[dict] = field(default_factory=list)

    def record(self, decision: str, **facts) -> None:
        """One of this launch's decisions, written down (§ 6.0, step 5;
        :func:`_record`)."""
        if self.records:
            _record(self.base, self.told, decision, **facts)

    def asked(self, said) -> None:
        """The questions this launch put, as asked, with their answers --
        those held from before the plan, then the send's own."""
        for facts in self.held:
            self.record("question", **facts)
        self.held = []
        self.record("question",
                    about=("submit" if self.mode == "submit" else
                           "run here, in the background" if self.background
                           else "run here, in the foreground"),
                    answer=said.words)

    def made_from(self) -> List[str]:
        """What this plan was made from, line by line -- each submission's
        line and where it runs, each member and what it follows, every file
        the send writes or copies, every file read where it lies.  The send
        makes the plan again and compares (§ 6.0, step 4)."""
        out = [f"{r.name}: {r.shown}" for r in self.skipped]
        for s in self.submissions:
            out.append(f"{s.name}: {' '.join(s.command)} "
                       f"(in {_rel(s.cwd, self.base)})")
            out += [m.said() for m in s.members]
        out += [line.replace(f"{self.base}{os.sep}", "")
                for line in self.writes.described()]
        return out + list(self.reads)

    def shown(self) -> List[JobResult]:
        """The plan as the question and a dry run show it: the trials passed
        over, what each member follows,
        each submission's exact line and queue, and the members riding
        it."""
        out: List[JobResult] = list(self.skipped)
        for s in self.submissions:
            out += list(s.preview)
            out += [JobResult(m.name, [], "takes", said=m.would())
                    for m in s.members if m.follows]
            out.append(JobResult(s.name, s.command, "planned",
                                 domain=s.domain, request=s.request()))
            if s.rides:
                # EACH MEMBER WITH THE COUNTS ITS WALK HANDS IT (`-np`,
                # `-omp`, :func:`_run_sh_args`) -- what runs it is seen too.
                out += [JobResult(m.name, [], "rides",
                                  said=" ".join([s.rides] + _counts_of(
                                      m.job.resources)))
                        for m in s.members]
        return out


#: `--gather` AT A FIRST LAUNCH: nothing to take anew -- the first launch
#: gathers, and a flag that would not be read is refused by name.
GATHER_AT_FIRST = ("--gather takes a transport rung's inputs anew when it is "
                   "launched again; its first launch gathers them from each "
                   "upstream rung's newest run already (job-system.md § 5.4).")


def _launched(where, names: RunNames) -> bool:
    """Was this run launched -- the one door's answer
    (`runrecord.launch_record`), asked in the folder it runs in with its
    stage's names, every gate of launch asking it.  A record that does not
    read is a refusal naming the file (:func:`plan_launch`): launching over
    it could send a job twice, and not launching could hide one in the
    queue."""
    from ..runrecord import launch_record
    return launch_record(where, names) is not None


def _redo(calc, job) -> str:
    """How a prepared stage is changed -- the one way back
    (`commands.way_back`), asked of its runs."""
    from .commands import way_back
    return way_back(job.name, calc=calc, launched=ever_launched(calc, job))


def _as_described(calc, job, *, described, written_at=None):
    """THE PREPARED STAGE IS STILL WHAT THE DESCRIPTION GIVES
    (`job-system.md` § 6.0, step 1) -- asked at a first launch and at a
    launch that takes inputs anew: the stage planned again as its prep
    planned it, quietly (`prep.plan_stage`: the description read and
    checked as prep reads it, then what its prep was told, `Job.asked`; a
    force-constant stage at ``written_at``, the `relax` run its prep took),
    and its deck, run script, header and row in
    `job-set.json` compared with what is on disk, their stamps masked (when
    and by which build each was written).  A difference -- a template value,
    a run card's ranks, a junction re-cited since the prep -- is refused,
    naming the way to prepare it as it is described now: an edit never runs
    unsaid on another stage's new files.  ``described`` gives the
    calculation as prep checks it, once for the whole launch
    (``_planned``)."""
    from .errors import as_prep_error, step_errors
    from .planned import Plan
    from .errors import PrepError
    from .prep import plan_stage
    base = Path(calc.base)
    asked = job.asked or {}
    try:
        try:
            d = described()
            now = plan_stage(base, d, "task", job.name,
                             allocation=Resources.from_dict(
                                 asked.get("allocation") or {}),
                             env=asked.get("env"),
                             emit_sbatch=asked.get("emit_sbatch", True),
                             plan=Plan(), continuation=written_at,
                             quiet=True)
        except step_errors() as exc:
            raise as_prep_error(exc) from exc
    except PrepError as exc:
        raise SubmitError(
            f"`{job.name}`, planned again as its prep planned it, is refused "
            f"now -- the description no longer gives what was prepared:\n  "
            f"{exc}\n  To prepare it as it is described now, "
            + _redo(calc, job)) from None
    # ITS DECKS, SCRIPTS AND ROW against what prep writes now -- the one
    # comparison (`prep.prepared_differs`).
    from .prep import prepared_differs
    differ = prepared_differs(calc, now.plan, job, now.job)
    # WHICH MOLBUILDER PREPARED IT, read off the deck's own record: a stage
    # prepared before an upgrade is said so by name (R-L15).
    deck = next((p for p in now.plan.writes()
                 if p.name == Path(job.script).name and p.is_file()), None)
    deck_read = deck is not None
    prepared_by = None
    if deck_read:
        from ..script_emit import written_by
        prepared_by = written_by(deck.read_text(errors="replace"))
    if differ:
        from .. import __version__
        from ..script_emit import molbuilder_words
        upgraded = (f"; it was prepared by {molbuilder_words(prepared_by)}, "
                    f"and this is molbuilder {__version__}"
                    if deck_read and prepared_by != __version__ else "")
        raise SubmitError(
            f"`{job.name}` was prepared from the description as it stood "
            f"then, and prep would write {', '.join(differ)} differently now "
            f"-- the template, its run card, or what it is composed from "
            f"changed since, or molbuilder did{upgraded} (job-system.md "
            f"§ 6.0).  Launched as it is, it would run the old setup on the "
            f"files it takes now.  To prepare it as it is described now, "
            + _redo(calc, job))
    return now


def _prep_files_back(prepared, run: Path, writes: Plan) -> None:
    """A FIRST LAUNCH'S RUN, emptied whole, given prep's own files back as
    they lie (`job-system.md` § 6.0, step 4): the ones ``prepared`` -- prep's
    plan, made again for the check (:func:`_as_described`) -- puts below
    ``run`` that the launch's opener has not brought from the stage's folder
    (a seeded progress log), each as prep wrote it."""
    top = Path(os.path.normpath(os.path.abspath(run)))
    for path in prepared.plan.writes():
        if (top in path.parents and not writes.is_file(path)
                and path.is_file()):
            writes.bytes(path, path.read_bytes())


def _plan_trial(jobset: JobSet, base: Path, job, *, mode: str, shape,
                named: bool = False):
    """Where a benchmark's trial runs -- a :class:`_Member` -- or, measured
    before, the result that passes it over by name.  A TRIAL is immutable
    once launched (`project-layout.md` § 1.5): a walk -- here, or a queue's
    shelves -- and a question to the scheduler pass the measured ones over;
    one the person NAMED (``named``) is refused when it is sent or run here,
    and a question to the scheduler says it already ran."""
    from .commands import command
    from ..paths import attempts_in
    from .materialize import bench_stage_of, run_dir
    from .naming import job_dir_names, run_names
    from ..runrecord import launch_record_path, next_run
    sh = shape
    names = run_names(jobset, job, sh)
    container = base / job_dir_names(jobset, sh)[job.name]
    ns = attempts_in(container)
    # WHERE A TRIAL'S FILES ARE: its attempt where the shape keeps one, the
    # trial's own folder on the flat layout (`materialize.run_dir`).
    where = run_dir(container)
    if not _launched(where, names):
        return _Member(job, container, where, bool(ns), where, names=names,
                       run=next_run(where, names), base=base)
    if mode == "ask" or not named:
        return (JobResult(job.name, [], "already run") if mode == "ask"
                else JobResult(job.name, [], "skipped",
                               said="skipped -- already launched"))
    stage = bench_stage_of(container)
    read_back = (command("summarize", "bench", stage, base=base)
                 if stage else "`summarize bench` on the sweep's stage")
    said = launch_record_path(where, names, FIRST_ATTEMPT)
    raise SubmitError(
        f"trial {job.name!r}: already launched -- {said} records it.  A "
        f"trial measures its point ONCE (project-layout.md § 1.5: immutable "
        f"once it has run).\n"
        f"  read what it measured:\n    {read_back}\n"
        f"  measure it again -- a prepared benchmark is not prepared again "
        f"(job-system.md § 5.0): "
        + rollback("the benchmark's prep", base=base))


def _plan_member(calc, job, *, writes: Plan,
                 cold: bool = False, from_attempt: Optional[str] = None,
                 gather: bool = False, described=None) -> "_Member":
    """**THE MEMBER PLANNER** -- one stage of a task: where it runs and what
    it takes, its run decided and filled in ``writes`` (the send writes it,
    never before the person has said yes), every refusal before anything is
    sent (`job-system.md` § 6.0, step 1).  Every launch asks it: a stage
    alone, and each member of a group.

    THE RUN'S LAYOUT DECIDES HOW IT IS FILLED, never the kind: one run
    folder (``run-<n>/``, `materialize.prepare_attempt`); a RUN OF POINTS --
    a transport rung swept over frames and voltages -- one run folder with a
    folder per point (`materialize.open_sweep_run`), which the points' walk
    then sends (:func:`_points_walk`); or, on the flat layout, the
    calculation's folder, each run by its number.  Whether it was launched
    is its launch record's answer (`materialize.ever_launched`); its run's
    number is decided here (`runrecord.next_run`), and handed to its run
    script.

    Its FIRST LAUNCH takes what it builds on (:func:`_first_launch`); a
    stage LAUNCHED BEFORE goes again, however it ended (user, 2026-10-07:
    *"run continue warm or cold is user's decision"*), from its newest
    launched run (:func:`_launch_again`).  ``calc`` is the calculation as
    the launch opened it (A16) -- its whole plan, never a set narrowed to
    what one launch sends; ``cold``, ``from_attempt`` and ``gather`` what
    was said at the launch."""
    import functools
    from ..paths import attempts_in
    from .naming import job_dir_names, run_names
    jobset, base, task = calc.jobset, Path(calc.base), calc.task
    sh = calc.shape
    names = run_names(jobset, job, sh)
    member = functools.partial(_Member, base=base, names=names)
    container = base / job_dir_names(jobset, sh)[job.name]
    ns = attempts_in(container)
    flat = not sh.keeps_attempts_as_directories
    if not ns and not flat:
        # A HIERARCHICAL stage with no run open would launch in its own
        # container, write no run.json, and be silently relaunchable --
        # everything § 1.5/1.6 exist to prevent.
        raise SubmitError(
            f"job {job.name!r}: no run is open under {container.name}/ -- a "
            f"hierarchical stage runs in run-<n>, never in its own container "
            f"(project-layout.md § 1.5, 1.6).  To open it, "
            + _redo(calc, job))
    first = not ever_launched(calc, job)
    # WHAT THE LAUNCH WAS TOLD, each flag where it is read: `--gather` takes
    # a transport rung's inputs anew, when it is launched again; `--from`
    # and `--gather` are two answers; the flat layout keeps no run to name.
    if gather and task.calculation != "transport":
        raise SubmitError(
            "--gather takes a transport rung's inputs anew from the "
            "rungs upstream; a stage of this calculation continues from "
            "a run, named with --from (job-system.md § 5.4).")
    if gather:
        from ..transport.stages import inputs_of
        if not inputs_of(task, job.name):
            raise SubmitError(
                f"--gather takes a transport rung's inputs anew from the "
                f"rungs upstream, and {job.name} takes none -- it is "
                f"launched again without it (job-system.md § 5.4, R-L14).")
    if first and gather:
        raise SubmitError(GATHER_AT_FIRST)
    if not first and from_attempt and gather:
        raise SubmitError(
            "--from and --gather are two answers to one question: --from "
            "names the run an independent stage continues from, --gather "
            "takes a transport rung's inputs anew from the rungs upstream "
            "(job-system.md § 5.4).")
    if not first and flat and (from_attempt or gather):
        from .materialize import FLAT_HAS_NO_ATTEMPTS
        raise SubmitError(FLAT_HAS_NO_ATTEMPTS)
    points = _a_sweep(calc, job.name)
    if first:
        return _first_launch(calc, job, member=member, names=names,
                             container=container, points=points,
                             writes=writes, cold=cold,
                             from_attempt=from_attempt, described=described)
    return _launch_again(calc, job, member=member, names=names,
                         container=container, ns=ns, points=points,
                         writes=writes, cold=cold, from_attempt=from_attempt,
                         gather=gather, described=described)


def _newest_launched(container: Path, names: RunNames, ns) -> Path:
    """A stage's newest run that was launched -- what a launch again
    continues: the newest, or the one before it when the newest was opened
    by a send that was refused, which this launch fills anew
    (`materialize.empty_run`)."""
    last = attempt_dir(container, ns[-1])
    return last if _launched(last, names) else attempt_dir(container, ns[-2])


def _first_launch(calc, job, *, member, names: RunNames, container: Path,
                  points: bool, writes: Plan, cold: bool, from_attempt,
                  described=None) -> "_Member":
    """``job``'s stage's FIRST LAUNCH (`job-system.md` § 6.0, step 1): what
    it takes, from the ready door -- the one door per kind, every kind's
    refusal with it (`ready.first_launch_takes`): the stage before it's
    newest run, which must have finished, ``from_attempt`` a run named,
    ``cold`` none; a transport rung's gather, each point its own; a
    force-constant stage's `relax` run its prep took -- refused before
    anything is sent while an upstream has not finished (R-L11); the prepared
    stage checked against the description (:func:`_as_described`); and its
    first run filled: prep's run, emptied whole, prep's own files put back
    as they lie (:func:`_prep_files_back`), and given what it takes now,
    with its record (`project-layout.md` § 1.6.3).  On the flat layout
    nothing is copied: the run's `.continued-from` is written with its
    launch record."""
    from ..runrecord import next_run
    from .ladder import force_constant_stage
    from .gather import gather_into
    from .materialize import open_sweep_run, prepare_attempt
    from .ready import first_launch_takes
    base, task, sh = Path(calc.base), calc.task, calc.shape
    stage = job.name
    cont, got, why = first_launch_takes(calc, stage, job,
                                        from_attempt=from_attempt, cold=cold)
    if why:
        raise SubmitError(why)
    prepared = _as_described(
        calc, job,
        written_at=cont if force_constant_stage(task, stage) else None,
        described=described)
    if not sh.keeps_attempts_as_directories:
        # A FLAT RUN takes what it builds on where its files lie: nothing is
        # copied, and its hand-over's record is written with its launch
        # record (`_record_launch`) -- the flat layout opens no run before
        # its launch.
        run = next_run(container, names)
        return member(job, container, container, False, container, run=run,
                      first=True, cold=cold, continuation=cont,
                      hands=_flat_hand(container, names, run, cont))
    if points:
        # A RUN OF POINTS: prep's run, each point given its own gather.
        sweep = open_sweep_run(calc, stage, plan=writes)
        _prep_files_back(prepared, sweep.dir, writes)
        took, gathered = _gather_sweep(base, sweep.dir, sweep.points, got,
                                       writes, names=names)
        return member(job, container, sweep.dir, True, sweep.dir,
                      run=FIRST_ATTEMPT, label=stage, first=True, cold=cold,
                      points=list(sweep.points), took=took, hands=gathered)
    try:
        opened = prepare_attempt(
            calc, stage,
            continue_from=(cont.source if cont is not None else None),
            cold=cont is None or cont.source is None,
            named=cont is not None and not cont.by_default,
            carry=(list(cont.carries) if cont is not None
                   and cont.carries is not None else None),
            plan=writes)
    except ValueError as exc:
        raise SubmitError(f"{stage}: {exc}") from None
    _prep_files_back(prepared, opened.dir, writes)
    inputs = [(src, fn) for _pt, ins in (got or ()) for src, fn in ins]
    run = next_run(opened.dir, names)
    gathered = gather_into(base, opened.dir, inputs, writes, names=names,
                           run=run)
    _shares(calc, job, container, opened.dir, None, None, writes)
    # READ FROM THE STAGE'S OWN FILES, prep's -- what its run is emptied and
    # filled from at the send -- as a launch again reads them.
    return member(job, container, opened.dir, True, container, run=run,
                  first=True, cold=cold, continuation=cont,
                  hands=_hands(opened.hand, gathered),
                  took=[{"file": fn, "from": src, "frame": None,
                         "volts": None} for src, fn in inputs])


def _launch_again(calc, job, *, member, names: RunNames, container: Path,
                  ns, points: bool, writes: Plan, cold: bool, from_attempt,
                  gather, described=None) -> "_Member":
    """A STAGE LAUNCHED BEFORE, launched again (`job-system.md` § 5.4, *A
    stage launched again*): warm by default -- the next run from its newest
    launched run (:func:`_newest_launched`; the flat layout runs again where
    its files are), a run of points taking over each point done there --
    or ``cold``, taking nothing from a run of its own: the next run opened
    empty but for the stage's files and the inputs its kind gathered, the
    flat run script told ``--cold --force``.  It takes its inputs anew when
    asked -- ``--from`` a run of the stage it builds on, ``--gather`` a
    transport rung's upstream -- checked against the description as a
    first launch is."""
    from ..runrecord import next_run
    from .gather import gather_into
    from .materialize import open_sweep_run, prepare_attempt
    base, task, sh = Path(calc.base), calc.task, calc.shape
    stage = job.name
    if not sh.keeps_attempts_as_directories:
        # A FLAT STAGE LAUNCHED AGAIN runs where its files are: warm, from
        # its own latest run, and its next run's record says so -- or cold,
        # its run script told to sweep them.  Its `.continued-from` is
        # written with its launch record, once the run is sent
        # (`_record_launch`): a refused send leaves no file of a run that
        # never started.
        cont = relaunch(calc, job, cold=cold)
        run = next_run(container, names)
        return member(job, container, container, False, container, run=run,
                      again=True, cold=cold, continuation=cont,
                      hands=_flat_hand(container, names, run, cont))
    ran = _newest_launched(container, names, ns)
    if points:
        # A RUN OF POINTS LAUNCHED AGAIN: its inputs its kind's, never named
        # (`--from` refused); what the newest launched run gathered, or
        # gathered anew; warm, the run continues from it, read at the point
        # (the walk takes each point done there over).
        from .continuation import cannot_be_named
        why = cannot_be_named(base, task, stage, from_attempt, False)
        if why:
            raise SubmitError(why)
        got = _gathered_now(calc, stage) if gather else None
        if gather:
            _as_described(calc, job, described=described)
        sweep = open_sweep_run(calc, stage, plan=writes)
        took: List[dict] = []
        if gather:
            took, hands = _gather_sweep(base, sweep.dir, sweep.points, got,
                                        writes, names=names)
        else:
            hands = _carry_sweep_gather(ran, sweep.dir, sweep.points, writes,
                                        base=base, stage=stage, names=names)
        if not cold:
            # THE RUN CONTINUES FROM THE STAGE'S NEWEST LAUNCHED RUN (rule 3)
            # -- its points taken over by the walk, each with its own record
            # (:func:`_points_walk`); said once at the run too, a hand-over
            # of no file, so its launch record carries it
            # (`project-layout.md` § 1.6.3) like every other run's.
            again = HandOver(sweep.dir, names, FIRST_ATTEMPT, CONTINUED,
                             source=RunRef.at(base, ran))
            hand_over(base, again, writes)
            hands += (again,)
        return member(job, container, sweep.dir, True, ran,
                      run=FIRST_ATTEMPT, label=stage, again=True, cold=cold,
                      points=list(sweep.points), ran=ran, took=took,
                      hands=hands)
    took = []
    if from_attempt:
        # A RUN NAMED, taken anew (`job-system.md` § 5.4): the one door, as a first launch asks
        # it -- what cannot be taken is refused (a transport rung, a run
        # that is not this calculation's); a force-constant stage's run is
        # its prep's.
        from .continuation import continuation_answer
        from .ladder import force_constant_stage
        if force_constant_stage(task, stage):
            raise SubmitError(
                f"--from {from_attempt!r}: `{stage}`'s deck is written at "
                f"the `relax` run its prep took ({job.written_at}) -- another "
                f"is said at its prep, which a launched stage is not given "
                f"again.  To change it, " + _redo(calc, job))
        cont, why = continuation_answer(calc, stage,
                                        from_attempt=from_attempt, cold=cold)
        if why:
            raise SubmitError(why)
        _as_described(calc, job, described=described)
        source, named_by = cont.source, True
    else:
        cont = relaunch(calc, job, cold=cold, ran=ran)
        source, named_by = (cont.source if cont is not None else None), False
    inputs = None
    if gather:
        # THE INPUTS ANEW (`job-system.md` § 5.4, R-L11): the upstreams' newest runs, under the
        # gather's gates, and the stage checked against the description.
        got = _gathered_now(calc, stage)
        _as_described(calc, job, described=described)
        inputs = [(src, fn) for _pt, ins in got for src, fn in ins]
    try:
        # THE ONE OPENER, planned (`project-layout.md` § 1.6.2): the next
        # run, the stage's files and what it carries from ``source`` --
        # nothing, cold.
        opened = prepare_attempt(calc, stage,
                                 continue_from=source, cold=source is None,
                                 named=named_by, plan=writes)
    except ValueError as e:
        if named_by:
            raise SubmitError(f"{stage}: {e}") from e
        # WARM IS IMPOSSIBLE -- its latest run left nothing the stage
        # takes.  Said, with the cold launch that is possible.
        raise SubmitError(
            f"{stage}: launched again warm, it continues from {source} "
            f"-- which is impossible here:\n  {e}\n  Launch it again cold "
            f"instead:\n"
            + block(launch_lines("task", stage, base=base,
                                 flags=("--cold",)))
            ) from e
    # WARM, THE STAGE'S OWN FILES STAND: what it carries from its own run
    # is not taken from the gather too (a device's `.DM`).
    own = set(opened.copied)
    run = next_run(opened.dir, names)
    if inputs is not None:
        inputs = [(src, fn) for src, fn in inputs if fn not in own]
        gathered = gather_into(base, opened.dir, inputs, writes, names=names,
                               run=run)
        took = [{"file": fn, "from": src, "frame": None, "volts": None}
                for src, fn in inputs]
    else:
        gathered = _carry_the_gather(ran, opened.dir, writes, base=base,
                                     stage=stage, names=names, run=run,
                                     keep=own)
    from ..transport.energies import SHARED_DIR
    _shares(calc, job, container, opened.dir, None, ran / SHARED_DIR, writes)
    # READ FROM THE STAGE'S OWN FILES, prep's -- what the new run is filled
    # from (`materialize.prepare_attempt`), so the deck checked, the header
    # shown and the line sent are one.
    return member(job, container, opened.dir, True, container, run=run,
                  again=True, cold=cold, continuation=cont, ran=ran,
                  hands=_hands(opened.hand, gathered), took=took)


def _carry_the_gather(source: Path, target: Path, writes: Plan, *,
                      base: Path, stage: str, names: RunNames, run: int,
                      keep=()) -> Optional[HandOver]:
    """A transport rung launched again takes the inputs its kind gathered
    for ``source`` -- a run, or a point of one -- one hand-over into
    ``target``, run ``run`` of the stage ``names`` names: each file from
    the run its record names, with that record (`.gathered-from`): what a
    stage builds on is decided at its first launch, and taken anew only
    when asked (``--gather``, `job-system.md` § 5.4).  ``keep`` are the
    files ``target`` takes from the stage's own run, warm, which stand:
    warm is its own latest run (a device's `.DM`), and the gather gives
    what it does not carry itself.  Returns the hand-over; ``None`` for a
    run that gathered nothing.

    WHAT A RUN CARRIES is its own record and, for an input it kept from the
    run before it (warm, so its record does not list it), what that run
    carries -- one hop at a time through `.continued-from`: a device
    launched warm, then cold, still takes the seed's density its first
    launch gathered (`job-system.md` § 5.4, *either way a transport rung
    keeps the inputs its first launch gathered*)."""
    from ..runrecord import carried_gather
    took = tuple((src, fn) for src, fn in carried_gather(base, source, names)
                 if fn not in keep)
    for src, fn in took:
        # FROM THE RUN THE RECORD NAMES, never from the source run's own
        # folder: a run is immutable (`project-layout.md` § 1.5), while the
        # engine overwrites a gathered input in the folder it runs in
        # (TranSIESTA writes its own density over the seed's `.DM`) --
        # copied from there, a cold launch would start warm under a record
        # naming the seed.
        if not (src.path(base) / fn).is_file():
            raise SubmitError(
                f"{_rel(source, base)}: {fn}, gathered for it from {src}, is "
                f"not there -- launched again, the stage takes the inputs "
                f"gathered for its run, and they are gone.  Gather them "
                f"anew, from the newest runs upstream:\n"
                + block(launch_lines("task", stage, base=base,
                                     flags=("--gather",))))
    if not took:
        return None
    h = HandOver(target, names, run, GATHERED, files=took)
    hand_over(base, h, writes)
    return h


def _the_queue(jobs) -> Tuple[Optional[str], Optional[str]]:
    """``(queue, its source)`` -- the queue the work's prep recorded
    (:func:`_placed_by_prep`), or ``(None, None)`` where it placed none."""
    placed = _placed_by_prep(jobs)
    return (placed.name, "its prep") if placed else (None, None)


# --------------------------------------------------------------------- #
#  the entry: plan, ask, send (`job-system.md` § 6.0)                   #
# --------------------------------------------------------------------- #

def _record(base: Path, told, decision: str, **facts) -> None:
    """A launch's decision in the calculation's ledger -- every refusal, the
    question and its answer, each submission as it goes -- written by the
    entry, whichever door called (`execution/architecture.md` § 2.1, floor
    7: `prep`'s and `launch`'s entries append their own) -- of the
    calculation the verb opened (`calculation.open`), the one a launch has."""
    from .ledger import record
    record(base, "launch", decision, **{**dict(told or {}), **facts})


def plan_launch(calc, *, mode: str,
                bench: "Optional[Tuple[JobSet, Path]]" = None,
                only: Optional[str] = None,
                side: Optional[str] = None,
                time_s: Optional[int] = None,
                trial_timeout_s: Optional[int] = None,
                told: Dict[str, object],
                record: bool = True,
                cold: bool = False,
                group: Optional[int] = None,
                from_attempt: Optional[str] = None,
                gather: bool = False,
                background: bool = False,
                held: Sequence[dict] = (),
                groups: Sequence[int] = (),
                inside: Optional[Inside] = None) -> LaunchPlan:
    """STEP 1 of `job-system.md` § 6.0 -- the whole launch of ``calc``, the
    calculation as the verb opened it (A16): its plan, or a benchmark's
    (``bench``: its set and the bench container it was read from) -- planned
    with nothing written: the work, its submissions and their members, the
    gates, the queue, the exact lines and every script the send will write.
    A refusal can only come from here, and every one is written down: the
    readers' own (a record that does not read, R-L21; a description, a
    template or a machine's record a step refuses) said in their words.

    ``mode`` is ``"submit"`` (the header prep wrote, sent to the scheduler),
    ``"ask"`` (the same line, asked about -- its kind's ``ask``)
    or ``"direct"`` (``bash`` here, the members in order; ``background``
    leaves it running, the default of the verb's run here, and otherwise the
    send waits for it -- `job-system.md` § 6.0).  ``only``
    names ONE job -- a ladder's stage (the only case: stages do not chain,
    `project-layout.md` § 1.6) or a sweep's trial -- by its name or ``#N``.
    The queue is the one the work's prep recorded; ``time_s`` is a submitted
    job's wall as the launch restates it, the one value a launch may
    (`commands.LAUNCH_FLAGS`); ``side`` (``"cpu"`` / ``"gpu"``) and
    ``trial_timeout_s`` are a grouped bench's.  ``held`` are the questions
    the verb asked before the plan (which group, which stage), each with its
    answer: written before a refusal, and otherwise after the send's save
    (``LaunchPlan.held``, § 6.0, step 4).
    ``told`` is what the verb was told -- the launch's ``kind``, ``stage``
    and ``trial``, and its ``flags`` as typed among it -- written on each of
    the launch's lines in the ledger, and said again by a refusal that
    offers the launch in another mode; required, so a library caller says
    what it launches as the verb does.  ``record`` is false for a dry run,
    which writes nothing (§ 6.0, step 3) -- a refusal is written down
    otherwise.  ``group``: a group of the description, by its number --
    its stages sent as its one job (`engines/stages.md` § 2,
    `project-layout.md` § 1.6.6).  ``from_attempt``, ``cold`` and
    ``gather`` speak for ONE stage (``only``, or a group holding one): at
    its first launch, the run it continues from and ``cold`` none; launched
    again, ``cold`` nothing of its own, ``from_attempt`` a run of the stage
    it builds on and ``gather`` a transport rung's inputs, taken anew
    (`job-system.md` § 5.4).  ``groups``, two or more: ONE job launching
    each in turn (:func:`_plan_sequence`, `job-system.md` § 6.0, *A
    sequence of groups*); ``inside``, a group's launch inside such a job
    (:class:`Inside`).

    THE WORK IS READ OFF WHAT IS LAUNCHED.  A sweep with no trial named,
    sent to (or asked of) a scheduler, goes as ONE job per resource shelf
    (`generator.md` § 4.3a) -- never as one job per trial, which would hand
    the scheduler a sweep at once: *"SLURM should never submit jobs in
    parallel.  Submission is manual and one by one"* (user, 2026-08-10); a
    sweep run here walks its unlaunched trials in one submission, as a
    shelf does on a queue; a transport sweep's per-point rung goes as one
    job walking its points; a stage and a named trial, one submission each.
    """
    base = Path(calc.base).resolve()
    # WHAT THE VERB GAVE, kept as given: the send makes the plan again from
    # the folder as it is with exactly these (§ 6.0, step 4).
    asked = dict(mode=mode, only=only, side=side, time_s=time_s,
                 trial_timeout_s=trial_timeout_s, told=told, cold=cold,
                 group=group, from_attempt=from_attempt, gather=gather,
                 background=background, groups=tuple(groups), inside=inside)
    try:
        # THE CALCULATION'S COPY OF ITS MACHINE'S RECORD READS, FIRST, for
        # every door in every mode (`configuration.md` M-3): a copy that does
        # not read is refused in its own words before any member is planned
        # -- and read ONCE for the whole launch (§ 3.2, *the machine*).
        machine = _machine_of(base)
        if inside is not None:
            inside = dataclasses.replace(inside, kind=machine.kind)
        jobset = calc.jobset if bench is None else bench[0]
        # A GROUP'S LAUNCH INSIDE A SEQUENCE'S JOB plans as a reader inside
        # that job (`joblist.inside`): the runs its job holds read by their
        # own files -- the gather's gates among them.
        from ..joblist import inside as _inside_job
        with _inside_job(inside.job if inside is not None else None,
                         inside.name if inside is not None else None):
            plan = _planned(calc, jobset, machine=machine, mode=mode,
                            only=only, side=side, time_s=time_s,
                            trial_timeout_s=trial_timeout_s, told=told,
                            cold=cold, group=group,
                            from_attempt=from_attempt, gather=gather,
                            groups=groups, inside=inside)
    except SubmitError as exc:
        if record:
            for facts in held:
                _record(base, told, "question", **facts)
            _record(base, told, "refused", reason=str(exc))
        raise
    except _refusal_kinds() as exc:
        # A READER'S REFUSAL, wherever the plan met it -- a record that does
        # not read (R-L21), a description or plan written before a rule, a
        # step's own -- is the launch's, in its reader's words, written down
        # like every refusal: one catch, not one per reader (A16: a
        # boundary translates the kind into its own).
        if record:
            for facts in held:
                _record(base, told, "question", **facts)
            _record(base, told, "refused", reason=str(exc))
        raise SubmitError(str(exc)) from None

    def again() -> LaunchPlan:
        # THE FOLDER AS IT IS NOW (`job-system.md` § 6.0, step 4): the
        # calculation opened again -- its description, template and plan --
        # and a benchmark's set read again from its container, so anything
        # changed since the plan is seen; planned with what the verb gave.
        from ..calculation import NotACalculation
        from ..calculation import open as open_calculation
        from .model import JobSet
        try:
            # THE DOOR'S OWN REFUSALS: not a calculation, or a record that
            # does not read (`calculation.open`, its readers' ValueErrors).
            fresh = open_calculation(base)
            again_bench = (None if bench is None
                           else (JobSet.of(bench[1]), bench[1]))
        except (NotACalculation, OSError, ValueError) as exc:
            raise SubmitError(str(exc)) from None
        gone = (bench[1] if again_bench is not None and again_bench[0] is None
                else base if bench is None and fresh.jobset is None else None)
        if gone is not None:
            raise SubmitError(f"the plan in {gone} is gone")
        return plan_launch(fresh, bench=again_bench, record=False, **asked)

    plan.remake = again
    plan.inside = inside
    plan.held = list(held)
    plan.told, plan.records = dict(told), record
    # the queue's source is the entry's decision, on every line
    plan.told.update(domain=plan.queue[0], domain_source=plan.queue[1])
    plan.background = bool(background) and mode == "direct"
    return plan


def _refusal_kinds() -> tuple:
    """What the readers a plan asks raise for the person to fix, each its
    own kind, said as the launch's refusal (:func:`plan_launch`): a record
    of a run that does not read (R-L21), a plan's job no file can be named
    for (`naming.NamingError`), and a calculation's own
    `warm-files.toml` that does not read (`warmfiles.WarmFilesError`) --
    never a plain error of code, which stays one."""
    from ..warmfiles import WarmFilesError
    from .naming import NamingError
    return (LaunchRecordError, NamingError, WarmFilesError)


def _planned(calc, jobset: JobSet, *, machine, mode, only, side, time_s,
             trial_timeout_s, told,
             cold: bool = False, group=None, from_attempt=None,
             gather: bool = False, groups: Sequence[int] = (),
             inside: Optional[Inside] = None) -> LaunchPlan:
    """:func:`plan_launch`'s body -- the plan, or the refusal it records;
    ``calc`` the calculation as the verb opened it (A16), ``jobset`` the
    set launched -- its ladder, or a sweep's from its bench container."""
    base, shape = Path(calc.base), calc.shape
    if only is not None:
        # Through the ONE resolver, so a name and a #N number reach the same
        # job here as at every other surface, and the refusal carries the
        # typeable spellings.
        from ..identity import resolve_stage_ref
        from .naming import stage_refs
        refs = stage_refs(jobset)
        try:
            only = resolve_stage_ref([refs[j.name] for j in jobset.jobs],
                                     only).name
        except ValueError as e:
            raise SubmitError(str(e))

    # A LADDER'S DESCRIPTION AND TEMPLATE are the calculation's, read once:
    # what each member is checked against and what its first launch takes.
    described = None
    if jobset.kind == KIND_LADDER:
        # AS PREP CHECKS IT, ONCE for the whole launch -- the first member
        # whose files are compared with prep's plan asks -- on the machine's
        # record the launch read.
        import functools
        described = functools.lru_cache(maxsize=None)(
            lambda: _checked(calc, machine))
    if len(groups) > 1:
        return _plan_sequence(calc, jobset, groups, machine=machine,
                              mode=mode, time_s=time_s, told=told,
                              one=bool(only or from_attempt or cold
                                       or gather),
                              described=described)
    members, number = ((_the_group(calc, group=group, only=only,
                                   one=bool(from_attempt or cold or gather)))
                       if jobset.kind == KIND_LADDER else (None, None))
    if inside is not None and number is not None:
        # THE GROUP THE SEQUENCE WAS SENT WITH, by its members: groups are
        # matched by who they are, never by their number (`job-system.md`,
        # *The task*), so a description renumbered since the job went is
        # refused here, not launched as another group (R-L46).
        from .group import same_members
        sent = next((g["stages"] for g in inside.record.groups
                     if int(g["number"]) == int(number)), None)
        now = calc.request.members(number)
        if sent is None or not same_members(sent, now):
            raise SubmitError(
                f"group {number} of the job request is "
                f"{', '.join(now) or 'none'}, "
                f"and the sequence {inside.name} was sent with "
                f"{', '.join(sent) if sent else 'no such group'} -- the "
                f"job request changed since the job went, so this group is "
                f"not launched in it (job-system.md § 6.0; R-L46).")
    if number is not None:
        # A GROUP A SEQUENCE STILL GOING HOLDS is that job's to launch
        # (`job-system.md` § 6.0, R-L45) -- but by its own walk.
        from .ready import held_by_sequence
        held = held_by_sequence(calc, number)
        if held is not None and (inside is None or inside.job != held[0].id):
            raise SubmitError(_held_by(held, number, base))
    if members is not None and len(members) == 1:
        only, members = members[0], None

    # THE QUEUE, the one the work's prep recorded (`job-system.md` § 6.0,
    # the placement) -- for the plan's lines and the ledger; each door
    # sends to it.
    domain, source = (_the_queue(
        [j for j in jobset.jobs
         if ((j.name in members) if members
             else (only is None or j.name == only))])
        if mode in ("submit", "ask") else (None, None))
    if members:
        plan = _plan_group(calc, members, mode=mode,
                           time_s=time_s, told=told, number=number,
                           described=described, machine=machine,
                           inside=inside)
    elif jobset.kind == KIND_SWEEP and only is None and mode in ("submit", "ask"):
        plan = _plan_shelves(jobset, base, shape=shape, mode=mode, side=side,
                             time_s=time_s, trial_timeout_s=trial_timeout_s,
                             told=told, machine=machine)
    elif jobset.kind == KIND_SWEEP and only is None:
        plan = _plan_bench_here(jobset, base, shape=shape,
                                trial_timeout_s=trial_timeout_s)
    else:
        plan = _plan_stage(calc, jobset, mode=mode, only=only,
                           time_s=time_s, told=told, cold=cold,
                           from_attempt=from_attempt, gather=gather,
                           described=described, machine=machine,
                           inside=inside)
    plan.queue = (domain, source)
    plan.engine = jobset.engine
    if number is not None and any(m.first for sub in plan.submissions
                                  for m in sub.members):
        # A GROUP'S FIRST LAUNCH: the save its send makes first (R-L24) --
        # the group named by its members, as every way back names it.
        from .group import named
        plan.save_before = f"launch task {named(calc.request, number)}"
    return plan


def _checked(calc, machine):
    """The calculation AS PREP CHECKS IT (`prep.Described`, `prep.checked`),
    for the member whose files are compared with prep's plan -- ``calc``
    as the launch opened it (A16), on ``machine``, the record the launch
    read (:func:`_machine_of`), never read again."""
    from .prep import Described, checked
    return checked(calc.base, Described(calc), target=None,
                   environment=machine)


def _the_group(calc, *, group, only,
               one: bool) -> Tuple[Optional[List[str]], Optional[int]]:
    """``(members, number)`` of what a task's launch sends, by the
    job request's groups (`execution/job-request.md` § 3.3) -- ``members`` the
    group's stages in ladder order, as their jobs name them, when ``group``
    names one, else ``None``; ``number`` the group launched, or the group
    ``only`` is in.  A name in any case is one name.  Refused: a group the
    job request does not hold, or not prepared; a launched group whose
    members the job request changed -- they are fixed; a group whose
    prepared jobs hold other members than it does now -- its prep forms it
    again (R-L8); ``one`` (``--from`` / ``--cold`` / ``--gather``) for a
    group of several (R-L9); a member's first launch alone while its group
    holds others -- a group goes whole the first time (R-L6)."""
    from ..identity import stage_key
    from .commands import launch_lines, prep_group_lines
    from .ready import regrouped
    jobset, base, request = calc.jobset, Path(calc.base), calc.request
    if group is None:
        if only is None:
            return None, None
        number = request.group_of(only)
        described = request.members(number)
        # ``only`` is a job of the plan: resolved against it (`_planned`).
        if ever_launched(calc, jobset.job(only)):
            return None, number
        if len(described) > 1:
            raise SubmitError(
                f"`{only}` was never launched, and its group {number} holds "
                f"{', '.join(f'`{n}`' for n in described if stage_key(n) != stage_key(only))}"
                f" too -- a group goes whole the first time; a member is "
                f"launched alone only again (job-system.md, *The task*):\n"
                + block(launch_lines("task", base=base, groups=[number])))
        why = regrouped(calc, only)
        if why:
            raise SubmitError(why)
        return None, number
    members = request.members(group)
    if not members:
        raise SubmitError(
            f"group {group}: the job request holds groups "
            f"{', '.join(map(str, request.numbers))}"
            f" (execution/job-request.md § 3.3).")
    # A LAUNCHED GROUP'S MEMBERS ARE FIXED (`job-system.md`, *The task*),
    # asked of the one rule and said in its words.
    from ..runrecord import LaunchRecordError
    from .ready import joined_launched, launched_without
    try:
        hit = launched_without(calc, group)
    except LaunchRecordError as exc:
        raise SubmitError(str(exc)) from None
    if hit:
        raise SubmitError(joined_launched(calc, group, *hit))
    ran = any(j is not None and ever_launched(calc, j)
              for j in map(jobset.job, members))
    missing = [m for m in members if jobset.job(m) is None]
    if missing:
        raise SubmitError(
            f"group {group}: {', '.join(f'`{m}`' for m in missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not prepared -- "
            f"prepare the group first:\n"
            + block(prep_group_lines([group], base=base)))
    if one and len(members) > 1:
        raise SubmitError(
            f"--from / --cold / --gather speak for one stage, and group "
            f"{group} holds {', '.join(f'`{m}`' for m in members)} -- its "
            f"members each take their default; a member launched again "
            f"alone takes them with --stage (job-system.md § 5.3).")
    if not ran:
        why = next((w for w in (regrouped(calc, m) for m in members)
                    if w), None)
        if why:
            raise SubmitError(why)
    return [jobset.job(m).name for m in members], group


def _a_sweep(calc, stage: str) -> bool:
    """Whether ``stage`` of ``calc`` is a transport rung that runs per point
    (`transport.stages.rung_points`: a frame and a voltage each) --
    launched as ONE job walking its run's points."""
    from ..transport.stages import frames_of, rung_points
    return bool(calc.task.calculation == "transport"
                and rung_points(calc.task, stage,
                                frames=frames_of(calc.base)))


def ask_launch(plan: LaunchPlan) -> List[JobResult]:
    """``--mode ask`` (`submission.md` S4): the scheduler asked about each
    submission -- its kind's ask on the very line the send would hand it --
    in place of the question; nothing is written or sent.  Past
    :data:`ASK_MAX_QUERIES` the rest are named, never dropped."""
    out: List[JobResult] = list(plan.skipped)
    for n, s in enumerate(plan.submissions):
        out += [JobResult(m.name, [], "takes", said=m.would())
                for m in s.members if m.follows]
        if n >= ASK_MAX_QUERIES:
            out.append(JobResult(s.name, [], "not asked"))
            continue
        out.append(dataclasses.replace(
            _ask(s.name, s.command, s.ask_in or s.cwd, kind=s.machine.kind,
                 domain=s.domain),
            request=s.request()))
    plan.record("asked", jobs=[{"job": r.name, "status": r.status,
                                **({"said": r.said} if r.said else {})}
                               for r in out])
    return out


def send_launch(plan: LaunchPlan, *, said) -> List[JobResult]:
    """STEPS 4 and 5 of `job-system.md` § 6.0 -- the plan, sent as it was
    shown, and written down; nothing is decided here.

    ``said`` is the person's answer to the one question the verb put --
    every launch is asked, a run here as a submission is (`ask.Said`;
    ``--yes`` the answer given in advance).  The question is written down
    as what was asked, with its answer, a *no* too, and a *no* sends
    nothing; with nobody to ask (``said.asked`` false) nothing goes and the
    launch is refused, written down (`submission.md` S4).  Then
    the folder is checked against the one the plan was made from
    (:func:`_same_folder`); the plan's writes are carried out -- every
    attempt opened and every script on disk before anything goes, so a
    scheduler's refusal costs one submission, never the scripts of those
    behind it -- and each submission goes out through ONE function
    (:func:`_go`), written down the moment it goes.  A shelf the scheduler
    refuses leaves the rest to go (``plan.tolerant``): its trials keep no
    launch record, so the next launch picks up exactly them.  Every refusal
    is written down."""
    # WHAT WAS ASKED, written as asked -- after the save when the launch
    # goes, so the state saved holds no line of it; with nothing saved, at
    # once.
    if not said.asked:
        why = ("not a terminal, so there is nobody to ask -- nothing was "
               "sent.  Pass --yes to go ahead with what is printed above "
               "without being asked.")
        plan.asked(said)
        plan.record("refused", reason=why)
        raise SubmitError(why)
    if not said:
        plan.asked(said)
        return []
    try:
        _same_folder(plan)
    except SubmitError as exc:
        plan.asked(said)
        plan.record("refused", reason=str(exc))
        raise
    if plan.save_before is not None:
        # A GROUP'S FIRST LAUNCH SAVES THE FOLDER'S STATE first -- the state
        # a redo of the group restores, every run it builds on in it
        # (`job-system.md` § 6.0, step 4; `checkpointing.md` § 9).  A state
        # that cannot be saved sends nothing.
        from ..checkpoint import CheckpointError, save_before
        from ..runtime_config import RuntimeConfigError
        try:
            kept = save_before(plan.base, plan.save_before,
                               engine=plan.engine)
        except (CheckpointError, RuntimeConfigError) as exc:
            why = (f"the folder's state could not be saved, so nothing was "
                   f"sent: {exc}")
            plan.asked(said)
            plan.record("refused", reason=why)
            raise SubmitError(why) from None
        plan.saved = kept.said()
        plan.asked(said)
        plan.record("saved", state=kept.state.short, note=kept.state.note,
                    new=kept.new)
    else:
        plan.asked(said)
    # THE OTHER WAY, ANSWERED: a run here goes as the person said --
    # waited on, or left in the background (`job-system.md` § 6.0).
    if said.other:
        plan.background = not plan.background
    plan.writes.carry_out()
    # WHAT EACH STAGE TAKES, recorded where it lands (`job-system.md`
    # § 5.4): the run it continues from and what came across (`continues`),
    # a transport rung's inputs (`gathers`), or nothing at a first launch
    # (`starts-cold`).
    for s in plan.submissions:
        for m in s.members:
            if m.continuation is not None:
                plan.record("continues", stage=m.name,
                            **m.continuation.ledger_facts(),
                            copied=list(m.carries),
                            written_as_ended=dict(m.warned),
                            **({"gathered": list(m.gathered)}
                               if m.gathered else {}))
            if m.took:
                plan.record("gathers", stage=m.name,
                            inputs=[{**t, "from": str(t["from"])}
                                    for t in m.took])
            elif m.gathered and m.continuation is None:
                # LAUNCHED AGAIN COLD, a transport rung still carries the
                # inputs its run was given, each from its run.
                plan.record("gathers", stage=m.name, carried=[
                    {"file": n, "from": str(src)} for h in m.hands
                    if h.record == GATHERED for src, n in h.files])
            elif m.first and m.continuation is None:
                plan.record("starts-cold", stage=m.name, asked=m.cold)
            for take in m.copied:
                # A POINT TAKEN OVER from the finished calculation this one
                # reuses (`engines/transport.md` § 2a.16).
                plan.record("taken over", stage=m.name,
                            point=take.point.words(), source=take.shown,
                            copied_from=take.copied_from)

    results: List[JobResult] = list(plan.skipped)
    refused: List[str] = []
    for s in plan.submissions:
        results += [JobResult(m.name, [], "takes", said=m.note())
                    for m in s.members if m.follows]
        try:
            results += _go(s, plan.record, background=plan.background,
                           inside=plan.inside)
        except SubmitError as exc:
            plan.record("refused", submission=s.name, reason=str(exc))
            if not plan.tolerant:
                raise
            refused.append(str(exc))
            results.append(JobResult(s.name, s.command, "scheduler refused",
                                     domain=s.domain, detail=str(exc)))
            results += [JobResult(m.name, [], "stays pending")
                        for m in s.members]
    if refused and len(refused) == len(plan.submissions):
        raise SubmitError("no shelf was accepted by the scheduler:\n"
                          + "\n".join(f"  {r}" for r in refused))
    return results


def _same_folder(plan: LaunchPlan) -> None:
    """The folder is the one the plan was made from (`job-system.md` § 6.0,
    step 4): the plan made again from it now, and compared with the one
    shown, line by line (:meth:`LaunchPlan.made_from`) -- a member launched,
    a run ended or a file changed since is refused, saying to launch again
    to see the new plan.  Nothing is written before this passes."""
    if plan.remake is None:
        return
    try:
        now = plan.remake()
    except SubmitError as exc:
        raise SubmitError(
            f"the folder changed since this launch was planned, and launched "
            f"now it is refused:\n  {exc}\n  Nothing was sent.  Launch again "
            f"to see the new plan.") from None
    was, got = plan.made_from(), now.made_from()
    if was == got:
        return
    from itertools import zip_longest
    a, b = next((x, y) for x, y in zip_longest(was, got) if x != y)
    raise SubmitError(
        f"the folder changed since this launch was planned:\n"
        f"  planned: {a or '(nothing)'}\n"
        f"  now:     {b or '(nothing)'}\n"
        f"  Nothing was sent.  Launch again to see the new plan.")


def _go(s: Submission, record, *, background: bool = False,
        inside: Optional[Inside] = None) -> List[JobResult]:
    """THE ONE SENDER (`job-system.md` § 6.0, step 4) -- a stage, a shelf,
    a chain, here or to the scheduler: the line run where the plan said,
    each member's launch recorded by the one writer (:func:`_record_launch`),
    and the ledger told the moment it goes -- a run here when it STARTS, a
    scheduler job when it is given its id.  ``background``: a run here is
    started and left -- its own session, so closing the terminal does not
    end it, and nothing attached to it, because everything its scripts print
    already goes to their own files (the run script's runwrap log, a walk's
    log) -- and the send does not wait.  A SEQUENCE OF GROUPS (``s.sequence``,
    `job-system.md` § 6.0) has no member of its own: its address names its
    groups, and its launch record gets the job's id once the scheduler
    gives it.  A group's launch ``inside`` a sequence's job records each run
    as that job's and adds its runs to that job's address, writing none of
    its own."""
    env = {**os.environ, LAUNCH_CLAIM_VAR: LAUNCH_CLAIM}
    # A RUN HERE WITH A WALL ON ITS CARD is told it the same way a sent job
    # is (`running-a-job.md` § 3.6); one stating none runs until done.
    if (s.direct and s.sent is not None and s.sent.time
            and _own_walls("direct", inside)):
        from ..scheduler.kinds import WALL_VAR
        from ..scheduler.quantities import parse_walltime
        env[WALL_VAR] = str(parse_walltime(s.sent.time))
    names = [m.name for m in s.members]
    if s.direct:
        # The launch-door claim rides the child ENV here: inheritance
        # survives forks and backgrounding, so a detached local run
        # launched through this verb never meets the gate's prompt.
        detached = (dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
                    if background else {})
        try:
            proc = subprocess.Popen(s.command, cwd=str(s.cwd), env=env,
                                    **detached)
        except OSError as exc:
            raise SubmitError(
                f"{s.name}: could not run {s.command[0]!r} ({exc})") from None
        # AT START, after the process exists: run.json answers "was this
        # launched?", and a record written on completion would leave a running
        # attempt reading as never launched for its whole runtime.  A failed
        # START records nothing -- Popen raising means no process exists.
        for m in s.members:
            _record_launch(m, mode="direct", command=s.command,
                           inside=inside)
        unsent = _stamped(s, None)
        from .. import joblist
        if inside is not None:
            # RIDES A SEQUENCE'S JOB: its runs added to that job's address
            # -- the job's to keep and to forget, never this launch's.
            job, unkept = None, _ride(inside, s)
        else:
            job, unkept = _keep(joblist.run_here(
                pid=proc.pid, background=background,
                calculation=_calculation_of(s), members=_job_members(s),
                sequence=_sequence_address(s)))
        record("launched", submission=s.name, command=s.command,
               members=names, **_member_facts(s),
               **({"job": job.id} if job is not None and unkept is None
                  else {}),
               **({"not_on_address": unkept}
                  if inside is not None and unkept else {}),
               **({"record_unwritten": unsent} if unsent else {}),
               **({"sequence": [g["number"] for g in s.sequence["groups"]]}
                  if s.sequence else {}),
               **({"background": True, "pid": proc.pid} if background
                  else {}))
        if background:
            # HOW TO STOP IT, said with it: `molbuilder jobs stop`, which
            # signals the run's process group -- a signal to the run script
            # alone would wait for the engine to end, bash running a trap
            # only after the command it waits on (`job-system.md` § 6.0,
            # § 6.0a).
            return ([JobResult(s.name, s.command, "started",
                               said="started in the background",
                               detail=f"pid {proc.pid}; to stop it: "
                                      + (f"molbuilder jobs stop {job.id}"
                                         if unkept is None else
                                         f"kill -- -{proc.pid} ({unkept})"))]
                    + ([JobResult(m.name, [], "rides", said=s.rides)
                        for m in s.members] if s.rides else []))
        try:
            rc = proc.wait()
        finally:
            # A RUN WAITED ON is shown by its own launch, which says how it
            # ended -- or, stopped at its terminal, its stop: its address is
            # done with (`job-system.md` § 6.0a).  A launch inside a
            # sequence's job wrote none.
            if job is not None:
                joblist.forget(job)
        # IT RAN HERE AND ENDED -- how it ended is the status door's, read
        # by the launch verb once every submission has gone (`_cli`,
        # `job-system.md`, *A run here, waited on*).
        return ([JobResult(s.name, s.command, "ran", returncode=rc,
                           detail="; ".join(w for w in (
                               unkept if inside is not None else None,
                               unsent) if w) or None)]
                + ([JobResult(m.name, [], "rides", said=s.rides)
                    for m in s.members] if s.rides else []))
    try:
        cp = subprocess.run(s.command, cwd=str(s.cwd), capture_output=True,
                            text=True, env=env)
    except OSError as exc:
        raise SubmitError(
            f"{s.name}: could not run {s.command[0]!r} ({exc})") from None
    kind = s.machine.kind
    if cp.returncode != 0:
        raise SubmitError(
            f"{kind.what} refused {s.name} (rc={cp.returncode}):\n"
            f"{cp.stderr.strip()}" + s.refusal_hint)
    # THE SCHEDULER TOOK IT (rc 0): the job is queued whatever its words
    # say, so the launch is recorded -- an id it did not print is recorded
    # unknown, and said, so the next launch never empties a run a queued
    # job is in.
    try:
        jid, unread = _job_id_of(kind, cp.stdout), None
    except SubmitError as exc:
        jid, unread = None, (f"{exc} -- the job is queued; the scheduler's "
                             f"own list of your jobs names it")
    for m in s.members:
        # WHERE IT RUNS, not the container: `_launched` reads the attempt,
        # so a record in the container would leave every grouped trial
        # reading *never launched*.
        _record_launch(m, mode="submit", command=s.command, job_id=jid,
                       placement=s.placement, sent=s.sent,
                       scheduler=s.scheduler_said(jid, m.run_dir))
    unsent = _stamped(s, jid)
    kept = {}
    unkept = None
    if jid is not None:
        from .. import joblist
        from ..scheduler.queue import cluster
        job, unkept = _keep(joblist.queue_job(
            job_id=jid, cluster=cluster(s.machine.scheduler),
            calculation=_calculation_of(s),
            members=_job_members(s), sequence=_sequence_address(s)))
        kept = {"job": job.id} if unkept is None else {}
    record("launched", submission=s.name, command=s.command, job_id=jid,
           domain=s.domain, members=names, **_member_facts(s), **kept,
           **({"sequence": [g["number"] for g in s.sequence["groups"]]}
              if s.sequence else {}),
           **({"record_unwritten": unsent} if unsent else {}))
    return ([JobResult(s.name, s.command, "submitted", job_id=jid,
                       domain=s.domain,
                       detail="; ".join(w for w in (unread, unkept, unsent)
                                        if w) or None)]
            + ([JobResult(m.name, [], "rides", said=s.rides, job_id=jid)
                for m in s.members] if s.rides else []))


def _stamped(s: "Submission", job_id: Optional[str]) -> Optional[str]:
    """A sequence's launch record stamped as its job went
    (`sequence.sent`) -- ``None``, or why it could not be, said on the
    launch's line and in the ledger: the job went all the same, and its
    runs read the record's facts, written before it went."""
    if not s.sequence:
        return None
    from .sequence import SequenceError, sent
    try:
        sent(s.sequence["record"], job_id=job_id)
    except (SequenceError, OSError) as exc:
        return ("its launch record was not stamped with when it went"
                + (f" and job {job_id}" if job_id else "") + f" ({exc})")
    return None


def _calculation_of(s: "Submission") -> Path:
    """The calculation a submission's job belongs to -- its members', or,
    for a sequence of groups, which has none, the folder its walk runs
    from: the calculation's own."""
    return s.members[0].base if s.members else Path(s.cwd)


def _sequence_address(s: "Submission") -> Optional[dict]:
    """What a sequence's job address says of it (`joblist.Job.sequence`):
    its name and groups; ``None`` for every other submission."""
    if not s.sequence:
        return None
    return {"name": s.sequence["name"], "groups": [
        {"number": g["number"], "stages": list(g["stages"])}
        for g in s.sequence["groups"]]}


def _ride(inside: Inside, s: "Submission") -> Optional[str]:
    """A group's launch inside a sequence's job: its runs added to that
    job's address (`joblist.add_members`) -- ``None``, or why they could
    not be, said on the launch's line: the job's runs record it all the
    same, and are read by their own records."""
    from .. import joblist
    try:
        joblist.add_members(inside.job, _job_members(s))
    except (joblist.JobFileError, OSError) as exc:
        return f"its runs are not on job {inside.job}'s address ({exc})"
    return None


def _job_members(s: "Submission") -> Dict[str, Dict[str, str]]:
    """Each member's run as a job's address names it (`joblist.Job`): as a
    record names a run (`runrecord.RunRef`) -- its folder, or where its
    stage's runs share one, its own name -- and its launch record's path
    from the calculation's folder, which its state is asked through
    (`job-system.md` § 6.0a)."""
    from ..runrecord import RunRef, launch_record_path
    return {m.name: {
        "run": str(RunRef.named(m.names.run_name(m.run)) if m.names.shared
                   else RunRef.at(m.base, m.run_dir)),
        "launch_record": _rel(launch_record_path(m.run_dir, m.names, m.run),
                              m.base)} for m in s.members}


def _keep(job):
    """``(job, None)`` -- the job's address written (`joblist.write`,
    `job-system.md` § 6.0a) -- or ``(job, why)`` when it could not be: the
    job has gone out already, so the send goes on, and its line says
    why."""
    from .. import joblist
    try:
        joblist.write(job)
    except OSError as exc:
        return job, (f"its address could not be kept, so `molbuilder jobs` "
                     f"will not list it: {exc}")
    return job, None


def _plan_shelves(jobset: JobSet, base: Path, *, machine, shape, mode: str,
                  side: Optional[str], time_s: Optional[int],
                  trial_timeout_s: Optional[int], told) -> LaunchPlan:
    """ONE scheduler job per RESOURCE SHELF of the sweep
    (`generator.md` § 4.3a).

    The standing rule -- *a scheduler is handed few, deliberate jobs* --
    is kept by construction: each shelf IS one job, and the value axes
    keep the shelf count small.  What the grouping replaces is one
    job PER TRIAL, which made an N-point sweep cost N queue waits; on an HPC
    a submission is expensive and unpredictable, and a benchmark's output is
    timing data, not the structure, so the trials ride one allocation in
    sequence.

    **The shelves are prep's** (`shelves.shelves_of`, the one partition):
    by each trial's GPU request, then by its exact ask, each named in full,
    ``bench-group-<side>-<cell>`` -- so the CPU shelf asks no GPU and
    devices are never held while CPU trials run -- each side's widest first,
    and each with the one header prep wrote and placed on the benchmark's
    one queue.  ``side`` (``"cpu"``/``"gpu"``) sends one side, the other
    staying pending for a later `launch bench`.

    The per-shelf pieces (:func:`_plan_shelf`): the walk of its trials still
    to run, planned here and written by the send, and every included trial's
    launch record, stamped with the ONE job id, so `status` and a later
    single-trial re-run see the truth.  ``ask`` asks the scheduler about
    each shelf -- the jobs the send would hand it.
    """
    from .shelves import ShelfError, shelves_of, sides_of
    try:
        sides = sides_of(jobset.jobs)
        shelves = shelves_of(jobset.jobs)
    except ShelfError as exc:
        # NOT SHELVED AS ITS PREP SHELVED THEM (R-L43): its way back.
        raise SubmitError(f"{exc}  "
                          + rollback("the benchmark's prep", base=base)
                          ) from None
    if side and not sides[side]:
        raise SubmitError(f"this sweep has no {side} trials to submit")
    plan = LaunchPlan(base, mode, [], tolerant=True)
    # THE TRIALS STILL TO RUN, the one answer the walk here reads too
    # (:func:`_bench_trials`) -- of the sides this launch sends, in the
    # sweep's order.
    sent = {j.name for this in ("cpu", "gpu")
            if not (side and this != side) for j in sides[this]}
    trials = {m.job.name: m for m in _bench_trials(
        jobset, base, plan, shape=shape, mode=mode,
        jobs=[j for j in jobset.jobs if j.name in sent])}
    for this, names, members in shelves:
        if side and this != side:
            continue
        pending = [trials[j.name] for j in members if j.name in trials]
        if not pending:
            continue            # this shelf already rode a group
        plan.submissions.append(_plan_shelf(
            jobset, base, pending, names, members, plan,
            trial_timeout_s=trial_timeout_s, told=told, time_s=time_s, machine=machine))

    if not plan.submissions:
        raise _all_launched(jobset, base, shape=shape, side=side,
                            count=len(sides[side]) if side
                            else len(jobset.jobs))
    return plan


def _one_container(pending: List["_Member"], base: Path) -> Path:
    """The one folder every trial of a benchmark's walk lies in -- its bench
    container, the trial's parent and not its attempt's (`project-layout.md`
    § 1.5a) -- from which a walk runs, here or on a queue (R-L43)."""
    containers = {m.container.parent for m in pending}
    if len(containers) != 1:
        raise SubmitError(
            "the sweep's trials do not share one container -- a walk of "
            f"them needs the one parent that sees them all; found "
            f"{sorted(str(c) for c in containers)}.  "
            + rollback("the benchmark's prep", base=base))
    return next(iter(containers))


def _all_launched(jobset: JobSet, base: Path, *, shape, count: int,
                  side: Optional[str] = None) -> SubmitError:
    """The refusal of a benchmark's walk with nothing left to run, and the
    next step: reading what its trials measured."""
    from .materialize import bench_stage_of
    from .naming import job_dir_names
    dirs = job_dir_names(jobset, shape)
    stage = bench_stage_of(base / dirs[jobset.jobs[0].name])
    return SubmitError(
        f"all {count} {side + ' ' if side else ''}trials are launched.  "
        f"next:\n    "
        + (_cmd("summarize", "bench", stage, base=base) if stage else
           "`summarize bench` on the sweep's stage"))


def _empty_scratch_fn(when: str) -> List[str]:
    """THE RUN'S SCRATCH FOLDER, EMPTIED (`engines/transport.md` § 2a.11,
    *What a run's points share*): the walk's ``empty_scratch`` -- the files
    the one answer names an engine's scratch (`warmfiles.scratch_families`,
    each family a glob of the shell's) and nothing else, and what went,
    with its size in du's words, to the log."""
    import shlex as _sh
    from ..warmfiles import scratch_families
    globs = " ".join("*" + _sh.quote(f[1:]) for f in scratch_families())
    return [
        "empty_scratch() {",
        '    _d="$1"',
        '    if [ ! -d "${_d}" ]; then',
        f'        echo "[group] {when} $(basename "${{_d}}")/ not there -- '
        f'nothing emptied" >> "$LOG"; return 0',
        "    fi",
        f'    _gone=$(cd "${{_d}}" && ls -d -- {globs} 2>/dev/null '
        f'| tr "\\n" " ")',
        '    if [ -n "${_gone}" ]; then',
        f'        _size=$(cd "${{_d}}" && du -ch -- {globs} 2>/dev/null '
        f'| tail -1 | cut -f1)',
        f'        (cd "${{_d}}" && rm -f -- {globs})',
        f'        echo "[group] {when} $(basename "${{_d}}")/ emptied -- '
        f'${{_gone}}(${{_size}})" >> "$LOG"',
        "    else",
        f'        echo "[group] {when} $(basename "${{_d}}")/ empty" >> "$LOG"',
        "    fi",
        "}",
    ]


class WalkCommand(str):
    """A walk step's command as shell words, run as they stand -- a group's
    launch in a sequence of groups (`job-system.md` § 6.0); every other
    step names a run script, which the walk runs with ``bash``."""


def _walk_script(name: str, steps, *, where: str, log: str,
                 member_word: str = "trial",
                 bound_s: Optional[int] = None,
                 stop_on_failure: bool = False,
                 chains: Optional[List[Any]] = None,
                 empties: Optional[Dict[int, str]] = None,
                 empty_at_end: Optional[str] = None,
                 walls: Optional[Sequence[Optional[int]]] = None) -> str:
    """THE ONE WALK SCRIPT -- what runs several members one after another in
    one submission: a benchmark's trials (a shelf sent to a queue, or its
    unlaunched trials run here, `generator.md` § 4.3a), a group's stages
    (`project-layout.md` § 1.6.6), a swept rung's points
    (`engines/transport.md` § 2a.11), a sequence's groups (`job-system.md`
    § 6.0).  ``steps`` are ``(name, folder, run script, its arguments,
    before)``: the folder from where the walk runs -- the run script run
    with ``bash``, or a :class:`WalkCommand`, shell words run as they stand
    (a group's launch) -- and ``before`` the files copied into it just
    before it starts -- ``(source, folder)`` pairs, a sweep's start (its
    `along` row); ``()`` for every other walk.  ``bound_s`` is a per-member bound, a member past
    it killed and read incomplete (a benchmark's).  A member that fails
    leaves the rest to run, and the walk exits nonzero when any failed --
    unless ``stop_on_failure`` (a device's points: the ones after a failed
    one would start from it), when the members after it IN ITS CHAIN are not
    run -- ``chains`` names each step's chain (a device point's frame,
    `engines/transport.md` § 2a.9), one chain when none is named -- and the
    walk goes on with the other chains.  Stopped by the person (Ctrl-C at a
    launch waited on, a lost terminal, ``molbuilder jobs stop`` of one left
    in the background, a cancel), it starts no further member.  What the
    scheduler tells the job it reads in molbuilder's names, which the job's
    header says once for everything it starts (`kinds.Kind.job_lines`).
    ``empties`` names, by the index of the step it stands before, a folder
    whose scratch family the walk empties there -- a device run's ``tsgf/``
    as each round begins -- and ``empty_at_end`` the folder emptied after
    the last step, a failed member's included; both from the folder the
    walk runs in (`engines/transport.md` § 2a.11).  Stopped by the person,
    the walk starts no further member and empties nothing more.  ``walls``
    are a walk run here's: each member's wall in seconds, told to it as a
    sent job's is (``MB_WALL_S``, `running-a-job.md` § 3.6) -- its own
    card's, ``None`` for one stating none; on a queue the job's line tells
    every member the job's wall, from the job's start.
    THE TWO-LAYER MODEL HOLDS (`job-system.md` § 6): this file orders, copies
    the starts and bounds; each member's own ``.run.sh`` activates its
    environment and launches its engine, exactly as when it runs alone."""
    when = "$(date '+%Y-%m-%dT%H:%M:%S')"
    lines = [
        "#!/usr/bin/env bash",
        f"# {GroupNames(name).name('.run.sh')} -- {where}, in sequence",
        "# (job-system.md § 6.0).  Regenerated at each launch.  THE TWO-LAYER",
        "# MODEL HOLDS (job-system.md § 6): this file is the launcher layer",
        "# only -- ordering, starts and bounds.  Env activation and the",
        "# engine launch stay in each member's own .run.sh, exactly as when",
        "# it runs alone; nothing here re-implements module load / source",
        "# activate.",
        "set -u",
        f'LOG="{log}"',
        # EVERYTHING IT SAYS IS IN ITS LOG -- bash's own errors and a start
        # that could not be copied among them -- so a walk left in the
        # background loses nothing (`job-system.md` § 6.0).
        'exec 2>> "$LOG"',
        f'echo "[group] {when} start {member_word}s={len(steps)} per-{member_word}-bound='
        f'{f"{bound_s}s" if bound_s else "none"} '
        'job=${MB_JOB_ID:-none} node=$(hostname) '
        'alloc_ntasks=${MB_NTASKS:-unset} '
        'alloc_cpus=${MB_CPUS_PER_TASK:-unset}" >> "$LOG"',
        "fails=0",
        # A CHAIN'S FLAG, where a failed member stops the ones after it.
        *([f"_chain_{i}=0" for i in range(len(dict.fromkeys(chains or [None])))]
          if stop_on_failure else []),
        # STOPPED BY THE PERSON -- Ctrl-C, a lost terminal, a cancel --
        # the walk starts no further member (bash runs this after the
        # running member returns; one under a per-member bound ends at it).
        f'_walk_stopped() {{ echo "[group] {when} stopped -- no further '
        f'{member_word}" >> "$LOG"; exit 130; }}',
        "trap _walk_stopped INT TERM HUP",
        *(_empty_scratch_fn(when) if empties or empty_at_end else []),
        "run_member() {",
        '    _name="$1"; _dir="$2"; shift 2',
        "    _t0=$(date +%s)",
        f'    echo "[group] {when} -> ${{_name}} starts" >> "$LOG"',
        (f'    ( cd "${{_dir}}" && timeout -k 30 {bound_s} '
         '"$@" ) >> "$LOG" 2>&1'
         if bound_s else
         '    ( cd "${_dir}" && "$@" ) >> "$LOG" 2>&1'),
        "    _rc=$?",
        '    if [ "${_rc}" -eq 124 ]; then',
        (f'        echo "[group] ${{_name}} hit the {bound_s}s '
         f'per-{member_word} bound -- killed; its artifacts read incomplete" >> "$LOG"'
         if bound_s else
         '        echo "[group] ${_name} killed (124)" >> "$LOG"'),
        "    fi",
        '    if [ "${_rc}" -ne 0 ]; then fails=$((fails+1)); fi',
        "    _t1=$(date +%s)",
        f'    echo "[group] {when} <- ${{_name}} finished rc=${{_rc}} '
        'took=$(( _t1 - _t0 ))s" >> "$LOG"',
        "    return ${_rc}",
        "}",
    ]
    import shlex as _shlex
    from ..scheduler.kinds import WALL_VAR
    chain_of = {c: i for i, c in enumerate(dict.fromkeys(chains or [None]))}
    for k, (member, folder, run_sh, args, before) in enumerate(steps):
        if empties and k in empties:
            lines.append(f'empty_scratch {_shlex.quote(empties[k])}')
        wall = walls[k] if walls else None
        what = (str(run_sh) if isinstance(run_sh, WalkCommand)
                else f'bash "{run_sh}"')
        call = ((f"{WALL_VAR}={int(wall)} " if wall else "")
                + f'run_member "{member}" "{folder}" {what}'
                + (f" {args}" if args else ""))
        # THE MEMBERS AFTER A FAILED ONE IN ITS CHAIN WOULD START FROM IT:
        # they are not run, and the walk goes on with the other chains.
        flag = (f"_chain_{chain_of[chains[k] if chains else None]}"
                if stop_on_failure else None)
        call += (f' || {{ echo "[group] {when} {member} failed -- the '
                 f'members after it in its chain would start from it, so '
                 f'they are not run" >> "$LOG"; {flag}=1; }}'
                 if flag else " || true")
        run = [call]
        if before:
            # ITS START, copied just before it runs (`along`): from a point
            # done before it, which may be one this very walk finished --
            # or it is not run, since its record names that start
            # (`.continued-from`, written with the plan).
            copies = " && ".join(f"cp -f {_shlex.quote(src)} "
                                 f"{_shlex.quote(dst)}/" for src, dst in before)
            run = ([f"if {copies}; then"]
                   + [f'    echo "[group] {member}: starts from {src}" '
                      f'>> "$LOG"' for src, _dst in before]
                   + [f"    {call}",
                      "else",
                      f'    echo "[group] {when} {member} not run -- its '
                      f'start could not be copied" >> "$LOG"',
                      "    fails=$((fails+1))"
                      + (f"; {flag}=1" if flag else ""),
                      "fi"])
        if not flag:
            lines += run
            continue
        lines += [f'if [ "${{{flag}}}" -eq 0 ]; then', *[f"    {r}" for r in run],
                  "else",
                  f'    echo "[group] {when} {member} not run -- its chain '
                  f'stopped at a failed member" >> "$LOG"',
                  "fi"]
    if empty_at_end:
        lines.append(f"empty_scratch {_shlex.quote(empty_at_end)}")
    lines += [f'echo "[group] {when} done fails=${{fails}}" >> "$LOG"',
              "exit $(( fails > 0 ))", ""]
    return "\n".join(lines)


def _no_run_script(what: str, run_name: str, folder: Path,
                   way: str) -> SubmitError:
    """A run script missing where its run is started -- the one refusal of
    every door that starts one (a stage, a trial), with its way back."""
    return SubmitError(f"{what}: {run_name} is not in {folder}.  To write "
                       f"it, {way}")


def _bench_trials(jobset: JobSet, base: Path, plan: LaunchPlan, *, shape,
                  mode: str, jobs=None) -> List[_Member]:
    """A benchmark's walk: its trials still to run -- of ``jobs``, all of
    them by default -- each through the trial's planner
    (:func:`_plan_trial`: where it runs, and whether it was launched) and
    its gates: its deck agrees with its launch, it starts cold, its run
    script is where it is run.  The walk here and a queue's shelves both
    send this answer; a trial launched before is passed over by name
    (``plan.skipped``)."""
    pending: List[_Member] = []
    for job in (jobset.jobs if jobs is None else jobs):
        m = _plan_trial(jobset, base, job, mode=mode, shape=shape)
        if isinstance(m, JobResult):
            plan.skipped.append(m)          # measured before
            continue
        # THE GATES GUARD EVERY DOOR (review 2026-08-21): a trial refused
        # when sent by name must not go silently by riding a walk.  And the
        # COLD gate (user, same day: "it is the submission that determines
        # the actual state of the run"): the pin baked the intent at prep;
        # here the deck itself is verified -- where it IS, the trial's
        # attempt (`project-layout.md` § 1.5a).
        try:
            check_trial_starts_cold(m.read_from, job)
        except TrialNotCold as e:
            raise SubmitError(str(e)) from e
        run_name = m.names.name(".run.sh")
        if not (m.read_from / run_name).exists():
            raise _no_run_script(f"trial {job.name!r}", run_name,
                                 m.read_from,
                                 rollback("the benchmark's prep", base=base))
        plan.reads += [_as_found(m.read_from / f, base)
                       for f in (job.script, run_name)]
        pending.append(m)
    return pending


def _walk_of(members: List[_Member], container: Path) -> list:
    """Each member as the walk runs it -- its name, the folder it runs in
    from the walk's own (its run: `project-layout.md` § 1.5a), its run
    script, and its own counts, the shield against what an allocation
    tells each run; no start to copy.  One answer for a benchmark's walk here,
    a queue's shelf and a group."""
    return [(m.name, str(m.run_dir.relative_to(container)),
             m.names.name(".run.sh"),
             " ".join(_run_sh_args(m.job.resources, m.run)), ())
            for m in members]


def _plan_bench_here(jobset: JobSet, base: Path, *, shape,
                     trial_timeout_s: Optional[int]) -> LaunchPlan:
    """A benchmark's trials run HERE (``--mode direct``, no trial named):
    one submission walking every trial not yet launched, in the sweep's
    order, through the one walk script (:func:`_walk_script`) -- as a shelf
    walks its trials on a queue -- each under the per-trial bound when one
    is given, and told its own card's wall; a trial launched before is
    passed over by name (:func:`_bench_trials`)."""
    plan = LaunchPlan(base, "direct", [])
    pending = _bench_trials(jobset, base, plan, shape=shape, mode="direct")
    if not pending:
        raise _all_launched(jobset, base, shape=shape,
                            count=len(jobset.jobs))
    # THE ONE PARENT THAT SEES EVERY TRIAL -- the benchmark's container,
    # as a shelf's walk runs from it on a queue.
    container = _one_container(pending, base)
    gn = GroupNames.of_trials()
    name = gn.stem
    log = f"{LAUNCH_DIR}/{gn.name('.log')}"
    open_container(base, container / LAUNCH_DIR, plan.writes)
    plan.writes.text(container / LAUNCH_DIR / gn.name(".run.sh"), _walk_script(
        name, _walk_of(pending, container),
        where="this benchmark's unlaunched trials, run here", log=log,
        bound_s=trial_timeout_s,
        walls=[_wall_s(m.job.resources) for m in pending]))
    plan.submissions.append(Submission(
        name, ["bash", f"{LAUNCH_DIR}/{gn.name('.run.sh')}"], container, True, pending,
        rides="rides the group", log=container / log))
    return plan


def _plan_shelf(jobset: JobSet, base: Path, pending: List[_Member],
                names: GroupNames, members, plan: LaunchPlan, *, machine,
                trial_timeout_s: Optional[int], told,
                time_s: Optional[int] = None) -> Submission:
    """One shelf's submission: the walk of ``pending`` -- its trials still
    to run, each planned and gated once (:func:`_bench_trials`) -- planned
    in ``plan`` and written by the send, and the line sending the header
    its prep wrote for all its trials, ``members``, as it is.  A dry run
    and an `ask` write nothing (`job-system.md` § 6.0, step 3).
    """
    container = _one_container(pending, base)
    # L3 (roadmap 7.10, user 2026-08-24): the shelf's own machinery -- its
    # walk, its header, its log, and the scheduler's stdout/err -- lives in
    # ``launch/`` beside the trial directories, not among them.
    launch_dir = container / LAUNCH_DIR
    name = names.stem
    header = launch_dir / names.name(".sbatch")
    # THE HEADER PREP WROTE, or no queue to send to: a benchmark prepared
    # with --no-sbatch, or for a machine that names no queue, has none --
    # said by the one answer (`_no_sbatch`); asked about, the same.
    if _needs_header(plan.mode, machine):
        if not header.is_file():
            raise SubmitError(_no_sbatch(
                name, f"{LAUNCH_DIR}/{names.name('.sbatch')}", base=base,
                told=told, machine=machine))
        plan.reads.append(_as_found(header, base))
    # THE ATTEMPT, NOT THE TRIAL (`project-layout.md` § 1.5a).  Each member's
    # ``run_dir`` is the one answer to "where does this trial run", and the
    # gates asked it (:func:`_bench_trials`).  Each trial is handed its own
    # -np / -omp: the shield against what the allocation tells each run.
    script = _walk_script(
        name, _walk_of(pending, container),
        where="ONE allocation, this shelf's unlaunched trials",
        log=f"{LAUNCH_DIR}/{names.name('.log')}", bound_s=trial_timeout_s)
    # THE ONE LINE (`_sbatch_request`): the shelf's header as prep wrote it
    # -- nothing chosen here.
    sent, placement, cmd, outputs_in = _sbatch_request(
        header=header, placement=_placed_by_prep(members),
        time_s=time_s, job_name=_scheduler_job_name(jobset, name),
        script=f"{LAUNCH_DIR}/{names.name('.sbatch')}", run_args=(),
        machine=machine)
    if plan.mode == "submit":
        open_container(base, launch_dir, plan.writes)
        plan.writes.text(launch_dir / names.name(".run.sh"), script)
    # A SHELF THE SCHEDULER REFUSES keeps its trials pending -- no launch
    # record -- so launching the benchmark again sends exactly them; a
    # shelf sent by hand would write no record, and the next launch would
    # measure its trials twice (the unit 11 review, 2026-10-05).
    hint = ("\n  Its trials stay pending: launch the benchmark again once "
            "the ask fits -- the shelves already queued are skipped.")
    return Submission(
        name, cmd, container, False, pending,
        placement=placement, sent=sent, rides="rides the group",
        header=header, refusal_hint=hint, machine=machine,
        outputs_in=outputs_in)


def _points_walk(plan: LaunchPlan, m: "_Member", calc, *, machine, mode: str,
                 time_s: Optional[int], told,
                 inside: Optional[Inside] = None) -> None:
    """A RUN OF POINTS SENT: one submission walking ``m``'s run -- its
    points as :func:`_plan_member` opened and filled them -- into ``plan``
    (`engines/transport.md` § 2a.11).

    **The run is the member**: one launch record, written at ``run-<n>/``
    by the one sender; each point is done or not done, read from its own
    files (`continuation.done`).  Launched again warm, each point done in
    the newest launched run is taken over (its files copied, its
    `.continued-from` naming the run that computed it, one hop) and the
    rest are walked; cold, or at a first launch, every point.  Each walked
    point starts from the closest voltage computed before it IN ITS FRAME
    (taken over, or walked earlier in this walk) -- the files the
    calculation's `along` row names and the rung itself declares (a
    device's `.TSDE`), copied just before it runs -- or, a frame's first,
    from its own gathered inputs: frames never start from one another
    (§ 2a.9 *Both axes*).  A rung that hands nothing forward
    (the transmission) walks on past a point that fails; one that does ends
    that frame's chain there, and the walk goes on with the other frames.
    THE WALK RUNS ROUND BY ROUND -- every frame's point at a voltage before
    the next voltage (§ 2a.11, *What a run's points share*) -- and, where
    the rung keeps the engine's scratch in a folder of the run's own
    (`Job.scratch_dir`), empties that folder as each round begins and when
    the walk ends.  The walk is the one walk script (:func:`_walk_script`); the header is its
    prep's, written for every point, their walls summed (`group.envelope`,
    `prep.prep_calculation`) -- the points done before walk on past.  Run
    here, each point is told its card's wall (`running-a-job.md` § 3.6)."""
    from ..runrecord import launch_record, next_run, read_continued_from
    from ..transport.energies import SHARED_DIR
    from ..transport.stages import hands_on, products_of
    from ..warmfiles import never_carried
    from .continuation import done
    jobset, base, task = calc.jobset, Path(calc.base), calc.task
    job, names, stage = m.job, m.names, m.job.name
    stage_dir, run, prev = m.container, m.run_dir, m.ran
    # WHAT NO COPY CARRIES (the one answer, `warmfiles.never_carried`): an
    # engine's scratch, recomputed by every run, so never taken over -- it
    # lies in the run's own folder for it (`Job.scratch_dir`), and a run
    # made before held it in its points -- and a file the catalogue calls
    # transient, a reading's claim speaking for a process not the new
    # run's.
    # WHAT A POINT MUST HOLD TO BE DONE -- the one list (`products_of`):
    # what the rungs downstream take, and what a point hands to the next
    # (`hands_on`, the one answer of that).
    products = products_of(stage, task.label, base_dir=base)
    hand = hands_on(stage, task.label, base)
    taken, walked = [], []
    prev_launch = launch_record(prev, names) if prev is not None else None
    # A FIRST LAUNCH IN A CALCULATION THAT REUSES A FINISHED ONE takes that
    # calculation's points over -- each done there, the same calculation,
    # given what it was given (`engines/transport.md` § 2a.16); --cold takes
    # none.
    across = (_across(calc, stage, m)
              if m.first and not m.cold and task.reuses else None)
    over = ({(t.point.frame, t.point.volts) for t in across.taken}
            if across is not None else set())
    for pdir, pt in m.points:
        if prev is not None and not m.cold:
            src = prev / pt.rel
            ok, _why = done(src, names, launch=prev_launch, products=products,
                            launched_in=prev)
            if ok:
                taken.append((pdir, pt, src))
                continue
        if (pt.frame, pt.volts) in over:
            continue
        walked.append((pdir, pt))
    if not walked and across is not None:
        raise SubmitError(
            f"{stage}: every point is one {across.run} computed the same, "
            f"so its first launch has nothing to run -- a calculation that "
            f"reuses a finished one answers a wider question "
            f"(engines/transport.md § 2a.16).  To run them all here, launch "
            f"it cold:\n"
            + block(launch_lines("task", stage, base=base, flags=("--cold",))))
    if not walked:
        raise SubmitError(
            f"{stage}: every point of {prev.relative_to(base)} is done, so a "
            f"warm launch has nothing to run -- to run them all again, "
            f"launch it cold:\n"
            + block(launch_lines("task", stage, base=base, flags=("--cold",))))
    # THE POINTS TAKEN OVER, each one hand-over: the done point's folder
    # whole -- its inputs, the engine's results the readers and the next
    # rungs use, the run's own account -- but not the engine's scratch;
    # its record the run that COMPUTED it, one hop (`engines/transport.md`
    # § 2a.11): a point taken over from a point itself taken over names the
    # run that ran it.
    for pdir, _v, src in taken:
        at = RunRef.at(base, src)
        # The source's own record is its ORIGIN only when it names a point
        # of another run (it was taken over itself); a point the source run
        # walked records its START there -- the point before it in that run
        # -- and the run that computed it is the source run.
        marker = read_continued_from(src, names, FIRST_ATTEMPT)
        origin = (marker if marker is not None
                  and not marker.within(RunRef.at(base, prev)) else at)
        hand_over(base, HandOver(
            pdir, names, FIRST_ATTEMPT, CONTINUED, source=origin,
            files=tuple((at, f.name) for f in sorted(src.iterdir())
                        if f.is_file() and not never_carried(f.name))),
            plan.writes)
        # ITS SHARED ENERGIES, with it (`engines/transport.md` § 2a.16).
        from .reuse import carry_shared
        carry_shared(src / SHARED_DIR, pdir / SHARED_DIR, plan.writes,
                     root=base)
    # THE POINTS TAKEN OVER FROM THE FINISHED CALCULATION, into their folders
    # beside what the gather gave them (`reuse.take_into`).
    if across is not None and across.taken:
        from .reuse import take_into
        take_into(calc, across.taken, m.points, plan.writes,
                  given=[t["file"] for t in m.took])
    # THE WALK: round by round -- every frame's point at a voltage, then the
    # next voltage (`engines/transport.md` § 2a.11) -- a point's start the
    # closest voltage before it in ITS frame (§ 2a.9), which ran a round
    # earlier or was taken over.
    order = [pt for _p, pt in m.points]
    rounds = list(dict.fromkeys(pt.volts for pt in order))

    def _before(pt):
        # THE CLOSEST VOLTAGE ALREADY COMPUTED, in this frame
        # (`engines/transport.md` § 1.1; the SIESTA manual: "copy the TSDE
        # from the closest, previously, calculated bias") -- of the points
        # before it in the list, the nearest in volts, the later on a tie.
        mine = [q for q in order if q.frame == pt.frame]
        earlier = mine[:mine.index(pt)]
        if not earlier:
            return None
        return min(reversed(earlier),
                   key=lambda q: abs(float(q.volts) - float(pt.volts)))
    walked.sort(key=lambda pp: (rounds.index(pp[1].volts),
                                order.index(pp[1])))
    # EVERY WALKED POINT THAT SHARES ENERGIES is given them (`_shares`).
    for pdir, pt in walked:
        _shares(calc, job, stage_dir / pt.rel, pdir, pt,
                prev / pt.rel / SHARED_DIR if prev is not None else None,
                plan.writes)
    # THE RUN'S SCRATCH FOLDER, emptied as each round begins -- before the
    # round's first point this walk runs -- and when the walk ends.
    scratch_at = (f"{run.name}/{job.scratch_dir}" if job.scratch_dir
                  else None)
    empties: Dict[int, str] = {}
    seen_rounds: set = set()
    steps, said, chains = [], [], []
    for pdir, pt in walked:
        if scratch_at and pt.volts not in seen_rounds:
            seen_rounds.add(pt.volts)
            empties[len(steps)] = scratch_at
        prior = _before(pt)
        tries = next_run(pdir, names)
        # ITS START, one hand-over decided now: its record written beside
        # it, so the record says what each point started from
        # (`engines/transport.md` § 2a.11) -- and its files copied by the
        # walk just before it runs, since the point before it may finish in
        # this very walk.
        start = None
        if hand and prior is not None:
            at = RunRef.at(base, run / prior.rel)
            start = HandOver(pdir, names, tries, CONTINUED, source=at,
                             files=tuple((at, f) for f in hand), walk=True)
            hand_over(base, start, plan.writes)
        steps.append((f"{stage}@{pt.rel}",
                      f"{run.name}/{pt.rel}", names.name(".run.sh"),
                      " ".join(_run_sh_args(job.resources, tries)),
                      start.walk_copies(base, stage_dir) if start else ()))
        chains.append(pt.frame)
        said.append(pt.words() + (f" (from {prior.words()}'s "
                                  f"{', '.join(hand)})" if start else ""))
        plan.reads += [_as_found(stage_dir / pt.rel / f, base)
                       for f in (job.script, names.name(".run.sh"))]
    walk = ((f"gathers {_gathered_from(m.took)}; " if m.took else "")
            + (f"takes over {', '.join(pt.words() for _p, pt, _s in taken)} "
               f"done in {prev.name}; " if taken else "")
            + (f"takes over "
               f"{', '.join(t.point.words() for t in across.taken)} from "
               f"{across.run}; " if across is not None and across.taken
               else "")
            + f"runs {', '.join(said)}")
    member = dataclasses.replace(m, read_from=stage_dir / walked[0][1].rel,
                                 walk=walk,
                                 copied=across.taken if across else ())
    gn = GroupNames.of_points(names.stem)
    launch_dir = stage_dir / LAUNCH_DIR
    log = f"{LAUNCH_DIR}/{gn.name('.log')}"
    script = _walk_script(
        gn.stem, steps, where=f"{stage}'s sweep, {run.name}",
        member_word="point", log=log,
        stop_on_failure=bool(hand), chains=chains,
        empties=empties, empty_at_end=scratch_at,
        walls=([_wall_s(job.resources)] * len(steps)
               if _own_walls(mode, inside)
               else None))
    if mode != "ask":
        open_container(base, launch_dir, plan.writes)
        plan.writes.text(launch_dir / gn.name(".run.sh"), script)
    if mode == "direct":
        plan.submissions.append(Submission(
            gn.stem, ["bash", f"{LAUNCH_DIR}/{gn.name('.run.sh')}"],
            stage_dir, True, [member], rides="rides the sweep",
            log=stage_dir / log))
        return
    # ---- submit / ask: one scheduler job walking the points ------------ #
    # THE WALK'S ONE HEADER, as its prep wrote and placed it: one
    # allocation walking every point, their walls summed (`group.envelope`,
    # `prep.prep_calculation`) -- or no queue to send to, said by the one
    # answer (`_no_sbatch`); asked about, the same.
    header = launch_dir / gn.name(".sbatch")
    if _needs_header(mode, machine):
        if not header.is_file():
            raise SubmitError(_no_sbatch(
                gn.stem, f"{LAUNCH_DIR}/{gn.name('.sbatch')}", base=base,
                told=told, machine=machine,
                redo=_redo(calc, job)))
        plan.reads.append(_as_found(header, base))
    sent, placement, cmd, outputs_in = _sbatch_request(
        header=header, placement=_placed_by_prep([job]), time_s=time_s,
        job_name=_scheduler_job_name(jobset, gn.stem),
        script=f"{LAUNCH_DIR}/{gn.name('.sbatch')}", run_args=(),
        machine=machine)
    plan.submissions.append(Submission(
        gn.stem, cmd, stage_dir, False, [member], placement=placement,
        sent=sent, rides="rides the sweep", header=header, machine=machine,
        outputs_in=outputs_in))


def _shares(calc, job, prepared: Path, folder: Path, pt, held,
            writes) -> None:
    """WHAT A TRANSMISSION POINT SHARES, given it at a launch that walks it
    -- a rung that runs once, its one point (``None``), the run itself
    (`engines/transport.md` § 2a.16): from ``held``, the newest run here's
    ``shared/`` at the point, when it holds them whole, else from the point
    the deck its prep wrote in ``prepared`` names (`reuse.shared_into`) --
    refused, in the launch's words, when neither is there."""
    from .reuse import SHARING_RUNG, shared_into
    if job.name != SHARING_RUNG:
        return
    try:
        shared_into(calc, folder, pt, prepared / job.script, held, writes)
    except ValueError as exc:
        raise SubmitError(f"{job.name}: {exc}") from None


def _across(calc, stage: str, m: "_Member"):
    """The points ``m``'s first launch takes over from the finished
    calculation this one reuses (`reuse.takes_over`), each matched by what
    this launch's gather gives it -- or ``None`` when that calculation is
    not there to read."""
    from .gather import ReadOnce
    from .reuse import finished_of, origin, takes_over
    finished, _gone = finished_of(calc)
    if finished is None:
        return None
    base = Path(calc.base)

    def given(pt):
        return {t["file"]: origin(base, t["from"].path(base))
                for t in m.took
                if (t["frame"], t["volts"]) == (pt.frame, pt.volts)}
    return takes_over(calc, finished, stage, [pt for _p, pt in m.points],
                      once=ReadOnce(calc), inputs=given)


def _gathered_now(calc, stage: str):
    """A transport rung's gather, read now (`gather.gather_sources`, its three
    gates) -- or the refusal, in the gather's words, before anything is
    sent (`job-system.md` § 6.0, step 1)."""
    from .errors import PrepError, user_error_as_prep
    from .gather import gather_sources
    try:
        with user_error_as_prep():
            return gather_sources(calc, stage)
    except PrepError as exc:
        raise SubmitError(str(exc)) from None


def _gather_sweep(base: Path, run: Path, points, got, writes, *,
                  names: RunNames) -> Tuple[List[dict], Tuple[HandOver, ...]]:
    """A swept run given its gather: what every point takes from one run
    into the run once -- the clean source a launch again copies from -- and
    each point's own into its folder (`engines/transport.md` § 2a.11), each
    one hand-over with its record; returns the gather as the ledger records
    it, and the run's own hand-over."""
    from .gather import gather_into
    by_point = {pt: inputs for pt, inputs in got}
    whole = gather_into(base, run,
                        list(dict.fromkeys(fi for inputs in by_point.values()
                                           for fi in inputs
                                           if all(fi in other for other
                                                  in by_point.values()))),
                        writes, names=names, run=FIRST_ATTEMPT)
    for pdir, pt in points:
        gather_into(base, pdir, by_point.get(pt, []), writes, names=names,
                    run=FIRST_ATTEMPT)
    return ([{"file": fn, "from": src,
              "frame": pt.frame if pt is not None else None,
              "volts": pt.volts if pt is not None else None}
             for pt, inputs in got for src, fn in inputs],
            _hands(whole))


def _carry_sweep_gather(prev_run: Path, run: Path, points, writes, *,
                        base: Path, stage: str, names: RunNames
                        ) -> Tuple[HandOver, ...]:
    """A run of points launched again takes what its newest launched run
    gathered -- never gathered again (`job-system.md` § 5.4): the run's own
    record into the new run, and each point's into its point, by the one
    carry (:func:`_carry_the_gather`).  Returns the run's own hand-over."""
    whole = _carry_the_gather(prev_run, run, writes, base=base, stage=stage,
                              names=names, run=FIRST_ATTEMPT)
    for pdir, pt in points:
        _carry_the_gather(prev_run / pt.rel, pdir, writes, base=base,
                          stage=stage, names=names, run=FIRST_ATTEMPT)
    return _hands(whole)


def _placed_on(placement, sent=None) -> Optional[dict]:
    """The queue prep placed the run on -> where this run was SENT, and with
    what wall and memory (``sent``, the request as sent: its wall the
    launch's ``--time`` when restated, `job-system.md` § 6.0) -- or ``None``
    when there was no placement (a direct run).

    The QUEUE half of `scheduler.md` R12: domain, partition, qos -- known
    the moment the scheduler accepts, and reachable before this field only by
    parsing the argv the same file records.  What the job LANDED ON is the
    monitor's to record, on the node, because a queued job has no node yet.
    """
    if placement is None:
        return None
    return {"domain": placement.name,
            "partition": placement.partition,
            "qos": placement.qos,
            **{f: getattr(sent, f) for f in ("time", "mem")
               if getattr(sent, f, None) not in (None, "")}}


def _hands(*hands: Optional[HandOver]) -> Tuple[HandOver, ...]:
    """The hand-overs a run takes, those there are."""
    return tuple(h for h in hands if h is not None)


def _flat_hand(folder: Path, names: RunNames, run: int,
               cont: Optional[Continuation]) -> Tuple[HandOver, ...]:
    """A flat run's hand-over -- the run it continues from; nothing is
    copied, the restart files being the folder's, and its record is written
    with its launch record (:func:`_record_launch`), since the flat layout
    opens no run before its launch (`job-system.md` § 5.4)."""
    ref = cont.ref if cont is not None else None
    return ((HandOver(folder, names, run, CONTINUED, source=ref),)
            if ref is not None else ())


def _member_facts(s: "Submission") -> Dict[str, object]:
    """What the ledger records of each member beyond its name
    (`feedback_decision_ledger`): the run it goes into, and a walked
    member's plan -- which points are taken over and which walked, from
    what (`_points_walk`'s ``walk``) -- so a sweep's history reads from the
    ledger, not the terminal alone."""
    runs = {m.name: _rel(m.run_dir, m.base) for m in s.members}
    walks = {m.name: m.walk for m in s.members if m.walk}
    return {"runs": runs, **({"walks": walks} if walks else {})}


def _record_launch(m: "_Member", *, mode: str, command: List[str],
                   job_id: Optional[str] = None, placement=None,
                   sent=None, scheduler: Optional[dict] = None,
                   inside: Optional[Inside] = None) -> None:
    """Write member ``m``'s launch record in the folder its run runs in,
    named by its stage's names (`runrecord.write_launch`) -- carrying what
    it continues from, its own hand-over (`job-system.md` § 5.4).  Where a
    stage's runs share the folder, that hand-over's record is written now
    too, with the launch record: the flat layout opens no run before its
    launch, so a refused send leaves no file of a run that never started.
    ``placement`` and ``scheduler`` are passed because submission is what
    knows where the job went and where its scheduler writes of it, and
    nothing downstream should have to work either out from a command
    line.  A run a group's launch ``inside`` a sequence's job sends records
    that job -- its mode, line, id, placement and scheduler as the
    sequence's launch record states them, and its name (`job-system.md`
    § 6.0, *A sequence of groups*)."""
    from ..runrecord import write_launch
    cont = m.continued
    if cont is not None and m.names.shared:
        hand_over(m.base, cont)
    said = (inside.run_said(m) if inside is not None else
            {"mode": mode, "command": command, "job_id": job_id,
             "placed_on": _placed_on(placement, sent),
             "scheduler": scheduler, "sequence": None})
    write_launch(m.run_dir, names=m.names, run=m.run, mode=said["mode"],
                 command=said["command"], job_id=said["job_id"],
                 continued_from=cont.source if cont is not None else None,
                 placed_on=said["placed_on"], scheduler=said["scheduler"],
                 sequence=said["sequence"])


def _plan_group(calc, names: List[str], *, machine, mode: str, told,
                number: int, time_s: Optional[int] = None,
                described=None, inside: Optional[Inside] = None
                ) -> LaunchPlan:
    """ONE submission walking a GROUP's stages -- the description's group
    ``number``, its members ``names`` in the ladder's order, prepared as one
    job (`project-layout.md` § 1.6.6).  Each is planned as a stage alone is
    (:func:`_plan_member`: at its first launch, what it takes; launched
    again, warm -- `--cold` speaks for one stage, R-L9), its deck agreeing
    with its launch; the walk runs each one's own run script in its own
    attempt, and one that fails does not stop the others -- each told its
    own card's wall when run here.  The header is the one the group's prep
    wrote (`launch/<group>.sbatch`), its members' shared request
    (`group.envelope`, refused at prep, R-P13) -- at the group's first
    launch checked against what prep would write now (`prep.group_header`,
    R-L16), as each member is (`job-system.md` § 6.0); the walker is
    written at the send, with each run's number."""
    from .group import names_for
    jobset, base, task = calc.jobset, Path(calc.base), calc.task
    jobs = [jobset.job(n) for n in names]
    gn = names_for(base, task, names)
    plan = LaunchPlan(base, mode, [])
    members: List[_Member] = []
    for job in jobs:
        m = _plan_member(calc, job, writes=plan.writes, described=described)
        plan.reads += [_as_found(m.read_from / f, base)
                       for f in (job.script, m.names.name(".run.sh"))]
        members.append(m)

    # THE ONE WALK SCRIPT: the members in order, each its own run in its own
    # run folder; one that fails does not stop the others -- none builds on
    # another.
    log = f"{LAUNCH_DIR}/{gn.name('.log')}"
    script = _walk_script(
        gn.stem,
        [(m.name, str(m.run_dir.relative_to(base)),
          m.names.name(".run.sh"),
          " ".join(_run_sh_args(m.job.resources, m.run,
                                cold=m.cold and not m.has_attempt)), ())
         for m in members],
        where="a group's one job: " + ", ".join(names),
        member_word="stage", log=log,
        walls=([_wall_s(m.job.resources) for m in members]
               if _own_walls(mode, inside) else None))
    launch_dir = base / LAUNCH_DIR
    if mode != "ask":
        open_container(base, launch_dir, plan.writes)
        plan.writes.text(launch_dir / gn.name(".run.sh"), script)
    if mode == "direct":
        plan.submissions.append(Submission(
            gn.stem, ["bash", f"{LAUNCH_DIR}/{gn.name('.run.sh')}"], base,
            True, members, rides="rides the group", log=base / log))
        return plan
    header = launch_dir / gn.name(".sbatch")
    # THE HEADER PREP WROTE, or no queue to send to -- asked about where the
    # machine has a scheduler to ask, as every door's is.
    if _needs_header(mode, machine):
        if not header.is_file():
            raise SubmitError(_no_sbatch(
                f"group {number}", f"{LAUNCH_DIR}/{gn.name('.sbatch')}",
                base=base, told=told, machine=machine,
                redo=_redo(calc, jobs[0])))
    if header.is_file() and any(m.first for m in members):
        # THE HEADER IS STILL WHAT PREP WOULD WRITE (R-L16): the members'
        # shared request, rendered now.
        from .errors import PrepError
        from .prep import as_written, group_header
        try:
            now = group_header(base, task, number, jobs,
                               described().environment)
        except PrepError as exc:
            raise SubmitError(str(exc)) from None
        if now is None or not as_written(header, now.text.encode("utf-8")):
            raise SubmitError(
                f"group {number}'s header, {LAUNCH_DIR}/{header.name}, is "
                f"not what prep would write now for its members' shared "
                f"request -- the header was edited, or the machine's record "
                f"it is rendered from changed, since the prep (job-system.md "
                f"§ 6.0).  To prepare it as it is described now, "
                + _redo(calc, jobs[0]))
    if header.is_file():
        plan.reads.append(_as_found(header, base))
    sent, placement, cmd, outputs_in = _sbatch_request(
        header=header, placement=_placed_by_prep(jobs), time_s=time_s,
        job_name=_scheduler_job_name(jobset, gn.stem),
        script=f"{LAUNCH_DIR}/{gn.name('.sbatch')}", run_args=(),
        machine=machine)
    # `--mode ask` ASKS OVER THE LINE ITSELF: the header prep wrote is the
    # one the send passes.
    plan.submissions.append(Submission(
        gn.stem, cmd, base, False, members, placement=placement, sent=sent,
        rides="rides the group", header=header, machine=machine,
        outputs_in=outputs_in))
    return plan


def _held_by(held, number: int, base) -> str:
    """The refusal of a group a sequence still going holds (R-L45): its job,
    what it is doing, and the line that stops it."""
    job, after, said = held
    return (f"group {number} goes in job {job.id} -- a sequence of groups "
            f"still going ({said}) launches it "
            + (f"after group {after}" if after is not None else "first")
            + "; launch it alone or in another sequence once that job has "
              "ended, or stop the job first:\n"
            + block([f"molbuilder jobs stop {job.id}"])
            + "\n  (job-system.md § 6.0, *A sequence of groups*; R-L45)")


def _plan_sequence(calc, jobset: JobSet, numbers: Sequence[int], *,
                   machine, mode: str, time_s: Optional[int], told,
                   one: bool, described) -> LaunchPlan:
    """A SEQUENCE OF GROUPS (`job-system.md` § 6.0, *A sequence of
    groups*): ONE submission whose walk launches each group named, in the
    description's order, as `launch task --group N --mode direct` launches
    it, on the job's node -- each planned when its turn comes, so each takes
    what it builds on from the runs the groups before it have just finished.

    Refused before anything is sent (R-L45): a flag that speaks for one
    stage (``one``); a group not prepared, or a group a sequence still going
    holds -- each group asked as its own launch asks it
    (:func:`_the_group`, `ready.held_by_sequence`); under ``submit`` or
    ``ask``,
    no header prep wrote for exactly these groups (`prep task --group …
    --one-job`), or one that is not what prep would write now.  The plan
    shows the first group's plan, made as its own launch makes it now -- its
    gates, its runs, what each takes -- and each group after it by its
    stages; it writes the sequence's walk and its launch record
    (`sequence.Record`), both before the job goes, since the job reads
    them."""
    import shlex
    import sys
    from ..runfiles import LAUNCH_DIR
    from .commands import block, prep_sequence_lines
    from .group import sequence_names
    from .ready import held_by_sequence
    from .sequence import Record, record_path, record_text
    base, task = Path(calc.base), calc.task
    if jobset.kind != KIND_LADDER:
        raise SubmitError("several groups go as one job in a task, and this "
                          "is a benchmark (job-system.md § 6.0).")
    numbers = sorted(set(int(n) for n in numbers))
    if one:
        raise SubmitError(
            "--stage, --cold, --from and --gather speak for one stage, and "
            f"groups {', '.join(map(str, numbers))} go as one job of "
            "several groups -- launch that stage alone (job-system.md "
            "§ 6.0, *A sequence of groups*; R-L45).")
    for n in numbers:
        # EACH GROUP AS ITS OWN LAUNCH ASKS IT: described, prepared, its
        # members not changed since -- and held by no sequence still going.
        _the_group(calc, group=n, only=None, one=False)
        held = held_by_sequence(calc, n)
        if held is not None:
            raise SubmitError(_held_by(held, n, base))
    groups = [{"number": n, "stages": list(calc.request.members(n))}
              for n in numbers]
    names = sequence_names(base, task, calc.request, numbers)
    words = ", ".join(map(str, numbers))
    # THE FIRST GROUP'S PLAN, made as its own launch makes it now, run here
    # inside the job -- its refusal is this launch's; nothing of it is
    # written by this send: its own launch, inside the job, plans and
    # writes it.
    first = _planned(calc, jobset, machine=machine, mode="direct", only=None,
                     side=None, time_s=None, trial_timeout_s=None, told=told,
                     group=numbers[0])
    preview = [r for r in first.shown() if r.status != "planned"]
    preview += [JobResult(f"group {g['number']} ({', '.join(g['stages'])})",
                          [], "takes",
                          said=(f"launched in this job when group "
                                f"{numbers[i]} has finished, taking what it "
                                f"builds on then"))
                for i, g in enumerate(groups[1:])]
    plan = LaunchPlan(base, mode, [])
    launch_dir = base / LAUNCH_DIR
    walk_name = f"{LAUNCH_DIR}/{names.name('.run.sh')}"
    log = f"{LAUNCH_DIR}/{names.name('.log')}"
    rec_path = record_path(base, names)
    # THE WALK: each group's launch, in order, the first that does not
    # finish ending it -- THE MOLBUILDER SENDING IT, its interpreter run
    # from the folder its package was imported from, which the job's node
    # must see; the calculation by its full path; the job's address as its
    # walk knows it (`job-system.md` § 6.0, *The walk*).
    import molbuilder as _mb
    home = Path(_mb.__file__).resolve().parent.parent
    steps = [(f"group {g['number']}", str(home), WalkCommand(
        " ".join([shlex.quote(sys.executable), "-m", "molbuilder", "jobset",
                  "launch", "task", "--group", str(g["number"]), "--bundle",
                  shlex.quote(str(base)), "--mode", "direct", "--foreground",
                  "--yes", "--in-sequence", shlex.quote(names.stem),
                  "--in-job", '"${MB_JOB_ID:-$(hostname)-$$}"'])), "", ())
        for g in groups]
    script = _walk_script(names.stem, steps,
                          where=f"groups {words} in sequence, one job",
                          member_word="group", log=log,
                          stop_on_failure=True, chains=[0] * len(steps))
    sequence = {"name": names.stem, "groups": groups, "record": rec_path}
    if mode == "direct":
        sub = Submission(names.stem, ["bash", walk_name], base, True, [],
                         log=base / log, sequence=sequence, preview=preview)
        rec = Record(groups=groups, mode="direct", command=sub.command,
                     placed_on=None, scheduler=None)
    else:
        header = launch_dir / names.name(".sbatch")
        jobs = [jobset.job(st) for g in groups for st in g["stages"]]
        if mode == "submit" and not machine.kind.queue:
            # NO QUEUE TO SEND TO: the one answer every door gives
            # (`_no_sbatch`), the run here as a command.
            raise SubmitError(_no_sbatch(
                f"groups {words}", f"{LAUNCH_DIR}/{header.name}", base=base,
                told=told, machine=machine))
        if _needs_header(mode, machine) and not header.is_file():
            raise SubmitError(
                f"groups {words} go as one job, and prep wrote no header "
                f"for them ({LAUNCH_DIR}/{header.name}) -- prepare them as "
                f"one job first:\n"
                + block(prep_sequence_lines(numbers, base=base))
                + "\n  (job-system.md § 6.0, *A sequence of groups*; R-L45)")
        if header.is_file():
            _sequence_header_is_prep_s(calc, numbers, groups, header,
                                       described)
            plan.reads.append(_as_found(header, base))
        sent, placement, cmd, outputs_in = _sbatch_request(
            header=header, placement=_placed_by_prep(jobs), time_s=time_s,
            job_name=_scheduler_job_name(jobset, names.stem),
            script=f"{LAUNCH_DIR}/{names.name('.sbatch')}", run_args=(),
            machine=machine)
        sub = Submission(names.stem, cmd, base, False, [],
                         placement=placement, sent=sent, header=header,
                         machine=machine, outputs_in=outputs_in,
                         sequence=sequence, preview=preview)
        plan.queue = _the_queue(jobs)
        rec = Record(
            groups=groups, mode="submit", command=cmd,
            placed_on=_placed_on(placement, sent),
            scheduler={"kind": machine.kind.name,
                       "outputs_in": os.path.relpath(
                           base / (outputs_in or ""), base)})
    if mode != "ask":
        open_container(base, launch_dir, plan.writes)
        plan.writes.text(base / walk_name, script)
        plan.writes.text(rec_path, record_text(rec))
    plan.engine = jobset.engine
    plan.submissions.append(sub)
    return plan


def _sequence_header_is_prep_s(calc, numbers, groups, header,
                               described) -> None:
    """THE SEQUENCE'S HEADER IS STILL WHAT PREP WOULD WRITE (R-L45, as a
    group's is, R-L16): each group's own request as its rows stand now, the
    sequence's one request rendered now (`prep.sequence_header`), compared
    with the file by the one test (`prep.as_written`)."""
    from ..transport.stages import frames_of, rung_points
    from .commands import block, prep_sequence_lines
    from .errors import PrepError
    from .prep import as_written, sequence_header
    base, task, jobset = Path(calc.base), calc.task, calc.jobset
    jobs_of = {g["number"]: [jobset.job(st) for st in g["stages"]]
               for g in groups}
    points_of = {n: (len(rung_points(task, js[0].name,
                                     frames=frames_of(base)))
                     if len(js) == 1 else 0) for n, js in jobs_of.items()}
    try:
        now, _less = sequence_header(base, task, calc.request, numbers,
                                     jobs_of, points_of,
                                     described().environment)
    except PrepError as exc:
        raise SubmitError(str(exc)) from None
    if now is None or not as_written(header, now.text.encode("utf-8")):
        raise SubmitError(
            f"the header of groups {', '.join(map(str, numbers))} as one "
            f"job, {header.relative_to(base)}, is not what prep would write "
            f"now -- a group was prepared again, or the header or the "
            f"machine's record changed, since that prep.  Prepare them as "
            f"one job again:\n" + block(prep_sequence_lines(numbers,
                                                            base=base))
            + "\n  (job-system.md § 6.0, *A sequence of groups*; R-L45)")


def _plan_stage(calc, jobset: JobSet, *, machine, mode: str,
                only: Optional[str], time_s: Optional[int], told,
                cold: bool = False,
                from_attempt: Optional[str] = None, gather: bool = False,
                described=None, inside: Optional[Inside] = None
                ) -> LaunchPlan:
    """The stage door's plan -- a task's stage, or a benchmark's named
    trial: one submission.

    Before anything is written: where it runs and what it takes
    (:func:`_plan_member`; a trial, :func:`_plan_trial`); a run of points
    sent as the points' walk (:func:`_points_walk`); else the deck/launch
    agreement and a trial's cold start, the wrapper the line runs -- where
    the line is run, or asked about when a scheduler is here to ask -- and,
    for a scheduler, the one line (:func:`_sbatch_request`): the header
    prep wrote, sent to the queue it recorded.

    A ladder is launched ONE stage at a time, in every mode -- direct
    running stages in order would be local chaining.  ``jobset`` is the set
    launched -- the calculation's plan, or a benchmark's sweep -- and stays
    whole: what this launch SENDS is the jobs it names (``only``), and every
    question about another stage is asked of the whole plan
    (``calc``, A16).
    """
    base, shape = Path(calc.base), calc.shape
    named = [j for j in jobset.jobs if only is None or j.name == only]
    # The no-chain rule, AT THE SEAM (U5, 2026-08-12): a guard only a
    # surface applies is one the next surface forgets.
    if jobset.kind == KIND_LADDER and len(named) > 1:
        raise SubmitError(
            "a task's launch sends one job -- a group, or one stage "
            "launched again -- and names which; with nothing named, `launch "
            "task` sends the group that can go:\n"
            + block(launch_lines("task", base=base))
            + "\n  Stages do not chain (project-layout.md § 1.6), in direct "
              "mode as much as submit: a stage that builds on another is "
              "launched after you have looked at the one it builds on.")

    plan = LaunchPlan(base, mode, [])
    for job in named:
        m = (_plan_trial(jobset, base, job, mode=mode, shape=shape,
                         named=only is not None)
             if jobset.kind == KIND_SWEEP else
             _plan_member(calc, job, writes=plan.writes,
                          cold=cold, from_attempt=from_attempt,
                          gather=gather, described=described))
        if isinstance(m, JobResult):
            plan.skipped.append(m)             # a trial measured before
            continue
        if m.points is not None:
            _points_walk(plan, m, calc, mode=mode,
                         time_s=time_s, told=told, machine=machine,
                         inside=inside)
            continue
        if jobset.kind == KIND_SWEEP:
            # the cold gate rides the named-trial door too (user,
            # 2026-08-21) -- one rule, every launch path
            try:
                check_trial_starts_cold(m.read_from, job)
            except TrialNotCold as e:
                raise SubmitError(str(e)) from e
        run_name = m.names.name(".run.sh")
        sbatch_name = m.names.name(".sbatch")
        plan.reads += [_as_found(m.read_from / f, base)
                       for f in (job.script, run_name, sbatch_name)]
        if mode == "direct":
            # The wrapper is required where it is RUN -- and a dry run is
            # this plan, shown: it says the refusal the launch would meet.
            if not (m.read_from / run_name).exists():
                # THE PREP THAT WROTE IT, again (`job-system.md` § 5.0).
                raise _no_run_script(
                    f"job {job.name!r}", run_name, m.read_from,
                    rollback("the benchmark's prep", base=base)
                    if jobset.kind == KIND_SWEEP else _redo(calc, job))
            # TOLD ITS CARD'S WALL, as a sent job is told its header's
            # (`running-a-job.md` § 3.6).
            plan.submissions.append(Submission(
                m.name, ["bash", run_name] + _run_sh_args(
                    job.resources, m.run,
                    cold=m.cold and not m.has_attempt),
                m.run_dir, True, [m], ask_in=m.read_from,
                sent=_Sent(time=job.resources.time or None)))
            continue
        # The header is required where it is SENT -- or ASKED about, when
        # the machine has a scheduler to ask: a question about a file that
        # does not exist answers nothing.
        if (_needs_header(mode, machine)
                and not (m.read_from / sbatch_name).exists()):
            raise SubmitError(_no_sbatch(
                f"job {job.name!r}", sbatch_name, base=base, told=told,
                machine=machine,
                redo=(None if jobset.kind == KIND_SWEEP
                      else _redo(calc, job))))
        sent, placement, cmd, outputs_in = _sbatch_request(
            header=m.read_from / sbatch_name,
            placement=_placed_by_prep([job]),
            time_s=time_s, job_name=_scheduler_job_name(jobset, job.name),
            script=sbatch_name,
            run_args=_run_sh_args(job.resources, m.run,
                                  cold=m.cold and not m.has_attempt),
            machine=machine)
        plan.submissions.append(Submission(
            m.name, cmd, m.run_dir, False, [m], placement=placement,
            sent=sent, ask_in=m.read_from, header=m.read_from / sbatch_name,
            machine=machine, outputs_in=outputs_in))
    return plan


__all__ = ["plan_launch", "ask_launch", "send_launch", "LaunchPlan",
           "Submission", "JobResult", "SubmitError"]
