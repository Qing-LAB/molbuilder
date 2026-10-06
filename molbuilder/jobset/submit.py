"""The launch door's engine -- the one entry `launch` runs: its plan, shown
and asked by the verb, then sent as it was shown (`docs/execution/
job-system.md` § 6.0; § 5.3, § 7; the wrapper contract is
`job-contracts.md` § 2).

``prep`` derives the work and lays out the tree; THIS module sends it, in the
order § 6.0 states:

1. **the plan** (:func:`plan_launch`), nothing written -- the work and its
   SUBMISSIONS, each one scheduler job or one process here walking one or
   more MEMBERS, each a prepared attempt: the attempt it runs in, what it
   follows and whether that run concluded; the gates; the queue; the exact
   line and every script the send will write.  Three shapes of work, one
   plan (:class:`LaunchPlan`):

   * **a stage** -- a ladder's stage, or a sweep's named trial.  A stage
     launched before is launched again by
     CONTINUING it (user, 2026-08-21): the next attempt opens from its
     latest, and a run that never concluded is followed only on the
     person's judgement;
   * **a benchmark's walk** -- ONE job per resource shelf of a sweep sent
     to a queue, or its unlaunched trials run here, the trials in sequence
     (`generator.md` § 4.3a, `_bench_walk`);
   * **a bias chain** -- one job walking a transport scan's points;

2. **shown** (:meth:`LaunchPlan.shown`) and 3. **asked**, by the verb --
   ``ask`` mode asks the scheduler about each submission instead
   (:func:`ask_launch`: ``sbatch --test-only`` on the same line), and a dry
   run stops there; neither writes anything;
4. **the send** (:func:`send_launch`), deciding nothing: the folder checked
   against the one the plan was made from, the plan's writes carried out,
   each submission out through ONE function (:func:`_go`), each member's
   ``run.json`` written by the one writer;
5. **the record** -- each submission handed to the verb's ledger the moment
   it goes.

Every door builds its scheduler line through ONE request
(:func:`_sbatch_request`: what prep baked, what was said at launch, admitted
on the queue, every value stated or refused).  The scheduler is handed one
job per invocation -- a grouped bench one per shelf.

*(Each door planned, then planned again to send, writing as it went, until
2026-10-05: nothing compared what was shown with what was sent, and a dry
run wrote ledger lines; W55 B5, D14.  This header described two paths, one
job per invocation, and a module that never inspects prior output, until
2026-10-01 -- each untrue of the body below it; W52.)*
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# The record's one spelling for a wall, written in ONE place.  A second
# formatter here would be a second answer to "what does a walltime look
# like on disk?" -- and there was one, until 2026-08-24, sitting beside a
# field a human spelling could reach.
from ..scheduler.quantities import slurm_time as _slurm_time

# NOTE: `job_dir_names` is NOT imported here.  It is the naming
# authority (materialize.py), and the places THIS module needs it import
# it locally beside the siblings they use.
from .agreement import (DeckLaunchMismatch, check_launch_matches_deck,
                        check_trial_starts_cold)
from .model import Job, JobSet, Resources
from .placement import one_process
from .continuation import Continuation
from .planned import Plan, found
from ..paths import attempt_dir
from ..runfiles import LAUNCH_DIR
from .commands import command as _cmd, rollback


class SubmitError(Exception):
    """A JobSet could not be submitted (bad mode, unknown domain, missing
    prepped wrapper, or sbatch failure)."""


@dataclass
class JobResult:
    """What happened to one job.  ``command`` is the exact line that ran or
    would run -- empty on a line that only reports.  In ``submit`` mode
    ``job_id`` is the SLURM id; in ``direct`` mode ``returncode`` is the
    process exit status.  ``status`` is one of:

    * ``submitted`` / ``ran`` / ``failed`` -- it went;
    * ``planned`` (dry-run) / ``asked`` (``--mode ask``: nothing was sent,
      and ``prediction`` carries what the scheduler said; ``command`` is the
      line that would be SENT) / ``not asked`` (past the query cap);
    * ``sbatch refused`` (this shelf was rejected; the rest still went) /
      ``stays pending`` (its shelf was refused, so it was never sent);
    * ``rides the group`` / ``rides the chain`` -- a trial or a bias point
      launched by its shelf's or its chain's one job;
    * ``already run`` (ask) / ``skipped -- already launched`` (direct) -- a
      trial measured before, passed over by name;
    * a line saying what a re-launch follows -- ``WOULD continue ...`` when
      planned or asked, ``concluded (...): continuing ...`` when sent."""
    name:       str
    command:    List[str]
    status:     str
    job_id:     Optional[str] = None
    returncode: Optional[int] = None
    #: What is known about this shelf's fate.  After a ``sbatch refused``,
    #: the scheduler's own words -- carried rather than raised, because ONE
    #: refused shelf must not cancel the shelves behind it, so the failure
    #: travels back as data (2026-08-30).  In a PREVIEW, what the record
    #: predicts instead: today, that this domain's per-user job cap will
    #: refuse some of this sweep (R14).  One meaning, two moments -- what we
    #: know about whether this will run.
    detail:     Optional[str] = None
    #: Only in ``ask`` mode.  ``None`` everywhere else, and a ``Prediction``
    #: whose ``start`` is ``None`` when SLURM declined to predict -- which
    #: is reported as *unknown*, never as *soon*.
    prediction: Optional[object] = None
    #: WHICH DOMAIN this was placed on, by name (2026-08-30).  The command
    #: carries ``-p`` and ``-q``, and on a cluster where several domains
    #: share one partition those flags do not say which domain was chosen:
    #: Sol's `debug` is (htc, debug) and its `htc` is (htc, public), so a
    #: preview showing ``-p htc`` reads as *htc* to anyone scanning it.
    #: It did, and the person who ran it believed a debug sweep had gone to
    #: the wrong queue.  The name is the fact; the flags are its rendering.
    domain:     Optional[str] = None
    #: What only the person can decide before this goes -- a re-launch over
    #: a run that never concluded (`project-layout.md` § 1.6.4) -- or
    #: ``None``.  The CLI shows it in the one question it asks, and asks
    #: with "no" as the answer Enter gives.
    judgement:  Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------- #
#  domain → -p/-q resolution (reuses runtime_config.get_routing)         #
# --------------------------------------------------------------------- #

def _sbatch_resource_flags(r: Resources, placement=None) -> List[str]:
    """The per-job ``sbatch`` flags — rendered by `scheduler.emit`, the ONE
    emitter (R1).

    Was a second writer beside `runwrap`'s header, each deciding for itself
    what queue and what wall to name.  That split is both Sol failures: a
    header naming ``htc/debug`` while this side asked for 38 minutes, and a
    header naming a queue while stating no wall at all.  Now both spellings
    come off one :class:`~molbuilder.scheduler.emit.Directives`, so there is
    nothing left for them to disagree about.

    CLI flags still WIN over the rendered header, which is what lets one
    ``.sbatch`` serve a whole sweep while each job gets its own ranks.
    """
    from ..scheduler.emit import Directives
    return Directives.of(placement, r).sbatch_flags()


def _run_sh_args(r: Resources) -> List[str]:
    """The per-job knobs for the local ``.run.sh`` (it accepts ``-np`` /
    ``-omp``; runwrap.py § arg-parsing)."""
    args: List[str] = []
    if r.mpi_np:
        args += ["-np", str(r.mpi_np)]
    if r.cpus_per_task:
        args += ["-omp", str(r.cpus_per_task)]
    return args


def _wrapper_name(script: str, suffix: str) -> str:
    """``bdt_stage1.fdf`` + ``.sbatch`` -> ``bdt_stage1.sbatch`` (mirrors
    runwrap's stem + suffix rule)."""
    return Path(script).stem + suffix


# --------------------------------------------------------------------- #
#  the two execution paths                                              #
# --------------------------------------------------------------------- #

def _parse_sbatch_id(stdout: str) -> str:
    """Extract the job id from ``Submitted batch job <id>``."""
    for tok in stdout.split():
        if tok.isdigit():
            return tok
    raise SubmitError(f"could not parse sbatch job id from: {stdout!r}")


def _scheduler_job_name(jobset: JobSet, name: str) -> str:
    """What `squeue` shows: the calculation, then the job within it --
    ``bdt_au/coarse``, ``bdt_au/G1K2C4``, ``bdt_au/bench-group-cpu``
    (`job-system.md` § 6).

    The id first, because that is the thing you are trying to tell apart
    when several calculations are queued at once, and the job second
    because within one calculation that is the question.  EVERY door names
    its job here -- the grouped bench and the bias chain spelled
    ``<calc>_<name>`` beside the stage's ``<calc>/<stage>`` until 2026-10-01
    (W52).
    """
    return f"{jobset.name}/{name}"


#: How many `sbatch --test-only` calls one `--mode ask` will make.
#:
#: Politeness, not a rule about queues: asking enqueues nothing, so the
#: one-at-a-time submission rule does not reach it.  A benchmark grid is
#: the case that matters and is usually a handful of shelves; past this the
#: rest are named as unasked rather than silently dropped.
ASK_MAX_QUERIES = 24

def _no_sbatch(what: str, name: str, *, base) -> str:
    """The one answer to *there is no scheduler header*, whichever door
    finds it missing (`job-system.md` § 6.1): prep withholds the
    ``.sbatch`` where the machine it prepped for names no queue, or where
    it was told to (``prep --no-sbatch``) -- each said with its own way
    back.  Three doors
    worded it three ways until 2026-10-01, one telling a person to "add a
    scheduler block" -- a premise § 6 retired -- and one to "run
    prep_jobset" (W52).  A machine with a queue is another machine, and a
    calculation is set to the machine of its first prep (`configuration.md`
    M-3): the way there is the state saved before that prep."""
    from .commands import takes_a_queue
    if takes_a_queue(base):
        # THE MACHINE NAMES A QUEUE, so prep was told to write no header
        # (`prep --no-sbatch`, `job-system.md` § 6.1's second answer) --
        # this blamed the machine until 2026-10-05 (the unit 11 review).
        return (f"{what}: there is no scheduler header ({name}) -- it was "
                f"prepped with --no-sbatch.  Run it here with --mode "
                f"direct; to send it to a queue, prep it again without "
                f"--no-sbatch -- a prepped stage is not prepped again: "
                + rollback("its prep", base=base))
    return (f"{what}: there is no scheduler header ({name}) -- the "
            f"machine it was prepped for names no queue (its record "
            f"says `workstation`, job-system.md § 6.1).  Run it here with "
            f"--mode direct.  For a machine with a queue, prep it for that "
            f"machine (--target, its record's name) -- a calculation is "
            f"set to the machine of its first prep, so "
            + rollback("its first prep", base=base))


def _sbatch_request(base: Path, *, envelope: Resources,
                    domain: Optional[str], mem: Optional[str],
                    time_s: Optional[int], label: str, job_name: str,
                    script: str, run_args: Sequence[str],
                    one_process: bool = False
                    ) -> Tuple[Resources, object, List[str]]:
    """THE ONE REQUEST, for every door that hands work to the scheduler --
    a stage, a grouped bench's shelf, a bias chain (`job-system.md` § 6).

    ``envelope`` is what `prep` baked; ``mem`` and ``time_s`` are what the
    person said at launch, and win.  The request is ADMITTED against the
    whole of it -- wall, cores, memory, the GPU count -- on the queue
    it is sent to (`scheduler.md` R9: what was admitted when the work was
    built is re-admitted when it is sent, against what this machine says
    now).  **Every value is stated** (`architecture.md` § 5.2): a wall or a
    memory stated nowhere is refused here, never the queue's ceiling or
    the scheduler's default -- prep refuses it first, and this is the door
    that would send it.

    Returns ``(envelope, placement, command)``: the envelope as sent, the
    `Placement` (``None`` on a machine with no menu -- the rendered header's
    own directives stand, R6), and the exact ``sbatch`` line, its resources
    as flags that win over the header.  The stage's door built the queue
    half of this alone until 2026-10-01: it sent the launch queue's
    ``-p/-q`` under the wall prep's header had worked out for another queue,
    and admitted nothing (W52).

    ``run_args`` are the run script's own ``-np`` / ``-omp``
    (`_run_sh_args`), set after the ``.sbatch`` name, which forwards them
    (``bash <base>.run.sh "$@"``): every launch hands every run its own
    counts (`job-system.md` § 6.1).  A submission whose script walks several
    members -- a benchmark's shelf, a bias scan -- hands each member its own
    inside that script, and passes none here.
    """
    from ..scheduler.quantities import parse_walltime
    if mem:
        envelope = dataclasses.replace(envelope, mem=mem)
    # What admission is asked to fit: the wall stated at launch, else the one
    # prep baked.  Unstated is None -- an unstated limit never bars (R3) --
    # and is refused below, once the request has a queue.
    needed_s = time_s
    if needed_s is None and envelope.time:
        try:
            needed_s = parse_walltime(str(envelope.time))
        except ValueError:
            raise SubmitError(
                f"prep baked time={envelope.time!r}, which does not parse "
                f"as a SLURM walltime.")
    # THE GPU REQUEST is the envelope's own -- whether, and how many (the
    # one door, `model.gpu_request`), never a side a caller passes beside it.
    gpus = _gpus(envelope, label)
    # THE ONE REQUEST a queue is asked, prep's and launch's alike
    # (`placement.request_of`): the envelope as it is sent -- its wall the
    # one stated at launch, else prep's.  Launch built its own until
    # 2026-10-05, and counted no cores for a PySCF run (``one_process``).
    from .placement import request_of
    asked = (dataclasses.replace(envelope, time=_slurm_time(needed_s))
             if needed_s is not None else envelope)
    placement = _place(base, request_of(asked, one_process=one_process),
                       gpu_side=gpus.uses, named=domain, label=label)
    # THE WALL: what was stated at launch, else what prep baked.  The target
    # queue's own ceiling stood in for neither until 2026-10-02 -- a wall
    # nobody stated -- and the scheduler's default would stand in now if
    # this let an unstated one through.
    if time_s is not None:
        envelope = _dc_replace_time(envelope, _slurm_time(time_s))
    # EVERY VALUE IS STATED, asked of the request AS SENT by the one answer
    # prep gives (`placement.launch_refusal`): a launch flag may state
    # what the description did not, and a prep with `--no-sbatch` asked
    # none of the three.  The envelope names the queue it is sent to.
    # Asked only where the request goes to a QUEUE: a machine with no menu
    # promised nothing (R6), and `ask` there must answer "no scheduler
    # here", as prep asks no wall of a run that writes no header.
    if placement is not None:
        from .. import runtime_config as _rc
        from .placement import launch_refusal
        envelope = dataclasses.replace(envelope, domain=placement.domain.name)
        why = launch_refusal(
            envelope, engine=None, header=True, shape=False,
            queues=[d.name for d in _rc.get_routing(project_dir=base)],
            at_launch=True)
        if why:
            raise SubmitError(f"{label or 'this job'} {why}")
    cmd = (["sbatch", "-J", job_name]
           + _sbatch_resource_flags(envelope, placement)
           # The launch-door claim, EXPLICIT on the command line: environment
           # inheritance alone is fragile (sites override SLURM's --export
           # policy), and the flag wins over site defaults, so the claim
           # reaches the job wherever it runs (job-contracts.md § 2.6).
           + ["--export", "ALL,MB_LAUNCHED_BY=jobset-launch", script]
           + list(run_args))
    return envelope, placement, cmd


def _ask(name: str, cmd: List[str], cwd: Path, *,
         domain: Optional[str] = None,
         script: Optional[str] = None) -> JobResult:
    """``sbatch --test-only`` on the line that WOULD be sent -- the scheduler
    answers instead of enqueueing.  Writes and records nothing.

    The result carries the line that would be SENT; ``--test-only`` goes
    first only in what is run, so it cannot be shadowed and is never shown
    as the line to send.  ``script`` stands in for the line's own when that
    one is written only at sending -- a grouped bench's shelf, a bias chain
    -- and is a header rendered for the same work: the flags, which win
    over any header, are the request asked about.  No scheduler here is an
    answer, not an error: on a workstation there is nothing to queue behind.
    """
    from .ask import Prediction, parse_test_only
    if shutil.which(cmd[0]) is None:
        return JobResult(name, cmd, "asked", domain=domain,
                         prediction=Prediction(no_scheduler=True,
                                               refused="no scheduler here"))
    try:
        cp = subprocess.run([cmd[0], "--test-only"] + cmd[1:-1]
                            + [script or cmd[-1]], cwd=str(cwd),
                            capture_output=True, text=True)
    except OSError as exc:
        # "I could not reach the scheduler" is what the person asked about.
        return JobResult(name, cmd, "asked", domain=domain,
                         prediction=Prediction(
                             refused=f"could not run {cmd[0]!r}: {exc}"))
    # A non-zero return is an answer too -- "this queue cannot take it" is
    # often the one worth reading.
    return JobResult(name, cmd, "asked", domain=domain,
                     prediction=parse_test_only((cp.stdout or "")
                                                + (cp.stderr or "")))


def _into_launch(header: str, name: str) -> str:
    """A grouped bench's or a bias chain's header, pointed into ``launch/``
    (L3): the delegated script and SLURM's own output.  COUNT-ASSERTED, so a
    change in the emitter's spelling fails here rather than scattering files
    among the runs -- one helper for both doors, each of which spelled it,
    one without the check (W52)."""
    for old, new in ((f"bash {name}.run.sh", f"bash launch/{name}.run.sh"),
                     ("#SBATCH -o slurm.%j.out",
                      "#SBATCH -o launch/slurm.%j.out"),
                     ("#SBATCH -e slurm.%j.err",
                      "#SBATCH -e launch/slurm.%j.err")):
        if header.count(old) != 1:
            raise SubmitError(
                f"the sbatch header no longer spells {old!r} exactly "
                f"once; the launch/ repointing needs updating.")
        header = header.replace(old, new)
    return header


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


@dataclass
class _Member:
    """One prepared attempt a submission runs, decided BEFORE anything is
    written (`job-system.md` § 6.0, step 1) -- a run is one member, a
    benchmark's shelf its pending trials, a bias scan its points: where it
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
    #: Where its deck and wrappers are read before it is sent: ``run_dir``,
    #: or for a continuation the attempt it continues -- the new one is
    #: filled with the same files (`materialize.prepare_attempt`).
    read_from: Path
    #: The set's kind, which says where its launch is recorded
    #: (`materialize.launch_record_at`).
    kind: str = "ladder"
    #: What the results and the ledger call it when the job's name does not
    #: -- a bias point's ``<stage>@<point>``.
    label: Optional[str] = None
    #: What it continues from when it is a stage launched again -- its own
    #: latest run, one `Continuation` from the one door
    #: (`continuation.relaunch`; `job-system.md` § 5.4) -- or ``None``.
    continuation: Optional[Continuation] = None
    #: The restart files that continuation copies, as the opener planned
    #: them -- what that run holds of what the stage declares.
    carries: List[str] = field(default_factory=list)
    #: The inputs a transport rung's kind gathered for that run, copied with
    #: their record (`.gathered-from`) -- never gathered again.
    gathered: List[str] = field(default_factory=list)
    #: The calculation's folder, so what the plan says names it.
    base: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.label or self.job.name

    @property
    def follows(self) -> bool:
        """It is a stage launched again, following a run of its own."""
        return self.continuation is not None

    def record_at(self) -> Tuple[Path, Optional[str]]:
        """Where its launch is recorded -- the one answer
        (`materialize.launch_record_at`)."""
        from .materialize import launch_record_at
        return launch_record_at(self.kind, self.job, self.container,
                                self.run_dir if self.has_attempt else None)

    def _where(self) -> str:
        """Where it runs again: its next attempt, or its folder."""
        return (f"into {self.run_dir.name}" if self.has_attempt
                else "in the same folder, where its files are")

    def _copied(self) -> List[str]:
        return list(self.carries) + list(self.gathered)

    def said(self) -> str:
        """The member as the plan holds it, in one line: where it runs, what
        it follows, how that run ended and what is copied from it."""
        line = f"{self.name}: runs in {_rel(self.run_dir, self.base)}"
        if self.continuation is not None:
            line += "; it " + self.continuation.line(self._copied(),
                                                     would=True)
        return line

    def judgement(self) -> Optional[str]:
        """What only the person can decide before this goes, or ``None``."""
        c = self.continuation
        if c is None or c.concluded is not None:
            return None
        return (f"{self.name}: {c.where()} was launched and never "
                f"CONCLUDED -- it may still be RUNNING, or it was "
                f"force-stopped (walltime, kill).\n"
                f"  Continuing reads its warm files AS THEY ARE: valid after "
                f"a forced stop, torn if it is still running.  Check the "
                f"queue, and `{_cmd('status', self.job.name, base=self.base)}`, "
                f"first.")

    def would(self) -> str:
        """What it follows, said before anything is sent."""
        return (f"WOULD launch it again {self._where()}: it "
                + self.continuation.line(self._copied(), would=True))

    def note(self) -> Optional[str]:
        """The line that says what this launch follows, once it is sent, or
        ``None``."""
        c = self.continuation
        if c is None:
            return None
        judged = ("" if c.concluded is not None else
                  " -- on your judgement: that run never concluded")
        return (f"launched again {self._where()}{judged}: it "
                + c.line(self._copied()) + ".  To start it afresh instead "
                "-- from the stage before it, or the structure: "
                + rollback("its prep", base=self.base))


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
    #: Run here (``bash``, waited for), or handed to the scheduler.
    direct: bool
    members: List[_Member]
    #: The `Placement` it was admitted on -- ``None`` for a run here, or on a
    #: machine with no menu (R6).
    placement: object = None
    #: The request as sent -- the job's resources under the launch's
    #: ``--time`` / ``--mem`` (:func:`_sbatch_request`): the wall and memory
    #: its ``run.json`` records beside the queue (`job-system.md` § 6.0).
    sent: Optional[Resources] = None
    #: What each member says in the results when the submission walks
    #: several (``rides the group``, ``rides the chain``); ``None`` for one
    #: run, which is the submission itself.
    rides: Optional[str] = None
    #: Where `ask` asks about it, and the header it asks over when the
    #: line's own is written only at the send -- a shelf's, a chain's.
    ask_in: Optional[Path] = None
    ask_script: Optional[str] = None
    #: What a refusal by the scheduler adds to the scheduler's own words.
    refusal_hint: str = ""
    #: Where a walk's trials write their output -- the one path its script
    #: writes and the plan shows (:func:`_bench_walk`).
    log: Optional[Path] = None

    @property
    def domain(self) -> Optional[str]:
        """The NAME of the queue it was placed on (:func:`_domain_name`)."""
        return _domain_name(self)


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
    #: one opener), a flat re-launch's marker, a shelf's or a chain's
    #: scripts.
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
    #: The queue the work goes to and where that came from -- --domain,
    #: or the queue its prep recorded (:func:`_the_queue`); ``(None,
    #: None)`` for a run here or a machine with no queues.
    queue: Tuple[Optional[str], Optional[str]] = (None, None)

    def record(self, decision: str, **facts) -> None:
        """One of this launch's decisions, written down (§ 6.0, step 5;
        :func:`_record`)."""
        if self.records:
            _record(self.base, self.told, decision, **facts)

    def made_from(self) -> List[str]:
        """What this plan was made from, line by line -- each submission's
        line and where it runs, each member and what it follows, every file
        the send writes or copies, every file read where it lies.  The send
        makes the plan again and compares (§ 6.0, step 4)."""
        out = [f"{r.name}: {r.status}" for r in self.skipped]
        for s in self.submissions:
            out.append(f"{s.name}: {' '.join(s.command)} "
                       f"(in {_rel(s.cwd, self.base)})")
            out += [m.said() for m in s.members]
        out += [line.replace(f"{self.base}{os.sep}", "")
                for line in self.writes.described()]
        return out + list(self.reads)

    def judgements(self) -> List[str]:
        """What only the person can decide before this goes."""
        return [j for s in self.submissions for m in s.members
                for j in (m.judgement(),) if j]

    def shown(self) -> List[JobResult]:
        """The plan as the question and a dry run show it: the trials passed
        over, what each member follows and what only the person can judge,
        each submission's exact line and queue -- with what the record
        predicts of its cap (R14) -- and the members riding it."""
        caps: Dict[str, str] = {}
        for n in submitted_cap_notes(self.submissions):
            caps[n.split(" takes ", 1)[0]] = n
        out: List[JobResult] = list(self.skipped)
        for s in self.submissions:
            out += [JobResult(m.name, [], m.would(), judgement=m.judgement())
                    for m in s.members if m.follows]
            out.append(JobResult(s.name, s.command, "planned",
                                 domain=s.domain, detail=caps.get(s.domain)))
            if s.rides:
                out += [JobResult(m.name, [], s.rides) for m in s.members]
        return out


def _launched(where, basename: Optional[str] = None) -> bool:
    """Was this run launched -- the one door's answer
    (`runrecord.launch_record`), every gate of launch asking it.  A record
    that does not read is a refusal naming the file: launching over it could
    send a job twice, and not launching could hide one in the queue."""
    from ..runrecord import LaunchRecordError, launch_record
    try:
        return launch_record(where, basename) is not None
    except LaunchRecordError as e:
        raise SubmitError(str(e)) from e


def _plan_member(jobset: JobSet, base: Path, job, *, mode: str,
                 writes: Plan, named: bool = False):
    """Where ``job`` runs and what it follows -- read, never written: a
    :class:`_Member`, or the result of a trial passed over by name.  A
    re-launch's next attempt is opened in ``writes`` by the one opener
    (`materialize.prepare_attempt`), and a flat re-launch's marker written
    there -- the send carries both out.

    THE SHAPE DECIDES, NOT THE KIND.  A hierarchical run -- ladder stage or
    sweep trial alike -- runs in ``run-<n>/``, because an attempt is
    immutable once it has run (`project-layout.md` § 1.5); a flat one keeps
    no attempt directories (§ 1.5a) and runs in its own container.  Whether
    it was launched is its launch record's answer -- ``run.json``, or a flat
    stage's ``<basename>.run.json`` (`materialize.launch_record_at`): a
    queued job has produced nothing yet, so absence of output proves
    nothing (§ 1.6).

    A LADDER STAGE LAUNCHED BEFORE is launched again by continuing it (user,
    2026-08-21: *"a run stopped due to the server running out of time, and
    you can submit again and by default it continues"*): the hierarchy
    opens the next attempt from the latest -- the one source that is never a
    guess -- and the flat layout simply runs again where its files are.  A
    run that never concluded is followed only on the person's judgement.
    A TRIAL is immutable once launched: a walk -- here, or a queue's
    shelves -- and a question to the scheduler pass the measured ones over
    by name; one the person NAMED (``named``) is refused when it is sent or
    run here -- a run here passed it over, exit 0, until 2026-10-05 (the
    unit 11 review) -- and a question to the scheduler says it already
    ran.
    """
    import functools
    from .commands import command
    from ..paths import attempts_in
    from .materialize import (bench_stage_of, job_dir_names, launch_record_at,
                              prepare_attempt, shape_of)
    from ..runrecord import launch_record_path, write_continued_from
    _member = functools.partial(_Member, kind=jobset.kind, base=base)
    sh = shape_of(jobset, base)
    container = base / job_dir_names(jobset, sh)[job.name]
    ns = attempts_in(container)
    if jobset.kind == "sweep":
        run = _trial_run_dir(container)
        where, basename = launch_record_at("sweep", job, container,
                                           run if ns else None)
        if not _launched(where, basename):
            return _member(job, container, run, bool(ns), run)
        if mode == "ask" or not named:
            return JobResult(job.name, [],
                             "already run" if mode == "ask"
                             else "skipped -- already launched")
        stage = bench_stage_of(base, container)
        read_back = (command("summarize", "bench", stage, base=base)
                     if stage else "`summarize bench` on the sweep's stage")
        if not ns:
            raise SubmitError(
                f"trial {job.name!r}: already launched -- "
                f"{launch_record_path(where, basename)} records it.  A "
                f"trial measures its point ONCE (project-layout.md § 1.5: "
                f"immutable once it has run); read the sweep back with "
                f"{read_back}.  To measure it again -- a prepped benchmark is "
                f"not prepped again (job-system.md § 5.0): "
                + rollback("the benchmark's prep", base=base))
        raise SubmitError(
            f"trial {job.name!r}: {run.name} has already been launched "
            f"({launch_record_path(where, basename)}).  A measurement is "
            f"immutable once it has run.\n"
            f"  read what it measured:\n    {read_back}\n"
            f"  measure it again -- a prepped benchmark is not prepped again "
            f"(job-system.md § 5.0): "
            + rollback("the benchmark's prep", base=base))
    if not ns:
        if sh is not None and sh.keeps_attempts_as_directories:
            # A HIERARCHICAL stage with no attempt open would launch in its
            # own container, write no run.json, and be silently
            # relaunchable -- everything § 1.5/1.6 exist to prevent.
            raise SubmitError(
                f"job {job.name!r}: no attempt is open under "
                f"{container.name}/ -- a hierarchical stage runs in run-<n>, "
                f"never in its own container (project-layout.md § 1.5, "
                f"1.6), and a prepped stage is not prepped again: "
                + rollback("its prep", base=base))
        where, basename = launch_record_at("ladder", job, container, None)
        if not _launched(where, basename):
            return _member(job, container, container, False, container)
        # A FLAT STAGE LAUNCHED AGAIN continues from its own latest run,
        # where its files are, and its record says so -- as the hierarchy's
        # next attempt names its own; one that does not continue from a run
        # of its own is refused (the one door, `continuation.relaunch`).
        # The marker prep left names what the stage's FIRST launch continued
        # from, the stage before it, and was recorded again (W55 D5).
        cont = _relaunched(base, job)
        if cont.run is not None:
            write_continued_from(where, cont.run, basename=basename,
                                 plan=writes)
        return _member(job, container, container, False, container,
                       continuation=cont)
    last = attempt_dir(container, ns[-1])
    if not _launched(last):
        return _member(job, container, last, True, last)
    cont = _relaunched(base, job)
    source = cont.source
    try:
        # THE ONE OPENER, planned (`project-layout.md` § 1.6.2): the next
        # attempt, the stage's files and what it carries from ``source`` --
        # written by the send, never before the person has said yes.
        opened = prepare_attempt(jobset, base, job.name,
                                 continue_from=source, named=False,
                                 plan=writes, shape=sh)
    except ValueError as e:
        # Continuing is impossible -- no state to carry, or the stage's deck
        # would not read it.  Both are SIGNALS (a launched run that left
        # nothing likely died at startup), so the door refuses with the
        # story rather than silently starting fresh.
        raise SubmitError(
            f"{job.name}: {source} was launched, so launching it again "
            f"continues from it -- but that is impossible here:\n  {e}\n"
            f"  Look at that run's logs.  To run the stage anew -- a "
            f"prepped stage is not prepped again (job-system.md § 5.0) -- "
            + rollback("its prep", base=base)) from e
    return _member(job, container, opened.dir, True, last, continuation=cont,
                   carries=list(opened.copied),
                   gathered=_carry_the_gather(last, opened.dir, writes,
                                              base=base))


def _relaunched(base: Path, job) -> Continuation:
    """What ``job``'s stage, launched again, continues from -- the one door
    (`continuation.relaunch`), asked with the calculation's description;
    its refusal, a stage that does not continue from a run of its own, is
    the launch's (`job-system.md` § 5.4, *A stage launched again*)."""
    from ..task import FILENAME, read_task
    from .continuation import relaunch
    desc = base / FILENAME
    try:
        task = read_task(desc)
    except Exception as exc:                                  # noqa: BLE001
        raise SubmitError(
            f"{job.name} was launched, and launching it again reads what it "
            f"continues from through its description -- {desc}: "
            f"{exc}") from None
    cont, why = relaunch(base, task, job)
    if why:
        raise SubmitError(why)
    return cont


def _carry_the_gather(source: Path, attempt: Path, writes: Plan, *,
                      base: Path) -> List[str]:
    """A transport rung launched again takes the inputs its kind gathered
    for ``source`` -- each file, copied from it into ``attempt``, with the
    record (`.gathered-from`) that says where each came from: what a stage
    builds on is decided once, at its prep, and never gathered again
    (`job-system.md` § 5.4).  ``[]`` for a run that gathered nothing.
    *(Until 2026-10-05 the next attempt of a single-bias device was opened
    without its electrodes' `.TSHS`.)*"""
    from ..runrecord import read_gathered_from, write_gathered_from
    took = read_gathered_from(source)
    for g in took:
        f = source / g["file"]
        if not f.is_file():
            raise SubmitError(
                f"{_rel(source, base)}: {g['file']}, gathered for it from "
                f"{g['from']}, is not there -- launched again, the stage "
                f"takes the inputs gathered for its run, and they are gone.  "
                f"A prepped stage is not prepped again: "
                + rollback("its prep", base=base))
        writes.copy(f, attempt / g["file"])
    if took:
        write_gathered_from(attempt, [(g["from"], g["file"]) for g in took],
                            plan=writes)
    return [g["file"] for g in took]


def _trial_run_dir(container):
    """**Where a trial's files are** — the attempt, or the trial itself.

    ONE ANSWER TO ONE QUESTION.  Since 2026-08-27 a trial runs in its
    attempt when the shape keeps one (`project-layout.md` § 1.5a: *"a sweep
    trial keeps attempts exactly as a stage does, and the SHAPE decides
    how"*), and § 1.6 puts everything it needs there -- the deck, the
    wrapper, the monitor.  **Flat keeps no attempt directories**, so the
    same call answers the container, and neither caller needs to know which
    shape it is looking at.

    The rule itself lives in the layout layer (`materialize.run_dir`);
    this is `submit`'s name for it, kept because the questions asked of it
    read better against a trial-shaped word than a generic one.
    """
    from .materialize import run_dir
    return run_dir(container)


def _gpus(resources, what: str):
    """A job's GPU request -- `model.gpu_request`, the one door every reader
    asks (`execution/architecture.md` § 3.2) -- or a `SubmitError` naming
    ``what`` when it disagrees with itself, which a job prep wrote never
    does (prep refuses it first).  *(`_job_wants_gpu` stood here until
    2026-10-03: a job was a GPU job by its count OR by `use_gpu`, else by a
    scan of its deck.)*"""
    from .model import GpuRequestError, gpu_request
    try:
        return gpu_request(resources)
    except GpuRequestError as exc:
        raise SubmitError(f"{what}: {exc}") from None


def _the_queue(jobs) -> Tuple[Optional[str], Optional[str]]:
    """``(queue, its source)`` the work's prep recorded -- a run's
    placement, which prep admitted; a trial's resources, the queue its
    prep was told (`job-system.md` § 6.0, the placement) -- or ``(None,
    None)`` when it recorded none.  Several are
    refused: one launch goes to one queue (a benchmark's GPU side names its
    own, ``--gpu-domain``).  *(The verb worked this out itself until
    2026-10-05: floor 7, which never works out a launch, `architecture.md`
    § 2.1.)*"""
    named = {(j.placement or {}).get("domain") or j.resources.domain
             for j in jobs} - {None, ""}
    # ADMITTED only where prep admitted it -- a run's placement; a
    # benchmark's trials carry the queue their prep was told, which is
    # admitted at this launch (`job-system.md` § 6.0, the placement)
    admitted = any((j.placement or {}).get("domain") for j in jobs)
    if len(named) > 1:
        raise SubmitError(
            f"the work being sent names more than one domain "
            f"({', '.join(sorted(named))}).  Name the one to use with "
            f"--domain, and --gpu-domain if a benchmark's GPU side differs.")
    if named:
        return named.pop(), ("its prep (the queue it admitted)" if admitted
                             else "its prep (the queue it named)")
    return None, None


def _no_queue_named(base: Path, jobs, *, mem: Optional[str],
                    time_s: Optional[int], one_proc: bool,
                    gpu: bool) -> Optional[str]:
    """The refusal for work sent to a queue that is named nowhere -- no
    ``--domain``, and none recorded at its prep -- with this machine's
    queues listed against what the work asks (the request the door admits,
    `placement.request_of`), so one can be named.  ``None`` on a machine
    with no queues, where the door's own answer stands (`scheduler.md`
    R6)."""
    from ..runtime_config import get_routing
    rows = get_routing(project_dir=base)
    if not rows:
        return None
    from ..scheduler import parse_mem_gb
    from ..scheduler.quantities import parse_walltime
    from .ask import Ask, queue_table
    from .placement import request_of
    cores = max((request_of(j.resources, one_process=one_proc).cores or 0
                 for j in jobs), default=0) or None
    gpus = (max((_gpus(j.resources, f"job {j.name!r}").count or 0
                 for j in jobs), default=0) or None) if gpu else None

    def most(read, field):
        got = []
        for j in jobs:
            v = getattr(j.resources, field, None)
            try:
                got.append(read(str(v)) if v else None)
            except ValueError:            # the door refuses it, by name
                got.append(None)
        return max((g for g in got if g is not None), default=None)

    wall = time_s if time_s is not None else most(parse_walltime, "time")
    gb = parse_mem_gb(mem) if mem else most(parse_mem_gb, "mem")
    return (queue_table(rows, Ask(time_s=wall, mem_gb=gb), cores=cores,
                        gpus=gpus)
            + "\nno --domain was given, and its prep recorded no queue: "
              "name one from the list above with `--domain`.")


# --------------------------------------------------------------------- #
#  the entry: plan, ask, send (`job-system.md` § 6.0)                   #
# --------------------------------------------------------------------- #

def _record(base: Path, told, decision: str, **facts) -> None:
    """A launch's decision in the calculation's ledger -- every refusal, the
    question and its answer, each submission as it goes -- written by the
    entry, whichever door called (`execution/architecture.md` § 2.1, floor
    7: `prep`'s and `launch`'s entries append their own); in a described
    calculation only, as prep's: a folder that is not one gets no ledger of
    ours."""
    from ..task import FILENAME
    from .ledger import record
    if (Path(base) / FILENAME).is_file():
        record(base, "launch", decision, **{**dict(told or {}), **facts})


def plan_launch(jobset: JobSet, base_dir, *, mode: str,
                only: Optional[str] = None,
                domain: Optional[str] = None,
                gpu_domain: Optional[str] = None,
                side: Optional[str] = None,
                mem: Optional[str] = None,
                time_s: Optional[int] = None,
                trial_timeout_s: Optional[int] = None,
                told: Optional[Dict[str, object]] = None,
                record: bool = True) -> LaunchPlan:
    """STEP 1 of `job-system.md` § 6.0 -- the whole launch of a prepped
    ``jobset`` rooted at ``base_dir``, planned with nothing written: the
    work, its submissions and their members, the gates, the queue, the exact
    lines and every script the send will write.  A refusal can only come
    from here.

    ``mode`` is ``"submit"`` (``sbatch``, its resources as flags over the
    rendered header), ``"ask"`` (the same line, asked with ``--test-only``)
    or ``"direct"`` (``bash`` here, in order, waiting for each).  ``only``
    names ONE job -- a ladder's stage (the only case: stages do not chain,
    `project-layout.md` § 1.6) or a sweep's trial -- by its name or ``#N``.
    ``domain`` names the queue, and ``gpu_domain`` the one a sweep's GPU
    side goes to when it differs; ``mem`` -- SLURM text, ``0`` for the whole
    node -- and ``time_s`` are what the person said at launch; ``side``
    (``"cpu"`` / ``"gpu"``) and ``trial_timeout_s`` are a grouped bench's.
    ``told`` is what the verb was told, written on each of the launch's
    lines in the ledger; ``record`` is false for a dry run, which writes
    nothing (§ 6.0, step 3) -- a refusal is written down otherwise.

    THE WORK IS READ OFF WHAT IS LAUNCHED.  A sweep with no trial named,
    sent to (or asked of) a scheduler, goes as ONE job per resource shelf
    (`generator.md` § 4.3a) -- never as one job per trial, which would hand
    the scheduler a sweep at once: *"SLURM should never submit jobs in
    parallel.  Submission is manual and one by one"* (user, 2026-08-10); a
    sweep run here walks its unlaunched trials in one submission, as a
    shelf does on a queue; a transport scan's per-point rung goes as one
    job walking its points; a stage and a named trial, one submission each.
    """
    base = Path(base_dir).resolve()
    try:
        plan = _planned(jobset, base, mode=mode, only=only, domain=domain,
                        gpu_domain=gpu_domain, side=side, mem=mem,
                        time_s=time_s, trial_timeout_s=trial_timeout_s,
                        told=told)
    except SubmitError as exc:
        if record:
            _record(base, told, "refused", reason=str(exc))
        raise
    plan.told, plan.records = dict(told or {}), record
    # the queue's source is the entry's decision, on every line
    plan.told.update(domain=plan.queue[0], domain_source=plan.queue[1])
    return plan


def _planned(jobset: JobSet, base: Path, *, mode, only, domain, gpu_domain,
             side, mem, time_s, trial_timeout_s, told) -> LaunchPlan:
    """:func:`plan_launch`'s body -- the plan, or the refusal it records."""
    errs = jobset.validate()
    if errs:
        raise SubmitError(
            "refusing to submit an invalid JobSet:\n  - " + "\n  - ".join(errs))
    if not base.is_dir():
        raise SubmitError(f"base dir not found (prep first): {base}")
    if mode not in ("submit", "ask", "direct"):
        raise SubmitError(
            f"unknown mode {mode!r}: must be 'submit' (SLURM), 'ask' (submit "
            f"nothing, report when it would start) or 'direct' (local)")
    if only is not None:
        # Through the ONE resolver, so a name and a #N number reach the same
        # job here as at every other surface, and the refusal carries the
        # typeable spellings.
        from ..identity import resolve_stage_ref
        from .materialize import stage_refs
        refs = stage_refs(jobset)
        try:
            only = resolve_stage_ref([refs[j.name] for j in jobset.jobs],
                                     only).name
        except ValueError as e:
            raise SubmitError(str(e))

    def again() -> LaunchPlan:
        return plan_launch(jobset, base, mode=mode, only=only, domain=told_q,
                           gpu_domain=gpu_domain, side=side, mem=mem,
                           time_s=time_s, trial_timeout_s=trial_timeout_s,
                           told=told, record=False)

    # THE QUEUE, decided here and nowhere else (`job-system.md` § 6.0, the
    # placement): --domain when typed, else the one the work's prep
    # recorded.  Each door refuses a queue named nowhere at its own
    # moment -- a stage after its header is found.
    told_q = domain
    source = "--domain flag" if domain else None
    if mode in ("submit", "ask") and not domain:
        domain, source = _the_queue(
            [j for j in jobset.jobs if only is None or j.name == only])

    if jobset.kind == "sweep" and only is None and mode in ("submit", "ask"):
        plan = _plan_shelves(jobset, base, mode=mode, domain=domain,
                             gpu_domain=gpu_domain, side=side, mem=mem,
                             time_s=time_s, trial_timeout_s=trial_timeout_s)
    elif jobset.kind == "sweep" and only is None:
        plan = _plan_bench_here(jobset, base,
                                trial_timeout_s=trial_timeout_s)
    else:
        task = _a_scan(jobset, base, only) if only is not None else None
        if task is not None and mode in ("submit", "ask") and not domain:
            why = _no_queue_named(
                base, [j for j in jobset.jobs if j.name == only], mem=mem,
                time_s=time_s, one_proc=one_process(jobset.engine),
                gpu=False)
            if why:
                raise SubmitError(why)
        plan = (_plan_chain(jobset, base, task, mode=mode, stage=only,
                            domain=domain, mem=mem, time_s=time_s)
                if task is not None else
                _plan_stage(jobset, base, mode=mode, domain=domain,
                            gpu_domain=gpu_domain, only=only, mem=mem,
                            time_s=time_s))
    plan.remake = again
    plan.queue = (domain, source)
    return plan


def _a_scan(jobset: JobSet, base: Path, stage: str):
    """The description, when ``stage`` is a transport rung a bias scan runs
    once per point (`transport.stages.scan_points`) -- launched as ONE job
    walking the points -- else ``None``."""
    if jobset.kind != "ladder":
        return None
    from ..task import FILENAME, read_task
    from ..transport.stages import scan_points
    desc = base / FILENAME
    if not desc.is_file():
        return None                       # a hand-built ladder: no scan
    try:
        task = read_task(desc)
    except Exception as exc:                                  # noqa: BLE001
        raise SubmitError(f"{desc}: {exc}") from None
    if task.calculation != "transport" or not scan_points(task, stage):
        return None
    return task


def ask_launch(plan: LaunchPlan) -> List[JobResult]:
    """``--mode ask`` (`submission.md` S4): the scheduler asked about each
    submission -- ``sbatch --test-only`` on the very line the send would
    hand it -- in place of the question; nothing is written or sent.  Past
    :data:`ASK_MAX_QUERIES` the rest are named, never dropped."""
    out: List[JobResult] = list(plan.skipped)
    for n, s in enumerate(plan.submissions):
        out += [JobResult(m.name, [], m.would(), judgement=m.judgement())
                for m in s.members if m.follows]
        if n >= ASK_MAX_QUERIES:
            out.append(JobResult(s.name, [], "not asked"))
            continue
        out.append(_ask(s.name, s.command, s.ask_in or s.cwd,
                        domain=s.domain, script=s.ask_script))
    plan.record("asked", jobs=[{"job": r.name, "status": r.status}
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
    launch is refused, written down (`project-layout.md` § 1.6.4).  Then
    the folder is checked against the one the plan was made from
    (:func:`_same_folder`); the plan's writes are carried out -- every
    attempt opened and every script on disk before anything goes, so a
    scheduler's refusal costs one submission, never the scripts of those
    behind it -- and each submission goes out through ONE function
    (:func:`_go`), written down the moment it goes.  A shelf the scheduler
    refuses leaves the rest to go (``plan.tolerant``): its trials keep no
    launch record, so the next launch picks up exactly them.  Every refusal
    is written down."""
    # WHAT WAS ASKED, as asked: the question was keyed on the mode until
    # 2026-10-06, so every run here read "follow a run that never
    # concluded" once a run here was asked at all (the unit 11 review).
    plan.record("question",
                about=(("submit" if plan.mode == "submit" else "run here")
                       + ("; follows a run that never concluded"
                          if plan.judgements() else "")),
                answer=said.words)
    if not said.asked:
        why = ("not a terminal, so there is nobody to ask -- nothing was "
               "sent.  Pass --yes to go ahead with what is printed above "
               "without being asked.")
        plan.record("refused", reason=why)
        raise SubmitError(why)
    if not said:
        return []
    try:
        _same_folder(plan)
    except SubmitError as exc:
        plan.record("refused", reason=str(exc))
        raise
    plan.writes.carry_out()
    # WHAT EACH STAGE LAUNCHED AGAIN CONTINUES FROM, recorded as prep's
    # hand-over is (`continues`; `job-system.md` § 5.4) -- its attempt
    # opened, what came across.
    for s in plan.submissions:
        for m in s.members:
            if m.continuation is not None:
                plan.record("continues", stage=m.name,
                            **m.continuation.ledger_facts(),
                            copied=list(m.carries),
                            **({"gathered": list(m.gathered)}
                               if m.gathered else {}))
    results: List[JobResult] = list(plan.skipped)
    refused: List[str] = []
    for s in plan.submissions:
        results += [JobResult(m.name, [], m.note())
                    for m in s.members if m.follows]
        try:
            results += _go(s, plan.record)
        except SubmitError as exc:
            plan.record("refused", submission=s.name, reason=str(exc))
            if not plan.tolerant:
                raise
            refused.append(str(exc))
            results.append(JobResult(s.name, s.command, "sbatch refused",
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


def _go(s: Submission, record) -> List[JobResult]:
    """THE ONE SENDER (`job-system.md` § 6.0, step 4) -- a stage, a shelf,
    a chain, here or to the scheduler: the line run where the plan said,
    each member's launch recorded by the one writer (:func:`_record_launch`),
    and the ledger told the moment it goes -- a run here when it STARTS, a
    scheduler job when it is given its id.  *(Four places sent until
    2026-10-05, each with its own record and its own wording of a
    refusal.)*"""
    env = {**os.environ, "MB_LAUNCHED_BY": "jobset-launch"}
    names = [m.name for m in s.members]
    if s.direct:
        # The launch-door claim rides the child ENV here: inheritance
        # survives forks and backgrounding, so a detached local run
        # launched through this verb never meets the gate's prompt.
        try:
            proc = subprocess.Popen(s.command, cwd=str(s.cwd), env=env)
        except OSError as exc:
            raise SubmitError(
                f"{s.name}: could not run {s.command[0]!r} ({exc})") from None
        # AT START, after the process exists: run.json answers "was this
        # launched?", and a record written on completion left a running
        # attempt reading as never launched for its whole runtime.  A failed
        # START records nothing -- Popen raising means no process exists.
        for m in s.members:
            where, basename = m.record_at()
            _record_launch(where, mode="direct", command=s.command,
                           basename=basename)
        record("launched", submission=s.name, command=s.command,
               members=names)
        rc = proc.wait()
        return ([JobResult(s.name, s.command, "ran" if rc == 0 else "failed",
                           returncode=rc)]
                + ([JobResult(m.name, [], s.rides) for m in s.members]
                   if s.rides else []))
    try:
        cp = subprocess.run(s.command, cwd=str(s.cwd), capture_output=True,
                            text=True, env=env)
    except OSError as exc:
        raise SubmitError(
            f"{s.name}: could not run {s.command[0]!r} ({exc})") from None
    if cp.returncode != 0:
        raise SubmitError(
            f"sbatch failed for {s.name} (rc={cp.returncode}):\n"
            f"{cp.stderr.strip()}" + s.refusal_hint)
    jid = _parse_sbatch_id(cp.stdout)
    for m in s.members:
        # WHERE IT RUNS, not the container: `_launched` reads the attempt,
        # so a record in the container left every grouped trial reading
        # *never launched*, and a re-launch re-submitted work that had
        # already measured its point.
        where, basename = m.record_at()
        _record_launch(where, mode="submit", command=s.command, job_id=jid,
                       placement=s.placement, sent=s.sent,
                       basename=basename)
    record("launched", submission=s.name, command=s.command, job_id=jid,
           domain=s.domain, members=names)
    return ([JobResult(s.name, s.command, "submitted", job_id=jid,
                       domain=s.domain)]
            + ([JobResult(m.name, [], s.rides, job_id=jid)
                for m in s.members] if s.rides else []))


def _group_envelope(jobs) -> "Resources":
    """The allocation one shelf's grouped job asks for.

    Since the shelf split (2026-08-21) every caller passes trials sharing
    ONE exact ask (`_shelf_key`: ranks, cores, GPUs), so this is the
    shelf's own ask read off its trials -- nothing is widened and nothing
    narrower exists inside a group.  The uniformity is ASSERTED rather
    than assumed: trials disagreeing here mean the shelf partition broke,
    and launching an allocation that fits only some of them would be the
    silent repair this module never makes.  *(Until the split this
    function computed the union of a whole side -- the widest trial's
    ranks/cores/devices -- and narrower trials idled the difference; the
    max() folds below survive as identities.)*
    """
    keys = {(j.resources.mpi_np, j.resources.cpus_per_task,
             j.resources.gres or "") for j in jobs}
    if len(keys) > 1:
        raise SubmitError(
            "the group's trials do not share one resource ask "
            f"({sorted(keys)}) -- the per-shelf partition is broken "
            "(generator.md § 4.3a); this is a bug, not a declaration "
            "problem.")
    n = max(j.resources.mpi_np for j in jobs)
    c = max(j.resources.cpus_per_task for j in jobs)
    gres = next((j.resources.gres for j in jobs if j.resources.gres), None)
    exclusive = any(j.resources.exclusive for j in jobs)
    # PREP'S ANSWERS RIDE THE TRIALS (resolve.py: `replace(allocation,
    # **machine)` -- the sweep delta only touches ranks/cores/gres, so a
    # `prep --mem/--time` is on every trial).  This function DISCARDED
    # both, which is half of how five jobs went to Sol with no --mem and
    # an invented 38-minute wall (62039301-05, 2026-08-24): the user's
    # prep-time statement existed and never reached the sbatch command.
    mems = {j.resources.mem for j in jobs}
    times = {j.resources.time for j in jobs}
    # ...and the GPU binding, the description's switch for the whole
    # calculation (`execution/gpu.md` G9) -- one value over a sweep, too.
    binds = {j.resources.gpu_binding for j in jobs}
    # ...and WHETHER the trials use the GPU: the envelope is a request like
    # any job's (`model.gpu_request`), so it carries both halves -- its
    # count above, and this.  A shelf's trials share one family by
    # construction (G = 0 is the CPU family, `prep_inputs.bench_inputs`).
    uses = {bool(j.resources.use_gpu) for j in jobs}
    if len(mems) > 1 or len(times) > 1 or len(binds) > 1 or len(uses) > 1:
        raise SubmitError(
            f"the group's trials disagree about mem/time/gpu_binding/use_gpu "
            f"({sorted(mems, key=str)} / {sorted(times, key=str)} / "
            f"{sorted(binds, key=str)} / {sorted(uses)}) -- prep bakes one "
            f"allocation over a sweep, so this is a bug, not a declaration "
            f"problem.")
    return Resources(mpi_np=n, cpus_per_task=c, gres=gres,
                     use_gpu=next(iter(uses)),
                     exclusive=exclusive, gpu_binding=next(iter(binds)),
                     mem=next(iter(mems)), time=next(iter(times)))


def _dc_replace_time(r: "Resources", time_str: str) -> "Resources":
    """The envelope with its wall set -- dataclasses.replace, named so the
    call site reads as what it does."""
    import dataclasses as _dc
    return _dc.replace(r, time=time_str)


def _plan_shelves(jobset: JobSet, base: Path, *, mode: str,
                  domain: Optional[str], gpu_domain: Optional[str],
                  side: Optional[str], mem: Optional[str],
                  time_s: Optional[int],
                  trial_timeout_s: Optional[int]) -> LaunchPlan:
    """ONE scheduler job per RESOURCE SHELF of the sweep (grouped
    2026-08-20; split per side, then per shelf, 2026-08-21 --
    `generator.md` § 4.3a).

    The standing rule -- *a scheduler is handed few, deliberate jobs* --
    is kept by construction: each shelf IS one job, and the value axes
    keep the shelf count small.  What the grouping replaces is one
    job PER TRIAL, which made an N-point sweep cost N queue waits; on an HPC
    a submission is expensive and unpredictable, and a benchmark's output is
    timing data, not the structure, so the trials ride one allocation in
    sequence.

    **The split** (§ 4.3a): trials partition by each trial's GPU request
    (:func:`sides_of`, the one door) -- a sweep whose trials all
    answer one way submits the single ``bench-group``; a sweep spanning
    both submits ``bench-group-cpu`` and ``bench-group-gpu``, so the CPU
    group's envelope asks no ``gres`` and devices are never held while CPU
    trials run.  The names come from the SET's composition, not from what
    is pending, so a side keeps its name across resubmissions.  ``domain``
    applies to both sides through `scheduler.place`; ``side`` (``"cpu"``/
    ``"gpu"``) sends one side -- and a side this machine cannot launch
    simply stays pending for a later `launch bench`, which is the
    cross-cluster lane.

    The per-shelf pieces (:func:`_plan_shelf`): the allocation, the
    sequencer and its header -- planned here, written by the send -- and
    every included trial's launch record, stamped with the ONE job id, so
    `status` and a later single-trial re-run see the truth.  ``ask`` asks
    the scheduler about each shelf -- the jobs the send would hand it (W52:
    a sweep was asked about trial by trial while submit sent shelves).
    """
    from .materialize import job_dir_names, shape_of
    dirs = job_dir_names(jobset, shape_of(jobset, base))
    if side not in (None, "cpu", "gpu"):
        raise SubmitError(f"--only takes cpu or gpu, not {side!r}")
    sides = sides_of(jobset)
    if side and not sides[side]:
        raise SubmitError(f"this sweep has no {side} trials to submit")
    mixed = bool(sides["cpu"]) and bool(sides["gpu"])
    plan = LaunchPlan(base, mode, [], tolerant=True)
    # THE TRIALS STILL TO RUN, the one answer the walk here reads too
    # (:func:`_bench_trials`) -- of the sides this launch sends, in the
    # sweep's order.
    sent = {j.name for this in ("cpu", "gpu")
            if not (side and this != side) for j in sides[this]}
    trials = {m.job.name: m for m in _bench_trials(
        jobset, base, plan, mode=mode,
        jobs=[j for j in jobset.jobs if j.name in sent])}
    for this in ("cpu", "gpu"):
        jobs = sides[this]
        if not jobs or (side and this != side):
            continue
        shelves: dict = {}
        for j in jobs:
            shelves.setdefault(_shelf_key(j), []).append(j)
        multi = len(shelves) > 1
        for key in sorted(shelves, key=_shelf_width, reverse=True):
            pending = [trials[j.name] for j in shelves[key]
                       if j.name in trials]
            if not pending:
                continue            # this shelf already rode a group
            name = ("bench-group"
                    + (f"-{this}" if mixed else "")
                    + (f"-{_shelf_token(key, shelves[key])}"
                       if multi else ""))
            named = (gpu_domain or domain) if this == "gpu" else domain
            if named is None:
                why = _no_queue_named(
                    base, [m.job for m in pending], mem=mem, time_s=time_s,
                    one_proc=one_process(jobset.engine),
                    gpu=(this == "gpu"))
                if why:
                    raise SubmitError(why)
            plan.submissions.append(_plan_shelf(
                jobset, base, pending, name, plan,
                gpu_side=(this == "gpu"),
                domain=named,
                trial_timeout_s=trial_timeout_s, mem=mem, time_s=time_s))

    if not plan.submissions:
        from .materialize import bench_stage_of
        stage = bench_stage_of(base, base / dirs[jobset.jobs[0].name])
        raise SubmitError(
            f"all {len(sides[side]) if side else len(jobset.jobs)} "
            f"{side + ' ' if side else ''}trials are launched.  next:\n    "
            + (_cmd("summarize", "bench", stage, base=base) if stage else
               "`summarize bench` on the sweep's stage"))
    return plan


def _place(base: Path, want, *, gpu_side: bool, named=None,
           label: str = ""):
    """This side's placement — `scheduler.place`, walked with THIS machine's
    menu (`execution/scheduler.md` § 5).

    Fetching the menu is this layer's job; deciding is not.  The routing walk
    lived here until 2026-08-23, which is why the CPU and GPU sides were
    written separately and disagreed: the GPU side looked only at the first
    gpu-capable row, and when that row's 15-minute ceiling could not hold a
    38-minute group it returned "no preference" and let the header's
    directives -- naming that same row -- stand.

    Returns ``None`` when this machine has no menu at all: nothing was
    promised, so the rendered header stands (R6).  Raises `SubmitError` when
    there IS a menu and nothing on it can take the request -- we hold the
    record that says the scheduler will refuse, so we say so here rather than
    spend a round trip finding out.
    """
    from .. import runtime_config as _rc
    from ..scheduler.place import place, Unplaceable
    # ``want`` is THE request (`placement.request_of`): the GPU count and no
    # card (`scheduler.md` R2a) -- a GPU job goes to a queue that has GPUs,
    # where a node holds as many as were asked.
    try:
        placed = place(_rc.get_routing(project_dir=base), want,
                       prefer_gpu=gpu_side, named=named)
        # R9's SECOND record.  Routing reads the calculation scope first, so
        # a prepped bundle routes against the snapshot beside it -- which is
        # right for reproducibility and useless as a re-check, because it is
        # the same record the request was built against.  What will actually
        # enforce the limits is THIS machine's own record, so the re-admission
        # reads that one.  When the two agree this costs a read; when they
        # disagree it is the whole point of the rule.
        if placed is not None:
            _reject_if_this_machine_says_no(placed, want, gpu_side, label,
                                            base)
        return placed
    except Unplaceable as exc:
        raise SubmitError(
            f"{label or 'this group'} cannot be placed on this machine:\n    "
            # `.message` -- `exc.reasons` is List[Refusal] since 2026-09-09.
            # Joining the objects raised TypeError instead of this SubmitError,
            # so an unplaceable group lost both the reasons and the remedies.
            + "\n    ".join(r.message for r in exc.reasons)
            + "\n  Nothing was submitted -- the scheduler would refuse it.  "
              "Change the wall or the memory with --time / --mem, or name "
              "another of the record's queues with --domain.  The ranks "
              "and cores are its prep's, and a prepped stage is not "
              "prepped again: " + rollback("its prep", base=base)) from None


def _reject_if_this_machine_says_no(placed, want, gpu_side: bool,
                                    label: str, base) -> None:
    """R9 -- the machine's OWN record has the last word.

    A bundle carries the record it was prepared against.  If it travelled, or
    if the machine has been re-probed since, the limits that will actually be
    enforced are the ones here.  The Au-BDT-Au sweep is the worked example:
    its cells were sized against a snapshot whose gpu rows said
    ``max_cores: None``, and Sol has since measured 48.

    Silent when this machine has no record of the queue in question -- absent
    evidence is not evidence of a smaller limit (R3).

    ``local_only=True`` -- this asks what the box RUNNING THIS PROCESS knows,
    never what the calculation is prepped for.  Bug found 2026-08-23: on a
    workstation carrying named targets (``environments/sol.json``) but no
    probe of its own, plain ``get_routing(project_dir=None)`` fell into
    `machine_for`'s C1 guard -- "several machines could be meant" -- and
    raised ``AmbiguousTarget`` out of a read-only re-check that names no
    target at all.  This question has nothing to do with C1: no wrapper is
    written here, nothing travels, there is nothing to be ambiguous about.
    """
    from .. import runtime_config as _rc
    from ..scheduler import admits
    here = _rc.get_routing(project_dir=None, local_only=True)
    if not here:
        return                       # no record of my own; nothing to add
    mine = [d for d in here if (d.partition, d.qos) == (placed.partition,
                                                        placed.qos)]
    if not mine:
        return                       # this machine does not know that queue
    why = admits(mine[0], want)
    if why:
        # THE REMEDY IS THE ASK, OR A NEW PREP.  It said "re-run `prep` here
        # so the trials are sized against what this machine offers" until
        # 2026-10-02 (W54 R6): prep keeps the calculation's snapshot (M-3)
        # and sizes nothing -- the record checks an ask and supplies none of
        # it (`architecture.md` § 5.2) -- so the same refusal came back.  A
        # calculation is set to the record of its first prep (M-3), so this
        # machine's reaches it through a new prep from a saved state.
        raise SubmitError(
            f"{label or 'this group'} was prepared against a record that "
            f"allowed it, but THIS machine does not:\n    "
            + "\n    ".join(i.message for i in why)
            + f"\n  This machine's record of the queue {mine[0].name!r} "
              f"differs from the one the calculation was prepared against, "
              f"and its limits are the ones enforced here.  Change the wall "
              f"or the memory with --time / --mem, or name another of the "
              f"record's queues with --domain (the ranks and cores are its "
              f"prep's, and a prepped stage is not prepped again) -- or, to "
              f"prepare against this machine's record (configuration.md "
              f"M-3), " + rollback("the calculation's first prep", base=base))


def _shelf_key(job: "Job"):
    """The exact resource ask that defines a group (§ 4.3a, 2026-08-21):
    trials grouped together must fit ONE allocation with nothing idle, so
    the key is everything the envelope would widen over."""
    r = job.resources
    return (r.mpi_np, r.cpus_per_task,
            _gpus(r, f"trial {job.name!r}").count or 0)


def _shelf_width(key) -> tuple:
    """Widest-first order across shelves: cores, then devices."""
    n, c, g = key
    return (n * max(c, 1), g)


#: The machine axes of a sweep coordinate, in the order they are spelled.
_MACHINE_AXES = ("G", "K", "C")


def _shelf_token(key, jobs=()) -> str:
    """A shelf's name qualifier -- ``G2K24C1`` -- appended when a side spans
    more than one shelf (`job-contracts.md` § 6.3: the ``-`` announces a
    qualifier; the token stays in [A-Za-z0-9_]).

    **THE SAME SPELLING ITS TRIALS CARRY, read off a trial** rather than
    derived a second way.  This produced ``g2n48c1`` until 2026-08-24 --
    lowercase, and ``n`` for the TOTAL rank count -- while the very
    directories that shelf's job launches were named ``bench-G2K24C1…``,
    where ``K`` is ranks PER GPU.  Same three facts, different letters,
    different case, sitting side by side in one listing; they coincide only
    at ``G1``, which is why nothing had misread them yet.  Two vocabularies
    for one thing is what § 6.3 exists to prevent.

    Every trial on a shelf shares one resource ask by construction, so any
    member answers -- and the value axes that DO differ between them are
    dropped, because the shelf is the machine cell, not the point.

    ``jobs`` empty (a hand-built set with no points) falls back to deriving
    the coordinate from the key, in the same spelling: ranks split evenly
    over the devices by ELPA's equal-share rule (`tuning.md` § 2.12), which
    the grid enforces, so ``K = n / g`` is exact where it applies.
    """
    from ..resolve import point_token
    for j in jobs or ():
        pt = getattr(j, "point", None) or {}
        if all(a in pt for a in _MACHINE_AXES):
            return point_token({a: pt[a] for a in _MACHINE_AXES})
    n, c, g = key
    if g and n % g:
        # An uneven split is a bug upstream (the grid drops those cells),
        # and `n // g` would name a rank count no trial has.  Say the
        # total instead of quietly rounding it.
        return point_token({"G": g, "K": 0, "C": c}).replace("K0", f"N{n}")
    return point_token({"G": g, "K": (n // g) if g else n, "C": c})


def _domain_name(prepared) -> Optional[str]:
    """The NAME of the domain a shelf was placed on, or ``None``.

    ``None`` means this machine has no menu (R6) -- the rendered header's
    own directives stand, and there is no domain to name.
    """
    placement = getattr(prepared, "placement", None)
    return getattr(placement, "name", None) or None


def submitted_cap_notes(plans) -> List[str]:
    """What a domain's per-user submitted-job cap says about THIS sweep
    (R14) -- one sentence per domain that cannot take it, ``[]`` otherwise.

    **A note, not a refusal**, and that is the design.  A refused shelf
    already costs nothing: its trials keep no launch record, so
    ``_launched`` leaves them pending and the next ``launch bench``
    picks up exactly them.  What was missing was not enforcement, it was
    being TOLD -- a sweep of six went to a queue that takes two, and the
    person learned the cap from four red refusals after saying yes.

    Silent when the record does not state a cap.  ``UNSET`` means the
    probe never asked (a record older than 2026-08-30) and ``None`` means
    it asked and the QoS states none; neither is a limit, and R3 forbids
    reading an unstated one as a bar.
    """
    from collections import Counter
    counts: Counter = Counter()
    caps: Dict[str, object] = {}
    for p in plans:
        name = _domain_name(p)
        if not name:
            continue
        counts[name] += 1
        placement = getattr(p, "placement", None)
        caps[name] = getattr(getattr(placement, "domain", None),
                             "max_submit_jobs", None)
    notes: List[str] = []
    for name, n in sorted(counts.items()):
        cap = caps.get(name)
        # `bool` is a subclass of `int`, so a stray True would compare as a
        # cap of 1 and invent a refusal.  A cap is a count or it is nothing.
        if type(cap) is not int or n <= cap:
            continue
        # SAID AS A CONDITION, NOT A PREDICTION.  What is known here is the
        # cap and the size of this sweep; what is NOT known is how many jobs
        # are already queued under that QoS -- with one already there, only
        # `cap - 1` of these get in.  Writing "the scheduler will accept 2"
        # would be the same overclaim this whole change exists to remove:
        # a sentence stated with more certainty than its evidence.
        notes.append(
            f"{name} takes {cap} submitted job(s) per user, and this sweep "
            f"is {n}. With nothing of yours already queued there, {cap} go "
            f"and {n - cap} come back QOSMaxSubmitJobPerUserLimit -- fewer "
            f"if you already hold some. A refused shelf's trials stay "
            f"pending, and re-running this launch picks up exactly them.")
    return notes


def _bench_walk(name: str, trials, *, where: str, log: str,
                bound_s: Optional[int] = None) -> str:
    """A BENCHMARK'S WALK -- the script that runs its trials one after
    another in one submission: a resource shelf sent to a queue
    (`generator.md` § 4.3a), or its unlaunched trials run here
    (`job-system.md` § 6.0).  ``trials`` are ``(name, folder, run script,
    its arguments)``, the folder from where the walk runs; ``bound_s`` the
    per-trial bound, a trial past it killed and read incomplete.  A trial
    that fails leaves the rest to run -- one bad point says nothing about
    the next -- and the walk exits nonzero when any failed; stopped by the
    person (Ctrl-C, a lost terminal, a cancel), it starts no further trial.
    THE TWO-LAYER
    MODEL HOLDS (`job-system.md` § 6): this file orders and bounds; each
    trial's own ``.run.sh`` activates its environment and launches its
    engine, exactly as when it runs alone.  The benchmark's own: a
    transport bias scan's walk is transport's, and shares nothing with it
    (user, 2026-10-05)."""
    when = "$(date '+%Y-%m-%dT%H:%M:%S')"
    lines = [
        "#!/usr/bin/env bash",
        f"# {name}.run.sh -- {where}, in sequence",
        "# (generator.md § 4.3a; job-system.md § 6.0).  Regenerated at each",
        "# launch.  THE TWO-LAYER MODEL HOLDS (job-system.md § 6): this file",
        "# is the launcher layer only -- ordering and bounds.  Env activation",
        "# and the engine launch stay in each trial's own .run.sh, exactly as",
        "# when a trial runs alone; nothing here re-implements module load /",
        "# source activate.",
        "set -u",
        f'LOG="{log}"',
        f'echo "[group] {when} start trials={len(trials)} per-trial-bound='
        f'{f"{bound_s}s" if bound_s else "none"} '
        'job=${SLURM_JOB_ID:-none} node=$(hostname) '
        'alloc_ntasks=${SLURM_NTASKS:-unset} '
        'alloc_cpus=${SLURM_CPUS_PER_TASK:-unset}" >> "$LOG"',
        "fails=0",
        # STOPPED BY THE PERSON -- Ctrl-C, a lost terminal, a scancel --
        # the walk starts no further trial (bash runs this after the
        # running trial returns; one under a per-trial bound ends at it).
        # Until 2026-10-05 the walk went on with the rest unseen (the
        # unit 11 review).
        f'_walk_stopped() {{ echo "[group] {when} stopped -- no further '
        'trial" >> "$LOG"; exit 130; }',
        "trap _walk_stopped INT TERM HUP",
        "run_trial() {",
        '    _name="$1"; _dir="$2"; shift 2',
        "    _t0=$(date +%s)",
        f'    echo "[group] {when} -> ${{_name}} starts" >> "$LOG"',
        (f'    ( cd "${{_dir}}" && timeout -k 30 {bound_s} '
         'bash "$@" ) >> "$LOG" 2>&1'
         if bound_s else
         '    ( cd "${_dir}" && bash "$@" ) >> "$LOG" 2>&1'),
        "    _rc=$?",
        '    if [ "${_rc}" -eq 124 ]; then',
        (f'        echo "[group] ${{_name}} hit the {bound_s}s '
         'per-trial bound -- killed; its artifacts read incomplete" >> "$LOG"'
         if bound_s else
         '        echo "[group] ${_name} killed (124)" >> "$LOG"'),
        "    fi",
        '    if [ "${_rc}" -ne 0 ]; then fails=$((fails+1)); fi',
        "    _t1=$(date +%s)",
        f'    echo "[group] {when} <- ${{_name}} finished rc=${{_rc}} '
        'took=$(( _t1 - _t0 ))s" >> "$LOG"',
        "    return 0    # one bad point says nothing about the next",
        "}",
    ]
    for trial, folder, run_sh, args in trials:
        lines.append(f'run_trial "{trial}" "{folder}" "{run_sh}"'
                     + (f" {args}" if args else ""))
    lines += [f'echo "[group] {when} done fails=${{fails}}" >> "$LOG"',
              "exit $(( fails > 0 ))", ""]
    return "\n".join(lines)


def sides_of(jobset: JobSet) -> Dict[str, List[Job]]:
    """A sweep's trials by the side they run on -- ``{"cpu": [...],
    "gpu": [...]}`` -- each trial's GPU request (`model.gpu_request`, the one
    door), read off the trial itself.  The grouped door splits on it.
    *(It read each
    trial's deck where the trial runs until 2026-10-03.)*"""
    sides: Dict[str, List[Job]] = {"cpu": [], "gpu": []}
    for j in jobset.jobs:
        sides["gpu" if _gpus(j.resources, f"trial {j.name!r}").uses
              else "cpu"].append(j)
    return sides


def _bench_trials(jobset: JobSet, base: Path, plan: LaunchPlan, *,
                  mode: str, jobs=None) -> List[_Member]:
    """A benchmark's walk: its trials still to run -- of ``jobs``, all of
    them by default -- each through the one member planner
    (:func:`_plan_member`: where it runs, and whether it was launched) and
    its gates: its deck agrees with its launch, it starts cold, its run
    script is where it is run.  The walk here and a queue's shelves both
    send this answer; a trial launched before is passed over by name
    (``plan.skipped``).  *(A queue's shelves read whether a trial was
    launched for themselves, and built its member a second way, until
    2026-10-06; the two agreed.)*"""
    pending: List[_Member] = []
    for job in (jobset.jobs if jobs is None else jobs):
        m = _plan_member(jobset, base, job, mode=mode, writes=plan.writes)
        if isinstance(m, JobResult):
            plan.skipped.append(m)          # measured before
            continue
        # ITS OWN COUNTS, STATED: a walk hands each trial its -np / -omp,
        # the shield against an allocation's SLURM_* variables -- a trial
        # that cannot state them is refused by name, never mis-measured.
        if not (job.resources.mpi_np and job.resources.cpus_per_task):
            raise SubmitError(
                f"trial {job.name!r} states no rank or core count -- a "
                f"benchmark's walk hands each trial its own (-np/-omp), "
                f"and a prepped benchmark is not prepped again: "
                + rollback("the benchmark's prep", base=base))
        # THE GATES GUARD EVERY DOOR (review 2026-08-21): a trial refused
        # when sent by name must not go silently by riding a walk.  And the
        # COLD gate (user, same day: "it is the submission that determines
        # the actual state of the run"): the pin baked the intent at prep;
        # here the deck itself is verified -- where it IS, the trial's
        # attempt (`project-layout.md` § 1.5a).
        try:
            check_launch_matches_deck(m.read_from, job)
            check_trial_starts_cold(m.read_from, job)
        except DeckLaunchMismatch as e:
            raise SubmitError(str(e)) from e
        run_name = _wrapper_name(job.script, ".run.sh")
        if not (m.read_from / run_name).exists():
            raise SubmitError(
                f"trial {job.name!r}: {run_name} is not in {m.read_from}, "
                f"and a prepped benchmark is not prepped again: "
                + rollback("the benchmark's prep", base=base))
        plan.reads += [_as_found(m.read_from / f, base)
                       for f in (job.script, run_name)]
        pending.append(m)
    return pending


def _walk_of(members: List[_Member], container: Path) -> list:
    """Each trial as a benchmark's walk runs it -- its name, the folder it
    runs in from the walk's own (its attempt: `project-layout.md` § 1.5a),
    its run script, and its own counts, the shield against an allocation's
    SLURM_* variables.  One answer for the walk here and a queue's shelf."""
    return [(m.name, str(m.run_dir.relative_to(container)),
             _wrapper_name(m.job.script, ".run.sh"),
             " ".join(_run_sh_args(m.job.resources))) for m in members]


def _plan_bench_here(jobset: JobSet, base: Path, *,
                     trial_timeout_s: Optional[int]) -> LaunchPlan:
    """A benchmark's trials run HERE (``--mode direct``, no trial named):
    one submission walking every trial not yet launched, in the sweep's
    order, through the benchmark's walk (:func:`_bench_walk`) -- as a shelf
    walks its trials on a queue -- each under the per-trial bound when one
    is given; a trial launched before is passed over by name
    (:func:`_bench_trials`).  *(Each trial ran as its own process, one
    after another from Python, until 2026-10-05.)*"""
    plan = LaunchPlan(base, "direct", [])
    pending = _bench_trials(jobset, base, plan, mode="direct")
    if not pending:
        from .materialize import bench_stage_of, job_dir_names, shape_of
        dirs = job_dir_names(jobset, shape_of(jobset, base))
        stage = bench_stage_of(base, base / dirs[jobset.jobs[0].name])
        raise SubmitError(
            f"all {len(jobset.jobs)} trials are launched.  next:\n    "
            + (_cmd("summarize", "bench", stage, base=base) if stage else
               "`summarize bench` on the sweep's stage"))
    # THE ONE PARENT THAT SEES EVERY TRIAL -- the benchmark's container,
    # as a shelf's walk runs from it on a queue.
    containers = {m.container.parent for m in pending}
    if len(containers) != 1:
        raise SubmitError(
            "the sweep's trials do not share one container -- a walk of "
            f"them needs the one parent that sees them all; found "
            f"{sorted(str(c) for c in containers)}")
    container = next(iter(containers))
    name = "bench-group"
    log = f"{LAUNCH_DIR}/{name}.log"
    plan.writes.text(container / LAUNCH_DIR / f"{name}.run.sh", _bench_walk(
        name, _walk_of(pending, container),
        where="this benchmark's unlaunched trials, run here", log=log,
        bound_s=trial_timeout_s))
    plan.submissions.append(Submission(
        name, ["bash", f"launch/{name}.run.sh"], container, True, pending,
        rides="rides the group", log=container / log))
    return plan


def _plan_shelf(jobset: JobSet, base: Path, pending: List[_Member],
                name: str, plan: LaunchPlan, *, gpu_side: bool,
                domain: Optional[str],
                trial_timeout_s: Optional[int], mem: Optional[str] = None,
                time_s: Optional[int] = None) -> Submission:
    """One shelf's submission, checked and placed, its sequencer and header
    planned in ``plan`` -- written by the send, never before.  ``pending``
    are its trials still to run, each planned and gated once
    (:func:`_bench_trials`).

    Every gate, the envelope, the placement and both scripts; the `sbatch`
    is the send's (:func:`_go`).  A dry run and an `ask` write nothing
    (`job-system.md` § 6.0, step 3), and an `ask` needs no header of the
    shelf's own: it asks over the first trial's.

    Widest-first ordering lives one level up since the shelf split
    (2026-08-21): every trial in a group shares one exact resource ask by
    construction, so within a group the enumeration (declaration) order
    stands, and the SHELVES submit widest first.
    """

    # THE CONTAINER IS THE TRIAL'S PARENT, NOT THE ATTEMPT'S.  With an
    # attempt layer the attempt's parent is `bench-<point>` -- one per
    # trial -- so the "they must share one container" check found N and
    # refused every grouped submission.  The container question belongs to
    # the trial's own folder, the files question to its attempt; they are
    # two questions and this asks each of the right thing.
    containers = {m.container.parent for m in pending}
    if len(containers) != 1:
        raise SubmitError(
            "the sweep's trials do not share one container -- a grouped "
            f"submission needs the one parent that sees them all; found "
            f"{sorted(str(c) for c in containers)}")
    container = next(iter(containers))
    # L3 (roadmap 7.10, user 2026-08-24): the group's own machinery -- this
    # sequencer, its .sbatch, its log, and SLURM's stdout/err -- lives in
    # ``launch/`` beside the trial directories, not among them.  Made when
    # the send writes the plan: a dry run and a declined question leave
    # nothing behind, an empty ``launch/`` included (W52).
    launch_dir = container / LAUNCH_DIR

    envelope = _group_envelope([m.job for m in pending])

    # THE ATTEMPT, NOT THE TRIAL.  The wrapper lives in `run-<n>` since the
    # attempt layer landed (`project-layout.md` § 1.5a, 2026-08-27), and the
    # sequencer went on naming the trial DIRECTORY -- so every grouped bench
    # `cd`ed one level too high and every trial died instantly with *"No
    # such file or directory"* (rc=127; Sol job 62372574).  Each member's
    # ``run_dir`` is the one answer to "where does this trial run", and the
    # gates asked it (:func:`_bench_trials`).  Each trial is handed its own
    # -np / -omp: the shield against the envelope's SLURM_* variables.
    script = _bench_walk(
        name, _walk_of(pending, container),
        where="ONE allocation, this shelf's unlaunched trials",
        log=f"{LAUNCH_DIR}/{name}.log", bound_s=trial_timeout_s)
    # THE ONE REQUEST (`_sbatch_request`): prep's envelope, what was said at
    # launch, admitted on this side's queue (R9), every value stated.  The
    # side IS the envelope's GPU request -- every trial on the shelf shares
    # it (`_group_envelope`), so nothing is re-derived here.
    envelope, placement, cmd = _sbatch_request(
        base, envelope=envelope, domain=domain, mem=mem,
        time_s=time_s, label=name,
        job_name=_scheduler_job_name(jobset, name),
        script=f"launch/{name}.sbatch", run_args=(),
        one_process=one_process(jobset.engine))

    if plan.mode == "submit":
        from ..runwrap import _render_sbatch_for
        # Rendered at the BUNDLE's scope, not the container's (review
        # 2026-08-21): the render derives its config/environment scope from
        # the script path's parent, and the calculation's environment.json
        # lives at the bundle root.  The pair this submission
        # is ALREADY routing to is handed to the header emitter, so the
        # .sbatch and the `sbatch -p/-q` on the command line cannot name
        # different queues.  The stem alone names the delegated run script,
        # so the header still runs `bash {name}.run.sh` from the container.
        header = _render_sbatch_for(base / f"{name}.sh",
                                    project_dir=base,
                                    resources=envelope,
                                    domain_pq=((placement.partition,
                                                placement.qos)
                                               if placement else None))
        if header is None:
            raise SubmitError(_no_sbatch(name, f"launch/{name}.sbatch",
                                         base=base))
        plan.writes.text(launch_dir / f"{name}.run.sh", script)
        plan.writes.text(launch_dir / f"{name}.sbatch",
                         _into_launch(header, name))
    elif plan.mode == "ask":
        # ASKED WITH THE FIRST TRIAL'S HEADER, the shelf's being written
        # only when it is sent: a benchmark prepped with --no-sbatch has
        # none, and a question about a file that does not exist answers
        # nothing -- refused, saying why (the scheduler's "Unable to open
        # file" stood here until 2026-10-06, the unit 11 review).
        first = pending[0]
        if not (first.read_from
                / _wrapper_name(first.job.script, ".sbatch")).exists():
            raise SubmitError(
                f"{name}: there is no header to ask the scheduler about -- "
                f"the benchmark was prepped with --no-sbatch, and a "
                f"shelf's own header is written when it is sent.  Send "
                f"it with --mode submit.")

    # A SHELF THE SCHEDULER REFUSES keeps its trials pending -- no launch
    # record -- so launching the benchmark again sends exactly them; a
    # shelf sent by hand would write no record, and the next launch would
    # measure its trials twice (the unit 11 review, 2026-10-05).
    hint = ("\n  Its trials stay pending: launch the benchmark again once "
            "the ask fits -- the shelves already queued are skipped.")
    first = pending[0]
    return Submission(
        name, cmd, container, False, pending,
        placement=placement, sent=envelope, rides="rides the group",
        ask_in=first.read_from,
        ask_script=_wrapper_name(first.job.script, ".sbatch"),
        refusal_hint=hint)


def _plan_chain(jobset: JobSet, base: Path, task, *, mode: str, stage: str,
                domain: Optional[str] = None, mem: Optional[str] = None,
                time_s: Optional[int] = None) -> LaunchPlan:
    """ONE submission that walks a transport bias scan's points in
    order (`archive/2026-09-01-transport-design.md` § 4.3; layout ruled 2026-08-29: plain
    v-dirs, one attempt ladder per point).

    The walker is the launcher layer only, exactly like the bench
    group's sequencer: it ``cd``s into each point's prepared attempt and
    runs the point's own ``.run.sh`` — env activation and the engine
    launch stay where they always live.  What it adds is the WARM CHAIN:
    before each point after the first, what the previous point left that
    a continuing device takes -- the device job's declaration, the NEGF
    density ``.TSDE`` among it, from the one restart-list door -- is copied
    forward, so ``V_{i+1}`` converges from ``V_i``'s state instead of from
    scratch.  And unlike the bench
    group it STOPS on a failed point: later points chain their density
    from this one, so walking on would converge from a state the
    failure poisoned — a benchmark's points are independent, a chain's
    are not.

    Every point's attempt must be OPEN and unlaunched (``prep run
    device`` opens them all); the deck/launch agreement gate guards this
    door like every other.  ``run.json`` lands in every point's attempt
    when the one job goes -- they are all launched by it.  The job's
    request is the one every door sends (:func:`_sbatch_request`); ``ask``
    asks the scheduler about it, over the first point's own header, since
    the chain's is written only when it is sent.  The walker and its header
    are planned here, and written by the send.
    """
    from ..task import bias_token
    from ..transport.stages import rung_containers, scan_points
    from .materialize import latest_attempt

    points = scan_points(task, stage)
    if len(points) < 2:
        raise SubmitError("not a bias scan -- the plain launch owns "
                          "a single-point device.")
    # The device chain warm-hands its declaration and STOPS on failure
    # (later points inherit the failed state); the transmission walk is
    # the same one-submission sequence over INDEPENDENT points -- no
    # hand-forward, and a bad point says nothing about the next, so the
    # walk continues and the exit code reports any failure (P6).
    warm = stage == "device"
    job = next((j for j in jobset.jobs if j.name == stage), None)
    if job is None:
        raise SubmitError(
            f"the {stage} stage is not in the plan -- run "
            f"`{_cmd('prep', 'run', stage, base=base)}` first.")
    # The stage's folder, from the one door (`materialize.stage_home`).
    from .materialize import stage_home
    home = stage_home(base, task, stage)
    token, stage_dir = home.token, home.dir
    launch_dir = stage_dir / LAUNCH_DIR
    run_name = _wrapper_name(job.script, ".run.sh")
    name = f"{Path(job.script).stem}-chain"

    plan = LaunchPlan(base, mode, [])
    attempts: List[Tuple[float, Path, Path]] = []
    # Each point's folder, from the one door (`rung_containers`, plan § 5w
    # K10) -- the folders prep wrote the point's deck and attempt into.
    for vdir, v in rung_containers(base, task, stage):
        att = latest_attempt(vdir)
        if att is None:
            raise SubmitError(
                f"bias point {bias_token(v)}: no attempt is open under "
                f"{token}/{bias_token(v)}/ -- the scan launches whole, so "
                f"every point needs one, and a prepped stage is not prepped "
                f"again: " + rollback("its prep", base=base))
        if _launched(att):
            raise SubmitError(
                f"bias point {bias_token(v)}: {att.relative_to(base)} "
                f"has already been launched.  An attempt is immutable once "
                f"it has run, and a prepped stage is not prepped again: "
                + rollback("its prep", base=base))
        try:
            check_launch_matches_deck(att, job)
        except DeckLaunchMismatch as e:
            raise SubmitError(str(e)) from e
        plan.reads += [_as_found(att / f, base)
                       for f in (job.script, run_name)]
        attempts.append((v, vdir, att))

    args = " ".join(_run_sh_args(job.resources))
    # WHAT A POINT TAKES FROM THE ONE BEFORE IT is what a continuing device
    # takes -- the device job's own declaration (`Job.warm`), which prep
    # recorded from the one restart-list door (`warmfiles.warm_list`,
    # `job-contracts.md` § 4.2a), the calculation's own list among it.
    # `.TSDE` was spelled here by hand until 2026-10-03 (plan W38, W36 ⑧).
    import shlex as _shlex
    handed = " ".join(_shlex.quote(w.name) for w in (job.warm or ()))
    lines = [
        "#!/usr/bin/env bash",
        f"# {name}.run.sh -- the bias chain: this scan's points in",
        "# sequence, each warm-started from what the previous point left",
        "# that a continuing device takes -- its declaration:",
        f"#   {handed or '(nothing)'}",
        "# (archive/2026-09-01-transport-design.md 4.3).  Regenerated at each launch.",
        "# STOPS on a failed point: later points chain their density",
        "# from this one, so walking on would converge from a state the",
        "# failure poisoned (a benchmark's points are independent; a",
        "# chain's are not).",
        "set -u",
        f'LOG="launch/{name}.log"',
        f'echo "[chain] $(date \'+%Y-%m-%dT%H:%M:%S\') start '
        f'points={len(attempts)} job=${{SLURM_JOB_ID:-none}} '
        'node=$(hostname)" >> "$LOG"',
        "prev=''",
        "fails=0",
        "run_point() {",
        '    _name="$1"; _dir="$2"; shift 2',
    ] + ([
        '    if [ -n "$prev" ]; then',
        '        _took=""',
        f'        for _f in {handed}; do',
        '            if [ -f "$prev/$_f" ]; then',
        '                cp "$prev/$_f" "$_dir/" && _took="$_took $_f"',
        "            fi",
        "        done",
        '        if [ -n "$_took" ]; then',
        '            echo "[chain] ${_name}: warm from $prev:$_took" >> "$LOG"',
        "        else",
        '            echo "[chain] ${_name}: nothing to take from $prev -- '
        'converging from scratch" >> "$LOG"',
        "        fi",
        "    fi",
    ] if warm and handed else []) + [
        '    echo "[chain] $(date \'+%Y-%m-%dT%H:%M:%S\') -> '
        '${_name} starts" >> "$LOG"',
        '    ( cd "${_dir}" && bash "$@" ) >> "$LOG" 2>&1',
        "    _rc=$?",
        '    if [ "${_rc}" -ne 0 ]; then',
    ] + ([
        '        echo "[chain] ${_name} FAILED rc=${_rc} -- the chain '
        'stops here; later points would inherit its state" >> "$LOG"',
        '        exit "${_rc}"',
    ] if warm else [
        '        echo "[chain] ${_name} FAILED rc=${_rc} -- independent '
        'points; the walk continues" >> "$LOG"',
        "        fails=$((fails+1))",
    ]) + [
        "    fi",
        '    echo "[chain] $(date \'+%Y-%m-%dT%H:%M:%S\') <- ${_name} '
        'done" >> "$LOG"',
        '    prev="${_dir}"',
        "}",
    ]
    for v, _vdir, att in attempts:
        # THE SAME IDIOM THE BENCH SEQUENCER USES.  Composing the path from
        # its parts was correct here and wrong there, and two spellings of
        # "where does this cd" is how the two came to disagree at all.
        rel = att.relative_to(stage_dir)
        lines.append(f'run_point "{bias_token(v)}" "{rel}" "{run_name}"'
                     + (f" {args}" if args else ""))
    lines += ['echo "[chain] $(date \'+%Y-%m-%dT%H:%M:%S\') done '
              'fails=${fails}" >> "$LOG"',
              'exit $(( fails > 0 ))', ""]

    members = [_Member(job, vdir, att, True, att,
                       label=f"{stage}@{bias_token(v)}", base=base)
               for v, vdir, att in attempts]
    if mode != "ask":
        plan.writes.text(launch_dir / f"{name}.run.sh", "\n".join(lines))
    if mode == "direct":
        plan.submissions.append(Submission(
            name, ["bash", f"launch/{name}.run.sh"], stage_dir, True, members,
            rides="rides the chain"))
        return plan

    # ---- submit / ask: one scheduler job, the group pattern in miniature #
    envelope, placement, cmd = _sbatch_request(
        base, envelope=job.resources, domain=domain,
        mem=mem, time_s=time_s, label=name,
        job_name=_scheduler_job_name(jobset, name),
        script=f"launch/{name}.sbatch", run_args=(),
        one_process=one_process(jobset.engine))
    if mode == "submit":
        from ..runwrap import _render_sbatch_for
        header = _render_sbatch_for(base / f"{name}.sh", project_dir=base,
                                    resources=envelope,
                                    domain_pq=((placement.partition,
                                                placement.qos)
                                               if placement else None))
        if header is None:
            raise SubmitError(_no_sbatch(name, f"launch/{name}.sbatch",
                                         base=base))
        plan.writes.text(launch_dir / f"{name}.sbatch",
                         _into_launch(header, name))
    plan.submissions.append(Submission(
        name, cmd, stage_dir, False, members, placement=placement,
        sent=envelope, rides="rides the chain", ask_in=attempts[0][2],
        ask_script=_wrapper_name(job.script, ".sbatch")))
    return plan


def _placed_on(placement, sent=None) -> Optional[dict]:
    """A `Placement` -> where this run was SENT, and with what wall and
    memory (``sent``, the request as sent: a launch flag changes them,
    `job-system.md` § 6.0) -- or ``None`` when there was no placement (a
    direct run).  *(The wall and memory were in the record's line alone
    until 2026-10-06.)*

    The QUEUE half of `scheduler.md` R12: domain, partition, qos -- known
    the moment ``sbatch`` accepts, and reachable before this field only by
    parsing the argv the same file records.  What the job LANDED ON is the
    monitor's to record, on the node, because a queued job has no node yet
    -- this dict carried a ``node_type`` until 2026-08-27, which was the
    domain's opinion of itself, not a fact about the run, and the probe
    never wrote it (R11).
    """
    if placement is None:
        return None
    d = getattr(placement, "domain", None)
    return {"domain": getattr(d, "name", None),
            "partition": placement.partition,
            "qos": placement.qos,
            **{f: getattr(sent, f) for f in ("time", "mem")
               if getattr(sent, f, None) not in (None, "")}}


def _record_launch(attempt: Path, *, mode: str, command: List[str],
                   job_id: Optional[str] = None, placement=None, sent=None,
                   basename: Optional[str] = None) -> None:
    """Write the launch record where `materialize.launch_record_at` says --
    the attempt's ``run.json``, or with ``basename`` a flat stage's own --
    carrying its provenance.

    ``continued_from`` is read back from what ``prep`` left -- the attempt's
    ``.continued-from``, or a flat stage's own (`continued_from_marker`) --
    rather than passed down: prep is what knows, and re-deriving it here
    would be a second answer to one question.  ``placement`` is passed for
    the opposite reason: submission is what knows where the job went, and
    nothing downstream should have to work it out from a command line.
    """
    from ..runrecord import read_continued_from, write_launch
    src = read_continued_from(attempt, basename)
    write_launch(attempt, mode=mode, command=command, job_id=job_id,
                 continued_from=src, placed_on=_placed_on(placement, sent),
                 basename=basename)


def _plan_stage(jobset: JobSet, base: Path, *, mode: str,
                domain: Optional[str], gpu_domain: Optional[str],
                only: Optional[str], mem: Optional[str],
                time_s: Optional[int]) -> LaunchPlan:
    """The stage door's plan -- a ladder's stage, or a sweep's named
    trial: one submission.

    For every job, before anything is written: where it runs and what it
    follows (:func:`_plan_member`), the deck/launch agreement and a trial's
    cold start, the wrapper the line runs -- where the line is run, or asked
    about when a scheduler is here to ask -- and, for a scheduler, the one
    request (:func:`_sbatch_request`): the queue admitted, the line exact.

    **There is no ``chain`` parameter** (deleted 2026-08-10, user): an
    opt-in typed before any stage has run is typed at the moment you know
    least.  A ladder is launched ONE stage at a time, in every mode -- direct
    running stages in order would be local chaining.
    """
    if only is not None:
        jobset = dataclasses.replace(
            jobset, jobs=[j for j in jobset.jobs if j.name == only])
    # The no-chain rule, AT THE SEAM (U5, 2026-08-12): a guard only a
    # surface applies is one the next surface forgets.
    if jobset.kind == "ladder" and len(jobset.jobs) > 1:
        raise SubmitError(
            "a ladder is launched ONE stage at a time; pass `only=<stage>`. "
            "Stages do not chain (project-layout.md § 1.6), in direct mode "
            "as much as submit: each stage is launched after you have "
            "looked at the one before it.")

    plan = LaunchPlan(base, mode, [])
    sbatch_here = shutil.which("sbatch") is not None
    for job in jobset.jobs:
        m = _plan_member(jobset, base, job, mode=mode, writes=plan.writes,
                         named=only is not None)
        if isinstance(m, JobResult):
            plan.skipped.append(m)             # a trial measured before
            continue
        try:
            check_launch_matches_deck(m.read_from, job)
            if jobset.kind == "sweep":
                # the cold gate rides the named-trial door too (user,
                # 2026-08-21) -- one rule, every launch path
                check_trial_starts_cold(m.read_from, job)
        except DeckLaunchMismatch as e:
            # M5: the refusal is the launch's -- the agreement floor states
            # the fact, this verb is what declines to act on it.
            raise SubmitError(str(e)) from e
        run_name = _wrapper_name(job.script, ".run.sh")
        sbatch_name = _wrapper_name(job.script, ".sbatch")
        plan.reads += [_as_found(m.read_from / f, base)
                       for f in (job.script, run_name, sbatch_name)]
        if mode == "direct":
            # The wrapper is required where it is RUN -- and a dry run is
            # this plan, shown: it says the refusal the launch would meet.
            if not (m.read_from / run_name).exists():
                # THE PREP THAT WROTE IT is not run again: the way back is
                # the state saved before it (`job-system.md` § 5.0).
                what = ("the benchmark's prep" if jobset.kind == "sweep"
                        else "its prep")
                raise SubmitError(
                    f"job {job.name!r}: {run_name} is not in "
                    f"{m.read_from}, and a prepped stage is not prepped "
                    f"again: " + rollback(what, base=base))
            plan.submissions.append(Submission(
                m.name, ["bash", run_name] + _run_sh_args(job.resources),
                m.run_dir, True, [m], ask_in=m.read_from))
            continue
        # The header is required where it is SENT -- or ASKED about, when a
        # scheduler is here to ask: a question about a file that does not
        # exist answers nothing.
        if ((mode == "submit" or sbatch_here)
                and not (m.read_from / sbatch_name).exists()):
            raise SubmitError(_no_sbatch(f"job {job.name!r}", sbatch_name,
                                         base=base))
        gpu = _gpus(job.resources, f"job {job.name!r}").uses
        named = gpu_domain if gpu and gpu_domain else domain
        if named is None:
            why = _no_queue_named(base, [job], mem=mem, time_s=time_s,
                                  one_proc=one_process(jobset.engine),
                                  gpu=gpu)
            if why:
                raise SubmitError(why)
        sent, placement, cmd = _sbatch_request(
            base, envelope=job.resources,
            domain=named,
            mem=mem, time_s=time_s, label=job.name,
            job_name=_scheduler_job_name(jobset, job.name),
            script=sbatch_name, run_args=_run_sh_args(job.resources),
            one_process=one_process(jobset.engine))
        plan.submissions.append(Submission(
            m.name, cmd, m.run_dir, False, [m], placement=placement,
            sent=sent, ask_in=m.read_from))
    return plan


__all__ = ["plan_launch", "ask_launch", "send_launch", "LaunchPlan",
           "Submission", "JobResult", "SubmitError"]
