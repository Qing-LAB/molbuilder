"""The launch door's engine -- hand a *prepped* :class:`JobSet` to the
scheduler, or run it here (`docs/execution/job-system.md` § 5.3, § 6, § 7;
the wrapper contract is `job-contracts.md` § 2).

``prep`` derives the work and lays out the tree; THIS module sends it, and
writes only what sending needs.  Three doors, one per shape of work:

  * **the job door** (:func:`submit_jobset`) -- a ladder's stage, a sweep's
    named trial, or (direct) a sweep's trials in turn.  A stage launched
    before is launched again by CONTINUING it (user, 2026-08-21): the next
    attempt opens from its latest, and a run that never concluded is
    followed only on the person's judgement;
  * **the grouped bench** (:func:`submit_bench_group`) -- ONE job per
    resource shelf of a sweep, its trials in sequence (`generator.md`
    § 4.3a);
  * **the bias chain** (:func:`submit_transport_chain`) -- one job walking a
    transport scan's points.

Three modes: ``submit`` (``sbatch``), ``ask`` (``sbatch --test-only`` on
the same line -- when would it start; nothing is written or recorded) and
``direct`` (``bash`` here, in order, waiting for each).  Every door builds
its scheduler line through ONE request (:func:`_sbatch_request`: what prep
baked, what was said at launch, admitted on the queue, and every value
stated or refused), and decides everything -- what it follows, the
deck/launch agreement, the queue, the header -- before the first write.
The scheduler is handed ONE job per invocation, a grouped bench one per
shelf (:func:`_refuse_batch_submission`).  ``dry_run`` returns the exact
command each job would get and writes nothing.

*(This header described two paths, one job per invocation, and a module
that never inspects prior output, until 2026-10-01 -- each untrue of the
body below it; W52.)*
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from ..paths import attempt_dir
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
    finds it missing (`job-system.md` § 6): prep withholds the ``.sbatch``
    only where the machine it prepped for names no queue.  Three doors
    worded it three ways until 2026-10-01, one telling a person to "add a
    scheduler block" -- a premise § 6 retired -- and one to "run
    prep_jobset" (W52).  A machine with a queue is another machine, and a
    calculation is set to the machine of its first prep (`configuration.md`
    M-3): the way there is the state saved before that prep."""
    return (f"{what}: there is no scheduler header ({name}) -- prep writes "
            f"one only where the machine it prepped for names a queue "
            f"(job-system.md § 6: a record saying `workstation`, or no "
            f"(partition, qos) pair to name).  Run it here with --mode "
            f"direct.  For a machine with a queue, prep it for that machine "
            f"(--target, its record's name) -- a calculation is set to the "
            f"machine of its first prep, so "
            + rollback("its first prep", base=base))


def _sbatch_request(base: Path, *, envelope: Resources, gpu_side: bool,
                    domain: Optional[str], mem: Optional[str],
                    time_s: Optional[int], label: str, job_name: str,
                    script: str) -> Tuple[Resources, object, List[str]]:
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
    placement = _place(base, gpu_side=gpu_side, needed_s=needed_s,
                       cores=(envelope.mpi_np or 0)
                             * max(envelope.cpus_per_task or 1, 1) or None,
                       mem=envelope.mem,
                       gpus=_gres_count(envelope.gres or ""),
                       named=domain, label=label)
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
            queues=[d.name for d in _rc.get_routing(project_dir=base)])
        if why:
            raise SubmitError(f"{label or 'this job'} {why}")
    cmd = (["sbatch", "-J", job_name]
           + _sbatch_resource_flags(envelope, placement)
           # The launch-door claim, EXPLICIT on the command line: environment
           # inheritance alone is fragile (sites override SLURM's --export
           # policy), and the flag wins over site defaults, so the claim
           # reaches the job wherever it runs (job-contracts.md § 2.6).
           + ["--export", "ALL,MB_LAUNCHED_BY=jobset-launch", script])
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


@dataclass
class _Plan:
    """One job of a launch, decided BEFORE anything is written: where it
    runs, where its deck and wrappers are read from now, what it continues
    from -- and the exact command, once every gate has passed."""
    job: Job
    #: The job's own directory: the stage's or the trial's container.
    container: Path
    #: Where it runs: its attempt -- for a continuation, the one it WILL
    #: open -- or, with no attempt layer, the container.
    run_dir: Path
    #: Whether the run has an attempt of its own (else the container is it).
    has_attempt: bool
    #: Where its deck and wrappers are read before it is sent: ``run_dir``,
    #: or for a continuation the attempt it continues -- the new one is
    #: filled with the same files (`materialize.prepare_attempt`), since a
    #: prep after a launch would have opened an attempt of its own.
    read_from: Path
    #: The launched attempt a ladder stage launched again continues from --
    #: "the natural workflow" (user, 2026-08-21).
    continues: Optional[str] = None
    #: The files that continuation carries, checked at planning.
    carries: List[str] = dataclasses.field(default_factory=list)
    #: A flat stage launched again: its files lie where it reads them.
    again: bool = False
    #: The conclusion line of what it follows -- ``None`` when that run was
    #: launched and never concluded, which the person judges, never
    #: molbuilder (`project-layout.md` § 1.6.4).
    concluded: Optional[str] = None
    #: Not launched, and why -- a trial already measured, under direct/ask.
    skip: Optional[str] = None
    command: List[str] = dataclasses.field(default_factory=list)
    placement: object = None
    #: The calculation's folder, so what the plan says names it.
    base: Optional[Path] = None

    @property
    def follows(self) -> bool:
        """It follows a launched run of its own stage."""
        return bool(self.continues or self.again)

    def judgement(self) -> Optional[str]:
        """What only the person can decide before this goes, or ``None``."""
        if not self.follows or self.concluded is not None:
            return None
        what = self.continues or "its last run, in this folder"
        return (f"{self.job.name}: {what} was launched and never "
                f"CONCLUDED -- it may still be RUNNING, or it was "
                f"force-stopped (walltime, kill).\n"
                f"  Continuing reads its warm files AS THEY ARE: valid after "
                f"a forced stop, torn if it is still running.  Check the "
                f"queue, and `{_cmd('status', self.job.name, base=self.base)}`, "
                f"first.")

    def note(self) -> Optional[str]:
        """The line that says what this launch follows, or ``None``."""
        if not self.follows:
            return None
        how = (f"concluded ({self.concluded})" if self.concluded is not None
               else "NOT concluded -- continued on your judgement")
        if self.again:
            return (f"{how}: launching it again in the same folder, where "
                    f"its files are")
        return (f"{how}: continuing {self.continues} -> {self.run_dir.name} "
                f"(carrying {', '.join(self.carries)}).  To start it afresh "
                f"instead (from the stage before it, or --cold from the "
                f"structure): " + rollback("its prep", base=self.base))


def _plan_job(jobset: JobSet, base: Path, job, *, mode: str) -> _Plan:
    """Where ``job`` runs and what it follows -- read, never written.

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
    A TRIAL is immutable once launched: under direct and ask the measured
    ones are passed over by name, and a named one is refused.
    """
    import functools
    from .commands import command
    from ..paths import attempts_in
    from .materialize import (bench_stage_of, continuation_files, job_dir_names, launch_record_at, shape_of)
    from ..runrecord import conclusion_line, launch_record_path, was_launched
    _plan = functools.partial(_Plan, base=base)
    sh = shape_of(jobset, base)
    container = base / job_dir_names(jobset, sh)[job.name]
    ns = attempts_in(container)
    stem = Path(job.script).stem
    if jobset.kind == "sweep":
        run = _trial_run_dir(container)
        where, basename = launch_record_at("sweep", job, container,
                                           run if ns else None)
        if not was_launched(where, basename):
            return _plan(job, container, run, bool(ns), run)
        if mode in ("direct", "ask"):
            return _plan(job, container, run, bool(ns), run,
                         skip=("already run" if mode == "ask"
                               else "skipped -- already launched"))
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
        if not was_launched(where, basename):
            return _plan(job, container, container, False, container)
        return _plan(job, container, container, False, container, again=True,
                     concluded=conclusion_line(container, stem))
    last = attempt_dir(container, ns[-1])
    if not was_launched(last):
        return _plan(job, container, last, True, last)
    source = str(last.relative_to(base))
    try:
        carries = continuation_files(jobset, base, job.name, source,
                                     named=False)
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
    return _plan(job, container, attempt_dir(container, ns[-1] + 1), True,
                 last, continues=source, carries=carries,
                 concluded=conclusion_line(last, stem))


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


def _job_wants_gpu(job_dir: Path, job) -> bool:
    """Whether this job asks for a GPU -- `runwrap._wants_gpu`, the one door
    (`gpu.md` G7), which reads ``resources.use_gpu`` first and the deck
    only for a job that states nothing.

    A sweep point that states ``gres`` outright is honoured unchanged -- the
    benchmark knows its own grid, and `bench/to_jobset.py` is where a GPU
    *count* is a swept parameter rather than a property of one deck.  *(This
    grepped a SIESTA keyword out of the deck until 2026-09-04, answering
    False for every PySCF GPU run and True for a SIESTA deck whose
    allocation said `use_gpu: false`.)*
    """
    if job.resources.gres:
        return True
    from ..runwrap import _wants_gpu                 # heavy; jobset stays light
    deck = Path(job_dir) / os.path.basename(job.script)
    return _wants_gpu(deck, job.resources)


def _send(jobset: JobSet, base: Path, p: _Plan, *, mode: str) -> List[JobResult]:
    """Send ONE planned job -- the first write of the launch.  A
    continuation opens its attempt only now, every refusal having already
    had its turn (W52: it was opened first until 2026-10-01, so a refusal
    after it left a fresh attempt behind)."""
    from .materialize import launch_record_at, prepare_attempt
    if p.skip:
        return [JobResult(p.job.name, [], p.skip)]
    out: List[JobResult] = []
    run = p.run_dir
    if p.continues:
        rep = prepare_attempt(jobset, base, p.job.name,
                              continue_from=p.continues, named=False)
        run = rep.dir
    if p.follows:
        out.append(JobResult(p.job.name, [], p.note()))
    where, basename = launch_record_at(jobset.kind, p.job, p.container,
                                       run if p.has_attempt else None)
    if p.again:
        # A FLAT STAGE LAUNCHED AGAIN continues from its own latest run,
        # where its files are (`job-system.md` § 5.0, row 7), and its record
        # says so -- as the hierarchy's next attempt names its own.  The
        # marker prep left names what the stage's FIRST launch continued
        # from, the stage before it, and was recorded again (W55 D5).
        from ..runfiles import latest_run, run_name
        from ..runrecord import continued_from_marker
        n = latest_run(where, basename)
        if n is not None:
            continued_from_marker(where, basename).write_text(
                run_name(basename, None, n) + "\n", encoding="utf-8")
    domain = getattr(getattr(p.placement, "domain", None), "name", None)
    if mode == "direct":
        # The launch-door claim rides the child ENV here: inheritance
        # survives forks and backgrounding, so a detached local run
        # launched through this verb never meets the gate's prompt.
        proc = subprocess.Popen(p.command, cwd=str(run),
                                env={**os.environ,
                                     "MB_LAUNCHED_BY": "jobset-launch"})
        # AT START, not after: run.json answers "was this launched?", and a
        # record written on completion left a running attempt reading as
        # never launched for its whole runtime.  A failed START records
        # nothing -- Popen raising means no process exists.
        _record_launch(where, mode="direct", command=p.command,
                       basename=basename)
        rc = proc.wait()
        out.append(JobResult(p.job.name, p.command,
                             "ran" if rc == 0 else "failed", returncode=rc))
        return out
    try:
        cp = subprocess.run(p.command, cwd=str(run), capture_output=True,
                            text=True,
                            env={**os.environ,
                                 "MB_LAUNCHED_BY": "jobset-launch"})
    except OSError as exc:
        raise SubmitError(
            f"job {p.job.name!r}: could not run {p.command[0]!r} ({exc})")
    if cp.returncode != 0:
        raise SubmitError(
            f"sbatch failed for job {p.job.name!r} (rc={cp.returncode}):\n"
            f"{cp.stderr.strip()}")
    jid = _parse_sbatch_id(cp.stdout)
    _record_launch(where, mode="submit", command=p.command, job_id=jid,
                   placement=p.placement, basename=basename)
    out.append(JobResult(p.job.name, p.command, "submitted", job_id=jid,
                         domain=domain))
    return out




def _group_envelope(jobs) -> "Resources":
    """The allocation one shelf's grouped job asks for.

    Since the shelf split (2026-08-21) every caller passes trials sharing
    ONE exact ask (`_shelf_key`: ranks, cores, gres), so this is the
    shelf's own ask read off its trials -- nothing is widened and nothing
    narrower exists inside a group.  The uniformity is ASSERTED rather
    than assumed: trials disagreeing here mean the shelf partition broke,
    and launching an allocation that fits only some of them would be the
    silent repair this module never makes.  *(Until the split this
    function computed the union of a whole side -- the widest trial's
    ranks/cores/devices -- and narrower trials idled the difference; the
    max() folds below survive as identities.)*
    """
    keys = {((j.resources.mpi_np or 1), (j.resources.cpus_per_task or 1),
             j.resources.gres or "") for j in jobs}
    if len(keys) > 1:
        raise SubmitError(
            "the group's trials do not share one resource ask "
            f"({sorted(keys)}) -- the per-shelf partition is broken "
            "(generator.md § 4.3a); this is a bug, not a declaration "
            "problem.")
    n = max((j.resources.mpi_np or 1) for j in jobs)
    c = max((j.resources.cpus_per_task or 1) for j in jobs)
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
    if len(mems) > 1 or len(times) > 1 or len(binds) > 1:
        raise SubmitError(
            f"the group's trials disagree about mem/time/gpu_binding "
            f"({sorted(mems, key=str)} / {sorted(times, key=str)} / "
            f"{sorted(binds, key=str)}) -- prep bakes one allocation over "
            f"a sweep, so this is a bug, not a declaration problem.")
    return Resources(mpi_np=n, cpus_per_task=c, gres=gres,
                     exclusive=exclusive, gpu_binding=next(iter(binds)),
                     mem=next(iter(mems)), time=next(iter(times)))


def _dc_replace_time(r: "Resources", time_str: str) -> "Resources":
    """The envelope with its wall set -- dataclasses.replace, named so the
    call site reads as what it does."""
    import dataclasses as _dc
    return _dc.replace(r, time=time_str)


def submit_bench_group(jobset: JobSet, base_dir, *,
                       gpu_domain: Optional[str] = None,
                       domain: Optional[str] = None,
                       dry_run: bool = False,
                       ask: bool = False,
                       trial_timeout_s: Optional[int] = None,
                       mem: Optional[str] = None,
                       time_s: Optional[int] = None,
                       only: Optional[str] = None) -> List[JobResult]:
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

    **The split** (§ 4.3a): trials partition by the DECK's own GPU answer
    (:func:`_job_wants_gpu`, the one door) -- a sweep whose trials all
    answer one way submits the single ``bench-group``; a sweep spanning
    both submits ``bench-group-cpu`` and ``bench-group-gpu``, so the CPU
    group's envelope asks no ``gres`` and devices are never held while CPU
    trials run.  The names come from the SET's composition, not from what
    is pending, so a side keeps its name across resubmissions.  ``domain``
    applies to both sides through `scheduler.place`; ``only`` (``"cpu"``/``"gpu"``)
    submits one side -- and a side this machine cannot launch simply stays
    pending for a later `launch bench`, which is the cross-cluster lane.

    The per-group pieces:

    * **the allocation** -- :func:`_group_envelope`: the shelf's own ask
      (identical across its trials by construction), sent through the one
      request (:func:`_sbatch_request`) -- admitted on its queue, the wall
      ``--time``, else what prep baked, else refused;
    * **the sequencer** -- ``launch/<name>.run.sh`` in the stage's bench
      container (the parent that sees every trial), regenerated from the
      trials STILL UNLAUNCHED at this submission.  Each trial runs in its own
      attempt through its own ``.run.sh`` (per-trial relabel, pins, monitor
      -- one home, untouched), under ``timeout`` when a per-trial bound is
      given; a trial that hits it is killed and reads ``incomplete``; the
      walk continues -- one bad point says nothing about the next.  It
      exits nonzero when any trial failed, so the scheduler's job state
      prompts a look at ``launch/<name>.log``;
    * **the launch records** -- every included trial's ``run.json`` is
      stamped with the ONE job id, so `status` and a later single-trial
      re-run see the truth.

    ``ask`` asks the scheduler about each shelf -- the jobs ``submit``
    sends, so a prediction is about the job that would go (W52: a sweep was
    asked about trial by trial while submit sent shelves).  Under ``ask``
    and ``dry_run`` nothing is written.
    """
    from .materialize import job_dir_names, shape_of
    from ..runrecord import was_launched
    dirs = job_dir_names(jobset, shape_of(jobset, base_dir))
    base = Path(base_dir)
    if only not in (None, "cpu", "gpu"):
        raise SubmitError(f"--only takes cpu or gpu, not {only!r}")
    sides = sides_of(jobset, base)
    if only and not sides[only]:
        raise SubmitError(f"this sweep has no {only} trials to submit")
    mixed = bool(sides["cpu"]) and bool(sides["gpu"])
    plans: List["_Prepared"] = []
    for side in ("cpu", "gpu"):
        jobs = sides[side]
        if not jobs or (only and side != only):
            continue
        # ONE GROUP PER RESOURCE SHELF (user, 2026-08-21: "lighter tasks
        # scheduled for heavy resource idling for hours is not a good use
        # of cpu time").  A single per-side group sized its allocation to
        # the WIDEST trial, so every narrower trial idled the difference
        # -- ~40% of the cores across a 32/64/128-rank matrix, and on the
        # GPU side idle DEVICES.  Trials sharing an exact resource ask
        # share one exact-fit allocation instead: nothing idles inside a
        # group, the value-axis cartesian still groups (its combos share
        # a shelf by construction, which is what keeps queue waits at
        # #shelves instead of #trials -- the 2026-08-20 grouping's point),
        # and the shelves submit WIDEST FIRST as independent jobs the
        # queue may even run concurrently.
        shelves: dict = {}
        for j in jobs:
            shelves.setdefault(_shelf_key(j), []).append(j)
        multi = len(shelves) > 1
        for key in sorted(shelves, key=_shelf_width, reverse=True):
            pending = [j for j in shelves[key]
                       if not was_launched(
                           _trial_run_dir(base / dirs[j.name]))]
            if not pending:
                continue            # this shelf already rode a group
            name = ("bench-group"
                    + (f"-{side}" if mixed else "")
                    + (f"-{_shelf_token(key, shelves[key])}"
                       if multi else ""))
            # ONE QUEUE PER SIDE.  A split sweep sends its CPU family and
            # its GPU family to different queues -- a cpu-only partition
            # cannot take the GPU side -- so one `--domain` cannot answer for
            # both, and using it for both is the guess this design removes.
            # `gpu_domain` REFINES rather than replaces: absent, the GPU
            # side takes `--domain` like everything else.  Requiring a second
            # flag from someone who already said `--only gpu` would be the
            # extra question this design exists to avoid; it is needed only
            # when the two sides genuinely go to different queues.
            plans.append(_prepare_side_group(
                jobset, base, dirs, pending, name,
                gpu_side=(side == "gpu"),
                domain=((gpu_domain or domain) if side == "gpu"
                        else domain),
                dry_run=(dry_run or ask),
                trial_timeout_s=trial_timeout_s,
                mem=mem, time_s=time_s))

    if not plans:
        from .materialize import bench_stage_of
        stage = bench_stage_of(base, base / dirs[jobset.jobs[0].name])
        raise SubmitError(
            f"all {len(sides[only]) if only else len(jobset.jobs)} "
            f"{only + ' ' if only else ''}trials are launched.  next:\n    "
            + (_cmd("summarize", "bench", stage, base=base) if stage else
               "`summarize bench` on the sweep's stage"))

    if ask:
        # EACH SHELF, as submit would send it.  Its own header is written
        # only when it is sent, so the question rides its first trial's --
        # rendered for the same work, and overruled by the same flags.
        out: List[JobResult] = []
        for n, p in enumerate(plans):
            if n >= ASK_MAX_QUERIES:
                out.append(JobResult(p.name, [], "not asked"))
                continue
            first = p.pending[0]
            out.append(_ask(p.name, p.cmd,
                            _trial_run_dir(base / dirs[first.name]),
                            domain=_domain_name(p),
                            script=_wrapper_name(first.script, ".sbatch")))
        return out

    if dry_run:
        # R14's prediction rides the shelves it is about: the note is a fact
        # about the DOMAIN, so every shelf on that domain carries the same
        # sentence and the caller dedupes -- the same shape the GPU-share
        # notes already use ("one warning per unstated fact, not per job").
        _caps = {}
        for _n in submitted_cap_notes(plans):
            _caps[_n.split(" takes ", 1)[0]] = _n
        return [r for p in plans
                for r in ([JobResult(p.name, p.cmd, "planned",
                                     domain=_domain_name(p),
                                     detail=_caps.get(_domain_name(p)))]
                          + [JobResult(j.name, [], "rides the group")
                             for j in p.pending])]

    # EVERY SHELF IS RENDERED BEFORE ANY IS SENT (above), AND ONE REFUSAL
    # DOES NOT CANCEL THE REST (here; user, 2026-08-30).  These are two
    # halves of one fault.  A Sol bench submitted its CPU group, had the
    # 4-GPU group refused -- *Requested node configuration is not
    # available*, for a card that partition does not stock -- and the
    # raise unwound the loop, so the 2-GPU group was neither written nor
    # sent.  Yet it was a perfectly valid ask, and independent: the
    # shelves are separate jobs the queue may even run concurrently.
    #
    # So a refusal is DATA on that shelf's result, not an exception over
    # the sweep.  Nothing is lost by continuing: the trials of a refused
    # shelf keep no launch record, so `was_launched` leaves them pending
    # and the next `launch bench` picks up exactly them.
    results: List[JobResult] = []
    refused: List[str] = []
    for p in plans:
        try:
            results += _launch_prepared(base, dirs, p)
        except SubmitError as exc:
            refused.append(p.name)
            results.append(JobResult(p.name, p.cmd, "sbatch refused",
                                     detail=str(exc)))
            results += [JobResult(j.name, [], "stays pending")
                        for j in p.pending]
    if refused and len(refused) == len(plans):
        # NOTHING went out.  There is no partial success to preserve, so
        # this is the plain failure it always was -- reported with every
        # shelf's reason rather than only the first.
        raise SubmitError(
            "no shelf was accepted by the scheduler:\n"
            + "\n".join(f"  {r.detail}" for r in results
                         if r.status == "sbatch refused"))
    return results


def _place(base: Path, *, gpu_side: bool, needed_s=None, cores=None,
           mem=None, gpus=None, named=None, label: str = ""):
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
    from ..scheduler import Request, parse_mem_gb
    from ..scheduler.place import place, Unplaceable
    # THE GPU COUNT, and no card (`scheduler.md` R2a): a GPU job goes to
    # a queue that has GPUs, where a node holds as many as were asked.
    want = Request(ranks=cores, cpus_per_task=1, gpus=gpus or None,
                   mem_gb=parse_mem_gb(mem), walltime_s=needed_s)
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
              "Change what is asked for (--time, --mem; the ranks and cores "
              "at prep), or name another of the record's queues with "
              "--domain.") from None


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
              f"and its limits are the ones enforced here.  Change what is "
              f"asked for (--time, --mem; the ranks and cores at prep), or "
              f"name another of the record's queues with --domain -- or, to "
              f"prepare against this machine's record (configuration.md "
              f"M-3), " + rollback("the calculation's first prep", base=base))


def _shelf_key(job: "Job"):
    """The exact resource ask that defines a group (§ 4.3a, 2026-08-21):
    trials grouped together must fit ONE allocation with nothing idle, so
    the key is everything the envelope would widen over."""
    r = job.resources
    return (r.mpi_np or 0, r.cpus_per_task or 0, r.gres or "")


def _gres_count(gres: str) -> int:
    """How many devices a ``--gres`` string asks for.

    Through `scheduler.quantities.parse_gres` -- the ONE reader of SLURM's
    gres spelling -- rather than the ``rsplit(":", 1)`` this was.  That
    read the last colon-separated token as the count, so
    ``gpu:a100:4,mps:400`` asked for four devices and reported 400, and
    the version-legal ``gpu:a100`` (one device, no count) raised and was
    caught as 1 by accident rather than by reading.
    """
    from ..scheduler.quantities import parse_gres
    return max(parse_gres(gres).values(), default=0)


def _shelf_width(key) -> tuple:
    """Widest-first order across shelves: cores, then devices."""
    n, c, gres = key
    return (n * max(c, 1), _gres_count(gres))


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
    n, c, gres = key
    g = _gres_count(gres)
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
    ``was_launched`` leaves them pending and the next ``launch bench``
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


@dataclass(frozen=True)
class _Prepared:
    """One shelf-job, rendered to disk and ready for `sbatch`.

    The submission is split in two -- render every shelf, THEN submit them
    -- because it used to render and submit each in turn, and a shelf whose
    sbatch failed took the loop down with it.  On 2026-08-30 a Sol bench
    left `launch/` holding two of its three script pairs: the CPU group had
    gone out, the 4-GPU group was refused by the scheduler, and the 2-GPU
    group had never been written, so the obvious recovery -- run the
    printed sbatch by hand -- answered *Unable to open file*.
    """
    name:      str
    cmd:       List[str]
    container: Path
    pending:   list
    #: The :class:`~molbuilder.scheduler.place.Placement` this shelf was
    #: bound to, or ``None`` when this machine has no menu (R6).  It carries
    #: the Domain itself, so the per-user job cap is already here -- which
    #: is why R14's check needed no new plumbing, only somebody to ask.
    placement: object
    gpu_side:  bool
    domain:    Optional[str]


def sides_of(jobset: JobSet, base_dir) -> Dict[str, List[Job]]:
    """A sweep's trials by the side they run on -- ``{"cpu": [...],
    "gpu": [...]}`` -- each trial's deck's own answer (:func:`_job_wants_gpu`,
    the one door), read where the trial RUNS (§ 1.6: the container holds no
    deck, so a trial that asks for a GPU without stating ``gres`` answered
    "cpu" by absence when the container was read).  The grouped door splits
    on it, and the launch verb asks it which queue each side needs."""
    from .materialize import job_dir_names, shape_of
    base = Path(base_dir)
    dirs = job_dir_names(jobset, shape_of(jobset, base))
    sides: Dict[str, List[Job]] = {"cpu": [], "gpu": []}
    for j in jobset.jobs:
        sides["gpu" if _job_wants_gpu(_trial_run_dir(base / dirs[j.name]), j)
              else "cpu"].append(j)
    return sides


def _prepare_side_group(jobset: JobSet, base: Path, dirs, pending,
                        name: str, *, gpu_side: bool,
                        domain: Optional[str], dry_run: bool,
                        trial_timeout_s: Optional[int],
                        mem: Optional[str] = None,
                        time_s: Optional[int] = None) -> "_Prepared":
    """One shelf's submission, checked, placed and WRITTEN -- but not sent.

    Every gate, the envelope, the placement and both scripts; the `sbatch`
    itself is :func:`_launch_prepared`.  Under ``dry_run`` nothing is
    written at all -- the flag's documented meaning is *print the exact
    command without launching* (`job-system.md` § 6), and the confirm
    preview walks this path before the person has said yes.

    Widest-first ordering lives one level up since the shelf split
    (2026-08-21): every trial in a group shares one exact resource ask by
    construction, so within a group the enumeration (declaration) order
    stands, and the SHELVES submit widest first.
    """

    # THE ENV-INHERITANCE SHIELD (user concern, 2026-08-20).  Inside the
    # allocation, SLURM_NTASKS / SLURM_CPUS_PER_TASK describe the ENVELOPE
    # (the widest trial), and the wrappers fall back to SLURM variables when
    # no flag is passed (running-a-job.md § 3.1-3.2) -- so a trial without
    # explicit knobs would silently measure the envelope's shape instead of
    # its own point.  Explicit -np/-omp flags win over every inherited
    # variable, so the sequencer passes both for every trial, and a trial
    # that cannot state them is refused BY NAME rather than mis-measured.
    unshaped = [j.name for j in pending
                if not (j.resources.mpi_np and j.resources.cpus_per_task)]
    if unshaped:
        raise SubmitError(
            "a grouped bench needs every trial's explicit rank/core shape "
            "(-np/-omp shield the trial from the allocation's SLURM_* "
            f"envelope); missing on: {', '.join(unshaped)}")

    # The deck/launch agreement gate guards THIS door too (review
    # 2026-08-21): a trial refused when submitted by name must not launch
    # silently by riding its group.  And the COLD gate (user, same day:
    # "it is the submission that determines the actual state of the
    # run"): the pin baked the intent at prep; here the artifact itself
    # is verified before it is launched.
    # WHERE THE DECK ACTUALLY IS.  A trial keeps attempts since 2026-08-27
    # (`project-layout.md` § 1.5a), so the deck sits in `run-<n>` and these
    # two gates read the container -- where they found NO deck, and
    # `check_trial_starts_cold`'s own doctrine is that *absence says
    # nothing*.  So the cold gate passed a WARM deck, silently, on the one
    # door that submits several trials at once, while the by-name door
    # still refused it.  Precisely the "guard-only-a-surface-applies"
    # failure this module names elsewhere; caught by
    # `test_submission_gates_the_cold_start_against_the_deck`.
    def _artifacts(j):
        return _trial_run_dir(base / dirs[j.name])

    for j in pending:
        try:
            check_launch_matches_deck(_artifacts(j), j)
            check_trial_starts_cold(_artifacts(j), j)
        except DeckLaunchMismatch as e:
            raise SubmitError(str(e)) from e

    # THE CONTAINER IS THE TRIAL'S PARENT, NOT THE ATTEMPT'S.  With an
    # attempt layer `_artifacts(j).parent` is `bench-<point>` -- one per
    # trial -- so the "they must share one container" check found N and
    # refused every grouped submission.  The container question belongs to
    # the naming authority (`dirs`), the artifacts question to the attempt;
    # they are two questions and this asks each of the right thing.
    containers = {(base / dirs[j.name]).parent for j in pending}
    if len(containers) != 1:
        raise SubmitError(
            "the sweep's trials do not share one container -- a grouped "
            f"submission needs the one parent that sees them all; found "
            f"{sorted(str(c) for c in containers)}")
    container = next(iter(containers))
    # L3 (roadmap 7.10, user 2026-08-24): the group's own machinery -- this
    # sequencer, its .sbatch, its log, and SLURM's stdout/err -- lives in
    # ``launch/`` beside the trial directories, not among them.  Made only
    # when the shelf is written (below): a dry run and a declined preview
    # leave nothing behind, an empty ``launch/`` included (W52).
    launch_dir = container / "launch"

    envelope = _group_envelope(pending)

    lines = [
        "#!/usr/bin/env bash",
        f"# {name}.run.sh -- ONE allocation, this shelf's unlaunched",
        "# trials in sequence (project-layout.md § 2.3.2, user 2026-08-20;",
        "# split per resource shelf 2026-08-21, generator.md § 4.3a).",
        "# Regenerated at each grouped submission.  THE TWO-LAYER MODEL",
        "# HOLDS (job-system.md § 6): this file is the launcher layer only",
        "# -- ordering and bounds.  Env activation and the engine launch",
        "# stay in each trial's own .run.sh, exactly as when a trial runs",
        "# alone; nothing here re-implements module load / source activate.",
        "set -u",
        f'LOG="launch/{name}.log"',
        f'echo "[group] $(date \'+%Y-%m-%dT%H:%M:%S\') start '
        f'trials={len(pending)} per-trial-bound='
        f'{f"{trial_timeout_s}s" if trial_timeout_s else "none"} '
        'job=${SLURM_JOB_ID:-none} node=$(hostname) '
        'alloc_ntasks=${SLURM_NTASKS:-unset} '
        'alloc_cpus=${SLURM_CPUS_PER_TASK:-unset}" >> "$LOG"',
        "fails=0",
        "run_trial() {",
        '    _name="$1"; _dir="$2"; shift 2',
        "    _t0=$(date +%s)",
        '    echo "[group] $(date \'+%Y-%m-%dT%H:%M:%S\') -> ${_name} starts" >> "$LOG"',
        (f'    ( cd "${{_dir}}" && timeout -k 30 {trial_timeout_s} '
         'bash "$@" ) >> "$LOG" 2>&1'
         if trial_timeout_s else
         '    ( cd "${_dir}" && bash "$@" ) >> "$LOG" 2>&1'),
        "    _rc=$?",
        '    if [ "${_rc}" -eq 124 ]; then',
        (f'        echo "[group] ${{_name}} hit the {trial_timeout_s}s '
         'per-trial bound -- killed; its artifacts read incomplete" >> "$LOG"'
         if trial_timeout_s else
         '        echo "[group] ${_name} killed (124)" >> "$LOG"'),
        "    fi",
        '    if [ "${_rc}" -ne 0 ]; then fails=$((fails+1)); fi',
        "    _t1=$(date +%s)",
        '    echo "[group] $(date \'+%Y-%m-%dT%H:%M:%S\') <- ${_name} '
        'finished rc=${_rc} took=$(( _t1 - _t0 ))s" >> "$LOG"',
        "    return 0    # one bad point says nothing about the next",
        "}",
    ]
    for j in pending:
        run_name = _wrapper_name(j.script, ".run.sh")
        # THE ATTEMPT, NOT THE TRIAL.  The wrapper lives in `run-<n>` since
        # the attempt layer landed (`project-layout.md` § 1.5a, 2026-08-27),
        # and this line went on naming the trial DIRECTORY -- so every
        # grouped bench `cd`ed one level too high and every trial died
        # instantly with *"No such file or directory"* (rc=127).  Sol job
        # 62372574, and every grouped bench since 2026-08-27.
        #
        # `_artifacts` is the one answer to "where are this trial's files",
        # and the gates above already ask it.  A ``trial_dirs`` list was
        # computed here for exactly this purpose, carrying the comment
        # *"the sequencer `cd`s into these, so they are the attempt too"* --
        # and nothing read it.  The intent was recorded, the value was
        # built, and the line that needed it went on computing its own.
        rel = _artifacts(j).relative_to(container)
        args = " ".join(_run_sh_args(j.resources))
        lines.append(
            f'run_trial "{j.name}" "{rel}" "{run_name}"'
            + (f" {args}" if args else ""))
    lines += [
        'echo "[group] $(date \'+%Y-%m-%dT%H:%M:%S\') done '
        'fails=${fails}" >> "$LOG"',
        'exit $(( fails > 0 ))',
        "",
    ]
    script = launch_dir / f"{name}.run.sh"
    # THE ONE REQUEST (`_sbatch_request`): prep's envelope, what was said at
    # launch, admitted on this side's queue (R9), every value stated.  The
    # side IS the GPU answer -- partitioned by the
    # deck's own word in `submit_bench_group`, so nothing is re-derived here.
    envelope, placement, cmd = _sbatch_request(
        base, envelope=envelope, gpu_side=gpu_side, domain=domain, mem=mem,
        time_s=time_s, label=name,
        job_name=_scheduler_job_name(jobset, name),
        script=f"launch/{name}.sbatch")

    prepared = _Prepared(name=name, cmd=cmd, container=container,
                         pending=list(pending), placement=placement,
                         gpu_side=gpu_side, domain=domain)
    if dry_run:
        return prepared

    from ..runwrap import _render_sbatch_for
    # Rendered at the BUNDLE's scope, not the container's (review
    # 2026-08-21): the render derives its config/environment scope from
    # the script path's parent, and the calculation's environment.json
    # lives at the bundle root.  The pair this submission
    # is ALREADY routing to is handed to the header emitter, so the .sbatch
    # and the `sbatch -p/-q` on the command line cannot name different
    # queues.  The stem alone names the delegated run script, so the
    # header still runs `bash {name}.run.sh` from the container.
    header = _render_sbatch_for(base / f"{name}.sh",
                                project_dir=base,
                                resources=envelope, env=None,
                                domain_pq=((placement.partition,
                                            placement.qos)
                                           if placement else None))
    if header is None:
        raise SubmitError(_no_sbatch(name, f"launch/{name}.sbatch",
                                     base=base))
    launch_dir.mkdir(parents=True, exist_ok=True)
    script.write_text("\n".join(lines), encoding="utf-8")
    (launch_dir / f"{name}.sbatch").write_text(_into_launch(header, name),
                                               encoding="utf-8")
    return prepared


def _launch_prepared(base: Path, dirs, prep: "_Prepared") -> List[JobResult]:
    """`sbatch` one prepared shelf, and stamp its trials' launch records.

    Separate from the render so that EVERY shelf is on disk before ANY is
    sent: a scheduler refusal then costs a submission, not the scripts of
    the shelves queued behind it.
    """
    name, cmd, container = prep.name, prep.cmd, prep.container
    gpu_side, domain, placement = prep.gpu_side, prep.domain, prep.placement
    pending = prep.pending
    cp = subprocess.run(cmd, cwd=str(container),
                        capture_output=True, text=True,
                        env={**os.environ,
                             "MB_LAUNCHED_BY": "jobset-launch"})
    if cp.returncode != 0:
        hint = ""
        if gpu_side and not domain:
            # Failure-time teaching, not a decision: when the default
            # directives cannot place a GPU group, name the menu rows
            # that could (generator.md § 4.3a) -- choosing one stays
            # the user's call, via --domain.
            from ..scheduler import domain_serves_gpu
            from .. import runtime_config as _rc
            able = [d.name for d in _rc.get_routing(project_dir=base)
                    if domain_serves_gpu(d)]
            if able:
                hint = (f"\n  The GPU group used the header's default "
                        f"directives; gpu-capable domains reachable here: "
                        f"{', '.join(able)} -- retry naming one with "
                        f"--domain.  "
                        f"The other side's launch stands; this side stays "
                        f"pending.")
        # Every shelf was rendered before any was sent, so the scripts a
        # by-hand retry needs are all on disk -- which they were not on
        # 2026-08-30, when this failure took its successor's .sbatch down
        # with it and `sbatch launch/...` answered "Unable to open file".
        hint += (f"\n  Every shelf's scripts are written under "
                 f"{container}/launch/ -- this one can be re-sent by hand "
                 f"once the ask fits.")
        raise SubmitError(
            f"sbatch failed for {name} (rc={cp.returncode}):\n"
            f"{cp.stderr.strip()}" + hint)
    jid = _parse_sbatch_id(cp.stdout)
    results = [JobResult(name, cmd, "submitted", job_id=jid)]
    for j in pending:
        # WHERE IT RAN, not the container.  `was_launched` reads the
        # attempt (`_trial_run_dir`), so recording in the container left
        # every grouped trial reading *never launched* -- and a re-launch
        # re-submitted work that had already measured its point.  The
        # single-job paths have always resolved this; only the grouped one
        # did not.
        _record_launch(_trial_run_dir(base / dirs[j.name]), mode="submit",
                       command=cmd, job_id=jid, placement=placement)
        results.append(JobResult(j.name, [], "rides the group",
                                 job_id=jid))
    return results


def submit_transport_chain(jobset: JobSet, base_dir, task, *,
                           mode: str, stage: str,
                           domain: Optional[str] = None,
                           dry_run: bool = False,
                           mem: Optional[str] = None,
                           time_s: Optional[int] = None
                           ) -> List[JobResult]:
    """ONE submission that walks a transport bias scan's points in
    order (`archive/2026-09-01-transport-design.md` § 4.3; layout ruled 2026-08-29: plain
    v-dirs, one attempt ladder per point).

    The walker is the launcher layer only, exactly like the bench
    group's sequencer: it ``cd``s into each point's prepared attempt and
    runs the point's own ``.run.sh`` — env activation and the engine
    launch stay where they always live.  What it adds is the WARM CHAIN:
    before each point after the first, the previous point's ``.TSDE``
    (the NEGF density) is copied forward, so ``V_{i+1}`` converges from
    ``V_i``'s state instead of from scratch.  And unlike the bench
    group it STOPS on a failed point: later points chain their density
    from this one, so walking on would converge from a state the
    failure poisoned — a benchmark's points are independent, a chain's
    are not.

    Every point's attempt must be OPEN and unlaunched (``prep run
    device`` opens them all); the deck/launch agreement gate guards this
    door like every other.  ``run.json`` lands in every point's attempt
    at start — they are all launched by this one command.  The job's
    request is the one every door sends (:func:`_sbatch_request`); ``ask``
    asks the scheduler about it, over the first point's own header, since
    the chain's is written only when it is sent.
    """
    from ..task import bias_token
    from ..transport.stages import rung_containers, scan_points
    from .materialize import latest_attempt
    from ..runrecord import was_launched

    if mode not in ("submit", "ask", "direct"):
        raise SubmitError(f"unknown mode {mode!r}: submit, ask or direct")
    if mode == "direct" and (domain or mem or time_s is not None):
        raise SubmitError(
            "--domain, --mem and --time are what a scheduler is asked for; "
            "'direct' runs it here, where none of them means anything.")
    points = scan_points(task, stage)
    if len(points) < 2:
        raise SubmitError("not a bias scan -- the plain launch owns "
                          "a single-point device.")
    # The device chain warm-hands the .TSDE and STOPS on failure
    # (later points inherit the failed state); the transmission walk is
    # the same one-submission sequence over INDEPENDENT points -- no
    # hand-forward, and a bad point says nothing about the next, so the
    # walk continues and the exit code reports any failure (P6).
    warm = stage == "device"
    job = next((j for j in jobset.jobs if j.name == stage), None)
    base = Path(base_dir).resolve()
    if job is None:
        raise SubmitError(
            f"the {stage} stage is not in the plan -- run "
            f"`{_cmd('prep', 'run', stage, base=base)}` first.")
    # The stage's folder, from the one door (`materialize.stage_home`).
    from .materialize import stage_home
    home = stage_home(base, task, stage)
    token, stage_dir = home.token, home.dir
    launch_dir = stage_dir / "launch"
    run_name = _wrapper_name(job.script, ".run.sh")
    name = f"{Path(job.script).stem}-chain"
    label = task.label

    attempts: List[Tuple[float, Path]] = []
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
        if was_launched(att):
            raise SubmitError(
                f"bias point {bias_token(v)}: {att.relative_to(base)} "
                f"has already been launched.  An attempt is immutable once "
                f"it has run, and a prepped stage is not prepped again: "
                + rollback("its prep", base=base))
        try:
            check_launch_matches_deck(att, job)
        except DeckLaunchMismatch as e:
            raise SubmitError(str(e)) from e
        attempts.append((v, att))

    args = " ".join(_run_sh_args(job.resources))
    lines = [
        "#!/usr/bin/env bash",
        f"# {name}.run.sh -- the bias chain: this scan's points in",
        "# sequence, each warm-started from the previous point's .TSDE",
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
        f'        if [ -f "$prev/{label}.TSDE" ]; then',
        f'            cp "$prev/{label}.TSDE" "$_dir/"',
        '            echo "[chain] ${_name}: warm from $prev" >> "$LOG"',
        "        else",
        f'            echo "[chain] ${{_name}}: no {label}.TSDE in '
        '$prev -- converging from scratch" >> "$LOG"',
        "        fi",
        "    fi",
    ] if warm else []) + [
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
    for v, att in attempts:
        # THE SAME IDIOM THE BENCH SEQUENCER USES.  Composing the path from
        # its parts was correct here and wrong there, and two spellings of
        # "where does this cd" is how the two came to disagree at all.
        rel = att.relative_to(stage_dir)
        lines.append(f'run_point "{bias_token(v)}" "{rel}" "{run_name}"'
                     + (f" {args}" if args else ""))
    lines += ['echo "[chain] $(date \'+%Y-%m-%dT%H:%M:%S\') done '
              'fails=${fails}" >> "$LOG"',
              'exit $(( fails > 0 ))', ""]

    if mode == "direct":
        cmd = ["bash", f"launch/{name}.run.sh"]
        if dry_run:
            return [JobResult(name, cmd, "planned")] + [
                JobResult(f"{stage}@{bias_token(v)}", [], "rides the chain")
                for v, _a in attempts]
        launch_dir.mkdir(parents=True, exist_ok=True)
        (launch_dir / f"{name}.run.sh").write_text("\n".join(lines),
                                                   encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(stage_dir),
                                env={**os.environ,
                                     "MB_LAUNCHED_BY": "jobset-launch"})
        # AT START, after the process exists -- the rule `_send` keeps: a
        # failed start records nothing (`project-layout.md` § 1.6.3).  Every
        # point was stamped before `Popen` until 2026-10-01 (W52).
        for _v, att in attempts:
            _record_launch(att, mode="direct", command=cmd)
        rc = proc.wait()
        return [JobResult(name, cmd, "ran" if rc == 0 else "failed",
                          returncode=rc)]

    # ---- submit / ask: one scheduler job, the group pattern in miniature #
    envelope, placement, cmd = _sbatch_request(
        base, envelope=job.resources,
        gpu_side=_job_wants_gpu(attempts[0][1], job), domain=domain,
        mem=mem, time_s=time_s, label=name,
        job_name=_scheduler_job_name(jobset, name),
        script=f"launch/{name}.sbatch")
    domain_name = getattr(getattr(placement, "domain", None), "name", None)
    if mode == "ask":
        return [_ask(name, cmd, attempts[0][1], domain=domain_name,
                     script=_wrapper_name(job.script, ".sbatch"))]
    if dry_run:
        return [JobResult(name, cmd, "planned", domain=domain_name)] + [
            JobResult(f"{stage}@{bias_token(v)}", [], "rides the chain")
            for v, _a in attempts]
    from ..runwrap import _render_sbatch_for
    header = _render_sbatch_for(base / f"{name}.sh", project_dir=base,
                                resources=envelope, env=None,
                                domain_pq=((placement.partition,
                                            placement.qos)
                                           if placement else None))
    if header is None:
        raise SubmitError(_no_sbatch(name, f"launch/{name}.sbatch",
                                     base=base))
    launch_dir.mkdir(parents=True, exist_ok=True)
    (launch_dir / f"{name}.run.sh").write_text("\n".join(lines),
                                               encoding="utf-8")
    (launch_dir / f"{name}.sbatch").write_text(_into_launch(header, name),
                                               encoding="utf-8")
    cp = subprocess.run(cmd, cwd=str(stage_dir), capture_output=True,
                        text=True,
                        env={**os.environ, "MB_LAUNCHED_BY": "jobset-launch"})
    if cp.returncode != 0:
        raise SubmitError(
            f"sbatch failed for {name} (rc={cp.returncode}):\n"
            f"{cp.stderr.strip()}")
    jid = _parse_sbatch_id(cp.stdout)
    results = [JobResult(name, cmd, "submitted", job_id=jid,
                         domain=domain_name)]
    for v, att in attempts:
        _record_launch(att, mode="submit", command=cmd, job_id=jid,
                       placement=placement)
        results.append(JobResult(f"{stage}@{bias_token(v)}", [],
                                 "rides the chain", job_id=jid))
    return results


def _placed_on(placement) -> Optional[dict]:
    """A `Placement` -> where this run was SENT, or ``None`` when there was
    no placement (a direct run).

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
            "qos": placement.qos}


def _record_launch(attempt: Path, *, mode: str, command: List[str],
                   job_id: Optional[str] = None, placement=None,
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
    from ..runrecord import continued_from_marker, write_run_launch
    src = None
    marker = continued_from_marker(attempt, basename)
    if marker.is_file():
        src = marker.read_text(encoding="utf-8").strip() or None
    write_run_launch(attempt, mode=mode, command=command, job_id=job_id,
                     continued_from=src, placed_on=_placed_on(placement),
                     basename=basename)


# --------------------------------------------------------------------- #
#  public entry point                                                   #
# --------------------------------------------------------------------- #

def _refuse_batch_submission(jobset: JobSet, base_dir: Path, *,
                             mode: str) -> None:
    """A scheduler is handed ONE job at a time (user rule, 2026-08-10).

    *"SLURM should never submit jobs in parallel.  Submission is manual and one
    by one.  It is a disaster to do parallel job submission on HPC."*  Firing N
    ``sbatch`` calls from one command puts N jobs in the queue that will start
    whenever the scheduler finds room -- together, if there is room -- and on a
    shared cluster that is antisocial at best.  For a **benchmark** it is worse:
    points that run concurrently contend for the same cores, memory bandwidth
    and interconnect, so the sweep measures contention rather than scaling and
    the numbers are quietly wrong.

    This is a rule about the **scheduler**, not about doing several things.
    ``--mode direct`` runs each job here, in order, waiting for each -- that is
    not submission at all, and it is untouched.

    **A second refusal stood here until 2026-08-10**: a hierarchical ladder
    could not be ``--chain``-ed, because continuing there means naming a run
    that has already finished and a chain has none to name.  It is gone
    because ``--chain`` is gone -- nothing chains, in either shape or either
    mode, so there is no longer a case to refuse.  What replaced it is
    ``_resolve_stage``, which will not act on a ladder at all without a named
    stage.
    """
    if len(jobset.jobs) <= 1:
        return

    # ``ask`` IS NOT GATED, and gating it was a misreading of this rule --
    # by its own words above: *a rule about the scheduler, not about doing
    # several things*.  `--test-only` enqueues nothing, so none of the harm
    # this prevents can happen: no job starts, nothing contends, nothing is
    # antisocial.
    #
    # And the sweep case is exactly where asking pays.  A grid's trials ask
    # for different shapes -- G1 schedules sooner than G4 -- so seeing all
    # of their waits side by side is what tells you which to submit.
    # Refusing that (caught by the user on a 4-trial bench, 2026-08-27) made
    # the feature useless precisely where it was most useful.
    #
    # The count is bounded by ASK_MAX_QUERIES instead, which is politeness
    # rather than a rule about queues.
    if mode == "submit" and len(jobset.jobs) > 1:
        names = ", ".join(j.name for j in jobset.jobs)
        from .materialize import bench_stage_of, job_dir_names, shape_of
        first = jobset.jobs[0].name
        if jobset.kind == "sweep":
            stage = bench_stage_of(base_dir, base_dir / job_dir_names(
                jobset, shape_of(jobset, base_dir))[first])
            one = (_cmd("launch", "bench", stage, first, base=base_dir,
                        flags=("--mode", "submit")) if stage else
                   f"`launch bench` on the sweep's stage, with {first}")
        else:
            one = _cmd("launch", "run", first, base=base_dir,
                       flags=("--mode", "submit"))
        raise SubmitError(
            f"refusing to hand {len(jobset.jobs)} jobs to the scheduler at "
            f"once ({names}).\n"
            "  Submission is one at a time, by hand.  Jobs queued together "
            "start together whenever the scheduler finds room, which on a "
            "shared cluster is antisocial -- and for a benchmark it is "
            "wrong, because points that run concurrently contend for the "
            "same cores and interconnect, so the sweep measures contention "
            "rather than scaling.\n"
            "  Name the one you mean -- the first, say:\n"
            f"    {one}\n"
            "  `--mode direct` is not affected: it runs them here, in order, "
            "waiting for each.")


def submit_jobset(jobset: JobSet, base_dir, *, mode: str,
                  domain: Optional[str] = None,
                  gpu_domain: Optional[str] = None,
                  dry_run: bool = False,
                  only: Optional[str] = None,
                  mem: Optional[str] = None,
                  time_s: Optional[int] = None,
                  continue_unconcluded: bool = False,
                  ) -> List[JobResult]:
    """Launch a prepped ``jobset`` rooted at ``base_dir`` -- the stage door,
    and a sweep's named trial or (direct) its trials in turn.

    ``mode`` is ``"submit"`` (SLURM ``sbatch``, its resources as flags over
    the rendered header), ``"ask"`` (``sbatch --test-only`` on the same
    line: when would it start) or ``"direct"`` (ordered local ``bash``).
    ``domain`` names the queue (submit and ask), and ``gpu_domain`` the one
    a sweep's GPU trial goes to when it differs -- where its shelf went, so a
    re-measured point lands with its group (review 2026-08-21); ``mem`` --
    SLURM text, ``0`` for the whole node -- and ``time_s`` are what the
    person said at launch.
    ``dry_run`` returns the exact command each job would get, writing
    nothing.  ``only`` names ONE job; for a ladder that is the only case --
    stages do not chain (`project-layout.md` § 1.6).

    **Every refusal before the first write** (W52): what each job follows,
    the deck/launch agreement, the queue's admission and the header are all
    decided first (:func:`_plan_job`, :func:`_sbatch_request`); only then is
    a continuation's attempt opened and the job sent (:func:`_send`).  A run
    that was launched and never concluded is followed only when
    ``continue_unconcluded`` records the person's judgement -- the CLI asks
    for it in the one question it puts before sending (`submission.md` S4).

    **There is no ``chain`` parameter** (deleted 2026-08-10, user): an
    opt-in typed before any stage has run is typed at the moment you know
    least.  :func:`_refuse_batch_submission` owns the standing rule that
    survives it -- a scheduler is handed one job at a time -- and lives here
    rather than in the CLI because a guard only a surface applies is one the
    next surface skips.
    """
    errs = jobset.validate()
    if errs:
        raise SubmitError(
            "refusing to submit an invalid JobSet:\n  - " + "\n  - ".join(errs))
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise SubmitError(f"base dir not found (prep first): {base}")
    if mode not in ("submit", "ask", "direct"):
        raise SubmitError(
            f"unknown mode {mode!r}: must be 'submit' (SLURM), 'ask' (submit "
            f"nothing, report when it would start) or 'direct' (local)")
    if mode == "direct" and (domain or gpu_domain or mem
                             or time_s is not None):
        raise SubmitError(
            "--domain, --mem and --time are what a scheduler is asked for; "
            "'direct' runs it here, where none of them means anything.")

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
        jobset = dataclasses.replace(
            jobset, jobs=[j for j in jobset.jobs if j.name == only])

    # The no-chain rule, AT THE SEAM (U5, 2026-08-12): a ladder is launched
    # one stage at a time, in EVERY mode -- direct running stages in order
    # would be local chaining.  A guard only a surface applies is one the
    # next surface forgets.
    if jobset.kind == "ladder" and len(jobset.jobs) > 1:
        raise SubmitError(
            "a ladder is launched ONE stage at a time; pass `only=<stage>`. "
            "Stages do not chain (project-layout.md § 1.6), in direct mode "
            "as much as submit: each stage is launched after you have "
            "looked at the one before it.")
    _refuse_batch_submission(jobset, base, mode=mode)

    plans = [_plan_job(jobset, base, j, mode=mode) for j in jobset.jobs]
    if not (dry_run or mode == "ask" or continue_unconcluded):
        for p in plans:
            if p.judgement():
                raise SubmitError(
                    p.judgement() + "\n  Then: re-run this launch with "
                    "--yes to record your judgement and continue.")

    # THE GATES AND THE EXACT COMMAND, for every job, before anything goes.
    sbatch_here = shutil.which("sbatch") is not None
    for p in plans:
        if p.skip:
            continue
        try:
            check_launch_matches_deck(p.read_from, p.job)
            if jobset.kind == "sweep":
                # the cold gate rides the named-trial door too (user,
                # 2026-08-21) -- one rule, every launch path
                check_trial_starts_cold(p.read_from, p.job)
        except DeckLaunchMismatch as e:
            # M5: the refusal is the launch's -- the agreement floor states
            # the fact, this verb is what declines to act on it.
            raise SubmitError(str(e)) from e
        if mode == "direct":
            run_name = _wrapper_name(p.job.script, ".run.sh")
            # The wrapper is required where it is RUN; a dry run prints the
            # command it would get (`job-system.md` § 6), as before.
            if not dry_run and not (p.read_from / run_name).exists():
                # THE PREP THAT WROTE IT is not run again: the way back is
                # the state saved before it (`job-system.md` § 5.0).
                what = ("the benchmark's prep" if jobset.kind == "sweep"
                        else "its prep")
                raise SubmitError(
                    f"job {p.job.name!r}: {run_name} is not in "
                    f"{p.read_from}, and a prepped stage is not prepped "
                    f"again: " + rollback(what, base=base))
            p.command = ["bash", run_name] + _run_sh_args(p.job.resources)
            continue
        sbatch_name = _wrapper_name(p.job.script, ".sbatch")
        # The header is required where it is SENT -- or ASKED about, when a
        # scheduler is here to ask: a question about a file that does not
        # exist answers nothing.  A dry run prints the command it would get.
        if ((mode == "submit" and not dry_run) or (mode == "ask"
                                                     and sbatch_here)) \
                and not (p.read_from / sbatch_name).exists():
            raise SubmitError(_no_sbatch(f"job {p.job.name!r}", sbatch_name,
                                         base=base))
        gpu = _job_wants_gpu(p.read_from, p.job)
        _env, p.placement, p.command = _sbatch_request(
            base, envelope=p.job.resources, gpu_side=gpu,
            domain=(gpu_domain if gpu and gpu_domain else domain),
            mem=mem, time_s=time_s, label=p.job.name,
            job_name=_scheduler_job_name(jobset, p.job.name),
            script=sbatch_name)

    def _domain(p):
        return getattr(getattr(p.placement, "domain", None), "name", None)

    if dry_run or mode == "ask":
        # A QUESTION MUST NOT WRITE (2026-08-28): asking -- or planning -- over
        # a launched stage opened run-<n+1> and copied its warm files, from a
        # run that could still be RUNNING, and the empty attempt then hid the
        # running one from `status`.  The would-be attempt is described, and
        # the question asked of the files it would be filled from.
        out: List[JobResult] = []
        asked = 0
        for p in plans:
            if p.skip:
                out.append(JobResult(p.job.name, [], p.skip))
                continue
            if p.follows:
                out.append(JobResult(
                    p.job.name, [],
                    (f"WOULD continue {p.continues} into {p.run_dir.name} "
                     f"(carrying {', '.join(p.carries)}), then launch it"
                     if p.continues else
                     "WOULD launch it again in the same folder, where its "
                     "files are"),
                    judgement=p.judgement()))
            if mode == "ask":
                if asked >= ASK_MAX_QUERIES:
                    # NO SILENT CAP: what was not asked is named.
                    out.append(JobResult(p.job.name, [], "not asked"))
                    continue
                asked += 1
                out.append(_ask(p.job.name, p.command, p.read_from,
                                domain=_domain(p)))
            else:
                out.append(JobResult(p.job.name, p.command, "planned",
                                     domain=_domain(p)))
        return out

    results: List[JobResult] = []
    for p in plans:
        results += _send(jobset, base, p, mode=mode)
    return results


__all__ = ["submit_jobset", "JobResult", "SubmitError"]
