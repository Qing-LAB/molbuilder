"""Submit engine — launch a *prepped* :class:`JobSet`
(docs/execution/job-system.md § 5.3-5.4; the wrapper contract is
job-contracts § 2).

``prep`` derives floor 3 from the described calculation and lays out the
tree (job-contracts § 6.3's naming); THIS engine launches ONE job per
invocation and renders nothing itself.  *(R8, 2026-08-12: this header
still introduced itself as "the keystone both producers feed", named a
bench migration that landed, a ``bench-<name>/`` tree that is now the
hand-built fallback only, and §§ 7-9 of a renumbered document.)*

Two execution paths, chosen by ``mode`` (== ``execution.mode``):

  * ``"submit"`` — SLURM.  Each job becomes an ``sbatch`` whose **per-job
    resources are CLI flags** (``-J``/``-n``/``-c``/``--gres``/``--mem``/
    ``-t``/``--exclusive`` + the domain's ``-p/-q``) over the (possibly
    shared) rendered ``.sbatch`` — exactly generalizing the benchmark launch
    line, so one rendered wrapper serves every point of a sweep.  **One job
    per invocation**, whatever the kind (§ 5.3, user rule 2026-08-10):
    handing a scheduler several at once is refused, because they would start
    together and — for a benchmark — measure contention rather than scaling.
  * ``"direct"`` — local shell, and NOT submission: each job's
    ``<stem>.run.sh`` is run in turn with its per-job knobs as args
    (``-np``/``-omp``), waiting for each, so nothing queues and nothing
    races.  Nothing is skipped on a failure either, because after 2026-08-10
    nothing here can depend on anything: a ladder arrives as ONE job, and a
    sweep's points are independent.

REUSE, not reinvention: prep renders via ``runwrap.write_run_wrapper``; this
engine adds only the cross-job concerns — per-job CLI overrides,
domain→``-p/-q`` resolution, and ordered local execution.

RESUME IS THE MODELING SOFTWARE'S JOB: this engine only launches.  It never
inspects prior output to auto-recover — the engine finds its own warm files
under the label it was given (``run-identity.md``), which `prep` copied into
the attempt; the decision to continue or switch stays the user's (assistant,
not nanny).  ``dry_run=True``
runs nothing: it returns the exact command line each job WOULD get, so the
plan is reviewable before anything is irreversible.
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
# authority (materialize.py) and very much alive -- six callers -- but
# the one place THIS module needs it, `_launch_dir`, imports it locally
# alongside three siblings that are not needed anywhere else.  A second,
# top-level import shadowed by that one sat here until 2026-08-10 doing
# nothing.
from .agreement import (DeckLaunchMismatch, check_launch_matches_deck,
                        check_trial_starts_cold)
from .model import Job, JobSet, Resources


class SubmitError(Exception):
    """A JobSet could not be submitted (bad mode, unknown domain, missing
    prepped wrapper, or sbatch failure)."""


@dataclass
class JobResult:
    """What happened to one job.  ``command`` is always populated (the exact
    line that ran / would run).  In ``submit`` mode ``job_id`` is the SLURM
    id; in ``direct`` mode ``returncode`` is the process exit status.
    ``status`` is one of ``submitted`` / ``ran`` / ``failed`` / ``skipped``
    / ``planned`` (dry-run) / ``asked`` (``--mode ask``: nothing was
    submitted and ``prediction`` carries what the scheduler said)."""
    name:       str
    command:    List[str]
    status:     str
    job_id:     Optional[str] = None
    returncode: Optional[int] = None
    #: Only in ``ask`` mode.  ``None`` everywhere else, and a ``Prediction``
    #: whose ``start`` is ``None`` when SLURM declined to predict -- which
    #: is reported as *unknown*, never as *soon*.
    prediction: Optional[object] = None

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


def _scheduler_job_name(jobset: JobSet, job) -> str:
    """What `squeue` shows: the calculation, then the stage within it.

    ``<jobset.name>/<job.name>`` -- the id first, because that is the thing you
    are trying to tell apart when several calculations are queued at once, and
    the stage second because within one calculation that is the question.

    A sweep's points and a ladder's stages both go through this, so one reading
    covers both: `bdt_au/coarse`, `bdt_au/G1K2C4`.
    """
    return f"{jobset.name}/{job.name}"


#: How many `sbatch --test-only` calls one `--mode ask` will make.
#:
#: Politeness, not a rule about queues: asking enqueues nothing, so the
#: one-at-a-time submission rule does not reach it.  A benchmark grid is
#: the case that matters and is usually a handful of trials; past this the
#: rest are named as unasked rather than silently dropped.
ASK_MAX_QUERIES = 24


def _submit_slurm(jobset: JobSet, base_dir: Path, *, domain: Optional[str],
                  dry_run: bool,
                  mem_gb: Optional[float] = None,
                  time_s: Optional[int] = None,
                  ask: bool = False) -> List[JobResult]:
    """SLURM path: ``sbatch`` **one** job, with its resources as CLI flags.

    **One per invocation** — :func:`_refuse_batch_submission` is what makes
    that true, and this loop keeps its shape only because a caller may narrow
    to a single job by several routes.  Until 2026-08-10 this docstring said
    *"a sweep submits with no dependency, so its jobs queue in parallel"*, and
    the code did exactly that: one command, N ``sbatch`` calls, every one of
    them racing the others for the same nodes.

    **No ``--dependency`` flag is ever emitted.** It was threaded here from
    ``job.depends_on`` until 2026-08-10; the field is deleted, so there is no
    edge to thread and nothing this loop can queue behind anything else.
    """
    results: List[JobResult] = []
    asked = 0                      # bounded by ASK_MAX_QUERIES
    if ask:
        from .materialize import job_dir_names, shape_of, was_launched
        _dirs = job_dir_names(jobset, shape_of(jobset, base_dir))
    for job in jobset.jobs:
        # ASKING ABOUT A TRIAL THAT ALREADY RAN IS NOT RE-RUNNING IT, and
        # this has to come BEFORE `_resolve_launch`, which is what raises.
        # § 1.5's immutability protects RESULTS -- a trial measures its
        # point once -- and a question creates nothing to protect them
        # from.  Refusing here killed the whole command over a trial
        # nobody proposed to run again.
        #
        # The useful case IS the part-run sweep: two of four done, when
        # would the other two start?  Skipped BY NAME, the way a direct
        # run already reports them.
        if ask and jobset.kind == "sweep" and was_launched(
                _trial_record_dir(base_dir / _dirs[job.name])):
            results.append(JobResult(job.name, [], "already run"))
            continue
        job_dir, attempt, sbatch_name = _resolve_launch(
            jobset, base_dir, job, ".sbatch")
        gpu = _job_wants_gpu(job_dir, job)
        # ONE WALK, both paths (2026-08-23).  A re-measured trial must land
        # where its group did (review 2026-08-21): without this the group
        # followed the per-side preference while a named trial fell to the
        # header's defaults -- a different partition, silently breaking the
        # compare-by-node-type premise.  A ladder keeps the defaults, so it
        # asks for no placement at all rather than getting a different one.
        placement = None
        if domain or jobset.kind == "sweep":
            placement = _place(Path(base_dir), gpu_side=gpu, named=domain,
                               label=job.name)

        # WHICH CALCULATION, then which stage.  `-J` carried the bare stage
        # name until 2026-08-10, so three concurrent ladders showed
        # `coarse coarse coarse` in `squeue` and no amount of looking told you
        # which was which -- the one place a scheduler shows you your own work,
        # and it showed you nothing.  `JobSet.name` is the id (run-identity.md
        # § 2), so it goes first and the stage qualifies it.
        cmd = ["sbatch", "-J", _scheduler_job_name(jobset, job)]
        resources = job.resources
        if time_s:
            import dataclasses as _dc
            resources = _dc.replace(resources, time=_slurm_time(time_s))
        if mem_gb:
            # Same gap as the grouped path (`_submit_side_group`), same
            # fix: nothing ever set `Resources.mem` for a single-job
            # submission either -- `job.resources` is whatever `prep`
            # baked (mpi_np/omp/gres from the sweep axes; never memory),
            # so `--mem` typed at `launch` was silently discarded here
            # too.  Overridden here rather than on `job.resources` itself
            # so the job's OWN prepped record is untouched -- only what
            # gets submitted changes.
            import dataclasses as _dc
            resources = _dc.replace(resources, mem=f"{mem_gb:g}G")
        cmd += _sbatch_resource_flags(resources, placement)
        # The launch-door claim, EXPLICIT on the command line: environment
        # inheritance alone is fragile (sites override SLURM's --export
        # policy), and the CLI flag wins over site defaults, so the claim
        # reaches the job's env wherever it runs (job-contracts.md § 2.6,
        # the Launch-door gate row).
        cmd += ["--export", "ALL,MB_LAUNCHED_BY=jobset-launch"]
        cmd.append(sbatch_name)          # relative; we cd into the job dir

        if ask and asked >= ASK_MAX_QUERIES:
            # NO SILENT CAP: what was not asked is named, or a partial
            # answer reads as a complete one.
            results.append(JobResult(job.name, [], "not asked"))
            continue
        if ask:
            asked += 1
            # `--test-only` goes FIRST so it cannot be shadowed, and the
            # rest of the line is untouched: the whole point is that the
            # question is about the command that would actually be sent.
            cmd = [cmd[0], "--test-only"] + cmd[1:]
            # NO SCHEDULER HERE IS AN ANSWER, NOT AN ERROR.  On a
            # workstation there is nothing to queue behind, so the honest
            # reply is "nothing to wait for" -- and `subprocess.run` would
            # otherwise raise FileNotFoundError and traceback, which is a
            # poor way to learn that your laptop is not a cluster.
            if shutil.which(cmd[0]) is None:
                from .ask import Prediction
                results.append(JobResult(
                    job.name, cmd, "asked",
                    prediction=Prediction(no_scheduler=True,
                                          refused="no scheduler here")))
                continue

        if dry_run:
            results.append(JobResult(job.name, cmd, "planned"))
            continue
        if not (job_dir / sbatch_name).exists():
            raise SubmitError(
                f"job {job.name!r}: {sbatch_name} not in {job_dir}. "
                "Run prep first (`molbuilder jobset prep`); if you already "
                "did, no scheduler is configured -- submit mode needs one "
                "(add a scheduler block to .molbuilder.json, or use "
                "--mode direct).")
        try:
            cp = subprocess.run(cmd, cwd=str(job_dir),
                                capture_output=True, text=True,
                                env={**os.environ,
                                     "MB_LAUNCHED_BY": "jobset-launch"})
        except OSError as exc:
            # `sbatch` vanished between the check above and here, or is not
            # executable.  A submission must fail loudly; an ASK reports it
            # as the answer, because "I could not reach the scheduler" is
            # what the person asked about.
            if not ask:
                raise SubmitError(
                    f"job {job.name!r}: could not run {cmd[0]!r} ({exc})")
            from .ask import Prediction
            results.append(JobResult(job.name, cmd, "asked",
                                     prediction=Prediction(
                                         refused=f"could not run "
                                                 f"{cmd[0]!r}: {exc}")))
            continue
        if ask:
            # NOTHING WAS SUBMITTED, so nothing is recorded: a launch record
            # says a job exists, and after this one does not.  A non-zero
            # return is not a failure here either -- "this queue cannot take
            # it" is an answer, and often the one worth reading.
            from .ask import parse_test_only
            pred = parse_test_only((cp.stdout or "") + (cp.stderr or ""))
            results.append(JobResult(job.name, cmd, "asked",
                                     prediction=pred))
            continue
        if cp.returncode != 0:
            results.append(JobResult(job.name, cmd, "failed",
                                     returncode=cp.returncode))
            raise SubmitError(
                f"sbatch failed for job {job.name!r} (rc={cp.returncode}):\n"
                f"{cp.stderr.strip()}")
        jid = _parse_sbatch_id(cp.stdout)
        rec = attempt if attempt is not None else (
            job_dir if jobset.kind == "sweep" else None)
        if rec is not None:
            _record_launch(rec, mode="submit", command=cmd, job_id=jid,
                           placement=placement)
        results.append(JobResult(job.name, cmd, "submitted", job_id=jid))
    return results


def _resolve_launch(jobset: JobSet, base_dir: Path, job, suffix: str):
    """The prologue both launch paths share: **where**, **which wrapper**, and
    **may this deck be launched like that**.

    *Named ``_staged_for_launch`` for about a minute, until
    `test_stage_vocabulary` refused it: it works for a sweep point as much as
    a ladder stage, so borrowing the project's core noun for "set up" is the
    collision that ledger exists to catch (§ 8c question 1).*

    § 9.4 names the defect this removes: *"`submit.py` grew `_launch_dir` and
    `_record_launch` **twice** — once in each of two near-identical loops"*.
    Both were called with identical arguments in both paths, so a fix to either
    had two sites and one of them would eventually be missed.

    Returns ``(job_dir, attempt, wrapper_name)``. ``attempt`` is ``None`` for a
    job with no attempt layer — a sweep point, or a ladder stage prepped
    without ``prep run`` — and that is the caller's signal not to write a
    ``run.json``.

    **The two loops are NOT merged**, and that is a judgement rather than an
    omission. § 9.6 lists *"one loop in `submit.py`"* as this object's gain;
    what it is actually against is the duplicated calls, and those are gone.
    The middles genuinely differ — one parses a scheduler id and raises, the
    other reads an exit status and propagates failure down the ladder — so
    folding them into a single loop would put a ``mode`` branch through the
    body and rebuild the shape the split avoids. *If that reading is wrong,
    this is the paragraph to argue with.*
    """
    job_dir, attempt = _launch_dir(jobset, base_dir, job)
    try:
        check_launch_matches_deck(job_dir, job)
        if jobset.kind == "sweep":
            # the cold gate rides the named-trial door too (user,
            # 2026-08-21) -- one rule, every launch path
            check_trial_starts_cold(job_dir, job)
    except DeckLaunchMismatch as e:
        # M5: the refusal is SUBMIT's -- the agreement floor states the
        # fact, this verb is what declines to act on it.
        raise SubmitError(str(e)) from e
    return job_dir, attempt, _wrapper_name(job.script, suffix)


def _trial_record_dir(container):
    """Where a trial's ``run.json`` is — the attempt, or the trial itself.

    `submit` writes the launch record where the job RAN, and since
    2026-08-27 a trial runs in its attempt when the shape keeps one
    (`project-layout.md` § 1.5a).  Asking the container instead reads
    *never launched* for a trial that has run, which is how a direct sweep
    came to re-run everything it had already measured.

    The same *latest attempt where there is one* rule `runstatus` and
    `summarize` use, in the one place `submit` needs it.
    """
    from .materialize import latest_attempt
    return latest_attempt(container) or container


def _launch_dir(jobset: JobSet, base_dir: Path, job) -> Tuple[Path, Optional[Path]]:
    """Where this job runs, and the attempt to record the launch into.

    Two layouts meet here.  A SWEEP has no attempt layer -- ``prep`` lays out
    ``bench-<name>/`` and the point runs there, as it always has.  A LADDER
    stage prepped with ``jobset prep run <stage>`` has ``<seq>_<name>/run-<n>/``,
    and that is where it runs, because an attempt is immutable once it has run
    (``project-layout.md`` § 1.5) and a re-run must not land on top of one.

    Refuses an attempt that has already been launched.  ``run.json`` is the only
    honest answer to *has this started?* -- a queued job has produced nothing
    yet, so absence of output proves nothing (§ 1.6).
    """
    from .materialize import (RUN_LAUNCH_FILE, attempts, job_dir_names,
                              shape_of, was_launched)
    sh = shape_of(jobset, base_dir)
    d = base_dir / job_dir_names(jobset, sh)[job.name]
    ns = attempts(d)
    if not ns:
        # C5 (2026-08-12, R2's missing half): a HIERARCHICAL ladder stage
        # with no attempt open used to fall through to (d, None) -- it
        # launched in its own container, wrote no run.json, and was
        # silently relaunchable, everything § 1.5/1.6 exist to prevent.
        # Only that case refuses: a sweep trial IS its own attempt (below),
        # and flat keeps no attempt directories at all.
        if (jobset.kind == "ladder" and sh is not None
                and sh.keeps_attempts_as_directories):
            raise SubmitError(
                f"job {job.name!r}: no attempt is open under {d.name}/ -- "
                f"a hierarchical stage runs in run-<n>, never in its own "
                f"container (project-layout.md § 1.5, 1.6).  Open one:\n"
                f"    molbuilder jobset prep run {job.name}")
        # An attempt-less dir (a sweep trial) is ITS OWN attempt, and
        # § 1.5's immutability applies to it the same way (R2, 2026-08-12:
        # until then a named trial -- or any direct re-invocation -- was
        # silently relaunchable in place, run.json overwritten; the rule
        # held only through the CLI's bare-form next-unlaunched skip, the
        # guard-only-a-surface-applies pattern this module names below).
        if was_launched(d):
            raise SubmitError(
                f"job {job.name!r}: already launched -- "
                f"{(d / RUN_LAUNCH_FILE)} records it.  A trial measures "
                f"its point ONCE (project-layout.md § 1.5: immutable once "
                f"it has run); read the sweep back with `molbuilder "
                f"jobset summarize bench <stage>`.  To measure this point "
                f"again, move the trial's directory aside yourself -- "
                f"molbuilder never deletes results.")
        return d, None
    last = d / f"run-{ns[-1]}"
    if was_launched(last):
        # A TRIAL IS REFUSED FOR THE SAME REASON AND ADVISED DIFFERENTLY.
        # `--from <attempt>` is a STAGE's remedy: it continues from what
        # the last attempt produced.  A trial does not continue -- it
        # measures its point again, from cold, and since 2026-08-27 it can
        # (`project-layout.md` § 1.5a): preparing opens `run-<n+1>` beside
        # this one and leaves it exactly as it is.
        #
        # Before that, this branch never saw a trial and the advice was
        # right for everything that reached it.
        if jobset.kind == "sweep":
            raise SubmitError(
                f"trial {job.name!r}: {last.name} has already been launched "
                f"({last / 'run.json'}).  A measurement is immutable once it "
                f"has run.\n"
                f"  read what it measured:  molbuilder jobset summarize "
                f"bench <stage>\n"
                f"  measure the point AGAIN: molbuilder jobset prep bench "
                f"<stage>  (opens {d.name}/run-{ns[-1] + 1}, leaving "
                f"{last.name} untouched)")
        raise SubmitError(
            f"job {job.name!r}: {last.name} has already been launched "
            f"({last / 'run.json'}).  An attempt is immutable once it has run; "
            f"prepare a fresh one:\n"
            f"    molbuilder jobset prep run {job.name} --from <attempt>")
    return last, last


def _job_wants_gpu(job_dir: Path, job) -> bool:
    """Whether this job asks for a GPU — **from its deck**, not from `gres`.

    `job-contracts.md § 6.2` derives the GPU request *"from `.fdf` + GPU type"*,
    and the two halves live in different places on purpose:

    * the **`.fdf`** carries the decision (`engines/stages.md § 5`: a GPU choice
      lands in the deck **and** the wrapper's env routing **and** a scheduler's
      `--gres`), and it travels with the bundle;
    * the **GPU type** is a fact about the cluster, and `job-system.md`
      decision #3 (*target isolation*) keeps cluster facts out of what you
      produce on a laptop.

    So the ladder producer leaves ``gres`` unset **and is right to** —
    ``siesta/stages.py`` says so in as many words: *"scheduler resources
    (domain/time/exclusive/mem/gres) resolve at submit."*  What was missing is
    that nothing here asked the deck instead: this read
    ``bool(job.resources.gres)``, which is always false for a ladder, so **a
    stage whose deck selects a GPU eigensolver was routed to the CPU
    partition.**  Its rendered ``.sbatch`` header already carried the right
    ``--gres`` (``runwrap`` derives it on the target from the same deck), so the
    job asked for a GPU on a partition that has none.

    A sweep point that states ``gres`` outright is honoured unchanged — the
    benchmark knows its own grid, and `bench/to_jobset.py` is where a GPU
    *count* is a swept parameter rather than a property of one deck.
    """
    if job.resources.gres:
        return True
    from ..runwrap import _fdf_requests_gpu          # heavy; jobset stays light
    deck = Path(job_dir) / os.path.basename(job.script)
    return deck.is_file() and _fdf_requests_gpu(deck)


def _run_direct(jobset: JobSet, base_dir: Path, *,
                dry_run: bool) -> List[JobResult]:
    """Local path: run each ``<stem>.run.sh`` here, in order, waiting for each.

    **Nothing is skipped because something else failed**, and after 2026-08-10
    nothing can be: a LADDER never reaches this loop with more than one job
    (`_resolve_stage` refuses to act on a ladder without a named stage), and a
    SWEEP's points are independent by definition, so one bad point says nothing
    about the next.  A `_blocked_by_a_failure` helper stood here to reproduce
    SLURM's ``afterok`` meaning locally; it went with the edges.
    """
    results: List[JobResult] = []
    from .materialize import job_dir_names, shape_of, was_launched
    dirs = job_dir_names(jobset, shape_of(jobset, base_dir))
    for job in jobset.jobs:
        # 2026-08-12 plan A6: a direct SWEEP resumes past what already ran.
        # A trial is immutable once launched (§ 1.5), and the submit
        # path's next-unlaunched pick already skips it -- but direct runs
        # the set in order, so without this the loop DIED at the first
        # launched trial and an interrupted sweep could never finish.
        # The skip is said out loud in the results.  Ladder stages keep
        # the refusal below: their re-run is a NEW attempt the user opens.
        if jobset.kind == "sweep" and was_launched(
                _trial_record_dir(base_dir / dirs[job.name])):
            results.append(JobResult(job.name, [],
                                     "skipped -- already launched"))
            continue
        job_dir, attempt, run_name = _resolve_launch(
            jobset, base_dir, job, ".run.sh")
        cmd = ["bash", run_name] + _run_sh_args(job.resources)
        if dry_run:
            results.append(JobResult(job.name, cmd, "planned"))
            continue
        if not (job_dir / run_name).exists():
            raise SubmitError(
                f"job {job.name!r}: {run_name} not in {job_dir} "
                "(run prep_jobset first).")
        # The launch-door claim rides the child ENV here: inheritance
        # survives forks and backgrounding, so a detached local run
        # launched through this verb never meets the gate's prompt.
        proc = subprocess.Popen(cmd, cwd=str(job_dir),
                                env={**os.environ,
                                     "MB_LAUNCHED_BY": "jobset-launch"})
        # AT START, not after: run.json answers "was this launched?", and a
        # record written on completion left a running attempt reading as
        # never launched for its whole runtime.  A failed START still
        # records nothing -- Popen raising means no process exists, and the
        # attempt is exactly as prepare left it (§ 1.6).
        rec = attempt if attempt is not None else (
            job_dir if jobset.kind == "sweep" else None)
        if rec is not None:
            _record_launch(rec, mode="direct", command=cmd)
        rc = proc.wait()
        if rc != 0:
            results.append(JobResult(job.name, cmd, "failed",
                                     returncode=rc))
        else:
            results.append(JobResult(job.name, cmd, "ran", returncode=0))
    return results




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
    if len(mems) > 1 or len(times) > 1:
        raise SubmitError(
            f"the group's trials disagree about mem/time "
            f"({sorted(mems)} / {sorted(times)}) -- prep bakes one "
            f"allocation over a sweep, so this is a bug, not a "
            f"declaration problem.")
    return Resources(mpi_np=n, cpus_per_task=c, gres=gres,
                     exclusive=exclusive,
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
                       trial_timeout_s: Optional[int] = None,
                       mem_gb: Optional[float] = None,
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
    applies to both sides through `scheduler.place` (a GPU side
    prefers the domain's ``gpu_partition``); ``only`` (``"cpu"``/``"gpu"``)
    submits one side -- and a side this machine cannot launch simply stays
    pending for a later `launch bench``, which is the cross-cluster lane.

    The per-group pieces:

    * **the allocation** -- :func:`_group_envelope`: the shelf's own ask
      (identical across its trials by construction); wall = pending x
      ``trial_timeout_s`` x 1.1 plus five minutes of startup margin;
    * **the sequencer** -- ``bench-group.run.sh``, written into the stage's
      ``bench/`` container (the parent that sees every trial), regenerated
      from the trials STILL UNLAUNCHED at this submission.  Each trial runs
      in its own directory through its own ``.run.sh`` (per-trial relabel,
      pins, monitor -- one home, untouched) under ``timeout``; a trial that
      hits the bound is killed and its artifacts read ``incomplete``; the
      walk continues -- one bad point says nothing about the next.  The
      script exits nonzero when any trial failed, so the scheduler's job
      state prompts a look at ``bench-group.log``;
    * **the launch records** -- every included trial's directory gets its
      ``run.json`` stamped with the ONE job id, so `status`, the picker and
      a later single-trial re-run all see the truth.
    """
    from .materialize import (job_dir_names, latest_attempt,
                              shape_of, was_launched)
    dirs = job_dir_names(jobset, shape_of(jobset, base_dir))
    base = Path(base_dir)
    if only not in (None, "cpu", "gpu"):
        raise SubmitError(f"--only takes cpu or gpu, not {only!r}")
    sides = {"cpu": [], "gpu": []}
    for j in jobset.jobs:
        sides["gpu" if _job_wants_gpu(base / dirs[j.name], j)
              else "cpu"].append(j)
    if only and not sides[only]:
        raise SubmitError(f"this sweep has no {only} trials to submit")
    mixed = bool(sides["cpu"]) and bool(sides["gpu"])
    results: List[JobResult] = []
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
                           _trial_record_dir(base / dirs[j.name]))]
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
            results += _submit_side_group(
                jobset, base, dirs, pending, name,
                gpu_side=(side == "gpu"),
                domain=((gpu_domain or domain) if side == "gpu"
                        else domain),
                dry_run=dry_run,
                trial_timeout_s=trial_timeout_s,
                mem_gb=mem_gb, time_s=time_s)
    if not results:
        raise SubmitError(
            f"all {len(sides[only]) if only else len(jobset.jobs)} "
            f"{only + ' ' if only else ''}trials are launched.  next: "
            f"molbuilder jobset summarize bench <stage>")
    return results


def _place(base: Path, *, gpu_side: bool, needed_s=None, cores=None,
           mem_gb=None, gpus=None, named=None, label: str = ""):
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
    want = Request(ranks=cores, cpus_per_task=1, gpus=gpus or None,
                   mem_gb=parse_mem_gb(mem_gb), walltime_s=needed_s)
    try:
        # WHICH AXIS DECIDES between queues that all fit is the site's to
        # say (`scheduler.placement_priority`, 2026-08-23).  Absent, `place`
        # supplies its own default and the display names it as a default --
        # the config does not pretend to have chosen.
        _sched = _rc.get_scheduler(project_dir=base) or {}
        placed = place(_rc.get_routing(project_dir=base), want,
                       prefer_gpu=gpu_side, named=named,
                       priority=_sched.get("placement_priority"))
        # R9's SECOND record.  Routing reads the calculation scope first, so
        # a prepped bundle routes against the snapshot beside it -- which is
        # right for reproducibility and useless as a re-check, because it is
        # the same record the request was built against.  What will actually
        # enforce the limits is THIS machine's own record, so the re-admission
        # reads that one.  When the two agree this costs a read; when they
        # disagree it is the whole point of the rule.
        if placed is not None:
            _reject_if_this_machine_says_no(placed, want, gpu_side, label)
        return placed
    except Unplaceable as exc:
        raise SubmitError(
            f"{label or 'this group'} cannot be placed on this machine:\n    "
            + "\n    ".join(exc.reasons)
            + "\n  Nothing was submitted -- the scheduler would refuse it.  "
              "Either run fewer trials per group, lower --trial-timeout, or "
              "name a longer domain with --domain <name>.") from None


def _reject_if_this_machine_says_no(placed, want, gpu_side: bool,
                                    label: str) -> None:
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
        raise SubmitError(
            f"{label or 'this group'} was prepared against a record that "
            f"allowed it, but THIS machine does not:\n    "
            + "\n    ".join(why)
            + f"\n  The bundle's snapshot and {mine[0].name}'s current record "
              f"disagree -- re-run `prep` here so the trials are sized "
              f"against what this machine actually offers.")


def _shelf_key(job: "Job"):
    """The exact resource ask that defines a group (§ 4.3a, 2026-08-21):
    trials grouped together must fit ONE allocation with nothing idle, so
    the key is everything the envelope would widen over."""
    r = job.resources
    return (r.mpi_np or 0, r.cpus_per_task or 0, r.gres or "")


def _gres_count(gres: str) -> int:
    try:
        return int(str(gres).rsplit(":", 1)[-1]) if gres else 0
    except ValueError:
        return 1


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


def _submit_side_group(jobset: JobSet, base: Path, dirs, pending,
                       name: str, *, gpu_side: bool,
                       domain: Optional[str], dry_run: bool,
                       trial_timeout_s: Optional[int],
                       mem_gb: Optional[float] = None,
                       time_s: Optional[int] = None) -> List[JobResult]:
    """One side's grouped submission -- the whole of the pre-split
    `submit_bench_group` body, run once per side with its own envelope,
    sequencer, and ``-p/-q`` resolution.

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
    from .materialize import latest_attempt
    def _artifacts(j):
        d = base / dirs[j.name]
        return latest_attempt(d) or d

    for j in pending:
        try:
            check_launch_matches_deck(_artifacts(j), j)
            check_trial_starts_cold(_artifacts(j), j)
        except DeckLaunchMismatch as e:
            raise SubmitError(str(e)) from e

    # The sequencer `cd`s into these, so they are the attempt too.
    trial_dirs = [_artifacts(j) for j in pending]
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
    # ``launch/`` beside the trial directories, not among them.  Before
    # this, five bench-group-* triples and ten slurm.%j files sat mixed
    # with the trial dirs in one listing.
    launch_dir = container / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)

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
        rel = (base / dirs[j.name]).name
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
    if mem_gb:
        # A launch-time --mem OVERRIDES what prep baked (the envelope
        # already carries prep's answer).  Found 2026-08-23 when job
        # 62039305 OOM'd at 24576M -- a per-GPU default nobody chose --
        # because neither prep's answer nor this flag ever reached the
        # sbatch command.
        import dataclasses as _dc
        envelope = _dc.replace(envelope, mem=f"{mem_gb:g}G")

    # What the admission is asked to fit: the wall the user stated at
    # launch, else the one prep baked.  Unstated is None -- an unstated
    # limit never bars (R3), and the wall it gets DEFAULTED to below (the
    # queue's own ceiling) fits that queue by construction.
    from ..scheduler.quantities import parse_walltime
    needed_s = time_s
    if needed_s is None and envelope.time:
        try:
            needed_s = parse_walltime(str(envelope.time))
        except ValueError:
            raise SubmitError(
                f"prep baked time={envelope.time!r}, which does not parse "
                f"as a SLURM walltime.")

    cmd = ["sbatch", "-J", f"{jobset.name}_{name}"]
    # The side IS the GPU answer -- partitioned by the deck's own word in
    # `submit_bench_group`, so nothing is re-derived here.  A named
    # --domain overrides; otherwise each side takes its capability-fitting
    # preference from the menu (`scheduler.place`, one walk).
    # R9 -- WHAT WAS ADMITTED WHEN THE WORK WAS BUILT IS RE-ADMITTED WHEN IT
    # IS SENT.  The envelope was decided at `prep`, against whatever the
    # machine record said THEN; this walk uses what it says NOW.  The two can
    # differ precisely in the case that matters: a bundle prepared on one
    # machine and rsync'd to another, or prepared before the machine's own
    # probe learned a limit.  The Au-BDT-Au sweep is the worked example --
    # its cells were sized against a record whose gpu rows said
    # `max_cores: None`, and Sol has since measured 48.
    placement = _place(base, gpu_side=gpu_side, needed_s=needed_s,
                       cores=(envelope.mpi_np or 0)
                             * max(envelope.cpus_per_task or 1, 1) or None,
                       mem_gb=envelope.mem,
                       gpus=_gres_count(envelope.gres or ""),
                       named=domain, label=name)
    # THE WALL, in the order the answers rank (user dictation,
    # 2026-08-24): what the user stated at launch (--time); else what
    # prep baked; else THE FULL AMOUNT THE TARGET QUEUE ALLOWS -- its own
    # ceiling, read off the placement.  NEVER a number this framework
    # invents -- no estimation, no per-trial arithmetic, no defaults
    # wearing a number's clothes.  Where the queue states no ceiling, no
    # wall is sent and the scheduler's default stands.  What this
    # replaces -- 15 invented minutes a trial, times slack, plus startup
    # -- sent five 38-minute jobs to Sol (62039301-05) for a system
    # nobody had sized.
    if time_s is not None:
        envelope = _dc_replace_time(envelope, _slurm_time(time_s))
    elif not envelope.time and placement is not None:
        from ..scheduler import domain_ceiling_s
        _ceil = domain_ceiling_s(placement.domain)
        if _ceil:
            envelope = _dc_replace_time(envelope, _slurm_time(_ceil))
    # The same shape as a single job: the rendered .sbatch header carries
    # the SITE directives (partition/qos/account/mail -- runwrap's one
    # header emitter, so the group cannot drift from what every trial
    # gets), and the CLI flags carry the envelope as overrides, exactly
    # the flags-win-over-header rule _sbatch_resource_flags documents.
    cmd += _sbatch_resource_flags(envelope, placement)
    cmd += ["--export", "ALL,MB_LAUNCHED_BY=jobset-launch"]
    cmd.append(f"launch/{name}.sbatch")

    if dry_run:
        return [JobResult(name, cmd, "planned")] + [
            JobResult(j.name, [], "rides the group") for j in pending]

    script.write_text("\n".join(lines), encoding="utf-8")
    from ..runwrap import _render_sbatch_for
    # Rendered at the BUNDLE's scope, not the container's (review
    # 2026-08-21): the render derives its config/environment scope from
    # the script path's parent, and the calculation's .molbuilder.json +
    # environment.json live at the bundle root -- a container-scoped
    # render missed both and refused ("no scheduler") or fell to the
    # machine scope.  The stem alone names the delegated run script, so
    # the header still runs `bash {name}.run.sh` from the container.
    # The pair this submission is ALREADY routing to, handed to the header
    # emitter so the .sbatch and the `sbatch -p/-q` on the command line
    # cannot name different queues.  Without it the header re-derived the
    # pair (or refused for want of a `scheduler` block) while the command
    # line used the one resolved above.
    header = _render_sbatch_for(base / f"{name}.sh",
                                project_dir=base,
                                resources=envelope, env=None,
                                domain_pq=((placement.partition,
                                            placement.qos)
                                           if placement else None))
    if header is not None:
        # Repoint the delegated script and SLURM's own output into
        # launch/ (L3).  Count-asserted: if the emitter's spelling of
        # either line changes, this fails loudly instead of silently
        # scattering files back among the trial dirs.
        for _old, _new in ((f"bash {name}.run.sh", f"bash launch/{name}.run.sh"),
                           ("#SBATCH -o slurm.%j.out",
                            "#SBATCH -o launch/slurm.%j.out"),
                           ("#SBATCH -e slurm.%j.err",
                            "#SBATCH -e launch/slurm.%j.err")):
            if header.count(_old) != 1:
                raise SubmitError(
                    f"the sbatch header no longer spells {_old!r} exactly "
                    f"once; the launch/ repointing needs updating.")
            header = header.replace(_old, _new)
    if header is None:
        raise SubmitError(
            "submit mode needs a queue, and this machine has neither a "
            "`scheduler` block in molbuilder.json nor any reachable "
            "submission domain in its probed environment.json -- run "
            "`molbuilder jobset probe --write` on this machine, use "
            "--mode direct, or add a scheduler block "
            "(running-a-job.md § 5.3).")
    (launch_dir / f"{name}.sbatch").write_text(header, encoding="utf-8")
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
                        f"{', '.join(able)} -- retry with --domain <name>.  "
                        f"The other side's launch stands; this side stays "
                        f"pending.")
        raise SubmitError(
            f"sbatch failed for {name} (rc={cp.returncode}):\n"
            f"{cp.stderr.strip()}" + hint)
    jid = _parse_sbatch_id(cp.stdout)
    results = [JobResult(name, cmd, "submitted", job_id=jid)]
    for j in pending:
        _record_launch(base / dirs[j.name], mode="submit",
                       command=cmd, job_id=jid, placement=placement)
        results.append(JobResult(j.name, [], "rides the group",
                                 job_id=jid))
    return results


def _placed_on(placement) -> Optional[dict]:
    """A `Placement` -> the four facts a later reader needs about where a run
    ran, or ``None`` when there was no placement (a direct run).

    ``node_type`` is the one that earns this: it is what decides whether a
    measurement taken here may be carried to a run somewhere else
    (`execution/submission.md` § 5), and it was reachable only by parsing the
    ``sbatch`` argv this file also records.
    """
    if placement is None:
        return None
    d = getattr(placement, "domain", None)
    return {"domain": getattr(d, "name", None),
            "partition": placement.partition,
            "qos": placement.qos,
            "node_type": getattr(d, "node_type", None)}


def _record_launch(attempt: Path, *, mode: str, command: List[str],
                   job_id: Optional[str] = None, placement=None) -> None:
    """Write ``run.json`` into the attempt, carrying its provenance.

    ``continued_from`` is read back from what ``prep`` copied in rather than
    passed down: prep is what knows, and re-deriving it here would be a second
    answer to one question.  ``placement`` is passed for the opposite reason:
    submission is what knows where the job went, and nothing downstream should
    have to work it out from a command line.
    """
    from .materialize import write_run_launch
    src = None
    marker = attempt / ".continued-from"
    if marker.is_file():
        src = marker.read_text(encoding="utf-8").strip() or None
    write_run_launch(attempt, mode=mode, command=command, job_id=job_id,
                     continued_from=src, placed_on=_placed_on(placement))


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
        raise SubmitError(
            f"refusing to hand {len(jobset.jobs)} jobs to the scheduler at "
            f"once ({names}).\n"
            "  Submission is one at a time, by hand.  Jobs queued together "
            "start together whenever the scheduler finds room, which on a "
            "shared cluster is antisocial -- and for a benchmark it is "
            "wrong, because points that run concurrently contend for the "
            "same cores and interconnect, so the sweep measures contention "
            "rather than scaling.\n"
            "  Name the one you mean:\n"
            f"    molbuilder jobset launch "
            f"{'bench <stage> <trial>' if jobset.kind == 'sweep' else 'run <stage>'}"
            " --mode submit\n"
            "  `--mode direct` is not affected: it runs them here, in order, "
            "waiting for each.")


def submit_jobset(jobset: JobSet, base_dir, *, mode: str,
                  domain: Optional[str] = None,
                  dry_run: bool = False,
                  only: Optional[str] = None,
                  mem_gb: Optional[float] = None,
                  time_s: Optional[int] = None,
                  ) -> List[JobResult]:
    """Launch a prepped ``jobset`` rooted at ``base_dir``.

    ``mode`` is ``"submit"`` (SLURM ``sbatch`` + per-job CLI flags) or
    ``"direct"`` (ordered local ``bash``).  ``domain`` is a
    probed domain name resolved to ``-p/-q`` (``submit`` mode only).
    ``dry_run`` returns the planned command for each job without launching.

    ``only`` names ONE job to launch.  For a ladder that is the ONLY case --
    stages do not chain, so each is submitted after you have looked at the last
    (``project-layout.md`` § 1.6), and ``_resolve_stage`` will not act on a
    ladder without a named stage.  There is nothing to thread: the run this one
    continues from has already finished and you have read it.

    **There is no ``chain`` parameter.**  It was deleted 2026-08-10 (user)
    together with ``depends_on`` / ``dep_kind`` / ``Carry``, in BOTH modes --
    not narrowed to the shape where it was safe.  An opt-in flag is typed
    before any stage has run, which is the moment you know least; the judgement
    belongs between two stages, where the evidence is.

    :func:`_refuse_batch_submission` still owns the standing rule that survives
    it -- **a scheduler is handed one job at a time** -- and lives here rather
    than in the CLI because a guard only a surface applies is one the next
    surface skips.

    Returns one :class:`JobResult` per job (the inform layer reads these).
    Refuses an invalid JobSet (same gate as prep/materialize).
    """
    errs = jobset.validate()
    if errs:
        raise SubmitError(
            "refusing to submit an invalid JobSet:\n  - " + "\n  - ".join(errs))
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise SubmitError(f"base dir not found (prep first): {base}")

    if only is not None:
        # Through the ONE resolver, so a name and a #N number reach the
        # same job here as they do at every other surface -- and the
        # refusal carries the typeable spellings (§ 8f).  This spelled its own lookup and
        # its own listing until 2026-08-10, which is the same defect
        # `prepare_attempt` had: a library entry point quietly speaking a
        # second vocabulary for the one question.
        from ..identity import resolve_stage_ref
        from .materialize import stage_refs
        refs = stage_refs(jobset)
        try:
            only = resolve_stage_ref([refs[j.name] for j in jobset.jobs],
                                     only).name
        except ValueError as e:
            raise SubmitError(str(e))
        # One job, on its own: `dataclasses.replace` would share the job
        # objects, which is what we want -- the SAME Job, just alone, so its
        # resources and script are untouched.  The dependency is dropped
        # because there is nothing left to wait for.
        import dataclasses as _dc
        lone = next(j for j in jobset.jobs if j.name == only)
        jobset = _dc.replace(jobset, jobs=[lone])

    # THE NATURAL WORKFLOW at the run door (user, 2026-08-21: "a run
    # stopped due to the server running out of time, and you can submit
    # again and by default it continues").  A LADDER stage whose latest
    # attempt has already been launched is not refused any more: the door
    # opens the next attempt CONTINUING from it -- the same
    # prepare_attempt primitive `prep run --from` uses, said out loud --
    # and then launches that.  § 1.6's "which run you continue from is
    # something you say" is AMENDED, not broken: the same stage's LATEST
    # attempt is the one source that is never a guess (a wall-killed
    # run's newest state is the state); an older attempt or another
    # stage's stays the explicit `--from` lane, and a fresh start stays
    # `prep run <stage>` first (an unlaunched attempt is reused, never
    # continued over).  Bench trials keep § 1.5's immutability refusal.
    # Under --dry-run nothing is created: the WOULD-continue line stands
    # in for that job's whole plan.
    continued: List[JobResult] = []
    if jobset.kind == "ladder":
        from .materialize import (attempts, job_dir_names, prepare_attempt,
                                  shape_of, was_launched)
        # Gated on the ATTEMPT LAYER itself, not on the shape flag: the
        # launched-attempt refusal downstream (`_launch_dir`) fires
        # whenever attempts exist, shape known or not, so this mirror
        # must too -- a shape-flag gate here left the hand-built-ladder
        # lane refusing where it should continue (found writing the pin).
        _sh = shape_of(jobset, base)
        if True:
            _names = job_dir_names(jobset, _sh)
            _skip = set()
            for _j in jobset.jobs:
                _d = base / _names[_j.name]
                _ns = attempts(_d)
                if not _ns or not was_launched(_d / f"run-{_ns[-1]}"):
                    continue
                if dry_run:
                    continued.append(JobResult(
                        _j.name, [],
                        f"WOULD continue {_names[_j.name]}/run-{_ns[-1]} "
                        f"into run-{_ns[-1] + 1} (warm), then launch it"))
                    _skip.add(_j.name)
                    continue
                try:
                    _rep = prepare_attempt(
                        jobset, base, _j.name,
                        continue_from=f"{_names[_j.name]}/run-{_ns[-1]}")
                except ValueError as _e:
                    # continuing is impossible -- no state to carry, or
                    # the stage's deck would not read it.  Both are
                    # SIGNALS (a launched run that left nothing likely
                    # died at startup), so the door refuses with the
                    # story rather than silently starting fresh.
                    raise SubmitError(
                        f"{_j.name}: run-{_ns[-1]} was launched, so "
                        f"re-submission continues by default -- but "
                        f"continuing is impossible here:\n  {_e}\n"
                        f"  Look at that run's logs; a FRESH attempt "
                        f"is:  molbuilder jobset prep run {_j.name}  "
                        f"(then submit).") from _e
                continued.append(JobResult(
                    _j.name, [],
                    f"continuing {_names[_j.name]}/run-{_ns[-1]} -> "
                    f"{_rep.dir.name} (copied: "
                    f"{', '.join(_rep.copied) or 'nothing to carry'}).  "
                    f"Fresh instead: prep run {_j.name} first."))
            if _skip:
                import dataclasses as _dc2
                jobset = _dc2.replace(jobset, jobs=[
                    j for j in jobset.jobs if j.name not in _skip])
                if not jobset.jobs:
                    return continued

    # The no-chain rule, AT THE SEAM (U5, 2026-08-12): a ladder is launched
    # one stage at a time, in EVERY mode -- direct running stages in order
    # would be local chaining, and a run that continues on its own can
    # spend a week refining a geometry you would have rejected in a minute
    # (project-layout.md § 1.6).  This was enforced only by the CLI's
    # `_resolve_stage`, so the loops below merely ASSUMED "a ladder never
    # reaches this with more than one job"; a library caller could hand
    # one straight in.  A guard only a surface applies is one the next
    # surface forgets.
    if jobset.kind == "ladder" and len(jobset.jobs) > 1:
        raise SubmitError(
            "a ladder is launched ONE stage at a time; pass `only=<stage>`. "
            "Stages do not chain (project-layout.md § 1.6), in direct mode "
            "as much as submit: each stage is launched after you have "
            "looked at the one before it.")

    # AFTER the narrowing, so `only` is what makes a launch single -- and
    # BEFORE either path, so a refusal costs nothing and a dry run previews
    # the real thing rather than a launch that would be refused.
    _refuse_batch_submission(jobset, base, mode=mode)

    if mode in ("submit", "ask"):
        # ONE PATH, so what is asked about and what would be submitted are
        # the same flags rather than two renderings that can drift.  The
        # only difference is `--test-only`, which makes SLURM answer
        # instead of enqueue.
        return continued + _submit_slurm(jobset, base, domain=domain,
                                         dry_run=dry_run, mem_gb=mem_gb,
                                         time_s=time_s, ask=(mode == "ask"))
    if mode == "direct":
        if domain:
            raise SubmitError(
                "domain is a SLURM-submit concept; it has no meaning in "
                "'direct' (local) mode.")
        return continued + _run_direct(jobset, base, dry_run=dry_run)
    raise SubmitError(
        f"unknown mode {mode!r}: must be 'submit' (SLURM), 'ask' (submit "
        f"nothing, report when it would start) or 'direct' (local)")


__all__ = ["submit_jobset", "JobResult", "SubmitError"]
