"""Submit engine — launch a *prepped* :class:`JobSet`
(docs/execution/job-system.md, § 7-9).

This is the keystone both producers feed: the SIESTA stage ladder and
(once it migrates) the benchmark sweep render to a JobSet, ``prep_jobset``
renders the launchers + lays out the ``point-<name>/`` tree, and THIS engine
launches them.  It assumes prep already ran (the wrappers are symlinked into
each job dir); it renders nothing itself.

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
    races.  A failure stops what depends on it — an explicit ``afterok``
    edge, or, for a ladder, everything after it, since stage N continues
    from stage N-1's warm files (:func:`_blocked_by_a_failure`).

REUSE, not reinvention: prep renders via ``runwrap.write_run_wrapper``; this
engine adds only the cross-job concerns — per-job CLI overrides, dependency
threading, domain→``-p/-q`` resolution, and ordered local execution.

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
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .materialize import job_dir_names
from .model import JobSet, Resources
from .prep import check_launch_matches_deck


class SubmitError(Exception):
    """A JobSet could not be submitted (bad mode, unknown domain, missing
    prepped wrapper, or sbatch failure)."""


@dataclass
class JobResult:
    """What happened to one job.  ``command`` is always populated (the exact
    line that ran / would run).  In ``submit`` mode ``job_id`` is the SLURM
    id; in ``direct`` mode ``returncode`` is the process exit status.
    ``status`` is one of ``submitted`` / ``ran`` / ``failed`` / ``skipped``
    / ``planned`` (dry-run)."""
    name:       str
    command:    List[str]
    status:     str
    job_id:     Optional[str] = None
    returncode: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------- #
#  domain → -p/-q resolution (reuses runtime_config.get_routing)         #
# --------------------------------------------------------------------- #

def _resolve_domain(domain: Optional[str], *, gpu: bool,
                    project_dir: Optional[Path]) -> Optional[tuple]:
    """Resolve a ``scheduler.routing`` domain name to ``(partition, qos)``
    for the ``sbatch`` CLI.  ``None`` domain → ``None`` (use the rendered
    header's default directives).  A GPU job prefers the domain's
    ``gpu_partition`` when set (slurm-integration.md § 4.3)."""
    if not domain:
        return None
    from .. import runtime_config as _rc
    routing = _rc.get_routing(project_dir=project_dir)
    for d in routing:
        if d["name"] == domain:
            part = (d.get("gpu_partition") or d["partition"]) if gpu \
                else d["partition"]
            return (part, d["qos"])
    names = ", ".join(d["name"] for d in routing) or "(none configured)"
    raise SubmitError(
        f"unknown submission domain {domain!r}; configured: {names} "
        "(scheduler.routing in .molbuilder.json, slurm-integration.md § 4.3)")


# --------------------------------------------------------------------- #
#  per-job resource → CLI flags (bench launch-line model)               #
# --------------------------------------------------------------------- #

def _sbatch_resource_flags(r: Resources) -> List[str]:
    """The per-job ``sbatch`` overrides.  CLI flags win over the rendered
    header's #SBATCH defaults, so one shared ``.sbatch`` serves every job.
    ``--exclusive`` and ``--mem`` are mutually exclusive (whole-node owns all
    memory — the ``--mem`` is meaningless and rejected by some sites), so
    exclusive suppresses ``--mem`` (slurm-integration.md § 4.3.1)."""
    flags: List[str] = []
    if r.mpi_np:
        flags += ["-n", str(r.mpi_np)]
    if r.cpus_per_task:
        flags += ["-c", str(r.cpus_per_task)]
    if r.gres:
        flags.append(f"--gres={r.gres}")
    if r.time:
        flags += ["-t", r.time]
    if r.exclusive:
        flags.append("--exclusive")
    elif r.mem:
        flags.append(f"--mem={r.mem}")
    return flags


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


def _submit_slurm(jobset: JobSet, base_dir: Path, *, domain: Optional[str],
                  dry_run: bool) -> List[JobResult]:
    """SLURM path: ``sbatch`` **one** job, with its resources as CLI flags.

    **One per invocation** — :func:`_refuse_batch_submission` is what makes
    that true, and this loop keeps its shape only because a caller may narrow
    to a single job by several routes.  Until 2026-08-10 this docstring said
    *"a sweep submits with no dependency, so its jobs queue in parallel"*, and
    the code did exactly that: one command, N ``sbatch`` calls, every one of
    them racing the others for the same nodes.

    The ``--dependency`` threading below is reachable only for a job whose set
    still declares an edge — a hand-built chained ladder — and never for more
    than one job at a time, so it can no longer be the mechanism by which a
    whole ladder is handed over at once.
    """
    results: List[JobResult] = []
    ids: Dict[str, str] = {}            # job.name -> slurm job id
    for job in jobset.jobs:
        job_dir, attempt, sbatch_name = _resolve_launch(
            jobset, base_dir, job, ".sbatch")
        gpu = _job_wants_gpu(job_dir, job)
        pq = _resolve_domain(domain, gpu=gpu, project_dir=base_dir)

        cmd = ["sbatch"]
        if job.depends_on is not None:
            dep_id = ids.get(job.depends_on)
            cmd.append(f"--dependency={job.dep_kind}:"
                       f"{dep_id if dep_id else '<' + job.depends_on + '>'}")
        cmd += ["-J", job.name]
        if pq:
            cmd += ["-p", pq[0], "-q", pq[1]]
        cmd += _sbatch_resource_flags(job.resources)
        cmd.append(sbatch_name)          # relative; we cd into the job dir

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
        cp = subprocess.run(cmd, cwd=str(job_dir),
                            capture_output=True, text=True)
        if cp.returncode != 0:
            results.append(JobResult(job.name, cmd, "failed",
                                     returncode=cp.returncode))
            raise SubmitError(
                f"sbatch failed for job {job.name!r} (rc={cp.returncode}):\n"
                f"{cp.stderr.strip()}")
        jid = _parse_sbatch_id(cp.stdout)
        ids[job.name] = jid
        if attempt is not None:
            _record_launch(attempt, mode="submit", command=cmd, job_id=jid)
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
    check_launch_matches_deck(job_dir, job)
    return job_dir, attempt, _wrapper_name(job.script, suffix)


def _launch_dir(jobset: JobSet, base_dir: Path, job) -> Tuple[Path, Optional[Path]]:
    """Where this job runs, and the attempt to record the launch into.

    Two layouts meet here.  A SWEEP has no attempt layer -- ``prep`` lays out
    ``point-<name>/`` and the point runs there, as it always has.  A LADDER
    stage prepped with ``jobset prep run <stage>`` has ``<seq>_<name>/run-<n>/``,
    and that is where it runs, because an attempt is immutable once it has run
    (``project-layout.md`` § 1.5) and a re-run must not land on top of one.

    Refuses an attempt that has already been launched.  ``run.json`` is the only
    honest answer to *has this started?* -- a queued job has produced nothing
    yet, so absence of output proves nothing (§ 1.6).
    """
    from .materialize import attempts, job_dir_names, shape_of, was_launched
    d = base_dir / job_dir_names(jobset, shape_of(jobset, base_dir))[job.name]
    ns = attempts(d)
    if not ns:
        return d, None
    last = d / f"run-{ns[-1]}"
    if was_launched(last):
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


def _blocked_by_a_failure(jobset: JobSet, job, failed: set) -> bool:
    """Should this job be skipped because something before it failed?

    **Two shapes, and the second was silently lost on 2026-08-10.**

    A job with an explicit ``afterok`` edge is skipped when its named producer
    failed — the SLURM dependency meaning reproduced locally (`job-system.md`
    § 5.3), and the only case this function handled.

    A **ladder** now declares no edges at all (P7 unit 2), so that test could
    never fire for one: `None in failed` is false, and a `--chain` whose first
    stage died went on to compute the next two from it. That is not a missing
    edge, it is the ladder's own meaning: **stage N continues from stage N-1's
    warm files**, so if N-1 failed there is nothing to continue from — at best
    stale state, at worst none, and either way the answer is garbage that
    reports success. `project-layout.md` § 1.6 argues chaining is dangerous
    because *"a chain that continues by itself can spend a week refining a
    geometry you would have rejected in a minute"*; continuing past a **failure**
    is that argument at its strongest.

    A **sweep** is untouched and must be: its points are independent by
    definition, so one bad point says nothing about the next.

    *This is the gap `on_nonconvergence: halt` used to cover. It was deleted
    with the edges it rendered into, and nothing replaced the intent — the
    defect was found by asking who still benefits from a mechanism whose
    stated justification had expired, rather than by a test.*
    """
    if job.depends_on is not None:
        return job.depends_on in failed and job.dep_kind == "afterok"
    return jobset.kind == "ladder" and bool(failed)


def _run_direct(jobset: JobSet, base_dir: Path, *,
                dry_run: bool) -> List[JobResult]:
    """Local path: run each ``<stem>.run.sh`` sequentially in dependency
    order with its per-job knobs.  An ``afterok`` edge whose producer failed
    skips the dependent (and, transitively, anything depending on it); an
    ``afterany`` edge runs regardless."""
    results: List[JobResult] = []
    failed: set = set()                 # job names that failed / were skipped
    for job in jobset.jobs:
        job_dir, attempt, run_name = _resolve_launch(
            jobset, base_dir, job, ".run.sh")
        cmd = ["bash", run_name] + _run_sh_args(job.resources)
        if _blocked_by_a_failure(jobset, job, failed):
            results.append(JobResult(job.name, cmd, "skipped"))
            failed.add(job.name)        # propagate down the chain
            continue
        if dry_run:
            results.append(JobResult(job.name, cmd, "planned"))
            continue
        if not (job_dir / run_name).exists():
            raise SubmitError(
                f"job {job.name!r}: {run_name} not in {job_dir} "
                "(run prep_jobset first).")
        cp = subprocess.run(cmd, cwd=str(job_dir))
        if attempt is not None:
            # AFTER the launch, so a failed start leaves the attempt exactly as
            # prepare left it -- still safe to prepare again (§ 1.6).
            _record_launch(attempt, mode="direct", command=cmd)
        if cp.returncode != 0:
            failed.add(job.name)
            results.append(JobResult(job.name, cmd, "failed",
                                     returncode=cp.returncode))
        else:
            results.append(JobResult(job.name, cmd, "ran", returncode=0))
    return results


def _record_launch(attempt: Path, *, mode: str, command: List[str],
                   job_id: Optional[str] = None) -> None:
    """Write ``run.json`` into the attempt, carrying its provenance.

    ``continued_from`` is read back from what ``prep`` copied in rather than
    passed down: prep is what knows, and re-deriving it here would be a second
    answer to one question.
    """
    from .materialize import write_run_launch
    src = None
    marker = attempt / ".continued-from"
    if marker.is_file():
        src = marker.read_text(encoding="utf-8").strip() or None
    write_run_launch(attempt, mode=mode, command=command, job_id=job_id,
                     continued_from=src)


# --------------------------------------------------------------------- #
#  public entry point                                                   #
# --------------------------------------------------------------------- #

def _refuse_batch_submission(jobset: JobSet, base_dir: Path, *, mode: str,
                             chain: bool) -> None:
    """The two things a launch may not do (user rule, 2026-08-10).

    **1. A scheduler is handed one job at a time.**  *"SLURM should never
    submit jobs in parallel.  Submission is manual and one by one.  It is a
    disaster to do parallel job submission on HPC."*  Firing N ``sbatch``
    calls from one command puts N jobs in the queue that will start whenever
    the scheduler finds room — together, if there is room — and on a shared
    cluster that is antisocial at best.  For a **benchmark** it is worse than
    antisocial: points that run concurrently contend for the same cores,
    memory bandwidth and interconnect, so the sweep measures contention rather
    than scaling and the numbers are quietly wrong.

    This is a rule about the **scheduler**, not about doing several things.
    ``--mode direct`` runs each job here, in order, waiting for each — that is
    not submission at all, and it is untouched.

    **2. A hierarchical ladder does not chain, in either mode.**
    `project-layout.md § 1`'s table is the reason, and it is structural rather
    than a matter of taste: continuing is *"free"* in **flat** — *"the next
    stage finds them lying there"*, because the warm files are one shared set
    at the root — while in the **hierarchy** *"you **name** the run, and its
    files are copied in."*  A named run must have already finished.  So a
    chained hierarchical launch would have to copy a file that does not exist
    yet, and the only way it ever appeared to work is the dangling carry
    symlinks P7 unit 2 removes.  Refusing says so instead of producing three
    stages that each started from the deck's own coordinates.
    """
    if len(jobset.jobs) <= 1 and not chain:
        return
    from .materialize import shape_of
    sh = shape_of(jobset, base_dir)

    if chain and sh is not None and sh.keeps_attempts_as_directories:
        raise SubmitError(
            "this calculation's shape is 'hierarchical', which does not "
            "chain.\n"
            "  Each stage there continues from a run you NAME -- `prep run "
            "<stage> --from 02_medium/run-0` -- and a chain has none to "
            "name, because the run it would continue from has not happened "
            "yet (project-layout.md § 1).  A chained hierarchical launch "
            "would start every stage from the deck's own coordinates and "
            "report success.\n"
            "  Run it a stage at a time, looking in between -- which is what "
            "this shape is for:\n"
            "    molbuilder jobset prep   run <stage> --from <attempt>\n"
            "    molbuilder jobset submit run <stage> --mode ...\n"
            "  A FLAT calculation does chain: its warm files are one shared "
            "set, so the next stage finds them lying there.")

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
            "    molbuilder jobset submit run <stage> --mode submit\n"
            "  `--mode direct` is not affected: it runs them here, in order, "
            "waiting for each.")


def submit_jobset(jobset: JobSet, base_dir, *, mode: str,
                  domain: Optional[str] = None,
                  dry_run: bool = False,
                  only: Optional[str] = None,
                  chain: bool = False) -> List[JobResult]:
    """Launch a prepped ``jobset`` rooted at ``base_dir``.

    ``mode`` is ``"submit"`` (SLURM ``sbatch`` + per-job CLI flags +
    dependency threading) or ``"direct"`` (ordered local ``bash``).
    ``domain`` is a ``scheduler.routing`` name resolved to ``-p/-q``
    (``submit`` mode only).  ``dry_run`` returns the planned command for each
    job without launching.

    ``only`` names ONE job to launch.  For a ladder that is the normal case --
    stages do not chain, so each is submitted after you have looked at the last
    (``project-layout.md`` § 1.6); the CLI refuses to act on a whole ladder
    without ``--chain``.  A job launched this way has no dependency to thread,
    because the thing it depended on has already finished and you have read it.

    ``chain`` is the caller saying *run the whole ladder back to back*.  It is
    honoured only where it can be honoured -- see
    :func:`_refuse_batch_submission`, which owns both standing refusals: **a
    scheduler is handed one job at a time**, and **a hierarchical ladder does
    not chain in either mode**.  Those live here rather than in the CLI because
    a guard only a surface applies is one the next surface skips.

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
        # Through the ONE resolver, so a name, a number and a token all reach
        # the same job here as they do at every other surface -- and the
        # refusal carries the ordinals (§ 8f).  This spelled its own lookup and
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
        lone = _dc.replace(next(j for j in jobset.jobs if j.name == only),
                           depends_on=None)
        jobset = _dc.replace(jobset, jobs=[lone])

    # AFTER the narrowing, so `only` is what makes a launch single -- and
    # BEFORE either path, so a refusal costs nothing and a dry run previews
    # the real thing rather than a launch that would be refused.
    _refuse_batch_submission(jobset, base, mode=mode, chain=chain)

    if mode == "submit":
        return _submit_slurm(jobset, base, domain=domain, dry_run=dry_run)
    if mode == "direct":
        if domain:
            raise SubmitError(
                "domain is a SLURM-submit concept; it has no meaning in "
                "'direct' (local) mode.")
        return _run_direct(jobset, base, dry_run=dry_run)
    raise SubmitError(
        f"unknown mode {mode!r}: must be 'submit' (SLURM) or 'direct' (local)")


__all__ = ["submit_jobset", "JobResult", "SubmitError"]
