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
    line, so one rendered wrapper serves every point of a sweep.  A ladder
    threads ``--dependency=<dep_kind>:<jobid>`` down the chain; a sweep
    submits with no dependency (jobs queue in parallel).
  * ``"direct"`` — local shell.  Each job's ``<stem>.run.sh`` is run in
    turn with its per-job knobs as args (``-np``/``-omp``), honoring
    ``dep_kind`` locally (an ``afterok`` edge whose producer failed skips
    the dependent and everything below it; ``afterany`` runs regardless —
    the SLURM dependency-kind semantics reproduced on a workstation).

REUSE, not reinvention: prep renders via ``runwrap.write_run_wrapper``; this
engine adds only the cross-job concerns — per-job CLI overrides, dependency
threading, domain→``-p/-q`` resolution, and ordered local execution.

RESUME IS THE MODELING SOFTWARE'S JOB (script-execution.md): this engine
only launches.  It never inspects prior output to auto-recover — the carry
symlinks prep laid let SIESTA/PySCF restart natively; the decision to
continue or switch stays the user's (assistant, not nanny).  ``dry_run=True``
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
    """SLURM path: sbatch each job with per-job CLI flags, threading
    ``--dependency`` from the producer's id.  A sweep (no ``depends_on``)
    submits with no dependency, so its jobs queue in parallel; a ladder
    chains."""
    results: List[JobResult] = []
    ids: Dict[str, str] = {}            # job.name -> slurm job id
    for job in jobset.jobs:
        job_dir, attempt = _launch_dir(jobset, base_dir, job)
        sbatch_name = _wrapper_name(job.script, ".sbatch")
        _check_launch_matches_deck(job_dir, job)
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


def _check_launch_matches_deck(job_dir: Path, job) -> None:
    """Refuse a launch the deck was not rendered for (P6 unit 2).

    `project-layout.md § 2.3.1` states the five steps and says the order is
    **forced**: *"Step 3 cannot precede step 1, because a deck carries values
    that depend on how it will be launched — a block size derived from the rank
    count … A parameter that depends on the launch cannot be decided before the
    launch is known."*

    Today step 3 happens at `molbuilder fdf`, on whatever machine typed it, and
    the rank count is resolved hours later by the wrapper. **The two halves of
    one ordered sequence run in different places, and nothing carried the
    first's answer to the third.** On 2026-08-10 that produced a deck rendered
    with no rank count — so ``BlockSize`` from the size-only branch — launched
    at ``-np 14``, and SIESTA refused at startup with *"You have too many
    processors for the system size"*.

    P4 unit 5 put the launch quantity **into** the deck (BENCH-MARKS carries
    ``mpi_np``), which is why that failure was diagnosable at all. **Recording
    is not agreeing**: this is the agreement, and it happens before the engine
    is started rather than being discovered by it.

    Three outcomes, and the middle one is the live defect:

    * deck ``auto`` + launch ``auto`` — both defer to the wrapper. Fine.
    * deck ``auto`` + launch ``N`` — the deck's launch-derived values were
      computed with **no** rank count, and now one is being imposed. Refused.
    * deck ``N`` + launch ``M`` — refused, with both numbers named.

    A deck with no BENCH-MARKS block says nothing about its launch, so there is
    nothing to disagree with and nothing is refused.
    """
    deck = Path(job_dir) / os.path.basename(job.script)
    if not deck.is_file():
        return
    from ..parse.scripts.bench_marks import _extract_bench_marks_dict
    marks = _extract_bench_marks_dict(deck.read_text(encoding="utf-8",
                                                     errors="replace"))
    if not marks or "mpi_np" not in marks:
        return
    rendered_for = marks["mpi_np"]                 # an int, or the str "auto"
    launching_at = job.resources.mpi_np            # an int, or None == auto
    if rendered_for == "auto" and launching_at is None:
        return
    if rendered_for == launching_at:
        return
    _fmt = lambda v: "auto" if v in ("auto", None) else str(v)   # noqa: E731
    raise SubmitError(
        f"job {job.name!r}: this deck was rendered for mpi_np "
        f"{_fmt(rendered_for)}, and you are launching it at "
        f"{_fmt(launching_at)}.\n"
        f"  {deck.name} derives values from the rank count -- BlockSize above "
        f"all -- so a deck rendered for one launch is wrong for another "
        f"(project-layout.md § 2.3.1: a parameter that depends on the launch "
        f"cannot be decided before the launch is known).\n"
        f"  Re-render the deck for this launch, or launch it at "
        f"{_fmt(rendered_for)}.  The deck records what it assumed in its "
        f"BENCH-MARKS block, which is what made this checkable.")


def _run_direct(jobset: JobSet, base_dir: Path, *,
                dry_run: bool) -> List[JobResult]:
    """Local path: run each ``<stem>.run.sh`` sequentially in dependency
    order with its per-job knobs.  An ``afterok`` edge whose producer failed
    skips the dependent (and, transitively, anything depending on it); an
    ``afterany`` edge runs regardless."""
    results: List[JobResult] = []
    failed: set = set()                 # job names that failed / were skipped
    for job in jobset.jobs:
        job_dir, attempt = _launch_dir(jobset, base_dir, job)
        _check_launch_matches_deck(job_dir, job)
        run_name = _wrapper_name(job.script, ".run.sh")
        cmd = ["bash", run_name] + _run_sh_args(job.resources)
        if job.depends_on in failed and job.dep_kind == "afterok":
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

def submit_jobset(jobset: JobSet, base_dir, *, mode: str,
                  domain: Optional[str] = None,
                  dry_run: bool = False,
                  only: Optional[str] = None) -> List[JobResult]:
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
