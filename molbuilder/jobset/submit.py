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
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    dirs = job_dir_names(jobset)
    for job in jobset.jobs:
        job_dir = base_dir / dirs[job.name]
        sbatch_name = _wrapper_name(job.script, ".sbatch")
        gpu = bool(job.resources.gres)
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
        results.append(JobResult(job.name, cmd, "submitted", job_id=jid))
    return results


def _run_direct(jobset: JobSet, base_dir: Path, *,
                dry_run: bool) -> List[JobResult]:
    """Local path: run each ``<stem>.run.sh`` sequentially in dependency
    order with its per-job knobs.  An ``afterok`` edge whose producer failed
    skips the dependent (and, transitively, anything depending on it); an
    ``afterany`` edge runs regardless."""
    results: List[JobResult] = []
    failed: set = set()                 # job names that failed / were skipped
    dirs = job_dir_names(jobset)
    for job in jobset.jobs:
        job_dir = base_dir / dirs[job.name]
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
        if cp.returncode != 0:
            failed.add(job.name)
            results.append(JobResult(job.name, cmd, "failed",
                                     returncode=cp.returncode))
        else:
            results.append(JobResult(job.name, cmd, "ran", returncode=0))
    return results


# --------------------------------------------------------------------- #
#  public entry point                                                   #
# --------------------------------------------------------------------- #

def submit_jobset(jobset: JobSet, base_dir, *, mode: str,
                  domain: Optional[str] = None,
                  dry_run: bool = False) -> List[JobResult]:
    """Launch a prepped ``jobset`` rooted at ``base_dir``.

    ``mode`` is ``"submit"`` (SLURM ``sbatch`` + per-job CLI flags +
    dependency threading) or ``"direct"`` (ordered local ``bash``).
    ``domain`` is a ``scheduler.routing`` name resolved to ``-p/-q``
    (``submit`` mode only).  ``dry_run`` returns the planned command for each
    job without launching.

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
