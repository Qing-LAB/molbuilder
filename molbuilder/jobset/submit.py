"""Submit engine — execute a *materialized* :class:`JobSet`
(docs/protocols/staged-execution.md § 2.1, § 8).

This is the keystone both producers feed: the SIESTA stage ladder and
(once it migrates) the benchmark sweep render to a JobSet, ``materialize``
lays out the ``point-<name>/`` dirs + shared/carry symlinks, and THIS
engine launches them.

Two execution paths, chosen by ``mode`` (== ``execution.mode``):

  * ``"submit"`` — SLURM.  Each job becomes an ``sbatch``; a ladder threads
    ``--dependency=<dep_kind>:<jobid>`` down the chain, a sweep submits all
    jobs with no dependency (they queue in parallel).  The selected
    ``domain`` resolves to ``sbatch -p/-q`` at submit time (mirrors the
    bench ``MB_GPU_PQ`` injection) so the same rendered ``.sbatch`` works
    for any domain.
  * ``"direct"`` — local shell.  Jobs run **sequentially** in dependency
    order via ``bash <name>.run.sh``; an ``afterok`` edge aborts the rest
    of the chain when its producer fails, an ``afterany`` edge runs the
    next job regardless (the SLURM dependency-kind semantics, honored
    locally so a workstation dry-run matches what the cluster would do).

REUSE, not reinvention: the per-job ``Resources`` → SLURM-flags translation
(``-n``/``-c``/``-t``/``--mem``/``--gres``/``--exclusive``) is
``runwrap.write_run_wrapper`` — the SAME path the single-job and bench
flows use.  This engine only adds the cross-job concerns: dependency
threading, domain→``-p/-q`` resolution, and ordered local execution.

RESUME IS THE MODELING SOFTWARE'S JOB (script-execution.md): this engine
only launches.  It never inspects prior output to auto-recover — the carry
symlinks materialize already laid let SIESTA/PySCF restart natively; the
decision to continue or switch stays the user's (assistant, not nanny).
``dry_run=True`` writes nothing and runs nothing: it returns the exact
command line each job WOULD get, so the plan is reviewable before anything
is irreversible.
"""

from __future__ import annotations

import dataclasses
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .materialize import job_dir_name
from .model import Job, JobSet


class SubmitError(Exception):
    """A JobSet could not be submitted (bad mode, unknown domain, missing
    materialized dir, scheduler required but absent, or sbatch failure)."""


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
#  per-job wrapper rendering (reuses runwrap.write_run_wrapper)          #
# --------------------------------------------------------------------- #

def _render_wrappers(job: Job, job_dir: Path, *, emit_sbatch: bool) -> Path:
    """Render ``<script>.run.sh`` (and ``.sbatch`` when ``emit_sbatch``) for
    one job's input, threading its :class:`Resources` through the SHARED
    translation in ``runwrap``.  Returns the script path inside ``job_dir``.

    The job's ``script`` is the materialized symlink to the rendered input
    (e.g. the ``.fdf``); the wrappers land next to it in the job dir, so the
    per-job ``-n``/``-c``/``--mem``/``--gres`` differ per point while the
    science input is shared.
    """
    from ..runwrap import write_run_wrapper

    script_path = job_dir / job.script
    if not script_path.exists():
        raise SubmitError(
            f"job {job.name!r}: input {job.script!r} not found in {job_dir} "
            "(was the JobSet materialized first?)")
    r = job.resources
    write_run_wrapper(
        script_path,
        mpi_np=r.mpi_np,
        cpus_per_task=r.cpus_per_task,
        time=r.time,
        gres=r.gres,
        mem=r.mem,
        exclusive=r.exclusive,
        emit_sbatch=emit_sbatch,
    )
    return script_path


# --------------------------------------------------------------------- #
#  the two execution paths                                              #
# --------------------------------------------------------------------- #

def _sbatch_cmd(sbatch_path: Path, *, dep: Optional[str],
                pq: Optional[tuple]) -> List[str]:
    """Build the ``sbatch`` argv: optional ``--dependency`` (threaded from
    the producer's job id) + optional ``-p/-q`` (the resolved domain) +
    the rendered ``.sbatch``."""
    cmd = ["sbatch"]
    if dep:
        cmd.append(f"--dependency={dep}")
    if pq:
        cmd += ["-p", pq[0], "-q", pq[1]]
    cmd.append(str(sbatch_path))
    return cmd


def _parse_sbatch_id(stdout: str) -> str:
    """Extract the job id from ``Submitted batch job <id>``."""
    for tok in stdout.split():
        if tok.isdigit():
            return tok
    raise SubmitError(f"could not parse sbatch job id from: {stdout!r}")


def _submit_slurm(jobset: JobSet, base_dir: Path, *, domain: Optional[str],
                  dry_run: bool) -> List[JobResult]:
    """SLURM path: sbatch each job, threading ``--dependency`` from the
    producer's id.  A sweep (no ``depends_on``) submits with no dependency,
    so its jobs queue in parallel; a ladder chains."""
    results: List[JobResult] = []
    ids: Dict[str, str] = {}            # job.name -> slurm job id
    for job in jobset.jobs:
        job_dir = base_dir / job_dir_name(job.name)
        gpu = bool(job.resources.gres)
        pq = _resolve_domain(domain, gpu=gpu, project_dir=base_dir)
        dep = None
        if job.depends_on is not None:
            dep_id = ids.get(job.depends_on)
            if dep_id is None:
                # producer wasn't submitted (dry-run shows the intent;
                # a real run would have it because order is validated).
                dep = f"{job.dep_kind}:<{job.depends_on}>"
            else:
                dep = f"{job.dep_kind}:{dep_id}"
        sbatch_path = job_dir / (Path(job.script).stem + ".sbatch")
        cmd = _sbatch_cmd(sbatch_path, dep=dep, pq=pq)
        if dry_run:
            results.append(JobResult(job.name, cmd, "planned"))
            continue
        _render_wrappers(job, job_dir, emit_sbatch=True)
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
    """Local path: run each ``<name>.run.sh`` sequentially in dependency
    order.  An ``afterok`` edge whose producer failed skips the dependent
    (and, transitively, anything depending on it); an ``afterany`` edge runs
    regardless — the SLURM dependency-kind semantics honored locally."""
    results: List[JobResult] = []
    failed: set = set()                 # job names that failed / were skipped
    for job in jobset.jobs:
        job_dir = base_dir / job_dir_name(job.name)
        run_sh = job_dir / (Path(job.script).stem + ".run.sh")
        cmd = ["bash", str(run_sh)]
        # honor the dependency kind: afterok blocks on a failed producer.
        if job.depends_on in failed and job.dep_kind == "afterok":
            results.append(JobResult(job.name, cmd, "skipped"))
            failed.add(job.name)        # propagate down the chain
            continue
        if dry_run:
            results.append(JobResult(job.name, cmd, "planned"))
            continue
        _render_wrappers(job, job_dir, emit_sbatch=False)
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
    """Execute a materialized ``jobset`` rooted at ``base_dir``.

    ``mode`` is ``"submit"`` (SLURM ``sbatch`` + dependency threading) or
    ``"direct"`` (ordered local ``bash`` execution).  ``domain`` is a
    ``scheduler.routing`` name resolved to ``-p/-q`` at submit time
    (``submit`` mode only).  ``dry_run`` returns the planned command for
    each job without writing wrappers or launching anything.

    Returns one :class:`JobResult` per job (the inform layer reads these).
    Refuses an invalid JobSet (same gate as materialize) so the engine
    can't act on a structurally-broken plan.
    """
    errs = jobset.validate()
    if errs:
        raise SubmitError(
            "refusing to submit an invalid JobSet:\n  - " + "\n  - ".join(errs))
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise SubmitError(f"base dir not found (materialize first): {base}")

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
