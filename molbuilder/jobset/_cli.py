"""``molbuilder jobset ...`` — operate on a bundle's ``job-set.json``
(docs/execution/job-system.md).

The thin CLI/bundle wrapper the framework needed: the engine-agnostic
target-side verbs over a persisted JobSet, mirroring ``molbuilder bench``.
Each command loads ``<bundle>/job-set.json`` and calls one engine:

  * ``plan``    -> ``render_plan``    (the chain + per-job resources, dry)
  * ``prep``    -> ``prep_jobset``    (render launchers + lay out point dirs)
  * ``submit``  -> ``submit_jobset``  (launch; ``--dry-run`` shows commands)

It owns no policy: the JobSet (produced on the host) is the source of truth;
these verbs just realize it on the target.  ``submit`` defaults to
``--dry-run`` off but PRINTS what it will do — the user picks the mode and
domain explicitly (assistant, not nanny; never a silent auto-submit).
"""

from __future__ import annotations

from pathlib import Path

import click

from .model import JobSet
from .plan import render_plan
from .prep import prep_jobset, PrepError
from .submit import submit_jobset, SubmitError
from .runstatus import jobset_status, render_status

_JOBSET_FILE = "job-set.json"


def _load(bundle: str) -> tuple:
    """Load ``<bundle>/job-set.json`` -> (JobSet, bundle Path).  A friendly
    error if it isn't there (the host producer writes it; see § 5)."""
    base = Path(bundle)
    jpath = base / _JOBSET_FILE
    if not jpath.is_file():
        raise click.ClickException(
            f"no {_JOBSET_FILE} in {base} -- nothing to do.  The host "
            "produces it (stages_to_jobset(...).write()); ship the bundle "
            "here first (staged-execution.md § 5).")
    try:
        return JobSet.load(jpath), base
    except ValueError as e:                      # bad schema / shape
        raise click.ClickException(str(e))


@click.group("jobset", short_help="run a job-set bundle (stage ladder / sweep)")
def jobset_group() -> None:
    """Operate on a bundle's ``job-set.json`` -- the engine-agnostic
    execution framework (staged-execution.md).  Produce the JobSet on the
    host; ``prep`` then ``submit`` it on the target."""


@jobset_group.command("plan", short_help="show the plan (jobs, resources, deps)")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False),
                default=".")
def plan_cmd(bundle: str) -> None:
    """Print the job-set plan: one row per job (resources, dependency,
    carry) + the order.  Reads only ``job-set.json`` -- changes nothing."""
    js, _ = _load(bundle)
    click.echo(render_plan(js))


@jobset_group.command("status", short_help="show per-stage status + resume point")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False),
                default=".")
def status_cmd(bundle: str) -> None:
    """Show each stage's run state (finished / running / failed / pending /
    not-started), which warm-restart files are present, and the FIRST
    incomplete stage (the one to resume from).  Read-only -- molbuilder
    informs; you decide whether to continue or switch (staged-execution.md
    § 10).  Reuses the same directory decoder as the Results tab."""
    js, base = _load(bundle)
    click.echo(render_status(jobset_status(js, base)))


@jobset_group.command("prep", short_help="render launchers + lay out job dirs")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False),
                default=".")
@click.option("--env", default=None,
              help="force one conda env for every job (default: auto-route "
                   "per script from the .fdf -- correct for a mixed CPU/GPU "
                   "ladder).")
@click.option("--sbatch/--no-sbatch", "emit_sbatch", default=True,
              help="emit .sbatch wrappers (default on; auto-skipped when no "
                   "scheduler is configured).")
def prep_cmd(bundle: str, env, emit_sbatch: bool) -> None:
    """Render each script's launcher (in the bundle root, from the real
    file), lay out ``point-<name>/`` dirs, and link the wrappers + carry
    files in.  Idempotent."""
    js, base = _load(bundle)
    try:
        dirs = prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch)
    except PrepError as e:
        raise click.ClickException(str(e))
    click.echo(f"prepped {len(dirs)} job dir(s) under {base}:")
    for d in dirs:
        click.echo(f"  {d.name}")
    click.echo("next: molbuilder jobset plan   (review)   then   "
               "molbuilder jobset submit --mode submit|direct")


@jobset_group.command("submit", short_help="launch the prepped job-set")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False),
                default=".")
@click.option("--mode", type=click.Choice(["submit", "direct"]), required=True,
              help="'submit' = SLURM sbatch chain; 'direct' = ordered local "
                   "bash (execution.mode, job-execution.md § 8.13).")
@click.option("--domain", default=None,
              help="scheduler.routing domain -> -p/-q (submit mode; EXPLICIT, "
                   "never auto-picked).")
@click.option("--dry-run", is_flag=True,
              help="print the exact command each job WOULD get; launch "
                   "nothing.")
def submit_cmd(bundle: str, mode: str, domain, dry_run: bool) -> None:
    """Launch the job-set: per-job ``sbatch`` (submit) or ordered local
    ``bash`` (direct).  Run ``prep`` first.  Use ``--dry-run`` to review the
    commands before anything is irreversible."""
    js, base = _load(bundle)
    try:
        results = submit_jobset(js, base, mode=mode, domain=domain,
                                dry_run=dry_run)
    except SubmitError as e:
        raise click.ClickException(str(e))
    verb = "WOULD run" if dry_run else "result"
    for r in results:
        tail = (f"job {r.job_id}" if r.job_id else
                (f"rc={r.returncode}" if r.returncode is not None else ""))
        click.echo(f"[{r.status:>9}] {r.name}  {tail}")
        click.echo(f"           {verb}: {' '.join(r.command)}")


__all__ = ["jobset_group"]
