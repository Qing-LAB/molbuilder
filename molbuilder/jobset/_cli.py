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
            "here first (job-system.md § 5.1).")
    try:
        return JobSet.load(jpath), base
    except ValueError as e:                      # bad schema / shape
        raise click.ClickException(str(e))


@click.group("jobset", short_help="run a job-set bundle (stage ladder / sweep)")
def jobset_group() -> None:
    """Operate on a bundle's ``job-set.json`` -- the engine-agnostic
    execution framework (job-system.md).  Produce the JobSet on the
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
    informs; you decide whether to continue or switch (job-system.md
    § 5.3).  Reuses the same directory decoder as the Results tab."""
    js, base = _load(bundle)
    click.echo(render_status(jobset_status(js, base)))


# --------------------------------------------------------------------- #
#  prep / submit -- the execution loop (job-system.md § 5.3)             #
#                                                                       #
#  One grammar: ``jobset <verb> <kind> [<stage>]``.  The KIND is a       #
#  positional and not a ``--bench`` flag because ``prep bench`` and      #
#  ``prep run`` are peers -- measuring and running are the same act over #
#  different parameters (project-layout.md § 2.3.1a).                    #
# --------------------------------------------------------------------- #

_KINDS = ("run", "bench")


def _check_kind(kind: str) -> None:
    """Refuse ``bench`` with a pointer rather than a shrug.

    The grammar has room for it and the fold-in is designed
    (staged-runs-architecture.md step 1c: ``bench generate`` + ``bench prep``
    become ``jobset prep bench``, and ``bench prep-run`` IS ``jobset prep run``
    written a second time).  Until that lands, saying where the working command
    lives beats pretending the word is unknown.
    """
    if kind == "bench":
        raise click.ClickException(
            "`jobset prep|submit bench` is not folded in yet -- the benchmark "
            "still has its own group:\n"
            "    molbuilder bench generate ...   (build the sweep)\n"
            "    molbuilder bench prep ...       (format it for this machine)\n"
            "    molbuilder bench summarize ...  (write the verdict)\n"
            "The grammar reserves `bench` because measuring and running are "
            "peers; see job-system.md § 5.3.")


def _resolve_stage(js, stage, chain: bool, verb: str):
    """Which jobs a verb acts on, and the refusal when that is ambiguous.

    A LADDER is a sequence you look at between steps, so acting on all of it is
    not a default -- ``--chain`` says you want it anyway (project-layout.md
    § 1.6).  A SWEEP has no such ordering: its points are independent, so the
    whole set is the ordinary thing and needs no flag.
    """
    names = [j.name for j in js.jobs]
    if stage is not None:
        if stage not in names:
            raise click.ClickException(
                f"no stage named {stage!r} in this job-set; it has: "
                f"{', '.join(names)}")
        return stage
    if js.kind == "ladder" and not chain:
        raise click.ClickException(
            f"this is a ladder, so `{verb}` acts on ONE stage: "
            f"{', '.join(names)}.\n"
            f"  molbuilder jobset {verb} run <stage>      one stage\n"
            f"  molbuilder jobset {verb} run --chain      the whole ladder, "
            f"unattended\n"
            "Stages do not chain by default because a chain that continues on "
            "its own can spend a week refining a geometry you would have "
            "rejected in a minute (project-layout.md § 1.6).")
    return None


@jobset_group.command("prep", short_help="set a stage up to run")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("stage", required=False, default=None)
@click.option("--bundle", "bundle", default=".",
              type=click.Path(exists=True, file_okay=False),
              help="the calculation folder (default: the current directory, "
                   "which is where a session is normally run from).")
@click.option("--from", "from_attempt", default=None, metavar="STAGE/run-N",
              help="the attempt this run continues from, e.g. "
                   "'01_coarse/run-0'.  Its warm files are COPIED in.  Which "
                   "run you continue from is something you say, never "
                   "something molbuilder guesses.")
@click.option("--cold", is_flag=True,
              help="start this run clean -- skip the copy.  With a directory "
                   "per attempt there is nothing to move aside.")
@click.option("--env", default=None,
              help="force one conda env for every job (default: auto-route "
                   "per script from the .fdf -- correct for a mixed CPU/GPU "
                   "ladder).")
@click.option("--sbatch/--no-sbatch", "emit_sbatch", default=True,
              help="emit .sbatch wrappers (default on; auto-skipped when no "
                   "scheduler is configured).")
def prep_cmd(kind: str, stage, bundle: str, from_attempt, cold: bool, env,
             emit_sbatch: bool) -> None:
    """Set a stage up to run, and report what was done.

    Renders the wrappers, then makes that stage's next ``run-<n>``, links the
    deck and shared package in, and copies in whatever it continues from.
    **Prep printing what it resolved is what makes submit a plain yes** -- it is
    the only place the chosen geometry and the rendered deck appear together.

    With no STAGE (and ``--chain`` semantics not applying to prep) it lays out
    every stage's container without opening an attempt, which is what a sweep
    wants and what a ladder needs before its first stage.
    """
    _check_kind(kind)
    js, base = _load(bundle)
    if (from_attempt or cold) and stage is None:
        raise click.ClickException(
            "--from / --cold describe ONE stage's attempt; name the stage:\n"
            "    molbuilder jobset prep run <stage> --from 01_coarse/run-0")
    try:
        dirs = prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch)
    except PrepError as e:
        raise click.ClickException(str(e))

    if stage is None:
        click.echo(f"prepped {len(dirs)} job dir(s) under {base}:")
        for d in dirs:
            click.echo(f"  {d.name}")
        click.echo("next: molbuilder jobset prep run <stage>   "
                   "(open its attempt)")
        return

    from .materialize import prepare_attempt
    try:
        rep = prepare_attempt(js, base, stage, continue_from=from_attempt,
                              cold=cold)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"prepared {rep['stage']}: {rep['dir'].relative_to(base)}"
               f"{'' if rep['fresh'] else '  (reused -- not launched yet)'}")
    click.echo(f"  linked: {', '.join(rep['linked'])}")
    if rep["copied"]:
        click.echo(f"  copied from {rep['continued_from']}: "
                   f"{', '.join(rep['copied'])}")
    elif rep["cold"]:
        click.echo("  cold start -- nothing copied in")
    else:
        click.echo("  nothing carried in (first stage, or none named)")
    click.echo(f"next: molbuilder jobset submit run {stage} "
               f"--mode submit|direct")


@jobset_group.command("submit", short_help="launch a prepped stage")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("stage", required=False, default=None)
@click.option("--bundle", "bundle", default=".",
              type=click.Path(exists=True, file_okay=False),
              help="the calculation folder (default: the current directory).")
@click.option("--mode", type=click.Choice(["submit", "direct"]), required=True,
              help="HOW to launch, which is a fact about this MACHINE and not "
                   "about the layout: 'direct' = run it here with bash; "
                   "'submit' = hand it to the scheduler molbuilder.json "
                   "configures.  Independent of the description's `shape` "
                   "(engines/stages.md § 6.7) -- a workstation running "
                   "hierarchical is ordinary.")
@click.option("--domain", default=None,
              help="scheduler.routing domain -> -p/-q (submit mode; EXPLICIT, "
                   "never auto-picked).")
@click.option("--chain", is_flag=True,
              help="run EVERY stage of a ladder back to back.  Off by design "
                   "(project-layout.md § 1.6); pass this to do it anyway, with "
                   "your eyes open.")
@click.option("--dry-run", is_flag=True,
              help="print the exact command each job WOULD get; launch "
                   "nothing.")
def submit_cmd(kind: str, stage, bundle: str, mode: str, domain, chain: bool,
               dry_run: bool) -> None:
    """Launch a prepped stage: local ``bash`` (direct) or the machine's
    submission system (submit).  Run ``prep`` first.  ``--dry-run`` shows the
    exact command before anything is irreversible."""
    _check_kind(kind)
    js, base = _load(bundle)
    only = _resolve_stage(js, stage, chain, "submit")
    try:
        results = submit_jobset(js, base, mode=mode, domain=domain,
                                dry_run=dry_run, only=only)
    except SubmitError as e:
        raise click.ClickException(str(e))
    verb = "WOULD run" if dry_run else "result"
    for r in results:
        tail = (f"job {r.job_id}" if r.job_id else
                (f"rc={r.returncode}" if r.returncode is not None else ""))
        click.echo(f"  {verb}  {r.name:<12} {' '.join(r.command)}  "
                   f"[{r.status}] {tail}".rstrip())
    if not dry_run:
        click.echo("next: molbuilder jobset status   (look before the next "
                   "stage)")
