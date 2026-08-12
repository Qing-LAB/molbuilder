"""``molbuilder jobset ...`` — operate on a bundle's ``job-set.json``
(docs/execution/job-system.md).

The thin CLI/bundle wrapper the framework needed: the engine-agnostic
target-side verbs over a persisted JobSet, mirroring ``molbuilder bench``.
Each command loads ``<bundle>/job-set.json`` and calls one engine:

  * ``plan``    -> ``render_plan``    (the jobs + per-job resources, dry)
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
from .runstatus import jobset_status, render_stage_status, render_status

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


#: The calculation folder, spelled the same way on every verb.
#:
#: It was a POSITIONAL on ``plan`` and ``status`` until 2026-08-10 while
#: ``prep``/``submit`` took ``--bundle``, so one word meant the folder on two
#: verbs and the stage on the other two.  `job-system.md` § 5.3 calls that *"a
#: defect of this section's own making"*: the grammar is
#: ``jobset <verb> <kind> [<stage>]``, and a positional that is sometimes a
#: path has no place in it.  `jobset status tight` answered *"Directory 'tight'
#: does not exist"*, which tells a user they mistyped a path they never meant.
_bundle_option = click.option(
    "--bundle", "bundle", default=".",
    type=click.Path(exists=True, file_okay=False),
    help="the calculation folder (default: the current directory, which is "
         "where a session is normally run from).")


@jobset_group.command("describe",
                      short_help="write the portable description (floor 2 only)")
@click.argument("structure", type=click.Path(exists=True, dir_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
@click.option("--shape", required=True,
              type=click.Choice(("flat", "hierarchical")),
              help="how the stages sit on disk. REQUIRED and never inferred -- "
                   "inferring it would hand you a directory tree you did not "
                   "ask for (engines/stages.md 6.7).")
@click.option("--stage-strategy", default=None,
              help="which shipped ladder to describe (e.g. publishable). Omit "
                   "for a calculation with a SINGLE parameter set, which is a "
                   "description with no stages at all (6.5).")
@click.option("--name", default=None, metavar="NAME",
              help="what you call this calculation; the label and the run id "
                   "derive from it. Default: the destination folder's name.")
@click.option("--engine", default="siesta", type=click.Choice(("siesta",)),
              help="whose parameters these are.")
@click.option("--psml-lib", default=None, metavar="DIR",
              type=click.Path(exists=True, file_okay=False),
              help="where to read pseudopotentials from. A path on THIS "
                   "machine: the files travel with the calculation, the path "
                   "does not.")
@click.option("--vacuum", type=float, default=None, metavar="ANGSTROM",
              help="isolation vacuum (A) per side on isolated axes. Needed "
                   "for a flat or linear molecule from a bare XYZ, which "
                   "otherwise has a degenerate cell.")
def describe_cmd(structure: str, dest: str, shape: str,
                 stage_strategy, name, engine: str, psml_lib, vacuum) -> None:
    """Write the portable description: the template, ``task.json``, and the
    data files.

    **Floor 2 only -- it renders no deck.**  A deck carries values that depend
    on how it will be launched, so rendering belongs to ``prep`` on the machine
    that will run it (project-layout.md 2.3.1).  What this writes names no
    machine and therefore means the same thing wherever you copy it.

    Names and values are validated HERE, on your laptop, not on the cluster: a
    stage name outside [A-Za-z0-9_]+, a duplicate stage, an override key the
    schema does not know, or a value outside its bounds is refused with the
    field named -- and refused before anything is written.
    """
    import dataclasses as _dc
    from pathlib import Path as _P

    from .. import load as _load_structure
    from ..config.siesta import SiestaConfig
    from ..describe import DescribeError, build_description, write_description
    from ..identity import normalise_id
    from ..siesta.input import _detect_species
    from ..siesta.stages import default_siesta_stages

    out_dir = _P(dest)
    run_name = name or out_dir.name

    try:
        struct = _load_structure(structure)
        if vacuum is not None:
            # The same channel the Modify -> Cell tab uses: the vacuum lives on
            # the STRUCTURE, not on the engine config, so every surface that
            # reads this structure sees the same isolation.
            struct = _dc.replace(struct, vacuum=(vacuum, vacuum, vacuum))

        stages = (tuple(default_siesta_stages(stage_strategy))
                  if stage_strategy else ())
        # The label goes through the SAME normaliser Task.label uses, so the
        # template's SystemLabel and the description's id cannot disagree
        # about what this calculation is called.
        label = normalise_id(run_name, what="name",
                             stage_names=tuple(s.name for s in stages))
        cfg = SiestaConfig(system_label=label, psml_lib=psml_lib)

        desc = build_description(
            struct, cfg, stages,
            engine=engine, shape=shape, name=run_name,
            source=str(structure),
            pseudo_species=_detect_species(struct.elements),
        )
        written = write_description(desc, out_dir, psml_lib=psml_lib)
    except DescribeError as e:
        raise click.ClickException(str(e))
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e))

    ladder = (f"{len(desc.task.stages)} stage(s): "
              f"{', '.join(s.name for s in desc.task.stages)}"
              if desc.task.stages else "one parameter set (no ladder)")
    click.echo(f"Described {desc.label!r} in {out_dir} -- {ladder}, "
               f"shape {shape}.", err=True)
    click.echo("  " + "\n  ".join(p.name for p in written), err=True)
    click.echo(
        f"\nIt names no machine. On the machine that will run it:\n"
        f"  cd {out_dir} && molbuilder jobset prep run <stage>", err=True)


@jobset_group.command("plan", short_help="show the plan (jobs, resources, deps)")
@_bundle_option
def plan_cmd(bundle: str) -> None:
    """Print the job-set plan: one row per job (resources, dependency,
    carry) + the order.  Reads only ``job-set.json`` -- changes nothing."""
    js, _ = _load(bundle)
    click.echo(render_plan(js))


@jobset_group.command("status", short_help="show per-stage status + resume point")
@click.argument("stage", required=False, default=None)
@_bundle_option
def status_cmd(stage, bundle: str) -> None:
    """Show each stage's run state (finished / running / failed / queued /
    pending / not-started), which warm-restart files are present, and the FIRST
    incomplete stage (the one to resume from).  Read-only -- molbuilder
    informs; you decide whether to continue or switch (job-system.md
    § 5.3).  Reuses the same directory decoder as the Results tab.

    With a STAGE -- by name, number or token, like every other verb -- it
    answers the other question instead: *what happened to this one*, with its
    attempts, its launch record and what it continued from.  That form is only
    answerable because a try is a directory and a launch is a record
    (project-layout.md § 1.5, § 1.6).
    """
    js, base = _load(bundle)
    status = jobset_status(js, base)
    if stage is None:
        click.echo(render_status(status))
        return
    click.echo(render_stage_status(status, _resolve_stage_name(js, stage)))


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


def _resolve_stage_name(js, stage: str) -> str:
    """The job ``stage`` names, through the ONE resolver (§ 8f).

    Split out from :func:`_resolve_stage` because two different questions were
    living in one function: *which job did the user name* (every verb that takes
    a STAGE asks this) and *may this verb act on the whole set* (only ``prep``
    and ``submit`` ask, and ``status`` legitimately may). Keeping them together
    would have made ``status <stage>`` either refuse a whole-ladder status or
    grow a second lookup -- and a second lookup is the thing § 8f is about.
    """
    from ..identity import resolve_stage_ref
    from .materialize import stage_refs
    refs = stage_refs(js)
    try:
        return resolve_stage_ref([refs[j.name] for j in js.jobs], stage).name
    except ValueError as e:
        raise click.ClickException(str(e))


def _resolve_stage(js, stage, verb: str):
    """Which jobs a verb acts on, and the refusal when that is ambiguous.

    A LADDER is a sequence you look at between steps, so acting on all of it is
    not merely off by default -- **there is no way to ask for it**.  ``--chain``
    was the way, and it was deleted 2026-08-10 (user) in both modes: whether a
    later stage should pick up an earlier one cannot be settled without
    reviewing the earlier one's result (`project-layout.md` § 1.6).

    A SWEEP has no such ordering: its points are independent, so the whole set
    is the ordinary thing to name here.

    **That still does not decide whether they may all be LAUNCHED**, and
    keeping the two apart is the point: this resolves *which jobs did you
    mean*, and ``submit_jobset`` owns *may this many go at once* -- a scheduler
    takes one per invocation.  A sweep resolves to all its points here and
    ``--mode submit`` still refuses to hand them over together, because the
    refusal has to hold for the web surface and any other caller, not only for
    what is typed.

    Both kinds go through the ONE resolver (§ 8f).  A sweep's refs simply carry
    no ordinal, so it resolves by name and the refusal stops offering numbers --
    the same code path, not a second one.  Until 2026-08-10 the sweep had its
    own lookup, its own refusal wording and its own listing format, so a user
    could be shown two vocabularies for one question.
    """
    from ..identity import render_stage_refs
    from .materialize import stage_refs
    if stage is not None:
        return _resolve_stage_name(js, stage)
    refs = stage_refs(js)
    ordered = [refs[j.name] for j in js.jobs]
    if js.kind == "ladder":
        raise click.ClickException(
            f"this is a ladder, so `{verb}` acts on ONE stage: "
            f"{render_stage_refs(ordered)}.\n"
            f"  molbuilder jobset {verb} run <stage>\n"
            "Stages do not chain, and there is no flag that makes them: a "
            "run that continues on its own can spend a week refining a "
            "geometry you would have rejected in a minute "
            "(project-layout.md § 1.6).")
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

    With no STAGE it lays out
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
    click.echo(f"prepared {rep.stage}: {rep.dir.relative_to(base)}"
               f"{'' if rep.fresh else '  (reused -- not launched yet)'}")
    click.echo(f"  linked: {', '.join(rep.linked)}")
    if rep.copied:
        click.echo(f"  copied from {rep.continued_from}: "
                   f"{', '.join(rep.copied)}")
    elif rep.cold:
        click.echo("  cold start -- nothing copied in")
    else:
        click.echo("  nothing carried in (first stage, or none named)")
    _echo_resolved(js, base, rep.stage, rep.dir)
    click.echo(f"next: molbuilder jobset submit run {stage} "
               f"--mode submit|direct")


def _echo_resolved(js, base, stage_name: str, attempt) -> None:
    """The resolved half of the prep report (P6 unit 6).

    `job-system.md` § 2.3.3: *"**Printing what it resolved is what makes
    `submit` a plain yes.**  It is the only place the measured numbers, the
    chosen geometry and the rendered deck appear together, which is exactly
    where a person should be looking before spending a week."*

    Two of those three are available today, and this prints those two rather
    than a placeholder for the third:

    * the **resources** this stage will be launched with;
    * the **deck's own claim** about the launch it was rendered for, and
      whether the two agree.

    **The second is the point.** P6 unit 2 made `submit` refuse a launch the
    deck was not rendered for — correctly, and at the last honest moment. But
    a refusal that first appears when you are committing cluster time is a
    surprise, and `prep` is the step that exists so there are none. Both read
    :func:`~molbuilder.jobset.prep.launch_agreement`, so the warning here and
    the refusal there cannot disagree.

    **What is still missing, and it is not this function's to invent.** The
    contract's report opens with the measured verdict
    (``reading 02_tight/bench/bench-result.json``) and says the deck was
    ``rendered`` — both of which wait on P6 unit 5 and on the deck moving into
    `prep`. Until then this reports what the deck *already says*, which is a
    weaker claim honestly made rather than the stronger one faked.
    """
    from .prep import launch_agreement
    job = next((j for j in js.jobs if j.name == stage_name), None)
    if job is None:
        return
    r = job.resources
    asks = [f"mpi_np {r.mpi_np if r.mpi_np else 'auto'}",
            f"omp {r.cpus_per_task if r.cpus_per_task else 'auto'}"]
    if r.continue_retries:
        asks.append(f"retries {r.continue_retries}")
    click.echo(f"  resources: {' | '.join(asks)}")

    a = launch_agreement(attempt, job)
    if a.verdict == "silent":
        return                       # the deck makes no claim; say nothing
    deck = Path(job.script).name
    if a.verdict == "agrees":
        click.echo(f"  {deck}: rendered for mpi_np {a.rendered_text} "
                   f"-- agrees with this launch")
        return
    click.echo(click.style(
        f"  {deck}: rendered for mpi_np {a.rendered_text}, but this launch "
        f"asks {a.launch_text}\n"
        f"    submit WILL REFUSE this -- a deck derives values from the rank "
        f"count (BlockSize above all), so one rendered for a different launch "
        f"is wrong for this one.  Re-render it, or launch at "
        f"{a.rendered_text}.", fg="yellow"), err=True)


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
@click.option("--dry-run", is_flag=True,
              help="print the exact command each job WOULD get; launch "
                   "nothing.")
def submit_cmd(kind: str, stage, bundle: str, mode: str, domain,
               dry_run: bool) -> None:
    """Launch a prepped stage: local ``bash`` (direct) or the machine's
    submission system (submit).  Run ``prep`` first.  ``--dry-run`` shows the
    exact command before anything is irreversible."""
    _check_kind(kind)
    js, base = _load(bundle)
    only = _resolve_stage(js, stage, "submit")
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
