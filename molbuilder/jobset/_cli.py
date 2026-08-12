"""``molbuilder jobset ...`` — the calculation's one grammar
(docs/execution/job-system.md § 5.3).

``describe`` writes the portable folder (floor 2); on the machine that runs
it, ``prep`` derives floor 3 and everything below (the five steps of
project-layout.md § 2.3.1), ``submit`` launches ONE job per invocation, and
``summarize`` reads a sweep's results back.  Nothing is produced on a host
and shipped — a bundle carrying a pre-made ``job-set.json`` is the legacy
route, and it narrows with every fold.

The verbs own no policy: ``submit`` PRINTS what it will do and the user
picks the mode and domain explicitly (assistant, not nanny; never a silent
auto-submit).
"""

from __future__ import annotations

from pathlib import Path

import click

from .ledger import record as _ledger
from .model import JobSet
from .plan import render_plan
from .prep import PrepError
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
            "describes it and `prep` derives it (job-system.md § 5.1); run "
            "`molbuilder jobset describe` first.")
    try:
        return JobSet.load(jpath), base
    except ValueError as e:                      # bad schema / shape
        raise click.ClickException(str(e))


@click.group("jobset", short_help="run a job-set bundle (stage ladder / sweep)")
def jobset_group() -> None:
    """The calculation's verbs, one grammar (job-system.md § 5.3):
    ``describe`` writes the portable folder; on the machine that runs it,
    ``prep`` derives everything else and ``submit`` launches one job per
    invocation.  Floor 3 (``job-set.json``) is DERIVED at prep, on the
    target -- nothing is produced on a host and shipped."""


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


def _check_kind(kind: str, js=None) -> None:
    """The KIND positional against the bundle's actual kind.

    ``bench`` stopped refusing on 2026-08-12 (plan step 6, u2): ``prep
    bench`` enumerates the grid on this machine and ``submit bench <trial>``
    launches ONE trial per invocation through the same resolver as
    everything else.  What remains checkable is AGREEMENT: a kind that
    contradicts the bundle's own is a typo about to act on the wrong thing.
    """
    if js is None:
        return
    actual = "bench" if js.kind == "sweep" else "run"
    if kind != actual:
        raise click.ClickException(
            f"this bundle's job set is a {js.kind}, which the grammar calls "
            f"{actual!r} -- and the command says {kind!r}.  The kind states "
            f"what the calculation IS, it does not switch modes "
            f"(job-system.md § 5.3).")


def _stage_bench_dir(base, stage):
    """The stage's ``bench/`` container (job-contracts.md § 6.3), resolved
    through the description — where its trials, its job-set and its verdict
    all live.  Returns ``(container_path, token)``; refuses an unknown
    stage with the ladder listed."""
    from ..task import FILENAME, read_task
    from .prep import _bench_container, _token_for
    desc = Path(base) / FILENAME
    if not desc.is_file():
        return None, None                    # hand-built set: no container
    task = read_task(desc)
    if not task.stages:
        return Path(base) / _bench_container(task, ""), ""
    if stage is None:
        raise click.ClickException(
            "which stage's benchmark? name it: "
            f"{', '.join(s.name for s in task.stages)}.")
    for s in task.stages:
        if s.name == stage:
            token = _token_for(task, s.name)
            return Path(base) / _bench_container(task, token), token
    raise click.ClickException(
        f"no stage named {stage!r} in this description. Available: "
        f"{', '.join(s.name for s in task.stages)}.")


def _pick_trial(js, base, trial, mode):
    """Which trial this invocation launches — § 2.3.2, decided 2026-08-12:
    ONE per invocation under submit.  Named → that one; bare → the NEXT
    UNLAUNCHED (launch recorded as run.json in the trial's dir), with the
    remaining count said out loud.  `--mode direct` is not submission and
    runs the set sequentially, as ever."""
    from .materialize import RUN_LAUNCH_FILE, job_dir_names, shape_of
    if trial is not None:
        if not any(j.name == trial for j in js.jobs):
            raise click.ClickException(
                f"no trial named {trial!r}. This sweep's trials: "
                f"{', '.join(j.name for j in js.jobs)}.")
        _ledger(base, "submit", "trial-picked", trial=trial,
                picked_by="named by the user")
        return trial
    if mode != "submit":
        return None                       # direct: the whole set, in order
    dirs = job_dir_names(js, shape_of(js, base))
    pending = [j.name for j in js.jobs
               if not (Path(base) / dirs[j.name] / RUN_LAUNCH_FILE).is_file()]
    if not pending:
        raise click.ClickException(
            f"all {len(js.jobs)} trials are launched.  next: "
            f"molbuilder jobset summarize bench <stage>")
    click.echo(f"next unlaunched trial: {pending[0]}  "
               f"({len(pending)} of {len(js.jobs)} remain)")
    _ledger(base, "submit", "trial-picked", trial=pending[0],
            picked_by="next unlaunched (run.json absent)",
            remaining=len(pending), total=len(js.jobs))
    return pending[0]


def _load_bench_set(base, stage):
    """The stage's OWN sweep record, from its container — or the root
    job-set for a hand-built (description-less) sweep."""
    container, _ = _stage_bench_dir(base, stage)
    if container is None:
        return _load(str(base))              # legacy/hand-built library sets
    jpath = container / _JOBSET_FILE
    if not jpath.is_file():
        raise click.ClickException(
            f"no {jpath.relative_to(Path(base))} -- this stage has no "
            f"prepped benchmark.  Run `molbuilder jobset prep bench "
            f"{stage}` first.")
    try:
        return JobSet.load(jpath), Path(base)
    except ValueError as e:
        raise click.ClickException(str(e))


def _ask_if_underway(base, stage) -> None:
    """§ 6's moment (run-identity.md, softened 2026-08-08): before writing,
    SAY what is in the folder — and when what is there says a run already
    HAPPENED (a launched attempt's ``run.json``, warm files at the root),
    ask before re-rendering over it (A3/U14, 2026-08-12).

    The default is YES and an unanswerable prompt PROCEEDS, saying so —
    the inverse of the verdict offer, deliberately: applying someone
    else's numbers silently is the thing silence must not do, while
    re-rendering decks is § 6's "ordinary thing to do" (warm files are
    never touched, nothing is renamed), so a scripted re-prep must not
    hang or die on a question it cannot hear.
    """
    from ..task import FILENAME, read_task
    desc = Path(base) / FILENAME
    if not desc.is_file():
        return
    task = read_task(desc)
    from .materialize import attempts, was_launched
    from .shape import Shape
    evidence = []
    if task.stages and stage is not None:
        from .prep import _token_for
        try:
            token = _token_for(task, stage)
        except Exception:
            token = None
        if token:
            sd = Shape.named(task.shape).stage_dir(token)
            d = Path(base) / sd if sd != "." else Path(base)
            for n in attempts(d):
                a = d / f"run-{n}"
                if was_launched(a):
                    evidence.append(
                        f"{a.relative_to(Path(base))}/ was launched "
                        f"(its run.json)")
    from ..validation.identity import warm_files_present
    warm = warm_files_present(base, task.label, task.engine)
    if warm:
        evidence.append("warm files at the root: " + ", ".join(warm))
    if not evidence:
        return
    click.echo("this calculation is already under way here:")
    for e in evidence:
        click.echo(f"    {e}")
    click.echo("  re-rendering replaces the DECKS only: the warm files are "
               "NOT touched and nothing is renamed (run-identity.md § 6).")
    if (Path(base) / ".git").is_dir():
        click.echo("  (a checkpoint repo exists -- `molbuilder checkpoint "
                   "save` first records the current state)")
    try:
        ok = click.confirm("  proceed (re-render the decks)?", default=True)
        answer = "yes" if ok else "no"
    except click.exceptions.Abort:
        click.echo("")
        click.echo("  no answer (non-interactive): proceeding -- § 6 warns, "
                   "it does not refuse.")
        ok, answer = True, "no answer (non-interactive) -> proceed"
    _ledger(base, "prep", "underway-ask", stage=stage,
            evidence=evidence, answer=answer)
    if not ok:
        raise click.ClickException(
            "stopped at your request -- nothing was re-rendered.")


def _offer_bench_verdict(base, allocation, stage=None):
    """§ 2.3.2: a verdict can always be FOUND — finding is not permission.

    A benchmark lives inside the calculation it measured, so `prep run` can
    always see ``bench-result.json``; it shows the choice and ASKS, every
    time, and a non-interactive shell's silence is No (same doctrine as the
    checkpoint question).  On yes, the measured machine half fills only the
    allocation fields the user did NOT state — your explicit flags stay
    yours — and the winning engine's eigensolver arrives as pins.  Returns
    ``(allocation, pins)``.
    """
    import dataclasses as _dc
    import json as _json
    container, _ = _stage_bench_dir(base, stage)
    path = ((container / "bench-result.json") if container is not None
            else Path(base) / "bench-result.json")
    if not path.is_file():
        return allocation, {}
    # Through the TYPED reader (U13): from_dict checks the schema by name
    # and major (persist.check_schema), so a stray artifact of the same
    # major cannot masquerade as a verdict -- raw json.loads checked
    # nothing.
    from ..bench.result import BenchResult
    try:
        res = BenchResult.from_dict(
            _json.loads(path.read_text(encoding="utf-8")))
    except ValueError as e:
        click.echo(f"  (bench-result.json unreadable -- ignored: {e})",
                   err=True)
        return allocation, {}
    choice = res.choice or {}
    rec = res.recommend or {}
    if not choice:
        return allocation, {}
    knobs = choice.get("knobs") or {}
    click.echo(f"a benchmark result exists for "
               f"{'stage ' + repr(stage) if stage else 'this calculation'}:")
    click.echo(f"    {choice.get('rationale', choice)}")
    if rec:
        click.echo(f"    sizing (measured on THAT machine -- a starting "
                   f"point, not a guarantee): {rec}")
    # `generated_at` is the artifact's own key -- this read `generated`
    # (a key nobody writes) until U13, so the measured-when line never
    # showed.
    when = res.generated_at or (res.environment or {}).get("detected_at")
    if when:
        click.echo(f"    measured: {when}")
    try:
        accepted = click.confirm("  use it?", default=False)
    except click.exceptions.Abort:
        # EOF / no stdin: SILENCE IS NO.  confirm() aborts on EOF, which
        # would kill a scripted prep outright -- the doctrine is that the
        # question is asked and an unanswered question declines.
        click.echo("")
        accepted = False
    try:
        _src = str(path.relative_to(Path(base)))
    except ValueError:
        _src = str(path)
    _ledger(base, "prep", "bench-verdict",
            stage=stage, source=_src, accepted=accepted)
    if not accepted:
        click.echo("  not applied -- your flags and defaults stand.")
        return allocation, {}
    stated = {}
    # The knobs already speak the exchange vocabulary (U13: summarize
    # writes the job-set's own field names), so this is a fill-in of what
    # the user did NOT state -- no renaming.
    for field_name in ("mpi_np", "cpus_per_task", "gres"):
        if (knobs.get(field_name) is not None
                and getattr(allocation, field_name) is None):
            stated[field_name] = knobs[field_name]
    if rec.get("mem_gb") and allocation.mem is None:
        stated["mem"] = f"{rec['mem_gb']}GB"
    if rec.get("time") and allocation.time is None:
        stated["time"] = rec["time"]
    if stated:
        allocation = _dc.replace(allocation, **stated)
    # The MECHANISM comes from the winner's own deck, recorded by
    # summarize (U13b) -- until then this pinned ELPA-1STAGE from
    # `engine == "gpu"` alone, inventing the mechanism the measurement
    # never named.
    mech = choice.get("mechanism") or {}
    if mech:
        pins = {k: v for k, v in (("enable_gpu", mech.get("enable_gpu")),
                                  ("diag_algorithm",
                                   mech.get("diag_algorithm")))
                if v is not None}
    else:
        pins = {}
        click.echo("  (this bench-result predates the mechanism record -- "
                   "no engine pins applied; re-run `jobset summarize "
                   "bench` to refresh it)", err=True)
    click.echo("  applied: "
               + (", ".join(f"{k}={v}" for k, v in stated.items()) or "(all "
                  "machine fields were stated explicitly -- flags win)")
               + f"; pins: {pins}")
    return allocation, pins


def _bench_inputs(base):
    """The benchmark specialisation's three inputs — `project-layout.md`
    § 2.3.1a's split, stated as data: WHERE the values come from (the GPU
    grid, enumerated from THIS machine's probed topology, as explicit
    points), the G/K/C → Resources translation, and the trial pins.  The
    framework — `prep`'s five steps — receives a longer list and never asks
    why (`generator.md` § 2).
    """
    from ..bench.adapters import _FALLBACK_KS, get_adapter, sweep_grid
    from ..resolve import MachineTranslation
    from .prep import _environment_for
    environment = _environment_for(base)
    topo = getattr(environment, "topology", None)
    gpn = getattr(topo, "gpus_per_node", None) or 0
    cps = getattr(topo, "cores_per_socket", None)
    gtype = getattr(topo, "gpu_type", None)
    if not gpn or not gtype:
        raise click.ClickException(
            f"prep bench enumerates the GPU grid (G × ranks-per-GPU × "
            f"cores), and this machine's probe found no GPU topology "
            f"(gpus_per_node={gpn!r}, gpu_type={gtype!r}).  Delete "
            f"environment.json to re-probe, or run the benchmark on the "
            f"target it is meant to measure -- the comparison is by node "
            f"type (asu-sol.md § 5.2).")
    ks = get_adapter(environment).sweep_K(topo) or list(_FALLBACK_KS)
    points = [{"G": g, "K": k, "C": c}
              for g, k, c in sweep_grid(gpn, cps, ks, None)]
    translation = MachineTranslation(
        axes=("G", "K", "C"),
        to_resources=lambda p, _env: {
            "mpi_np": p["G"] * p["K"], "cpus_per_task": p["C"],
            "gres": f"gpu:{gtype}:{p['G']}"})
    # The trial pins -- what `transform_fdf` used to SPLICE into a finished
    # deck, now schema values resolved like any other (`template.md` § 8.1:
    # rebuild and render, never splice): capped SCF, single point, forced
    # cold, the GPU eigensolver.  ``SCF.MustConverge`` has NO schema field
    # (the splice invented it); a capped trial therefore reports its
    # nonconvergence honestly rather than being silenced -- adding the field
    # is a recorded vocabulary gap (template.md § 7), not this unit's scope.
    pins = {"max_scf_iter": 5, "relax_steps": 0, "restart": "clean",
            "diag_algorithm": "ELPA-1STAGE", "enable_gpu": True,
            "parallel_block_size": 256}
    return points, pins, translation


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
@click.option("--np", "mpi_np", type=int, default=None, metavar="N",
              help="MPI ranks to ASK FOR. Part of the allocation, not of the "
                   "description -- how a job is scheduled depends on how much "
                   "you ask for, so this is yours to choose per run.")
@click.option("--cpus-per-task", type=int, default=None, metavar="C",
              help="cores per rank (OMP). sbatch -c.")
@click.option("--gpus", "gres", default=None, metavar="TYPE:N",
              help="GPUs, by type -- e.g. a100:1, a100.40gb:1, a30:2. The MIG "
                   "slices are separate askable types, not a smaller ask of "
                   "the same one.")
@click.option("--time", "time_", default=None, metavar="D-HH:MM:SS",
              help="wall time to ask for.")
@click.option("--mem", default=None, metavar="SIZE",
              help="memory for the whole job, e.g. 80GB. '0' asks for all of "
                   "the node's.")
@click.option("--max-memory-mb", type=int, default=None, metavar="MB",
              help="per-rank cap, baked into the wrapper as ulimit -v.")
@click.option("--domain", default=None, metavar="NAME",
              help="which named domain to run in (a scheduler.routing entry -- "
                   "a partition and a QOS together, with its own limits).")
@click.option("--sbatch/--no-sbatch", "emit_sbatch", default=True,
              help="emit .sbatch wrappers (default on; auto-skipped when no "
                   "scheduler is configured).")
def prep_cmd(kind: str, stage, bundle: str, from_attempt, cold: bool, env,
             mpi_np, cpus_per_task, gres, time_, mem, max_memory_mb,
             domain, emit_sbatch: bool) -> None:
    """Set a stage up to run, and report what was done.

    Renders the wrappers, then makes that stage's next ``run-<n>``, links the
    deck and shared package in, and copies in whatever it continues from.
    **Prep printing what it resolved is what makes submit a plain yes** -- it is
    the only place the chosen geometry and the rendered deck appear together.

    With no STAGE it lays out
    every stage's container without opening an attempt, which is what a sweep
    wants and what a ladder needs before its first stage.
    """
    # A DESCRIBED calculation is "a template PLUS task.json"
    # (project-layout.md § 2.1), and `prep` builds everything else from the
    # two.  **Both are required to take this route, and the template is the
    # load-bearing half**: a bundle from before `describe` existed carries a
    # `task.json` and FINISHED DECKS but no template, and for it steps 2 and 3
    # have already happened elsewhere.  Routing on `task.json` alone sent those
    # bundles down a path that then asked for a template they never had.
    from pathlib import Path as _P
    from ..task import FILENAME as _TASK
    base = _P(bundle).resolve()
    if not ((base / _TASK).is_file() and any(base.glob("*.template.toml"))):
        # Described-only (U2/U4, 2026-08-12): the pre-made-bundle arm that
        # stood here had NO producer left -- `describe` is the only writer
        # of floor 2, and the old bench bundles died in step 6 u5.  A
        # folder without the pair gets the next step, not a guess.
        raise click.ClickException(
            f"{base} is not a described calculation -- no task.json + "
            "template pair.  `prep` derives everything from those two "
            "(project-layout.md § 2.1); run `molbuilder jobset describe` "
            "first.  (Hand-built job-sets remain launchable: `submit` and "
            "`status` read job-set.json directly.)")
    if (from_attempt or cold) and stage is None:
        raise click.ClickException(
            "--from / --cold describe ONE stage's attempt; name the stage:\n"
            "    molbuilder jobset prep run <stage> --from 01_coarse/run-0")
    try:
        # The ALLOCATION -- what you ask the scheduler for on THIS prep.
        # Assembled here and nowhere else, so a value cannot reach the wrapper
        # by two roads (generator.md § 4.1).
        from .model import Resources as _Alloc
        allocation = _Alloc(mpi_np=mpi_np, cpus_per_task=cpus_per_task,
                            gres=gres, time=time_, mem=mem,
                            max_memory_mb=max_memory_mb, domain=domain)
        # The five steps, from the DESCRIPTION -- `prep` resolves the
        # machine, resolves the parameters, and renders the decks itself.
        # `bench` is the same call with a longer step 2: the grid as
        # explicit points, the G/K/C translation, and the trial pins
        # (§ 2.3.1a -- benchmarking is prep whose parameters are a set).
        sweep = pins = translation = None
        if kind == "bench":
            if stage is None:
                raise click.ClickException(
                    "prep bench measures ONE stage's configuration; "
                    "name it:\n    molbuilder jobset prep bench <stage>")
            sweep, pins, translation = _bench_inputs(base)
        elif stage is not None:
            # § 2.3.2's other half: a run prepped where a verdict sits
            # is OFFERED it -- asked, never applied silently.
            allocation, verdict_pins = _offer_bench_verdict(
                base, allocation, stage=stage)
            pins = verdict_pins or None
        if kind == "run":
            # § 6's say-what-is-there, and the A3 ask when a run already
            # happened here (U14).
            _ask_if_underway(base, stage)
        from .prep import prep_calculation
        dirs = prep_calculation(base, stage, allocation=allocation,
                                env=env, emit_sbatch=emit_sbatch,
                                sweep=sweep, pins=pins,
                                translation=translation)
        # `prep` WROTE floor 3 as part of those five steps; read it back
        # rather than keeping a second copy in hand, so what the attempt
        # setup below sees is exactly what landed on disk.  A sweep's
        # record lives in the stage's bench/ container (§ 6.3), a run's
        # at the root -- read each from its own home.
        js, _ = (_load_bench_set(str(base), stage) if kind == "bench"
                 else _load(str(base)))
    except PrepError as e:
        raise click.ClickException(str(e))

    # WHERE the effective config came from -- part of "prep prints what it
    # resolved" (job-system.md § 2.3.3): a person debugging a machine
    # difference reads which file supplied each setting, at the moment it
    # took effect (user request 2026-08-12; secrets excluded by design).
    # The same facts go to the bundle's LEDGER: the terminal is gone when
    # a job misbehaves hours later; the bundle is not.
    from ..runtime_config import config_provenance, format_provenance
    prov = config_provenance(project_dir=base)
    click.echo(format_provenance(prov))
    def _rel(d):
        try:
            return str(_P(d).resolve().relative_to(_P(base).resolve()))
        except ValueError:
            return str(d)
    _ledger(base, "prep", "prepped", kind=kind, stage=stage,
            job_dirs=sorted(_rel(d) for d in dirs), provenance=prov)

    if kind == "bench":
        # Trials RUN in their own directories; the attempt machinery below
        # is the run-kind's shape (a stage's run-N), and a sweep's jobs are
        # named by coordinate, not by the stage the ladder names.
        where = f" for stage {stage!r}" if stage else ""
        click.echo(f"prepped {len(dirs)} trial dir(s){where} under {base}:")
        for d in dirs:
            click.echo(f"  {d.name}")
        click.echo("next: molbuilder jobset submit bench <trial> "
                   "-- one trial per invocation")
        return

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
    from .agreement import disagreement_note, launch_agreement
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
    _ledger(base, "prep", "launch-agreement", stage=stage_name,
            verdict=a.verdict, rendered_for=a.rendered_text,
            launching_at=a.launch_text)
    if a.verdict == "agrees":
        click.echo(f"  {deck}: rendered for mpi_np {a.rendered_text} "
                   f"-- agrees with this launch")
        return
    # The WHY and the remedy are the agreement module's one wording
    # (disagreement_note); this surface adds only its framing -- what will
    # happen next if nothing changes.
    click.echo(click.style(
        f"  {deck}: rendered for mpi_np {a.rendered_text}, but this launch "
        f"asks {a.launch_text}\n"
        f"    submit WILL REFUSE this -- " + disagreement_note(a),
        fg="yellow"), err=True)


@jobset_group.command("summarize",
                      short_help="read a sweep's results -> bench-result.json")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("stage", required=False, default=None)
@click.option("--bundle", "bundle", default=".",
              type=click.Path(exists=True, file_okay=False),
              help="the calculation folder (default: the current directory).")
def summarize_cmd(kind: str, stage, bundle: str) -> None:
    """Read the trials' artifacts and write ``bench-result.json`` — a
    recommendation, not a decision (`project-layout.md` § 2.3.2): you read
    it, you decide.

    **Asynchronous by design** (user, 2026-08-12): trials are ordinary jobs
    whose results land in the same logs as any run — the ``.out`` timer, the
    monitor's samples.  Run this after the queue drains; a trial that has
    produced nothing yet reports ``state=unknown`` and one started but
    unfinished ``incomplete`` — never a failure of the set.
    Discovery is keyed by ``job-set.json``'s own data, never by parsing
    directory names back (`job-contracts.md` § 6.3).
    """
    if kind != "bench":
        raise click.ClickException(
            "summarize reads a BENCH sweep's measurements.  A run's own "
            "outputs are the calculation's results -- `jobset status` and "
            "the Watch tab are their readers (job-system.md § 5.3).")
    js, base = _load_bench_set(bundle, stage)
    _check_kind(kind, js)
    from ..bench.summarize import (run_summarize_jobset,
                                   summary_text, utc_now_iso)
    container, _ = _stage_bench_dir(base, stage)
    # The container's job-set holds ONLY this stage's trials (U1), so the
    # SET is the scope and the verdict goes back where they live -- there
    # is no name filter anywhere (U12).  A description-less sweep has no
    # stages to name at all:
    if container is None and stage is not None:
        raise click.ClickException(
            f"this sweep carries no description, so it has no stage named "
            f"{stage!r} -- run `molbuilder jobset summarize bench` bare.")
    res, out_path = run_summarize_jobset(
        js, base,
        out=(container / "bench-result.json") if container is not None
            else None,
        now_iso=utc_now_iso())
    click.echo(summary_text(res, out_path))
    _ledger(base, "summarize", "verdict-written", stage=stage,
            out=str(out_path), points=len(res.points),
            choice=(res.choice or None))
    click.echo("next: molbuilder jobset prep run <stage>   (it will FIND "
               "this verdict and ASK)")


@jobset_group.command("submit", short_help="launch a prepped stage")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("stage", required=False, default=None)
@click.argument("trial", required=False, default=None)
@click.option("--bundle", "bundle", default=".",
              type=click.Path(exists=True, file_okay=False),
              help="the calculation folder (default: the current directory).")
@click.option("--mode", type=click.Choice(["submit", "direct"]), default=None,
              help="HOW to launch, which is a fact about this MACHINE and not "
                   "about the layout: 'direct' = run it here with bash; "
                   "'submit' = hand it to the scheduler molbuilder.json "
                   "configures.  Independent of the description's `shape` "
                   "(engines/stages.md § 6.7) -- a workstation running "
                   "hierarchical is ordinary.  **Defaults to `execution.mode` "
                   "in molbuilder.json**, which is what running-a-job.md § 5.4 "
                   "says gates submission; pass it only to override that.")
@click.option("--domain", default=None,
              help="scheduler.routing domain -> -p/-q (submit mode; EXPLICIT, "
                   "never auto-picked).")
@click.option("--dry-run", is_flag=True,
              help="print the exact command each job WOULD get; launch "
                   "nothing.")
def submit_cmd(kind: str, stage, trial, bundle: str, mode: str, domain,
               dry_run: bool) -> None:
    """Launch a prepped stage: local ``bash`` (direct) or the machine's
    submission system (submit).  Run ``prep`` first.  ``--dry-run`` shows the
    exact command before anything is irreversible.

    ``--mode`` falls back to ``execution.mode``.  `running-a-job.md` § 5.4 says
    that setting is what gates submission, and until 2026-08-11 only ``bench``
    consulted it while this verb demanded the flag on every call -- so the
    config said one thing and the command required another.
    """
    mode_source = "--mode flag"
    if mode is None:
        from ..runtime_config import get_execution
        try:
            mode = (get_execution() or {}).get("mode")
        except Exception as exc:
            # A malformed config -- unreadable file, or an execution.mode
            # that names neither launch mode (get_execution validates; ONE
            # place defines what a mode is) -- is ITS OWN error.  Swallowing
            # it here told the user to set a value they may already have set.
            raise click.ClickException(
                f"execution.mode could not be resolved from molbuilder.json: "
                f"{exc}\n  Fix the config, or pass --mode explicitly for "
                f"this call.") from exc
        if not mode:
            # Unset is a refusal, never a derivation: deciding `submit` from
            # a DETECTED scheduler would gate submission on detection, which
            # running-a-job.md § 5.4 forbids.
            raise click.ClickException(
                "no --mode, and molbuilder.json sets no `execution.mode`.\n"
                "  'direct' runs it here with bash; 'submit' hands it to the "
                "scheduler.  Set execution.mode once for this machine, or pass "
                "--mode for this call (running-a-job.md § 5.4).")
        mode_source = "execution.mode (config)"
    if kind == "bench":
        # The stage's own sweep record, from its bench/ container (§ 6.3).
        js, base = _load_bench_set(bundle, stage)
    else:
        if trial is not None:
            raise click.ClickException(
                "a TRIAL names a benchmark point; `submit run` takes a "
                "stage only (job-system.md § 5.3).")
        js, base = _load(bundle)
    _check_kind(kind, js)
    # The same provenance line prep printed, at the LAST moment before the
    # launch -- the mode above may have come from config, and this names
    # which file said so (user request 2026-08-12).
    from ..runtime_config import config_provenance, format_provenance
    prov = config_provenance(project_dir=base)
    click.echo(format_provenance(prov))
    if kind == "bench" and js.kind == "sweep":
        only = _pick_trial(js, base, trial, mode)
    else:
        only = _resolve_stage(js, stage, "submit")
    try:
        results = submit_jobset(js, base, mode=mode, domain=domain,
                                dry_run=dry_run, only=only)
    except SubmitError as e:
        _ledger(base, "submit", "refused", kind=kind, stage=stage,
                trial=trial, mode=mode, mode_source=mode_source,
                reason=str(e))
        raise click.ClickException(str(e))
    _ledger(base, "submit", "launched", kind=kind, stage=stage,
            mode=mode, mode_source=mode_source, dry_run=dry_run,
            provenance=prov,
            jobs=[{"job": r.name, "status": r.status, "job_id": r.job_id,
                   "returncode": r.returncode} for r in results])
    verb = "WOULD run" if dry_run else "result"
    for r in results:
        tail = (f"job {r.job_id}" if r.job_id else
                (f"rc={r.returncode}" if r.returncode is not None else ""))
        click.echo(f"  {verb}  {r.name:<12} {' '.join(r.command)}  "
                   f"[{r.status}] {tail}".rstrip())
    if not dry_run:
        click.echo("next: molbuilder jobset status   (look before the next "
                   "stage)")
