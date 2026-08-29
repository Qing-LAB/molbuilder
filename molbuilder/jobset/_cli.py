"""``molbuilder jobset ...`` — the calculation's one grammar
(docs/execution/job-system.md § 5.3).

``init`` writes the portable folder (floor 2); on the machine that runs
it, ``prep`` derives floor 3 and everything below (the five steps of
project-layout.md § 2.3.1), ``launch`` launches ONE job per invocation, and
``summarize`` reads a sweep's results back.  Nothing is produced on a host
and shipped — a bundle carrying a pre-made ``job-set.json`` is the legacy
route, and it narrows with every fold.

The verbs own no policy: ``launch`` PRINTS what it will do and the user
picks the mode and domain explicitly (assistant, not nanny; never a silent
auto-submit).

STAGE CONTRACT: ``docs/engines/stages.md`` § 6 — a bare ``§ 6.x`` below means
that document, because this module has no numbered sections of its own and
its grammar reference (``job-system.md``) has no § 6.  Stated here once rather
than repeated at each citation, the way ``task.py`` anchors the same contract.
The rule those citations lean on hardest is § 6.5: **a job always has at least
one stage**, so every verb that acts on a stage is given the stage's name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import click

from .ledger import record as _ledger
from .model import JobSet
from .plan import render_plan
from .prep import PrepError
from .submit import submit_jobset, SubmitError
from .runstatus import jobset_status, render_stage_status, render_status

#: A11: the name comes from the module that writes the file.
from .model import FILENAME as _JOBSET_FILE


def _load(bundle: str) -> tuple:
    """Load ``<bundle>/job-set.json`` -> (JobSet, bundle Path).  A friendly
    error if it isn't there (the host producer writes it; see § 5)."""
    base = Path(bundle)
    jpath = base / _JOBSET_FILE
    if not jpath.is_file():
        raise click.ClickException(
            f"no {_JOBSET_FILE} in {base} -- nothing to do.  The host "
            "describes it and `prep` derives it (job-system.md § 5.1); run "
            "`molbuilder jobset init` first.")
    try:
        return JobSet.load(jpath), base
    except ValueError as e:                      # bad schema / shape
        raise click.ClickException(str(e))


#  The group docstring below IS the `jobset --help` text, so it names verbs a
#  user is about to type.  It said ``describe`` for days after that verb became
#  ``init`` -- help recommending a command the CLI rejects.  Kept honest by
#  `test_machines_listing.py::test_jobset_help_names_live_verbs`, which
#  resolves every verb the text names against the registered commands.  (The
#  note lives here rather than in the docstring: a user reading --help needs
#  the verbs, not our rename history.)
@click.group("jobset", short_help="run a job-set bundle (stage ladder / sweep)")
def jobset_group() -> None:
    """The calculation's verbs, one grammar (job-system.md § 5.3):
    ``init`` writes the portable folder; on the machine that runs it,
    ``prep`` derives everything else and ``launch`` launches one job per
    invocation.  Floor 3 (``job-set.json``) is DERIVED at prep, on the
    target -- nothing is produced on a host and shipped.

    ``probe`` measures a machine and ``machines`` lists the records that
    measuring produced, which is how a calculation is prepared for a
    machine other than this one (preparing-for-another-machine.md)."""
    _echo_config_root()


def _echo_config_root() -> None:
    """The ONE line every jobset verb opens with -- where ``molbuilder.json``
    resolves from before anything else runs (user, 2026-08-23: *"this gives
    the user some root information of the starting point of config
    information"*).

    Not the full ``config:`` provenance block -- that already exists
    (`config_provenance`/`format_provenance`, on ``prep``/``launch``) and
    answers a bigger question, *what took effect*, for the commands that need
    it.  This answers a narrower one that every verb needs even when it never
    reads a `molbuilder.json` section: *which file is the starting point*.  A
    person hitting the ``script_generation.activation`` refusal should not
    have to already know that ``~/.config/molbuilder/molbuilder.json`` is the
    file to edit -- the first line of output says so.

    A Click GROUP callback runs once, before the subcommand's own output, and
    is skipped for ``--help`` and the bare group (verified empirically --
    Click has no documented guarantee either way).  That is exactly the
    placement: every real invocation, no help-text noise.
    """
    import click as _click
    from ..runtime_config import CONFIG_FILENAME, machine_config_path
    path, via = machine_config_path()
    state = "found" if path.is_file() else "not found -- defaults in effect"
    _click.echo(f"{CONFIG_FILENAME}: {path} ({state}, via {via})")


#: The calculation folder, spelled the same way on every verb.
#:
#: It was a POSITIONAL on ``plan`` and ``status`` until 2026-08-10 while
#: ``prep``/``launch`` took ``--bundle``, so one word meant the folder on two
#: verbs and the stage on the other two.  `job-system.md` § 5.3 calls that *"a
#: defect of this section's own making"*: the grammar is
#: ``jobset <verb> <kind> [<stage>]``, and a positional that is sometimes a
#: path has no place in it.  `jobset status tight` answered *"Directory 'tight'
#: does not exist"*, which tells a user they mistyped a path they never meant.
def _bundle_hint(base) -> str:
    """`` --bundle <path from the projects root>``, or "" when the caller
    is already standing in it.

    A printed next-step has to be a command the user can paste.  Naming
    the bundle is what makes it work from anywhere, and omitting it when
    the cwd already IS the bundle keeps the common case short.
    """
    from ..projects import projects_root
    try:
        here = Path(base).resolve()
        if here == Path.cwd().resolve():
            return ""
        rel = here.relative_to(Path(projects_root()).resolve())
    except (ValueError, OSError):
        return ""
    return f" --bundle {rel}"


def _resolve_bundle(ctx, param, value, *, must_exist: bool = True):
    """``--bundle`` names a calculation, and a calculation is always inside
    the projects tree (user, 2026-08-22).

      * not given            -> the working directory
      * anything else        -> read from the projects root, uniformly
      * either way           -> it must be INSIDE the projects root

    **Uniform, with no escape hatch.**  A first cut let ``./x`` and ``../x``
    mean *"beside me"*, mirroring `psml_lib`'s rule.  That was the wrong
    borrowing: the two fields denote different kinds of thing.  `psml_lib`
    points at a LIBRARY OF DATA that legitimately lives anywhere -- a
    shared pseudopotential collection, ``/opt``, a home directory -- so its
    spellings must be able to leave the tree.  ``--bundle`` points at a
    CALCULATION, and a calculation outside the projects tree is not a
    calculation molbuilder manages.  One anchor, therefore, and a fence.

    **The fence is the point, not a side effect.**  `..` segments and
    absolute paths are both resolved and then checked, so no spelling
    reaches outside the tree -- the same containment the sidebar backend
    applies to every path it serves (`files.py`'s "outside every configured
    root").  Two doors onto the same tree that disagreed about what is
    reachable would be one door too many.
    """
    from ..projects import OutsideRoot, contain, projects_root
    root = Path(projects_root()).expanduser()
    raw = str(value)
    p = Path(raw).expanduser()
    if p.is_absolute():
        candidate = p
    elif raw == ".":
        candidate = Path.cwd()
    else:
        candidate = root / p

    # THE fence is `projects.contain`, shared with the sidebar backend.
    # This function had its own copy for one revision -- a second fence
    # around the same tree, and the weaker of the two.
    try:
        real = contain(candidate, root)
    except OutsideRoot as exc:
        if raw == ".":
            raise click.BadParameter(
                f"the working directory ({exc.candidate}) is not inside the "
                f"projects tree ({exc.root}), so it names no calculation.  "
                f"Give the job's path from the projects root, e.g. "
                f"--bundle <project>/<topic>/<calculation>.")
        raise click.BadParameter(
            f"{raw!r} does not name a calculation inside the projects tree "
            f"({exc.root}): {exc.reason}.  Paths are read from the tree's "
            f"root and may not leave it, e.g. "
            f"--bundle <project>/<topic>/<calculation>.")

    if must_exist and not real.is_dir():
        raise click.BadParameter(
            f"{raw!r} is read from the projects root, and {real} is not a "
            f"directory.  Give the job's path from the projects root, e.g. "
            f"--bundle <project>/<topic>/<calculation>.")
    if not must_exist and real.exists() and not real.is_dir():
        raise click.BadParameter(
            f"{raw!r} names {real}, which exists and is not a directory.")
    return str(real)


#: ONE declaration for every verb.  Three verbs re-declared this option
#: inline until 2026-08-22, so the anchor rule would have had to be added in
#: four places -- and the help text had already drifted into two wordings.
#:
#: ``must_exist`` is the ONLY axis on which the verbs differ: five of them
#: act on a calculation that is already there, and `init` creates one.  That
#: is a parameter of the one option, never a second option with its own
#: semantics -- the fence, the anchor rule and the refusal text stay single.
def _bundle_option(must_exist: bool = True):
    return click.option(
        "--bundle", "bundle", default=".",
        callback=(_resolve_bundle if must_exist
                  else _resolve_bundle_may_be_new),
        type=click.Path(file_okay=False),
        help="the calculation, as a path from the PROJECTS ROOT -- e.g. "
             "`--bundle <project>/<topic>/<calculation>` -- which works from "
             "any directory.  Omit it to use the current directory.  Either "
             "way it must be inside the projects tree "
             "(job-contracts.md 2.5b)."
             + ("" if must_exist else "  It need not exist yet."))


def _resolve_structure(ctx, param, value):
    """`--structure` is a tree address too (`job-contracts.md` § 2.5b).

    The same anchor and the same fence as `--bundle`; only the shape at the
    end differs, because this one names a file.  A structure belongs in
    `<project>/structure/` -- which is where the sidebar picks it from, so
    the CLI and the browser cite the same thing the same way.
    """
    from ..projects import OutsideRoot, contain, projects_root
    root = Path(projects_root()).expanduser()
    raw = str(value)
    p = Path(raw).expanduser()
    candidate = p if p.is_absolute() else root / p
    try:
        real = contain(candidate, root)
    except OutsideRoot as exc:
        raise click.BadParameter(
            f"{raw!r} does not name a structure inside the projects tree "
            f"({exc.root}): {exc.reason}.  Paths are read from the tree's "
            f"root, e.g. --structure <project>/structure/water.xyz.")
    if not real.is_file():
        raise click.BadParameter(
            f"{raw!r} is read from the projects root, and {real} is not a "
            f"file.")
    return str(real)


def _resolve_bundle_may_be_new(ctx, param, value):
    """`--bundle` for the verb that CREATES the calculation.

    Same anchor, same fence, one difference: the folder may not be there
    yet.  `projects.contain` already declines to check existence -- that is
    why it can serve both -- so this is the existence check being skipped,
    not a second resolution rule.
    """
    return _resolve_bundle(ctx, param, value, must_exist=False)


@jobset_group.command("init",
                      short_help="create the calculation (floor 2 only)")
@_bundle_option(must_exist=False)
@click.option("--structure", "structure", required=True, metavar="PATH",
              callback=_resolve_structure,
              help="the structure to describe, as a path from the PROJECTS "
                   "ROOT -- e.g. `--structure <project>/structure/water.xyz`. "
                   "It lives in the tree like everything else a calculation "
                   "cites (job-contracts.md 2.5b).")
@click.option("--shape", required=True,
              type=click.Choice(("flat", "hierarchical")),
              help="how the stages sit on disk. REQUIRED and never inferred -- "
                   "inferring it would hand you a directory tree you did not "
                   "ask for (engines/stages.md 6.7).")
@click.option("--stage-strategy", default=None,
              help="which shipped ladder to describe (e.g. publishable). Omit "
                   "for a calculation with a SINGLE parameter set, which is "
                   "described as ONE stage named 'coarse' -- named and "
                   "tokened like any other (6.5).")
@click.option("--name", default=None, metavar="NAME",
              help="what you call this calculation; the label and the run id "
                   "derive from it. Default: the destination folder's name.")
@click.option("--engine", default="siesta",
              type=click.Choice(("siesta", "pyscf")),
              help="whose parameters these are.")
@click.option("--psml-lib", default=None, metavar="DIR",
              # NO click-level exists check, deliberately (2026-08-28): the
              # anchor rule lives in pseudos.resolve_psml_lib (job-contracts
              # § 2.5a) and describe.py validates through it with the
              # teaching refusal.  A click Path(exists=True) checked the
              # WORKING DIRECTORY instead, so from the repo root the two
              # validators refused each other's accepted spelling -- click
              # rejecting the bare in-tree name, the resolver rejecting the
              # cwd-relative one click demanded.  One fact, one door.
              help="where to read pseudopotentials from. A path on THIS "
                   "machine: the files travel with the calculation, the "
                   "path does not.  A bare name (`pseudopotential`) means "
                   "the projects tree the calculation lives in; `./x` means "
                   "beside the calculation; an absolute path means itself.")
@click.option("--vacuum", type=float, default=None, metavar="ANGSTROM",
              help="isolation vacuum (A) per side on isolated axes. Needed "
                   "for a flat or linear molecule from a bare XYZ, which "
                   "otherwise has a degenerate cell.")
@click.option("--calculation", "calculation", default="optimization",
              show_default=True, metavar="TYPE",
              help="which KIND of calculation this describes -- the key "
                   "into the engine's warm-file vocabulary (job-contracts "
                   "4.2a).  The engine's sections define what is legal, "
                   "checked where the vocabulary is read.")
def init_cmd(structure: str, bundle: str, shape: str,
                 stage_strategy, name, engine: str, psml_lib, vacuum,
                 calculation: str) -> None:
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
    from ..config.siesta import SIESTA_STAGE_NAMES, SiestaConfig
    from ..describe import DescribeError, build_description, write_description
    from ..identity import normalise_id
    from ..siesta.input import _detect_species
    from ..pyscf.stages import default_pyscf_stages
    from ..siesta.stages import default_siesta_stages
    from ..task import Stage

    out_dir = _P(bundle)
    run_name = name or out_dir.name

    try:
        struct = _load_structure(structure)
        if vacuum is not None:
            # The same channel the Modify -> Cell tab uses: the vacuum lives on
            # the STRUCTURE, not on the engine config, so every surface that
            # reads this structure sees the same isolation.
            struct = _dc.replace(struct, vacuum=(vacuum, vacuum, vacuum))

        # § 6.5 (2026-08-16): a job always has at least one stage.  Without
        # ``--stage-strategy`` the ladder is ONE stage carrying no overrides
        # -- the calculation that is just the template -- rather than the
        # stage-less shape that used to mean the same thing.  One shape, so
        # the artifact names, the tokens and the directories are the same
        # whether a job has one rung or three.
        # THE LADDER IS THE ENGINE'S, AND THE SHAPE OF IT IS NOT.
        # Both engines describe a ladder the same way -- a tuple of ``Stage``
        # with per-rung ``overrides`` -- and differ only in which values those
        # rungs carry, which is the engine's own preset table.
        _ladder = {"siesta": default_siesta_stages,
                   "pyscf": default_pyscf_stages}[engine]
        _one = SIESTA_STAGE_NAMES[1]        # the shared ladder vocabulary
        if calculation == "vibration":
            # The vibration kind's own ladder (spectra-migration plan § 2):
            # ONE freq stage, PySCF first.  A --stage-strategy names the
            # optimization ladder's tiers, which grade nothing here.
            from ..pyscf.stages import vibration_stages
            if engine != "pyscf":
                raise click.ClickException(
                    f"calculation 'vibration' is PySCF-first (the plan's "
                    f"engine-agnostic shape admits others later); engine "
                    f"{engine!r} has no vibration deck yet.")
            if stage_strategy:
                raise click.ClickException(
                    "--stage-strategy names the optimization ladder's "
                    "tiers; a vibration calculation has one `freq` stage "
                    "whose geometry criteria default to the tight tier "
                    "in its template.")
            stages = tuple(vibration_stages())
        else:
            stages = (tuple(_ladder(stage_strategy))
                      if stage_strategy
                      else (Stage(name=_one, enabled=True, overrides={}),))
        # The label goes through the SAME normaliser Task.label uses, so the
        # template's SystemLabel and the description's id cannot disagree
        # about what this calculation is called.
        label = normalise_id(run_name, what="name",
                             stage_names=tuple(s.name for s in stages))
        # The config the template is written from -- the engine's own class,
        # carrying the one identity field each spells differently.
        if engine == "pyscf":
            from ..config.pyscf import PySCFConfig
            cfg = PySCFConfig(job_name=label)
        else:
            cfg = SiestaConfig(system_label=label, psml_lib=psml_lib)

        desc = build_description(
            struct, cfg, stages,
            engine=engine, shape=shape, name=run_name,
            source=str(structure), calculation=calculation,
            pseudo_species=_detect_species(struct.elements),
        )
        # The struct AS DESCRIBED travels: --vacuum replaced it in memory,
        # and a raw copy of the source dropped that choice on the floor
        # (prep re-rendered the 3 A-default cell over it, found 2026-08-12).
        written = write_description(desc, out_dir, psml_lib=psml_lib,
                                    struct=struct)
    except DescribeError as e:
        raise click.ClickException(str(e))
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e))

    # Always a ladder now (§ 6.5), so there is no second phrasing: one stage
    # reports as "1 stage: coarse", not as "one parameter set (no ladder)".
    ladder = (f"{len(desc.task.stages)} stage(s): "
              f"{', '.join(s.name for s in desc.task.stages)}")
    click.echo(f"Described {desc.label!r} in {out_dir} -- {ladder}, "
               f"shape {shape}.", err=True)
    click.echo("  " + "\n  ".join(p.name for p in written), err=True)
    click.echo(
        f"\nIt names no machine. On the machine that will run it:\n"
        f"  cd {out_dir} && molbuilder jobset prep run <stage>", err=True)


@jobset_group.command("plan",
                      short_help="show the plan (jobs, warm files, resources)")
@_bundle_option()
def plan_cmd(bundle: str) -> None:
    """Print the job-set plan: one row per job — its seq, input deck, warm
    files and resources, in ladder order.  Reads only ``job-set.json`` --
    changes nothing.  (Stages do not chain and nothing here is a
    dependency: what a stage continues from is said at ``prep --from``,
    project-layout.md § 1.6.)"""
    js, _ = _load(bundle)
    click.echo(render_plan(js))


@jobset_group.command("status", short_help="show per-stage status + resume point")
@click.argument("stage", required=False, default=None)
@_bundle_option()
def status_cmd(stage, bundle: str) -> None:
    """Show each stage's run state (finished / running / failed / queued /
    pending / not-started), which warm-restart files are present, and the FIRST
    incomplete stage (the one to resume from).  Read-only -- molbuilder
    informs; you decide whether to continue or switch (job-system.md
    § 5.3).  Reuses the same directory decoder as the Results tab.

    With a STAGE -- its name, or #N its number, like every other verb -- it
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
    bench`` enumerates the grid on this machine and ``launch bench` <trial>``
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
    """The stage's bench container (job-contracts.md § 6.3), resolved
    through the description — where its trials, its job-set and its verdict
    all live.  Returns ``(container_path, token)``; refuses an unknown
    stage with the ladder listed, and a bare invocation the same way —
    § 6.5 gives every description a ladder, so there is always a stage to
    name and never a bare form to fall back to."""
    from ..task import FILENAME, read_task
    from .materialize import bench_container
    from .prep import token_for
    from .shape import Shape
    desc = Path(base) / FILENAME
    if not desc.is_file():
        return None, None                    # hand-built set: no container
    task = read_task(desc)
    sh = Shape.named(task.shape)
    if stage is None:
        raise click.ClickException(
            "which stage's benchmark? name it: "
            f"{', '.join(s.name for s in task.stages)}.")
    for s in task.stages:
        if s.name == stage:
            token = token_for(task, s.name)
            return Path(base) / bench_container(sh, token), token
    raise click.ClickException(
        f"no stage named {stage!r} in this description. Available: "
        f"{', '.join(s.name for s in task.stages)}.")


# ``_bench_positionals`` lived here until 2026-08-16.  It re-bound a lone
# name after ``bench`` to the TRIAL, because a stage-less calculation owned
# no stage to name (final review A-4).  With § 6.5's rule that every
# description has at least one stage, the two positionals after ``bench``
# always mean (stage, trial) and the re-binding can never fire.


def _pick_trial(js, base, trial):
    """Which trial this invocation launches.  NAMED → that one (how a single
    point is re-run); refused by name against the sweep's own list.  Bare →
    ``None`` (--mode direct runs the whole set, in order), and
    bare-under-submit never reaches here at all: the
    dispatch routes it to the grouped door (one exact-fit job per resource
    shelf, § 2.3.2).  A next-unlaunched picker arm stood here for the
    pre-grouping shape; its own docstring called it unreachable, and it
    retired 2026-08-21 (R2-4) with its imports.
    """
    if trial is not None:
        if not any(j.name == trial for j in js.jobs):
            raise click.ClickException(
                f"no trial named {trial!r}. This sweep's trials: "
                f"{', '.join(j.name for j in js.jobs)}.")
        _ledger(base, "launch", "trial-picked", trial=trial,
                picked_by="named by the user")
        return trial
    return None       # bare: --mode direct runs the whole set, in order
                      # (bare-under-submit routes to the grouped door
                      # upstream and never reaches here)


def _load_bench_set(base, stage):
    """The stage's OWN sweep record, from its container — or the root
    job-set for a hand-built (description-less) sweep."""
    container, _ = _stage_bench_dir(base, stage)
    if container is None:
        return _load(str(base))              # legacy/hand-built library sets
    jpath = container / _JOBSET_FILE
    if not jpath.is_file():
        # the bare form is the stageless spelling -- interpolating a None
        # here told the user to run "prep bench None" (2026-08-12 plan A4)
        verb = ("molbuilder jobset prep bench"
                + (f" {stage}" if stage is not None else ""))
        raise click.ClickException(
            f"no {jpath.relative_to(Path(base))} -- "
            f"{'this stage has' if stage is not None else 'this calculation has'} "
            f"no prepped benchmark.  Run `{verb}` first.")
    try:
        return JobSet.load(jpath), Path(base)
    except ValueError as e:
        raise click.ClickException(str(e))


def _ask_if_underway(base, stage, *, bench_container=None) -> None:
    """§ 6's moment (run-identity.md, softened 2026-08-08): before writing,
    SAY what is in the folder — and when what is there says a run already
    HAPPENED (a launched attempt's ``run.json``, warm files at the root),
    ask before re-rendering over it (2026-08-12 plan A3/U14).

    ``bench_container`` NARROWS the evidence to a sweep's launched trials
    (2026-08-12 plan A7; narrowed 2026-08-21, user: "bench always starts cold
    -- there is no point of asking"): `prep bench` re-renders the very
    decks a QUEUED trial's symlinks point at, so THAT is worth a
    question -- while the run's launched attempts and the root's warm
    files, which the run-side ask weighs, cannot be touched by
    re-rendering relabelled cold trial decks and are not asked about.

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
    from .materialize import RUN_LAUNCH_FILE, attempts, was_launched
    from .shape import Shape
    evidence = []
    # A BENCH prep weighs ONE kind of evidence only (user, 2026-08-21:
    # "bench always starts cold -- there is no point of asking"): launched
    # trials in this stage's container, whose decks a queued job may be
    # reading mid-flight (2026-08-12 plan A7).  The run's own attempts
    # and the root's warm files are irrelevant to it -- trial decks are
    # relabelled and forced cold, so re-rendering them cannot touch the
    # run -- and asking about them made every bench re-prep beside a
    # launched run a question with no stakes.
    # No `task.stages` test: `read_task` above refuses a description whose
    # ladder is absent OR empty (`task.py`, engines/stages.md 6.5), so on
    # this path it is always non-empty.  The TYPE is weaker -- it permits
    # `stages=None` when `varies` is None too -- which is why the invariant
    # is named here against the reader that establishes it rather than
    # assumed from the dataclass.
    if bench_container is None and stage is not None:
        from .prep import token_for
        try:
            token = token_for(task, stage)
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
    # (2026-08-12 plan A7 added a second arm here for a STAGELESS calculation,
    # whose attempts sat at the ROOT where the stage gate above could not
    # see them.  § 6.5 retired that shape on 2026-08-16: every attempt now
    # lives under its stage, so the gate above sees all of them.)
    if bench_container is not None:
        # DEPTH-AGNOSTIC, and the NAME comes from the `bench-` component.
        # A trial keeps attempts since 2026-08-27 (`project-layout.md`
        # § 1.5a), so its record is at `bench-<point>/run-<n>/run.json` in
        # hierarchical and `bench-<point>/run.json` in flat.  `**` matches
        # zero or more directories, so one glob answers for both; taking
        # `p.parent.name` would have named the ATTEMPT (`run-0`) rather
        # than the trial in the layout that has one.
        def _trial_name(rec: Path) -> str:
            for part in reversed(rec.parts):
                if part.startswith("bench-"):
                    return part
            return rec.parent.name

        launched = [_trial_name(p) for p in
                    sorted(Path(bench_container)
                           .glob(f"bench-*/**/{RUN_LAUNCH_FILE}"))]
        if launched:
            evidence.append(
                f"launched trial(s) in "
                f"{Path(bench_container).relative_to(Path(base))}/: "
                + ", ".join(launched))
    if bench_container is None:
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



def _measured_on(root: Path) -> Dict[Tuple, str]:
    """The machine KINDS this benchmark's verdict was measured on.

    ``{kind: brief}`` from ``bench-result.json``'s own points -- the record
    `prep run` is about to apply, which carries each trial's ``machine``
    since the monitor's ``[MACHINE]`` line landed (`scheduler.md` R12).
    Reading the record rather than re-walking launch files is the one-door
    rule; the previous reader globbed ``run.json`` for a ``node_type`` the
    probe never wrote, so the S3 check it fed had never once fired.

    ``{}`` means *no trial says* -- a record from before the line -- and a
    reader must not turn that into a match.  Kinds, not hostnames: six
    trials on six identical boxes are one entry (R11).
    """
    import json as _json
    from ..bench.result import BenchResult, machine_brief, machine_kind
    try:
        res = BenchResult.from_dict(_json.loads(
            (Path(root) / "bench-result.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}
    kinds: Dict[Tuple, str] = {}
    for p in res.points:
        k = machine_kind(p.machine)
        if k is not None:
            kinds.setdefault(k, machine_brief(p.machine))
    return kinds


def _refuse_if_measured_elsewhere(base, root, stage) -> None:
    """Refuse to apply a verdict its own measurement cannot support.

    **A refusal, not a warning** (S3).  A warning about a number that is
    already wrong asks the person to do the comparison the framework was
    holding both halves of.  Two grounds, from the P4 table
    (`plans/machine-identity-plan.md`):

    * **the trials ran on several kinds of node** -- the verdict ranked
      measurements of different machines against each other, so there is
      no single basis to carry anywhere (and this is a different fact
      from *unknown*, which stays silent);
    * **the target positively rules the measured kind out** -- its menu
      lists machines and none has the measured core count.  CORES, and
      only cores, because it is the one fact both sides state in one
      vocabulary: the menu names devices in gres tokens, the measurement
      in model names, and a bridge between those would be a guess.

    Silent on the honest unknowns -- no machine recorded, or a target row
    with no ``node_types`` -- because *cannot tell* is not a match (R3).
    A no-``gres`` job may land on ANY node of a queue, GPU nodes
    included, so a menu that contains the measured core count anywhere
    rules nothing out.
    """
    from ..runtime_config import get_routing
    from ..scheduler.place import candidates
    from .summarize import RUN_CONFIG_NAME
    measured = _measured_on(root)
    if not measured:
        return
    ways_out = (f"  Either re-run the benchmark where the run will go, or "
                f"state the allocation yourself with flags -- an explicit "
                f"ask is always honoured, and deleting {RUN_CONFIG_NAME} "
                f"declines the verdict outright.")
    if len(measured) > 1:
        raise click.ClickException(
            f"this benchmark's trials ran on {len(measured)} kinds of node "
            f"({', '.join(sorted(measured.values()))}) -- a verdict that "
            f"ranked different machines against each other has no single "
            f"measurement to carry (execution/submission.md S3).\n"
            + ways_out)
    (kind, brief), = measured.items()
    rows = candidates(get_routing(project_dir=Path(base)), prefer_gpu=False)
    row = rows[0] if rows else None
    groups = list(getattr(row, "node_types", None) or [])
    offered = {str(g.get("cores")) for g in groups
               if isinstance(g, dict) and g.get("cores") is not None}
    cores = kind[0]
    if not offered or cores in offered:
        return
    raise click.ClickException(
        f"this benchmark was measured on {brief} and every machine in "
        f"{getattr(row, 'name', '?')!r}'s menu has "
        f"{' or '.join(sorted(offered))} cores -- the numbers do not "
        f"carry.\n"
        f"  Seconds-per-cycle, peak memory and the walltime derived from "
        f"them describe the hardware they were taken on; applying them to "
        f"a different kind of node is not conservative, it is meaningless "
        f"(execution/submission.md S3).\n"
        + ways_out)





#: WHAT YOU MAY TYPE FOR THE TWO ASKS -- said ONCE, because `prep` and
#: `launch` each take a `--time` and a `--mem` and each used to describe
#: them differently: `prep --time` advertised `D-HH:MM:SS` while
#: `launch --time` advertised "4h, 90m, or a bare number of minutes", and
#: both accepted all of it.  Same tool, same flag name, two stories about
#: what is allowed -- the defect roadmap 7.11 already recorded for `--mem`
#: ("two flags of one name disagreeing about a spelling one of them
#: advertises") and which `--time` was never swept for.
#:
#: These describe the HUMAN edge only.  What the file stores and what
#: reaches `sbatch` is SLURM's own spelling, always, and neither is any of
#: the person's business here (`engines/stages.md` § 6.8a).
TIME_METAVAR = "DURATION"
TIME_HELP = ("wall-clock limit -- `4h`, `90m`, `2-00:00:00`, or a bare "
             "number of minutes.  Unstated, the target queue's own ceiling "
             "is requested -- the full amount the cluster allows there.  "
             "Never derived, never estimated.")
MEM_METAVAR = "SIZE"
MEM_HELP = ("how much TOTAL memory this needs -- `128G`, `80GB`, `0.5T`, or "
            "a bare number of GB.  `0` asks for all of the node's.  "
            "Unstated means the scheduler's own default decides, which is "
            "how a 64-core job came to ask for 128 GB nobody had chosen.")


def _duration(text):
    """`scheduler.quantities.parse_duration`, refusing in click's voice."""
    from ..scheduler.quantities import parse_duration
    try:
        return parse_duration(text)
    except ValueError as e:
        raise click.ClickException(f"--time: {e}")


def _memory(text):
    """`scheduler.quantities.parse_memory`, refusing in click's voice."""
    from ..scheduler.quantities import parse_memory
    try:
        return parse_memory(text)
    except ValueError as e:
        raise click.ClickException(f"--mem: {e}")


def _apply_run_config(base, allocation, stage=None, engine=None):
    """§ 2.3.2: a verdict can always be FOUND — finding is not permission.

    Permission is ``run-config.toml``, the editable proposal `summarize`
    writes beside the record: `prep run` applies what the file says to the
    allocation fields the user did NOT state — your explicit flags stay
    yours — and the file's ``[pins]`` arrive as engine pins.  Editing the
    file is the answer; deleting it declines.  *(Until 2026-08-19 this was
    an interactive ``use it? [y/N]`` — the doctrine is unchanged, the
    answer moved into the tree where a scripted prep can carry it and a
    re-prep weeks later still finds it.)*

    With neither file nor flags, the wrapper's runtime policy sizes the
    launch (`running-a-job.md` § 3) — and that is SAID, per engine, never
    implied (user, 2026-08-19).  Returns ``(allocation, pins)``.
    """
    import dataclasses as _dc
    import json as _json
    from .summarize import RUN_CONFIG_NAME, read_run_config
    container, _ = _stage_bench_dir(base, stage)
    root = container if container is not None else Path(base)
    cfg_path = root / RUN_CONFIG_NAME
    result_path = root / "bench-result.json"

    def _policy_note(applied_any):
        # The explicit no-input default: nothing applied and no
        # launch-shape flag stated -> the wrapper's runtime policy
        # decides, and prep names the policy instead of going quiet.
        if applied_any or any(getattr(allocation, f) is not None
                              for f in ("mpi_np", "cpus_per_task", "gres")):
            return
        if engine == "pyscf":
            click.echo(
                "  no benchmark verdict and no thread flags -- PySCF's "
                "wrapper resolves the OMP thread count at\n"
                "  run time (-omp flag > OMP_NUM_THREADS > the scheduler's "
                "allocation > this node's physical\n"
                "  cores; running-a-job.md § 3).")
        else:
            click.echo(
                "  no benchmark verdict and no rank/thread flags -- the "
                "wrapper sizes the launch at run time on\n"
                "  the machine it lands on (SIESTA: MPI over all physical "
                "cores, clamped to the atom count; a\n"
                "  GPU deck follows the ELPA-CUDA placement policy -- "
                "running-a-job.md § 3)."
                + (f"\n  To measure instead of guess:  molbuilder jobset "
                   f"prep bench {stage}" if stage else ""))

    if not cfg_path.is_file():
        if result_path.is_file():
            # A record with no proposal beside it is one of two stories:
            # summarize concluded nothing (say the census -- roadmap
            # § 0.1 B4), or a verdict exists and the proposal was deleted
            # or never written (deleting declined it; point at summarize).
            from ..bench.result import BenchResult
            try:
                res = BenchResult.from_dict(
                    _json.loads(result_path.read_text(encoding="utf-8")))
            except ValueError as e:
                click.echo(f"  (bench-result.json unreadable -- ignored: "
                           f"{e})", err=True)
                _policy_note(False)
                return allocation, {}
            whose = ("stage " + repr(stage)) if stage else "this calculation"
            if res.choice:
                click.echo(
                    f"  (a bench verdict exists for {whose} but no "
                    f"{RUN_CONFIG_NAME} -- if you deleted it, that "
                    f"declined it;\n   `jobset summarize bench` writes a "
                    f"fresh proposal, or state flags yourself)")
            else:
                by_state = {}
                for p_ in res.points:
                    by_state[p_.state] = by_state.get(p_.state, 0) + 1
                census = ", ".join(f"{n} {s}"
                                   for s, n in sorted(by_state.items()))
                click.echo(
                    f"  (a benchmark record exists for {whose} but "
                    f"concludes nothing -- {census or 'no points'}; "
                    f"submit its trials and `jobset summarize bench` "
                    f"again)")
        _policy_note(False)
        return allocation, {}

    try:
        cfg = read_run_config(cfg_path, engine=engine)
    except ValueError as e:
        # A file the user edited into an unreadable state STOPS the prep:
        # skipping it would silently discard a decision they wrote down.
        raise click.ClickException(str(e))
    # S3 -- A MEASUREMENT IS NOT PORTABLE BY DEFAULT
    # (`execution/submission.md` § 5).  Numbers measured on one kind of node
    # do not describe another: a seconds-per-cycle taken on a 48-core GPU node
    # says nothing about a 128-core CPU node, and a walltime derived from it
    # is not conservative or optimistic, it is meaningless.  The measured
    # kind comes from the verdict's own record -- the monitor's [MACHINE]
    # line, carried through BenchPoint.machine -- since 2026-08-27; the
    # declared `node_type` this read before was never written by the probe,
    # so the check had never fired (machine-identity-plan.md P4).
    _refuse_if_measured_elsewhere(base, root, stage)
    stated = {}
    for field_name in ("mpi_np", "cpus_per_task", "gres", "mem", "time"):
        if (cfg["resources"].get(field_name) is not None
                and getattr(allocation, field_name) is None):
            stated[field_name] = cfg["resources"][field_name]
    if stated:
        allocation = _dc.replace(allocation, **stated)
    pins = dict(cfg["pins"])
    try:
        _src = str(cfg_path.relative_to(Path(base)))
    except ValueError:
        _src = str(cfg_path)
    _ledger(base, "prep", "run-config", stage=stage, source=_src,
            applied=stated, pins=pins)
    click.echo(f"  applied {_src}: "
               + (", ".join(f"{k}={v}" for k, v in stated.items())
                  or "(every field it names was stated explicitly -- "
                     "flags win)")
               + (f"; pins: "
                  + ", ".join(f"{k}={v}" for k, v in pins.items())
                  if pins else "")
               + "\n  (edit or delete the file to change this)")
    return allocation, pins


def _declared_execution_pins(base, engine):
    """`task.json` ``bench``, read as the OVERRIDE LANE it is (user rule,
    2026-08-20; `generator.md` § 4.3a): every non-machine entry overrides
    the template -- several points = an axis to try, ONE point = the value
    in force, applied at prep as a pin for the bench's trials and the run
    alike.  Nothing migrates between files: the description stays exactly
    as edited, and prep is where a declaration is resolved.

    Returns ``(pins, axes, value_axes)``: the one-point non-machine values
    as a pins dict, the machine-answered entries untouched (the grid's
    axes), and the MULTI-point non-machine entries as value axes --
    ``{name: [points...]}`` -- which `_bench_inputs` crosses with the
    machine grid so every trial's deck carries its coordinates
    (§ 4.3a's built rule, 2026-08-21; refused by name until then).

    Refused BY NAME, never repaired -- through the ONE shape checker
    (`validation/task.py::_bench_points_fit_their_items`, R2-5 dedup
    2026-08-21): an enum value outside the item's choices, a non-bool on
    a bool item, a repeated point.  This function carried its own copy
    of those rules; the copies had already diverged (a duplicated
    allocation point -- ``mpi_np: [8, 8, 16]`` -- was refused at
    describe and accepted here, because the local copy classified
    allocation entries before checking them).  The dispatch runs the
    full preflight BEFORE this helper, so in the CLI flow the refusal
    normally lands there with every finding listed; the call here is
    the backstop for direct callers, first-refusal shaped as the CLI's
    errors are.

    MEMBERSHIP is not this function's question (one door): every bench key
    must be an ``execution`` item, and `validation/task.py`'s
    ``_bench_names_a_speed_knob`` owns that rule -- a non-execution name
    was refused upstream and an unknown name here is simply skipped.
    Point SHAPES are `read_task`'s (task.py `_bench_from_obj`): every
    value arrives as a non-empty tuple of scalars, so no scalar/empty
    arms exist here.
    """
    from ..task import FILENAME as TASK_FILENAME, read_task
    from ..validation.task import _bench_points_fit_their_items
    from .. import template as _T

    task = read_task(Path(base) / TASK_FILENAME)
    declared = {k: list(v) for k, v in (task.bench or {}).items()}
    if not declared:
        return {}, {}, {}

    shape_errors = [i for i in _bench_points_fit_their_items(task)
                    if i.severity == "error"]
    if shape_errors:
        raise click.ClickException(shape_errors[0].message)

    # THE ENGINE IS THE DESCRIPTION'S OWN (task.engine): the caller
    # passed the same value, but reading it here keeps this door and
    # the shape checker it calls keyed on one source.
    items = {i.name: i
             for i in _T.select(_T.catalogue(), engine=str(task.engine))
             if "execution" in (i.category or ())}
    pins, axes, value_axes = {}, {}, {}
    for name, pts in declared.items():
        it = items.get(name)
        if it is None:
            continue     # membership is the preflight's refusal, upstream
        if it.allocation:
            axes[name] = pts                 # machine-answered: the grid's
            continue                         # business, never a value
        if len(pts) > 1:
            value_axes[name] = pts           # a VALUE AXIS (§ 4.3a): the
            continue                         # grid multiplies per point
        pins[name] = pts[0]
    return pins, axes, value_axes


#: What makes a trial a MEASUREMENT rather than a run -- schema values
#: resolved like any other (`template.md` § 8.1: rebuild and render, never
#: splice): capped SCF, single point, forced cold, run-once, and
#: ``scf_must_converge: False`` so the cap ends CLEAN.  Applied OVER every
#: declared value (one-point pins and value-axis coordinates alike), and a
#: value AXIS naming one of these is refused -- its trials would be one
#: measurement under many labels.  One spelling for both uses (the pins
#: and that refusal); the per-key whys sit at the application site.
#: ``max_scf_iter: 3`` since 2026-08-21 (user, from measured experience:
#: iterations 3-5 agree within seconds on a 444-atom junction, and the
#: bench reads SCALING and DEPENDENCY -- the knee -- not tight rankings):
#: iteration 1 never forms a timing delta, the iter-2 delta is dropped as
#: warm-up-adjacent, iteration 3 is the one clean sample.  It was 5 (a
#: three-sample mean) from 2026-08-19; older 5-iteration records still
#: average, the reader is shape-blind (tuning.md § 2.12).
_MEASUREMENT_PINS = {"max_scf_iter": 3, "relax_steps": 0, "restart": "clean",
                     "continue_retries": 0, "scf_must_converge": False}


def _gpu_inventory(base):
    """The cluster's GPU ``(per-node count, type)`` from THE gpu domain
    row (`scheduler.place.candidates` -- one walk, shared with the cap and
    the routing, so the grid's device count, the cap and the submission
    can never read three different rows) -- § 4.3a's fallback when THIS
    node's probe has none (a login node).

    ``(None, None)`` when that row records no inventory.  Refuses when it
    records SEVERAL types: choosing one would be a ranking, and the probe
    buried ``best_gpu_type`` for exactly that (scheduler/probe.py, N3) --
    the remedy is curating the row down to the type this bench measures.
    """
    from ..runtime_config import get_routing
    from ..scheduler.place import candidates
    # CAPABILITY, not duration: `prep` asks which nodes have devices, not how
    # long a job may run, so it passes no wall and the answer is the menu's
    # own recommendation -- the first gpu-capable row (R7).
    rows = candidates(get_routing(project_dir=Path(base)), prefer_gpu=True)
    row = rows[0] if rows else None
    # `Domain.devices`, never `row.gpu`: the column has two spellings and this
    # read the map one only, so the documented hand-declared row
    # (`{type, per_node, mem_gb}`) refused below naming its own KEYS as GPU
    # types.  One reader, in the record (`scheduler/record._read_devices`).
    inv = row.devices if row is not None else ()
    if not inv:
        return None, None
    if len(inv) > 1:
        raise click.ClickException(
            f"domain {row.name!r} records several GPU types "
            f"({', '.join(sorted(d.type for d in inv))}), and choosing one "
            f"is not the machine's call.  Edit that row in environment.json "
            f"to keep the type this benchmark should measure.")
    dev = inv[0]
    return (dev.per_node or None), dev.type


def _gpu_core_cap(base):
    """``(max_cores, domain name)`` of THE gpu domain row
    (`scheduler.place.candidates`; ``(None, None)`` when it states no cap --
    the queue then teaches).  Probed since 2026-08-21
    (§ 4.3a): sinfo reports one row per node group, so the GPU nodes'
    own core count is on their row even inside a mixed partition -- on
    Sol, GPU nodes take 48 cores where standard nodes take 128.  The
    row stays the user's to edit.
    """
    from ..runtime_config import get_routing
    from ..scheduler.place import candidates
    # CAPABILITY, not duration: `prep` asks which nodes have devices, not how
    # long a job may run, so it passes no wall and the answer is the menu's
    # own recommendation -- the first gpu-capable row (R7).
    rows = candidates(get_routing(project_dir=Path(base)), prefer_gpu=True)
    row = rows[0] if rows else None
    if row is not None and row.max_cores:
        return int(row.max_cores), str(row.name)
    return None, None


def _bench_inputs(base, target):
    """The benchmark specialisation's three inputs — `project-layout.md`

    ``target`` is REQUIRED, and that is the point (2026-08-24).  It read
    ``target=None`` until a caller forgot it: the browser's prep door called
    ``_bench_inputs(dest)``, Python filled the default, and the grid was
    enumerated against "no machine named" -- which is a MEANINGFUL state
    (one record, no ambiguity) so nothing raised, and it failed only on a
    machine holding two records, only on the write path.  This function
    exists to read a SPECIFIC machine's hardware; letting that be omitted
    made the one fact it needs the one fact a caller could forget.  Pass
    ``None`` deliberately to mean "this machine".

    § 2.3.1a's split, stated as data: WHERE the values come from (the grid,
    enumerated from THIS machine's probed topology, as explicit points), the
    point → Resources translation, and the trial pins.  The framework —
    `prep`'s five steps — receives a longer list and never asks why
    (`generator.md` § 2).

    **The grid is RESOLVED here, at prep — and the description may DECLARE
    it** (`generator.md` § 4.3a, user-settled 2026-08-17, wired 2026-08-19).
    ``task.json``'s ``bench`` names the points to try — *"try 4, 8 and 16
    ranks"* is true on every cluster, so it is portable and belongs with the
    calculation; what those points MEAN on this machine is resolved here.
    With no declaration the machine proposes: the grid is enumerated from
    the probed topology, exactly as before.  (Until 2026-08-19 the
    declaration was read by nothing — this function always enumerated, so
    declaring ``{mpi_np: [1,2,3]}`` produced eleven machine-chosen K×C
    trials, and the user had no say in what was measured.)

    **Whether this is a GPU grid is the DESCRIPTION's answer, not this
    function's assumption** (2026-08-17).  `web/task-setup.md` § 6.2 —
    *"use GPU or not is set up only at the Job Prep UI"* — makes ``use_gpu``
    a value the person chose, carried in the template like any other; and
    § 6.2 is equally explicit that the eigensolver is NOT the same question
    (``diag_algorithm`` is a `budget` item on the parameter tab).  This
    function pinned ``use_gpu=True`` and ``diag_algorithm='ELPA-1STAGE'``
    flat, so every trial measured a GPU regardless of what was asked for —
    and on a machine with no GPU the whole verb refused, which made a
    CPU benchmark impossible to run at all.  Both pins are gone: the
    description answers, and the grid follows its answer.

    **A multi-point non-machine entry is a VALUE AXIS** (§ 4.3a, built
    2026-08-21): its points multiply the machine grid, each point carries
    its coordinates (the resolver's parameter lane applies them per
    trial), and ``use_gpu`` with two points is the grid-FAMILY axis --
    the grid enumerates once per flag, G=0 holding the CPU family's
    device coordinate.  See the section for the cap, naming, and
    split-submission halves of the rule.
    """
    from ..bench.grid import _FALLBACK_KS, sweep_K, sweep_grid
    from ..resolve import MachineTranslation
    from ..task import FILENAME as TASK_FILENAME, read_task
    from ..template import (read_template, template_path,
                            select as template_select)
    from .prep import _environment_for
    # The grid is enumerated from the TARGET's topology, not from whatever
    # box you happen to be typing on (P2, 2026-08-17).  Without this a
    # benchmark prepped on a workstation for a cluster measured the
    # workstation -- 20 cores, no GPU -- and said nothing.
    environment = _environment_for(base, target)
    topo = getattr(environment, "topology", None)
    gpn = getattr(topo, "gpus_per_node", None) or 0
    cps = getattr(topo, "cores_per_socket", None)
    gtype = getattr(topo, "gpu_type", None)

    task = read_task(Path(base) / TASK_FILENAME)
    # THE SEAM REFUSAL, BY NAME (E-J1, restored 2026-08-21).  The bench
    # lane speaks SIESTA's vocabulary today: the measurement pins name
    # SiestaConfig fields (`max_scf_iter`, `restart`, ...), and the GPU
    # question is read under SIESTA's `use_gpu`.  A PySCF description
    # used to be stopped only by ACCIDENT -- those pins failing resolve
    # with a message blaming settings the user never wrote -- and the
    # accident evaporates the day PySCFConfig grows any same-named
    # field, after which a `use_gpu` sweep silently enumerates a CPU
    # grid (generator.md § 12.1 row 9's recorded hazard).
    if str(task.engine) != "siesta":
        raise click.ClickException(
            f"this description's engine is {task.engine!r}, and the "
            f"benchmark lane only speaks SIESTA today: its measurement "
            f"pins (a capped-SCF probe run) name SIESTA settings, so a "
            f"{task.engine} bench would measure nothing meaningful.  "
            f"Benchmark support for other engines is a recorded design "
            f"(generator.md § 12.1 row 9); for now, size the run from "
            f"the engine's own scaling guidance in docs/engines/tuning.md.")
    tmpl = read_template(
        template_path(Path(base), task.label).read_text(encoding="utf-8"))
    # Through `select` -- `template.md` § 8.0 owns the rule.  What it cost
    # HERE: the hand-rolled comprehension ignored ``engines``, so on a PySCF
    # description it read the GPU flag as absent and enumerated a CPU grid,
    # silently -- § 2.2's predicted failure exactly.
    #
    # TWO names answer one question until § 6.3's settled merge is renamed:
    # SIESTA's `use_gpu`, PySCF's `use_gpu`.  Spelling both here is the
    # honest encoding of "an un-renamed pair stays two items", and it collapses
    # to one line when the rename lands.
    #
    # ``select`` rather than ``one`` because the question is *is it on?*: a
    # template that never carried the item answers "no", while ``one`` RAISES
    # on a name the file never had -- right for a caller that NEEDS the item,
    # wrong for one asking whether it exists.
    #
    # THE NAME IS SIESTA'S, AND THAT IS A DEPENDENCY RATHER THAN A CHOICE.
    # The GPU question has no engine-agnostic name yet: `template.md` § 6.3's
    # merge of ``use_gpu`` / ``use_gpu`` is RULED and not yet renamed, so
    # today two names answer one question.  Writing an engine->name table here
    # would put that un-landed rename in a second place to maintain.  Reading
    # SIESTA's name flat is SAFE because the seam refusal above already
    # stopped every non-SIESTA description by name (E-J1, restored
    # 2026-08-21) -- the un-renamed pair can no longer make a `use_gpu`
    # sweep enumerate a CPU grid.  The engine-agnostic bench remains
    # § 12.1 row 9's recorded design.
    # THE DECLARED OVERRIDE LANE, split before anything is decided (user
    # rule, 2026-08-20): one-point non-machine entries are pins -- values
    # in force for every trial -- and the machine-answered entries are the
    # grid's axes.  A declared use_gpu pin OVERRIDES the template's
    # answer below, which is what makes the machine card's choice reach
    # the sweep without touching the template file.
    declared_pins, declared_axes, value_axes = _declared_execution_pins(
        base, task.engine)

    on_gpu = any(i.name == "use_gpu" and bool(i.value)
                 for i in template_select(tmpl, engine=task.engine))
    if "use_gpu" in declared_pins:
        on_gpu = bool(declared_pins["use_gpu"])

    # THE GRID-FAMILY AXIS (§ 4.3a, user 2026-08-21): use_gpu with two
    # points enumerates the machine grid once per flag -- the CPU family
    # holds the device count at G=0, the GPU family ranges it -- and the
    # flag rides each point as an ordinary value coordinate, so the deck's
    # answer and the point's family agree by construction (submit reads
    # the deck, `_job_wants_gpu`, and splits the groups from that answer).
    gpu_flags = None
    if "use_gpu" in value_axes:
        gpu_flags = [bool(v) for v in value_axes.pop("use_gpu")]
    families = gpu_flags if gpu_flags is not None else [bool(on_gpu)]
    mixed = gpu_flags is not None

    # What makes a trial a MEASUREMENT (the pins below) must win over what
    # it measures -- so an axis NAMING a measurement pin would render its
    # trials identical under different labels: one measurement, twice.
    _measured = sorted(set(value_axes) & set(_MEASUREMENT_PINS))
    if _measured:
        raise click.ClickException(
            f"task.json declares {', '.join(_measured)} as a value axis, "
            f"and the benchmark pins "
            f"{'it' if len(_measured) == 1 else 'them'} on every trial (a "
            f"trial is a measurement -- generator.md § 4.3a).  Its points "
            f"would render identical decks under different labels.  Drop "
            f"the entry.")

    if any(families) and (not gpn or not gtype):
        # No GPU on THIS node -- but the cluster behind a login node may
        # still have one: the probe records each partition's gres
        # inventory on its domain row (§ 4.3a), and that answers here.
        _gpn, _gtype = _gpu_inventory(base)
        if _gpn:
            gpn, gtype = _gpn, _gtype
        else:
            raise click.ClickException(
                f"this description asks for the GPU (use_gpu = "
                f"{'a cpu-vs-gpu axis' if mixed else 'true'}), so the "
                f"benchmark enumerates a GPU grid (G × ranks-per-GPU × "
                f"cores) -- and this machine's probe found no GPU topology "
                f"(gpus_per_node={gpn!r}, gpu_type={gtype!r}) and no "
                f"domain row with a recorded GPU inventory.  Delete "
                f"environment.json to re-probe, run `jobset probe --write` "
                f"on the cluster's login node, or run the benchmark on "
                f"the target it is meant to measure -- the comparison is "
                f"by node type (asu-sol.md § 5.2).")

    # The axes come from the split above -- the value entries already left
    # as pins or value axes, so what remains is machine-answered by
    # construction.  (The
    # raw read + unknown-axes refusal that stood here moved into
    # `_declared_execution_pins`, which refuses by name with the § 4.3a
    # story: non-execution items, bad enum/bool values, and the
    # multi-point value axis that is recorded rather than built.)
    declared = {k: list(v) for k, v in declared_axes.items()}
    _KNOWN_AXES = ("mpi_np", "omp_threads", "gpu_count")
    _unresolvable = sorted(k for k in declared if k not in _KNOWN_AXES)
    if _unresolvable:
        # `max_memory_mb` (and PySCF's `threads`) are machine-answered
        # execution items TOO -- the helper hands every allocation item
        # through as an axis, and the grid resolves exactly these two.
        # Without this refusal a declared memory axis was silently ignored
        # (the static review's catch; the pre-split code refused it here).
        raise click.ClickException(
            f"task.json declares bench axes this machine translation does "
            f"not know: {', '.join(_unresolvable)}.  The axes a sweep can "
            f"resolve today are {', '.join(_KNOWN_AXES)} "
            f"(generator.md § 4.3a).")
    sockets = getattr(topo, "sockets", None) or 1
    cores_total = (sockets * cps) if cps else None
    # gpu_count alone does not declare a RANK grid: without mpi_np /
    # omp_threads the K x C half stays the machine's proposal, filtered
    # to the declared device counts below.
    grid_declared = bool(declared.get("mpi_np") or declared.get("omp_threads"))
    if grid_declared:
        # THE DECLARED GRID (§ 4.3a).  ``mpi_np`` is the TOTAL rank count a
        # point runs -- the same meaning it has everywhere else -- and
        # ``omp_threads`` the cores per rank.  A point the machine cannot
        # hold is refused BY NAME, not clamped: a clamped point would
        # measure a configuration nobody declared.
        ranks = [int(v) for v in declared.get("mpi_np") or [1]]
        cores = [int(v) for v in declared.get("omp_threads") or [1]]
        for r in ranks:
            for c in cores:
                if cores_total and r * c > cores_total:
                    raise click.ClickException(
                        f"declared bench point mpi_np={r}, omp_threads={c} "
                        f"needs {r * c} cores and this machine's probe "
                        f"found {cores_total} "
                        f"({sockets} socket(s) x {cps} cores).  Trim the "
                        f"declaration in task.json, or benchmark on the "
                        f"machine it is meant to measure.")

    # THE DECLARED DEVICE COUNTS (user, 2026-08-21: "explicit is what we
    # need").  Declared, gpu_count is exact: those G values and no others,
    # each its own shelf.  A count the machine does not have is refused by
    # name (the same rule as a rank count the machine cannot hold); a
    # (mpi_np, G) pair that cannot split EVENLY is dropped by name below
    # -- ELPA's own rule is the same rank count on every device
    # (tuning.md § 2.12), and refusing the whole prep would deny the
    # divisible cells and the CPU family a rank count they hold fine.
    gpu_counts = ([int(v) for v in declared.get("gpu_count")]
                  if declared.get("gpu_count") else None)
    if gpu_counts and not any(families):
        raise click.ClickException(
            "task.json declares gpu_count, but this bench resolves to the "
            "CPU family (use_gpu is false and not an axis) -- the "
            "device counts would be silently ignored.  Declare use_gpu "
            "= [true] (or the [true, false] axis), or drop gpu_count.")
    if gpu_counts and any(families):
        _over = sorted(g for g in gpu_counts if g > (gpn or 0))
        if _over:
            raise click.ClickException(
                f"task.json declares gpu_count = {_over!r} and this "
                f"machine's record holds {gpn or 0} device(s) per node.  "
                f"Trim the declaration, or benchmark on the machine it "
                f"is meant to measure.")

    def _family_cells(fam):
        """The machine cells of ONE family, as (G, K, C) -- G=0 is the CPU
        family's held coordinate (plain ranks), G>=1 the device count with
        G*K == the total rank count."""
        if grid_declared:
            if fam:
                counts = gpu_counts or range(1, gpn + 1)
                cells = sorted({(g, r // g, c)
                                for r in ranks for c in cores
                                for g in counts if r % g == 0})
                if gpu_counts:
                    bad = sorted({(r, g) for r in ranks for g in gpu_counts
                                  if r % g})
                    if bad:
                        click.echo(
                            "  dropped (ranks must split EVENLY over the "
                            "devices -- ELPA's equal-share rule, "
                            "tuning.md § 2.12): "
                            + ", ".join(f"mpi_np={r} x gpu_count={g}"
                                        for r, g in bad))
                return cells
            return [(0, r, c) for r in ranks for c in cores]
        ks = sweep_K(topo) or list(_FALLBACK_KS)
        # ONE enumeration, both grids (`bench/grid.py`: the single source
        # of truth for the sweep grid, so no two consumers can define it
        # differently).  On CPU there is no device to range over, so G is
        # held at 0 and dropped from a single-family coordinate below.
        # A declared gpu_count FILTERS the probed grid to exactly those
        # device counts.
        return [(g if fam else 0, k, c)
                for g, k, c in sweep_grid(gpn if fam else 1, cps, ks, None)
                if not (fam and gpu_counts) or g in gpu_counts]

    # THE PER-FAMILY CAP (§ 4.3a): when the menu's gpu-capable row states
    # max_cores (probed since 2026-08-21 from the GPU node group's own
    # sinfo row; still hand-editable), a GPU cell that exceeds it is
    # DROPPED BY NAME,
    # never silently and never by refusing the prep -- refusal would deny
    # the CPU family a rank count only the GPU nodes cannot hold.
    cap, cap_dom = (_gpu_core_cap(base) if any(families) else (None, None))
    cells = []
    for fam in families:
        fcells = _family_cells(fam)
        if fam and cap:
            dropped = [(g, k, c) for g, k, c in fcells if g * k * c > cap]
            fcells = [(g, k, c) for g, k, c in fcells if g * k * c <= cap]
            if dropped:
                click.echo(
                    f"  dropped from the GPU family (domain {cap_dom!r} "
                    f"allows {cap} cores/node): "
                    + ", ".join(f"G{g}K{k}C{c} ({g * k * c} cores)"
                                for g, k, c in dropped))
        if fam and fcells:
            # Checks 1-3 of the GPU-sharing note (user, 2026-08-23): ALWAYS
            # state ranks/GPU, warn past MPS's 48-client ceiling, note past
            # this stack's ~4-rank tuned point.  Check 4 (node-fit) is the
            # cap-drop just above -- this does not re-derive it, only
            # states the sharing fact for whatever survived it.  ONE
            # function (`ask.gpu_share_notes`) so this and the submission
            # display can never disagree about the arithmetic.
            from .ask import gpu_share_notes
            shares = sorted({(g, k) for g, k, c in fcells})
            bits = []
            for g, k in shares:
                notes = gpu_share_notes(g, k)
                flag = ""
                if len(notes) > 1:
                    flag = ("  <- WARNING, past MPS's ceiling"
                           if "WARNING" in notes[1] else
                           "  <- past the tuned point")
                bits.append(f"G{g}K{k}: {k} rank(s)/GPU{flag}")
            click.echo(f"  GPU sharing in this family: " + ", ".join(bits))
        if fam and not fcells:
            # every GPU cell fell to the cap or the even-split rule --
            # the drops were echoed by name above, so this names the
            # consequence.
            if mixed:
                click.echo(
                    "  NOTE: no GPU cell survived the drops above -- "
                    "this sweep measures only the CPU family.")
            else:
                raise click.ClickException(
                    "every GPU cell was dropped (see the lines above: "
                    "the domain's core cap and/or the even-split rule).  "
                    "Adjust mpi_np / gpu_count in task.json, or "
                    "benchmark where the GPU nodes are larger.")
        cells.extend((fam, cell) for cell in fcells)
    # THE VALUE-AXIS CARTESIAN (§ 4.3a): every machine cell is crossed
    # with the remaining declared value axes, in declaration order, and
    # each point CARRIES its coordinates -- the resolver's ordinary
    # parameter lane applies them to that trial's config (provenance
    # "sweep"), and the coordinate rides the trial's name and, as data,
    # `job-set.json`'s per-trial ``point``.
    if not cells:
        raise click.ClickException(
            "no bench cell survived the declaration on this machine -- "
            "see the drop/refusal lines above.")
    combos = [{}]
    for name, vals in value_axes.items():
        combos = [{**c, name: v} for c in combos for v in vals]

    points = []
    for fam, (g, k, c) in cells:
        if mixed:
            coord = {"G": g, "K": k, "C": c, "use_gpu": fam}
        elif on_gpu:
            coord = {"G": g, "K": k, "C": c}
        else:
            coord = {"K": k, "C": c}
        points.extend({**coord, **vc} for vc in combos)

    if mixed:
        # ONE translation serves both families: G=0 maps to plain ranks
        # and NO gres, G>=1 to G*K ranks plus the device ask -- which is
        # what lets `launch` put the CPU group on an allocation that
        # holds no device (§ 4.3a's split submission).
        translation = MachineTranslation(
            axes=("G", "K", "C"),
            to_resources=lambda p, _env: (
                {"mpi_np": p["G"] * p["K"], "cpus_per_task": p["C"],
                 "gres": f"gpu:{gtype}:{p['G']}"} if p["G"] else
                {"mpi_np": p["K"], "cpus_per_task": p["C"]}))
    elif on_gpu:
        translation = MachineTranslation(
            axes=("G", "K", "C"),
            to_resources=lambda p, _env: {
                "mpi_np": p["G"] * p["K"], "cpus_per_task": p["C"],
                "gres": f"gpu:{gtype}:{p['G']}"})
    else:
        translation = MachineTranslation(
            axes=("K", "C"),
            to_resources=lambda p, _env: {
                "mpi_np": p["K"], "cpus_per_task": p["C"]})
    # The trial pins -- what `transform_fdf` used to SPLICE into a finished
    # deck, now schema values resolved like any other (`template.md` § 8.1:
    # rebuild and render, never splice): capped SCF, single point, forced
    # cold -- and ``scf_must_converge: False``, the switch that makes the
    # cap CLEAN (item added 2026-08-19: until then the keyword had no
    # schema field, the retired splicer used to invent the line, and every
    # properly-capped trial ended ABNORMAL_TERMINATION, classified
    # incomplete, and could never win -- `choose_winner` ranks only
    # completed points, so a sweep could not produce a verdict at all).
    #
    # What is pinned here is what makes a trial a MEASUREMENT rather than a
    # run.  What the calculation IS -- the GPU, the eigensolver, the block
    # size -- is the description's, and pinning it here would measure a
    # configuration nobody asked to run.
    #
    # ``continue_retries: 0`` is what makes the trial run ONCE.  Without it
    # the capped SCF above guarantees non-convergence, the wrapper's retry
    # budget reads that as a failure and re-runs, and `summarize` reads the
    # HIGHEST run index -- so every trial that retried was timed on its second
    # run and every trial that did not was timed on its first.  Those are not
    # comparable, which is the one thing a sweep exists to be.  (Until the
    # `restart: clean` group was written out rather than omitted, the second
    # run was also WARM, so the second measurement was of a different
    # calculation as well as a different run.)
    # The declared values ride UNDER the measurement pins: what makes a
    # trial a measurement (capped SCF, forced cold, run-once) must win
    # over any declaration -- one-point declarations and value-axis
    # coordinates alike (the resolver applies pins over a point's
    # parameters, and the overlap refusal above bars an AXIS on a pin).
    pins = {**declared_pins, **_MEASUREMENT_PINS}
    return points, pins, translation


def _resolve_stage_name(js, stage: str) -> str:
    """The job ``stage`` names, through the ONE resolver (§ 8f).

    Split out from :func:`_resolve_stage` because two different questions were
    living in one function: *which job did the user name* (every verb that takes
    a STAGE asks this) and *may this verb act on the whole set* (only ``prep``
    and ``launch`` ask, and ``status`` legitimately may). Keeping them together
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


# ``_lone_stageless_job`` lived here until 2026-08-16: the door's answer
# when there was no stage name to type.  `engines/stages.md` § 6.5 now says
# every description carries at least one stage, and one stage is named and
# tokened like any other, so there is always a name to type and the bare
# verbs have nothing to fall back to.  Deleted rather than left inert --
# a helper whose docstring cites a rule that now says the opposite is worse
# than no helper.


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
@_bundle_option()
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
@click.option("--time", "time_", default=None, metavar=TIME_METAVAR,
              help=TIME_HELP)
@click.option("--mem", default=None, metavar=MEM_METAVAR, help=MEM_HELP)
@click.option("--max-memory-mb", type=int, default=None, metavar="MB",
              help="per-rank cap, baked into the wrapper as ulimit -v.")
@click.option("--domain", default=None, metavar="NAME",
              help="which named domain to run in (a PROBED domain from "
                   "environment.json -- a partition and a QOS together, "
                   "with its own limits).")
@click.option("--target", default=None, metavar="NAME",
              help="which MACHINE this is for -- a record written by "
                   "`jobset probe --write --name NAME`.  Omit when there is "
                   "one; naming it is how a bench prepped on a workstation "
                   "measures the cluster instead of the desk.")
@click.option("--sbatch/--no-sbatch", "emit_sbatch", default=True,
              help="emit .sbatch wrappers (default on; auto-skipped when no "
                   "scheduler is configured).")
@click.option("--pipeline-log", "pipeline_log", is_flag=True, default=False,
              help="write a step-by-step record of what each step received, "
                   "decided and produced, beside this prep's STAGE-PLAN.md. "
                   "Answers 'where did this value come from?' without "
                   "re-running anything. Off by default; no generated file "
                   "differs either way.")
def prep_cmd(kind: str, stage, bundle: str, from_attempt, cold: bool, env,
             mpi_np, cpus_per_task, gres, time_, mem, max_memory_mb,
             domain, target, emit_sbatch: bool, pipeline_log: bool) -> None:
    """Set a stage up to run, and report what was done.

    Renders the wrappers, then makes that stage's next ``run-<n>``, links the
    deck and shared package in, and copies in whatever it continues from.
    **Prep printing what it resolved is what makes submit a plain yes** -- it is
    the only place the chosen geometry and the rendered deck appear together.

    A STAGE is required on a ladder — bare ``prep run`` is refused by
    resolve with the ladder listed by name (`engines/stages.md` § 6.5;
    until 2026-08-21 the stage-less form died earlier, on the BENCHMARK's
    "which stage's benchmark?" question — review B1).
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
    from ..template import find_template as _find_template
    base = _P(bundle).resolve()
    # `find_template` rather than a glob: it REFUSES a folder holding two
    # rather than answering from the first one alphabetically.
    if not ((base / _TASK).is_file() and _find_template(base) is not None):
        # Described-only (U2/U4, 2026-08-12): the pre-made-bundle arm that
        # stood here had NO producer left -- `describe` is the only writer
        # of floor 2, and the old bench bundles died in step 6 u5.  A
        # folder without the pair gets the next step, not a guess.
        raise click.ClickException(
            f"{base} is not a described calculation -- no task.json + "
            "template pair.  `prep` derives everything from those two "
            "(project-layout.md § 2.1); run `molbuilder jobset init` "
            "first.  (Hand-built job-sets remain launchable: `launch` and "
            "`status` read job-set.json directly.)")
    if (from_attempt or cold) and stage is None:
        # Every description has a ladder (§ 6.5), so an attempt always
        # belongs to a named stage.  Until 2026-08-16 this was conditional:
        # a stage-LESS calculation had exactly one attempt line, so the bare
        # flags were unambiguous there (R1).  That shape is gone.
        raise click.ClickException(
            "--from / --cold describe ONE stage's attempt; name the "
            "stage:\n"
            "    molbuilder jobset prep run <stage> --from "
            "01_coarse/run-0")
    try:
        # The ALLOCATION -- what you ask the scheduler for on THIS prep.
        # Assembled here and nowhere else, so a value cannot reach the wrapper
        # by two roads (generator.md § 4.1).
        from .model import Resources as _Alloc
        allocation = _Alloc(mpi_np=mpi_np, cpus_per_task=cpus_per_task,
                            gres=gres, time=time_, mem=mem,
                            max_memory_mb=max_memory_mb, domain=domain)
        # § 6.6's preflight, at its LIVE moment (R5, 2026-08-12).  It ran
        # only inside `describe`, BEFORE the fingerprint stamp -- where the
        # description had just been built from one schema and could not
        # mismatch itself.  The hazard the fingerprint row exists for is
        # HERE: prep on a machine whose molbuilder differs from the
        # description's author.  Errors refuse before anything renders; the
        # fingerprint row warns and proceeds (§ 6.6's one non-refusal), and
        # the note lands in the ledger.
        from ..task import read_task as _rt_preflight
        from ..template import SUFFIX as _TPL_SUFFIX
        from ..validation.task import preflight as _preflight
        _pf_task = _rt_preflight(base / _TASK)
        # THE stage grammar, at the door (user-settled 2026-08-21): a
        # ladder stage is named by its NAME, or by `#N` (its assigned
        # number -- the NN in its 02_tight directory).  Resolved here,
        # pre-produce, through the ONE resolver with refs built from the
        # FULL ladder (token_for's own ordinal rule), so `prep run #2`
        # and `status #2` cannot disagree about which stage that is.
        if stage is not None and getattr(_pf_task, "stages", None):
            from ..identity import StageRef, resolve_stage_ref
            _refs = StageRef.ladder([_s.name for _s in _pf_task.stages])
            try:
                stage = resolve_stage_ref(_refs, stage).name
            except ValueError as _e:
                raise click.ClickException(str(_e))
        # The template rides along when it is where prep will look for it,
        # which adds § 6.4/§ 6.6a's SEQUENCE warnings (resolved stages) to
        # the preflight -- reachable on a production path since 2026-08-13
        # (A-8).  Absent or unreadable, prep's own refusal owns the story.
        _tpl_file = base / f"{_pf_task.label}{_TPL_SUFFIX}"
        _pf_issues = _preflight(
            _pf_task,
            template_text=(_tpl_file.read_text(encoding="utf-8")
                           if _tpl_file.is_file() else None))
        _pf_errs = [i for i in _pf_issues if i.severity == "error"]
        _pf_warns = [i for i in _pf_issues if i.severity != "error"]
        for _i in _pf_warns:
            click.echo(f"note: {_i.message}", err=True)
        if _pf_warns:
            _ledger(base, "prep", "preflight-report", stage=stage,
                    notes=[_i.message for _i in _pf_warns])
        if _pf_errs:
            raise click.ClickException(
                "the description fails its own preflight "
                "(engines/stages.md § 6.6):\n  - "
                + "\n  - ".join(_i.message for _i in _pf_errs))

        # The five steps, from the DESCRIPTION -- `prep` resolves the
        # machine, resolves the parameters, and renders the decks itself.
        # `bench` is the same call with a longer step 2: the grid as
        # explicit points, the G/K/C translation, and the trial pins
        # (§ 2.3.1a -- benchmarking is prep whose parameters are a set).
        sweep = pins = translation = None
        if kind == "bench":
            if stage is None:
                # A benchmark measures ONE stage's configuration, and there
                # is always a stage to name (§ 6.5).  Until 2026-08-16 this
                # refusal was conditional, because a stage-LESS calculation
                # had no name to give and its bare bench measured the one
                # parameter set (2026-08-12 plan A4).
                raise click.ClickException(
                    "prep bench measures ONE stage's configuration; "
                    "name it:\n    molbuilder jobset prep bench <stage>")
            # The SAME which-machine refusal the run arm gives (below):
            # both are answers only the user has, and this arm reached
            # `_environment_for` before the run arm's catch -- so an
            # unnamed target on a two-record machine leaked a TRACEBACK
            # here while `prep run` spoke plainly (workflow.md § 9: a
            # gate refuses with the reason, never a stack trace; found
            # live 2026-08-28).
            from ..scheduler import AmbiguousTarget, UnknownTarget
            try:
                sweep, pins, translation = _bench_inputs(base, target)
            except (UnknownTarget, AmbiguousTarget) as exc:
                raise click.ClickException(str(exc))
        elif kind == "run":
            # § 2.3.2's other half: the stage's run-config.toml (the
            # editable proposal summarize wrote) fills the allocation
            # fields the user did not state; with neither file nor
            # flags, the wrapper's runtime policy is NAMED, not implied.
            #
            # SKIPPED when no stage is named (review 2026-08-21, B1): a
            # verdict is per stage, and reaching for it stage-less died
            # with the BENCHMARK's "which stage's benchmark?" question --
            # falling through instead lets resolve give the right
            # refusal, the ladder listed by name.
            verdict_pins = None
            if stage is not None:
                allocation, verdict_pins = _apply_run_config(
                    base, allocation, stage=stage, engine=_pf_task.engine)
            # The declaration's one-point values pin the run too (user
            # rule, 2026-08-20), UNDER the measured verdict: template <
            # declaration < run-config < flags (§ 4.3a's precedence).
            # value_axes pin nothing at `prep run` (§ 4.3a): the
            # verdict's run-config answers; absent one, the template
            # stands.
            declared_pins, _axes, _value_axes = _declared_execution_pins(
                base, _pf_task.engine)
            pins = {**declared_pins, **(verdict_pins or {})} or None
        if kind == "run":
            # § 6's say-what-is-there, and the 2026-08-12 plan A3/U14 ask
            # when a run already happened here.
            _ask_if_underway(base, stage)
        else:
            # A7: `prep bench` re-renders the decks a QUEUED trial's links
            # point at -- same moment, wider evidence (launched trials in
            # the stage's container).  An unresolvable container is left
            # to prep's own refusal.  (Until 2026-08-13 the stageless arm
            # hardcoded `base/"bench"` here -- a second spelling of the
            # container rule, the C-c habit.)
            try:
                cont, _tok = _stage_bench_dir(base, stage)
            except click.ClickException:
                cont = None
            _ask_if_underway(base, stage, bench_container=cont)
        from .prep import prep_calculation
        from ..scheduler import AmbiguousTarget, UnknownTarget
        try:
            dirs = prep_calculation(base, stage, allocation=allocation,
                                    env=env, emit_sbatch=emit_sbatch,
                                    sweep=sweep, pins=pins,
                                    translation=translation, target=target,
                                    pipeline_log=pipeline_log)
        except (UnknownTarget, AmbiguousTarget) as exc:
            # Both are the same class of refusal -- WHICH machine is this
            # for -- and both are answers only the user has
            # (`preparing-for-another-machine.md` § 4).
            raise click.ClickException(str(exc))
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
            # the path from the bundle, not the bare attempt name -- with
            # the attempt layer every trial's dir ENDS in run-<n>, and a
            # list reading "run-0, run-0" names nobody (user, 2026-08-28)
            click.echo(f"  {_rel(d)}")
        # The REAL grammar (E-J2, 2026-08-21 review): `launch bench
        # <stage>` -- typed with a trial name, the trial binds as the
        # STAGE and is refused; and the shipped submission is the
        # grouped one, not one-per-invocation.
        #
        # The hint names the BUNDLE, because `--bundle` reads it from the
        # projects root, so the line works from wherever the user is
        # standing (job-contracts.md § 2.5b).
        _b = _bundle_hint(base)
        click.echo(f"next: molbuilder jobset launch bench "
                   f"{stage or '<stage>'}{_b}   (grouped: one job per "
                   f"resource shelf)")
        click.echo(f"then: molbuilder jobset summarize bench "
                   f"{stage or '<stage>'}{_b}"
                   f"  -- writes bench-result.json + run-config.toml "
                   f"(the editable proposal `prep run` applies)")
        return

    # `stage` is not None here: § 6.5 gives every description a ladder, so a
    # bare `prep run` is refused by `resolve` -- with the ladder listed --
    # long before this point.  A listing branch stood here until 2026-08-16,
    # printing "prepped N job dir(s)" and pointing at `prep run <stage>`; it
    # was unreachable (prep_calculation raises first), and `job-system.md`'s
    # grammar table advertised it as a shipped form.  Both are gone.
    attempt_target = stage

    from .materialize import prepare_attempt, shape_of
    sh = shape_of(js, base)
    if (sh is not None and not sh.keeps_attempts_as_directories
            and not from_attempt and not cold):
        # FLAT keeps no attempt directories, so a flat prep is COMPLETE
        # right here: the wrappers are rendered, the shared warm set lies
        # in the root, and continuing is free (project-layout.md § 1).
        # The attempt tail below is the CLI's OWN addition -- until A2
        # (2026-08-12) it ran anyway and prepare_attempt's flat refusal
        # turned a SUCCESSFUL prep into exit 1.  An explicit --from or
        # --cold still falls through: the user asked for attempt
        # machinery flat does not have, and prepare_attempt owns that
        # refusal's text.
        click.echo(f"prepped {len(dirs)} job dir(s) under {base}  "
                   "(flat: no attempt to open; runs are told apart by "
                   "the wrapper's output index)")
        click.echo("next: molbuilder jobset launch run "
                   + (f"{stage} " if stage is not None else "")
                   + f"--mode submit|direct{_bundle_hint(base)}")
        return
    try:
        rep = prepare_attempt(js, base, attempt_target,
                              continue_from=from_attempt,
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
    click.echo("next: molbuilder jobset launch run "
               + (f"{stage} " if stage is not None else "")
               + f"--mode submit|direct{_bundle_hint(base)}")


def _echo_resolved(js, base, stage_name: str, attempt) -> None:
    """The resolved half of the prep report (P6 unit 6).

    `job-system.md` § 2.3.3: *"**Printing what it resolved is what makes
    `launch` a plain yes.**  It is the only place the measured numbers, the
    chosen geometry and the rendered deck appear together, which is exactly
    where a person should be looking before spending a week."*

    Two of those three are available today, and this prints those two rather
    than a placeholder for the third:

    * the **resources** this stage will be launched with;
    * the **deck's own claim** about the launch it was rendered for, and
      whether the two agree.

    **The second is the point.** P6 unit 2 made `launch` refuse a launch the
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
@_bundle_option()
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
    from .summarize import (run_summarize_jobset,
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
    res, out_path, run_config = run_summarize_jobset(
        js, base,
        out=(container / "bench-result.json") if container is not None
            else None,
        now_iso=utc_now_iso(), stage=stage)
    click.echo(summary_text(res, out_path, run_config=run_config,
                            stage=stage))
    _ledger(base, "summarize", "verdict-written", stage=stage,
            out=str(out_path), points=len(res.points),
            choice=(res.choice or None),
            run_config=str(run_config[0]), run_config_status=run_config[1])


@jobset_group.command("launch", short_help="launch a prepped stage")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("stage", required=False, default=None)
@click.argument("trial", required=False, default=None)
@_bundle_option()
@click.option("--mode", type=click.Choice(["submit", "direct", "ask"]),
              default=None,
              help="HOW to launch, which is a fact about this MACHINE and not "
                   "about the layout: 'direct' = run it here with bash; "
                   "'submit' = hand it to the scheduler molbuilder.json "
                   "configures; **'ask' = submit NOTHING and report when it "
                   "would start** (`sbatch --test-only`), so you can change "
                   "the domain or the resources and ask again before "
                   "committing.  'ask' needs a login node -- there is no "
                   "prediction without the cluster.  Independent of the description's `shape` "
                   "(engines/stages.md § 6.7) -- a workstation running "
                   "hierarchical is ordinary.  **Defaults to `execution.mode` "
                   "in molbuilder.json**, which is what running-a-job.md § 5.4 "
                   "says gates submission; pass it only to override that.")
@click.option("--domain", default=None,
              help="probed domain (environment.json) -> -p/-q (submit mode). "
                   "OVERRIDES the per-side preference a grouped bench applies "
                   "(cpu group prefers a cpu-only domain, gpu group a "
                   "gpu-capable one -- generator.md § 4.3a); combine with "
                   "--only to place one side at a time.")
@click.option("--dry-run", is_flag=True,
              help="print the exact command each job WOULD get; launch "
                   "nothing.")
@click.option("--time", "time_text", default=None, metavar=TIME_METAVAR,
              help=TIME_HELP)
@click.option("--mem", "mem_text", default=None, metavar=MEM_METAVAR,
              help=MEM_HELP)
@click.option("--gpu-domain", "gpu_domain", default=None, metavar="NAME",
              help="the queue the GPU side of a SPLIT sweep goes to, when "
                   "it differs from --domain.  A cpu-only partition cannot "
                   "take the GPU side, so one queue cannot always answer for "
                   "both.  Omit it and the GPU side takes --domain.")
@click.option("--yes", "-y", "auto_yes", is_flag=True,
              help="submit without showing the request first.  Without it "
                   "you see exactly what will be asked for, and say so.")
@click.option("--trial-timeout", "trial_timeout_min", default=None,
              metavar="MINUTES",
              help="bench + submit mode: kill any single trial after this "
                   "many minutes so the rest of the group still runs; the "
                   "killed trial reads incomplete.  Unstated, no per-trial "
                   "bound exists -- each trial runs until the job's wall.")
@click.option("--only", "only_side", default=None,
              type=click.Choice(["cpu", "gpu"]),
              help="bench + submit mode: submit just this side of a sweep "
                   "that spans CPU and GPU trials (generator.md § 4.3a).  "
                   "The other side stays pending; a later `launch bench`` "
                   "collects it -- here or on the cluster that reaches it.")
def submit_cmd(kind: str, stage, trial, bundle: str, mode: str, domain,
               dry_run: bool, time_text, mem_text, gpu_domain,
               auto_yes, trial_timeout_min, only_side) -> None:
    """Launch a prepped stage: local ``bash`` (direct) or the machine's
    submission system (submit).  Run ``prep`` first.  ``--dry-run`` shows the
    exact command before anything is irreversible.

    ``--mode`` falls back to ``execution.mode``.  `running-a-job.md` § 5.4 says
    that setting is what gates submission, and until 2026-08-11 only ``bench``
    consulted it while this verb demanded the flag on every call -- so the
    config said one thing and the command required another.
    """
    mode_source = "--mode flag"
    domain_source = "--domain flag" if domain else None
    # The BUNDLE's scope gates its own launches (R3, 2026-08-12): this
    # called get_execution() with no project_dir, so a calculation's
    # .molbuilder.json execution block never reached submission while the
    # provenance echo below claimed it did (running-a-job § 5.2: the
    # project scope wins).
    _execn = {}
    if mode is None or domain is None:
        from ..runtime_config import get_execution
        try:
            _execn = get_execution(project_dir=Path(bundle)) or {}
        except Exception as exc:
            # A malformed config -- unreadable file, or an execution.mode
            # that names neither launch mode (get_execution validates; ONE
            # place defines what a mode is) -- is ITS OWN error.  Swallowing
            # it here told the user to set a value they may already have set.
            raise click.ClickException(
                f"the execution block could not be resolved from config: "
                f"{exc}\n  Fix the config, or pass --mode/--domain "
                f"explicitly for this call.") from exc
    if mode is None:
        mode = _execn.get("mode")
        if not mode:
            # Unset is a refusal, never a derivation: deciding `launch` from
            # a DETECTED scheduler would gate submission on detection, which
            # running-a-job.md § 5.4 forbids.
            raise click.ClickException(
                "no --mode, and molbuilder.json sets no `execution.mode`.\n"
                "  'direct' runs it here with bash; 'submit' hands it to the "
                "scheduler.  Set execution.mode once for this machine, or pass "
                "--mode for this call (running-a-job.md § 5.4).")
        mode_source = "execution.mode (config)"
    if only_side and (kind != "bench" or mode != "submit"
                      or trial is not None):
        # Mirrors the domain refusal below: a side filter rides the GROUPED
        # submission and nothing else (generator.md § 4.3a).  A named
        # trial IS the selection, so pairing it with --only would be a
        # filter silently ignored (review 2026-08-21).
        raise click.ClickException(
            "--only picks a side of a grouped bench submission; it has no "
            "meaning for `launch run``, --mode direct, or a named trial.")
    if kind == "bench":
        # the stage's own sweep record, from its bench container (§ 6.3)
        js, base = _load_bench_set(bundle, stage)
    else:
        if trial is not None:
            raise click.ClickException(
                "a TRIAL names a benchmark point; `launch run`` takes a "
                "stage only (job-system.md § 5.3).")
        js, base = _load(bundle)
    _check_kind(kind, js)

    # ------------------------------------------------------------------ #
    #  Which queue -- asked ONCE, of everyone who already answered        #
    # ------------------------------------------------------------------ #
    # NOBODY GUESSES THE QUEUE (user, 2026-08-23).  Which queue to spend a
    # day of wall-clock in is a judgement about priority, contention and
    # what else is running -- none of it on the machine's record, all of it
    # the person's.  So this never DERIVES a queue; it asks, in order,
    # everyone the person has already told:
    #
    #   1. --domain on this call        -- said about this launch
    #   2. the bundle's own resources   -- said about THIS WORK at prep,
    #                                      which is where the Task-setup tab
    #                                      puts it (task.json `allocation`)
    #   3. execution.domain in config   -- said once about this machine
    #
    # Most specific wins.  (2) was consulted by NOTHING until 2026-08-24: a
    # person picked `htc` in the browser, `prep` baked it into every job's
    # `resources.domain`, and `launch` then printed the queue table and
    # refused for want of a --domain -- asking a question its own artifact
    # had answered.  R9 says what was admitted when the work was built is
    # re-admitted when it is sent, and it cannot be re-admitted unread.
    #
    # Reading it is not inferring it: the value is there because a person
    # put it there, so S5 holds.  It is still ADMITTED below like any other
    # -- a bundle prepped elsewhere and rsync'd here names a queue this
    # machine may not have, and that refuses with the reason rather than
    # silently retargeting.
    if domain is None and mode == "submit":
        _baked = {j.resources.domain for j in js.jobs if j.resources.domain}
        if len(_baked) == 1:
            domain = _baked.pop()
            domain_source = "the bundle (prep baked it)"
        elif len(_baked) > 1:
            # The cpu/gpu split, which --domain / --gpu-domain already
            # model.  Two baked answers are not one answer, so name them
            # rather than pick between them.
            raise click.ClickException(
                "this bundle's jobs name more than one domain ("
                + ", ".join(sorted(_baked)) + ").  Name the one to use with "
                "--domain, and --gpu-domain if the GPU side differs.")
    if domain is None and mode == "submit" and _execn.get("domain"):
        # § 5.4's other half: execution.domain is the machine's default
        # routing -- documented, returned, and never consulted until R3.
        # ONLY for the submit mode (mode resolves first, above): domain is
        # a SLURM concept the seam refuses in 'direct', so pouring the
        # config default in unconditionally made `--mode direct` impossible
        # on any machine whose config records its routing (2026-08-12 plan A3).
        # An EXPLICIT --domain with --mode direct still reaches the seam's
        # refusal -- a stated contradiction is an error, a config default
        # is not.
        domain = _execn["domain"]
        domain_source = "execution.domain (config)"
    if domain is None and mode == "submit":
        from .ask import Ask, queue_table
        from ..runtime_config import get_routing
        _rows = get_routing(project_dir=Path(base) if base else None)
        if _rows:
            click.echo(queue_table(
                _rows, Ask(time_s=_duration(time_text),
                           mem_gb=_memory(mem_text)),
                cores=None, gpu=(only_side == "gpu")))
            raise click.ClickException(
                "no --domain, so no queue was chosen.  Pick one from the "
                "list above with `--domain <name>`, name it in the "
                "description's `allocation.domain` so the bundle carries it, "
                "or set `execution.domain` in molbuilder.json to answer this "
                "once for this machine.")
    # The same provenance line prep printed, at the LAST moment before the
    # launch -- the mode above may have come from config, and this names
    # which file said so (user request 2026-08-12).
    from ..runtime_config import config_provenance, format_provenance
    prov = config_provenance(project_dir=base)
    click.echo(format_provenance(prov))
    try:
        if (kind == "bench" and js.kind == "sweep"
                and mode == "submit" and trial is None):
            # ONE grouped job per resource shelf (§ 2.3.2 user 2026-08-20;
            # split per shelf 2026-08-21, generator.md § 4.3a): each
            # shelf's trials ride one exact-fit allocation in sequence,
            # each under the per-trial bound.  A named trial still submits
            # alone below -- how a single point is re-run.
            from .submit import submit_bench_group
            # The decision entry records the SWEEP considered -- the group's
            # actual members are the still-unlaunched subset, which the
            # "launched" entry below records per job ("rides the group").
            # The key said `trials` until 2026-08-20 and read as the ride
            # list, which it was not (milestone review, N1).
            # ONE QUESTION, ONE ANSWER, ONE OUTPUT (`jobset/ask.py`).
            # The person states the TOTAL -- how long they are willing to wait
            # -- and the per-trial bound is arithmetic on top of it.  Stating
            # the bound directly still works: it is the same answer, said the
            # other way.
            from .ask import Ask, confirm
            _ask = Ask(time_s=_duration(time_text),
                       mem_gb=_memory(mem_text))
            # NOTHING IS DERIVED (user dictation, 2026-08-24).  --time is
            # the wall, said or defaulted to the target queue's ceiling
            # inside `_submit_side_group`; --trial-timeout is itself or
            # absent.  The bench_bound arithmetic that turned a "budget"
            # into a per-trial bound -- slack factor, startup margin,
            # one-minute floor, a 15-minute default when nothing was said
            # -- is DELETED, not bypassed: it manufactured the 38-minute
            # walls of jobs 62039301-05 from a default nobody chose.
            _bound_s = (int(trial_timeout_min) * 60
                        if trial_timeout_min is not None else None)
            # NOTHING IS SUBMITTED UNSEEN (S4) -- and what is seen is the
            # REAL thing: the exact sbatch command of every shelf-job,
            # from the same code that will submit it, not a summary
            # computed a second way.  The summary this replaces said
            # "10 trial(s) -> 170 min total" while five separate jobs
            # went out at 38 minutes EACH (62039301-05): a sweep splits
            # per resource shelf, and a display that ignores the split
            # describes a submission that never happens.
            _plan = submit_bench_group(
                js, base, domain=domain, gpu_domain=gpu_domain,
                dry_run=True,
                trial_timeout_s=_bound_s, mem_gb=_ask.mem_gb,
                time_s=_ask.time_s,
                only=only_side)
            _lines = ["about to submit:"]
            _planned = [r for r in _plan if r.status == "planned"]
            for _r in _planned:
                _lines.append(f"  {_r.name}")
                _lines.append(f"    {' '.join(_r.command)}")
            # ONE warning per unstated FACT, not per job.  These describe
            # the ask -- prep bakes one allocation over a sweep, so every
            # shelf-job says the same thing -- and a sweep of eight
            # shelves printed the same two sentences eight times each.
            _warn = []
            # GPU SHARING, in the one output a person approves (user
            # 2026-08-23: "explicitly note ... how many task will be
            # sharing the gpu ... and warn if that number is exceedingly
            # high").  Read off the very command about to be sent -- its
            # `-n` and its `--gres` -- so the number shown is the number
            # requested.  `runwrap` prints the same arithmetic at RUN
            # time; this is the half that arrives while changing it is
            # still free.
            from .ask import gpu_share_notes as _gsn
            for _r in _planned:
                _g = next((a for a in _r.command
                           if a.startswith("--gres=gpu")), None)
                if not _g:
                    continue
                try:
                    _ng = int(_g.rsplit(":", 1)[-1])
                    _nr = int(_r.command[_r.command.index("-n") + 1])
                except (ValueError, IndexError):
                    continue
                for _l in _gsn(_ng, _nr):
                    # Deduped on the STORED spelling: several shelves can
                    # share one rank/GPU ratio, and the fact is about the
                    # ratio, not about which shelf happens to carry it.
                    _line = "  " + _l.strip()
                    if _line not in _warn:
                        _warn.append(_line)
            if any(not any(a.startswith("--mem") for a in r.command)
                   for r in _planned):
                _warn.append(
                    "  MEMORY NOT STATED -- the scheduler's own default "
                    "decides (a per-core or per-GPU rate; on some sites "
                    "far below the node).  State it at prep (--mem) or "
                    "here (--mem).")
            if any("-t" not in r.command for r in _planned):
                _warn.append(
                    "  time not stated and this queue declares no ceiling "
                    "-- the scheduler's default stands.")
            _lines.extend(_warn)
            _lines.append(
                f"  per-trial bound: "
                + (f"{_bound_s // 60} min" if _bound_s else
                   "none -- each trial runs until the wall"))
            # Under --dry-run the results echo below prints these very
            # commands, so showing the plan here too would say everything
            # twice.  The WARNINGS are the part only this block carries,
            # so they are kept either way -- an unstated --mem is exactly
            # what a dry run is for finding.
            if dry_run:
                for _l in _warn + [_lines[-1]]:
                    click.echo(_l)
            elif not confirm("\n".join(_lines), auto_yes=auto_yes):
                click.echo("nothing submitted.")
                return
            _ledger(base, "launch", "bench-grouped",
                    trial_timeout_s=_bound_s,
                    time_s=_ask.time_s,
                    mem_gb=_ask.mem_gb,
                    only=only_side,
                    sweep=[j.name for j in js.jobs])
            if dry_run:
                results = _plan
            else:
                results = submit_bench_group(
                    js, base, domain=domain, gpu_domain=gpu_domain,
                    dry_run=False,
                    trial_timeout_s=_bound_s, mem_gb=_ask.mem_gb,
                    time_s=_ask.time_s,
                    only=only_side)
        else:
            if kind == "bench" and js.kind == "sweep":
                only = _pick_trial(js, base, trial)
            else:
                only = _resolve_stage(js, stage, "launch")
            # Same ask, the single-job door (2026-08-23): this branch never
            # built an Ask at all, so --mem typed here was silently
            # dropped before it could even reach submit_jobset -- the
            # grouped-sweep branch's `_ask` was the only one, and only a
            # trial-less grouped submit ever ran through it.
            # --yes is the recorded judgement over an attempt that was
            # launched and never CONCLUDED (project-layout.md 1.6, the
            # other file): still running and force-stopped look the same
            # on disk, so the refusal below names both and the person
            # decides -- molbuilder never decides over them.
            results = submit_jobset(js, base, mode=mode, domain=domain,
                                    dry_run=dry_run, only=only,
                                    mem_gb=_memory(mem_text),
                                    time_s=_duration(time_text),
                                    continue_unconcluded=auto_yes)
    except SubmitError as e:
        _ledger(base, "launch", "refused", kind=kind, stage=stage,
                trial=trial, mode=mode, mode_source=mode_source,
                reason=str(e))
        raise click.ClickException(str(e))
    _ledger(base, "launch", "launched", kind=kind, stage=stage,
            mode=mode, mode_source=mode_source,
            domain=domain, domain_source=domain_source, dry_run=dry_run,
            provenance=prov,
            jobs=[{"job": r.name, "status": r.status, "job_id": r.job_id,
                   "returncode": r.returncode} for r in results])
    if mode == "ask":
        # NOTHING WAS SUBMITTED.  The line the scheduler was asked about is
        # the line that WOULD be sent -- same flags, plus --test-only --
        # so what is printed here and what launch would do cannot drift.
        from .ask import prediction_table
        import dataclasses as _dcs
        preds = [_dcs.replace(r.prediction, label=r.name)
                 for r in results if r.prediction is not None]
        ran = [r.name for r in results if r.status == "already run"]
        skipped = [r.name for r in results if r.status == "not asked"]
        click.echo("")
        if preds:
            click.echo(prediction_table(preds))
        if ran:
            # Said plainly, not as a refusal: asking about a finished trial
            # creates nothing, so there is nothing to warn about -- it is
            # simply a question with no point left in it.
            click.echo(f"  already run, so not asked: " + ", ".join(ran))
        if skipped:
            from .submit import ASK_MAX_QUERIES
            click.echo(f"  NOT asked (past {ASK_MAX_QUERIES} queries): "
                       + ", ".join(skipped))
        would = [r for r in results
                 if r.status.startswith("WOULD continue")]
        for r in would:
            # the stage is launched; a re-submission would continue it.
            # Said as a WOULD: asking created nothing (the attempt is
            # opened at launch), and the answer to "when would it start"
            # is the same envelope either way.
            click.echo(f"  {r.name}: {r.status}")
        if not preds:
            # NOTHING WAS ASKED, so do not end on "launch it when the answer
            # suits you" -- there is no answer, and the line would read as
            # though there were.
            if not would:
                click.echo("\n  nothing left to ask about in this sweep.")
            if ran:
                click.echo("  read what they measured: "
                           "`molbuilder jobset summarize bench <stage>`")
            return
        asked_cmd = next((r.command for r in results if r.command), None)
        click.echo("")
        # "nothing to wait for" followed by an sbatch preview reads as a
        # contradiction (user, 2026-08-28): on a scheduler-less machine
        # nothing WOULD be sent, so nothing is previewed.
        if asked_cmd and not all(p.no_scheduler for p in preds):
            click.echo("  would send: " + " ".join(asked_cmd))
        # DO NOT SAY "--mode submit" WHEN THERE IS NOTHING TO SUBMIT TO.
        # The table has just said there is no scheduler here; following it
        # with "launch it with --mode submit when the answer suits you"
        # contradicts that in the next breath, and points at a mode this
        # machine cannot run.
        if all(p.no_scheduler for p in preds):
            click.echo("  run it here instead: the same command with "
                       "`--mode direct`.")
        else:
            click.echo("  launch it with the same command and "
                       "`--mode submit` when the answer suits you.")
        return

    verb = "WOULD run" if dry_run else "result"
    for r in results:
        tail = (f"job {r.job_id}" if r.job_id else
                (f"rc={r.returncode}" if r.returncode is not None else ""))
        # a skipped trial was not run and WOULD not be -- its verb says so;
        # a trial riding the group is launched BY the group's one command
        v = ("skip     " if r.status.startswith("skipped")
             else "rides    " if r.status == "rides the group"
             else verb)
        click.echo(f"  {v}  {r.name:<12} {' '.join(r.command)}  "
                   f"[{r.status}] {tail}".rstrip())
    if not dry_run:
        click.echo("next: molbuilder jobset status   (look before the next "
                   "stage)")


# --------------------------------------------------------------------- #
#  probe-scheduler -- MOVED here from `molbuilder bench` on 2026-08-17.  #
#                                                                        #
#  It was the last inhabitant of a command group whose four lifecycle    #
#  verbs were deleted in the 2026-08-12 fold, and it is not a benchmark  #
#  verb at all: it reads a live SLURM cluster and proposes a `scheduler` #
#  config block.  Leaving it there kept a group alive whose name no      #
#  longer described anything, so `molbuilder bench` is gone and every    #
#  verb now lives under `jobset` (user, 2026-08-17).                     #
# --------------------------------------------------------------------- #

#: Marks a key a row does not carry, so the field diff below can tell
#: *absent* from ``null`` -- the very distinction the record keeps.
_ABSENT = object()


def _domains_shown(before, probed):
    """What the domains consent question SHOWS -- judgeable, never two
    identical name lists (2026-08-28: the real change was field-level,
    the prompt printed only names, and the user was asked to judge the
    invisible).

    Name lists when the domain SET changed; otherwise the changed FIELDS
    per domain, with identical changes grouped ("all 9 domains: ...").
    ``null``/``absent`` are printed as the record means them.
    """
    b_names = [d.name for d in before]
    p_names = [d.name for d in probed]
    if b_names != p_names:
        return b_names, p_names

    def _say(v):
        return ("null" if v is None else
                "absent" if v is _ABSENT else repr(v))

    by_change = {}                        # change text -> [domain names]
    for b, p in zip(before, probed):
        br, pr = b.to_row(), p.to_row()
        moved = sorted(k for k in set(br) | set(pr)
                       if br.get(k, _ABSENT) != pr.get(k, _ABSENT))
        if moved:
            desc = ", ".join(f"{k} {_say(br.get(k, _ABSENT))} -> "
                             f"{_say(pr.get(k, _ABSENT))}" for k in moved)
            by_change.setdefault(desc, []).append(b.name)
    bits = []
    for desc, names in by_change.items():
        who = (f"all {len(names)} domains" if len(names) == len(before)
               else ", ".join(names))
        bits.append(f"{who}: {desc}")
    return "(same domains)", "; ".join(bits) or "(no visible change)"


def _probe_consent_merge(before, probed, *, yes: bool):
    """N3+ (roadmap § 0.2): a probe over an EXISTING record asks per
    difference which value survives -- consent, never a clobber.

    ``--yes`` takes every probed value (scripts).  Otherwise each
    difference is a question defaulting to No, and EOF -- a scripted
    probe without ``--yes`` -- keeps the record for that and every
    remaining difference: an unanswered question declines, the standing
    doctrine.  The record's declared facts survive a weaker probe the
    same way (a login node that cannot see GPUs probes ``None``, and No
    keeps your declared 4).  ``detected_at``/``source`` follow the new
    probe either way: the kept values were re-CONFIRMED now, and the
    stamp says when the record was last looked at.

    Returns the record to write.
    """
    import dataclasses as _dc

    diffs = []                                # (name, recorded, probed, keep)
    if before.scheduler != probed.scheduler:
        diffs.append(("scheduler", before.scheduler, probed.scheduler,
                      lambda: setattr(probed, "scheduler",
                                      before.scheduler)))
    for f in _dc.fields(probed.topology):
        b = getattr(before.topology, f.name)
        pv = getattr(probed.topology, f.name)
        if b != pv:
            diffs.append((
                f"topology.{f.name}", b, pv,
                lambda n=f.name, v=b: setattr(probed.topology, n, v)))
    if before.site.partition != probed.site.partition:
        diffs.append(("site.partition", before.site.partition,
                      probed.site.partition,
                      lambda: setattr(probed.site, "partition",
                                      before.site.partition)))
    # The reachable-domain SET is one fact -- a per-row question would ask
    # about a menu nobody composed row by row.
    if [d.to_row() for d in before.domains] != \
            [d.to_row() for d in probed.domains]:
        shown_b, shown_p = _domains_shown(before.domains, probed.domains)
        diffs.append((
            "domains", shown_b, shown_p,
            lambda: setattr(probed, "domains", list(before.domains))))

    if not diffs:
        click.echo("\nthe record already says this -- refreshing "
                   "detected_at only.")
        return probed

    click.echo(f"\nthe record disagrees with this probe in {len(diffs)} "
               f"place(s).  Per difference: take the probed value?  "
               f"(No keeps the record)")
    took, kept, dead = [], [], False
    for fname_, b, pv, keep in diffs:
        if yes:
            take = True
        elif dead:
            take = False
        else:
            try:
                take = click.confirm(
                    f"  {fname_}: recorded {b!r} -> probed {pv!r} -- "
                    f"take probed?", default=False)
            except click.exceptions.Abort:
                click.echo("\n  no answer -- keeping the record for this "
                           "and every remaining difference (silence is "
                           "no; --yes takes them all).")
                dead, take = True, False
        (took if take else kept).append(fname_)
        if not take:
            keep()
    click.echo("  " + "; ".join(filter(None, [
        f"took probed: {', '.join(took)}" if took else "",
        f"kept recorded: {', '.join(kept)}" if kept else ""])))
    return probed


@jobset_group.command("machines",
                     short_help="list the machine records prep can target")
def cmd_machines() -> None:
    """Which machines a calculation can be prepared FOR, and where each lives.

    The answer to *"did the record I copied over actually arrive, and does it
    parse?"* -- which until 2026-08-22 could only be had by running ``prep
    --target`` and reading the refusal, because this list existed only in the
    browser (`preparing-for-another-machine.md` § 5).

    Prints the same list `GET /api/task-setup/machines` serves, from the same
    function, so the terminal and the tab cannot disagree.  An unreadable
    record is PRINTED and marked, never skipped: the user wrote it, and a
    silently-dropped record looks exactly like one that was never copied.

    The path is shown because it is the destination of the copy -- a record
    written on the cluster by ``probe --write --name sol`` is carried here by
    copying it to the path this command prints.
    """
    from molbuilder.scheduler import (known_machines, choice_required,
                                        environments_dir)
    machines = known_machines()
    for m in machines:
        mark = " " if m["readable"] else "!"
        name = m["name"] + ("" if m["kind"] == "target" else "  [local]")
        click.echo(f"{mark} {name:24} {m['summary']}")
        click.echo(f"    {m['path']}")
        if m["readable"] and m["detected_at"]:
            click.echo(f"    measured {m['detected_at']}")
    named = [m for m in machines if m["kind"] == "target"]
    click.echo("")
    if not named:
        click.echo("No named targets yet.  To prepare for another machine, "
                   "run this on THAT machine:")
        click.echo("    molbuilder jobset probe --write --name <name>")
        click.echo(f"then copy the file it writes into:\n    "
                   f"{environments_dir()}/")
    elif choice_required(machines):
        # NAME THE LOCAL SPELLING HERE TOO.  The listing above shows
        # `(this machine)`, which is a LABEL and not something anyone can
        # type -- so a reader told that `prep` "requires --target <name>"
        # was left with no name for the box in front of them.  The same
        # gap the ambiguity refusal had (`record.LOCAL_TARGET`, 2026-08-24);
        # a hint that names only half the options is half a hint.
        from ..scheduler.record import LOCAL_TARGET
        click.echo("More than one machine could be meant, so `prep` requires "
                   "`--target <name>` (being asked costs one flag; being "
                   "given the wrong one costs a queue wait).")
        click.echo(f"    --target {LOCAL_TARGET}   is this machine.")


@jobset_group.command("probe",
                     short_help="record a machine's capability -> "
                                "environment.json (--name for a cluster you "
                                "prep FOR)")
@click.option("--out", default=None,
              type=click.Path(file_okay=False, resolve_path=True),
              help="directory to write environment.json into with --write "
                   "(default: the per-user machine scope).")
@click.option("--write", "do_write", is_flag=True, default=False,
              help="write the probed record (shows a diff + confirms).")
@click.option("--name", default=None, metavar="NAME",
              help="RECORD THIS AS A NAMED TARGET you can prep FOR, rather "
                   "than as the machine you are standing on.  Run `probe "
                   "--write --name sol` on Sol's login node, copy the file it "
                   "writes to the directory `jobset machines` prints, and "
                   "`prep --target sol` sizes for Sol from anywhere.  Without "
                   "it the record describes THIS box, which is the wrong "
                   "answer whenever you prep on a laptop for a cluster.")
@click.option("--yes", is_flag=True, default=False,
              help="with --write: take every probed value without asking "
                   "(scripts).  Without it, each difference against an "
                   "existing record is asked about, and silence keeps the "
                   "record.")
@click.option("--set", "sets", multiple=True, metavar="KEY=VALUE",
              help="declare a topology fact the probe cannot see from here "
                   "(M-1's declared door -- e.g. describing a cluster from a "
                   "workstation): --set gpus_per_node=4 --set gpu_type=a100. "
                   "Repeatable; wins over detection; the record's source "
                   "says 'flag'.")
@click.option("--scheduler", "scheduler_flag", default=None,
              type=click.Choice(["slurm", "workstation"]),
              help="force the scheduler kind instead of detecting it "
                   "(source 'flag').")
def cmd_probe_scheduler(out, do_write: bool, name, yes: bool,
                        sets, scheduler_flag) -> None:
    """Record what a machine IS -- cores, GPUs, scheduler, and on a cluster
    every (partition, QoS) you may actually submit to, with its wall.

    \b
    TWO USES, and the second is the one people miss:
      the machine you are ON      probe --write
      a cluster you prep FOR      probe --write --name sol

    The named form answers *"I describe calculations on my laptop and run them
    on Sol"*.  Run it ON Sol's login node, copy the file it writes to the
    directory `jobset machines` prints, and `prep --target sol` sizes for Sol
    from anywhere.  **Without it, a bench prepped on a laptop is measured
    against the laptop's cores and queues, silently** -- which is the whole
    reason named records exist.

    The unnamed form writes ``environment.json`` at the machine scope, so one
    probe serves every calculation here (`configuration.md` § 5).

    **Facts only.** Which partition you want, the account, and the policy no
    probe can invent (``gpu.exclusive``, ``gpu.mem``) stay yours, in
    ``molbuilder.json`` -- M-1.  Until 2026-08-17 this verb proposed a whole
    ``scheduler`` config block and defaulted your partition to the cheapest one
    it found; a probe choosing on your behalf is what that rule removed.

    Run it on a login node for a cluster; on a workstation it records the same
    shape with no domains (M-2), rather than refusing.
    """
    import getpass
    from datetime import datetime, timezone
    from pathlib import Path

    from ..scheduler import (Domain, FILENAME, machine_scope_path,
                             read_environment, resolve_environment,
                             write_environment)
    # `_run` is PRIVATE to the record module and the package does not
    # re-export it -- a private name comes from the module that defines it,
    # or it is not private.  (The blanket rename of 2026-08-23 pointed this
    # at the package, where it correctly did not exist.)
    from ..scheduler.record import _run
    from ..scheduler.probe import (derive_domains, parse_allowed_qos,
                                    parse_qos, parse_sinfo)

    user = getpass.getuser()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # The DECLARED half of M-1, typed by the schema itself: a fact you
    # cannot probe from here (the ordinary case is describing a cluster
    # from a workstation) arrives by flag and wins over detection, and the
    # record's source says so.  An unknown key or a mistyped value is
    # refused by name -- a silently-dropped declaration is a decision the
    # user wrote down and nobody obeyed.
    from ..scheduler import topology_field_types
    types = topology_field_types()
    overrides = {}
    for s in sets:
        key, sep, raw = s.partition("=")
        if not sep:
            raise click.ClickException(
                f"--set takes KEY=VALUE, got {s!r}")
        if key not in types:
            raise click.ClickException(
                f"--set knows no topology field {key!r} -- it knows "
                f"{', '.join(sorted(types))}")
        try:
            overrides[key] = types[key](raw)
        except ValueError:
            raise click.ClickException(
                f"--set {key}: {raw!r} is not {types[key].__name__}")

    # The NODE probe first -- scheduler, topology, default partition.  It is
    # the same call `prep` step 1 makes, so the two cannot disagree about what
    # this machine is.
    env = resolve_environment(now_iso=now, overrides=overrides or None,
                              scheduler_override=scheduler_flag)

    # HOW THIS MACHINE ENTERS ITS ENVIRONMENT TRAVELS WITH THE RECORD
    # (2026-08-24).  A wrapper is generated on one machine and executed on
    # another; the record is what carries the target across, and activation
    # is as much a fact about the target as its core count.  Probing Sol
    # records `module load mamba` / `source activate`; copying that record
    # to the workstation is then SUFFICIENT to generate a wrapper that runs
    # on Sol.  Without it, `prep --target sol` had Sol's queues and the
    # workstation's conda hook, and every job died sourcing a path that
    # exists on neither the cluster nor anywhere else it was sent.
    import dataclasses as _dc
    from ..runtime_config import get_script_generation as _gsg
    try:
        _sg = _gsg(project_dir=None)
        _sg_rec = {k: v for k, v in
                   (("preamble", _sg.get("preamble")),
                    ("activation", _sg.get("activation")))
                   if v}
    except Exception:      # pragma: no cover - a broken config is its own error
        _sg_rec = {}
    # WHICH ENVIRONMENTS EXIST HERE travels too -- the other half of the
    # pair.  `conda env list` enumerates without entering, so this is free
    # from whatever env the probe itself runs in.
    try:
        from ..diagnostics import get_capabilities as _gc
        _envs_here = sorted(_gc().conda_envs or ())
    except Exception:      # pragma: no cover - enumeration is best-effort
        _envs_here = []
    # AND WHAT THEY WERE BUILT FOR.  An env name is not portable, so the
    # list above means nothing without the instruction set it was seen on
    # (user, 2026-08-26: "we should know our compiled/installed
    # architecture").  `platform.machine()` -- the machine running the
    # probe, which is the machine those envs live on.
    _env_arch = None
    if _envs_here:
        import platform as _pl
        _env_arch = _pl.machine() or None
    if _sg_rec or _envs_here:
        env = _dc.replace(env, script_generation=_sg_rec or {},
                          conda_envs=_envs_here, env_arch=_env_arch)
    else:
        notes_sg = ("this machine states no script_generation, so the "
                    "record carries none -- a bundle prepped ELSEWHERE for "
                    "this machine will be refused until it does")

    notes = []
    # ``%m`` (memory per node, MB) added 2026-08-23 -- the ceiling
    # `Domain.max_mem_gb` has wanted since the row was designed
    # (`execution/submission.md` § 8, step 1).
    sinfo_txt = _run(["sinfo", "-h", "-o", "%P|%30l|%D|%40G|%c|%m"])
    if sinfo_txt is None:
        # NOT a refusal.  M-2: a workstation records its capability in the same
        # shape a cluster does.  This verb used to exit 2 here, which left the
        # one machine that most needs a stated ceiling with no record at all.
        if env.scheduler == "slurm":
            # Declared-slurm from a machine without sinfo (describing a
            # cluster from a workstation): the domains are not probeable
            # from here, and saying "workstation record" would contradict
            # the scheduler the user just declared.
            notes.append("no sinfo reachable from here, so no domains "
                         "were probed -- run `jobset probe` on the "
                         "cluster's login node to fill them, or the "
                         "record rides with none.")
        else:
            notes.append("no sinfo, so no scheduler domains -- this is a "
                         "workstation record (topology only).")
    else:
        parts = parse_sinfo(sinfo_txt)
        # MaxTRES added 2026-08-27 (R13): the QoS's per-job cpu cap was
        # in this very table all along, and the format list never asked
        # for it -- a field you did not request is not an absence the
        # record may report as silence.
        qos = parse_qos(_run(["sacctmgr", "-nP", "show", "qos",
                              "format=Name,MaxWall,MaxTRES,Flags"]) or "")
        allowed = parse_allowed_qos(
            _run(["sacctmgr", "-nP", "show", "assoc", f"user={user}",
                  "format=QOS"]) or "")
        # THE SECOND COMMAND, and it needs one: `sinfo` has no format code
        # for DefMemPerCPU, and that is the number SLURM grants per core when
        # a job states no --mem -- the one that turned 64 cores into a 128 G
        # ask nobody made.  Absent scontrol, the rows simply carry no
        # per-core default, which reads as "this machine does not say"
        # rather than as zero (R3).
        from ..scheduler.probe import parse_scontrol_partitions
        policy = parse_scontrol_partitions(
            _run(["scontrol", "show", "partition"]) or "")
        for _p in parts:
            _pol = policy.get(_p.name)
            if _pol is not None:
                _p.def_mem_per_cpu_mb = _pol.def_mem_per_cpu_mb
                _p.max_cpus_per_node = _pol.max_cpus_per_node
                # the record writes null-when-asked vs absent-when-not
                # (probe.py's absent-vs-null note, 2026-08-28)
                _p.policy_queried = True
        rows, notes = derive_domains(parts, qos, allowed)
        # AFTER `derive_domains`, which REASSIGNS `notes` -- appending before
        # it silently dropped this line, which is the class of bug the note
        # itself is about: a measurement that quietly did not happen.
        if not policy:
            notes.append("scontrol was not reachable, so no per-core memory "
                         "default was measured -- a job that states no --mem "
                         "will get whatever SLURM decides and molbuilder "
                         "cannot show you the number in advance.")
        env.domains = [Domain(**r) for r in rows]
        env.source["domains"] = "sinfo+sacctmgr"
        click.echo(f"Probed (user={user}): {len(parts)} partitions; "
                   f"allowed QoS: {', '.join(sorted(allowed)) or '(unknown)'}")

    t = env.topology
    click.echo(f"\nMachine: scheduler={env.scheduler}"
               f"  cores/socket={t.cores_per_socket}  sockets={t.sockets}"
               f"  gpus/node={t.gpus_per_node}  gpu={t.gpu_type or '-'}"
               f"  mem={t.mem_total_gb or '-'} GB")
    if env.domains:
        click.echo("\nReachable domains (submit with --domain <name>):")
        for d in env.domains:
            click.echo(f"  {d.name:<10} <= {str(d.max_time):<12} "
                       f"{d.partition}/{d.qos}")
    if notes:
        click.echo("\nNotes / assumptions (read before --write):")
        for n in notes:
            click.echo(f"  - {n}")

    # A named target is a record ABOUT another machine, kept beside this
    # machine's rather than replacing it (P2): a workstation holds both its own
    # capability and the cluster's, and `prep --target NAME` says which.
    if name:
        from ..scheduler import environments_dir
        from ..scheduler.record import LOCAL_TARGET
        if name == LOCAL_TARGET:
            # RESERVED: `--target this` means the box you are on, so a
            # record by that name would make the flag ambiguous -- and it
            # is the one name whose meaning nothing can override.
            raise click.ClickException(
                f"{LOCAL_TARGET!r} is reserved: `--target {LOCAL_TARGET}` "
                f"already means this machine, so a record called that "
                f"could never be prepped for.  Give it the machine's own "
                f"name (`--name sol`); this machine's own record needs no "
                f"--name at all.")
        target = Path(out) if out else environments_dir()
    else:
        target = Path(out) if out else machine_scope_path().parent
    fname = f"{name}.json" if name else FILENAME
    if not do_write:
        click.echo(f"\n(dry run -- nothing written. Re-run with --write to "
                   f"record this in {target / fname}.)")
        return

    before = read_environment(target / fname)
    if before is None:
        # Nothing to clobber: one consent creates the record.
        if not yes:
            try:
                click.confirm(f"Write this record to {target / fname}?",
                              abort=True)
            except click.exceptions.Abort:
                click.echo("\n  no answer -- nothing written "
                           "(silence is no).")
                return
    else:
        env = _probe_consent_merge(before, env, yes=yes)
    target.mkdir(parents=True, exist_ok=True)
    path = write_environment(env, target / fname)
    if name:
        click.echo(f"wrote {path}\n"
                   f"  use it with `molbuilder jobset prep run <stage> "
                   f"--target {name}`.")
    else:
        click.echo(f"wrote {path}\n"
                   f"  `prep` snapshots it into each calculation; what you "
                   f"WANT from this machine stays in molbuilder.json.")
