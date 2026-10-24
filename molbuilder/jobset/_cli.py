"""``molbuilder jobset ...`` — the calculation's one grammar
(docs/execution/job-system.md § 5.3).

`molbuilder describe` writes the portable folder (floor 2); on the machine
that runs it, ``prep`` derives floor 3 and everything below (the five steps of
project-layout.md § 2.3.1), ``launch`` sends ONE job -- a group, a stage
launched again, or a benchmark's trials, one job per resource shelf to the
queue, each in turn here -- and ``summarize`` reads a sweep's results back.  Nothing is produced on a host
and shipped.

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

from .model import KIND_SWEEP

from pathlib import Path
from typing import Optional

import click

from .ledger import record as _ledger
from ..transport.export import FORMATS as _EXPORT_FORMATS
from .model import JobSet
from .errors import PrepError
from .submit import SubmitError
from .runstatus import jobset_status, render_stage_status, render_status

#: A11: the name comes from the module that writes the file.
from .model import FILENAME as _JOBSET_FILE
# EVERY ENGINE MOLBUILDER KNOWS, the catalogue's (`runfiles.engines`) -- the
# one list `--engine` offers, never restated here.
from ..runfiles import engines as _rf_engines


def _sweeps_in(base: Path) -> list:
    """The bench containers below ``base`` holding a sweep's set, as
    relative paths -- asked for (`materialize.sweep_set_dirs`, the search
    counterpart of `bench_container`) and read before they are named: one
    that will not load is left out rather than described as something it
    might not be."""
    from .materialize import sweep_set_dirs
    sweeps = []
    for d in sweep_set_dirs(base):
        try:
            s = JobSet.of(d)
            if s is not None and s.kind == KIND_SWEEP:
                sweeps.append(str(d.relative_to(base)))
        except (OSError, ValueError):  # unreadable: not evidence of anything
            pass
    return sweeps


def _nothing_prepared(calc) -> str:
    """R-L1, said once: a described calculation with no run prepared --
    THE EXACT COMMAND, not a placeholder (user, 2026-08-20: a detected
    problem carries the invocation that repairs it): its first group's
    prep, and a benchmark prepared beside it named, since it is not the run
    (`job-system.md` § 4)."""
    from .commands import command, words_for
    base, task = Path(calc.base), calc.task
    sweeps = _sweeps_in(base)
    first = ((calc.request.group_of(task.stages[0].name),)
             if task.stages else ())
    what = (f"no {_JOBSET_FILE} in {base}" if calc.jobset is None else
            f"{_JOBSET_FILE} in {base} holds none of the description's "
            f"stages")
    bench = (f"\nA benchmark is prepared in "
             f"{', '.join(sweeps[:3])}{'...' if len(sweeps) > 3 else ''}"
             f" -- its own verbs are `launch bench` and `summarize "
             f"bench`." if sweeps else "")
    return (f"{what}: this calculation is described "
            f"but no run is prepared yet.  `prep` derives the set from the "
            f"description (job-system.md § 4):\n"
            f"    {command('prep', *words_for('task', groups=first), base=base)}"
            + bench)


def _load(bundle: str) -> tuple:
    """``(JobSet, folder)`` of a folder that holds no description -- a
    benchmark's bench container, which `status` reads against its
    calculation.  **When it holds no set the refusal reads the folder
    first**: a hand-built sweep in a container below, one of a
    calculation's own folders (its root named, the one door's sentence), or
    nothing here at all."""
    base = Path(bundle)
    try:
        js = JobSet.of(base)
    except ValueError as e:                      # bad schema / shape
        raise click.ClickException(str(e))
    if js is not None:
        return js, base
    sweeps = _sweeps_in(base)
    if sweeps:
        from ..task import FILENAME as _TASK_FILE
        where = ", ".join(sweeps[:3]) + ("..." if len(sweeps) > 3 else "")
        raise click.ClickException(
            f"no {_JOBSET_FILE} at the root of {base}, and no "
            f"{_TASK_FILE}: a hand-built sweep is prepared in {where}, "
            f"and its verbs are `launch bench` and `summarize bench` -- "
            f"which read a sweep through its description, so a "
            f"description-less one is launched and read by hand.")
    from .. import calcdirs
    from ..calculation import refusal
    root = calcdirs.root_of(base)
    if root is not None and Path(root).resolve() != base.resolve():
        raise click.ClickException(refusal(base))
    raise click.ClickException(
        f"no {_JOBSET_FILE} in {base} -- nothing to do.  The host "
        "describes it and `prep` derives it (job-system.md § 4); run "
        "`molbuilder describe` first.")


def _flags_may_stand(calc, allocation) -> None:
    """Each launch flag stated is a value of its item, asked of the setting
    door in a prep flag's home (`engines/template.md` § 5.4) as it is typed
    -- a count of ranks, threads or GPUs above 0, a memory cap above 0 --
    refused naming the flag."""
    from ..scheduler.quantities import parse_gres_flag
    from ..template import accepts, catalogue, select
    from .model import AS_RESOURCE
    from .placement import _LAUNCH_WORDS
    engine, kind = str(calc.task.engine), str(calc.task.calculation)
    for it in select(catalogue(), engine=engine, allocation=True):
        field = AS_RESOURCE.get(it.name, it.name)
        said = getattr(allocation, field, None)
        if said in (None, ""):
            continue
        clause = accepts(it, parse_gres_flag(said) if field == "gres"
                         else said, "flag", engine=engine, kind=kind)
        if clause:
            flag = _LAUNCH_WORDS[field][1].split()[0]
            raise click.ClickException(f"{flag} {said}{clause}.")


def _calc_of(folder):
    """The calculation a verb works on, opened ONCE through its door
    (`calculation.open`, A16) -- a folder that is not one, or a record that
    does not read, is the verb's refusal in the door's words."""
    from ..calculation import NotACalculation, open as _open
    try:
        return _open(folder)
    except (NotACalculation, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None


@click.group("jobset", short_help="run a job-set bundle (stage ladder / sweep)")
def jobset_group() -> None:
    """The calculation's verbs, one grammar (job-system.md § 5.3).
    Describing is not one of them: ``describe``, molbuilder's own command,
    writes the portable folder; on the machine that runs it,
    ``prep`` derives everything else and ``launch`` launches one stage's run,
    or a benchmark's trials.  Floor 3 (``job-set.json``) is DERIVED at prep,
    on the target -- nothing is produced on a host and shipped.

    ``probe`` measures a machine and ``machines`` lists the records that
    measuring produced, which is how a calculation is prepared for a
    machine other than this one (preparing-for-another-machine.md)."""
    _echo_config_root()


def _echo_config_root() -> None:
    """The TWO lines every jobset verb opens with -- where ``molbuilder.json``
    resolves from, and where the projects tree is, before anything else runs
    (user, 2026-08-23: *"this gives the user some root information of the
    starting point of config information"*; extended to the tree 2026-09-12:
    *"where we are looking for the project dir should always be displayed in
    molbuilder execution such that user will not need to be implicitly believe
    where it is"*).

    The tree line names its SOURCE, not just its path, because four things can
    answer (an explicit argument, ``$MOLBUILDER_PROJECTS``, ``paths.projects``,
    the default) and a bare path cannot say which did.  It prints
    unconditionally, including when the default wins: silence-when-default
    would still leave a person believing rather than knowing, which is the
    thing the instruction rules out.

    Not the full ``config:`` provenance block -- that already exists
    (`config_provenance`/`format_provenance`, on ``prep``/``launch``) and
    answers a bigger question, *what took effect*, for the commands that need
    it.  This answers a narrower one that every verb needs even when it never
    reads a `molbuilder.json` section: *which file is the starting point*.  A
    person hitting the activation refusal should not
    have to already know that the config directory is the
    file to edit -- the first line of output says so.

    A Click GROUP callback runs once, before the subcommand's own output.  It
    is skipped for the GROUP's own ``--help`` and for the bare group, and it
    RUNS for a subcommand's ``--help``: Click invokes the group's callback
    before it parses the subcommand, whose eager ``--help`` fires only then --
    so ``jobset prep --help`` prints these two lines above its usage
    (measured 2026-09-29).  Every real invocation carries them.
    """
    import click as _click
    from ..placement import machine_config_warnings
    from ..runtime_config import CONFIG_FILENAME, machine_config_path
    path = machine_config_path()
    state = "found" if path.is_file() else "not found -- defaults in effect"
    _click.echo(f"{CONFIG_FILENAME}: {path} ({state})")
    # A cwd file LOSES silently otherwise (configuration.md § 2.1a).  The line
    # above already names the resolved path; this says what that path is
    # standing in front of, which is the half a reader cannot infer.  To
    # stderr, so it reaches a person without entering piped output.
    for warning in machine_config_warnings():
        _click.echo(warning, err=True)
    from ..projects import projects_root_with_source
    _click.echo(projects_root_with_source().describe())


#: The calculation folder, spelled the same way on every verb.
def _resolve_bundle(ctx, param, value, *, must_exist: bool = True):
    """``--bundle`` names a calculation, and a calculation is always inside
    the projects tree (user, 2026-08-22).

      * not given            -> the working directory
      * anything else        -> read from the projects root, uniformly
      * either way           -> it must be INSIDE the projects root

    **Uniform, with no escape hatch.**  `psml_lib`
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


#: ONE declaration for every verb.
#:
#: ``must_exist`` is the ONLY axis on which the verbs differ: five of them
#: act on a calculation that is already there, and `molbuilder describe` creates one.  That
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
    if value is None:
        # transport's legitimate absence -- required-per-calculation is
        # decided in the command body, where the refusal can name why.
        return None
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


def _init_transport(*, out_dir, shape, run_name, engine, slots_opt,
                    bias_opt, structure, psml_lib, vacuum,
                    low_bias_approximation=None,
                    stage_strategy) -> None:
    """`init --calculation transport` -- task.json and the calculation's
    template, from the junction citation (engines/transport.md § 3.1): a
    finished relaxation run, or a structure pair recording its optimization.

    The structure and the cell arrive at prep from the citation, so every
    option that would supply them here is refused naming that rule rather
    than silently ignored.  The pseudopotentials come with a cited run and
    from ``--psml-lib`` for a cited pair -- refused for the first, required
    for the second.
    """
    from pathlib import Path as _P

    from ..describe import DescribeError
    from ..task import Stage, Task, derive_run, fresh_varies
    # The ladder is the transport module's fact (one door): five stages
    # in dependency order, every one (`task.fixed_by_kind`).
    from ..transport.stages import TRANSPORT_STAGES

    for given, flag, why in (
            (structure, "--structure", "its structure IS the junction "
             "citation, copied in at prep"),
            (vacuum, "--vacuum", "the cell comes with the cited junction"),
            (stage_strategy, "--stage-strategy", "the composite's five "
             "stages are fixed by design (seed, electrode_L, electrode_R, "
             "device, transmission)")):
        if given is not None and given != ():
            raise click.ClickException(
                f"{flag} does not apply to a transport calculation -- "
                f"{why} (engines/transport.md § 3.1).")
    # THE ONE SENTENCE every door refuses an engine with (`transport.stages.
    # why_no_transport`, `engines/transport.md` § 3.8.8).
    from ..transport.stages import why_no_transport
    no = why_no_transport(engine)
    if no:
        raise click.ClickException(f"engine {engine!r}: {no}.")
    # shape: the codec owns the transport==hierarchical pairing (task.py
    # __post_init__); the Task construction below surfaces it by name.

    slots = {}
    for entry in slots_opt:
        name_, sep, cite = entry.partition("=")
        if not sep or not name_ or not cite:
            raise click.ClickException(
                f"--slot {entry!r}: spell it NAME=PATH, e.g. "
                f"--slot junction=<project>/<topic>/<calc>/<stage>/run-N "
                f"-- a finished relaxation run's folder, or a structure "
                f"pair's .xyz (engines/transport.md § 3.1).")
        slots[name_] = cite
    if set(slots) != {"junction"}:
        raise click.ClickException(
            "a transport calculation takes exactly one slot, `junction` "
            "-- the relaxed junction it composes from: a finished "
            "relaxation run of molbuilder's own, or a structure pair that "
            "records its optimization (engines/transport.md § 3.1).")

    # THE CITATION DOOR prep composes through (`compose.resolve_citation`):
    # the tree fence, then § 3.1's condition by what the path is -- a
    # citation that cannot compose is refused at init, naming what is
    # missing (strict composition, Q2).
    from ..projects import projects_root
    from ..transport.compose import CitedPair, ComposeError, resolve_citation
    root = _P(projects_root()).expanduser().resolve()
    raw = _P(slots["junction"]).expanduser()
    if raw.is_absolute():
        try:
            raw = raw.resolve().relative_to(root)
        except ValueError:
            raise click.ClickException(
                f"--slot junction={slots['junction']}: {raw} is not inside "
                f"the projects tree ({root}); give its path from the tree's "
                f"root.")
    try:
        resolved, cited = resolve_citation(str(raw), root)
    except ComposeError as exc:
        raise click.ClickException(str(exc))
    citation = str(resolved.relative_to(root))
    pair = isinstance(cited, CitedPair)
    dest = (out_dir if out_dir.is_absolute()
            else _P(_resolve_bundle(None, None, str(out_dir),
                                    must_exist=False)))
    from ..transport.compose import refuse_inside_citation
    try:
        refuse_inside_citation(dest, cited.path)
    except ComposeError as exc:
        raise click.ClickException(str(exc))

    # THE SINGLE BIAS IS STATED, `[0.0]` -- the one spelling both roads
    # write (`engines/transport.md` § 2a.10; the tab writes the same), never
    # an empty list a reader has to fill in (2026-10-18).
    bias = _bias_of(bias_opt) if bias_opt else (0.0,)

    stages = tuple(Stage(name=n, overrides={})
                   for n in TRANSPORT_STAGES)
    try:
        task = Task(
            engine=engine, shape=shape,
            run=derive_run(run_name, citation,
                           stage_names=TRANSPORT_STAGES),
            structure=None, calculation="transport",
            slots={"junction": citation}, bias=bias,
            low_bias_approximation=low_bias_approximation,
            varies=fresh_varies(stages), stages=stages)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # THE TEMPLATE'S TEXT FIRST -- before the folder exists -- so a
    # citation it refuses (one that carried a net charge, ES7,
    # `science/chemistry-correctness.md` § 2a; a pair with no
    # pseudopotential directory, or a run with one, § 3.1) leaves nothing
    # behind.
    from ..transport.citation_defaults import (check_pseudopotentials,
                                               transport_template_text)
    try:
        check_pseudopotentials(cited, psml_lib)
        _tmpl_text = transport_template_text(
            resolved, label=task.label,
            **({"psml_lib": str(psml_lib)} if psml_lib else {}))
    except ValueError as exc:
        raise click.ClickException(str(exc))
    # THE DESCRIPTION'S OWN CHECK (gate ③), which every describe runs
    # (`workflow.md` § 9) -- refusing before the folder exists, and saying its warnings
    # (a bias point outside its recommended range) the way `prep` says its
    # notes.
    from ..describe import Description, write_description
    from ..issues import ValidationError
    from ..validation.task import preflight, refuse_on_error
    desc = Description(task=task, template=_tmpl_text)
    try:
        for issue in refuse_on_error(preflight(task,
                                               template_text=_tmpl_text,
                                               request=desc.request)):
            click.echo(f"note: {issue.message}", err=True)
    except ValidationError as exc:
        raise click.ClickException(str(exc))
    # THE DESCRIPTION AND ITS TEMPLATE, every file or none, through the one
    # writer every describe takes (`describe.write_description`) -- the
    # Transport tab's describe door writes through it too.
    #
    # The template is transport's shared baseline, like every other kind's
    # (`engines/transport.md` § 2a.3): its values are DEFAULTED FROM THE
    # CITED RUN, not sealed to it (§ 2a.7), and its TEXT comes through the
    # same door the describe door uses (`citation_defaults.
    # transport_template_text`): a `citation` row the cited directory does
    # not answer stays VALUELESS (§ 3.8.3).
    try:
        written = write_description(desc, dest)
    except (DescribeError, OSError) as exc:
        raise click.ClickException(str(exc))

    click.echo(f"Described transport '{task.run.name}' in {dest} -- "
               f"composes {citation}.")
    for path in written:
        click.echo(f"  {path.name}")
    click.echo("")
    click.echo("The template carries the shared electronic description, "
               "filled in from the "
               + ("record of the optimization your pair carries"
                  if pair else "run you cited")
               + " -- change anything in it and every stage follows, "
               "because there is one of it.  The structure arrives at prep "
               "from the citation, and the pseudopotentials "
               + ("from the directory you named" if pair
                  else "from the run you cited")
               + ".  On the machine that will run it:")
    # THE FIRST COMMAND A PERSON COPIES: the groups `prep task` offers --
    # every one, before anything has run.
    from .commands import prep_group_lines
    from .ready import task_state
    click.echo("  " + prep_group_lines(task_state(_calc_of(dest)).prep_offer,
                                       base=dest)[0])


def _bias_of(bias_opt: str) -> tuple:
    """``--bias``'s volts, as a description holds them."""
    try:
        return tuple(float(v) for v in bias_opt.split(","))
    except ValueError:
        raise click.ClickException(
            f"--bias {bias_opt!r}: a comma-separated list of volts, "
            f"e.g. --bias 0.0,0.2,0.4.")


def _init_reusing(*, out_dir, reuses, engine, shape, bias_opt,
                  low_bias_approximation, window_opt, finer,
                  refused) -> None:
    """`describe --calculation transport --reuses <finished>` -- the new
    calculation written from the finished one's records as they read now
    (`engines/transport.md` § 2a.16, *Describing it*): its description,
    with ``reuses`` naming it, its template and its job request, its
    composed junction and its pseudopotentials.  ``--bias`` and
    ``--low-bias-approximation`` set what they set on any describe; a flag
    naming what the finished calculation fixes -- its junction, its label,
    its pseudopotentials -- is refused, naming why.  ``--window a,b``
    (``--finer n``) fits the transmission's window to the finished grid
    (`transport.window.fitted`), written as the transmission's own values
    and said where it is fitted."""
    import dataclasses

    from ..describe import DescribeError, Description
    from ..issues import ValidationError
    from ..projects import projects_root
    from ..transport.reuse import ReuseError, reused, write_reusing
    from ..transport.stages import why_no_transport
    from ..validation.task import preflight, refuse_on_error
    if refused:
        flag, why = next(iter(refused.items()))
        raise click.ClickException(
            f"{flag} does not apply beside --reuses -- {why} "
            f"(engines/transport.md § 2a.16).")
    root = Path(projects_root()).expanduser().resolve()
    try:
        finished, citation = reused(reuses, tree_root=root)
    except ReuseError as exc:
        raise click.ClickException(str(exc))
    # WHAT A FLAG SETS, as on any describe: the voltages with their
    # treatment, or the treatment alone.
    said = {}
    if bias_opt:
        said = {"bias": _bias_of(bias_opt),
                "low_bias_approximation": low_bias_approximation}
    elif low_bias_approximation is not None:
        said = {"low_bias_approximation": low_bias_approximation}
    engine = engine or finished.task.engine
    no = why_no_transport(engine)
    if no:
        raise click.ClickException(f"engine {engine!r}: {no}.")
    try:
        task = dataclasses.replace(
            finished.task, reuses=citation, engine=engine,
            shape=shape or finished.task.shape, **said)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    if finer != 1 and not window_opt:
        raise click.ClickException(
            "--finer cuts the window fitted by --window finer -- name the "
            "window too (engines/transport.md § 2a.16).")
    if window_opt:
        # THE WINDOW FITTED TO THE FINISHED GRID, the transmission's own
        # values -- where the page's *use these* writes it
        # (`engines/transport.md` § 2a.16).
        from ..task import Stage
        from ..transport.window import fitted
        from .reuse import SHARING_RUNG, finished_window
        try:
            lo, hi = (float(v) for v in window_opt.split(","))
            fit = fitted(finished_window(finished), lo, hi, finer)
        except ValueError as exc:
            raise click.ClickException(f"--window {window_opt}: {exc}")
        window = {"transmission_emin_ev": fit.emin,
                  "transmission_emax_ev": fit.emax,
                  "transmission_n_points": fit.points}
        task = dataclasses.replace(
            task, varies=tuple(dict.fromkeys((*task.varies, *window))),
            stages=tuple(Stage(s.name, {**s.overrides, **window})
                         if s.name == SHARING_RUNG else s
                         for s in task.stages))
        click.echo(f"transmission window: {fit.words()}")
    desc = Description(task=task, template=finished.template_text,
                       request=finished.request)
    try:
        for issue in refuse_on_error(preflight(task,
                                               template_text=desc.template,
                                               request=desc.request)):
            click.echo(f"note: {issue.message}", err=True)
    except ValidationError as exc:
        raise click.ClickException(str(exc))
    dest = (out_dir if out_dir.is_absolute()
            else Path(_resolve_bundle(None, None, str(out_dir),
                                      must_exist=False)))
    try:
        written, notes = write_reusing(desc, dest, finished, tree_root=root)
    except (ReuseError, DescribeError, OSError) as exc:
        raise click.ClickException(str(exc))
    for note in notes:
        click.echo(f"note: {note}", err=True)
    click.echo(f"Described transport '{task.run.name}' in {dest} -- reuses "
               f"{citation}.")
    for path in written:
        click.echo(f"  {path.name}")
    # WHAT IT WILL REUSE, AND WHAT RUNS -- the one answer prep composes
    # (`reuse.decide`), said once it is described (`engines/transport.md`
    # § 2a.16, *Extending it*): every group, nothing written.
    from .reuse import decide
    try:
        calc_now = _calc_of(dest)
        manifest = decide(calc_now, calc_now.request.numbers)
    except PrepError as exc:
        click.echo(f"note: what it will reuse is not known yet -- {exc}",
                   err=True)
    else:
        if manifest is not None:
            click.echo("")
            click.echo("\n".join(manifest.lines(dest)))
    from .commands import prep_group_lines
    from .ready import task_state
    click.echo("")
    click.echo("Change what the new question asks -- on Task setup, or in "
               "its task.json and template -- then, on the machine that "
               "will run it:")
    click.echo("  " + prep_group_lines(task_state(_calc_of(dest)).prep_offer,
                                       base=dest)[0])


@click.command("describe",
               short_help="describe a calculation: its description and "
                          "its job request (floor 2 only)")
@_bundle_option(must_exist=False)
@click.option("--structure", "structure", default=None, metavar="PATH",
              callback=_resolve_structure,
              help="the structure to describe, as a path from the PROJECTS "
                   "ROOT -- e.g. `--structure <project>/structure/water.xyz`. "
                   "It lives in the tree like everything else a calculation "
                   "cites (job-contracts.md 2.5b).  Required for every "
                   "calculation EXCEPT transport, whose structure IS its "
                   "junction citation (--slot).")
@click.option("--shape", default=None,
              type=click.Choice(("flat", "hierarchical")),
              help="how the stages sit on disk. REQUIRED and never inferred -- "
                   "inferring it would hand you a directory tree you did not "
                   "ask for (engines/stages.md 6.7) -- but beside --reuses, "
                   "which describes the finished calculation's.")
@click.option("--stage-strategy", default=None,
              help="which shipped ladder to describe (e.g. publishable). Omit "
                   "for a calculation with a SINGLE parameter set, which is "
                   "described as ONE stage named 'coarse' -- named and "
                   "tokened like any other (6.5).")
@click.option("--name", default=None, metavar="NAME",
              help="what you call this calculation; the label and the run id "
                   "derive from it. Default: the destination folder's name.")
@click.option("--engine", default=None,
              type=click.Choice(_rf_engines()),
              help="whose parameters these are. REQUIRED -- never assumed "
                   "-- but beside --reuses, which describes the finished "
                   "calculation's.")
@click.option("--psml-lib", default=None, metavar="DIR",
              # NO click-level exists check: the anchor rule lives in
              # pseudos.resolve_psml_lib (job-contracts § 2.5a) and
              # describe.py validates through it with the teaching refusal.
              help="where to read pseudopotentials from -- a path INSIDE "
                   "the projects tree, measured from the tree root (the "
                   "convention is `pseudopotential`).  The files travel "
                   "with the calculation; the path does not.  A transport "
                   "calculation takes it for a cited structure pair, whose "
                   "pseudopotentials are not beside it.")
@click.option("--vacuum", type=float, default=None, metavar="ANGSTROM",
              help="isolation vacuum (A) per side on isolated axes. Needed "
                   "for a flat or linear molecule from a bare XYZ, which "
                   "otherwise has a degenerate cell.")
@click.option("--calculation", "calculation", required=True, metavar="TYPE",
              help="which KIND of calculation this describes -- "
                   "optimization, vibration or transport; the key into the "
                   "engine's warm-file vocabulary (job-contracts 4.2a).  "
                   "REQUIRED -- never assumed.")
@click.option("--slot", "slots_opt", multiple=True, metavar="NAME=CITATION",
              help="a composite input (transport only): "
                   "--slot junction=<path> -- a finished relaxation run's "
                   "folder, or a structure pair's .xyz that records its "
                   "optimization (engines/transport.md § 3.1).  Named "
                   "explicitly, never picked.")
@click.option("--bias", "bias_opt", default=None, metavar="V0,V1,...",
              help="transport only: the bias points in volts, starting at "
                   "0.0 (engines/transport.md 2a.10).  Several state "
                   "--low-bias-approximation or --no-low-bias-approximation.")
@click.option("--low-bias-approximation/--no-low-bias-approximation",
              "low_bias_approximation", default=None,
              help="transport only, with several --bias points: with the "
                   "approximation, the device and the transmission run once, "
                   "at 0 V, and each voltage's current is computed from T(E, 0) "
                   "(linear response); without it, every voltage gets its own "
                   "device SCF and transmission (engines/transport.md 2a.10).  "
                   "Required with several points, never inferred.")
@click.option("--window", "window_opt", default=None, metavar="LO,HI",
              help="with --reuses: the transmission window wanted, in eV "
                   "-- fitted to the finished calculation's grid, its "
                   "spacing and alignment, extended outward by whole "
                   "spacings until the range is covered, so every finished "
                   "energy stays a point (engines/transport.md 2a.16).")
@click.option("--finer", default=1, type=int, metavar="N",
              help="with --window: the grid N times finer, an odd number, "
                   "so every finished energy stays a point.")
@click.option("--reuses", default=None, metavar="FOLDER",
              help="transport only: a finished transport calculation's "
                   "folder, from the projects root.  The new calculation "
                   "starts as its description and its composed junction, "
                   "whole; its prep copies in every group the finished one "
                   "already computed (engines/transport.md 2a.16).")
def describe_cmd(structure, bundle: str, shape: str,
                 stage_strategy, name, engine: str, psml_lib, vacuum,
                 calculation: str, slots_opt, bias_opt,
                 low_bias_approximation, window_opt, finer, reuses) -> None:
    """Write the portable description and its job request: the template,
    ``task.json``, ``job-request.json`` (each stage's group, nothing else
    asked yet), and the data files.  Describing is not a job verb, so it is
    not under `molbuilder jobset` (execution/job-request.md 9).

    **Floor 2 only -- it renders no deck.**  A deck carries values that depend
    on how it will be launched, so rendering belongs to ``prep`` on the machine
    that will run it (project-layout.md 2.3.1).  What this writes names no
    machine and therefore means the same thing wherever you copy it.

    Names and values are validated HERE, on your laptop, not on the cluster: a
    stage name outside [A-Za-z0-9_]+, a duplicate stage, an override key the
    schema does not know, or a value outside its bounds is refused with the
    field named -- and refused before anything is written.
    """
    from pathlib import Path as _P

    from ..workingcopy_structure import StructureCodec
    from ..config.siesta import SiestaConfig
    from ..describe import DescribeError, build_description, write_description
    from ..identity import normalise_id
    from ..chemistry import species_order
    from ..pyscf.stages import default_pyscf_stages
    from ..siesta.stages import default_siesta_stages
    from .ladder import fresh_stages

    out_dir = _P(bundle)
    run_name = name or out_dir.name
    # ONE CALCULATION PER FOLDER (user, 2026-10-01: refuse), the one check
    # every door that describes asks (`describe.refuse_described`) --
    # before anything is read or written.
    from ..describe import refuse_described
    try:
        refuse_described(out_dir)
    except DescribeError as exc:
        raise click.ClickException(str(exc))

    if reuses is not None:
        if calculation != "transport":
            raise click.ClickException(
                "--reuses belongs to --calculation transport alone "
                "(engines/transport.md § 2a.16).")
        _init_reusing(
            out_dir=out_dir, reuses=reuses, engine=engine, shape=shape,
            bias_opt=bias_opt, low_bias_approximation=low_bias_approximation,
            window_opt=window_opt, finer=finer,
            refused={flag: why for flag, given, why in (
                ("--slot", slots_opt, "the junction is the finished "
                 "calculation's, carried with its record"),
                ("--name", name, "the label is the finished calculation's: "
                 "SystemLabel names every file a deck reads and writes, and "
                 "decks under two labels are never the same calculation"),
                ("--psml-lib", psml_lib, "the pseudopotentials are the "
                 "finished calculation's, copied with it"),
                ("--structure", structure, "its structure IS the junction "
                 "citation"),
                ("--vacuum", vacuum, "the cell comes with the junction"),
                ("--stage-strategy", stage_strategy, "the composite's five "
                 "stages are fixed by design")) if given not in (None, ())})
        return
    if window_opt is not None or finer != 1:
        raise click.ClickException(
            "--window and --finer fit a transmission window to a finished "
            "calculation's grid -- they go with --reuses "
            "(engines/transport.md § 2a.16).")
    for flag, given in (("--engine", engine), ("--shape", shape)):
        if given is None:
            raise click.ClickException(
                f"{flag} is required -- never assumed or inferred.")
    if calculation == "transport":
        _init_transport(out_dir=out_dir, shape=shape, run_name=run_name,
                        engine=engine, slots_opt=slots_opt,
                        bias_opt=bias_opt,
                        low_bias_approximation=low_bias_approximation,
                        structure=structure,
                        psml_lib=psml_lib, vacuum=vacuum,
                        stage_strategy=stage_strategy)
        return
    if slots_opt or bias_opt or low_bias_approximation is not None:
        raise click.ClickException(
            "--slot / --bias / --low-bias-approximation belong to --calculation "
            "transport alone "
            "(engines/transport.md § 3.1).")
    if structure is None:
        raise click.ClickException(
            "--structure is required -- it is what this calculation "
            "describes (transport is the one kind without it: its "
            "structure IS the junction citation).")

    try:
        # THE PAIR, through the one door: the geometry with its regions,
        # frozen atoms and cell -- frame 0 of a frame set, and said
        # (`model/structure.md` § 2.3, *Which frame*).
        said: list = []
        # A CALCULATION'S STRUCTURE IS IN THE USER ORDER (`model/overview.md`
        # § 2.2a, I1): a file in an engine's order is read restored, said,
        # and one of a run whose order nothing states is refused (I9).
        whole = StructureCodec().load(structure, frames=True, said_out=said,
                                      in_user_order=True, a_copy=True)
        for notice in said:            # what the read says (structure.md § 2.3)
            click.echo(notice.message)
        if whole.n_frames > 1:
            from ..structure import frame_words
            click.echo(f"{Path(structure).name} holds {whole.n_frames} "
                       f"frames; {frame_words(0, whole.n_frames)} was taken")
        struct = StructureCodec.frame_choice(whole)
        if vacuum is not None:
            # The same channel the Modify -> Cell tab uses: the vacuum lives on
            # the STRUCTURE, not on the engine config, so every surface that
            # reads this structure sees the same isolation.
            # THROUGH THE DOOR: `dataclasses.replace` does not dispatch to
            # `Structure.replace`, and would share the loaded structure's
            # `cell` and `info` dict by reference.
            struct = struct.replace(vacuum=(vacuum, vacuum, vacuum))

        # § 6.5 (2026-08-16): a job always has at least one stage.  Without
        # ``--stage-strategy`` the ladder is ONE stage carrying no overrides
        # -- the calculation that is just the template.  One shape, so
        # the artifact names, the tokens and the directories are the same
        # whether a job has one rung or three.
        # THE LADDER IS THE ENGINE'S, AND THE SHAPE OF IT IS NOT.
        # Both engines describe a ladder the same way -- a tuple of ``Stage``
        # with per-rung ``overrides`` -- and differ only in which values those
        # rungs carry, which is the engine's own preset table.
        _ladder = {"siesta": default_siesta_stages,
                   "pyscf": default_pyscf_stages}[engine]
        if calculation == "vibration":
            # The vibration kind's own ladder, read from the person's one
            # statement (`engines/vibration.md` § 2.2, § 5.2a): a fresh
            # description carries the catalogue's own value of the box --
            # unticked -- so SIESTA gets `relax` then `freq`, PySCF `freq`
            # alone.  A --stage-strategy names the optimization ladder's
            # tiers, which grade nothing here: the convergence settings are
            # the template's, recommended tight (`template.md` § 6.3a).
            from ..pyscf.stages import VIBRATION_ENGINES
            from ..template import catalogue as _catalogue, one as _item
            if engine not in VIBRATION_ENGINES:
                raise click.ClickException(
                    f"calculation 'vibration' runs on pyscf (analytic "
                    f"Hessian, intensities) or siesta (force constants, "
                    f"frequencies and mode shapes only); engine "
                    f"{engine!r} has no vibration deck.")
            if stage_strategy:
                raise click.ClickException(
                    "--stage-strategy names the optimization ladder's "
                    "tiers; a vibration calculation's ladder is read from "
                    "its own template -- `freq`, with a `relax` stage "
                    "before it on SIESTA unless the structure is stated "
                    "relaxed (engines/vibration.md 2.2) -- and its "
                    "convergence settings are the template's own.")
            _box = _item(_catalogue(), "already_relaxed", engine=engine)
            stages = fresh_stages(
                calculation, engine,
                already_relaxed=bool(_box.default if _box else False))
        else:
            # THE FRESH LADDER'S ONE OWNER (`ladder.fresh_stages`), or the
            # tiers a --stage-strategy names.
            stages = (tuple(_ladder(stage_strategy)) if stage_strategy
                      else fresh_stages(calculation, engine,
                                        already_relaxed=False))
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
        # The kind's own recommendations become the template's values
        # (`template.md` § 6.3a): a vibration's relaxation settings start
        # tight, and the person changes them there like any other value.
        from ..template import apply_recommended as _apply_recommended
        cfg = _apply_recommended(cfg, calculation, engine=engine)

        # WHAT THE PERSON STATED here is theirs (`template.md` § 6.6
        # obligation 2): the calculation's name, and the pseudopotential
        # folder when given; the kind's recommendations are nobody's choice.
        _said = {("job_name" if engine == "pyscf" else "system_label"):
                 "person"}
        if engine != "pyscf" and psml_lib:
            _said["psml_lib"] = "person"
        desc = build_description(
            struct, cfg, stages,
            engine=engine, shape=shape, name=run_name,
            source=str(structure), calculation=calculation,
            pseudo_species=species_order(struct.elements),
            value_sources=_said,
        )
        # The struct AS DESCRIBED travels: --vacuum replaced it in memory,
        # and a raw copy of the source would drop that choice.
        written = write_description(desc, out_dir, psml_lib=psml_lib,
                                    struct=struct)
    except DescribeError as e:
        raise click.ClickException(str(e))
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e))

    # Always a ladder (§ 6.5), so one stage and several share one phrasing.
    ladder = (f"{len(desc.task.stages)} stage(s): "
              f"{', '.join(s.name for s in desc.task.stages)}")
    click.echo(f"Described {desc.label!r} in {out_dir} -- {ladder}, "
               f"shape {shape}.", err=True)
    click.echo("  " + "\n  ".join(p.name for p in written), err=True)
    # THE FIRST COMMAND A PERSON COPIES, real: the groups `prep task`
    # offers.
    from .commands import prep_group_lines
    from .ready import task_state
    click.echo(
        "\nIt names no machine. On the machine that will run it:\n  "
        + prep_group_lines(task_state(_calc_of(out_dir)).prep_offer,
                           base=out_dir)[0], err=True)


@jobset_group.command("status", short_help="every stage's state, and each "
                                           "group's next step")
@click.argument("stage", required=False, default=None)
@_bundle_option()
def status_cmd(stage, bundle: str) -> None:
    """Show every stage of the description, with its group -- its run state
    (finished / running / failed / queued / pending), or the ready door's
    (described; prepared, ready to launch or waiting for what it names),
    which warm-restart
    files are present -- and each group's next step, the line that moves
    it, the first group not finished marked.  The stages are listed from the moment `molbuilder describe` writes them, the
    ones not prepared yet among them (job-system.md § 5.3).  Read-only --
    molbuilder informs; you decide whether to continue or switch.  Reuses
    the same directory decoder as the Results tab.

    With a STAGE -- its name, or '#N' its number (quoted: bash reads a bare
    # as a comment), like every other verb -- it
    answers the other question instead: *what this one is and what happened
    to it* -- its deck, what it carries, its resources, its attempts, its
    launch record and what it continued from.  That form is only answerable
    because a try is a directory and a launch is a record
    (project-layout.md § 1.5, § 1.6).
    """
    from ..calcdirs import described as _described
    base = Path(bundle)
    if _described(base):
        # THE DESCRIPTION'S LADDER (job-system.md § 5.3, 2026-10-01): every
        # stage it names, before the first prep and as prep reaches each --
        # the calculation through its one door (A16).
        calc = _calc_of(base)
        js, sweep = calc.jobset, None
        name = _described_stage(stage, calc)
    else:
        js, base = _load(bundle)
        # A BENCH FOLDER is read against its calculation: its sweep names
        # its trials from there, and what it prints names that.  A plan with
        # no description beside it is no calculation of molbuilder's (A16).
        from .materialize import bench_owner
        owner = bench_owner(base) if js.kind == KIND_SWEEP else None
        if owner is None:
            from ..calculation import refusal
            raise click.ClickException(refusal(base))
        base = owner[0]
        calc, sweep = _calc_of(base), js
        name = (_resolve_stage_name(js, stage) if stage is not None
                else None)
    try:
        status = jobset_status(calc, sweep=sweep)
    except (OSError, ValueError, KeyError) as exc:
        # A refusal, in the verb's voice -- never a traceback.
        raise click.ClickException(str(exc))
    if name is None:
        click.echo(render_status(status))
        return
    click.echo(render_stage_status(status, name))


# --------------------------------------------------------------------- #
#  prep / launch -- the execution loop (job-system.md § 5.3)             #
#                                                                       #
#  One grammar: ``jobset <verb> task [--group N ...]`` and               #
#  ``jobset <verb> bench <stage> [<trial>]`` (job-system.md § 5.3).  The  #
#  KIND is a positional: the task -- the calculation, its groups named    #
#  with --group (a stage launched again alone, --stage), the answer to    #
#  the question its verbs ask -- or a benchmark of one stage             #
#  (project-layout.md § 2.3.1a).                                         #
# --------------------------------------------------------------------- #

_KINDS = ("task", "bench")


def _stage_words(kind: str, words, stages, verb: str, *, groups=()):
    """``(stage, trial, stages)`` from what a verb was given: a benchmark's
    stage and trial by position; a task's groups by ``--group`` and its one
    stage by ``--stage``, no positional -- each refused by name where it
    does not belong (`job-system.md` § 5.3)."""
    if kind == "bench":
        if stages or groups:
            raise click.ClickException(
                f"--{'stage' if stages else 'group'} names what a task "
                f"{verb}s; `{verb} bench` takes its stage by position: "
                f"`{verb} bench <stage>`.")
        if len(words) > (2 if verb == "launch" else 1):
            raise click.ClickException(
                f"`{verb} bench` takes a stage"
                + (" and, optionally, one trial." if verb == "launch"
                   else "."))
        return (words[0] if words else None,
                words[1] if len(words) > 1 else None, [])
    if words:
        raise click.ClickException(
            f"`{verb} task` names its groups with --group N"
            + (f", or one stage with --stage {words[0]}"
               if verb == "launch" else "")
            + " (job-system.md § 5.3).")
    return (None, None, list(stages))


def _check_kind(kind: str, js=None) -> None:
    """The KIND positional against the bundle's actual kind.

    ``prep bench`` enumerates the grid on this machine and ``launch bench
    <stage>`` launches its trials -- or the one named -- through the same
    resolver as everything else.  What is checkable is AGREEMENT: a kind that
    contradicts the bundle's own is a typo about to act on the wrong thing.
    """
    if js is None:
        return
    actual = "bench" if js.kind == KIND_SWEEP else "task"
    if kind != actual:
        raise click.ClickException(
            f"this bundle's job set is a {js.kind}, which the grammar calls "
            f"{actual!r} -- and the command says {kind!r}.  The kind states "
            f"what the calculation IS, it does not switch modes "
            f"(job-system.md § 5.3).")


def _described_stage(stage, calc):
    """The stage a verb was given, as the description spells it, through
    the ONE resolver -- its name in any case, or ``#N``
    (`identity.resolve_stage_ref`, `job-system.md` § 5.3) -- and what the
    verb then finds, records and PRINTS (a pasted ``#2`` would be a comment
    in bash).  Asked by the verbs that name a stage the description holds
    whether or not it is prepared: the bench verbs and `status`.  ``None`` stays ``None``, and a folder with no description has
    no ladder to resolve against: its name stands."""
    from ..identity import StageRef, resolve_stage_ref
    if stage is None:
        return stage
    try:
        from .naming import ladder_homes
        return resolve_stage_ref(
            [StageRef(h.seq, h.name)
             for h in ladder_homes(calc.base, calc.task)], stage).name
    except ValueError as e:
        raise click.ClickException(str(e))


def _refuse_unprepared(calc, stage) -> None:
    """A stage the description holds and no prep has prepared is refused by
    NAME, with its prep and its launch -- what `status` says of it.
    Prepared is the one door's answer (`Calculation.job`)."""
    from .commands import block, run_first
    if stage is None:
        return
    if calc.job(stage) is None:
        raise click.ClickException(
            f"stage {stage!r} is not prepared yet -- prep it, then launch "
            f"it:\n" + block(run_first(stage, calc=calc)))


def _stage_bench_dir(calc, stage, verb: str = "launch"):
    """The stage's bench container (job-contracts.md § 6.3), resolved
    through the description — where its trials, its job-set and its verdict
    all live.  Returns ``(container_path, token)``.  ``stage`` is the
    description's own spelling -- the verb resolved it first
    (:func:`_described_stage`); a bare invocation is refused with the ladder
    listed — § 6.5 gives every description a ladder, so there is always a
    stage to name and never a bare form to fall back to."""
    from ..paths import bench_container
    from .naming import stage_home
    from ..paths import Shape
    base, task = calc.base, calc.task
    sh = Shape.named(task.shape)
    # A CALCULATION THAT HAS NO BENCHMARK says so first -- `prep bench`'s own
    # answer (`bench_refusal`) -- or the command offered next is refused in
    # turn.
    from .prep_inputs import bench_refusal
    why = bench_refusal(str(task.engine), str(task.calculation))
    if why:
        raise click.ClickException(why)
    if stage is None:
        # THE STAGES WITH A BENCHMARK TO ACT ON -- a prepared one -- or, with
        # none, the prep that makes one.
        from .commands import command, name_a_stage
        from .naming import described_refs
        refs = described_refs(base, task)
        benched = [r for r in refs if JobSet.of(Path(base) / bench_container(
            sh, stage_home(base, task, r.name).token)) is not None]
        if not benched:
            raise click.ClickException(
                "no stage has a prepared benchmark yet -- prep one first:\n"
                "    " + command("prep", "bench", refs[0].name, base=base))
        raise click.ClickException(
            "which stage's benchmark? "
            + name_a_stage(verb, "bench", benched, base=base))
    token = stage_home(base, task, stage).token
    return Path(base) / bench_container(sh, token), token


def _pick_trial(js, trial):
    """Which trial this invocation launches.  NAMED → that one (how a single
    point is re-run); refused by name against the sweep's own list.  Bare →
    ``None``: under --mode direct the whole set runs, in order, and sent to
    a scheduler it goes one job per resource shelf (`submit.plan_launch`,
    § 2.3.2).  The trial named rides the launch's own lines in the ledger.
    """
    if trial is not None and not any(j.name == trial for j in js.jobs):
        raise click.ClickException(
            f"no trial named {trial!r}. This sweep's trials: "
            f"{', '.join(j.name for j in js.jobs)}.")
    return trial


def _load_bench_set(calc, stage, verb: str = "launch"):
    """``(the stage's OWN sweep record, its container)`` -- the plan's door
    asked with the container (A16)."""
    base = calc.base
    container, _ = _stage_bench_dir(calc, stage, verb)
    try:
        js = JobSet.of(container)
    except ValueError as e:
        raise click.ClickException(str(e))
    if js is None:
        from .commands import command
        raise click.ClickException(
            f"no plan in {container.relative_to(base)}/ -- this stage has no "
            f"prepared benchmark.  Prep it first:\n    "
            + command("prep", "bench", stage, base=base))
    return js, container


#: WHAT YOU MAY TYPE FOR THE TWO ASKS -- said ONCE, prep's; launch's
#: `--time` takes the same spelling (`TIME_METAVAR`).
#:
#: These describe the HUMAN edge only.  What the file stores and what
#: reaches the scheduler is the record's one spelling (`task.py`:
#: ``D-HH:MM:SS``), always, and neither is any of the person's business here
#: (`engines/stages.md` § 6.8a).
TIME_METAVAR = "DURATION"
TIME_HELP = ("wall-clock limit -- `4h`, `90m`, `2-00:00:00`, or a bare "
             "number of minutes.  A run on a scheduler states one -- here, "
             "or `allocation.time` / the run card's `time` in "
             "job-request.json -- "
             "or prep refuses.  Never derived, never estimated "
             "(architecture.md § 5.2).")
MEM_METAVAR = "SIZE"
MEM_HELP = ("how much TOTAL memory this needs -- `128G`, `80GB`, `0.5T`, or "
            "a bare number of GB.  `0` asks for all of the node's.  A run "
            "on a scheduler states one -- here, or `allocation.mem` in "
            "job-request.json -- or prep refuses (running-a-job.md "
            "§ 5.3.1).")


def _duration(text):
    """`scheduler.quantities.parse_duration`, refusing in click's voice."""
    from ..scheduler.quantities import parse_duration
    try:
        return parse_duration(text)
    except ValueError as e:
        raise click.ClickException(f"--time: {e}")


def _resolve_stage_name(js, stage: str) -> str:
    """The job ``stage`` names, through the ONE resolver (§ 8f) -- the
    question every verb that takes a STAGE asks."""
    from ..identity import resolve_stage_ref
    from .naming import stage_refs
    refs = stage_refs(js)
    try:
        return resolve_stage_ref([refs[j.name] for j in js.jobs], stage).name
    except ValueError as e:
        raise click.ClickException(str(e))


@jobset_group.command("prep", short_help="prepare the task's groups, "
                                          "or a benchmark")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("words", nargs=-1)
@click.option("--group", "groups", multiple=True, type=int, metavar="N",
              help="task: a group to prepare, by its number in "
                   "job-request.json -- "
                   "repeated for several, each one job (engines/stages.md "
                   "2).  A group prepared and never launched is prepared "
                   "again.")
@_bundle_option()
@click.option("--from", "from_attempt", default=None,
              metavar="NN_relax/run-N",
              help="the `relax` run every force-constant stage this prep "
                   "prepares is written at, by its folder -- e.g. "
                   "'01_relax/run-0'.  Without it, relax's newest run, which "
                   "must have finished (engines/vibration.md 5.2a).  What "
                   "any other stage takes from a run is its first launch's "
                   "(`launch --from`).")
@click.option("--env", default=None,
              help="force one conda env for every job (default: each job's "
                   "env by what it asks for -- the GPU env for a GPU job, "
                   "the CPU env otherwise -- which a mixed CPU/GPU ladder "
                   "needs).")
@click.option("--np", "mpi_np", type=int, default=None, metavar="N",
              help="MPI ranks for this prep -- the launch shape, which "
                   "the job request's run card states for every prep "
                   "(project-layout.md D2); prep renders the deck for it.  "
                   "Stated nowhere, a SIESTA run is refused.")
@click.option("--cpus-per-task", type=int, default=None, metavar="C",
              help="cores per rank (OMP threads), the header's cores per "
                   "rank.  Stated "
                   "nowhere -- here or the run card's `omp_threads` / "
                   "`threads` -- a run is refused.")
@click.option("--gpus", "gres", default=None, metavar="N",
              help="how many GPUs -- a count; which card a node carries is "
                   "the machine's business (scheduler.md R2a).")
@click.option("--time", "time_", default=None, metavar=TIME_METAVAR,
              help=TIME_HELP)
@click.option("--mem", default=None, metavar=MEM_METAVAR, help=MEM_HELP)
@click.option("--max-memory-mb", type=int, default=None, metavar="MB",
              help="the run's memory cap in MB: a SIESTA run script's "
                   "ulimit -v per rank, a PySCF deck's mol.max_memory.")
@click.option("--domain", default=None, metavar="NAME",
              help="which named domain to run in (a PROBED domain from "
                   "the target's record -- a partition and a QOS together, "
                   "with its own limits).  On a scheduler a run names one "
                   "-- here, its run card's `domain`, or `allocation.domain` "
                   "in job-request.json (a benchmark: here, or "
                   "`allocation.domain`) "
                   "-- or prep refuses and lists them.")
@click.option("--target", default=None, metavar="NAME",
              help="which MACHINE this is for -- a record written by "
                   "`jobset probe --write --name NAME`, or `this` for this "
                   "machine's own.  Omit when there is one; naming it is how "
                   "a bench prepared on a workstation measures the cluster "
                   "instead of the desk.")
@click.option("--sbatch/--no-sbatch", "emit_sbatch", default=True,
              help="emit .sbatch wrappers (default on; withheld where the "
                   "target's record says `workstation` -- job-system.md "
                   "§ 6).  --no-sbatch writes none, and then no queue, wall "
                   "or memory is asked for.")
@click.option("--one-job", "one_job", is_flag=True,
              help="task: the groups named also as ONE job that launches "
                   "each in turn, when the one before has finished -- its "
                   "one header written too: one queue, the largest request "
                   "any group asks for, the walls summed.  A group named "
                   "that is launched already is read as prepared.  Then "
                   "`launch task` with the same --group lines sends it "
                   "(job-system.md 6.0, A sequence of groups).")
def prep_cmd(kind: str, words, groups, bundle: str, from_attempt, env,
             mpi_np, cpus_per_task, gres, time_, mem, max_memory_mb,
             domain, target, emit_sbatch: bool, one_job: bool) -> None:
    """Set the task's groups up to run -- each group one job -- and report
    what was done; or a benchmark of one stage.

    Renders each stage's deck and its wrappers, makes its first ``run-<n>``
    with the deck and the shared package in it, and writes each group's one
    header.  What a stage takes from another stage's run is its first
    launch's to take -- a force-constant stage's `relax` geometry alone is
    written into its deck here.  **Prep printing what it resolved is what
    makes the launch a plain yes.**

    Bare ``prep task`` shows the groups and asks which, every one not
    prepared and preparable offered; ``--group`` answers without a terminal
    (`job-system.md`, *The task*).
    """
    # THE ONE ENTRY (`job-system.md` § 5.3, plan W38 F7): the Task setup
    # tab's Prep buttons call it too.  This verb collects what the person
    # said -- the flags -- and prints the answer; the act itself is
    # `prep.prep_task`'s, and it asks nothing.
    from ..scheduler.quantities import (canonical_mem, canonical_time,
                                        parse_gres_flag)
    from .model import Resources as _Alloc
    from .prep import prep_task
    base = Path(bundle).resolve()
    try:
        bench_stage, _trial, _stages = _stage_words(kind, words, (), "prep",
                                                    groups=groups)
        if one_job and kind != "task":
            raise click.ClickException(
                "--one-job sends a task's groups as one job; a benchmark's "
                "trials are sent by their shelves (job-system.md 6.0, "
                "R-P22).")
        # A SPELLING THAT IS NO AMOUNT is refused in the verb's voice,
        # naming the flag -- through the same readers the record uses
        # (`Resources`) -- before anything is asked, so an answer is never
        # thrown away.
        for _flag, _said, _read in (("--time", time_, canonical_time),
                                    ("--mem", mem, canonical_mem),
                                    ("--gpus", gres,
                                     lambda g: g and parse_gres_flag(g))):
            try:
                _read(_said)
            except ValueError as e:
                raise click.ClickException(f"{_flag}: {e}")
    except click.ClickException as e:
        # EVERY REFUSAL IS WRITTEN DOWN (`job-system.md` § 5.0, rule 3): the
        # verb's own, before the entry -- in a described calculation only.
        from ..calcdirs import described as _described
        if _described(base):
            _ledger(base, "prep", "refused", kind=kind,
                    groups=list(groups) or None, reason=e.format_message())
        raise
    # THE CALCULATION, OPENED ONCE for the whole prep (A16).
    calc = _calc_of(base)
    allocation = _Alloc(mpi_np=mpi_np, cpus_per_task=cpus_per_task,
                        gres=gres, time=time_, mem=mem,
                        max_memory_mb=max_memory_mb, domain=domain)
    try:
        _flags_may_stand(calc, allocation)
    except click.ClickException as e:
        _ledger(base, "prep", "refused", kind=kind,
                groups=list(groups) or None, reason=e.format_message())
        raise
    asked = None
    if kind == "task":
        # THE TASK'S QUESTION (`job-system.md`, *The task*): the groups,
        # each prepared, launched or waiting, then which to prepare --
        # answered here, or by --group.
        groups, asked = _ask_which_groups(calc, list(groups))

    shown: list = []

    def _show(findings, notes):
        # The preflight's notes and what the inputs said, ONCE, before the
        # decks are written -- the entry hands them over as soon as it has
        # them (`on_found`) -- or from a refusal that carries them
        # (`prep.PrepError`), which the terminal prints before its sentence.
        if shown:
            return
        shown.append(True)
        for issue in findings:
            click.echo(f"note: {issue.message}", err=True)
        for line in notes:
            click.echo(line)

    try:
        answers = prep_task(calc, kind, groups=groups, stage=bench_stage,
                            target=target, allocation=allocation,
                            from_attempt=from_attempt, env=env,
                            emit_sbatch=emit_sbatch, on_found=_show,
                            asked=asked, one_job=one_job)
    except PrepError as e:
        _show(e.findings, e.notes)
        raise click.ClickException(str(e))
    from .prep import opened_no_run
    ladder_next = opened_no_run(answers)
    # WHAT THE PREP DID, ONCE -- the state it saved, its config, its
    # manifest -- with the first stage it wrote, not one it left as it stood.
    lead = next((i for i, a in enumerate(answers) if a.unchanged is None), 0)
    for i, ans in enumerate(answers):
        _echo_prep_answer(ans, base, first=(i == lead),
                          last=(i == len(answers) - 1),
                          ladder_next=ladder_next)


def _launch_unit(calc, *, told: dict, dry_run: bool = False):
    """``(what, asked)`` -- what `launch task` sends with nothing named
    (`job-system.md`, *The task*): the groups prepared and never launched
    whose upstream has finished, by the ready door (`ready.launch_offer`)
    -- one is sent (its number); several, the question which; none, the
    stages launched before, offered to launch again (a stage's name),
    beside what each waiting group waits for -- and the question put, with
    its answer, which the launch writes down after its save (``None`` when
    none was put).  With nobody to ask, refused, naming each line."""
    from .ask import choose
    from .commands import block, launch_lines
    from .ready import task_state
    base = calc.base
    try:
        state = task_state(calc)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    gs = state.groups
    # A GROUP A SEQUENCE STILL GOING HOLDS is that job's (R-L45): not
    # offered, nor any stage of it -- the group answer's (`held`), its
    # line saying where it goes.
    in_held = {st for g in gs if g.held for st in g.members}
    ready = state.launch_offer
    if len(ready) == 1:
        return ready[0], None
    text = "the task's groups:\n" + state.text()
    if ready:
        offered = [str(n) for n in ready]
        question = "which group to launch?"
        lines_for = [ln for n in ready
                     for ln in launch_lines("task", base=base, groups=[n])]
    else:
        again = [a.stage for g in gs for a in g.answers
                 if a.launched and a.stage not in in_held]
        if not again:
            if not any(a.prepared for g in gs for a in g.answers):
                raise click.ClickException(_nothing_prepared(calc))
            raise click.ClickException(
                text + "\nno prepared group can go yet -- each one above "
                "waits for what it says.")
        text += ("\nno prepared group can go; launched before, and "
                 "launched again warm or --cold:")
        offered = again
        question = "which stage to launch again?"
        lines_for = [ln for st in again
                     for ln in launch_lines("task", st, base=base)]
    picked = choose(text, offered[:1] if ready else [], question=question)
    # THE QUESTION AND ITS ANSWER, written down as prep's are -- by the
    # launch, after its save (`submit.LaunchPlan.held`); a dry run writes
    # nothing (§ 6.0, step 3).
    asked = (dict(about=question, offered=offered, answer=picked.words)
             if picked.asked and picked.picks and not dry_run else None)
    if not picked.asked or len(picked.picks) != 1:
        if asked is not None:
            from .ledger import record
            record(base, "launch", "question", **{**told, **asked})
        raise click.ClickException(
            text + "\nname one -- one job per launch:\n" + block(lines_for))
    one = picked.picks[0]
    if ready:
        try:
            return int(one), asked
        except ValueError:
            from .ledger import record
            if asked is not None:
                record(base, "launch", "question", **{**told, **asked})
            raise click.ClickException(
                f"{one!r}: a group is named by its number -- "
                f"{', '.join(offered)}.") from None
    return one, asked


def _ask_which_groups(calc, named) -> tuple:
    """``(groups, asked)`` -- the groups `prep task` prepares: the groups
    shown every time (`ready.groups_text`); the ones --group ``named``, or
    else the question -- every group not prepared and preparable offered,
    Enter's answer (`ready.prep_offer`), the question and its answer
    ``asked``, which the entry writes down after its save.  With
    nobody to ask, refused, naming the line that picks them; with none to
    offer, the groups shown and none named, ``([], None)``: `prep task` with
    no group named records a junction's order or refuses, naming the groups
    (`prep._prep_groups`; `engines/transport.md` § 1.1)."""
    from .ask import choose
    from .commands import block, prep_group_lines
    from .ledger import record
    from .ready import task_state
    base = calc.base

    def _refuse(why: str) -> click.ClickException:
        # EVERY REFUSAL IS WRITTEN DOWN (`job-system.md` § 5.0, rule 3).
        record(base, "prep", "refused", kind="task",
               groups=list(named) or None, reason=why)
        return click.ClickException(why)

    # A DESCRIPTION OR TEMPLATE THAT DOES NOT READ is refused in its own
    # words, as prep refuses it.
    try:
        state = task_state(calc, verdict=True)
    except ValueError as exc:
        raise _refuse(str(exc)) from None
    gs = state.groups
    text = "the task's groups:\n" + state.text()
    if named:
        click.echo(text)
        return list(named), None
    pre = state.prep_offer
    if not pre:
        # NO GROUP TO OFFER: prep, asked with none named, records a
        # junction's order or refuses, naming the groups (`prep._prep_groups`;
        # `engines/transport.md` § 1.1, *No group left to prepare*).
        click.echo(text)
        return [], None
    # WHAT EACH GROUP OFFERED COPIES, AND WHAT IT RUNS, for a calculation
    # that reuses a finished one -- so picking a group is agreeing to its
    # copy (`engines/transport.md` § 2a.16, *The manifest*).
    from .reuse import decide
    try:
        reuse = decide(calc, pre)
    except PrepError as exc:
        raise _refuse(str(exc)) from None
    if reuse is not None:
        text += "\n" + "\n".join(reuse.lines(base))
    picked = choose(text, [str(n) for n in pre],
                    question="which group(s) to prepare?")
    if not picked.asked:
        raise _refuse(
            "nobody to ask which groups to prepare (not a terminal) -- "
            "the ones offered:\n" + block(prep_group_lines(pre, base=base)))
    asked = dict(about="which group(s) to prepare", offered=pre,
                 answer=picked.words)
    try:
        answer = [int(w) for w in picked.picks]
    except ValueError:
        record(base, "prep", "question", **asked)
        raise _refuse(f"{picked.words}: groups are named by their numbers "
                      f"-- {', '.join(str(g.number) for g in gs)}.") from None
    return answer, asked


def _header_said(lines) -> str:
    """A header's request in one line -- `Directives.said`, the one
    rendering both doors show."""
    from ..scheduler.emit import Directives
    return Directives.said(lines)


def _echo_prep_answer(ans, base, *, first: bool = True,
                      last: bool = True, ladder_next: bool = False) -> None:
    """The prep report, from the entry's answer -- `job-system.md` § 5.3:
    *"prep prints what it resolved, which is what makes launch a plain
    yes"*.  Where the configuration came from, what was written, the run
    and its own files, what its first launch will take, what it will launch
    with, whether the deck agrees -- then the next command, naming the
    bundle so it works from anywhere (job-contracts.md § 2.5b).  The Task
    setup tab shows the same answer.  The stages of the groups prepared are
    reported one after another, the saved state with the ``first``, and
    after the ``last`` the launch with nothing named -- it sends the group
    that can go, asks which when several can, and says what each waits for
    when none can (`job-system.md`, *The task*).  ``ladder_next``: the
    prep opened no run of its own (`prep.opened_no_run`), so what comes
    next is the ladder's own step.
    """
    def say_next(line):
        click.echo(line)

    if ans.recorded:
        # NO GROUP NAMED: the calculation's own record, and nothing else.
        if ans.saved:
            click.echo(ans.saved)
        click.echo(ans.recorded)
        return
    if ans.unchanged:
        # A GROUP ALREADY AS THIS PREP PLANS IT: nothing written for it --
        # said, with what comes next (`job-system.md`, *The task*, 2).
        click.echo(ans.unchanged)
        if last:
            _echo_next_step(ans, base, ladder_next=ladder_next)
        return

    from ..runtime_config import format_provenance
    from .ledger import rel_to as _rel
    if ans.saved and first:
        # THE STATE A REDO RESTORES, named where the person reads it --
        # saved now, or the one the folder already stood at; once.
        click.echo(ans.saved)
    if ans.provenance is not None and first:
        # WHERE the effective config came from (user request 2026-08-12;
        # secrets excluded by design) -- also in the bundle's ledger, since
        # the terminal is gone when a job misbehaves hours later.
        click.echo(format_provenance(ans.provenance))
    if ans.copied is not None and first:
        # THE MANIFEST, once: what this calculation copied in from the
        # finished one it reuses, and what it runs (`engines/transport.md`
        # § 2a.16).
        click.echo("\n".join(ans.copied.lines(base)))
    stage = ans.stage
    from .commands import block, command, launch_lines
    if ans.kind == "bench":
        where = f" for stage {stage!r}" if stage else ""
        click.echo(f"prepared {len(ans.dirs)} trial dir(s){where} under "
                   f"{base}:")
        for d in ans.dirs:
            # the path from the bundle, not the bare attempt name -- with the
            # attempt layer every trial's dir ENDS in run-<n> (user,
            # 2026-08-28)
            click.echo(f"  {_rel(base, d)}")
        if ans.placement:
            # THE ONE QUEUE IT MEASURES ON, placed at prep and recorded on
            # every trial (`job-system.md` § 6.0) -- said as a run's is.
            from .placement import placement_line
            click.echo("  " + placement_line(ans.placement))
        for shelf in ans.shelves:
            # EACH SHELF AS ITS HEADER ASKS IT -- the one job its launch
            # sends (`shelves.shelves_of`).
            click.echo(f"  {shelf['name']}  {shelf['said']}")
            if shelf["fit"]:
                click.echo(f"    {shelf['fit']}")
        if ans.continuation is not None:
            # A FORCE-CONSTANT STAGE'S TRIALS are written at the geometry of
            # the relax run it builds on -- which one, said (§ 5.2a).
            click.echo("  every trial's geometry: "
                       + ans.continuation.line())
        _echo_pipeline_log(ans, base)
        # ONE COMMAND A LINE, prose after `#`; the report's `execution`
        # block is the person's to copy into task.json (job-system.md § 7).
        say_next("next -- launch the sweep (to the queue it goes as one "
                 "job per resource shelf):\n"
                 + block(launch_lines("bench", stage, base=base)))
        say_next("then -- a report to read; its `execution` block is "
                 "yours to copy onto a run card in job-request.json:\n"
                 + block([command("summarize", "bench", stage,
                                  base=base)]))
        return
    # THE LAYOUT'S OWN LINES first -- the folder, the attempt or the points
    # -- then what every layout says alike: what its first launch will take,
    # the resources, the pipeline log, what to do next.
    if ans.flat:
        click.echo(f"prepared {stage}: under {base}  (flat: no attempt to "
                   f"open; each run carries the number launch gives it)")
    elif ans.points:
        for att, pt in ans.points:
            click.echo(f"prepared {stage} @ {pt.rel}: "
                       f"{Path(att).relative_to(base)}")
    elif ans.copied is not None and ans.copied.copies_of(stage):
        # COPIED IN: its runs are the finished calculation's, launched runs
        # of this one (`engines/transport.md` § 2a.16).
        click.echo(f"prepared {stage}: copied in -- "
                   + ", ".join(str(c.run.relative_to(base))
                               for c in ans.copied.copies_of(stage))
                   + f" from {ans.copied.name}; its next launch opens its "
                     f"next run with these files")
    elif ans.launched:
        # PREPARED AGAIN FOR ITS JOB REQUEST (`execution/job-request.md`
        # § 5): its runs kept, its next launch opening its next run.
        click.echo(f"prepared {stage} again: launched -- its runs are kept, "
                   f"and its next launch opens its next run with these "
                   f"files")
    else:
        rep = ans.attempt
        click.echo(f"prepared {rep.stage}: {rep.dir.relative_to(base)}"
                   f"{'' if rep.fresh else '  (reused -- not launched yet)'}")
        click.echo(f"  brought in: {', '.join(rep.brought)}")
    # PREPARED AGAIN, never launched: what of it differed from what this
    # prep writes now (`job-system.md`, *The task*, 2).
    if ans.differs:
        click.echo("  prepared again -- differed from what was prepared: "
                   + ", ".join(ans.differs))
    # A FORCE-CONSTANT STAGE's deck is written at a `relax` run -- which,
    # and what it was (`engines/vibration.md` § 5.2a).
    if ans.continuation is not None:
        click.echo("  written at: " + ans.continuation.line())
    # WHAT ITS FIRST LAUNCH WILL TAKE (`job-system.md` § 5.0, *a preview*):
    # nothing is copied at prep.
    for t in ans.takes:
        click.echo("  " + t["line"])
    if not ans.takes and ans.continuation is None:
        click.echo("  takes nothing from another run"
                   + ("" if ans.linked else
                      " (the first stage, or one that starts clean)"))
    from .placement import placement_line, stated_line
    if ans.placement:
        # WHERE IT WAS PLACED, where each value came from, and the FIT of
        # what it sends -- a run of points', its walk's (`job-system.md`
        # § 6.0; `scheduler.md` R6) -- the lines the tab shows too.
        click.echo("  " + placement_line(ans.placement))
        if ans.fit:
            click.echo("  " + ans.fit)
    elif stated_line(ans.sources):
        # NO HEADER -- where each value it states came from, all the same.
        click.echo("  " + stated_line(ans.sources))
    if ans.need is not None:
        # THE MEMORY IT NEEDS at the ranks it is prepared with, beside the
        # ask -- a hint, refusing nothing (`job-request.md` § 3.1a).
        click.echo("  memory need: " + ans.need.line)
    if ans.launch and ans.launch.get("header") and ans.group is None:
        # EVERY VALUE IT IS SENT WITH, as the header its launch sends carries
        # it -- a run of points', its walk's -- the line above saying where
        # each came from.  A member of a group of several is sent in the
        # group's one job, said after its last member.
        click.echo("  " + (f"one job walking its {len(ans.points)} points "
                           f"(their walls summed) -- " if ans.points else "")
                   + ans.launch["said"])
    for line in ans.notes:
        # WHAT ITS REQUEST MEANS, said where a value can still change -- its
        # ranks over its GPUs (`ask.gpu_share_notes`) -- and what a rendering
        # says and does not refuse (`gpu.md` G6).
        click.echo("  " + line)
    if ans.resources is not None:
        r = ans.resources
        # WHAT IS STATED, and nothing else: no launch value is left for
        # the wrapper to decide (`architecture.md` § 5.2), and a PySCF run
        # has no rank count at all.
        asks = [f"{word} {r[key]}" for word, key in (("mpi_np", "mpi_np"),
                                                     ("omp", "cpus_per_task"))
                if r.get(key)]
        if r.get("max_memory_mb"):
            asks.append(f"memory cap {r['max_memory_mb']} MB")
        if r.get("continue_retries"):
            asks.append(f"retries {r['continue_retries']}")
        if asks:
            click.echo(f"  resources: {' | '.join(asks)}")
    _echo_pipeline_log(ans, base)
    g = ans.group
    if g is not None and ans.stage == g["members"][-1]:
        # A GROUP'S ONE JOB, once, after its last member: the header its
        # first launch sends, not a member's own (`job-system.md` § 5.0,
        # checkpoint 6).
        click.echo(f"group {g['number']} -- one job: "
                   f"{', '.join(g['members'])}, one after another (their "
                   f"walls summed, the largest memory)"
                   + (f"; its header: {g['header']}" if g["header"] else ""))
        if g["launch"]["header"]:
            click.echo("  " + g["launch"]["said"])
            click.echo("  " + g["launch"]["fit"])
        click.echo("  memory need: " + g["need"]["line"])
    seq = ans.sequence
    if seq is not None:
        # THE SEQUENCE'S ONE JOB, once, after the last answer: its header,
        # what it launches with, its fit, and each group asking less than
        # it (`job-system.md` § 6.0, *A sequence of groups*).
        nums = ", ".join(map(str, seq["numbers"]))
        click.echo(f"groups {nums} -- one job, each group launched when the "
                   f"one before has finished (the largest request any "
                   f"asks, their walls summed)"
                   + (f"; its header: {seq['header']}" if seq["header"]
                      else ""))
        if seq["launch"]["header"]:
            click.echo("  " + seq["launch"]["said"])
            click.echo("  " + seq["launch"]["fit"])
        click.echo("  memory need: " + seq["need"]["line"])
        for line in seq["less"]:
            click.echo("  " + line)
    if not last:
        return
    _echo_next_step(ans, base, ladder_next=ladder_next)


def _echo_next_step(ans, base, *, ladder_next: bool) -> None:
    """What comes next, after a prep's last answer: a sequence of groups'
    one job; the ladder's own step when the prep opened no run of its own
    -- every group copied in, already as it plans it, or launched and
    prepared again for its job request (`job-system.md`,
    *The next step*); else the launch with nothing named."""
    from .commands import block, launch_lines
    seq = ans.sequence
    if seq is not None:
        click.echo("next -- send them as one job:\n"
                   + block(launch_lines("task", base=base,
                                        groups=seq["numbers"])))
        return
    if ladder_next:
        from .runstatus import jobset_status, render_steps
        click.echo(render_steps(jobset_status(_calc_of(base))))
        return
    click.echo("next -- launch the group that can go, each group one job:\n"
               + block(launch_lines("task", base=base)))


def _echo_pipeline_log(ans, base) -> None:
    """Where this prep's pipeline log is -- every prep writes one
    (`script-preparation.md` § 4.5); a refused prep's answer may not say.
    A line of its own, not indented under what the prep listed: the log is
    the prep's record, beside its `STAGE-PLAN.md`, not one of the trials or
    a fact of the attempt."""
    if ans.pipeline_log is not None:
        click.echo(f"pipeline log: {Path(ans.pipeline_log).relative_to(base)}")


@jobset_group.command("summarize",
                      short_help="summarize results that exist: a sweep's "
                                 "trials, a transport's bias points, a "
                                 "vibration's displacement sweep")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("words", nargs=-1)
@_bundle_option()
@click.option("--tolerance-cm1", "tolerance_cm1", type=float, default=None,
              help="a SIESTA vibration's displacement sweep: flag every mode "
                   "whose frequency spreads by more than this across the "
                   "force-constant stages (cm^-1).  Without it nothing is "
                   "flagged -- the numbers are stated and the judgement is "
                   "yours (engines/vibration.md 5.9).")
def summarize_cmd(kind: str, words, bundle: str,
                  tolerance_cm1: Optional[float]) -> None:
    """Summarize results that exist -- never derive a run's own
    (`job-system.md` § 5, the verb table).  Three summaries, by what was
    described:

    \b
    * ``summarize bench`` reads a benchmark's trials and writes
      ``bench-result.json`` -- a recommendation, not a decision
      (`project-layout.md` § 2.3.2): you read it, you decide;
    * ``summarize task`` on a transport calculation reads its transmission
      points into ``<label>.transport.json`` (`engines/transport.md`
      § 2a.12);
    * ``summarize task`` on a SIESTA vibration with two or more
      force-constant stages compares them into ``<label>.fc-sweep.json``
      (`engines/vibration.md` § 5.9), ``--tolerance-cm1`` flagging a mode
      whose spread exceeds it.

    **Asynchronous by design** (user, 2026-08-12): each reads what has
    landed.  A trial that has produced nothing yet reports ``state=unknown``
    and one started but unfinished ``incomplete`` -- never a failure of the
    set; a transmission point or a force-constant stage without its result
    reads as pending, and one whose run failed says so.  Discovery is keyed
    by the description and ``job-set.json``'s own data, never by parsing
    directory names back (`job-contracts.md` § 6.3).
    """
    # A TASK'S SUMMARY reads the whole calculation, and names no stage
    # (`job-system.md` § 5.3); a benchmark's, its stage.
    if kind == "task" and words:
        raise click.ClickException(
            "`summarize task` reads the whole calculation -- it names no "
            "stage.")
    stage, _trial, _ = _stage_words(kind, words, (), "summarize")
    if kind != "bench":
        # THE TRANSPORT COMPOSITE'S DELIVERABLE (`engines/transport.md`
        # § 2a.12): `summarize task` on a transport calculation reads the
        # transmission attempts back into <label>.transport.json and
        # prints the I-V table.  Asynchronous like the bench reader: a
        # point that has not run yet reads as pending, never a failure.
        from pathlib import Path as _P

        # THE CALCULATION, opened once (A16); one that does not read is
        # said in its reader's words, never read as no description.
        _calc = _calc_of(_P(bundle))
        _tt = _calc.task
        # WHICH SUMMARY, by the one rule the task's next step asks too
        # (`summarize.task_summary`).
        from .summarize import task_summary
        _summary = task_summary(_tt) if _tt is not None else None
        _is_vibration = _tt is not None and _tt.calculation == "vibration"
        if tolerance_cm1 is not None and not _is_vibration:
            raise click.ClickException(
                "--tolerance-cm1 is a displacement sweep's -- a SIESTA "
                "vibration's force-constant stages compared "
                "(engines/vibration.md 5.9)")
        if _summary == "transport":
            from ..transport.record import (RecordError, collect_record,
                                            iv_table_text, write_record)
            from ..transport.tbtnc import read_points
            # EVERY POINT'S DOS PER ORBITAL, read where it is not kept: the
            # record and the data file hold each region's (`model/parse.md`
            # § 5e) -- said as it goes; a file another reading holds is not
            # read again.
            readings = read_points(_calc, say=click.echo)
            try:
                rec = collect_record(_calc)
            except RecordError as e:
                raise click.ClickException(str(e))
            # THE SAME RESULTS AS DATA, on their grid, beside the report
            # (`engines/transport.md` § 2a.12, *the data file*) -- written
            # first, as the one of the two that can refuse, so a refusal
            # leaves the pair the last summarize wrote.
            from ..transport.datafile import DataFileError, write_data_file
            try:
                data = write_data_file(_P(bundle), _tt, rec,
                                       readings=readings)
            except DataFileError as e:
                raise click.ClickException(str(e))
            out = write_record(_P(bundle), rec)
            click.echo(iv_table_text(rec))
            click.echo(f"-> {out}")
            click.echo(f"-> {data}")
            _ledger(_P(bundle), "summarize", "transport-record",
                    out=str(out), data=str(data), points=len(rec["points"]),
                    pending=len(rec["pending"]), failed=len(rec["failed"]),
                    readings=readings)
            return
        if _is_vibration and _summary is None:
            # A VIBRATION WITH NO SWEEP TO COMPARE, said in the rule's words.
            from ..spectra.displacement_sweep import why_no_sweep
            raise click.ClickException(why_no_sweep(_tt))
        if _summary == "displacement sweep":
            # A DISPLACEMENT SWEEP'S SUMMARY (engines/vibration.md § 5.9): the
            # force-constant stages' results, which their jobs wrote, read
            # where they are and compared -- nothing derived that a run did
            # not already write, nothing moved.  Every stage is compared, so
            # a stage name is not asked for (`summarize task` takes none).
            from ..spectra.displacement_sweep import (SweepError,
                                                      collect_sweep,
                                                      sweep_table_text,
                                                      write_sweep)
            try:
                rec = collect_sweep(_calc, tolerance_cm1=tolerance_cm1)
            except SweepError as e:
                raise click.ClickException(str(e))
            out = write_sweep(_P(bundle), rec)
            click.echo(sweep_table_text(rec))
            click.echo(f"-> {out}")
            _ledger(_P(bundle), "summarize", "displacement-sweep",
                    out=str(out), stages=[x["name"] for x in rec["stages"]],
                    pending=len(rec["pending"]), failed=len(rec["failed"]),
                    tolerance_cm1=tolerance_cm1)
            return
        raise click.ClickException(
            "summarize summarizes results that exist: a BENCH sweep's "
            "measurements, a transport calculation's bias points (`summarize "
            "task`, into <label>.transport.json), and a SIESTA vibration's "
            "force-constant stages (`summarize task`, into "
            "<label>.fc-sweep.json).  A run's own outputs are the "
            "calculation's results -- `jobset status` and the Results tab "
            "are their readers (job-system.md § 5.3) -- and a vibration's "
            "run writes its <label>.spectra.json itself, on both engines "
            "(engines/vibration.md § 5.5).")
    if tolerance_cm1 is not None:
        raise click.ClickException(
            "--tolerance-cm1 is a displacement sweep's -- a SIESTA "
            "vibration's force-constant stages compared "
            "(engines/vibration.md 5.9)")
    calc = _calc_of(bundle)
    base = calc.base
    stage = _described_stage(stage, calc)
    js, container = _load_bench_set(calc, stage, "summarize")
    _check_kind(kind, js)
    from .summarize import (run_summarize_jobset,
                                   summary_text, utc_now_iso)
    # The container's job-set holds ONLY this stage's trials (U1), so the
    # SET is the scope and the verdict goes back where they live -- there
    # is no name filter anywhere (U12).
    from ..runfiles import BENCH_RESULT_FILE
    res, out_path, report = run_summarize_jobset(
        calc, js, out=container / BENCH_RESULT_FILE,
        now_iso=utc_now_iso(), stage=stage)
    click.echo(summary_text(res, out_path, report=report, stage=stage,
                            base=base, request=calc.request))
    _ledger(base, "summarize", "verdict-written", stage=stage,
            out=str(out_path), points=len(res.points),
            choice=res.choice,
            reported=report is not None)


def _refuse_flags_without_effect(*, kind: str, mode: str, trial, typed,
                                 each_mode=lambda: "") -> None:
    """A launch flag that would not be read is REFUSED by name, never
    ignored -- each as the one table says who reads it
    (`commands.LAUNCH_FLAGS`): its modes, the launches that take it, a
    benchmark's grouped walk alone -- with the launch as typed for each mode
    (``each_mode``, R-L4)."""
    from .commands import LAUNCH_FLAGS
    said = {flag for flag, _value in typed}
    for row in LAUNCH_FLAGS:
        if row.flag not in said:
            continue
        if kind not in row.kinds:
            raise click.ClickException(
                f"{row.flag}: {row.what} -- `launch {kind}` does not read "
                f"it.")
        if row.grouped and trial is not None:
            raise click.ClickException(
                f"{row.flag}: {row.what} -- a named trial is sent alone, and "
                f"is the selection.")
        if mode not in row.modes:
            raise click.ClickException(
                f"{row.flag}: {row.what} -- `--mode {mode}` does not read "
                f"it.\n" + each_mode())


def _show_and_ask(plan, *, dry_run: bool, auto_yes: bool,
                  footer=(), here: bool = False, background: bool = False):
    """**Nothing is submitted unseen** (`submission.md` S4): the exact
    ``sbatch`` line of every job about to be sent -- by the code that sends
    it -- what each follows, what only the person can judge, and what the
    request leaves to the scheduler; then the one question.  ``True`` to go.

    Every door asks it, the stage's included.  Under ``--dry-run`` the
    commands are printed by the results that
    follow, so only what the plan alone carries is said here, and nothing
    is asked: ``None``.  Otherwise the answer (`ask.Said`).  ``--yes``
    skips the question, never the output.  A run ``here`` is shown and
    asked as a submission is (user, 2026-10-05), the question offering the
    other way of running it -- in the foreground, or in the background
    (user, 2026-10-16; `job-system.md` § 6.0)."""
    from .ask import confirm
    lines = [("about to start here, in the background:" if background
              else "about to run here, waited on:") if here
             else "about to submit:"]
    for r in plan:
        if r.status == "planned":
            # NAME THE QUEUE, not only its header's lines: where several
            # queues share one partition, `-p htc` reads as *htc* to anyone
            # scanning (2026-08-30: a debug sweep was believed to have gone
            # wrong).
            lines.append(f"  {r.name}"
                         + (f"   -> domain {r.domain}" if r.domain else ""))
            lines.append(f"    {' '.join(r.command)}")
            if r.request:
                lines.append(f"    {_header_said(r.request)}")
        elif r.status == "takes":
            lines.append(f"  {r.name}: {r.said}")
    # WHAT THE REQUEST MEANS -- a GPU's ranks, a queue's job cap -- was said
    # at prep, where a value can still change (`job-system.md` § 6.0).
    lines += list(footer)
    if dry_run:
        for line in footer:
            click.echo(line)
        return None
    if not here:
        return confirm("\n".join(lines), auto_yes=auto_yes,
                       question="submit this?")
    return confirm("\n".join(lines), auto_yes=auto_yes,
                   question=("launch in the background?" if background
                             else "launch in the foreground?"),
                   other=(("f", "foreground") if background
                          else ("b", "background")))


@jobset_group.command("launch", short_help="launch what is prepared")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("words", nargs=-1)
@click.option("--group", "groups", multiple=True, type=int, metavar="N",
              help="task: the group to launch, by its number in "
                   "job-request.json "
                   "-- its stages as one job (engines/stages.md 2).  A "
                   "group goes whole the first time.  Repeated: the groups "
                   "as ONE job that launches each in turn, when the one "
                   "before has finished -- its header the one `prep task "
                   "--group ... --one-job` wrote (job-system.md 6.0, A "
                   "sequence of groups).")
@click.option("--stage", "stages", multiple=True, metavar="NAME",
              help="task: ONE stage, by name or '#N' -- a member launched "
                   "again alone, or a group holding only it.")
@_bundle_option()
@click.option("--mode", type=click.Choice(["submit", "direct", "ask"]),
              default=None,
              help="HOW to launch, which is a fact about this MACHINE and not "
                   "about the layout: 'direct' = run it here with bash; "
                   "'submit' = hand it to this machine's scheduler; "
                   "**'ask' = submit NOTHING and report when it "
                   "would start** (the scheduler asked about the line "
                   "submit would send, its kind's ask -- "
                   "scheduler/kinds.toml), so you can change the request at "
                   "prep and "
                   "ask again before committing.  'ask' needs a login "
                   "node -- there is no prediction without the cluster.  "
                   "**Defaults to `launch.mode` in molbuilder.json** "
                   "(running-a-job.md § 5.4); pass it only to override that.")
@click.option("--dry-run", is_flag=True,
              help="print the exact command each job WOULD get; launch "
                   "nothing, write nothing.")
@click.option("--time", "time_text", default=None, metavar=TIME_METAVAR,
              help="submit and ask: a submitted job's wall, restated on the "
                   "line that sends it -- for a run launched again that ran "
                   "out of it.  Every other value -- the queue, ranks, cores, "
                   "GPUs, memory -- is the header prep wrote, sent as it is "
                   "(job-system.md 6.0).")
@click.option("--yes", "-y", "auto_yes", is_flag=True,
              help="take what is shown without being asked.  The request is "
                   "still printed -- --yes skips the question, never the "
                   "output (submission.md S4).")
@click.option("--cold", is_flag=True,
              help="ONE stage: at its first launch, it takes nothing from "
                   "the stage before it -- it starts from the calculation's "
                   "structure; launched again, nothing from a run of its own "
                   "(on the flat layout its run script removes the restart "
                   "files lying in the folder).  Without it, a first launch "
                   "takes the stage "
                   "before it's newest run, which must have finished, and a "
                   "launch again continues from its own latest run, however "
                   "it ended (job-system.md 5.4).")
@click.option("--from", "from_attempt", default=None,
              metavar="NN_STAGE/run-N",
              help="ONE stage: the run it continues from, by its folder -- "
                   "e.g. '01_coarse/run-0' -- at its first launch, or "
                   "launched again, taken as said; its restart files are "
                   "COPIED in.  The flat layout and a transport rung have "
                   "no run to name (job-system.md 5.4).")
@click.option("--gather", is_flag=True,
              help="ONE transport rung launched again: take its inputs anew "
                   "-- the newest run of each rung upstream, under the "
                   "gather's gates -- instead of those its last run was "
                   "given (engines/transport.md 6.1).")
@click.option("--trial-timeout", "trial_timeout_min", default=None,
              type=click.IntRange(min=1), metavar="MINUTES",
              help="a benchmark's walk (`launch bench <stage>`, here or "
                   "sent to its queue): kill any single trial after this many "
                   "minutes so the rest of the walk still runs; the killed "
                   "trial reads incomplete.  Unstated, no per-trial bound "
                   "exists -- each trial runs until it ends, or the job's "
                   "wall.")
@click.option("--foreground", is_flag=True,
              help="--mode direct: run it here and wait for it, the launch "
                   "ending with how the run ended.  Without it a run here "
                   "starts in the background and the launch returns at once, "
                   "leaving it running -- closing the terminal does not end "
                   "it, `molbuilder jobs stop` does -- and nothing is lost: "
                   "the run script copies everything it prints into its "
                   ".runwrap log in the run folder, a walk appends each "
                   "member's output to its log under launch/, and `jobset "
                   "status` follows the run.  Either way the question "
                   "offers the other one (job-system.md 6.0).")
@click.option("--in-sequence", "in_sequence", default=None, metavar="NAME",
              help="typed by a sequence of groups' walk alone: this group's "
                   "launch runs inside the sequence NAME's job, each run "
                   "recording that job (job-system.md 6.0).")
@click.option("--in-job", "in_job", default=None, metavar="ID",
              help="typed by a sequence of groups' walk alone, with "
                   "--in-sequence: the job's address -- a queue job's id, "
                   "or a walk run here's <host>-<pid>.")
@click.option("--only", "only_side", default=None,
              type=click.Choice(["cpu", "gpu"]),
              help="a grouped bench: send just this side of a sweep that "
                   "spans CPU and GPU trials (generator.md § 4.3a).  The "
                   "other side stays pending; a later `launch bench` sends "
                   "it.")
def submit_cmd(kind: str, words, groups, stages, bundle: str, mode: str,
               dry_run: bool, time_text, auto_yes, cold, from_attempt,
               gather, foreground, trial_timeout_min, in_sequence, in_job,
               only_side) -> None:
    """Launch a prepared group, or a stage again: run it here (direct) --
    in the background, unless ``--foreground`` or your answer waits for it
    -- hand it to the machine's scheduler (submit), or ask the scheduler
    when it would start (ask).  Run ``prep`` first.  Before anything is sent
    the exact line is shown and you are asked; ``--dry-run`` shows it and
    sends nothing.

    ``--mode`` falls back to ``launch.mode`` (`running-a-job.md` § 5.4).
    """
    # EVERY REFUSAL IS WRITTEN DOWN (`job-system.md` § 6.0, step 5): the
    # entry writes its own, and the verb the ones it says before it calls
    # the entry -- in a described calculation only, a folder that is not one
    # getting no ledger of ours; a dry run writes nothing.  WHAT THE VERB
    # WAS TOLD, one record filled in as the verb learns it -- the kind, what
    # was named (a benchmark's stage and one trial by position; a task's
    # group by --group, or one stage by --stage, § 5.3), the mode and where
    # it came from, the flags as typed, which config files answered -- on
    # every line, the verb's refusals and the entry's alike.
    groups = sorted(set(groups))
    group = groups[0] if len(groups) == 1 else None
    told = {"kind": kind, "stage": list(stages) or None,
            "group": (group if len(groups) < 2 else list(groups)),
            "trial": None, "mode": mode,
            "flags": _typed(click.get_current_context()),
            # A GROUP'S LAUNCH INSIDE A SEQUENCE'S JOB, on every line.
            **({"in_sequence": in_sequence, "in_job": in_job}
               if in_sequence is not None else {})}
    try:
        stage, trial, stages = _stage_words(
            kind, words, stages, "launch", groups=groups)
        if (in_sequence is None) != (in_job is None) or (
                in_sequence is not None
                and (kind != "task" or group is None or mode != "direct"
                     or not foreground or not auto_yes or stages)):
            raise click.ClickException(
                "--in-sequence and --in-job are typed by a sequence of "
                "groups' walk alone: both, with one --group, --mode direct "
                "--foreground --yes (job-system.md 6.0; R-L46).")
        if kind == "task" and len(stages) > 1:
            raise click.ClickException(
                "--stage names ONE stage -- a member launched again alone; "
                "stages launched together are a group, by its number: "
                "--group N (job-system.md § 5.3).")
        if kind == "task" and stages and groups:
            raise click.ClickException(
                "--group and --stage are two answers to one question: the "
                "group to launch, or one stage (job-system.md § 5.3).")
        if kind == "task" and stages:
            stage = stages[0]
        told.update(stage=stage, trial=trial)
        # A GROUP'S LAUNCH INSIDE A SEQUENCE'S JOB reads, from its plan to
        # the last line it prints, as a reader inside that job
        # (`joblist.inside`, `job-system.md` § 6.0).
        from ..joblist import inside
        with inside(in_job if in_sequence is not None else None,
                    in_sequence):
            _launch(told, kind, stage, trial, bundle, mode, dry_run,
                    time_text, auto_yes, trial_timeout_min, only_side, cold,
                    group, from_attempt=from_attempt, gather=gather,
                    foreground=foreground,
                    several=tuple(groups) if len(groups) > 1 else (),
                    in_sequence=in_sequence, in_job=in_job)
    except SubmitError as e:
        # the entry's refusal, which the entry wrote down
        raise click.ClickException(str(e)) from None
    except click.ClickException as e:
        from ..calcdirs import described as _described
        if not dry_run and _described(bundle):
            _ledger(bundle, "launch", "refused", reason=e.format_message(),
                    **told)
        raise


def _typed(ctx) -> list:
    """THE LAUNCH FLAGS AS TYPED, ``[flag, value]`` in the help's order --
    each row of the one table (`commands.LAUNCH_FLAGS`) the line gave a
    value: recorded with what the verb was told, refused where it would not
    be read, and carried by every line that offers this launch again in
    another mode (`commands.launch_with`)."""
    from .commands import LAUNCH_FLAGS
    rows = {r.flag for r in LAUNCH_FLAGS}
    out = []
    for param in ctx.command.params:
        flag = next((o for o in param.opts if o in rows), None)
        value = ctx.params.get(param.name)
        if flag is not None and value not in (None, False):
            out.append([flag, value])
    return out


def _how_runs_ended(results, plan, calc, sweep):
    """A RUN HERE, WAITED ON, IS SAID AS IT ENDED (`job-system.md`, *A run
    here, waited on*): every member of a submission that ran here is read
    through the one status door -- the rows `jobset status` prints, a swept
    rung's from its points, a benchmark's trials against its calculation, as
    `status` reads them -- and a submission any member of which did not
    finish is ``failed``, the row's own words its ``detail``.  It runs
    after the send, so nothing here is a refusal: a status that does not
    read is said on each submission's line, and the launch ends with 1.
    A group's launch inside a sequence's job reads as a reader inside that
    job (`joblist.inside`, set around the whole launch by the verb): its
    runs by their own files, its walk having returned (`job-system.md`
    § 6.0, *A sequence of groups*)."""
    if not any(r.status == "ran" for r in results):
        return results
    from dataclasses import replace
    from ..identity import stage_key
    from .runstatus import jobset_status
    try:
        # THE ROWS ARE THE DESCRIPTION'S LADDER (A16: the calculation as
        # the verb opened it); a sweep's are its own trials.
        rows = {stage_key(s.ref.name): s for s in jobset_status(
            calc, sweep=sweep).stages}
    except (OSError, ValueError, KeyError) as exc:
        return [replace(r, status="unread",
                        detail=f"the run ended, and how it ended could not "
                               f"be read: {exc}")
                if r.status == "ran" else r for r in results]
    # A SEQUENCE OF GROUPS has no member of its own: every stage of its
    # groups is read, a group it never reached said as its row stands.
    members = {s.name: ([st for g in s.sequence["groups"]
                         for st in g["stages"]] if s.sequence
                        else [m.job.name for m in s.members])
               for s in plan.submissions}
    out = []
    for r in results:
        if r.status != "ran":
            out.append(r)
            continue
        said = []
        for name in members.get(r.name, []):
            row = rows.get(stage_key(name))
            if row is None:
                said.append(f"{name}: not in the status")
            elif row.state != "finished":
                said.append(f"{name}: {row.state}"
                            + (f" -- {row.detail}" if row.detail else ""))
        out.append(replace(r, status="failed", detail="; ".join(said))
                   if said else r)
    return out


def _launch(told: dict, kind: str, stage, trial, bundle: str, mode: str,
            dry_run: bool, time_text, auto_yes, trial_timeout_min, only_side,
            cold: bool, group=None, *, from_attempt=None,
            gather: bool = False, foreground: bool = False,
            several: tuple = (), in_sequence: Optional[str] = None,
            in_job: Optional[str] = None) -> None:
    """The launch verb's body (:func:`submit_cmd`, which writes down any
    refusal it raises).  ``told`` is what the verb was told, filled in as
    the body learns it: the mode and where it came from, the stage as the
    description spells it, which config files answered -- every line of
    the ledger carries it."""
    mode_source = "--mode flag"
    typed = told["flags"]

    # ------------------------------------------------------------------ #
    #  Which work -- before which queue, which is read off this work      #
    # ------------------------------------------------------------------ #
    # A QUESTION PUT BEFORE THE PLAN -- which group -- is the launch's to
    # write, after its save (`submit.LaunchPlan.held`).
    held_q = None
    # THE CALCULATION, OPENED ONCE for the whole launch (A16): the plan
    # sent, every gate, the send and what it says after read this one.
    calc = _calc_of(bundle)
    bench = None
    if kind == "bench":
        # the stage's own sweep record, from its bench container (§ 6.3) --
        # with the folder it lies in, which the send reads it again from
        stage = told["stage"] = _described_stage(stage, calc)
        bench = _load_bench_set(calc, stage, "launch")
        js = bench[0]
    else:
        js = calc.jobset
        if js is None:
            raise click.ClickException(_nothing_prepared(calc))
    base = calc.base
    _check_kind(kind, js)
    # This machine's `launch.mode` when no --mode is given (running-a-job
    # § 5.4), and every flag the mode refuses -- before any question, so
    # an answer is never thrown away; the refusal says the launch as typed
    # with each mode.
    words = tuple(w for w in (stage, trial) if w is not None)
    groups = list(several) or ([group] if group is not None else [])

    def each_mode() -> str:
        # THE LAUNCH AS TYPED, with each mode this machine takes -- a run
        # here's line without a queue's flags, a queue's without a run
        # here's (`commands.launch_with`).
        from .commands import block, launch_with, takes_a_queue
        return block([launch_with(kind, *words, base=base, mode=m,
                                  typed=typed, groups=groups)
                      for m in (("direct", "submit") if takes_a_queue(base)
                                else ("direct",))])
    if mode is None:
        from ..runtime_config import get_launch_mode
        try:
            mode = get_launch_mode()
        except Exception as exc:
            # A malformed config is ITS OWN error.
            raise click.ClickException(
                f"the launch block could not be resolved from config: "
                f"{exc}\n  Fix the config (running-a-job.md § 5.4).") from exc
        if not mode:
            # Unset is a refusal, never a derivation: deciding `submit` from
            # a DETECTED scheduler would gate submission on detection, which
            # running-a-job.md § 5.4 forbids.  THE WAY ON AS COMMANDS, one
            # per mode the machine takes.
            raise click.ClickException(
                "no --mode, and molbuilder.json sets no `launch.mode`.\n"
                "  'direct' runs it here with bash; 'submit' hands it to the "
                "scheduler.  Set launch.mode once for this machine "
                "(running-a-job.md § 5.4), or say the mode on the line:\n"
                + each_mode())
        mode_source = "launch.mode (config)"
    told.update(mode=mode, mode_source=mode_source)
    _refuse_flags_without_effect(kind=kind, mode=mode, trial=trial,
                                 typed=typed, each_mode=each_mode)
    time_s = _duration(time_text)
    if kind == "bench":
        only = _pick_trial(js, trial)            # None: every trial
    else:
        # The description's spelling from here on -- what the ledger
        # records and every line prints (plan § 5w K12) -- and a stage it
        # holds that is not prepared, said so (W55 D4).
        if several:
            # SEVERAL GROUPS, one job launching each in turn: the entry
            # plans it (`submit._plan_sequence`).
            pass
        elif group is None and stage is None:
            # NOTHING NAMED: the prepared groups whose upstream has
            # finished, one sent and several asked; none, the stages
            # launched before (`job-system.md`, *The task*, D3).
            picked, held_q = _launch_unit(calc, told=told, dry_run=dry_run)
            if isinstance(picked, int):
                group = told["group"] = picked
            else:
                stage = picked
        if group is None and not several:
            stage = told["stage"] = _described_stage(stage, calc)
            _refuse_unprepared(calc, stage)
            only = stage = told["stage"] = _resolve_stage_name(js, stage)
        else:
            only = None
    # THE WORDS EVERY PRINTED LAUNCH REPEATS: the stage and its trial; a
    # group by its number (``groups``).
    words = tuple(w for w in (stage, trial) if w is not None)
    groups = list(several) or ([group] if group is not None else [])
    grouped = kind == "bench" and trial is None

    # THE QUEUE is the one the work's prep recorded (`submit.plan_launch`;
    # floor 7 never works out a launch, `architecture.md` § 2.1).
    # The same provenance line prep printed, at the LAST moment before the
    # launch -- the mode above may have come from config, and this names
    # which file said so (user request 2026-08-12).
    from ..runtime_config import config_provenance, format_provenance
    prov = config_provenance(project_dir=base)
    click.echo(format_provenance(prov))
    from .submit import ask_launch, plan_launch, send_launch

    # A grouped bench's per-trial bound.  NOTHING IS DERIVED (user
    # dictation, 2026-08-24): --trial-timeout is itself or absent.
    _bound_s = (trial_timeout_min * 60 if trial_timeout_min is not None
                else None)

    # 1 · THE PLAN, made once (`job-system.md` § 6.0): what is shown, what
    # the question is about and what the send carries out are one object.  The
    # entry writes its own decisions down, with what this verb was told on
    # each line; a dry run writes nothing.
    told.update(provenance=prov,
                # A LAUNCH'S WALL, as typed: the value it restates is
                # recorded in the launch record too (§ 6.0).
                **({"time": time_text} if time_text else {}),
                **({"trial_timeout_s": _bound_s, "side": only_side}
                   if grouped else {}))
    # WHAT WAS ASKED BEFORE THE PLAN goes with it: written before its
    # refusal, or after the send's save (`plan_launch`'s ``held``).
    inside = None
    if in_sequence is not None:
        # A GROUP'S LAUNCH INSIDE A SEQUENCE'S JOB (`job-system.md` § 6.0):
        # its record read from the calculation's `launch/` (R-L46).
        from .submit import Inside, SubmitError as _SE
        try:
            inside = Inside.of(base, in_sequence, in_job)
        except _SE as exc:
            raise click.ClickException(str(exc)) from None
    plan = plan_launch(calc, bench=bench, mode=mode, only=only,
                       side=only_side,
                       time_s=time_s, trial_timeout_s=_bound_s,
                       told=told, record=not dry_run, cold=cold,
                       group=group, from_attempt=from_attempt,
                       gather=gather, background=not foreground,
                       held=[held_q] if held_q is not None else [],
                       groups=several, inside=inside)
    from .commands import command
    if mode == "ask":
        # NOTHING IS SAVED OR SENT: what was asked is written now.
        for facts in plan.held:
            plan.record("question", **facts)
        results = ask_launch(plan)
        # NOTHING WAS SUBMITTED.  The line the scheduler was asked about is
        # the line that WOULD be sent, the header its prep wrote, plus its
        # kind's ask -- so what is printed here and what launch would do
        # cannot drift.
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
            # creates nothing, so there is nothing to warn about.
            click.echo("  already run, so not asked: " + ", ".join(ran))
        if skipped:
            from .submit import ASK_MAX_QUERIES
            click.echo(f"  NOT asked (past {ASK_MAX_QUERIES} queries): "
                       + ", ".join(skipped))
        for r in results:
            if r.status == "takes":
                # a stage launched before: asking created nothing (the
                # attempt opens at launch), and the line asked about is
                # the one that attempt would send.
                click.echo(f"  {r.name}: {r.said}")
        if not preds:
            click.echo("\n  nothing left to ask about here.")
            if ran:
                click.echo("  read what they measured:\n    "
                           + command("summarize", "bench", stage, base=base))
            return
        click.echo("")
        # "nothing to wait for" followed by an sbatch preview reads as a
        # contradiction (user, 2026-08-28): on a scheduler-less machine
        # nothing WOULD be sent, so nothing is previewed.
        # THE LAUNCH AS A COMMAND, its flags as typed (`commands.
        # launch_with`).
        from .commands import block, launch_with
        if not all(p.no_scheduler for p in preds):
            # EVERY LINE ASKED ABOUT, and every value of the header it sends
            # (`submission.md` S4) -- what the answer is an answer about.
            for r in results:
                if r.prediction is None:
                    continue
                click.echo(f"  would send {r.name}"
                           + (f"   -> domain {r.domain}" if r.domain else "")
                           + f"\n    {' '.join(r.command)}")
                if r.request:
                    click.echo(f"    {_header_said(r.request)}")
            click.echo("  to send it, when the answer suits you:\n"
                       + block([launch_with(kind, *words, base=base,
                                            mode="submit", typed=typed,
                                            groups=groups)]))
        else:
            click.echo("  run it here instead:\n"
                       + block([launch_with(kind, *words, base=base,
                                            mode="direct", typed=typed,
                                            groups=groups)]))
        return

    # 2 · SHOWN, and 3 · ASKED -- once, for the whole plan.  A dry run stops
    # here and writes nothing, the ledger included (§ 6.0, step 3).
    shown = plan.shown()
    # WHERE A WALK'S OUTPUT GOES: each member's into the walk's log -- a
    # benchmark's trials, a group's stages, a sweep's points -- so the
    # terminal shows nothing while it runs (the unit 11 review)
    from .ledger import rel_to
    walks = ["  the walk's output: " + rel_to(base, s.log)
             for s in plan.submissions if s.direct and s.log]
    # WHERE A RUN HERE IS READ, in the background or not -- the question
    # may choose either: its own folder, the run script's .runwrap log
    # holding everything it prints, the engine's .out its own output.
    walks += ["  its output: " + rel_to(base, m.run_dir) + "/ -- the "
              ".runwrap log holds everything the run script prints, "
              "the .out the engine's own"
              for s in plan.submissions if s.direct and not s.log
              for m in s.members]
    footer = walks + (["  per-trial bound: "
               + (f"{_bound_s // 60} min" if _bound_s else
                  "none -- each trial runs until it ends"
                  + ("" if mode == "direct" else ", or the job's wall"))]
              if grouped else [])
    if dry_run:
        _show_and_ask(shown, dry_run=True, auto_yes=auto_yes, footer=footer)
        results = shown
    else:
        # EVERY LAUNCH IS SHOWN AND ASKED, a run here as a submission is
        # (user, 2026-10-05) -- `--yes` the answer given in advance.  With
        # nobody to ask the entry refuses it, written down, so a script is
        # never told it went (`submission.md` S4).
        answer = _show_and_ask(shown, dry_run=False, auto_yes=auto_yes,
                               footer=footer, here=(mode == "direct"),
                               background=plan.background)
        # 4 · THE SEND, and 5 · THE RECORD -- the answer written down, a *no*
        # too (W55 D6), and each submission the moment it goes: a run here
        # when it starts, a scheduler job when it has its id.
        try:
            results = send_launch(plan, said=answer)
        finally:
            if plan.saved:
                # THE STATE A REDO OF THE GROUP RESTORES, named where the
                # person reads it -- a send the scheduler then refuses too --
                # as prep names its own (`checkpointing.md` § 9).
                click.echo(plan.saved)
        if not answer:
            click.echo("nothing submitted." if mode == "submit"
                       else "nothing launched.")
            return
        if not plan.background:
            results = _how_runs_ended(results, plan, calc,
                                      js if kind == "bench" else None)

    verb = (("WOULD start in the background" if plan.background
             else "WOULD run") if dry_run
            else "result")
    for r in results:
        tail = (f"job {r.job_id}" + (f" -- {r.detail}" if r.detail else "")
                if r.job_id else
                (f"rc={r.returncode}"
                 + (f" -- {r.detail}"
                    if r.status in ("failed", "unread") and r.detail
                    else "")
                 if r.returncode is not None else
                 (r.detail or "") if r.status in ("started", "submitted")
                 else ""))
        # a skipped trial was not run and WOULD not be -- its verb says so;
        # a trial or a point riding a shelf or a chain is launched BY that
        # one command
        v = ("skip     " if r.status == "skipped"
             else "rides    " if r.status == "rides"
             # WHAT A MEMBER TAKES, said beside the line that sends it --
             # no command of its own
             else "takes    " if r.status == "takes"
             else "pending  " if r.status == "stays pending"
             # A RUN HERE THAT DID NOT FINISH says so in its first word.
             else "FAILED   " if r.status == "failed"
             else "UNREAD   " if r.status == "unread"
             else verb)
        click.echo(f"  {v}  {r.name:<12} {' '.join(r.command)}  "
                   f"[{r.shown}] {tail}".rstrip())
        if r.request:
            click.echo(f"  {'':9}  {_header_said(r.request)}")
    # A PARTIAL SUBMISSION SAYS SO, LOUDLY.  One shelf refused by the
    # scheduler does not cancel the others, so the run can end with some
    # jobs queued and some not; the refused shelves' trials stay pending,
    # so re-running `launch` picks up exactly them.
    _refused = [r for r in results if r.status == "scheduler refused"]
    if _refused:
        click.echo("")
        click.echo(f"  {len(_refused)} group(s) the scheduler refused -- "
                   f"their trials stay pending, the rest are queued:")
        for r in _refused:
            click.echo(f"    {r.name}")
            for _l in (r.detail or "").splitlines():
                click.echo(f"      {_l}")
        click.echo("  Re-run this launch after fixing the ask; the groups "
                   "already queued are skipped.")
    if not dry_run and kind == "bench":
        # ONE COMMAND A LINE, the sentence above it (`commands`).
        click.echo("next -- read the measurements once they have run:\n    "
                   + command("summarize", "bench", stage, base=base))
    elif not dry_run:
        # THE NEXT STEP, read after what was sent (`job-system.md`, *The
        # task*, *The next step*): every group, and the line that moves it
        # -- after the send, said, never a refusal.
        from .runstatus import jobset_status, render_steps
        try:
            click.echo(render_steps(jobset_status(calc)))
        except (OSError, ValueError, KeyError) as exc:
            click.echo(f"the next step could not be read: {exc}\n    "
                       + command("status", base=base))
    # A RUN HERE THAT DID NOT FINISH ENDS THE LAUNCH WITH 1 (`job-system.md`,
    # *A run here, waited on*): how it ended is on its line above, in the
    # status door's words, so a script trusting the exit status is told.
    if not dry_run and any(r.status in ("failed", "unread")
                           for r in results):
        click.get_current_context().exit(1)


# --------------------------------------------------------------------- #
#  probe -- this machine's record, read off the machine.                #
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
    probe: the kept values were re-CONFIRMED now, and the stamp says when
    the record was last looked at -- but a DECLARED fact kept stays
    declared, its section's ``source`` keeping ``flag`` (user, 2026-10-03,
    `configuration.md` M-6).

    Returns the record to write.
    """
    import dataclasses as _dc

    diffs = []                                # (name, recorded, probed, keep)
    # THE SCHEDULER IS ONE FACT -- its kind and where its commands are,
    # asked about as one block, said in words; kept, it keeps the note of
    # how it arrived, as the record has it.
    def _keep_scheduler():
        probed.scheduler = before.scheduler
        if "scheduler" in before.source:
            probed.source["scheduler"] = before.source["scheduler"]
    if before.scheduler != probed.scheduler:
        diffs.append(("scheduler", before.scheduler.said(),
                      probed.scheduler.said(), _keep_scheduler))
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
    # HOW THE MACHINE ENTERS ITS ENVIRONMENT is this machine's
    # molbuilder.json's, which molbuilder runs with here: copied as it
    # stands, asked nothing, and said when it changed (`configuration.md`
    # § 4, M-6).
    entered = before.env_init != probed.env_init
    if entered:
        click.echo(f"\nenv_init: copied from this machine's molbuilder.json "
                   f"-- recorded {before.env_init!r} -> {probed.env_init!r}")
    # THE THREE MEASURED FACTS THAT TRAVEL WITH A RECORD
    # (`diagnostics.local_facts`: which envs it holds, what they were built
    # for, whether its manager's `run` enters one) are asked about like
    # every other difference.
    for _f in ("conda_envs", "env_arch", "manager_run"):
        b, pv = getattr(before, _f), getattr(probed, _f)
        if b != pv:
            diffs.append((_f, b, pv,
                          lambda n=_f, v=b: setattr(probed, n, v)))
    # The reachable-domain SET is one fact -- a per-row question would ask
    # about a menu nobody composed row by row.
    if [d.to_row() for d in before.domains] != \
            [d.to_row() for d in probed.domains]:
        shown_b, shown_p = _domains_shown(before.domains, probed.domains)
        diffs.append((
            "domains", shown_b, shown_p,
            lambda: setattr(probed, "domains", list(before.domains))))

    if not diffs:
        click.echo("\nnothing else differs -- refreshing detected_at."
                   if entered else
                   "\nthe record already says this -- refreshing "
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
    # A DECLARED FACT KEPT STAYS DECLARED.  `source` is noted per section
    # (`scheduler`, `topology`, `site`, `domains`); where a kept value's
    # section was declared in the record, the new note keeps `flag`.
    for section in {name.split(".")[0] for name in kept} & set(before.source):
        was = before.source.get(section, "").split("+")
        now = probed.source.get(section) or "unknown"
        if "flag" in was and "flag" not in now.split("+"):
            probed.source[section] = ("flag" if now == "unknown"
                                      else f"{now}+flag")
    click.echo("  " + "; ".join(filter(None, [
        f"took probed: {', '.join(took)}" if took else "",
        f"kept recorded: {', '.join(kept)}" if kept else ""])))
    return probed


@jobset_group.command("migrate",
                      short_help="rewrite what an older molbuilder wrote in "
                                 "a calculation, each change said")
@_bundle_option()
def migrate_cmd(bundle: str) -> None:
    """Rewrite what an older molbuilder wrote in a calculation -- the road
    every reader names when it refuses such a file: the description and the
    plans, the copy of the machine's record, a flat calculation's run files
    and last run, a transport lead's files on its own stem, and the
    template's older, renamed and retired items and the run settings it
    held, a value moved onto the run card.

    What the calculation was is KEPT: every value carried, every blank left
    blank, each change printed with the rule it follows; a change that would
    alter the calculation is refused by name, writing nothing.  Each old file
    rewritten stays beside its new one as ``<name>.pre-<rule>``, the earliest
    rule it was written before; a file renamed keeps every byte."""
    from ..calcdirs import described as _described
    from .migrate import CURRENT, MigrateError, migrate_state
    try:
        said = migrate_state(bundle)
    except MigrateError as exc:
        # A REFUSAL IS WRITTEN DOWN, in a described calculation's own ledger
        # (`architecture.md` § 2.1, floor 7).
        if _described(bundle):
            _ledger(bundle, "migrate", "refused", reason=str(exc))
        raise click.ClickException(str(exc)) from None
    if not said:
        # CURRENT: said, written down, and no failure.
        if _described(bundle):
            _ledger(bundle, "migrate", "current", reason=CURRENT)
        click.echo(f"{bundle} is current -- {CURRENT}.")
        return
    click.echo(f"migrated {bundle}:")
    for line in said:
        click.echo(f"  {line}")


@jobset_group.command("export",
                      short_help="a transport calculation's every result, "
                                 "as NetCDF or as CSV tables")
@click.option("--format", "fmt", required=True,
              type=click.Choice(list(_EXPORT_FORMATS)),
              help="netcdf: the data file, every variable saying what it "
                   "is; csv: its tables, one per set of axes, and a README.")
@click.option("--to", "to", required=True, type=click.Path(file_okay=False),
              help="the folder it is written into -- never one inside the "
                   "calculation.")
@_bundle_option()
def export_cmd(bundle: str, fmt: str, to: str) -> None:
    """Every result of a transport calculation as it stands -- its data
    file, ``<label>.transport.nc``, or the same as tables,
    ``<label>.transport-csv/`` -- from what is already there, no data file
    read: the points without their DOS by region said, and written into the
    folder named, never into the calculation (engines/transport.md 2a.12,
    *Export*).  A ladder in progress exports what exists, a point not done
    an empty cell; what is there already under that name is refused before
    anything is composed, naming it."""
    from pathlib import Path as _P
    from ..persist import write_bytes
    from ..transport.export import ExportError, export, name_of
    from ..units import size_words
    calc = _calc_of(_P(bundle))
    base, dest_dir = _P(calc.base).resolve(), _P(to).resolve()
    if dest_dir == base or base in dest_dir.parents:
        raise click.ClickException(
            f"{to} is inside the calculation -- an export is written "
            f"outside it, its own records untouched; name another folder "
            f"with --to")
    dest = dest_dir / name_of(calc, fmt)
    if dest.exists():
        raise click.ClickException(
            f"{dest} is there already -- move it aside, or name another "
            f"folder with --to; nothing was written")
    try:
        got = export(calc, fmt)
    except ExportError as exc:
        raise click.ClickException(str(exc)) from None
    files = ({dest: got.files[got.name]} if fmt == "netcdf" else
             {dest / rel: data for rel, data in got.files.items()})
    # WRITTEN OUTSIDE THE CALCULATION: an export changes nothing of it, so
    # its ledger has no line of it.  A group's tables in its own folder.
    try:
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_bytes(path, data)
    except OSError as exc:
        raise click.ClickException(
            f"writing {dest} stopped: {exc} -- what is there is not the "
            f"whole export; remove it and export again") from None
    size = sum(len(d) for d in files.values())
    click.echo(f"wrote {dest} ({size_words(size)}"
               + (f", {len(files) - 1} tables and README.txt)"
                  if fmt == "csv" else ")"))
    if got.said:
        click.echo(f"note: {got.said}")


@jobset_group.command("scratch",
                      short_help="the engine's scratch a calculation's runs "
                                 "hold -- listed; --empty removes it")
@click.option("--empty", is_flag=True,
              help="remove it from every run that is not queued or running, "
                   "after one question naming how many files and how much.")
@click.option("--yes", "-y", "auto_yes", is_flag=True,
              help="answer that question yes in advance.")
@_bundle_option()
def scratch_cmd(bundle: str, empty: bool, auto_yes: bool) -> None:
    """The engine's scratch a calculation's runs hold -- the files the
    engines' vocabularies name scratch (TranSIESTA's leads' Green's
    functions, `*.TSGFL` / `*.TSGFR`): recomputed by every run, never
    carried, never saved by a checkpoint.  A run's walk empties them as it
    ends; a run that stopped first, or one prepared before 2.22.0, leaves
    them behind.  Listed with each file's stage and size; ``--empty``
    removes them from every run not queued or running, after one question
    (job-system.md § 5.3)."""
    from ..units import size_words
    from .ask import confirm
    from .commands import command
    from .runstatus import scratch_answer
    calc = _calc_of(bundle)
    base = Path(calc.base)
    answer = scratch_answer(calc)
    if not answer.found:
        click.echo(f"no engine scratch in {base}.")
        return
    click.echo("the engine's scratch its runs hold (never saved, "
               "checkpointing.md S1):")
    for f in answer.found:
        click.echo(f"  {f.stage or '-':<14} {size_words(f.size):>10}  "
                   f"{f.path.relative_to(base)}")
    click.echo(f"  {len(answer.found)} file(s), "
               f"{size_words(sum(f.size for f in answer.found))}")
    # A RUN GOING KEEPS ITS SCRATCH, each said with why (`scratch_answer`).
    for f, why in answer.kept:
        click.echo(f"  kept: {f.path.relative_to(base)} -- {why}")
    if not empty:
        if answer.free:
            click.echo("to remove it from every run not queued or running:"
                       "\n    " + command("scratch", base=base,
                                         flags=("--empty",)))
        return
    if not answer.free:
        click.echo("nothing to remove.")
        return
    go = answer.free
    size = sum(f.size for f in go)
    said = confirm(f"about to remove {len(go)} file(s), {size_words(size)}:",
                   auto_yes=auto_yes, question="remove them?", default=False,
                   unanswered="nothing removed")
    _ledger(base, "scratch", "question",
            about="remove the engine's scratch", answer=said.words)
    if not said.asked:
        raise click.ClickException(
            "nobody to ask, and no --yes: nothing was removed.")
    if not said.yes:
        click.echo("nothing was removed.")
        return
    # A FILE GONE SINCE IT WAS LISTED is passed over, and what was removed
    # is what is said.
    gone, freed = [], 0
    for f in go:
        try:
            f.path.unlink()
        except FileNotFoundError:
            continue
        gone.append(str(f.path.relative_to(base)))
        freed += f.size
    _ledger(base, "scratch", "emptied", bytes=freed, files=gone)
    click.echo(f"removed {len(gone)} file(s), {size_words(freed)}.")


@jobset_group.command("machines",
                     short_help="list the machine records prep can target")
def cmd_machines() -> None:
    """Which machines a calculation can be prepared FOR, and where each lives.

    The answer to *"did the record I copied over actually arrive, and does it
    parse?"* (`preparing-for-another-machine.md` § 5).

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
        # THE NAME IS THE PERSON'S, so it is asked for in words -- a
        # `<name>` in a command is a redirect to bash.
        click.echo("No named targets yet.  To prepare for another machine, "
                   "run `molbuilder jobset probe --write --name` on THAT "
                   "machine with the name you will prep it by (`--target`), "
                   "then copy the file it writes into:\n    "
                   f"{environments_dir()}/")
    elif choice_required(machines):
        # NAME THE LOCAL SPELLING HERE TOO, so the reader has a name to type
        # for the box in front of them (`record.LOCAL_TARGET`).
        from ..scheduler.record import LOCAL_TARGET
        # ONLY BEFORE THE FIRST PREP: a calculation that holds its snapshot
        # has its answer, and is asked nothing (`record.machine_for`'s C1).
        click.echo("More than one machine could be meant, so a calculation's "
                   "first `prep` asks which, with `--target` (being asked "
                   "costs one flag; being given the wrong one costs a queue "
                   "wait); once prepared, it keeps the record it took.")
        click.echo(f"    --target {LOCAL_TARGET}   # this machine")


def _scheduler_kinds():
    """The kinds ``--scheduler`` may name: `scheduler/kinds.toml`'s."""
    from ..scheduler.kinds import names
    return names()


@jobset_group.command("probe",
                     short_help="record a machine's capability -> "
                                "environment.json (--name for a cluster you "
                                "prep FOR)")
@click.option("--write", "do_write", is_flag=True, default=False,
              help="write the probed record (shows a diff + confirms).")
@click.option("--name", default=None, metavar="NAME",
              help="write THIS machine's record as environments/NAME.json, "
                   "the name another machine preps it by.  The probe always "
                   "measures the machine it runs on: run `probe --write "
                   "--name sol` on Sol's login node, copy the file it writes "
                   "to the directory `jobset machines` prints, and `prep "
                   "--target sol` sizes for Sol from there.  Without it the "
                   "record is this machine's own environment.json.")
@click.option("--yes", is_flag=True, default=False,
              help="with --write: take every probed value without asking "
                   "(scripts).  Without it, each difference against an "
                   "existing record is asked about, and silence keeps the "
                   "record.")
@click.option("--set", "sets", multiple=True, metavar="KEY=VALUE",
              help="declare a fact the probe cannot see where it runs "
                   "(M-1's declared door): a topology fact -- e.g. a login "
                   "node whose compute nodes hold the GPUs: --set "
                   "gpus_per_node=4 -- or where one of the scheduler's "
                   "commands is, for a site's own wrapper: --set "
                   "scheduler.submit=PATH.  Repeatable; wins over "
                   "detection; the record's source says 'flag'.")
@click.option("--scheduler", "scheduler_flag", default=None,
              type=click.Choice(list(_scheduler_kinds())),
              help="name the scheduler's kind instead of detecting it "
                   "(source 'flag'); its commands are still looked for on "
                   "PATH (scheduler/kinds.toml).")
def cmd_probe(do_write: bool, name, yes: bool,
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
    from anywhere.  **Without it, a bench prepared on a laptop is measured
    against the laptop's cores and queues, silently** -- which is the whole
    reason named records exist.

    The unnamed form writes ``environment.json`` at the machine scope, so one
    probe serves every calculation here (`configuration.md` § 5).

    **Facts only** -- M-1: a probe never chooses on your behalf.  Which queue
    a job uses is that job's own statement (`allocation.domain`, --domain).
    What the probe cannot see is declared to it: --set, --scheduler.  How a
    shell enters an environment here is copied from this machine's
    molbuilder.json (`env_init`), into whichever record it writes.

    Run it on a login node for a cluster; on a workstation it records the same
    shape with no domains (M-2), rather than refusing.
    """
    import getpass
    from datetime import datetime, timezone

    from ..scheduler import (machine_scope_path, read_environment,
                             resolve_environment, write_environment)

    user = getpass.getuser()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # The DECLARED half of M-1, typed by the schema itself: a fact the probe
    # cannot see where it runs (the ordinary case: a login node whose compute
    # nodes hold the GPUs) arrives by flag and wins over detection, and the
    # record's source says so.  An unknown key or a mistyped value is
    # refused by name -- a silently-dropped declaration is a decision the
    # user wrote down and nobody obeyed.
    from ..scheduler import topology_field_types
    from ..scheduler.kinds import ROLES
    types = topology_field_types()
    overrides, commands = {}, {}
    for s in sets:
        key, sep, raw = s.partition("=")
        if not sep:
            raise click.ClickException(
                f"--set takes KEY=VALUE, got {s!r}")
        if key.startswith("scheduler."):
            # WHERE ONE OF THE SCHEDULER'S COMMANDS IS (M-2): a site's own
            # wrapper, or a command not on PATH where the probe runs.
            role = key[len("scheduler."):]
            if role not in ROLES:
                raise click.ClickException(
                    f"--set knows no scheduler command {role!r} -- it knows "
                    f"{', '.join(f'scheduler.{r}' for r in ROLES)}")
            if not raw.strip():
                raise click.ClickException(f"--set {key}: names no command")
            commands[role] = raw.strip()
            continue
        if key not in types:
            raise click.ClickException(
                f"--set knows no topology field {key!r} -- it knows "
                f"{', '.join(sorted(types))}, and "
                f"{', '.join(f'scheduler.{r}' for r in ROLES)}")
        try:
            overrides[key] = types[key](raw)
        except ValueError:
            raise click.ClickException(
                f"--set {key}: {raw!r} is not {types[key].__name__}")
    from ..scheduler.record import LOCAL_TARGET
    if name == LOCAL_TARGET:
        # RESERVED: `--target this` means the box you are on, so a record by
        # that name would make the flag ambiguous -- and it is the one name
        # whose meaning nothing can override.  Refused with what else was
        # typed, before anything is probed.
        raise click.ClickException(
            f"{LOCAL_TARGET!r} is reserved: `--target {LOCAL_TARGET}` "
            f"already means this machine, so a record called that could "
            f"never be prepared for.  Give it the machine's own name (`--name "
            f"sol`); this machine's own record needs no --name at all.")

    # HOW A SHELL ENTERS AN ENVIRONMENT HERE IS REQUIRED, stated in this
    # machine's molbuilder.json (`configuration.md` § 4; user, 2026-10-03:
    # "error when no env_init is present. this is required explicitly").
    # Every record carries the copy and every prep for the machine reads it,
    # so a record is never written without one -- nor keeps one from before.
    from ..runtime_config import get_env_init, machine_config_path
    declared = get_env_init()
    if do_write and not declared.get("activation"):
        raise click.ClickException(
            f"this machine's molbuilder.json ({machine_config_path()}) states "
            f"no `env_init.activation` -- how a shell enters an environment "
            f"here, which every record this probe writes carries and every "
            f"prep for this machine reads (configuration.md § 4).  It is "
            f"required: `molbuilder envs init-config` asks for it, or state "
            f"it in that file; then probe again.")

    # The NODE probe first -- scheduler, topology, default partition: the one
    # node prober, the one `envs init-config` seeds this machine's record
    # through too, so the two cannot disagree about what this machine is.
    try:
        env = resolve_environment(now_iso=now, overrides=overrides or None,
                                  scheduler_override=scheduler_flag,
                                  commands=commands)
    except ValueError as exc:
        # A SCHEDULER THAT CANNOT BE RECORDED -- a kind with a queue whose
        # submit command is neither found nor declared, or commands named
        # for a kind with none (`record.Scheduler`).
        raise click.ClickException(str(exc)) from None

    # THE FOUR FACTS THAT TRAVEL WITH A RECORD -- how this machine enters
    # its environment, which envs exist here, what they were built for, and
    # whether its manager's `run` enters one.  `diagnostics.local_facts`
    # owns them and states why.  The first is THIS machine's
    # `env_init`, declared in its molbuilder.json and copied into whichever
    # record this writes, this machine's or a named one (user, 2026-10-02) --
    # required, and checked above.
    from ..diagnostics import local_facts as _local_facts
    env, facts_notes = _local_facts(env, declared)

    # THE QUEUES -- `record.probe_queues`, the one queue probe; `envs
    # init-config` seeds this machine's record through it too (M-3).
    from ..scheduler.record import probe_queues
    notes, summary = probe_queues(env, user)
    if summary:
        click.echo(summary)

    # A named target is a record ABOUT another machine, kept beside this
    # machine's rather than replacing it (P2): a workstation holds both its own
    # capability and the cluster's, and `prep --target NAME` says which.
    # WHICH FILE, asked for -- not a directory plus a re-spelled name.
    if name:
        from ..scheduler import named_environment_path
        record = named_environment_path(name)
    else:
        record = machine_scope_path()
    target = record.parent
    fname = record.name
    # THE COPY IS WHOLE (user, 2026-10-02: "simply a copy"): a preamble
    # removed from molbuilder.json leaves the record too, asked about like
    # every other difference.
    before = read_environment(target / fname)

    t = env.topology
    click.echo(f"\nMachine: scheduler={env.scheduler.said()}"
               f"  cores/socket={t.cores_per_socket}  sockets={t.sockets}"
               f"  gpus/node={t.gpus_per_node}  gpu={t.gpu_type or '-'}"
               f"  mem={t.mem_total_gb or '-'} GB")
    if env.domains:
        click.echo("\nReachable domains (a job names one: its "
                   "allocation.domain, or prep --domain):")
        for d in env.domains:
            click.echo(f"  {d.name:<10} <= {str(d.max_time):<12} "
                       f"{d.partition}/{d.qos or '-'}")
    # AFTER the queue probe, whose notes this extends (`record.probe_queues`
    # builds them after `derive_domains`, which reassigns its list).
    notes.extend(facts_notes)
    if notes:
        click.echo("\nNotes / assumptions (read before --write):")
        for n in notes:
            click.echo(f"  - {n}")

    if not do_write:
        # THE LINE THAT WRITES WHAT WAS SHOWN, its flags as typed
        # (`job-system.md` § 5.3).
        import shlex
        from ..scheduler.record import probe_command
        again = (probe_command(name)
                 + "".join(f" --set {shlex.quote(s)}" for s in sets)
                 + (f" --scheduler {scheduler_flag}" if scheduler_flag else "")
                 + (" --yes" if yes else ""))
        click.echo(f"\n(dry run -- nothing written.  To record this in "
                   f"{target / fname}:)\n    {again}")
        return

    if before is None:
        # Nothing to clobber: one consent creates the record -- unless a file
        # IS there and does not read, which is said, never treated as absent
        # (`read_environment` answers both with `None`; `--yes` skips the
        # question, never this line).
        dest = target / fname
        if dest.exists() and not dest.is_file():
            # NOT A FILE AT ALL -- a directory: a record cannot replace it,
            # and the write would end in a traceback (the fix-6 review).
            raise click.ClickException(
                f"{dest} is there and is not a file -- move it aside, and "
                f"probe again.")
        there = dest.exists()
        if there:
            from ..scheduler.record import why_unread
            click.echo(f"\n{dest} is there and does not read -- "
                       f"{why_unread(dest)} -- writing replaces it.")
        if not yes:
            try:
                ok = click.confirm("Replace it with this record?" if there
                                   else f"Write this record to {dest}?",
                                   default=False)
            except click.exceptions.Abort:
                click.echo("\n  no answer -- nothing written "
                           "(silence is no).")
                return
            if not ok:
                click.echo("  nothing written.")
                return
    else:
        env = _probe_consent_merge(before, env, yes=yes)
    # 0700, through the one creator: `jobset probe --write` is the FIRST
    # command the seeded `environments/README` tells a person to run on a
    # target, so it may create the config directory every later secret lands in.
    from ..config_dir import ensure_private_dir
    ensure_private_dir(target)
    path = write_environment(env, target / fname)
    if name:
        click.echo(f"wrote {path}\n"
                   f"  prep for that machine with `--target {name}`, "
                   f"wherever this record is copied.")
    else:
        click.echo(f"wrote {path}\n"
                   f"  `prep` snapshots it into each calculation; what you "
                   f"WANT from this machine stays in molbuilder.json.")
