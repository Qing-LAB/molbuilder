"""``molbuilder jobset ...`` — the calculation's one grammar
(docs/execution/job-system.md § 5.3).

``init`` writes the portable folder (floor 2); on the machine that runs
it, ``prep`` derives floor 3 and everything below (the five steps of
project-layout.md § 2.3.1), ``launch`` launches ONE stage's run -- or a
benchmark's trials, one job per resource shelf to the queue, each in turn
here -- and ``summarize`` reads a sweep's results back.  Nothing is produced on a host
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
from typing import Optional

import click

from .ledger import record as _ledger
from .model import JobSet
from .errors import PrepError
from .submit import submit_jobset, SubmitError
from .runstatus import jobset_status, render_stage_status, render_status

#: A11: the name comes from the module that writes the file.
from .model import FILENAME as _JOBSET_FILE


def _load(bundle: str) -> tuple:
    """Load ``<bundle>/job-set.json`` -> (JobSet, bundle Path).

    **When it is absent the refusal reads the folder first**, because the
    same absence means three different things and only one of them is
    "nothing has happened here": a sweep keeps its set in the bench
    container rather than the root, a described calculation is simply not
    prepped yet, and a bare directory has not been described at all.  Shared
    by the verbs that read a job set -- `launch`, a hand-built sweep's bench
    verbs, `status` where no description stands -- so what it says is about
    the STATE it found and not about the verb that arrived.
    """
    base = Path(bundle)
    jpath = base / _JOBSET_FILE
    if not jpath.is_file():
        # WHAT IS HERE DECIDES WHAT TO SAY.  This answered "nothing to do ...
        # run `jobset init` first" for every absence, which is wrong twice on
        # a calculation that HAS been prepped: a sweep's set lives in its
        # bench container, one directory down, so `status` reported nothing
        # while `summarize` could read the whole thing -- and `init` is not
        # the verb for a folder that is already described.  A refusal that
        # names the wrong verb costs more than one that names none.
        from ..task import FILENAME as _TASK_FILE
        described = (base / _TASK_FILE).is_file()
        # WHERE A SWEEP'S SET LIVES IS ASKED FOR, not spelled here.
        # `materialize.sweep_set_paths` is the search counterpart of
        # `bench_container`, and lives beside it so the layout has one home.
        # This walked the whole tree with `rglob` -- every attempt directory
        # full of engine output -- to find files that can only be in two
        # places; spelling those two places HERE would have been a third copy
        # of the layout rule, which is the habit, not the fix.
        from .materialize import sweep_set_paths
        found = sweep_set_paths(base)
        # AND IT IS READ BEFORE IT IS NAMED.  Calling every set below the root
        # "a sweep" is the message claiming what it has not checked; `kind` is
        # in the file.  One that will not load is left out entirely rather
        # than described as something it might not be.
        sweeps = []
        for _p in found:
            try:
                if JobSet.load(_p).kind == "sweep":
                    sweeps.append(str(_p.parent.relative_to(base)))
            except Exception:          # unreadable: not evidence of anything
                pass
        from .commands import bundle_flag, command
        if described:
            # THE EXACT COMMAND, not a placeholder (user, 2026-08-20: a
            # detected problem carries the invocation that repairs it).  The
            # rung is in hand -- the description is right there -- so naming
            # `<stage>` would be this refusal declining to read a file it has
            # already found.  A benchmark prepped here is named beside it --
            # it is not the run (W52: the run was refused with a note about
            # the sweep that named no way to prep the run).  A description
            # that does not read leaves the stage out: `prep` then says what
            # is wrong with it, and never a `<stage>` nobody can type.
            _rung = None
            try:
                from ..task import read_task as _read_task
                _t = _read_task(base / _TASK_FILE)
                _rung = next((st.name for st in (_t.stages or ())
                              if st.enabled is not False), None)
            except Exception:      # a description mid-edit is its own error
                pass
            bench = (f"\nA benchmark is prepped in "
                     f"{', '.join(sweeps[:3])}{'...' if len(sweeps) > 3 else ''}"
                     f" -- its own verbs are `launch bench` and `summarize "
                     f"bench`." if sweeps else "")
            raise click.ClickException(
                f"no {_JOBSET_FILE} in {base}: this calculation is described "
                f"but no run is prepped yet.  `prep` derives the set from "
                f"the description (job-system.md § 4):\n"
                f"    {command('prep', 'run', _rung, base=base)}" + bench)
        if sweeps:
            where = ", ".join(sweeps[:3]) + ("..." if len(sweeps) > 3 else "")
            raise click.ClickException(
                f"no {_JOBSET_FILE} at the root of {base}, and no "
                f"{_TASK_FILE}: a hand-built sweep is prepped in {where}, "
                f"and its verbs are `launch bench` and `summarize bench` -- "
                f"which read a sweep through its description, so a "
                f"description-less one is launched and read by hand.")
        # INSIDE A CALCULATION -- one of its stage or attempt folders -- is
        # not "nothing here": the folder says which calculation it belongs
        # to (`calcdirs.root_of`, `project-layout.md` § 1.4a), and that is
        # where the verb works (W52: it was told to run `init`, which would
        # describe a new calculation inside an attempt).
        from .. import calcdirs
        root = calcdirs.root_of(base)
        if root is not None and Path(root).resolve() != base.resolve():
            raise click.ClickException(
                f"{base} is a folder of the calculation at {root}; the "
                f"verbs work on the calculation -- name it with "
                f"`--bundle`:{bundle_flag(root) or ' (run it from there)'}")
        raise click.ClickException(
            f"no {_JOBSET_FILE} in {base} -- nothing to do.  The host "
            "describes it and `prep` derives it (job-system.md § 4); run "
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
    #
    # THROUGH THE ONE FUNCTION.  This loop was where the pair was FOUND, and
    # `machine_config_warnings` was extracted from it -- and then this site
    # kept calling the two halves itself, so the placement findings added to
    # that function later reached `serve` and not the jobset verbs (D6).
    for warning in machine_config_warnings():
        _click.echo(warning, err=True)
    from ..projects import projects_root_with_source
    _click.echo(projects_root_with_source().describe())


#: The calculation folder, spelled the same way on every verb.
#:
#: It was a POSITIONAL on ``plan`` and ``status`` until 2026-08-10 while
#: ``prep``/``launch`` took ``--bundle``, so one word meant the folder on two
#: verbs and the stage on the other two.  `job-system.md` § 5.3 calls that *"a
#: defect of this section's own making"*: the grammar is
#: ``jobset <verb> <kind> [<stage>]``, and a positional that is sometimes a
#: path has no place in it.  `jobset status tight` answered *"Directory 'tight'
#: does not exist"*, which tells a user they mistyped a path they never meant.
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
                    stage_strategy) -> None:
    """`init --calculation transport` -- floor 2 is task.json ALONE.

    The structure, the pseudos and the electronic template all arrive at
    prep from the junction citation (archive/2026-09-01-transport-design.md 4.1, Q5:
    one template governs everything, physically enforced by deriving) --
    so every option that would supply them here is refused naming that
    rule rather than silently ignored.
    """
    from pathlib import Path as _P

    from ..task import Stage, Task, derive_run, write_task
    from ..task import FILENAME as TASK_FILENAME
    # The ladder is the transport module's fact (one door): five stages
    # in dependency order, per-stage `enabled` for the seed's Q4 skip.
    from ..transport.stages import TRANSPORT_STAGES

    for given, flag, why in (
            (structure, "--structure", "its structure IS the junction "
             "citation, copied in at prep"),
            (psml_lib, "--psml-lib", "the pseudopotentials travel with "
             "the cited junction"),
            (vacuum, "--vacuum", "the cell comes with the cited junction"),
            (stage_strategy, "--stage-strategy", "the composite's five "
             "stages are fixed by design (seed, electrode_L, electrode_R, "
             "device, transmission)")):
        if given is not None and given != ():
            raise click.ClickException(
                f"{flag} does not apply to a transport calculation -- "
                f"{why} (archive/2026-09-01-transport-design.md 4.1).")
    if engine != "siesta":
        raise click.ClickException(
            f"transport is SIESTA-first (TranSIESTA); engine {engine!r} "
            f"has no transport ladder yet.")
    # shape: the codec owns the transport==hierarchical pairing (task.py
    # __post_init__); the Task construction below surfaces it by name.

    slots = {}
    for entry in slots_opt:
        name_, sep, cite = entry.partition("=")
        if not sep or not name_ or not cite:
            raise click.ClickException(
                f"--slot {entry!r}: spell it NAME=DIRECTORY, e.g. "
                f"--slot junction=<project>/<topic>/<calc>/<stage>/run-N "
                f"-- any directory whose files satisfy the citation "
                f"condition (archive/2026-09-01-transport-design.md 4.1b).")
        slots[name_] = cite
    if set(slots) != {"junction"}:
        raise click.ClickException(
            "a transport calculation takes exactly one slot, `junction` "
            "-- the directory holding the relaxed junction it composes "
            "from: a finished relaxation's .fdf+.XV together, or a "
            "labeled .xyz+.molstruct.json pair "
            "(archive/2026-09-01-transport-design.md 4.1b).")

    # The cited directory goes through the SAME tree fence every
    # calculation path uses (2.5b) and is classified against the 4.1b
    # FILE condition right here -- a citation that cannot compose is
    # refused at init, naming the missing file (strict composition, Q2).
    resolved = _resolve_bundle(None, None, slots["junction"],
                               must_exist=True)
    from ..projects import projects_root
    citation = str(_P(resolved).relative_to(projects_root()))
    from ..transport.compose import ComposeError, classify_citation
    from .. import template as _T
    try:
        classify_citation(_P(resolved))
    except ComposeError as exc:
        raise click.ClickException(str(exc))

    bias = ()
    if bias_opt:
        try:
            bias = tuple(float(v) for v in bias_opt.split(","))
        except ValueError:
            raise click.ClickException(
                f"--bias {bias_opt!r}: a comma-separated list of volts, "
                f"e.g. --bias 0.0,0.2,0.4.")

    stages = tuple(Stage(name=n, enabled=True, overrides={})
                   for n in TRANSPORT_STAGES)
    try:
        task = Task(
            engine=engine, shape=shape,
            run=derive_run(run_name, citation,
                           stage_names=TRANSPORT_STAGES),
            structure=None, calculation="transport",
            slots={"junction": citation}, bias=bias,
            varies=(), stages=stages)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # THE TEMPLATE'S TEXT FIRST -- before the folder exists -- so a
    # citation it refuses (one that carried a net charge, ES7,
    # `science/chemistry-correctness.md` § 2a) leaves nothing behind: an
    # empty folder was left until the M6 review.
    from ..transport.citation_defaults import transport_template_text
    try:
        _tmpl_text = transport_template_text(_P(resolved), label=task.label)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    # THE DESCRIPTION'S OWN CHECK (gate ③), which every describe runs
    # (`workflow.md` § 9) and this door skipped until 2026-09-30 (plan
    # § 5w K3) -- refusing before the folder exists, and saying its warnings
    # (a bias point outside its recommended range) the way `prep` says its
    # notes.
    from ..issues import ValidationError
    from ..validation.task import preflight, refuse_on_error
    try:
        for issue in refuse_on_error(preflight(task,
                                               template_text=_tmpl_text)):
            click.echo(f"note: {issue.message}", err=True)
    except ValidationError as exc:
        raise click.ClickException(str(exc))
    dest = out_dir if out_dir.is_absolute() else         _P(_resolve_bundle(None, None, str(out_dir), must_exist=False))
    dest.mkdir(parents=True, exist_ok=True)
    write_task(dest / TASK_FILENAME, task)

    # THE TEMPLATE -- transport's shared baseline, like every other kind's
    # (TR1).  A transport folder carried no template at all until
    # 2026-09-16, so the Class A values had nowhere to live and the stage
    # table had nothing to inherit from (`engines/transport.md` § 2a.3).
    #
    # Its values are DEFAULTED FROM THE CITED RUN, not sealed to it
    # (§ 2a.7): the person may change any of them afterwards, and a change
    # applies to every stage at once because there is one template.
    # ...and its TEXT comes through the same door the browser's describe
    # door uses (`citation_defaults.transport_template_text`): a `citation`
    # row the cited directory does not answer stays VALUELESS
    # (`engines/transport.md` § 3.8.3).  This wrote the class default into
    # such rows until 2026-09-24, so a hand-built structure described on
    # the CLI claimed a basis no run had said.
    # THE ONE DOOR that forms this name (`template.template_path`).  Six
    # call sites spelled it by hand, in two incompatible ways, until
    # 2026-08-17; `test_doc_claims` walks the AST to keep it at one, and
    # caught this line the day it was written.
    tmpl = _T.template_path(dest, task.label)
    tmpl.write_text(_tmpl_text, encoding="utf-8")

    click.echo(f"Described transport '{task.run.name}' in {dest} -- "
               f"composes {citation}.")
    click.echo(f"  {TASK_FILENAME}")
    click.echo(f"  {tmpl.name}")
    click.echo("")
    click.echo("The template carries the shared electronic description, "
               "filled in from the run you cited -- change anything in it "
               "and every stage follows, because there is one of it.  The "
               "structure and pseudopotentials still arrive at prep from "
               "the citation.  On the machine that will run it:")
    # THE FIRST COMMAND A PERSON COPIES, as `init` prints it for every other
    # calculation (W52: a `cd` to this host's path, which is not a path on
    # the machine the text names).
    from .commands import command
    from .materialize import described_refs
    click.echo("  " + command("prep", "run", described_refs(dest, task)[0].name,
                              base=dest))


@jobset_group.command("init",
                      short_help="create the calculation (floor 2 only)")
@_bundle_option(must_exist=False)
@click.option("--structure", "structure", default=None, metavar="PATH",
              callback=_resolve_structure,
              help="the structure to describe, as a path from the PROJECTS "
                   "ROOT -- e.g. `--structure <project>/structure/water.xyz`. "
                   "It lives in the tree like everything else a calculation "
                   "cites (job-contracts.md 2.5b).  Required for every "
                   "calculation EXCEPT transport, whose structure IS its "
                   "junction citation (--slot).")
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
              help="where to read pseudopotentials from -- a path INSIDE "
                   "the projects tree, measured from the tree root (the "
                   "convention is `pseudopotential`).  The files travel "
                   "with the calculation; the path does not.")
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
@click.option("--slot", "slots_opt", multiple=True, metavar="NAME=CITATION",
              help="a composite input (transport only): "
                   "--slot junction=<dir> (a directory whose files "
                   "satisfy archive/2026-09-01-transport-design.md 4.1b).  The "
                   "attempt is named explicitly, never picked "
                   "(archive/2026-09-01-transport-design.md, ruling Q1).")
@click.option("--bias", "bias_opt", default=None, metavar="V0,V1,...",
              help="transport only: the bias sweep in volts, starting at "
                   "0.0 -- each point warm-starts from the previous one's "
                   ".TSDE (archive/2026-09-01-transport-design.md 4.3).")
def init_cmd(structure, bundle: str, shape: str,
                 stage_strategy, name, engine: str, psml_lib, vacuum,
                 calculation: str, slots_opt, bias_opt) -> None:
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
    from pathlib import Path as _P

    from ..workingcopy_structure import StructureCodec
    from ..config.siesta import SIESTA_STAGE_NAMES, SiestaConfig
    from ..describe import DescribeError, build_description, write_description
    from ..identity import normalise_id
    from ..chemistry import species_order
    from ..pyscf.stages import default_pyscf_stages
    from ..siesta.stages import default_siesta_stages
    from ..task import Stage

    out_dir = _P(bundle)
    run_name = name or out_dir.name
    # ONE CALCULATION PER FOLDER (user, 2026-10-01: refuse).  A folder that
    # already holds a description is changed in Task setup or its
    # task.json; `init` over it re-described it silently, leaving whatever
    # was prepped there describing a calculation that no longer existed
    # (W52, P1a-12).  Refused before anything is read or written.
    from ..task import FILENAME as _TASK
    if (out_dir / _TASK).is_file():
        raise click.ClickException(
            f"{out_dir} already holds a calculation ({_TASK}) -- one "
            f"calculation per folder.  Change it in Task setup or its "
            f"{_TASK}, or describe the new one in a folder of its own.")

    if calculation == "transport":
        _init_transport(out_dir=out_dir, shape=shape, run_name=run_name,
                        engine=engine, slots_opt=slots_opt,
                        bias_opt=bias_opt, structure=structure,
                        psml_lib=psml_lib, vacuum=vacuum,
                        stage_strategy=stage_strategy)
        return
    if slots_opt or bias_opt:
        raise click.ClickException(
            "--slot / --bias belong to --calculation transport alone "
            "(archive/2026-09-01-transport-design.md 4.1).")
    if structure is None:
        raise click.ClickException(
            "--structure is required -- it is what this calculation "
            "describes (transport is the one kind without it: its "
            "structure IS the junction citation).")

    try:
        # THE PAIR, through the one door.  This called the top-level
        # `load()` until 2026-09-07 -- a reader that predated the
        # sidecar and returned the geometry alone, so a description
        # was born without the regions, the frozen atoms or the cell
        # the author had set in the viewer.
        struct = StructureCodec().load(structure)
        if vacuum is not None:
            # The same channel the Modify -> Cell tab uses: the vacuum lives on
            # the STRUCTURE, not on the engine config, so every surface that
            # reads this structure sees the same isolation.
            # THROUGH THE DOOR.  `dataclasses.replace` does not dispatch to
            # `Structure.replace` on this interpreter, so it re-passed every
            # mutable field BY REFERENCE -- the derived structure shared its
            # `cell` and its `info` dict with the loaded one.
            struct = struct.replace(vacuum=(vacuum, vacuum, vacuum))

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
            # The vibration kind's own ladder, read from the person's one
            # statement (`engines/vibration.md` § 2.2, § 5.2a): a fresh
            # description carries the catalogue's own value of the box --
            # unticked -- so SIESTA gets `relax` then `freq`, PySCF `freq`
            # alone.  A --stage-strategy names the optimization ladder's
            # tiers, which grade nothing here: the convergence settings are
            # the template's, recommended tight (`template.md` § 6.3a).
            from ..pyscf.stages import VIBRATION_ENGINES, vibration_stages
            from ..template import catalogue as _catalogue, one as _one
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
            _box = _one(_catalogue(), "already_relaxed", engine=engine)
            stages = tuple(vibration_stages(
                engine, already_relaxed=bool(_box.default if _box else False)))
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
    # THE FIRST COMMAND A PERSON COPIES, real (W52: it printed `prep run
    # <stage>`, which bash refuses, beside a `cd` to this host's path).
    from .commands import command
    from .materialize import described_refs
    first = described_refs(out_dir, desc.task)[0].name
    click.echo(
        "\nIt names no machine. On the machine that will run it:\n  "
        + command("prep", "run", first, base=out_dir), err=True)


@jobset_group.command("status", short_help="show per-stage status + resume point")
@click.argument("stage", required=False, default=None)
@_bundle_option()
def status_cmd(stage, bundle: str) -> None:
    """Show every stage of the description -- its run state (finished /
    running / failed / queued / pending / not-started), which warm-restart
    files are present -- and the FIRST incomplete stage (the one to resume
    from).  The stages are listed from the moment `init` writes them, the
    ones not prepped yet among them (job-system.md § 5.3).  Read-only --
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
    from ..task import FILENAME as _TASK_FILE
    from .runstatus import stage_continuation
    base = Path(bundle)
    task = None
    if (base / _TASK_FILE).is_file():
        # THE DESCRIPTION'S LADDER (job-system.md § 5.3, 2026-10-01): every
        # stage it names, before the first prep and as prep reaches each.
        jpath = base / _JOBSET_FILE
        try:
            js = JobSet.load(jpath) if jpath.is_file() else None
        except ValueError as e:
            raise click.ClickException(str(e))
        name = _described_stage(base, stage)
        from ..task import read_task
        try:
            task = read_task(base / _TASK_FILE)
        except Exception as exc:                        # noqa: BLE001
            raise click.ClickException(f"{_TASK_FILE}: {exc}")
    else:
        js, base = _load(bundle)
        if js.kind == "sweep":
            # A BENCH FOLDER is read against its calculation: its sweep names
            # its trials from there, and what it prints names that (W52).
            from .materialize import bench_owner
            owner = bench_owner(base)
            if owner is not None:
                base = owner[0]
        name = (_resolve_stage_name(js, stage) if stage is not None
                else None)
    try:
        status = jobset_status(js, base)
    except (OSError, ValueError, KeyError) as exc:
        # A refusal, in the verb's voice -- never a traceback (W52).
        raise click.ClickException(str(exc))
    if name is None:
        click.echo(render_status(status))
        return
    row = next(s for s in status.stages if s.name == name)
    click.echo(render_stage_status(
        status, name,
        stage_continuation(base, task, name)
        if task is not None and not row.prepped else None))


# --------------------------------------------------------------------- #
#  prep / launch -- the execution loop (job-system.md § 5.3)             #
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
    bench`` enumerates the grid on this machine and ``launch bench <stage>``
    launches its trials -- or the one named -- through the same resolver as
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


def _described_stage(base, stage):
    """The stage a verb was given, as the description spells it, through
    the ONE resolver -- its name in any case, or ``#N``
    (`identity.resolve_stage_ref`, `job-system.md` § 5.3) -- and what the
    verb then finds, records and PRINTS (a pasted ``#2`` would be a comment
    in bash).  Asked by the verbs that name a stage the description holds
    whether or not it is prepped: the bench verbs, which matched the exact
    name in a lookup of their own until K12, so `launch bench '#2'` refused
    the stage `prep bench '#2'` had just prepared (plan § 5w K12), and
    `status`.  ``None`` stays ``None``, and a folder with no description has
    no ladder to resolve against: its name stands."""
    from ..identity import StageRef, resolve_stage_ref
    from ..task import FILENAME, read_task
    desc = Path(base) / FILENAME
    if stage is None or not desc.is_file():
        return stage
    try:
        from .materialize import ladder_homes
        return resolve_stage_ref(
            [StageRef(h.seq, h.name)
             for h in ladder_homes(base, read_task(desc))], stage).name
    except ValueError as e:
        raise click.ClickException(str(e))


def _refuse_disabled(base, stage) -> None:
    """A stage the description disables is never launched -- a folder it
    left from before is kept as it is (`task.stage_disabled`; user,
    2026-10-03, Q1: "never allow use")."""
    from ..task import FILENAME, read_task, stage_disabled
    desc = Path(base) / FILENAME
    if stage is None or not desc.is_file():
        return
    why = stage_disabled(read_task(desc), stage)
    if why:
        raise click.ClickException(why)


def _refuse_unprepped(base, stage) -> None:
    """A stage the description holds and no prep has prepared is refused by
    NAME, with its prep and its launch -- what `status` says of it.  It read
    *no stage named ... in this job-set* (W55 D4): the job-set holds the
    prepped stages, and the description all of them.  Prepped is the prep
    entry's own answer (`prep.prepped_already`)."""
    from ..task import FILENAME, read_task
    from .commands import block, run_first
    from .prep import prepped_already
    desc = Path(base) / FILENAME
    if stage is None or not desc.is_file():
        return
    if prepped_already(base, read_task(desc), "run", stage) is None:
        raise click.ClickException(
            f"stage {stage!r} is not prepped yet -- prep it, then launch "
            f"it:\n" + block(run_first(stage, base=base)))


def _stage_bench_dir(base, stage, verb: str = "launch"):
    """The stage's bench container (job-contracts.md § 6.3), resolved
    through the description — where its trials, its job-set and its verdict
    all live.  Returns ``(container_path, token)``.  ``stage`` is the
    description's own spelling -- the verb resolved it first
    (:func:`_described_stage`); a bare invocation is refused with the ladder
    listed — § 6.5 gives every description a ladder, so there is always a
    stage to name and never a bare form to fall back to."""
    from ..task import FILENAME, read_task
    from ..paths import bench_container
    from .materialize import stage_home
    from ..paths import Shape
    desc = Path(base) / FILENAME
    if not desc.is_file():
        return None, None                    # hand-built set: no container
    task = read_task(desc)
    sh = Shape.named(task.shape)
    # A CALCULATION THAT HAS NO BENCHMARK says so first -- `prep bench`'s own
    # answer (`bench_refusal`) -- or the command offered next is refused in
    # turn (W52: `launch bench` on PySCF offered `prep bench`).
    from .prep_inputs import bench_refusal
    why = bench_refusal(task)
    if why:
        raise click.ClickException(why)
    if stage is None:
        # THE STAGES WITH A BENCHMARK TO ACT ON -- a prepped one -- or, with
        # none, the prep that makes one (W52: every stage was offered, the
        # disabled and the never-benched, and the first was then refused).
        from .commands import command, name_a_stage
        from .materialize import described_refs
        refs = described_refs(base, task)
        benched = [r for r in refs if (Path(base) / bench_container(
            sh, stage_home(base, task, r.name).token) / _JOBSET_FILE).is_file()]
        if not benched:
            raise click.ClickException(
                "no stage has a prepped benchmark yet -- prep one first:\n"
                "    " + command("prep", "bench", refs[0].name, base=base))
        raise click.ClickException(
            "which stage's benchmark? "
            + name_a_stage(verb, "bench", benched, base=base))
    token = stage_home(base, task, stage).token
    return Path(base) / bench_container(sh, token), token


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


def _load_bench_set(base, stage, verb: str = "launch"):
    """The stage's OWN sweep record, from its container — or the root
    job-set for a hand-built (description-less) sweep."""
    container, _ = _stage_bench_dir(base, stage, verb)
    if container is None:
        return _load(str(base))              # legacy/hand-built library sets
    jpath = container / _JOBSET_FILE
    if not jpath.is_file():
        from .commands import command
        raise click.ClickException(
            f"no {jpath.relative_to(Path(base))} -- this stage has no "
            f"prepped benchmark.  Prep it first:\n    "
            + command("prep", "bench", stage, base=base))
    try:
        return JobSet.load(jpath), Path(base)
    except ValueError as e:
        raise click.ClickException(str(e))


# `_ask_if_underway` stood here until 2026-09-29, and its successors
# `prep.underway_evidence` and `_ask_underway` until 2026-10-02: a prepped
# stage is refused now.  `_ask_save`, the question that followed, went
# 2026-10-03: prep saves the folder's state always, and says so.


# `_refuse_if_measured_elsewhere` and `_measured_on` stood here until
# 2026-09-04.  They read `bench-result.json` to refuse applying a verdict
# measured on a different machine kind (`submission.md` S3).  Nothing in
# production ever called them -- the only caller was a test -- because the
# premise died on 2026-09-02: see step 2 of `prep_inputs.prep_run_inputs`,
# THERE IS NO SECOND RUNG.  No verdict reaches a launch on its own any more, so there
# is no boundary left to cross, and a guard against a route that does not
# exist is a guard nobody can trip.


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
             "number of minutes.  A run on a scheduler states one -- here, "
             "or `allocation.time` / the run card's `time` in task.json -- "
             "or prep refuses; on launch it overrides what prep baked.  "
             "Never derived, never estimated (architecture.md § 5.2).")
MEM_METAVAR = "SIZE"
MEM_HELP = ("how much TOTAL memory this needs -- `128G`, `80GB`, `0.5T`, or "
            "a bare number of GB.  `0` asks for all of the node's.  A run "
            "on a scheduler states one -- here, or `allocation.mem` in "
            "task.json -- or prep refuses; on launch it overrides what prep "
            "baked (running-a-job.md § 5.3.1).")


def _duration(text):
    """`scheduler.quantities.parse_duration`, refusing in click's voice."""
    from ..scheduler.quantities import parse_duration
    try:
        return parse_duration(text)
    except ValueError as e:
        raise click.ClickException(f"--time: {e}")


def _memory(text):
    """A stated memory as the record writes it -- SLURM's spelling, ``0``
    for all of the node's (`scheduler.quantities.canonical_mem`) -- refusing
    in click's voice.  ``launch`` read it as a number of gigabytes until
    2026-10-01 and so refused the ``0`` this help advertises (W52)."""
    from ..scheduler.quantities import canonical_mem
    try:
        return canonical_mem(text)
    except ValueError as e:
        raise click.ClickException(f"--mem: {e}")


# THE ASSEMBLY a prep receives -- `_declared_execution_pins` through
# `bench_inputs`, the bench grid's cell checks and the run's condition --
# moved to `jobset/prep_inputs.py` on 2026-09-29: the conductor's own
# assembly, beside it (`architecture.md` § 2.1's note, A7, A12), so the one
# prep entry can call it and the Task setup tab no longer reaches across to
# this module.


def _resolve_stage_name(js, stage: str) -> str:
    """The job ``stage`` names, through the ONE resolver (§ 8f).

    Split out from :func:`_resolve_stage` because two different questions were
    living in one function: *which job did the user name* (every verb that takes
    a STAGE asks this) and *may this verb act on the whole set* (only
    ``launch`` asks -- `prep` takes its stage through the prep entry -- and
    ``status`` legitimately may). Keeping them together
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


def _resolve_stage(js, stage, verb: str, *, base):
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
    from .commands import name_a_stage
    from .materialize import stage_refs
    if stage is not None:
        return _resolve_stage_name(js, stage)
    refs = stage_refs(js)
    ordered = [refs[j.name] for j in js.jobs]
    if js.kind == "ladder":
        raise click.ClickException(
            # WHAT YOU CAN TYPE, never the token (`job-system.md` § 5.3):
            # `01_coarse` listed here was refused when typed back, until K12.
            f"this is a ladder, so `{verb} run` acts on ONE stage; "
            + name_a_stage(verb, "run", ordered, base=base) + "\n"
            "Stages do not chain, and there is no flag that makes them: a "
            "run that continues on its own can spend a week refining a "
            "geometry you would have rejected in a minute "
            "(project-layout.md § 1.6).")
    return None


@jobset_group.command("prep", short_help="set a stage up to run")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("stage", required=False, default=None)
@_bundle_option()
@click.option("--from", "from_attempt", default=None,
              metavar="NN_STAGE/run-N",
              help="the attempt this run continues from, by its folder -- "
                   "e.g. '01_coarse/run-0'.  Its warm files are COPIED in.  "
                   "Without it, a continuing stage takes the newest attempt "
                   "of the stage before it, which must have concluded "
                   "(job-system.md 5.4).  The flat layout and a bias scan "
                   "have no single attempt to name.")
@click.option("--cold", is_flag=True,
              help="start this run from the calculation's structure -- "
                   "nothing is copied in.  On the flat layout a stage starts "
                   "clean by its run card's `restart: clean` instead.")
@click.option("--env", default=None,
              help="force one conda env for every job (default: each job's "
                   "env by what it asks for -- the GPU env for a GPU job, "
                   "the CPU env otherwise -- which a mixed CPU/GPU ladder "
                   "needs).")
@click.option("--np", "mpi_np", type=int, default=None, metavar="N",
              help="MPI ranks for this prep -- the launch shape, which "
                   "task.json's `execution` states for every prep "
                   "(project-layout.md D2); prep renders the deck for it.  "
                   "Stated nowhere, a SIESTA run is refused.")
@click.option("--cpus-per-task", type=int, default=None, metavar="C",
              help="cores per rank (OMP threads). sbatch -c.  Stated "
                   "nowhere -- here or the run card's `omp_threads` / "
                   "`threads` -- a run is refused.")
@click.option("--gpus", "gres", default=None, metavar="N",
              help="how many GPUs -- a count; which card a node carries is "
                   "the machine's business (scheduler.md R2a).")
@click.option("--time", "time_", default=None, metavar=TIME_METAVAR,
              help=TIME_HELP)
@click.option("--mem", default=None, metavar=MEM_METAVAR, help=MEM_HELP)
@click.option("--max-memory-mb", type=int, default=None, metavar="MB",
              help="per-rank cap, baked into the wrapper as ulimit -v.")
@click.option("--domain", default=None, metavar="NAME",
              help="which named domain to run in (a PROBED domain from "
                   "the target's record -- a partition and a QOS together, "
                   "with its own limits).  On a scheduler a run names one "
                   "-- here, or `allocation.domain` in task.json -- or prep "
                   "refuses and lists them.")
@click.option("--target", default=None, metavar="NAME",
              help="which MACHINE this is for -- a record written by "
                   "`jobset probe --write --name NAME`, or `this` for this "
                   "machine's own.  Omit when there is one; naming it is how "
                   "a bench prepped on a workstation measures the cluster "
                   "instead of the desk.")
@click.option("--sbatch/--no-sbatch", "emit_sbatch", default=True,
              help="emit .sbatch wrappers (default on; withheld where the "
                   "target's record says `workstation` -- job-system.md "
                   "§ 6).  --no-sbatch writes none, and then no queue, wall "
                   "or memory is asked for.")
def prep_cmd(kind: str, stage, bundle: str, from_attempt, cold: bool, env,
             mpi_np, cpus_per_task, gres, time_, mem, max_memory_mb,
             domain, target, emit_sbatch: bool) -> None:
    """Set a stage up to run, and report what was done.

    Renders the deck and its wrappers, makes that stage's next ``run-<n>``,
    copies the deck and the shared package in, and copies in what it
    continues from.  **Prep printing what it resolved is what makes the launch
    a plain yes** -- it is the only place the chosen geometry and the rendered
    deck appear together.

    A STAGE is required on a ladder — bare ``prep run`` is refused before
    anything is read of the machine or written, offering the stages by name
    and the command for the first (`engines/stages.md` § 6.5; W52).
    """
    # THE ONE ENTRY (`job-system.md` § 5.3, plan W38 F7): the Task setup
    # tab's Prep buttons call it too.  This verb collects what the person
    # said -- the flags -- and prints the answer; the act itself is
    # `prep.prep_stage`'s, and it asks nothing (its one question, *already
    # under way*, was retired on 2026-10-02).
    from ..scheduler.quantities import (canonical_mem, canonical_time,
                                        parse_gres_flag)
    from .model import Resources as _Alloc
    from .prep import prep_stage
    base = Path(bundle).resolve()
    # A SPELLING THAT IS NO AMOUNT is refused in the verb's voice, naming the
    # flag -- through the same readers the record uses (`Resources`), which
    # raised it as a traceback until 2026-10-01 (W52).
    for _flag, _said, _read in (("--time", time_, canonical_time),
                                ("--mem", mem, canonical_mem),
                                ("--gpus", gres,
                                 lambda g: g and parse_gres_flag(g))):
        try:
            _read(_said)
        except ValueError as e:
            raise click.ClickException(f"{_flag}: {e}")
    allocation = _Alloc(mpi_np=mpi_np, cpus_per_task=cpus_per_task,
                        gres=gres, time=time_, mem=mem,
                        max_memory_mb=max_memory_mb, domain=domain)

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
        ans = prep_stage(base, kind, stage, target=target,
                         allocation=allocation, from_attempt=from_attempt,
                         cold=cold, env=env, emit_sbatch=emit_sbatch,
                         on_found=_show)
    except PrepError as e:
        _show(e.findings, e.notes)
        raise click.ClickException(str(e))
    _echo_prep_answer(ans, base)


def _echo_prep_answer(ans, base) -> None:
    """The prep report, from the entry's answer -- `job-system.md` § 5.3:
    *"prep prints what it resolved, which is what makes launch a plain
    yes"*.  Where the configuration came from, what was written, the attempt
    and what it carries, what it will launch with, whether the deck agrees
    -- then the next command, naming the bundle so it works from anywhere
    (job-contracts.md § 2.5b).  The Task setup tab shows the same answer.
    """
    def say_next(line):
        click.echo(line)

    from ..runtime_config import format_provenance
    from .ledger import rel_to as _rel
    if ans.saved:
        # THE STATE A REDO RESTORES, named where the person reads it --
        # saved now, or the one the folder already stood at.
        click.echo(ans.saved)
    if ans.provenance is not None:
        # WHERE the effective config came from (user request 2026-08-12;
        # secrets excluded by design) -- also in the bundle's ledger, since
        # the terminal is gone when a job misbehaves hours later.
        click.echo(format_provenance(ans.provenance))
    stage = ans.stage
    from .commands import block, command, launch_lines
    if ans.kind == "bench":
        where = f" for stage {stage!r}" if stage else ""
        click.echo(f"prepped {len(ans.dirs)} trial dir(s){where} under "
                   f"{base}:")
        for d in ans.dirs:
            # the path from the bundle, not the bare attempt name -- with the
            # attempt layer every trial's dir ENDS in run-<n> (user,
            # 2026-08-28)
            click.echo(f"  {_rel(base, d)}")
        _echo_pipeline_log(ans, base)
        # ONE COMMAND A LINE, prose after `#` (W52: a `(note)` after the
        # command, which bash cannot parse; and the report described as a
        # proposal `prep run` applies -- nothing applies it, the person
        # copies its `execution` block into task.json, job-system.md § 7).
        say_next("next -- launch the sweep (to the queue it goes as one "
                 "job per resource shelf):\n"
                 + block(launch_lines("bench", stage, base=base)))
        say_next("then -- a report to read; its `execution` block is "
                 "yours to copy into task.json:\n"
                 + block([command("summarize", "bench", stage,
                                  base=base)]))
        return
    next_line = "next:\n" + block(launch_lines("run", stage, base=base))
    if ans.flat:
        click.echo(f"prepped {len(ans.dirs)} job dir(s) under {base}  "
                   "(flat: no attempt to open; runs are told apart by "
                   "the wrapper's output index)")
        if ans.continuation is not None:
            click.echo("  " + ans.continuation.line())
        _echo_pipeline_log(ans, base)
        say_next(next_line)
        return
    if ans.points:
        from ..task import bias_token as _bias_token
        for att, v, got in ans.points:
            at = f" @ {_bias_token(v)}" if v is not None else ""
            click.echo(f"prepared {stage}{at}: {Path(att).relative_to(base)}")
            for src, fn in got:
                click.echo(f"  gathered: {fn} <- {src}")
        _echo_pipeline_log(ans, base)
        say_next("next -- one job walks the points in order:\n"
                 + block(launch_lines("run", stage, base=base)))
        return
    rep = ans.attempt
    click.echo(f"prepared {rep.stage}: {rep.dir.relative_to(base)}"
               f"{'' if rep.fresh else '  (reused -- not launched yet)'}")
    click.echo(f"  brought in: {', '.join(rep.brought)}")
    # WHAT IT STARTS FROM, said (`job-system.md` § 5.4) -- one line, read off
    # the answer both doors print: the run it continues from (which, by
    # default or named, what it was, what came across); a cold start asked
    # for; the files a transport rung gathered (said below); or nothing from
    # another run -- a linked stage's too: its kind's first rung, or the
    # structure as given ("its input is prep's own" stood here until
    # 2026-10-05, for rungs that take nothing).
    if ans.continuation is not None:
        click.echo("  " + ans.continuation.line(rep.copied))
    elif ans.cold:
        click.echo("  cold start -- nothing copied in")
    elif not ans.gathered:
        click.echo("  takes nothing from another run"
                   + ("" if ans.linked else
                      " (the first stage, or one that starts clean)"))
    for src, fn in ans.gathered:
        click.echo(f"  gathered: {fn} <- {src}")
    if ans.resources is not None:
        r = ans.resources
        # WHAT IS STATED, and nothing else: no launch value is left for
        # the wrapper to decide (`architecture.md` § 5.2), and a PySCF run
        # has no rank count at all -- `mpi_np auto` claimed one.
        asks = [f"{word} {r[key]}" for word, key in (("mpi_np", "mpi_np"),
                                                     ("omp", "cpus_per_task"))
                if r.get(key)]
        if r.get("continue_retries"):
            asks.append(f"retries {r['continue_retries']}")
        if asks:
            click.echo(f"  resources: {' | '.join(asks)}")
    a = ans.agreement
    if a is not None and a.verdict == "agrees":
        click.echo(f"  {ans.deck}: rendered for mpi_np {a.rendered_text} "
                   f"-- agrees with this launch")
    elif a is not None:
        # The WHY and the remedy are the agreement module's one wording
        # (disagreement_note); this surface adds only its framing -- what
        # will happen next if nothing changes.
        from .agreement import disagreement_note
        click.echo(click.style(
            f"  {ans.deck}: rendered for mpi_np {a.rendered_text}, but this "
            f"launch asks {a.launch_text}\n"
            f"    launch WILL REFUSE this -- " + disagreement_note(a),
            fg="yellow"), err=True)
    _echo_pipeline_log(ans, base)
    say_next(next_line)


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
@click.argument("stage", required=False, default=None)
@_bundle_option()
@click.option("--tolerance-cm1", "tolerance_cm1", type=float, default=None,
              help="a SIESTA vibration's displacement sweep: flag every mode "
                   "whose frequency spreads by more than this across the "
                   "force-constant stages (cm^-1).  Without it nothing is "
                   "flagged -- the numbers are stated and the judgement is "
                   "yours (engines/vibration.md 5.9).")
def summarize_cmd(kind: str, stage, bundle: str,
                  tolerance_cm1: Optional[float]) -> None:
    """Summarize results that exist -- never derive a run's own
    (`job-system.md` § 5, the verb table).  Three summaries, by what was
    described:

    \b
    * ``summarize bench`` reads a benchmark's trials and writes
      ``bench-result.json`` -- a recommendation, not a decision
      (`project-layout.md` § 2.3.2): you read it, you decide;
    * ``summarize run`` on a transport calculation reads its transmission
      points into ``<label>.transport.json`` (`engines/transport.md`
      § 2a.12);
    * ``summarize run`` on a SIESTA vibration with two or more
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
    if kind != "bench":
        # THE TRANSPORT COMPOSITE'S DELIVERABLE (archive/2026-09-01-transport-design.md
        # § 7 P6): `summarize run` on a transport calculation reads the
        # transmission attempts back into <label>.transport.json and
        # prints the I-V table.  Asynchronous like the bench reader: a
        # point that has not run yet reads as pending, never a failure.
        from pathlib import Path as _P

        from ..task import FILENAME as _TASKF
        from ..task import read_task as _rt_sum
        _tt = None
        try:
            _tt = _rt_sum(_P(bundle) / _TASKF)
        except Exception:
            _tt = None
        _is_vibration = _tt is not None and _tt.calculation == "vibration"
        if tolerance_cm1 is not None and not _is_vibration:
            raise click.ClickException(
                "--tolerance-cm1 is a displacement sweep's -- a SIESTA "
                "vibration's force-constant stages compared "
                "(engines/vibration.md 5.9)")
        if _tt is not None and _tt.calculation == "transport":
            from ..transport.record import (RecordError, collect_record,
                                            iv_table_text, write_record)
            try:
                rec = collect_record(_P(bundle), _tt)
            except RecordError as e:
                raise click.ClickException(str(e))
            out = write_record(_P(bundle), rec)
            click.echo(iv_table_text(rec))
            click.echo(f"-> {out}")
            _ledger(_P(bundle), "summarize", "transport-record",
                    out=str(out), points=len(rec["points"]),
                    pending=len(rec.get("pending", ())))
            return
        if _is_vibration:
            # A DISPLACEMENT SWEEP'S SUMMARY (engines/vibration.md § 5.9): the
            # force-constant stages' results, which their jobs wrote, read
            # where they are and compared -- nothing derived that a run did
            # not already write, nothing moved.  Every stage is compared, so
            # a stage name is not asked for.
            if stage is not None:
                raise click.ClickException(
                    f"a displacement sweep compares every force-constant "
                    f"stage; name none (it was given {stage!r}).")
            from ..spectra.displacement_sweep import (SweepError,
                                                      collect_sweep,
                                                      sweep_table_text,
                                                      write_sweep)
            try:
                rec = collect_sweep(_P(bundle), _tt,
                                    tolerance_cm1=tolerance_cm1)
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
            "run`, into <label>.transport.json), and a SIESTA vibration's "
            "force-constant stages (`summarize run`, into "
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
    stage = _described_stage(bundle, stage)
    js, base = _load_bench_set(bundle, stage, "summarize")
    _check_kind(kind, js)
    from .summarize import (run_summarize_jobset,
                                   summary_text, utc_now_iso)
    container, _ = _stage_bench_dir(base, stage, "summarize")
    # The container's job-set holds ONLY this stage's trials (U1), so the
    # SET is the scope and the verdict goes back where they live -- there
    # is no name filter anywhere (U12).  A description-less sweep has no
    # stages to name at all:
    if container is None and stage is not None:
        from .commands import command
        raise click.ClickException(
            f"this sweep carries no description, so it has no stage named "
            f"{stage!r} -- run it bare:\n    "
            + command("summarize", "bench", base=base))
    from ..runfiles import BENCH_RESULT_FILE
    res, out_path, report = run_summarize_jobset(
        js, base,
        out=(container / BENCH_RESULT_FILE) if container is not None
            else None,
        now_iso=utc_now_iso(), stage=stage)
    click.echo(summary_text(res, out_path, report=report, stage=stage,
                            base=base))
    _ledger(base, "summarize", "verdict-written", stage=stage,
            out=str(out_path), points=len(res.points),
            choice=(res.choice or None),
            reported=report is not None)


def _refuse_flags_without_effect(*, kind: str, mode: str, trial,
                                 domain, time_text, mem_text, gpu_domain,
                                 trial_timeout_min, only_side) -> None:
    """A launch flag that would not be read is REFUSED by name, never
    ignored (W52: ``--trial-timeout`` was read by the grouped bench alone,
    ``--gpu-domain`` likewise, and ``--time``/``--mem`` meant nothing to a
    direct run -- each silently, while ``--only`` was refused in exactly
    those places as *a filter silently ignored*)."""
    flags = {"--domain": domain, "--time": time_text, "--mem": mem_text,
             "--gpu-domain": gpu_domain,
             "--trial-timeout": trial_timeout_min, "--only": only_side}

    def said(*names):
        return [n for n in names if flags[n] is not None]

    if mode == "direct":
        got = said(*flags)
        if got:
            raise click.ClickException(
                f"{', '.join(got)}: what a scheduler is asked for -- "
                f"`--mode direct` runs it here, where "
                f"{'it means' if len(got) == 1 else 'they mean'} nothing.")
    if kind != "bench":
        got = said("--gpu-domain", "--trial-timeout", "--only")
        if got:
            raise click.ClickException(
                f"{', '.join(got)}: a benchmark's -- `launch run` sends one "
                f"stage, to the queue --domain names.")
    elif trial is not None:
        got = said("--trial-timeout", "--only")
        if got:
            raise click.ClickException(
                f"{', '.join(got)}: a grouped bench's -- a named trial is "
                f"sent alone, and is the selection.")


def _show_and_ask(plan, *, dry_run: bool, auto_yes: bool,
                  footer=()):
    """**Nothing is submitted unseen** (`submission.md` S4): the exact
    ``sbatch`` line of every job about to be sent -- by the code that sends
    it -- what each follows, what only the person can judge, and what the
    request leaves to the scheduler; then the one question.  ``True`` to go.

    Every door asks it, the stage's included (user, 2026-10-01: the queue and
    the wall are decided at launch, after prep's printout, so they were never
    seen).  Under ``--dry-run`` the commands are printed by the results that
    follow, so only what the plan alone carries is said here, and nothing
    is asked: ``None``.  Otherwise the answer (`ask.Said`).  ``--yes``
    skips the question, never the output."""
    from .ask import confirm, gpu_share_notes
    from ..scheduler.quantities import parse_gres_flag
    planned = [r for r in plan if r.status == "planned"]
    lines = ["about to submit:"]
    for r in plan:
        if r.status == "planned":
            # NAME THE DOMAIN, not only the flags: where several domains
            # share one partition, `-p htc` reads as *htc* to anyone scanning
            # (2026-08-30: a debug sweep was believed to have gone wrong).
            lines.append(f"  {r.name}"
                         + (f"   -> domain {r.domain}" if r.domain else ""))
            lines.append(f"    {' '.join(r.command)}")
        elif r.status.startswith("WOULD"):
            lines.append(f"  {r.name}: {r.status}")
    warn = []
    for r in planned:
        # GPU SHARING, read off the very command about to be sent (user
        # 2026-08-23) -- its `-n` and its `--gres`, through the one reader.
        g = next((a for a in r.command if a.startswith("--gres=")), None)
        try:
            ng = parse_gres_flag(g.split("=", 1)[1]) if g else 0
            nr = int(r.command[r.command.index("-n") + 1])
        except (ValueError, IndexError):
            ng = 0
        for line in (gpu_share_notes(ng, nr) if ng else ()):
            line = "  " + line.strip()
            if line not in warn:
                warn.append(line)
    for r in planned:
        # R14 -- the cap this sweep will meet, said while no is still free.
        if r.detail and "  " + r.detail.strip() not in warn:
            warn.append("  " + r.detail.strip())
    judged = [r.judgement for r in plan if r.judgement]
    lines += warn + list(footer) + ["  " + j for j in judged]
    if dry_run:
        for line in warn + list(footer) + ["  " + j for j in judged]:
            click.echo(line)
        return None
    # A JUDGEMENT ONLY THE PERSON CAN MAKE is not made by Enter.
    return confirm("\n".join(lines), auto_yes=auto_yes,
                   default=not judged)


@jobset_group.command("launch", short_help="launch a prepped stage")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("stage", required=False, default=None)
@click.argument("trial", required=False, default=None)
@_bundle_option()
@click.option("--mode", type=click.Choice(["submit", "direct", "ask"]),
              default=None,
              help="HOW to launch, which is a fact about this MACHINE and not "
                   "about the layout: 'direct' = run it here with bash; "
                   "'submit' = hand it to this machine's scheduler; "
                   "**'ask' = submit NOTHING and report when it "
                   "would start** (`sbatch --test-only` on the line submit "
                   "would send), so you can change the queue or the request "
                   "and ask again before committing.  'ask' needs a login "
                   "node -- there is no prediction without the cluster.  "
                   "**Defaults to `launch.mode` in molbuilder.json** "
                   "(running-a-job.md § 5.4); pass it only to override that.")
@click.option("--domain", default=None, metavar="NAME",
              help="the queue -- a domain of this machine's record, sent as "
                   "-p/-q (submit and ask).  Unstated: the one prep baked "
                   "for this stage; with none, the queues are listed and "
                   "nothing is sent.  A grouped "
                   "bench's sides take it too; --only places one side at a "
                   "time.")
@click.option("--dry-run", is_flag=True,
              help="print the exact command each job WOULD get; launch "
                   "nothing, write nothing.")
@click.option("--time", "time_text", default=None, metavar=TIME_METAVAR,
              help=TIME_HELP)
@click.option("--mem", "mem_text", default=None, metavar=MEM_METAVAR,
              help=MEM_HELP)
@click.option("--gpu-domain", "gpu_domain", default=None, metavar="NAME",
              help="the queue a benchmark's GPU side goes to -- a grouped "
                   "bench's GPU shelves, or a named GPU trial -- when it "
                   "differs from --domain.  A cpu-only partition cannot take "
                   "the GPU side, so one queue cannot always answer for "
                   "both.  Omit it and the GPU side takes --domain.")
@click.option("--yes", "-y", "auto_yes", is_flag=True,
              help="take what is shown without being asked.  The request is "
                   "still printed -- --yes skips the question, never the "
                   "output (submission.md S4) -- and it is also your "
                   "recorded judgement to continue a run that was launched "
                   "and never concluded (project-layout.md § 1.6.4).")
@click.option("--trial-timeout", "trial_timeout_min", default=None,
              type=click.IntRange(min=1), metavar="MINUTES",
              help="a grouped bench (`launch bench <stage>`, submit mode): "
                   "kill any single trial after this many minutes so the "
                   "rest of the group still runs; the killed trial reads "
                   "incomplete.  Unstated, no per-trial bound exists -- each "
                   "trial runs until the job's wall.")
@click.option("--only", "only_side", default=None,
              type=click.Choice(["cpu", "gpu"]),
              help="a grouped bench: send just this side of a sweep that "
                   "spans CPU and GPU trials (generator.md § 4.3a).  The "
                   "other side stays pending; a later `launch bench` "
                   "collects it -- here or on the cluster that reaches it.")
def submit_cmd(kind: str, stage, trial, bundle: str, mode: str, domain,
               dry_run: bool, time_text, mem_text, gpu_domain,
               auto_yes, trial_timeout_min, only_side) -> None:
    """Launch a prepped stage: run it here (direct), hand it to the machine's
    scheduler (submit), or ask the scheduler when it would start (ask).
    Run ``prep`` first.  Before anything is sent the exact ``sbatch`` line is
    shown and you are asked; ``--dry-run`` shows it and sends nothing.

    ``--mode`` falls back to ``launch.mode`` (`running-a-job.md` § 5.4).
    """
    mode_source = "--mode flag"
    domain_source = "--domain flag" if domain else None
    # This machine's `launch.mode` when no --mode is given (running-a-job
    # § 5.4).
    if mode is None:
        from ..runtime_config import get_launch_mode
        try:
            mode = get_launch_mode()
        except Exception as exc:
            # A malformed config is ITS OWN error.  Swallowing it here told
            # the user to set a value they may already have set.
            raise click.ClickException(
                f"the launch block could not be resolved from config: "
                f"{exc}\n  Fix the config (running-a-job.md § 5.4).") from exc
        if not mode:
            # Unset is a refusal, never a derivation: deciding `submit` from
            # a DETECTED scheduler would gate submission on detection, which
            # running-a-job.md § 5.4 forbids.
            raise click.ClickException(
                "no --mode, and molbuilder.json sets no `launch.mode`.\n"
                "  'direct' runs it here with bash; 'submit' hands it to the "
                "scheduler.  Set launch.mode once for this machine, or pass "
                "--mode for this call (running-a-job.md § 5.4).")
        mode_source = "launch.mode (config)"
    _refuse_flags_without_effect(
        kind=kind, mode=mode, trial=trial, domain=domain,
        time_text=time_text, mem_text=mem_text, gpu_domain=gpu_domain,
        trial_timeout_min=trial_timeout_min, only_side=only_side)

    # ------------------------------------------------------------------ #
    #  Which work -- before which queue, which is read off this work      #
    # ------------------------------------------------------------------ #
    if kind == "bench":
        # the stage's own sweep record, from its bench container (§ 6.3)
        stage = _described_stage(bundle, stage)
        _refuse_disabled(bundle, stage)
        js, base = _load_bench_set(bundle, stage, "launch")
    else:
        if trial is not None:
            raise click.ClickException(
                "a TRIAL names a benchmark point; `launch run` takes a "
                "stage only (job-system.md § 5.3).")
        js, base = _load(bundle)
    _check_kind(kind, js)
    if kind == "bench":
        only = _pick_trial(js, base, trial)      # None: every trial
    else:
        # The description's spelling from here on -- what the ledger
        # records and every line prints (plan § 5w K12) -- and a stage it
        # holds that is not prepped, said so (W55 D4).
        stage = _described_stage(bundle, stage)
        _refuse_disabled(base, stage)
        _refuse_unprepped(base, stage)
        only = stage = _resolve_stage(js, stage, "launch", base=base)
    launching = [j for j in js.jobs if only is None or j.name == only]
    grouped = kind == "bench" and trial is None and mode in ("submit", "ask")
    mem = _memory(mem_text)
    time_s = _duration(time_text)

    # ------------------------------------------------------------------ #
    #  Which queue -- asked ONCE, of everyone who already answered        #
    # ------------------------------------------------------------------ #
    # NOBODY GUESSES THE QUEUE (user, 2026-08-23; `submission.md` S5).  In
    # order, most specific first: --domain on this call; the work's own
    # resources -- what prep baked for THIS stage (W52: every stage's row
    # was read, so a queue named at one stage's prep routed another).  There
    # is no machine-wide queue: `execution.domain` in molbuilder.json stood
    # in for one until 2026-10-02.  For `ask` as for `submit`, so the line
    # asked about is the line that would be sent (W52: ask resolved no queue
    # at all).  Reading the baked value is not inferring it -- a person put
    # it there -- and it is ADMITTED like any other: a bundle prepped
    # elsewhere may name a queue this machine lacks.
    slurm = mode in ("submit", "ask")
    if slurm and domain is None:
        _baked = {j.resources.domain for j in launching
                  if j.resources.domain}
        if len(_baked) == 1:
            domain = _baked.pop()
            domain_source = "the bundle (prep baked it)"
        elif len(_baked) > 1:
            raise click.ClickException(
                "the trials being sent name more than one domain ("
                + ", ".join(sorted(_baked)) + ").  Name the one to use with "
                "--domain, and --gpu-domain if the GPU side differs.")
    if slurm and domain is None:
        # A queue is needed unless the only side being sent is named by
        # --gpu-domain.
        from .submit import sides_of
        _gpu_only = bool(gpu_domain) and kind == "bench" and (
            (grouped and (only_side == "gpu"
                          or not sides_of(js)["cpu"]))
            or (only is not None
                and any(j.name == only for j in sides_of(js)["gpu"])))
        from ..runtime_config import get_routing
        _rows = [] if _gpu_only else get_routing(
            project_dir=Path(base) if base else None)
        if _rows:
            from .ask import Ask, queue_table
            from ..scheduler import parse_mem_gb
            # THE SAME REQUEST the door will admit -- the cores and the GPU
            # side included -- so the table cannot say yes where the
            # submission says no (`submission.md` § 3).
            _cores = max(((j.resources.mpi_np or 0)
                          * max(j.resources.cpus_per_task or 1, 1)
                          for j in launching), default=0) or None
            # ...each job's GPU REQUEST, through the one door the
            # submission asks (`model.gpu_request`) -- whether, and how many;
            # never a stand-in of one.
            from .submit import _gpus as _request
            try:
                _asks = [_request(j.resources, f"job {j.name!r}")
                         for j in launching]
            except SubmitError as e:
                raise click.ClickException(str(e))
            _gpu = only_side == "gpu" or (bool(_asks)
                                          and all(a.uses for a in _asks))
            _gpus = (max((a.count or 0 for a in _asks), default=0) or None
                     if _gpu else None)
            # ...and the WALL AND MEMORY the door admits: what launch
            # states, else what prep baked (`submit._sbatch_request`) -- the
            # flags alone showed a queue as fitting that the door then
            # refused (W55 D7).  The most any job being sent asks.
            from ..scheduler.quantities import parse_walltime

            def _most(read, field):
                got = []
                for j in launching:
                    v = getattr(j.resources, field, None)
                    try:
                        got.append(read(str(v)) if v else None)
                    except ValueError:        # the door refuses it, by name
                        got.append(None)
                return max((g for g in got if g is not None), default=None)
            _wall = (time_s if time_s is not None
                     else _most(parse_walltime, "time"))
            _mem = parse_mem_gb(mem) if mem else _most(parse_mem_gb, "mem")
            click.echo(queue_table(_rows, Ask(time_s=_wall, mem_gb=_mem),
                                   cores=_cores, gpus=_gpus))
            raise click.ClickException(
                "no --domain, so no queue was chosen.  Name one from the "
                "list above with `--domain` -- a queue named in the "
                "description's `allocation.domain` reaches a stage at its "
                "prep.")
    # The same provenance line prep printed, at the LAST moment before the
    # launch -- the mode above may have come from config, and this names
    # which file said so (user request 2026-08-12).
    from ..runtime_config import config_provenance, format_provenance
    prov = config_provenance(project_dir=base)
    click.echo(format_provenance(prov))
    from .ask import confirm
    from .submit import submit_bench_group, submit_transport_chain

    def _answered(said, about: str) -> bool:
        # EACH QUESTION AND ITS ANSWER IS WRITTEN DOWN, a *no* too
        # (`job-system.md` § 5.0, agreement 6) -- a declined launch left no
        # line until W55 D6.
        _ledger(base, "launch", "question", kind=kind, stage=stage,
                about=about, answer=said.words)
        return bool(said)

    try:
        if grouped:
            # ONE grouped job per resource shelf (§ 2.3.2, user 2026-08-20;
            # split per shelf 2026-08-21, generator.md § 4.3a): each shelf's
            # trials ride one exact-fit allocation in sequence.  A named
            # trial still goes alone -- how a single point is re-run.
            # NOTHING IS DERIVED (user dictation, 2026-08-24): --time is the
            # wall, said or the target queue's ceiling; --trial-timeout is
            # itself or absent.
            _bound_s = (trial_timeout_min * 60
                        if trial_timeout_min is not None else None)

            def _group(**kw):
                return submit_bench_group(
                    js, base, domain=domain, gpu_domain=gpu_domain,
                    trial_timeout_s=_bound_s, mem=mem, time_s=time_s,
                    only=only_side, **kw)

            if mode == "ask":
                results = _group(ask=True)
            else:
                plan = _group(dry_run=True)
                said = _show_and_ask(
                    plan, dry_run=dry_run, auto_yes=auto_yes,
                    footer=["  per-trial bound: "
                            + (f"{_bound_s // 60} min" if _bound_s else
                               "none -- each trial runs until the wall")])
                if said is not None and not _answered(said, "send"):
                    click.echo("nothing submitted.")
                    return
                if dry_run:
                    results = plan
                else:
                    # The decision records the SWEEP considered; the group's
                    # members are the still-unlaunched subset, which the
                    # "launched" entry records per job.
                    _ledger(base, "launch", "bench-grouped",
                            trial_timeout_s=_bound_s, time_s=time_s,
                            mem=mem, only=only_side,
                            sweep=[j.name for j in js.jobs])
                    results = _group(dry_run=False)
        else:
            # A TRANSPORT BIAS SCAN launches as ONE job walking the points
            # (archive/2026-09-01-transport-design.md 4.3) -- the chain's own door.
            _chain_task = None
            if kind == "run":
                from ..task import FILENAME as _TASKF
                from ..task import read_task as _rt_chain
                try:
                    _chain_task = _rt_chain(Path(base) / _TASKF)
                except Exception:
                    _chain_task = None        # a hand-built ladder: no scan
            _chain_scan = ()
            if (_chain_task is not None
                    and _chain_task.calculation == "transport"):
                from ..transport.stages import scan_points as _scan
                _chain_scan = _scan(_chain_task, only)

            def _send(**kw):
                if _chain_scan:
                    return submit_transport_chain(
                        js, base, _chain_task, mode=mode, stage=only,
                        domain=domain, mem=mem, time_s=time_s, **kw)
                return submit_jobset(
                    js, base, mode=mode, domain=domain,
                    gpu_domain=gpu_domain, only=only, mem=mem,
                    time_s=time_s, **kw)

            plan = _send(dry_run=True)
            judged = any(r.judgement for r in plan)
            if mode == "ask" or dry_run:
                results = plan
                if dry_run and mode != "ask":
                    _show_and_ask(plan, dry_run=True, auto_yes=auto_yes)
            elif mode == "submit":
                if not _answered(_show_and_ask(plan, dry_run=False,
                                               auto_yes=auto_yes), "send"):
                    click.echo("nothing submitted.")
                    return
                if _chain_scan:
                    _ledger(base, "launch", "bias-chain",
                            points=len(plan) - 1, mode=mode)
                    results = _send(dry_run=False)
                else:
                    results = _send(dry_run=False,
                                    continue_unconcluded=judged)
            else:
                # DIRECT runs what was typed, here -- the question S4 puts is
                # the scheduler's.  What only the person can judge is still
                # asked: following a run that may still be running.
                if judged and not _answered(confirm(
                        "\n".join(r.judgement for r in plan if r.judgement),
                        auto_yes=auto_yes, default=False),
                        "follow a run that never concluded"):
                    click.echo("nothing launched.")
                    return
                if _chain_scan:
                    _ledger(base, "launch", "bias-chain",
                            points=len(plan) - 1, mode=mode)
                    results = _send(dry_run=False)
                else:
                    results = _send(dry_run=False,
                                    continue_unconcluded=judged)
    except SubmitError as e:
        _ledger(base, "launch", "refused", kind=kind, stage=stage,
                trial=trial, mode=mode, mode_source=mode_source,
                reason=str(e))
        raise click.ClickException(str(e))
    # WHAT HAPPENED, by its own name: a question and a dry run send nothing,
    # and the ledger said "launched" for both until 2026-10-01 (W52).
    _ledger(base, "launch",
            "asked" if mode == "ask" else "planned" if dry_run else "launched",
            kind=kind, stage=stage, mode=mode, mode_source=mode_source,
            domain=domain, domain_source=domain_source, dry_run=dry_run,
            provenance=prov,
            jobs=[{"job": r.name, "status": r.status, "job_id": r.job_id,
                   "returncode": r.returncode} for r in results])
    from .commands import command
    if mode == "ask":
        # NOTHING WAS SUBMITTED.  The line the scheduler was asked about is
        # the line that WOULD be sent -- same flags, plus --test-only -- so
        # what is printed here and what launch would do cannot drift.
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
            if r.status.startswith("WOULD"):
                # a stage launched before: asking created nothing (the
                # attempt opens at launch), and the line asked about is
                # the one that attempt would send.
                click.echo(f"  {r.name}: {r.status}")
                if r.judgement:
                    click.echo("  " + r.judgement)
        if not preds:
            click.echo("\n  nothing left to ask about here.")
            if ran:
                click.echo("  read what they measured:\n    "
                           + command("summarize", "bench", stage, base=base))
            return
        asked = next((r for r in results if r.prediction is not None), None)
        click.echo("")
        # "nothing to wait for" followed by an sbatch preview reads as a
        # contradiction (user, 2026-08-28): on a scheduler-less machine
        # nothing WOULD be sent, so nothing is previewed.
        if not all(p.no_scheduler for p in preds):
            click.echo("  would send: " + " ".join(asked.command))
            click.echo("  launch it with the same command and "
                       "`--mode submit` when the answer suits you.")
        else:
            click.echo("  run it here instead: the same command with "
                       "`--mode direct`.")
        return

    verb = "WOULD run" if dry_run else "result"
    for r in results:
        tail = (f"job {r.job_id}" if r.job_id else
                (f"rc={r.returncode}" if r.returncode is not None else ""))
        # a skipped trial was not run and WOULD not be -- its verb says so;
        # a trial or a point riding a shelf or a chain is launched BY that
        # one command
        v = ("skip     " if r.status.startswith("skipped")
             else "rides    " if r.status.startswith("rides the")
             else verb)
        click.echo(f"  {v}  {r.name:<12} {' '.join(r.command)}  "
                   f"[{r.status}] {tail}".rstrip())
    # A PARTIAL SUBMISSION SAYS SO, LOUDLY.  One shelf refused by the
    # scheduler does not cancel the others, so the run can end with some
    # jobs queued and some not; the refused shelves' trials stay pending,
    # so re-running `launch` picks up exactly them.
    _refused = [r for r in results if r.status == "sbatch refused"]
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
    if not dry_run:
        # ONE COMMAND A LINE, the sentence above it (`commands`).
        click.echo("next -- read the measurements once they have run:\n    "
                   + command("summarize", "bench", stage, base=base)
                   if kind == "bench" else
                   "next -- look before the next stage:\n    "
                   + command("status", base=base))


# --------------------------------------------------------------------- #
#  probe -- this machine's record, read off the machine.  It came from   #
#  `molbuilder bench` on 2026-08-17 as `probe-scheduler`, the last        #
#  inhabitant of a group whose lifecycle verbs went in the 2026-08-12     #
#  fold (every verb lives under `jobset` -- user, 2026-08-17), and        #
#  proposed a `scheduler` config block; that block is refused since      #
#  2026-10-02, and the probe writes the machine's record instead.        #
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
    # THE THREE FACTS THAT TRAVEL WITH A RECORD (`diagnostics.local_facts`:
    # how the machine enters its environment, which envs it holds, what they
    # were built for) are asked about like every other difference: a record
    # edited by hand is a person's answer, and a probe replaced it unasked
    # whatever was answered (W52).
    for _f in ("env_init", "conda_envs", "env_arch"):
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
                      short_help="rewrite a pre-2026-09-28 calculation's "
                                 "charge and spin items")
@_bundle_option()
def migrate_cmd(bundle: str) -> None:
    """Rewrite this calculation's template into the electronic state's items
    (science/chemistry-correctness.md § 2a): PySCF's `spin` and the R/U inside
    its `method`, SIESTA's `spin_treatment` spellings and `spin_total`.

    What the run was is KEPT -- every old value becomes a stated one -- and
    each change is printed.  The old template stays beside the new one as
    `<name>.pre-m6`.  prep refuses a template that still carries an old item,
    naming this command."""
    from .migrate import MigrateError, migrate_state
    try:
        said = migrate_state(bundle)
    except MigrateError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"migrated {bundle}:")
    for line in said:
        click.echo(f"  {line}")


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
        # THE NAME IS THE PERSON'S, so it is asked for in words -- a
        # `<name>` in a command is a redirect to bash (W52).
        click.echo("No named targets yet.  To prepare for another machine, "
                   "run `molbuilder jobset probe --write --name` on THAT "
                   "machine with the name you will prep it by (`--target`), "
                   "then copy the file it writes into:\n    "
                   f"{environments_dir()}/")
    elif choice_required(machines):
        # NAME THE LOCAL SPELLING HERE TOO.  The listing above shows
        # `(this machine)`, which is a LABEL and not something anyone can
        # type -- so a reader told that `prep` "requires --target <name>"
        # was left with no name for the box in front of them.  The same
        # gap the ambiguity refusal had (`record.LOCAL_TARGET`, 2026-08-24);
        # a hint that names only half the options is half a hint.
        from ..scheduler.record import LOCAL_TARGET
        # ONLY BEFORE THE FIRST PREP: a calculation that holds its snapshot
        # has its answer, and is asked nothing (`record.machine_for`'s C1;
        # W52: this said `prep` requires the flag, always).
        click.echo("More than one machine could be meant, so a calculation's "
                   "first `prep` asks which, with `--target` (being asked "
                   "costs one flag; being given the wrong one costs a queue "
                   "wait); once prepped, it keeps the record it took.")
        click.echo(f"    --target {LOCAL_TARGET}   # this machine")


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
              help="declare a topology fact the probe cannot see where it "
                   "runs (M-1's declared door -- e.g. a login node whose "
                   "compute nodes hold the GPUs): --set gpus_per_node=4. "
                   "Repeatable; wins over detection; the record's source "
                   "says 'flag'.")
@click.option("--scheduler", "scheduler_flag", default=None,
              type=click.Choice(["slurm", "workstation"]),
              help="force the scheduler kind instead of detecting it "
                   "(source 'flag').")
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
    from anywhere.  **Without it, a bench prepped on a laptop is measured
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
    from ..scheduler.record import LOCAL_TARGET
    if name == LOCAL_TARGET:
        # RESERVED: `--target this` means the box you are on, so a record by
        # that name would make the flag ambiguous -- and it is the one name
        # whose meaning nothing can override.  Refused with what else was
        # typed, before anything is probed.
        raise click.ClickException(
            f"{LOCAL_TARGET!r} is reserved: `--target {LOCAL_TARGET}` "
            f"already means this machine, so a record called that could "
            f"never be prepped for.  Give it the machine's own name (`--name "
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
    env = resolve_environment(now_iso=now, overrides=overrides or None,
                              scheduler_override=scheduler_flag)

    # THE THREE FACTS THAT TRAVEL WITH A RECORD -- how this machine enters
    # its environment, which envs exist here, and what they were built
    # for.  `diagnostics.local_facts` owns them and states why; `envs
    # init-config` became the second caller 2026-09-08, which is what
    # took them out of this function.  The first is THIS machine's
    # `env_init`, declared in its molbuilder.json and copied into whichever
    # record this writes, this machine's or a named one (user, 2026-10-02) --
    # required, and checked above.
    from ..diagnostics import local_facts as _local_facts
    env, facts_notes = _local_facts(env, declared)

    # THE QUEUES -- `record.probe_queues`, the one queue probe; `envs
    # init-config` seeds this machine's record through it too (M-3).  It was
    # inline here until 2026-10-02, and init-config seeded a cluster with none.
    from ..scheduler.record import probe_queues
    notes, summary = probe_queues(env, user)
    if summary:
        click.echo(summary)

    # A named target is a record ABOUT another machine, kept beside this
    # machine's rather than replacing it (P2): a workstation holds both its own
    # capability and the cluster's, and `prep --target NAME` says which.
    # WHICH FILE, asked for -- not a directory plus a re-spelled name.  This
    # used to read `target = machine_scope_path().parent` / `fname =
    # f"{name}.json" if name else FILENAME`: the record's own resolver taken
    # apart to get a directory, and its filename typed again beside it, with
    # the bare `FILENAME` imported into a surface to do it (A11, I1).  Moving
    # either file would have moved the reader and left this writer behind.
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
    click.echo(f"\nMachine: scheduler={env.scheduler}"
               f"  cores/socket={t.cores_per_socket}  sockets={t.sockets}"
               f"  gpus/node={t.gpus_per_node}  gpu={t.gpu_type or '-'}"
               f"  mem={t.mem_total_gb or '-'} GB")
    if env.domains:
        click.echo("\nReachable domains (a launch names one with "
                   "--domain):")
        for d in env.domains:
            click.echo(f"  {d.name:<10} <= {str(d.max_time):<12} "
                       f"{d.partition}/{d.qos}")
    # AFTER the queue probe, whose notes this extends (`record.probe_queues`
    # builds them after `derive_domains`, which reassigns its list).  This
    # line was composed and never shown: `notes_sg` was assigned and read
    # by nothing, so the one machine that most needed the warning -- the
    # one with no activation -- was the one told nothing.
    notes.extend(facts_notes)
    if notes:
        click.echo("\nNotes / assumptions (read before --write):")
        for n in notes:
            click.echo(f"  - {n}")

    if not do_write:
        click.echo(f"\n(dry run -- nothing written. Re-run with --write to "
                   f"record this in {target / fname}.)")
        return

    if before is None:
        # Nothing to clobber: one consent creates the record -- unless a file
        # IS there and does not read, which is said, never treated as absent
        # (W52: `read_environment` answers both with `None`, and a newer
        # schema or a hand-fixed record was replaced unasked; `--yes` skips
        # the question, never this line).
        dest = target / fname
        if dest.exists() and not dest.is_file():
            # NOT A FILE AT ALL -- a directory: a record cannot replace it,
            # and the write would end in a traceback (the fix-6 review).
            raise click.ClickException(
                f"{dest} is there and is not a file -- move it aside, and "
                f"probe again.")
        there = dest.exists()
        if there:
            import os as _os
            why = ("cannot be read here (permissions)"
                   if not _os.access(dest, _os.R_OK) else
                   "does not read as a record this molbuilder knows")
            click.echo(f"\n{dest} is there and {why} -- writing replaces "
                       f"it.")
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
    # 0700, through the one creator.  This was a bare `mkdir` with no mode, and
    # `jobset probe --write` is the FIRST command the seeded `environments/
    # README` tells a person to run on a target -- so the config directory that
    # every later secret lands in was created world-readable, and `envs
    # init-config` then reported it "kept" (A4).
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
