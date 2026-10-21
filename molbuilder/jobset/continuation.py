"""What a stage continues from -- which run, and what that run was
(`execution/job-system.md` § 5.4, plan W37): the stage before it in an
independent ladder, and the `relax` run a vibration's force-constant stage
builds on (`engines/vibration.md` § 5.2a, W38 F9).

**Module:** L2 (jobset).  ONE answer, asked by `launch` at a stage's first
launch (`job-system.md` § 6.0), by `prep` for the `relax` run a force-constant
stage's deck is written at -- the default, or what a run named by ``--from``
is -- and by `status` through the ready door, so none says a different thing
about the same run.
Reads the folder; writes
nothing; raises nothing the person can fix -- a refusal is an answer, which
`prep` turns into its own error and `status` prints; a bug raises.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

from ..runrecord import LaunchRecordError


@dataclasses.dataclass(frozen=True)
class Continuation:
    """WHICH RUN A STAGE CONTINUES FROM, and what that run was -- printed by
    the launch that takes it (a force-constant stage's, by its prep too)
    and recorded in the decision ledger."""
    #: The stage the run belongs to.
    stage: str
    #: Its attempt, from the calculation folder (``01_coarse/run-0``) --
    #: ``None`` on the flat layout, where the run's files lie in the folder.
    source: Optional[str]
    #: The stage before it, newest (the default) -- or named by ``--from``.
    by_default: bool
    #: The line its conclusion said (`runrecord.ending`), or ``None``: the
    #: run has not ended on its own.
    concluded: Optional[str]
    #: Its run state, `run_status`'s word (finished, failed, running, ...).
    state: Optional[str]
    #: The relaxation's verdict, ``None`` when the run relaxed nothing.
    converged: Optional[bool]
    #: A force-constant stage building on its `relax` (`engines/vibration.md`
    #: § 5.2a), not a stage continuing from the one before it.
    linked: bool = False
    #: The files it carries into the attempt, by the pair's rule
    #: (`model.warm_carry`) of what the run holds -- counted against the
    #: stage's job, ``None`` until then.  On the flat layout nothing is
    #: copied: the files lie in the folder.
    carries: Optional[Tuple[str, ...]] = None
    #: On the flat layout, the run by the name every file of it carries
    #: (`runfiles.run_name`, ``H2_01_coarse-run1``) -- what the stage's own
    #: ``.continued-from`` records -- or ``None``: an attempt names it.
    run: Optional[str] = None
    #: The stage's OWN latest run: a stage launched again (:func:`relaunch`;
    #: `job-system.md` § 5.4, *A stage launched again*).
    own: bool = False
    #: On the flat layout, the run whose restart files lie in the folder --
    #: the last run of any stage there (`runrecord.last_run`) -- when it is
    #: not :attr:`run`; and how it ended, in words.  What the run reads, and
    #: so what its record names (`job-system.md` § 5.4; user, 2026-10-12).
    written_by: Optional[str] = None
    written_by_is: Optional[str] = None

    @property
    def ref(self):
        """The run, as its hand-over records it (`runrecord.RunRef`): its
        attempt's folder, or on the flat layout the run whose restart files
        lie in the folder -- the last run there -- by its name; ``None`` for
        a flat stage never launched, which no hand-over takes from."""
        from ..runrecord import RunRef
        if self.source is not None:
            return RunRef.read(self.source, shared=False)
        name = self.written_by or self.run
        return RunRef.named(name) if name else None

    def where(self) -> str:
        """The run, as a person reads it: its attempt, or -- on the flat
        layout -- the stage's latest run in the folder, by its name; and when
        another run was the last here, the restart files that one left,
        which are what it reads (`job-system.md` § 5.4)."""
        if self.source is not None:
            return self.source
        # A FORCE-CONSTANT STAGE'S RUN IS THE ONE ITS DECK IS WRITTEN AT, the
        # run its prep took -- not necessarily `relax`'s latest.
        which = "run its deck is written at" if self.linked else "latest run"
        mine = (f"{self.stage}'s {which}"
                + (f", {self.run}," if self.run else "") + " in this folder")
        if not self.written_by:
            return mine
        return (f"the restart files {self.written_by} left in this folder "
                f"({self.written_by_is}) -- the last run here; "
                f"{self.stage}'s {which} is {self.run}")

    def line(self, copied=(), *, would: bool = False, warned=None) -> str:
        """``continues from 01_coarse/run-0 (the stage before it; concluded
        rc=0 at ...; converged): copied H2.XV, ...`` -- what both doors
        print; ``would``, a preview's: what it would copy.  ``warned`` is
        ``{name: why}`` for a copied file written as its run ended
        (`HandOver.written_as_ended`, `running-a-job.md` § 3.6), said after
        the file's name."""
        warned = warned or {}
        facts = [("named" if not self.by_default else
                  "its own latest run" if self.own else
                  "the relaxation it builds on" if self.linked else
                  "the stage before it"),
                 _what_it_is(self.concluded, self.state)]
        if self.converged is not None:
            # NOT CONVERGED, said for what it costs: a force-constant stage
            # measured off a stationary point reports imaginary frequencies
            # (W38 F9).
            facts.append("converged" if self.converged else
                         "NOT converged -- expect imaginary frequencies"
                         if self.linked else
                         "NOT converged -- taken as it stands")
        in_place = [n for n in warned if n not in copied]
        return (f"continues from {self.where()} ({'; '.join(facts)})"
                + (f": {'would copy' if would else 'copied'} "
                   + ", ".join(n + (f" -- WARNING: {warned[n]}"
                                    if n in warned else "")
                               for n in copied) if copied else "")
                + "".join(f"; {n} in the folder -- WARNING: {warned[n]}"
                          for n in in_place))

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def ledger_facts(self) -> dict:
        """The decision-ledger line's facts -- the run's stage as
        ``from_stage``, beside the stage being prepared.  What came across is
        the line's ``copied``, read off the attempt opened."""
        d = self.as_dict()
        d["from_stage"] = d.pop("stage")
        d.pop("carries", None)
        return d


def _what_it_is(concluded: Optional[str], state: Optional[str]) -> str:
    """A run's conclusion, in words: its marker's line when it concluded --
    FAILED beside it when its run failed -- else what its state says."""
    if concluded is not None:
        return ("concluded " + concluded).strip() + (
            ", FAILED" if state == "failed" else "")
    return {"pending": "NOT launched yet",
            "queued": "NOT concluded -- queued",
            "running": "NOT concluded -- still running",
            "unreadable": "its launch record does not read -- whether it "
                          "ran cannot be told"}.get(
        state, "NOT concluded -- stopped without its conclusion marker")


def flat_files(base, task, cont: Optional[Continuation]
               ) -> Optional[Continuation]:
    """On the flat layout, ``cont`` with the run whose restart files lie in
    the folder -- the last run of any stage there (`runrecord.last_run`) --
    when it is not the run the rule took: what the stage reads, and so what
    its record names, said beside the rule's run (`job-system.md` § 5.4;
    user, 2026-10-12).  A record there that does not read raises
    `runrecord.LaunchRecordError`, which each caller says as its refusal."""
    from ..runfiles import parse_run_name
    from ..runrecord import last_run
    if cont is None or cont.source is not None:
        return cont
    last = last_run(base, task.label)
    if last is None or str(last) == cont.run:
        return cont
    from .materialize import ladder_homes
    named = parse_run_name(str(last), task.label)
    home = next((h for h in ladder_homes(base, task)
                 if h.token == named.stage), None)
    if home is None:
        said = "of a stage this description no longer holds"
    else:
        concluded, state, _c, _d = read_run(base, task, home.name,
                                             Path(base), verdict=False,
                                             run=named.run)
        said = _what_it_is(concluded, state)
    return dataclasses.replace(cont, written_by=str(last), written_by_is=said)


def usable(state: Optional[str]) -> bool:
    """THE RULE FOR A RUN TO BUILD ON (`execution/architecture.md` § 3.2,
    `job-system.md` § 5.4): the one status door says it FINISHED -- the
    door `status` asks (`parse.dirs.run_status`, read by :func:`read_run`),
    from every fact of the run: it ended on its own with exit code 0, and
    nothing in its output says the engine stopped (user, 2026-10-06: "the
    handover should call a unified api that tells the true status of
    previous stage that views all facts from exit code, log etc.").  The
    default of every hand-over -- a continuing stage, the frequency stage's
    geometry, a transport rung's inputs; a run named with ``--from`` is
    taken as said, and a structure stated relaxed needs no run, so neither
    asks it."""
    return state == "finished"


def done(folder, names, *, launch=None, products=(),
         launched_in=None) -> Tuple[bool, str]:
    """THE DONE-DOOR (`engines/transport.md` § 2a.11): whether the run, or a
    swept run's point, in ``folder`` is DONE -- ``(True, "done")`` -- or what
    it is instead, in the run door's words.  Done is three facts, all read
    from the folder's own files: it **finished** (the one status door,
    :func:`usable`); its run **converged** where it has an SCF or a
    relaxation (the same scan, `runstatus.converged_of`); and it **holds**
    ``products`` -- what the rungs downstream take from it.  ``launch`` is
    the launch record the folder's run was sent with (a point's is its
    run's), and ``launched_in`` the folder it lies in when not ``folder``
    (a point's run's folder; `run_status`)."""
    from ..parse.dirs import run_status
    from .runstatus import converged_of
    folder = Path(folder)
    if not folder.is_dir():
        return False, "not opened"
    st = run_status(folder, names.stem, launch=launch,
                    launched_in=launched_in)
    if st.state in ("pending", "queued"):
        # NOTHING RAN HERE YET: queued and pending are the RUN's state, said
        # on its row; a point is done or not done.
        return False, "not run"
    if not usable(st.state):
        return False, st.detail or st.state
    if st.verdict is not None and not st.verdict[1]:
        return False, f"finished, not converged ({converged_of(st)})"
    missing = [_said(p) for p in products if not _held(folder, p)]
    if missing:
        return False, f"finished without {', '.join(missing)}"
    return True, "done"


def _held(folder: Path, product) -> bool:
    """Does ``folder`` hold ``product`` -- a file's name, or several sets of
    names of which one set, whole, must be there (a polarized transmission's
    two channel files stand where the unpolarized one does;
    `transport.stages.products_of`)?"""
    if isinstance(product, str):
        return (folder / product).is_file()
    return any(all((folder / f).is_file() for f in group) for group in product)


def _said(product) -> str:
    """``product`` as a message names it: the file, or each set joined."""
    if isinstance(product, str):
        return product
    return " or ".join(" and ".join(group) for group in product)


def read_run(base: Path, task, stage: str, attempt: Path,
             *, verdict: bool = True, run: Optional[int] = None
             ) -> Tuple[Optional[str], Optional[str], Optional[bool],
                        Optional[str]]:
    """``(concluded, state, converged, detail)`` of one run of ``stage``
    -- ``detail`` the status door's own words for the state -- the
    line its conclusion said, through the one door (`runrecord.ending`;
    ``None`` when it has not ended on its own), its state through the one
    door (`parse.dirs.run_status`) and its relaxation's verdict
    (`parse.contract.relaxation_of`), read from the run's own files in
    either shape, the run door's (`runs.run_of`): its engine output at the
    run's index, then its progress log -- where a PySCF run writes its
    steps, which its stdout cannot stand in for.  The run is the newest in
    ``attempt`` by the one rule (`runrecord.newest_run`): where a stage's
    runs share the folder, its newest launched; in a run's own folder, its
    newest try -- or ``run``, a run named (a force-constant stage's
    `relax` run, recorded at its prep).  ``verdict=False``
    skips the relaxation's parse -- a list of runs needs each one's
    conclusion, not a parse of each output."""
    from ..parse.contract import relaxation_of
    from ..runs import concluded as _concluded, run_of
    from ..runrecord import ending, newest_run
    from .materialize import stage_home
    home = stage_home(base, task, stage)
    token, names = home.token, home.names(task.shape)
    run = run if run is not None else newest_run(attempt, names)
    concluded = ending(attempt, names.stem, run).line
    # ITS STATE, through the one door, asked with its launch record -- a
    # record that does not read is the state `unreadable`, said, never taken
    # for a run that stopped (`runs.concluded`, which asks the scheduler
    # once where it can, `running-a-job.md` § 4.2).
    from ..paths import run_folder_of
    st = _concluded(attempt, names, base=base, run=run,
                   launched_in=run_folder_of(attempt))
    state, detail = st.state, st.detail
    rec = None
    if verdict:
        # THIS STAGE'S RUN, the run door's, in either shape -- an attempt's
        # own, or this stage's among a flat folder's -- read in the order the
        # run door reads a run (`runfiles.result_roles`): its engine output
        # at the run's index, then its progress log, which a SIESTA run never
        # writes into and a PySCF run's output no parser reads.  The
        # force-constant stage reads its relax run's record from that same
        # output (`prep._vibration_stage_geometry`), so the line and the deck
        # say one verdict.  Where a stage's runs share the folder, the run
        # is the newest launched -- the one whose record was read above.
        _run = run_of(attempt, stage=token)
        if _run is not None and run is not None:
            _run = dataclasses.replace(_run, run=run)
        paths = ([_run.stdout,
                  _run.file(".molwatch.log",
                            _run.run if names.numbered(".molwatch.log")
                            else None)]
                 if _run is not None else [])
        from ..parse.errors import ParseError
        for path in (q for q in paths if q is not None):
            try:
                rec = relaxation_of(path)
            except (OSError, ValueError, ParseError):
                # AN OUTPUT THAT DOES NOT READ says no verdict -- the line
                # says its state, as the status door gave it; a defect in a
                # reader is not caught.
                rec = None
            if rec is not None:
                break
    return (concluded, state,
            None if rec is None else bool(rec["converged"]), detail)


def continuation_answer(calc, stage: str, *, from_attempt=None,
                        cold: bool = False, verdict: bool = True,
                        bench: bool = False
                        ) -> Tuple[Optional[Continuation], Optional[str]]:
    """``(continuation, refusal)`` for ``stage`` (`job-system.md` § 5.4).

    First, what cannot be taken at all is refused, before prep writes
    anything (:func:`cannot_be_named`): a run that is not one of this
    calculation's, ``--from`` with ``--cold``, and -- this door answers a
    first launch -- either on the flat layout or a bias sweep.  **Named** (``--from``): taken as said, for any kind -- what
    the run was is read and reported; `prepare_attempt` refuses only what
    cannot be done (no restart files in it).  **By default**, for a
    continuing stage of an independent ladder (`stage_before`): the NEWEST
    attempt of the stage before it, which must have ended on its own
    with exit code 0 (:func:`usable`) -- an older one never stands in,
    because a stage re-launched to tighten is the run the person means.  Otherwise the refusal, worded by
    what the run's state says and naming the commands that work on this
    layout.  ``(None, None)`` for ``--cold``, a linked kind's default, the
    first stage, a stage that starts clean, and one the description does
    not hold (which prep refuses at its checkpoint 2).  ``verdict=False`` leaves the
    relaxation unread -- for a reader that never prints it.

    A force-constant stage with no `relax` before it builds on nothing: the
    structure as given, when it is stated relaxed -- else refused, whatever
    was asked (`engines/vibration.md` § 5.2a's table).

    ``calc`` is the calculation as the verb read it (A16) -- prep's, with
    its plan as prep's overlay leaves it (`Calculation.as_planned`).
    ``bench`` asks for a benchmark of the stage, which takes no ``--from``
    and is offered none."""
    from ..identity import command_stage
    base, task = Path(calc.base), calc.task
    template_text = calc.template_text
    if force_constant_stage(task, stage) and relax_stage_of(task) is None:
        try:
            cfg = _stage_config(task, stage, template_text)
        except unreadable() as exc:
            return None, (f"what `{stage}` builds on cannot be read: "
                          f"{exc}")
        why = unrelaxed_refusal(
            task, stage, stated=bool(getattr(cfg, "already_relaxed", False)))
        if why:
            return None, why
    refused = cannot_be_named(base, task, stage, from_attempt, cold)
    if refused:
        return None, refused
    if cold:
        return None, None
    if from_attempt:
        # A RUN NAMED is taken as said, for any kind -- what it was is
        # reported and recorded, a linked stage's as much as any.
        attempt = base / from_attempt
        named = command_stage(Path(from_attempt).parts[0])
        concluded, state, converged, _ = read_run(
            base, task, named, attempt, verdict=verdict)
        return Continuation(stage=named, source=str(Path(from_attempt)),
                            by_default=False, concluded=concluded, state=state,
                            converged=converged,
                            linked=force_constant_stage(task, stage)), None
    try:
        prev, linked = source_stage(task, stage, template_text)
        if prev is None:
            return None, None
        return _by_default(calc, stage, prev, verdict=verdict,
                           linked=linked, bench=bench)
    except unreadable() as exc:
        # RAISES NOTHING THE PERSON CAN FIX, as this module promises: a
        # template prep would refuse is a refusal here too, said.  A
        # `TypeError` is a bug, and looks like one.
        return None, (f"what `{stage}` continues from cannot be read: "
                      f"{exc}")


def unreadable() -> tuple:
    """What a description, its template or a run's records raise when they
    do not read -- the person's to fix, said as a refusal."""
    from ..resolve import ResolveError
    from .errors import PrepError
    return (ValueError, KeyError, OSError, ResolveError, PrepError)


def unrelaxed_refusal(task, stage: str, *, stated: bool) -> Optional[str]:
    """§ 5.2a's row *no `relax` stage, the structure not stated relaxed*
    (`engines/vibration.md`): the box says *relax first* and the ladder
    holds nothing that would, so the description contradicts itself -- refused
    with the two ways out rather than measured at a geometry nobody chose
    (§ 2.2).  ``None`` when the ladder holds a `relax`, or the structure is
    stated relaxed."""
    from ..pyscf.stages import VIBRATION_RELAX_STAGE
    if relax_stage_of(task) is not None or stated:
        return None
    return (f"the structure is not stated to be relaxed (already_relaxed is "
            f"false in the template) and this ladder has no "
            f"`{VIBRATION_RELAX_STAGE}` stage to relax it -- a harmonic "
            f"analysis off a stationary point reports the wrong frequencies "
            f"(engines/vibration.md 2.2).  Either add the "
            f"`{VIBRATION_RELAX_STAGE}` stage before `{stage}` (Task setup, "
            f"or task.json) and run it first, or state already_relaxed = true "
            f"in the template; the finish then measures the forces at this "
            f"geometry and says whether the statement held.")


def cannot_be_named(base: Path, task, stage: str, from_attempt,
                     cold: bool) -> Optional[str]:
    """Why ``--from`` / ``--cold`` cannot be taken here -- said before
    anything is written -- or ``None``.  Every door that takes them asks
    through this."""
    if not (from_attempt or cold):
        return None
    if from_attempt and cold:
        return ("--from and --cold are two answers to one question -- name "
                "the run it continues from, or start it from the "
                "calculation's structure.")
    from pathlib import PurePosixPath
    from ..identity import command_stage
    from ..paths import Shape
    from .materialize import FLAT_HAS_NO_ATTEMPTS
    if not Shape.named(task.shape).keeps_attempts_as_directories:
        # A STAGE OF AN INDEPENDENT LADDER starts clean by its run card; a
        # linked one builds on what its kind says, and is told no way it
        # does not have.
        return FLAT_HAS_NO_ATTEMPTS + (
            "  A stage starts clean by its run card's `restart: clean`."
            if _independent(task) else "")
    if getattr(task, "calculation", None) == "transport":
        # A TRANSPORT RUNG'S INPUTS ARE ITS KIND'S: gathered from the rungs
        # upstream (`gather.gather_sources`), never named.
        return ("--from / --cold name what a stage continues from; a "
                "transport rung takes its inputs from the rungs upstream, "
                "gathered by its kind at its first launch, and anew with "
                "--gather when it is launched again (engines/transport.md "
                "§ 6.1, job-system.md § 5.4).")
    if (from_attempt and not _independent(task)
            and not force_constant_stage(task, stage)):
        # A LINKED KIND'S FIRST RUNG -- a vibration's `relax`, PySCF's one
        # rung -- builds on the calculation's structure; a run of another
        # stage is not its input (a `freq` run's `.XV` is a displaced
        # geometry).
        return (f"--from {from_attempt!r}: `{stage}` builds on the "
                f"calculation's structure, its kind's first rung -- no run "
                f"of another stage is its input (engines/vibration.md "
                f"5.2a).")
    if from_attempt:
        p = PurePosixPath(str(from_attempt))
        if p.is_absolute() or ".." in p.parts:
            return (f"--from names a run of this calculation, by its folder "
                    f"(01_coarse/run-0): {from_attempt!r}")
        if not (base / from_attempt).is_dir():
            return (f"--from {from_attempt!r}: no such attempt in this "
                    f"calculation.  Name an attempt directory that has "
                    f"already run, e.g. '01_coarse/run-0'.")
        head = p.parts[0]
        try:
            named = command_stage(head)
        except ValueError:
            named = None
        if named not in {s.name for s in task.stages}:
            return (f"--from {from_attempt!r}: {head!r} is not a stage "
                    f"folder of this calculation -- name a run as "
                    f"<NN>_<stage>/run-<n>.")
        # A RUN THAT WAS NEVER LAUNCHED holds nothing to continue from -- and
        # one opened by a refused send is the very run this launch fills.
        from ..runrecord import LaunchRecordError, launch_record
        from ..runs import run_of
        r = run_of(base / from_attempt)
        try:
            ran = (r is not None and r.run is not None
                   and launch_record(r.folder, r.names, r.run) is not None)
        except LaunchRecordError as exc:
            return f"--from {from_attempt!r}: {exc}"
        if not ran:
            return (f"--from {from_attempt!r}: that run was never launched "
                    f"-- name a run that has run, e.g. '01_coarse/run-0'.")
    # A FORCE-CONSTANT STAGE BUILDS ON ITS LADDER'S `relax`, or on nothing
    # (`engines/vibration.md` § 5.2a's table): a run of `relax` may be
    # named; no other run, and no start from the structure while there is a
    # `relax` to measure at.
    if force_constant_stage(task, stage):
        relax = relax_stage_of(task)
        if relax is None:
            if from_attempt:
                return (f"--from {from_attempt!r}: nothing in this ladder "
                        f"relaxes -- `{stage}` measures the structure as "
                        f"given, stated relaxed (engines/vibration.md "
                        f"5.2a).  To build on a relaxation, add the "
                        f"`relax` stage before it and run it first.")
        elif cold:
            return (f"--cold: `{stage}` builds on `{relax}` -- it measures "
                    f"at the geometry `{relax}` reached.  To measure the "
                    f"structure as given, remove `{relax}` and state the "
                    f"structure relaxed (`already_relaxed` in the "
                    f"template) (engines/vibration.md 5.2a).")
        elif from_attempt and named != relax:
            return (f"--from {from_attempt!r}: `{stage}` builds on "
                    f"`{relax}` -- name one of its runs, "
                    f"<NN>_{relax}/run-<n> (engines/vibration.md 5.2a).")
    return None


def _by_default(calc, stage: str, prev: str, *, verdict: bool,
                linked: bool = False, bench: bool = False
                ) -> Tuple[Optional[Continuation], Optional[str]]:
    """The default for ``stage``, which builds on ``prev`` -- the stage
    before it, or (``linked``) the `relax` a force-constant stage builds on:
    the newest attempt, or the refusal (`continuation_answer`).  A linked
    stage is offered no start from the structure (`engines/vibration.md`
    § 5.2a), and a benchmark no earlier run (it takes no ``--from``)."""
    from ..paths import Shape
    from ..paths import attempt_dir as _adir
    from ..paths import attempts_in
    from .materialize import latest_attempt, stage_home
    base, task, jobset = Path(calc.base), calc.task, calc.jobset
    sh = Shape.named(task.shape)
    # THE STAGE BEFORE'S HOME, from the one door: its names and its folder.
    home = stage_home(base, task, prev)
    container = home.dir
    flat = not sh.keeps_attempts_as_directories

    # THE COMMANDS, from the one composer (`commands`): each names the
    # calculation and states the mode where its config sets none.  A choice
    # of run is said where it is taken: a force-constant stage's at its
    # prep, any other stage's at its launch (`job-system.md` § 5.4).
    from ..identity import stage_key
    from .commands import (block, command, launch_lines, launch_pending,
                           run_first, words_for)
    from .group import number_of
    run_prev = block(run_first(prev, calc=calc))
    g_prev, g_stage = number_of(task, prev), number_of(task, stage)
    # A CHOICE OF RUN A LAUNCH SAYS FOR ONE STAGE (`job-system.md` § 5.4):
    # a member of a group of several, whose first launch takes the group
    # whole, is offered it once it is a group of its own -- a benchmark
    # measures one stage, never a group.
    mates = [s.name for s in task.stages if s.group == g_stage
             and stage_key(s.name) != stage_key(stage)]
    alone = linked or bench or not mates

    def _taking(*flags) -> List[str]:
        if linked:
            return [command("prep", *words_for("task", groups=[g_stage]),
                            base=base, flags=flags)]
        return launch_lines("task", stage, base=base, flags=flags)
    # THE WAY OUT THAT WORKS HERE (the W37 review's first finding): `--cold`
    # names an attempt-less start, which the flat layout -- one folder, no
    # attempts -- refuses; there a stage starts clean by its run card.
    clean = ("" if linked else
             f"or start `{stage}` clean: set its run card's `restart` to "
             f"`clean` (Task setup, or task.json)\n" if flat else
             f"or start `{stage}` from the calculation's structure --\n"
             + block(_taking("--cold")) + "\n" if alone else
             f"or choose what `{stage}` starts from -- an earlier run of "
             f"`{prev}` (--from) or the calculation's structure (--cold): a "
             f"launch says that for one stage, and `{stage}` is in group "
             f"{g_stage} with {', '.join(f'`{m}`' for m in mates)}, so give "
             f"it a group of its own first (Task setup, or task.json) and "
             f"prepare that group\n")
    rule = ("(engines/vibration.md § 5.2a)" if linked
            else "(job-system.md § 5.4)")
    # WHAT IT BUILDS ON, in the words of the ladder's kind.
    lead = (f"`{stage}` builds on `{prev}`" if linked else
            f"`{stage}` continues from the stage before it, `{prev}`")

    run = None
    if flat:
        # PREPARED, by the one door (`ready.job_of`): its row in the plan.
        from .ready import job_of
        if job_of(jobset, prev) is None:
            return None, (f"{lead}, which has not run yet.  Run it "
                          f"first --\n{run_prev}\n{clean}{rule}")
        attempt, source = base, None
        # THE RUN BY ITS NAME, which every file of it carries -- the
        # stage's newest, by the one rule (`runrecord.newest_run`), as its
        # state is read.
        from ..runrecord import newest_run
        names = home.names(task.shape)
        n = newest_run(base, names)
        run = names.run_name(n) if n is not None else None
    else:
        latest = latest_attempt(container) if container.is_dir() else None
        if latest is None:
            return None, (f"{lead}, which has not run yet.  Run it "
                          f"first --\n{run_prev}\n{clean}{rule}")
        attempt, source = latest, str(latest.relative_to(base))
    concluded, state, converged, detail = read_run(
        base, task, prev, attempt, verdict=verdict)
    if usable(state):
        # ON THE FLAT LAYOUT, what it reads is the folder's restart files,
        # as the last run there left them (:func:`flat_files`) -- a record
        # of another run that does not read is the refusal, in its words.
        from ..runrecord import LaunchRecordError
        try:
            return flat_files(base, task, Continuation(
                stage=prev, source=source, by_default=True,
                concluded=concluded, state=state, converged=converged,
                linked=linked, run=run)), None
        except LaunchRecordError as exc:
            return None, str(exc)

    # REFUSED -- worded by what the run's state says, with a command for
    # each way on (§ 5.3: what molbuilder prints, you can type): prep's run
    # not launched goes with its group; a run a refused send opened, and
    # one launched, with the stage alone again (D4).
    what = (f"the newest attempt of `{prev}`, {source}," if source
            else f"`{prev}`'s latest run,")
    launch_prev = block(launch_pending(prev, g_prev,
                                       attempt.name if source else None,
                                       base=base)
                        if state == "pending"
                        else launch_lines("task", prev, base=base))
    why, first = state_remedy(concluded, state, launch_prev,
                              detail=detail)
    other = None
    if not (flat or bench) and alone:
        # AN EARLIER RUN THAT CAN STAND IN WHEN ASKED FOR: the newest one
        # to build on, typed out -- never a placeholder.  Its ending is the
        # question, not its relaxation.
        for n in reversed(attempts_in(container)):
            a = _adir(container, n)
            if a != attempt and usable(read_run(
                    base, task, prev, a, verdict=False)[1]):
                other = str(a.relative_to(base))
                break
    alt = (f"or continue from an earlier run of `{prev}` that finished --\n"
           + block(_taking("--from", other)) + "\n"
           if other else "")
    return None, (f"`{stage}` {'builds on' if linked else 'continues from'} "
                  f"{what} {why}.  {first}\n{alt}{clean}{rule}")


def relaunch(calc, job, *, cold: bool = False, ran=None
             ) -> Optional[Continuation]:
    """What ``job``'s stage, LAUNCHED AGAIN, continues from -- the door
    `launch` asks, as prep asks :func:`continuation_answer`
    (`job-system.md` § 5.4, *A stage launched again*).  Any stage is
    launched again, however its last run ended: warm, from its OWN latest
    run -- ``ran``, its newest attempt launched, as launch found it, or on
    the flat layout its latest run in the folder, by the name every file of
    it carries -- when its kind resumes and it takes something from a run
    (`Job.relaunch_continues`); cold (``cold``), or a stage that takes
    nothing, continues from nothing: ``None``.  How that run ended is read
    in the run door's order (:func:`read_run`) and said, never judged.
    ``calc`` is the calculation as the launch opened it (A16)."""
    from ..paths import Shape
    from ..runrecord import newest_run
    from .materialize import stage_home
    base, task = Path(calc.base), calc.task
    stage = job.name
    if cold or not job.relaunch_continues:
        return None
    home = stage_home(base, task, stage)
    if Shape.named(task.shape).keeps_attempts_as_directories:
        concluded, state, converged, _ = read_run(base, task, stage, ran)
        return Continuation(stage=stage,
                            source=str(Path(ran).relative_to(base)),
                            by_default=True, concluded=concluded, state=state,
                            converged=converged, own=True)
    names = home.names(task.shape)
    n = newest_run(base, names)
    concluded, state, converged, _ = read_run(base, task, stage, base)
    return flat_files(base, task, Continuation(
        stage=stage, source=None, by_default=True, concluded=concluded,
        state=state, converged=converged, own=True,
        run=names.run_name(n) if n is not None else None))


def state_remedy(concluded: Optional[str], state: Optional[str],
                 launch_block: str, *,
                 detail: Optional[str] = None) -> Tuple[str, str]:
    """``(why, what to do first)`` for a run a stage would build on and may
    not: worded by what the run's state says, the way on a command you can
    type (`job-system.md` § 5.3) -- the hand-over's default and a transport
    rung's gather say it alike.  A run that ended with an error, or stopped,
    is LAUNCHED again, never prepared again (`job-system.md` § 5.0, § 5.4)."""
    again = f"Launch it again --\n{launch_block}"
    # A RECORD THAT DOES NOT READ is said as its reader says it -- which
    # names the way out (`jobset migrate`) -- never read as an ending.
    # Its words come last: they end on the command to type.
    if state == "unreadable":
        return "whose records do not read", detail
    # FAILED, IN THE STATUS DOOR'S WORDS -- why, as `status` says it: a
    # run that exited 0 while its output says the engine stopped is one
    # (`usable`)
    if concluded is not None or state == "failed":
        return (f"which failed -- {detail or f'concluded ({concluded})'}",
                again)
    if state == "pending":
        return "which has not been launched", f"Launch it --\n{launch_block}"
    if state in ("queued", "running"):
        return f"which is {state}", "Let it finish"
    return ("which stopped without its conclusion marker -- killed, or out "
            "of time (project-layout.md § 1.6)", again)


def force_constant_stage(task, stage: str) -> bool:
    """A force-constant stage of a SIESTA vibration -- `freq`, or any stage
    after `relax` in a displacement sweep, asked of its role and never of a
    name (`pyscf.stages.vibration_render_kind`, `engines/vibration.md`
    § 5.2a).  A PySCF vibration's one rung relaxes inside its deck."""
    if (str(getattr(task, "engine", "")) != "siesta"
            or getattr(task, "calculation", None) != "vibration"):
        return False
    from ..pyscf.stages import vibration_render_kind
    return vibration_render_kind(stage) == "vibration"


def relax_stage_of(task) -> Optional[str]:
    """The ladder's relaxation rung, which every force-constant stage
    builds on, or ``None`` -- the structure is then measured as given, and
    must be stated relaxed (`engines/vibration.md` § 5.2a).  By the one role
    rule, the name in any case (plan § 5w K12)."""
    from ..pyscf.stages import vibration_render_kind
    return next((s.name for s in task.stages
                 if vibration_render_kind(s.name) != "vibration"), None)


def source_stage(task, stage: str, template_text: Optional[str] = None
                  ) -> Tuple[Optional[str], bool]:
    """``(the stage whose run ``stage`` builds on by default, linked)``: the
    stage before it in an independent ladder (:func:`stage_before`), or the
    `relax` a force-constant stage builds on (linked) -- ``(None, ...)``
    when it builds on none."""
    if force_constant_stage(task, stage):
        return relax_stage_of(task), True
    return stage_before(task, stage, template_text), False


def _independent(task) -> bool:
    """An INDEPENDENT ladder, whose stages continue one from another: a kind
    without rung roles (`paths.KIND_ROLES`).  A linked stage's input is
    its kind's: a transport rung's gather (`gather.gather_sources`),
    and a force-constant stage builds on `relax` (:func:`source_stage`)."""
    from ..paths import KIND_ROLES
    return task.calculation not in KIND_ROLES


def _ladder(task, template_text: Optional[str] = None) -> list:
    """``[(stage, resolved config)]`` of the described stages, as `prep`
    resolves them (`resolve.resolved_ladder`: template ⊕ stage overrides ⊕
    the run card) -- from ``template_text`` as the verb read it (A16);
    ``[]`` with none."""
    from ..resolve import resolved_ladder
    from .engines import engine_seam
    if template_text is None:
        return []
    return resolved_ladder(template_text, task,
                           engine_seam(str(task.engine)).config_cls)


def _stage_config(task, stage: str, template_text: Optional[str] = None):
    """``stage``'s resolved config (:func:`_ladder`), or ``None``."""
    from ..identity import stage_key
    return next((c for n, c in _ladder(task, template_text)
                 if stage_key(n) == stage_key(stage)), None)


def stage_before(task, stage: str,
                  template_text: Optional[str] = None) -> Optional[str]:
    """The stage ``stage`` continues from by default -- the one
    before it in the ladder, resolved as `prep` resolves it (template ⊕
    stage overrides ⊕ the run card, so a card's ``restart: clean`` is read)
    -- or ``None``: a linked kind, the first stage, one that starts clean,
    one the description does not hold (which prep refuses at its checkpoint
    2)."""
    from ..identity import continues
    if not _independent(task):
        return None
    ladder = _ladder(task, template_text)
    names = [n for n, _c in ladder]
    if stage not in names:
        return None
    i = names.index(stage)
    if i == 0 or not continues(ladder[i][1]):
        return None
    return names[i - 1]


def continue_from_choices(calc, stage: str) -> Optional[dict]:
    """What Task setup's **Continue from** offers ``stage`` -- the default
    and, when the verb that takes it would refuse it, why; every run of the
    stage before it with what it was; whether the structure (``--cold``) is
    a choice here -- or ``None`` when the stage continues from nothing by
    default (`job-system.md` § 5.4, `web/task-setup.md` § 11).  The answer
    its first launch acts on -- a force-constant stage's prep -- served
    before it does, without the relaxation's verdict (`continuation_answer`
    reads it): a folder is answered every time it is opened, and parsing
    each stage's output there is the cost of a preview nobody asked for."""
    from ..paths import Shape
    from ..paths import attempt_dir as _adir
    from ..paths import attempts_in
    from .materialize import stage_home
    base, task = Path(calc.base), calc.task
    if force_constant_stage(task, stage) and relax_stage_of(task) is None:
        # NO `relax` TO BUILD ON: the structure as given when it is stated
        # relaxed -- nothing to choose -- else the refusal prep gives, here
        # before anything is written (`engines/vibration.md` § 5.2a).
        _got, why = continuation_answer(calc, stage, verdict=False)
        return ({"from_stage": None, "linked": True, "default": None,
                 "refused": why, "runs": [], "cold": False}
                if why else None)
    prev, linked = source_stage(task, stage, calc.template_text)
    if prev is None:
        return None
    got, refused = _by_default(calc, stage, prev, verdict=False,
                               linked=linked)
    sh = Shape.named(task.shape)
    runs = []
    if sh.keeps_attempts_as_directories:
        container = stage_home(base, task, prev).dir
        for n in (reversed(attempts_in(container)) if container.is_dir()
                  else ()):
            a = _adir(container, n)
            c, s, _v, _d = read_run(base, task, prev, a, verdict=False)
            runs.append({"source": str(a.relative_to(base)),
                         "what": _what_it_is(c, s)})
    return {"from_stage": prev,
            # A FORCE-CONSTANT STAGE builds on its ladder's relaxation, not
            # always the stage before it -- the page's words follow.
            "linked": linked,
            "default": (dict(got.as_dict(), line=got.line())
                        if got is not None else None),
            "refused": refused, "runs": runs,
            # THE STRUCTURE, where the layout keeps attempts and the stage
            # may start from it -- a force-constant stage may not while
            # it builds on `relax` (`engines/vibration.md` § 5.2a).
            "cold": sh.keeps_attempts_as_directories and not linked}


def point_rows(run, names, task, stage: str, *, launch, products, base):
    """A SWEPT RUN'S POINTS AS `status` READS THEM -- `StageStatus.points`'
    rows for ``run``: the folder each ran in, whether it is done, what it
    started from (its `.continued-from`) and what it alone took (its
    `.gathered-from`, beyond the run's own gather).  The one per-point
    reader (`engines/transport.md` § 2a.11): the status door asks it for a
    stage's latest run, the record for the run a transmission gathered
    (§ 2a.12), so the two never read a point two ways.  Returns ``(rows,
    running)``, the second whether any point not done is running."""
    from ..parse.dirs import run_status
    from ..transport.stages import points_in
    from ..runrecord import read_continued_from
    gathered = gathered_lines(run)
    pts, running = [], False
    for pdir, pt in points_in(run, task, stage, base=base):
        ok, why = done(pdir, names, launch=launch, products=products,
                       launched_in=run)
        # ITS RUN'S STATE AND VERDICT, the one status door's -- a point done
        # finished; decided on these, never on the reason's words.
        opened = pdir.is_dir()
        st = (run_status(pdir, names.stem, launch=launch, launched_in=run)
              if opened else None)
        state = ("finished" if ok else st.state if st is not None
                 else "pending")
        converged = (st.verdict[1] if st is not None
                     and st.verdict is not None else None)
        running = running or state == "running"
        unread = None
        try:
            src = (read_continued_from(pdir, names, 0) if pdir.is_dir()
                   else None)
            src = str(src) if src is not None else None
        except LaunchRecordError as exc:
            # A RECORD THAT DOES NOT READ is said on its point's row, in
            # its reader's words -- never read as a start, and never the
            # whole status.
            src, unread = None, str(exc)
        took = [g for g in gathered_lines(pdir) if g not in gathered]
        pts.append({**pt.to_wire(),
                    "point": pt.words(), "folder": f"{run.name}/{pt.rel}",
                    "state": state, "done": ok, "why": why,
                    "opened": opened, "converged": converged,
                    "started_from": src, "start_unread": unread,
                    "took": took})
    return pts, running


def points_summary(pts) -> str:
    """THE COUNT AND ITS AXES, in full every time (`engines/transport.md`
    § 2a.11): *k of N points done (F frames × V voltages)*."""
    n_done = sum(p["done"] for p in pts)
    n_frames = len({p["frame"] for p in pts})
    n_volts = len({p["bias_v"] for p in pts})
    return (f"{n_done} of {len(pts)} points done ({n_frames} "
            f"frame{'' if n_frames == 1 else 's'} × {n_volts} "
            f"voltage{'' if n_volts == 1 else 's'})")


def points_verdict(pts) -> "tuple[str, str]":
    """A run of points' state once nothing in it runs -- THE RUN'S STATE IS
    ITS POINTS' RUNS': finished when every one finished -- converged or not,
    which is the converged column's (`job-system.md`, rule 4) -- else
    failed; the first point not done says why.  The one rule, asked by
    :func:`points_state` and by the transport record for the run its
    transmission gathered (`engines/transport.md` § 2a.12)."""
    summary = points_summary(pts)
    state = ("finished" if all(p["state"] == "finished" for p in pts)
             else "failed")
    first = next((p for p in pts if not p["done"]), None)
    return state, (summary if first is None
                   else f"{summary}; {first['point']}: {first['why']}")


def taken_over(p, here) -> bool:
    """Whether a point was taken over from an earlier run -- its record
    names the run that computed it (one hop, `engines/transport.md`
    § 2a.11), a run outside ``here`` (`runrecord.RunRef`)."""
    from ..runrecord import RunRef
    said = p.get("started_from")
    return bool(said) and not RunRef.read(said, shared=False).within(here)


def _walked_here(pts, run: Path, base: Path) -> list:
    """The points ``run`` walks itself -- every point but those it took
    over from an earlier run (:func:`taken_over`)."""
    from ..runrecord import RunRef
    here = RunRef.at(base, run)
    return [p for p in pts if not (p["done"] and taken_over(p, here))]


def gathered_lines(folder) -> List[str]:
    """``<file> <- <run>`` for each entry of ``folder``'s ``.gathered-from``
    (`runrecord.read_gathered_from`), in the order taken; ``[]`` when there
    is none."""
    from ..runrecord import read_gathered_from
    try:
        return [f"{fn} <- {src}" for src, fn in read_gathered_from(folder)]
    except LaunchRecordError as exc:
        # SAID ON ITS ROW, in its reader's words -- never the whole status.
        return [str(exc)]


def points_state(run, names, task, stage: str, *, launch, products, base
                 ) -> "Tuple[str, str, list]":
    """A LAUNCHED RUN OF POINTS' STATE -- ``(state, detail, rows)``, the one
    rule the status door and the gather ask (`engines/transport.md`
    § 2a.11): ``running`` while a point runs, or between two points while a
    job launched from this machine that holds the run is going
    (`joblist.is_going`); ``queued`` while nothing the run walks itself has
    started; else the points' verdict (:func:`points_verdict`) --
    ``finished``, or ``failed`` naming the first point not done, a chain
    that stopped among them: a point never reached is no point queued."""
    pts, running = point_rows(run, names, task, stage, launch=launch,
                              products=products, base=base)
    summary = points_summary(pts)
    if running:
        return "running", summary, pts
    if all(p["opened"] and p["state"] in ("pending", "queued")
           for p in _walked_here(pts, run, base)):
        # NOTHING THIS RUN WALKS HAS RUN YET -- the points it took over
        # from an earlier run are that run's, done before it was sent --
        # unless its job is over: cancelled in the queue, or a process
        # that died before its first point.
        if not _ended_before_a_point(base, run, names):
            jid = (launch or {}).get("job_id")
            return "queued", (f"queued as job {jid}" if jid
                              else "launched, nothing run yet"), pts
        state, detail = points_verdict(pts)
        return state, (detail + " -- its job ended before any point "
                       "started"), pts
    if any(not p["done"] for p in pts) and _walk_going(base, run):
        return "running", summary + " (between two points)", pts
    state, detail = points_verdict(pts)
    return state, detail, pts


def _ended_before_a_point(base, run, names) -> bool:
    """Whether a run of points none of whose points started has ended --
    asked once of what knows: a job launched from this machine that holds
    the run (`joblist.is_going`), else the run door, which asks the
    scheduler where the job was sent from here and writes its answer into
    the launch record (`runs.concluded`, `running-a-job.md` § 4.2);
    ``False`` where neither can tell."""
    from ..joblist import Asked, Job, every, is_going
    from ..parse.dirs.job import LIVE_STATES
    from ..paths import run_folder_of
    from ..runrecord import RunRef
    from ..runs import concluded
    want = RunRef.at(base, run)
    asked = Asked()
    for job in every():
        if (not isinstance(job, Job)
                or Path(job.calculation) != Path(base).resolve()):
            continue
        if any(str(run_folder_of(m["run"]) or m["run"]) == str(want)
               for m in job.members.values()):
            going = is_going(job, asked)[0]
            if going is not None:
                return not going
    try:
        return concluded(run, names, base=base,
                         launched_in=run).state not in LIVE_STATES
    except (OSError, ValueError):
        return False


def stage_run_state(base, task, stage: str, run) -> "Tuple[str, str]":
    """``(state, detail)`` of one run of ``stage`` -- THE ONE DOOR for a
    run's state, whichever kind: a run of points by :func:`points_state`,
    any other run by the run door (:func:`read_run`).  Asked by `jobset
    scratch` of each run that holds an engine's scratch."""
    from ..runrecord import launch_record
    from ..transport.stages import frames_of, products_of, rung_points
    from .materialize import stage_home
    names = stage_home(base, task, stage).names(task.shape)
    if rung_points(task, stage, frames=frames_of(base)):
        launch = launch_record(run, names)
        if launch is None:
            return "pending", "prepared, not launched"
        state, detail, _rows = points_state(
            run, names, task, stage, launch=launch,
            products=products_of(stage, task.label, base_dir=base),
            base=base)
        return state, detail
    _ended, state, _conv, detail = read_run(base, task, stage, run,
                                            verdict=False)
    return state, detail


def _walk_going(base, run) -> bool:
    """Whether a job launched from this machine that holds ``run`` is
    going -- its process, or its scheduler's list (`joblist.is_going`, the
    one answer `molbuilder jobs` asks); ``False`` where none is on file or
    this machine cannot tell, the points' verdict then said."""
    from ..joblist import Asked, Job, every, is_going
    from ..paths import run_folder_of
    from ..runrecord import RunRef
    want = RunRef.at(base, run)
    asked = Asked()
    for job in every():
        if (not isinstance(job, Job)
                or Path(job.calculation) != Path(base).resolve()):
            continue
        holds = any(str(run_folder_of(m["run"]) or m["run"]) == str(want)
                    for m in job.members.values())
        if holds and is_going(job, asked)[0] is True:
            return True
    return False
