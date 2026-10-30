"""What a stage continues from -- which run, and what that run was
(`execution/job-system.md` § 5.4, plan W37): the stage before it in an
independent ladder.

**Module:** L2 (jobset).  ONE answer, asked by `launch` at a stage's first
launch (`job-system.md` § 6.0) -- the default, or what a run named by
``--from`` is -- and by `status` through the ready door, so none says a
different thing about the same run.
Reads the folder; writes
nothing; raises nothing the person can fix -- a refusal is an answer, which
`launch` turns into its own error and `status` prints; a bug raises.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

from ..calcdirs import RunPlace
from ..runrecord import LaunchRecordError
from .ladder import independent, inputs_of, stage_before


@dataclasses.dataclass(frozen=True)
class Continuation:
    """WHICH RUN A STAGE CONTINUES FROM, and what that run was -- printed by
    the launch that takes it and recorded in the decision ledger."""
    #: The stage the run belongs to.
    stage: str
    #: Its attempt, from the calculation folder (``01_coarse/run-0``).
    source: str
    #: The stage before it, newest (the default) -- or named by ``--from``.
    by_default: bool
    #: The line its conclusion said (`runrecord.ending`), or ``None``: the
    #: run has not ended on its own.
    concluded: Optional[str]
    #: Its run state, `run_status`'s word (finished, failed, running, ...).
    state: Optional[str]
    #: The relaxation's verdict, ``None`` when the run relaxed nothing.
    converged: Optional[bool]
    #: The files it carries into the attempt, by the pair's rule
    #: (`model.warm_carry`) of what the run holds -- counted against the
    #: stage's job, ``None`` until then.
    carries: Optional[Tuple[str, ...]] = None
    #: The stage's OWN latest run: a stage launched again (:func:`relaunch`;
    #: `job-system.md` § 5.4, *A stage launched again*).
    own: bool = False

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
                  "the stage before it"),
                 _what_it_is(self.concluded, self.state)]
        if self.converged is not None:
            facts.append("converged" if self.converged else
                         "NOT converged -- taken as it stands")
        return (f"continues from {self.source} ({'; '.join(facts)})"
                + (f": {'would copy' if would else 'copied'} "
                   + ", ".join(n + (f" -- WARNING: {warned[n]}"
                                    if n in warned else "")
                               for n in copied) if copied else ""))

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


def usable(state: Optional[str]) -> bool:
    """THE RULE FOR A RUN TO BUILD ON (`execution/architecture.md` § 3.2,
    `job-system.md` § 5.4): the one status door says it FINISHED -- the
    door `status` asks (`parse.dirs.run_status`, read by :func:`read_run`),
    from every fact of the run: it ended on its own with exit code 0, and
    nothing in its output says the engine stopped (user, 2026-10-06: "the
    handover should call a unified api that tells the true status of
    previous stage that views all facts from exit code, log etc.").  The
    default of every hand-over -- a continuing stage, a transport rung's
    inputs; a run named with ``--from`` is taken as said, so it does not
    ask it."""
    return state == "finished"


def done(folder, names, *, launch=None, must_hold=(),
         launched_in=None, verdict_counts: bool = True) -> Tuple[bool, str]:
    """THE DONE-DOOR (`engines/transport.md` § 2a.11): whether the run, or a
    swept run's point, in ``folder`` is DONE -- ``(True, "done")`` -- or what
    it is instead, in the run door's words.  Done is three facts, all read
    from the folder's own files: it **finished** (the one status door,
    :func:`usable`); its run **converged** where it has an SCF or a
    relaxation (the same scan, `RunStatus.verdict`); and it **holds**
    what it must -- ``must_hold``, ``[(product, words)]``, its kind's
    declaration's (`points.PointsOf.must_hold`; a transport rung's,
    `transport.stages.must_hold`): what the rungs downstream take from it,
    and where it shares energies, its ``shared/`` with its ``.copied-from``
    (`engines/transport.md` § 2a.16).  ``launch`` is
    the launch record the folder's run was sent with (a point's is its
    run's), and ``launched_in`` the folder it lies in when not ``folder``
    (a point's run's folder; `run_status`).  ``verdict_counts`` False where
    the kind says a point's verdict is not its done -- a vibration's
    relaxation call, done finished, converged or not
    (`points.PointsOf.verdict_counts`).  A point whose own step failed
    concluded failed, so finished says it (`execution/runs-of-points.md`
    § 4.1).  What a finished point lacks is said in the one rule's words
    (`_run_ending.not_done_why`), which the job's side asks too
    (`runpoints.rows_of`)."""
    from ..parse.dirs import run_status
    from ..parse.engines._run_ending import holds, not_done_why
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
    why = not_done_why(st.verdict, counts=verdict_counts,
                       missing=[words for product, words in must_hold
                                if not holds(folder, product)])
    if why:
        return False, why
    return True, "done"


def read_run(base: Path, task, stage: str, attempt: Path,
             *, verdict: bool = True
             ) -> Tuple[Optional[str], Optional[str], Optional[bool],
                        Optional[str]]:
    """``(concluded, state, converged, detail)`` of one run of ``stage``
    -- ``detail`` the status door's own words for the state -- the
    line its conclusion said, through the one door (`runrecord.ending`;
    ``None`` when it has not ended on its own), its state through the one
    door (`parse.dirs.run_status`) and its relaxation's verdict
    (`parse.contract.relaxation_of`), read from the run's own files, the
    run door's (`runs.run_of`): its engine output at the run's newest try,
    then its progress log -- where a PySCF run writes its steps, which its
    stdout cannot stand in for.  The try is the newest in ``attempt`` by
    the one rule (`runrecord.newest_run`).  ``verdict=False``
    skips the relaxation's parse -- a list of runs needs each one's
    conclusion, not a parse of each output."""
    from ..parse.contract import relaxation_of
    from ..runs import concluded as _concluded, run_of
    from ..runrecord import ending, newest_run
    from .naming import stage_home
    home = stage_home(base, task, stage)
    names = home.names
    run = newest_run(attempt, names)
    concluded = ending(attempt, names.stem, run).line
    # ITS STATE, through the one door, asked with its launch record -- a
    # record that does not read is the state `unreadable`, said, never taken
    # for a run that stopped (`runs.concluded`, which asks the scheduler
    # once where it can, `running-a-job.md` § 4.2).
    st = _concluded(attempt, names, base=base, run=run)
    state, detail = st.state, st.detail
    rec = None
    if verdict:
        # THIS STAGE'S RUN, the run door's, read in the order the run door
        # reads a run (`runfiles.result_roles`): its engine output at its
        # newest try, then its progress log, which a SIESTA run never
        # writes into and a PySCF run's output no parser reads.
        _run = run_of(attempt)
        paths = ([_run.stdout, _run.file(".molwatch.log")]
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

    First, what cannot be taken at all is refused, before anything is
    written (:func:`cannot_be_named`): a run that is not one of this
    calculation's, ``--from`` with ``--cold``, and ``--from`` for a kind
    whose stages take their input from the kind (transport, a vibration).
    **Named** (``--from``): taken as said -- what
    the run was is read and reported; `prepare_attempt` refuses only what
    cannot be done (no restart files in it).  **By default**, for a
    continuing stage of an independent ladder (`stage_before`): the NEWEST
    attempt of the stage before it, which must have ended on its own
    with exit code 0 (:func:`usable`) -- an older one never stands in,
    because a stage re-launched to tighten is the run the person means.
    Otherwise the refusal, worded by what the run's state says and naming
    the commands that work.  ``(None, None)`` for ``--cold``, a stage that builds on no
    stage before it, the first stage, a stage that starts clean, and one
    the description does not hold (which prep refuses at its checkpoint 2).
    ``verdict=False`` leaves the relaxation unread -- for a reader that
    never prints it.

    ``calc`` is the calculation as the verb read it (A16) -- prep's, with
    its plan as prep's overlay leaves it (`Calculation.as_planned`).
    ``bench`` asks for a benchmark of the stage, which takes no ``--from``
    and is offered none."""
    from ..identity import command_stage
    base, task = Path(calc.base), calc.task
    template_text = calc.template_text
    refused = cannot_be_named(base, task, stage, from_attempt, cold)
    if refused:
        return None, refused
    if cold:
        return None, None
    if from_attempt:
        # A RUN NAMED is taken as said -- what it was is reported and
        # recorded.
        attempt = base / from_attempt
        named = command_stage(Path(from_attempt).parts[0])
        concluded, state, converged, _ = read_run(
            base, task, named, attempt, verdict=verdict)
        return Continuation(stage=named, source=str(Path(from_attempt)),
                            by_default=False, concluded=concluded, state=state,
                            converged=converged), None
    try:
        prev = stage_before(task, stage, template_text)
        if prev is None:
            return None, None
        return _by_default(calc, stage, prev, verdict=verdict, bench=bench)
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
    if inputs_of(task) is not None:
        # A LINKED RUNG'S INPUTS ARE ITS KIND'S: gathered from the rungs
        # upstream (the gather its pair names), never named.
        from ..kind_table import kind
        return (f"--from / --cold name what a stage continues from; "
                f"{kind(task.calculation).with_article}'s rung takes its "
                f"inputs from the rungs upstream, gathered by its kind at "
                f"its first launch, and anew with --gather when it is "
                f"launched again (job-system.md § 5.4).")
    if not independent(task):
        # ANY OTHER LINKED STAGE -- a vibration's, one run in phases on
        # SIESTA, PySCF's one deck -- builds on the calculation's structure;
        # a run of another stage is not its input (a `freq` run's `.XV` is
        # a displaced geometry), and a first launch has nothing to start
        # cold from (`--cold` is a launch again's).
        return (f"{'--from ' + repr(from_attempt) if from_attempt else '--cold'}"
                f": `{stage}` builds on the calculation's structure -- no "
                f"run of another stage is its input, and its first launch "
                f"takes nothing to start cold from (job-system.md § 5.4).")
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
            ran = r is not None and launch_record(r.folder) is not None
        except LaunchRecordError as exc:
            return f"--from {from_attempt!r}: {exc}"
        if not ran:
            return (f"--from {from_attempt!r}: that run was never launched "
                    f"-- name a run that has run, e.g. '01_coarse/run-0'.")
    return None


def _by_default(calc, stage: str, prev: str, *, verdict: bool,
                bench: bool = False
                ) -> Tuple[Optional[Continuation], Optional[str]]:
    """The default for ``stage``, which continues from ``prev``, the stage
    before it: the newest attempt, or the refusal (`continuation_answer`).
    A benchmark is offered no earlier run (it takes no ``--from``)."""
    from ..paths import attempt_dir as _adir
    from ..paths import attempts_in
    from .materialize import latest_attempt
    from .naming import stage_home
    base, task = Path(calc.base), calc.task
    # THE STAGE BEFORE'S FOLDER, from the one door.
    container = stage_home(base, task, prev).dir

    # THE COMMANDS, from the one composer (`commands`): each names the
    # calculation and states the mode where its config sets none.  A choice
    # of run is said where it is taken, at the stage's launch
    # (`job-system.md` § 5.4).
    from ..identity import stage_key
    from .commands import block, launch_lines, launch_pending, run_first
    run_prev = block(run_first(prev, calc=calc))
    g_prev = calc.request.group_of(prev)
    g_stage = calc.request.group_of(stage)
    # A CHOICE OF RUN A LAUNCH SAYS FOR ONE STAGE (`job-system.md` § 5.4):
    # a member of a group of several, whose first launch takes the group
    # whole, is offered it once it is a group of its own -- a benchmark
    # measures one stage, never a group.
    mates = [n for n in calc.request.members(g_stage)
             if stage_key(n) != stage_key(stage)]
    alone = bench or not mates

    def _taking(*flags) -> List[str]:
        return launch_lines("task", stage, base=base, flags=flags)
    # THE WAY OUT THAT WORKS HERE (the W37 review's first finding).
    clean = (f"or start `{stage}` from the calculation's structure --\n"
             + block(_taking("--cold")) + "\n" if alone else
             f"or choose what `{stage}` starts from -- an earlier run of "
             f"`{prev}` (--from) or the calculation's structure (--cold): a "
             f"launch says that for one stage, and `{stage}` is in group "
             f"{g_stage} with {', '.join(f'`{m}`' for m in mates)}, so give "
             f"it a group of its own first (Task setup, or job-request.json) and "
             f"prepare that group\n")
    rule = "(job-system.md § 5.4)"
    lead = f"`{stage}` continues from the stage before it, `{prev}`"

    attempt = latest_attempt(container) if container.is_dir() else None
    if attempt is None:
        return None, (f"{lead}, which has not run yet.  Run it "
                      f"first --\n{run_prev}\n{clean}{rule}")
    source = str(attempt.relative_to(base))
    concluded, state, converged, detail = read_run(
        base, task, prev, attempt, verdict=verdict)
    if usable(state):
        return Continuation(stage=prev, source=source, by_default=True,
                            concluded=concluded, state=state,
                            converged=converged), None

    # REFUSED -- worded by what the run's state says, with a command for
    # each way on (§ 5.3: what molbuilder prints, you can type): prep's run
    # not launched goes with its group; a run a refused send opened, and
    # one launched, with the stage alone again (D4).
    what = f"the newest attempt of `{prev}`, {source},"
    launch_prev = block(launch_pending(prev, g_prev, attempt.name, base=base)
                        if state == "pending"
                        else launch_lines("task", prev, base=base))
    why, first = state_remedy(concluded, state, launch_prev,
                              detail=detail)
    other = None
    if not bench and alone:
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
    return None, (f"`{stage}` continues from {what} {why}.  {first}\n"
                  f"{alt}{clean}{rule}")


def relaunch(calc, job, *, ran, cold: bool = False
             ) -> Optional[Continuation]:
    """What ``job``'s stage, LAUNCHED AGAIN, continues from -- the door
    `launch` asks, as a first launch asks :func:`continuation_answer`
    (`job-system.md` § 5.4, *A stage launched again*).  Any stage is
    launched again, however its last run ended: warm, from its OWN latest
    run -- ``ran``, its newest attempt launched, as launch found it -- when
    its kind resumes and it takes something from a run
    (`Job.relaunch_continues`); cold (``cold``), or a stage that takes
    nothing, continues from nothing: ``None``.  How that run ended is read
    in the run door's order (:func:`read_run`) and said, never judged.
    ``calc`` is the calculation as the launch opened it (A16)."""
    base, task = Path(calc.base), calc.task
    stage = job.name
    if cold or not job.relaunch_continues:
        return None
    concluded, state, converged, _ = read_run(base, task, stage, ran)
    return Continuation(stage=stage, source=str(Path(ran).relative_to(base)),
                        by_default=True, concluded=concluded, state=state,
                        converged=converged, own=True)


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


def continue_from_choices(calc, stage: str) -> Optional[dict]:
    """What Task setup's **Continue from** offers ``stage`` -- the default
    and, when the verb that takes it would refuse it, why; every run of the
    stage before it with what it was; whether the structure (``--cold``) is
    a choice here -- or ``None`` when the stage continues from nothing by
    default (`job-system.md` § 5.4, `web/task-setup.md` § 11).  The answer
    its first launch acts on -- served before it does, without the
    relaxation's verdict (`continuation_answer` reads it): a folder is
    answered every time it is opened, and parsing
    each stage's output there is the cost of a preview nobody asked for."""
    from ..paths import attempt_dir as _adir
    from ..paths import attempts_in
    from .naming import stage_home
    base, task = Path(calc.base), calc.task
    prev = stage_before(task, stage, calc.template_text)
    if prev is None:
        return None
    got, refused = _by_default(calc, stage, prev, verdict=False)
    runs = []
    container = stage_home(base, task, prev).dir
    for n in (reversed(attempts_in(container)) if container.is_dir()
              else ()):
        a = _adir(container, n)
        c, s, _v, _d = read_run(base, task, prev, a, verdict=False)
        runs.append({"source": str(a.relative_to(base)),
                     "what": _what_it_is(c, s)})
    return {"from_stage": prev,
            "default": (dict(got.as_dict(), line=got.line())
                        if got is not None else None),
            "refused": refused, "runs": runs,
            # THE STRUCTURE, a choice for every stage that continues: a
            # launch's --cold opens its run with nothing taken.
            "cold": True}


def point_done(declared, run, pdir, pt, names, *, task, stage: str, base,
               launch) -> Tuple[bool, str]:
    """Whether the point ``pt`` of a run of points, in ``pdir`` below
    ``run``, is DONE -- the done door (:func:`done`) asked with its kind's
    declaration (`points.PointsOf`): what the point must hold, whether its
    verdict counts (`execution/runs-of-points.md` § 4.2).  The one question
    status (:func:`point_rows`) and launch's walk ask of a point."""
    return done(pdir, names, launch=launch, launched_in=run,
                must_hold=declared.must_hold(stage, task.label, base, pt,
                                             pdir),
                verdict_counts=declared.verdict_counts(stage, pt))


def point_rows(run, names, task, stage: str, *, launch, base, index=None):
    """A SWEPT RUN'S POINTS AS `status` READS THEM -- `StageStatus.points`'
    rows for ``run``: the folder each ran in, whether it is done, what it
    started from (its `.continued-from`), what it is a copy of -- a point
    taken over from the finished calculation this one reuses, or of a run
    copied in whole (`runrecord.copied_of`, `engines/transport.md`
    § 2a.16) -- and what it alone took (its `.gathered-from`, beyond the
    run's own gather).  The one per-point
    reader (`engines/transport.md` § 2a.11): the status door asks it for a
    stage's latest run, the record for the run a transmission gathered
    (§ 2a.12), so the two never read a point two ways.  Each row in the
    job's own shape too (`runpoints.rows_of`, `execution/runs-of-points.md`
    § 4.7): ``rel``, ``words``, ``phase``, ``round``, ``needs`` -- launch's,
    from the run's index (``index``, read here when not handed in) --
    ``ended`` and ``ran``, so one rule answers both sides.  Returns
    ``(rows, running)``, the second whether any point not done is
    running."""
    from ..parse.dirs.job import FINAL_STATES
    from ..parse.engines._run_ending import verdict_words
    from ..runpoints import entry_of, index_at
    from ..runs import concluded
    from ..runrecord import copied_of, read_continued_from
    from .points import points_of
    declared = points_of(task)
    gathered = gathered_lines(run)
    if index is None:
        index = index_at(run, names) or {}
    pts, running = [], False
    for pdir, pt in declared.points_in(run, task, stage, base):
        ok, why = point_done(declared, run, pdir, pt, names, task=task,
                             stage=stage, base=base, launch=launch)
        # ITS RUN'S STATE AND VERDICT, through the one door every reader of
        # a run's state asks (`runs.concluded`: the run's record, and the
        # scheduler where this machine sent its job, its answer written
        # once) -- a point done finished; decided on these, never on the
        # reason's words.
        opened = pdir.is_dir()
        st = (concluded(pdir, names, base=base, launched_in=run)
              if opened else None)
        state = ("finished" if ok else st.state if st is not None
                 else "pending")
        verdict = st.verdict if st is not None else None
        converged = verdict[1] if verdict is not None else None
        running = running or state == "running"
        unread = None
        try:
            src = read_continued_from(pdir) if pdir.is_dir() else None
            src = str(src) if src is not None else None
        except LaunchRecordError as exc:
            # A RECORD THAT DOES NOT READ is said on its point's row, in
            # its reader's words -- never read as a start, and never the
            # whole status.
            src, unread = None, str(exc)
        try:
            copied = copied_of(base, pdir) if opened else None
        except LaunchRecordError as exc:
            copied, unread = None, str(exc)
        took = [g for g in gathered_lines(pdir) if g not in gathered]
        entry = entry_of(index, pt.rel) or {}
        pts.append({**pt.to_wire(),
                    # THE JOB'S SHAPE (`runpoints.rows_of`): which point,
                    # its words, phase, round and needs as launch listed it.
                    "rel": pt.rel, "words": pt.words(),
                    "phase": entry.get("phase"), "round": entry.get("round"),
                    "needs": list(entry.get("needs") or ()),
                    "ended": state in FINAL_STATES,
                    "ran": state == "finished", "not_run": None,
                    "point": pt.words(),
                    "folder": RunPlace.of(run).walked(pt.rel),
                    "state": state, "done": ok, "why": why,
                    # ITS RUN'S STATE IN THE STATUS DOOR'S WORDS.
                    "detail": st.detail if st is not None else None,
                    "opened": opened, "converged": converged,
                    # THE VERDICT IN ITS OWN WORDS -- `geometry NO` for a
                    # relaxation, `SCF NO` for an SCF (`job-system.md`,
                    # rule 4).
                    "verdict_words": verdict_words(verdict),
                    "started_from": src, "start_unread": unread,
                    # WHAT LAUNCH TOOK IT OVER FROM, its run's own index
                    # (`runpoints.taken_from`).
                    "taken_from": entry.get("taken_from") or None,
                    "copied_from": copied, "took": took})
    return pts, running


def points_verdict(pts, summary: str, *, record, index
                   ) -> "tuple[str, str]":
    """A run of points' state once nothing in it runs -- the job's own rule
    (`runpoints.run_state`, `execution/runs-of-points.md` § 4.4) over
    `status`'s rows: by its kind's ``finished`` (``record``, the kind
    table's), every point done or every point ran; a finish step that
    exited non-zero (the run's ``index``), failed.  Its words after
    ``summary``, the count in the kind's words.  Asked by
    :func:`points_state` and by the transport record for the run its
    transmission gathered (`engines/transport.md` § 2a.12)."""
    from ..runpoints import run_state
    state, why = run_state(pts, {"finished": record.finished,
                                 "unit": record.unit,
                                 "finish": (index or {}).get("finish")})
    return state, (f"{summary}; {why}" if why else summary)


def count_of(task, run, names, pts, index=None) -> str:
    """A RUN OF POINTS' COUNT in its kind's words -- the kind table's
    record's ``count`` part over `status`'s rows and the facts launch wrote
    into the run's index (``index``, read here when not handed in), the
    job's count's own words (`execution/runs-of-points.md` § 4.7)."""
    from ..kind_table import part
    from ..runpoints import index_at
    from .points import record_of
    rec = record_of(task)
    if index is None:
        index = index_at(run, names) or {}
    facts = index.get("facts") or {}
    return part(rec.count, record=f"{task.calculation} on {task.engine}",
                key="points.count")(pts, facts)


def taken_over(p) -> bool:
    """Whether a point was taken over from an earlier run -- its run's own
    index names the run that computed it (one hop, `engines/transport.md`
    § 2a.11; :func:`point_rows`' ``taken_from``), as launch decided it: a
    `relax` call continued warm from its own last run ran here, and is not
    (`engines/vibration.md` § 5.11 § 6)."""
    return bool(p.get("taken_from"))


def _walked_here(pts) -> list:
    """The points a run walks itself -- every point but those it took
    over from an earlier run (:func:`taken_over`) and those copies of the
    finished calculation's points (`engines/transport.md` § 2a.16): taken
    over from it, or a run copied in whole, which walks none."""
    return [p for p in pts if not (p["done"] and (taken_over(p)
                                                  or p.get("copied_from")))]


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



def never_reached(pts) -> list:
    """``pts`` of a run of points that has ended, each point still pending
    or queued -- one its walk never ran -- failed, its why the job's own
    (`runpoints.why_not_run`, `execution/runs-of-points.md` § 4.5): *it
    builds on <words>, which failed*, or `runpoints.NEVER_REACHED`."""
    from ..runpoints import why_not_run
    pts = list(pts)
    out = []
    for p in pts:
        if p["state"] in ("pending", "queued"):
            why = why_not_run(pts, p["rel"])
            p = dict(p, state="failed", ended=True, ran=False, why=why,
                     not_run=why)
        out.append(p)
    return out


def points_state(run, names, task, stage: str, *, launch, base
                 ) -> "Tuple[str, str, list]":
    """A LAUNCHED RUN OF POINTS' STATE -- ``(state, detail, rows)``, the one
    rule the status door and the gather ask (`engines/transport.md`
    § 2a.11): ``running`` while a point runs, or between two points while a
    job launched from this machine that holds the run is going
    (`joblist.is_going`); ``queued`` while nothing the run walks itself has
    started (a run that walks none -- a run copied in whole -- is read by
    its points); else the points' verdict (:func:`points_verdict`) --
    ``finished``, or ``failed`` naming the first point not done, a chain
    that stopped among them: a point never reached is failed
    (:func:`never_reached`), no point queued.
    A run the job the reader runs inside sent (:func:`_own_job_s`) is read
    by its points alone: its walk has returned, and the job, the reader's
    own, is still going (`job-system.md` § 6.0, *A sequence of groups*)."""
    from ..runpoints import index_at
    from .points import points_of, record_of
    declared, record = points_of(task), record_of(task)
    index = index_at(run, names) or {}
    pts, running = point_rows(run, names, task, stage, launch=launch,
                              base=base, index=index)
    summary = count_of(task, run, names, pts, index)
    if running:
        return "running", summary, pts
    walked = _walked_here(pts)
    if walked and all(p["opened"] and p["state"] in ("pending", "queued")
                      for p in walked):
        # NOTHING THIS RUN WALKS HAS RUN YET -- the points it took over
        # from an earlier run are that run's, done before it was sent --
        # unless its job is over: cancelled in the queue, or a process
        # that died before its first point.
        own = _own_job_s(launch)
        if not own and not _ended_before_a_point(base, run, names):
            jid = (launch or {}).get("job_id")
            return "queued", (f"queued as job {jid}" if jid
                              else "launched, nothing run yet"), pts
        pts = never_reached(pts)
        state, detail = points_verdict(pts, count_of(task, run, names, pts,
                                                     index),
                                       record=record, index=index)
        return state, (detail + (" -- its walk ended before any point "
                                 "started" if own else
                                 " -- its job ended before any point "
                                 "started")), pts
    if (any(not p["done"] for p in pts) and not _own_job_s(launch)
            and _walk_going(base, run)):
        return "running", summary + " (between two points)", pts
    # THE WALK HAS ENDED: a point it never reached is failed.
    pts = never_reached(pts)
    summary = count_of(task, run, names, pts, index)
    state, detail = points_verdict(pts, summary, record=record, index=index)
    # WHAT THE RUN NEEDS BEYOND ITS POINTS, by its kind (`points.PointsOf.
    # also_needs`): a vibration's finish, which writes the result after its
    # last call (`engines/vibration.md` § 5.11 § 7) -- still running while
    # the job that holds the run is going (its last call's record step, or
    # the finish, at work).
    if state == "finished":
        why = declared.also_needs(run, names, task, stage, base)
        if why:
            if not _own_job_s(launch) and _walk_going(base, run):
                return "running", summary + " (after its last point)", pts
            return "failed", f"{detail}; {why}", pts
    return state, detail, pts


def _own_job_s(launch) -> bool:
    """Whether the job the reader runs inside (`joblist.inside_job`) sent
    the run whose launch record is ``launch`` -- its sequence by name, and
    its job by id where a queue gave one (run here, none: two going jobs of
    one name cannot be, their groups held): then its walk has returned
    before the reader reads it, and nothing asks the job, the reader's
    own."""
    from ..joblist import inside_job
    got = inside_job()
    if got is None or not isinstance(launch, dict):
        return False
    job, name = got
    return (launch.get("sequence") == name
            and launch.get("job_id") in (None, job))


def _ended_before_a_point(base, run, names) -> bool:
    """Whether a run of points none of whose points started has ended --
    asked once of what knows: a run here launched from this machine that
    holds the run, by its process (`joblist.is_going`); else the run door,
    which asks the scheduler where this machine sent the job and writes its
    answer into the launch record (`runs.concluded`, `running-a-job.md`
    § 4.2) -- a queue job's answer written by this very look, so the run
    reads ended once its address is gone; ``False`` where neither can
    tell."""
    from ..joblist import Asked, is_going
    from ..parse.dirs.job import LIVE_STATES
    from ..runs import concluded
    asked = Asked()
    for job in jobs_holding(base, run):
        if job.how != "here":
            continue
        going = is_going(job, asked)[0]
        if going is not None:
            return not going
    try:
        return concluded(run, names, base=base,
                         launched_in=run).state not in LIVE_STATES
    except (OSError, ValueError):
        return False


def stage_run_state(calc, stage: str, run) -> "Tuple[str, str]":
    """``(state, detail)`` of one run of ``stage`` of ``calc`` -- THE ONE
    DOOR for a run's state, whichever kind: a run of points by
    :func:`points_state`, any other run by the run door (:func:`read_run`).
    Asked by `jobset scratch` of each run that holds an engine's scratch."""
    from ..runrecord import launch_record
    from .naming import stage_home
    from .points import points_of
    base, task = calc.base, calc.task
    names = stage_home(base, task, stage).names
    declared = points_of(task)
    if declared is not None and declared.rung_points(calc, stage):
        launch = launch_record(run)
        if launch is None:
            return "pending", "prepared, not launched"
        state, detail, _rows = points_state(
            run, names, task, stage, launch=launch, base=base)
        return state, detail
    _ended, state, _conv, detail = read_run(base, task, stage, run,
                                            verdict=False)
    return state, detail


def _walk_going(base, run) -> bool:
    """Whether a job launched from this machine that holds ``run`` is
    going -- its process, or its scheduler's list (`joblist.is_going`, the
    one answer `molbuilder jobs` asks); ``False`` where none is on file or
    this machine cannot tell, the points' verdict then said."""
    from ..joblist import Asked, is_going
    asked = Asked()
    return any(is_going(job, asked)[0] is True
               for job in jobs_holding(base, run))


def jobs_holding(base, run):
    """The jobs launched from this machine (`joblist.every`) that hold
    ``run`` of the calculation in ``base`` -- a member of each runs in it,
    or in a point of it (`paths.run_folder_of`)."""
    from ..joblist import Job, every
    from ..paths import run_folder_of
    from ..runrecord import RunRef
    want, here = str(RunRef.at(base, run)), Path(base).resolve()
    for job in every():
        if (isinstance(job, Job) and Path(job.calculation) == here
                and any(str(run_folder_of(m["run"])) == want
                        for m in job.members.values())):
            yield job
