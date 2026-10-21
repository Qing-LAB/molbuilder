"""THE READY DOOR -- where a stage and its group stand between the
description and a run, and what a stage's first launch takes
(`job-system.md`, *The task*; `architecture.md` § 3.2).

**A stage is described, prepared or launched**, each answered here from the
doors that already decide it.  **Described** -- not prepared: preparable,
but a SIESTA force-constant stage while the `relax` run its deck would be
written at has not finished (`continuation.continuation_answer`, the one
value hand-over, `engines/vibration.md` § 5.2a).  **Prepared**, never
launched: **ready to launch** when what its first launch takes is there,
else **waits for** what it names -- the door that decides what it takes, the
same one launch asks: an independent stage's continuation (the stage before
it, its newest run finished), a transport rung's gather
(`gather.gather_sources`, its three gates), a force-constant stage's `relax`
run its prep took (`Job.written_at`).  **Launched**: the run doors answer.
Nothing here decides a second way, and nothing is written.

**A group** is the stages one number holds (`engines/stages.md` § 2),
matched to what is prepared by its members, never its number: prepared as
described when every member's job holds exactly those members.

Readers: `prep task`'s groups and offer, `launch task`'s gate and offer,
`status`'s rows and next step, Task setup's Prep and its group tabs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Readiness:
    """One stage's answer.  ``ready`` is the question its state asks: a
    stage not prepared, whether it can be prepared now; one prepared and
    never launched, whether it can be launched now -- with what it takes,
    or ``why`` it waits."""
    stage: str
    group: Optional[int] = None
    prepared: bool = False
    launched: bool = False
    ready: bool = False
    #: What it takes, one line each: for a stage not prepared, what its
    #: first launch will take, said before any run is read (a force-constant
    #: stage, the `relax` run its deck would be written at); for one
    #: prepared, what its launch would take now.  ``[]`` for a stage that
    #: builds on nothing.
    takes: List[str] = field(default_factory=list)
    #: The refusal of the door that decides what it takes -- what it waits
    #: for, and what to do.
    why: Optional[str] = None
    #: Which wait ``why`` is, for a stage prepared and not ready to launch
    #: (`job-system.md`, *The task*): ``"waits"`` -- a run it builds on is
    #: not there yet; ``"regrouped"`` -- its job's members changed;
    #: ``"unreadable"`` -- its launch record does not read; ``"again"`` --
    #: it is prepared again before it can be launched.  ``None`` otherwise.
    blocked: Optional[str] = None
    #: The members of the job it is prepared in (`Job.members`, or itself
    #: alone) -- ``None`` when not prepared.
    job_members: Optional[List[str]] = None
    #: A stage prepared and never launched by ANOTHER molbuilder, in words
    #: (``molbuilder 2.17.1``) -- its first launch plans it again as prep
    #: would now and refuses it if prep would write it differently (R-L15),
    #: so its next step says so -- ``None`` when this one prepared it.
    prepared_by: Optional[str] = None

    @property
    def state(self) -> str:
        """``described`` / ``prepared`` / ``launched`` -- the word `status`
        shows for a stage the run doors do not answer."""
        return ("launched" if self.launched else
                "prepared" if self.prepared else "described")

    @property
    def detail(self) -> str:
        """One line beside the state: what a ready stage takes, the first
        line of what one waiting waits for."""
        if self.launched:
            return ""
        from .commands import reason_of
        first = reason_of(self.why or "")
        if self.prepared:
            if not self.ready:
                return {"regrouped": "its job's members changed: ",
                        "unreadable": "its launch record does not read: ",
                        "again": "prepare its group again: "}.get(
                            self.blocked, "waits for: ") + first
            return ("ready to launch"
                    + (f" (prepared by {self.prepared_by}: its launch "
                       f"compares it with what prep writes now)"
                       if self.prepared_by else "")
                    + " -- " + ("; ".join(self.takes) if self.takes else
                                "takes nothing from another run"))
        if not self.ready:
            return "waits before it can be prepared: " + first
        return "; ".join(self.takes) if self.takes else "builds on nothing"


@dataclass
class GroupAnswer:
    """One group of the description: its number, its members in ladder
    order, each member's answer, and where the group stands."""
    number: int
    members: List[str]
    answers: List[Readiness]

    @property
    def launched(self) -> bool:
        """A member was launched: the group's job is fixed."""
        return any(a.launched for a in self.answers)

    @property
    def prepared(self) -> bool:
        """Every member prepared, each in the one job these members make --
        matched by membership (`engines/stages.md` § 2)."""
        from .group import same_members
        return all(a.prepared and same_members(a.job_members, self.members)
                   for a in self.answers)

    @property
    def changed(self) -> bool:
        """A member was prepared in a job these members no longer make: its
        prep forms the group again (`web/task-setup.md` § 11)."""
        from .group import same_members
        return not self.launched and any(
            a.prepared and not same_members(a.job_members, self.members)
            for a in self.answers)

    @property
    def preparable(self) -> bool:
        """It may be prepared now: no member launched, and every member
        preparable -- a force-constant stage once its `relax` has finished."""
        return not self.launched and all(
            a.prepared or a.ready for a in self.answers)

    @property
    def prepared_by(self) -> Optional[str]:
        """Another molbuilder that prepared a member, in words, or ``None``
        (:attr:`Readiness.prepared_by`)."""
        return next((a.prepared_by for a in self.answers if a.prepared_by),
                    None)

    @property
    def launchable(self) -> bool:
        """Its first launch may go now: prepared as described, never
        launched, and what each member takes is there."""
        return (self.prepared and not self.launched
                and all(a.ready for a in self.answers))

    @property
    def why(self) -> Optional[str]:
        """What it waits for: the first member's refusal."""
        return next((a.why for a in self.answers if a.why), None)


def held_by_sequence(calc, number: int, *, asked=None
                     ) -> Optional[Tuple[object, Optional[int], str]]:
    """THE ONE ANSWER to *does a sequence still going hold group ``number``*
    (`job-system.md` § 6.0, *A sequence of groups*): ``(its job, the group
    before it in the job -- ``None`` for the job's first --, the job's state
    in words)`` for a job launched from this machine whose sequence names
    the group's stages, which is GOING, and which has not launched the
    group yet -- neither on its address nor by the group's own runs, whose
    launch records name the sequence that sent them (`run.json`'s
    ``sequence``: in that very queue job, or run here after it went); a job
    this machine cannot ask about holds nothing.  ``None`` otherwise.
    Asked by the next step, by every launch and prep of a group, and by the
    bare launch's offer."""
    from ..identity import stage_key
    from ..joblist import Asked, Job, every, is_going
    from .group import group_members, same_members
    base, task = Path(calc.base), calc.task
    stages = group_members(task, number)
    keys = {stage_key(s) for s in stages}
    asked = asked if asked is not None else Asked()
    for job in every():
        if (not isinstance(job, Job) or job.sequence is None
                or Path(job.calculation) != base.resolve()):
            continue
        groups = job.sequence["groups"]
        at = next((i for i, g in enumerate(groups)
                   if same_members(g["stages"], stages)), None)
        if at is None or keys & {stage_key(m) for m in job.members}:
            continue
        if _sent_by(calc, stages, job):
            continue
        going, said = is_going(job, asked)
        if going is not True:
            continue
        return (job, groups[at - 1]["number"] if at else None, said)
    return None


def _sent_by(calc, stages, job) -> bool:
    """Whether a member of ``stages`` has a newest run the sequence job
    ``job`` sent -- its launch record naming that sequence, and that job by
    its id (a queue job) or launched after it went (a walk run here)."""
    from datetime import datetime
    from ..runrecord import LaunchRecordError
    from .materialize import newest_launch

    def when(text):
        try:
            return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        except ValueError:
            return None
    went = when(job.launched_at)
    for st in stages:
        row = calc.job(st)
        if row is None:
            continue
        try:
            rec = newest_launch(calc, row)
        except LaunchRecordError:
            continue
        if not rec or rec.get("sequence") != job.sequence["name"]:
            continue
        if job.how == "queue":
            if str(rec.get("job_id")) == str(job.job_id):
                return True
        else:
            at = when(rec.get("launched_at"))
            if at is not None and went is not None and at >= went:
                return True
    return False


def launched_already(calc, stage: str) -> Optional[str]:
    """Why ``stage`` is not prepared again -- it was launched -- or ``None``
    (`job-system.md` § 5.0, checkpoint 2a): earlier runs' facts are read
    from the stage's job and its description (the optimizer a ``.CG``
    follows, a swept rung's points), so its runs must keep the deck and job
    they ran with.  The way back is the state saved before its group's first
    launch, every run it builds on in it (`job-system.md`, *How the
    checkpoint supports you*).  ``calc`` is the calculation as the verb
    read it (A16)."""
    from .commands import redo_launched
    from .materialize import ever_launched
    job = calc.job(stage)
    if job is None:
        return None
    from ..runrecord import LaunchRecordError
    try:
        launched = ever_launched(calc, job)
    except LaunchRecordError as exc:
        return str(exc)
    if not launched:
        return None
    return (f"stage {stage!r} was launched -- a launched stage is not "
            f"prepared again: its runs' facts are read from its job and its "
            f"description, so they must stay what its runs ran with "
            f"(job-system.md § 5.0).  To change it, "
            + redo_launched(stage, calc=calc))


def launched_stage(calc) -> Optional[str]:
    """The first of the description's stages, in its order, that has been
    launched (`materialize.ever_launched`, the one rule) -- or ``None``;
    ``calc`` the calculation as the verb read it (A16).  A launch record
    that does not read raises
    :class:`~molbuilder.runrecord.LaunchRecordError`, in its reader's
    words."""
    from .materialize import ever_launched
    if calc.jobset is None:
        return None
    for st in calc.task.stages:
        job = calc.job(st.name)
        if job is not None and ever_launched(calc, job):
            return st.name
    return None


def launched_without(calc, number: int) -> Optional[Tuple[str, List[str]]]:
    """``(launched, joined)`` -- a stage of group ``number`` that was
    launched, and the stages the group holds now that it was launched
    without -- or ``None``.  A launched group's members are fixed
    (`job-system.md`, *The task*).  A launch record that does not read
    raises, as the one door does (`materialize.ever_launched`)."""
    from ..identity import stage_key
    from .group import group_members, members_of
    from .materialize import ever_launched
    task, jobset = calc.task, calc.jobset
    held = group_members(task, number)
    for m in held:
        job = jobset.job(m) if jobset else None
        if job is None or not ever_launched(calc, job):
            continue
        joined = [h for h in held if not any(
            stage_key(h) == stage_key(x) for x in members_of(job))]
        if joined:
            return m, joined
    return None


def joined_launched(calc, number: int, launched: str,
                    joined: List[str]) -> str:
    """The way on for stages put into a launched group: a group of their own,
    or the group formed anew from the state saved before its first launch."""
    from .commands import redo_launched
    said = ", ".join(f"`{h}`" for h in joined)
    one = len(joined) == 1
    return (f"group {number} holds {said} beside `{launched}`, which was "
            f"launched without {'it' if one else 'them'} -- a launched "
            f"group's members are fixed: give {said} a group of "
            f"{'its' if one else 'their'} own (Task setup, or task.json), or, "
            f"to form the group anew, "
            + redo_launched(launched, calc=calc))


def regrouped(calc, stage: str) -> Optional[str]:
    """Why ``stage``'s prepared job (in ``calc``'s plan) is no longer the
    description's group -- the members it was prepared with, the group's
    now, and the prep that forms the job again, every group it needs named
    (`group.formed_again`) -- or ``None``.  Groups are matched by their
    members (`engines/stages.md` § 2): the ready door says it, and launch
    refuses it.  Where a group that prep would name holds a stage beside one
    launched without it, that is said instead (:func:`joined_launched`)."""
    from ..identity import stage_key
    from ..runrecord import LaunchRecordError
    from .commands import block, prep_group_lines
    from .group import (formed_again, group_members, members_of, number_of,
                        same_members)
    task, jobset, base = calc.task, calc.jobset, calc.base
    g = number_of(task, stage)
    job = jobset.job(stage)
    if g is None or job is None:
        return None
    members = group_members(task, g)
    had = members_of(job)
    if same_members(had, members):
        return None
    others = [m for m in had if stage_key(m) != stage_key(stage)]
    was = (f"in one job with {', '.join(f'`{m}`' for m in others)}"
           if others else "as a job of its own")
    said = (f"`{stage}` was prepared {was}, and its group {g} is "
            f"{', '.join(f'`{m}`' for m in members)} now -- groups are "
            f"matched by their members")
    numbers = formed_again(task, jobset.jobs, [g])
    for n in numbers:
        try:
            hit = launched_without(calc, n)
        except LaunchRecordError as exc:
            return str(exc)
        if hit:
            return said + "; " + joined_launched(calc, n, *hit)
    return (said + ", so its prep forms the job again "
            "(engines/stages.md § 2):\n"
            + block(prep_group_lines(numbers, base=base)))


def first_launch_takes(calc, stage: str, job, *,
                       from_attempt: Optional[str] = None,
                       cold: bool = False, verdict: bool = True):
    """``(continuation, gather, refusal)`` -- what ``stage``'s first launch
    takes (`job-system.md` § 6.0, step 1), from the one door per kind, read
    now, every kind's refusal with it: a force-constant stage's `relax` run
    its prep took (``job.written_at``) with that run's files -- ``cold`` and
    ``from_attempt`` refused, its run its prep's; a transport rung's
    gather, the three gates (`engines/transport.md` § 6.1), each point its
    own for a rung that runs per point -- its inputs its kind's, never
    named; an independent stage's continuation -- the stage before it's
    newest run, which must have finished, ``from_attempt`` a run named,
    ``cold`` none.  The continuation carries the files it would copy (the
    pair's rule, `materialize.continuation_files`), so a run holding none is
    refused here, before anything is sent.  ``calc`` is the calculation as
    the verb read it (A16): every question about another stage -- prepared,
    its newest run, its job -- is its plan's, the whole plan, never a set
    narrowed to what one launch sends."""
    import dataclasses
    from .continuation import Continuation, continuation_answer, read_run
    from .ladder import force_constant_stage, relax_stage_of
    from .materialize import continuation_files
    from .errors import PrepError, user_error_as_prep
    from .gather import gather_sources
    base, task, shape = Path(calc.base), calc.task, calc.shape
    if force_constant_stage(task, stage):
        relax = relax_stage_of(task)
        if cold or (from_attempt and relax is None):
            # IT MEASURES AT THE GEOMETRY ITS `relax` REACHED, or the
            # structure stated relaxed (§ 5.2a's table): nothing to start
            # cold from, and no run to name.
            from .continuation import cannot_be_named
            return None, None, (
                cannot_be_named(base, task, stage, from_attempt, cold)
                or f"--cold: `{stage}` takes nothing to start cold from -- "
                   f"it measures the structure as given.")
        if relax is None:
            return None, None, None
        if from_attempt:
            from .commands import block, command, words_for
            from .continuation import cannot_be_named
            from .group import number_of
            # A RUN ON THE FLAT LAYOUT, which keeps none to name, is
            # refused as such -- never told to a prep that would refuse it
            # too.
            if not shape.keeps_attempts_as_directories:
                return None, None, cannot_be_named(base, task, stage,
                                                   from_attempt, False)
            return None, None, (
                f"--from {from_attempt!r}: `{stage}`'s deck is written at the "
                f"`relax` run its prep took ({job.written_at}), and its first "
                f"launch takes that run's files -- another `relax` run is "
                f"said at its prep (engines/vibration.md § 5.2a):\n"
                + block([command("prep", *words_for(
                    "task", groups=[number_of(task, stage)]), base=base,
                    flags=("--from", from_attempt))]))
        if job.written_at is None:
            from .commands import way_back
            return None, None, (
                f"`{stage}` was prepared before its prep recorded the "
                f"`{relax}` run its deck is written at (2.7.0), so its "
                f"launch cannot say whose files it takes.  To launch it, "
                + way_back(stage, calc=calc, launched=False))
        flat = not shape.keeps_attempts_as_directories
        where = base if flat else base / job.written_at
        # THE RUN ITS PREP TOOK, the one its record names -- on the flat
        # layout by its name (`runfiles.parse_run_name`), never the newest.
        from ..runfiles import parse_run_name
        named = parse_run_name(job.written_at, task.label) if flat else None
        if flat and named is None:
            from .commands import way_back
            return None, None, (
                f"`{stage}`'s record names the `{relax}` run its deck is "
                f"written at as {job.written_at!r}, which is no run of "
                f"`{task.label}` in this folder.  To launch it, "
                + way_back(stage, calc=calc, launched=False))
        concluded, state, converged, _ = read_run(
            base, task, relax, where, verdict=verdict,
            run=named.run if named is not None else None)
        cont = Continuation(stage=relax,
                            source=None if flat else job.written_at,
                            by_default=True, concluded=concluded,
                            state=state, converged=converged, linked=True,
                            run=job.written_at if flat else None)
        if flat:
            # WHAT IT READS is the folder's restart files, as the last run
            # there left them (`continuation.flat_files`) -- a record of
            # another run that does not read is the refusal, in its words.
            from ..runrecord import LaunchRecordError
            from .continuation import flat_files
            try:
                return (dataclasses.replace(flat_files(base, task, cont),
                                            carries=()), None, None)
            except LaunchRecordError as exc:
                return None, None, str(exc)
        try:
            names = continuation_files(calc, stage, job.written_at,
                                       named=False)
        except ValueError as exc:
            return None, None, str(exc)
        return (dataclasses.replace(cont, carries=tuple(
            n for n in names if (where / n).is_file())), None, None)
    if task.calculation == "transport":
        # ITS INPUTS ARE ITS KIND'S, never named: `--from` / `--cold`
        # refused by the one door, as a swept rung's are.
        from .continuation import cannot_be_named
        why = cannot_be_named(base, task, stage, from_attempt, cold)
        if why:
            return None, None, why
        try:
            # THE REFUSALS THE GATHER SAYS, translated as prep translates.
            with user_error_as_prep():
                got = gather_sources(calc, stage)
        except PrepError as exc:
            return None, None, str(exc)
        return None, got, None
    cont, refused = continuation_answer(calc, stage,
                                        from_attempt=from_attempt, cold=cold,
                                        verdict=verdict)
    if refused:
        return None, None, refused
    if cont is None or cont.source is None:
        # FROM NOTHING, or on the flat layout from files lying in the folder.
        return (dataclasses.replace(cont, carries=()) if cont is not None
                else None), None, None
    try:
        names = continuation_files(calc, stage, cont.source,
                                   named=not cont.by_default)
    except ValueError as exc:
        return None, None, str(exc)
    return (dataclasses.replace(cont, carries=tuple(
        n for n in names if (base / cont.source / n).is_file())), None, None)


def at_launch(calc, stage: str, *, job=None) -> List[dict]:
    """What ``stage``'s first launch will take, said before any run is read
    (`job-system.md` § 5.0, *a preview*) -- ``[{"from", "files", "line"}]``,
    the line *at launch: takes <files> from <stage>'s newest run, which must
    have finished*: a transport rung one per stage upstream, its files by
    the ladder's DAG; an independent stage its restart files (``job``'s,
    when prepared; ``files`` ``None`` before) from the stage before it, by
    the pair's rule the launch copies by (`model.warm_carry`) against that
    stage's job in the calculation's plan, ``calc`` as the verb read it
    (A16); ``[]`` for a stage that builds on nothing.  A
    force-constant stage's is the `relax` run its deck is written at
    (:func:`readiness`)."""
    task = calc.task
    if task.calculation == "transport":
        from ..transport.stages import inputs_of
        by: Dict[str, List[str]] = {}
        for up, fn in inputs_of(task, stage):
            by.setdefault(up, []).append(fn)
        return [{"from": up, "files": fns,
                 "line": f"at launch: takes {', '.join(fns)} from `{up}`'s "
                         f"newest run, which must have finished"}
                for up, fns in by.items()]
    from .ladder import stage_before
    prev = stage_before(task, stage, calc.template_text)
    if prev is None:
        return []
    files = (carried_from(calc, job, prev)
             if job is not None and job.warm else None)
    return [{"from": prev, "files": files,
             "line": f"at launch: takes "
                     f"{', '.join(files) if files else 'its restart files'} "
                     f"from `{prev}`'s newest run, which must have finished"}]


def carried_from(calc, job, source: str) -> List[str]:
    """The restart files ``job`` takes from a run of ``source`` -- the
    pair's rule (`model.warm_carry`), the one the launch copies by, against
    ``source``'s job in the calculation's plan, ``calc`` as the verb read it
    (A16): a file conditioned on the pair is said only when both jobs are
    there to agree on it."""
    from .model import warm_carry
    return warm_carry(job, calc.job(source))


def written_at_takes(cont, files=None) -> dict:
    """What a force-constant stage's first launch takes -- ``{"from",
    "files", "line"}`` as :func:`at_launch`'s items: ``files`` (its job's
    restart files, ``None`` before it is prepared) from the `relax` run its
    deck is written at, ``cont`` (`engines/vibration.md` § 5.2a)."""
    return {"from": cont.stage, "files": files,
            "line": f"at launch: takes "
                    f"{', '.join(files) if files else 'its restart files'} "
                    f"from the `{cont.stage}` run its deck is written at, "
                    f"{cont.where()}"}


def takes_sentence(items: List[dict]) -> str:
    """:func:`at_launch`'s items as one sentence -- the one item's line, or
    for several upstreams *at launch: takes <files> from <stage>; ... --
    each from its newest run, which must have finished*."""
    if len(items) == 1:
        return items[0]["line"]
    return ("at launch: takes "
            + "; ".join(f"{', '.join(t['files'])} from `{t['from']}`"
                        for t in items)
            + " -- each from its newest run, which must have finished")


def readiness(calc, stage: str, *, verdict: bool = True) -> Readiness:
    """``stage``'s answer (`job-system.md`, *The task*), each run taken by
    default: for a force-constant stage not prepared, the `relax` run its
    prep would write it at; for a stage prepared, what its first launch
    would take.  ``verdict=False``
    leaves a relaxation's convergence unread -- for `status`, read far more
    often than a prep or a launch.  ``calc`` is the calculation as the verb
    opened it (`calculation.open`, A16): its description, template and
    plan, read once."""
    from .continuation import continuation_answer
    from .ladder import force_constant_stage
    from .materialize import ever_launched
    task, jobset = calc.task, calc.jobset
    from .group import number_of
    group = number_of(task, stage)
    job = calc.job(stage)
    if job is not None:
        from .group import members_of
        members = members_of(job)
        from ..runrecord import LaunchRecordError
        try:
            launched = ever_launched(calc, job)
        except LaunchRecordError as exc:
            # A RECORD THAT DOES NOT READ is this stage's answer, in its
            # reader's words, which name the way out -- the ladder's other
            # rows stand.
            return Readiness(stage, group, prepared=True, why=str(exc),
                             blocked="unreadable", job_members=members)
        if launched:
            return Readiness(stage, group, prepared=True, launched=True,
                             job_members=members)
        why = regrouped(calc, stage)
        if why:
            return Readiness(stage, group, prepared=True, why=why,
                             blocked="regrouped", job_members=members)
        cont, gather, why = first_launch_takes(calc, stage, job,
                                               verdict=verdict)
        if why:
            # A FORCE-CONSTANT STAGE PREPARED BEFORE ITS PREP RECORDED THE
            # RUN its deck is written at is prepared again; every other
            # refusal waits for a run it builds on.
            again = (job.written_at is None
                     and force_constant_stage(task, stage))
            return Readiness(stage, group, prepared=True, why=why,
                             blocked="again" if again else "waits",
                             job_members=members)
        return Readiness(stage, group, prepared=True, ready=True,
                         takes=_takes_lines(cont, gather),
                         job_members=members,
                         prepared_by=_prepared_by_another(calc, job))
    if group is not None and jobset is not None:
        # PUT INTO A LAUNCHED GROUP it was not launched with: its prep, and
        # its group's launch, are refused -- said here, with the way on.
        from ..runrecord import LaunchRecordError
        try:
            hit = launched_without(calc, group)
        except LaunchRecordError as exc:
            return Readiness(stage, group, why=str(exc))
        if hit and stage in hit[1]:
            return Readiness(stage, group, why=joined_launched(
                calc, group, *hit))
    if force_constant_stage(task, stage):
        # THE ONE VALUE HAND-OVER: its deck is written at a `relax` run,
        # which must have finished before it can be prepared.
        cont, why = continuation_answer(calc, stage, verdict=verdict)
        if why:
            return Readiness(stage, group, why=why)
        return Readiness(stage, group, ready=True,
                         takes=([written_at_takes(cont)["line"]]
                                if cont is not None else []))
    from .continuation import unreadable
    try:
        items = at_launch(calc, stage)
    except unreadable() as exc:
        # A TEMPLATE PREP WOULD REFUSE is said, never raised: it waits for
        # the fix before it can be prepared.
        return Readiness(stage, group,
                         why=f"what `{stage}` continues from cannot be read: "
                             f"{exc}")
    return Readiness(stage, group, ready=True,
                     takes=[takes_sentence(items)] if items else [])


def _prepared_by_another(calc, job) -> Optional[str]:
    """The molbuilder that prepared ``job``'s files, in words, when it is
    not this one -- read off its deck's own record (`script_emit.written_by`)
    where prep wrote it: the stage's folder, or a run of points' first
    point (`transport.stages.point_folders`) -- else ``None``: this one wrote
    it, or the deck is not there to say."""
    from .. import __version__
    from ..script_emit import molbuilder_words, written_by
    from ..transport.stages import point_folders
    from .naming import job_dir_names
    try:
        folder = (Path(calc.base)
                  / job_dir_names(calc.jobset, calc.shape)[job.name])
        points = (point_folders(calc.base, calc.task, job.name)
                  if calc.task.calculation == "transport" else [])
        text = ((points[0][0] if points else folder) / job.script
                ).read_text(errors="replace")
    except (OSError, ValueError, KeyError):
        return None
    by = written_by(text)
    return None if by == __version__ else molbuilder_words(by)


def _takes_lines(cont, gather) -> List[str]:
    """What a first launch would take now, one line each: the run it
    continues from, with the files it would copy; or, for a transport rung,
    one line per file (a file every point takes from one run said once; a
    file each point takes from its own point of one run, that run and the
    count)."""
    if cont is not None:
        return [cont.line(cont.carries or (), would=True)]
    srcs: Dict[str, list] = {}
    for _point, inputs in (gather or ()):
        for src, fn in inputs:
            if src not in srcs.setdefault(fn, []):
                srcs[fn].append(src)
    out = []
    for fn, where in srcs.items():
        if len(where) == 1:
            out.append(f"{fn} <- {where[0]}")
            continue
        # THE RUN a point lies in (`RunRef.run_folder`), the folders below
        # it its point's levels (a frame, a voltage).
        runs = {str(w.run_folder()) for w in where}
        out.append(f"{fn} <- {runs.pop()} ({len(where)} points)"
                   if len(runs) == 1
                   else f"{fn} <- " + ", ".join(str(w) for w in where))
    return out


def ladder(calc, *, verdict: bool = True) -> List[Readiness]:
    """Every stage of the description, in ladder order, with its answer --
    ``calc`` the calculation as the verb opened it (A16)."""
    from .naming import ladder_homes
    return [readiness(calc, h.name, verdict=verdict)
            for h in ladder_homes(calc.base, calc.task)]


@dataclass
class TaskState:
    """THE TASK AS THE READY DOOR READS IT, ONCE (`job-system.md`, *The
    task*): the description, its template's text, the plan, every stage's
    answer in ladder order and every group's -- what `prep task` shows
    before it asks, what `launch task` sends with nothing named, what
    `status` lists and what Task setup's folder answer carries."""
    task: object
    template_text: Optional[str]
    jobset: object
    stages: List[Readiness]
    groups: List[GroupAnswer]

    @property
    def prep_offer(self) -> List[int]:
        """The groups `prep task` offers (:func:`prep_offer`)."""
        return prep_offer(self.groups)

    @property
    def launch_offer(self) -> List[int]:
        """The groups `launch task` sends with nothing named
        (:func:`launch_offer`)."""
        return launch_offer(self.groups)

    def text(self) -> str:
        """The groups as both verbs show them (:func:`groups_text`)."""
        return groups_text(self.groups)


def task_state(calc, *, verdict: bool = False) -> TaskState:
    """The calculation's :class:`TaskState` -- ``calc`` as the verb opened
    it (`calculation.open`, A16), every file read once there.  ``verdict``
    reads a relaxation's convergence where a stage takes one (a prep's
    question); `status` and the folder answer leave it unread."""
    answers = ladder(calc, verdict=verdict)
    return TaskState(calc.task, calc.template_text, calc.jobset, answers,
                     groups(calc.task, answers))


def groups(task, answers: List[Readiness]) -> List[GroupAnswer]:
    """The description's groups, in order, each with its members' answers
    (``answers``, :func:`ladder`'s)."""
    by = {a.stage: a for a in answers}
    out: List[GroupAnswer] = []
    for st in task.stages:
        if not out or out[-1].number != st.group:
            out.append(GroupAnswer(st.group, [], []))
        out[-1].members.append(st.name)
        out[-1].answers.append(by[st.name])
    return out


def prep_offer(gs: List[GroupAnswer]) -> List[int]:
    """The groups `prep task` offers pre-selected (`job-system.md`, *The
    task*): every one not prepared as described and preparable."""
    return [g.number for g in gs if not g.prepared and g.preparable]


def launch_offer(gs: List[GroupAnswer]) -> List[int]:
    """The groups `launch task` sends with nothing named (`job-system.md`, *The task*): prepared as
    described, never launched, and what each member takes is there."""
    return [g.number for g in gs if g.launchable]


def groups_text(gs: List[GroupAnswer]) -> str:
    """The groups as `prep task` and `launch task` show them: each group's
    number, then each member, where it stands, and what it takes or waits
    for."""
    width = max([len(a.stage) for g in gs for a in g.answers] + [5])
    out = []
    for g in gs:
        out.append(f"  group {g.number}"
                   + ("  (launched)" if g.launched else
                      "  (prepared, not launched)" if g.prepared else
                      "  (prepared, its members changed)" if g.changed
                      else ""))
        for a in g.answers:
            out.append(f"    {a.stage:<{width}}  {a.state:<9}  {a.detail}")
    return "\n".join(out)


__all__ = ["Readiness", "GroupAnswer", "TaskState", "task_state",
           "readiness", "launched_already", "launched_without",
           "joined_launched", "regrouped", "launched_stage",
           "first_launch_takes",
           "at_launch", "written_at_takes", "takes_sentence", "ladder", "groups", "prep_offer", "launch_offer",
           "groups_text"]
