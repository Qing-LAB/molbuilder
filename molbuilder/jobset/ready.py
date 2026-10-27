"""THE READY DOOR -- where a stage and its group stand between the
description and a run, and what a stage's first launch takes
(`job-system.md`, *The task*; `architecture.md` § 3.2).

**A stage is described, prepared or launched**, each answered here from the
doors that already decide it.  **Described** -- not prepared: preparable.
**Prepared**, never launched: **ready to launch** when what its first
launch takes is there, else **waits for** what it names -- the door that
decides what it takes, the same one launch asks: an independent stage's
continuation (the stage before it, its newest run finished), a transport
rung's gather (`gather.gather_sources`, its three gates); a SIESTA
vibration's one run takes nothing from another stage
(`engines/vibration.md` § 5.11).  **Launched**: the run doors answer.
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
    #: first launch will take, said before any run is read; for one
    #: prepared, what its launch would take now.  ``[]`` for a stage that
    #: builds on nothing.
    takes: List[str] = field(default_factory=list)
    #: The refusal of the door that decides what it takes -- what it waits
    #: for, and what to do.
    why: Optional[str] = None
    #: Which wait ``why`` is, for a stage prepared and not ready to launch
    #: (`job-system.md`, *The task*): ``"waits"`` -- a run it builds on is
    #: not there yet; ``"regrouped"`` -- its job's members changed;
    #: ``"unreadable"`` -- its launch record does not read.  ``None``
    #: otherwise.
    blocked: Optional[str] = None
    #: The members of the job it is prepared in (`Job.members`, or itself
    #: alone) -- ``None`` when not prepared.
    job_members: Optional[List[str]] = None
    #: A stage prepared and never launched by ANOTHER molbuilder, in words
    #: (``molbuilder 2.17.1``), said beside its state -- ``None`` when this
    #: one prepared it.
    prepared_by: Optional[str] = None
    #: CHANGED SINCE PREPARED (`job-system.md`, *The task*): each value its
    #: records give now otherwise than its prep recorded, ``name: was →
    #: now`` (`prepared_with.changes`) -- ``[]`` when none does, and for a
    #: stage not prepared.
    changed: list = field(default_factory=list)      # `prepared_with.Difference`
    #: Prepared, and its prep recorded no values (prepared before 2.41.0):
    #: nothing is compared, and that is said.
    unrecorded: bool = False

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
        from .commands import reason_of
        from .prepared_with import UNRECORDED_SAID
        if self.launched:
            # A LAUNCHED STAGE SAYS IT TOO (`job-system.md`, *The task*,
            # the launched row): its runs computed the description as it
            # was; the way on is its group's step.
            return ("changed since prepared: "
                    + "; ".join(d.said for d in self.changed)
                    if self.changed else "")
        first = reason_of(self.why or "")
        if self.prepared and self.changed:
            return ("changed since prepared: "
                    + "; ".join(d.said for d in self.changed)
                    + " -- prepare it again")
        if self.prepared:
            if not self.ready:
                return {"regrouped": "its job's members changed: ",
                        "unreadable": "its launch record does not read: "
                        }.get(self.blocked, "waits for: ") + first
            return ("ready to launch"
                    + (f" (prepared by {self.prepared_by})"
                       if self.prepared_by else "")
                    + " -- " + ("; ".join(self.takes) if self.takes else
                                "takes nothing from another run")
                    + (f"; {UNRECORDED_SAID}" if self.unrecorded else ""))
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
    #: WHY A LAUNCHED GROUP IS NOT PREPARED AGAIN on its layout -- the flat
    #: one, whose runs share their folder's files -- in the rule's own words
    #: (:func:`launched_already`); ``None`` where it may be, for its job
    #: request (`execution/job-request.md` § 5).
    not_again: Optional[str] = None
    #: A LAUNCHED GROUP WHOSE MEMBERS CHANGED -- a stage put into it, or one
    #: launched with it taken out (:func:`launched_without`) -- in the rule's
    #: own words and way on (:func:`joined_launched`); ``None`` otherwise.
    #: Such a group is offered by no verb (`job-system.md`, *The task*).
    members_changed: Optional[str] = None
    #: A SEQUENCE STILL GOING HOLDS IT -- ``(its job, the group before it in
    #: the job or None, the job's state in words)``, :func:`held_by_sequence`'s
    #: answer, asked of a group every member of which is prepared -- or
    #: ``None``.  That job's to launch: no verb prepares or launches it
    #: meanwhile (`job-system.md` § 6.0, R-P23, R-L45).
    held: Optional[tuple] = None
    #: THE GROUPS ITS PREP NAMES -- it, and every group a member of its
    #: prepared job is in now, wherever that job's members are no longer one
    #: group (`group.formed_again`; R-P6).
    prep_with: List[int] = field(default_factory=list)

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
        """It may be prepared now: every member preparable, no sequence
        still going holding it, and no member launched, or, on the
        hierarchical layout, a launched one prepared again for its job
        request (`execution/job-request.md` § 5)."""
        return (self.not_again is None
                and self.members_changed is None
                and self.held is None
                and all(a.prepared or a.ready for a in self.answers))

    @property
    def since_prepared(self) -> list:
        """CHANGED SINCE PREPARED: each value that differs, by member --
        ``tight: mesh_cutoff: 300 → 400`` (:attr:`Readiness.changed`), each
        keeping what it is of (`prepared_with.Difference.kind`)."""
        from .prepared_with import Difference
        return [Difference(d.kind, f"{a.stage}: {d.said}")
                for a in self.answers for d in a.changed]

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
        return (self.prepared and not self.launched and self.held is None
                and all(a.ready for a in self.answers))

    @property
    def why(self) -> Optional[str]:
        """What it waits for: a launched group's changed members, or why it
        is not prepared again, or the sequence holding it, else the first
        member's refusal."""
        return (self.members_changed or self.not_again
                or (held_why(self.number, self.held) if self.held else None)
                or next((a.why for a in self.answers if a.why), None))

    @property
    def said(self) -> str:
        """Where it stands, by its state's name in the diagram
        (`job-system.md`, *The task*) -- the words every verb and surface
        shows it in."""
        return ("launched, its members changed" if self.members_changed else
                "prepared, its members changed" if self.changed else
                f"goes in job {self.held[0].id} -- "
                + (f"after group {self.held[1]}" if self.held[1] is not None
                   else "its first group") if self.held else
                "launched, its members fixed" if self.launched else
                "prepared, never launched" if self.prepared else
                "not prepared")


def held_why(number: int, held: tuple) -> str:
    """Why group ``number`` is not prepared while the sequence ``held``
    (:func:`held_by_sequence`'s answer) is going, and the way on -- the one
    sentence prep refuses with and the group answer says (R-P23)."""
    job, _after, state = held
    return (f"group {number} goes in job {job.id} -- a sequence of groups "
            f"still going ({state}) launches it with the files it was "
            f"prepared with, and preparing it again would rewrite them under "
            f"the job.  Prepare it once that job has ended, or stop the job "
            f"first: molbuilder jobs stop {job.id}  (job-system.md R-P23)")


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
    from .group import same_members
    base = Path(calc.base)
    stages = calc.request.members(number)
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
    from ..runrecord import LaunchRecordError, stamp_time
    from .materialize import newest_launch
    went = stamp_time(job.launched_at)
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
            at = stamp_time(rec.get("launched_at"))
            if at is not None and went is not None and at >= went:
                return True
    return False


def launched_already(calc, stage: str) -> Optional[str]:
    """Why ``stage`` is not prepared again -- launched, on the FLAT layout --
    or ``None`` (`execution/job-request.md` § 5; `job-system.md` § 5.0,
    checkpoint 2a): there a stage's runs share its folder's deck and run
    script, so a deck rendered anew would rewrite what an earlier run's
    record reads as its own.  On the hierarchical layout each run holds its
    own copy, and a launched stage is prepared again for its job request --
    its deck the same calculation, which the prep asks of the deck it
    renders.  The way back is the state saved before its group's first
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
    if not launched or calc.shape.keeps_attempts_as_directories:
        return None
    return (f"stage {stage!r} was launched, and on the flat layout a launched "
            f"stage is not prepared again: its runs share its folder's deck "
            f"and run script, which its runs' records read as theirs "
            f"(execution/job-request.md § 5).  To change it, "
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


def launched_without(calc, number: int
                     ) -> Optional[Tuple[str, List[str], List[str]]]:
    """``(launched, joined, left)`` -- a stage of group ``number`` that was
    launched, the stages the group holds now that it was launched without,
    and the stages it was launched with that the group holds no longer --
    or ``None``.  A launched group's members are fixed (`job-system.md`,
    *The task*; `execution/job-request.md` § 5).  A launch record that does
    not read raises, as the one door does (`materialize.ever_launched`)."""
    from ..identity import stage_key
    from .group import members_of
    from .materialize import ever_launched
    jobset = calc.jobset
    held = calc.request.members(number)
    for m in held:
        job = jobset.job(m) if jobset else None
        if job is None or not ever_launched(calc, job):
            continue
        had = members_of(job)
        joined = [h for h in held if not any(
            stage_key(h) == stage_key(x) for x in had)]
        left = [x for x in had if not any(
            stage_key(x) == stage_key(h) for h in held)]
        if joined or left:
            return m, joined, left
    return None


def joined_launched(calc, number: int, launched: str,
                    joined: List[str], left: List[str]) -> str:
    """The way on for a launched group whose members changed -- the one
    composer every verb and surface says it in (`job-system.md`, *The
    task*): a stage put into it, a group of its own; a stage taken out of
    it, put back beside the one launched with it; the one launched,
    launched again alone; or the group formed anew from the state saved
    before its first launch."""
    from .commands import block, launch_lines, redo_launched

    def said(names):
        return ", ".join(f"`{h}`" for h in names)
    what, ways = [], []
    if joined:
        one = len(joined) == 1
        what.append(f"group {number} holds {said(joined)} beside "
                    f"`{launched}`, which was launched without "
                    f"{'it' if one else 'them'}")
        ways.append(f"give {said(joined)} a group of "
                    f"{'its' if one else 'their'} own (Task setup, or "
                    f"job-request.json)")
    if left:
        one = len(left) == 1
        what.append(f"`{launched}` was launched in one job with "
                    f"{said(left)}, and group {number} now holds "
                    f"`{launched}` without {'it' if one else 'them'}")
        # A STAGE TAKEN OUT OF THE DESCRIPTION goes back through it: the
        # job request holds no stage the description does not.
        ways.append(f"put {said(left)} back beside `{launched}` in one "
                    f"group (Task setup)")
    return ("; ".join(what) + " -- a launched group's members are fixed: "
            + " and ".join(ways) + "; or launch "
            f"`{launched}` again alone --\n"
            + block(launch_lines("task", launched, base=calc.base))
            + "\n  or, to form the group anew, "
            + redo_launched(launched, calc=calc))


def why_group_fixed(calc, stage: str) -> str:
    """Why a launched stage's group is not changed where the groups are
    edited (`web/task-setup.md` § 5), and the restore that changes it."""
    from .commands import redo_launched
    return (f"`{stage}` was launched -- its group is fixed; to change it, "
            + redo_launched(stage, calc=calc))


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
    from .group import formed_again, members_of, same_members
    jobset, base = calc.jobset, calc.base
    g = calc.request.group_of(stage)
    job = jobset.job(stage)
    if g is None or job is None:
        return None
    members = calc.request.members(g)
    had = members_of(job)
    if same_members(had, members):
        return None
    others = [m for m in had if stage_key(m) != stage_key(stage)]
    was = (f"in one job with {', '.join(f'`{m}`' for m in others)}"
           if others else "as a job of its own")
    said = (f"`{stage}` was prepared {was}, and its group {g} is "
            f"{', '.join(f'`{m}`' for m in members)} now -- groups are "
            f"matched by their members")
    numbers = formed_again(calc.request, jobset.jobs, [g])
    for n in numbers:
        try:
            hit = launched_without(calc, n)
        except LaunchRecordError as exc:
            return str(exc)
        if hit:
            return said + "; " + joined_launched(calc, n, *hit)
    return (said + ", so its prep forms the job again "
            "(engines/stages.md § 2) --\n"
            + block(prep_group_lines(numbers, base=base)))


def first_launch_takes(calc, stage: str, job, *,
                       from_attempt: Optional[str] = None,
                       cold: bool = False, verdict: bool = True):
    """``(continuation, gather, refusal)`` -- what ``stage``'s first launch
    takes (`job-system.md` § 6.0, step 1), from the one door per kind, read
    now, every kind's refusal with it: a transport rung's gather, the three
    gates (`engines/transport.md` § 6.1), each point its own for a rung that
    runs per point -- its inputs its kind's, never named; a SIESTA
    vibration's run nothing -- its calls build on each other inside the run
    (`engines/vibration.md` § 5.11), ``cold`` and ``from_attempt`` refused
    by the one door (`continuation.cannot_be_named`); an independent
    stage's continuation -- the stage before it's newest run, which must
    have finished, ``from_attempt`` a run named, ``cold`` none.  The
    continuation carries the files it would copy (the pair's rule,
    `materialize.continuation_files`), so a run holding none is refused
    here, before anything is sent.  ``calc`` is the calculation as the verb
    read it (A16): every question about another stage -- prepared, its
    newest run, its job -- is its plan's, the whole plan, never a set
    narrowed to what one launch sends."""
    import dataclasses
    from .continuation import continuation_answer
    from .materialize import continuation_files
    from .errors import PrepError, user_error_as_prep
    from .gather import gather_sources
    base, task = Path(calc.base), calc.task
    if task.calculation == "transport":
        # ITS INPUTS ARE ITS KIND'S, never named: `--from` / `--cold`
        # refused by the one door, as a swept rung's are -- but a rung of
        # points of a calculation that reuses a finished one, whose first
        # launch takes that calculation's points over: `--cold` takes none
        # (`engines/transport.md` § 2a.16).
        from ..transport.stages import frames_of, rung_points
        from .continuation import cannot_be_named
        why = (None if cold and not from_attempt and task.reuses
               and rung_points(task, stage, frames=frames_of(base))
               else cannot_be_named(base, task, stage, from_attempt, cold))
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
    (A16); ``[]`` for a stage that builds on nothing."""
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
    default: for a stage prepared, what its first launch would take.
    ``verdict=False``
    leaves a relaxation's convergence unread -- for `status`, read far more
    often than a prep or a launch.  ``calc`` is the calculation as the verb
    opened it (`calculation.open`, A16): its description, template and
    plan, read once."""
    from .materialize import ever_launched
    jobset = calc.jobset
    group = calc.request.group_of(stage)
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
        # CHANGED SINCE PREPARED -- its records now against what its prep
        # recorded, a read (`prepared_with.changes`; `job-system.md`, *The
        # task*): said, never a refusal.
        from .prepared_with import changes
        recorded, changed = changes(calc, stage)
        since = dict(changed=changed, unrecorded=not recorded)
        if launched:
            return Readiness(stage, group, prepared=True, launched=True,
                             job_members=members, **since)
        why = regrouped(calc, stage)
        if why:
            return Readiness(stage, group, prepared=True, why=why,
                             blocked="regrouped", job_members=members,
                             **since)
        cont, gather, why = first_launch_takes(calc, stage, job,
                                               verdict=verdict)
        if why:
            # A REFUSAL WAITS for a run the stage builds on.
            return Readiness(stage, group, prepared=True, why=why,
                             blocked="waits", job_members=members, **since)
        return Readiness(stage, group, prepared=True, ready=True,
                         takes=_takes_lines(cont, gather),
                         job_members=members,
                         prepared_by=_prepared_by_another(calc, job),
                         **since)
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
    from .naming import job_dir_names
    from .points import rung_points_of
    try:
        folder = (Path(calc.base)
                  / job_dir_names(calc.jobset, calc.shape)[job.name])
        points = rung_points_of(calc, job.name)
        text = ((folder / points[0].rel if points else folder) / job.script
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
    gs = groups(calc.request, answers)
    from ..joblist import Asked
    from .group import formed_again
    jobs = calc.jobset.jobs if calc.jobset is not None else []
    asked = Asked()
    for g in gs:
        g.prep_with = formed_again(calc.request, jobs, [g.number])
        if all(a.prepared for a in g.answers):
            g.held = held_by_sequence(calc, g.number, asked=asked)
    # A LAUNCHED GROUP'S MEMBERS ARE FIXED: one whose members changed says
    # so, and one its layout does not prepare again says why -- each rule
    # asked once here for every reader (`job-system.md`, *The task*).
    from ..runrecord import LaunchRecordError
    for g in gs:
        if not g.launched:
            continue
        g.not_again = next((w for w in (launched_already(calc, m)
                                        for m in g.members) if w), None)
        try:
            hit = launched_without(calc, g.number)
        except LaunchRecordError as exc:
            g.members_changed = str(exc)
            continue
        if hit:
            g.members_changed = joined_launched(calc, g.number, *hit)
    return TaskState(calc.task, calc.template_text, calc.jobset, answers, gs)


def groups(request, answers: List[Readiness]) -> List[GroupAnswer]:
    """The job request's groups, in order, each with its members' answers
    (``answers``, :func:`ladder`'s)."""
    by = {a.stage: a for a in answers}
    out: List[GroupAnswer] = []
    for name, entry in request.stages.items():
        if not out or out[-1].number != entry.group:
            out.append(GroupAnswer(entry.group, [], []))
        out[-1].members.append(name)
        out[-1].answers.append(by[name])
    return out


def prep_offer(gs: List[GroupAnswer]) -> List[int]:
    """The groups `prep task` offers pre-selected (`job-system.md`, *The
    task*): every one not prepared as described, or never launched and
    changed since prepared, and preparable."""
    return [g.number for g in gs
            if (not g.prepared or (g.since_prepared and not g.launched))
            and g.preparable]


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
        # A GROUP A SEQUENCE HOLDS names where to follow it, as `status`'s
        # step does (`job-system.md`, *The next step*).
        out.append(f"  group {g.number}  ({g.said}"
                   + ("; molbuilder jobs status" if g.held else "") + ")")
        for a in g.answers:
            out.append(f"    {a.stage:<{width}}  {a.state:<9}  {a.detail}")
    return "\n".join(out)


__all__ = ["Readiness", "GroupAnswer", "TaskState", "task_state",
           "readiness", "launched_already", "launched_without",
           "joined_launched", "why_group_fixed", "regrouped",
           "launched_stage",
           "first_launch_takes",
           "at_launch", "takes_sentence", "ladder", "groups", "prep_offer", "launch_offer",
           "groups_text"]
