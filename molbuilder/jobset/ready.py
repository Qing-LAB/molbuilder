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
from typing import Dict, List, Optional


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
            return "ready to launch -- " + ("; ".join(self.takes)
                                            if self.takes else
                                            "takes nothing from another run")
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
    def launchable(self) -> bool:
        """Its first launch may go now: prepared as described, never
        launched, and what each member takes is there."""
        return (self.prepared and not self.launched
                and all(a.ready for a in self.answers))

    @property
    def why(self) -> Optional[str]:
        """What it waits for: the first member's refusal."""
        return next((a.why for a in self.answers if a.why), None)


def job_of(jobset, stage: str):
    """The job ``stage`` is prepared as -- its row in the calculation's
    plan, matched as every reader matches a stage's name
    (`identity.stage_key`) -- or ``None``, as it is with no plan.  PREPARED
    is that row (`job-system.md` § 5.0): the one door every reader asks,
    with the plan as the verb read it (`JobSet.of`, A16)."""
    return jobset.job(stage) if jobset is not None else None


def launched_already(base, task, stage: str, jobset) -> Optional[str]:
    """Why ``stage`` is not prepared again -- it was launched -- or ``None``
    (`job-system.md` § 5.0, checkpoint 2a): earlier runs' facts are read
    from the stage's job and its description (the optimizer a ``.CG``
    follows, a swept rung's points), so its runs must keep the deck and job
    they ran with.  The way back is the state saved before its group's first
    launch, every run it builds on in it (`job-system.md`, *How the
    checkpoint supports you*).  ``jobset`` is the plan as the verb read
    it (A16)."""
    from ..paths import Shape
    from .commands import redo_launched
    from .materialize import ever_launched
    js = jobset
    job = job_of(js, stage)
    if job is None:
        return None
    from ..runrecord import LaunchRecordError
    try:
        launched = ever_launched(js, base, job, Shape.named(task.shape))
    except LaunchRecordError as exc:
        return str(exc)
    if not launched:
        return None
    return (f"stage {stage!r} was launched -- a launched stage is not "
            f"prepared again: its runs' facts are read from its job and its "
            f"description, so they must stay what its runs ran with "
            f"(job-system.md § 5.0).  To change it, "
            + redo_launched(stage, base=base, task=task, jobset=js))


def launched_stage(base, task, jobset) -> Optional[str]:
    """The first of ``task``'s stages, in the description's order, that has
    been launched (`materialize.ever_launched`, the one rule) -- or
    ``None``; ``jobset`` the plan as the verb read it (A16).  A launch
    record that does not read raises
    :class:`~molbuilder.runrecord.LaunchRecordError`, in its reader's
    words."""
    from ..paths import Shape
    from .materialize import ever_launched
    js = jobset
    if js is None:
        return None
    shape = Shape.named(task.shape)
    for st in task.stages:
        job = job_of(js, st.name)
        if job is not None and ever_launched(js, base, job, shape):
            return st.name
    return None


def first_launch_takes(base, task, stage: str, job, jobset, *,
                       template_text: Optional[str] = None,
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
    refused here, before anything is sent."""
    import dataclasses
    from .continuation import (Continuation, continuation_answer,
                               force_constant_stage, read_run,
                               relax_stage_of)
    from ..paths import Shape
    from .materialize import continuation_files
    from .errors import PrepError, user_error_as_prep
    from .gather import gather_sources
    base = Path(base)
    shape = Shape.named(task.shape)
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
                + way_back(stage, base=base, task=task, launched=False,
                           jobset=jobset))
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
                + way_back(stage, base=base, task=task, launched=False,
                           jobset=jobset))
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
            names = continuation_files(jobset, base, stage, job.written_at,
                                       named=False, shape=shape)
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
                got = gather_sources(base, task, stage, jobset=jobset,
                                     template_text=template_text)
        except PrepError as exc:
            return None, None, str(exc)
        return None, got, None
    cont, refused = continuation_answer(base, task, stage, jobset=jobset,
                                        from_attempt=from_attempt, cold=cold,
                                        verdict=verdict,
                                        template_text=template_text)
    if refused:
        return None, None, refused
    if cont is None or cont.source is None:
        # FROM NOTHING, or on the flat layout from files lying in the folder.
        return (dataclasses.replace(cont, carries=()) if cont is not None
                else None), None, None
    try:
        names = continuation_files(jobset, base, stage, cont.source,
                                   named=not cont.by_default, shape=shape)
    except ValueError as exc:
        return None, None, str(exc)
    return (dataclasses.replace(cont, carries=tuple(
        n for n in names if (base / cont.source / n).is_file())), None, None)


def at_launch(task, stage: str, *, job=None, jobset=None,
              template_text: Optional[str] = None) -> List[dict]:
    """What ``stage``'s first launch will take, said before any run is read
    (`job-system.md` § 5.0, *a preview*) -- ``[{"from", "files", "line"}]``,
    the line *at launch: takes <files> from <stage>'s newest run, which must
    have finished*: a transport rung one per stage upstream, its files by
    the ladder's DAG; an independent stage its restart files (``job``'s,
    when prepared; ``files`` ``None`` before) from the stage before it, by
    the pair's rule the launch copies by (`model.warm_carry`) against that
    stage's job in ``jobset``, the plan as the verb read it; ``[]`` for a
    stage that builds on nothing.  A
    force-constant stage's is the `relax` run its deck is written at
    (:func:`readiness`)."""
    if task.calculation == "transport":
        from ..transport.stages import inputs_of
        by: Dict[str, List[str]] = {}
        for up, fn in inputs_of(task, stage):
            by.setdefault(up, []).append(fn)
        return [{"from": up, "files": fns,
                 "line": f"at launch: takes {', '.join(fns)} from `{up}`'s "
                         f"newest run, which must have finished"}
                for up, fns in by.items()]
    from .continuation import stage_before
    prev = stage_before(task, stage, template_text)
    if prev is None:
        return []
    files = (carried_from(job, prev, jobset=jobset)
             if job is not None and job.warm else None)
    return [{"from": prev, "files": files,
             "line": f"at launch: takes "
                     f"{', '.join(files) if files else 'its restart files'} "
                     f"from `{prev}`'s newest run, which must have finished"}]


def carried_from(job, source: str, *, jobset=None) -> List[str]:
    """The restart files ``job`` takes from a run of ``source`` -- the
    pair's rule (`model.warm_carry`), the one the launch copies by, against
    ``source``'s job in ``jobset``, the plan as the verb read it: a file
    conditioned on the pair is said only when both jobs are there to agree
    on it."""
    from .model import warm_carry
    return warm_carry(job, job_of(jobset, source))


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
    from .continuation import continuation_answer, force_constant_stage
    from .materialize import ever_launched
    base, task = Path(calc.base), calc.task
    template_text, jobset = calc.template_text, calc.jobset
    from .group import number_of
    group = number_of(task, stage)
    job = job_of(jobset, stage)
    if job is not None:
        from .group import members_of
        members = members_of(job)
        from ..paths import Shape
        from ..runrecord import LaunchRecordError
        try:
            launched = ever_launched(jobset, base, job,
                                     Shape.named(task.shape))
        except LaunchRecordError as exc:
            # A RECORD THAT DOES NOT READ is this stage's answer, in its
            # reader's words, which name the way out -- the ladder's other
            # rows stand.
            return Readiness(stage, group, prepared=True, why=str(exc),
                             blocked="unreadable", job_members=members)
        if launched:
            return Readiness(stage, group, prepared=True, launched=True,
                             job_members=members)
        from .group import regrouped
        why = regrouped(task, jobset, stage, base=base)
        if why:
            return Readiness(stage, group, prepared=True, why=why,
                             blocked="regrouped", job_members=members)
        cont, gather, why = first_launch_takes(
            base, task, stage, job, jobset, template_text=template_text,
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
                         job_members=members)
    if group is not None and jobset is not None:
        # PUT INTO A LAUNCHED GROUP it was not launched with: its prep, and
        # its group's launch, are refused -- said here, with the way on.
        from ..runrecord import LaunchRecordError
        from .group import joined_launched, launched_without
        try:
            hit = launched_without(task, jobset, group, base=base)
        except LaunchRecordError as exc:
            return Readiness(stage, group, why=str(exc))
        if hit and stage in hit[1]:
            return Readiness(stage, group, why=joined_launched(
                group, *hit, base=base, task=task, jobset=jobset))
    if force_constant_stage(task, stage):
        # THE ONE VALUE HAND-OVER: its deck is written at a `relax` run,
        # which must have finished before it can be prepared.
        cont, why = continuation_answer(base, task, stage, verdict=verdict,
                                        template_text=template_text,
                                        jobset=jobset)
        if why:
            return Readiness(stage, group, why=why)
        return Readiness(stage, group, ready=True,
                         takes=([written_at_takes(cont)["line"]]
                                if cont is not None else []))
    from .continuation import unreadable
    try:
        items = at_launch(task, stage, template_text=template_text)
    except unreadable() as exc:
        # A TEMPLATE PREP WOULD REFUSE is said, never raised: it waits for
        # the fix before it can be prepared.
        return Readiness(stage, group,
                         why=f"what `{stage}` continues from cannot be read: "
                             f"{exc}")
    return Readiness(stage, group, ready=True,
                     takes=[takes_sentence(items)] if items else [])


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
    from .materialize import ladder_homes
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
    """The groups `launch task` sends with nothing named (L1): prepared as
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
           "readiness", "job_of", "launched_already", "launched_stage",
           "first_launch_takes",
           "at_launch", "written_at_takes", "takes_sentence", "ladder", "groups", "prep_offer", "launch_offer",
           "groups_text"]
