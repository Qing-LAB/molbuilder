"""A GROUP: the stages that run as one job, by the number each states in
`task.json` (`engines/stages.md` § 2, `project-layout.md` § 1.6.6; plan
§ 5aa).

Stages that do not build on one another -- a transport ladder's seed and its
two leads -- wait in the queue once instead of once each.  Each member is
prepared as it would be alone and keeps its own attempt, deck, run script
and records; the group is written on each member's job (`Job.members`) with
one header for the group's job, and launch sends that job, walking the
members in the ladder's order.

This module answers the group's questions, each in one place:

* **does a stage build on another** (:func:`upstream_of`) -- the one fact
  the ladder already states: a transport rung's inputs
  (`transport.stages.stage_inputs`), a force-constant stage's `relax`, an
  independent stage's stage before (`continuation.source_stage`) -- and so
  **the numbers a new description takes** (:func:`default_groups`,
  :func:`numbered`) and **whether the stated ones keep it**
  (:func:`check_groups`);
* **can the members share one allocation** (:func:`envelope`) -- one queue,
  one count of ranks, cores per rank and GPUs; the wall the sum of theirs,
  the memory the largest;
* **what the group's files are called** (:func:`names_for`,
  `runfiles.GroupNames.of_stages`) -- one
  ``GroupNames`` from the members' tokens, in order.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Set, Tuple

from ..runfiles import GroupNames

if TYPE_CHECKING:
    from .model import Resources


class GroupError(ValueError):
    """The stages named cannot be one group -- the message says which and
    why, in the person's words."""


#: WHAT ONE JOB'S MEMBERS SHARE (`project-layout.md` § 1.6.6) -- each
#: `Resources` field, by its word.  A run card's key is read through the
#: exchange map (`model.AS_RESOURCE`: SIESTA's `omp_threads` and PySCF's
#: `threads` are `cpus_per_task`, `gpu_count` the count in `gres`); the GPU
#: binding is the calculation's (`allocation.gpu_binding`).  The wall is
#: the members' summed and the memory the largest, so neither is here.
SHARED = {"domain": "queue", "mpi_np": "ranks",
          "cpus_per_task": "cores per rank", "gres": "GPUs",
          "use_gpu": "GPU use", "gpu_binding": "GPU binding"}


def shared_card_keys(names) -> List[str]:
    """Of the run-card ``names`` (`stages.md` § 6.8d), those one job's
    members share (:data:`SHARED`, read through the exchange map) -- the
    rows Task setup's group card writes into every member
    (`web/task-setup.md` § 6.2b)."""
    from .model import AS_RESOURCE
    return [n for n in names if AS_RESOURCE.get(n, n) in SHARED]


def card_shares(task, stage: str) -> dict:
    """``stage``'s run card (`Task.run_condition`) on what a job's members
    share (:data:`SHARED`), by field -- ``None`` where it states nothing."""
    from .model import AS_RESOURCE
    card = {AS_RESOURCE.get(k, k): v
            for k, v in task.run_condition(stage).items()}
    return {f: card.get(f) for f in SHARED}


def upstream_of(base, task, stage: str,
                template_text: Optional[str] = None) -> Set[str]:
    """The stages ``stage`` builds on, as the ladder states it: a transport
    rung's inputs, a force-constant stage's `relax`, an independent stage's
    stage before it (none for one set `restart: clean`, or the first)."""
    if getattr(task, "calculation", None) == "transport":
        from ..transport.stages import inputs_of
        return {src for src, _f in inputs_of(task, stage)}
    from .continuation import source_stage
    src, _linked = source_stage(base, task, stage, template_text)
    return {src} if src else set()


def linked(base, task, stage: str,
           template_text: Optional[str] = None) -> bool:
    """Whether what ``stage`` builds on is its KIND's -- a transport rung's
    inputs, a force-constant stage's `relax` -- which no run card changes;
    ``False`` for an independent ladder's stage, which builds on the stage
    before it only while its run card's `restart` continues
    (`job-system.md` § 5.4)."""
    if getattr(task, "calculation", None) == "transport":
        return True
    from .continuation import source_stage
    return bool(source_stage(base, task, stage, template_text)[1])


def may_join(base, task, stage: str, members: Sequence[str],
             template_text: Optional[str] = None) -> bool:
    """Whether ``stage`` may share the job ``members`` make, the group
    above it, by default (`engines/stages.md` § 2: the defaults repeat the
    number above wherever that is allowed): it builds on none of them
    (:func:`upstream_of`), and its run card states what theirs do on what a
    job shares (:func:`card_shares`) -- else prep would refuse the group
    (`job-system.md` R-P13).  The one rule both writers of numbers ask --
    a new description's (:func:`default_groups`) and an older one's
    (`migrate`)."""
    return (bool(members)
            and not upstream_of(base, task, stage, template_text)
            & set(members)
            and card_shares(task, stage) == card_shares(task, members[0]))


def default_groups(task, template_text: Optional[str] = None
                   ) -> Tuple[int, ...]:
    """The numbers a new description takes (`engines/stages.md` § 2): the
    first stage group 1; each later one the number above it wherever it
    may join that group (:func:`may_join`), else one more -- a transport
    ladder 1 1 1 2 3, an optimization ladder 1 2 3, stages set `restart:
    clean` 1 1 1."""
    out: List[int] = []
    members: List[str] = []
    for st in task.stages:
        if out and may_join(None, task, st.name, members, template_text):
            out.append(out[-1])
            members.append(st.name)
        else:
            out.append(out[-1] + 1 if out else 1)
            members = [st.name]
    return tuple(out)


def numbered(task, template_text: Optional[str] = None):
    """``task`` with every stage's group stated: as given when its stages
    state them, else :func:`default_groups` -- what a new description is
    numbered with as it is made (`describe.Description`), before any writer
    writes it."""
    if all(st.group is not None for st in task.stages):
        return task
    import dataclasses
    return dataclasses.replace(task, stages=tuple(
        dataclasses.replace(st, group=g)
        for st, g in zip(task.stages, default_groups(task, template_text))))


def check_groups(base, task, template_text: Optional[str] = None
                 ) -> Optional[str]:
    """Why the stated groups cannot run -- a stage sharing a group with one
    it builds on, both named -- or ``None`` (`engines/stages.md` § 6.6):
    one job runs its members one after another with nothing handed between
    them.  A ladder not numbered yet has nothing to check: the numbers
    :func:`numbered` gives it keep the rule by construction."""
    if any(st.group is None for st in task.stages):
        return None
    seen: dict = {}
    for st in task.stages:
        members = seen.setdefault(st.group, [])
        fed = upstream_of(base, task, st.name, template_text) & set(members)
        if fed:
            named = [f"`{n}`" for n in members if n in fed]
            on = (", ".join(named[:-1]) + " and " + named[-1]
                  if len(named) > 1 else named[0])
            # THE OTHER WAY OUT, where there is one: an independent stage
            # builds on the stage before it only while it continues, so set
            # clean it builds on nothing; a linked stage's input is its
            # kind's, whatever its run card says.
            kind_s = linked(base, task, st.name, template_text)
            return (f"`{st.name}` is in group {st.group} with {on}, which "
                    f"it builds on: one job runs its members one after "
                    f"another with nothing handed between them, so "
                    f"`{st.name}` would start before you have looked at "
                    f"what {on} produced (engines/stages.md § 2).  Give "
                    f"`{st.name}` a new group -- one more than the group "
                    f"above" + ("." if kind_s else
                                f"; or, for it to start from its own deck "
                                f"rather than from {on}, set its run "
                                f"card's `restart` to `clean`."))
        members.append(st.name)
    return None


def envelope(jobs: Sequence) -> "Resources":
    """The one allocation the members share, or :class:`GroupError` naming
    the member that cannot: one queue (its placement's domain), one count of
    ranks, cores per rank and GPUs, one GPU binding; the wall the sum of the
    members' walls (they run one after another), the memory the largest."""
    from ..scheduler.quantities import parse_memory, parse_walltime, slurm_time
    first = jobs[0]

    def shape(j):
        # the queue the placement admitted, else the one stated; no GPU
        # and GPU use off are one answer each
        r = j.resources
        got = {w: getattr(r, f) for f, w in SHARED.items()}
        got.update({"queue": (j.placement or {}).get("domain") or r.domain,
                    "GPUs": r.gres or None, "GPU use": bool(r.use_gpu)})
        return got

    want = shape(first)
    for j in jobs[1:]:
        got = shape(j)
        differ = [k for k in want if got[k] != want[k]]
        if differ:
            raise GroupError(
                f"`{j.name}` cannot share `{first.name}`'s job: its "
                + ", ".join(f"{k} {got[k]!r} against {want[k]!r}"
                            for k in differ)
                + ".  A group runs in one allocation; state the same for "
                  "each (project-layout.md § 1.6.6), or give them groups of "
                  "their own.")
    walls = [j.resources.time for j in jobs]
    wall = (slurm_time(sum(parse_walltime(str(w)) for w in walls))
            if all(walls) else None)
    # `0` ASKS FOR ALL OF THE NODE'S (`quantities.MEM_HELP`) -- the largest
    # ask there is, so a member stating it gives the group it.
    mems = [str(j.resources.mem) for j in jobs if j.resources.mem]
    mem = ("0" if "0" in mems
           else max(mems, key=parse_memory) if mems else None)
    r = first.resources
    import dataclasses
    return dataclasses.replace(
        r, time=wall, mem=mem,
        exclusive=any(j.resources.exclusive for j in jobs))


def number_of(task, stage: str) -> Optional[int]:
    """The group ``stage`` is in, as the description numbers it -- a name in
    any case one name (`engines/stages.md` § 2) -- or ``None`` for a stage
    the description does not hold.  The one lookup."""
    from ..identity import stage_key
    return next((st.group for st in task.stages
                 if stage_key(st.name) == stage_key(stage)), None)


def named(task, number: int) -> str:
    """Group ``number`` in words, by its members as well as its number --
    ``group 2 (device)`` -- since a group is matched by its members and a
    number shifts when a stage above it is removed (`engines/stages.md`
    § 2): what a saved state's note and a way back name it by."""
    return f"group {number} ({', '.join(group_members(task, number))})"


def members_of(job) -> List[str]:
    """The members of the job ``job`` is prepared in, in order (`Job.members`,
    or itself alone)."""
    return list(job.members) if job.members else [job.name]


def same_members(a: Sequence[str], b: Sequence[str]) -> bool:
    """Whether two member lists are one job's -- the same stages in the same
    order, a name in any case one name (`engines/stages.md` § 2): how a
    prepared job is matched to the description's group."""
    from ..identity import stage_key
    return [stage_key(m) for m in a] == [stage_key(m) for m in b]


def formed_again(task, jobs, numbers) -> List[int]:
    """The groups a prep of ``numbers`` names: those, and every group a
    member of a prepared job (``jobs``) of theirs is in now, wherever that
    job's members are no longer one group -- a job is formed again whole,
    in one prep (`job-system.md`, *The task*, step 3) -- until none is
    left out."""
    from ..identity import stage_key
    group_of = {stage_key(st.name): st.group for st in task.stages}
    jobs_of = {stage_key(j.name): members_of(j) for j in jobs}
    out: Set[int] = set(numbers)
    while True:
        more = set()
        for n in out:
            members = group_members(task, n)
            for st in members:
                job = jobs_of.get(stage_key(st))
                if job is not None and not same_members(job, members):
                    more |= {group_of[stage_key(m)] for m in job
                             if stage_key(m) in group_of}
        if more <= out:
            return sorted(out)
        out |= more


def group_members(task, number: int) -> List[str]:
    """The stages group ``number`` holds, in ladder order (`engines/stages.md`
    § 2) -- the one lookup."""
    return [st.name for st in task.stages if st.group == number]


def launched_without(task, jobset, number: int, *, base
                     ) -> Optional[Tuple[str, List[str]]]:
    """``(launched, joined)`` -- a stage of group ``number`` that was
    launched, and the stages the group holds now that it was launched
    without -- or ``None``.  A launched group's members are fixed
    (`job-system.md`, *The task*).  A launch record that does not read
    raises, as the one door does (`materialize.ever_launched`)."""
    from ..identity import stage_key
    from .materialize import ever_launched
    held = group_members(task, number)
    for m in held:
        job = jobset.job(m) if jobset else None
        if job is None or not ever_launched(jobset, base, job):
            continue
        joined = [h for h in held if not any(
            stage_key(h) == stage_key(x) for x in members_of(job))]
        if joined:
            return m, joined
    return None


def joined_launched(number: int, launched: str, joined: List[str], *, base,
                    task) -> str:
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
            + redo_launched(launched, base=base, task=task))


def regrouped(task, jobset, stage: str, *, base) -> Optional[str]:
    """Why ``stage``'s prepared job (in ``jobset``) is no longer the
    description's group -- the members it was prepared with, the group's
    now, and the prep that forms the job again, every group it needs named
    (:func:`formed_again`) -- or ``None``.  Groups are matched by their
    members (`engines/stages.md` § 2): the ready door says it, and launch
    refuses it.  Where a group that prep would name holds a stage beside one
    launched without it, that is said instead (:func:`joined_launched`)."""
    from ..identity import stage_key
    from ..runrecord import LaunchRecordError
    from .commands import block, prep_group_lines
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
            hit = launched_without(task, jobset, n, base=base)
        except LaunchRecordError as exc:
            return str(exc)
        if hit:
            return said + "; " + joined_launched(n, *hit, base=base,
                                                 task=task)
    return (said + ", so its prep forms the job again "
            "(engines/stages.md § 2):\n"
            + block(prep_group_lines(numbers, base=base)))


def header_of(base, jobset, job, shape):
    """The header of the job ``job`` is prepared in -- a group of several's,
    in the calculation's `launch/`, named by its members' own files, their
    rows in ``jobset`` (`materialize.run_names`) -- or ``None`` for a job of
    one.  What a prep removes when that job is formed again, or when a
    member's row leaves the plan."""
    from pathlib import Path
    from ..runfiles import LAUNCH_DIR
    from .materialize import run_names
    members = members_of(job)
    if len(members) < 2:
        return None
    jobs = [jobset.job(m) for m in members]
    return Path(base) / LAUNCH_DIR / GroupNames.of_stages(jobset.name, [
        run_names(jobset, j, shape).stage
        for j in jobs if j is not None]).name(".sbatch")


def names_for(base, task, members: Sequence[str]) -> GroupNames:
    """The names of the one job ``members`` make, as the description gives
    them -- its header, walker and log in `launch/`: the label and each
    member's token (`materialize.stage_home`).  What prep writes and launch
    sends; a job already written is named by its rows (:func:`header_of`)."""
    from .materialize import stage_home
    return GroupNames.of_stages(task.label, [stage_home(base, task, n).token
                                             for n in members])


__all__ = ["GroupError", "SHARED", "shared_card_keys", "card_shares",
           "upstream_of", "linked", "may_join",
           "default_groups", "numbered",
           "check_groups", "envelope", "number_of", "named", "members_of",
           "same_members", "formed_again", "regrouped", "header_of",
           "group_members", "launched_without", "joined_launched",
           "names_for"]
