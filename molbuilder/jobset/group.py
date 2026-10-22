"""A GROUP: the stages that run as one job, by the number each states in
the job request (`execution/job-request.md` § 3.3, `project-layout.md`
§ 1.6.6; plan § 5aa).

Stages that do not build on one another -- a transport ladder's seed and its
two leads -- wait in the queue once instead of once each.  Each member is
prepared as it would be alone and keeps its own attempt, deck, run script
and records; the group is written on each member's job (`Job.members`) with
one header for the group's job, and launch sends that job, walking the
members in the ladder's order.

This module answers the group's questions, each in one place:

* **does a stage build on another** (:func:`upstream_of`) -- the one fact
  the description already states: a transport rung's inputs
  (`transport.stages.stage_inputs`), a force-constant stage's `relax`, an
  independent stage's stage before (`ladder.source_stage`) -- and so
  **the numbers a new calculation takes** (:func:`default_groups`,
  :func:`fresh_groups`) and **whether the stated ones keep it**
  (:func:`check_groups`);
* **can the members share one allocation** (:func:`envelope`) -- one queue,
  one count of ranks, cores per rank and GPUs; the wall the sum of theirs,
  the memory the largest;
* **what the group's files are called** (:func:`names_for`,
  `runfiles.GroupNames.of_stages`) -- one
  ``GroupNames`` from the members' tokens, in order.

The numbers are the job request's; what a stage builds on is the
description's.  A function that asks both takes both, and reads each as
itself (A17).
"""
from __future__ import annotations

from dataclasses import dataclass
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



def card_shares(request, stage: str) -> dict:
    """``stage``'s run card (`JobRequest.run_condition`) on what a job's
    members share (:data:`SHARED`), by field, as the run takes it: a setting
    no card states at the catalogue's one default (`template.default_of`),
    a queue none states at the allocation's (`job-request.md` § 3.2a) --
    ``None`` where nothing answers, and everywhere for a ladder with no
    request yet."""
    if request is None:
        return {f: None for f in SHARED}
    from .. import template as _T
    from .model import AS_RESOURCE
    kind = str(request.calculation)
    taken = {it.name: _T.default_of(it, kind)
             for it in _T.select(_T.catalogue(), engine=str(request.engine))
             if _T.is_run_setting(it)}
    taken["domain"] = request.allocation.domain or None
    taken.update(request.run_condition(stage))
    card = {AS_RESOURCE.get(k, k): v for k, v in taken.items()}
    return {f: card.get(f) for f in SHARED}


def upstream_of(task, stage: str,
                template_text: Optional[str] = None) -> Set[str]:
    """The stages ``stage`` builds on, as the description states it: a
    transport rung's inputs, a force-constant stage's `relax`, an
    independent stage's stage before it (none for one set `restart: clean`,
    or the first)."""
    if getattr(task, "calculation", None) == "transport":
        from ..transport.stages import inputs_of
        return {src for src, _f in inputs_of(task, stage)}
    from .ladder import source_stage
    src, _linked = source_stage(task, stage, template_text)
    return {src} if src else set()


def linked(task, stage: str,
           template_text: Optional[str] = None) -> bool:
    """Whether what ``stage`` builds on is its KIND's -- a transport rung's
    inputs, a force-constant stage's `relax` -- which no override changes;
    ``False`` for an independent ladder's stage, which builds on the stage
    before it only while its `restart` continues (`job-system.md` § 5.4)."""
    if getattr(task, "calculation", None) == "transport":
        return True
    from .ladder import source_stage
    return bool(source_stage(task, stage, template_text)[1])


def may_join(task, request, stage: str, members: Sequence[str],
             template_text: Optional[str] = None) -> bool:
    """Whether ``stage`` may share the job ``members`` make, the group
    above it, by default (`execution/job-request.md` § 3.3: the defaults
    repeat the number above wherever that is allowed): it builds on none of
    them (:func:`upstream_of`), and its run card states what theirs do on
    what a job shares (:func:`card_shares`) -- else prep would refuse the
    group (`job-system.md` R-P13).  The one rule every writer of numbers
    asks -- a new calculation's (:func:`default_groups`) and an older one's
    (`migrate`)."""
    return (bool(members)
            and not upstream_of(task, stage, template_text)
            & set(members)
            and card_shares(request, stage)
            == card_shares(request, members[0]))


def why_not_join(task, request, stage: str, members: Sequence[str],
                 template_text: Optional[str] = None) -> Optional[str]:
    """Why ``stage`` may not share the job ``members`` make -- the sentence
    the Task setup page shows beside its row (`web/task-setup.md` § 5),
    said here and nowhere else -- or ``None`` where :func:`may_join` says
    it may.  The two conditions :func:`may_join` asks, each in words."""
    if not members:
        return None
    fed = upstream_of(task, stage, template_text) & set(members)
    if fed:
        named_ = ", ".join(f"`{n}`" for n in members if n in fed)
        kind_s = linked(task, stage, template_text)
        return (f"`{stage}` builds on {named_}, in the group above: one job "
                f"runs its members one after another with nothing handed "
                f"between them"
                + ("" if kind_s else
                   " -- or, for it to start from its own deck, set its "
                   "`restart` override to `clean`"))
    if card_shares(request, stage) != card_shares(request, members[0]):
        return (f"`{stage}`'s run card states other values than the group "
                f"above's on what one job shares; set them alike, or give "
                f"it a new group")
    return None


def default_groups(task, request=None,
                   template_text: Optional[str] = None) -> Tuple[int, ...]:
    """The numbers a new calculation takes (`execution/job-request.md`
    § 3.3): the first stage group 1; each later one the number above it
    wherever it may join that group (:func:`may_join`), else one more -- a
    transport ladder 1 1 1 2 3, an optimization ladder 1 2 3, stages set
    `restart: clean` 1 1 1.  ``request`` is the run cards they are asked
    against; ``None`` for a ladder that states none."""
    out: List[int] = []
    members: List[str] = []
    for st in task.stages:
        if out and may_join(task, request, st.name, members, template_text):
            out.append(out[-1])
            members.append(st.name)
        else:
            out.append(out[-1] + 1 if out else 1)
            members = [st.name]
    return tuple(out)


@dataclass(frozen=True)
class _Fresh:
    """A ladder before it is a description -- what :func:`default_groups`
    reads of one: the kind, the engine and the stages."""
    calculation: str
    engine: str
    stages: tuple


def fresh_groups(stages, *, engine: str, calculation: str,
                 template_text: str) -> Tuple[int, ...]:
    """The groups of ``stages``, a fresh ladder (`ladder.fresh_stages`), by
    :func:`default_groups` -- the rule a new calculation is numbered with,
    asked before any description holds the ladder: the hand-over's proposal
    (`web/handover-procedure.md`), which Task setup shows as what will be
    written."""
    like = _Fresh(calculation=str(calculation), engine=str(engine),
                  stages=tuple(stages))
    return default_groups(like, None, template_text)


def check_groups(task, request, template_text: Optional[str] = None
                 ) -> Optional[str]:
    """Why the stated groups cannot run -- a stage sharing a group with one
    it builds on, both named -- or ``None`` (`execution/job-request.md`
    § 3.3): one job runs its members one after another with nothing handed
    between them.  The numbers are the request's; what a stage builds on,
    the description's."""
    seen: dict = {}
    for st in task.stages:
        number = request.group_of(st.name)
        members = seen.setdefault(number, [])
        fed = upstream_of(task, st.name, template_text) & set(members)
        if fed:
            names = [f"`{n}`" for n in members if n in fed]
            on = (", ".join(names[:-1]) + " and " + names[-1]
                  if len(names) > 1 else names[0])
            # THE OTHER WAY OUT, where there is one: an independent stage
            # builds on the stage before it only while it continues, so set
            # clean it builds on nothing; a linked stage's input is its
            # kind's, whatever its overrides say.
            kind_s = linked(task, st.name, template_text)
            return (f"`{st.name}` is in group {number} with {on}, which "
                    f"it builds on: one job runs its members one after "
                    f"another with nothing handed between them, so "
                    f"`{st.name}` would start before you have looked at "
                    f"what {on} produced (execution/job-request.md § 3.3).  "
                    f"Give `{st.name}` a new group -- one more than the "
                    f"group above" + ("." if kind_s else
                                      f"; or, for it to start from its own "
                                      f"deck rather than from {on}, set its "
                                      f"`restart` override to `clean`."))
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
        # the queue the placement bound, else the one stated; no GPU
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


def own_request(jobs: Sequence, points: int = 0) -> "Resources":
    """The request a group's own job sends -- what its own header asks
    (`job-system.md` § 6.0, *The placement*): its members' shared request
    (:func:`envelope`) for several; a run of points', every point's wall
    (``points`` of them); a stage's own, alone.  Its queue the placement's
    domain, where prep placed it."""
    import dataclasses
    if len(jobs) > 1:
        r = envelope(jobs)
    elif points:
        r = envelope([jobs[0]] * points)
    else:
        r = jobs[0].resources
    domain = (jobs[0].placement or {}).get("domain") or r.domain
    return dataclasses.replace(r, domain=domain)


def sequence_request(requests: Sequence[Tuple[int, "Resources"]]
                     ) -> "Resources":
    """THE ONE REQUEST OF A SEQUENCE OF GROUPS (`job-system.md` § 6.0, *A
    sequence of groups*) -- ``requests``, each group's number and its own
    request (:func:`own_request`), in order -- or :class:`GroupError` naming
    what they cannot share: one queue, every group's; the LARGEST ranks,
    cores per rank and memory any group asks for; the walls summed; the
    node whole when any asks it; the GPUs the groups that use them ask
    alike -- one count, one binding.  A group asking less runs with its own
    card inside it; one using no GPU runs on its cores."""
    import dataclasses
    from ..scheduler.quantities import (parse_gres_flag, parse_memory,
                                        parse_walltime, slurm_time)
    from .model import gpu_request
    queues = {n: r.domain for n, r in requests}
    if len(set(queues.values())) > 1:
        raise GroupError(
            "one job sits in one queue, and these groups name "
            + ", ".join(f"group {n} {q!r}" for n, q in queues.items())
            + " -- state one queue on their run cards, or send each alone")
    gpu = [(n, r) for n, r in requests if gpu_request(r).uses]
    # UNSTATED IS THE RULE'S, ON (`model.Resources.gpu_binding`): only an
    # explicit off differs.
    bindings = {n: r.gpu_binding is not False for n, r in gpu}
    if len(set(bindings.values())) > 1:
        raise GroupError(
            "one job binds its GPUs one way, and these groups ask "
            + ", ".join(f"group {n} {'on' if b else 'off'}"
                        for n, b in bindings.items())
            + " -- state the same `gpu_binding`, or send each alone")
    # ONE SET OF GPUS: a group that uses GPUs runs its ranks over every GPU
    # the job holds (the run script's rank helper spreads them over what the
    # scheduler gives), so the groups that use them ask the same count.
    counts = {n: parse_gres_flag(r.gres) for n, r in gpu}
    if len(set(counts.values())) > 1:
        raise GroupError(
            "one job holds one set of GPUs, which every group using GPUs "
            "runs on, and these groups ask "
            + ", ".join(f"group {n} {c}" for n, c in counts.items())
            + " -- state the same count, or send each alone")
    rs = [r for _n, r in requests]

    def largest(field: str):
        got = [getattr(r, field) for r in rs]
        return None if any(v is None for v in got) else max(got)
    gres = max((r.gres for _n, r in gpu),
               key=lambda g: parse_gres_flag(g) or 0, default=None)
    walls = [r.time for r in rs]
    mems = [str(r.mem) for r in rs if r.mem]
    return dataclasses.replace(
        rs[0], domain=rs[0].domain,
        mpi_np=largest("mpi_np"), cpus_per_task=largest("cpus_per_task"),
        gres=gres, use_gpu=bool(gpu),
        gpu_binding=(gpu[0][1].gpu_binding if gpu
                     else rs[0].gpu_binding),
        time=(slurm_time(sum(parse_walltime(str(w)) for w in walls))
              if all(walls) else None),
        mem=("0" if "0" in mems
             else max(mems, key=parse_memory) if mems else None),
        exclusive=any(r.exclusive for r in rs))


def sequence_names(base, task, request, numbers: Sequence[int]
                   ) -> GroupNames:
    """The names of a sequence of groups' one job -- its header, walk, log
    and launch record in `launch/`: the label and every member's token,
    the groups in order (`runfiles.GroupNames.of_sequence`)."""
    from .naming import stage_home
    return GroupNames.of_sequence(task.label, [
        stage_home(base, task, n).token
        for g in sorted(numbers) for n in request.members(g)])


def number_of(request, stage: str) -> Optional[int]:
    """The group ``stage`` is in, as the job request numbers it -- a name in
    any case one name -- or ``None`` for a stage it does not hold.  The one
    lookup (`JobRequest.group_of`)."""
    return request.group_of(stage)


def named(request, number: int) -> str:
    """Group ``number`` in words, by its members as well as its number --
    ``group 2 (device)`` -- since a group is matched by its members and a
    number shifts when a stage above it is removed
    (`execution/job-request.md` § 3.3): what a saved state's note and a way
    back name it by."""
    return f"group {number} ({', '.join(request.members(number))})"


def members_of(job) -> List[str]:
    """The members of the job ``job`` is prepared in, in order (`Job.members`,
    or itself alone)."""
    return list(job.members) if job.members else [job.name]


def same_members(a: Sequence[str], b: Sequence[str]) -> bool:
    """Whether two member lists are one job's -- the same stages in the same
    order, a name in any case one name (`engines/stages.md` § 2): how a
    prepared job is matched to the job request's group."""
    from ..identity import stage_key
    return [stage_key(m) for m in a] == [stage_key(m) for m in b]


def formed_again(request, jobs, numbers) -> List[int]:
    """The groups a prep of ``numbers`` names: those, and every group a
    member of a prepared job (``jobs``) of theirs is in now, wherever that
    job's members are no longer one group -- a job is formed again whole,
    in one prep (`job-system.md`, *The task*, step 3) -- until none is
    left out."""
    from ..identity import stage_key
    group_of = {stage_key(n): e.group for n, e in request.stages.items()}
    jobs_of = {stage_key(j.name): members_of(j) for j in jobs}
    out: Set[int] = set(numbers)
    while True:
        more = set()
        for n in out:
            members = request.members(n)
            for st in members:
                job = jobs_of.get(stage_key(st))
                if job is not None and not same_members(job, members):
                    more |= {group_of[stage_key(m)] for m in job
                             if stage_key(m) in group_of}
        if more <= out:
            return sorted(out)
        out |= more


def group_members(request, number: int) -> List[str]:
    """The stages group ``number`` holds, in ladder order -- the one lookup
    (`JobRequest.members`)."""
    return request.members(number)


def header_of(calc, job):
    """The header of the job ``job`` is prepared in -- a group of several's,
    in the calculation's `launch/`, named by its members' own files, their
    rows in ``calc``'s plan (`naming.run_names`) -- or ``None`` for a
    job of one.  What a prep removes when that job is formed again, or when
    a member's row leaves the plan."""
    from pathlib import Path
    from ..runfiles import LAUNCH_DIR
    from .naming import run_names
    base, jobset, shape = calc.base, calc.jobset, calc.shape
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
    member's token (`naming.stage_home`).  What prep writes and launch
    sends; a job already written is named by its rows (:func:`header_of`)."""
    from .naming import stage_home
    return GroupNames.of_stages(task.label, [stage_home(base, task, n).token
                                             for n in members])


__all__ = ["GroupError", "SHARED", "card_shares",
           "upstream_of", "linked", "may_join",
           "default_groups", "fresh_groups",
           "check_groups", "envelope", "number_of", "named", "members_of",
           "same_members", "formed_again", "header_of",
           "group_members",
           "names_for", "own_request", "sequence_names", "sequence_request"]
