"""A GROUP: stages prepared together to share one job (`project-layout.md`
§ 1.6.6; plan § 5u.1 step 5, TD2).

Stages that do not build on one another -- a transport ladder's seed and its
two leads -- wait in the queue once instead of once each.  You group them at
prep, by naming them; each is prepared as it would be alone and keeps its own
attempt, deck, run script and records; the group is written on each
member's job (`Job.group`) with one header for the group's job, and launch
sends that job, walking the members in the order named.

This module answers the group's three questions, each in one place:

* **does any member build on another** (:func:`upstream_of`) -- the one fact
  the ladder already states: a transport rung's inputs
  (`transport.stages.stage_inputs`), a force-constant stage's `relax`, an
  independent stage's stage before (`continuation._source_stage`);
* **can the members share one allocation** (:func:`envelope`) -- one queue,
  one count of ranks, cores per rank and GPUs; the wall the sum of theirs,
  the memory the largest;
* **what the group's files are called** (:func:`names_of`) -- one
  ``GroupNames`` from the members' tokens, in order.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Set

from ..runfiles import GroupNames


class GroupError(ValueError):
    """The stages named cannot be one group -- the message says which and
    why, in the person's words."""


def upstream_of(base, task, stage: str,
                template_text: Optional[str] = None) -> Set[str]:
    """The stages ``stage`` builds on, as the ladder states it: a transport
    rung's inputs, a force-constant stage's `relax`, an independent stage's
    stage before it (none for one set `restart: clean`, or the first)."""
    if getattr(task, "calculation", None) == "transport":
        from ..transport.stages import stage_inputs
        with_seed = "seed" in {s.name for s in task.stages}
        return {src for src, _f in stage_inputs(stage, task.label,
                                                 with_seed=with_seed)}
    from .continuation import _source_stage
    src, _linked = _source_stage(base, task, stage, template_text)
    return {src} if src else set()


def refuse_feeding(base, task, stages: Sequence[str],
                   template_text: Optional[str] = None) -> Optional[str]:
    """Why ``stages`` cannot be one group -- a member another member builds
    on, named -- or ``None``."""
    named = set(stages)
    for s in stages:
        fed = upstream_of(base, task, s, template_text) & named
        if fed:
            return (f"`{s}` builds on `{', '.join(sorted(fed))}`, so they "
                    f"cannot share one job: whether `{s}` should run depends "
                    f"on what `{', '.join(sorted(fed))}` produced, which you "
                    f"look at first (project-layout.md § 1.6.6).  Prep them "
                    f"apart.")
    return None


def envelope(jobs: Sequence) -> "Resources":
    """The one allocation the members share, or :class:`GroupError` naming
    the member that cannot: one queue (its placement's domain), one count of
    ranks, cores per rank and GPUs, one GPU binding; the wall the sum of the
    members' walls (they run one after another), the memory the largest."""
    from ..scheduler.quantities import parse_memory, parse_walltime, slurm_time
    from .model import Resources
    first = jobs[0]

    def shape(j):
        r = j.resources
        return {"queue": (j.placement or {}).get("domain") or r.domain,
                "ranks": r.mpi_np, "cores per rank": r.cpus_per_task,
                "GPUs": r.gres or None, "GPU use": bool(r.use_gpu),
                "GPU binding": r.gpu_binding}

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
                  "each (project-layout.md § 1.6.6), or prep them apart.")
    walls = [j.resources.time for j in jobs]
    wall = (slurm_time(sum(parse_walltime(str(w)) for w in walls))
            if all(walls) else None)
    mems = [j.resources.mem for j in jobs if j.resources.mem]
    mem = (max(mems, key=lambda m: parse_memory(str(m))) if mems else None)
    r = first.resources
    import dataclasses
    return dataclasses.replace(
        r, time=wall, mem=mem,
        exclusive=any(j.resources.exclusive for j in jobs))


def names_of(label: str, tokens: Sequence[str]) -> GroupNames:
    """The group's files' names: the calculation's label and every member's
    token, in order -- ``chain-t_group-01_seed-02_electrode_L-03_electrode_R``."""
    return GroupNames(f"{label}_group-{'-'.join(tokens)}")


def group_of(jobset, stage: str) -> Optional[List[str]]:
    """The group ``stage`` was prepared in, or ``None``."""
    job = next((j for j in jobset.jobs if j.name == stage), None)
    return list(job.group) if job is not None and job.group else None


__all__ = ["GroupError", "upstream_of", "refuse_feeding", "envelope",
           "names_of", "group_of"]
