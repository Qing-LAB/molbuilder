"""A SIESTA VIBRATION'S RUN OF POINTS, DECLARED -- its points are its
SIESTA calls (`engines/vibration.md` § 5.11), and this is the declaration
the kind table's record names for the pair (``declaration``,
`execution/runs-of-points.md` § 2.2).

MODULE  jobset.vibration_points (floor 4, beside `jobset.points`): the
        phases, the calls and a call's entry are `siesta.vibration_phases`'s,
        which travels beside the job and holds no part of a higher floor;
        what a reader of the run's points asks is answered here, from the
        phases table (`PHASES`) -- no phase is named.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ..calcdirs import RunPlace
from ..runfiles import COORDINATES_FILE, ROLE_SPECTRA, compose
from ..runpoints import index_at, index_of, read_index
from ..siesta.vibration_phases import (Call, calls_of, must_hold, phase,
                                       point_entry)
from .points import Container, PointsOf, Start


def _job_phases(calc, stage: str) -> Tuple[str, ...]:
    """The phases the stage's job was prepared with, as its row in the
    calculation's plan records them (`model.Job.phases`, written by prep)
    -- the plan as ``calc`` holds it: a prep's, with its rows merged in,
    else the one on disk; ``()`` for a stage prepared without phases, or
    none prepared."""
    job = calc.job(stage)
    return tuple(getattr(job, "phases", ()) or ())


def _prepared(phases) -> Tuple[Call, ...]:
    """The calls prep prepares, one per phase whose calls it does not leave
    to a planner -- in the phases' order."""
    return tuple(Call(p, p) for p in phases if not phase(p).planned)


def _declared_rung_points(calc, stage: str, *, read=None
                          ) -> Tuple[Call, ...]:
    """The calls known before the job -- one each for the phases the stage
    runs whose calls prep prepares (the relaxation, the force constants);
    a planned phase's calls exist only once their planner has run."""
    return _prepared(_job_phases(calc, stage))


def _declared_points_in(run, task, stage: str, base) -> List[Tuple[Path, Call]]:
    """A run's calls -- those its index lists, each with its folder below
    the run; before launch wrote the index, the calls prep opened (the
    plan's row's phases, `Job.phases`).  The index is the run's one,
    written by the launch that opened the run (`runpoints.index_of`, which
    refuses an older molbuilder's naming `jobset migrate`)."""
    from .naming import stage_home
    place = RunPlace.of(run)
    path = index_of(place.run_dir, stage_home(base, task, stage).names)
    if path is None:
        from .model import JobSet
        plan = JobSet.of(base)
        job = plan.job(stage) if plan is not None else None
        return [(place.point(c.rel), c)
                for c in _prepared(getattr(job, "phases", None) or ())]
    return [(place.point(c.rel), c) for c in calls_of(read_index(path))]


def _entry(calc, stage: str, point: Call, deck) -> dict:
    """The call's entry in the run's index -- the one shape the force
    constants' step writes too (`vibration_phases.point_entry`)."""
    e = point_entry(point, calc.task.label)
    return {k: e[k] for k in ("words", "phase", "round", "facts")}


def _must_hold(stage: str, label: str, base, point: Call, folder,
               deck=None):
    return must_hold(point, label)


def _start_of(calc, stage: str, point: Call, points) -> Start:
    """A CALL'S START AND NEEDS, its phase's row (`PHASES`; § 5.11 § 3):
    the seed -- the file of the phase it starts from -- from that phase's
    call where the run holds one, and the calls of the phases it needs."""
    rule = phase(point.phase)
    first = {}
    for q in points:
        first.setdefault(q.phase, q)
    source = first.get(rule.seed[0]) if rule.seed is not None else None
    return Start(source=source,
                 files=((compose(calc.task.label, rule.seed[1]),)
                        if source is not None else ()),
                 needs=tuple(first[n] for n in rule.needs if n in first))


def _containers(calc, stage: str) -> Tuple[Container, ...]:
    """The folders a run holds for its planned calls, opened with the run
    (§ 5.11 § 2): each planned phase the stage's job runs -- its folder,
    the call whose step plans its calls, and the coordinates its planning
    writes into each, which a launch again copies into one it runs
    again."""
    return tuple(Container(point=Call(p), planner=Call(rule.planned_by,
                                                       rule.planned_by),
                           inputs=(COORDINATES_FILE,))
                 for p in _job_phases(calc, stage)
                 if (rule := phase(p)).planned)


def _run_facts(calc, stage: str, *, carried, deck) -> dict:
    """The run's facts, for its index: the phases it walks (its job's row,
    `Job.phases`) -- what the count's words read -- and, where a launch
    again took the force constants over, the selection they applied, which
    the optical calls taken with them are its (§ 5.11 § 7)."""
    facts = {"phases": list(_job_phases(calc, stage))}
    if carried and carried.get("selection") is not None:
        facts["selection"] = carried["selection"]
    return facts


def _also_needs(run, names, task, stage: str, base):
    """What the run needs beyond its calls done: its finish step ended (the
    index's entry; one that exited non-zero is the run's state's own rule,
    `runpoints.run_state`) and its result on disk (§ 5.11 § 7)."""
    index = index_at(run, names)
    if index is None:
        return "no index of its calls"
    if (index.get("finish") or {}).get("ended") is None:
        return "the finish did not end"
    from ..runs import on_label_at
    if on_label_at(run, ROLE_SPECTRA) is None:
        return "the finish left no result at the run's root"
    return None


#: THE DECLARATION, named by the kind table's record (the pair's
#: ``points.declaration``): a SIESTA vibration run's points are its SIESTA
#: calls -- the relaxation's and the force constants' prepared, the optical
#: ones planned in the job.
POINTS = PointsOf(rung_points=_declared_rung_points,
                  points_in=_declared_points_in,
                  entry=_entry,
                  point_of=Call.from_entry,
                  must_hold=_must_hold,
                  start_of=_start_of,
                  verdict_counts=lambda stage, point: phase(
                      point.phase).verdict_counts,
                  relaunch_of=lambda stage, point: phase(
                      point.phase).relaunch,
                  containers=_containers,
                  run_facts=_run_facts,
                  also_needs=_also_needs)
