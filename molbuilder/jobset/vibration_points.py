"""A SIESTA VIBRATION'S RUN OF POINTS, DECLARED -- its points are its
SIESTA calls (`engines/vibration.md` § 5.11 § 11), and this is the
declaration the kind table names for the pair (``points``,
`execution/architecture.md` § 3.3).

MODULE  jobset.vibration_points (floor 4, beside `jobset.points`): the
        phases, the calls and the index are `siesta.vibration_phases`'s,
        which travels beside the job and holds no part of a higher floor;
        what a reader of the run's points asks -- its calls, their folders,
        what a call must hold to be done, what the run needs beyond them --
        is answered here, from them.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..runfiles import FIRST_ATTEMPT, ROLE_SPECTRA
from ..siesta.vibration_phases import (FC, OPTICAL, RELAX, Call, calls_of,
                                       entry_of, index_path, phase,
                                       products_of, read_index, summary)
from .points import PointsOf


def _job_phases(calc, stage: str) -> Tuple[str, ...]:
    """The phases the stage's job was prepared with, as its row in the
    calculation's plan records them (`model.Job.phases`, written by prep)
    -- the plan as ``calc`` holds it: a prep's, with its rows merged in,
    else the one on disk; ``()`` for a stage prepared without phases, or
    none prepared."""
    job = calc.job(stage)
    return tuple(getattr(job, "phases", ()) or ())


def _declared_rung_points(calc, stage: str, *, read=None
                          ) -> Tuple[Call, ...]:
    """The calls known before the job -- one each for the relaxation and
    the force constants the stage runs (:func:`_job_phases`); the optical
    calls exist only once the force constants have run."""
    return tuple(Call(p, p) for p in _job_phases(calc, stage)
                 if p in (RELAX, FC))


def _declared_points_in(run, task, stage: str, base) -> List[Tuple[Path, Call]]:
    """A run's calls -- those its index lists (:func:`read_index`), each
    with its folder below the run; before launch wrote the index, the calls
    prep opened, the plan's row's phases (`Job.phases`).  The index is the
    run's one, written at its first launch (a run folder holds one run of
    points; a call's retries are the call's own)."""
    from .naming import stage_home
    run = Path(run)
    names = stage_home(base, task, stage).names(task.shape)
    path = index_path(run, names, FIRST_ATTEMPT)
    if not path.is_file():
        # BEFORE LAUNCH WROTE THE INDEX: the calls prep opened -- the
        # plan's row's phases (`Job.phases`), as a transport rung lists its
        # prepared points.
        from .model import JobSet
        plan = JobSet.of(base)
        job = plan.job(stage) if plan is not None else None
        return [(run / Call(p, p).rel, Call(p, p))
                for p in (getattr(job, "phases", None) or ())
                if p in (RELAX, FC)]
    return [(run / c.rel, c) for c in calls_of(read_index(path))]


def _declared_hands_on(stage: str, label: str, base) -> list:
    return []


def _declared_containers(calc, stage: str) -> Tuple[str, ...]:
    """The folders a run holds besides its calls, opened with the run
    (§ 5.11 § 2): the optical calls' container, when the stage's job runs
    the optical phase -- the plan step puts each call below it."""
    return (OPTICAL,) if OPTICAL in _job_phases(calc, stage) else ()


def _declared_summary(rows, run, names) -> str:
    """The count's words for `status` (`points.PointsOf.summary`): the
    phases the run walks read off its index."""
    path = index_path(run, names, FIRST_ATTEMPT)
    try:
        phases = read_index(path).get("phases") or ()
    except (OSError, ValueError):
        phases = ()
    return summary(rows, phases)


def _record_says(run, names, point: Call) -> Optional[str]:
    """What the run's index says against ``point`` being done: its record
    step's failure, where it failed (§ 5.11 § 7) -- ``None`` with no index
    yet, or no entry."""
    path = index_path(run, names, FIRST_ATTEMPT)
    if not path.is_file():
        return None
    row = entry_of(read_index(path), point) or {}
    return row.get("record_error") or None


def _also_needs(run, names, task, stage: str, base) -> Optional[str]:
    """What the run needs beyond its calls done: its finish ended with exit
    0 (the index's entry) and its result on disk (§ 5.11 § 7)."""
    path = index_path(run, names, FIRST_ATTEMPT)
    if not path.is_file():
        return "no index of its calls"
    fin = (read_index(path).get("finish") or {})
    if fin.get("ended") is None:
        return "the finish did not end"
    if fin.get("exit") not in (0, "0"):
        return f"the finish exited {fin.get('exit')}: no result"
    from ..runs import on_label_at
    if on_label_at(run, ROLE_SPECTRA) is None:
        return "the finish left no result at the run's root"
    return None


#: THE DECLARATION, named by the kind table (the pair's ``points``): a
#: SIESTA vibration run's points are its SIESTA calls -- the relaxation's and
#: the force constants' known at launch, the optical ones planned in the job.
POINTS = PointsOf(kind="vibration",
                   rung_points=_declared_rung_points,
                   points_in=_declared_points_in,
                   products_of=products_of,
                   hands_on=_declared_hands_on,
                   summary=_declared_summary,
                   converges=lambda stage: True,
                   verdict_counts=lambda stage, point: phase(
                       point.phase).verdict_counts,
                   record_says=_record_says,
                   also_needs=_also_needs,
                   containers=_declared_containers,
                   planned_in_job=True, one_wall=True, unit="call")
