"""A RUNG OF POINTS, DECLARED -- which calculation kinds run a rung per
point, and the doors every reader of its points asks (`job-system.md`, *The
task*, *a run of points*; `engines/transport.md` § 2a.11;
`engines/vibration.md` § 5.11 § 11).

MODULE  jobset.points (floor 2: reads the description's kind; each kind's
        own module is imported when its declaration is asked for)
ROLE    one table, keyed by the calculation kind, so that no reader of a
        run's points asks ``calculation == "transport"``: prep's conductor
        writes one deck per prepared folder, launch walks a run's points and
        takes the done ones over, `status`, Task setup, the Results ladder
        and the record read each point's own files -- each through the
        kind's declaration (:class:`PointsOf`), never through its name.
USED-BY `jobset.submit`, `jobset.continuation`, `jobset.runstatus`,
        `jobset.ready`, `jobset.prepared_with`, `runs`

A transport calculation's points are a frame of the cited set and a voltage,
decided at prep (`transport.stages`); a SIESTA vibration's are the SIESTA
calls of its one run -- its phases' -- the first known at launch, the
optical ones planned in the job (`siesta.vibration_phases`).  Both are read
alike: a point is a folder below the run holding that point's deck, run
script, output and records.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class PointsOf:
    """How a kind's rungs run per point -- every door a reader asks.

    Each callable takes the description (``task``), the stage's name and
    the calculation's folder (``base``) where it needs them; a point is the
    kind's own type (`transport.stages.Point`, `siesta.vibration_phases.
    Call`), with ``rel`` (its folder below the stage or a run), ``words()``
    and ``to_wire()`` alike (`job-system.md`, *A point*)."""
    kind: str
    #: ``(calc, stage, *, read=None) -> (point, ...)``: the rung's points
    #: as PREPARED -- the folders prep writes a deck into below the stage,
    #: what a first launch opens -- ``()`` for a rung that runs once.
    #: ``calc`` is the calculation whole (`calculation.Calculation`, A16):
    #: its plan as the caller holds it -- a prep's, with its rows merged in
    #: (a vibration's phases are its job's row) -- else the one on disk;
    #: ``read`` reads a record's text where the file is a prep's plan's,
    #: not yet on disk (the junction's record).
    rung_points: Callable
    #: ``(run_dir, task, stage, base) -> [(folder, point)]``: a run's points,
    #: each with its folder below the run -- the prepared ones (transport);
    #: those the run's own record lists, planned in the job (a vibration's
    #: optical calls).
    points_in: Callable
    #: ``(stage, label, base, point=None) -> products``: what a point must
    #: hold to be DONE, the list the done door is asked with
    #: (`continuation.done`) -- by the stage alone (transport), or by the
    #: point, whose phase decides it (a vibration's calls).
    products_of: Callable
    #: ``(stage, label, base) -> [file]``: what a point hands the next
    #: point's start -- ``[]`` where points start from nothing of each
    #: other's.
    hands_on: Callable
    #: ``(rows, run_dir, names) -> str``: the count and its axes, in the
    #: kind's words -- *k of N points done (F frames × V voltages)*; *relax
    #: done · fc done · optical 7 of 11* (the phases the run walks read off
    #: its index).
    summary: Callable
    #: ``stage -> bool``: whether a point of ``stage`` converges an SCF --
    #: the converged column (`job-system.md`, rule 4); the transmission,
    #: a product of the device's, converges nothing.
    converges: Callable
    #: ``(run_dir, names, task, stage, base) -> Optional[str]``: what a run
    #: of points needs BEYOND its points done to be finished -- ``None``
    #: when nothing, else why not: a vibration's finish, which writes the
    #: result after its last call (`engines/vibration.md` § 5.11 § 7).
    also_needs: Callable = lambda run, names, task, stage, base: None
    #: ``(calc, stage) -> (rel, ...)``: the folders a run holds besides its
    #: points, opened with the run as containers
    #: (`materialize.open_sweep_run`) -- a vibration's optical calls'
    #: container, below which the job plans each call
    #: (`engines/vibration.md` § 5.11 § 2); ``()`` for a kind with none.
    containers: Callable = lambda calc, stage: ()
    #: Whether the run's points are decided in the job, after a call has
    #: run, rather than at prep -- the walk's shape (`submit`): a list of
    #: members, or the phases' loop.
    planned_in_job: bool = False
    #: ``(stage, point) -> bool``: whether a point's VERDICT counts toward
    #: done -- a run that finished without converging is not done where it
    #: does (`continuation.done`); a vibration's relaxation call is done
    #: finished, converged or not (`engines/vibration.md` § 5.11 § 5, § 6).
    verdict_counts: Callable = lambda stage, point: True
    #: Whether the run's points SHARE THE STAGE'S ONE WALL -- the card's
    #: wall is the run's, every point counting from the run's start (a
    #: vibration's calls, `engines/vibration.md` § 5.11 § 7a) -- rather than
    #: each point told the card's wall from its own start, the job's wall
    #: their sum (a bias sweep's points, `engines/transport.md` § 2a.11).
    one_wall: bool = False

    def walls(self, n: int) -> int:
        """How many of the card's walls a run of ``n`` points asks: one
        where its points share it, else one each; ``0`` for none."""
        return min(n, 1) if self.one_wall else n


#: THE KINDS THAT RUN A RUNG PER POINT, each by where its declaration is --
#: ``(engine, calculation) -> "module:NAME"``, the module imported when
#: asked, so this floor imports no kind.  A kind not here runs every rung
#: once.
DECLARED: Dict[Tuple[str, str], str] = {
    ("siesta", "transport"): "molbuilder.transport.stages:POINTS",
    ("siesta", "vibration"): "molbuilder.siesta.vibration_phases:POINTS",
}


def points_of(task) -> Optional[PointsOf]:
    """``task``'s kind's declaration (:data:`DECLARED`), or ``None`` for a
    kind whose every rung runs once."""
    key = (str(getattr(task, "engine", "")),
           str(getattr(task, "calculation", "")))
    where = DECLARED.get(key)
    if where is None:
        return None
    module, _, name = where.partition(":")
    return getattr(importlib.import_module(module), name)


def walls_of(task, n: int) -> int:
    """How many of the card's walls a run of ``n`` points of ``task``'s
    kind asks (:meth:`PointsOf.walls`) -- ``n`` for a kind whose every rung
    runs once, where ``n`` is a group's count."""
    pts = points_of(task)
    return pts.walls(n) if pts is not None else n


def rung_points_of(calc, stage: str) -> tuple:
    """``stage``'s points as prepared, by its kind's declaration -- ``()``
    for a kind with none, or a rung that runs once (:attr:`PointsOf.
    rung_points`)."""
    pts = points_of(calc.task)
    return tuple(pts.rung_points(calc, stage)) if pts is not None else ()


__all__ = ["PointsOf", "DECLARED", "points_of", "walls_of",
           "rung_points_of"]
