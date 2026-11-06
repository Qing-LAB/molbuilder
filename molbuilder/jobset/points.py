"""A RUN OF POINTS, DECLARED -- what a pair's points are, as the code parts
every reader of them asks (`execution/runs-of-points.md` § 2.2).

MODULE  jobset.points (floor 4: a run of points; each pair's declaration
        is named by its record in the kind table, `kind_table.RunOfPoints`,
        and imported when asked for)
ROLE    one door, the pair's declaration (:class:`PointsOf`), so that no
        reader of a run's points asks a kind's name: prep's conductor
        writes one deck per prepared point, launch's one walk
        (`submit._points_walk`) takes the done ones over and walks the
        rest, `status`, Task setup, the Results ladder and the record read
        each point's own files -- each through the declaration, never
        through the kind.  What never changes for the pair -- what one
        point is called, when its run is finished, its wall, its words, its
        own steps -- is the kind table's record (:func:`record_of`).
USED-BY `jobset.submit`, `jobset.continuation`, `jobset.runstatus`,
        `jobset.ready`, `jobset.prepared_with`, `jobset.materialize`
        (`open_sweep_run`), `jobset.prep` (`walk_words`, `walls_of`),
        `runs`

A transport rung's points are a frame of the cited set and a voltage,
decided at prep (`transport.points`); a SIESTA vibration's are the
SIESTA calls of its one run -- its phases' -- the relaxation's and the force
constants' prepared, the optical ones planned in the job
(`spectra.vibration_points`).  Both are read alike: a point is a folder below
the run holding that point's deck, run script, output and records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from ..kind_table import WALL_ONE
from ..runpoints import AGAIN


@dataclass(frozen=True)
class Start:
    """A POINT'S START AND NEEDS (`runs-of-points.md` § 1.2): the point it
    starts from -- its ``source`` -- and the files it takes from it, one
    hand-over the walk copies just before the point runs; and the points it
    needs to have RAN before it runs, its source among them.  An empty start
    begins from what its run gathered, or from what prep wrote."""
    source: object = None
    files: Tuple[str, ...] = ()
    needs: Tuple[object, ...] = ()


@dataclass(frozen=True)
class Container:
    """A FOLDER BELOW THE RUN THAT HOLDS PLANNED POINTS (`runs-of-points.md`
    § 1.1): its ``point`` -- the kind's point type for the folder itself,
    whose ``rel`` is the folder and whose ``prepared`` folder holds the deck
    its points run -- its ``planner``, the point whose after-step plans its
    points, and the ``inputs`` its planning writes into each planned point
    (what a launch again copies into one it runs again, never its
    outputs)."""
    point: object
    planner: object
    inputs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PointsOf:
    """HOW A KIND'S RUNGS RUN PER POINT -- each part one question
    (`runs-of-points.md` § 2.2).

    A point is the kind's own type (`transport.stages.Point`,
    `siesta.vibration_phases.Call`), answering ``rel`` (its folder below
    the run), ``prepared`` (its prepared folder below the stage),
    ``words()`` and ``to_wire()``."""
    #: ``(calc, stage, *, read=None) -> (point, ...)``: the rung's PREPARED
    #: points, in the listing order -- what prep writes a deck for below the
    #: stage and a first launch opens -- ``()`` for a rung that runs once.
    #: ``calc`` is the calculation whole (`calculation.Calculation`, A16),
    #: its plan as the caller holds it; ``read`` reads a record's text
    #: where the file is a prep's plan's, not yet on disk.
    rung_points: Callable
    #: ``(run_dir, calc, stage) -> [(folder, point)]``: the points a run
    #: holds, each with its folder below the run.  ``calc`` is the
    #: calculation whole, as :attr:`rung_points` takes it: what a kind
    #: reads of the calculation -- its plan, before a launch listed the
    #: run's points -- is the caller's, never read a second time (A16).
    points_in: Callable
    #: ``(calc, stage, point, deck) -> {words, phase, round, facts}``: the
    #: kind's part of the point's entry in the run's index -- ``deck`` the
    #: path of the deck prep wrote for it.
    entry: Callable
    #: ``(entry) -> point``: the point an entry of the run's index is -- a
    #: planned point a launch again takes with its planner.
    point_of: Callable
    #: ``(calc, stage, point, folder, deck=None) ->
    #: [(product, words)]``: what the point in ``folder`` must hold to be
    #: DONE, each with the words its absence is said in -- the list the
    #: done door is asked with (`continuation.done`) and the index carries
    #: for the job; ``deck`` a prepared deck's path where no run holds the
    #: point's deck yet (launch writing the index).
    must_hold: Callable
    #: ``(calc, stage, point, points) -> Start``: its start and its needs,
    #: among ``points``, the run's in the listing order.
    start_of: Callable
    #: ``(point) -> bool``: whether its VERDICT counts toward done
    #: -- a vibration's relaxation is done finished, converged or not.
    verdict_counts: Callable = lambda point: True
    #: ``(point) -> "again" | "continue"``: what a launch again does
    #: with it not done (`runpoints.AGAIN`, `runpoints.CONTINUE`).
    relaunch_of: Callable = lambda point: AGAIN
    #: ``(calc, stage) -> (Container, ...)``: the folders below a run that
    #: hold its planned points; ``()`` for a kind with none.
    containers: Callable = lambda calc, stage: ()
    #: ``(calc, stage, *, carried, deck) -> dict``: the kind's facts about
    #: the run, for its index -- ``carried`` those of the run whose planner
    #: a launch again took over, ``deck`` a prepared point's deck's path.
    run_facts: Callable = (lambda calc, stage, *, carried, deck: {})
    #: ``(calc, stage, run_dir, index) -> Optional[str]``: what the
    #: run needs BEYOND its points done to be finished -- ``None`` when
    #: nothing, else why not: a vibration's finish step, which writes the
    #: result once no call is left (`engines/vibration.md` § 5.11 § 7).
    also_needs: Callable = lambda calc, stage, run, index: None
    #: ``(point) -> bool``: whether a stage that continues from another
    #: (`ladder.stage_before`) takes ``point`` over from that stage's run
    #: -- never computing it itself -- where that run has it done: a
    #: displacement sweep's later stage, the relaxation
    #: (`engines/vibration.md` § 5.9).
    carried: Callable = lambda point: False
    #: ``(calc, stage) -> bool``: whether ``stage`` runs as a run of points,
    #: asked before any prep as after: a kind whose rungs' points are known
    #: only once prepared says so here.
    runs_points: Callable = (lambda calc, stage: None)
    #: ``() -> point | None``: the one point a run of a stage is
    #: when the rung runs once -- a transport rung's, at 0 V and no frame
    #: (`engines/transport.md` § 2a.10) -- what :func:`point_of` answers
    #: for the run of a rung whose folder opens its calculation's report at
    #: a point; ``None`` for a kind with none.
    one_point: Callable = lambda: None


def record_of(task):
    """``task``'s pair's run of points, as the kind table states it
    (`kind_table.RunOfPoints`, `runs-of-points.md` § 2.1) -- ``None`` for
    a pair whose every rung runs once."""
    from ..kind_table import run_of_points
    return run_of_points(str(task.calculation), str(task.engine))


def points_of(task) -> Optional[PointsOf]:
    """``task``'s pair's declaration -- the part its record names
    (``declaration``, `execution/architecture.md` § 3.3) -- or ``None`` for
    a pair whose every rung runs once."""
    from ..kind_table import part_of
    return part_of(str(task.calculation), str(task.engine), "points")


def walls_of(task, n: int) -> int:
    """How many of the card's walls a run of ``n`` points of ``task``'s
    kind asks: one where its points share ONE wall, else one each
    (`runs-of-points.md` § 1.2); ``0`` for none.  ``n`` for a kind whose
    every rung runs once, where ``n`` is a group's count."""
    rec = record_of(task)
    return min(n, 1) if rec is not None and rec.wall == WALL_ONE else n


def carried_of(calc, stage: str) -> tuple:
    """The points ``stage``'s run takes over from the run it continues from
    -- its prepared points its kind carries (:attr:`PointsOf.carried`: a
    displacement sweep's later stage, the relaxation) -- ``()`` for a kind
    with none, or a stage not prepared."""
    pts = points_of(calc.task)
    return (tuple(pt for pt in pts.rung_points(calc, stage) if pts.carried(pt))
            if pts is not None else ())


def point_of(calc, folder):
    """THE POINT A FOLDER IS (`execution/architecture.md` § 3.2) -- by its
    kind's declaration, of the folder ``calcdirs.where`` places: a point's
    folder below a run of points, its point (the declaration's walk of the
    run, :attr:`PointsOf.points_in`, matched by its folder); a transmission
    point's ``shared/``, the point it lies in; the run of a rung that runs
    once whose folder opens its calculation's report at a point (the kind
    table's role ``opens``), its one point (:attr:`PointsOf.one_point`) --
    every other run-once rung has none (`web/results.md` § 2.4).  ``None``
    for any other folder, a benchmark trial's, and a folder no calculation
    claims.  ``calc`` is the opened calculation the folder lies in, as the
    caller holds it (A16).  The point object, served by its one wire form
    (``to_wire()``)."""
    from .. import calcdirs
    from ..identity import stage_name
    from ..kind_table import kind
    from ..runrecord import LaunchRecordError
    # A ``shared/`` FOLDER IS READ WITH THE POINT IT LIES IN
    # (`calcdirs.read_with`, `engines/transport.md` § 2a.16).
    at = calcdirs.where(calcdirs.read_with(folder))
    if at.root is None or at.run_folder is None or at.trial is not None:
        return None
    declared = points_of(calc.task)
    if declared is None:
        return None
    stage = stage_name(at.stage)
    if at.what == calcdirs.POINT:
        try:
            return next((pt for _f, pt in declared.points_in(calc, stage, at.run_folder) if pt.rel == at.point), None)
        except LaunchRecordError:
            # ITS RUN'S RECORDS DO NOT READ -- an older molbuilder's index:
            # no point, the run's own row saying why (`continuation.
            # read_points`), never the whole answer.
            return None
    if (at.what == calcdirs.ATTEMPT and at.role == calcdirs.RUN
            and kind(str(calc.task.calculation)).opens_report(stage)):
        return declared.one_point()
    return None


def rung_points_of(calc, stage: str) -> tuple:
    """``stage``'s points as prepared, by its kind's declaration -- ``()``
    for a kind with none, or a rung that runs once (:attr:`PointsOf.
    rung_points`)."""
    pts = points_of(calc.task)
    return tuple(pts.rung_points(calc, stage)) if pts is not None else ()


def runs_points(calc, stage: str) -> bool:
    """Whether ``stage`` of the calculation runs as a run of points, whether or
    not it is prepared yet (`runs-of-points.md` § 1.2: its wall is typed before
    its first prep): the kind's own word (:attr:`PointsOf.runs_points`), else
    whether it has points now."""
    pts = points_of(calc.task)
    if pts is None:
        return False
    said = pts.runs_points(calc, stage)
    return bool(pts.rung_points(calc, stage)) if said is None else bool(said)


__all__ = ["Start", "Container", "PointsOf", "record_of", "points_of",
           "walls_of", "carried_of", "point_of", "rung_points_of",
           "runs_points"]
