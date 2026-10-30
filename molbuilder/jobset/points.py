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
decided at prep (`jobset.transport_points`); a SIESTA vibration's are the
SIESTA calls of its one run -- its phases' -- the relaxation's and the force
constants' prepared, the optical ones planned in the job
(`jobset.vibration_points`).  Both are read alike: a point is a folder below
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
    #: ``(run_dir, task, stage, base) -> [(folder, point)]``: the points a
    #: run holds, each with its folder below the run.
    points_in: Callable
    #: ``(calc, stage, point, deck) -> {words, phase, round, facts}``: the
    #: kind's part of the point's entry in the run's index -- ``deck`` the
    #: path of the deck prep wrote for it.
    entry: Callable
    #: ``(entry) -> point``: the point an entry of the run's index is -- a
    #: planned point a launch again takes with its planner.
    point_of: Callable
    #: ``(stage, label, base, point, folder, deck=None) ->
    #: [(product, words)]``: what the point in ``folder`` must hold to be
    #: DONE, each with the words its absence is said in -- the list the
    #: done door is asked with (`continuation.done`) and the index carries
    #: for the job; ``deck`` a prepared deck's path where no run holds the
    #: point's deck yet (launch writing the index).
    must_hold: Callable
    #: ``(calc, stage, point, points) -> Start``: its start and its needs,
    #: among ``points``, the run's in the listing order.
    start_of: Callable
    #: ``(stage, point) -> bool``: whether its VERDICT counts toward done
    #: -- a vibration's relaxation is done finished, converged or not.
    verdict_counts: Callable = lambda stage, point: True
    #: ``(stage, point) -> "again" | "continue"``: what a launch again does
    #: with it not done (`runpoints.AGAIN`, `runpoints.CONTINUE`).
    relaunch_of: Callable = lambda stage, point: AGAIN
    #: ``(calc, stage) -> (Container, ...)``: the folders below a run that
    #: hold its planned points; ``()`` for a kind with none.
    containers: Callable = lambda calc, stage: ()
    #: ``(calc, stage, *, carried, deck) -> dict``: the kind's facts about
    #: the run, for its index -- ``carried`` those of the run whose planner
    #: a launch again took over, ``deck`` a prepared point's deck's path.
    run_facts: Callable = (lambda calc, stage, *, carried, deck: {})
    #: ``(run_dir, names, task, stage, base) -> Optional[str]``: what the
    #: run needs BEYOND its points done to be finished -- ``None`` when
    #: nothing, else why not: a vibration's finish step, which writes the
    #: result once no call is left (`engines/vibration.md` § 5.11 § 7).
    also_needs: Callable = lambda run, names, task, stage, base: None


def record_of(task):
    """``task``'s pair's run of points, as the kind table states it
    (`kind_table.RunOfPoints`, `runs-of-points.md` § 2.1) -- ``None`` for
    a pair whose every rung runs once."""
    from ..kind_table import run_of_points
    return run_of_points(str(getattr(task, "calculation", "")),
                         str(getattr(task, "engine", "")))


def points_of(task) -> Optional[PointsOf]:
    """``task``'s pair's declaration -- the part its record names
    (``declaration``, `execution/architecture.md` § 3.3) -- or ``None`` for
    a pair whose every rung runs once."""
    from ..kind_table import part_of
    return part_of(str(getattr(task, "calculation", "")),
                   str(getattr(task, "engine", "")), "points")


def walls_of(task, n: int) -> int:
    """How many of the card's walls a run of ``n`` points of ``task``'s
    kind asks: one where its points share ONE wall, else one each
    (`runs-of-points.md` § 1.2); ``0`` for none.  ``n`` for a kind whose
    every rung runs once, where ``n`` is a group's count."""
    rec = record_of(task)
    return min(n, 1) if rec is not None and rec.wall == WALL_ONE else n


def rung_points_of(calc, stage: str) -> tuple:
    """``stage``'s points as prepared, by its kind's declaration -- ``()``
    for a kind with none, or a rung that runs once (:attr:`PointsOf.
    rung_points`)."""
    pts = points_of(calc.task)
    return tuple(pts.rung_points(calc, stage)) if pts is not None else ()


__all__ = ["Start", "Container", "PointsOf", "record_of", "points_of",
           "walls_of", "rung_points_of"]
