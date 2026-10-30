"""A TRANSPORT RUNG'S RUN OF POINTS, DECLARED -- a frame of the cited set
and a voltage each, decided at prep (`engines/transport.md` § 2a.11), and
this is the declaration the kind table's record names for the pair
(``declaration``, `execution/runs-of-points.md` § 2.2).

MODULE  jobset.transport_points (floor 4, beside `jobset.points`): the
        rungs' points, their folders, what each must hold and hands on are
        `transport.stages`'s, which lower floors read; what a reader of a
        run's points asks is answered here, from them.
"""
from __future__ import annotations

from pathlib import Path

from ..transport.stages import (STAGE_FACT, Point, frames_of, hands_on,
                                must_hold, points_in, rung_points)
from .points import PointsOf, Start


def _deck_text(deck):
    """A deck's text, its includes read in (`parse.fdf.read_deck`)."""
    from ..parse.fdf import read_deck
    return read_deck(Path(deck))


def _declared_rung_points(calc, stage: str, *, read=None):
    return rung_points(calc.task, stage, frames=frames_of(calc.base, read=read))


def _declared_points_in(run, task, stage: str, base):
    return points_in(run, task, stage, base=base)


def _entry(calc, stage: str, point: Point, deck) -> dict:
    """The point's entry in the run's index (`runs-of-points.md` § 2.3):
    its words, the rung its one phase, its voltage its round, and what it
    is -- its frame and voltage, and the energy grid its deck describes
    (`energies.deck_window`), what a transmission round's outcome puts its
    printed energies on."""
    from ..transport.energies import deck_window
    return {"words": point.words(), "phase": stage, "round": point.round,
            "facts": {**point.facts(),
                      "window": (list(deck_window(_deck_text(deck)) or ())
                                 or None)}}


def _must_hold(stage: str, label: str, base, point, folder, deck=None):
    return must_hold(stage, label, folder, base_dir=base,
                     deck_text=_deck_text(deck) if deck is not None else None)


def _start_of(calc, stage: str, point: Point, points) -> Start:
    """A POINT'S START (`engines/transport.md` § 2a.9, § 2a.11): the closest
    voltage computed before it IN ITS FRAME -- of the points before it in
    the listing, the nearest in volts, the later on a tie (the SIESTA
    manual: "copy the TSDE from the closest, previously, calculated bias")
    -- and the files the rung hands on (`stages.hands_on`: the device's
    ``.TSDE``); it needs that point.  A frame's first voltage, and every
    point of a rung that hands nothing on (the transmission), starts from
    what its run gathered: frames never start from one another."""
    hand = hands_on(stage, calc.task.label, calc.base)
    if not hand:
        return Start()
    mine = [q for q in points if q.frame == point.frame]
    earlier = mine[:mine.index(point)]
    if not earlier:
        return Start()
    prior = min(reversed(earlier),
                key=lambda q: abs(float(q.volts) - float(point.volts)))
    return Start(source=prior, files=tuple(hand), needs=(prior,))


def _run_facts(calc, stage: str, *, carried, deck) -> dict:
    """The run's facts, for its index (`runs-of-points.md` § 2.3): what its
    points compute (the kind table's ``computes``), the leads' pair and
    each lead as their ``TS.Elecs`` blocks name them, the spin the decks
    state (`record.spin_of`) and the frames' weights, where the cited set
    states them (`frameset.read`) -- what a round's outcome reads."""
    from ..frameset import read as read_frame_set
    from ..parse.fdf import parse_fdf_params
    from ..transport.compose import recorded_junction
    from ..transport.record import spin_of
    from ..transport.transiesta import LEAD_NAMES, LEAD_PAIR
    recorded = recorded_junction(Path(calc.base))
    rows = (read_frame_set(recorded.structure) if recorded is not None
            else None)
    return {"computes": STAGE_FACT[stage], "pair": LEAD_PAIR,
            "leads": list(LEAD_NAMES),
            "spin": spin_of(parse_fdf_params(_deck_text(deck),
                                             source=Path(deck).name)),
            "weights": (list(rows.weights) if rows is not None
                        and rows.weights is not None else None)}


#: THE DECLARATION, named by the kind table's record (the pair's
#: ``points.declaration``): a transport rung's points are a frame of the
#: cited set and a voltage each, decided at prep.
POINTS = PointsOf(rung_points=_declared_rung_points,
                  points_in=_declared_points_in,
                  entry=_entry,
                  point_of=Point.from_entry,
                  must_hold=_must_hold,
                  start_of=_start_of,
                  run_facts=_run_facts)
