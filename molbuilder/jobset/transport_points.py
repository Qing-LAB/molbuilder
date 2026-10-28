"""A TRANSPORT RUNG'S RUN OF POINTS, DECLARED -- a frame of the cited set
and a voltage each, decided at prep (`engines/transport.md` § 2a.11), and
this is the declaration the kind table names for the pair (``points``,
`execution/architecture.md` § 3.3).

MODULE  jobset.transport_points (floor 4, beside `jobset.points`): the
        rungs' points, their folders, inputs and products are
        `transport.stages`'s, which lower floors read; what a reader of a
        run's points asks is answered here, from them.
"""
from __future__ import annotations

from ..transport.stages import (STAGE_FACT, frames_of, hands_on, points_in,
                                points_summary, products_of, rung_points)
from .points import PointsOf


def _declared_rung_points(calc, stage: str, *, read=None):
    return rung_points(calc.task, stage, frames=frames_of(calc.base, read=read))


def _declared_summary(rows, run, names) -> str:
    return points_summary(rows)


def _declared_points_in(run, task, stage: str, base):
    return points_in(run, task, stage, base=base)


def _declared_products(stage: str, label: str, base, point=None):
    return products_of(stage, label, base_dir=base)


def _declared_hands_on(stage: str, label: str, base):
    return hands_on(stage, label, base)


def _converges(stage: str) -> bool:
    return STAGE_FACT.get(stage, "scf") != "product"


#: THE DECLARATION, named by the kind table (the pair's ``points``): a
#: transport rung's points are a frame of the cited set and a voltage each,
#: decided at prep.
POINTS = PointsOf(rung_points=_declared_rung_points,
                  points_in=_declared_points_in,
                  products_of=_declared_products,
                  hands_on=_declared_hands_on,
                  summary=_declared_summary,
                  converges=_converges)
