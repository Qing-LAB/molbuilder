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

from ..transport.points_index import UNIT, summary
from ..transport.stages import (STAGE_FACT, frames_of, hands_on, points_in,
                                products_of, rung_points)
from .points import PointsOf


def _declared_rung_points(calc, stage: str, *, read=None):
    return rung_points(calc.task, stage, frames=frames_of(calc.base, read=read))


def _declared_summary(rows, run, names) -> str:
    return summary(rows)


def _declared_points_in(run, task, stage: str, base):
    return points_in(run, task, stage, base=base)


def _declared_products(stage: str, label: str, base, point=None):
    return products_of(stage, label, base_dir=base)


def _taken_from(run, names, point):
    """The run ``point`` was taken over from, as launch wrote it into the
    run's index (`engines/transport.md` § 2a.11)."""
    from ..transport.points_index import entry_of, index_path, read_index
    path = index_path(run, names)
    if not path.is_file():
        return None
    return (entry_of(read_index(path), point.rel) or {}).get(
        "taken_from") or None


def _run_index(calc, stage: str, run, names, points):
    """THE RUN'S INDEX, as launch writes it (`engines/transport.md` § 2a.11;
    `transport.points_index`): every point of the run in the walk's order --
    its folder, words and frame's, voltage, the point it needs, the run it
    was taken over from, the energy grid its deck describes
    (`energies.deck_window`) and what it must hold to be done, each with the
    words its absence is said in (`continuation.must_hold`, asked of its
    prepared deck) -- with what the rung's points compute (the
    kind table's ``computes``), the leads' pair, the spin the decks state
    (`record.spin_of`) and the frames' weights, where the cited set states
    them (`frameset.read`).  ``(path, index)``."""
    from pathlib import Path
    from ..frameset import read as read_frame_set
    from ..parse.fdf import parse_fdf_params, read_deck
    from ..structure import frame_words
    from ..transport.compose import recorded_junction
    from ..transport.energies import deck_window
    from ..transport.points_index import add_point, empty_index, index_path
    from ..transport.record import spin_of
    from ..transport.transiesta import LEAD_NAMES, LEAD_PAIR
    from .continuation import must_hold
    base, task = Path(calc.base), calc.task
    recorded = recorded_junction(base)
    rows = (read_frame_set(recorded.structure) if recorded is not None
            else None)
    script = calc.job(stage).script
    decks = {rp.point.rel: read_deck(Path(rp.prepared) / script)
             for rp in points}
    index = empty_index(
        stage=stage, computes=STAGE_FACT[stage], pair=LEAD_PAIR,
        leads=LEAD_NAMES,
        spin=spin_of(parse_fdf_params(decks[points[0].point.rel],
                                      source=script)),
        weights=rows.weights if rows is not None else None)
    products = products_of(stage, task.label, base_dir=base)
    for rp in points:
        pt = rp.point
        deck = decks[pt.rel]
        add_point(index, rel=pt.rel, words=pt.words(), frame=pt.frame,
                  frame_words=(frame_words(pt.frame, pt.of)
                               if pt.frame is not None else None),
                  bias_v=pt.bias_v, needs=rp.needs,
                  must_hold=must_hold(rp.prepared, products, deck_text=deck),
                  verdict_counts=POINTS.verdict_counts(stage, pt),
                  window=deck_window(deck), taken_from=rp.taken_from)
    return index_path(run, names), index


#: THE DECLARATION, named by the kind table (the pair's ``points``): a
#: transport rung's points are a frame of the cited set and a voltage each,
#: decided at prep.
POINTS = PointsOf(rung_points=_declared_rung_points,
                  points_in=_declared_points_in,
                  products_of=_declared_products,
                  hands_on=hands_on,
                  summary=_declared_summary,
                  taken_from=_taken_from,
                  run_index=_run_index,
                  unit=UNIT)
