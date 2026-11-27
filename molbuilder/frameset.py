"""A MULTI-FRAME STRUCTURE'S ROWS, AS WRITTEN -- what it states of itself and
of each frame, and the one row a combination of its frames reads
(`model/structure.md` § 2.2f).

MODULE  frameset (L2 by the index, `architecture.md`; it imports only the
        Structure and the standard library)
ROLE    :func:`read` -- the structure's rows and each frame's as written,
        each row's meaning (:meth:`FrameRows.meaning`), and each frame's
        ``weight``; the weights' rule, its one home -- asked of a structure
        (:func:`weights_of`), of its rows (:func:`read`'s ``weights`` and
        ``warnings``), of weights given (:func:`sum_broken`); the weights
        assigned
        (:func:`with_weights`, :func:`equal_weights`)
USED-BY the frames' post-process (`frame_average`: the weighted average, the
        data file's frame part, what a page is answered, the frames
        averaged at weights given); the transport record
        (`transport.record`), its run's points (`transport.points`) and its
        citation (`transport.compose`); a vibration's set written, read
        back and its rows' weights (`vibration_frames`); a data file's
        frame set's rows (`postprocess.frame_rows`,
        :meth:`FrameRows.meaning` a script's to ask) and the weights
        assigned (`postprocess.with_weights`, `equal_weights`); the report
        log's reader (`parse.instruments.reports`, :func:`finite_number`)

**A calculation over the frames needs the frames and nothing else** (user,
2026-10-18: "the frame can of course carry meta data, and why you treat it as
transport's job to determine what should or should not be there?").  Every
row the structure and its frames carry is its maker's: carried as written,
never checked, never computed from -- but one: each frame's ``weight``, its
share of the frames' combined result, **required** (the user, 2026-11-26:
"weight is explicitly calculated and saved with multiframe data, and is
required, and will be checked for normalization"): every frame states one as
a finite number and they sum to 1 within :data:`WEIGHT_SUM_TOLERANCE` --
never filled in or rescaled here.  A set that breaks the rule is refused by
name wherever its frames are averaged or cited, by this one rule
(:func:`weights_of` of a structure, :func:`read` of its rows); :func:`read`
answers the weights only where they meet it, :attr:`FrameRows.warnings` the
rule's words where they do not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .structure import frame_words

#: The one row a combination of the frames reads: each frame's share of it.
WEIGHT_ROW = "weight"

#: How far the weights' sum may be from 1 -- past it a set is refused by name
#: (:func:`weights_of`); the record states it beside the average.
WEIGHT_SUM_TOLERANCE = 1e-6

#: THE WAY ON for a set whose weights break the rule -- the two doors that
#: assign them, as a script reaches them.
_ASSIGN_THEM = ("assign them: postprocess.with_weights(structure, weights), "
                "or postprocess.equal_weights(structure) for 1/F on every "
                "frame, frame 0 included")


#: WHY ONE STRUCTURE IS NOT WEIGHED -- the rule binds a frame set alone.
_ONE_STRUCTURE = ("one structure has no frames to weigh: a frame set holds "
                  "two frames or more")


class FrameWeightsError(ValueError):
    """A frame set's weights break the rule (`model/structure.md` § 2.2f):
    the message names the frames, a weight that is no number, or the sum,
    and the way on -- ready to surface verbatim."""


@dataclass(frozen=True)
class FrameRows:
    """A multi-frame structure's rows, as written: the structure's
    (``structure``, ``[{name, value, unit, note}]``), each frame's (``frames``,
    one such list per frame), the frames' weights where they meet the rule
    -- every frame's a finite number, their sum 1 within
    :data:`WEIGHT_SUM_TOLERANCE` -- (``weights``, else ``None``), and how
    they break it where they do (``warnings``)."""
    structure: Tuple[Dict[str, Any], ...]
    frames: Tuple[Tuple[Dict[str, Any], ...], ...]
    weights: Optional[Tuple[float, ...]]
    warnings: Tuple[str, ...]
    # ``weights`` every frame's where they meet the rule, else ``None`` and
    # ``warnings`` the rule's words -- one sentence.

    @property
    def weight_sum(self) -> Optional[float]:
        return math.fsum(self.weights) if self.weights is not None else None

    def value(self, name: str, frame: Optional[int] = None) -> Any:
        """Row ``name``'s value as written -- the structure's, or frame
        ``frame``'s -- or ``None`` where it states none."""
        rows = self.structure if frame is None else self.frames[frame]
        return next((r["value"] for r in rows if r["name"] == name), None)

    def meaning(self, name: str, *, of_frames: bool = False
                ) -> Optional[str]:
        """Row ``name``'s meaning as written (its ``note``) -- the
        structure's, or, ``of_frames``, the first frame's stating one, a
        frame row's one meaning (`model/structure.md` § 2.2f) -- or ``None``
        where none is written."""
        sets = self.frames if of_frames else (self.structure,)
        return next((str(r["note"]) for rows in sets for r in rows
                     if r["name"] == name and str(r.get("note") or "")),
                    None)


def finite_number(value: Any) -> Optional[float]:
    """``value`` as a finite number, or ``None`` -- true/false is not a
    number here, though Python counts it as one.  The one rule every
    reader of a frame's row asks (`frame_average`, `vibration_frames`), and
    the report log's reader of a report's sent time."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        v = float(value)
    except OverflowError:
        # AN INTEGER PAST ANY FLOAT is no finite number either.
        return None
    return v if math.isfinite(v) else None


def read(structure) -> Optional[FrameRows]:
    """``structure``'s rows and each frame's, as written, its frames'
    weights read by their rule -- or ``None`` for a structure that states
    no rows at all: one structure, or frames that state no rows, no weight
    among them (:func:`weights_said` says so)."""
    n = structure.n_frames
    srows = tuple(dict(r) for r in structure.customized_rows())
    frows = tuple(tuple(dict(r) for r in structure.customized_rows(f))
                  for f in range(n))
    if not srows and not any(frows):
        return None
    weights, warnings = _weights(frows, n)
    return FrameRows(structure=srows, frames=frows, weights=weights,
                     warnings=warnings)


def _weights(frows, n: int) -> Tuple[Optional[Tuple[float, ...]],
                                     Tuple[str, ...]]:
    """THE WEIGHTS' RULE (`model/structure.md` § 2.2f), its one home: each
    frame's :data:`WEIGHT_ROW` as a number where every frame states one as a
    finite number and they sum to 1 within :data:`WEIGHT_SUM_TOLERANCE` --
    else ``None`` and how they break it, one sentence, the frames named
    from 1."""
    if n < 2:
        return None, ()
    stated = [next((r["value"] for r in rows if r["name"] == WEIGHT_ROW),
                   None) for rows in frows]
    if all(v is None for v in stated):
        return None, (f"no frame states a {WEIGHT_ROW}",)
    gone = [frame_words(f, n) for f, v in enumerate(stated) if v is None]
    bad = [f"{frame_words(f, n)}'s is {v!r}" for f, v in enumerate(stated)
           if v is not None and finite_number(v) is None]
    if gone or bad:
        why = "; ".join(
            ([f"{', '.join(gone)} {'states' if len(gone) == 1 else 'state'} "
              f"none"] if gone else []) + bad)
        return None, (f"the frames' {WEIGHT_ROW} rows do not give every frame "
                      f"a number ({why})",)
    weights = tuple(float(v) for v in stated)
    broken = sum_broken(weights)
    return (None, (broken,)) if broken is not None else (weights, ())


def sum_broken(weights: Sequence[float]) -> Optional[str]:
    """How ``weights`` -- numbers, one per frame -- break the rule's sum, one
    sentence; ``None`` where they sum to 1 within
    :data:`WEIGHT_SUM_TOLERANCE`.  The rule's one sum, asked of a set's rows
    (:func:`_weights`) and of weights a person gives
    (`frame_average.averaged_at`)."""
    total = math.fsum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        return (f"the frames' weights sum to {total!r}, not 1 within "
                f"{WEIGHT_SUM_TOLERANCE:g}")
    return None


def weights_said(rows: Optional[FrameRows], n: int) -> Tuple[str, ...]:
    """How ``n`` frames' weights break the rule, as :func:`read` says it --
    ``rows``' warnings, or, for frames stating no rows at all (``rows``
    ``None``), that none states a weight; nothing for one structure or
    weights that meet it."""
    if rows is not None:
        return tuple(rows.warnings)
    return _weights(((),) * n, n)[1] if n >= 2 else ()


def weights_of(structure) -> Tuple[float, ...]:
    """``structure``'s frames' weights, by the rule (`model/structure.md`
    § 2.2f) -- the rule asked of a structure: by a transport citation and
    the motion toolkit's reader (its writer and re-weighting ask
    :func:`read`'s answer of the rows, the same rule).
    Refused by name (:class:`FrameWeightsError`): one structure, which has
    no frames to weigh; a frame stating none, one that is no finite number,
    a sum off 1 -- with the way on, the weights assigned."""
    n = structure.n_frames
    if n < 2:
        raise FrameWeightsError(_ONE_STRUCTURE)
    frows = tuple(tuple(dict(r) for r in structure.customized_rows(f))
                  for f in range(n))
    weights, why = _weights(frows, n)
    if weights is None:
        raise FrameWeightsError(
            f"{why[0]}: every frame's {WEIGHT_ROW} is required, their sum 1 "
            f"-- {_ASSIGN_THEM}")
    return weights


def with_weights(structure, weights: Sequence[Any]):
    """``structure`` with each frame's :data:`WEIGHT_ROW` set to
    ``weights`` -- one per frame, frame 0 first, dimensionless, its meaning
    the frame's share of the frames' average -- a new structure, the
    given one unchanged; every other row as it was.  Refused by name
    (:class:`FrameWeightsError`): one structure; weights not one per frame;
    and what the rule refuses (:func:`weights_of`)."""
    n = structure.n_frames
    if n < 2:
        raise FrameWeightsError(_ONE_STRUCTURE)
    given = list(weights) if isinstance(weights, (list, tuple)) else (
        list(weights.tolist()) if hasattr(weights, "tolist") else None)
    if given is None or len(given) != n:
        raise FrameWeightsError(
            f"the structure holds {n} frames: the weights are one per frame, "
            f"frame 0 first -- "
            + (f"a {type(weights).__name__} given" if given is None
               else f"{len(given)} given"))
    out = structure.copy()
    for f, w in enumerate(given):
        # A NUMPY NUMBER as Python's, so the row holds a plain number.
        w = w.item() if hasattr(w, "item") else w
        out.set_customized(WEIGHT_ROW, w, unit="1", frame=f,
                           note="the frame's share of the frames' average")
    weights_of(out)
    return out


def equal_weights(structure):
    """``structure`` with every frame weighing the same, ``1/F`` -- frame
    0 included (:func:`with_weights`); a set whose frame 0 is left out of
    the average gives its weights itself, frame 0's 0."""
    n = structure.n_frames
    return with_weights(structure, [1.0 / n] * n if n >= 2 else [])


__all__ = ["FrameRows", "FrameWeightsError", "WEIGHT_ROW",
           "WEIGHT_SUM_TOLERANCE", "equal_weights", "finite_number", "read",
           "sum_broken", "weights_of", "weights_said", "with_weights"]
