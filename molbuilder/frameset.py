"""A MULTI-FRAME STRUCTURE'S ROWS, AS WRITTEN -- what it states of itself and
of each frame, and the one row a combination of its frames reads
(`model/structure.md` § 2.2f).

MODULE  frameset (L1; the Structure and the standard library)
ROLE    :func:`read` -- the structure's rows and each frame's as written, and
        each frame's ``weight``
USED-BY the frames' post-process (`frame_average`: the weighted average, the
        data file's frame part, what a page is answered)

**A calculation over the frames needs the frames and nothing else** (user,
2026-10-18: "the frame can of course carry meta data, and why you treat it as
transport's job to determine what should or should not be there?").  Every
row the structure and its frames carry is its maker's: carried as written,
never checked, never computed from -- but one: each frame's ``weight``, its
share of the frames' combined result, read where every frame states one as a
number.  A weight missing from some frames or from all, or weights that do
not sum to 1, are said in :attr:`FrameRows.warnings` -- the average is then
not taken, or taken at the weights as written -- never refused.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .structure import frame_words

#: The one row a combination of the frames reads: each frame's share of it.
WEIGHT_ROW = "weight"

#: How far the weights' sum may be from 1 before it is said -- the record
#: states it beside the average.
WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FrameRows:
    """A multi-frame structure's rows, as written: the structure's
    (``structure``, ``[{name, value, unit}]``), each frame's (``frames``,
    one such list per frame), the frames' weights where every frame states
    one as a number (``weights``, else ``None``), and what is said of them
    (``warnings``)."""
    structure: Tuple[Dict[str, Any], ...]
    frames: Tuple[Tuple[Dict[str, Any], ...], ...]
    weights: Optional[Tuple[float, ...]]
    warnings: Tuple[str, ...]

    @property
    def weight_sum(self) -> Optional[float]:
        return math.fsum(self.weights) if self.weights is not None else None

    def value(self, name: str, frame: Optional[int] = None) -> Any:
        """Row ``name``'s value as written -- the structure's, or frame
        ``frame``'s -- or ``None`` where it states none."""
        rows = self.structure if frame is None else self.frames[frame]
        return next((r["value"] for r in rows if r["name"] == name), None)

    def structure_rows(self) -> Dict[str, Any]:
        """The structure's rows, ``{name: value}``, as written."""
        return {r["name"]: r["value"] for r in self.structure}


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite number, or ``None`` -- true/false is not a
    number here, though Python counts it as one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def read(structure) -> Optional[FrameRows]:
    """``structure``'s rows and each frame's, as written, its frames'
    weights read -- or ``None`` for a structure that states no rows at all:
    one structure, or frames each its own curve."""
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
    """Each frame's :data:`WEIGHT_ROW` as a number, where every frame states
    one -- else ``None`` -- and what is said of them."""
    if n < 2:
        return None, ()
    stated = [next((r["value"] for r in rows if r["name"] == WEIGHT_ROW),
                   None) for rows in frows]
    if all(v is None for v in stated):
        return None, (f"no frame states a {WEIGHT_ROW}: no average is taken "
                      f"-- each frame's curve is its own",)
    gone = [frame_words(f, n) for f, v in enumerate(stated) if v is None]
    bad = [f"{frame_words(f, n)}'s is {v!r}" for f, v in enumerate(stated)
           if v is not None and _number(v) is None]
    if gone or bad:
        why = "; ".join(
            ([f"{', '.join(gone)} {'states' if len(gone) == 1 else 'state'} "
              f"none"] if gone else []) + bad)
        return None, (f"the frames' {WEIGHT_ROW} rows do not give every frame "
                      f"a number ({why}): no average is taken -- each frame's "
                      f"curve is its own",)
    weights = tuple(float(v) for v in stated)
    total = math.fsum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        return weights, (f"the frames' weights sum to {total!r}, not 1: the "
                         f"average is taken at the weights as written",)
    return weights, ()


__all__ = ["FrameRows", "WEIGHT_ROW", "WEIGHT_SUM_TOLERANCE", "read"]
