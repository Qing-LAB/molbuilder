"""A TRANSMISSION WINDOW FITTED TO A FINISHED GRID (`engines/transport.md`
§ 2a.16, *A wider window, fitted to the finished grid*): in a calculation
that reuses a finished one, the person states the range wanted -- and, to go
finer, how many times finer, an odd number -- and the window is fitted to
the finished point's grid, as its deck states it: its spacing, over that
factor, and its alignment, extended outward by whole spacings until the
range asked is covered.  Every finished energy is then a point of the new
grid, and every new energy lies at the same spacing (:func:`fitted`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Fitted:
    """A window fitted to a finished grid, and what was asked of it."""
    #: the range asked, and how many times finer
    lo: float
    hi: float
    finer: int
    #: the window: its ends and its number of energies, mid-rule
    emin: float
    emax: float
    points: int
    #: the finished grid's energies, every one kept
    kept: int

    @property
    def spacing(self) -> float:
        return (self.emax - self.emin) / self.points

    def words(self) -> str:
        """``-3…3 eV asked, fitted to the finished grid -- -3.00748…3.00748
        eV at 603 points, 0.00997506 eV apart, every one of its 401
        energies kept``."""
        finer = f", {self.finer} times finer" if self.finer > 1 else ""
        return (f"{self.lo:g}…{self.hi:g} eV asked{finer}, fitted to the "
                f"finished grid -- {self.emin:.5f}…{self.emax:.5f} eV at "
                f"{self.points} points, {self.spacing:.6g} eV apart, every "
                f"one of its {self.kept} energies kept")


def fitted(window: Tuple[float, float, int], lo: float, hi: float,
           finer: int = 1) -> Fitted:
    """THE WINDOW FITTED to the finished grid ``window`` -- its ends and its
    number of energies, mid-rule, as the finished point's deck states them
    (`jobset.reuse.finished_window`) -- covering ``lo``…``hi`` eV, its
    spacing the finished one over ``finer``: the finished window cut
    ``finer`` to a spacing, extended by whole spacings each side until its
    end energies reach the range asked -- the finished window itself, end
    for end, when it already does at the same spacing.
    :class:`ValueError` names what cannot be fitted: a range that is not
    one, a factor that is not an odd whole number."""
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"the window asked, {lo:g}…{hi:g} eV, is not a "
                         f"range -- give two energies, the first below the "
                         f"second")
    if (isinstance(finer, bool) or not isinstance(finer, int) or finer < 1
            or finer % 2 == 0):
        raise ValueError(f"how many times finer must be an odd whole "
                         f"number (1, 3, 5 …) so every finished energy "
                         f"stays a point of the finer grid -- {finer!r} is "
                         f"not")
    a, b, n = float(window[0]), float(window[1]), int(window[2])
    h = (b - a) / n / finer
    # WHOLE SPACINGS OUTWARD until the end energies -- half a spacing inside
    # each end -- cover the range asked.
    left = max(0, math.ceil((a + h / 2.0 - lo) / h - 1e-9))
    right = max(0, math.ceil((hi - (b - h / 2.0)) / h - 1e-9))
    return Fitted(lo=float(lo), hi=float(hi), finer=finer,
                  emin=a - left * h, emax=b + right * h,
                  points=n * finer + left + right, kept=n)


__all__ = ["Fitted", "fitted"]
