"""THE CONDUCTANCE AT THE FERMI LEVEL -- G/G0 = T(E_F), read off a
transmission on energies measured from E_F (`engines/transport.md` § 2a.4).

MODULE  transport.conductance (stdlib only; travels in ``mb_monitor.pyz``,
        `runwrap.members`)
ROLE    the one rule, the one rule of the transmission it is read on -- per
        spin channel (:func:`per_spin_channel`) -- and the spin a deck's
        parameters state (:func:`spin_of`), asked by the record
        for every point (`transport.record`) and by a transmission run's
        report step for its round's milestone (`execution/run-reports.md`
        § 2.9): so the number a message says is the record's rule, on the
        values it is given; and the frames' averaged conductance at their
        weights (:func:`frames_average`).
USED-BY `transport.record`, `transport.run_words`, `transport.tbtnc`,
        `transport.points`, `frame_average`
"""
from __future__ import annotations

import math
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence

try:                                        # inside molbuilder
    from ..runfiles import (SPIN_NON_POLARIZED, SPIN_POLARIZED,
                            TBT_SPIN_CHANNELS)
except ImportError:                         # beside a job
    from runfiles import (  # type: ignore[no-redef]
        SPIN_NON_POLARIZED, SPIN_POLARIZED, TBT_SPIN_CHANNELS)


def per_spin_channel(curves: Dict[str, Sequence[float]], spin: str
                     ) -> List[float]:
    """T(E) PER SPIN CHANNEL, the transmission G = G0 · T(E_F) is read on
    (`engines/transport.md` § 2a.12): the unpolarized channel's as it is; a
    polarized point's two averaged, (T↑ + T↓)/2.  ``curves`` holds each
    channel the run wrote, its T on the one grid; a channel the deck's
    ``spin`` has TBtrans write and ``curves`` lacks is :class:`ValueError`,
    in the words that say the point not done."""
    want = TBT_SPIN_CHANNELS[spin]
    missing = [c for c in want if c not in curves]
    if missing:
        raise ValueError(
            f"the deck is spin-polarized and the run wrote no "
            f"{' or '.join(missing)} channel transmission"
            if spin == SPIN_POLARIZED else "the run wrote no transmission file")
    return [math.fsum(vals) / len(vals)
            for vals in zip(*(curves[c] for c in want))]


def at_fermi(energies: Sequence[float], trans: Sequence[float]
             ) -> Optional[float]:
    """T interpolated linearly at E = 0 -- the grid is E − E_F, ascending,
    as TBtrans writes it -- the conductance in G0; ``None`` where the
    window does not straddle E_F: never an extrapolated number."""
    e = [float(x) for x in energies]
    t = [float(x) for x in trans]
    if not e or min(e) > 0 or max(e) < 0:
        return None
    j = bisect_right(e, 0.0)
    if j >= len(e):
        return t[-1]
    i = j - 1
    if e[j] == e[i]:
        return t[i]
    return t[i] + (t[j] - t[i]) * (0.0 - e[i]) / (e[j] - e[i])


def spin_of(params) -> str:
    """The spin a deck's parameters state (`parse.fdf.parse_fdf_params`) --
    ``non-polarized`` or ``polarized`` -- in any word SIESTA accepts for it,
    SIESTA's own default, non-polarized, where the deck states none: the
    one rule the record and a transport run's index read it by."""
    return (SPIN_POLARIZED if params.spin_treatment == "unrestricted"
            else SPIN_NON_POLARIZED)


def frames_average(weights: Sequence[float], values: Sequence[float]
                   ) -> float:
    """THE FRAMES' AVERAGED CONDUCTANCE, one value a frame each at the
    frames' stated weights, ``sum_f W_f G_f`` (`engines/transport.md`
    § 2a.12) -- the one rule the record's averaged conductance
    (`frame_average`) and a round's report line (`transport.run_words`,
    beside the job) take."""
    return math.fsum(float(w) * float(x) for w, x in zip(weights, values))


__all__ = ["at_fermi", "frames_average", "per_spin_channel", "spin_of"]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.runfiles",)
