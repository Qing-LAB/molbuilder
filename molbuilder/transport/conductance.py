"""THE CONDUCTANCE AT THE FERMI LEVEL -- G/G0 = T(E_F), read off a
transmission on energies measured from E_F (`engines/transport.md` § 2a.4).

MODULE  transport.conductance (stdlib only; travels in ``mb_monitor.pyz``,
        `runwrap.MONITOR_COMPANIONS`)
ROLE    the one rule, and the one rule of the transmission it is read on --
        per spin channel (:func:`per_spin_channel`) -- asked by the record
        for every point (`transport.record`) and by a transmission run's
        report step for its round's milestone (`execution/run-reports.md`
        § 2.9): so the number a message says is the record's rule, on the
        values it is given.
USED-BY `transport.record`, `transport.run_words`
"""
from __future__ import annotations

import math
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence

#: THE CHANNELS TBtrans WRITES for a deck's spin -- `siesta_grammar.
#: TBT_CHANNELS`' names: both of a polarized device (`TBT.Spin` fixed at 0,
#: `engines/transport.md` § 6.1), the one of any other.
SPIN_CHANNELS = {"polarized": ("up", "down"),
                 "non-polarized": ("unpolarized",)}


def per_spin_channel(curves: Dict[str, Sequence[float]], spin: str
                     ) -> List[float]:
    """T(E) PER SPIN CHANNEL, the transmission G = G0 · T(E_F) is read on
    (`engines/transport.md` § 2a.12): the unpolarized channel's as it is; a
    polarized point's two averaged, (T↑ + T↓)/2.  ``curves`` holds each
    channel the run wrote, its T on the one grid; a channel the deck's
    ``spin`` has TBtrans write and ``curves`` lacks is :class:`ValueError`,
    in the words that say the point not done."""
    want = SPIN_CHANNELS[spin]
    missing = [c for c in want if c not in curves]
    if missing:
        raise ValueError(
            f"the deck is spin-polarized and the run wrote no "
            f"{' or '.join(missing)} channel transmission"
            if spin == "polarized" else "the run wrote no transmission file")
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


__all__ = ["SPIN_CHANNELS", "at_fermi", "per_spin_channel"]
