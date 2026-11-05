"""A TRANSMISSION POINT'S ENERGIES (`engines/transport.md` § 2a.16, *A
transmission point's energies*): its grid (:func:`grid`); the rule of whether
a window reaches a bias (:func:`window_short_of`); the contours a
point computes when it shares energies with a finished point, and their
record in its deck (:class:`Contours`); and **the one door that reads a
point's energies and every value on them** -- from its own run and its
``shared/``, recursively, in the grid's order (:func:`joined`) -- asked by
the record, the data file, a selection's PDOS and the sharing test alike.

Two energies are one when they lie within :data:`SAME_ENERGY_EV`: the
energies of two runs agree to rounding (10⁻¹⁵ eV), and an energy a few
10⁻⁶ eV away is another (the transmission's narrowest features are as wide
as its broadening, 10⁻⁴ eV at the catalogue's defaults).

**It travels beside a job** in the monitor's bundle
(`runwrap.monitor_companions`), so a transmission run's report step joins a
point's energies by this same rule (`transport.run_words`): imported from
the package here, from its copy there; reading a deck stays the package's.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Two energies are one energy within this, in eV (§ 2a.16).
SAME_ENERGY_EV = 1e-9

#: The folder a point's shared energies lie in, beside its own files
#: (`project-layout.md` § 1.4a, its role `calcdirs.SHARED`).
SHARED_DIR = "shared"

#: The molbuilder block of a transmission deck that records its sharing:
#: which finished point, the grid, and the contours -- its name the deck
#: record's (`deck_record`).
try:                                        # inside molbuilder
    from ..deck_record import BLOCK_SHARED, read_json_block  # noqa: E402
except ImportError:                         # beside a job
    from deck_record import BLOCK_SHARED, read_json_block  # type: ignore[no-redef]  # noqa: E402


def spacing(window: Tuple[float, float, int]) -> float:
    """A window's spacing, ``(to − from) / points`` -- the distance between
    its mid-rule energies, and each energy's weight in TBtrans's integral of
    the current (`m_integrate.f90`, `m_tbt_save.F90`)."""
    lo, hi, n = window
    return (float(hi) - float(lo)) / int(n)


#: HOW FAR PAST THE BIAS WINDOW the transmission must reach for a current
#: computed from T(E, 0) to be whole: the Fermi functions' tails, in kT
#: (`engines/transport.md` § 2a.10; the window gate, plan § 5x P2).
WINDOW_TAILS_KT = 5.0


def window_short_of(emin_ev: float, emax_ev: float, volts: float,
                    kt_ev: float) -> Optional[str]:
    """ONE RULE for the transmission window against a bias
    (`engines/transport.md` § 2a.10, P2): the window must reach
    ±(|V|/2 + :data:`WINDOW_TAILS_KT`·kT) -- both leads' Fermi tails at
    μ = ±V/2 -- or the current integrated over it is cut short without a
    word from TBtrans.  ``None`` when it reaches; else the sentence naming
    the reach and the fix, said alike by the record (a voltage it does not
    integrate), the transmission deck's gate and the description's
    preflight."""
    half = abs(float(volts)) / 2.0
    reach = half + WINDOW_TAILS_KT * float(kt_ev)
    lo, hi = float(emin_ev), float(emax_ev)
    if reach > hi or -reach < lo:
        return (f"the transmission's sampled window [{lo:g}, {hi:g}] eV "
                f"does not reach "
                f"±{reach:.3f} eV for {float(volts):g} V (V/2 + "
                f"{WINDOW_TAILS_KT:g} kT at kT = {float(kt_ev):.4f} eV); "
                f"widen it -- transmission_emin_ev / transmission_emax_ev")
    return None


def sampled_window(emin_ev: float, emax_ev: float,
                   n_points: int) -> Tuple[float, float]:
    """The first and last energy TBtrans samples on a window of ``n_points``
    -- the mid-rule's points, half a spacing inside each edge
    (`m_integrate.f90`; the deck's ``method mid-rule``): what
    :func:`window_short_of` is asked with by the transmission deck's gate
    and the description's preflight, as the record asks it with the energies
    the run wrote -- one rule, on the energies the current is integrated
    over."""
    lo, hi = float(emin_ev), float(emax_ev)
    h = spacing((lo, hi, n_points))
    return lo + h / 2.0, hi - h / 2.0


def grid(emin: float, emax: float, points: int) -> List[float]:
    """The energies a window computes: ``points`` mid-rule energies, half a
    spacing inside each end (`m_integrate.f90`; the deck's ``method
    mid-rule``)."""
    h = spacing((emin, emax, points))
    return [float(emin) + (i + 0.5) * h for i in range(int(points))]


def deck_window(text: str) -> Optional[Tuple[float, float, int]]:
    """The grid a transmission deck describes -- ``(from, to, points)``, the
    numbers tbtrans computed its energies from: its sharing record's
    (:class:`Contours`), or its window's own block; ``None`` for a deck
    stating neither."""
    c = Contours.of_deck(text)
    if c is not None:
        return c.window
    from ..parse.fdf import fdf_key, read_fdf
    rows = {r[0].lower(): r
            for r in read_fdf(text)[1].get(fdf_key("TBT.Contour.window")) or ()}
    return ((float(rows["from"][1]), float(rows["from"][4]),
             int(rows["points"][1]))
            if "from" in rows and "points" in rows else None)


def _index(sorted_energies: Sequence[float], e: float) -> Optional[int]:
    """Where ``e`` is among ``sorted_energies``, within
    :data:`SAME_ENERGY_EV`, or ``None``."""
    i = bisect.bisect_left(sorted_energies, e - SAME_ENERGY_EV)
    if i < len(sorted_energies) and abs(sorted_energies[i] - e) \
            <= SAME_ENERGY_EV:
        return i
    return None


def same_grid(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two points' energies are one grid: as many, each the same
    energy within :data:`SAME_ENERGY_EV` -- a point that shares energies
    holds the finished point's, which agree with its own run's to rounding,
    never bit for bit (§ 2a.16)."""
    return len(a) == len(b) and all(abs(float(x) - float(y))
                                    <= SAME_ENERGY_EV for x, y in zip(a, b))


def in_common(energies: Sequence[float], held: Sequence[float]) -> int:
    """How many of ``energies`` one of ``held`` is, within
    :data:`SAME_ENERGY_EV`."""
    held_sorted = sorted(held)
    return sum(1 for e in energies if _index(held_sorted, e) is not None)


@dataclass(frozen=True)
class Contours:
    """A point that shares energies with a finished point: the line
    contours of the energies it computes -- ``(from, to, points)`` each, at
    full precision, mid-rule -- the finished point (``source``,
    tree-relative), the grid described (``window``: from, to, points) and
    how many of its energies the finished point holds (``held``).  Its deck
    writes it, and is read back by it (:meth:`of_deck`)."""
    lines: Tuple[Tuple[float, float, int], ...]
    source: str
    window: Tuple[float, float, int]
    held: int

    @property
    def computed(self) -> int:
        return sum(n for _a, _b, n in self.lines)

    def to_record(self) -> dict:
        return {"from": self.source,
                "window": {"from": self.window[0], "to": self.window[1],
                           "points": self.window[2]},
                "held": self.held,
                "lines": [{"from": a, "to": b, "points": n}
                          for a, b, n in self.lines]}

    @classmethod
    def of_record(cls, rec: dict) -> "Contours":
        w = rec["window"]
        return cls(lines=tuple((float(x["from"]), float(x["to"]),
                                int(x["points"])) for x in rec["lines"]),
                   source=str(rec["from"]),
                   window=(float(w["from"]), float(w["to"]),
                           int(w["points"])),
                   held=int(rec["held"]))

    @classmethod
    def of_deck(cls, text: str) -> Optional["Contours"]:
        """The sharing a deck records, or ``None`` for a deck that computes
        its window whole."""
        rec = read_json_block(text, BLOCK_SHARED)
        return cls.of_record(rec) if rec is not None else None


def shared_contours(energies: Sequence[float], held: Sequence[float], *,
                    window: Tuple[float, float, int], source: str
                    ) -> Optional[Contours]:
    """THE ENERGIES A POINT COMPUTES when it shares with a finished point
    holding ``held``: those of its grid (``window``) the finished point
    does not hold, as line contours -- each a run of them at one spacing:
    where the finished grid's spacing is a whole number of the new one's,
    each class of new energies between the finished ones at the finished
    spacing, else each run at the new spacing -- or ``None`` when the two
    grids hold no energy in common, or the new one adds none."""
    held_sorted = sorted(held)
    mine = list(energies)
    new = [i for i, e in enumerate(mine) if _index(held_sorted, e) is None]
    common = len(mine) - len(new)
    if not common or not new:
        return None
    h = spacing(window)
    step = 1
    if len(held_sorted) > 1:
        ratio = (held_sorted[1] - held_sorted[0]) / h
        if abs(ratio - round(ratio)) < 1e-6 and round(ratio) > 1:
            step = int(round(ratio))
    runs: List[List[int]] = []
    left = set(new)
    if step > 1:
        # EACH CLASS of new energies at the finished spacing, a run where
        # its members follow one another.
        for r in range(step):
            cls_ = sorted(i for i in left if i % step == r)
            for run in _consecutive(cls_, step):
                if len(run) > 1:
                    runs.append(run)
                    left -= set(run)
    runs += _consecutive(sorted(left), 1)
    lines = []
    for run in sorted(runs, key=lambda r: mine[r[0]]):
        s = (run[1] - run[0]) if len(run) > 1 else 1
        lines.append((mine[run[0]] - s * h / 2.0,
                      mine[run[-1]] + s * h / 2.0, len(run)))
    return Contours(lines=tuple(lines), source=source, window=window,
                    held=common)


def _consecutive(indices: Sequence[int], step: int) -> List[List[int]]:
    """``indices`` cut into runs whose members are ``step`` apart."""
    out: List[List[int]] = []
    for i in indices:
        if out and i - out[-1][-1] == step:
            out[-1].append(i)
        else:
            out.append([i])
    return out


#: The values of a reader's answer that do not run along the energies --
#: never joined, whatever their length.
_NOT_ALONG_E = frozenset(("atoms", "outside_device"))


def joined(folder: Path, read: Callable[[Path], Dict], *,
           window_of: Optional[Callable[[Path], Optional[
               Tuple[float, float, int]]]] = None) -> Dict:
    """THE ONE JOIN: what ``read`` answers of the point in ``folder`` --
    ``energy_ev`` and every value along it -- joined with what it answers of
    the point's ``shared/``, recursively, an energy the point computed
    itself taken from its own run, the whole in the grid's order; and, for
    a point that shares energies, only the energies of the grid it
    describes (`engines/transport.md` § 2a.16).  A quantity one side
    holds and the other does not is dropped: no point holds a value on part
    of its grid.  ``window_of`` answers that grid for a folder, ``None`` for
    a point that does not share -- by default the sharing its deck records
    (:func:`sharing_of`); beside a job, which reads no deck, the run's
    index's (`transport.run_words`)."""
    own = read(folder)
    shared = Path(folder) / SHARED_DIR
    if not shared.is_dir():
        return own
    other = joined(shared, read, window_of=window_of)
    out = _merge(own, other)
    if window_of is None:
        contours = sharing_of(folder)
        window = contours.window if contours is not None else None
    else:
        window = window_of(Path(folder))
    if window is not None:
        out = _on(out, grid(*window))
    return out


def _deck_text(folder: Path) -> Optional[str]:
    """The deck of the point in ``folder``, read through the run door
    (`runs.run_of`), or ``None``."""
    from ..parse.fdf import read_deck
    from ..runs import run_of
    run = run_of(folder)
    deck = run.deck if run is not None else None
    return read_deck(deck) if deck is not None else None


def sharing_of(folder: Path) -> Optional[Contours]:
    """The sharing the deck of the point in ``folder`` records, or
    ``None``."""
    text = _deck_text(folder)
    return Contours.of_deck(text) if text is not None else None


def window_of(folder: Path) -> Optional[Tuple[float, float, int]]:
    """The grid the deck of the point in ``folder`` describes
    (:func:`deck_window`), or ``None``."""
    text = _deck_text(folder)
    return deck_window(text) if text is not None else None


def _pick(a, b, rows, na: int, nb: int):
    """``a`` (along ``na`` energies) and ``b`` (along ``nb``), one value
    each row of ``rows`` -- ``(side, index)`` -- recursively."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _pick(a[k], b[k], rows, na, nb) for k in a
                if k in b and k not in _NOT_ALONG_E}
    if (isinstance(a, list) and isinstance(b, list)
            and len(a) == na and len(b) == nb
            and all(isinstance(x, (int, float)) for x in a + b)):
        return [a[i] if side == 0 else b[i] for side, i in rows]
    if (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
            and a and all(isinstance(x, list) for x in a + b)):
        return [_pick(x, y, rows, na, nb) for x, y in zip(a, b)]
    return a


def _merge(own: Dict, other: Dict) -> Dict:
    ea, eb = list(own["energy_ev"]), list(other["energy_ev"])
    mine = sorted(ea)
    rows = sorted([(e, 0, i) for i, e in enumerate(ea)]
                  + [(e, 1, j) for j, e in enumerate(eb)
                     if _index(mine, e) is None])
    out = _pick(own, other, [(s, i) for _e, s, i in rows], len(ea), len(eb))
    out["energy_ev"] = [e for e, _s, _i in rows]
    for key in _NOT_ALONG_E:
        if key in own:
            out[key] = own[key]
    return out


def _on(answer: Dict, energies: Sequence[float]) -> Dict:
    """``answer`` restricted to ``energies``."""
    want = sorted(energies)
    e = answer["energy_ev"]
    keep = [i for i, x in enumerate(e) if _index(want, x) is not None]
    rows = [(0, i) for i in keep]
    out = _pick(answer, answer, rows, len(e), len(e))
    out["energy_ev"] = [e[i] for i in keep]
    for key in _NOT_ALONG_E:
        if key in answer:
            out[key] = answer[key]
    return out


def held(folder: Path, label: str) -> List[float]:
    """Every energy the point in ``folder`` holds -- its own and those it
    shared, on its grid (:func:`joined`) -- read at full precision from its
    data files, through the parser (`tbtnc.energies_of`), never the five
    decimals its text files print."""
    from .tbtnc import energies_of
    return sorted(joined(folder, lambda f: {"energy_ev":
                                            energies_of(f, label)})
                  ["energy_ev"])



#: WHAT TRAVELS WITH THESE ENERGIES beside the job
#: (`runwrap.monitor_companions`): the sharing record's block, named by the
#: deck record.
TRAVELS_WITH = ("molbuilder.deck_record",)

__all__ = ["SAME_ENERGY_EV", "SHARED_DIR", "WINDOW_TAILS_KT", "spacing",
           "window_short_of", "sampled_window", "grid", "deck_window",
           "same_grid", "in_common",
           "Contours", "shared_contours", "joined", "sharing_of",
           "window_of", "held"]
