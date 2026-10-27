"""A TRANSMISSION POINT'S ENERGIES (`engines/transport.md` § 2a.16, *A
transmission point's energies*): its grid (:func:`grid`); the contours a
point computes when it shares energies with a finished point, and their
record in its deck (:class:`Contours`); and **the one door that reads a
point's energies and every value on them** -- from its own run and its
``shared/``, recursively, in the grid's order (:func:`joined`) -- asked by
the record, the data file, a selection's PDOS and the sharing test alike.

Two energies are one when they lie within :data:`SAME_ENERGY_EV`: the
energies of two runs agree to rounding (10⁻¹⁵ eV), and an energy a few
10⁻⁶ eV away is another (the transmission's narrowest features are as wide
as its broadening, 10⁻⁴ eV at the catalogue's defaults).
"""
from __future__ import annotations

import bisect
import re
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
from ..deck_record import BLOCK_SHARED  # noqa: E402


def spacing(window: Tuple[float, float, int]) -> float:
    """A window's spacing, ``(to − from) / points`` -- the distance between
    its mid-rule energies, and each energy's weight in TBtrans's integral of
    the current (`m_integrate.f90`, `m_tbt_save.F90`)."""
    lo, hi, n = window
    return (float(hi) - float(lo)) / int(n)


def grid(emin: float, emax: float, points: int) -> List[float]:
    """The energies a window computes: ``points`` mid-rule energies, half a
    spacing inside each end (`m_integrate.f90`; the deck's ``method
    mid-rule``)."""
    h = spacing((emin, emax, points))
    return [float(emin) + (i + 0.5) * h for i in range(int(points))]


#: A transmission deck's own window (`transport.deck._emit_tbt_window`).
_WINDOW_LINE = re.compile(
    r"%block[ \t]+TBT\.Contour\.window.*?from[ \t]+(\S+)[ \t]+eV[ \t]+to"
    r"[ \t]+(\S+)[ \t]+eV.*?points[ \t]+(\d+)", re.S | re.I)


def deck_window(text: str) -> Optional[Tuple[float, float, int]]:
    """The grid a transmission deck describes -- ``(from, to, points)``, the
    numbers tbtrans computed its energies from: its sharing record's
    (:class:`Contours`), or its window's own block; ``None`` for a deck
    stating neither."""
    c = Contours.of_deck(text)
    if c is not None:
        return c.window
    m = _WINDOW_LINE.search(text)
    return ((float(m.group(1)), float(m.group(2)), int(m.group(3)))
            if m else None)


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
        from ..deck_record import read_json_block
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
    h = (window[1] - window[0]) / window[2]
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


def joined(folder: Path, read: Callable[[Path], Dict]) -> Dict:
    """THE ONE JOIN: what ``read`` answers of the point in ``folder`` --
    ``energy_ev`` and every value along it -- joined with what it answers of
    the point's ``shared/``, recursively, an energy the point computed
    itself taken from its own run, the whole in the grid's order; and, for
    a point whose deck records its sharing, only the energies of the grid
    it describes (`engines/transport.md` § 2a.16).  A quantity one side
    holds and the other does not is dropped: no point holds a value on part
    of its grid."""
    own = read(folder)
    shared = Path(folder) / SHARED_DIR
    if not shared.is_dir():
        return own
    other = joined(shared, read)
    out = _merge(own, other)
    contours = sharing_of(folder)
    if contours is not None:
        out = _on(out, grid(*contours.window))
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


__all__ = ["SAME_ENERGY_EV", "SHARED_DIR", "BLOCK_SHARED", "spacing",
           "grid", "deck_window", "same_grid", "in_common",
           "Contours", "shared_contours", "joined", "sharing_of",
           "window_of", "held"]
