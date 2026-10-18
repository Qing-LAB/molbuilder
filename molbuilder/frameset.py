"""A MULTI-FRAME STRUCTURE'S ROWS, AS WRITTEN -- what it states of itself and
of each frame, the vocabulary a mode's frame set is written in, and the one
row a calculation over the frames reads (`model/structure.md` § 2.2f).

MODULE  frameset (L1; the Structure and the standard library)
ROLE    the vocabulary of a mode's frame set as data -- every row such a set
        states, where it lives, its kind and unit and how a page says it
        (:data:`ROWS`), the per-atom masses channel (:data:`MASS_CHANNEL`)
        and the ``info`` cluster (:data:`INFO_KEY`) -- and its one reader,
        :func:`read`: the structure's rows and each frame's as written, and
        each frame's ``weight``
USED-BY the transport citation door (`transport.compose`, whose
        `frame_set_answer` -- the describe door's and Task setup's -- is
        `FrameRows.to_wire`), the transport record (`transport.record`,
        `transport.average`), the data file (`transport.datafile`)

**A calculation over the frames needs the frames and nothing else** (user,
2026-10-18: "the frame can of course carry meta data, and why you treat it as
transport's job to determine what should or should not be there?").  Every
row the structure and its frames carry is its maker's: carried and shown as
written, never checked, never computed from -- the vocabulary below only
names how a page says the rows it knows.  But one: each frame's ``weight``,
its share of the combined result, read where every frame states one as a
number.  A weight missing from some frames, or weights that do not sum to 1,
are said in :attr:`FrameRows.warnings` -- the average is then not taken, or
taken at the weights as written -- never refused.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .structure import frame_words

#: Each atom's mass, the value channel a mode's coordinate is weighted by --
#: a mode's maker's (`model/structure-annotations.md`).
MASS_CHANNEL = "mass_amu"

#: Where the mode came from: ``info.vibration = {run, result_sha256}``.
INFO_KEY = "vibration"
INFO_FIELDS = ("run", "result_sha256")

#: The one row a calculation over the frames reads: each frame's share of
#: the combined result.
WEIGHT_ROW = "weight"


@dataclass(frozen=True)
class Row:
    """One row of a mode's frame set's vocabulary: its ``name`` (its unit in
    it), ``where`` it lives (``structure`` -- the structure's rows -- or
    ``frame``, each frame's), its ``kind`` (``given``, ``measured`` from the
    coordinates, or ``derived`` from the others), the ASCII spelling of its
    ``unit`` (``None``: a number without one) -- the file's -- and how a
    page says it: its ``words`` and its ``page_unit`` (the name stays one
    hover away, `engines/transport.md` § 3.8.1)."""
    name: str
    where: str
    kind: str
    unit: Optional[str]
    words: str
    page_unit: Optional[str]


#: THE VOCABULARY, as data -- `model/structure.md` § 2.2f's table: the rows a
#: mode's frame set is written in, and how a page says each.
ROWS: Tuple[Row, ...] = (
    Row("mode_index_1based", "structure", "given", None,
        "Mode", None),
    Row("frequency_cm1", "structure", "given", "cm-1",
        "Frequency", "cm⁻¹"),
    Row("temperature_k", "structure", "given", "K",
        "Temperature", "K"),
    Row("zero_point_amplitude_amu12_ang", "structure", "given",
        "amu^1/2 angstrom", "Zero-point amplitude", "amu½·Å"),
    Row("sigma_amu12_ang", "structure", "derived", "amu^1/2 angstrom",
        "Thermal spread σ", "amu½·Å"),
    Row("displacement_ang", "frame", "measured", "angstrom",
        "Displacement", "Å"),
    Row("max_atom_displacement_ang", "frame", "measured", "angstrom",
        "Largest atom move", "Å"),
    Row("q_amu12_ang", "frame", "derived", "amu^1/2 angstrom",
        "Position along the mode, q", "amu½·Å"),
    Row("node_sigma", "frame", "derived", None,
        "q in units of σ", None),
    Row(WEIGHT_ROW, "frame", "given", None,
        "Weight", None),
)
STRUCTURE_ROWS = tuple(r for r in ROWS if r.where == "structure")
FRAME_ROWS = tuple(r for r in ROWS if r.where == "frame")
_KNOWN = {r.name: r for r in ROWS}

#: How far the weights' sum may be from 1 before it is said -- the record
#: states it beside the average.
WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FrameRows:
    """A multi-frame structure's rows, as written: the structure's
    (``structure``, ``[{name, value, unit}]``), each frame's (``frames``,
    one such list per frame), the frames' weights where every frame states
    one as a number (``weights``, else ``None``), what is said of them
    (``warnings``), and -- where the structure carries them -- each atom's
    mass on :data:`MASS_CHANNEL` and ``info.vibration``."""
    structure: Tuple[Dict[str, Any], ...]
    frames: Tuple[Tuple[Dict[str, Any], ...], ...]
    weights: Optional[Tuple[float, ...]]
    warnings: Tuple[str, ...]
    masses_amu: Optional[Tuple[Any, ...]] = None
    vibration: Optional[Mapping[str, Any]] = None

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

    def _columns(self) -> List[str]:
        """Every row name the frames state: the vocabulary's first, in its
        order, then the others as they are first written."""
        stated: List[str] = []
        for rows in self.frames:
            for r in rows:
                if r["name"] not in stated:
                    stated.append(r["name"])
        known = [r.name for r in FRAME_ROWS if r.name in stated]
        return known + [n for n in stated if n not in known]

    def to_wire(self) -> Dict[str, Any]:
        """The rows as a page shows them without knowing their names:
        ``{structure: [{name, value, unit, kind, words, page_unit}], columns:
        [{name, unit, kind, words, page_unit}], frames: [[a value per column,
        None where a frame states none] per frame], weights, weight_sum,
        warnings}``, and ``masses_amu`` and ``vibration`` where the structure
        carries them -- a row the vocabulary knows said in its words, any
        other by its name and unit as written (`engines/transport.md`
        § 3.8.1)."""
        srows = sorted(self.structure, key=lambda r: (
            [s.name for s in STRUCTURE_ROWS].index(r["name"])
            if r["name"] in _KNOWN else len(STRUCTURE_ROWS)))
        units = {r["name"]: r.get("unit") for rows in self.frames
                 for r in rows}
        names = self._columns()
        out: Dict[str, Any] = {
            "structure": [_said(r["name"], r.get("unit"), value=r["value"])
                          for r in srows],
            "columns": [_said(n, units.get(n)) for n in names],
            "frames": [[self.value(n, f) for n in names]
                       for f in range(len(self.frames))],
            "weights": list(self.weights) if self.weights is not None
            else None,
            "weight_sum": self.weight_sum,
            "warnings": list(self.warnings),
        }
        if self.masses_amu is not None:
            out["masses_amu"] = list(self.masses_amu)
        if self.vibration is not None:
            out["vibration"] = dict(self.vibration)
        return out


def _said(name: str, unit: Any, **value) -> Dict[str, Any]:
    """A row as a page says it: the vocabulary's words, kind and unit for a
    row it knows, else its name and its unit as written."""
    known = _KNOWN.get(name)
    if known is not None:
        return {"name": name, **value, "unit": known.unit,
                "kind": known.kind, "words": known.words,
                "page_unit": known.page_unit}
    return {"name": name, **value, "unit": unit, "kind": "as written",
            "words": name, "page_unit": unit}


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
    channel = (structure.annotations or {}).get(MASS_CHANNEL)
    data = getattr(channel, "data", None) or {}
    masses = (tuple(data.get(i) for i in range(structure.n_atoms))
              if channel is not None else None)
    info = (structure.info or {}).get(INFO_KEY)
    return FrameRows(structure=srows, frames=frows, weights=weights,
                     warnings=warnings, masses_amu=masses,
                     vibration=dict(info) if isinstance(info, Mapping)
                     else None)


def _weights(frows, n: int) -> Tuple[Optional[Tuple[float, ...]],
                                     Tuple[str, ...]]:
    """Each frame's :data:`WEIGHT_ROW` as a number, where every frame states
    one -- else ``None`` -- and what is said of them."""
    if n < 2:
        return None, ()
    stated = [next((r["value"] for r in rows if r["name"] == WEIGHT_ROW),
                   None) for rows in frows]
    if all(v is None for v in stated):
        return None, ()
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


__all__ = ["FRAME_ROWS", "FrameRows", "INFO_FIELDS", "INFO_KEY",
           "MASS_CHANNEL", "ROWS", "Row", "STRUCTURE_ROWS", "WEIGHT_ROW",
           "WEIGHT_SUM_TOLERANCE", "read"]
