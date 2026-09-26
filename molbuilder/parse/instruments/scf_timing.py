"""``<base>-runN.scf-timing.log`` — seconds per SCF iteration.

The wrapper tees every SCF row of either phase -- SIESTA's ``scf:`` and
TranSIESTA's ``ts-scf:`` -- into this file with an epoch stamp in front
(`running-a-job.md` § 4.1), so consecutive deltas of one phase ARE its
per-iteration durations.  Nothing else in the run states that number.

*(This logic lived in `bench/result.py::parse_scf_timing` until
2026-09-04, where it opened the file and read bytes directly.  It moves
here for `parse.md` § 5c's reason: being the wrapper's output rather
than the engine's is not a reason to read it a different way.)*
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from molbuilder.parse.base import FileParser
from molbuilder.parse.engines import siesta_grammar as _G
from molbuilder.parse.types import InstrumentResult

from ._helpers import build_instrument_result


def scf_timing_metrics(text: str) -> Dict[str, Any]:
    """``{"s_per_iter": float|None, "iters_measured": int}`` -- one level,
    as `model/parse.md` § 5c has every instrument's metrics.

    Each line is ``<epoch> <iscf> <the row as SIESTA printed it>``, and the
    row says its own phase (`siesta_grammar.scf_row`).  A phase is timed only
    from consecutive rows OF THAT PHASE: a device's step from its last
    periodic row to its first NEGF row holds TranSIESTA's switch and the whole
    first NEGF iteration -- a row prints at the end of its iteration -- and
    is neither phase's rate.  The FIRST delta of a phase is dropped when two
    or more are present: iteration 1->2 can still carry warm-up.  Under the
    capped 3-iteration trial (2026-08-21) that leaves ONE clean sample --
    iteration 3 -- which the user's measured experience says reads the
    scaling story as well as the older 5-iteration mean.

    A run with both phases also states each one's figures --
    ``s_per_iter_<phase>``, ``iters_measured_<phase>``, ``rows_<phase>`` --
    and its headline is the NEGF loop's: the part a device runs until it
    converges, 27.5 s an iteration beside the initialization's 97 s on
    2026-09-25.  A line with no row text reads as periodic, which is what
    every tee before the NEGF phase was recorded wrote.
    """
    rows: List[Tuple[float, str]] = []
    for line in text.splitlines():
        parts = line.split(None, 2)
        if not parts:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        if not math.isfinite(t):                 # reject nan/inf tokens
            continue
        row = _G.scf_row(parts[2]) if len(parts) == 3 else None
        rows.append((t, row.phase if row else _G.PHASE_PERIODIC))
    phases = [ph for ph in (_G.PHASE_PERIODIC, _G.PHASE_NEGF)
              if any(p == ph for _, p in rows)]
    if not phases:
        return {"s_per_iter": None, "iters_measured": 0}
    figures = {ph: _phase_figures(rows, ph) for ph in phases}
    out: Dict[str, Any] = dict(figures[phases[-1]])
    if len(phases) > 1:
        for ph in phases:
            out[f"s_per_iter_{ph}"] = figures[ph]["s_per_iter"]
            out[f"iters_measured_{ph}"] = figures[ph]["iters_measured"]
            out[f"rows_{ph}"] = sum(1 for _, p in rows if p == ph)
    return out


def _phase_figures(rows: List[Tuple[float, str]],
                   phase: str) -> Dict[str, Any]:
    """One phase's seconds per iteration, from its own consecutive rows;
    only forward (positive) deltas are real iteration durations."""
    deltas = [b[0] - a[0] for a, b in zip(rows, rows[1:])
              if a[1] == b[1] == phase and b[0] - a[0] > 0]
    measured = deltas[1:] if len(deltas) >= 2 else deltas
    return {"s_per_iter": (round(sum(measured) / len(measured), 1)
                           if measured else None),
            "iters_measured": len(measured)}


class ScfTimingFileParser(FileParser):
    """The wrapper's SCF-timing tee."""

    name   = "scf-timing"
    label  = "wrapper SCF-timing log (.scf-timing.log)"
    hint   = "files ending in .scf-timing.log"
    output = InstrumentResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".scf-timing.log") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> InstrumentResult:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return build_instrument_result(metrics=scf_timing_metrics(text),
                                       parser_name=cls.name, source=path)
