"""``<base>-runN.scf-timing.log``'s rows -- seconds per SCF iteration, by phase.

The reading half of the SCF-timing instrument (`scf_timing.py` registers the
parser that wraps it).  **Stdlib only, and it travels beside every job**
(`runwrap.MONITOR_COMPANIONS`, `execution/run-reports.md` § 2.3): the monitor
reports a run's iterations and rate with this, the same function the Results
tab's record and the benchmark read the file with.  It stood inside the
registered parser's module until 2026-09-26, where the parse types it needs
would have come with it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

try:                                        # inside molbuilder
    from ..engines import siesta_grammar as _G
except ImportError:                         # beside a job, as the monitor's
    import siesta_grammar as _G


def scf_timing_metrics(text: str) -> Dict[str, Any]:
    """``{"s_per_iter": float|None, "iters_measured": int, "rows": int}`` --
    one level,
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

    **A step boundary is not an iteration either**, for the same reason: the
    delta INTO a row whose iteration is 1 spans the previous SCF's end, the
    forces, the move and the new step's setup, and it is dropped.  It was
    averaged in until 2026-09-26, and a relaxation read 0.5 s an iteration
    against 0.3 s within its SCFs (149 boundaries of about 1.9 s).

    ``rows`` is the headline phase's SCF rows so far -- the monitor's
    iteration count (`execution/run-reports.md` § 2.3).  A run with both
    phases also states each one's figures -- ``s_per_iter_<phase>``,
    ``iters_measured_<phase>``, ``rows_<phase>`` --
    and its headline is the NEGF loop's: the part a device runs until it
    converges, 27.5 s an iteration beside the initialization's 97 s on
    2026-09-25.  A line with no row text reads as periodic, which is what
    every tee before the NEGF phase was recorded wrote.
    """
    rows: List[Tuple[float, str, bool]] = []
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
        # Iteration 1 begins an SCF: the tee's own iteration column, else
        # the row's.  An unreadable one begins nothing.
        try:
            first = int(parts[1]) == 1 if len(parts) > 1 else False
        except ValueError:
            first = bool(row and row.iscf == 1)
        rows.append((t, row.phase if row else _G.PHASE_PERIODIC, first))
    phases = [ph for ph in (_G.PHASE_PERIODIC, _G.PHASE_NEGF)
              if any(r[1] == ph for r in rows)]
    if not phases:
        return {"s_per_iter": None, "iters_measured": 0, "rows": 0}
    figures = {ph: _phase_figures(rows, ph) for ph in phases}
    out: Dict[str, Any] = dict(figures[phases[-1]])
    if len(phases) > 1:
        for ph in phases:
            out[f"s_per_iter_{ph}"] = figures[ph]["s_per_iter"]
            out[f"iters_measured_{ph}"] = figures[ph]["iters_measured"]
            out[f"rows_{ph}"] = sum(1 for r in rows if r[1] == ph)
    return out


def _phase_figures(rows: List[Tuple[float, str, bool]],
                   phase: str) -> Dict[str, Any]:
    """One phase's seconds per iteration, from its own consecutive rows
    within one SCF; only forward (positive) deltas are real iteration
    durations."""
    deltas = [b[0] - a[0] for a, b in zip(rows, rows[1:])
              if a[1] == b[1] == phase and not b[2] and b[0] - a[0] > 0]
    measured = deltas[1:] if len(deltas) >= 2 else deltas
    return {"s_per_iter": (round(sum(measured) / len(measured), 1)
                           if measured else None),
            "iters_measured": len(measured),
            "rows": sum(1 for r in rows if r[1] == phase)}
