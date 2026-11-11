"""Seconds per SCF iteration, by phase -- ONE rule, and each engine's rows
from the file that stamps them (`model/parse.md` § 5c).

A run's SCF rows are stamped as they are written: the SIESTA family's by the
wrapper's tee, into ``<base>.scf-timing.log``; PySCF's by its deck, as
the last column of each ``scf_history`` row of its progress log,
``<label>_<NN>_<stage>.molwatch.log``.  Each file has its row reader here,
and :func:`timing_figures` times either the same way; :func:`timing_of` reads
a file by what it IS.  The run record, the trajectory viewer, the benchmark
and the monitor all ask :func:`timing_of`, so every surface states one number
for one run.

**Stdlib only, and it travels beside every job**
(`runwrap.monitor_companions`, `execution/run-reports.md` § 2.3).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:                                        # inside molbuilder
    from ... import runfiles as _rf
    from ..engines import molwatch_grammar as _MG
    from ..engines import siesta_grammar as _G
except ImportError:                         # beside a job, as the monitor's
    import runfiles as _rf
    import molwatch_grammar as _MG
    import siesta_grammar as _G

#: One stamped SCF row: ``(epoch, phase, begins an SCF)``.
Row = Tuple[float, str, bool]

#: EVERY PHASE A RUN'S STAMPED ROWS CAN CARRY, in each row reader's family's
#: words -- a SIESTA-family device's periodic initialization and its NEGF
#: loop, a PySCF run's one SCF -- under which a run with several states each
#: phase's own figures (:func:`timing_figures`).
PHASES: Tuple[str, ...] = (_G.PHASE_PERIODIC, _G.PHASE_NEGF, _MG.SCF_PHASE)

def scf_row_patterns(role: str) -> Optional[Tuple[str, str]]:
    """How an output in ``role`` prints an SCF row -- ``(a row's ERE, its
    prefix's ERE)``, what its catalogue row names for ``scf_rows`` (the
    SIESTA family's grammar's: its engine prints no time per row, so the run
    script stamps them as they print) -- or ``None``: it names none.  WHICH
    output the run script stamps is the file catalogue's to say too -- the
    timing file's row names the output it is cut from
    (`runfiles.cut_from_output`; `running-a-job.md` § 4.1)."""
    return _rf.reader(role, "scf_rows")


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
    forces, the move and the new step's setup, and it is dropped.  Measured:
    averaged in, a relaxation read 0.5 s an iteration against 0.3 s within
    its SCFs (149 boundaries of about 1.9 s).

    ``rows`` is the headline phase's SCF rows so far -- the monitor's
    iteration count (`execution/run-reports.md` § 2.3).  A run with both
    phases also states each one's figures -- ``s_per_iter_<phase>``,
    ``iters_measured_<phase>``, ``rows_<phase>`` --
    and its headline is the NEGF loop's: the part a device runs until it
    converges, 27.5 s an iteration beside the initialization's 97 s on
    2026-09-25.  A line with no row text reads as periodic.
    """
    rows: List[Row] = []
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
    return timing_figures(rows)


def progress_log_timing_metrics(text: str) -> Dict[str, Any]:
    """The same figures from a PySCF progress log: each ``scf_history``
    row's stamp -- the deck's epoch when that cycle finished, read by the
    grammar's one row reader (`molwatch_grammar.scf_history_row`) -- one
    ``scf_history`` block per SCF, its first row the one that begins it, in
    the run's one phase (`molwatch_grammar.SCF_PHASE`).  A row with no stamp
    is not timed.

    THE ROWS ALONE, not the log's reading pass: the monitor asks this at
    every wake, and the pass builds every step's coordinates and forces --
    the same figures at three times the cost (38 ms against 116 ms on a
    2.5 MB log of 300 steps)."""
    rows: List[Row] = []
    first = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == _MG.SCF_HISTORY_BEGIN:
            first = True
            continue
        if stripped == _MG.SCF_HISTORY_END:
            first = False
            continue
        row = _MG.scf_history_row(line)
        if row is None:
            continue
        t = row.get("wall_clock_s")
        if t is not None and math.isfinite(t):
            rows.append((t, _MG.SCF_PHASE, first))
        first = False
    return timing_figures(rows)


def timing_roles() -> Tuple[str, ...]:
    """The roles :func:`timing_file` chooses among: the timing files' own
    (the rows naming a ``timing`` reader) and every output the run script
    cuts a tee from (`runfiles.cut_from_output`) -- what a caller holding a
    run hands it."""
    return _rf.read_roles("timing") + tuple(
        r for r in _rf.stdout_roles() if _rf.cut_from_output(r))


def timing_file(files) -> Optional[Path]:
    """THE RUN'S TIMING FILE, out of its files ``{role: path}``
    (:func:`timing_roles` the ones it reads) -- or ``None``.  A run whose
    files hold an output the run script cuts a tee from
    (`runfiles.cut_from_output` -- a SIESTA-family ``.out``) is timed by
    that tee alone, ``None`` before its first row: its progress log is
    prep's seed, which SIESTA never writes.  Any other run is timed by the
    progress log its deck stamps (a PySCF run's) -- a SIESTA run that has
    not written its ``.out`` yet included, whose seed holds no row.  The one choice the run record, the monitor, the Results
    load and the benchmark summary ask (`model/parse.md` § 5c)."""
    for role, path in files.items():
        tee = _rf.cut_from_output(role) if path is not None else None
        if tee is not None:
            return files.get(tee)
    return next((files[r] for r in _rf.read_roles("timing")
                 if files.get(r) is not None
                 and r not in {_rf.cut_from_output(o)
                               for o in _rf.stdout_roles()}), None)


def timing_of(path) -> Dict[str, Any]:
    """The figures from the timing file ``path``, by its role's row reader;
    ``{}`` when it is no timing file or cannot be read."""
    if path is None:
        return {}
    p = Path(path)
    role = _rf.role_of(p.name)
    read = _rf.reader(role, "timing") if role else None
    if read is None:
        return {}
    try:
        return read(p.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}


def timing_figures(rows: List[Row]) -> Dict[str, Any]:
    """The rule, for any engine's rows (`model/parse.md` § 5c): each phase
    timed from its own consecutive rows within one SCF; the headline is the
    last phase's, and with several each states its own.  Phases in the order
    the run entered them -- a device's periodic initialization, then its NEGF
    loop."""
    phases = list(dict.fromkeys(r[1] for r in rows))
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


def _phase_figures(rows: List[Row], phase: str) -> Dict[str, Any]:
    """One phase's seconds per iteration, from its own consecutive rows
    within one SCF; only forward (positive) deltas are real iteration
    durations."""
    deltas = [b[0] - a[0] for a, b in zip(rows, rows[1:])
              if a[1] == b[1] == phase and not b[2] and b[0] - a[0] > 0]
    measured = deltas[1:] if len(deltas) >= 2 else deltas
    # KEPT TO 0.1 ms, rounded only where it is shown: a fast run's rate is
    # under a tenth of a second, which one decimal would print as 0.0.
    return {"s_per_iter": (round(sum(measured) / len(measured), 4)
                           if measured else None),
            "iters_measured": len(measured),
            "rows": sum(1 for r in rows if r[1] == phase)}


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.monitor_companions`).
TRAVELS_WITH = ("molbuilder.parse.engines.molwatch_grammar",
                "molbuilder.runfiles",
                "molbuilder.parse.engines.siesta_grammar")
