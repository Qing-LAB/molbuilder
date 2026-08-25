"""The browser side of the two-clock rule (docs/model/parse.md § 2a).

``core.js`` receives two per-frame time series and they are not
interchangeable: ``wall_clock_s`` is an absolute epoch and may be shown
as a date; ``elapsed_s`` counts from the run's start and may be shown as
a duration.  Feeding one to the other's formatter is the defect that
made a six-minute SIESTA run display "last result Dec 31, 5:06 PM".

Two guards here:
  * ``cumulativeElapsed`` — a pure helper, run under node and checked
    directly.
  * the run-state badge — checked at source level, because its clock
    selection sits inside a large DOM render function that cannot be
    called without a full browser.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT   = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/trajectory/core.js"


def _run_cycle_clock(expr: str):
    """Extract ``cumulativeElapsed`` from the module and evaluate ``expr``."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = MODULE.read_text()
    ix  = src.index("function cumulativeElapsed")
    end = src.index("function fmtElapsed", ix)
    fn  = src[ix:end].rstrip()
    full = fn + "\nconsole.log(JSON.stringify(" + expr + "));"
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestCumulativeElapsed:
    """The per-iteration ladder divides this value by `cumulative_calls`.

    That division is arithmetic on a DURATION, so the accessor must yield
    only a cumulative elapsed -- never an absolute epoch.  A first version
    took "whichever clock the cycle carries", which made the ladder fire on
    molwatch cycles (they have no `cumulative_calls`, so the raw epoch fell
    through to the display) and a PySCF run read
    "~489276.7h/iter (from SIESTA iter-1 timer)".
    """

    def test_a_siesta_cycle_reports_its_cumulative_timer(self):
        assert _run_cycle_clock(
            "cumulativeElapsed({cycle: 2, elapsed_s: 75.4})") == 75.4

    def test_an_epoch_is_NOT_a_duration_and_must_not_match(self):
        """The regression, stated as the rule it broke: dividing
        1761396030 by a call count is arithmetic on a date."""
        assert _run_cycle_clock(
            "cumulativeElapsed({cycle: 2, wall_clock_s: 1761396030.0})"
            " === null") is True

    def test_a_cycle_with_no_timing_is_null_not_zero(self):
        """A missing measurement must not read as 'at t=0' -- that is how
        it becomes a plotted point at the origin."""
        assert _run_cycle_clock("cumulativeElapsed({cycle: 1}) === null") is True
        assert _run_cycle_clock("cumulativeElapsed(null) === null") is True

    def test_non_finite_is_rejected(self):
        """SIESTA's Fortran column overflow yields NaN on the Time field;
        NaN must not pass as a measurement.

        Compared with ``=== null`` rather than for a null RESULT:
        JSON.stringify turns NaN into the token `null`, so a test that
        round-trips the value through JSON cannot tell the two apart and
        passes even when NaN leaks through.
        """
        assert _run_cycle_clock(
            "cumulativeElapsed({elapsed_s: NaN}) === null") is True
        assert _run_cycle_clock(
            "cumulativeElapsed({elapsed_s: Infinity}) === null") is True


class TestBadgeReadsTheRightClock:
    """Source-level guard on the run-state badge's clock selection."""

    def test_timestamp_comes_from_the_epoch_series(self):
        src = MODULE.read_text()
        assert "const clockSeries   = state.data.wall_clock_s || [];" in src, (
            "the badge's 'last result at' must read the EPOCH series; if "
            "this moved, check it did not move onto elapsed_s")
        assert "const lastWall = lastFinite(clockSeries);" in src

    def test_duration_comes_from_the_elapsed_series(self):
        src = MODULE.read_text()
        assert "const elapsedSeries = state.data.elapsed_s || [];" in src
        assert "const elapsed  = lastFinite(elapsedSeries);" in src

    def test_the_old_ambiguous_field_is_gone(self):
        """`wall_times` carried an epoch from one engine and elapsed
        seconds from another.  It must not come back."""
        src = MODULE.read_text()
        assert "wall_times" not in src, (
            "wall_times is the ambiguous field the two-clock rule "
            "replaced (docs/model/parse.md § 2a)")
