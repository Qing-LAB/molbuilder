"""The browser side of the two-clock rule (docs/model/parse.md § 2a).

``core.js`` receives two per-frame time series and they are not
interchangeable: ``wall_clock_s`` is an absolute epoch and may be shown
as a date; ``elapsed_s`` counts from the run's start and may be shown as
a duration.  Feeding one to the other's formatter is the defect that
made a six-minute SIESTA run display "last result Dec 31, 5:06 PM".

Two guards here:
  * ``cycleClock`` — a pure helper, run under node and checked directly.
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
    """Extract ``cycleClock`` from the module and evaluate ``expr``."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = MODULE.read_text()
    ix  = src.index("function cycleClock")
    end = src.index("function fmtElapsed", ix)
    fn  = src[ix:end].rstrip()
    full = fn + "\nconsole.log(JSON.stringify(" + expr + "));"
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestCycleClock:
    """Per-cycle timing reads whichever clock the engine reported.

    A DIFFERENCE between two cycles of one run is identical either way
    (the origin cancels), which is what makes one accessor legitimate
    here where it would not be for an absolute display.
    """

    def test_siesta_cycle_reports_elapsed(self):
        assert _run_cycle_clock(
            "cycleClock({cycle: 2, elapsed_s: 75.4})") == 75.4

    def test_pyscf_cycle_reports_epoch(self):
        assert _run_cycle_clock(
            "cycleClock({cycle: 2, wall_clock_s: 1761396030.0})"
        ) == 1761396030.0

    def test_elapsed_wins_when_somehow_both_present(self):
        """No engine emits both; if one ever does, the run-relative
        value is the one that cannot be confused with a date."""
        assert _run_cycle_clock(
            "cycleClock({elapsed_s: 5.0, wall_clock_s: 1761396030.0})"
        ) == 5.0

    def test_neither_clock_is_null_not_zero(self):
        """A cycle with no timing must not read as 'at t=0' — that is
        how a missing measurement becomes a plotted point at the
        origin."""
        assert _run_cycle_clock("cycleClock({cycle: 1}) === null") is True
        assert _run_cycle_clock("cycleClock(null) === null") is True

    def test_non_finite_is_rejected(self):
        """SIESTA's Fortran column overflow yields NaN on the Time
        field; NaN must not pass as a measurement.

        Compared with ``=== null`` rather than for a null RESULT:
        JSON.stringify turns NaN into the token `null`, so a test that
        round-trips the value through JSON cannot tell the two apart
        and passes even when NaN leaks through.
        """
        assert _run_cycle_clock(
            "cycleClock({elapsed_s: NaN}) === null") is True
        assert _run_cycle_clock(
            "cycleClock({elapsed_s: Infinity}) === null") is True
        assert _run_cycle_clock(
            "cycleClock({wall_clock_s: NaN}) === null") is True


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
