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
    """Evaluate ``expr`` against the module's pure clock helpers under node.

    The extraction window runs from ``cumulativeElapsed`` to ``fmtElapsed``
    and so covers ``badgeClocks`` too -- both are module-level pure
    functions, extracted for the same reason and living side by side.
    """
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
    """The badge's clock selection, RUN -- `badgeClocks(state)`.

    These four assertions read `core.js` as TEXT until 2026-09-06 and checked
    for the spelling of the four lines that made the choice::

        assert "const clockSeries   = state.data.wall_clock_s || [];" in src

    Behaviour-blind in both directions.  Rename `clockSeries` consistently and
    it fails while the badge is perfect.  Swap the two clocks at their point
    of USE -- leaving those four declarations byte-identical -- and it passes
    while the badge formats a duration as a date and an epoch as a duration.
    That second one was measured: **233 tests green** with the clocks swapped.

    The decision now lives in a pure function beside `cumulativeElapsed`,
    which was extracted for exactly this reason, so these run it instead.
    """

    def test_each_clock_lands_in_its_own_slot(self):
        """The whole rule in one payload: two values that cannot be mistaken
        for each other, and each must come out of the right field.

        MUTATION THIS MUST FAIL AGAINST: swap `elapsed` and
        `lastResultEpoch` in the returned object, or read either from the
        other's series.
        """
        got = _run_cycle_clock(
            "badgeClocks({data: {elapsed_s: [10, 245],"
            "                    wall_clock_s: [1761396029, 1761396030]}})")
        assert got["elapsed"] == 245, (
            "the DURATION must be the last elapsed_s -- a 4m05s run")
        assert got["lastResultEpoch"] == 1761396030, (
            "the TIMESTAMP must be the last wall_clock_s -- an absolute epoch")

    def test_an_epoch_is_never_offered_as_the_duration(self):
        """The failure this file exists for, from the other side: a run with
        only an epoch has NO duration to show, and 1.76e9 is not one."""
        got = _run_cycle_clock(
            "badgeClocks({data: {wall_clock_s: [1761396030]}})")
        assert got["elapsed"] is None, (
            "an epoch leaked into the duration slot -- the badge would read "
            "'55 years' where it should read nothing")
        assert got["lastResultEpoch"] == 1761396030

    def test_a_duration_is_never_offered_as_the_timestamp(self):
        """P-T3: `wall_clock_s` may never be derived from `elapsed_s`,
        because the file does not contain the missing addend.  With no epoch
        and no mtime the answer is *nothing*, not the elapsed value -- which
        formatted as a date is the "Dec 31, 5:06 PM" badge."""
        got = _run_cycle_clock("badgeClocks({data: {elapsed_s: [360]}})")
        assert got["lastResultEpoch"] is None, (
            "an elapsed duration was offered as an absolute time")
        assert got["elapsed"] == 360

    def test_mtime_is_the_fallback_and_only_for_the_epoch(self):
        """A raw SIESTA `.out` with no molwatch hooks carries no clock of its
        own, so the file's own mtime stands in -- deliberately, which is what
        P-T2 means by `None` being a correct answer a consumer can act on."""
        got = _run_cycle_clock(
            "badgeClocks({mtime: 1761396030, data: {elapsed_s: [360]}})")
        assert got["lastResultEpoch"] == 1761396030
        assert got["elapsed"] == 360, "mtime must not touch the duration"

        # and the run's own clock beats it when there is one
        got = _run_cycle_clock(
            "badgeClocks({mtime: 1, data: {wall_clock_s: [1761396030]}})")
        assert got["lastResultEpoch"] == 1761396030, (
            "mtime is a FALLBACK -- it must not override a run's own clock")

    def test_the_last_FINITE_sample_wins_not_the_last(self):
        """A trailing null or NaN is a frame that reported no time, not a
        run that ended at zero.  Both series are read the same way."""
        got = _run_cycle_clock(
            "badgeClocks({data: {elapsed_s: [10, 245, null],"
            "                    wall_clock_s: [1761396030, NaN]}})")
        assert got["elapsed"] == 245
        assert got["lastResultEpoch"] == 1761396030

    def test_a_run_with_no_clocks_at_all_says_nothing(self):
        """Empty is not zero: a badge showing `0s` on a run that reported no
        time is a measurement it never made."""
        got = _run_cycle_clock("badgeClocks({data: {}})")
        assert got == {"elapsed": None, "lastResultEpoch": None}
