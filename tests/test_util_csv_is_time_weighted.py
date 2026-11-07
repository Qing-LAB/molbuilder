"""``util.csv`` is change-gated, so its mean must be over TIME: the
arithmetic of `parse.instruments.util_csv._time_weighted`, on analytic input.
What the reader makes of a file a real monitor wrote is
`test_util_csv_is_time_weighted_e2e.py`.

The monitor writes a row only when a metric moves past its threshold, or
when a 300 s keepalive elapses (``monitor.py``: ``util_change_frac`` /
``util_keepalive_s``).  Rows are therefore deliberately not uniformly
spaced -- that is the whole point, it keeps the file small on a long run.

``util_csv_metrics`` took ``sum/len`` over those rows until 2026-08-25, which
weights a one-second startup sample exactly as heavily as five minutes of
steady state.  Found on a real Au-BDT-Au trial: 316 s, five rows, one of
them covering 300 s.  Reported 31.5%; true figure 40.3%.

Worst on the SHORTEST runs, which is why it survived: dense trials came
within a point of correct.

A short real run has no such row -- its longest measured here is 14 s, a CO2
force-constant call's 24 s -- so the keepalive's own shape stays here, as the
numbers it was measured as.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.instruments.util_csv import _time_weighted


def test_a_long_steady_tail_outweighs_a_short_startup():
    """THE LIVE SHAPE.  Four startup samples five seconds apart, then one
    row covering the next 300 s -- exactly what change-gating produces."""
    series = [(0, 7.5), (5, 27.1), (10, 41.1), (15, 41.0), (316, 40.8)]
    got = _time_weighted(series)
    unweighted = sum(c for _e, c in series) / len(series)
    assert got == pytest.approx(40.3, abs=0.2), (
        f"got {got}; the unweighted mean would be {unweighted:.1f}")
    assert got > unweighted + 5, (
        "the startup transient still dominates -- this is the defect")


def test_uniform_samples_are_unaffected():
    """The control: when rows ARE evenly spaced the two agree, which is
    why dense trials never showed the bug."""
    series = [(0, 10.0), (10, 20.0), (20, 30.0), (30, 40.0)]
    # 10,20,30 each held for 10 s; the final sample ends the window.
    assert _time_weighted(series) == pytest.approx(20.0, abs=0.1)
