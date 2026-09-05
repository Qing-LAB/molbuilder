"""``util.csv`` is change-gated, so its mean must be over TIME.

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
"""
from __future__ import annotations

import pytest

from molbuilder.parse.instruments.util_csv import util_csv_metrics

_HEADER = "epoch,iso,cpu_pct,mem_gb\n"


def _csv(rows):
    return _HEADER + "".join(
        f"{e},2026-08-25T00:00:00,{c},10.0\n" for e, c in rows)


def test_a_long_steady_tail_outweighs_a_short_startup():
    """THE LIVE SHAPE.  Four startup samples five seconds apart, then one
    row covering the next 300 s -- exactly what change-gating produces."""
    rows = [(0, 7.5), (5, 27.1), (10, 41.1), (15, 41.0), (316, 40.8)]
    got = util_csv_metrics(_csv(rows))["cpu_mean_pct"]
    unweighted = sum(c for _e, c in rows) / len(rows)
    assert got == pytest.approx(40.3, abs=0.2), (
        f"got {got}; the unweighted mean would be {unweighted:.1f}")
    assert got > unweighted + 5, (
        "the startup transient still dominates -- this is the defect")


def test_uniform_samples_are_unaffected():
    """The control: when rows ARE evenly spaced the two agree, which is
    why dense trials never showed the bug."""
    rows = [(0, 10.0), (10, 20.0), (20, 30.0), (30, 40.0)]
    got = util_csv_metrics(_csv(rows))["cpu_mean_pct"]
    # 10,20,30 each held for 10 s; the final sample ends the window.
    assert got == pytest.approx(20.0, abs=0.1)


def test_a_single_row_still_answers():
    """One sample has no interval.  It is still the only thing measured,
    and reporting it beats reporting nothing."""
    assert util_csv_metrics(_csv([(5, 33.0)]) )["cpu_mean_pct"] == \
        pytest.approx(33.0)


def test_gpu_sm_is_weighted_the_same_way():
    """`gpu<N>_sm` runs through the identical path -- it was the same
    unweighted mean, and a GPU that idles at startup is the common case."""
    txt = ("epoch,iso,cpu_pct,gpu0_sm\n"
           "0,x,50,5\n5,x,50,90\n300,x,50,90\n")
    assert util_csv_metrics(txt)["gpu_sm_mean_pct"] == pytest.approx(88.6, abs=0.5)


def test_a_hole_in_one_column_does_not_shift_the_others_onto_it():
    """A CELL THAT DOES NOT PARSE MUST NOT MOVE ITS NEIGHBOURS.

    `nvidia-smi` writes ``[N/A]`` for utilisation on MIG instances and on
    some drivers, and a transient failure writes an empty cell.  The reader
    collected each column into its own list, skipping cells it could not
    parse -- so a column with a hole came out SHORTER than the epoch column
    and every later reading silently took an earlier row's timestamp.

    Harmless while the mean was `sum/len` (alignment does not matter to a
    plain average) and load-bearing the moment it became time-weighted.
    The failure mode was the bad kind: no exception, no warning, a
    plausible number.  Here the truth is 36.7% and the shifted read was
    50.0% -- a GPU that idled for two thirds of the window reported as
    half busy.
    """
    txt = ("epoch,iso,cpu_pct,gpu0_sm\n"
           "0,x,50,10\n"
           "10,x,50,[N/A]\n"      # the hole
           "20,x,50,90\n"
           "30,x,50,90\n")
    got = util_csv_metrics(txt)["gpu_sm_mean_pct"]
    # 10 held 0->20 (nothing observed at 10), 90 held 20->30.
    assert got == pytest.approx((10 * 20 + 90 * 10) / 30, abs=0.1), (
        f"got {got}; 50.0 means the 90s were paired with the wrong epochs")
    # The intact column in the same file must be untouched by its neighbour.
    assert util_csv_metrics(txt)["cpu_mean_pct"] == pytest.approx(50.0, abs=0.1)


def test_the_window_is_not_truncated_by_a_short_column():
    """The same defect seen through `monitored_elapsed_s`'s eyes.

    The old pairing also shortened the SPAN: with one hole it divided by
    the interval up to the second-to-last epoch instead of the whole
    window, so the mean was scaled by a factor that depended on how many
    cells happened to be unreadable.
    """
    txt = ("epoch,iso,cpu_pct,gpu0_sm\n"
           "0,x,50,80\n100,x,50,[N/A]\n200,x,50,80\n")
    assert util_csv_metrics(txt)["monitored_elapsed_s"] == pytest.approx(200.0)
    assert util_csv_metrics(txt)["gpu_sm_mean_pct"] == pytest.approx(80.0, abs=0.1)
