"""The ``util.csv`` a real run's monitor wrote reads as a mean over TIME
(`parse.instruments.util_csv`; `run-reports.md` § 2.1; `process/testing.md`
§ 0, tier 3).

The monitor writes a row when a metric moved past its threshold, a keepalive
elapsed or the run ended, so its rows are not evenly spaced, and a column can
have a hole: it has no CPU reading to write in its first row, or a last.  The
reader weighs each sample by the interval until the next sample of its column:
the last ends the window, and a hole is absent from that column alone.  Every
expected number here is that rule worked on the file's rows, read as plain csv,
and no reader of ours.

WHAT A SHORT RUN REACHES, measured 2026-11-06: an H2 stage's file has rows ten
seconds apart and a last 13 or 14 s after them, each with a hole in the first
row's CPU, and a warm stage's has one CPU reading.  So a row standing for
longer than its neighbours is reached, by about 1.4 to 1 -- not the keepalive's
300 s (`test_util_csv_is_time_weighted.py`, on analytic input).

THE SILENT FAILURES: a mean over rows, which weighs a brief spike as a long
steady stretch; a column with a hole paired with the wrong epochs; a window
measured from a column's own rows.
"""
from __future__ import annotations

import csv
import io

import pytest

from _road import conda_hook, env_available, gpu_visible
from molbuilder.parse.instruments.util_csv import util_csv_metrics

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


def _util_csv(run, stage: str, n: int = 0) -> str:
    """The text of the file run ``n`` of ``stage``'s monitor wrote, asked of
    the run door (`runs.run_of`)."""
    from molbuilder.runfiles import ROLE_UTIL_CSV
    from molbuilder.runs import run_of
    return run_of(run.attempt(stage, n)).file(ROLE_UTIL_CSV).read_text(
        encoding="utf-8")


def _series(text: str, column: str) -> list:
    """``column`` as ``(epoch, value)`` where it holds a value, off the
    file's rows."""
    return [(float(row["epoch"]), float(row[column]))
            for row in csv.DictReader(io.StringIO(text)) if row[column]]


def _over_time(series: list) -> float:
    """The rule: each sample weighs the interval until the next, the last
    ends the window, and one sample alone is what was measured."""
    if len(series) == 1:
        return series[0][1]
    held = zip(series, series[1:])
    return (sum(value * (later - at) for (at, value), (later, _) in held)
            / (series[-1][0] - series[0][0]))


def test_a_real_monitors_file_reads_as_a_mean_over_time(real_h2_layered):
    """Each file an H2 run's monitor wrote: its CPU mean is the rule worked on
    its rows, and its window runs from its first row to its last, whichever
    columns hold a value there, a hole in the CPU column included.  (How far a
    mean over rows would differ is this machine's speed: the shape that shows it
    plainly, one row covering 300 s, is the analytic test's.)"""
    files = [_util_csv(real_h2_layered, stage, n)
             for stage, n in (("coarse", 0), ("coarse", 1), ("coarse", 2),
                              ("medium", 0))]
    holes = []
    for text in files:
        got = util_csv_metrics(text)
        rows = list(csv.DictReader(io.StringIO(text)))
        cpu = _series(text, "cpu_pct")
        assert got["cpu_mean_pct"] == pytest.approx(_over_time(cpu), abs=0.051)
        assert got["monitored_elapsed_s"] == pytest.approx(
            float(rows[-1]["epoch"]) - float(rows[0]["epoch"]))
        holes.append(len(cpu) < len(rows))
    assert all(holes), files


@pytest.mark.skipif(not (env_available("molbuilder-siesta-gpu")
                         and gpu_visible()),
                    reason="needs an NVIDIA GPU and the molbuilder-siesta-gpu env")
def test_a_gpu_column_is_weighted_the_same_way(real_h2_gpu):
    """`gpu<N>_sm` of the file a real GPU run's monitor wrote: the rule worked
    on its rows -- it was the same mean over rows, and a GPU that idles at
    startup is the common case."""
    text = _util_csv(real_h2_gpu, "coarse")
    assert util_csv_metrics(text)["gpu_sm_mean_pct"] == pytest.approx(
        _over_time(_series(text, "gpu0_sm")), abs=0.051), text
