"""**The exact number when the monitor wrote one, the samples when it did not**
*(user ruling, 2026-09-03)*.

Two files describe one fact and each is better at something:

| | basis | availability |
|---|---|---|
| the monitor's `[UTIL-SUMMARY]` | **every tick** — its accumulator runs unconditionally | only if the monitor reached its terminal branch |
| `util.csv` | a **change-gated** subset — a row lands when a metric moves ≥ 10% or a 300 s keepalive elapses | always |

So the summary is the right number and the csv is the one that is always there.
A trial the scheduler **kills** has a csv and no summary — and that is the
trial a benchmark most needs to read, which is why the csv path cannot simply
be dropped.

**The pair was collapsed into the csv alone on 2026-08-19** after two readers
of one fact diverged.  That lesson holds, and it is why the choice lives in
ONE function (`_utilisation`) rather than at each call site: there is
still exactly one thing to be wrong.

**The gap between the two is real, not theoretical.**  The fixture below is
change-gated the way a real series is, and the two answers differ by three
points — the csv weights a value that held for one interval the same as one
that held for two.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.instruments import utilisation as _resolve
from molbuilder.parse.instruments.monitor import monitor_metrics
from molbuilder.parse.instruments.util_csv import util_csv_metrics


def _utilisation(monitor_log, csv_text):
    """The pair, resolved -- what `parse_utilisation` did in one call
    before the readers became registered parsers (`parse.md` § 5c)."""
    return _resolve(monitor_metrics(monitor_log), util_csv_metrics(csv_text))


#: A change-gated series: 50% for the first interval, 90% for the next two.
#: Time-weighted over the ROWS this is 70%; over every TICK the monitor took,
#: it is 73% — and only the monitor can know that, because the ticks between
#: the rows were never written down.
_CSV = (
    "epoch,iso,cpu_pct,mem_gb,gpu0_sm_pct\n"
    "0,x,50.0,1.0,10.0\n"
    "100,x,90.0,2.0,90.0\n"
    "200,x,90.0,2.0,90.0\n"
)

_MONITOR_FINISHED = (
    "[t] [MACHINE] node=n1 cores=8 mem_gb=32 gpu=none\n"
    "[t] [UTIL-SUMMARY] cpu mean=73% (50-90); gpu0 sm mean=61% (10-90)"
    " -> mixed (GPU not saturated)\n"
)

#: What a KILLED trial leaves: the first line, never the last.
_MONITOR_KILLED = "[t] [MACHINE] node=n1 cores=8 mem_gb=32 gpu=none\n"


def test_the_monitors_own_mean_wins_when_it_is_there():
    got = _utilisation(_MONITOR_FINISHED, _CSV)
    assert got["cpu_mean_pct"] == 73.0, (
        "the summary's mean is over every tick; the csv's is over a gated "
        "subset, and reading the subset when the exact figure is on disk is "
        "what this ruling changed")
    assert got["gpu_sm_mean_pct"] == 61.0
    assert got["util_basis"] == "monitor-summary"


def test_a_killed_trial_still_answers_from_the_samples():
    """The case the fallback exists for.

    The summary is written only in the monitor's terminal branch, so a
    scheduler kill takes it — and takes it from exactly the run somebody is
    trying to understand.
    """
    got = _utilisation(_MONITOR_KILLED, _CSV)
    assert got["cpu_mean_pct"] == pytest.approx(70.0), (
        "with no summary the samples must still answer")
    assert got["util_basis"] == "util-csv", (
        "a reconstruction must not be labelled as the monitor's own figure")


def test_the_two_sources_actually_disagree():
    """The mutation guard for the pair above.

    If the fixture's csv happened to reconstruct 73% too, both tests would
    pass no matter which branch ran.  They differ by three points, so the
    assertions above can only be satisfied by reading the intended source.
    """
    assert util_csv_metrics(_CSV)["cpu_mean_pct"] != 73.0


def test_what_the_summary_does_not_carry_still_comes_from_the_csv():
    """Peak RSS, the sampled wall window and peak VRAM appear on no summary
    line, so preferring it must not lose them."""
    got = _utilisation(_MONITOR_FINISHED, _CSV)
    assert got["peak_rss_gb"] == 2.0
    assert got["wall_s"] == pytest.approx(200.0)


def test_neither_source_says_nothing_rather_than_something_invented():
    assert _utilisation("", "") == {}
    assert _utilisation(_MONITOR_KILLED, "") == {}, (
        "no samples and no summary is no answer, not a zero")


def test_a_multi_gpu_summary_reduces_the_way_the_csv_does():
    """Both sources report ONE gpu figure, and it is the max across devices.
    Two different reductions would make the answer depend on which file was
    read — the divergence this one door exists to prevent."""
    mon = ("[t] [UTIL-SUMMARY] cpu mean=50% (10-90); gpu0 sm mean=20% (1-40); "
           "gpu1 sm mean=80% (60-95) -> mixed (GPU not saturated)\n")
    assert _utilisation(mon, _CSV)["gpu_sm_mean_pct"] == 80.0
