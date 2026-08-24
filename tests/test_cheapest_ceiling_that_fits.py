"""Placement takes the cheapest ceiling that FITS — on every stated axis.

`scheduler.md` § 5's graph has said *"cheapest ceiling that fits"* since it was
drawn, and the code took **the first admitting row** from a menu ordered by
**walltime alone**.  So the choice was *"the shortest queue that says yes"*,
which is not the same thing and differs exactly where it costs: a 38-minute
job needing 128 GB could land on the partition built for 2 TB work, wait
behind it, and pay a scheduling penalty for memory nobody chose
(`submission.md` § 3).

Nothing here changes what is ADMITTED.  A row that does not fit was refused
before this ordering is consulted; this only decides which of the survivors is
the one to take.
"""
from __future__ import annotations

import pytest

from molbuilder.scheduler import Domain
from molbuilder.scheduler.admit import Request
from molbuilder.scheduler.place import _excess, place


def _sol():
    """Sol-shaped, in the order the probe writes it: by walltime, cheapest
    wall first.  `highmem` therefore sits BEFORE `general`, which is what made
    the old first-that-fits rule reach for it."""
    return [
        Domain(name="debug", partition="htc", qos="debug",
               max_time="0-00:15:00", max_cores=128, max_mem_gb=251.0),
        Domain(name="htc", partition="htc", qos="public",
               max_time="0-04:00:00", max_cores=128, max_mem_gb=251.0),
        Domain(name="highmem", partition="highmem", qos="public",
               max_time="2-00:00:00", max_cores=128, max_mem_gb=2002.0),
        Domain(name="general", partition="general", qos="public",
               max_time="7-00:00:00", max_cores=128, max_mem_gb=502.9),
    ]


def _where(request):
    return place(_sol(), request, prefer_gpu=False).domain.name


def test_the_job_that_prompted_this_lands_on_the_ordinary_queue():
    """64 cores, 128 GB, 38 minutes.  `debug` cannot hold 38 minutes, and of
    the three that can, `htc` is the tightest fit on every axis."""
    assert _where(Request(ranks=64, walltime_s=2280, mem_gb=128)) == "htc"


def test_the_case_where_first_that_fits_and_cheapest_fit_DISAGREE():
    """**The discriminating case, and the first version of this file had
    none.**

    Every other test here passes under the old rule too: with the menu ordered
    by walltime, the shortest queue that admits is usually also the tightest,
    so the two rules agree and a test cannot tell them apart.  Reverting to
    `fits[0]` passed all eight assertions — verified by mutation 2026-08-23.

    They disagree exactly when an EARLY row (short wall) is wasteful on
    another axis and a LATER row is tighter overall.  A 1-day 128 GB job meets
    `highmem` first — 2-day wall, 2 TB — and `general` is the better queue on
    every axis that matters.
    """
    assert _where(Request(ranks=64, walltime_s=86400, mem_gb=128)) == "general"
    # ...and under the old rule it would have been the first that admitted:
    fits = [d for d in _sol()
            if d.name in ("highmem", "general")]
    assert fits[0].name == "highmem", (
        "the menu no longer puts highmem before general, so this test has "
        "stopped discriminating -- find another case or delete it")


def test_a_job_that_genuinely_needs_the_big_memory_still_gets_it():
    """The check the fix must not break: tightest-fit is not smallest-queue.
    Nothing else admits 900 GB, so `highmem` is correct and stays correct."""
    assert _where(Request(ranks=64, walltime_s=2280, mem_gb=900)) == "highmem"


def test_a_long_job_takes_the_long_queue():
    """A 4.6-day wall rules out everything but `general`, whose ceiling is the
    only one that holds it — tightness never overrides admission."""
    assert _where(Request(ranks=64, walltime_s=400000, mem_gb=128)) == "general"


def test_a_quick_small_job_can_still_reach_debug():
    """`debug` is the cheapest thing on the machine to get, and a job that
    fits inside its 15 minutes should have it."""
    assert _where(Request(ranks=8, walltime_s=600, mem_gb=16)) == "debug"


# --------------------------------------------------------------------- #
#  the ordering key itself                                              #
# --------------------------------------------------------------------- #

def test_an_unstated_ceiling_is_not_a_tight_one():
    """R3 in the sort key.  A row that declares nothing has not said it fits
    snugly — it has said nothing — and it sorts AFTER rows whose fit is
    measurable.  Otherwise an unmeasured queue wins every comparison by
    silence, which is the loudest possible way to be wrong quietly.
    """
    r = Request(ranks=64, walltime_s=2280, mem_gb=128)
    known = Domain(name="k", partition="p", qos="q", max_time="0-04:00:00",
                   max_cores=128, max_mem_gb=251.0)
    silent = Domain(name="s", partition="p", qos="q")
    assert _excess(known, r) < _excess(silent, r)
    assert _excess(silent, r)[0] == 3, "three ceilings unknown, and counted"


def test_a_dimension_the_ASK_does_not_state_is_not_compared():
    """Both halves are required.  A request that says nothing about memory
    cannot prefer a queue on memory grounds."""
    r = Request(ranks=64, walltime_s=2280)          # no mem_gb
    tight = Domain(name="t", partition="p", qos="q", max_time="0-04:00:00",
                   max_cores=128, max_mem_gb=251.0)
    huge = Domain(name="h", partition="p", qos="q", max_time="0-04:00:00",
                  max_cores=128, max_mem_gb=2002.0)
    assert _excess(tight, r) == _excess(huge, r)


def test_the_menus_own_order_still_breaks_a_tie():
    """`min` is stable, so among equally tight rows the menu's recommendation
    survives as the tie-break rather than being overruled (R7)."""
    a = Domain(name="a", partition="p", qos="q", max_time="0-04:00:00",
               max_cores=128, max_mem_gb=251.0)
    b = Domain(name="b", partition="p", qos="q", max_time="0-04:00:00",
               max_cores=128, max_mem_gb=251.0)
    r = Request(ranks=64, walltime_s=2280, mem_gb=128)
    assert place([a, b], r, prefer_gpu=False).domain.name == "a"
    assert place([b, a], r, prefer_gpu=False).domain.name == "b"


def test_ordering_never_rescues_a_request_nothing_admits():
    """It orders survivors; it does not create them."""
    from molbuilder.scheduler.place import Unplaceable
    with pytest.raises(Unplaceable):
        _where(Request(ranks=64, walltime_s=2280, mem_gb=99999))
