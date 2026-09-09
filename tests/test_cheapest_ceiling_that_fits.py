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


def test_a_quick_small_job_can_still_reach_debug():
    """`debug` is the cheapest thing on the machine to get, and a job that
    fits inside its 15 minutes should have it."""
    assert _where(Request(ranks=8, walltime_s=600, mem_gb=16)) == "debug"


def test_memory_decides_between_queues_that_are_otherwise_alike():
    """**Cores are the requirement; memory is the chooser** (user, 2026-08-23).

    Wall-clock depends most critically on the core count, so cores are not
    traded away — `admits` has already guaranteed them.  These jobs are core
    bound and do not press against memory ceilings, so the queue offering
    LESS memory is the easier one to allocate, and asking for a partition you
    do not need buys a longer wait and nothing else.

    Here both queues hold the wall and the cores, and only memory differs.
    The lean one must win.
    """
    lean = Domain(name="lean", partition="p", qos="q", max_time="0-04:00:00",
                  max_cores=128, max_mem_gb=251.0)
    fat = Domain(name="fat", partition="p", qos="q", max_time="0-04:00:00",
                 max_cores=128, max_mem_gb=2002.0)
    r = Request(ranks=64, walltime_s=2280, mem_gb=128)
    assert place([fat, lean], r, prefer_gpu=False).domain.name == "lean", (
        "the fatter queue was chosen even though both fit -- a 2 TB "
        "allocation for a 128 G job is a longer wait and nothing else")


def test_memory_outranks_a_much_larger_walltime_difference():
    """**Why the key is lexicographic and not a sum.**

    The first version added the three ratios equally.  On a Sol-shaped menu
    the walltime ratios span 6x-76x while memory spans 2x-16x, so the sum was
    a walltime sort in disguise and memory could never decide.  Averaging
    quantities whose spreads differ by an order of magnitude is a way of
    choosing by the loudest one.

    Here the lean queue has a *far* longer wall ceiling -- which the old sum
    would have punished heavily -- and it is still the right answer, because
    the wall only has to FIT and the memory is what costs.
    """
    lean_long = Domain(name="lean", partition="p", qos="q",
                       max_time="7-00:00:00", max_cores=128, max_mem_gb=251.0)
    fat_short = Domain(name="fat", partition="p", qos="q",
                       max_time="0-04:00:00", max_cores=128,
                       max_mem_gb=2002.0)
    r = Request(ranks=64, walltime_s=2280, mem_gb=128)
    assert place([fat_short, lean_long], r,
                 prefer_gpu=False).domain.name == "lean"


def test_the_wall_still_has_to_fit_though():
    """Memory chooses only among queues that already admit; a queue whose
    wall is too short never reaches the comparison."""
    lean_short = Domain(name="lean", partition="p", qos="q",
                        max_time="0-00:15:00", max_cores=128,
                        max_mem_gb=251.0)
    fat_ok = Domain(name="fat", partition="p", qos="q", max_time="0-04:00:00",
                    max_cores=128, max_mem_gb=2002.0)
    r = Request(ranks=64, walltime_s=2280, mem_gb=128)
    assert place([lean_short, fat_ok], r, prefer_gpu=False).domain.name == "fat"


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


def test_ordering_never_rescues_a_request_nothing_admits():
    """It orders survivors; it does not create them."""
    from molbuilder.scheduler.place import Unplaceable
    with pytest.raises(Unplaceable):
        _where(Request(ranks=64, walltime_s=2280, mem_gb=99999))


# --------------------------------------------------------------------- #
#  the order is the SITE's, not the code's                              #
# --------------------------------------------------------------------- #

def _pair():
    """Two queues that both fit, differing on memory and walltime in opposite
    directions — so whichever axis leads decides, and the test can see it."""
    lean_long = Domain(name="lean-long", partition="p", qos="q",
                       max_time="7-00:00:00", max_cores=128, max_mem_gb=251.0)
    fat_short = Domain(name="fat-short", partition="p", qos="q",
                       max_time="0-04:00:00", max_cores=128,
                       max_mem_gb=2002.0)
    return [fat_short, lean_long], Request(ranks=64, walltime_s=2280,
                                           mem_gb=128)


@pytest.mark.parametrize("order,winner", [
    (None,                             "lean-long"),   # the default
    (["cores", "memory", "walltime"],  "lean-long"),
    (["memory", "cores", "walltime"],  "lean-long"),
    (["walltime", "memory"],           "fat-short"),
])
def test_the_declared_priority_decides(order, winner):
    """**The order is tunable** (user, 2026-08-23).  A site whose jobs press
    on a different axis says so and gets a different queue — the same menu,
    the same request, a different answer, because the preference changed."""
    menu, req = _pair()
    assert place(menu, req, prefer_gpu=False,
                 priority=order).domain.name == winner


def test_an_unknown_axis_is_refused_rather_than_dropped():
    """A preference silently ignored looks honoured and is not — which is the
    failure this whole document exists to remove."""
    from molbuilder.scheduler.place import check_priority
    with pytest.raises(ValueError):
        check_priority(["memroy"])          # a plausible typo


def test_the_site_can_set_it_in_the_config(tmp_path, monkeypatch):
    """It is a PREFERENCE, so it lives in molbuilder.json and never in the
    machine record (M-1): the record measures what the queues offer, this
    says which of those facts matters most here."""
    import json
    from molbuilder.runtime_config import PROJECT_CONFIG_FILENAME, get_scheduler
    (tmp_path / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "scheduler": {"kind": "slurm",
                      "directives": {"partition": "htc", "qos": "public"},
                      "placement_priority": ["memory", "cores"]}}))
    got = get_scheduler(project_dir=tmp_path)
    assert got["placement_priority"] == ["memory", "cores"]


def test_a_bad_priority_in_the_config_refuses_the_config(tmp_path):
    import json
    from molbuilder.runtime_config import (PROJECT_CONFIG_FILENAME,
                                           RuntimeConfigError, get_scheduler)
    (tmp_path / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "scheduler": {"kind": "slurm",
                      "directives": {"partition": "htc", "qos": "public"},
                      "placement_priority": ["gpu"]}}))
    with pytest.raises(RuntimeConfigError) as e:
        get_scheduler(project_dir=tmp_path)
    assert "placement_priority" in str(e.value)
