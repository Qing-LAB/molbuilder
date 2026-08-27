"""A partition is a QUEUE, not a machine type.

Measured on ASU Sol 2026-08-27: `htc` is 51 nodes of 48 cores with A100s,
3 of 64 with MIG slices, and 134 of 128 with no device at all. `general` and
`public` have the same shape, and what separates the three is the wall clock
— 4 h, 7 d, 14 d — not the hardware.

User: *list available machine types explicitly and allow cpu request to fit
that range instead of one lowest fit. That's more reasonable and what the
user would expect.*

**Why one number could not work.** `max_cores` was a MINIMUM across GPU
groups for a gpu-capable partition and a MAXIMUM across all groups for a
cpu-only one, and `admits` compared both as one ceiling. Either reading is
wrong somewhere: a floor refuses work the wide nodes would run, a ceiling
admits work most nodes cannot hold. The groups are the measurement; any
scalar over them is an opinion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_scheduler_probe import _sol                      # noqa: E402

from molbuilder.jobset.ask import Ask, queue_table         # noqa: E402
from molbuilder.scheduler.admit import Request, admits     # noqa: E402
from molbuilder.scheduler.probe import derive_domains, parse_sinfo  # noqa: E402
from molbuilder.scheduler.record import Domain             # noqa: E402


def _domains():
    return {d.name: d for d in
            (Domain.from_row(r) for r in derive_domains(*_sol())[0]) if d}


# --------------------------------------------------------------------- #
#  the measurement survives                                              #
# --------------------------------------------------------------------- #

def test_every_machine_in_the_queue_is_listed(): 
    htc = _domains()["htc"]
    assert {(t["cores"], t["nodes"]) for t in htc.node_types} == \
        {(48, 51), (64, 3), (128, 134)}


def test_the_device_rides_with_the_machine_that_has_it():
    """Not with the queue. `htc`'s `gpu` column says a100 and a100.20gb
    exist somewhere in it; only the per-machine rows say the A100s are on
    the 48-core nodes and nothing is on the 128-core ones."""
    by_cores = {t["cores"]: t for t in _domains()["htc"].node_types}
    assert by_cores[48]["gpu"] == {"a100": 4}
    assert by_cores[64]["gpu"] == {"a100.20gb": 16}
    assert "gpu" not in by_cores[128]


def test_a_single_shape_queue_lists_exactly_one():
    assert [(t["cores"], t["nodes"])
            for t in _domains()["highmem"].node_types] == [(128, 11)]


def test_two_domains_over_one_partition_see_the_same_machines():
    """`debug` and `htc` are two QoS over the partition `htc` — same nodes,
    different clock. That is what makes a partition a queue."""
    d = _domains()
    assert d["debug"].partition == d["htc"].partition == "htc"
    assert d["debug"].node_types == d["htc"].node_types
    assert d["debug"].max_time != d["htc"].max_time


# --------------------------------------------------------------------- #
#  admission: refuse only what nothing here can hold                     #
# --------------------------------------------------------------------- #

def test_a_64_rank_cpu_ask_is_ADMITTED_where_a_floor_refused_it():
    """**The case that caught the collapsed field.** `htc`'s GPU nodes have
    48 cores, so a floor refused this — but its 134 CPU-only nodes have 128,
    and SLURM will not place a job on a node too small; it waits for one
    that fits."""
    assert admits(_domains()["htc"], Request(ranks=64)) == []


def test_an_ask_no_machine_can_hold_is_refused_by_name():
    """R10 — name what WOULD fit. A refusal that stops at "no" leaves the
    person guessing at a number the record is already holding."""
    why = admits(_domains()["htc"], Request(ranks=256))
    assert why and "256" in why[0]
    assert "128" in why[0], f"the refusal must name the largest: {why[0]}"


def test_the_widest_machine_is_the_ceiling_for_every_queue():
    for name, row in _domains().items():
        widest = max((t["cores"] for t in row.node_types or []
                      if t.get("cores")), default=None)
        if not widest:
            continue
        assert admits(row, Request(ranks=widest)) == [], \
            f"{name}: its own largest machine was refused"
        assert admits(row, Request(ranks=widest + 1)), \
            f"{name}: an ask past every machine was admitted"


def test_the_machines_bound_the_ask_even_with_no_max_cores():
    """**Found by mutation, 2026-08-27.** With `max_cores` derived as the
    widest machine, reading it and reading `node_types` agree on every
    probed record — so a mutation that ignored `node_types` entirely passed
    all 36 tests. They only differ on a HAND-WRITTEN row, which is exactly
    the case `Domain` exists to serve equally: *"one type for both ways a
    fact arrives."*

    Someone who lists the machines they know about has stated a limit, and
    it must bind without their having to also compute a scalar from it.
    """
    hand = Domain(name="site", partition="p", qos="public",
                  node_types=[{"cores": 32, "nodes": 4},
                              {"cores": 96, "nodes": 2}])
    assert hand.max_cores is None
    assert admits(hand, Request(ranks=96)) == []
    why = admits(hand, Request(ranks=97))
    assert why, "a stated set of machines must bound the ask"
    assert "96" in why[0]


def test_the_refusal_names_the_MACHINE_not_just_the_number():
    """R4 — *a refusal names the numbers* — and R10 — *name what would
    fit*. Knowing 128 is the ceiling is less useful than knowing it is 134
    nodes: that is the difference between "trim this" and "this will
    queue"."""
    why = admits(_domains()["htc"], Request(ranks=256))
    assert "134 node(s) of 128" in why[0], why[0]


def test_a_record_that_lists_no_machines_never_bars():
    """R3 — *an unstated limit never bars.* Every record written before
    2026-08-27 has no `node_types`, and must keep working from
    `max_cores`."""
    old = Domain(name="htc", partition="htc", qos="public", max_cores=48)
    assert old.node_types is None
    assert admits(old, Request(ranks=48)) == []
    assert admits(old, Request(ranks=64)), "max_cores still bounds"
    silent = Domain(name="x", partition="x", qos="public")
    assert admits(silent, Request(ranks=100000)) == [], \
        "a domain that states nothing must refuse nothing"


# --------------------------------------------------------------------- #
#  what a person reads                                                   #
# --------------------------------------------------------------------- #

def test_the_table_shows_the_machines_not_one_number():
    rows = [d for d in _domains().values()]
    out = queue_table(rows, Ask(), cores=64)
    assert "48 cores  x51 node(s)  a100 x4" in out
    assert "128 cores  x134 node(s)" in out


def test_a_machine_too_small_for_THIS_ask_is_marked_not_hidden():
    """It is why the queue is slower than its node count suggests: a
    64-core ask on `htc` fits 137 of 188 nodes, and **none of the A100
    ones** — so "64 cores with an A100" is impossible there, which no
    single figure could have said."""
    out = queue_table([_domains()["htc"]], Ask(), cores=64)
    line = [ln for ln in out.splitlines() if "48 cores" in ln][0]
    assert "(too small)" in line
    assert "128 cores" in out and "(too small)" not in \
        [ln for ln in out.splitlines() if "128 cores" in ln][0]


def test_a_single_shape_queue_prints_no_extra_lines():
    """The `cores` column already said it. Repeating it on every row that
    has nothing to disclose is noise."""
    out = queue_table([_domains()["highmem"]], Ask())
    assert "128 cores  x11" not in out
    assert "highmem" in out


def test_the_table_still_refuses_to_choose():
    """`queue_table`'s own rule, unchanged: *it shows what exists, marks
    what fits, and the person picks.* More detail must not become a
    recommendation."""
    out = queue_table(list(_domains().values()), Ask(), cores=64)
    assert "choose one with --domain" in out
    for word in ("recommended", "best", "fastest", "you should"):
        assert word not in out.lower()
