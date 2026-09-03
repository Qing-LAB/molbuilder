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


def test_a_gpu_ask_is_bounded_by_the_machines_that_HAVE_one():
    """**`execution/scheduler.md` R3's second half.** "SLURM will not
    place a job on a node too small" is exactly as true of devices: a
    ``--gres`` job can only land on a node that has one.  On `htc` the
    device-bearing machines top out at 64 cores while the queue's widest
    is 128 — so 64 ranks + a GPU fits (the MIG nodes), 65 does not, and
    the refusal names the device-side ceiling rather than the queue's."""
    htc = _domains()["htc"]
    assert admits(htc, Request(ranks=64, gpus=1)) == []
    why = admits(htc, Request(ranks=65, gpus=1))
    assert why, "65 ranks + a device fits no device-bearing machine"
    assert "64" in why[0] and "128" not in why[0], (
        f"the ceiling must be the widest machine WITH a device: {why[0]}")


def test_the_same_ranks_without_a_device_still_enjoy_the_wide_nodes():
    """**The mutation guard for P6.** A device filter that always applies
    would re-break the 64-rank CPU trial R3's corollary was written for —
    the two asks must keep different ceilings."""
    htc = _domains()["htc"]
    assert admits(htc, Request(ranks=65)) == [], (
        "a CPU ask was bounded by the GPU nodes' size — the device filter "
        "is applying to requests that name no device")
    assert admits(htc, Request(ranks=128)) == []


def test_a_machine_list_that_names_no_devices_does_not_bar_a_gpu_ask():
    """Silence never bars (R3): a hand-written list that does not say
    which nodes hold the devices is not claiming none do — the unfiltered
    widest stands, and the device count question stays with the `gpu`
    column."""
    hand = Domain(name="site", partition="p", qos="public",
                  gpu={"type": "a100", "per_node": 4},
                  node_types=[{"cores": 96, "nodes": 2}])
    assert admits(hand, Request(ranks=96, gpus=1)) == []
    # ...and the unfiltered machines still BOUND the device ask: an empty
    # filter must fall back to them, not to nothing.  (First caught as a
    # mutation survivor: dropping the fallback slid through because
    # max_cores caught probed rows, and this hand row has none.)
    why = admits(hand, Request(ranks=97, gpus=1))
    assert why and "96" in why[0], (
        "a device ask slipped past the stated machines when none of them "
        "named a device -- the filter emptied the list instead of "
        "standing aside")


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
    assert "48 cores  x51 node(s)" in out
    assert "128 cores  x134 node(s)" in out
    assert "a100" in out, "the cards available at a size ride with it"


def test_the_machines_are_grouped_by_SIZE_not_by_gres_row():
    """**Found against the real Sol record, 2026-08-27.**

    `sinfo` reports one row per *gres group*, and the freshly probed `htc`
    has FOURTEEN — nine of them 48-core rows differing only in which card
    they carry. Printed one-per-group the menu ran to 68 lines and stopped
    being readable, which is the opposite of showing what exists.

    Sizes are what a person picking a machine chooses between; the cards
    available AT each size ride along, because that is the pairing no
    single figure could state.
    """
    from molbuilder.jobset.ask import _machine_lines
    many = Domain(name="x", partition="x", qos="public", node_types=[
        {"cores": 48, "nodes": 51, "gpu": {"a100": 4}},
        {"cores": 48, "nodes": 8, "gpu": {"l40": 4}},
        {"cores": 48, "nodes": 2, "gpu": {"a30": 3}},
        {"cores": 128, "nodes": 134},
    ])
    lines = _machine_lines(many, cores=64)
    assert len(lines) == 2, f"one line per SIZE, got {len(lines)}: {lines}"
    small = [ln for ln in lines if "48 cores" in ln][0]
    assert "x61 node(s)" in small, "node counts must be summed across groups"
    for card in ("a100", "a30", "l40"):
        assert card in small, f"{card} was dropped when its group merged"
    assert "(too small)" in small
    assert "(too small)" not in [ln for ln in lines if "128" in ln][0]


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


# --------------------------------------------------------------------- #
#  the maximum core range, and why the count rides with it               #
# --------------------------------------------------------------------- #

def test_the_range_is_what_the_largest_ask_runs_from_to():
    """User's name for it, 2026-08-27: *maximum core range*. Each machine
    has a maximum -- its own core count -- and a queue holding several has
    a range across them."""
    from molbuilder.jobset.ask import core_range
    assert core_range(_domains()["htc"]) == "48-128"


def test_a_queue_of_one_machine_shows_a_NUMBER_not_a_range():
    """*They could equal to each other. So it's a tight range.* A range
    whose ends are equal is a number, and printing `128-128` would invite
    the reader to look for a difference that is not there."""
    from molbuilder.jobset.ask import core_range
    assert core_range(_domains()["highmem"]) == "128"
    assert core_range(_domains()["lightwork"]) == "64"


def test_the_low_end_is_NOT_a_floor_on_what_you_may_ask():
    """**Why it is not called a minimum** (user caught the name himself).
    *Minimum cores* reads as *you must ask at least this*, and that is
    flatly wrong: measured on Sol, a `-c 4` job gets exactly 4 cores on a
    48-core node. You can always ask for less than a machine has.
    """
    htc = _domains()["htc"]
    assert admits(htc, Request(ranks=4)) == [], \
        "an ask below the smallest machine must not be refused"
    assert admits(htc, Request(ranks=1)) == []


def test_the_fitting_node_COUNT_rides_with_the_range():
    """**The range alone misleads.** Reading `48-128` you would take 128
    for the rare extreme; on `htc` it is 134 of 188 nodes -- the COMMON
    machine, with the 48-core GPU nodes in the minority. So a large CPU ask
    there costs almost nothing in scheduling, which is the opposite of what
    the range implies on its own."""
    from molbuilder.jobset.ask import fits_how_many
    htc = _domains()["htc"]
    assert "137 of 188" in fits_how_many(htc, 64)
    assert "134 of 188" in fits_how_many(htc, 128)
    # NO leading arrow: `->` already means *this queue is refused* one
    # indent out, and two arrows at two indents saying different things is
    # a table you have to decode.  Only visible by reading the rendering.
    assert not fits_how_many(htc, 64).startswith("->")


def test_an_ask_nothing_can_hold_says_none_rather_than_a_percentage():
    from molbuilder.jobset.ask import fits_how_many
    assert "none" in fits_how_many(_domains()["htc"], 256)


def test_nothing_is_printed_when_every_machine_fits():
    """A line saying *all of them* on every row is noise. It appears only
    when there is something to disclose."""
    from molbuilder.jobset.ask import fits_how_many
    assert fits_how_many(_domains()["htc"], 48) == ""
    assert fits_how_many(_domains()["htc"], None) == ""
    assert fits_how_many(_domains()["highmem"], 128) == ""


def test_the_queues_differ_and_the_table_shows_by_how_much():
    """The decision this exists to support: the same 64-core ask lands on
    94% of `general` and 72% of `htc`. Neither number is on the record as
    a scalar; both fall out of the machines."""
    from molbuilder.jobset.ask import fits_how_many
    d = _domains()
    assert "94%" in fits_how_many(d["general"], 64)
    assert "72%" in fits_how_many(d["htc"], 64)


def test_an_old_record_shows_its_one_number_and_no_fit_line():
    """R3 again: a record with no `node_types` states a range of one thing
    it knows, and claims nothing about node counts it never measured."""
    from molbuilder.jobset.ask import core_range, fits_how_many
    old = Domain(name="htc", partition="htc", qos="public", max_cores=48)
    assert core_range(old) == "48"
    assert fits_how_many(old, 32) == ""


# --------------------------------------------------------------------- #
#  the machine card a person picks from                                  #
# --------------------------------------------------------------------- #

def test_the_range_has_ONE_spelling():
    """`ask`'s queue table and the browser's machine card must not write a
    range two ways. The arithmetic lives in `scheduler.quantities`, beside
    `human_wall`, because it answers the same kind of question: how is this
    measurement stated to a person."""
    from molbuilder.scheduler.quantities import core_range as q_range
    from molbuilder.jobset.ask import core_range as ask_range
    htc = _domains()["htc"]
    assert ask_range(htc) == q_range([48, 64, 128]) == "48-128"


def test_an_older_record_still_reads_its_one_figure():
    """**R3, and the reason this fix is safe to ship before anyone
    re-probes.** Every record on disk predates `node_types`; the card must
    fall back to what it does have rather than going blank."""
    from molbuilder.scheduler.quantities import core_range, machine_sizes
    from molbuilder.scheduler.record import Domain
    old = [Domain(name="htc", partition="htc", qos="public", max_cores=48)]
    assert machine_sizes(old) == []
    assert core_range(machine_sizes(old)) == ""


def test_the_card_stops_showing_one_arbitrary_nodes_cores():
    """**Caught in the browser, 2026-08-27.** The machine picker read
    "sol · slurm · 64 cores" — `sockets x cores_per_socket`, which is
    whichever node `sinfo` printed first — for a cluster whose machines are
    48, 64 AND 128. A person choosing a machine was shown a number no
    partition guarantees."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/scheduler/record.py").read_text()
    block = src[src.index("bits = [env.scheduler"):]
    block = block[:block.index("_mem =")]
    assert "if spread:" in block, "the range is not preferred over the node"
    assert "elif total:" in block, "the older-record fallback is gone"
