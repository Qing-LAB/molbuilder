"""WHICH card a request names, compared against what a queue stocks.

`execution/scheduler.md` R2 (admission is total over the named limits) and
R3's second half (the ceiling is among the machines that offer what was
asked).  § 7 has shown ``Request(..., gpus=2, gpu_type="a100", ...)`` in
the caller's view since the contract was written; admission got the field
on 2026-08-30.

**What its absence cost.**  A benchmark prepped on ASU Sol baked
``--gres=gpu:a100.40gb:4`` -- the card on whatever node the probe had run
on -- and placement sent it to `public`, which stocks a100, a100.20gb and
a30 and no a100.40gb anywhere.  Every declared limit but this one was
compared, so the group was admitted here and refused by the scheduler
(*Requested node configuration is not available*) after the group ahead of
it had already gone out.

The types are NOT interchangeable and the suffix is not decoration: Sol
registers `a100` on 48-core nodes (52 of them in `public`) and
`a100.40gb` on 64-core nodes (four, and none in `public`) -- disjoint
groups, different machines.  ``--gpus``' own help calls the MIG slices
"separate askable types, not a smaller ask of the same one".
"""
from __future__ import annotations

import pytest

from molbuilder.scheduler.admit import Request, admits
from molbuilder.scheduler.place import Unplaceable, place
from molbuilder.scheduler.record import Domain


def _dom(name, **kw):
    return Domain.from_row({"name": name, "partition": name,
                            "qos": "public", **kw})


#: Sol's `public`, as its probe writes it: 107 standard 128-core nodes and
#: 52 GPU nodes of 48 cores, and the a100.20gb MIG rows that made a
#: count-only check admit any 4-device ask.
PUBLIC = _dom("public", max_time="7-00:00:00", max_cores=128,
              gpu={"a100": 4, "a30": 3, "a100.20gb": 16},
              node_types=[
                  {"cores": 128, "nodes": 107, "mem_gb": 503.5},
                  {"cores": 48, "nodes": 52, "gpu": {"a100": 4}},
                  {"cores": 48, "nodes": 5, "gpu": {"a30": 3}},
                  {"cores": 48, "nodes": 2, "gpu": {"a100.20gb": 16}},
              ])

#: Sol's `general`: the only long queue stocking a100.40gb, and it stocks
#: it on FOUR 64-core nodes while holding a 128-core node with an h200.
GENERAL = _dom("general", max_time="14-00:00:00", max_cores=128,
               gpu={"a100": 4, "a100.40gb": 4, "h200": 1},
               node_types=[
                   {"cores": 128, "nodes": 61, "mem_gb": 503.2},
                   {"cores": 128, "nodes": 1, "gpu": {"h200": 1}},
                   {"cores": 64, "nodes": 4, "gpu": {"a100.40gb": 4}},
                   {"cores": 48, "nodes": 9, "gpu": {"a100": 4}},
               ])


class TestTheCardIsCompared:

    def test_a_queue_that_does_not_stock_the_card_refuses_it(self):
        """THE BUG, in one line.  The count fit (public offers 16
        a100.20gb slices) and the card did not exist."""
        why = admits(PUBLIC, Request(ranks=48, cpus_per_task=1, gpus=4,
                                     gpu_type="a100.40gb"))
        assert why, ("public stocks no a100.40gb; admitting this is what "
                     "sent an unrunnable sbatch to the scheduler")
        assert "a100.40gb" in why[0]
        # R4/R10 -- the refusal names what IS here, so the way out is in
        # the message rather than in `sinfo`.
        assert "a100" in why[0] and "a30" in why[0]

    def test_the_card_it_does_stock_is_admitted(self):
        assert admits(PUBLIC, Request(ranks=48, cpus_per_task=1, gpus=4,
                                      gpu_type="a100")) == []

    def test_naming_no_card_names_nothing_to_refuse(self):
        """R7/R3: a caller that does not know the type asks without one,
        and an unstated field never bars."""
        assert admits(PUBLIC, Request(ranks=48, cpus_per_task=1,
                                      gpus=4)) == []

    def test_a_terse_row_never_bars_on_a_card(self):
        """R3.  Plenty of records describe a queue without enumerating its
        gres; silence is not a claim to stock nothing."""
        terse = _dom("terse", max_time="7-00:00:00", gpu_partition="gpu")
        assert admits(terse, Request(ranks=8, cpus_per_task=1, gpus=1,
                                     gpu_type="a100.40gb")) == []


class TestTheCeilingIsAmongMachinesWithThatCard:

    def test_the_widest_node_bearing_the_named_card_governs(self):
        """`general` holds a 128-core node with an h200 and four 64-core
        nodes with a100.40gb.  A 128-rank a100.40gb trial could only land
        on the h200 node, which carries no a100.40gb at all."""
        why = admits(GENERAL, Request(ranks=128, cpus_per_task=1, gpus=4,
                                      gpu_type="a100.40gb"))
        assert why and "64" in why[0], why
        assert "a100.40gb" in why[0]
        # ...and the same shape at 48 ranks fits those very nodes.
        assert admits(GENERAL, Request(ranks=48, cpus_per_task=1, gpus=4,
                                       gpu_type="a100.40gb")) == []

    def test_a_missing_card_states_no_core_ceiling_for_it(self):
        """A queue with no such node must not report a ceiling *with* that
        card -- "public's largest machine with a100.40gb has 48" describes
        a machine that does not exist.  One true reason, not two."""
        why = admits(PUBLIC, Request(ranks=128, cpus_per_task=1, gpus=4,
                                     gpu_type="a100.40gb"))
        assert not any("largest machine with a100.40gb" in w for w in why), \
            why
        assert any("offers" in w for w in why)

    def test_naming_a_card_never_LOOSENS_the_core_ceiling(self):
        """A record that does not say which nodes hold its devices is
        SILENT about that, and narrowing silence by type yields silence --
        so the wider answer must stand, not vanish.

        Caught in review, 2026-08-30, in this very change: the "no machine
        here carries that card" short-circuit fired on an empty
        device-bearing list too, which is every record with no
        ``node_types`` -- all of them before 2026-08-27, and every
        hand-declared row.  Naming a card then REMOVED the ceiling, and a
        4096-rank trial was admitted on a 48-core queue.
        """
        big = Request(ranks=4096, cpus_per_task=1, gpus=2, gpu_type="a100")
        # (a) max_cores only -- the pre-node_types record shape.
        old = _dom("old", max_cores=48, gpu={"a100": 4})
        assert admits(old, big), "the max_cores ceiling must still bar it"
        # (b) node_types listed, but none says which machines hold devices.
        quiet = _dom("quiet", max_cores=64, gpu={"a100": 4},
                     node_types=[{"cores": 64, "nodes": 10}])
        assert admits(quiet, big), "the widest-node ceiling must still bar it"
        # And naming the card must never admit MORE than not naming it.
        untyped = Request(ranks=4096, cpus_per_task=1, gpus=2)
        for row in (old, quiet):
            assert len(admits(row, big)) >= len(admits(row, untyped)), row.name

    def test_the_largest_node_group_is_the_one_named(self):
        """R10 names what would fit, and a queue listing a 1-node group
        and a 51-node group of the same width must not report the one."""
        row = _dom("htc", max_cores=128, gpu={"a100": 4},
                   node_types=[{"cores": 48, "nodes": 1,
                                "gpu": {"a100": 2}},
                               {"cores": 48, "nodes": 51,
                                "gpu": {"a100": 4}}])
        why = admits(row, Request(ranks=128, cpus_per_task=1, gpus=2,
                                  gpu_type="a100"))
        assert why and "51 node(s) of 48" in why[0], why


class TestTheCountIsOfTheCardAsked:

    def test_a_richer_other_type_does_not_answer_for_this_one(self):
        """`public` offers 16 a100.20gb slices and 3 a30 per node.  Read
        as one number, the 16 admitted every 4-device ask."""
        why = admits(PUBLIC, Request(ranks=12, cpus_per_task=1, gpus=4,
                                     gpu_type="a30"))
        assert why and "3" in why[0], why
        assert "a30" in why[0]

    def test_the_count_of_the_named_card_is_enough(self):
        assert admits(PUBLIC, Request(ranks=12, cpus_per_task=1, gpus=16,
                                      gpu_type="a100.20gb")) == []


class TestPlacementRoutesByTheCard:

    def test_the_queue_without_the_card_is_not_chosen(self):
        """THE SUBMISSION THAT FAILED, placed again.  48 ranks, four
        a100.40gb, seven days: `public` has the wall and not the card,
        `general` has both."""
        req = Request(ranks=48, cpus_per_task=1, gpus=4,
                      gpu_type="a100.40gb", walltime_s=7 * 24 * 3600)
        placed = place([PUBLIC, GENERAL], req, prefer_gpu=True)
        assert placed.name == "general"

    def test_asking_for_the_card_public_stocks_reaches_public(self):
        req = Request(ranks=48, cpus_per_task=1, gpus=4, gpu_type="a100",
                      walltime_s=7 * 24 * 3600)
        assert place([PUBLIC, GENERAL], req, prefer_gpu=True).name \
            in ("public", "general")

    def test_no_queue_stocks_it_and_the_refusal_says_so(self):
        req = Request(ranks=8, cpus_per_task=1, gpus=1, gpu_type="mi300x",
                      walltime_s=3600)
        with pytest.raises(Unplaceable) as e:
            place([PUBLIC, GENERAL], req, prefer_gpu=True)
        assert all("mi300x" in r for r in e.value.reasons)


class TestTheGresStringIsReadByOneDoor:

    def test_the_type_and_the_count_come_from_the_same_reader(self):
        """`_gres_count` split on the last colon, so a trailing ``mps:400``
        was read as the device count.  Both now go through
        `quantities.parse_gres`, which the contract already owns."""
        from molbuilder.jobset.submit import _gres_count, _gres_type
        assert _gres_type("gpu:a100.40gb:4") == "a100.40gb"
        assert _gres_count("gpu:a100.40gb:4") == 4
        assert _gres_count("gpu:a100:4,mps:400") == 4
        assert _gres_type("gpu:a100:4,mps:400") == "a100"
        assert _gres_type("gpu:4") == "gpu"        # untyped
        assert (_gres_type(""), _gres_count("")) == (None, 0)


class TestTheGridFitCheckReportsHonestly:
    """`_cells_this_machine_holds` -- caught in review, 2026-08-30."""

    def test_a_cell_no_queue_could_take_is_never_reported_as_kept(
            self, monkeypatch):
        """The kept/crossed split is on the REASONS, so a cell offered to
        an empty pool -- a GPU cell on a cluster with no gpu-capable queue
        -- must carry one.  With no reason it read as *this machine can
        hold it*, which is the opposite of the truth."""
        import molbuilder.runtime_config as rc
        from molbuilder.jobset._cli import _cells_this_machine_holds
        cpu_only = [_dom("cpuonly", max_cores=128,
                         node_types=[{"cores": 128, "nodes": 10}])]
        monkeypatch.setattr(rc, "get_routing", lambda **k: cpu_only)
        out = _cells_this_machine_holds(
            ".", [(True, (2, 4, 1)), (False, (0, 4, 1))], "a100")
        gpu_cell, cpu_cell = out[0], out[1]
        assert gpu_cell[3], "a GPU cell with nowhere to go must be crossed out"
        assert "gpu-capable" in gpu_cell[3][0]
        assert not cpu_cell[3] and cpu_cell[2] == ("cpuonly",)

    def test_a_count_refusal_outranks_a_wrong_queue_one(self):
        """`_rank_reasons` demotes *"gaudi offers hl225"* -- which only says
        the wrong queue was asked.  It keyed on the word "offers", which the
        COUNT refusal also uses, so the one naming the number to change was
        demoted with it."""
        from molbuilder.jobset._cli import _rank_reasons
        first = _rank_reasons([
            "needs a100 but gaudi offers hl225",
            "needs 4 a30 GPUs but public offers at most 3 a30"])[0]
        assert "at most 3" in first

    def test_a_stated_card_survives_an_unrelated_probe_failure(
            self, monkeypatch, tmp_path):
        """`_gpu_type_for_bench` asks two questions -- *what did you state*
        and *what did this box measure* -- and the second raises on its own
        account: on a workstation carrying named targets `machine_for` asks
        which machine was meant.  Sharing one try block discarded the
        stated answer over a failure that had nothing to do with it."""
        import molbuilder.runtime_config as rc
        import molbuilder.scheduler as sched
        from molbuilder.jobset._cli import _gpu_type_for_bench

        monkeypatch.setattr(rc, "get_scheduler",
                            lambda **k: {"gpu": {"default_type": "a100"}})

        def _ambiguous(*a, **k):
            raise RuntimeError("several machines could be meant")
        monkeypatch.setattr(sched, "machine_for", _ambiguous)

        topo = type("T", (), {"gpu_type": "a100.40gb"})()
        assert _gpu_type_for_bench(tmp_path, topo) == "a100", (
            "the person stated a100; a probe failure must not silently "
            "hand the bench the card this node happens to carry")
