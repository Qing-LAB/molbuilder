"""A GPU job's admission: the queue has GPUs, and one of its nodes holds as
many as were asked -- that is the whole GPU check.

PINS: ``docs/execution/scheduler.md`` R2a (user, 2026-10-01: *"the only thing
you check is the gpu dependent task is scheduled to the group/domain that
actually claimed to have it, and has the capacity as requested (gpu_number).
that's it"*; *"we never claimed any card type"*) and R3's device half (a GPU
job's core ceiling is the widest machine that HAS GPUs, and silence never
bars).  No card is compared: a GPU ask is a count, and which card a node
carries is the machine's business.

API-LEVEL, ON A MEASURED FIXTURE: the queues are ASU Sol's `public` and
`general` as their probe wrote them (2026-08-30), and admission is asked
directly -- the road cannot put a chosen queue menu in front of it.

PREVENTS: a GPU job placed on a queue with no GPUs; one admitted where no node
holds as many as it asks; its core ceiling taken from machines without GPUs,
or loosened by a record that does not say which nodes hold them.  *(This file
was `test_gres_type_admission.py` until 2026-10-01 and pinned a card
comparison; the card is gone.)*
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
#: GPU nodes of 48 cores, the most GPUs on one node being the 16 MIG slices.
PUBLIC = _dom("public", max_time="7-00:00:00", max_cores=128,
              gpu={"a100": 4, "a30": 3, "a100.20gb": 16},
              node_types=[
                  {"cores": 128, "nodes": 107, "mem_gb": 503.5},
                  {"cores": 48, "nodes": 52, "gpu": {"a100": 4}},
                  {"cores": 48, "nodes": 5, "gpu": {"a30": 3}},
                  {"cores": 48, "nodes": 2, "gpu": {"a100.20gb": 16}},
              ])

#: A queue with no GPUs at all.
CPU_ONLY = _dom("htc", max_time="0-04:00:00", max_cores=128,
                node_types=[{"cores": 128, "nodes": 10}])


def test_a_gpu_job_goes_only_to_a_queue_that_has_gpus():
    """Offered a CPU queue and a GPU queue, a GPU job is placed on the GPU
    one; offered only the CPU queue, it is refused, saying so."""
    req = Request(ranks=8, cpus_per_task=1, gpus=1, walltime_s=3600)
    assert place([CPU_ONLY, PUBLIC], req, prefer_gpu=True).name == "public"
    with pytest.raises(Unplaceable) as e:
        place([CPU_ONLY], req, prefer_gpu=True)
    assert [r.limit for r in e.value.reasons] == ["no_queue"]


def test_a_node_must_hold_as_many_gpus_as_asked():
    """The most GPUs one node of `public` holds is 16: sixteen is admitted,
    seventeen refused, naming the sixteen -- whatever the cards."""
    assert admits(PUBLIC, Request(ranks=8, cpus_per_task=1, gpus=16)) == []
    why = admits(PUBLIC, Request(ranks=8, cpus_per_task=1, gpus=17))
    assert [(r.limit, r.allowed) for r in why] == [("gpus", 16)], why


def test_a_queue_that_lists_no_inventory_never_bars_on_the_count():
    """R3.  Plenty of records describe a queue without enumerating its
    GPUs; silence is not a claim to hold none."""
    terse = _dom("terse", max_time="7-00:00:00", gpu_partition="gpu")
    assert admits(terse, Request(ranks=8, cpus_per_task=1, gpus=8)) == []


def test_a_gpu_jobs_core_ceiling_is_the_widest_node_with_gpus():
    """`public`'s widest machine has 128 cores and no GPU; its GPU nodes
    have 48.  A 64-rank GPU job could land on none of them -- refused,
    naming the 48 of the larger GPU group."""
    why = admits(PUBLIC, Request(ranks=64, cpus_per_task=1, gpus=2))
    assert [(r.limit, r.allowed) for r in why] == [("cores", 48)], why
    assert "52 node(s) of 48" in why[0].note, why[0].note


def test_asking_for_a_gpu_never_loosens_the_core_ceiling():
    """A record that does not say which nodes hold its GPUs is SILENT about
    it, so the wider ceiling stands rather than vanishing -- on the record
    shape before `node_types` (max_cores only) and on one whose machine list
    names no GPU."""
    big = Request(ranks=4096, cpus_per_task=1, gpus=2)
    old = _dom("old", max_cores=48, gpu={"a100": 4})
    quiet = _dom("quiet", max_cores=64, gpu={"a100": 4},
                 node_types=[{"cores": 64, "nodes": 10}])
    for row in (old, quiet):
        assert [r.limit for r in admits(row, big)] == ["cores"], row.name


def test_the_gpu_count_is_read_by_one_reader():
    """`_gres_count` split on the last colon, so a trailing ``mps:400`` was
    read as the device count; it reads through `quantities.parse_gres`."""
    from molbuilder.jobset.submit import _gres_count
    assert _gres_count("gpu:4") == 4
    assert _gres_count("gpu:a100:4,mps:400") == 4
    assert _gres_count("") == 0


def test_a_gpu_cell_with_no_gpu_queue_is_crossed_out(monkeypatch):
    """The bench's cell check: a GPU cell offered to a machine whose queues
    have no GPUs carries the refusal, never reads as kept; a CPU cell fits."""
    import molbuilder.runtime_config as rc
    from molbuilder.jobset.prep_inputs import _cells_this_machine_holds
    monkeypatch.setattr(rc, "get_routing", lambda **k: [CPU_ONLY])
    gpu_cell, cpu_cell = _cells_this_machine_holds(
        ".", [(True, (2, 4, 1)), (False, (0, 4, 1))])
    assert [r.limit for r in gpu_cell[3]] == ["no_queue"]
    assert not cpu_cell[3] and cpu_cell[2] == ("htc",)
