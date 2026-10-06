"""FIELD TEST -- a machine record probed ON the target machine, held to the
record's contract (`docs/process/testing.md` § 0, tier 2).

Goal: molbuilder reads what its probe wrote on a real machine -- a cluster's
queues, its policy caps, its nodes -- the way every verb will read it.
Contract: `configuration.md` M-2 (one shape; a fact not detected is null,
never a guess), `execution/scheduler.md` R13/R14 (a question asked is
answered, a value or null, in every row it was asked of).

Nothing here is scheduler text: the record is the probe's, made on the
machine (`molbuilder jobset probe --write`, there), and these checks assert
only what the contract says of any record -- never what one site's SLURM
prints.  Run by the person, never in the basic suite:

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from molbuilder.scheduler.admit import domain_ceiling_s
from molbuilder.scheduler.record import read_environment

RECORD = Path(os.environ.get("MOLBUILDER_FIELD_RECORD", "")).expanduser()


@pytest.fixture(scope="module")
def raw():
    """The record as the probe wrote it."""
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def env():
    """The record as every verb reads it -- the one reader."""
    got = read_environment(RECORD)
    assert got is not None, f"{RECORD} does not read as a machine record"
    return got


def test_the_record_reads_and_states_its_scheduler(env):
    assert env.scheduler in ("slurm", "workstation")


def test_each_machine_fact_is_a_number_or_unknown(env):
    """M-2: a fact the probe could not detect is null -- never a guess,
    never a zero standing for 'unknown'."""
    t = env.topology
    for name in ("sockets", "cores_per_socket", "threads_per_core"):
        v = getattr(t, name)
        assert v is None or (isinstance(v, int) and v > 0), (name, v)
    g = t.gpus_per_node
    assert g is None or (isinstance(g, int) and g >= 0), ("gpus_per_node", g)


def test_a_cluster_lists_its_queues_each_named_whole(env):
    """A cluster's record lists the queues this account may use, each with
    its name, partition and QoS -- the three a job is sent with."""
    if env.scheduler != "slurm":
        pytest.skip("a workstation lists no queues")
    assert env.domains, "the probe found no queue this account may use"
    for d in env.domains:
        assert d.name and d.partition and d.qos, d


def test_every_wall_a_queue_states_reads(env):
    """A queue's wall, when stated, is one the scheduler takes."""
    for d in env.domains or ():
        if d.max_time is not None:
            assert domain_ceiling_s(d) is not None, (d.name, d.max_time)


def test_the_answers_of_one_question_arrive_together(raw):
    """R13: the per-node core cap and the per-core memory default come from
    one `scontrol show partition` block -- a row that carries one carries
    the other, a value or null (asked; none stated)."""
    for row in raw.get("domains") or []:
        asked = "max_cpus_per_node" in row
        assert asked == ("default_mem_per_core_gb" in row), row["name"]
