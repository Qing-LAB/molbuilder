"""The two memory facts, measured — `execution/submission.md` § 4.

**Why this exists.** `Domain.max_mem_gb` and `default_mem_per_core_gb` have
been on the record since the row was designed and **neither was ever filled by
any probe**.  So nothing could tell a 128 GB ask anything at all: not that a
queue holds less, not that the number came from a per-core default nobody
chose.  A 64-core job asked for 128 GB because SLURM grants 2 GB a core, and
that number was invisible to every surface that could have shown it.

**They are two different facts and the file keeps them apart.**  The ceiling
says *what a node has*; the per-core default says *what you get when you ask
for nothing*.  Reading one as the other is how a GPU card's memory and a
queue's memory came to be confused elsewhere (`gpu.md` § 1).
"""
from __future__ import annotations

import pytest

from molbuilder.scheduler.probe import (derive_domains, parse_scontrol_partitions,
                                        parse_sinfo)

#: Sol-shaped: htc 128-core/251G, general 48-core GPU/503G, highmem 2 TB.
_SINFO = (
    "htc|4:00:00|40|(null)|128|257000\n"
    "general|7-00:00:00|30|gpu:a100:4|48|515000\n"
    "highmem|2-00:00:00|4|(null)|128|2050000\n"
)

_SCONTROL = """PartitionName=htc
   AllowGroups=ALL DefMemPerCPU=2048 MaxMemPerNode=UNLIMITED
   Nodes=cg[1-40]
PartitionName=general
   DefMemPerCPU=2048
PartitionName=highmem
   DefMemPerCPU=16384
"""


def _rows():
    parts = parse_sinfo(_SINFO)
    defmem = parse_scontrol_partitions(_SCONTROL)
    for p in parts:
        p.def_mem_per_cpu_mb = defmem.get(p.name)
    rows, _notes = derive_domains(parts, {"public": (None, None)}, {"public"})
    return {r["name"]: r for r in rows}


# --------------------------------------------------------------------- #
#  the ceiling — what a node has                                         #
# --------------------------------------------------------------------- #

def test_the_memory_ceiling_is_measured_onto_every_row():
    rows = _rows()
    assert rows["htc"]["max_mem_gb"] == pytest.approx(251.0, abs=0.1)
    assert rows["highmem"]["max_mem_gb"] == pytest.approx(2002.0, abs=1.0)


def test_a_partition_whose_nodes_differ_promises_the_SMALLEST():
    """A ceiling that over-promises is the one that sends a job to a queue it
    does not fit -- the same rule the core count already uses."""
    parts = parse_sinfo("mixed|1:00:00|10|(null)|64|64000\n"
                        "mixed|1:00:00|10|(null)|64|512000\n")
    assert parts[0].mem_mb == 64000


def test_an_older_record_says_nothing_rather_than_zero():
    """R3.  A probe written before `%m` produced five columns; a reader must
    not mistake *did not measure* for *has none*, or every such queue looks
    too small for every job."""
    for text in ("p|1:00:00|4|(null)",              # the 2026-08-21 format
                 "p|1:00:00|4|(null)|128"):         # the pre-08-23 format
        [p] = parse_sinfo(text)
        assert p.mem_mb is None
    rows = {r["name"]: r for r in derive_domains(
        parse_sinfo("p|1:00:00|4|(null)|128"),
        {"public": (None, None)}, {"public"})[0]}
    assert "max_mem_gb" not in rows["p"], (
        "an unmeasured ceiling was written onto the row, so admission would "
        "compare an ask against a number nobody measured")


def test_a_zero_from_sinfo_is_unknown_and_not_a_ceiling_of_zero():
    """**The mutation the first version of this file missed.**

    `sinfo` prints ``0`` for a node group whose memory it cannot report.
    Writing that onto the row as a ceiling would make admission refuse EVERY
    job on that queue -- the loudest possible reading of *we do not know*,
    and the exact inversion R3 forbids.  Caught 2026-08-23 by mutating
    ``if part.mem_mb:`` to ``is not None`` and watching nothing fail.
    """
    parts = parse_sinfo("p|1:00:00|4|(null)|128|0\n")
    rows = {r["name"]: r for r in derive_domains(
        parts, {"public": (None, None)}, {"public"})[0]}
    assert "max_mem_gb" not in rows["p"], (
        "a queue reporting 0 MB was given a ceiling of 0 GB, so every job "
        "would be refused there for needing more memory than nothing")


def test_a_zero_per_core_default_is_unknown_too():
    """Same rule on the other fact: 0 GB per core is not a policy, it is a
    partition that did not answer."""
    parts = parse_sinfo(_SINFO)
    for pt in parts:
        pt.def_mem_per_cpu_mb = 0
    rows = {r["name"]: r for r in derive_domains(
        parts, {"public": (None, None)}, {"public"})[0]}
    assert "default_mem_per_core_gb" not in rows["htc"]


# --------------------------------------------------------------------- #
#  the per-core default — what you get for asking nothing                #
# --------------------------------------------------------------------- #

def test_the_per_core_default_is_read_from_scontrol():
    """`sinfo` has no format code for it, which is why this is a second
    command rather than a wider one."""
    assert parse_scontrol_partitions(_SCONTROL) == {
        "htc": 2048, "general": 2048, "highmem": 16384}


def test_the_number_that_made_a_64_core_job_ask_for_128G():
    """The arithmetic nobody was shown, now checkable."""
    rows = _rows()
    per_core = rows["htc"]["default_mem_per_core_gb"]
    assert per_core == pytest.approx(2.0)
    assert 64 * per_core == pytest.approx(128.0), (
        "this is the 128 G that chose a queue without anyone deciding it")


@pytest.mark.parametrize("body,why", [
    ("PartitionName=p\n   DefMemPerNode=64000\n",
     "a per-NODE default is not a per-core one, and dividing it would "
     "invent a number"),
    ("PartitionName=p\n   DefMemPerCPU=UNLIMITED\n",
     "UNLIMITED states no number"),
    ("PartitionName=p\n   Nodes=n[1-4]\n",
     "a partition that sets neither says nothing"),
])
def test_a_partition_that_states_no_per_core_default_maps_to_none(body, why):
    """R3 again: ``None`` is *this partition does not say*, never zero."""
    assert parse_scontrol_partitions(body) == {"p": None}, why


def test_a_row_carries_no_per_core_default_when_scontrol_is_silent():
    parts = parse_sinfo(_SINFO)          # def_mem_per_cpu_mb left unset
    rows = {r["name"]: r for r in derive_domains(
        parts, {"public": (None, None)}, {"public"})[0]}
    assert "default_mem_per_core_gb" not in rows["htc"]


# --------------------------------------------------------------------- #
#  the two are not one                                                   #
# --------------------------------------------------------------------- #

def test_the_ceiling_and_the_default_are_different_numbers():
    """The confusion this file exists to prevent.  On highmem they differ by
    a factor of 125; a reader that took one for the other would either refuse
    every job or promise a node it does not have."""
    rows = _rows()
    hm = rows["highmem"]
    assert hm["max_mem_gb"] > 2000 and hm["default_mem_per_core_gb"] == 16.0
    assert hm["max_mem_gb"] != hm["default_mem_per_core_gb"]


# --------------------------------------------------------------------- #
#  the row must become an OBJECT — the step this file did not test       #
# --------------------------------------------------------------------- #

def test_every_row_the_probe_builds_constructs_a_Domain():
    """**The gap that shipped a crash.**

    `derive_domains` returns dicts and `jobset probe --write` does
    ``Domain(**row)``.  This file tested the dicts thoroughly and never the
    step between, so adding `default_mem_per_core_gb` to the row without
    adding it to `Domain` passed every test here and died on the first real
    probe:

        TypeError: Domain.__init__() got an unexpected keyword argument
                   'default_mem_per_core_gb'

    Testing the shape a producer emits, without testing that the consumer
    accepts it, is a loop left open -- the same defect class as a field
    declared and never read, one step earlier.
    """
    from molbuilder.scheduler import Domain
    parts = parse_sinfo(_SINFO)
    defmem = parse_scontrol_partitions(_SCONTROL)
    for p in parts:
        p.def_mem_per_cpu_mb = defmem.get(p.name)
    rows, _notes = derive_domains(parts, {"public": (None, None)}, {"public"})
    assert rows, "the fixture produced no rows, so this proves nothing"
    for row in rows:
        d = Domain(**row)                      # exactly what `probe --write` does
        assert d.name and d.partition and d.qos


def test_the_columns_the_probe_writes_are_all_KNOWN_to_the_record():
    """The same loop, stated as a rule rather than as one construction: a
    column the probe emits that `Domain` does not name would land in `extra`
    on read and crash on `Domain(**row)` -- present in the file, invisible to
    the reader, fatal to the writer."""
    from molbuilder.scheduler import Domain
    parts = parse_sinfo(_SINFO)
    for p in parts:
        p.def_mem_per_cpu_mb = 2048
    rows, _ = derive_domains(parts, {"public": (None, None)}, {"public"})
    emitted = {k for row in rows for k in row}
    unknown = sorted(emitted - set(Domain._KNOWN))
    assert not unknown, (
        f"the probe writes {unknown}, which `Domain` does not declare -- add "
        f"the field (and say what it is for), or stop writing the column")
