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
    policy = parse_scontrol_partitions(_SCONTROL)
    for p in parts:
        pol = policy.get(p.name)
        if pol is not None:
            p.def_mem_per_cpu_mb = pol.def_mem_per_cpu_mb
            p.max_cpus_per_node = pol.max_cpus_per_node
    rows, _notes = derive_domains(parts, {}, {"public"})
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
        {}, {"public"})[0]}
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
        parts, {}, {"public"})[0]}
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
        parts, {}, {"public"})[0]}
    assert "default_mem_per_core_gb" not in rows["htc"]


# --------------------------------------------------------------------- #
#  the per-core default — what you get for asking nothing                #
# --------------------------------------------------------------------- #

def test_the_per_core_default_is_read_from_scontrol():
    """`sinfo` has no format code for it, which is why this is a second
    command rather than a wider one."""
    got = parse_scontrol_partitions(_SCONTROL)
    assert {n: pol.def_mem_per_cpu_mb for n, pol in got.items()} == {
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
    assert parse_scontrol_partitions(body)["p"].def_mem_per_cpu_mb \
        is None, why


def test_a_row_carries_no_per_core_default_when_scontrol_is_silent():
    parts = parse_sinfo(_SINFO)          # def_mem_per_cpu_mb left unset
    rows = {r["name"]: r for r in derive_domains(
        parts, {}, {"public"})[0]}
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
    policy = parse_scontrol_partitions(_SCONTROL)
    for p in parts:
        pol = policy.get(p.name)
        if pol is not None:
            p.def_mem_per_cpu_mb = pol.def_mem_per_cpu_mb
            p.max_cpus_per_node = pol.max_cpus_per_node
    rows, _notes = derive_domains(parts, {}, {"public"})
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
    rows, _ = derive_domains(parts, {}, {"public"})
    emitted = {k for row in rows for k in row}
    unknown = sorted(emitted - set(Domain._KNOWN))
    assert not unknown, (
        f"the probe writes {unknown}, which `Domain` does not declare -- add "
        f"the field (and say what it is for), or stop writing the column")


# --------------------------------------------------------------------- #
#  the policy ceilings — what you may ASK, beside what the node HAS      #
# --------------------------------------------------------------------- #

def test_the_policy_cap_rides_the_row_and_admission_reads_it():
    """R13, end to end: `lightwork`'s suspected 8-core cap could be neither
    confirmed nor denied because the probe never asked.  Now `scontrol`'s
    ``MaxCPUsPerNode`` lands on the row and `admits` compares BOTH ceilings
    -- the smaller governs, so 128-core nodes do not launder a 9-core ask
    past an 8-core policy."""
    from molbuilder.scheduler import Domain
    from molbuilder.scheduler.admit import Request, admits
    body = ("PartitionName=lightwork\n"
            "   DefMemPerCPU=2048 MaxCPUsPerNode=8\n")
    parts = parse_sinfo("lightwork|1-00:00:00|3|(null)|128|515000\n")
    pol = parse_scontrol_partitions(body)["lightwork"]
    assert pol.max_cpus_per_node == 8
    parts[0].def_mem_per_cpu_mb = pol.def_mem_per_cpu_mb
    parts[0].max_cpus_per_node = pol.max_cpus_per_node
    parts[0].policy_queried = True
    rows, _ = derive_domains(parts, {}, {"public"})
    row = rows[0]
    assert row["max_cpus_per_node"] == 8
    assert row["max_cores"] == 128, "the hardware ceiling must survive"
    d = Domain(**row)
    assert admits(d, Request(ranks=8)) == []
    why = admits(d, Request(ranks=9))
    assert why and "8" in why[0] and "policy" in why[0], (
        f"a 9-core ask slid past an 8-core policy on 128-core nodes: {why}")


def test_a_partition_with_no_stated_cap_bars_nothing_new():
    """R3: UNLIMITED and absence both say *no policy stated*, and the
    hardware ceiling alone governs -- the probe asking a new question must
    not make old records or permissive partitions stricter."""
    for body in ("PartitionName=p\n   MaxCPUsPerNode=UNLIMITED\n",
                 "PartitionName=p\n   DefMemPerCPU=2048\n"):
        assert parse_scontrol_partitions(body)["p"].max_cpus_per_node \
            is None, body


def test_asked_and_unstated_writes_NULL_never_absence():
    """Absent-vs-null (checkpointing.md S3), found the day a fresh Sol
    record arrived with the key missing and nothing could say whether the
    probe had asked the new question or predated it.  A record that was
    ASKED writes the key -- null when no cap is stated -- so `lightwork`'s
    suspected cap can actually be settled by reading the record."""
    parts = parse_sinfo("lightwork|1-00:00:00|3|(null)|128|515000\n")
    pol = parse_scontrol_partitions(
        "PartitionName=lightwork\n   MaxCPUsPerNode=UNLIMITED\n")["lightwork"]
    parts[0].max_cpus_per_node = pol.max_cpus_per_node
    parts[0].policy_queried = True
    from molbuilder.scheduler.probe import QosLimit
    rows, _ = derive_domains(parts, {"public": QosLimit(None, None, None)},
                             {"public"})
    row = rows[0]
    assert "max_cpus_per_node" in row and row["max_cpus_per_node"] is None, (
        "asked-and-unstated must be a null in the record, not a missing key")
    assert "max_cpus_per_job" in row and row["max_cpus_per_job"] is None

    # and NEVER asked stays absent -- the old records' honest shape
    parts2 = parse_sinfo("lightwork|1-00:00:00|3|(null)|128|515000\n")
    rows2, _ = derive_domains(parts2, {}, {"public"})
    assert "max_cpus_per_node" not in rows2[0]
    assert "max_cpus_per_job" not in rows2[0]


def test_the_qos_cap_lands_on_the_row_too():
    """R13's other half: a QoS ``MaxTRESPerJob=cpu=N`` caps the job
    wherever it lands, so it rides the (partition, qos) row."""
    from molbuilder.scheduler.probe import QosLimit
    parts = parse_sinfo("htc|4:00:00|10|(null)|128|515000\n")
    rows, _ = derive_domains(
        parts, {"public": QosLimit(None, None, 96)}, {"public"})
    assert rows[0]["max_cpus_per_job"] == 96


def test_the_null_survives_the_trip_to_disk_and_back():
    """The 2026-08-28 leak, at the door it leaked through: `derive` wrote
    the null, `Domain.to_row` dropped every ``None``, and the record a
    REAL Sol probe wrote could not show the question was asked.  The
    tri-state must survive derive -> Domain -> json -> Domain -> row."""
    import json as _json

    from molbuilder.scheduler.probe import QosLimit
    from molbuilder.scheduler.record import Domain

    def _asked_rows():
        parts = parse_sinfo("lightwork|1-00:00:00|3|(null)|128|515000\n")
        pol = parse_scontrol_partitions(
            "PartitionName=lightwork\n   MaxCPUsPerNode=UNLIMITED\n")
        parts[0].max_cpus_per_node = pol["lightwork"].max_cpus_per_node
        parts[0].policy_queried = True
        rows, _ = derive_domains(
            parts, {"public": QosLimit(None, None, None)}, {"public"})
        return rows

    on_disk = _json.loads(_json.dumps(Domain(**_asked_rows()[0]).to_row()))
    for k in ("max_cpus_per_node", "max_cpus_per_job"):
        assert k in on_disk and on_disk[k] is None, (
            f"asked-and-uncapped must land as null, lost at {k}")
    # and a RELOADED record still says asked -- the second write too
    again = Domain.from_row(on_disk).to_row()
    assert again["max_cpus_per_node"] is None
    assert again["max_cpus_per_job"] is None

    # never-asked stays OFF the disk through the same trip
    parts2 = parse_sinfo("lightwork|1-00:00:00|3|(null)|128|515000\n")
    rows2, _ = derive_domains(parts2, {}, {"public"})
    disk2 = Domain(**rows2[0]).to_row()
    assert "max_cpus_per_node" not in disk2, (
        "a question never asked must not be reported as answered")
    assert "max_cpus_per_job" not in disk2


def test_a_numeric_cap_still_rides_the_row_to_disk():
    """The tri-state's third value: a real cap is a number on disk."""
    from molbuilder.scheduler.probe import QosLimit
    from molbuilder.scheduler.record import Domain
    parts = parse_sinfo("htc|4:00:00|10|(null)|128|515000\n")
    rows, _ = derive_domains(
        parts, {"public": QosLimit(None, None, 96)}, {"public"})
    assert Domain(**rows[0]).to_row()["max_cpus_per_job"] == 96


def test_the_consent_question_names_the_field_that_moved():
    """2026-08-28: the probe said *disagrees in 1 place* and showed two
    IDENTICAL name lists -- the change was field-level, the display was
    name-level, and the user was asked to judge the invisible.  When the
    domain SET is unchanged, the question must name the moved fields."""
    from molbuilder.jobset._cli import _domains_shown
    from molbuilder.scheduler.record import Domain

    old = [Domain.from_row({"name": "lightwork", "partition": "lightwork",
                            "qos": "public"})]
    new = [Domain.from_row({"name": "lightwork", "partition": "lightwork",
                            "qos": "public", "max_cpus_per_job": None,
                            "max_cpus_per_node": None})]
    shown_b, shown_p = _domains_shown(old, new)
    assert shown_b == "(same domains)"
    assert "max_cpus_per_job absent -> null" in shown_p
    assert "max_cpus_per_node absent -> null" in shown_p

    # a changed SET still shows the two name lists, as before
    shown_b2, shown_p2 = _domains_shown(old, new + [Domain.from_row(
        {"name": "htc", "partition": "htc", "qos": "public"})])
    assert shown_b2 == ["lightwork"] and shown_p2 == ["lightwork", "htc"]
