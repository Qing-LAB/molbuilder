"""`jobset probe --write` on a machine that HAS a scheduler.

**Why this file exists.** The probe has two paths and only one of them is
reachable from a workstation: without `sinfo` it records topology and stops.
Every test of the memory work exercised the parsers and the row DICTS; nothing
ran the verb down the cluster path, where the rows become `Domain` objects.

So `derive_domains` gained a column, `Domain` did not, and

    TypeError: Domain.__init__() got an unexpected keyword argument
               'default_mem_per_core_gb'

reached a user on the first real run — on Sol, because no machine here could
produce it. A parser test that stops before the object is a loop left open.

The scheduler commands are faked at `record._run`, which is the one door the
probe shells out through, so this is the verb's own code path with the
cluster's answers substituted.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

_SINFO = (
    "htc|4:00:00|40|(null)|128|257000\n"
    "general|7-00:00:00|30|gpu:a100:4|48|515000\n"
    "highmem|2-00:00:00|4|(null)|128|2050000\n"
)
_SCONTROL = """PartitionName=htc
   DefMemPerCPU=2048
PartitionName=general
   DefMemPerCPU=2048
PartitionName=highmem
   DefMemPerCPU=16384
"""
_QOS = "public|||\ndebug|00:15:00||\n"
_ASSOC = "public,debug\n"


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """This box, answering as a login node would."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    (tmp_path / "home").mkdir(exist_ok=True)

    from molbuilder.scheduler import record

    def _fake(cmd, timeout=10.0):
        head = " ".join(cmd[:2])
        if head.startswith("sinfo"):
            return _SINFO
        if head.startswith("scontrol show"):
            return _SCONTROL
        if "qos" in cmd:
            return _QOS
        if "assoc" in cmd:
            return _ASSOC
        return None

    monkeypatch.setattr(record, "_run", _fake)
    monkeypatch.setattr("molbuilder.jobset._cli._run", _fake, raising=False)
    return tmp_path


def _probe(*args):
    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, ["probe", *args])


def test_the_verb_survives_the_cluster_path(cluster):
    """**The regression.**  Not the parsers — the verb, all the way to the
    objects it builds."""
    r = _probe("--write", "--yes")
    assert r.exit_code == 0, r.output + str(r.exception)


def test_a_named_target_is_written_where_machines_says_it_should_be(cluster):
    """`probe --write --name sol` is the command a person runs ON the
    cluster, and the file it writes is the one they copy back."""
    r = _probe("--write", "--yes", "--name", "sol")
    assert r.exit_code == 0, r.output + str(r.exception)
    from molbuilder.scheduler import named_environments
    assert "sol" in named_environments(), r.output


def test_the_written_record_carries_the_memory_facts(cluster):
    """The whole point of measuring them: they have to survive to the file a
    later `prep` reads."""
    _probe("--write", "--yes", "--name", "sol")
    from molbuilder.scheduler import named_environments
    body = json.loads(named_environments()["sol"].read_text())
    rows = {d["name"]: d for d in body["domains"]}
    assert rows["htc"]["max_mem_gb"] == pytest.approx(251.0, abs=0.5)
    assert rows["htc"]["default_mem_per_core_gb"] == pytest.approx(2.0)
    assert rows["highmem"]["max_mem_gb"] > 2000


def test_the_written_record_reads_back_as_objects(cluster):
    """The step that crashed: the file becomes `Domain`s, not just JSON."""
    _probe("--write", "--yes", "--name", "sol")
    from molbuilder.scheduler import named_environments, read_environment
    env = read_environment(named_environments()["sol"])
    assert env is not None and env.domains
    for d in env.domains:
        assert d.name and d.partition and d.qos
    got = {d.name: d.default_mem_per_core_gb for d in env.domains}
    assert got.get("htc") == pytest.approx(2.0)


def test_a_cluster_without_scontrol_says_so_rather_than_guessing(
        tmp_path, monkeypatch):
    """`sinfo` has no format code for the per-core default, so it is a second
    command.  When it is unreachable the rows simply carry no default — *this
    machine does not say*, never zero — and the verb SAYS so, because a
    measurement that quietly did not happen is the thing this whole round was
    about."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    (tmp_path / "home").mkdir(exist_ok=True)
    from molbuilder.scheduler import record

    def _fake(cmd, timeout=10.0):
        if cmd[0] == "sinfo":
            return _SINFO
        if cmd[0] == "scontrol":
            return None                      # not reachable from here
        if "qos" in cmd:
            return _QOS
        if "assoc" in cmd:
            return _ASSOC
        return None

    monkeypatch.setattr(record, "_run", _fake)
    monkeypatch.setattr("molbuilder.jobset._cli._run", _fake, raising=False)
    r = _probe("--write", "--yes", "--name", "sol")
    assert r.exit_code == 0, r.output + str(r.exception)
    assert "scontrol was not reachable" in r.output, (
        "the probe measured no per-core default and did not say so")
    from molbuilder.scheduler import named_environments
    body = json.loads(named_environments()["sol"].read_text())
    for d in body["domains"]:
        assert "default_mem_per_core_gb" not in d, (
            "an unmeasured default was written as a number anyway")
