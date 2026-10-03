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
    """This box, answering as a login node would -- set up the way molbuilder
    sets one up, its molbuilder.json stating `env_init` (`configuration.md`
    § 4: the probe requires it)."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    _states_its_env_init()

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
    return tmp_path


def _states_its_env_init():
    """This machine's molbuilder.json, as `envs init-config` leaves it."""
    from molbuilder.runtime_config import write_config_scope
    write_config_scope({"env_init": {"activation": "source activate",
                                     "preamble": "module load mamba"}})


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


@pytest.mark.parametrize("writer", ["probe", "init-config"])
def test_the_written_record_reads_back_as_objects(cluster, writer):
    """The step that crashed: the file becomes `Domain`s, not just JSON.

    BOTH writers of this machine's record take the cluster path, stamped and
    signed as the probe's: `envs init-config` seeded a cluster as `slurm` with
    no queues and no `detected_at` until 2026-10-02 (R4; `configuration.md`
    M-3, "through the same prober"), and every record said prep wrote it
    (R13)."""
    if writer == "probe":
        r = _probe("--write", "--yes")
    else:
        from molbuilder.envs._cli import envs_group
        r = CliRunner().invoke(envs_group, [
            "init-config", "--activation", "source activate", "--yes"])
    assert r.exit_code == 0, r.output + str(r.exception)
    from molbuilder.scheduler import machine_scope_path, read_environment
    env = read_environment(machine_scope_path())
    assert env is not None and env.domains
    assert env.detected_at and env.tool == "jobset-probe@1"
    for d in env.domains:
        assert d.name and d.partition and d.qos
    got = {d.name: d.default_mem_per_core_gb for d in env.domains}
    assert got.get("htc") == pytest.approx(2.0)


@pytest.mark.parametrize("writer", ["probe", "init-config"])
@pytest.mark.parametrize("unanswered,said", [
    ("scontrol", "scontrol was not reachable"),
    ("MaxSubmitJobsPerUser", "did not accept MaxSubmitJobsPerUser"),
])
def test_a_measurement_that_did_not_happen_is_said(tmp_path, monkeypatch,
                                                   unanswered, said, writer):
    """`sinfo` has no format code for the per-core default, so it is a second
    command; `sacctmgr` rejects a whole query over one unknown column, so the
    submit cap is asked with a fall-back.  When either goes unanswered the
    record simply does not say -- never zero -- and the verb SAYS so, because
    a measurement that quietly did not happen is the thing this whole round
    was about.  (The submit-cap line was appended before `derive_domains`
    reassigned the notes, and was never shown, until 2026-10-02 -- and `envs
    init-config`, seeding through the same queue probe, dropped every note.)"""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    from molbuilder.scheduler import record

    def _fake(cmd, timeout=10.0):
        if any(unanswered in c for c in cmd):
            return None                      # not answered from here
        if cmd[0] == "sinfo":
            return _SINFO
        if cmd[0] == "scontrol":
            return _SCONTROL
        if "qos" in cmd:
            return _QOS
        if "assoc" in cmd:
            return _ASSOC
        return None

    monkeypatch.setattr(record, "_run", _fake)
    if writer == "probe":
        _states_its_env_init()
        r = _probe("--write", "--yes")
    else:
        from molbuilder.envs._cli import envs_group
        r = CliRunner().invoke(envs_group, [
            "init-config", "--activation", "source activate", "--yes"])
    assert r.exit_code == 0, r.output + str(r.exception)
    assert said in r.output, f"{writer} measured nothing there and did not say so"
    if unanswered == "scontrol":
        from molbuilder.scheduler import machine_scope_path
        body = json.loads(machine_scope_path().read_text())
        for d in body["domains"]:
            assert "default_mem_per_core_gb" not in d, (
                "an unmeasured default was written as a number anyway")
