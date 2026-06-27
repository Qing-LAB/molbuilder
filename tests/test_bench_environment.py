"""Tests for the benchmark environment probe + scheduler adapters
(molbuilder/bench/environment.py, adapters.py)."""
from __future__ import annotations

import pytest

from molbuilder.bench import adapters
from molbuilder.bench import environment as env_mod
from molbuilder.bench.environment import (
    Environment, Site, Topology, _parse_gres, _parse_lscpu,
    _parse_nvidia_smi_l, _parse_scontrol_node, resolve_environment,
)


# --------------------------------------------------------------------- #
#  pure parsers                                                         #
# --------------------------------------------------------------------- #

_SCONTROL = """\
NodeName=gpu01 Arch=x86_64 CoresPerSocket=24
   CPUAlloc=0 CPUTot=48 RealMemory=515558
   Gres=gpu:a100:4(S:0-1)
   Sockets=2 ThreadsPerCore=1 State=IDLE
"""

_LSCPU = """\
Architecture:            x86_64
CPU(s):                  96
Socket(s):               2
Core(s) per socket:      24
Thread(s) per core:      2
NUMA node(s):            8
"""

_NVIDIA_L = """\
GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-aaaa)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-bbbb)
"""


def test_parse_scontrol_node():
    t = _parse_scontrol_node(_SCONTROL)
    assert t.sockets == 2
    assert t.cores_per_socket == 24
    assert t.threads_per_core == 1
    assert t.gpus_per_node == 4
    assert t.gpu_type == "a100"
    assert t.mem_total_gb == pytest.approx(503.5, abs=0.1)


def test_parse_lscpu_numa_per_socket():
    t = _parse_lscpu(_LSCPU)
    assert (t.sockets, t.cores_per_socket, t.threads_per_core) == (2, 24, 2)
    assert t.numa_per_socket == 4          # 8 NUMA / 2 sockets


def test_parse_nvidia_smi_l():
    assert _parse_nvidia_smi_l(_NVIDIA_L) == (2, "a100")
    assert _parse_nvidia_smi_l("") == (None, None)


@pytest.mark.parametrize("gres,expect", [
    ("gpu:a100:4", (4, "a100")),
    ("gpu:a100:4(S:0-1)", (4, "a100")),
    ("gpu:4", (4, None)),
    ("(null)", (None, None)),
    ("", (None, None)),
])
def test_parse_gres(gres, expect):
    assert _parse_gres(gres) == expect


# --------------------------------------------------------------------- #
#  Environment JSON round-trip (the persisted contract, § 5.2)          #
# --------------------------------------------------------------------- #


def test_environment_json_round_trip():
    env = Environment(
        scheduler="slurm",
        topology=Topology(sockets=2, cores_per_socket=24, gpus_per_node=4,
                          gpu_type="a100", mem_total_gb=503.0),
        site=Site(partition="public", qos="public"),
        source={"scheduler": "path:sbatch", "topology": "scontrol",
                "site": "sinfo"},
        detected_at="2026-06-27T18:30:00Z",
    )
    d = env.to_dict()
    assert d["schema"] == env_mod.SCHEMA
    assert d["topology"]["cores_per_socket"] == 24
    # null kept, never omitted (consumers distinguish absent from unknown)
    assert "account" in d["site"] and d["site"]["account"] is None

    back = Environment.from_dict(d)
    assert back.scheduler == "slurm"
    assert back.topology.gpu_type == "a100"
    assert back.site.partition == "public"
    assert back.source["topology"] == "scontrol"


def test_environment_from_dict_rejects_major_mismatch():
    with pytest.raises(ValueError, match="schema mismatch"):
        Environment.from_dict({"schema": "molbuilder/environment@2",
                               "scheduler": "slurm"})


def test_environment_from_dict_tolerates_extra_and_missing_keys():
    env = Environment.from_dict({
        "schema": "molbuilder/environment@1",
        "scheduler": "workstation",
        "topology": {"cores_per_socket": 16, "future_field": "ignored"},
        # site missing entirely
    })
    assert env.topology.cores_per_socket == 16
    assert env.site.partition is None


# --------------------------------------------------------------------- #
#  detection (guarded subprocess; overrides win)                        #
# --------------------------------------------------------------------- #


def test_detect_scheduler_env(monkeypatch):
    monkeypatch.setattr(env_mod.shutil, "which", lambda _x: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert env_mod.detect_scheduler() == ("slurm", "env:SLURM_*")


def test_detect_scheduler_workstation(monkeypatch):
    monkeypatch.setattr(env_mod.shutil, "which", lambda _x: None)
    for k in list(__import__("os").environ):
        if k.startswith("SLURM_"):
            monkeypatch.delenv(k, raising=False)
    assert env_mod.detect_scheduler() == ("workstation", "no-sbatch")


def test_resolve_environment_overrides_win(monkeypatch):
    # Force workstation + stub the probes so the test is hermetic, then
    # confirm declared overrides win and the source is marked 'flag'.
    monkeypatch.setattr(env_mod, "_run", lambda *a, **k: None)
    env = resolve_environment(
        scheduler_override="workstation",
        overrides={"cores_per_socket": 24, "gpus_per_node": 4,
                   "gpu_type": "a100"},
        now_iso="2026-06-27T00:00:00Z",
    )
    assert env.scheduler == "workstation"
    assert env.topology.cores_per_socket == 24
    assert env.topology.gpus_per_node == 4
    assert env.source["scheduler"] == "flag"
    assert "flag" in env.source["topology"]
    assert env.detected_at == "2026-06-27T00:00:00Z"


# --------------------------------------------------------------------- #
#  adapters                                                             #
# --------------------------------------------------------------------- #


def test_divisors():
    assert adapters.divisors(24) == [1, 2, 3, 4, 6, 8, 12, 24]
    assert adapters.divisors(0) == []
    assert adapters.divisors(None) == []


def test_sweep_K_is_socket_divisors():
    a = adapters.SlurmAdapter()
    assert a.sweep_K(Topology(cores_per_socket=24)) == \
        [1, 2, 3, 4, 6, 8, 12, 24]
    # unknown cores -> empty (caller falls back to a declared default)
    assert a.sweep_K(Topology()) == []


def test_get_adapter_selects_by_scheduler():
    slurm = get = adapters.get_adapter(Environment(scheduler="slurm"))
    assert slurm.name == "slurm"
    ws = adapters.get_adapter(Environment(scheduler="workstation"))
    assert ws.name == "workstation"
    with pytest.raises(ValueError, match="no scheduler adapter"):
        adapters.get_adapter(Environment(scheduler="pbs"))


def _sweep(adapter, env, **kw):
    return adapter.format_bench(env, **kw)["job-gpu-sweep.sh"]


def test_format_bench_slurm_emits_sbatch_grid():
    env = Environment(scheduler="slurm",
                      topology=Topology(cores_per_socket=24, gpus_per_node=2,
                                        gpu_type="a100"))
    s = _sweep(adapters.SlurmAdapter(), env)
    # K=8 -> c=24/8=3; single + dual GPU lines present.
    assert "sbatch --gres=gpu:a100:1 -n 8 -c 3 job-gpu.sbatch" in s
    assert "sbatch --gres=gpu:a100:2 -n 16 -c 3 job-gpu.sbatch" in s
    assert "QUEUE IN PARALLEL" in s
    # multi-GPU caveat on G>=2 lines
    assert "do NOT add --gpu-bind" in s


def test_format_bench_workstation_emits_direct_grid():
    env = Environment(scheduler="workstation",
                      topology=Topology(cores_per_socket=24, gpus_per_node=1,
                                        gpu_type="a100"))
    s = _sweep(adapters.WorkstationAdapter(), env)
    # direct launch, no sbatch, picks the GPU + drives the launcher knobs
    assert "sbatch" not in s
    assert ("CUDA_VISIBLE_DEVICES=0 MOLBUILDER_MPI_NP=8 "
            "MOLBUILDER_OMP_NUM_THREADS=3 ./job-gpu.run.sh") in s
    assert "SEQUENTIALLY" in s


def test_format_bench_unknown_cores_uses_fallback_ks():
    env = Environment(scheduler="slurm",
                      topology=Topology(gpus_per_node=1))   # no cores/socket
    s = _sweep(adapters.SlurmAdapter(), env)
    assert "fallback set" in s
    assert "-n 8" in s            # K=8 from the fallback set
    # without cores/socket there is no -c; the launcher defaults it
    assert "-c" not in s.split("\n\n", 1)[1]


def test_format_run_still_pending():
    with pytest.raises(NotImplementedError):
        adapters.SlurmAdapter().format_run(None, None,
                                           Environment(scheduler="slurm"))
