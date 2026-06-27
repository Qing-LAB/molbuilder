"""Tests for the prep-bench driver (molbuilder/bench/prep.py)."""
from __future__ import annotations

import json
import subprocess

import pytest

from molbuilder.bench import environment as env_mod
from molbuilder.bench import prep
from molbuilder.bench.environment import Environment


@pytest.fixture(autouse=True)
def _no_subprocess(monkeypatch):
    # Keep detection hermetic: no real scontrol/lscpu/nvidia-smi.  Tests
    # drive topology via overrides + scheduler_override.
    monkeypatch.setattr(env_mod, "_run", lambda *a, **k: None)


def test_run_prep_bench_writes_environment_and_sweep(tmp_path):
    env, written = prep.run_prep_bench(
        tmp_path,
        overrides={"cores_per_socket": 24, "gpus_per_node": 4,
                   "gpu_type": "a100"},
        scheduler_override="slurm",
        now_iso="2026-06-27T00:00:00Z")

    names = {p.name for p in written}
    assert names == {"environment.json", "job-gpu-sweep.sh"}

    # environment.json is valid + round-trips
    envp = tmp_path / "environment.json"
    doc = json.loads(envp.read_text())
    assert doc["schema"] == "molbuilder/environment@1"
    back = Environment.from_dict(doc)
    assert back.scheduler == "slurm"
    assert back.topology.cores_per_socket == 24
    assert back.source["topology"] == "flag"          # overrides used

    # the sweep is topology-sized, valid bash, executable
    sweep = tmp_path / "job-gpu-sweep.sh"
    assert sweep.stat().st_mode & 0o111               # +x
    text = sweep.read_text()
    assert "K values = 1,2,3,4,6,8,12,24" in text     # divisors of 24
    assert "sbatch --gres=gpu:a100:4 -n" in text      # 4 GPUs detected
    r = subprocess.run(["bash", "-n", str(sweep)], capture_output=True)
    assert r.returncode == 0, r.stderr


def test_run_prep_bench_workstation(tmp_path):
    env, _ = prep.run_prep_bench(
        tmp_path,
        overrides={"cores_per_socket": 8, "gpus_per_node": 1,
                   "gpu_type": "rtx"},
        scheduler_override="workstation",
        now_iso="2026-06-27T00:00:00Z")
    assert env.scheduler == "workstation"
    text = (tmp_path / "job-gpu-sweep.sh").read_text()
    assert "SEQUENTIALLY" in text                     # workstation mode
    assert "&& sbatch" not in text                    # direct launch


def test_run_prep_bench_creates_out_dir(tmp_path):
    out = tmp_path / "bundle"                          # does not exist yet
    _, written = prep.run_prep_bench(
        out, overrides={"cores_per_socket": 4, "gpus_per_node": 1},
        scheduler_override="workstation", now_iso="t")
    assert out.is_dir()
    assert (out / "environment.json").is_file()


def test_overrides_from_drops_none():
    assert prep._overrides_from(24, None, None) == {"cores_per_socket": 24}
    assert prep._overrides_from(None, None, None) is None


def test_main_standalone(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(env_mod, "_run", lambda *a, **k: None)
    rc = prep.main(["--out", str(tmp_path), "--scheduler", "workstation",
                    "--cores-per-socket", "16", "--gpus-per-node", "2",
                    "--gpu-type", "a100"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "prep-bench: detected target" in out
    assert "scheduler : workstation" in out
    assert (tmp_path / "environment.json").is_file()
    assert (tmp_path / "job-gpu-sweep.sh").is_file()


def test_utc_now_iso_format():
    s = prep.utc_now_iso()
    assert s.endswith("Z") and "T" in s and len(s) == 20
