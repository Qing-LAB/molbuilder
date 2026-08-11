"""Tests for the prep-run driver (molbuilder/bench/prep_run.py)."""
from __future__ import annotations

import json
import subprocess

import pytest

from molbuilder import environment as env_mod
from molbuilder.bench import prep_run


@pytest.fixture(autouse=True)
def _no_subprocess(monkeypatch):
    monkeypatch.setattr(env_mod, "_run", lambda *a, **k: None)


def _write_bench_result(tmp_path, choice):
    p = tmp_path / "bench-result.json"
    p.write_text(json.dumps({
        "schema": "molbuilder/bench-result@1",
        "environment": {"schema": "molbuilder/environment@1",
                        "scheduler": "slurm"},     # where it was measured
        "system": {"engine": "siesta", "n_atoms": 444},
        "points": [], "choice": choice, "recommend": {},
    }))
    return p


def test_prep_run_reresolves_choice_locally(tmp_path):
    # Winner from a Sol benchmark (gpu K=8); prep-run on a DIFFERENT
    # machine (24 cores/socket here) re-resolves c = 24//8 = 3.
    brp = _write_bench_result(
        tmp_path, {"engine": "gpu",
                   "knobs": {"gpus": 1, "ranks_per_gpu": 8,
                             "cores_per_rank": 99}})  # stale bench c
    env, choice, out_path = prep_run.run_prep_run(
        brp, script_base="prod", scheduler_override="slurm",
        overrides={"cores_per_socket": 24, "gpus_per_node": 4,
                   "gpu_type": "a100"},
        now_iso="t")

    assert out_path == tmp_path / "run-production.sh"
    assert out_path.stat().st_mode & 0o111            # executable
    text = out_path.read_text()
    assert "sbatch --gres=gpu:a100:1 -n 8 -c 3 prod.sbatch" in text
    assert "re-resolved to 3" in text                 # 24//8, not the 99
    r = subprocess.run(["bash", "-n", str(out_path)], capture_output=True)
    assert r.returncode == 0, r.stderr


def test_prep_run_workstation_direct_launch(tmp_path):
    brp = _write_bench_result(
        tmp_path, {"engine": "gpu", "knobs": {"gpus": 1, "ranks_per_gpu": 4}})
    env, choice, out_path = prep_run.run_prep_run(
        brp, script_base="prod", scheduler_override="workstation",
        overrides={"cores_per_socket": 8, "gpus_per_node": 1,
                   "gpu_type": "rtx"}, now_iso="t")
    text = out_path.read_text()
    assert "&& sbatch" not in text
    assert "MB_NP=4" in text          # the var the wrapper honours (not MOLBUILDER_MPI_NP)
    assert "./prod.run.sh" in text


def test_prep_run_no_choice_raises(tmp_path):
    brp = _write_bench_result(tmp_path, {})           # no winner
    with pytest.raises(ValueError, match="no 'choice'"):
        prep_run.run_prep_run(brp, scheduler_override="workstation")


def test_prep_run_rejects_unsafe_script_base(tmp_path):
    brp = _write_bench_result(
        tmp_path, {"engine": "cpu", "knobs": {"ranks": 8}})
    with pytest.raises(ValueError, match="unsafe script_base"):
        prep_run.run_prep_run(brp, script_base="x; rm -rf /",
                              scheduler_override="slurm")


def test_run_prep_run_core(tmp_path, monkeypatch):
    # The on-target shim calls `molbuilder bench prep-run`, which calls this
    # core (execution/job-system.md § 7); there is no standalone entry to test.
    monkeypatch.setattr(env_mod, "_run", lambda *a, **k: None)
    brp = _write_bench_result(
        tmp_path, {"engine": "gpu", "knobs": {"gpus": 1, "ranks_per_gpu": 8},
                   "rationale": "gpu-k8 fastest"})
    env, choice, out_path = prep_run.run_prep_run(
        brp, script_base="prod", scheduler_override="workstation",
        overrides={"cores_per_socket": 16, "gpus_per_node": 1,
                   "gpu_type": "a100"})
    assert choice["engine"] == "gpu"
    assert (tmp_path / "run-production.sh").is_file()
    summary = prep_run._summary(env, choice, out_path)
    assert "prep-run: production run formatted" in summary
