"""Execution mode -- the run-vs-submit launch policy (execution/job-system.md § 5.3 -- which records that this key has no live
contract of its own).

Covers the four moving parts of the feature:
* ``get_execution`` reads + validates ``execution`` from .molbuilder.json
  (and the key SURVIVES ``_normalise``'s top-level allowlist);
* ``resolve_mode`` / ``resolve_launch_adapter`` pick the launch mechanism
  independently of the detected scheduler;
* ``bake_run_bench`` routes the CPU baseline through the SAME adapter as the
  sweep (so CPU + GPU are symmetric -- submit under submit, bash under direct);
* ``render_bench_plan`` states the resolved mode; ``run_prep_bench`` honors it.
"""

import json

import pytest

from molbuilder.bench import adapters
from molbuilder.bench.adapters import (SlurmAdapter, WorkstationAdapter,
                                       resolve_launch_adapter, resolve_mode)
from molbuilder.environment import Environment, Topology
from molbuilder.bench.generate import bake_run_bench, render_bench_plan
from molbuilder.bench.prep import run_prep_bench
from molbuilder.runtime_config import (PROJECT_CONFIG_FILENAME,
                                       RuntimeConfigError, get_execution)


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Isolated cwd + $HOME + XDG for every test in this file.

    ``get_execution`` deep-merges the server-wide scope, which is CWD-first
    with an XDG fallback — so without this the verdicts here depended on the
    developer's repo-root ``molbuilder.json`` (found 2026-08-12; the same
    class as the inert-fixture bug in test_prep_calculation.py)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()


# ---- get_execution: read, default, validate, survive _normalise ------ #

def _write_project(tmp_path, cfg):
    (tmp_path / PROJECT_CONFIG_FILENAME).write_text(json.dumps(cfg))


def test_get_execution_reads_mode_and_submit_via(tmp_path):
    _write_project(tmp_path, {"execution": {"mode": "submit",
                                            "submit_via": "slurm"}})
    out = get_execution(project_dir=tmp_path)
    assert out == {"mode": "submit", "submit_via": "slurm", "domain": None}


def test_get_execution_absent_is_none_mode(tmp_path):
    _write_project(tmp_path, {})
    out = get_execution(project_dir=tmp_path)
    assert out["mode"] is None
    assert out["submit_via"] == "slurm"          # default backend


def test_get_execution_survives_normalise_allowlist(tmp_path):
    # The key must NOT be dropped by _normalise's top-level allowlist
    # (the bug found while wiring this up): a sibling known key present.
    _write_project(tmp_path, {"scheduler": {"kind": "slurm"},
                              "execution": {"mode": "direct"}})
    assert get_execution(project_dir=tmp_path)["mode"] == "direct"


def test_get_execution_bad_mode_raises(tmp_path):
    _write_project(tmp_path, {"execution": {"mode": "fling"}})
    with pytest.raises(RuntimeConfigError, match="execution.mode"):
        get_execution(project_dir=tmp_path)


def test_get_execution_non_object_raises(tmp_path):
    _write_project(tmp_path, {"execution": "submit"})
    with pytest.raises(RuntimeConfigError, match="'execution' must be an"):
        get_execution(project_dir=tmp_path)


# ---- resolve_mode / resolve_launch_adapter --------------------------- #

def test_resolve_mode_explicit_wins():
    assert resolve_mode("submit") == "submit"
    assert resolve_mode("direct") == "direct"


def test_resolve_mode_refuses_unset():
    """Unset is a REFUSAL, never a derivation from the detected scheduler
    (`running-a-job.md` § 5.4 — the mode, not the scheduler, gates
    submission).  Until 2026-08-12 this function DERIVED (slurm → submit)
    and this test pinned the derivation — the one door that disagreed with
    `jobset submit`'s; the environment left the signature with the rule, so
    the derivation cannot quietly return."""
    with pytest.raises(ValueError, match=r"never derived"):
        resolve_mode(None)


def test_resolve_launch_adapter_submit_picks_by_name_not_detection():
    # mode=submit picks SlurmAdapter BY NAME, whatever machine this is --
    # "submit from an interactive shell".  The adapter selection takes no
    # Environment at all since 2026-08-12, which is the stronger form of
    # "independent of detection".
    a, rmode = resolve_launch_adapter(mode="submit", submit_via="slurm")
    assert isinstance(a, SlurmAdapter) and rmode == "submit"


def test_resolve_launch_adapter_direct_picks_workstation():
    a, rmode = resolve_launch_adapter(mode="direct")
    assert isinstance(a, WorkstationAdapter) and rmode == "direct"


def test_resolve_launch_adapter_unknown_submit_via_raises():
    with pytest.raises(ValueError, match="submit_via"):
        resolve_launch_adapter(mode="submit", submit_via="pbs")


# ---- bake_run_bench: CPU baseline follows the adapter ----------------- #

def test_bake_run_bench_submit_uses_sbatch_for_cpu(tmp_path):
    p = bake_run_bench(tmp_path, SlurmAdapter(), cpu_np=64, mode="submit")
    text = p.read_text()
    assert "sbatch -J job-cpu -n 64 job-cpu.sbatch" in text  # CPU submits, named
    assert "bash job-gpu-sweep.sh" in text             # sweep still invoked
    assert "SUBMITTED" in text                          # tail tells the truth


def test_bake_run_bench_direct_uses_bash_for_cpu(tmp_path):
    p = bake_run_bench(tmp_path, WorkstationAdapter(), cpu_np=8, mode="direct")
    text = p.read_text()
    assert "MB_NP=8 ./job-cpu.run.sh" in text           # CPU runs in-shell
    assert "sbatch" not in text


# ---- render_bench_plan states the mode ------------------------------- #

_MANIFEST = {"engine": "siesta", "description": "d", "measured": "m",
             "points": {"cpu": {"mpi_np": 8, "solver": "CPU"},
                        "gpu": {"gpu_k": 2, "gpus": 1, "solver": "GPU"}}}


def test_plan_launch_line_submit():
    env = Environment(scheduler="slurm",
                      topology=Topology(sockets=2, cores_per_socket=10,
                                        gpus_per_node=1, gpu_type="a100"))
    plan = render_bench_plan(env, _MANIFEST, [1, 2], mode="submit")
    assert "Launch: submit via slurm" in plan


def test_plan_launch_line_direct():
    env = Environment(scheduler="workstation",
                      topology=Topology(sockets=2, cores_per_socket=10,
                                        gpus_per_node=1, gpu_type="rtx"))
    plan = render_bench_plan(env, _MANIFEST, [1, 2], mode="direct")
    assert "Launch: direct bash" in plan


# ---- run_prep_bench honors mode in the baked sweep ------------------- #

def test_prep_submit_mode_bakes_sbatch_sweep_on_workstation(tmp_path):
    # Force a workstation topology via override but mode=submit: the sweep
    # must use sbatch per point regardless of the detected scheduler.
    env, _ = run_prep_bench(
        tmp_path, scheduler_override="workstation",
        overrides={"cores_per_socket": 10, "gpus_per_node": 1,
                   "gpu_type": "a100"},
        ks=[1], cs=[1], mode="submit", submit_via="slurm")
    sweep = (tmp_path / "job-gpu-sweep.sh").read_text()
    assert "--gres=gpu:a100:1" in sweep
    assert "sbatch ${MB_GPU_PQ:-} -J job-gpu-G1K1C1" in sweep   # § 4.3/§ 4.4


def test_prep_direct_mode_bakes_bash_sweep(tmp_path):
    run_prep_bench(
        tmp_path, scheduler_override="slurm",
        overrides={"cores_per_socket": 10, "gpus_per_node": 1,
                   "gpu_type": "a100"},
        ks=[1], cs=[1], mode="direct")
    sweep = (tmp_path / "job-gpu-sweep.sh").read_text()
    # The launch line (inside `( cd point... && ... )`) is direct, not sbatch.
    # ("sbatch" still appears in the _mb_point symlink helper -- not a launch.)
    assert "&& CUDA_VISIBLE_DEVICES=0 MB_NP=1" in sweep
    assert "&& sbatch" not in sweep
