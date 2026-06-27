"""Tests for the SLURM ``.sbatch`` submission-layer emitter
(``runwrap.render_sbatch`` / ``write_sbatch`` + the ``write_run_wrapper``
wiring).

Authoritative design: docs/protocols/slurm-integration.md
  § 3  two-layer model (header delegates to the unchanged .run.sh)
  § 5  block-by-block header
  § 6  value-source matrix
  § 7.4/7.5.1/8  GPU gating + 1-rank-per-GPU + enforce-binding
  § 10 refuse-to-emit / skip-when-no-scheduler
  § 13 testing strategy (L1/L2)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder import diagnostics, runwrap
from molbuilder.diagnostics import Capabilities
from molbuilder.runwrap import WrapperError, render_sbatch, write_sbatch


_SCHED = {
    "kind": "slurm",
    "directives": {
        "partition": "public", "qos": "public",
        "mail_type": "ALL", "mail_user": "%u@asu.edu", "export": "NONE",
    },
    "gpu": {"partition": "public", "default_type": "a100", "exclusive": True},
    "defaults": {"time": "0-04:00:00", "cpus_per_task": None, "mem": None},
}


@pytest.fixture(autouse=True)
def _caps():
    """Synthetic Capabilities so no real ``conda env list`` runs."""
    diagnostics.set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs=frozenset({"molbuilder-siesta", "molbuilder-siesta-gpu"}),
    ))
    yield


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project dir carrying the asu-sol script_generation + scheduler
    config (project scope), with an isolated HOME so the server-wide
    lookup chain doesn't leak a real ~/.config file."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / ".molbuilder.json").write_text(json.dumps({
        "script_generation": {
            "preamble": "module load mamba/latest",
            "activation": "source activate",
        },
        "scheduler": _SCHED,
    }))
    return tmp_path


# --------------------------------------------------------------------- #
#  render_sbatch -- pure header rendering                               #
# --------------------------------------------------------------------- #


def test_cpu_header_shape(tmp_path):
    fdf = tmp_path / "cpu-np64.fdf"
    fdf.write_text("NumberOfAtoms 444\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=64)
    assert "#SBATCH -J cpu-np64" in txt
    assert "#SBATCH -N 1" in txt
    assert "#SBATCH -n 64" in txt
    assert "#SBATCH -p public" in txt          # NOT general (§ 7.0)
    assert "#SBATCH -q public" in txt
    assert "#SBATCH -t 0-04:00:00" in txt
    assert "#SBATCH -o slurm.%j.out" in txt
    assert "#SBATCH --export=NONE" in txt
    # CPU job: no GPU lines, no exclusive.
    assert "--gres" not in txt
    assert "--exclusive" not in txt
    # Delegation body, unchanged launcher (§ 3).
    assert 'bash cpu-np64.run.sh "$@"' in txt


def test_gpu_header_shape(tmp_path):
    fdf = tmp_path / "gpu-2a100.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=2, cpus_per_task=12,
                        gpu=True, gpu_count=2, exclusive=False)
    assert "#SBATCH -n 2" in txt
    assert "#SBATCH -c 12" in txt
    assert "#SBATCH --gres=gpu:a100:2" in txt
    assert "#SBATCH --gres-flags=enforce-binding" in txt   # § 7.5.1
    # exclusive=False override honoured (benchmark sweep, § 11.2 / D9).
    assert "--exclusive" not in txt


def test_gpu_exclusive_defaults_from_config(tmp_path):
    fdf = tmp_path / "prod.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    # exclusive=None => use scheduler.gpu.exclusive (True for asu-sol).
    txt = render_sbatch(fdf, _SCHED, ntasks=1, gpu=True)
    assert "#SBATCH --exclusive" in txt


def test_gpu_count_defaults_to_ntasks(tmp_path):
    fdf = tmp_path / "g.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=3, gpu=True)  # 1 rank/GPU default
    assert "#SBATCH --gres=gpu:a100:3" in txt


def test_gpu_ranks_independent_of_gpu_count(tmp_path):
    """K ranks sharing FEWER GPUs via MPS: -n is the RANK count, --gres is
    the GPU count -- they are independent (the load-balance model)."""
    fdf = tmp_path / "g.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=8, cpus_per_task=3,
                        gpu=True, gpu_count=1)        # 8 ranks, 1 GPU
    assert "#SBATCH -n 8" in txt                       # ranks, NOT 1
    assert "#SBATCH -c 3" in txt
    assert "#SBATCH --gres=gpu:a100:1" in txt


def test_mem_emitted_when_set(tmp_path):
    fdf = tmp_path / "m.fdf"
    fdf.write_text("x\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=8, mem="120G")
    assert "#SBATCH --mem=120G" in txt


def test_cpus_omitted_when_unset(tmp_path):
    fdf = tmp_path / "c.fdf"
    fdf.write_text("x\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=20)
    assert "#SBATCH -c " not in txt


def test_ntasks_must_be_positive(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("x\n")
    with pytest.raises(WrapperError, match="ntasks"):
        render_sbatch(fdf, _SCHED, ntasks=0)


def test_missing_partition_refuses(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("x\n")
    bad = {"kind": "slurm", "directives": {"qos": "public"}}
    with pytest.raises(WrapperError, match="partition"):
        render_sbatch(fdf, bad, ntasks=4)


def test_bad_gres_rejected(tmp_path):
    fdf = tmp_path / "x.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    with pytest.raises(WrapperError, match="gres"):
        runwrap._parse_gres("a100x2")


@pytest.mark.parametrize("spec,expect", [
    ("gpu:a100:2", ("a100", 2)),
    ("a100:4", ("a100", 4)),
    ("2", (None, 2)),
])
def test_parse_gres_forms(spec, expect):
    assert runwrap._parse_gres(spec) == expect


# --------------------------------------------------------------------- #
#  bash -n validity                                                     #
# --------------------------------------------------------------------- #


def test_rendered_sbatch_is_valid_bash(tmp_path):
    fdf = tmp_path / "gpu-2a100.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    # write_sbatch runs bash -n internally; a malformed header raises.
    p = write_sbatch(fdf, _SCHED, ntasks=2, cpus_per_task=12,
                     gpu=True, gpu_count=2, exclusive=False)
    assert p.name == "gpu-2a100.sbatch"
    assert oct(p.stat().st_mode)[-3:] == "644"


# --------------------------------------------------------------------- #
#  write_run_wrapper wiring (§ 15 B)                                    #
# --------------------------------------------------------------------- #


def test_wrapper_emits_sbatch_when_scheduler_configured(project):
    fdf = project / "cpu-np64.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .false.\n")
    runwrap.write_run_wrapper(fdf, mpi_np=64)
    sbatch = project / "cpu-np64.sbatch"
    assert sbatch.is_file()
    txt = sbatch.read_text()
    assert "#SBATCH -n 64" in txt
    assert "--gres" not in txt          # CPU .fdf -> no GPU lines


def test_wrapper_gpu_fdf_emits_gres(project):
    fdf = project / "gpu.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, mpi_np=2, gres="gpu:a100:2",
                              cpus_per_task=12, exclusive=False)
    txt = (project / "gpu.sbatch").read_text()
    assert "#SBATCH --gres=gpu:a100:2" in txt
    assert "#SBATCH -c 12" in txt
    assert "#SBATCH --gres-flags=enforce-binding" in txt


def test_wrapper_gpu_K_ranks_share_one_gpu(project):
    """The benchmark case: 8 (or 4) ranks share ONE A100 via MPS ->
    -n must be the rank count (8), --gres the GPU count (1)."""
    fdf = project / "gpu-k8.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, mpi_np=8, gres="gpu:a100:1",
                              cpus_per_task=3, exclusive=False)
    txt = (project / "gpu-k8.sbatch").read_text()
    assert "#SBATCH -n 8" in txt                  # ranks (was wrongly 1)
    assert "#SBATCH -c 3" in txt
    assert "#SBATCH --gres=gpu:a100:1" in txt     # one GPU, shared


def test_gpu_fdf_auto_gres_without_cli(project):
    """A .fdf with Diag.ELPA.GPU .true. emits a GPU header even without
    --gres: the GPU count defaults to ONE (ranks share it via MPS) and
    -n is the rank count -- not 1-rank-per-GPU."""
    fdf = project / "auto.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, mpi_np=2)
    txt = (project / "auto.sbatch").read_text()
    assert "#SBATCH --gres=gpu:a100:1" in txt      # default 1 GPU
    assert "#SBATCH -n 2" in txt                    # ranks share it


def test_no_scheduler_no_sbatch(tmp_path, monkeypatch):
    """§ 10: with no scheduler block, only .run.sh is emitted."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / ".molbuilder.json").write_text(json.dumps({
        "script_generation": {"activation": "source activate"},
    }))
    fdf = tmp_path / "local.fdf"
    fdf.write_text("NumberOfAtoms 10\n")
    runwrap.write_run_wrapper(fdf, mpi_np=4)
    assert (tmp_path / "local.run.sh").is_file()
    assert not (tmp_path / "local.sbatch").exists()


def test_emit_sbatch_false_suppresses(project):
    fdf = project / "x.fdf"
    fdf.write_text("NumberOfAtoms 10\n")
    runwrap.write_run_wrapper(fdf, mpi_np=4, emit_sbatch=False)
    assert (project / "x.run.sh").is_file()
    assert not (project / "x.sbatch").exists()
