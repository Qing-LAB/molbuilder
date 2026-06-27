"""Tests for ``molbuilder bench generate`` -- the CPU-only + GPU-only
benchmark bundle generator (molbuilder/bench/generate.py)."""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from molbuilder.bench.generate import (
    generate_bench_bundle, render_gpu_sweep_helper, transform_fdf,
)

# A compact-but-complete SIESTA fdf: enough for the run-wrapper to parse
# NumberOfAtoms + species, and carrying the directives the transform flips.
_INPUT_FDF = """\
SystemName          src
SystemLabel         src
NumberOfAtoms       4
PAO.BasisSize       TZP
MeshCutoff          300 Ry
LatticeConstant     1.0 Ang
%block LatticeVectors
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
%endblock LatticeVectors
%block ChemicalSpeciesLabel
1 6 C
%endblock ChemicalSpeciesLabel
%block AtomicCoordinatesAndAtomicSpecies
0.0 0.0 0.0 1
0.1 0.0 0.0 1
0.0 0.1 0.0 1
0.0 0.0 0.1 1
%endblock AtomicCoordinatesAndAtomicSpecies
MD.NumCGsteps       200
BlockSize           64
DM.UseSaveDM        .true.
Diag.ELPA.GPU       .true.
"""

_SCHEDULER_CONFIG = {
    "script_generation": {"preamble": "module load mamba",
                          "activation": "source activate"},
    "scheduler": {
        "kind": "slurm",
        "directives": {"partition": "public", "qos": "public",
                       "export": "NONE"},
        "gpu": {"partition": "public", "default_type": "a100",
                "exclusive": True, "mem": "64G"},
        "defaults": {"time": "0-04:00:00", "cpus_per_task": 8, "mem": None},
        "mem_model": {"node_mem_gb": 500, "safety": 1.3, "extra_gb": 0},
    },
}


# --------------------------------------------------------------------- #
#  transform_fdf                                                        #
# --------------------------------------------------------------------- #


def test_transform_cpu_strips_gpu_directives():
    out = transform_fdf(_INPUT_FDF, label="job-cpu", gpu=False,
                        block_size=8, max_scf=5)
    assert re.search(r"^SystemName\s+job-cpu", out, re.MULTILINE)
    assert re.search(r"^SystemLabel\s+job-cpu", out, re.MULTILINE)
    assert re.search(r"^MaxSCFIterations\s+5", out, re.MULTILINE)
    assert re.search(r"^DM\.UseSaveDM\s+\.false\.", out, re.MULTILINE)
    assert re.search(r"^BlockSize\s+8", out, re.MULTILINE)
    # Capped run must exit 0 (not abort -> SLURM FAILED).
    assert re.search(r"^SCF\.MustConverge\s+\.false\.", out, re.MULTILINE)
    # CPU bundle is plain diagon -- NO ELPA/GPU directives.
    assert "Diag.ELPA.GPU" not in out
    assert "Diag.Algorithm" not in out


def test_transform_normalizes_variant_spelled_directives():
    # SIESTA treats . - _ as interchangeable and labels as case-
    # insensitive.  A legacy-spelled input must be replaced IN PLACE, not
    # duplicated, and GPU directives must be stripped for the CPU bundle
    # regardless of spelling (audit 2026-06-27 B-2).
    src = ("SystemName x\nNumberOfAtoms 1\n"
           "DM-UseSaveDM .true.\nDiag-ELPA-GPU .true.\n"
           "max_scf_iterations 99\n")
    cpu = transform_fdf(src, label="job-cpu", gpu=False,
                        block_size=8, max_scf=5)
    # No conflicting duplicate left behind, original variant gone.
    assert "DM-UseSaveDM .true." not in cpu
    assert cpu.count(".true.") == 0          # both flipped/stripped
    assert "Diag-ELPA-GPU" not in cpu and "Diag.ELPA.GPU" not in cpu
    # The variant MaxSCFIterations was replaced in place, not appended.
    assert len(re.findall(r"(?im)^\s*max[._-]?scf", cpu)) == 1
    assert re.search(r"(?im)^MaxSCFIterations\s+5", cpu)


def test_transform_gpu_adds_elpa_directives():
    out = transform_fdf(_INPUT_FDF, label="job-gpu", gpu=True,
                        block_size=256, max_scf=5)
    assert re.search(r"^SystemName\s+job-gpu", out, re.MULTILINE)
    assert re.search(r"^BlockSize\s+256", out, re.MULTILINE)
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE", out, re.MULTILINE)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.", out, re.MULTILINE)
    assert re.search(r"^DM\.UseSaveDM\s+\.false\.", out, re.MULTILINE)


def test_transform_appends_missing_directives():
    # An fdf with none of the flipped keys still gets them all.
    minimal = "SystemName x\nNumberOfAtoms 1\n"
    out = transform_fdf(minimal, label="job-gpu", gpu=True,
                        block_size=128, max_scf=3)
    assert re.search(r"^MaxSCFIterations\s+3", out, re.MULTILINE)
    assert re.search(r"^BlockSize\s+128", out, re.MULTILINE)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.", out, re.MULTILINE)


# --------------------------------------------------------------------- #
#  GPU sweep helper                                                     #
# --------------------------------------------------------------------- #


def test_sweep_helper_matrix_and_calculator(tmp_path):
    helper = tmp_path / "job-gpu-sweep.sh"
    helper.write_text(render_gpu_sweep_helper(gpus_per_node=4,
                                              cores_per_socket=24))
    helper.chmod(helper.stat().st_mode | stat.S_IEXEC)

    # Valid bash.
    subprocess.run(["bash", "-n", str(helper)], check=True)

    # No-args matrix: a single-GPU K=4 line with -n 4 -c 6.
    matrix = subprocess.run(["bash", str(helper)], capture_output=True,
                            text=True, check=True).stdout
    assert "sbatch --gres=gpu:a100:1 -n 4 -c 6 job-gpu.sbatch" in matrix
    # K=16 single-GPU is in the sweep with c=1 (OMP off).
    assert "sbatch --gres=gpu:a100:1 -n 16 -c 1 job-gpu.sbatch" in matrix
    # multi-GPU lines carry the no-NCCL / no-gpu-bind caveat.
    assert "do NOT add --gpu-bind" in matrix

    # Calculator form: G=2 K=4 -> -n 8 -c 6.
    one = subprocess.run(["bash", str(helper), "2", "4"],
                         capture_output=True, text=True, check=True).stdout
    assert "--gres=gpu:a100:2 -n 8 -c 6" in one
    # K=16 dual-GPU -> -n 32 -c 1.
    k16 = subprocess.run(["bash", str(helper), "2", "16"],
                         capture_output=True, text=True, check=True).stdout
    assert "--gres=gpu:a100:2 -n 32 -c 1" in k16


# --------------------------------------------------------------------- #
#  generate_bench_bundle (full, with a scheduler configured)            #
# --------------------------------------------------------------------- #


def _make_src(tmp_path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "input.fdf").write_text(_INPUT_FDF)
    (src / "C.psml").write_text("<psml></psml>")   # presence is enough
    return src / "input.fdf"


def _make_out_with_config(tmp_path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    (out / ".molbuilder.json").write_text(json.dumps(_SCHEDULER_CONFIG))
    return out


def test_generate_bundle_emits_both_jobs(tmp_path):
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)

    out_dir, written = generate_bench_bundle(
        fdf, out, cpu_np=64, gpu_gpus=1, gpu_k=4,
        gpus_per_node=4, cores_per_socket=24)

    names = {p.name for p in written}
    for expect in ("job-cpu.fdf", "job-gpu.fdf", "job-cpu.run.sh",
                   "job-gpu.run.sh", "job-cpu.sbatch", "job-gpu.sbatch",
                   "job-gpu-sweep.sh", "README.md", "C.psml"):
        assert expect in names, f"missing {expect}"
        assert (out_dir / expect).is_file()

    # GPU fdf routes to GPU; CPU fdf does not.
    assert "Diag.ELPA.GPU" in (out_dir / "job-gpu.fdf").read_text()
    assert "Diag.ELPA.GPU" not in (out_dir / "job-cpu.fdf").read_text()


def test_generate_cpu_sbatch_has_estimated_mem(tmp_path):
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)
    out_dir, _ = generate_bench_bundle(fdf, out, cpu_np=64)
    cpu_sbatch = (out_dir / "job-cpu.sbatch").read_text()
    # The system-aware estimator wrote a concrete --mem (not the null
    # default) and annotated it.
    assert re.search(r"^#SBATCH --mem=\d+G", cpu_sbatch, re.MULTILINE)
    assert "auto-estimated" in cpu_sbatch


def test_generate_gpu_sbatch_has_gres_and_ranks(tmp_path):
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)
    out_dir, _ = generate_bench_bundle(fdf, out, gpu_gpus=1, gpu_k=4,
                                       cores_per_socket=24)
    gpu_sbatch = (out_dir / "job-gpu.sbatch").read_text()
    assert "--gres=gpu:a100:1" in gpu_sbatch
    assert re.search(r"^#SBATCH -n 4", gpu_sbatch, re.MULTILINE)   # K*G
    assert re.search(r"^#SBATCH -c 6", gpu_sbatch, re.MULTILINE)   # 24/K


def test_generate_rejects_non_fdf(tmp_path):
    bad = tmp_path / "input.txt"
    bad.write_text("not an fdf")
    with pytest.raises(ValueError, match="must be a .fdf"):
        generate_bench_bundle(bad, tmp_path / "out")


def test_generate_ships_prep_lib_verbatim(tmp_path):
    import molbuilder.bench as _benchpkg
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)
    out_dir, written = generate_bench_bundle(fdf, out)

    src = Path(_benchpkg.__file__).parent
    # the 6 stdlib modules are copied byte-for-byte into mbbench/
    for m in ("environment", "adapters", "result", "prep", "summarize",
              "prep_run"):
        shipped = out_dir / "mbbench" / f"{m}.py"
        assert shipped.is_file()
        assert shipped.read_bytes() == (src / f"{m}.py").read_bytes()
    assert (out_dir / "mbbench" / "__init__.py").is_file()
    # executable shims for each on-target driver
    for shim in ("prep-bench", "bench-summarize", "prep-run"):
        p = out_dir / shim
        assert p.is_file() and (p.stat().st_mode & 0o111)


def test_shipped_prep_lib_runs_with_no_molbuilder(tmp_path):
    # The headline guarantee: the bundle's prep-bench runs on a target with
    # NO molbuilder importable.  Run the shim in a clean env from the bundle
    # dir; if any shipped module secretly needed molbuilder, import would
    # fail (molbuilder is not pip-installed -- it lives in the repo).
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)
    out_dir, _ = generate_bench_bundle(fdf, out)

    env = {"PATH": os.environ.get("PATH", ""),
           "HOME": str(tmp_path)}            # NO PYTHONPATH -> no repo/molbuilder
    # sanity: molbuilder is indeed unreachable in this clean env
    chk = subprocess.run(
        [sys.executable, "-c", "import molbuilder"],
        cwd=str(tmp_path), env=env, capture_output=True)
    assert chk.returncode != 0, "test invalid: molbuilder importable here"

    r = subprocess.run(
        [sys.executable, "prep-bench", "--scheduler", "workstation",
         "--cores-per-socket", "8", "--gpus-per-node", "1",
         "--gpu-type", "a100"],
        cwd=str(out_dir), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out_dir / "environment.json").is_file()
    sweep = (out_dir / "job-gpu-sweep.sh").read_text()
    assert "K values = 1,2,4,8" in sweep      # divisors of 8
