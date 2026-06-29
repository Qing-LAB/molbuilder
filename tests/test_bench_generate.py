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
    generate_bench_bundle, render_sweep_placeholder, transform_fdf,
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


def test_transform_cpu_uses_elpa_without_gpu_flag():
    # Apples-to-apples: the CPU point uses the SAME solver as GPU
    # (ELPA-1STAGE) but NOT the CUDA toggle -- only hardware differs.
    out = transform_fdf(_INPUT_FDF, label="job-cpu", gpu=False,
                        block_size=8, max_scf=5)
    assert re.search(r"^SystemName\s+job-cpu", out, re.MULTILINE)
    assert re.search(r"^MaxSCFIterations\s+5", out, re.MULTILINE)
    assert re.search(r"^DM\.UseSaveDM\s+\.false\.", out, re.MULTILINE)
    assert re.search(r"^BlockSize\s+8", out, re.MULTILINE)
    assert re.search(r"^SCF\.MustConverge\s+\.false\.", out, re.MULTILINE)
    # Same eigensolver as the GPU point ...
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE", out, re.MULTILINE)
    # ... but NOT the CUDA toggle (that's the only difference).
    assert "Diag.ELPA.GPU" not in out


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


def test_sweep_placeholder_reminds_to_prep(tmp_path):
    # The baked sweep is a placeholder: running it before prep-bench must
    # remind the user (exit non-zero), not silently produce a broken sweep.
    helper = tmp_path / "job-gpu-sweep.sh"
    helper.write_text(render_sweep_placeholder())
    helper.chmod(helper.stat().st_mode | stat.S_IEXEC)
    subprocess.run(["bash", "-n", str(helper)], check=True)        # valid bash
    r = subprocess.run(["bash", str(helper)], capture_output=True, text=True)
    assert r.returncode != 0                                       # reminds
    assert "Run ./prep-bench first" in r.stderr


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


def test_generate_emits_portable_inputs_not_wrappers(tmp_path):
    # generate writes the TARGET-NEUTRAL bundle: fdf inputs + manifest +
    # sweep placeholder + README + pseudos.  The run wrappers are baked at
    # PREP, on the target (job-execution.md § 7) -- so they are absent here.
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)

    out_dir, written = generate_bench_bundle(
        fdf, out, cpu_np=64, gpu_gpus=1, gpu_k=4,
        gpus_per_node=4, cores_per_socket=24)

    names = {p.name for p in written}
    for expect in ("job-cpu.fdf", "job-gpu.fdf", "bench-manifest.json",
                   "job-gpu-sweep.sh", "README.md", "C.psml"):
        assert expect in names, f"missing {expect}"
        assert (out_dir / expect).is_file()
    # wrappers are NOT baked at generate (portable bundle)
    assert not (out_dir / "job-cpu.run.sh").exists()
    assert not (out_dir / "job-gpu.run.sh").exists()
    assert not (out_dir / "job-cpu.sbatch").exists()

    # GPU fdf routes to GPU; CPU fdf does not.
    assert "Diag.ELPA.GPU" in (out_dir / "job-gpu.fdf").read_text()
    assert "Diag.ELPA.GPU" not in (out_dir / "job-cpu.fdf").read_text()

    # the manifest carries the chosen knobs verbatim (§ 7.3)
    man = json.loads((out_dir / "bench-manifest.json").read_text())
    assert man["schema"] == "molbuilder/bench-manifest@1"
    # engine-neutral schema (multi-engine readiness): top-level engine +
    # per-job `script` (NOT a SIESTA-flavored `fdf` key)
    assert man["engine"] == "siesta"
    assert man["jobs"]["cpu"]["script"] == "job-cpu.fdf"
    assert man["jobs"]["gpu"]["script"] == "job-gpu.fdf"
    assert "fdf" not in man["jobs"]["cpu"]
    assert man["jobs"]["cpu"]["mpi_np"] == 64
    assert man["jobs"]["cpu"]["cpus_per_task"] == 1
    assert man["jobs"]["gpu"]["gpu_gpus"] == 1
    assert man["jobs"]["gpu"]["gpu_k"] == 4


#  Activation resolution -- the hole that dead-ended a fresh user:        #
#  generate must NOT require a hand-authored .molbuilder.json.             #


def _no_server_config(monkeypatch):
    """Force the server-wide config layer empty so these tests exercise
    the project / auto-detect paths regardless of the host machine."""
    import molbuilder.runtime_config as rc
    monkeypatch.setattr(rc, "_read_server_wide", lambda: {})


def test_generate_cold_start_is_target_neutral(tmp_path, monkeypatch):
    # No config, no flags -> generate must NOT autodetect the host's conda
    # and must NOT write a .molbuilder.json.  Activation is decided at PREP,
    # on the target (job-execution.md § 7); generate stays target-neutral.
    import molbuilder.runtime_config as rc
    _no_server_config(monkeypatch)
    # Even if a host conda is detectable, generate must not consult it.
    monkeypatch.setattr(rc, "detect_conda_activation",
                        lambda: {"activation": "conda activate",
                                 "preamble": 'source "/x/conda.sh"'})
    fdf = _make_src(tmp_path)
    out = tmp_path / "out"; out.mkdir()           # NO .molbuilder.json
    out_dir, _ = generate_bench_bundle(fdf, out)
    assert not (out_dir / "job-cpu.run.sh").exists()
    assert not (out_dir / ".molbuilder.json").exists()
    assert (out_dir / "bench-manifest.json").is_file()


def test_generate_explicit_activation_persisted_for_prep(tmp_path, monkeypatch):
    # An HPC target: generate persists the explicit activation/preamble into
    # .molbuilder.json for `bench prep` to consume on the cluster.  It does
    # NOT bake a wrapper here (that happens at prep).
    _no_server_config(monkeypatch)
    fdf = _make_src(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    out_dir, _ = generate_bench_bundle(
        fdf, out, activation="source activate", preamble="module load mamba")
    cfg = json.loads((out_dir / ".molbuilder.json").read_text())
    assert cfg["script_generation"]["activation"] == "source activate"
    assert cfg["script_generation"]["preamble"] == "module load mamba"
    assert not (out_dir / "job-cpu.run.sh").exists()


def test_generate_preamble_only_persists_preamble(tmp_path, monkeypatch):
    # --preamble WITHOUT --activation: persist the preamble; leave activation
    # unset so prep resolves it (workstation autodetect / HPC explicit).  No
    # host-conda autodetect at generate.
    _no_server_config(monkeypatch)
    fdf = _make_src(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    out_dir, _ = generate_bench_bundle(fdf, out, preamble="module load mamba")
    cfg = json.loads((out_dir / ".molbuilder.json").read_text())
    assert cfg["script_generation"]["preamble"] == "module load mamba"
    assert "activation" not in cfg["script_generation"]
    assert not (out_dir / "job-cpu.run.sh").exists()


def test_generate_help_documents_molbuilder_json_explicitly():
    # config file must be explicit in --help (where it lives + an example)
    from click.testing import CliRunner
    from molbuilder.bench._cli import bench_group
    out = CliRunner().invoke(bench_group, ["generate", "-h"]).output
    assert ".molbuilder.json" in out
    assert "WHERE" in out and "WHAT" in out          # location + content
    assert '"script_generation"' in out               # a concrete example
    assert "config.md" in out                          # pointer to full ref


def test_detect_conda_activation_finds_local_conda():
    # Smoke: the tests run inside a conda env, so detection must work.
    from molbuilder.runtime_config import detect_conda_activation
    got = detect_conda_activation()
    assert got is not None
    assert got["activation"] == "conda activate"
    assert got["preamble"].startswith("source ")


# NOTE: the sbatch CONTENT tests (estimated --mem, -c 1 / -n for CPU,
# gpu --gres / -n=K*G / -c=cores//K, --exclusive) moved to
# tests/test_bench_prep.py::TestBakeTargetWrappers -- the wrappers are now
# baked at PREP, on the target, from the detected topology (job-execution.md
# § 7).  They are no longer produced by `generate`.


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


def test_shipped_summarize_and_prep_run_shims_standalone(tmp_path):
    # The bench-summarize + prep-run shims also run with no molbuilder.
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)
    out_dir, _ = generate_bench_bundle(fdf, out)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}

    # fake one swept, completed GPU point
    d = out_dir / "point-G1K8"
    d.mkdir()
    (d / "job-gpu-run0.scf-timing.log").write_text(
        "1000.0 1 scf:1\n1010.0 2 scf:2\n1020.0 3 scf:3\n1030.0 4 scf:4\n")
    (d / "job-gpu.monitor.log").write_text(
        "[t] [UTIL-SUMMARY] cpu mean=40% (..); gpu0 sm mean=91% (..) "
        "-> GPU-bound (..)\n")
    (d / "job-gpu-run0.out").write_text(">> End of run: completed\n")

    br = out_dir / "bench-result.json"
    r1 = subprocess.run(
        [sys.executable, "bench-summarize", "--bundle", str(out_dir),
         "--out", str(br)],
        cwd=str(out_dir), env=env, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    assert br.is_file()

    r2 = subprocess.run(
        [sys.executable, "prep-run", "--bench-result", str(br),
         "--script-base", "prod", "--scheduler", "workstation",
         "--cores-per-socket", "16", "--gpus-per-node", "1",
         "--gpu-type", "a100", "--out", str(out_dir / "run-production.sh")],
        cwd=str(out_dir), env=env, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert (out_dir / "run-production.sh").is_file()
