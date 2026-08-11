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


def test_transform_cpu_uses_elpa_with_gpu_flag_explicitly_false():
    # Apples-to-apples: the CPU point uses the SAME solver as GPU
    # (ELPA-1STAGE) but with the CUDA toggle EXPLICITLY OFF -- only
    # hardware differs.  The flag must be set .false., NOT omitted: the
    # ELPA-CUDA build defaults to GPU, so an omitted flag let the CPU
    # baseline initialize CUDA and crash (Sol job 57852377, 2026-06-29).
    out = transform_fdf(_INPUT_FDF, label="job-cpu", gpu=False,
                        block_size=8, max_scf=5)
    assert re.search(r"^SystemName\s+job-cpu", out, re.MULTILINE)
    assert re.search(r"^MaxSCFIterations\s+5", out, re.MULTILINE)
    assert re.search(r"^DM\.UseSaveDM\s+\.false\.", out, re.MULTILINE)
    assert re.search(r"^BlockSize\s+8", out, re.MULTILINE)
    assert re.search(r"^SCF\.MustConverge\s+\.false\.", out, re.MULTILINE)
    # Same eigensolver as the GPU point ...
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE", out, re.MULTILINE)
    # ... with the CUDA toggle EXPLICITLY .false. (that's the difference).
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.false\.", out, re.MULTILINE)


def test_transform_normalizes_variant_spelled_directives():
    # SIESTA treats . - _ as interchangeable and labels as case-
    # insensitive.  A legacy-spelled input must be replaced IN PLACE, not
    # duplicated; for the CPU bundle the GPU toggle is flipped to .false.
    # in place regardless of spelling (audit 2026-06-27 B-2; CUDA-crash
    # fix 2026-06-29).
    src = ("SystemName x\nNumberOfAtoms 1\n"
           "DM-UseSaveDM .true.\nDiag-ELPA-GPU .true.\n"
           "max_scf_iterations 99\n")
    cpu = transform_fdf(src, label="job-cpu", gpu=False,
                        block_size=8, max_scf=5)
    # No conflicting duplicate left behind, original variant gone.
    assert "DM-UseSaveDM .true." not in cpu
    assert cpu.count(".true.") == 0          # both flipped to .false.
    # The variant spelling is replaced in place by the canonical .false.;
    # exactly one GPU-toggle line, set OFF, no duplicate.
    assert "Diag-ELPA-GPU .true." not in cpu
    assert len(re.findall(r"(?im)^\s*Diag[._-]ELPA[._-]GPU", cpu)) == 1
    assert re.search(r"(?im)^Diag\.ELPA\.GPU\s+\.false\.", cpu)
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
    # PREP, on the target (execution/job-system.md § 7) -- so they are absent here.
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)

    out_dir, written = generate_bench_bundle(
        fdf, out, cpu_np=64, gpu_gpus=1, gpu_k=4,
        gpus_per_node=4, cores_per_socket=24)

    names = {p.name for p in written}
    for expect in ("job-cpu.fdf", "job-gpu.fdf", "bench-manifest.json",
                   "job-gpu-sweep.sh", "README.md", "C.psml",
                   "prep-bench", "run-bench", "bench-summarize", "prep-run"):
        assert expect in names, f"missing {expect}"
        assert (out_dir / expect).is_file()
    # wrappers are NOT baked at generate (portable bundle)
    assert not (out_dir / "job-cpu.run.sh").exists()
    assert not (out_dir / "job-gpu.run.sh").exists()
    assert not (out_dir / "job-cpu.sbatch").exists()
    # NO stdlib copy shipped (§ 8.3): molbuilder is on the target by contract
    assert not (out_dir / "mbbench").exists()

    # GPU fdf enables the CUDA toggle; CPU fdf sets it EXPLICITLY .false.
    # (NOT omitted -- the ELPA-CUDA build defaults to GPU and the CPU
    # baseline crashed in CUDA when it was omitted; Sol job 57852377).
    import re as _re
    assert _re.search(r"(?im)^Diag\.ELPA\.GPU\s+\.true\.",
                      (out_dir / "job-gpu.fdf").read_text())
    assert _re.search(r"(?im)^Diag\.ELPA\.GPU\s+\.false\.",
                      (out_dir / "job-cpu.fdf").read_text())

    # self-describing manifest@2 (§ 8.6): human-readable, engine-neutral
    # `script` key, `points` (not `jobs`)
    man = json.loads((out_dir / "bench-manifest.json").read_text())
    assert man["schema"] == "molbuilder/bench-manifest@2"
    assert man["engine"] == "siesta"
    assert "description" in man and "measured" in man
    assert man["points"]["cpu"]["script"] == "job-cpu.fdf"
    assert man["points"]["gpu"]["script"] == "job-gpu.fdf"
    assert man["points"]["cpu"]["role"] == "baseline"
    assert man["points"]["cpu"]["mpi_np"] == 64
    assert man["points"]["cpu"]["cpus_per_task"] == 1
    assert man["points"]["gpu"]["gpus"] == 1
    assert man["points"]["gpu"]["gpu_k"] == 4


def test_generate_ships_self_bootstrapping_shims(tmp_path):
    # §8.3: the on-target entry points are bash shims that SELF-BOOTSTRAP the
    # molbuilder env (no manual activation) then run the CLI -- no stdlib copy.
    fdf = _make_src(tmp_path)
    out = _make_out_with_config(tmp_path)
    out_dir, _ = generate_bench_bundle(fdf, out)
    for shim, sub in (("prep-bench", "prep"),
                      ("bench-summarize", "summarize"),
                      ("prep-run", "prep-run")):
        t = (out_dir / shim).read_text()
        assert t.startswith("#!/usr/bin/env bash")
        # bootstrap: env activation (workstation autodetect / HPC preamble) +
        # invocation resolution into $_mb_run, then the actual call.
        assert "MB_HOST_ENV" in t                 # baked host-env (overridable)
        assert "MB_REPO" in t                     # explicit repo escape hatch
        assert 'import molbuilder' in t           # importability gate
        assert "conda activate" in t              # workstation activation
        assert '"preamble"' in t                  # reads HPC preamble from config (not hardcoded)
        assert "python -m molbuilder" in t        # fallback invocation
        assert f'exec $_mb_run bench {sub} "$@"' in t
        assert (out_dir / shim).stat().st_mode & 0o111
    rb = (out_dir / "run-bench").read_text()
    assert "job-cpu.run.sh" in rb and "job-gpu-sweep.sh" in rb
    assert "run ./prep-bench first" in rb         # guards the un-prepped state


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
    # on the target (execution/job-system.md § 7); generate stays target-neutral.
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
    """CONTRACT: ``bench generate --help`` must document the config file the
    bundle depends on -- name it, say WHERE it goes and WHAT it holds, show a
    concrete example, and point at the full reference -- so a user reading only
    the help can produce a runnable bundle.

    The pointer is checked by RESOLVING it, not by matching a filename.  This
    test used to assert the literal string ``config.md``; the 2026-07 docs
    migration retired that file (it is under docs/archive/old_docs/ now) and the
    help was correctly repointed at ``docs/execution/running-a-job.md``, whose
    § 5.2 is the home of ``script_generation``.  So the contract held and only
    the assertion had rotted.  Resolving the path instead pins what actually
    matters -- the help sends you somewhere that EXISTS -- and it survives the
    next reorganisation while still catching a dangling pointer, which a
    filename match can never do."""
    import re
    from pathlib import Path
    from click.testing import CliRunner
    from molbuilder.bench._cli import bench_group
    out = CliRunner().invoke(bench_group, ["generate", "-h"]).output
    assert ".molbuilder.json" in out
    assert "WHERE" in out and "WHAT" in out          # location + content
    assert '"script_generation"' in out               # a concrete example
    # A pointer to the full reference, and it must resolve.
    repo = Path(__file__).resolve().parents[1]
    refs = re.findall(r"docs/[\w/.-]+\.md", out)
    assert refs, (
        "the help no longer points at any doc for the config file; a user who "
        "reads only --help has nowhere to go for the full reference")
    for ref in refs:
        assert (repo / ref).is_file(), (
            f"--help points at {ref!r}, which does not exist -- a dangling "
            f"pointer is worse than none.  Update the help text (and prefer "
            f"the doc that owns the section, e.g. execution/running-a-job.md "
            f"§ 5.2 for script_generation).")


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
# baked at PREP, on the target, from the detected topology (job-system.md
# § 7).  They are no longer produced by `generate`.


def test_generate_rejects_non_fdf(tmp_path):
    bad = tmp_path / "input.txt"
    bad.write_text("not an fdf")
    with pytest.raises(ValueError, match="must be a .fdf"):
        generate_bench_bundle(bad, tmp_path / "out")


# NOTE (§ 8.2/§ 8.3): the old "shipped stdlib prep-lib runs with no molbuilder"
# tests were retired -- molbuilder is installed on every target (the § 3.4
# contract), so the bundle no longer ships an mbbench/ copy and the shims call
# `molbuilder bench …` (see test_generate_ships_thin_molbuilder_shims).  The
# summarize / prep-run logic is exercised by tests/test_bench_result.py and the
# prep bake/plan flow by tests/test_bench_prep.py.
