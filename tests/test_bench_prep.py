"""Tests for the prep-bench driver (molbuilder/bench/prep.py)."""
from __future__ import annotations

import json
import re
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


def test_summary_surfaces_readiness_with_gpu(tmp_path):
    # prep must POINT at the existing envs readiness checks (job-execution.md
    # § 3.4); with GPUs detected, both doctor and the GPU-env validate show.
    env, written = prep.run_prep_bench(
        tmp_path, overrides={"cores_per_socket": 24, "gpus_per_node": 4,
                             "gpu_type": "a100"},
        scheduler_override="slurm", now_iso="t")
    out = prep._summary(env, written)
    assert "molbuilder envs doctor" in out
    assert "molbuilder envs validate molbuilder-siesta-gpu" in out
    # point only -- never auto-run / auto-install (assistant, not nanny)
    assert "envs install" not in out


def test_summary_readiness_validate_is_gpu_gated(tmp_path):
    # no GPUs -> doctor still shows, the GPU-env validate line does not.
    env, written = prep.run_prep_bench(
        tmp_path, overrides={"cores_per_socket": 8, "gpus_per_node": 0},
        scheduler_override="workstation", now_iso="t")
    out = prep._summary(env, written)
    assert "molbuilder envs doctor" in out
    assert "molbuilder envs validate" not in out


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


def test_parse_ks():
    assert prep._parse_ks("8,16") == [8, 16]
    assert prep._parse_ks("8, 16 ,32") == [8, 16, 32]
    assert prep._parse_ks(None) is None
    assert prep._parse_ks("") is None


def test_gpu_ks_override_in_sweep(tmp_path):
    # --gpu-ks 8,16 -> exactly G x {8,16}; K=16 underuses a 24-core socket
    # (c=1, 8 idle) and is flagged but still emitted.
    prep.run_prep_bench(
        tmp_path,
        overrides={"cores_per_socket": 24, "gpus_per_node": 2,
                   "gpu_type": "a100"},
        scheduler_override="slurm", ks=[8, 16], now_iso="t")
    text = (tmp_path / "job-gpu-sweep.sh").read_text()
    assert "K values = 8,16" in text
    assert "sbatch --gres=gpu:a100:1 -n 8 -c 3 job-gpu.sbatch" in text
    assert "sbatch --gres=gpu:a100:2 -n 32 -c 1 job-gpu.sbatch" in text  # G2K16
    assert "8 idle cores" in text                 # K=16 doesn't divide 24
    # no K=4/K=2 points (we asked for 8,16 only)
    assert "K=4 " not in text


def test_gpu_instance_scaling_allows_oversubscription(tmp_path):
    # GPU-instance / np scaling on a small box: K beyond cores/socket is
    # ALLOWED (ranks share the GPU via MPS) and flagged, NOT skipped, so
    # the np-scaling curve isn't truncated at the core count.
    prep.run_prep_bench(
        tmp_path,
        overrides={"cores_per_socket": 6, "gpus_per_node": 1,
                   "gpu_type": "rtx3060"},
        scheduler_override="workstation", ks=[1, 2, 4, 8], now_iso="t")
    text = (tmp_path / "job-gpu-sweep.sh").read_text()
    for k in (1, 2, 4, 8):
        assert f"point-G1K{k}" in text            # full ladder emitted
    assert "INVALID" not in text                  # K=8 > 6 not rejected
    assert "OVERSUBSCRIBED: K=8 > cores/socket=6" in text


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


# --------------------------------------------------------------------- #
#  bake_target_wrappers -- prep-time wrapper baking (job-execution.md §7) #
#  The wrappers moved from generate to prep; THIS is where one portable   #
#  bundle gets specialised for the machine it will run on.                #
# --------------------------------------------------------------------- #

_BAKE_FDF = """\
SystemName          src
SystemLabel         src
NumberOfAtoms       4
%block ChemicalSpeciesLabel
1 6 C
%endblock ChemicalSpeciesLabel
%block AtomicCoordinatesAndAtomicSpecies
0.0 0.0 0.0 1
0.1 0.0 0.0 1
0.0 0.1 0.0 1
0.0 0.0 0.1 1
%endblock AtomicCoordinatesAndAtomicSpecies
Diag.ELPA.GPU       .true.
"""

_HPC_CONFIG = {
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


def _make_bundle(tmp_path, *, hpc_config=False):
    """Generate a portable bundle (fdf + manifest, NO wrappers)."""
    from molbuilder.bench.generate import generate_bench_bundle
    src = tmp_path / "src"; src.mkdir()
    (src / "input.fdf").write_text(_BAKE_FDF)
    (src / "C.psml").write_text("<psml></psml>")
    out = tmp_path / "bundle"; out.mkdir()
    if hpc_config:
        (out / ".molbuilder.json").write_text(json.dumps(_HPC_CONFIG))
    out_dir, _ = generate_bench_bundle(
        src / "input.fdf", out, cpu_np=64, gpu_gpus=1, gpu_k=4)
    assert not (out_dir / "job-cpu.run.sh").exists()   # not baked at generate
    return out_dir


def _env(scheduler, *, cores=24, gpus=1, gtype="a100"):
    return env_mod.resolve_environment(
        overrides={"cores_per_socket": cores, "gpus_per_node": gpus,
                   "gpu_type": gtype},
        scheduler_override=scheduler, now_iso="t")


def test_bake_workstation_autodetects_conda(tmp_path, monkeypatch):
    # SAME bundle, workstation target: prep autodetects conda and bakes
    # `conda activate` into the run wrappers (no scheduler -> .run.sh only).
    import molbuilder.runtime_config as rc
    monkeypatch.setattr(rc, "detect_conda_activation",
                        lambda: {"activation": "conda activate",
                                 "preamble": 'source "/x/conda.sh"'})
    from molbuilder.bench.generate import bake_target_wrappers
    out_dir = _make_bundle(tmp_path)                   # no shipped config
    bake_target_wrappers(out_dir, _env("workstation"))
    run = (out_dir / "job-cpu.run.sh").read_text()
    assert "conda activate" in run
    assert (out_dir / "job-gpu.run.sh").is_file()
    # workstation: no scheduler -> no sbatch
    assert not (out_dir / "job-cpu.sbatch").exists()
    # it specialised THIS dir's config for the workstation
    cfg = json.loads((out_dir / ".molbuilder.json").read_text())
    assert cfg["script_generation"]["activation"] == "conda activate"


def test_bake_hpc_uses_shipped_config_and_detected_topology(tmp_path):
    # SAME generator, HPC target: prep uses the shipped explicit activation
    # and bakes sbatch headers from the DETECTED topology (gres type + cores).
    from molbuilder.bench.generate import bake_target_wrappers
    out_dir = _make_bundle(tmp_path, hpc_config=True)
    bake_target_wrappers(out_dir, _env("slurm", cores=24, gpus=1, gtype="a100"))

    run = (out_dir / "job-gpu.run.sh").read_text()
    assert "module load mamba" in run and "source activate" in run

    gpu_sbatch = (out_dir / "job-gpu.sbatch").read_text()
    assert "--gres=gpu:a100:1" in gpu_sbatch          # detected gpu_type
    assert re.search(r"^#SBATCH -n 4", gpu_sbatch, re.MULTILINE)   # K*G
    assert re.search(r"^#SBATCH -c 6", gpu_sbatch, re.MULTILINE)   # 24//K

    cpu_sbatch = (out_dir / "job-cpu.sbatch").read_text()
    assert re.search(r"^#SBATCH -n 64\b", cpu_sbatch, re.MULTILINE)
    assert re.search(r"^#SBATCH -c 1\b", cpu_sbatch, re.MULTILINE)
    assert re.search(r"^#SBATCH --mem=\d+G", cpu_sbatch, re.MULTILINE)


def test_bake_gres_follows_detected_gpu_type(tmp_path):
    # The gres TYPE is the DETECTED one (not a generate-time hardcoded a100);
    # the COUNT is the manifest knob (--gpu-gpus=1); -c is detected cores//K.
    from molbuilder.bench.generate import bake_target_wrappers
    out_dir = _make_bundle(tmp_path, hpc_config=True)
    bake_target_wrappers(out_dir, _env("slurm", cores=16, gpus=2, gtype="h100"))
    gpu_sbatch = (out_dir / "job-gpu.sbatch").read_text()
    assert "--gres=gpu:h100:1" in gpu_sbatch          # type detected, count from manifest
    assert re.search(r"^#SBATCH -c 4", gpu_sbatch, re.MULTILINE)   # 16//4


def test_bake_workstation_emits_no_sbatch_even_with_hpc_config(tmp_path,
                                                               monkeypatch):
    # Consistency (§ 7.5): a bundle carrying an HPC scheduler block, prepped
    # on a WORKSTATION, must NOT emit stray SLURM .sbatch files -- the
    # DETECTED scheduler gates emission, not the shipped config block.
    import molbuilder.runtime_config as rc
    monkeypatch.setattr(rc, "detect_conda_activation",
                        lambda: {"activation": "conda activate",
                                 "preamble": 'source "/x/conda.sh"'})
    from molbuilder.bench.generate import bake_target_wrappers
    out_dir = _make_bundle(tmp_path, hpc_config=True)   # ships scheduler block
    bake_target_wrappers(out_dir, _env("workstation"))
    assert (out_dir / "job-cpu.run.sh").is_file()
    assert not (out_dir / "job-cpu.sbatch").exists()    # no stray SLURM file
    assert not (out_dir / "job-gpu.sbatch").exists()
    # and the activation was specialised to the workstation
    assert "conda activate" in (out_dir / "job-cpu.run.sh").read_text()


def test_bake_hpc_without_config_raises(tmp_path):
    # HPC target, no shipped activation -> refuse with a pointer (clean job
    # shell, nothing to autodetect; job-execution.md § 3.3 row M / § 7.5).
    import molbuilder.runtime_config as rc
    from molbuilder.bench.generate import bake_target_wrappers
    out_dir = _make_bundle(tmp_path)                   # NO .molbuilder.json
    with pytest.raises(rc.RuntimeConfigError, match="4.4.2|script_generation"):
        bake_target_wrappers(out_dir, _env("slurm"))


def test_bake_missing_manifest_raises(tmp_path):
    from molbuilder.bench.generate import bake_target_wrappers
    d = tmp_path / "empty"; d.mkdir()
    with pytest.raises(FileNotFoundError, match="bench-manifest"):
        bake_target_wrappers(d, _env("workstation"))
