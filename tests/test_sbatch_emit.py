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
import math
import re
import subprocess
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


def test_gpu_mem_default_applies_to_gpu_not_cpu(tmp_path):
    """``scheduler.gpu.mem`` is a GPU-only default (Sol's GPU 24 GB default
    is tight); CPU jobs must NOT inherit it (they keep the generous
    partition default)."""
    sched = dict(_SCHED)
    sched["gpu"] = dict(_SCHED["gpu"], mem="64G")
    sched["defaults"] = dict(_SCHED["defaults"], mem=None)
    gfdf = tmp_path / "g.fdf"; gfdf.write_text("Diag.ELPA.GPU .true.\n")
    gtxt = render_sbatch(gfdf, sched, ntasks=8, gpu=True, gpu_count=1)
    assert "#SBATCH --mem=64G" in gtxt              # GPU gets gpu.mem
    cfdf = tmp_path / "c.fdf"; cfdf.write_text("x\n")
    ctxt = render_sbatch(cfdf, sched, ntasks=64)    # CPU job
    assert "--mem" not in ctxt                       # NOT capped


def test_explicit_mem_overrides_gpu_mem(tmp_path):
    sched = dict(_SCHED)
    sched["gpu"] = dict(_SCHED["gpu"], mem="64G")
    fdf = tmp_path / "g.fdf"; fdf.write_text("Diag.ELPA.GPU .true.\n")
    txt = render_sbatch(fdf, sched, ntasks=8, gpu=True, gpu_count=1, mem="120G")
    assert "#SBATCH --mem=120G" in txt and "64G" not in txt


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


def test_wrapper_gpu_has_socket_affinity_block(project):
    # GPU launcher carries the § 7.5.2 socket co-location logic: pin under
    # a whole-node (--exclusive) cpuset, WARN on a shared cross-socket
    # allocation, exec via the $_pin prefix.  write_run_wrapper bash -n's
    # the rendered wrapper, so reaching here = it parses.
    fdf = project / "g.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, mpi_np=4, gres="gpu:a100:1",
                              cpus_per_task=6)
    runsh = (project / "g.run.sh").read_text()
    assert "socket co-location" in runsh
    assert "socket-pin -> GPU socket" in runsh        # the pin branch
    assert "WARN cross-socket" in runsh               # the warn branch
    assert "exec $_pin siesta" in runsh               # numactl-or-nothing
    assert "physical_package_id" in runsh


def test_wrapper_cpu_has_no_socket_affinity_block(project):
    # CPU jobs have no GPU to co-locate against.
    fdf = project / "c.fdf"
    fdf.write_text("NumberOfAtoms 444\n")
    runwrap.write_run_wrapper(fdf, mpi_np=8)
    runsh = (project / "c.run.sh").read_text()
    assert "socket co-location" not in runsh
    assert "exec $_pin siesta" not in runsh


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


# A complete-enough fdf so the memory estimator parses a real system
# (N_orb > 0) and the runtime mem-audit block is emitted.
_PARSEABLE_FDF = """\
NumberOfAtoms 4
MeshCutoff 300 Ry
PAO.BasisSize TZP
LatticeConstant 1.0 Ang
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
"""


def test_wrapper_cpu_emits_runtime_mem_audit(project):
    # CPU SIESTA .fdf -> the launcher carries the estimate-vs-allocation
    # audit (recomputes for the runtime rank count; WARNs on OOM risk).
    fdf = project / "job.fdf"
    fdf.write_text(_PARSEABLE_FDF)
    runwrap.write_run_wrapper(fdf, mpi_np=64)
    runsh = (project / "job.run.sh").read_text()
    assert "_mb_mem_est=$(awk" in runsh
    assert "memory          : estimated" in runsh
    assert "EXCEEDS allocation" in runsh           # the WARN branch
    # Reads the SLURM allocation, with a /proc/meminfo fallback.
    assert "SLURM_MEM_PER_NODE" in runsh and "MemTotal" in runsh


def test_runtime_mem_audit_awk_reproduces_the_formula(project):
    # The baked awk must compute ceil(safety*(fixed + c_rank*n)) +
    # ceil(extra), floored, capped -- the same formula as the Python
    # estimator.  Extract the awk's constants, RUN it for several rank
    # counts, and check against an independent Python evaluation.
    fdf = project / "job.fdf"
    fdf.write_text(_PARSEABLE_FDF)
    runwrap.write_run_wrapper(fdf, mpi_np=64)
    runsh = (project / "job.run.sh").read_text()
    line = next(ln for ln in runsh.splitlines()
                if ln.strip().startswith("_mb_mem_est=$(awk"))

    def _const(flag):
        m = re.search(rf"-v {flag}=([0-9.]+)", line)
        return float(m.group(1))
    f, pr, s, e = _const("f"), _const("pr"), _const("s"), _const("e")
    fl, cap = _const("fl"), _const("cap")

    def _py(n):
        raw = s * (f + pr * n)
        est = math.ceil(raw) + math.ceil(e)
        est = max(est, fl)
        if cap > 0 and est > cap:
            est = cap
        return int(est)

    for n in (1, 4, 8, 32, 64, 256):
        out = subprocess.run(
            ["bash"], text=True, capture_output=True,
            input=f'_mpi_np={n}\n{line}\necho "$_mb_mem_est"')
        assert out.returncode == 0, out.stderr
        assert int(out.stdout.strip()) == _py(n), f"mismatch at n={n}"


def test_workstation_gpu_knobs_match_launcher_contract(project):
    # CONTRACT: the env vars the bench/run adapter emits for a workstation
    # GPU point MUST be exactly the ones the generated launcher reads, or
    # the rank/omp count is silently wrong (params valid bash, wrong
    # meaning).  This pins producer (adapter) <-> consumer (launcher).
    from molbuilder.bench.adapters import WorkstationAdapter
    from molbuilder.bench.environment import Environment, Topology

    fdf = project / "g.fdf"
    fdf.write_text(_PARSEABLE_FDF + "Diag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, mpi_np=4, gres="gpu:a100:1",
                              cpus_per_task=6)
    launcher = (project / "g.run.sh").read_text()
    # the launcher actually honours these on a workstation (no SLURM):
    assert "MOLBUILDER_MPI_NP" in launcher
    assert "MOLBUILDER_OMP_NUM_THREADS" in launcher

    env = Environment(scheduler="workstation",
                      topology=Topology(cores_per_socket=24, gpus_per_node=1,
                                        gpu_type="a100"))
    sweep = WorkstationAdapter().format_bench(env)["job-gpu-sweep.sh"]
    # the adapter drives those SAME knobs (+ CUDA_VISIBLE_DEVICES for G):
    assert "MOLBUILDER_MPI_NP=" in sweep
    assert "MOLBUILDER_OMP_NUM_THREADS=" in sweep
    assert "CUDA_VISIBLE_DEVICES=" in sweep
    run = WorkstationAdapter().format_run(
        {"engine": "gpu", "knobs": {"gpus": 1, "ranks_per_gpu": 8}}, env,
        script_base="prod")["run-production.sh"]
    assert "MOLBUILDER_MPI_NP=8" in run


def test_wrapper_gpu_has_no_mem_audit(project):
    # GPU jobs size memory from gpu.mem, not the CPU model -> no audit.
    fdf = project / "job.fdf"
    fdf.write_text(_PARSEABLE_FDF + "Diag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, mpi_np=4, gres="gpu:a100:1",
                              cpus_per_task=6)
    runsh = (project / "job.run.sh").read_text()
    assert "_mb_mem_est=$(awk" not in runsh


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


def _min_cpu_fdf(tmp_path):
    """Minimal SIESTA .fdf carrying the fields the mem estimator reads."""
    fdf = tmp_path / "job.fdf"
    fdf.write_text(
        "SystemName test\n"
        "NumberOfAtoms 100\n"
        "NumberOfSpecies 1\n"
        "PAO.BasisSize DZP\n"
        "MeshCutoff 300 Ry\n"
        "LatticeConstant 1.0 Ang\n"
        "%block LatticeVectors\n"
        "20.0 0.0 0.0\n0.0 20.0 0.0\n0.0 0.0 20.0\n"
        "%endblock LatticeVectors\n"
        "%block kgrid_Monkhorst_Pack\n"
        "2 0 0 0.0\n0 2 0 0.0\n0 0 1 0.0\n"
        "%endblock kgrid_Monkhorst_Pack\n"
        "%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        + "".join(f"{i*0.1} 0.0 0.0 1\n" for i in range(100))
        + "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    return fdf


def test_cpu_siesta_mem_auto_estimated_when_unset(tmp_path):
    """A CPU SIESTA .fdf with no explicit/config mem gets a system-aware
    --mem (prevents the high-np OOM), with the breakdown as a comment."""
    sched = dict(_SCHED)
    sched["defaults"] = dict(_SCHED["defaults"], mem=None)
    sched["mem_model"] = {"node_mem_gb": 500}
    fdf = _min_cpu_fdf(tmp_path)
    txt = render_sbatch(fdf, sched, ntasks=64)
    assert "#SBATCH --mem=" in txt
    assert "auto-estimated from problem size" in txt
    # more ranks -> more replication -> >= memory
    lo = render_sbatch(fdf, sched, ntasks=8)
    def _mem(t): return int([l for l in t.splitlines()
                             if l.startswith("#SBATCH --mem=")][0]
                            .split("=")[1].rstrip("G"))
    assert _mem(txt) >= _mem(lo)


def test_explicit_mem_skips_estimate(tmp_path):
    sched = dict(_SCHED); sched["defaults"] = dict(_SCHED["defaults"], mem=None)
    fdf = _min_cpu_fdf(tmp_path)
    txt = render_sbatch(fdf, sched, ntasks=64, mem="120G")
    assert "#SBATCH --mem=120G" in txt
    assert "auto-estimated" not in txt


def test_node_mem_cap_clamps_estimate(tmp_path):
    sched = dict(_SCHED); sched["defaults"] = dict(_SCHED["defaults"], mem=None)
    sched["mem_model"] = {"node_mem_gb": 16}     # tiny cap
    fdf = _min_cpu_fdf(tmp_path)
    txt = render_sbatch(fdf, sched, ntasks=64)
    mem = int([l for l in txt.splitlines()
               if l.startswith("#SBATCH --mem=")][0].split("=")[1].rstrip("G"))
    assert mem <= 16
