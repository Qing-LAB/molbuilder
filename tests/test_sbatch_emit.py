"""Tests for the SLURM ``.sbatch`` submission-layer emitter
(``runwrap.render_sbatch`` + the ``render_wrappers`` / ``write_run_wrapper``
wiring).  ``write_sbatch`` was a second writer with no production caller and
went on 2026-08-18 (roadmap P6); one test below asserts its absence.

Authoritative design: docs/execution/job-system.md  two-layer model (header delegates to the unchanged .run.sh)
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
from molbuilder.runwrap import WrapperError, render_sbatch
from molbuilder.jobset.model import Resources


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


def test_mail_user_percent_pattern_is_flagged_not_dropped(tmp_path):
    # %u doesn't expand in --mail-user (only -o/-e/-i).  We KEEP the
    # user's value (don't twist their config) but flag it explicitly.
    fdf = tmp_path / "j.fdf"; fdf.write_text("NumberOfAtoms 1\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=8)        # _SCHED has %u
    assert '#SBATCH --mail-user="%u@asu.edu"' in txt   # kept verbatim
    assert "do NOT expand in --mail-user" in txt        # but flagged


def test_mail_user_real_address_not_flagged(tmp_path):
    sched = json.loads(json.dumps(_SCHED))
    sched["directives"]["mail_user"] = "me@asu.edu"
    fdf = tmp_path / "j.fdf"; fdf.write_text("NumberOfAtoms 1\n")
    txt = render_sbatch(fdf, sched, ntasks=8)
    assert '#SBATCH --mail-user="me@asu.edu"' in txt
    assert "do NOT expand" not in txt


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


def test_an_absent_gpu_count_defaults_to_one_device(tmp_path):
    """**G5** (`execution/gpu.md`): one default for an absent ask, and it is
    **1 device** — not the rank count.

    *Replaces* `test_gpu_count_defaults_to_ntasks`, which asserted
    ``ntasks=3 -> --gres=gpu:a100:3``: the *one rank per GPU* model D12e
    retired on 2026-08-13.  It was the only thing keeping that model alive —
    `_render_sbatch_for`, the one production caller, already defaulted to 1,
    so the two disagreed one function apart and the wrong one was the branch
    a direct caller reached.  Note the test immediately below has asserted
    the REPLACING model — ranks and devices independent — the whole time.
    """
    fdf = tmp_path / "g.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    txt = render_sbatch(fdf, _SCHED, ntasks=3, gpu=True)
    assert "#SBATCH --gres=gpu:a100:1" in txt, (
        "an absent gpu_count must ask for ONE device; deriving it from the "
        "rank count is the retired 1-rank-per-GPU model (gpu.md G5)")
    # ...and the rank count is untouched by the device default.
    assert "#SBATCH -n 3" in txt or "#SBATCH --ntasks=3" in txt


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


def test_gpu_memory_falls_back_to_shared_default(tmp_path):
    """A GPU job uses defaults.mem only when scheduler.gpu.mem is absent."""
    sched = dict(_SCHED)
    sched["defaults"] = dict(_SCHED["defaults"], mem="200G")
    gfdf = tmp_path / "g.fdf"; gfdf.write_text("Diag.ELPA.GPU .true.\n")
    gtxt = render_sbatch(gfdf, sched, ntasks=8, gpu=True, gpu_count=1,
                         exclusive=False)
    cfdf = tmp_path / "c.fdf"; cfdf.write_text("x\n")
    ctxt = render_sbatch(cfdf, sched, ntasks=64)
    # With no gpu.mem configured, both job types fall back to defaults.mem.
    assert "#SBATCH --mem=200G" in gtxt
    assert "#SBATCH --mem=200G" in ctxt


def test_gpu_mem_config_key_is_used(tmp_path):
    """With nothing to size from (no defaults.mem, no parseable system),
    a GPU job requests the scheduler.gpu.mem FLOOR — not the site's
    tight per-GPU default."""
    sched = dict(_SCHED)
    sched["gpu"] = dict(_SCHED["gpu"], mem="64G")
    sched["defaults"] = dict(_SCHED["defaults"], mem=None)
    gfdf = tmp_path / "g.fdf"; gfdf.write_text("Diag.ELPA.GPU .true.\n")
    gtxt = render_sbatch(gfdf, sched, ntasks=8, gpu=True, gpu_count=1,
                         exclusive=False)
    assert "#SBATCH --mem=64G" in gtxt


def test_cpu_job_ignores_gpu_mem_floor(tmp_path):
    """gpu.mem / mem_cap_per_gpu are GPU-band knobs; a CPU job with no
    defaults.mem and no parseable system emits NO --mem at all (the
    partition default), not the GPU floor."""
    sched = dict(_SCHED)
    sched["gpu"] = dict(_SCHED["gpu"], mem="64G", mem_cap_per_gpu="128G")
    sched["defaults"] = dict(_SCHED["defaults"], mem=None)
    cfdf = tmp_path / "c.fdf"; cfdf.write_text("x\n")
    ctxt = render_sbatch(cfdf, sched, ntasks=64)
    assert "--mem=64G" not in ctxt
    assert "#SBATCH --mem=" not in ctxt


def test_explicit_mem_beats_the_gpu_band(tmp_path):
    """An explicit --mem is the operator's judgment: never floored,
    never capped."""
    sched = dict(_SCHED)
    sched["gpu"] = dict(_SCHED["gpu"], mem="64G", mem_cap_per_gpu="128G")
    gfdf = tmp_path / "g.fdf"; gfdf.write_text("Diag.ELPA.GPU .true.\n")
    gtxt = render_sbatch(gfdf, sched, ntasks=8, gpu=True, gpu_count=1,
                         mem="470G", exclusive=False)
    assert "#SBATCH --mem=470G" in gtxt
    assert "CAPPED" not in gtxt


def test_gpu_mem_below_floor_is_raised(tmp_path):
    """A sized value under the floor is raised to it (and says so)."""
    sched = dict(_SCHED)
    sched["gpu"] = dict(_SCHED["gpu"], mem="64G", mem_cap_per_gpu="128G")
    sched["defaults"] = dict(_SCHED["defaults"], mem="8G")
    gfdf = tmp_path / "g.fdf"; gfdf.write_text("Diag.ELPA.GPU .true.\n")
    gtxt = render_sbatch(gfdf, sched, ntasks=8, gpu=True, gpu_count=1,
                         exclusive=False)
    assert "#SBATCH --mem=64G" in gtxt
    assert "raised from 8G" in gtxt


def test_gpu_mem_above_cap_is_capped(tmp_path):
    """A sized value over the per-GPU cap is capped to cap x n_gpus —
    the node's proportional host-RAM share (backfill + CHE)."""
    sched = dict(_SCHED)
    sched["gpu"] = dict(_SCHED["gpu"], mem="64G", mem_cap_per_gpu="128G")
    sched["defaults"] = dict(_SCHED["defaults"], mem="470G")
    gfdf = tmp_path / "g.fdf"; gfdf.write_text("Diag.ELPA.GPU .true.\n")
    gtxt = render_sbatch(gfdf, sched, ntasks=8, gpu=True, gpu_count=1,
                         exclusive=False)
    assert "#SBATCH --mem=128G" in gtxt
    assert "CAPPED" in gtxt and "470G" in gtxt


def test_gpu_mem_cap_scales_with_gpu_count(tmp_path):
    """Two GPUs = two proportional shares: the cap doubles."""
    sched = dict(_SCHED)
    sched["gpu"] = dict(_SCHED["gpu"], mem="64G", mem_cap_per_gpu="128G")
    sched["defaults"] = dict(_SCHED["defaults"], mem="470G")
    gfdf = tmp_path / "g.fdf"; gfdf.write_text("Diag.ELPA.GPU .true.\n")
    gtxt = render_sbatch(gfdf, sched, ntasks=8, gpu=True, gpu_count=2,
                         exclusive=False)
    assert "#SBATCH --mem=256G" in gtxt


def test_explicit_mem_overrides_default(tmp_path):
    sched = dict(_SCHED)
    sched["defaults"] = dict(_SCHED["defaults"], mem="64G")
    fdf = tmp_path / "g.fdf"; fdf.write_text("Diag.ELPA.GPU .true.\n")
    txt = render_sbatch(fdf, sched, ntasks=8, gpu=True, gpu_count=1,
                        mem="120G", exclusive=False)
    assert "#SBATCH --mem=120G" in txt and "64G" not in txt


def test_exclusive_ignores_mem_takes_whole_node(tmp_path):
    """Exclusive owns the whole node -> --mem=0; any configured/explicit mem
    is ignored, and the script says so (§ 4.3.1)."""
    sched = dict(_SCHED)
    sched["defaults"] = dict(_SCHED["defaults"], mem="120G")
    fdf = tmp_path / "g.fdf"; fdf.write_text("Diag.ELPA.GPU .true.\n")
    txt = render_sbatch(fdf, sched, ntasks=8, gpu=True, gpu_count=1,
                        exclusive=True)
    assert "#SBATCH --exclusive" in txt
    assert "#SBATCH --mem=0" in txt
    assert "#SBATCH --mem=120G" not in txt
    assert "IGNORED" in txt                                # loud comment


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
    """The header parses as shell before anything writes it.

    Asked ``write_sbatch`` until 2026-08-18 -- a second writer with no
    production caller, deleted with P6.  The gate itself did not move: it runs
    inside ``render_wrappers``, which is where the text is produced, and the
    mode the file lands with is asserted where the writing happens
    (``test_wrapper_emits_sbatch_when_scheduler_configured`` below)."""
    fdf = tmp_path / "gpu-2a100.fdf"
    fdf.write_text("Diag.ELPA.GPU .true.\n")
    text = render_sbatch(fdf, _SCHED, ntasks=2, cpus_per_task=12,
                         gpu=True, gpu_count=2, exclusive=False)
    runwrap._validate_rendered_wrapper(text, fdf)   # raises if bash rejects it


# --------------------------------------------------------------------- #
#  write_run_wrapper wiring (§ 15 B)                                    #
# --------------------------------------------------------------------- #


def test_wrapper_emits_sbatch_when_scheduler_configured(project):
    fdf = project / "cpu-np64.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .false.\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=64))
    sbatch = project / "cpu-np64.sbatch"
    assert sbatch.is_file()
    txt = sbatch.read_text()
    assert "#SBATCH -n 64" in txt
    assert "--gres" not in txt          # CPU .fdf -> no GPU lines


def test_wrapper_gpu_fdf_emits_gres(project):
    fdf = project / "gpu.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=2, gres="gpu:a100:2", cpus_per_task=12, exclusive=False))
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
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=4, gres="gpu:a100:1", cpus_per_task=6))
    runsh = (project / "g.run.sh").read_text()
    assert "socket co-location" in runsh
    assert "socket-pin -> GPU socket" in runsh        # the pin branch
    assert "WARN cross-socket" in runsh               # the warn branch
    assert "exec $_pin siesta" in runsh               # numactl-or-nothing
    assert "physical_package_id" in runsh
    assert "MB_NO_SOCKET_PIN" in runsh                # the A/B disable toggle


def test_wrapper_cpu_has_no_socket_affinity_block(project):
    # CPU jobs have no GPU to co-locate against.
    fdf = project / "c.fdf"
    fdf.write_text("NumberOfAtoms 444\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=8))
    runsh = (project / "c.run.sh").read_text()
    assert "socket co-location" not in runsh
    assert "exec $_pin siesta" not in runsh


def test_wrapper_gpu_K_ranks_share_one_gpu(project):
    """The benchmark case: 8 (or 4) ranks share ONE A100 via MPS ->
    -n must be the rank count (8), --gres the GPU count (1)."""
    fdf = project / "gpu-k8.fdf"
    fdf.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=8, gres="gpu:a100:1", cpus_per_task=3, exclusive=False))
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
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=2))
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
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=64))
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
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=64))
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

    fdf = project / "g.fdf"
    fdf.write_text(_PARSEABLE_FDF + "Diag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=4, gres="gpu:a100:1", cpus_per_task=6))
    launcher = (project / "g.run.sh").read_text()
    # The launcher's LAUNCH honours MB_NP / OMP_NUM_THREADS (`_mpi_np` reads
    # MB_NP/SLURM_NTASKS; `_omp_threads` reads OMP_NUM_THREADS).  NOTE: a baked
    # explicit mpi_np makes MOLBUILDER_MPI_NP a no-op for the launch (it only
    # sets the shadowed auto-default) -- so the workstation sweep MUST use
    # MB_NP, not MOLBUILDER_MPI_NP (bug fixed 2026-06-28).
    assert "MB_NP" in launcher
    assert "OMP_NUM_THREADS" in launcher

    # (the bash-sweep emitters `format_bench`/`format_run` were DELETED
    # 2026-08-12, step 6 u5: trials are rendered by `jobset prep bench` --
    # each wrapper carrying its OWN translated resources is pinned in
    # test_prep_bench_fold.test_each_trials_resources_carry_its_own_
    # coordinate -- and a verdict reaches prep through
    # `_apply_run_config`, pinned in ..._applies_the_proposal_file.)


def test_wrapper_gpu_has_no_mem_audit(project):
    # The runtime mem-audit block is not emitted in this path (independent of
    # the header mem; GPU jobs use gpu.mem rather than the CPU estimator).
    fdf = project / "job.fdf"
    fdf.write_text(_PARSEABLE_FDF + "Diag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=4, gres="gpu:a100:1", cpus_per_task=6))
    runsh = (project / "job.run.sh").read_text()
    assert "_mb_mem_est=$(awk" not in runsh


def test_no_scheduler_no_sbatch(tmp_path, monkeypatch):
    """§ 10: with no scheduler block, only .run.sh is emitted.

    Isolation is HOME **and cwd**: the machine scope reads cwd-first
    (running-a-job.md § 5.2), so without the chdir this test's verdict
    depended on the developer's own molbuilder.json at the repo root."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "home").mkdir()
    (tmp_path / ".molbuilder.json").write_text(json.dumps({
        "script_generation": {"activation": "source activate"},
    }))
    fdf = tmp_path / "local.fdf"
    fdf.write_text("NumberOfAtoms 10\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=4))
    assert (tmp_path / "local.run.sh").is_file()
    assert not (tmp_path / "local.sbatch").exists()


def test_emit_sbatch_false_suppresses(project):
    fdf = project / "x.fdf"
    fdf.write_text("NumberOfAtoms 10\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=4), emit_sbatch=False)
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
