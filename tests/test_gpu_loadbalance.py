"""Static-text contract tests for the GPU load-balance + --dry-run
additions to the SIESTA run-wrapper (slurm-integration.md § 7.5.1).

These assert on the GENERATED bash (no execution) -- the docstrings of
the extracted block-emitters are the contracts; these tests pin them:

  * _gpu_loadbalance_block       -> $_ranks_per_gpu derivation
  * _gpu_per_rank_launcher_block -> rank<->GPU helper + SLURM-trust
  * _siesta_resolved_log_block   -> always-on launch audit log
  * _siesta_dry_run_block        -> --dry-run preview, side-effect-free
  * OMP precedence honors SLURM_CPUS_PER_TASK
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder import runwrap
from molbuilder.diagnostics import Capabilities, set_capabilities


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    """Activation config (refuse-to-emit contract) + synthetic caps."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"activation": "source activate"}
    }))
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs=frozenset({"molbuilder-siesta", "molbuilder-siesta-gpu"}),
    ))
    yield


def _gpu(tmp_path: Path, np: int = 4) -> str:
    f = tmp_path / "g.fdf"
    f.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    return runwrap.render_run_wrapper(f, mpi_np=np)


def _cpu(tmp_path: Path, np: int = 20) -> str:
    f = tmp_path / "c.fdf"
    f.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .false.\n")
    return runwrap.render_run_wrapper(f, mpi_np=np)


# --------------------------------------------------------------------- #
#  Load-balance block (§ 7.5.1)                                         #
# --------------------------------------------------------------------- #


def test_gpu_wrapper_has_loadbalance_block(tmp_path):
    t = _gpu(tmp_path)
    assert "GPU load-balance" in t
    assert "_ranks_per_gpu=$(( _mpi_np / _ngpu ))" in t
    assert 'grep -c "^GPU "' in t  # GPU count probe


def test_cpu_wrapper_has_no_loadbalance_block(tmp_path):
    t = _cpu(tmp_path)
    assert "GPU load-balance" not in t
    assert "_ranks_per_gpu" not in t


# --------------------------------------------------------------------- #
#  Per-rank launcher: rank<->GPU mapping + SLURM-trust                  #
# --------------------------------------------------------------------- #


def test_gpu_wrapper_writes_per_rank_helper(tmp_path):
    t = _gpu(tmp_path)
    assert "_rank_helper=\".mb-rank-launch-$$.sh\"" in t
    assert "<<'HELPEREOF'" in t
    # block-distributed rank -> GPU: derive the ALLOCATED GPU list from
    # CUDA_VISIBLE_DEVICES (robust under SLURM cgroups), index into it.
    assert 'IFS=, read -ra _gpus <<< "$CUDA_VISIBLE_DEVICES"' in t
    assert "_idx=$(( _lr * _ngpu / _ls ))" in t
    assert "_gpu=${_gpus[$_idx]}" in t
    assert "export CUDA_VISIBLE_DEVICES=$_gpu" in t
    assert '_siesta_target="bash $_rank_helper"' in t


def test_single_unified_exit_trap(tmp_path):
    """Bug fix: a second ``trap ... EXIT`` would REPLACE the first, so all
    teardown (per-rank helper + MPS daemon) must route through ONE
    unified trap.  Assert exactly one trap COMMAND + the cleanup wiring."""
    t = _gpu(tmp_path)
    trap_cmds = [ln for ln in t.splitlines() if ln.strip().startswith("trap ")]
    assert trap_cmds == ["trap _mb_cleanup EXIT"], trap_cmds
    assert "_mb_cleanup() {" in t
    assert "_mps_started=1" in t          # MPS sets the flag, not its own trap
    assert '[ "${_mps_started:-0}" = "1" ]' in t


def test_gpu_wrapper_trusts_slurm_cpuset(tmp_path):
    t = _gpu(tmp_path)
    # under SLURM, drop the manual numactl/map-by (P1, § 7.5.1.b)
    assert 'if [ -n "${SLURM_JOB_ID:-}" ]; then' in t
    assert "trusting scheduler cpuset" in t
    # ... and there must be no double-quote seam artifact
    assert '""(no manual' not in t


def test_launch_line_uses_siesta_target(tmp_path):
    t = _gpu(tmp_path)
    assert ('_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np '
            '$_mpirun_bind $_siesta_target"' in t)


def test_cpu_wrapper_target_is_bare_siesta(tmp_path):
    t = _cpu(tmp_path)
    assert '_siesta_target="siesta"' in t
    # No per-rank helper is WRITTEN in CPU mode (the shared _mb_cleanup
    # function still references ${_rank_helper:-} harmlessly -- it no-ops
    # when unset -- so assert on the WRITE, not the bare name).
    assert '_rank_helper=".mb-rank-launch' not in t
    assert "HELPEREOF" not in t


# --------------------------------------------------------------------- #
#  MPS gating: per-GPU sharing, and never during --dry-run             #
# --------------------------------------------------------------------- #


def test_mps_keyed_on_ranks_per_gpu_and_not_dry_run(tmp_path):
    t = _gpu(tmp_path)
    assert ('[ "$_use_mps_default" = "1" ] && [ "$_ranks_per_gpu" -ge 2 ] '
            '&& [ "${_dry_run:-0}" != "1" ]' in t)


# --------------------------------------------------------------------- #
#  Resolved-launch audit log (always)                                  #
# --------------------------------------------------------------------- #


def test_resolved_launch_logged(tmp_path):
    t = _gpu(tmp_path)
    assert '_log INFO "resolved launch :' in t
    assert '_log INFO "gpu placement   :' in t  # GPU mode adds placement


def test_cpu_resolved_launch_logged_without_gpu_placement(tmp_path):
    t = _cpu(tmp_path)
    assert '_log INFO "resolved launch :' in t
    assert "gpu placement" not in t


# --------------------------------------------------------------------- #
#  --dry-run preview                                                    #
# --------------------------------------------------------------------- #


def test_dry_run_flag_and_block(tmp_path):
    t = _gpu(tmp_path)
    assert "--dry-run|--dryrun)" in t
    assert "_dry_run=1; shift ;;" in t
    assert "molbuilder DRY RUN (no SIESTA launch)" in t
    # GPU mode: per-rank mapping preview + exit before launch
    assert "Rank -> GPU mapping (block-distributed)" in t
    assert "_dry_run complete" not in t  # (sanity: it's the log msg form)
    assert '_log INFO "dry-run complete; no SIESTA launched"' in t
    assert "exit 0" in t


def test_dry_run_present_for_cpu_without_gpu_mapping(tmp_path):
    t = _cpu(tmp_path)
    assert "molbuilder DRY RUN (no SIESTA launch)" in t
    assert "Rank -> GPU mapping" not in t   # no GPU section in CPU mode


def test_pyscf_has_dry_run(tmp_path):
    p = tmp_path / "q.py"
    p.write_text("# fake\n")
    t = runwrap.render_run_wrapper(p)
    assert "--dry-run|--dryrun)" in t
    assert "molbuilder DRY RUN (no PySCF launch)" in t


# --------------------------------------------------------------------- #
#  OMP precedence honors SLURM_CPUS_PER_TASK                           #
# --------------------------------------------------------------------- #


def test_omp_honors_slurm_cpus_per_task(tmp_path):
    t = _gpu(tmp_path)
    assert ('_omp_threads="${OMP_NUM_THREADS:-'
            '${SLURM_CPUS_PER_TASK:-$_omp_threads_default}}"' in t)


# --------------------------------------------------------------------- #
#  SCF per-iteration timing instrument (§ 11.0b, item D)               #
# --------------------------------------------------------------------- #


def test_scf_timing_instrument_present(tmp_path):
    """Both CPU and GPU SIESTA wrappers carry the per-iteration timing
    instrument: the _mb_scf_tee filter, a per-run .scf-timing.log paired
    with the .out, the piped launch + PIPESTATUS, and a wall-time log."""
    for t in (_gpu(tmp_path), _cpu(tmp_path)):
        assert "_mb_scf_tee() {" in t
        assert '/^[ \\t]*scf:[ \\t]*[0-9]/' in t      # iteration-line match
        assert '_scf_timing_log="${_out_file%.out}.scf-timing.log"' in t
        assert '| _mb_scf_tee "$_out_file" "$_scf_timing_log"' in t
        # PIPESTATUS so awk never masks SIESTA's exit code
        assert "_siesta_exit=${PIPESTATUS[0]}" in t
        assert 'SIESTA wall time:' in t


def test_propor_diagnostic_still_reads_out_after_timing(tmp_path):
    """The timing pipe still writes the .out, so the propor-error
    diagnostic that greps $_out_file keeps working."""
    t = _gpu(tmp_path)
    assert 'grep -aq "propor: ERROR" "$_out_file"' in t


def test_pyscf_has_no_scf_timing(tmp_path):
    p = tmp_path / "q.py"
    p.write_text("# fake\n")
    t = runwrap.render_run_wrapper(p)
    assert "_mb_scf_tee" not in t
    assert "scf-timing.log" not in t
