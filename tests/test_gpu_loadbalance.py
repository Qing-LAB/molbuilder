"""Static-text contract tests for the GPU load-balance + --dry-run
additions to the SIESTA run-wrapper (execution/running-a-job.md § 3.3).

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
from molbuilder.jobset.model import Resources


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
    unified cleanup.  The contract is one EXIT trap -- the TERM/INT trap
    added for D17 (2026-08-12: a walltime SIGTERM leaked the MPS daemon
    with no 'killed' line) clobbers nothing and routes through the SAME
    cleanup, guarded idempotent so signal-then-exit runs it once."""
    t = _gpu(tmp_path)
    trap_cmds = [ln.strip() for ln in t.splitlines()
                 if ln.strip().startswith("trap ")]
    exit_traps = [c for c in trap_cmds if c.endswith(" EXIT")]
    assert exit_traps == ["trap _mb_cleanup EXIT"], trap_cmds
    assert "trap _mb_on_signal TERM INT" in trap_cmds, trap_cmds
    assert "_mb_cleanup() {" in t
    assert '[ "${_mb_cleanup_ran:-0}" = "1" ]' in t   # idempotence guard
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


def test_mps_keyed_on_any_shared_gpu_and_not_dry_run(tmp_path):
    """User decision 2026-08-13: MPS whenever ranks exceed GPUs -- the
    floor-division gate (`_ranks_per_gpu >= 2`) missed the uneven split
    (3 ranks / 2 GPUs shared GPU0 by time-slicing, no funnel)."""
    t = _gpu(tmp_path)
    assert ('[ "$_use_mps_default" = "1" ] '
            '&& [ "$_mpi_np" -gt "${_ngpu:-0}" ] '
            '&& [ "${_ngpu:-0}" -ge 1 ] '
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


def test_env_bootstrap_disables_nounset(tmp_path):
    """conda activate.d hooks (e.g. cuda-nvcc's unbound NVCC_PREPEND_FLAGS)
    abort under `set -u`; the env bootstrap (preamble + activation) must
    run under `set +u`, restored to `set -u` afterwards.  (Real Sol GPU
    blocker, 2026-06-26.)"""
    t = _gpu(tmp_path)
    lines = t.splitlines()
    i_su  = next(i for i, l in enumerate(lines) if l == "set +u")
    i_act = next(i for i, l in enumerate(lines)
                 if l.startswith("source activate"))
    i_ru  = next(i for i, l in enumerate(lines)
                 if l == "set -u" and i > i_su)
    assert i_su < i_act < i_ru            # +u ... activate ... -u


def test_propor_diagnostic_scans_runwrap_log(tmp_path):
    """SIESTA's propor/abort errors go to STDERR (-> the runwrap log), not
    stdout, so the diagnostic must grep the runwrap log too."""
    t = _gpu(tmp_path)
    assert 'grep -aq "propor: ERROR" "$_out_file" "$_runwrap_log"' in t


def test_timing_read_guarded_when_no_output(tmp_path):
    """If SIESTA crashes before any scf: output, the .scf-timing.log never
    exists -- the total/N read must guard the file, not error noisily."""
    t = _gpu(tmp_path)
    assert 'if [ -f "$_scf_timing_log" ]; then' in t


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
        # Reliable benchmark metric = total wall / N SCF iters (SIESTA's
        # own per-scf time is unreliable; per-line stamps are buffered).
        assert '_n_scf=$(wc -l < "$_scf_timing_log"' in t
        assert "total/N -- the reliable metric" in t


def test_propor_diagnostic_still_reads_out_after_timing(tmp_path):
    """The timing pipe still writes the .out, so the propor-error
    diagnostic that greps $_out_file keeps working."""
    t = _gpu(tmp_path)
    assert 'grep -aq "propor: ERROR" "$_out_file"' in t


def test_pyscf_has_no_scf_timing(tmp_path):
    """The SCF-timing instrument is SIESTA's: a PySCF wrapper emits
    neither the tee nor its block.

    Pinned by the instrument's own marker and header, not by the bare
    string ``scf-timing.log`` (2026-08-13): since the cold-sweep
    exception list derives from ``identity.OUR_FILE_PATTERNS`` -- one
    enumeration for both engines -- that NAME appears in every
    wrapper's sweep exceptions, which is correct and harmless (it says
    *if such a file exists, leave it alone*).  The substring assert
    read that as the instrument and failed on a wrapper that has none.
    """
    p = tmp_path / "q.py"
    p.write_text("# fake\n")
    t = runwrap.render_run_wrapper(p)
    assert "_mb_scf_tee" not in t
    assert "SCF per-iteration timing instrument" not in t
    assert "_scf_timing_log=" not in t


# --------------------------------------------------------------------- #
#  Background monitor wiring (§ 11.0b, item F)                         #
# --------------------------------------------------------------------- #


def test_wrapper_launches_low_priority_monitor(tmp_path):
    """SIESTA wrappers background the SELF-CONTAINED mb_monitor.py at
    nice 19 with the JOB's own python (no molbuilder install needed),
    guarded by MB_MONITOR + the shipped file, watching the wrapper PID."""
    t = _gpu(tmp_path)
    assert 'if [ "${MB_MONITOR:-1}" = "1" ]' in t
    assert "[ -f mb_monitor.py ]" in t
    # R9: the interpreter is PROBED (python3-first) -- bare `python`
    # does not exist on python3-only hosts, and the backgrounded 127 was
    # swallowed while the log claimed a live pid.
    assert '_mb_py="$(command -v python3 || command -v python' in t
    assert 'nice -n 19 "$_mb_py" mb_monitor.py' in t     # shipped, not -m
    assert "python -m molbuilder monitor" not in t       # NOT the package form
    assert "--watch-pid $$" in t
    assert "--interval \"${MB_MONITOR_INTERVAL:-5}\"" in t
    # cleaned up by the single unified EXIT trap
    assert '[ -n "${_monitor_pid:-}" ] && kill "$_monitor_pid"' in t


def test_wrapper_ships_standalone_monitor(tmp_path):
    """write_run_wrapper drops a verbatim, stdlib-only copy of the monitor
    next to a SIESTA job as mb_monitor.py (runnable with the job's python).
    A PySCF job gets none."""
    import json
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "source activate"}}))
    fdf = tmp_path / "j.fdf"
    fdf.write_text("NumberOfAtoms 10\nDiag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, resources=Resources(mpi_np=2), emit_sbatch=False)
    shipped = tmp_path / "mb_monitor.py"
    assert shipped.is_file()
    src = (shipped).read_text()
    assert "def run_monitor(" in src and "def main(" in src
    # It is the stdlib-only module -- no molbuilder/numpy imports.
    assert "import numpy" not in src
    assert "import molbuilder" not in src
    # PySCF job: no monitor shipped (siesta-only instrument).
    py = tmp_path / "q.py"; py.write_text("# fake\n")
    (tmp_path / "mb_monitor.py").unlink()
    runwrap.write_run_wrapper(py, resources=Resources(), emit_sbatch=False)
    assert not (tmp_path / "mb_monitor.py").exists()


def test_monitor_killed_in_unified_cleanup(tmp_path):
    """The monitor kill lives in _mb_cleanup (the ONE EXIT trap), not its
    own trap (which would clobber the others).  The D17 signal trap is
    not a second EXIT trap -- see test_single_unified_exit_trap."""
    t = _gpu(tmp_path)
    trap_cmds = [ln.strip() for ln in t.splitlines()
                 if ln.strip().startswith("trap ")]
    exit_traps = [c for c in trap_cmds if c.endswith(" EXIT")]
    assert exit_traps == ["trap _mb_cleanup EXIT"], trap_cmds
