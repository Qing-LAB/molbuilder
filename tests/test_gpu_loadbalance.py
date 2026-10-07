"""Static-text contract tests for the GPU load-balance + --dry-run
additions to the SIESTA run-wrapper (execution/running-a-job.md § 3.3).

These assert on the GENERATED bash.
"""
from __future__ import annotations

import sys
from pathlib import Path


from molbuilder import runwrap
from molbuilder.jobset.model import Resources
from molbuilder.diagnostics import Capabilities
import pytest
from molbuilder.diagnostics import set_capabilities
from molbuilder.runfiles import RunNames


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    """This machine's record (its activation: the refuse-to-emit contract)
    + synthetic caps."""
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT: without naming the directory the
    # write lands in a file nothing opens, and the test passes having
    # configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    # The record follows the config root -- and carries the activation, the
    # probe's copy the generator reads (`configuration.md` § 4).
    from conftest import write_machine_record
    write_machine_record(env_init={"activation": "source activate"})
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/usr/bin/conda",
        conda_envs=frozenset({"molbuilder-siesta", "molbuilder-siesta-gpu"}),
    ))
    yield


def _gpu(tmp_path: Path, np: int = 4) -> str:
    names = RunNames.of("g", "01_coarse", "hierarchical")
    f = tmp_path / names.name(".fdf")
    f.write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    # a GPU job as `resolve` writes one: told, with its count (`gpu.md` G5)
    return runwrap.render_run_wrapper(
        f, names=names,
        resources=Resources(mpi_np=np, cpus_per_task=1, use_gpu=True,
                               gres="gpu:1"))


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


# --------------------------------------------------------------------- #
#  Background monitor wiring (§ 11.0b, item F)                         #
# --------------------------------------------------------------------- #


def test_wrapper_ships_standalone_monitor(tmp_path):
    """write_run_wrapper drops ONE file next to the job, mb_monitor.pyz --
    a Python zip application holding a verbatim, stdlib-only copy of the
    monitor with the framework modules it reads the run through, each its
    own file, runnable with the job's python -- beside every engine's job
    (`run-reports.md` § 2.3)."""
    names = RunNames.of("j", "01_coarse", "hierarchical")
    fdf = tmp_path / names.name(".fdf")
    fdf.write_text("NumberOfAtoms 10\nDiag.ELPA.GPU .true.\n")
    runwrap.write_run_wrapper(fdf, names=names,
                              resources=Resources(mpi_np=2, cpus_per_task=1),
                              emit_sbatch=False)
    shipped = tmp_path / runwrap.MONITOR_BUNDLE
    assert shipped.is_file()
    import zipfile
    with zipfile.ZipFile(shipped) as z:
        assert set(z.namelist()) == {"__main__.py",
                                     *runwrap.MONITOR_COMPANIONS}
        src = z.read("mb_monitor.py").decode("utf-8")
    assert "def run_monitor(" in src and "def main(" in src
    # STDLIB-ONLY IS PROVEN BY RUNNING IT, not by reading it: run the artifact under the condition it exists for.  `molbuilder` and
    # `numpy` are denied at the import system, which is what a compute node
    # does by simply not having them, and the shipped file has to reach
    # argparse anyway.  A real top-level import of either fails this with
    # `ImportError: numpy is not installed on a compute node`.
    blocked = tmp_path / "_blocked_run.py"
    blocked.write_text(
        "import sys, runpy\n"
        "class _Deny:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('molbuilder', 'numpy'):\n"
        "            raise ImportError(name + ' is not installed on a "
        "compute node')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Deny())\n"
        "sys.argv = ['mb_monitor.pyz', '--help']\n"
        "runpy.run_path('mb_monitor.pyz', run_name='__main__')\n"
    )
    import subprocess as _sp
    cp = _sp.run([sys.executable, str(blocked)], cwd=str(tmp_path),
                 capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, (
        "the shipped monitor does not start on a machine without molbuilder "
        "or numpy -- which is every compute node it runs on:\n"
        + cp.stderr[-2000:])
    # A PySCF job gets the SAME monitor and the same readers: one monitor,
    # every engine -- and, one file, nothing of it can be left behind.
    q = RunNames.of("q", "01_coarse", "hierarchical")
    py = tmp_path / q.name(".py"); py.write_text("# fake\n")
    shipped.unlink()
    runwrap.write_run_wrapper(py, names=q,
                              resources=Resources(cpus_per_task=1),
                              emit_sbatch=False)
    assert shipped.read_bytes() == runwrap.monitor_bundle()
