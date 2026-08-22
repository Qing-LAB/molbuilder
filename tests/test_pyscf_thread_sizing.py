"""The generated PySCF script sizes its threads from the ALLOCATION.

`_mb_count_physical_cores()` counts the whole node.  On a workstation
that is right -- the node IS the allocation.  Under a scheduler it is
wrong and expensively so: a job given 8 cores of a 128-core node used to
start 128 OpenMP threads, which the cgroup then time-slices onto the 8 it
granted.  The job runs slower than an honest 8 would, and the thrashing
is charged to it.  Same code, correct on one posture and wrong on the
other, because it asked the machine instead of the allocation.

These run the EMITTED source rather than inspecting it: the resolution
has to hold when Python executes it, not merely appear in the text.
"""
from __future__ import annotations

import os
import subprocess
import sys

from molbuilder.runtime_info import emit_threading_setup_lines
from molbuilder.jobset.model import Resources

_PROBE = "\nimport os\nprint(os.environ['OMP_NUM_THREADS'], _MB_THREADS_FROM, sep='|')\n"


def _resolve(env: dict, threads=None):
    """Execute the emitted setup under ``env``; return (threads, source)."""
    src = "\n".join(emit_threading_setup_lines(threads)) + _PROBE
    e = {k: v for k, v in os.environ.items()
         if k not in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK",
                      "PBS_NCPUS", "NSLOTS")}
    e.update(env)
    cp = subprocess.run([sys.executable, "-c", src],
                        capture_output=True, text=True, env=e, timeout=60)
    assert cp.returncode == 0, cp.stderr
    n, whence = cp.stdout.strip().splitlines()[-1].split("|")
    return int(n), whence


def test_a_scheduler_allocation_wins_over_the_node():
    """THE bug.  8 granted cores must give 8 threads, not the node's."""
    n, whence = _resolve({"SLURM_CPUS_PER_TASK": "8"})
    assert n == 8
    assert whence == "SLURM_CPUS_PER_TASK"


def test_an_exported_thread_count_wins_over_everything():
    n, whence = _resolve({"OMP_NUM_THREADS": "3", "SLURM_CPUS_PER_TASK": "8"})
    assert n == 3 and whence == "OMP_NUM_THREADS"


def test_a_workstation_still_sizes_from_the_node():
    """No scheduler: the node IS the allocation, so counting it is right."""
    n, whence = _resolve({})
    assert n >= 1
    assert whence == "node physical cores"


def test_an_explicit_config_value_is_honoured_and_says_so():
    n, whence = _resolve({"SLURM_CPUS_PER_TASK": "8"}, threads=5)
    assert n == 5 and "config" in whence


def test_a_garbage_allocation_variable_falls_through():
    """A scheduler variable that is not a positive integer is not a
    number of cores; fall through rather than crash or believe it."""
    n, whence = _resolve({"SLURM_CPUS_PER_TASK": "not-a-number"})
    assert whence == "node physical cores" and n >= 1
    n, whence = _resolve({"SLURM_CPUS_PER_TASK": "0"})
    assert whence == "node physical cores"


def test_the_log_says_where_the_number_came_from():
    """A run that sized itself from the node when it should have read the
    allocation must be distinguishable in the log from one that was
    correctly told that many -- otherwise the failure is invisible."""
    src = "\n".join(emit_threading_setup_lines(None))
    assert "_MB_THREADS_FROM" in src
    assert "from {_MB_THREADS_FROM}" in src


# --------------------------------------------------------------------- #
#  The wrapper side (P1b)                                               #
# --------------------------------------------------------------------- #


def _pyscf_wrapper(tmp_path, monkeypatch):
    """Render a PySCF run-wrapper in an isolated cwd + HOME."""
    import json
    from molbuilder import runwrap
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"script_generation": {"preamble": "true",
                               "activation": "source activate"}}))
    (tmp_path / "job.py").write_text("print('hi')\n")
    return runwrap.render_run_wrapper(tmp_path / "job.py", resources=Resources())


def test_the_wrapper_accepts_the_flags_submit_actually_sends(
        tmp_path, monkeypatch):
    """`jobset launch` hands EVERY run script `-np N -omp M`
    (submit._run_sh_args).  The PySCF parser used to reject both as
    unknown and exit 1, so `submit --mode direct` on a PySCF job with
    resources set died before Python started -- on the workstation
    posture, where direct mode is the normal way to run."""
    t = _pyscf_wrapper(tmp_path, monkeypatch)
    assert "-omp|--omp)" in t, "-omp must be parsed, not rejected"
    assert "-np|--np)" in t, "-np must be tolerated, not rejected"
    # ...and -np must be visibly ignored rather than silently swallowed:
    # PySCF has no MPI ranks, and a user who asked for some should hear.
    assert "OpenMP-only" in t


def test_the_wrapper_exports_the_thread_count_it_resolved(
        tmp_path, monkeypatch):
    """One layer decides.  The script keeps the same chain for a hand
    run, but when the wrapper is in play its answer is the answer."""
    t = _pyscf_wrapper(tmp_path, monkeypatch)
    assert 'export OMP_NUM_THREADS="$_omp_threads"' in t
    # The allocation is consulted BEFORE the node.
    assert t.index("SLURM_CPUS_PER_TASK") < t.index('_omp_from="node physical')


def test_the_wrapper_banner_states_where_the_count_came_from(
        tmp_path, monkeypatch):
    t = _pyscf_wrapper(tmp_path, monkeypatch)
    assert "OMP threads : $_omp_threads (from $_omp_from" in t


def test_both_engines_share_one_core_probe(tmp_path, monkeypatch):
    """`how many cores does this machine have` has one answer; two
    engines probing separately is how they come to disagree."""
    from molbuilder.runwrap import _phys_cores_probe_block
    probe = _phys_cores_probe_block()
    assert "_phys_cores=" in probe and "_cps=" in probe
    assert probe in _pyscf_wrapper(tmp_path, monkeypatch)


# ------------------------------------------------------------------ #
#  The two chains are ONE policy in two languages                     #
#                                                                     #
#  The wrapper resolves and exports OMP_NUM_THREADS; the script keeps  #
#  the same chain for ``python job.py`` run by hand.  The wrapper's    #
#  comment claimed the two were "identical" while it was missing       #
#  PBS_NCPUS and NSLOTS -- and because the wrapper EXPORTS the         #
#  variable, the script's chain saw it already set and never reached   #
#  its own PBS step.  Net effect under qsub: the engine got the whole  #
#  node, which is the 128-threads-for-8-cores bug the block exists to  #
#  prevent, in the one configuration where NOT using the wrapper would #
#  have been correct.                                                  #
#                                                                     #
#  A comment cannot hold two languages in step.  This does.            #
# ------------------------------------------------------------------ #

_CHAIN = ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "PBS_NCPUS", "NSLOTS")


def test_the_script_consults_the_whole_chain_in_order():
    from molbuilder.runtime_info import emit_threading_setup_lines
    text = "\n".join(emit_threading_setup_lines(threads=None))
    at = [text.index(f"'{v}'") for v in _CHAIN]
    assert at == sorted(at), "script chain out of order"
    assert text.index("_mb_count_physical_cores(), 'node physical cores'") \
        > max(at), "the node must be the last resort, not an early answer"


def test_the_wrapper_consults_the_same_chain_in_the_same_order(
        tmp_path, monkeypatch):
    """Every scheduler variable the script knows, the wrapper must know.

    Anything the wrapper omits is not merely unhandled -- it is
    OVERWRITTEN, because exporting OMP_NUM_THREADS disables the script's
    own lookup of it.
    """
    t = _pyscf_wrapper(tmp_path, monkeypatch)
    at = [t.index(f'"${{{v}:-}}"') for v in _CHAIN[1:]]
    assert at == sorted(at), "wrapper chain out of order"
    for v in _CHAIN[1:]:
        assert f'_omp_from="{v}"' in t, (
            f"the wrapper never consults {v}; under that scheduler it "
            f"exports the node's core count and the script -- which DOES "
            f"consult it -- is pre-empted by the export"
        )
    assert t.index('_omp_from="node physical cores"') > max(at)
