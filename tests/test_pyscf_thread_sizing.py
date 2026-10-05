"""A PySCF run sizes its threads from the ALLOCATION.

The node's physical core count is right on a workstation -- the node IS the
allocation.  Under a scheduler it is wrong and expensively so: a job given 8
cores of a 128-core node used to start 128 OpenMP threads, which the cgroup
then time-sliced onto the 8 it granted.  The job runs slower than an honest 8
would, and the thrashing is charged to it.  Same code, correct on one posture
and wrong on the other, because it asked the machine instead of the
allocation.

The script's chain is `runtime_info.threads_for`, the function it imports
from `mb_pyscf.pyz` (`engines/pyscf.md` § 3), asked here directly; the run
script's chain is built from the same list, `runtime_info.THREAD_SOURCES`
(`running-a-job.md` § 3.2).  *(Until 2026-10-05 both chains were text --
the script's emitted by `runtime_info`, run here in a subprocess -- and two
tests held their texts in the same order.)*
"""
from __future__ import annotations

import os

from molbuilder.jobset.model import Resources
from molbuilder.runtime_info import cap_threads, threads_for

#: A machine with no scheduler variable set.
_BARE = {}


def test_a_scheduler_allocation_wins_over_the_node():
    """THE bug.  8 granted cores must give 8 threads, not the node's."""
    assert threads_for(environ={"SLURM_CPUS_PER_TASK": "8"}) == \
        (8, "SLURM_CPUS_PER_TASK")


def test_an_exported_thread_count_wins_over_everything():
    assert threads_for(environ={"OMP_NUM_THREADS": "3",
                                "SLURM_CPUS_PER_TASK": "8"}) == \
        (3, "OMP_NUM_THREADS")


def test_a_workstation_still_sizes_from_the_node():
    """No scheduler: the node IS the allocation, so counting it is right."""
    n, whence = threads_for(environ=_BARE)
    assert n >= 1 and whence == "node physical cores"


def test_an_explicit_config_value_is_honoured_and_says_so():
    n, whence = threads_for(5, environ={"SLURM_CPUS_PER_TASK": "8"})
    assert n == 5 and "config" in whence


def test_a_garbage_allocation_variable_falls_through():
    """A scheduler variable that is not a positive integer is not a
    number of cores; fall through rather than crash or believe it."""
    for said in ("not-a-number", "0"):
        n, whence = threads_for(environ={"SLURM_CPUS_PER_TASK": said})
        assert whence == "node physical cores" and n >= 1, said


def test_the_caps_are_set_and_an_exported_value_wins(monkeypatch):
    """What the script runs before numpy loads: OpenMP at the run's count,
    every BLAS at one -- and a value already exported is left as it is."""
    for var in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "PBS_NCPUS",
                "NSLOTS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        # Recorded first, so what `cap_threads` sets is taken away after.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    n, physical = cap_threads(6)
    assert (n, os.environ["OMP_NUM_THREADS"]) == (6, "6")
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "2", "an exported value wins"
    assert physical >= 1


# --------------------------------------------------------------------- #
#  The wrapper side (P1b)                                               #
# --------------------------------------------------------------------- #


def _pyscf_wrapper(tmp_path, monkeypatch):
    """Render a PySCF run-wrapper in an isolated cwd (conftest isolates the config root), its thread
    count stated (4) as every run's is (`architecture.md` § 5.2)."""
    from molbuilder import runwrap
    from molbuilder.scheduler import Environment, Topology
    monkeypatch.chdir(tmp_path)
    # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
    # working-directory step, which is gone (configuration.md § 2.1a) --
    # without naming the directory the write lands in a file nothing
    # opens, and the test passes having configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    # This machine's record, carrying the activation the generator reads
    # (`configuration.md` § 4).
    (tmp_path / "environment.json").write_text(Environment(
        scheduler="workstation", topology=Topology(),
        env_init={"preamble": "true",
                           "activation": "source activate"}).to_json())
    (tmp_path / "job.py").write_text("print('hi')\n")
    return runwrap.render_run_wrapper(tmp_path / "job.py",
                                      resources=Resources(cpus_per_task=4))


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
    # The allocation is consulted BEFORE the count stated at prep -- the
    # wrapper's last rung since 2026-10-02, when the node's core count
    # stopped standing in for a thread count nobody stated.
    assert t.index("SLURM_CPUS_PER_TASK") < t.index(
        '_omp_threads="4"; _omp_from="stated at prep"')
    assert '_omp_from="node physical' not in t


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


# Retired 2026-10-05: the two tests that held the script's chain and the
# run script's in the same order, text against text -- both are built from
# one list now, `runtime_info.THREAD_SOURCES`.
