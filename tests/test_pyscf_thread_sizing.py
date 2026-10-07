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
(`running-a-job.md` § 3.2).
"""
from __future__ import annotations

import os

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
