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
