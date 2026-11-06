"""FIELD TEST -- the monitor reads this job's own numbers, from the cgroup of
whichever generation the target has (`execution/run-reports.md` § 2.1a).

Goal: a percentage is a fraction of what the job holds.  What a cgroup holds is
read where it is, so the first test runs on any Linux target, and the others
only inside a scheduled job, whose request they compare the readings with:

    srun --mem=8G -c 4 env MOLBUILDER_FIELD_RECORD=<the record> \\
        python tools/testrun.py run tests/field/test_the_monitor_reads_its_job.py

Measured on ASU Sol 2026-08-26, cgroup v1: the task's cgroup carried the no-limit
sentinel while the job's, one level up, carried the real `--mem=8G`.
"""
from __future__ import annotations

import os

import pytest

from molbuilder import monitor as M

in_a_job = pytest.mark.skipif("SLURM_JOB_ID" not in os.environ,
                              reason="not inside a scheduled job")


def test_the_readings_are_this_processes_cgroup():
    if not M._cgroup_paths():
        pytest.skip("this machine states no cgroup for this process")
    assert M._read_cpu_used_ns()[1].startswith("cgroup-")
    assert M._read_mem_used_gb()[1].startswith("cgroup-")


def test_a_limit_is_never_more_than_the_machine_holds():
    """`scheduler.md` R3: an unstated limit never bars, and a no-limit value is
    an absence wearing a number (v1's `2**63 - 4096`, v2's `max`)."""
    limit = M._read_mem_limit_gb()
    pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert limit is None or limit <= pages / 1073741824.0 + 0.01


@in_a_job
def test_the_limit_is_the_memory_the_job_asked_for():
    asked_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if asked_mb is None:
        pytest.skip("this job did not ask with --mem")
    limit = M._read_mem_limit_gb()
    if limit is None:
        pytest.skip("this site's scheduler enforces no memory by a cgroup")
    assert limit == pytest.approx(int(asked_mb) / 1024, abs=0.01)


@in_a_job
def test_the_denominator_is_the_cores_the_job_holds():
    asked = os.environ.get("SLURM_CPUS_PER_TASK")
    if asked is None:
        pytest.skip("this job did not ask with -c")
    assert M._alloc_cores()[0] == int(asked)


@in_a_job
def test_the_peak_is_the_kernels_own():
    """Where the kernel keeps one, it is no copy of what is held now: memory
    held and let go stays in the peak."""
    size = 256 * 1024 * 1024
    block = bytearray(size)
    block[::4096] = bytes(1 for _ in range(0, size, 4096))
    del block
    used = M._read_mem_used_gb()[0]
    peak = M._read_mem_peak_gb()
    if peak is None:
        pytest.skip("this kernel keeps no peak")
    assert peak > used
