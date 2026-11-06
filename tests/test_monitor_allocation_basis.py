"""What a reported percentage is a fraction OF.

`execution/run-reports.md` § 2.1a.  The rule, in the user's words:
*we do need to know if we are using the cpu allocated effectively; we don't
really care if additional cpus are used because it is unpredictable if they
are there.*

Two reasons it matters, and the second is the one that bites. A fraction
taken over the whole node measures the cluster rather than the calculation --
48 ranks on a 128-core node cannot move the node-wide reading past 37.5%, so
the Au-BDT-Au sweep's 32.2% read as idleness when the job was at ~86% of what
it held. And a job that looks starved argues for a bigger machine, which is
the queue this practice exists to stay out of.

What a cgroup holds is read where it is: this machine's here, and a scheduled
job's in `tests/field/test_the_monitor_reads_its_job.py`, on the target.
"""
from __future__ import annotations

import time

from molbuilder import monitor as M


# --------------------------------------------------------------------- #
#  the readings                                                          #
# --------------------------------------------------------------------- #

def test_what_this_machine_reads_says_where_it_came_from():
    """**A percentage whose denominator is invisible is how this went wrong in
    the first place.** Whatever this machine is, each reading is a number and
    names the source that answered, and the basis line says the same names, so
    a number can be checked."""
    used_ns, cpu_from = M._read_cpu_used_ns()
    used_gb, mem_from = M._read_mem_used_gb()
    assert used_ns >= 0 and used_gb >= 0
    line = M.measurement_provenance()
    assert f"cpu time [{cpu_from}]" in line
    assert f"mem [{mem_from}]" in line
    assert "core(s) [" in line


def test_an_unreadable_proc_file_is_empty_not_an_error(tmp_path, monkeypatch):
    """No cgroup is an ordinary state -- a workstation, a container without
    the file. It must fall through to the node reading, never raise."""
    monkeypatch.setattr(M, "_PROC_CGROUP", str(tmp_path / "absent"))
    assert M._cgroup_paths() == {}
    assert M._read_cpu_used_ns()[1] == "node"


# --------------------------------------------------------------------- #
#  the denominator                                                       #
# --------------------------------------------------------------------- #

def test_the_allocation_is_the_denominator_not_the_node(monkeypatch):
    """Measured on Sol: a `-c 4` job reports exactly 4 through the affinity
    mask on a node with 48 or 128 cores."""
    monkeypatch.setattr(M.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    assert M._alloc_cores() == (4, "affinity")


def test_the_schedulers_grant_answers_when_the_mask_does_not(monkeypatch):
    """The cores the scheduler granted, in molbuilder's name -- as the job's
    header exports it (`scheduler/kinds.py`, `JOB_VARIABLES`)."""
    monkeypatch.setattr(M.os, "sched_getaffinity",
                        lambda _pid: (_ for _ in ()).throw(OSError()))
    monkeypatch.setenv("MB_CPUS_ON_NODE", "16")
    assert M._alloc_cores() == (16, "MB_CPUS_ON_NODE")


def test_the_node_is_the_last_rung_and_says_so(monkeypatch):
    monkeypatch.setattr(M.os, "sched_getaffinity",
                        lambda _pid: (_ for _ in ()).throw(OSError()))
    monkeypatch.delenv("MB_CPUS_ON_NODE", raising=False)
    monkeypatch.delenv("MB_CPUS_PER_TASK", raising=False)
    n, rung = M._alloc_cores()
    assert rung == "node" and n >= 1


# --------------------------------------------------------------------- #
#  what the percentage actually comes out as                             #
# --------------------------------------------------------------------- #

def test_one_busy_core_of_four_reads_as_25_percent(monkeypatch):
    """The arithmetic, end to end, with time and cpu-time both controlled.

    One core fully busy for 10 s on a 4-core allocation is 25%. On the SAME
    numbers with the node's 128 cores as the denominator it would read 0.8%
    -- which is the reading that made a saturated job look idle.
    """
    monkeypatch.setattr(M.os, "sched_getaffinity", lambda _pid: set(range(4)))
    ticks = iter([100.0, 110.0])
    cpu_ns = iter([(0, "cgroup-v1"), (10 * 10**9, "cgroup-v1")])
    monkeypatch.setattr(M, "_read_cpu_used_ns", lambda: next(cpu_ns))
    monkeypatch.setattr(M, "_read_mem_used_gb", lambda: (1.0, "cgroup-v1"))
    monkeypatch.setattr(M, "_gpu_present", lambda: False)

    s = M._make_default_sampler(lambda: next(ticks))
    assert s().cpu_pct == 25.0


def test_the_denominator_is_fixed_once_and_cannot_drift(monkeypatch):
    """Resolved at construction, never per tick: a percentage that changed
    denominator mid-series would silently change meaning, and nothing in the
    file would show it."""
    calls = []

    def _alloc():
        calls.append(1)
        return 4, "affinity"

    monkeypatch.setattr(M, "_alloc_cores", _alloc)
    monkeypatch.setattr(M, "_read_cpu_used_ns", lambda: (0, "cgroup-v1"))
    monkeypatch.setattr(M, "_read_mem_used_gb", lambda: (1.0, "cgroup-v1"))
    monkeypatch.setattr(M, "_gpu_present", lambda: False)
    s = M._make_default_sampler(lambda: time.time())
    for _ in range(5):
        s()
    assert len(calls) == 1, "the allocation was re-read during the run"


# --------------------------------------------------------------------- #
#  saying which rung answered                                            #
# --------------------------------------------------------------------- #

def test_a_process_tree_is_named_even_when_the_last_read_met_an_exit(
        monkeypatch):
    """A run started directly is read from its process tree, and the line
    says so however one last read went: at the end the wrapper spawns
    `sleep`s while it waits for the monitor, and a read that meets one
    exiting returns nothing.  A PySCF water run closed on "cpu time
    [unavailable]" (2026-09-26, `test_vibration_e2e.py`).  API-level: the
    race is timing, which the road reaches only by chance."""
    monkeypatch.setattr(M, "_read_tree_cpu_ns", lambda root, skip: None)
    monkeypatch.setattr(M, "_read_tree_mem_gb", lambda root, skip: (0.5, "Pss"))
    line = M.measurement_provenance(M._Basis(8, "launched on", tree=1))
    assert "cpu time [process tree]" in line, line


def test_the_basis_line_admits_when_it_is_the_NODE(tmp_path, monkeypatch):
    """The fallback must be visible. A node-wide number presented without
    that label is exactly the defect this change exists to end."""
    monkeypatch.setattr(M, "_PROC_CGROUP", str(tmp_path / "absent"))
    line = M.measurement_provenance()
    assert "cpu time [node]" in line
    assert "mem [node]" in line


def test_the_basis_line_says_when_no_limit_is_stated(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_PROC_CGROUP", str(tmp_path / "absent"))
    assert "limit not stated" in M.measurement_provenance()
