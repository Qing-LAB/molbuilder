"""What a reported percentage is a fraction OF.

`plans/bench-and-junction-plan.md` § 2.12. The rule, in the user's words:
*we do need to know if we are using the cpu allocated effectively; we don't
really care if additional cpus are used because it is unpredictable if they
are there.*

Two reasons it matters, and the second is the one that bites. A fraction
taken over the whole node measures the cluster rather than the calculation --
48 ranks on a 128-core node cannot move the node-wide reading past 37.5%, so
the Au-BDT-Au sweep's 32.2% read as idleness when the job was at ~86% of what
it held. And a job that looks starved argues for a bigger machine, which is
the queue this practice exists to stay out of.

**The fixtures are the real thing.** Every cgroup-v1 value below was measured
on an ASU Sol `htc`/`debug` node on 2026-08-26, including the `2**63 - 4096`
no-limit sentinel on the task cgroup and the enforced 8 GiB one level up.
This workstation is cgroup v2, so without fixtures the v1 path -- the one that
actually runs on the cluster -- could not be tested at all.
"""
from __future__ import annotations

import time

import pytest

from molbuilder import monitor as M


# The measured Sol job: /slurm/uid_961166/job_62238108/step_0/task_0
SOL_V1 = """\
12:rdma:/
11:freezer:/slurm/uid_961166/job_62238108/step_0
10:cpuset:/slurm/uid_961166/job_62238108/step_0/task_0
9:hugetlb:/
8:devices:/slurm/uid_961166/job_62238108/step_0/task_0
7:cpu,cpuacct:/slurm/uid_961166/job_62238108/step_0/task_0
6:perf_event:/
5:pids:/system.slice/slurmd.service
4:memory:/slurm/uid_961166/job_62238108/step_0/task_0
3:blkio:/system.slice/slurmd.service
2:net_cls,net_prio:/
1:name=systemd:/system.slice/slurmd.service
"""
SOL_TASK = "/slurm/uid_961166/job_62238108/step_0/task_0"
SOL_JOB = "/slurm/uid_961166/job_62238108"
V2_LINE = "0::/user.slice/user-1000.slice/session-1.scope\n"


@pytest.fixture
def sol(tmp_path, monkeypatch):
    """A cgroup-v1 tree holding exactly what Sol reported."""
    proc = tmp_path / "proc_cgroup"
    proc.write_text(SOL_V1)
    monkeypatch.setattr(M, "_PROC_CGROUP", str(proc))
    monkeypatch.setattr(M, "_CG", str(tmp_path / "cg"))

    cpu = tmp_path / "cg" / "cpu,cpuacct" / SOL_TASK.lstrip("/")
    cpu.mkdir(parents=True)
    (cpu / "cpuacct.usage").write_text("52132694\n")

    task = tmp_path / "cg" / "memory" / SOL_TASK.lstrip("/")
    task.mkdir(parents=True)
    (task / "memory.usage_in_bytes").write_text("1736704\n")
    (task / "memory.max_usage_in_bytes").write_text("2805760\n")
    (task / "memory.limit_in_bytes").write_text("9223372036854771712\n")

    job = tmp_path / "cg" / "memory" / SOL_JOB.lstrip("/")
    (job / "memory.limit_in_bytes").write_text("8589934592\n")
    return tmp_path


# --------------------------------------------------------------------- #
#  reading the layout                                                    #
# --------------------------------------------------------------------- #

def test_the_v1_layout_is_read_per_controller(sol):
    """v1 writes one line per hierarchy. `cpu,cpuacct` must be reachable
    under its joined spelling (which is also the mount directory) AND under
    each controller name, because callers ask for one or the other."""
    cg = M._cgroup_paths()
    assert cg["memory"] == SOL_TASK
    assert cg["cpu,cpuacct"] == SOL_TASK
    assert cg["cpuacct"] == SOL_TASK
    assert cg["cpu"] == SOL_TASK
    assert "" not in cg, "a v1 tree must not look like v2"


def test_the_v2_layout_is_read_as_one_path(tmp_path, monkeypatch):
    proc = tmp_path / "pc"
    proc.write_text(V2_LINE)
    monkeypatch.setattr(M, "_PROC_CGROUP", str(proc))
    cg = M._cgroup_paths()
    assert cg[""] == "/user.slice/user-1000.slice/session-1.scope"


def test_an_unreadable_proc_file_is_empty_not_an_error(tmp_path, monkeypatch):
    """No cgroup is an ordinary state -- a workstation, a container without
    the file. It must fall through to the node reading, never raise."""
    monkeypatch.setattr(M, "_PROC_CGROUP", str(tmp_path / "absent"))
    assert M._cgroup_paths() == {}
    assert M._read_cpu_used_ns()[1] == "node"


# --------------------------------------------------------------------- #
#  the limit, and the sentinel that is not one                           #
# --------------------------------------------------------------------- #

def test_the_limit_comes_from_the_JOB_cgroup_not_the_task(sol):
    """MEASURED: the task cgroup carries the no-limit sentinel while the job
    cgroup one level up carries the real `--mem=8G`. A reader that asks the
    task level gets a number that is not a limit."""
    assert M._read_mem_limit_gb() == 8.0


def test_the_sentinel_reads_as_no_limit_stated(sol, tmp_path):
    """`scheduler.md` R3 -- *an unstated limit never bars* -- and a sentinel
    is an absence wearing a number. Dividing by `2**63` silently reports 0%
    of everything."""
    job = tmp_path / "cg" / "memory" / SOL_JOB.lstrip("/")
    (job / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert M._read_mem_limit_gb() is None


def test_job_cgroup_strips_the_step_and_task(sol):
    assert M._job_cgroup(SOL_TASK) == SOL_JOB
    assert M._job_cgroup(SOL_JOB) == SOL_JOB, "already a job path: unchanged"


# --------------------------------------------------------------------- #
#  the readings themselves                                               #
# --------------------------------------------------------------------- #

def test_cpu_time_is_the_jobs_own(sol):
    used = M._read_cpu_used_ns()
    assert used == (52132694, "cgroup-v1")


def test_memory_is_the_jobs_own_not_the_nodes(sol):
    """`MemTotal - MemAvailable` was every process on the machine, so on a
    shared node it measured other people's jobs as much as this one's."""
    gb, rung = M._read_mem_used_gb()
    assert rung == "cgroup-v1"
    assert gb == round(1736704 / 1073741824.0, 2)


def test_the_kernels_own_peak_file_is_the_one_read(sol):
    """v1 maintains a running peak, so `peak_rss_gb` stops being the largest
    value a 10-second sampler happened to catch.

    Asserted against the measured Sol bytes. Note they are ~2.8 MB and GB is
    kept to two decimals, so both peak and current round to 0.0 here -- the
    reading is exact, the DISPLAY resolution is 10 MB. That is deliberate and
    ample for a calculation whose memory is counted in GB, and the separate
    test below is the one that checks peak and current can differ.
    """
    assert M._read_mem_peak_gb() == round(2805760 / 1073741824.0, 2)


def test_the_peak_is_not_a_copy_of_current(tmp_path, monkeypatch):
    """At the scale a real calculation runs at, the two are different
    numbers -- which is what makes the kernel counter worth reading."""
    proc = tmp_path / "pc"
    proc.write_text(SOL_V1)
    monkeypatch.setattr(M, "_PROC_CGROUP", str(proc))
    monkeypatch.setattr(M, "_CG", str(tmp_path / "cg"))
    d = tmp_path / "cg" / "memory" / SOL_TASK.lstrip("/")
    d.mkdir(parents=True)
    (d / "memory.usage_in_bytes").write_text(str(12 * 1073741824))
    (d / "memory.max_usage_in_bytes").write_text(str(31 * 1073741824))
    cur, _ = M._read_mem_used_gb()
    assert cur == 12.0
    assert M._read_mem_peak_gb() == 31.0


# --------------------------------------------------------------------- #
#  the denominator                                                       #
# --------------------------------------------------------------------- #

def test_the_allocation_is_the_denominator_not_the_node(monkeypatch):
    """Measured on Sol: a `-c 4` job reports exactly 4 through the affinity
    mask on a node with 48 or 128 cores."""
    monkeypatch.setattr(M.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    assert M._alloc_cores() == (4, "affinity")


def test_slurm_answers_when_the_mask_does_not(monkeypatch):
    monkeypatch.setattr(M.os, "sched_getaffinity",
                        lambda _pid: (_ for _ in ()).throw(OSError()))
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")
    assert M._alloc_cores() == (16, "SLURM_CPUS_ON_NODE")


def test_the_node_is_the_last_rung_and_says_so(monkeypatch):
    monkeypatch.setattr(M.os, "sched_getaffinity",
                        lambda _pid: (_ for _ in ()).throw(OSError()))
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
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

def test_the_basis_line_names_every_rung(sol, monkeypatch):
    """**A percentage whose denominator is invisible is how this went wrong
    in the first place.** With two cgroup generations in play, a number that
    does not say where it came from cannot be checked at all."""
    monkeypatch.setattr(M.os, "sched_getaffinity", lambda _pid: set(range(4)))
    line = M.measurement_provenance()
    assert "cpu% of 4 core(s) [affinity]" in line
    assert "cpu time [cgroup-v1]" in line
    assert "mem [cgroup-v1]" in line
    assert "limit 8 GB" in line
    assert "peak" in line


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
