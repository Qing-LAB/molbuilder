"""The [MACHINE] line — what kind of node was under this run.

`scheduler.md` R12: where a job was SENT is `run.json`'s (written at
sbatch-accept, when no node exists yet); what it LANDED ON is the monitor's,
written on the compute node as the log's first line.

The trap this guards would produce a feature that runs, reports, and means
nothing:

* **T2** — the node's size, not the allocation.  ``_alloc_cores()`` reads
  the affinity mask, which the scheduler shrinks to what the job was given;
  a machine record built from it would report a different "machine" per
  trial of a rank-scaling sweep.  On an unconstrained workstation the two
  sources agree, so the guard must constrain affinity itself or it tests
  nothing.

The device COUNT is the allocation's -- inside a scheduled job the device
cgroup shows only granted GPUs -- so only the MODEL may appear: asked of a real
device list in `tests/field/test_the_machine_line.py`.
"""
import os

import pytest

from molbuilder import monitor


# ---------------------------------------------------------------- T2 guard

def test_cores_is_the_node_not_the_allocation():
    """Restrict this process to ONE cpu; the machine must still report all.

    This is the mutation-catcher: swap ``os.cpu_count()`` for
    ``_alloc_cores()`` in ``machine_identity`` and this fails.  Without the
    restriction it cannot fail — on an unconstrained box both read the same
    number (measured 40 = 40 on the dev workstation), which is exactly how
    trap T2 would slip through a local suite.
    """
    if not hasattr(os, "sched_setaffinity"):        # pragma: no cover
        pytest.skip("no affinity control on this platform")
    full = os.sched_getaffinity(0)
    if len(full) < 2:                               # pragma: no cover
        pytest.skip("one visible cpu; the two sources cannot disagree")
    try:
        os.sched_setaffinity(0, {min(full)})
        assert monitor._alloc_cores()[0] == 1, (
            "the fixture failed: affinity was set but _alloc_cores does "
            "not see it, so this test is proving nothing")
        assert int(monitor.machine_identity()["cores"]) == os.cpu_count(), (
            "machine cores followed the ALLOCATION: a rank-scaling sweep "
            "would report a different machine per trial (T2)")
    finally:
        os.sched_setaffinity(0, full)



def test_no_tool_and_no_device_read_the_same(monkeypatch):
    """`nvidia-smi` absent and `nvidia-smi` reporting nothing are one
    answer — this run could not touch a device — so both must yield the
    same record, not an error for one of them."""
    def absent(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(monitor.subprocess, "run", absent)
    assert monitor._gpu_models() == []
    assert monitor.machine_identity()["gpu"] == "none"
    assert monitor._gpu_present() is False
