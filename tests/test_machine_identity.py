"""The [MACHINE] line — what kind of node was under this run.

`scheduler.md` R12: where a job was SENT is `run.json`'s (written at
sbatch-accept, when no node exists yet); what it LANDED ON is the monitor's,
written on the compute node as the log's first line.

The guards here are named for the two traps in `archive/2026-09-01-machine-identity-plan.md`
§ 2, because each would produce a feature that runs, reports, and means
nothing:

* **T2** — the node's size, not the allocation.  ``_alloc_cores()`` reads
  the affinity mask, which the scheduler shrinks to what the job was given;
  a machine record built from it would report a different "machine" per
  trial of a rank-scaling sweep.  On an unconstrained workstation the two
  sources agree, so the guard must constrain affinity itself or it tests
  nothing.
* the device COUNT is the allocation's — inside a scheduled job the device
  cgroup shows only granted GPUs — so only the MODEL may appear.
"""
import os
import subprocess

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


# ------------------------------------------------------- model, never count

def test_gpu_models_dedup_and_never_a_count(monkeypatch):
    two_of_a_kind = (
        "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-aaa)\n"
        "GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-bbb)\n"
        "GPU 2: NVIDIA H200 (UUID: GPU-ccc)\n")

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = two_of_a_kind
        assert cmd[:2] == ["nvidia-smi", "-L"]
        return R()

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)
    models = monitor._gpu_models()
    assert models == ["NVIDIA A100-SXM4-80GB", "NVIDIA H200"], (
        "distinct models in first-seen order — a repeated model must not "
        "repeat, because two visible A100s is a fact about the ALLOCATION")
    assert (monitor.machine_line().split("gpu=", 1)[1]
            == "NVIDIA A100-SXM4-80GB, NVIDIA H200"), (
        "the gpu field must be exactly the model list — a count prefix "
        "(e.g. '2x') would be the allocation's count, not the node's")


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


# ------------------------------------------------------------- line format

def test_machine_line_puts_the_model_last(monkeypatch):
    """Device models contain spaces, so the reader takes everything after
    ``gpu=`` as the model list — legal only while gpu stays the last field.

    **Asked of the REAL reader.** This carried its own copy of the
    reader's regex until 2026-09-04 — a third spelling of one pattern,
    beside the writer and the parser — so a change to the READER left it
    green while every trial silently lost its machine. The parser is
    `parse/instruments/monitor.py`; feeding it the writer's own output is
    the only form of this test that fails in both directions.
    """
    from molbuilder.parse.instruments.monitor import monitor_metrics

    monkeypatch.setattr(monitor, "_gpu_models",
                        lambda: ["NVIDIA A100-SXM4-80GB"])
    line = monitor.machine_line()
    m = monitor_metrics(f"[2026-01-01T00:00:00] [MACHINE] {line}")["machine"]
    assert m, f"un-parseable [MACHINE] payload: {line!r}"
    assert m["gpu"] == "NVIDIA A100-SXM4-80GB"


def test_the_machine_is_the_logs_first_line(tmp_path):
    """Written at START, not in the terminal block: a monitor killed with
    its allocation must still have said where it died.  First line, so the
    reader's cheapest scan finds it."""
    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"
    it = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    last = [0.0]

    def clock():                       # holds the last value when drained
        last[0] = next(it, last[0])
        return last[0]

    monitor.run_monitor(out, timing, log, interval=1,
                        watch_pid=999_999_999,      # gone on tick 1
                        sleep=lambda s: None, clock=clock)
    first = log.read_text(encoding="utf-8").splitlines()[0]
    assert "[MACHINE]" in first, (
        "the machine record moved off the first line; a killed run "
        "loses it if it waits for the terminal block")
    assert f"node={os.uname().nodename[:128]}" in first


def test_a_monitor_missing_its_companion_still_monitors(tmp_path, monkeypatch):
    """`config_dir.py` travels beside the shipped monitor; when a staging
    defect loses it, WHERE reports go cannot be answered -- and the answer
    is *reports off* (`run-reports.md`: absent is off), never startup
    death.  Death cost every [STATUS], the util series and the [MACHINE]
    record of every production run for two days, silently
    (stderr goes to /dev/null under the wrapper)."""
    def _no_companion():
        raise ModuleNotFoundError("config_dir")
    monkeypatch.setattr(monitor, "_config_dir", _no_companion)
    monkeypatch.delenv("MOLBUILDER_NOTIFY_FILE", raising=False)

    out = tmp_path / "j.out"
    out.write_text("scf:   1   -100.5\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -100.5\n")
    log = tmp_path / "j.monitor.log"
    it = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]); last = [0.0]

    def clock():
        last[0] = next(it, last[0]); return last[0]

    monitor.run_monitor(out, timing, log, interval=1,
                        watch_pid=999_999_999,
                        sleep=lambda s: None, clock=clock)
    text = log.read_text(encoding="utf-8")
    assert "[MACHINE]" in text and "[MONITOR]" in text, (
        "the monitor died over its missing notify companion")
    assert "reports off" in text, (
        "the absence must be SAID in the log the user reads")
