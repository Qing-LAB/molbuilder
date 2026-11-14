"""L1 tests for ``web/blueprints/system_load.py`` -- the bottom-strip
load-monitor backend.

The GPU half changed shape on 2026-08-28: it was an in-process ``pynvml``
call from request threads, and a request thread that entered the driver
during another user's GPU work never came back -- NVML has no timeout
anywhere in its API, a thread blocked in a driver ioctl cannot be
cancelled from Python, and the whole web server froze for eight hours
behind one page widget.  The query is a SUBPROCESS with a hard timeout
now, behind a small cache, so the guards here are named for the property
each protects:

  * the no-block property: a query that hangs is abandoned, and that is a
    REPORTED, temporary failure of the inquiry, never a failure of the system
    (user rule, 2026-08-28);
  * a CPU-only box is a choice, not a fault: quiet;
  * the cache means N open pages cost one query per TTL, not N.

What the tool prints when it answers is read from the tool, where it runs:
`tests/field/test_the_load_widget.py`.
"""
from __future__ import annotations

import time

import pytest

from molbuilder.web.blueprints import system_load as sl
from support.machines import (DRIVER_MISMATCH, a_gpu_tool,
                              gpu_reading_asked_anew)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Every test starts with an expired cache and no residue."""
    gpu_reading_asked_anew(monkeypatch)


# ------------------------------------------------------------ the snapshot

def test_snapshot_returns_documented_shape():
    """Public ``snapshot()`` returns the dict the widget consumes.  THE
    SILENT FAILURE: a key renamed -- the load card skips what it cannot find,
    so it drops the line (the load average's) without a word."""
    snap = sl.snapshot()
    for key in ("cpu_pct",
                "cpu_count_physical", "cpu_count_logical",
                "loadavg_1m", "loadavg_5m", "loadavg_15m",
                "ram_pct", "ram_used_gb", "ram_total_gb",
                "gpus", "gpu_error"):
        assert key in snap, f"snapshot missing required key {key!r}"
    assert isinstance(snap["cpu_pct"],      (int, float))
    assert isinstance(snap["ram_pct"],      (int, float))
    assert isinstance(snap["gpus"], list)
    for key in ("loadavg_1m", "loadavg_5m", "loadavg_15m"):
        v = snap[key]
        assert v is None or isinstance(v, (int, float))
    assert snap["cpu_count_physical"] >= 1
    assert snap["cpu_count_logical"]  >= snap["cpu_count_physical"]


def test_a_field_the_card_does_not_expose_reads_None(tmp_path, monkeypatch):
    """A card prints ``[N/A]`` for what it does not expose -- a consumer
    card's power, a MIG instance's utilization: no number made up, and its
    other fields kept, by both samplers, the server's load widget and the
    monitor beside a job (`runtime_info.gpu_reading`, the one reading).
    THE SILENT FAILURE: the monitor read every field with float(), so one
    `[N/A]` dropped the whole sample -- every GPU's -- for the run."""
    from molbuilder import monitor
    a_gpu_tool(tmp_path, monkeypatch,
               "case \"$*\" in\n"
               "  *index,*) echo '0, [N/A], [N/A], 2048' ;;\n"
               "  *) echo 'Card, [N/A], [N/A], 2048, 8192, [N/A], 40, "
               "[N/A], [N/A]' ;;\n"
               "esac")
    sample = monitor._sample_gpus()
    assert sample == [(0, None, None, 2.0)]
    # ...and the run's end-of-run average takes it -- the watch loop adds
    # every tick's sample, and a utilization the card never gave is no
    # number to add.
    acc = monitor.UtilAccum()
    acc.add(monitor.UtilSample(epoch=0.0, cpu_pct=50.0, mem_gb=1.0,
                               gpus=sample))
    from molbuilder.monitor_log import GPU_MEAN
    assert f"gpu0{GPU_MEAN}" not in acc.summary(), acc.summary()
    gpus, err = sl._gpu_snapshot()
    assert err is None, err
    assert (gpus[0]["util_pct"], gpus[0]["mem_used_mb"]) == (None, 2048.0)


# ------------------------------------------- the no-block property itself

def test_a_query_that_hangs_is_abandoned_and_said(tmp_path, monkeypatch):
    """THE property this module exists to have (2026-08-28): the driver call
    happens in a child the server can abandon.  One held driver with no timeout
    and every request thread is gone again.  A temporary failure of the
    inquiry, never of the system: the snapshot completes, and the reason rides
    ``gpu_error`` where the cells would have been."""
    a_gpu_tool(tmp_path, monkeypatch, "exec sleep 60")
    monkeypatch.setattr(sl, "_SMI_TIMEOUT_S", 0.2)
    started = time.monotonic()
    gpus, err = sl._gpu_snapshot()
    assert time.monotonic() - started < 10
    assert gpus == []
    assert err and "timed out" in err
    assert sl.snapshot()["gpu_error"] == err


#: The tool on a host whose driver and library diverged, as measured (its
#: reason on stdout), and the same with its reason on stderr.
_EITHER_STREAM = [DRIVER_MISMATCH, DRIVER_MISMATCH.replace("\n", " >&2\n", 1)]


@pytest.mark.parametrize("tool", _EITHER_STREAM, ids=["stdout", "stderr"])
def test_the_reason_is_read_off_whichever_stream_carried_it(tmp_path, monkeypatch,
                                                            tool):
    """nvidia-smi puts its failure on STDOUT, and the reason must survive.

    Measured 2026-09-18 on a host whose driver and library had diverged:
    exit 18, an EMPTY stderr, and *"Failed to initialize NVML:
    Driver/library version mismatch"* on stdout.  Reading only stderr
    rendered that as ``nvidia-smi exited 18: no message`` -- the module's own
    rule is that degrading gracefully is right and degrading SILENTLY is not,
    and the reason was being dropped one line short of the widget.  An empty
    list with no reason cannot be told from a CPU-only box.
    """
    a_gpu_tool(tmp_path, monkeypatch, tool)
    gpus, err = sl._gpu_snapshot()
    assert gpus == []
    assert "Driver/library version mismatch" in err and "18" in err, (
        f"the reason nvidia-smi gave must reach the widget; got {err!r}")
    assert "no message" not in err


def test_a_cpu_only_box_is_a_choice_not_a_fault(tmp_path, monkeypatch):
    """No ``nvidia-smi`` on PATH: nothing is wrong, the widget drops its
    GPU cells, and nothing cries wolf."""
    monkeypatch.setenv("PATH", str(tmp_path))
    gpus, err = sl._gpu_snapshot()
    assert gpus == [] and err is None


# ------------------------------------------------------------- the cache

def test_open_pages_share_one_query_per_ttl(tmp_path, monkeypatch):
    """Several widgets polling must not become several subprocesses --
    the second call inside the TTL serves the cached answer."""
    asked = tmp_path / "asked"
    a_gpu_tool(tmp_path, monkeypatch, f'echo once >> "{asked}"')
    first, _ = sl._gpu_snapshot()
    second, _ = sl._gpu_snapshot()
    assert asked.read_text().count("once") == 1, (
        "the cache did not serve the second poll")
    assert second == first


# ---------------------------------------------------------- the endpoint


def test_snapshot_endpoint_handles_psutil_failure(monkeypatch):
    from molbuilder.web.app import create_app

    def _boom():
        raise RuntimeError("simulated psutil crash")

    monkeypatch.setattr(sl, "snapshot", _boom)
    client = create_app(config={}).test_client()
    r = client.get("/api/system/load")
    assert r.status_code == 500
    body = r.get_json()
    assert body["ok"] is False
    assert "simulated psutil crash" in body["error"]


def test_the_endpoint_carries_the_reason_through_json(tmp_path, monkeypatch):
    """The widget reads ``gpu_error`` off the response, not off a global
    -- so the field has to survive the jsonify layer: the reason the tool
    gave, as the sampler composed it, on a host whose driver and library
    diverged (`support.machines.DRIVER_MISMATCH`)."""
    from molbuilder.web.app import create_app

    a_gpu_tool(tmp_path, monkeypatch, DRIVER_MISMATCH)
    client = create_app(config={}).test_client()
    body = client.get("/api/system/load").get_json()
    assert body["ok"] is True
    assert body["data"]["gpu_error"] == (
        "nvidia-smi exited 18: "
        "Failed to initialize NVML: Driver/library version mismatch")
