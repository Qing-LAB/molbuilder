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

  * the no-block property -- the subprocess call carries the timeout;
  * a timeout is a REPORTED, temporary failure of the inquiry, never a
    failure of the system (user rule, 2026-08-28);
  * a CPU-only box is a choice, not a fault: quiet;
  * the cache means N open pages cost one query per TTL, not N.
"""
from __future__ import annotations

import subprocess
import types

import pytest

from molbuilder.web.blueprints import system_load as sl


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Every test starts with an expired cache and no residue."""
    monkeypatch.setattr(sl, "_smi_cache",
                        {"t": 0.0, "gpus": [], "error": None})


def _smi_result(stdout="", rc=0, stderr=""):
    r = types.SimpleNamespace()
    r.returncode, r.stdout, r.stderr = rc, stdout, stderr
    return r


_ROW = ("NVIDIA GeForce RTX 3060 Ti, 37, 12, 4867, 8192, 118.4, 61, "
        "1890, 6801\n")


# ------------------------------------------------------------ the snapshot

def test_snapshot_returns_documented_shape():
    """Public ``snapshot()`` returns the dict the widget consumes."""
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


def test_a_parsed_row_carries_the_fields_the_widget_draws(monkeypatch):
    monkeypatch.setattr(sl.subprocess, "run",
                        lambda *a, **kw: _smi_result(_ROW))
    gpus, err = sl._gpu_snapshot()
    assert err is None and len(gpus) == 1
    g = gpus[0]
    assert g["name"] == "NVIDIA GeForce RTX 3060 Ti"
    assert g["util_pct"] == 37.0 and g["util_mem_pct"] == 12.0
    assert g["mem_used_mb"] == 4867.0 and g["mem_total_mb"] == 8192.0
    assert g["mem_pct"] == pytest.approx(59.4, abs=0.1)
    assert g["power_w"] == 118.4 and g["temp_c"] == 61
    assert g["sm_clock_mhz"] == 1890 and g["mem_clock_mhz"] == 6801


def test_a_field_the_card_lacks_reads_None_not_a_crash(monkeypatch):
    """Consumer cards print ``[N/A]`` for fields they do not expose."""
    row = "Some GPU, 5, 1, 100, 1000, [N/A], 40, [Not Supported], 800\n"
    monkeypatch.setattr(sl.subprocess, "run",
                        lambda *a, **kw: _smi_result(row))
    gpus, err = sl._gpu_snapshot()
    assert err is None
    assert gpus[0]["power_w"] is None
    assert gpus[0]["sm_clock_mhz"] is None
    assert gpus[0]["mem_clock_mhz"] == 800


# ------------------------------------------- the no-block property itself

def test_the_query_always_carries_the_hard_timeout(monkeypatch):
    """THE property this module exists to have (2026-08-28): the driver
    call happens in a child the server can abandon.  A ``run`` without a
    timeout is the in-process bug wearing a subprocess costume."""
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw); seen["cmd"] = cmd
        return _smi_result(_ROW)

    monkeypatch.setattr(sl.subprocess, "run", fake_run)
    sl._gpu_snapshot()
    assert seen.get("timeout") == sl._SMI_TIMEOUT_S, (
        "the GPU query ran with no hard timeout -- one held driver and "
        "every request thread is gone again")
    assert seen["cmd"][0] == "nvidia-smi"


def test_a_timeout_is_reported_and_the_server_keeps_serving(monkeypatch):
    """A temporary failure of the inquiry, never of the system: the
    snapshot completes, the reason rides ``gpu_error``, and the widget
    prints it where the cells would have been."""
    def hang(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi",
                                        timeout=sl._SMI_TIMEOUT_S)
    monkeypatch.setattr(sl.subprocess, "run", hang)
    gpus, err = sl._gpu_snapshot()
    assert gpus == []
    assert err and "timed out" in err
    snap = sl.snapshot()                 # the system did not fail
    assert snap["gpu_error"] == err


def test_a_cpu_only_box_is_a_choice_not_a_fault(monkeypatch):
    """No ``nvidia-smi`` on PATH: nothing is wrong, the widget drops its
    GPU cells, and nothing cries wolf."""
    def absent(*a, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(sl.subprocess, "run", absent)
    gpus, err = sl._gpu_snapshot()
    assert gpus == [] and err is None


def test_a_tool_that_fails_is_loud_on_the_wire(monkeypatch):
    """The case that hid for five weeks in 2026: driver broken, empty
    list indistinguishable from a CPU-only box.  The reason must reach
    ``gpu_error``."""
    monkeypatch.setattr(
        sl.subprocess, "run",
        lambda *a, **kw: _smi_result(
            "", rc=15, stderr="NVML/RM version mismatch"))
    gpus, err = sl._gpu_snapshot()
    assert gpus == []
    assert err and "version mismatch" in err and "15" in err


# ------------------------------------------------------------- the cache

def test_open_pages_share_one_query_per_ttl(monkeypatch):
    """Several widgets polling must not become several subprocesses --
    the second call inside the TTL serves the cached answer."""
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return _smi_result(_ROW)

    monkeypatch.setattr(sl.subprocess, "run", counting)
    first, _ = sl._gpu_snapshot()
    second, _ = sl._gpu_snapshot()
    assert calls["n"] == 1, "the cache did not serve the second poll"
    assert second == first


# ---------------------------------------------------------- the endpoint

def test_endpoint_returns_envelope(monkeypatch):
    from molbuilder.web.app import create_app
    client = create_app(config={}).test_client()
    r = client.get("/api/system/load")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert isinstance(body["data"]["cpu_pct"], (int, float))


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


def test_the_endpoint_carries_the_reason_through_json(monkeypatch):
    """The widget reads ``gpu_error`` off the response, not off a global
    -- so the field has to survive the jsonify layer."""
    from molbuilder.web.app import create_app

    monkeypatch.setattr(sl, "_gpu_snapshot",
                        lambda: ([], "GPU query timed out after 2s"))
    client = create_app(config={}).test_client()
    body = client.get("/api/system/load").get_json()
    assert body["ok"] is True
    assert body["data"]["gpu_error"] == "GPU query timed out after 2s"
