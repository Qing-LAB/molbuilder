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

import os
import time

import pytest

from molbuilder.web.blueprints import system_load as sl


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Every test starts with an expired cache and no residue."""
    monkeypatch.setattr(sl, "_smi_cache",
                        {"t": 0.0, "gpus": [], "error": None})


def _a_tool(tmp_path, monkeypatch, body):
    """``nvidia-smi`` on this test's PATH, doing ``body``."""
    tool = tmp_path / "nvidia-smi"
    tool.write_text(f"#!/bin/sh\n{body}\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


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


def test_a_field_the_card_does_not_expose_reads_None():
    """Consumer cards print ``[N/A]`` for what they do not expose: no number
    made up."""
    assert sl._num("[N/A]") is None


# ------------------------------------------- the no-block property itself

def test_a_query_that_hangs_is_abandoned_and_said(tmp_path, monkeypatch):
    """THE property this module exists to have (2026-08-28): the driver call
    happens in a child the server can abandon.  One held driver with no timeout
    and every request thread is gone again.  A temporary failure of the
    inquiry, never of the system: the snapshot completes, and the reason rides
    ``gpu_error`` where the cells would have been."""
    _a_tool(tmp_path, monkeypatch, "exec sleep 60")
    monkeypatch.setattr(sl, "_SMI_TIMEOUT_S", 0.2)
    started = time.monotonic()
    gpus, err = sl._gpu_snapshot()
    assert time.monotonic() - started < 10
    assert gpus == []
    assert err and "timed out" in err
    assert sl.snapshot()["gpu_error"] == err


@pytest.mark.parametrize("to_stream", ["", " >&2"], ids=["stdout", "stderr"])
def test_the_reason_is_read_off_whichever_stream_carried_it(tmp_path, monkeypatch,
                                                            to_stream):
    """nvidia-smi puts its failure on STDOUT, and the reason must survive.

    Measured 2026-09-18 on a host whose driver and library had diverged:
    exit 18, an EMPTY stderr, and *"Failed to initialize NVML:
    Driver/library version mismatch"* on stdout.  Reading only stderr
    rendered that as ``nvidia-smi exited 18: no message`` -- the module's own
    rule is that degrading gracefully is right and degrading SILENTLY is not,
    and the reason was being dropped one line short of the widget.  An empty
    list with no reason cannot be told from a CPU-only box.
    """
    _a_tool(tmp_path, monkeypatch,
            "echo 'Failed to initialize NVML: Driver/library version mismatch'"
            f"{to_stream}\nexit 18")
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
    _a_tool(tmp_path, monkeypatch, f'echo once >> "{asked}"')
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
