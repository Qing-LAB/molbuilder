"""L1 tests for ``web/blueprints/system_load.py`` -- the bottom-strip
load-monitor backend.

Cheap, no-I/O, no GPU dependency.  Verifies the snapshot helper:

  * Returns the documented dict shape (cpu_pct + ram_* + gpus).
  * Degrades to ``gpus: []`` when NVML wasn't init'd (CPU-only host).
  * Field types match the JS widget's expectations (floats, not None).
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import sys
import types

import pytest

from molbuilder.web.blueprints import system_load as sl

_MISSING = object()


@contextlib.contextmanager
def _reimported_with(fake_pynvml):
    """Re-run the import-time NVML probe against a substitute ``pynvml``.

    The probe runs ONCE, at import -- which is exactly why a host that
    breaks after the server starts stays broken for the life of the
    process -- so the only honest way to exercise either branch is to
    make the import happen again.  Pass ``None`` to make ``import
    pynvml`` raise ImportError (CPython treats a ``None`` entry in
    ``sys.modules`` as "this import fails").

    Reloading rebinds the module's globals in place and the finally
    reloads it once more against the real environment, so the module
    other tests see is the true one again.
    """
    saved = sys.modules.get("pynvml", _MISSING)
    sys.modules["pynvml"] = fake_pynvml
    try:
        importlib.reload(sl)
        yield
    finally:
        if saved is _MISSING:
            sys.modules.pop("pynvml", None)
        else:
            sys.modules["pynvml"] = saved
        importlib.reload(sl)


def test_snapshot_returns_documented_shape():
    """Public ``snapshot()`` returns the dict the widget consumes."""
    snap = sl.snapshot()
    # CPU/RAM fields always present even on a CPU-only host -- psutil
    # works everywhere we support.
    for key in ("cpu_pct",
                "cpu_count_physical", "cpu_count_logical",
                "loadavg_1m", "loadavg_5m", "loadavg_15m",
                "ram_pct", "ram_used_gb", "ram_total_gb",
                "gpus", "gpu_error"):
        assert key in snap, f"snapshot missing required key {key!r}"
    assert isinstance(snap["cpu_pct"],      (int, float))
    assert isinstance(snap["ram_pct"],      (int, float))
    assert isinstance(snap["ram_used_gb"],  (int, float))
    assert isinstance(snap["ram_total_gb"], (int, float))
    assert isinstance(snap["gpus"], list)
    # Load avg may be None on non-POSIX hosts but is a float here.
    for key in ("loadavg_1m", "loadavg_5m", "loadavg_15m"):
        v = snap[key]
        assert v is None or isinstance(v, (int, float)), \
            f"loadavg field {key!r} has unexpected type: {type(v)}"
    # Physical < logical when SMT is on; equal otherwise.  Both > 0.
    assert snap["cpu_count_physical"] >= 1
    assert snap["cpu_count_logical"]  >= snap["cpu_count_physical"]


def test_snapshot_gpu_entries_have_required_fields():
    """When NVML is available + a GPU is present, each entry carries
    the fields the JS sparkline draws + the hover tooltip cites."""
    snap = sl.snapshot()
    for gpu in snap["gpus"]:
        # ``error`` is the failure-marker shape -- one GPU's probe
        # failed but the snapshot kept going.  Don't assert the
        # success-shape on it.
        if "error" in gpu:
            assert "name" in gpu and "index" in gpu
            continue
        for key in ("index", "name", "util_pct",
                    "mem_used_mb", "mem_total_mb", "mem_pct"):
            assert key in gpu, f"GPU entry missing {key!r}: {gpu!r}"
        assert isinstance(gpu["util_pct"], (int, float))
        assert 0.0 <= gpu["mem_pct"]  <= 100.0
        assert 0.0 <= gpu["util_pct"] <= 100.0


def test_endpoint_returns_envelope(monkeypatch):
    """/api/system/load wraps the snapshot in the documented
    ``{ok, data}`` envelope (web-api.md § 1.1.1)."""
    from molbuilder.web.app import create_app
    app = create_app(config={})
    client = app.test_client()
    r = client.get("/api/system/load")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert "data" in body
    # Spot-check one field round-trips through the JSON layer cleanly.
    assert isinstance(body["data"]["cpu_pct"], (int, float))


def test_snapshot_endpoint_handles_psutil_failure(monkeypatch):
    """If psutil somehow raises, the endpoint returns ok=false with
    a useful message rather than 500-ing the JSON parser."""
    from molbuilder.web.app import create_app

    def _boom():
        raise RuntimeError("simulated psutil crash")

    monkeypatch.setattr(sl, "snapshot", _boom)
    app = create_app(config={})
    client = app.test_client()
    r = client.get("/api/system/load")
    assert r.status_code == 500
    body = r.get_json()
    assert body["ok"] is False
    assert "simulated psutil crash" in body["error"]


def test_a_host_with_no_gpu_support_reports_no_fault(caplog):
    """``pynvml`` absent is a CHOICE, not a breakage.

    Nobody installed ``molbuilder[gpu]``, so nothing is wrong: the wire
    carries no reason, the widget stays silent, and the log does not cry
    wolf on every start of every CPU-only server.
    """
    with caplog.at_level(logging.INFO, logger=sl.__name__):
        with _reimported_with(None):     # None => `import pynvml` raises
            assert sl._NVML_OK is False
            assert sl._GPU_ERROR is None
            assert sl.snapshot()["gpu_error"] is None
            warnings = [r for r in caplog.records
                        if r.levelno >= logging.WARNING
                        and r.name == sl.__name__]
            assert warnings == [], \
                f"a CPU-only host should not warn: {[r.message for r in warnings]}"


def test_a_driver_that_will_not_start_is_a_warning_and_reaches_the_wire(caplog):
    """``pynvml`` present but NVML refusing to start is a BROKEN HOST.

    This is the case that hid: an empty ``gpus`` list looks identical to
    a CPU-only box, so the strip silently dropped three cells and the
    only trace was an INFO line nobody runs the server verbose enough to
    see.  The reason must now be loud in the log AND present on the wire
    for the widget to print.
    """
    fake = types.ModuleType("pynvml")

    def _mismatch():
        raise RuntimeError("RM has detected an NVML/RM version mismatch.")

    fake.nvmlInit = _mismatch

    with caplog.at_level(logging.INFO, logger=sl.__name__):
        with _reimported_with(fake):
            assert sl._NVML_OK is False
            # The reason names the exception type AND its message -- the
            # type alone ("RuntimeError") tells a user nothing.
            assert sl._GPU_ERROR is not None
            assert "version mismatch" in sl._GPU_ERROR
            assert "RuntimeError" in sl._GPU_ERROR
            # On the wire, beside the empty list it explains.
            snap = sl.snapshot()
            assert snap["gpus"] == []
            assert snap["gpu_error"] == sl._GPU_ERROR
            # And in the log, at a level the operator actually sees.
            warnings = [r for r in caplog.records
                        if r.levelno >= logging.WARNING
                        and r.name == sl.__name__]
            assert len(warnings) == 1, \
                f"expected exactly one warning, got {[r.message for r in warnings]}"
            assert "version mismatch" in warnings[0].getMessage()


def test_the_endpoint_carries_the_reason_through_json(monkeypatch):
    """The widget reads ``gpu_error`` off the response, not off a global
    -- so the field has to survive the jsonify layer."""
    from molbuilder.web.app import create_app

    monkeypatch.setattr(sl, "_GPU_ERROR",
                        "NVMLError_LibRmVersionMismatch: mismatch")
    client = create_app(config={}).test_client()
    body = client.get("/api/system/load").get_json()
    assert body["ok"] is True
    assert body["data"]["gpu_error"] == \
        "NVMLError_LibRmVersionMismatch: mismatch"


def test_gpu_name_decodes_bytes_and_str():
    """NVML's ``nvmlDeviceGetName`` returns bytes in some pynvml
    versions and str in others.  Helper must normalise to str without
    crashing on either."""
    # Stub a handle object the wrapper accepts.  The real signature
    # takes a c_void_p; the helper only calls one library function.
    sentinel = object()

    if sl.pynvml is None:
        pytest.skip("pynvml not importable; helper unreachable")

    monkeypatched = {"value": b"NVIDIA Bytes GPU"}
    real_fn = sl.pynvml.nvmlDeviceGetName
    sl.pynvml.nvmlDeviceGetName = lambda h: monkeypatched["value"]
    try:
        assert sl._gpu_name(sentinel) == "NVIDIA Bytes GPU"
        monkeypatched["value"] = "NVIDIA Str GPU"
        assert sl._gpu_name(sentinel) == "NVIDIA Str GPU"
    finally:
        sl.pynvml.nvmlDeviceGetName = real_fn
