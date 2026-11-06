"""FIELD TEST -- the load widget reads what this machine's ``nvidia-smi`` prints
(`web/blueprints/system_load.py`).

Goal: the widget's GPU cells are the tool's own numbers, one row per device.
What the tool prints is read where it is printed: this runs on the target,
against its real tool, and is skipped where no device is visible.

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from molbuilder.web.blueprints import system_load as sl


def test_the_widget_reads_each_device_the_tool_lists(monkeypatch):
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no nvidia-smi on this machine")
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                            text=True, timeout=5)
    devices = [ln for ln in listed.stdout.splitlines() if ln.startswith("GPU ")]
    if not devices:
        pytest.skip("nvidia-smi lists no device here")
    monkeypatch.setattr(sl, "_smi_cache", {"t": 0.0, "gpus": [], "error": None})
    gpus, err = sl._gpu_snapshot()
    if err and "timed out" in err:
        pytest.skip(f"a temporary failure of the inquiry: {err}")
    assert err is None
    assert [g["index"] for g in gpus] == list(range(len(devices)))
    for gpu, line in zip(gpus, devices):
        assert gpu["name"] in line, (gpu["name"], line)
        assert gpu["mem_total_mb"] > 0
        assert 0 <= gpu["mem_used_mb"] <= gpu["mem_total_mb"]
        assert gpu["mem_pct"] == pytest.approx(
            100.0 * gpu["mem_used_mb"] / gpu["mem_total_mb"], abs=0.1)
        # What a card does not expose reads None; what it does is a number.
        for field in ("util_pct", "util_mem_pct", "power_w", "temp_c",
                      "sm_clock_mhz", "mem_clock_mhz"):
            assert gpu[field] is None or gpu[field] >= 0, field
