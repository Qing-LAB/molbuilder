"""Unit tests for ``lib/trajectory/result-list.js`` driven via node.

We run the JS module under Node in a minimal stub environment
(window + document.createElement) and exercise the pure helpers
(parseDir, formatRelativeTime, _labelForResult).  The DOM-bound
mount() path is covered indirectly by the backend's
TestResultList integration tests + the Playwright suite (#178);
this file pins the pure-derivation logic without needing a
browser.

These tests are deliberately not Playwright -- they're cheap, run
in <1s, and would catch regressions in the helper functions
without requiring chromium / network."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/trajectory/result-list.js"


def _run_node(snippet: str) -> dict:
    """Run a JS snippet under Node with the result-list module
    pre-loaded.  ``snippet`` must end with a line that assigns
    ``OUT`` to a JSON-serialisable value (string / number / object
    / array).  We then print JSON.stringify(OUT) so the Python
    side can parse it back.

    Returns the parsed object (dict / list / primitive).
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    # Minimal window/document/fetch stubs so the IIFE runs without
    # touching real APIs.  parseDir and formatRelativeTime only
    # touch globals via the IIFE root closure; the helpers
    # themselves don't.
    bootstrap = """
        // Minimal DOM stubs.  parseDir + formatRelativeTime don't
        // touch the DOM; they live in the closure root namespace.
        global.window = global;
        global.document = {
            createElement: () => ({
                setAttribute: () => {},
                appendChild: () => {},
            }),
            getElementById: () => null,
        };
        global.AbortController = function () {
            this.signal = {};
            this.abort = () => {};
        };
        global.fetch = () => Promise.resolve({ ok: false });
    """
    full = bootstrap + "\n" + MODULE.read_text() + "\n" + snippet
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", full],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestParseDir:

    def test_posix_path(self):
        out = _run_node(
            "console.log(JSON.stringify("
            "window.molbuilder.trajectoryResultList.parseDir("
            "'/projects/job/out-run0.out')"
            "))"
        )
        assert out == {"dir": "/projects/job", "name": "out-run0.out"}

    def test_windows_path(self):
        out = _run_node(
            "console.log(JSON.stringify("
            "window.molbuilder.trajectoryResultList.parseDir("
            "'C:\\\\projects\\\\job\\\\out.out')"
            "))"
        )
        assert out == {"dir": "C:\\projects\\job", "name": "out.out"}

    def test_no_separator(self):
        out = _run_node(
            "console.log(JSON.stringify("
            "window.molbuilder.trajectoryResultList.parseDir('bare.out')"
            "))"
        )
        assert out == {"dir": "", "name": "bare.out"}

    def test_empty_string(self):
        out = _run_node(
            "console.log(JSON.stringify("
            "window.molbuilder.trajectoryResultList.parseDir('')"
            "))"
        )
        assert out == {"dir": "", "name": ""}


class TestFormatRelativeTime:

    def test_just_now(self):
        """Within 60 s -> ``Ns ago``."""
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.trajectoryResultList."
            "formatRelativeTime(now - 7);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "7s ago"

    def test_minutes(self):
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.trajectoryResultList."
            "formatRelativeTime(now - 180);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "3m ago"

    def test_hours(self):
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.trajectoryResultList."
            "formatRelativeTime(now - 7200);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "2h ago"

    def test_days(self):
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.trajectoryResultList."
            "formatRelativeTime(now - 2 * 86400);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "2d ago"

    def test_future_timestamp_clamped(self):
        """Clock skew: a 'future' mtime should clamp to 0s, not
        produce a negative number string."""
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.trajectoryResultList."
            "formatRelativeTime(now + 100);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "0s ago"

    def test_null_returns_empty(self):
        out = _run_node(
            "const r = window.molbuilder.trajectoryResultList."
            "formatRelativeTime(null);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == ""

    def test_nan_returns_empty(self):
        out = _run_node(
            "const r = window.molbuilder.trajectoryResultList."
            "formatRelativeTime(Number.NaN);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == ""


class TestLabelForResult:

    def test_runN_entry(self):
        """``run 1, 5m ago`` tail for a runN entry."""
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.trajectoryResultList."
            "_labelForResult({"
            "  name: 'job-run1.out',"
            "  mtime: now - 300,"
            "  run_index: 1"
            "});\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "job-run1.out (run 1, 5m ago)"

    def test_non_runN_entry(self):
        """Files without a runN suffix carry ``single`` in their tag."""
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.trajectoryResultList."
            "_labelForResult({"
            "  name: 'plain.out',"
            "  mtime: now - 600,"
            "  run_index: null"
            "});\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "plain.out (single, 10m ago)"


class TestAPISurface:

    def test_module_exposes_named_helpers(self):
        """Pin the public-surface contract: a future refactor that
        renames parseDir / formatRelativeTime / _labelForResult
        would break callers."""
        out = _run_node(
            "const api = window.molbuilder.trajectoryResultList;\n"
            "console.log(JSON.stringify({"
            "  hasMount:               typeof api.mount === 'function',"
            "  hasParseDir:            typeof api.parseDir === 'function',"
            "  hasFormatRelativeTime:  typeof api.formatRelativeTime === 'function',"
            "  hasLabelForResult:      typeof api._labelForResult === 'function'"
            "}));"
        )
        assert out == {
            "hasMount":               True,
            "hasParseDir":            True,
            "hasFormatRelativeTime":  True,
            "hasLabelForResult":      True,
        }
