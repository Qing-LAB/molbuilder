"""Unit tests for ``lib/results/file-picker.js`` driven via node.

The tab-level result-file picker replaced the in-trajectory
``lib/trajectory/result-list.js`` on 2026-06-01.  These tests cover
its pure helpers (``parseDir``, ``formatRelativeTime``,
``filterToResultFiles``, ``_labelForResult``) under Node in a
minimal stub environment -- no Chromium, no network.

The DOM-bound mount() path (subscriber wiring, fetch, auto-pick) is
covered by ``tests/test_results_blueprint.py`` integration smoke
tests + the Playwright suite (#178).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT   = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/results/file-picker.js"


def _run_node(snippet: str) -> object:
    """Run a JS snippet under Node with the file-picker module
    pre-loaded.  ``snippet`` must end with a line that
    JSON.stringifies its output to stdout.  Returns the parsed
    value."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    # Minimal window/document/fetch stubs so the IIFE runs without
    # touching real APIs.
    bootstrap = """
        global.window = global;
        global.document = {
            createElement: () => ({
                setAttribute: () => {},
                appendChild: () => {},
            }),
            getElementById: () => null,
            querySelector: () => null,
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
        capture_output=True, text=True, timeout=15,
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
            "const r = window.molbuilder.resultsFilePicker.parseDir("
            "'/projects/job/out-run0.out');\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == {"dir": "/projects/job", "name": "out-run0.out"}

    def test_windows_path(self):
        out = _run_node(
            "const r = window.molbuilder.resultsFilePicker.parseDir("
            "'C:\\\\projects\\\\job\\\\out.out');\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == {"dir": "C:\\projects\\job", "name": "out.out"}

    def test_no_separator(self):
        out = _run_node(
            "const r = window.molbuilder.resultsFilePicker.parseDir("
            "'bare.out');\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == {"dir": "", "name": "bare.out"}

    def test_empty_string(self):
        out = _run_node(
            "const r = window.molbuilder.resultsFilePicker.parseDir('');\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == {"dir": "", "name": ""}


class TestFormatRelativeTime:

    def test_just_now(self):
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.resultsFilePicker."
            "formatRelativeTime(now - 7);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "7s ago"

    def test_minutes(self):
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.resultsFilePicker."
            "formatRelativeTime(now - 180);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "3m ago"

    def test_hours(self):
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.resultsFilePicker."
            "formatRelativeTime(now - 7200);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "2h ago"

    def test_days(self):
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.resultsFilePicker."
            "formatRelativeTime(now - 2 * 86400);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "2d ago"

    def test_future_timestamp_clamped(self):
        """Clock skew: a 'future' mtime clamps to 0s, not negative."""
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.resultsFilePicker."
            "formatRelativeTime(now + 100);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "0s ago"

    def test_null_returns_empty(self):
        out = _run_node(
            "const r = window.molbuilder.resultsFilePicker."
            "formatRelativeTime(null);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == ""


class TestFilterToResultFiles:
    """The directory-listing filter is the core of the tab-level
    picker.  It must:
      * keep only entries whose ``kind === 'file'``;
      * keep only files for which ``pickResult(path)`` returns truthy;
      * sort by mtime descending;
      * break ties on name deterministically.
    """

    def test_filters_by_pick_result(self):
        """Files for which pickResult returns falsy must be dropped."""
        out = _run_node(
            "const entries = [\n"
            "  {name: 'a.out',          kind: 'file', mtime: 100},\n"
            "  {name: 'b.config',       kind: 'file', mtime: 200},\n"
            "  {name: 'c.spectra.json', kind: 'file', mtime: 300},\n"
            "];\n"
            "function pickResult(p) {\n"
            "  return p.endsWith('.out') || p.endsWith('.spectra.json')"
            "    ? { name: 'fake' }\n"
            "    : null;\n"
            "}\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  entries, '/proj', pickResult);\n"
            "console.log(JSON.stringify(r.map(x => x.name)));"
        )
        assert out == ["c.spectra.json", "a.out"]

    def test_filters_out_directories(self):
        out = _run_node(
            "const entries = [\n"
            "  {name: 'subdir',  kind: 'directory', mtime: 999},\n"
            "  {name: 'foo.out', kind: 'file',      mtime: 100},\n"
            "];\n"
            "function pickResult(_) { return { name: 'fake' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  entries, '/proj', pickResult);\n"
            "console.log(JSON.stringify(r.map(x => x.name)));"
        )
        assert out == ["foo.out"]

    def test_sorts_newest_first(self):
        out = _run_node(
            "const entries = [\n"
            "  {name: 'old.out',    kind: 'file', mtime: 100},\n"
            "  {name: 'middle.out', kind: 'file', mtime: 500},\n"
            "  {name: 'newest.out', kind: 'file', mtime: 900},\n"
            "];\n"
            "function pickResult(_) { return { name: 'fake' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  entries, '/proj', pickResult);\n"
            "console.log(JSON.stringify(r.map(x => x.name)));"
        )
        assert out == ["newest.out", "middle.out", "old.out"]

    def test_tiebreak_by_name(self):
        """Filesystems with 1s mtime resolution can report identical
        mtimes for two files written back-to-back.  The picker's
        sort must produce the same order on every page load."""
        out = _run_node(
            "const entries = [\n"
            "  {name: 'zzz.out', kind: 'file', mtime: 500},\n"
            "  {name: 'aaa.out', kind: 'file', mtime: 500},\n"
            "  {name: 'mmm.out', kind: 'file', mtime: 500},\n"
            "];\n"
            "function pickResult(_) { return { name: 'fake' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  entries, '/proj', pickResult);\n"
            "console.log(JSON.stringify(r.map(x => x.name)));"
        )
        # Alphabetical on tie -- aaa before mmm before zzz.
        assert out == ["aaa.out", "mmm.out", "zzz.out"]

    def test_builds_absolute_path(self):
        """The picker calls projects.setShared(dir, path), so each
        entry must carry the FULL path -- not just the basename."""
        out = _run_node(
            "const entries = [{name: 'foo.out', kind: 'file', mtime: 100}];\n"
            "function pickResult(_) { return { name: 'fake' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  entries, '/projects/myjob', pickResult);\n"
            "console.log(JSON.stringify(r[0]));"
        )
        assert out["name"] == "foo.out"
        assert out["path"] == "/projects/myjob/foo.out"
        assert out["mtime"] == 100

    def test_handles_missing_mtime(self):
        """A broken-symlink entry has ``mtime: null``; the sort
        must not crash and the entry must land at the end."""
        out = _run_node(
            "const entries = [\n"
            "  {name: 'good.out',   kind: 'file', mtime: 100},\n"
            "  {name: 'broken.out', kind: 'file', mtime: null},\n"
            "];\n"
            "function pickResult(_) { return { name: 'fake' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  entries, '/proj', pickResult);\n"
            "console.log(JSON.stringify(r.map(x => x.name)));"
        )
        # null mtime sorts to end (treated as -Infinity).
        assert out == ["good.out", "broken.out"]

    def test_empty_input_returns_empty(self):
        out = _run_node(
            "function pickResult(_) { return { name: 'fake' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  [], '/proj', pickResult);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == []

    def test_invalid_input_returns_empty(self):
        """Defensive: non-array input must not throw."""
        out = _run_node(
            "function pickResult(_) { return { name: 'fake' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  null, '/proj', pickResult);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == []

    def test_uses_backslash_separator_for_windows_paths(self):
        """Windows-style dir paths build entries with backslash sep
        so projects.setShared receives a path that round-trips
        through parseDir without losing the separator type."""
        out = _run_node(
            "const entries = [{name: 'foo.out', kind: 'file', mtime: 100}];\n"
            "function pickResult(_) { return { name: 'fake' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.filterToResultFiles("
            "  entries, 'C:\\\\projects\\\\myjob', pickResult);\n"
            "console.log(JSON.stringify(r[0].path));"
        )
        assert out == "C:\\projects\\myjob\\foo.out"


class TestLabelForResult:

    def test_with_mtime(self):
        out = _run_node(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.resultsFilePicker._labelForResult({"
            "  name: 'job.out', mtime: now - 300"
            "});\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "job.out (5m ago)"

    def test_without_mtime(self):
        out = _run_node(
            "const r = window.molbuilder.resultsFilePicker._labelForResult({"
            "  name: 'plain.out', mtime: null"
            "});\n"
            "console.log(JSON.stringify(r));"
        )
        # No timestamp suffix when mtime is missing.
        assert out == "plain.out"


class TestAPISurface:

    def test_module_exposes_named_helpers(self):
        """Pin the public-surface contract: a future refactor that
        renames any helper would break callers."""
        out = _run_node(
            "const api = window.molbuilder.resultsFilePicker;\n"
            "console.log(JSON.stringify({"
            "  hasMount:               typeof api.mount === 'function',"
            "  hasParseDir:            typeof api.parseDir === 'function',"
            "  hasFormatRelativeTime:  typeof api.formatRelativeTime === 'function',"
            "  hasFilterToResultFiles: typeof api.filterToResultFiles === 'function',"
            "  hasLabelForResult:      typeof api._labelForResult === 'function'"
            "}));"
        )
        assert out == {
            "hasMount":               True,
            "hasParseDir":            True,
            "hasFormatRelativeTime":  True,
            "hasFilterToResultFiles": True,
            "hasLabelForResult":      True,
        }
