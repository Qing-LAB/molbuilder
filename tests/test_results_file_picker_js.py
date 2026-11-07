"""Unit tests for ``lib/results/file-picker.js`` driven via node.

What stays here is the picker's own logic on what a presenter answers: how
entries are bucketed into headings and the headings ordered, and an entry's
label.  The presenters are stand-ins that answer a heading; the entries are
names and times no line here reads inside.

What the picker makes of a real run folder's listing -- the files it lists,
newest first, the path whole; a run as one entry, its master absorbing its
parts; the heading built from the folder's answer -- is asked of runs made
in the pass: `test_the_viewers_follow_a_real_run_e2e.py`.
"""
from __future__ import annotations

from pathlib import Path

from tests._node_esm import run_node


ROOT   = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/results/file-picker.js"
# Loaded before the picker, as `results.html` loads it: the one "ago".
TIME_AGO = ROOT / "molbuilder/web/static/lib/time-ago.js"


def _run(snippet: str) -> object:
    """``snippet`` against the picker as `results.html` loads it -- the one
    "ago" first (`_node_esm.run_node`)."""
    return run_node([TIME_AGO, MODULE], snippet)


class TestGroupResultFiles:
    """Pre-filtered ``filterToResultFiles`` output, bucketed by
    inspector ``resultCategory(path)`` into ``[{label, entries}]``.
    Group order = newest entry's mtime descending (so the group
    holding the user's most-recent run floats to top).  Entries
    inside each group preserve input order (caller feeds newest-
    first, so no per-group resort needed)."""

    def test_buckets_by_category(self):
        """Each entry's category comes from ``pickResult(path).resultCategory``."""
        out = _run(
            "const entries = [\n"
            "  {path: '/p/a.out',         name: 'a.out',         mtime: 300},\n"
            "  {path: '/p/b.spectra.json', name: 'b.spectra.json', mtime: 200},\n"
            "  {path: '/p/c.out',         name: 'c.out',         mtime: 100},\n"
            "];\n"
            "function pickResult(p) {\n"
            "  if (p.endsWith('.out'))         return { name: 'trajectory',"
            "    displayName: 'Traj', resultCategory: (_) => 'SIESTA optimization' };\n"
            "  if (p.endsWith('.spectra.json')) return { name: 'spectra',"
            "    displayName: 'Spec', resultCategory: (_) => 'PySCF spectrum' };\n"
            "  return null;\n"
            "}\n"
            "const groups = window.molbuilder.resultsFilePicker.groupResultFiles("
            "  entries, pickResult);\n"
            "console.log(JSON.stringify(groups.map(g => ({"
            "  label: g.label,"
            "  names: g.entries.map(e => e.name)"
            "}))));"
        )
        assert out == [
            {"label": "SIESTA optimization", "names": ["a.out", "c.out"]},
            {"label": "PySCF spectrum",      "names": ["b.spectra.json"]},
        ]

    def test_groups_sorted_by_newest_entry(self):
        """Group with the newest file overall comes first regardless
        of insertion order in the input list."""
        out = _run(
            "const entries = [\n"
            "  {path: '/p/old.spectra.json', name: 'old.json', mtime: 100},\n"
            "  {path: '/p/new.out',          name: 'new.out',  mtime: 999},\n"
            "];\n"
            "function pickResult(p) {\n"
            "  if (p.endsWith('.out'))         return { name: 't',"
            "    displayName: 't', resultCategory: (_) => 'SIESTA optimization' };\n"
            "  if (p.endsWith('.spectra.json')) return { name: 's',"
            "    displayName: 's', resultCategory: (_) => 'PySCF spectrum' };\n"
            "  return null;\n"
            "}\n"
            "const groups = window.molbuilder.resultsFilePicker.groupResultFiles("
            "  entries, pickResult);\n"
            "console.log(JSON.stringify(groups.map(g => g.label)));"
        )
        # SIESTA group's newest (999) > PySCF group's newest (100), so
        # SIESTA group first.
        assert out == ["SIESTA optimization", "PySCF spectrum"]

    def test_falls_back_to_display_name(self):
        """An inspector without ``resultCategory`` uses ``displayName``."""
        out = _run(
            "const entries = [{path: '/p/x.foo', name: 'x.foo', mtime: 100}];\n"
            "function pickResult(_) { return {"
            "  name: 'i', displayName: 'Fallback label' /* no resultCategory */"
            "}; }\n"
            "const groups = window.molbuilder.resultsFilePicker.groupResultFiles("
            "  entries, pickResult);\n"
            "console.log(JSON.stringify(groups.map(g => g.label)));"
        )
        assert out == ["Fallback label"]

    def test_isolates_throwing_resultCategory(self):
        """An inspector whose resultCategory throws still gets bucketed
        (via its displayName) -- a buggy inspector must not lose its
        files from the dropdown."""
        out = _run(
            "const entries = [{path: '/p/x.foo', name: 'x.foo', mtime: 100}];\n"
            "function pickResult(_) { return {"
            "  name: 'i', displayName: 'Fallback',"
            "  resultCategory: () => { throw new Error('boom'); }"
            "}; }\n"
            "const groups = window.molbuilder.resultsFilePicker.groupResultFiles("
            "  entries, pickResult);\n"
            "console.log(JSON.stringify(groups.map(g => g.label)));"
        )
        assert out == ["Fallback"]

    def test_empty_input_returns_empty(self):
        out = _run(
            "function pickResult(_) { return { name: 'i', displayName: 'd' }; }\n"
            "const r = window.molbuilder.resultsFilePicker.groupResultFiles("
            "  [], pickResult);\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == []


class TestLabelForResult:

    def test_with_mtime(self):
        out = _run(
            "const now = Date.now()/1000;\n"
            "const r = window.molbuilder.resultsFilePicker._labelForResult({"
            "  name: 'job.out', mtime: now - 300"
            "});\n"
            "console.log(JSON.stringify(r));"
        )
        assert out == "job.out (5m ago)"
