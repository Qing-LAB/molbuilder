"""L2 Node-driven test: the trajectory CSV export redacts
user-identifying path prefixes so the downloaded ``*_plots.csv``
header doesn't leak the OS username.

User report (2026-06-14)
========================

The CSV download button on the Results tab produced a header like:

    # molbuilder — trajectory plot data export
    # generated:    2026-06-14T22:14:33.108Z
    # source path:  /home/qqing/molbuilder/projects/BDT/run.out
    # parser:       siesta
    ...

The ``/home/qqing/`` prefix is sensitive (OS-level username).  The
fix: redact it in the JS-side CSV builder so the CSV reader sees a
header like:

    # source path:  ~/molbuilder/projects/BDT/run.out

Path structure past the username segment is preserved verbatim --
the project layout, staged-relaxation naming, etc. are useful for
scientific provenance and don't disclose identity.

What this file pins
===================

The ``_redactSourcePath`` helper is exported on
``window.molbuilder.trajectoryInspector._redactSourcePath`` for
test-only use (the JSDoc comment in the source notes it's not part
of the inspector's public API).  This file drives it via a small
Node harness with a series of POSIX + macOS + Windows + pytest-tmp
inputs and asserts each one redacts to the expected shape.

A future regression that drops the redaction call from
``_buildPlotCsv`` (or weakens the regexes) fails here loudly.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_CORE = _ROOT / "molbuilder/web/static/lib/trajectory/core.js"


def _have_node():
    return shutil.which("node") is not None


pytestmark = pytest.mark.skipif(
    not _have_node(),
    reason="node not installed; skipping L2 Node-driven JS test",
)


# Cases: (input, expected_output, why).
# Keep ``why`` short -- it becomes the failure message + the docstring
# of the parametrize case.
_CASES = [
    # POSIX home (Linux): username segment after /home/ is redacted.
    (
        "/home/qqing/molbuilder/projects/BDT/run.out",   # not-a-fixture
        "~/molbuilder/projects/BDT/run.out",             # not-a-fixture
        "Linux home prefix redacted to ~",
    ),
    # POSIX home with a different username -- must still match.
    (
        "/home/alice/work/foo.out",
        "~/work/foo.out",
        "Linux home, any username, redacted",
    ),
    # macOS home (/Users/<u>/...).
    (
        "/Users/bob/Documents/molbuilder/foo.out",
        "~/Documents/molbuilder/foo.out",
        "macOS home prefix redacted to ~",
    ),
    # pytest tmpdir (/tmp/pytest-of-<u>/...) — common in CI logs.
    (
        "/tmp/pytest-of-qqing/pytest-272/test_x/foo.out",
        "<tmp>/pytest-272/test_x/foo.out",
        "pytest tmpdir username segment redacted to <tmp>",
    ),
    # Windows home (C:\Users\<u>\...).
    (
        r"C:\Users\carol\Documents\molbuilder\foo.out",
        r"~\Documents\molbuilder\foo.out",
        "Windows home prefix redacted to ~",
    ),
    # Windows home with mixed-case drive.
    (
        r"d:\Users\dave\foo.out",
        r"~\foo.out",
        "Windows home, lowercase drive letter, redacted",
    ),
    # Non-home POSIX path: unchanged.
    (
        "/opt/molbuilder/share/foo.out",
        "/opt/molbuilder/share/foo.out",
        "Non-home POSIX path preserved verbatim",
    ),
    # Empty / unknown placeholder: unchanged (no infinite loop, no
    # crash).
    (
        "(unknown)",
        "(unknown)",
        "Sentinel '(unknown)' preserved unchanged",
    ),
    # Empty string: unchanged.
    (
        "",
        "",
        "Empty string preserved",
    ),
]


def _run_node(script: str) -> str:
    """Run a Node one-liner and return stdout."""
    proc = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout


def test_redact_source_path_module_export_present():
    """The redaction helper must be exported on the
    ``trajectoryInspector`` namespace; ``_buildPlotCsv`` uses it
    directly inside the inspector and the test exercise needs it
    on the module surface."""
    src = _CORE.read_text(encoding="utf-8")
    assert "_redactSourcePath: _redactSourcePath" in src, (
        "trajectory/core.js no longer exports _redactSourcePath on "
        "the inspector namespace; restore the export so this test "
        "can drive the function or move the helper into a separate "
        "module."
    )


@pytest.mark.parametrize(
    "raw, expected, description",
    _CASES,
    ids=[c[2] for c in _CASES],
)
def test_redaction_pattern(raw, expected, description):
    """Drive ``_redactSourcePath`` via Node + assert exact output.

    Each case pins one redaction shape.  A failure means the regex
    no longer matches the documented pattern OR the replacement
    string drifted.
    """
    # Strip the IIFE wrapper for Node execution: just inline the
    # function body via JSON-encoded arg so we don't have to source
    # the whole module.  The module's IIFE captures ``window`` /
    # ``this`` which Node doesn't have; sandboxing into a snippet is
    # cleaner than loading the whole 2700-line file.
    src = _CORE.read_text(encoding="utf-8")

    # Extract the function source by anchored slicing.
    marker_begin = "function _redactSourcePath(p) {"
    marker_end_token = "        return p;\n    }"
    if marker_begin not in src or marker_end_token not in src:
        pytest.fail(
            "Could not locate _redactSourcePath in trajectory/core.js "
            "via anchored markers; the function may have been "
            "renamed or moved.  Update this test's markers."
        )
    fn_start = src.index(marker_begin)
    fn_end = src.index(marker_end_token, fn_start) + len(marker_end_token)
    fn_src = src[fn_start:fn_end]

    raw_json = json.dumps(raw)
    expected_json = json.dumps(expected)
    script = (
        fn_src
        + f"\nconst got = _redactSourcePath({raw_json});"
        + f"\nconst want = {expected_json};"
        + "\nif (got !== want) {"
        + "\n  console.error(JSON.stringify({got, want}));"
        + "\n  process.exit(1);"
        + "\n}"
    )
    try:
        _run_node(script)
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"_redactSourcePath({raw!r}) did not yield {expected!r}.\n"
            f"Case: {description}\n"
            f"Node stderr: {exc.stderr.strip()}"
        )


def test_csv_builder_calls_redaction():
    """Source-text guard: ``_buildPlotCsv`` must call
    ``_redactSourcePath`` on ``ctx.sourcePath`` before writing the
    CSV header.  Pre-fix the function inlined ``ctx.sourcePath ||
    "(unknown)"`` straight into the header, leaking the home dir.
    """
    src = _CORE.read_text(encoding="utf-8")
    # Sniff a tight pattern: the assignment immediately after the
    # function header must invoke _redactSourcePath on ctx.sourcePath.
    needle = "_redactSourcePath(\n            ctx.sourcePath"
    assert needle in src, (
        "_buildPlotCsv no longer applies _redactSourcePath to "
        "ctx.sourcePath before writing to the CSV header.  The bare-"
        "path version leaked the OS username via the CSV download; "
        "restore the redaction call."
    )
