"""L2 Node-driven test: the trajectory CSV export redacts
user-identifying path prefixes so the downloaded ``*_plots.csv``
header doesn't leak the OS username.

User report (2026-06-14)
========================

The CSV download button on the Results tab produced a header like:

    # molbuilder — trajectory plot data export
    # generated:    2026-06-14T22:14:33.108Z
    # source path:  /home/u/molbuilder/projects/BDT/run.out
    # parser:       siesta
    ...

The ``/home/u/`` prefix is sensitive (OS-level username).  The
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
        "/home/u/molbuilder/projects/BDT/run.out",   # not-a-fixture
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
        "/tmp/pytest-of-u/pytest-272/test_x/foo.out",
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


_STATIC = _CORE.parents[2]

#: A DOM thin enough to import the module under Node.  `core.js` mounts
#: nothing at import time -- loading it is a no-op by design ("safe to include
#: on any page that might need the inspector later") -- so the stub only has
#: to survive the module's top level.
_DOM_STUB = """
globalThis.window = globalThis;
globalThis.document = {
    createElement: () => ({ style: {}, classList: { add() {}, remove() {} },
                            appendChild() {}, setAttribute() {},
                            addEventListener() {} }),
    querySelector: () => null, querySelectorAll: () => [],
    addEventListener() {},
};
globalThis.molbuilder = {};
"""


def _load_and_call(raw):
    """Load `core.js` and call the exported redactor on *raw*."""
    from _node_esm import run_node
    out = run_node(
        [_CORE], "console.log(JSON.stringify({v: "
                 "globalThis.molbuilder.trajectoryInspector._redactSourcePath("
                 + json.dumps(raw) + ")}));",
        globals_js=_DOM_STUB, static_root=_STATIC)
    return out["v"]


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


@pytest.mark.parametrize(
    "raw,expected,description", _CASES,
    ids=[c[2].replace(" ", "_") for c in _CASES],
)
def test_redaction_pattern(raw, expected, description):
    """Drive ``_redactSourcePath`` through the module's own export.

    CONVERTED 2026-09-06 (`plans/plan.md` § 5h).  This used to EXTRACT the
    function's source by anchored slicing -- ``marker_end_token = "
    return p;\\n    }"``, eight spaces and a newline -- run that text under
    Node, and separately assert the export string appeared in the file.  Two
    problems, and they compounded: re-indent the function and the slice breaks
    with "the function may have been renamed"; and because the slice never
    touched the export, the export existed *"so this test can drive the
    function"* while no test drove it that way.  Removing it failed only the
    string pin, never a real check.

    Now the module is LOADED (`tests/_node_esm.run_node`, with `static_root`
    so its browser-absolute `/static/...` import resolves) and the function is
    called on the namespace production reaches it through.  The separate
    export pin is deleted: it is genuinely redundant now, because these cases
    cannot run without the export.
    """
    out = _load_and_call(raw)
    assert out == expected, (
        f"_redactSourcePath({raw!r}) -> {out!r}, expected {expected!r}\n"
        f"Case: {description}")

# `test_csv_builder_calls_redaction` stood here and searched core.js for the
# literal `_redactSourcePath(\n            ctx.sourcePath` -- twelve spaces of
# indentation included.  Re-wrapping that one call broke it while the CSV
# stayed clean, and no arrangement of text can answer the question that
# matters: did the file that reached the user's disk carry their login.  It
# does now, in tests/test_inspector_registry_e2e.py::
# test_the_exported_csv_does_not_carry_the_users_name, which mounts a real
# run, clicks Export, and reads the downloaded bytes.  What stays HERE is the
# pattern table above: one path in, one path out, no browser needed.
