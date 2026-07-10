"""Atom index-base conversion helper (data-vocabulary.md § 3.1).
Pure logic, run under Node (no browser)."""
import json, shutil, subprocess
from pathlib import Path
import pytest

_NODE = shutil.which("node")
_MODULE = (Path(__file__).resolve().parent.parent
           / "molbuilder/web/static/lib/molview/_atom-index.js")
pytestmark = pytest.mark.skipif(_NODE is None, reason="node not available")


def _run(script: str) -> str:
    full = f"const M = require({json.dumps(str(_MODULE))});\n" + script
    r = subprocess.run([_NODE, "-e", full], capture_output=True, text=True)
    assert r.returncode == 0, f"node failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout.strip()


def test_to_from_display_roundtrip():
    _run("""
      const A = require("assert");
      A.strictEqual(M.toDisplay(0), 1);
      A.strictEqual(M.fromDisplay(1), 0);
      A.strictEqual(M.fromDisplay(M.toDisplay(41)), 41);
      console.log("ok");
    """)


def test_shift_expression_ranges_singletons_whitespace():
    out = _run("""console.log(M.shiftExpression("1-4, 6, 10-11", -1));""")
    assert out == "0-3, 5, 9-10"


def test_shift_expression_forward():
    out = _run("""console.log(M.shiftExpression("0-3, 5", 1));""")
    assert out == "1-4, 6"


def test_shift_expression_leaves_unrecognised_and_nonstring():
    _run("""
      const A = require("assert");
      A.strictEqual(M.shiftExpression("Au,C", -1), "Au, C");  // not indices
      A.strictEqual(M.shiftExpression(42, -1), 42);           // non-string
      console.log("ok");
    """)


def test_viewer_index_labels_conform_to_l1():
    """The standalone 3D viewer (mol-viewer-embed.js) inlines its atom-index
    label because it can't import L1; bind that inline convention to L1's
    toDisplay contract so the two can't drift (atom-annotations.md § 5.1).
    If L1's display delta changes, this fails until the viewer is updated."""
    import re
    delta = int(_run("console.log(String(M.toDisplay(7) - 7));"))
    assert delta == 1, f"L1 toDisplay delta is {delta}; update the viewer + this test"
    viewer = (Path(__file__).resolve().parent.parent
              / "molbuilder/web/static/lib/mol-viewer-embed.js").read_text()
    # Both atom-index label sites must display the SAME i + delta L1 defines.
    assert re.search(rf"String\(\s*i\s*\+\s*{delta}\s*\)", viewer), (
        "auto atom-index label must apply L1's display delta (i + %d)" % delta)
    assert re.search(rf"String\(\s*idx\s*\+\s*{delta}\s*\)", viewer), (
        "picked-atom index label must apply L1's display delta (idx + %d)" % delta)
