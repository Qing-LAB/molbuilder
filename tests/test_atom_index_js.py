"""Atom index-base conversion helper (data-vocabulary.md § 3.1).
Pure logic, run under Node (no browser)."""
import json, shutil, subprocess
from pathlib import Path
import pytest

_NODE = shutil.which("node")
_MODULE = (Path(__file__).resolve().parent.parent
           / "molbuilder/web/static/lib/workspace/_atom-index.js")
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
