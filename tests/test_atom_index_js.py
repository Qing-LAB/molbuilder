"""Atom index-base conversion helper (data-vocabulary.md § 3.1).
Pure logic, run under Node (no browser).

Loaded via the shared ES-module-capable harness (tests/_node_esm.py) and read through the
GLOBAL ``molbuilder.atomIndexModel`` -- which the file publishes whether it's a classic IIFE
or a native ES module -- so this test is stable across the MolView ESM migration.
"""
import re
from pathlib import Path

from _node_esm import run_node

_ROOT   = Path(__file__).resolve().parent.parent
_MODULE = _ROOT / "molbuilder/web/static/lib/molview/_atom-index.js"
_EMBED  = _ROOT / "molbuilder/web/static/lib/viewer/mol-viewer-embed.js"

_M = "globalThis.molbuilder.atomIndexModel"   # published by classic OR ESM form


def test_to_from_display_roundtrip():
    out = run_node([_MODULE], f"""
        const M = {_M};
        console.log(JSON.stringify({{
            display0: M.toDisplay(0), internal1: M.fromDisplay(1),
            roundtrip: M.fromDisplay(M.toDisplay(41)),
        }}));
    """)
    assert out == {"display0": 1, "internal1": 0, "roundtrip": 41}


def test_shift_expression_ranges_singletons_whitespace():
    out = run_node([_MODULE], f'console.log(JSON.stringify({_M}.shiftExpression("1-4, 6, 10-11", -1)));')
    assert out == "0-3, 5, 9-10"


def test_shift_expression_forward():
    out = run_node([_MODULE], f'console.log(JSON.stringify({_M}.shiftExpression("0-3, 5", 1)));')
    assert out == "1-4, 6"


def test_shift_expression_leaves_unrecognised_and_nonstring():
    out = run_node([_MODULE], f"""
        const M = {_M};
        console.log(JSON.stringify({{ elems: M.shiftExpression("Au,C", -1), num: M.shiftExpression(42, -1) }}));
    """)
    assert out == {"elems": "Au, C", "num": 42}


def test_viewer_index_labels_conform_to_l1():
    """The standalone 3D viewer (mol-viewer-embed.js) inlines its atom-index label because it
    can't import L1; bind that inline convention to L1's toDisplay so the two can't drift."""
    delta = run_node([_MODULE], f'console.log(JSON.stringify({_M}.toDisplay(7) - 7));')
    assert delta == 1, f"L1 toDisplay delta is {delta}; update the viewer + this test"
    viewer = _EMBED.read_text()
    assert re.search(rf"String\(\s*i\s*\+\s*{delta}\s*\)", viewer), (
        "auto atom-index label must apply L1's display delta (i + %d)" % delta)
    assert re.search(rf"String\(\s*idx\s*\+\s*{delta}\s*\)", viewer), (
        "picked-atom index label must apply L1's display delta (idx + %d)" % delta)
