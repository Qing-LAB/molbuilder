"""Unit tests for ``lib/selection/measurements.js`` — the shared
xyz / distance / angle helpers used by the Modify-tab Selection
panel and the Results-tab trajectory inspector.

Run by spawning Node with a tiny harness that loads the module
(it has a ``module.exports`` branch alongside the
``window.molbuilder.selection.measurements`` registration).

The math here is small but load-bearing — both consumers display
the result to the user, and drift between the two (different
decimal counts, off-by-one indexing) is the original sin that
prompted the extraction.  Pin every contract.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "molbuilder/web/static/lib/selection/measurements.js"


def _run_node(snippet: str) -> object:
    if shutil.which("node") is None:
        pytest.skip("Node not available in this environment")
    harness = (
        f"const m = require({json.dumps(str(MOD))});\n"
        + snippet
    )
    out = subprocess.run(
        ["node", "-e", harness],
        check=True, capture_output=True, text=True,
        timeout=30,
    )
    return json.loads(out.stdout)


WATER = [
    [0.000, 0.000, 0.000],   # 0 = O
    [0.957, 0.000, 0.000],   # 1 = H
    [-0.239, 0.927, 0.000],  # 2 = H
]
WATER_META = [
    {"index": 0, "element": "O", "atom_name": ""},
    {"index": 1, "element": "H", "atom_name": ""},
    {"index": 2, "element": "H", "atom_name": ""},
]


def test_empty_selection_returns_null():
    assert _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([], {json.dumps(WATER_META)}, {json.dumps(WATER)})"
        f"));"
    ) is None


def test_four_atoms_returns_null():
    assert _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0,1,2,0], {json.dumps(WATER_META)}, {json.dumps(WATER)})"
        f"));"
    ) is None


def test_xyz_single_atom():
    """One atom selected → kind='xyz', xyz coords match positions[idx],
    display uses 1-based label + 3-decimal coords."""
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0], {json.dumps(WATER_META)}, {json.dumps(WATER)})"
        f"));"
    )
    assert r["kind"]    == "xyz"
    assert r["indices"] == [0]
    assert r["labels"]  == ["O #1"]
    assert r["xyz"]     == [0.0, 0.0, 0.0]
    assert r["display"] == "O #1 — (0.000, 0.000, 0.000) Å"


def test_distance_two_atoms():
    """Two atoms → distance.  Water O–H bond is 0.957 Å exactly per
    the fixture coords; the math should round-trip to 4 decimals."""
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0, 1], {json.dumps(WATER_META)}, {json.dumps(WATER)})"
        f"));"
    )
    assert r["kind"]      == "distance"
    assert r["indices"]   == [0, 1]
    assert abs(r["valueAng"] - 0.957) < 1e-6
    assert r["display"]   == "|O #1 – H #2| = 0.9570 Å"


def test_angle_with_pickorder_uses_middle_click_as_vertex():
    """When pickOrder is supplied (the user's click sequence), the
    SECOND click is the vertex — the chemist's convention "pick A,
    then B (vertex), then C" so the angle is ∠A-B-C."""
    # Right-angle geometry: atoms at +x, origin, +y.  Pick order
    # [0, 1, 2] (clicked +x first, origin second, +y third) → the
    # vertex is the origin at index 1; 90° is the answer.
    pos = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    meta = [{"element": "X"}] * 3
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0,1,2], {json.dumps(meta)}, {json.dumps(pos)},"
        f"           [0,1,2])"
        f"));"
    )
    assert r["vertexIndex"] == 1
    assert abs(r["valueDeg"] - 90.0) < 1e-9

    # Same atoms, but the user clicked them as [+y, origin, +x] →
    # pickOrder=[2, 1, 0].  Vertex is still index 1 (the middle
    # click); angle is still 90°.
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0,1,2], {json.dumps(meta)}, {json.dumps(pos)},"
        f"           [2,1,0])"
        f"));"
    )
    assert r["vertexIndex"] == 1
    assert abs(r["valueDeg"] - 90.0) < 1e-9

    # User picked the +x atom in the middle — vertex is index 0
    # now, angle 45° (between origin→+x and +y→+x).
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0,1,2], {json.dumps(meta)}, {json.dumps(pos)},"
        f"           [1,0,2])"
        f"));"
    )
    assert r["vertexIndex"] == 0
    assert abs(r["valueDeg"] - 45.0) < 1e-9


def test_angle_falls_back_to_geometric_vertex_without_pickorder():
    """When pickOrder is missing or stale (filter eval, session
    restore), fall back to the geometric heuristic — vertex = atom
    minimising sum-of-distances to the other two.  For water this
    picks O regardless of selection order."""
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0, 1, 2], {json.dumps(WATER_META)}, {json.dumps(WATER)})"
        f"));"
    )
    assert r["kind"] == "angle"
    assert r["vertexIndex"] == 0
    assert abs(r["valueDeg"] - 104.5) < 0.5


def test_angle_uses_geometric_fallback_when_pickorder_has_duplicates():
    """A pickOrder that repeats an atom (corrupt or post-merge
    leftover) would, without the dedup guard, feed two identical
    vectors to angleDeg and emit NaN.  Pin that the helper falls
    back to the geometric vertex instead so the readout stays
    sane."""
    pos = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    meta = [{"element": "X"}] * 3
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0,1,2], {json.dumps(meta)}, {json.dumps(pos)},"
        f"           [0,0,1])"
        f"));"
    )
    # Geometric fallback → vertex at the origin (index 1) → 90°.
    assert r["vertexIndex"] == 1
    assert abs(r["valueDeg"] - 90.0) < 1e-9


def test_angle_uses_geometric_fallback_when_pickorder_is_stale():
    """A pickOrder that doesn't match the selection set (different
    atoms, different length, or junk) is treated as missing — the
    geometric heuristic kicks in.  Pin that path so a future
    refactor doesn't silently honour a corrupt pickOrder."""
    pos = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    meta = [{"element": "X"}] * 3
    # pickOrder length mismatch.
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0,1,2], {json.dumps(meta)}, {json.dumps(pos)},"
        f"           [0,1])"
        f"));"
    )
    assert r["vertexIndex"] == 1  # geometric fallback

    # pickOrder names atoms not in the selection.
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0,1,2], {json.dumps(meta)}, {json.dumps(pos)},"
        f"           [0,1,99])"
        f"));"
    )
    assert r["vertexIndex"] == 1


def test_distance_uses_pickorder_for_label_direction():
    """Even though distance is symmetric, the label "A – B" reads
    naturally as the user's click direction.  pickOrder=[1, 0]
    should label H first, O second; selection=[0, 1] (sorted)
    alone would give the reverse."""
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0, 1], {json.dumps(WATER_META)}, {json.dumps(WATER)},"
        f"           [1, 0])"
        f"));"
    )
    assert r["display"] == "|H #2 – O #1| = 0.9570 Å"


def test_angle_collinear_atoms_returns_180():
    """Three collinear atoms — the angle is 180°.  Cosine clamp must
    handle the boundary cleanly (a floating-point dot product slightly
    over 1 would otherwise NaN out of acos)."""
    pos = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    meta = [{"element": "X"}] * 3
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0, 1, 2], {json.dumps(meta)}, {json.dumps(pos)})"
        f"));"
    )
    assert abs(r["valueDeg"] - 180.0) < 1e-9


def test_atom_name_overrides_element_in_label():
    """Atom_name (e.g. PDB ``OE1``) is more informative than the
    raw element + index for users reading off a structure — when
    present, the label must use it."""
    meta = [
        {"index": 0, "element": "O", "atom_name": "OE1"},
        {"index": 1, "element": "C", "atom_name": "CD"},
    ]
    pos = [[0,0,0], [1,0,0]]
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0,1], {json.dumps(meta)}, {json.dumps(pos)})"
        f"));"
    )
    assert r["labels"] == ["OE1 #1", "CD #2"]


def test_out_of_range_selection_index_returns_null():
    """If a selection index sits outside the positions array (op
    just changed the atom count, positions hasn't caught up), bail
    cleanly rather than indexing into ``undefined``."""
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([5, 6], {json.dumps(WATER_META)}, {json.dumps(WATER)})"
        f"));"
    )
    assert r is None


def test_missing_positions_returns_null():
    """Caller didn't pass a positions array (or passed null) —
    return null rather than throwing."""
    r = _run_node(
        f"console.log(JSON.stringify("
        f" m.compute([0], {json.dumps(WATER_META)}, null)"
        f"));"
    )
    assert r is None
