"""Render-pipeline compose -- assembles § 6.3 layers 2-3 (selection/isolate +
k-grid) into the displayed positions + a global sourceIndex.

Node unit test: selection/isolate filtering, k-grid tiling of the visible set,
and the LOCAL->GLOBAL sourceIndex remap so element/decoration lookup stays right.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    ROOT / "molbuilder/web/static/lib/molview/kgrid.js",
    ROOT / "molbuilder/web/static/lib/molview/render-pipeline.js",
]


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    full = ("global.window = global;\n"
            + "\n".join(m.read_text() for m in MODULES) + "\n" + snippet)
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n"
                    f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# 3 atoms on the x-axis; a 10 A cubic cell.
_SETUP = (
    "const cr = global.molbuilder.molview.computeRender;\n"
    "const coords = [[0,0,0],[1,0,0],[2,0,0]];\n"
    "const cell = [[10,0,0],[0,10,0],[0,0,10]];\n"
)


def test_no_view_shows_all_atoms_identity_source():
    out = _run_node(_SETUP + "console.log(JSON.stringify(cr(coords, {}, null)));")
    assert out["positions"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert out["sourceIndex"] == [0, 1, 2]


def test_isolate_filters_to_selection():
    out = _run_node(_SETUP + """
        console.log(JSON.stringify(cr(coords, { isolate: true, indices: [0, 2] }, null)));
    """)
    assert out["positions"] == [[0, 0, 0], [2, 0, 0]]
    assert out["sourceIndex"] == [0, 2]        # global indices preserved


def test_selection_without_isolate_does_not_filter():
    out = _run_node(_SETUP + """
        console.log(JSON.stringify(cr(coords, { isolate: false, indices: [0] }, null)));
    """)
    assert out["positions"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]   # all visible
    assert out["sourceIndex"] == [0, 1, 2]


def test_kgrid_tiles_all_visible_atoms():
    out = _run_node(_SETUP + """
        console.log(JSON.stringify(cr(coords, { kgrid: { enabled: true, dims: [2,1,1] } }, cell)));
    """)
    # original cell (3 atoms) then image shifted by a=[10,0,0].
    assert out["positions"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0],
                                [10, 0, 0], [11, 0, 0], [12, 0, 0]]
    assert out["sourceIndex"] == [0, 1, 2, 0, 1, 2]


def test_isolate_then_kgrid_tiles_only_selected_with_global_source():
    out = _run_node(_SETUP + """
        console.log(JSON.stringify(cr(coords,
            { isolate: true, indices: [1], kgrid: { enabled: true, dims: [2,1,1] } }, cell)));
    """)
    # visible = {atom 1}; tiled 2x along a.  sourceIndex stays GLOBAL (1), not the
    # tile-local 0.
    assert out["positions"] == [[1, 0, 0], [11, 0, 0]]
    assert out["sourceIndex"] == [1, 1]


# ---- mountKgridRender: THE module k-grid render controller ---------- #

_CTL_SETUP = """
    function makeStore(initial) {
        let state = initial; const subs = [];
        return {
            getState: () => state,
            subscribe: (fn) => { subs.push(fn); return () => {
                const i = subs.indexOf(fn); if (i >= 0) subs.splice(i, 1); }; },
            _set: (p) => { state = Object.assign({}, state, p); subs.forEach(fn => fn(state)); },
            _subCount: () => subs.length,
        };
    }
    const store = makeStore({ kgrid: { enabled: false, dims: [2,1,1] },
                              indices: [], isolate: false });
    const handleCalls = [];
    const handle = { setStructure: (o) => handleCalls.push(o) };
    const ctl = global.molbuilder.molview.mountKgridRender(handle, store, {
        getUnit: () => ({ coords: [[0,0,0],[1,0,0]], elements: ['C','H'], xyz: 'UNIT' }),
        getCell: () => [[10,0,0],[0,10,0],[0,0,10]],
    });
"""


def test_mount_kgrid_render_tiles_on_enable_restores_on_disable():
    out = _run_node(_CTL_SETUP + """
        const afterMount = handleCalls.length;                     // off -> no-op
        store._set({ kgrid: { enabled: true, dims: [2,1,1] } });   // enable -> tile
        const enabledN = handleCalls[handleCalls.length-1].xyz.split('\\n')[0];
        store._set({ kgrid: { enabled: false, dims: [2,1,1] } });  // disable -> restore
        const restored = handleCalls[handleCalls.length-1].xyz;
        ctl.dispose();
        console.log(JSON.stringify({
            afterMount, enabledN, restored, subs: store._subCount(),
        }));
    """)
    assert out["afterMount"] == 0        # k-grid off at mount -> no render
    assert out["enabledN"] == "4"        # 2 unit atoms x 2x1x1 = 4-atom supercell
    assert out["restored"] == "UNIT"     # disable restores the unit-cell xyz
    assert out["subs"] == 0              # dispose() unsubscribed


def test_mount_kgrid_render_no_cell_does_not_tile():
    out = _run_node(_CTL_SETUP.replace(
        "getCell: () => [[10,0,0],[0,10,0],[0,0,10]],",
        "getCell: () => null,") + """
        store._set({ kgrid: { enabled: true, dims: [2,1,1] } });   // enabled but no cell
        console.log(JSON.stringify({ calls: handleCalls.length }));
    """)
    assert out["calls"] == 0             # no cell -> nothing tiles (inert, not crash)
