"""molview.mountRender -- "molview owns the render" (molview-module.md §14).

Node integration test: loads the REAL kgrid + render-pipeline + render modules and drives
mountRender with a stubbed viewer handle + workspace + store.  Pins that the render reads
the atoms from the STORE (the same source the panel lists -- so viewer and panel can never
diverge), builds the xyz from those atoms (not a stale canvas text), re-draws only when the
STRUCTURE changes (not on a selection click), and tiles by the WORKSPACE's k-grid dims.
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
    ROOT / "molbuilder/web/static/lib/molview/render.js",
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


# Stub handle + workspace + store.  The ATOMS live on the STORE (as they do in the real
# store snapshot) -- getUnit reads them there.  The workspace supplies the cell + k-grid dims
# [2,1,1] (deliberately != the store's kgrid.dims, to prove dims come from the workspace).
_HARNESS = """
    const calls = [];
    const handle = { setStructure: (o) => calls.push(o), getAtomCoords: () => [[0,0,0],[1,0,0]] };
    const wsSubs = [], storeSubs = [];
    let storeState = { kgrid: { enabled: false, dims: [1,1,1] }, indices: [], isolate: false,
                       atoms: [{ element:'C', x:0, y:0, z:0 }, { element:'H', x:1, y:0, z:0 }] };
    const cell = [[10,0,0],[0,10,0],[0,0,10]];
    const workspace = {
        getUnitCellInfo: () => ({ value: cell }),
        getKgrid: () => [2, 1, 1],
        subscribe: (fn) => { wsSubs.push(fn);
            return () => { const i = wsSubs.indexOf(fn); if (i>=0) wsSubs.splice(i,1); }; },
    };
    const store = {
        getState: () => storeState,
        subscribe: (fn) => { storeSubs.push(fn);
            return () => { const i = storeSubs.indexOf(fn); if (i>=0) storeSubs.splice(i,1); }; },
    };
    const mountRender = global.molbuilder.molview.mountRender;
"""


def test_render_draws_the_structure_from_the_store_atoms_on_mount():
    out = _run_node(_HARNESS + """
        mountRender(handle, workspace, store, {});
        console.log(JSON.stringify({ n: calls.length, head: calls[0].xyz.split('\\n')[0] }));
    """)
    assert out["n"] == 1              # one base draw
    assert out["head"] == "2"        # 2 atoms, read from the STORE (not workspace.getStructure)


def test_render_redraws_on_structure_change_not_on_selection_click():
    out = _run_node(_HARNESS + """
        mountRender(handle, workspace, store, {});
        const afterMount = calls.length;
        // selection-only change (same atoms) -> must NOT redraw the base
        storeState = Object.assign({}, storeState, { indices: [0] });
        storeSubs.forEach((fn) => fn());
        const afterClick = calls.length;
        // an atoms change (a load) -> MUST redraw, from the new atoms
        storeState = Object.assign({}, storeState, { atoms: [{ element:'O', x:5, y:0, z:0 }] });
        storeSubs.forEach((fn) => fn());
        console.log(JSON.stringify({ afterMount, afterClick, afterLoad: calls.length,
                                     loadHead: calls[calls.length-1].xyz.split('\\n')[0] }));
    """)
    assert out["afterMount"] == 1
    assert out["afterClick"] == 1     # a selection click did NOT redraw the base
    assert out["afterLoad"] == 2      # an atoms change (load) DID redraw
    assert out["loadHead"] == "1"    # the new 1-atom structure


def test_render_tiles_when_kgrid_enabled_using_workspace_dims():
    out = _run_node(_HARNESS + """
        mountRender(handle, workspace, store, {});
        storeState = Object.assign({}, storeState, { kgrid: { enabled: true, dims: [1,1,1] } });
        storeSubs.forEach((fn) => fn());   // enable toggle -> tile
        console.log(JSON.stringify({ head: calls[calls.length-1].xyz.split('\\n')[0] }));
    """)
    # dims come from ws.getKgrid()=[2,1,1], NOT the store's [1,1,1]: 2 store atoms x 2 = 4
    assert out["head"] == "4"


def test_render_dispose_unsubscribes_everything():
    out = _run_node(_HARNESS + """
        const r = mountRender(handle, workspace, store, {});
        const beforeStore = storeSubs.length, beforeWs = wsSubs.length;
        r.dispose();
        console.log(JSON.stringify({ beforeStore, beforeWs,
                                     afterStore: storeSubs.length, afterWs: wsSubs.length }));
    """)
    # kg + the structure-redraw both subscribe to the store; the redraw also subscribes to ws
    assert out["beforeStore"] == 2 and out["afterStore"] == 0
    assert out["beforeWs"] == 1 and out["afterWs"] == 0
