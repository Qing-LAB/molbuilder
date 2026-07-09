"""molview.mountRender -- "molview owns the render" (molview-module.md §14).

Node integration test: loads the REAL kgrid + render-pipeline + render modules and drives
mountRender with a stubbed viewer handle + workspace + store.  Pins that the render reads
the structure through ws.* (no local copy), draws on mount, re-draws on a workspace change,
and tiles by the WORKSPACE's k-grid dims (not the store's).
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


# A stubbed viewer handle + workspace + store.  The workspace returns a 2-atom unit cell +
# k-grid dims [2,1,1]; the store's kgrid.dims is deliberately [1,1,1] so the test proves the
# tiling dims come from ws.getKgrid(), not the store.  subscribe()s are recorded, not fired.
_HARNESS = """
    const calls = [];
    const handle = { setStructure: (o) => calls.push(o), getAtomCoords: () => [[0,0,0],[1,0,0]] };
    const wsSubs = [], storeSubs = [];
    let storeState = { kgrid: { enabled: false, dims: [1,1,1] }, indices: [], isolate: false };
    const cell = [[10,0,0],[0,10,0],[0,0,10]];
    const workspace = {
        getStructure: () => ({ text: '2\\nx\\nC 0 0 0\\nH 1 0 0\\n',
                               atoms: [{element:'C',x:0,y:0,z:0},{element:'H',x:1,y:0,z:0}] }),
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


def test_render_draws_the_unit_cell_from_the_workspace_on_mount():
    out = _run_node(_HARNESS + """
        mountRender(handle, workspace, store, {});
        console.log(JSON.stringify({ n: calls.length, head: calls[0].xyz.split('\\n')[0] }));
    """)
    assert out["n"] == 1              # one base draw
    assert out["head"] == "2"        # the 2-atom unit cell, read from ws.getStructure()


def test_render_redraws_on_workspace_change():
    out = _run_node(_HARNESS + """
        mountRender(handle, workspace, store, {});
        const afterMount = calls.length;
        wsSubs.forEach((fn) => fn());   // a workspace change (load / edit)
        console.log(JSON.stringify({ afterMount, afterChange: calls.length }));
    """)
    assert out["afterMount"] == 1
    assert out["afterChange"] == 2   # re-drew on the workspace change


def test_render_tiles_when_kgrid_enabled_using_workspace_dims():
    out = _run_node(_HARNESS + """
        mountRender(handle, workspace, store, {});
        storeState = { kgrid: { enabled: true, dims: [1,1,1] }, indices: [], isolate: false };
        storeSubs.forEach((fn) => fn());   // enable toggle -> tile
        console.log(JSON.stringify({ head: calls[calls.length-1].xyz.split('\\n')[0] }));
    """)
    # dims come from ws.getKgrid()=[2,1,1], NOT the store's [1,1,1]: 2 unit atoms x 2 = 4
    assert out["head"] == "4"


def test_render_dispose_unsubscribes_everything():
    out = _run_node(_HARNESS + """
        const r = mountRender(handle, workspace, store, {});
        const beforeWs = wsSubs.length, beforeStore = storeSubs.length;
        r.dispose();
        console.log(JSON.stringify({ beforeWs, beforeStore,
                                     afterWs: wsSubs.length, afterStore: storeSubs.length }));
    """)
    assert out["beforeWs"] == 1 and out["afterWs"] == 0        # ws subscription torn down
    assert out["beforeStore"] == 1 and out["afterStore"] == 0  # k-grid's store sub torn down
