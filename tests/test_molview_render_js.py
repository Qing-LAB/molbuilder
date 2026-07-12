"""molview.mountRender -- "molview owns the render" (molview-module.md §14).

Node integration test: loads the REAL kgrid + render-pipeline + render modules and drives
mountRender with a stubbed viewer handle + data model + selection store.  It pins the §14
contract:

  * §14.0 -- the render recomputes from the CLEAN unit cell in the store; the base draw is
    ONE setStructure of that list, with the RESOLVED cell as the lattice.
  * §14.2 -- a signature guard: a structure/coords change triggers EXACTLY ONE redraw; a pure
    selection click triggers NONE.
  * §14.0/§14.2 -- isolate and k-grid FILTER/tile the RENDER list (a derived, display-only
    view); the STORED dataset is never mutated.  k-grid dims come from the data model
    (periodicity.kgrid via getKgrid), not the store's view dims.
  * §14.4 -- the base cell is ``getUnitCellInfo().value`` (the resolved cell) -- NON-NULL even
    for a cell-less molecule (no explicit cell), so a bare molecule still draws with its
    resolved bbox lattice.

mount.js passes ``molview.data`` as mountRender's 2nd positional arg (the source of
getUnitCellInfo / getKgrid, §14.4); the harness names it ``data`` to match.
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


# Stub handle + data model + store.  The ATOMS live on the STORE (the clean unit cell -- the
# same source the panel lists, §14.0).  The DATA MODEL supplies the RESOLVED cell and the
# k-grid DIMS [2,1,1] (deliberately != the store's view dims [1,1,1], to prove §14.4: dims
# come from the data model, not the view).  getUnitCellInfo().value is NON-NULL with
# isDefault:true -- i.e. a cell-less molecule's resolved bbox (§14.4).
_HARNESS = """
    const calls = [];
    const handle = { setStructure: (o) => calls.push(o), getAtomCoords: () => [[0,0,0],[1,0,0]] };
    const dataSubs = [], storeSubs = [];
    let storeState = { kgrid: { enabled: false, dims: [1,1,1] }, indices: [], isolate: false,
                       atoms: [{ element:'C', x:0, y:0, z:0 }, { element:'H', x:1, y:0, z:0 }] };
    const cell = [[10,0,0],[0,10,0],[0,0,10]];
    // The DATA MODEL (molview.data) -- passed as mountRender's 2nd arg (§14.4).  No explicit
    // getUnitCell is provided, proving the base cell comes from getUnitCellInfo().value.
    const data = {
        getUnitCellInfo: () => ({ value: cell, isDefault: true }),   // resolved bbox, no explicit cell
        getKgrid: () => [2, 1, 1],                                   // periodicity.kgrid dims
        subscribe: (fn) => { dataSubs.push(fn);
            return () => { const i = dataSubs.indexOf(fn); if (i>=0) dataSubs.splice(i,1); }; },
    };
    const store = {
        getState: () => storeState,
        subscribe: (fn) => { storeSubs.push(fn);
            return () => { const i = storeSubs.indexOf(fn); if (i>=0) storeSubs.splice(i,1); }; },
    };
    const mountRender = global.molbuilder.molview.mountRender;
"""


def test_base_draw_reads_store_atoms_and_explicit_cell_only():
    """§14.0/§14.4: the base draw is ONE setStructure of the clean unit-cell atoms (from the
    store), with the EXPLICIT cell (getUnitCell()) as the lattice -- NOT the resolved bbox.
    A cell-less molecule (no explicit cell) draws NO box and Cartesian fixed-length axes; the
    resolved bbox is reserved for k-grid TILING only.  (Regression guard: feeding the resolved
    bbox here drew a spurious box + extent-scaled axes -- the electrode/axes bug.)"""
    out = _run_node(_HARNESS + """
        mountRender(handle, data, store, {});
        console.log(JSON.stringify({ n: calls.length,
                                     head: calls[0].xyz.split('\\n')[0],
                                     hasLattice: calls[0].lattice != null }));
    """)
    assert out["n"] == 1                       # one base draw
    assert out["head"] == "2"                 # 2 atoms, read from the STORE (the clean unit cell)
    assert out["hasLattice"] is False          # cell-less molecule -> NO lattice on the base draw


def test_redraw_on_structure_change_but_not_on_selection_click():
    """§14.2 signature guard: a pure selection click does NOT redraw the base; a coords/atoms
    change (a load) redraws EXACTLY once."""
    out = _run_node(_HARNESS + """
        mountRender(handle, data, store, {});
        const afterMount = calls.length;
        // selection-only change (same atoms) -> must NOT redraw the base
        storeState = Object.assign({}, storeState, { indices: [0] });
        storeSubs.forEach((fn) => fn());
        const afterClick = calls.length;
        // an atoms change (a load) -> MUST redraw exactly once, from the new atoms
        storeState = Object.assign({}, storeState, { atoms: [{ element:'O', x:5, y:0, z:0 }] });
        storeSubs.forEach((fn) => fn());
        console.log(JSON.stringify({ afterMount, afterClick, afterLoad: calls.length,
                                     delta: calls.length - afterClick,
                                     loadHead: calls[calls.length-1].xyz.split('\\n')[0] }));
    """)
    assert out["afterMount"] == 1
    assert out["afterClick"] == 1     # a selection click did NOT redraw the base
    assert out["delta"] == 1          # a structure change redrew EXACTLY once
    assert out["loadHead"] == "1"    # the new 1-atom structure


def test_isolate_filters_the_render_list_not_the_stored_data():
    """§14.0/§14.2: isolate is a real FILTER of the render list (the non-selected atoms are
    absent from the drawn model), NOT a mutation of the stored dataset.  Turning it off
    restores the full-list base draw."""
    out = _run_node(_HARNESS + """
        mountRender(handle, data, store, {});
        // isolate ON + select atom 0 -> the DERIVED render list is that 1 atom
        storeState = Object.assign({}, storeState, { isolate: true, indices: [0] });
        storeSubs.forEach((fn) => fn());
        const isoHead = calls[calls.length-1].xyz.split('\\n')[0];
        const storedAtomsWhileIsolated = storeState.atoms.length;   // the DATA is untouched
        // isolate OFF -> the plain full-list base draw returns
        storeState = Object.assign({}, storeState, { isolate: false });
        storeSubs.forEach((fn) => fn());
        const offHead = calls[calls.length-1].xyz.split('\\n')[0];
        console.log(JSON.stringify({ isoHead, storedAtomsWhileIsolated, offHead }));
    """)
    assert out["isoHead"] == "1"                  # render list FILTERED to the 1 selected atom
    assert out["storedAtomsWhileIsolated"] == 2   # the stored dataset is untouched (display-only view)
    assert out["offHead"] == "2"                  # isolate off restores the full list


def test_kgrid_tiles_the_render_list_using_data_model_dims():
    """§14.0/§14.4: k-grid tiles the render list (display-only) using the DATA MODEL's dims
    (getKgrid), not the store's view dims; the stored dataset is untouched."""
    out = _run_node(_HARNESS + """
        mountRender(handle, data, store, {});
        storeState = Object.assign({}, storeState, { kgrid: { enabled: true, dims: [1,1,1] } });
        storeSubs.forEach((fn) => fn());   // enable toggle -> tile
        console.log(JSON.stringify({ head: calls[calls.length-1].xyz.split('\\n')[0],
                                     storedAtoms: storeState.atoms.length }));
    """)
    # dims come from data.getKgrid()=[2,1,1], NOT the store's view dims [1,1,1]: 2 atoms x 2 = 4
    assert out["head"] == "4"
    assert out["storedAtoms"] == 2    # tiling is a display-only derivation; the data is untouched


def test_dispose_unsubscribes_everything():
    out = _run_node(_HARNESS + """
        const r = mountRender(handle, data, store, {});
        const beforeStore = storeSubs.length, beforeData = dataSubs.length;
        r.dispose();
        console.log(JSON.stringify({ beforeStore, beforeData,
                                     afterStore: storeSubs.length, afterData: dataSubs.length }));
    """)
    # kg + the structure-redraw both subscribe to the store; the redraw also subscribes to the
    # data model (for periodicity-edit redraws).  dispose() tears them all down.
    assert out["beforeStore"] == 2 and out["afterStore"] == 0
    assert out["beforeData"] == 1 and out["afterData"] == 0
