"""Measurement overlay -- module decoration (molview-module.md §15).

Node unit test: driven by the SELECTION (pickOrder), it shows position / distance
/ angle for 1 / 2 / 3 atoms and hides for 0 or >=4; coords come from a
coordsProvider (frame-independent store); dispose removes the overlay.  The real
measurements.js math is loaded (true integration).
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    ROOT / "molbuilder/web/static/lib/selection/measurements.js",
    ROOT / "molbuilder/web/static/lib/molview/measurement-overlay.js",
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


# Stubs: a document that makes capturable elements, a viewerHost, a sync-notify
# store, and a coordsProvider.  3 atoms; the overlay's element is overlays[0].
_HARNESS = """
    global.document = { createElement: () => ({ className:"", hidden:false,
        textContent:"", dataset:{}, parentNode:null }) };
    const overlays = [];
    const viewerHost = {
        appendChild: (e) => { e.parentNode = viewerHost; overlays.push(e); },
        removeChild: (e) => { e.parentNode = null; const i = overlays.indexOf(e); if (i>=0) overlays.splice(i,1); },
    };
    function makeStore(state) {
        const subs = [];
        return { getState: () => state,
                 subscribe: (fn) => { subs.push(fn); fn(state); return () => { subs.length = 0; }; },
                 set: (s) => { state = s; subs.slice().forEach((fn) => fn(state)); } };
    }
    const coords = [[0,0,0],[1,0,0],[0,1,0]];
    const atoms  = [{index:0,element:"H"},{index:1,element:"C"},{index:2,element:"O"}];
    const mk = global.molbuilder.molview.mountMeasurementOverlay;
    function snap() { const e = overlays[0] || {}; return { hidden: !!e.hidden, kind: e.dataset && e.dataset.kind, text: e.textContent }; }
"""


def test_overlay_shows_position_distance_angle_by_selection_count():
    out = _run_node(_HARNESS + """
        const store = makeStore({ pickOrder: [], indices: [], atoms });
        mk(viewerHost, { store, coordsProvider: () => coords });
        const seen = { start: snap() };
        store.set({ pickOrder: [0],     indices: [0],     atoms }); seen.one   = snap();
        store.set({ pickOrder: [0,1],   indices: [0,1],   atoms }); seen.two   = snap();
        store.set({ pickOrder: [0,1,2], indices: [0,1,2], atoms }); seen.three = snap();
        store.set({ pickOrder: [0,1,2,0], indices: [0,1,2], atoms }); seen.four = snap();
        console.log(JSON.stringify(seen));
    """)
    assert out["start"]["hidden"] is True                      # 0 atoms -> hidden
    assert out["one"]["hidden"] is False and out["one"]["kind"] == "xyz"
    assert out["two"]["hidden"] is False and out["two"]["kind"] == "distance"
    assert out["three"]["hidden"] is False and out["three"]["kind"] == "angle"
    assert out["four"]["hidden"] is True                       # 4 atoms -> hidden
    # each shown state carries a non-empty readout
    for k in ("one", "two", "three"):
        assert out[k]["text"]


def test_distance_readout_uses_coords_from_provider():
    # atoms 0 and 1 are 1.0 A apart along x; a distance readout must reflect that.
    out = _run_node(_HARNESS + """
        const store = makeStore({ pickOrder: [0,1], indices: [0,1], atoms });
        mk(viewerHost, { store, coordsProvider: () => coords });
        console.log(JSON.stringify(snap()));
    """)
    assert out["kind"] == "distance"
    assert "1.0" in out["text"]     # 1.000 A (exact format is measurements.js's)


def test_dispose_removes_the_overlay():
    out = _run_node(_HARNESS + """
        const store = makeStore({ pickOrder: [0], indices: [0], atoms });
        const h = mk(viewerHost, { store, coordsProvider: () => coords });
        const before = overlays.length;
        h.dispose();
        console.log(JSON.stringify({ before, after: overlays.length }));
    """)
    assert out["before"] == 1 and out["after"] == 0


def test_overlay_hidden_while_kgrid_enabled():
    """molview-module.md §14.3/§15: the measurement overlay STANDS DOWN while k-grid
    tiling is on -- the duplicated atoms make a unit-cell-indexed readout meaningless.
    A valid 2-atom pick that WOULD show a distance must stay hidden when kgrid.enabled."""
    out = _run_node(_HARNESS + """
        const store = makeStore({ pickOrder: [0,1], indices: [0,1], atoms,
                                  kgrid: { enabled: true } });
        mk(viewerHost, { store, coordsProvider: () => coords });
        console.log(JSON.stringify(snap()));
    """)
    assert out["hidden"] is True     # valid 2-atom pick, but k-grid on -> hidden


def test_measurement_works_under_isolate_via_global_rekey():
    """molview-module.md §14.3: under ISOLATE the viewer draws ONLY the selected atoms, so
    coordsProvider() returns them in visible (filtered) order.  The overlay must RE-KEY those
    coords back to GLOBAL atom index so the distance is between the right atoms -- isolate
    KEEPS the measurement (only k-grid pauses it).

    Select global atoms 1 & 2 -- coords (1,0,0) and (0,1,0) -- with isolate on; the viewer
    then draws just those two, so coordsProvider returns [[1,0,0],[0,1,0]] (visible order).
    The distance must be |(1,0,0)-(0,1,0)| = sqrt(2) ~ 1.414, proving the re-key mapped the
    filtered coords onto picks 1 & 2 (a naive positions[1]/positions[2] read of the filtered
    array would measure the wrong pair or hide)."""
    out = _run_node(_HARNESS + """
        // isolate on, atoms 1 & 2 selected -> viewer draws only those, in visible order.
        const filtered = [coords[1], coords[2]];   // what getAtomCoords returns under isolate
        const store = makeStore({ pickOrder: [1,2], indices: [1,2], atoms,
                                  isolate: true, kgrid: { enabled: false } });
        mk(viewerHost, { store, coordsProvider: () => filtered });
        console.log(JSON.stringify(snap()));
    """)
    assert out["hidden"] is False, "measurement must stay visible under isolate"
    assert out["kind"] == "distance"
    assert "1.41" in out["text"], (
        f"distance must be sqrt(2)~1.414 (re-keyed to atoms 1 & 2); got {out['text']!r}")


def test_angle_vertex_is_pickOrder_second_not_index_order():
    """molview-module.md §15: the angle VERTEX is pickOrder[1] -- the 2nd atom the
    user CLICKED -- not the middle-by-index.  Same three atoms, two different click
    orders => two different angles (45 vs 90 deg), proving the vertex follows the
    pick order.  coords: 0=(0,0,0) 1=(1,0,0) 2=(0,1,0)."""
    out = _run_node(_HARNESS + """
        const store = makeStore({ pickOrder: [0,1,2], indices: [0,1,2], atoms });
        mk(viewerHost, { store, coordsProvider: () => coords });
        const vAt1 = snap().text;                              // vertex = atom 1 -> 45 deg
        store.set({ pickOrder: [1,0,2], indices: [0,1,2], atoms });
        const vAt0 = snap().text;                              // vertex = atom 0 (2nd pick) -> 90 deg
        console.log(JSON.stringify({ vAt1, vAt0 }));
    """)
    assert "45" in out["vAt1"], f"vertex=atom1 should give 45 deg; got {out['vAt1']!r}"
    assert "90" in out["vAt0"], f"vertex=atom0 (2nd pick) should give 90 deg; got {out['vAt0']!r}"
