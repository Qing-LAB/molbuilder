"""Measurement overlay -- module decoration (atom-annotations.md § 6.4).

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
