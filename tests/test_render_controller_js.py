"""Render-pipeline controller -- runs computeRender + FrameSet against an embed
(atom-annotations.md § 6.3).

Node unit test with a captured embed handle + a controllable stub store: initial
render, re-render on store change (k-grid, isolate), frame scrub, and dispose
stopping further renders.  The real molview compose layers (kgrid + pipeline +
frameset) are loaded so this is a true integration of the pieces.
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
    ROOT / "molbuilder/web/static/lib/molview/frameset.js",
    ROOT / "molbuilder/web/static/lib/molview/render-controller.js",
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


# A captured embed handle + a stub store with SYNCHRONOUS notify (deterministic).
_HARNESS = """
    const mv = global.molbuilder.molview;
    const captured = [];
    const handle = { setStructure: (o) => captured.push(o.xyz) };
    function makeStore(state) {
        const subs = [];
        return {
            getState: () => state,
            subscribe: (fn) => { subs.push(fn); fn(state); return () => { subs.length = 0; }; },
            set: (s) => { state = s; subs.slice().forEach((fn) => fn(state)); },
        };
    }
    const cell = [[10,0,0],[0,10,0],[0,0,10]];
"""


def test_initial_render_builds_xyz_from_elements_and_coords():
    out = _run_node(_HARNESS + """
        const fs = mv.createFrameSet([[[0,0,0],[1,0,0]]]);   // 1 frame, 2 atoms
        const store = makeStore({ indices: [], isolate: false, kgrid: { enabled:false, dims:[1,1,1] } });
        mv.createRenderController({ handle, frameSet: fs, store, elements: ["H","C"], cell });
        console.log(JSON.stringify({ n: captured.length, xyz: captured[captured.length-1] }));
    """)
    assert out["n"] == 1
    assert out["xyz"] == "2\n\nH 0.000000 0.000000 0.000000\nC 1.000000 0.000000 0.000000"


def test_store_change_re_renders_with_kgrid_tiling():
    out = _run_node(_HARNESS + """
        const fs = mv.createFrameSet([[[0,0,0],[1,0,0]]]);
        const store = makeStore({ indices: [], isolate: false, kgrid: { enabled:false, dims:[1,1,1] } });
        mv.createRenderController({ handle, frameSet: fs, store, elements: ["H","C"], cell });
        store.set({ indices: [], isolate: false, kgrid: { enabled:true, dims:[2,1,1] } });
        console.log(JSON.stringify({ n: captured.length, xyz: captured[captured.length-1] }));
    """)
    assert out["n"] == 2   # initial + one re-render on the store change
    # 4 atoms: original H,C then the +[10,0,0] image (same elements via sourceIndex)
    assert out["xyz"] == ("4\n\nH 0.000000 0.000000 0.000000\nC 1.000000 0.000000 0.000000\n"
                          "H 10.000000 0.000000 0.000000\nC 11.000000 0.000000 0.000000")


def test_isolate_re_renders_to_selected_only():
    out = _run_node(_HARNESS + """
        const fs = mv.createFrameSet([[[0,0,0],[1,0,0],[2,0,0]]]);
        const store = makeStore({ indices: [], isolate: false, kgrid: { enabled:false, dims:[1,1,1] } });
        mv.createRenderController({ handle, frameSet: fs, store, elements: ["H","C","O"], cell });
        store.set({ indices: [1], isolate: true, kgrid: { enabled:false, dims:[1,1,1] } });
        console.log(JSON.stringify({ xyz: captured[captured.length-1] }));
    """)
    assert out["xyz"] == "1\n\nC 1.000000 0.000000 0.000000"   # only global atom 1


def test_set_frame_scrubs_coords():
    out = _run_node(_HARNESS + """
        const fs = mv.createFrameSet([[[0,0,0]], [[1,0,0]], [[2,0,0]]]);   // 3 frames, 1 atom
        const store = makeStore({ indices: [], isolate: false, kgrid: { enabled:false, dims:[1,1,1] } });
        const ctl = mv.createRenderController({ handle, frameSet: fs, store, elements: ["H"], cell });
        ctl.setFrame(2);
        console.log(JSON.stringify({ n: captured.length, xyz: captured[captured.length-1] }));
    """)
    assert out["n"] == 2
    assert out["xyz"] == "1\n\nH 2.000000 0.000000 0.000000"   # frame 2 coords


def test_dispose_stops_re_render():
    out = _run_node(_HARNESS + """
        const fs = mv.createFrameSet([[[0,0,0]]]);
        const store = makeStore({ indices: [], isolate: false, kgrid: { enabled:false, dims:[1,1,1] } });
        const ctl = mv.createRenderController({ handle, frameSet: fs, store, elements: ["H"], cell });
        ctl.dispose();
        store.set({ indices: [], isolate: true, kgrid: { enabled:false, dims:[1,1,1] } });
        console.log(JSON.stringify({ n: captured.length }));
    """)
    assert out["n"] == 1   # only the initial render; dispose detached the subscription
