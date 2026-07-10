"""Frame-scoped overlays for MolView (molview-module.md §14.5.1).

Node unit test with a STUBBED embed handle (records setArrows/setLabels) + store + workspace.
Pins: force arrows are built from the current frame's forces (start=atom, end=atom+force,
gated by showForces + presence of forces); index labels toggle via setLabels; a store change
(a frame swap) rebuilds; dispose clears both.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "molbuilder/web/static/lib/molview/frame-overlays.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    full = "global.window = global;\n" + MOD.read_text() + "\n" + snippet
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\nstderr:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# A stubbed handle (records the last setArrows/setLabels), a sync-notify store whose atoms are
# the current frame's coords, and a workspace whose currentForces() returns the frame's forces.
_HARNESS = """
    const handle = { arrows: null, labels: null,
        setArrows: (a) => { handle.arrows = a; },
        setLabels: (l) => { handle.labels = l; } };
    let _atoms  = [{x:0,y:0,z:0},{x:1,y:0,z:0}];
    let _forces = [[0.5,0,0],[0,0,0]];
    const subs = [];
    const store = { getState: () => ({ atoms: _atoms }),
                    subscribe: (fn) => { subs.push(fn); return () => {}; },
                    notify: () => subs.forEach(fn => fn()) };
    const workspace = { currentForces: () => _forces };
    let showForces = false, showIndices = false;
    const mount = global.molbuilder.molview.mountFrameOverlays;
    const ov = mount(handle, workspace, store,
        { getShowForces: () => showForces, getShowIndices: () => showIndices });
"""


def test_force_arrows_built_from_the_current_frames_forces_when_on():
    out = _run_node(_HARNESS + """
        showForces = true; ov.refresh();
        console.log(JSON.stringify(handle.arrows));
    """)
    # atom 0 has force [0.5,0,0] -> arrow start (0,0,0) -> end (0.5,0,0); atom 1's force is ~0.
    assert len(out) == 1
    assert out[0]["start"] == [0, 0, 0] and out[0]["end"] == [0.5, 0, 0]


def test_forces_off_clears_the_arrows():
    out = _run_node(_HARNESS + """
        showForces = true; ov.refresh();
        showForces = false; ov.refresh();
        console.log(JSON.stringify({ arrows: handle.arrows }));
    """)
    assert out["arrows"] == []


def test_index_labels_toggle_via_setLabels():
    out = _run_node(_HARNESS + """
        showIndices = true; ov.refresh();  const on = handle.labels;
        showIndices = false; ov.refresh(); const off = handle.labels;
        console.log(JSON.stringify({ on, off }));
    """)
    assert out["on"] == {"atoms": "all", "format": "index"}
    assert out["off"] is False


def test_a_frame_swap_rebuilds_the_arrows_from_the_new_frame():
    out = _run_node(_HARNESS + """
        showForces = true; ov.refresh();
        // simulate setFrame: new coords + new forces, then the store notifies.
        _atoms  = [{x:2,y:0,z:0},{x:1,y:0,z:0}];
        _forces = [[1.0,0,0],[0,0,0]];
        store.notify();
        console.log(JSON.stringify(handle.arrows));
    """)
    # atom 0 now at x=2 with force 1.0 -> arrow (2,0,0) -> (3,0,0)
    assert out[0]["start"] == [2, 0, 0] and out[0]["end"] == [3, 0, 0]


def test_dispose_clears_arrows_and_labels():
    out = _run_node(_HARNESS + """
        showForces = true; showIndices = true; ov.refresh();
        ov.dispose();
        console.log(JSON.stringify({ arrows: handle.arrows, labels: handle.labels }));
    """)
    assert out["arrows"] == [] and out["labels"] is False
