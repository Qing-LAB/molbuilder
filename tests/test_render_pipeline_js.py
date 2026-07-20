"""Render-pipeline compose -- assembles molview-module.md §14 layer 2
(selection/isolate) into the displayed positions + a global sourceIndex.

Node unit test: selection/isolate filtering and the LOCAL->GLOBAL sourceIndex
remap so element/decoration lookup stays right.  (k-grid tiling is gone: k-grid
is a reciprocal-space sampling knob on SiestaConfig, not a real-space repeat --
structure-periodicity.md.)
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
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


# 3 atoms on the x-axis.
_SETUP = (
    "const cr = global.molbuilder.molview.computeRender;\n"
    "const coords = [[0,0,0],[1,0,0],[2,0,0]];\n"
)


def test_no_view_shows_all_atoms_identity_source():
    out = _run_node(_SETUP + "console.log(JSON.stringify(cr(coords, {})));")
    assert out["positions"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert out["sourceIndex"] == [0, 1, 2]


def test_isolate_filters_to_selection():
    out = _run_node(_SETUP + """
        console.log(JSON.stringify(cr(coords, { isolate: true, indices: [0, 2] })));
    """)
    assert out["positions"] == [[0, 0, 0], [2, 0, 0]]
    assert out["sourceIndex"] == [0, 2]        # global indices preserved


def test_selection_without_isolate_does_not_filter():
    out = _run_node(_SETUP + """
        console.log(JSON.stringify(cr(coords, { isolate: false, indices: [0] })));
    """)
    assert out["positions"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]   # all visible
    assert out["sourceIndex"] == [0, 1, 2]


# ---- mountIsolateRender: THE module isolate render controller ------- #

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
    const store = makeStore({ indices: [], isolate: false });
    const handleCalls = [];
    const handle = { setStructure: (o) => handleCalls.push(o) };
    const ctl = global.molbuilder.molview.mountIsolateRender(handle, store, {
        getUnit: () => ({ coords: [[0,0,0],[1,0,0],[2,0,0]],
                          elements: ['C','H','O'], xyz: 'UNIT' }),
        getCell: () => [[10,0,0],[0,10,0],[0,0,10]],
    });
"""


def test_isolate_filters_the_render_list_on_enable_restores_on_disable():
    # The controller holds NO store subscription -- the render streamline reads the flag
    # + selection and calls refresh() (render.js onStoreChange).  So the test drives it the
    # same way: set the flag, then refresh().
    out = _run_node(_CTL_SETUP + """
        const afterMount = handleCalls.length;                      // off -> no-op
        store._set({ isolate: true, indices: [1] }); ctl.refresh(); // enable -> filter
        const enabledN = handleCalls[handleCalls.length-1].xyz.split('\\n')[0];
        store._set({ isolate: false }); ctl.refresh();              // disable -> restore
        const restored = handleCalls[handleCalls.length-1].xyz;
        ctl.dispose();
        console.log(JSON.stringify({
            afterMount, enabledN, restored, subs: store._subCount(),
        }));
    """)
    assert out["afterMount"] == 0        # isolate off at mount -> no derived draw
    assert out["enabledN"] == "1"        # only the 1 selected atom is drawn
    assert out["restored"] == "UNIT"     # disable restores the unit-cell xyz
    assert out["subs"] == 0              # controller holds NO subscription of its own


def test_isolate_with_empty_selection_does_not_derive():
    out = _run_node(_CTL_SETUP + """
        store._set({ isolate: true, indices: [] }); ctl.refresh();  // isolate on, nothing selected
        console.log(JSON.stringify({ calls: handleCalls.length }));
    """)
    assert out["calls"] == 0             # no "selected only" of zero atoms -> plain base draw
