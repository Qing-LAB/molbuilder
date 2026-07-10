"""Overlay controller for MolView (molview-module.md §14.5.1).

MolView is a VIEWER: it DRAWS overlays it is HANDED; it does NOT generate them.  The consumer
computes arrows (from its own force data, with its OWN normalization) + labels and pushes them
via setArrows / setLabels; this controller forwards them to the embed verbatim and re-applies
them across a redraw (a per-frame setStructure clears embed overlays).  These tests pin:
draw-what-handed (verbatim), persist-across-redraw, dispose clears + stops re-applying, and that
the controller reads NO force data / synthesizes NO geometry (the design the viewer must NOT do).
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


# A stubbed embed handle (records the last setArrows/setLabels) + a sync-notify store.  The
# overlay controller takes ONLY (handle, store) -- no workspace, no force data.
_HARNESS = """
    const handle = { arrows: null, labels: null,
        setArrows: (a) => { handle.arrows = a; },
        setLabels: (l) => { handle.labels = l; } };
    const subs = [];
    const store = { subscribe: (fn) => { subs.push(fn); return () => {}; },
                    notify: () => subs.forEach(fn => fn()) };
    const ov = global.molbuilder.molview.mountOverlays(handle, store);
"""


def test_setArrows_draws_exactly_what_it_is_handed():
    """The consumer hands opaque arrow specs; the viewer draws them VERBATIM (no build, no
    normalize)."""
    out = _run_node(_HARNESS + """
        const arrows = [{start:[0,0,0], end:[1,0,0], color:'#f0a020', radius:0.05}];
        ov.setArrows(arrows);
        console.log(JSON.stringify(handle.arrows));
    """)
    assert out == [{"start": [0, 0, 0], "end": [1, 0, 0], "color": "#f0a020", "radius": 0.05}]


def test_setLabels_draws_exactly_what_it_is_handed():
    out = _run_node(_HARNESS + """
        ov.setLabels({atoms:'all', format:'index'});
        const on = handle.labels;
        ov.setLabels(false);
        console.log(JSON.stringify({on, off: handle.labels}));
    """)
    assert out["on"] == {"atoms": "all", "format": "index"}
    assert out["off"] is False


def test_overlays_persist_verbatim_across_a_redraw():
    """A per-frame setStructure clears the embed's overlays; a store change re-applies the
    CONSUMER's last-set arrows/labels EXACTLY -- the controller never recomputes them."""
    out = _run_node(_HARNESS + """
        ov.setArrows([{start:[0,0,0], end:[2,0,0]}]);
        ov.setLabels({atoms:'all', format:'index'});
        handle.arrows = 'CLEARED'; handle.labels = 'CLEARED';   // the redraw wiped them
        store.notify();                                          // a redraw happened
        console.log(JSON.stringify({arrows: handle.arrows, labels: handle.labels}));
    """)
    assert out["arrows"] == [{"start": [0, 0, 0], "end": [2, 0, 0]}]
    assert out["labels"] == {"atoms": "all", "format": "index"}


def test_controller_reads_no_force_data_and_synthesizes_nothing():
    """(task e2) The viewer must NOT pull force data or build arrow geometry.  Even with a
    workspace exposing currentForces present, the controller never touches it, and with no
    consumer-set overlay the drawn arrows stay empty."""
    out = _run_node(_HARNESS + """
        let forcesRead = 0;
        global.workspace = { currentForces: () => { forcesRead++; return [[1,0,0]]; } };
        store.notify(); store.notify();       // redraws, but the consumer set NO overlay
        console.log(JSON.stringify({forcesRead, arrows: handle.arrows}));
    """)
    assert out["forcesRead"] == 0             # never reached for force data
    assert out["arrows"] == []                # nothing synthesized (empty, not computed)


def test_dispose_clears_overlays_and_stops_reapplying():
    out = _run_node(_HARNESS + """
        ov.setArrows([{start:[0,0,0], end:[1,0,0]}]);
        ov.setLabels({atoms:'all', format:'index'});
        ov.dispose();
        const cleared = {arrows: handle.arrows, labels: handle.labels};
        handle.arrows = 'X'; handle.labels = 'X';
        store.notify();                        // must NOT re-apply after dispose
        console.log(JSON.stringify({cleared,
            afterNotify: {arrows: handle.arrows, labels: handle.labels}}));
    """)
    assert out["cleared"] == {"arrows": [], "labels": False}
    assert out["afterNotify"] == {"arrows": "X", "labels": "X"}
