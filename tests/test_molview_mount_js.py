"""molview.mount -- the owner-facing handle (molview-module.md §18 / §D).

Node unit test with a STUBBED workspace + panel: the handle's read/notify surface
(getStructure / getSelection / onChange) reads the molecule THROUGH the workspace and
gives the owner ONE change channel -- exposing NO internals (no els / store / viewer).
The WRITE side (load / save / undo) lands in B2.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOUNT = ROOT / "molbuilder/web/static/lib/molview/mount.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    full = "global.window = global;\n" + MOUNT.read_text() + "\n" + snippet
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n"
                    f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# A stubbed workspace + panel + host, enough to drive molview.mount.  The workspace
# records subscribe()s so the test can fire a change; getStructure/getSelection return
# fixed data the handle must surface.
_HARNESS = """
    const wsSubs = [];
    const structure = { text: 'XYZ', atoms: [{ index: 0, x: 1 }] };
    const store = { getState: () => ({ indices: [1, 2] }), subscribe: () => (() => {}) };
    const workspace = {
        selection: store,
        getStructure: () => structure,
        subscribe: (fn) => { wsSubs.push(fn);
            return () => { const i = wsSubs.indexOf(fn); if (i >= 0) wsSubs.splice(i, 1); }; },
    };
    // Stub the panel module mount() composes (view-controls host is absent in the stub).
    global.molbuilder.selection = { mountPanel: async () => ({ panel: {}, dispose: () => {} }) };
    // A fused-card host stub (classList says it IS the card; no sub-hosts).
    const host = { classList: { contains: () => true }, querySelector: () => null };
    const mount = global.molbuilder.molview.mount;
"""


def test_handle_exposes_only_the_read_notify_surface_no_internals():
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then((h) => {
            console.log(JSON.stringify({ keys: Object.keys(h).sort(),
                                         text: h.getStructure().text }));
        });
    """)
    assert out["text"] == "XYZ"                      # getStructure reads through the workspace
    # ONLY the §D surface -- no els / store / viewerHandle leaked
    assert out["keys"] == ["dispose", "getSelection", "getStructure", "onChange"]


def test_handle_getSelection_returns_a_copy():
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then((h) => {
            const a = h.getSelection();
            a.push(999);                    // mutate the returned array
            const b = h.getSelection();     // must be a fresh copy
            console.log(JSON.stringify({ a, b }));
        });
    """)
    assert out["a"] == [1, 2, 999]
    assert out["b"] == [1, 2]              # unaffected -> getSelection returned a copy


def test_handle_onChange_is_the_one_workspace_change_channel():
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then((h) => {
            let n = 0;
            const off = h.onChange(() => { n++; });
            wsSubs.forEach((fn) => fn());   // a workspace change -> owner notified
            const afterFire = n;
            off();
            wsSubs.forEach((fn) => fn());   // no listeners now
            console.log(JSON.stringify({ afterFire, afterOff: n, subCount: wsSubs.length }));
        });
    """)
    assert out["afterFire"] == 1     # onChange fired on the workspace change
    assert out["afterOff"] == 1      # off() stopped further notifications
    assert out["subCount"] == 0      # unsubscribed cleanly


def test_dispose_tears_down_onChange_subscriptions():
    out = _run_node(_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then((h) => {
            h.onChange(() => {});
            h.onChange(() => {});
            const before = wsSubs.length;
            h.dispose();                    // must tear down onChange subs it handed out
            console.log(JSON.stringify({ before, after: wsSubs.length }));
        });
    """)
    assert out["before"] == 2
    assert out["after"] == 0          # dispose() unsubscribed both onChange listeners
