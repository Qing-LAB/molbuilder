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
MODULES = [
    ROOT / "molbuilder/web/static/lib/molview/kgrid.js",
    ROOT / "molbuilder/web/static/lib/molview/render-pipeline.js",
    ROOT / "molbuilder/web/static/lib/molview/render.js",
    ROOT / "molbuilder/web/static/lib/molview/mount.js",
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
    // A PRE-BUILT fused-card host stub (it IS the card + carries a .molview-panel), so mount
    // takes the wire-only path -- the B1 read/notify handle is the same on both paths.
    const host = { classList: { contains: () => true },
                   querySelector: (sel) => (sel === '.molview-panel' ? {} : null) };
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


# ---- The full-component path: an EMPTY host -> molview builds + embeds + owns the render -- #

_BUILD_HARNESS = """
    // Minimal DOM stub (build path uses direct refs -- no querySelector on the built tree).
    function mkEl(tag) {
        const el = { tagName: tag, className: '', textContent: '', children: [], parentNode: null };
        const cls = new Set();
        el.classList = { add: (x) => cls.add(x), contains: (x) => cls.has(x),
            toggle: (x) => { if (cls.has(x)) { cls.delete(x); return false; } cls.add(x); return true; } };
        el.appendChild = (c) => { el.children.push(c); c.parentNode = el; return c; };
        el.setAttribute = () => {}; el.addEventListener = () => {}; el.removeEventListener = () => {};
        el.querySelector = () => null; el.closest = () => null;
        return el;
    }
    global.document = { createElement: mkEl };
    // Stub the viewer: embed() synchronously reports ready with a recording handle.
    const setStructureCalls = []; let embeddedInto = null;
    global.molbuilder.viewer = { embed: (host, opts) => {
        embeddedInto = host;
        const h = { setStructure: (o) => setStructureCalls.push(o), getAtomCoords: () => [[0,0,0]] };
        if (opts && opts.onReady) opts.onReady(h);
        return h;
    } };
    global.molbuilder.selection = { mountPanel: async () => ({ panel: {}, dispose: () => {} }) };
    const workspace = {
        selection: { getState: () => ({ indices: [], kgrid: { enabled: false, dims: [1,1,1] }, isolate: false,
                                atoms: [{ element: 'O', x: 0, y: 0, z: 0 }] }),
                     subscribe: () => (() => {}) },
        getStructure: () => ({ text: '1\\nx\\nO 0 0 0\\n', atoms: [{ element: 'O', x: 0, y: 0, z: 0 }] }),
        getUnitCellInfo: () => ({ value: null }),
        getKgrid: () => [1, 1, 1],
        subscribe: () => (() => {}),
    };
    const host = mkEl('div');   // EMPTY host -> full-component build path
    const mount = global.molbuilder.molview.mount;
"""


def test_empty_host_builds_card_embeds_viewer_and_owns_render():
    out = _run_node(_BUILD_HARNESS + """
        mount(host, workspace, { mode: 'modify' }).then((h) => {
            const card = host.children[0];
            console.log(JSON.stringify({
                builtCard:  !!card && card.className === 'molview-card',
                embedded:   embeddedInto !== null,
                baseDrawn:  setStructureCalls.length >= 1,
                baseHead:   setStructureCalls.length ? setStructureCalls[0].xyz.split('\\n')[0] : null,
                handleKeys: Object.keys(h).sort(),
            }));
        });
    """)
    assert out["builtCard"] is True     # molview built the fused card into the empty host
    assert out["embedded"] is True      # it embedded the viewer itself
    assert out["baseDrawn"] is True     # the render loop drew the structure (mountRender)
    assert out["baseHead"] == "1"      # the 1-atom unit cell, read from ws.getStructure()
    # still ONLY the read/notify handle surface -- no internals leaked by the build path
    assert out["handleKeys"] == ["dispose", "getSelection", "getStructure", "onChange"]
