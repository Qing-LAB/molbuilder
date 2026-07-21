"""Node unit test for ``lib/molview/selection/mount-panel.js`` -- the reusable fused
selection-panel composition (Phase 5 S2).

Verifies the wiring (partial fetched + inserted, the CALLER's store forwarded to
both selectionPanel.mount and viewerAdapter.attach, adapter attached to the
given viewer handle) with stubbed panel/adapter/fetch -- the DOM/network
integration itself is owned by the Playwright E2E.
"""
from __future__ import annotations

from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/molview/selection/mount-panel.js"


def _run_node(snippet: str) -> object:
    # mount-panel.js is now a native ES module; the shared ESM harness imports
    # it (running its transitional ``window.molbuilder.*`` publish) and the
    # snippet drives it through the global exactly as before.
    return run_node([MODULE], snippet)


def test_mountpanel_forwards_store_and_attaches_handle():
    out = _run_node("""
        const calls = { mount: null, attach: null };
        const STORE  = { _tag: "ephemeral" };
        const HANDLE = { setOverlays: function () {}, _tag: "viewer" };
        // The module IIFE already created global.molbuilder{.selection.mountPanel};
        // ADD to it (don't overwrite -- that would drop mountPanel).
        global.molbuilder.molview.selection.panel = {
            mount: function (host, opts) { calls.mount = opts; return { _tag: "panel" }; },
        };
        global.molbuilder.molview.selection.viewerAdapter = {
            attach: function (h, opts) {
                calls.attach = { handle: h._tag, store: opts.store._tag, mode: opts.mode };
                return { _tag: "adapter" };
            },
        };
        global.fetch = function () {
            return Promise.resolve({ ok: true,
                text: function () { return Promise.resolve("<div id='selection-atom-list'></div>"); } });
        };
        const host = { innerHTML: "" };
        (async () => {
            const r = await global.molbuilder.molview.selection.mountPanel(
                host, { store: STORE, viewerHandle: HANDLE, mode: "readonly" });
            console.log(JSON.stringify({
                htmlInserted:  host.innerHTML.indexOf("selection-atom-list") >= 0,
                mountStoreTag: calls.mount && calls.mount.store && calls.mount.store._tag,
                mountMode:     calls.mount && calls.mount.mode,
                attach:        calls.attach,
                panel:         r.panel && r.panel._tag,
                adapter:       r.adapterHandle && r.adapterHandle._tag,
            }));
        })();
    """)
    assert out["htmlInserted"] is True
    # the CALLER's store reaches both mount and attach -- not the singleton
    assert out["mountStoreTag"] == "ephemeral"
    assert out["mountMode"] == "readonly"
    assert out["attach"] == {"handle": "viewer", "store": "ephemeral", "mode": "readonly"}
    assert out["panel"] == "panel"
    assert out["adapter"] == "adapter"


def test_mountpanel_renders_banner_on_fetch_failure():
    out = _run_node("""
        let created = null;
        global.document = { createElement: function () {
            return { setAttribute: function () {},
                     set textContent(v) { created = v; }, get textContent() { return created; } };
        } };
        global.molbuilder.molview.selection.panel = {
            mount: function () { throw new Error("must not mount on fetch failure"); } };
        global.fetch = function () { return Promise.resolve({ ok: false, status: 503 }); };
        const host = { innerHTML: "seed", appendChild: function () {} };
        (async () => {
            const r = await global.molbuilder.molview.selection.mountPanel(host, { viewerHandle: {} });
            console.log(JSON.stringify({
                panel: r.panel, adapter: r.adapterHandle, banner: created, cleared: host.innerHTML }));
        })();
    """)
    assert out["panel"] is None
    assert out["adapter"] is None
    assert "503" in out["banner"]
    assert out["cleared"] == ""   # host cleared before the banner is appended
