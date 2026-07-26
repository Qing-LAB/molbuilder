"""The selection glow is a SHAPE that tracks the trajectory by re-placement (§8.1).

The engine hands the embed WHICH drawn atoms are selected (setSelectionHalo); the embed draws a
translucent addSphere per selected atom -- a SHAPE, not a model restyle. So:
  * selecting an atom adds ONE shape and does NOT rebuild the molecule's geometry (no setStyle on
    the model, no second model) -- the cheap-click property;
  * a frame swap RE-PLACES the shape at the shown frame's position (removeShape + addSphere), so
    the glow tracks moving atoms, still with no model rebuild;
  * clearing the selection removes the shape.

History: an earlier design rendered the highlight as a DUPLICATE movie model to ride the native
swap; it occluded the atom (opaque imposter) and its non-occluding fix (depthWrite=false) reset on
every frame. A dim-all/pop atom-style scheme rebuilt the whole molecule on every click. The shape
glow avoids both -- it never touches the molecule's geometry. See molview-render-streamline.md §8.1.

Guard-clean: it spies the 3Dmol calls the embed makes, never reads render state.
"""
from __future__ import annotations

import threading

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

_BOOT_TIMEOUT_MS = 5000


@pytest.fixture(scope="module")
def flask_server():
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# Spy the 3Dmol calls the embed makes (via the molbuilder.viewer.create factory, wrapped BEFORE
# embed()). No selectedAtoms / _viewer3dmol render reads -- just call bookkeeping.
_PROBE = r"""
() => {
    const vapi = window.molbuilder.viewer;
    const origCreate = vapi.create;
    const spy = { addModelsAsFrames: 0, addSphere: 0, setStyle: 0, removeShape: 0 };
    vapi.create = function () {
        const v = origCreate.apply(this, arguments);
        const wrap = (name, hit) => {
            if (typeof v[name] !== "function") return;
            const orig = v[name].bind(v);
            v[name] = function () { try { hit(); } catch (_) {} return orig.apply(v, arguments); };
        };
        wrap("addModelsAsFrames", () => spy.addModelsAsFrames++);
        wrap("addSphere",         () => spy.addSphere++);
        wrap("setStyle",          () => spy.setStyle++);
        wrap("removeShape",       () => spy.removeShape++);
        return v;
    };

    const host = document.createElement("div");
    host.style.width = "400px"; host.style.height = "300px";
    document.body.appendChild(host);
    const h = window.molbuilder.viewer.embed(host, {
        xyz: "3\n\nO 0 0 0\nH 5 0 0\nH 8 0 0\n",
        card: { bare: true, showInfoLine: false },
    });
    // Two-frame native movie: atom 1 moves 5 -> 6.
    h.setAnimation({ kind: "trajectory", frames: [[[0,0,0],[5,0,0],[8,0,0]], [[0,0,0],[6,0,0],[8,0,0]]] });
    const afterMovie = { addModelsAsFrames: spy.addModelsAsFrames };

    // Glow atom 1 -> ONE addSphere, NO setStyle on the model, NO second model.
    const beforeSel = Object.assign({}, spy);
    h.setSelectionHalo([1]);
    const afterSel = Object.assign({}, spy);

    // Frame swap: the glow must RE-PLACE (removeShape + addSphere), still no model rebuild.
    const beforeSwap = Object.assign({}, spy);
    h.setAnimationFrame(1);
    const afterSwap = Object.assign({}, spy);

    // Clear: the glow shape is removed.
    const beforeClear = Object.assign({}, spy);
    h.setSelectionHalo([]);
    const afterClear = Object.assign({}, spy);

    vapi.create = origCreate;
    h.dispose(); host.remove();
    return { afterMovie, beforeSel, afterSel, beforeSwap, afterSwap, beforeClear, afterClear };
}
"""


class TestSelectionGlowIsAShape:
    def test_glow_is_a_shape_that_tracks_frames_without_rebuilding_the_model(
            self, page, flask_server):
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#molview-host .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate(_PROBE)

        # Exactly ONE native movie -- the structure. No second (halo) model is ever built.
        assert out["afterMovie"]["addModelsAsFrames"] == 1

        # Selecting an atom adds exactly ONE glow shape and does NOT restyle the model
        # (no molecule geometry rebuild) and does NOT add a model.
        assert out["afterSel"]["addSphere"] - out["beforeSel"]["addSphere"] == 1, (
            "selecting one atom must add exactly one glow shape")
        assert out["afterSel"]["setStyle"] == out["beforeSel"]["setStyle"], (
            "selecting must NOT restyle the molecule (no geometry rebuild)")
        assert out["afterSel"]["addModelsAsFrames"] == out["beforeSel"]["addModelsAsFrames"], (
            "selecting must NOT add a second model")

        # A frame swap RE-PLACES the glow shape (remove + re-add) at the new position, and STILL
        # does not rebuild the model.
        assert out["afterSwap"]["removeShape"] - out["beforeSwap"]["removeShape"] >= 1, (
            "a frame swap must re-place the glow (remove the old shape)")
        assert out["afterSwap"]["addSphere"] - out["beforeSwap"]["addSphere"] >= 1, (
            "a frame swap must re-place the glow (add it at the new position)")
        assert out["afterSwap"]["setStyle"] == out["beforeSwap"]["setStyle"], (
            "a frame swap must NOT restyle the molecule")

        # Clearing the selection removes the glow shape.
        assert out["afterClear"]["removeShape"] - out["beforeClear"]["removeShape"] >= 1, (
            "clearing the selection must remove the glow shape")
