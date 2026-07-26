"""Trajectory halos render as a SECOND movie model that rides the native swap (§8.1 step 1).

History: halos used to be free-standing addSphere shapes re-placed every frame, which drifted
off the atoms during playback (bug fixed 2026-07-25) and — even reconciled — capped a large
selection at ~2 fps.  §8.1 step 1 renders trajectory halos as a DUPLICATE movie model
(setStyle translucent spheres on the highlighted atoms, setClickable(false)) that 3Dmol's
setFrame swaps ALONGSIDE the main model, so the halos ride the native swap for free and a
click is one setStyle on the delta (head-to-head: 35 ms click / 88 fps playback vs ~2 fps for
shapes — molview-render-streamline.md §8.1).

This pins the MECHANISM (guard-clean — it spies the 3Dmol calls the embed makes, never reads
render state): applying a halo to a trajectory builds a second model, draws NO halo spheres,
marks that model non-clickable (so picks hit the real atom), and a frame swap neither rebuilds
the model nor adds a shape — it just rides the movie.
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
# embed()).  No selectedAtoms / _viewer3dmol render reads — just call bookkeeping.
_PROBE = r"""
() => {
    const vapi = window.molbuilder.viewer;
    const origCreate = vapi.create;
    const spy = { addModelsAsFrames: 0, addSphere: 0, setClickableFalse: 0 };
    window.__spy = spy;
    vapi.create = function () {
        const v = origCreate.apply(this, arguments);
        const wrap = (name, hit) => {
            if (typeof v[name] !== "function") return;
            const orig = v[name].bind(v);
            v[name] = function () { try { hit(arguments); } catch (_) {} return orig.apply(v, arguments); };
        };
        wrap("addModelsAsFrames", () => spy.addModelsAsFrames++);
        wrap("addSphere",         () => spy.addSphere++);
        wrap("setClickable", (a) => { if (a[1] === false) spy.setClickableFalse++; });
        return v;
    };

    const host = document.createElement("div");
    host.style.width = "400px"; host.style.height = "300px";
    document.body.appendChild(host);
    const h = window.molbuilder.viewer.embed(host, {
        xyz: "2\n\nO 0 0 0\nH 5 0 0\n",
        card: { bare: true, showInfoLine: false },
    });
    // Two-frame native movie: atom 0 moves 0 -> 5.
    h.setAnimation({ kind: "trajectory", frames: [[[0,0,0],[5,0,0]], [[5,0,0],[5,0,0]]] });
    // A selection halo on atom 0 -> should build + style the SECOND (halo) model.
    h.setOverlays({ atoms: [{ indices: [0], halo: { color: "yellow", radius: 0.7 } }] });
    const afterSelect = { addModelsAsFrames: spy.addModelsAsFrames,
                          addSphere: spy.addSphere,
                          setClickableFalse: spy.setClickableFalse };

    // Frame swap: the halo model must ride the movie — no rebuild, no shape re-placement.
    const beforeSwap = { addModelsAsFrames: spy.addModelsAsFrames, addSphere: spy.addSphere };
    h.setAnimationFrame(1);
    const afterSwap = { addModelsAsFrames: spy.addModelsAsFrames, addSphere: spy.addSphere };

    vapi.create = origCreate;
    h.dispose(); host.remove();
    return { afterSelect, beforeSwap, afterSwap };
}
"""


class TestTrajectoryHaloSecondModel:
    def test_halo_is_a_second_model_that_rides_the_movie(self, page, flask_server):
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#molview-host .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate(_PROBE)

        sel = out["afterSelect"]
        # A halo on a trajectory builds a SECOND native movie (main movie + halo model = 2 calls).
        assert sel["addModelsAsFrames"] >= 2, (
            f"expected a second (halo) model built; addModelsAsFrames={sel['addModelsAsFrames']}")
        # Halos are NOT free-standing shapes any more — they live on the model's style.
        assert sel["addSphere"] == 0, (
            f"trajectory halos must not be addSphere shapes; got {sel['addSphere']}")
        # The halo model is non-clickable so a pick lands on the real atom, not the surround.
        assert sel["setClickableFalse"] >= 1, "halo model must be setClickable(false)"

        # The decisive property: a frame swap RIDES the movie — no rebuild, no shape re-placement.
        assert out["afterSwap"]["addModelsAsFrames"] == out["beforeSwap"]["addModelsAsFrames"], (
            "a frame swap must NOT rebuild the halo model (it rides the native swap)")
        assert out["afterSwap"]["addSphere"] == 0, (
            "a frame swap must NOT add halo shapes (the halo model rides the movie)")
