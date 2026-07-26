"""Overlay halos/markers must be FRAME-AWARE on a trajectory (regression).

The bug (reported 2026-07-25, visible in the Results tab where trajectories play):
selection/region/frozen HALOS + glyph MARKERS are drawn as 3Dmol SHAPES at the
atom's current position, so a frame swap must RE-PLACE them.  Two paths that should
have repainted them on a frame change both skipped it:

  * the engine re-hands the shown frame's overlay spec via ``setOverlays`` every
    frame, but that spec is UNCHANGED while the selection holds, so setOverlays'
    idempotence bail (``_equalNormalised -> return``) drops the redraw; and
  * ``_postFramePositionRedraw`` (the per-frame hook for the native movie) redrew
    labels + pick halos but OMITTED the overlay halos/markers.

So the halos were drawn once at frame 0 and never moved, drifting off the atoms as
the trajectory played.  (3Dmol's setFrame DOES advance the model atoms' x/y/z to the
shown frame, so the coordinate source was never the issue -- the redraw was.)
The fix repaints overlay halos/markers in ``_postFramePositionRedraw`` on every swap.

This pins it with REAL 3Dmol: it spies the ``center`` the embed hands to
``addSphere`` (an INPUT capture -- not a 3Dmol data read, so no paint race) while a
two-frame trajectory whose atom 0 moves (0,0,0) -> (5,0,0) is scrubbed.  Before the
fix the frame-1 halo is never repainted (stays at frame 0); after, it lands at x≈5.
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


# The embed handle drives a native 3Dmol movie; we spy addSphere's center arg via a
# createViewer wrapper installed BEFORE embed() so every halo redraw is captured.
_PROBE = r"""
() => {
    // The embed builds its 3Dmol viewer via molbuilder.viewer.create(canvas); wrap that
    // factory (BEFORE embed()) so every viewer's addSphere records the center it is handed.
    const vapi = window.molbuilder.viewer;
    const origCreate = vapi.create;
    vapi.create = function () {
        const v = origCreate.apply(this, arguments);
        const origSphere = v.addSphere.bind(v);
        v.addSphere = function (spec) {
            try { window.__halo.push(spec && spec.center); } catch (_) {}
            return origSphere(spec);
        };
        return v;
    };
    window.__halo = [];
    const host = document.createElement("div");
    host.style.width = "400px"; host.style.height = "300px";
    document.body.appendChild(host);
    const h = window.molbuilder.viewer.embed(host, {
        xyz: "2\n\nO 0 0 0\nH 5 0 0\n",
        card: { bare: true, showInfoLine: false },
    });
    // Two-frame movie: atom 0 moves 0->5 along x; atom 1 stays at 5.
    h.setAnimation({
        kind: "trajectory",
        frames: [[[0, 0, 0], [5, 0, 0]], [[5, 0, 0], [5, 0, 0]]],
    });
    // A selection halo on atom 0 -- setOverlays draws it at the shown frame (0).
    window.__halo = [];
    h.setOverlays({ atoms: [{ indices: [0], halo: { color: "yellow", radius: 0.7 } }] });
    const f0 = window.__halo.slice();

    // The frame swap must RE-PLACE the halo at frame 1 (atom 0 -> x=5).  The bug left it
    // frozen at frame 0: setOverlays idempotence-bails on the unchanged spec, and (pre-fix)
    // _postFramePositionRedraw omitted the overlay-halo redraw -> the halo never repainted.
    window.__halo = [];
    h.setAnimationFrame(1);
    const f1 = window.__halo.slice();

    h.dispose(); host.remove();
    vapi.create = origCreate;
    return { f0, f1 };
}
"""


def _max_x(centers):
    xs = [c["x"] for c in centers if c and "x" in c]
    return max(xs) if xs else None


class TestOverlayHaloTracksFrame:
    def test_selection_halo_follows_the_shown_frame(self, page, flask_server):
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#molview-host .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate(_PROBE)

        # setOverlays drew the halo at frame 0 -- proves the harness/spy works.
        assert out["f0"], "no halo drawn by setOverlays at frame 0 (harness broken)"
        x0 = _max_x(out["f0"])
        assert x0 is not None and abs(x0) < 0.5, (
            f"frame-0 halo center x should be ≈0; got {x0}")

        # The discriminator: a frame swap must REPAINT the halo at the new position.
        # The bug skipped that repaint entirely, so f1 is empty (halo frozen at frame 0).
        assert out["f1"], (
            "frame-1 halo was never repainted -- it stayed frozen at frame 0 "
            "(setOverlays idempotence-bail + missing _postFramePositionRedraw redraw)")
        x1 = _max_x(out["f1"])
        # Atom 0 moved to x=5, so the repainted halo must land at x≈5.
        assert x1 is not None and abs(x1 - 5.0) < 0.5, (
            f"frame-1 halo center x should track the atom to ≈5; got {x1}")
