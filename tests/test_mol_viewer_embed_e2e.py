"""End-to-end Playwright tests for the embedded MolViewer
(``window.molbuilder.viewer.embed``).

Why this file exists
====================

The 2026-06-02 stage-5 migration of the Build + Modify viewers to
the embed contract shipped a CSS bug that left both tabs' viewer
canvases visibly BLANK in the browser.  The pre-existing
``test_build_e2e.py`` + ``test_modify_e2e.py`` suites passed 227 /
227 because they asserted on program state (atom counts via
``handle.getAtomCount()``, info-line text, JS-error-free boot) --
NONE asserted on what the user actually sees: a non-zero-sized
viewer canvas with a visible WebGL element.

The bug was a CSS layout collapse: the embed's bare-mode wrapper
had no sizing rules, so the inner ``.mol-viewer-canvas``'s
inline ``height: 100%`` resolved against a 0-height parent.
3Dmol mounted on a 0x0 canvas.  A 30-second eyeball check in a
browser would have caught it; the test suite did not.

This file closes that loophole.  Every test here asserts on
DIMENSIONS or VISIBILITY of the rendered viewer canvas -- not on
program state.  The pattern is the canonical regression-test
recipe for visual-rendering bugs (per
docs/protocols/playwright-tests.md § 3 "Assertions").

Tests
=====

  * Build (/) tab -- mounts the embed inside ``#viewer``; the
    inner ``.mol-viewer-canvas`` must have non-zero width AND
    height after the page boot.
  * Modify (/modify) tab -- same.
  * 3Dmol's WebGL canvas must actually render: a ``<canvas>``
    element inside the ``.mol-viewer-canvas`` with non-zero
    dimensions.
"""
from __future__ import annotations

import threading

import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Live Flask server fixture                                            #
# --------------------------------------------------------------------- #


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


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


_BOOT_TIMEOUT_MS = 5000


def _canvas_dimensions(page, host_selector):
    """Return the bounding-rect width + height of the embed's
    canvas div inside ``host_selector``.  Returns ``(w, h)`` as
    floats.  0 means the canvas collapsed -- the exact failure
    mode this file exists to catch."""
    page.wait_for_selector(
        f"{host_selector} .mol-viewer-canvas",
        timeout=_BOOT_TIMEOUT_MS,
    )
    box = page.evaluate(
        """(sel) => {
            const el = document.querySelector(sel + ' .mol-viewer-canvas');
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return { w: r.width, h: r.height };
        }""",
        host_selector,
    )
    assert box is not None, (
        f"no .mol-viewer-canvas under {host_selector} -- the embed "
        f"didn't mount at all"
    )
    return box["w"], box["h"]


def _has_webgl_canvas(page, host_selector):
    """True iff there's a ``<canvas>`` element inside the embed's
    canvas div, and that canvas has non-zero dimensions.  3Dmol
    creates this canvas when it initialises its WebGL context."""
    return page.evaluate(
        """(sel) => {
            const host = document.querySelector(sel + ' .mol-viewer-canvas');
            if (!host) return false;
            const canvas = host.querySelector('canvas');
            if (!canvas) return false;
            const r = canvas.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
        }""",
        host_selector,
    )


# --------------------------------------------------------------------- #
#  Tests — Build (/) tab                                                #
# --------------------------------------------------------------------- #


class TestBuildViewerDimensions:
    """The 2026-06-02 stage-5 migration regressed this; pin it."""

    def test_build_viewer_canvas_has_nonzero_dimensions(
            self, page, flask_server):
        """``.mol-viewer-canvas`` inside ``#viewer`` must have
        non-zero width AND height after the page boots.  A 0-width
        or 0-height canvas is the visual symptom of the CSS
        wrapper collapse bug (#198 stage 5 hotfix)."""
        page.goto(f"{flask_server}/")
        page.wait_for_selector("#viewer", timeout=_BOOT_TIMEOUT_MS)
        w, h = _canvas_dimensions(page, "#viewer")
        assert w > 0, f"viewer canvas width is {w}; expected > 0"
        assert h > 0, f"viewer canvas height is {h}; expected > 0"

    def test_build_viewer_renders_3dmol_canvas(
            self, page, flask_server):
        """Beyond the embed's wrapper having dimensions, 3Dmol's
        OWN ``<canvas>`` element must mount inside the wrapper
        with non-zero dimensions.  This catches the case where
        the embed wrapper has size but 3Dmol's WebGL context
        failed to initialise (e.g. a regression that called
        ``viewer.create`` on a 0x0 host before this commit's
        double-rAF resize fix)."""
        page.goto(f"{flask_server}/")
        page.wait_for_selector("#viewer", timeout=_BOOT_TIMEOUT_MS)
        # Give 3Dmol a beat to mount its WebGL canvas + the
        # embed's double-rAF resize() + render() to fire.
        page.wait_for_timeout(500)
        assert _has_webgl_canvas(page, "#viewer"), (
            "3Dmol WebGL canvas missing or 0x0 inside .mol-viewer-canvas; "
            "viewer is visually blank"
        )


# --------------------------------------------------------------------- #
#  Tests — Modify (/modify) tab                                         #
# --------------------------------------------------------------------- #


class TestModifyViewerDimensions:

    def test_modify_viewer_canvas_has_nonzero_dimensions(
            self, page, flask_server):
        """Same property as the Build test but for /modify, which
        has its own viewer card with aspect-ratio + min-height
        CSS that the bare-mode wrapper must pass through."""
        page.goto(f"{flask_server}/modify")
        page.wait_for_selector("#viewer", timeout=_BOOT_TIMEOUT_MS)
        w, h = _canvas_dimensions(page, "#viewer")
        assert w > 0, f"modify viewer canvas width is {w}; expected > 0"
        assert h > 0, f"modify viewer canvas height is {h}; expected > 0"

    def test_modify_viewer_renders_3dmol_canvas(
            self, page, flask_server):
        page.goto(f"{flask_server}/modify")
        page.wait_for_selector("#viewer", timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(500)
        assert _has_webgl_canvas(page, "#viewer"), (
            "3Dmol WebGL canvas missing or 0x0 in /modify; "
            "viewer is visually blank"
        )

    def test_modify_viewer_respects_host_aspect_ratio(
            self, page, flask_server):
        """The host ``#viewer.viewer`` has aspect-ratio 1/1; the
        embed's bare-mode wrapper must pass the host's dimensions
        through unchanged.  Width should approximately equal
        height (small tolerance for sub-pixel rounding + the
        max-height: min(60vh, 560px) clamp)."""
        page.goto(f"{flask_server}/modify")
        page.wait_for_selector("#viewer", timeout=_BOOT_TIMEOUT_MS)
        w, h = _canvas_dimensions(page, "#viewer")
        # min-height: 320px → both should be at least 320.
        assert min(w, h) >= 320, (
            f"viewer collapsed below the host's min-height: ({w}, {h})"
        )
        # The host's max-width is 560 px and aspect-ratio is 1/1;
        # the canvas should be reasonably square (within the
        # min(60vh, 560px) height clamp).
        ratio = max(w, h) / max(1, min(w, h))
        assert ratio < 2.5, (
            f"viewer aspect ratio {ratio:.2f} is well off the 1:1 host"
        )
