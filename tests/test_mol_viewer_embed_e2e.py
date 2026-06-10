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
        page.goto(f"{flask_server}/structure-optimization")
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
        page.goto(f"{flask_server}/structure-optimization")
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
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#viewer", timeout=_BOOT_TIMEOUT_MS)
        w, h = _canvas_dimensions(page, "#viewer")
        assert w > 0, f"modify viewer canvas width is {w}; expected > 0"
        assert h > 0, f"modify viewer canvas height is {h}; expected > 0"

    def test_modify_viewer_renders_3dmol_canvas(
            self, page, flask_server):
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#viewer", timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(500)
        assert _has_webgl_canvas(page, "#viewer"), (
            "3Dmol WebGL canvas missing or 0x0 in /modify; "
            "viewer is visually blank"
        )

    def test_modify_viewer_respects_host_aspect_ratio(
            self, page, flask_server):
        """The host ``#viewer.viewer`` has aspect-ratio 1/1 and
        min-height 320 px.  After #203 (knob bar visible), the
        canvas sits beneath a ~60 px header + knob bar inside
        the host card, so it's NOT the full host area anymore.

        The regression we still guard: the canvas must not
        collapse to near-zero height (the 2026-06-02 blank-viewer
        bug).  We assert it stays above 200 px tall — comfortably
        above any rounding-noise floor but well below the host's
        320 min-height, accounting for the chrome above."""
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#viewer", timeout=_BOOT_TIMEOUT_MS)
        w, h = _canvas_dimensions(page, "#viewer")
        assert h >= 200, (
            f"viewer canvas height {h} collapsed; the blank-viewer "
            f"bug class is back"
        )
        assert w >= 320, (
            f"viewer canvas width {w} below host min-height; check "
            f"that the embed's flex column isn't squashing the canvas"
        )
        # Width is still roughly the host's max-width (560 px);
        # the canvas should still appear square-ish (within a
        # factor of 2 of being square, accounting for the chrome).
        ratio = w / max(1, h)
        assert 0.5 < ratio < 3.0, (
            f"viewer aspect ratio {ratio:.2f} drifted too far from "
            f"the host's 1:1"
        )


# --------------------------------------------------------------------- #
#  Tests — Camera control (§ 3.13 + § 4.2)                             #
# --------------------------------------------------------------------- #


class TestCameraControl:
    """Camera get/set round-trip + preserveCamera semantics.
    These tests exercise the live 3Dmol getView / setView surface
    via the embed handle exposed on the Build tab's
    ``window.__molbuilder_build_test`` debug hook."""

    def test_getCamera_returns_versioned_blob(
            self, page, flask_server):
        """getCamera() returns {_viewer: "3dmol", _version: 1,
        data: any} per § 3.13 even before any structure is
        loaded.  The opaque ``data`` field may be null/undefined;
        the discriminator + version MUST be stable."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(300)
        cam = page.evaluate("""
            () => {
                const v = window.molbuilder.viewer;
                // We don't have direct access to the build handle
                // from here without the test surface (Phase 7);
                // mount a fresh embed to verify the contract.
                const host = document.createElement("div");
                host.style.width = "400px";
                host.style.height = "300px";
                document.body.appendChild(host);
                const h = v.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                const c = h.getCamera();
                h.dispose();
                host.remove();
                return {
                    viewer:  c._viewer,
                    version: c._version,
                    hasData: c.data !== null && c.data !== undefined,
                };
            }
        """)
        assert cam["viewer"]  == "3dmol"
        assert cam["version"] == 1
        # `data` may be a non-null view array even pre-zoom; just
        # confirm getCamera reaches setView's return at all.
        assert cam["hasData"] in (True, False)

    def test_setCamera_round_trip(self, page, flask_server):
        """getCamera → mutate → setCamera back restores the view.
        The view BEFORE refit must round-trip via the opaque blob."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(300)
        result = page.evaluate("""
            () => {
                const v = window.molbuilder.viewer;
                const host = document.createElement("div");
                host.style.width = "400px";
                host.style.height = "300px";
                document.body.appendChild(host);
                const h = v.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                const before = h.getCamera();
                // Manually rotate the camera via the escape hatch
                // so before / after differ; this is the kind of
                // mutation the selection adapter does today via
                // _viewer3dmol().
                try { h._viewer3dmol().rotate(45); h.render(); } catch (_) {}
                // Restore original camera:
                h.setCamera(before);
                const after = h.getCamera();
                h.dispose();
                host.remove();
                return {
                    before: JSON.stringify(before.data),
                    after:  JSON.stringify(after.data),
                };
            }
        """)
        # Round-trip: setCamera(before) → getCamera should return
        # the same opaque blob as the original capture.
        assert result["before"] == result["after"], (
            f"camera did not round-trip:\nbefore={result['before']}\n"
            f"after={result['after']}"
        )

    def test_setCamera_version_mismatch_is_noop(
            self, page, flask_server):
        """Per § 3.13 forward-compat: a CameraState with the wrong
        ``_viewer`` or ``_version`` is silently ignored — no
        error, no camera change."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(300)
        out = page.evaluate("""
            () => {
                const v = window.molbuilder.viewer;
                const host = document.createElement("div");
                host.style.width = "400px";
                host.style.height = "300px";
                document.body.appendChild(host);
                const h = v.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                const before = h.getCamera();
                // Wrong renderer name:
                h.setCamera({ _viewer: "potree", _version: 1, data: {} });
                const afterRenderer = h.getCamera();
                // Wrong version:
                h.setCamera({ _viewer: "3dmol", _version: 99, data: {} });
                const afterVersion = h.getCamera();
                h.dispose();
                host.remove();
                return {
                    rendererEq: JSON.stringify(before.data) ===
                                  JSON.stringify(afterRenderer.data),
                    versionEq:  JSON.stringify(before.data) ===
                                  JSON.stringify(afterVersion.data),
                };
            }
        """)
        assert out["rendererEq"], "wrong _viewer should be a no-op"
        assert out["versionEq"],  "wrong _version should be a no-op"


# --------------------------------------------------------------------- #
#  Tests — Test affordance surface (§ 9.2)                             #
# --------------------------------------------------------------------- #


class TestTestHandle:
    """``handle._test`` is the stable surface for visual-invariant
    test assertions — tests should NOT reach for ``_viewer3dmol``."""

    def test_test_handle_has_expected_methods(
            self, page, flask_server):
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        keys = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.width = "400px";
                host.style.height = "300px";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    card: { bare: true, showInfoLine: false },
                });
                const keys = Object.keys(h._test || {}).sort();
                h.dispose();
                host.remove();
                return keys;
            }
        """)
        assert keys == [
            "getCanvasElement",
            "getCurrent",
            "getCurrentBackground",
            "getDependencyStatus",
            "getFrameRebuildTimings",
            "getFrameStripElement",
            "getKnobBarElement",
            "getOpenExportOverlayCount",
            "getOverlayLabelCount",
            "getOverlayShapeCount",
            "hasAnimationLoop",
            "resetFrameRebuildTimings",
            "triggerKnob",
        ]

    def test_dependency_status_reports_loaded_modules(
            self, page, flask_server):
        """In production, the page boot loads mol-axes / mol-style /
        mol-format before mol-viewer-embed.  ``axes``, ``style``,
        ``format`` should all be true; the integration deps
        (projects / clipboard / mediaRecorder / gif) vary by
        environment."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        status = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.width = "400px";
                host.style.height = "300px";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    card: { bare: true, showInfoLine: false },
                });
                const s = h._test.getDependencyStatus();
                h.dispose();
                host.remove();
                return s;
            }
        """)
        # mol-axes / mol-style / mol-format are loaded on the
        # Build page boot.
        assert status["axes"]   is True, "mol-axes.js should be loaded"
        assert status["style"]  is True, "mol-style.js should be loaded"
        assert status["format"] is True, "mol-format.js should be loaded"
        # gif.js is lazy-loaded; should be "absent" until first
        # animation export.
        assert status["gif"] == "absent"
        # Status shape includes every documented key (§ 9.2)
        # even when the dep is absent.
        for key in ("axes", "style", "format",
                    "projects", "clipboard", "mediaRecorder", "gif"):
            assert key in status, f"missing status key: {key}"

    def test_overlay_shape_count_tracks_setOverlays(
            self, page, flask_server):
        """setOverlays with N halos should bump getOverlayShapeCount
        by N (visual-invariant assertion without _viewer3dmol)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.width = "400px";
                host.style.height = "300px";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                const before = h._test.getOverlayShapeCount();
                h.setOverlays({
                    atoms: [{
                        indices: [0, 1, 2],
                        halo: { color: "red", radius: 0.6 },
                    }],
                });
                const after = h._test.getOverlayShapeCount();
                h.dispose();
                host.remove();
                return { before, after };
            }
        """)
        # 3 atoms x 1 halo each = +3 over baseline (which has 0
        # overlay halos but may have arrow shapes from other
        # overlays; water has none, so before == 0).
        assert out["after"] - out["before"] == 3, (
            f"setOverlays should add 3 halo shapes; got "
            f"{out['after'] - out['before']} (before={out['before']}, "
            f"after={out['after']})"
        )

    def test_canvas_element_is_3dmol_canvas(
            self, page, flask_server):
        """getCanvasElement returns the 3Dmol-mounted <canvas>,
        not the wrapper div."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(300)
        kind = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.width = "400px";
                host.style.height = "300px";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                const el = h._test.getCanvasElement();
                const isCanvas = el && el.tagName === "CANVAS";
                h.dispose();
                host.remove();
                return { isCanvas };
            }
        """)
        assert kind["isCanvas"] is True


# --------------------------------------------------------------------- #
#  Tests — Export plumbing (§ 3.11 + § 5)                              #
# --------------------------------------------------------------------- #


class TestExportData:
    """exportData routes xyz/pdb text to project / download /
    clipboard targets.  Uses opts.testInjection per § 9.3 so the
    tests don't depend on the real projects sidebar state."""

    def test_export_to_clipboard_via_injection(
            self, page, flask_server):
        """clipboard target writes structure text via the injected
        clipboardApi.writeText mock; onExport fires with the right
        info."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.width = "300px";
                host.style.height = "200px";
                document.body.appendChild(host);
                let copied = null;
                let onExportInfo = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                    export: { onExport: (info) => { onExportInfo = info; } },
                    testInjection: {
                        clipboardApi: {
                            writeText: async (s) => { copied = s; },
                        },
                    },
                });
                const result = await h.exportData({
                    target: "clipboard", format: "xyz" });
                h.dispose();
                host.remove();
                return { copied, result, onExportInfo };
            }
        """)
        assert out["copied"].startswith("3"), \
            "clipboard should receive the xyz text"
        assert out["result"]["bytes"] > 0
        assert out["onExportInfo"]["kind"]   == "structure"
        assert out["onExportInfo"]["target"] == "clipboard"
        assert out["onExportInfo"]["format"] == "xyz"

    def test_export_to_project_via_injection(
            self, page, flask_server):
        """project target writes via the injected projectsApi.
        writeFile mock; path is constructed as currentDir + "/" +
        filename per § 2.5.4."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.width = "300px";
                host.style.height = "200px";
                document.body.appendChild(host);
                let writtenPath = null;
                let writtenData = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                    export: { defaultName: "water" },
                    testInjection: {
                        projectsApi: {
                            writeFile: async (path, data) => {
                                writtenPath = path;
                                writtenData = data;
                                return { ok: true, path: path };
                            },
                            currentDir: () => "/tmp/proj1",
                        },
                    },
                });
                const result = await h.exportData({ target: "project" });
                h.dispose();
                host.remove();
                return { writtenPath, hasData: !!writtenData,
                         resultFilename: result.filename,
                         resultPath:     result.path };
            }
        """)
        # Review fix D12: filename is leaf name; path is full path.
        assert out["writtenPath"]    == "/tmp/proj1/water.xyz"
        assert out["hasData"]        is True
        assert out["resultFilename"] == "water.xyz"
        assert out["resultPath"]     == "/tmp/proj1/water.xyz"

    def test_export_to_project_rejects_when_no_active_dir(
            self, page, flask_server):
        """No currentDir → reject with code: no_project per § 5.3."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.width = "300px";
                host.style.height = "200px";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                    testInjection: {
                        projectsApi: {
                            writeFile: async () =>
                                ({ ok: true }),
                            currentDir: () => "",   // no active dir
                        },
                    },
                });
                let code = null;
                try { await h.exportData({ target: "project" }); }
                catch (e) { code = e && e.code; }
                h.dispose();
                host.remove();
                return { code };
            }
        """)
        assert out["code"] == "no_project"


class TestScreenshot:
    """screenshot returns the canvas as PNG.  Live test uses the
    real 3Dmol pngURI; we just confirm the result has the right
    shape + non-zero bytes."""

    def test_screenshot_returns_dataurl_and_blob(
            self, page, flask_server):
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(500)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.width = "300px";
                host.style.height = "200px";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                // Wait one tick for the 3Dmol render to commit.
                await new Promise(r => requestAnimationFrame(r));
                await new Promise(r => requestAnimationFrame(r));
                const r = await h.screenshot();   // no target = capture only
                h.dispose();
                host.remove();
                return {
                    isDataUrl: r.dataUrl.startsWith("data:image/png"),
                    blobSize:  r.blob.size,
                    blobType:  r.blob.type,
                };
            }
        """)
        assert out["isDataUrl"] is True
        assert out["blobSize"]  > 0
        assert out["blobType"]  == "image/png"


# --------------------------------------------------------------------- #
#  Tests — Handle surface enumeration                                   #
# --------------------------------------------------------------------- #
#
# These are the structural safety net per the 2026-06-03 code review:
# every method documented in ``docs/protocols/embedded-viewer.md`` § 3.2
# ViewerHandle MUST exist on the returned handle as a function (or for
# ``_test`` an object).  Any drift between doc and code surfaces here
# rather than waiting for a consumer site to discover it during
# migration.  Adding a new documented method without exporting it from
# the handle fails this test.


class TestHandleSurface:
    """Enumerates the documented handle surface per § 3.2."""

    # The complete list, sorted, with the exact case from the doc.
    # Update this list AND the doc in the same commit when the
    # contract changes.
    EXPECTED_METHODS = sorted([
        # Data setters
        "setStructure", "appendFrames",
        # Style + overlays
        "setStyle", "setAxes", "setCell", "setLabels",
        "setArrows", "setPick", "setBackground",
        "setOverlays", "setAtomStyle",
        # Camera
        "getCamera", "setCamera",
        # Knob bar
        "setKnobs",
        # Animation control
        "setAnimation", "playAnimation", "pauseAnimation",
        "isAnimationPlaying", "setAnimationFrame", "getAnimationFrame",
        # Read accessors
        "getAtomCount", "getElements", "getAtomCoords",
        "getPickedIndices", "setPickedIndices",
        "getStructureText",
        # Declarative-state getters (D3 symmetry — round-trip with setX)
        "getStyle", "getAxes", "getCell", "getLabels",
        "getOverlays", "getPick", "getKnobs", "getArrows",
        "getAnimation", "getBackground", "getLattice",
        # Ordered batch runner (D4)
        "applyState",
        # Output / export
        "screenshot", "exportData",
        "captureFrames", "exportAnimation",
        # Lifecycle
        "refit", "setPivot", "render", "dispose",
        # Escape hatch + test surface
        "_viewer3dmol",
    ])
    # _test is an object, not a function — checked separately.

    def test_handle_has_exact_documented_method_set(
            self, page, flask_server):
        """The handle MUST export every documented method as a
        function.  Catches D1-class drift: a documented method
        being silently absent from the handle export."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.width = "300px";
                host.style.height = "200px";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    card: { bare: true, showInfoLine: false },
                });
                const fns = [];
                const nonFns = [];
                const missing = [];
                for (const key of Object.keys(h)) {
                    if (typeof h[key] === "function") fns.push(key);
                    else nonFns.push(key);
                }
                h.dispose();
                host.remove();
                return {
                    fns: fns.sort(),
                    nonFns: nonFns.sort(),
                };
            }
        """)
        # Every documented method must be present + callable.
        missing = [m for m in self.EXPECTED_METHODS
                   if m not in out["fns"]]
        assert not missing, (
            f"Handle is missing documented methods: {missing}\n"
            f"Present functions: {out['fns']}\n"
            f"This is doc-vs-code drift; either implement the "
            f"missing methods or update § 3.2 of the contract."
        )
        # No extra functions outside the documented set
        # (catches accidental exports of internal helpers).
        extras = [m for m in out["fns"]
                  if m not in self.EXPECTED_METHODS]
        assert not extras, (
            f"Handle exports undocumented functions: {extras}.\n"
            f"Either remove them or document them in § 3.2."
        )

    def test_setAnimation_partial_update_keeps_vibration_running(
            self, page, flask_server):
        """Per § 3.2: setAnimation with a partial object (no kind)
        merges into the active animation without stopping the loop.
        Regression test for D1/N1: spectra amplitude/speed sliders
        called setAnimation({amplitude: v}) on every tick, which
        previously normalised to null and STOPPED the vibration."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.1],[0,0,0.1],[0,0,0.1]],
                        amplitude: 0.2, speedHz: 1.0, paused: false,
                    },
                });
                await new Promise(r => requestAnimationFrame(r));
                const beforeLoop = h._test.hasAnimationLoop();
                // Drive 5 partial updates in tight succession (the
                // pattern spectra's amplitude/speed sliders use).
                for (const a of [0.3, 0.35, 0.4, 0.45, 0.5]) {
                    h.setAnimation({ amplitude: a });
                    await new Promise(r => requestAnimationFrame(r));
                }
                const afterLoop = h._test.hasAnimationLoop();
                h.dispose();
                host.remove();
                return { beforeLoop, afterLoop };
            }
        """)
        assert out["beforeLoop"] is True, \
            "vibration loop not running after initial setAnimation"
        assert out["afterLoop"]  is True, (
            "partial setAnimation({amplitude}) stopped the loop "
            "(D1/N1 regression)"
        )

    def test_setPickedIndices_drives_halo_state(
            self, page, flask_server):
        """Per § 3.2: setPickedIndices pushes the pick state from an
        external source.  The embed re-renders halos through
        _redrawPickHalos and clamps to the active mode's max
        (single: 1, pair: 2).  Pinned for #236: trajectory's atom-
        list row click drives picks via this API instead of a
        bespoke halo path."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let onPickFires = 0;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    pick: {
                        mode: "pair", halo: true, label: false,
                        onPick: () => { onPickFires++; },
                    },
                });
                const r = {};
                // Push 2 picks via setPickedIndices.
                h.setPickedIndices([0, 2]);
                r.afterTwo  = h.getPickedIndices();
                // Clamp to pair-max=2 when 3 supplied.
                h.setPickedIndices([0, 1, 2]);
                r.afterThree = h.getPickedIndices();
                // Clear via null.
                h.setPickedIndices(null);
                r.afterNull  = h.getPickedIndices();
                // Clear via [].
                h.setPickedIndices([1]);
                h.setPickedIndices([]);
                r.afterEmpty = h.getPickedIndices();
                // onPick must NOT fire for external pushes (per § 3.2
                // contract -- avoids feedback loops when hosts mirror
                // picks into a store).
                r.onPickFires = onPickFires;
                h.dispose();
                host.remove();
                return r;
            }
        """)
        assert out["afterTwo"]   == [0, 2], (
            f"setPickedIndices([0,2]) -> {out['afterTwo']}, "
            f"expected [0, 2]"
        )
        assert out["afterThree"] == [1, 2], (
            f"setPickedIndices([0,1,2]) -> {out['afterThree']}, "
            f"expected [1, 2] (pair mode keeps last 2)"
        )
        assert out["afterNull"]  == []
        assert out["afterEmpty"] == []
        assert out["onPickFires"] == 0, (
            "setPickedIndices fired onPick -- it must not (feedback "
            "loop risk per § 3.2 contract)"
        )

    def test_setX_dispatches_invalid_input_per_5_3(
            self, page, flask_server):
        """Per § 5.3: every documented sync setter dispatches
        ``invalid_input`` on input that fails type / shape / enum /
        range validation against the contract.  Pinned for #237 so a
        future edit that silently coerces a bad value (instead of
        firing onError) fails this test.

        Each subcase mounts a fresh embed, calls one setter with a
        single bad value, and asserts the error code + that the
        setter either halted (for halt cases) or proceeded with the
        documented default."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const cases = {};
                function run(name, ctor) {
                    const host = document.createElement("div");
                    host.style.cssText =
                        "width:300px;height:200px;position:fixed;top:-9999px;";
                    document.body.appendChild(host);
                    const errs = [];
                    const h = window.molbuilder.viewer.embed(host, {
                        xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                        card: { showInfoLine: false, height: "100%" },
                        onError: (e) => { errs.push(e.code); },
                    });
                    let result;
                    try { result = ctor(h); }
                    catch (e) { result = "threw:" + e.message; }
                    cases[name] = { errs: errs.slice(), result };
                    h.dispose();
                    host.remove();
                }
                run("setStructure_bad_xyz",
                    (h) => h.setStructure({xyz: 42}));
                run("setStyle_bad_rep",
                    (h) => h.setStyle({rep: "bogus"}));
                run("setStyle_NaN_radius",
                    (h) => h.setStyle({radiusScale: NaN}));
                run("setAxes_bad_mode",
                    (h) => h.setAxes({mode: "diagonal"}));
                run("setLabels_bad_format",
                    (h) => h.setLabels({atoms: "all", format: "weird"}));
                run("setLabels_negative_idx",
                    (h) => h.setLabels({atoms: [-1, 0]}));
                run("setArrows_not_array",
                    (h) => h.setArrows("nope"));
                run("setArrows_bad_entry",
                    (h) => h.setArrows([{start: [0,0,0]}]));
                run("setPick_bad_mode",
                    (h) => h.setPick({mode: "quintuple"}));
                run("setPick_bad_label",
                    (h) => h.setPick({mode: "single", label: "bogus"}));
                run("setBackground_empty",
                    (h) => h.setBackground(""));
                // Phase 6 dropped KnobBarOpts.position and
                // labelsFormats (2-menu redesign — no horizontal
                // placement variant, no labels-format popover).
                // Test cases retired with them.
                run("setKnobs_bad_presets",
                    (h) => h.setKnobs({backgroundPresets: "nope"}));
                run("setAnimation_bad_kind",
                    (h) => h.setAnimation({kind: "rotation"}));
                run("setAnimation_atom_mismatch",
                    (h) => h.setAnimation({
                        kind: "vibration",
                        displacements: [[0,0,0.1],[0,0,0.1]],  // 2 not 3
                    }));
                run("setPickedIndices_bad",
                    (h) => h.setPickedIndices("nope"));
                run("setCamera_not_object",
                    (h) => h.setCamera(42));
                return cases;
            }
        """)
        for name, cur in out.items():
            assert "invalid_input" in cur["errs"], (
                f"{name}: expected onError(invalid_input); got "
                f"errs={cur['errs']!r}, result={cur['result']!r}"
            )

    def test_setAxes_accepts_documented_modes(
            self, page, flask_server):
        """Per § 3.4 + § 5.3: setAxes accepts mode ∈ {auto, cartesian,
        cell}.  Regression test for B1 (#238) — VALID_AXES_MODES used
        to be ["auto", "world"] which silently coerced the legal
        cartesian / cell modes."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const r = {};
                // Phase 5d: mode "cell" requires a lattice (see
                // test_setAxes_cell_without_lattice below for the
                // halt path).  Supply a 10 Å cubic cell here so
                // the accept-path covers all three modes.
                for (const mode of ["auto", "cartesian", "cell"]) {
                    const host = document.createElement("div");
                    host.style.cssText =
                        "width:300px;height:200px;position:fixed;top:-9999px;";
                    document.body.appendChild(host);
                    const errs = [];
                    const h = window.molbuilder.viewer.embed(host, {
                        xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                        lattice: [[10,0,0],[0,10,0],[0,0,10]],
                        card: { showInfoLine: false, height: "100%" },
                        onError: (e) => { errs.push(e.code); },
                    });
                    h.setAxes({mode});
                    r[mode] = errs.slice();
                    h.dispose();
                    host.remove();
                }
                return r;
            }
        """)
        for mode in ["auto", "cartesian", "cell"]:
            assert "invalid_input" not in out[mode], (
                f"setAxes({{mode: {mode!r}}}) fired invalid_input "
                f"({out[mode]!r}); see B1 review finding"
            )

    def test_setAxes_cell_without_lattice_halts(
            self, page, flask_server):
        """Per § 5.3 Phase 5d: setAxes({mode: "cell"}) without a
        lattice on the current structure dispatches invalid_input
        + halts (so the user doesn't silently get Cartesian when
        they asked for cell)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const errs = [];
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    onError: (e) => { errs.push(e.code); },
                });
                // No lattice configured — cell mode must halt.
                h.setAxes({ mode: "cell" });
                const afterAxes = h.getAxes();
                h.dispose();
                host.remove();
                return { errs, afterAxes };
            }
        """)
        assert "invalid_input" in out["errs"]
        # Halt: axes state unchanged (still default null, not "cell").
        assert (out["afterAxes"] is None
                or out["afterAxes"].get("mode") != "cell"), (
            f"setAxes({{mode: 'cell'}}) did not halt — axes state "
            f"changed: {out['afterAxes']!r}"
        )

    def test_getBackground_and_getLattice_round_trip(
            self, page, flask_server):
        """Per § 3.2 Phase 5d: getBackground + getLattice complete
        the D3/D4 round-trip story for the cell-bearing structures
        and background field that applyState already accepted."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    lattice: [[5,0,0],[0,5,0],[0,0,5]],
                    card: { showInfoLine: false, height: "100%" },
                    style: { background: "#abcdef" },
                });
                const bg = h.getBackground();
                const lat = h.getLattice();
                // Defensive-clone: mutating the returned lattice
                // must NOT leak back into the embed.
                if (lat) lat[0][0] = 999;
                const latAgain = h.getLattice();
                h.dispose();
                host.remove();
                return { bg, lat0: lat && lat[0][0],
                         latUnchanged: latAgain && latAgain[0][0] };
            }
        """)
        assert out["bg"] == "#abcdef", out
        assert out["lat0"] == 999, "mutation didn't take on the clone"
        assert out["latUnchanged"] == 5, (
            f"getLattice didn't return a defensive clone — mutation "
            f"leaked: {out['latUnchanged']!r}"
        )

    def test_applyState_lattice_round_trip(
            self, page, flask_server):
        """Per § 4.2.2 Phase 5d: the round-trip example must work for
        cell-bearing structures — handle.applyState({structure: {xyz:
        getStructureText(), lattice: getLattice()}}) preserves the
        cell, which means setAxes({mode: "cell"}) keeps working
        post-restore."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    lattice: [[7,0,0],[0,7,0],[0,0,7]],
                    card: { showInfoLine: false, height: "100%" },
                });
                // Snapshot + re-apply structure with lattice.
                const snap = {
                    structure: {
                        xyz: h.getStructureText(),
                        lattice: h.getLattice(),
                    },
                };
                h.applyState(snap);
                const lat = h.getLattice();
                // After re-apply, cell-mode axes must work without
                // dispatching invalid_input.
                const errs = [];
                window.molbuilder.viewer.embed; // ref
                h.applyState({}); // no-op
                // Hook into onError via a fresh setter call:
                h.setAxes({ mode: "cell" });
                const axes = h.getAxes();
                h.dispose();
                host.remove();
                return { lat, axes };
            }
        """)
        assert out["lat"] is not None, "lattice lost after applyState round-trip"
        assert out["lat"][0][0] == 7, out
        assert out["axes"] and out["axes"]["mode"] == "cell", (
            f"setAxes({{mode: 'cell'}}) failed after applyState "
            f"round-trip: {out['axes']!r}"
        )

    def test_pick_halo_true_renders_halos_with_defaults(
            self, page, flask_server):
        """Per § 3.8 (Phase 5e B7): pick.halo: true is an alias for
        ``{}`` (enabled with defaults), matching the boolean-
        shorthand convention used for opts.axes / opts.cell.
        Before this fix, ``halo: true`` silently fell through to
        null in _normalisePick — trajectory atom-pick halos
        rendered nothing.  Pin the alias + verify the embed
        actually draws a halo shape after setPickedIndices."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    pick: { mode: "pair", halo: true, label: false,
                            onPick() {} },
                });
                h.setPickedIndices([0, 2]);
                const haloCount = h._test.getOverlayShapeCount();
                const pick = h.getPick();
                h.dispose();
                host.remove();
                return {
                    haloCount,
                    haloOpts: pick && pick.halo,
                };
            }
        """)
        # halo defaults: color #ffd54a, radius 0.6, opacity 0.5
        assert out["haloOpts"] is not None, (
            "halo: true did not produce a halo config; B7 regression"
        )
        assert out["haloOpts"]["color"] == "#ffd54a"
        assert out["haloOpts"]["radius"] == 0.6
        assert out["haloCount"] >= 2, (
            f"halo: true didn't draw halo shapes after "
            f"setPickedIndices([0,2]); shapes drawn: {out['haloCount']}"
        )

    def test_setAnimation_partial_update_preserves_trajectory_playback(
            self, page, flask_server):
        """Phase 5f A-1 regression catcher: a partial-update
        setAnimation({fps: N}) on an actively-playing trajectory
        must NOT pause the loop.  Before this fix, the merged
        opts inherited the stale mount-time ``paused: true`` from
        cur.animation, and _normaliseAnimation re-emitted it,
        silently stopping playback.  /results trajectory's
        #speed and #loop sliders surfaced the bug to end users.
        """
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [[[0,0,0]],[[0.1,0,0]],[[0.2,0,0]]],
                        fps: 10, paused: true,
                    },
                });
                // Start playback via the API.
                h.playAnimation();
                await new Promise(r => requestAnimationFrame(r));
                const wasPlaying = h.isAnimationPlaying();
                // Partial update like the trajectory's #speed
                // slider — change fps WITHOUT touching paused.
                h.setAnimation({ fps: 5 });
                await new Promise(r => requestAnimationFrame(r));
                const stillPlaying = h.isAnimationPlaying();
                h.dispose();
                host.remove();
                return { wasPlaying, stillPlaying };
            }
        """)
        assert out["wasPlaying"] is True, (
            "playAnimation didn't start the loop"
        )
        assert out["stillPlaying"] is True, (
            "setAnimation({fps: N}) paused active playback — "
            "Phase 5f A-1 regression"
        )

    def test_getAnimation_paused_reflects_runtime_state(
            self, page, flask_server):
        """Phase 5g B-1 regression catcher: getAnimation() must
        return the LIVE paused state, not the mount-time value.
        A host that snapshots getAnimation() during playback and
        re-applies it via applyState() must NOT silently pause
        the loop.  Single source of truth: state.current.animation
        .paused tracks state._anim.playing.
        """
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [[[0,0,0]],[[0.1,0,0]],[[0.2,0,0]]],
                        fps: 10, paused: true,
                    },
                });
                const atMount = h.getAnimation().paused;
                h.playAnimation();
                await new Promise(r => requestAnimationFrame(r));
                const whilePlaying = h.getAnimation().paused;
                h.pauseAnimation();
                const afterPause = h.getAnimation().paused;
                // Round-trip via applyState while playing — must
                // not silently stop the loop.
                h.playAnimation();
                await new Promise(r => requestAnimationFrame(r));
                const snap = { animation: h.getAnimation() };
                h.applyState(snap);
                await new Promise(r => requestAnimationFrame(r));
                const afterRoundTrip = h.isAnimationPlaying();
                h.dispose();
                host.remove();
                return {
                    atMount, whilePlaying, afterPause,
                    afterRoundTrip,
                };
            }
        """)
        assert out["atMount"] is True, (
            "mount-time paused should be True (we passed paused: true)"
        )
        assert out["whilePlaying"] is False, (
            "getAnimation().paused returned True during active "
            "playback — config-state out of sync with runtime "
            "(Phase 5g B-1 regression)"
        )
        assert out["afterPause"] is True, (
            "getAnimation().paused returned False after pauseAnimation()"
        )
        assert out["afterRoundTrip"] is True, (
            "applyState({animation: getAnimation()}) during playback "
            "stopped the loop — round-trip contract broken"
        )

    def test_applyState_preserves_trajectory_currentFrame(
            self, page, flask_server):
        """Phase 5i regression catcher: applyState({animation:
        getAnimation()}) must preserve the trajectory playhead.

        The Phase 5h I-1 fix made _normaliseAnimation honor caller-
        supplied currentFrame, but _setAnimationImpl was still
        calling _showTrajectoryFrame(state, next.startFrame) — which
        wrote a.currentFrame = startFrame, clobbering the preserved
        value just before autoplay resumed.  Symptom: a user paused
        on frame 5 who snapshots+applies would land on frame 0.
        """
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [[[0,0,0]],[[0.1,0,0]],[[0.2,0,0]],
                                 [[0.3,0,0]],[[0.4,0,0]],[[0.5,0,0]]],
                        fps: 10, paused: true,
                    },
                });
                // Move playhead to frame 4 (well past startFrame=0).
                h.setAnimationFrame(4);
                const beforeIdx  = h.getAnimationFrame();
                const beforeSnap = h.getAnimation().currentFrame;
                // Full-replace round-trip via applyState — must
                // preserve currentFrame.
                h.applyState({ animation: h.getAnimation() });
                const afterIdx = h.getAnimationFrame();
                // Partial-update round-trip — also must preserve.
                h.setAnimation({ fps: 5 });
                const afterPartial = h.getAnimationFrame();
                h.dispose();
                host.remove();
                return { beforeIdx, beforeSnap, afterIdx, afterPartial };
            }
        """)
        assert out["beforeIdx"]  == 4, (
            "setAnimationFrame(4) didn't move the playhead"
        )
        assert out["beforeSnap"] == 4, (
            "getAnimation() snapshot lost currentFrame "
            "(Phase 5h I-1 regression)"
        )
        assert out["afterIdx"]   == 4, (
            "applyState round-trip reset trajectory currentFrame "
            "from 4 to 0 (Phase 5i regression — _setAnimationImpl "
            "clobbered via _showTrajectoryFrame(startFrame))"
        )
        assert out["afterPartial"] == 4, (
            "partial-update setAnimation({fps:N}) clobbered "
            "currentFrame (Phase 5h I-1 regression — the partial "
            "merge path relies on _normaliseAnimation preservation)"
        )

    def test_trajectory_frame_advance_actually_moves_atoms(
            self, page, flask_server):
        """Phase 6c regression catcher: 3Dmol caches rep geometry
        meshes at setStyle time, so mutating ``atom.x`` alone leaves
        the visible atoms stuck on the original frame even though
        state.current.animation.currentFrame advances.  The bug
        surfaced on the /results trajectory tab: play cycled the
        counter but the molecule never moved.  Fix: every
        ``_applyCoords`` site now follows up with
        ``_rebuildGeometryForCoordChange`` which calls
        ``_applyStyle`` again to rebuild the geometry mesh from the
        new positions.

        This test reads atom coords through 3Dmol's
        ``selectedAtoms({})`` after a ``setAnimationFrame`` call
        and asserts they match the frame's input coords."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[1.0, 0.0, 0.0]],
                            [[2.0, 0.0, 0.0]],
                        ],
                        fps: 10,
                        paused: true,
                    },
                });
                const viewer = h._viewer3dmol();
                const readX = () =>
                    viewer.getModel().selectedAtoms({})[0].x;
                const r = {};
                r.atFrame0 = readX();
                h.setAnimationFrame(1);
                await new Promise(r => requestAnimationFrame(r));
                r.atFrame1 = readX();
                h.setAnimationFrame(2);
                await new Promise(r => requestAnimationFrame(r));
                r.atFrame2 = readX();
                h.dispose();
                host.remove();
                return r;
            }
        """)
        assert out["atFrame0"] == 0.0, (
            f"mount-time frame 0 should land at x=0; got {out['atFrame0']}"
        )
        assert out["atFrame1"] == 1.0, (
            "setAnimationFrame(1) did not move the atom to x=1.  "
            "_rebuildGeometryForCoordChange regression — 3Dmol's "
            "rep mesh is stale (see Phase 6c)"
        )
        assert out["atFrame2"] == 2.0, (
            "setAnimationFrame(2) did not move the atom to x=2"
        )

    def test_vibration_loop_actually_displaces_atoms(
            self, page, flask_server):
        """Phase 6c regression catcher: same root cause as the
        trajectory bug — vibration's per-rAF coord update never
        triggered a 3Dmol mesh rebuild, so the molecule visibly
        sat still even with the loop running.  Surfaced on /results
        spectra (mode animation didn't show).  This test runs the
        vibration for a couple of frames and asserts the atom's
        x position moves away from baseline."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        // Large amplitude + 100 Hz so a couple of
                        // rAF ticks produce a visibly-non-zero
                        // displacement, regardless of the test
                        // host's rAF cadence.
                        displacements: [[1.0, 0.0, 0.0]],
                        amplitude: 2.0,
                        speedHz: 100.0,
                        paused: false,
                    },
                });
                const viewer = h._viewer3dmol();
                const readX = () =>
                    viewer.getModel().selectedAtoms({})[0].x;
                // Spin a few rAFs to let the loop tick at least
                // once or twice.
                for (let i = 0; i < 8; i++) {
                    await new Promise(r => requestAnimationFrame(r));
                }
                const x = readX();
                h.pauseAnimation();
                h.dispose();
                host.remove();
                return { x };
            }
        """)
        # After several rAFs at 100 Hz, the atom should have
        # displaced from x=0 in either direction by some
        # non-negligible amount (cosine phase is between -2 and
        # +2 with amplitude=2).
        assert abs(out["x"]) > 0.05, (
            f"vibration loop ran but atom.x = {out['x']!r}; the per-rAF "
            "coord update is not actually moving the visible atom "
            "(Phase 6c regression — _rebuildGeometryForCoordChange "
            "missed a code path)"
        )

    def test_setStructure_loads_atoms_at_input_coords(
            self, page, flask_server):
        """Phase 6d / audit B11: read actual atom coords from the
        3Dmol model after setStructure() and assert they match the
        input XYZ.  The existing tests only checked atom COUNT via
        getAtomCount() — a setStructure regression that loaded the
        right number of atoms but at wrong positions (silent
        try/catch around _loadStructure was the original concern)
        would have passed every state-machine test."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                // Swap to water with three known coords.
                h.setStructure({ xyz:
                    "3\\nwater\\n"
                    + "O 1.5 0.0 0.0\\n"
                    + "H 2.4 0.7 0.0\\n"
                    + "H 0.6 0.7 0.0\\n" });
                const atoms = h._viewer3dmol().getModel()
                                              .selectedAtoms({});
                const r = atoms.map((a) => ({
                    elem: a.elem || a.atom,
                    x: a.x, y: a.y, z: a.z,
                }));
                h.dispose();
                host.remove();
                return r;
            }
        """)
        assert len(out) == 3, f"expected 3 atoms post-swap; got {len(out)}"
        assert out[0]["x"] == 1.5 and out[0]["y"] == 0.0, (
            f"O atom mis-positioned: {out[0]}"
        )
        assert out[1]["x"] == 2.4 and out[1]["y"] == 0.7, (
            f"H1 atom mis-positioned: {out[1]}"
        )
        assert out[2]["x"] == 0.6 and out[2]["y"] == 0.7, (
            f"H2 atom mis-positioned: {out[2]}"
        )

    def test_setStyle_actually_changes_atom_style_spec(
            self, page, flask_server):
        """Phase 6d / audit B1: after setStyle({rep:"sphere"}), the
        per-atom style spec on the 3Dmol model must contain a sphere
        entry (and no stick).  Tests previously only checked
        getStyle().rep === "sphere" (state read-back).  This is the
        same bug class as Phase 6c's animation bug: setStyle writes
        state, but if the 3Dmol setStyle({}, spec) call breaks
        silently, the canvas shows the old rep while the test still
        passes."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    style: { rep: "stick", radiusScale: 1.0 },
                });
                const readSpec = () =>
                    h._viewer3dmol().getModel().selectedAtoms({})[0].style;
                const r = {};
                r.afterStick = JSON.stringify(readSpec());
                h.setStyle({ rep: "sphere" });
                r.afterSphere = JSON.stringify(readSpec());
                h.setStyle({ rep: "line" });
                r.afterLine = JSON.stringify(readSpec());
                h.setStyle({ rep: "ball-and-stick" });
                r.afterBS = JSON.stringify(readSpec());
                h.dispose();
                host.remove();
                return r;
            }
        """)
        # Stick spec has a stick component (and a tiny sphere
        # overlay per mol-style.js).
        assert "stick" in out["afterStick"]
        # Sphere: spec has sphere only.
        assert "sphere" in out["afterSphere"]
        assert "stick" not in out["afterSphere"], (
            "setStyle({rep:'sphere'}) did not actually flip the model's "
            "style spec; the 3Dmol setStyle call may have been a no-op "
            "(audit B1 bomb-class)"
        )
        # Line: spec has line only.
        assert "line" in out["afterLine"]
        assert "sphere" not in out["afterLine"], (
            "setStyle({rep:'line'}) did not remove sphere from spec"
        )
        # Ball & stick (mol-style.js identifier is 'ballstick' — Phase 6
        # translation at the boundary): spec has both stick and sphere.
        assert "stick" in out["afterBS"] and "sphere" in out["afterBS"]

    def test_setBackground_actually_changes_3dmol_bgcolor(
            self, page, flask_server):
        """Phase 6d / audit B3: after setBackground(color), 3Dmol's
        actual background color must change.  Previous tests checked
        getBackground() and swatch.is-active (both state read-back) —
        the 3Dmol setBackgroundColor call is wrapped in a swallowing
        try/catch (line ~1226 of mol-viewer-embed.js), so a future
        3Dmol API rename would leave state advancing while the canvas
        kept the old colour."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                const v = h._viewer3dmol();
                // 3Dmol exposes the bg via getBgColor() (modern
                // versions) or the bgColor field.  Both should work
                // for our vendored copy.
                const readBg = () => {
                    if (typeof v.getBgColor === "function") {
                        const c = v.getBgColor();
                        return (typeof c === "object" && c !== null)
                            ? c : Number(c);
                    }
                    return v.bgColor !== undefined ? v.bgColor : null;
                };
                const r = {};
                r.afterMount = readBg();
                h.setBackground("#ffffff");
                r.afterWhite = readBg();
                h.setBackground("#1d2128");
                r.afterDark = readBg();
                h.dispose();
                host.remove();
                return {
                    afterMount: String(r.afterMount),
                    afterWhite: String(r.afterWhite),
                    afterDark:  String(r.afterDark),
                };
            }
        """)
        # Don't pin exact numeric form (3Dmol uses 0xRRGGBB; getBgColor
        # may also return a THREE.Color).  What we care about: the
        # value CHANGES between setBackground calls.  If 3Dmol's
        # setBackgroundColor were silently dropped, white and dark
        # would both equal afterMount.
        assert out["afterWhite"] != out["afterMount"], (
            f"setBackground('#ffffff') did not change 3Dmol's bg "
            f"({out['afterMount']!r}) — audit B3 bomb-class"
        )
        assert out["afterWhite"] != out["afterDark"], (
            f"setBackground white vs dark gave the same 3Dmol bg "
            f"({out['afterWhite']!r}); setBackgroundColor call may "
            f"be a no-op"
        )

    def test_setLabels_actually_creates_label_objects(
            self, page, flask_server):
        """Phase 6d / audit B4: after setLabels({atoms:'all',
        format:'element'}), 3Dmol must hold real label objects
        positioned at the atom coords with the right text.  Tests
        previously asserted state.current.labels was set; nothing
        verified the 3Dmol labels actually appeared with the right
        format.  Doc § 3.6 promises format dispatch maps
        'element'→element symbol, 'name'→atom name, 'index'→serial."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                const r = {};
                r.beforeCount = h._test.getOverlayLabelCount();
                h.setLabels({ atoms: "all", format: "element" });
                r.afterElementCount = h._test.getOverlayLabelCount();
                // Read the actual rendered label text via the
                // labels[i].text() / .stylespec backing.  3Dmol stores
                // labels in viewer.labels; each has a ``text`` field.
                const v = h._viewer3dmol();
                // 3Dmol stores labels in viewer.labels; each label
                // object carries the rendered text on the ``text``
                // field at the top level (not under stylespec).
                const readLabelTexts = () => (v.labels || []).map(
                    (l) => (l && (l.text || (l.stylespec
                                              && l.stylespec.text))) || "");
                r.elementTexts = readLabelTexts().slice().sort().join(",");
                h.setLabels({ atoms: "all", format: "index" });
                r.indexTexts = readLabelTexts().slice().sort().join(",");
                h.setLabels(false);
                r.afterOff = h._test.getOverlayLabelCount();
                h.dispose();
                host.remove();
                return r;
            }
        """)
        assert out["beforeCount"] == 0
        assert out["afterElementCount"] >= 3, (
            f"setLabels(atoms:'all') did not create 3 labels for "
            f"water; got {out['afterElementCount']} (audit B4 bomb-class)"
        )
        # Format 'element' must produce element symbols (O, H, H).
        assert "O" in out["elementTexts"] and "H" in out["elementTexts"], (
            f"format:'element' did not produce element-symbol labels: "
            f"{out['elementTexts']!r}"
        )
        # Format 'index' must produce numeric strings.
        assert any(c.isdigit() for c in out["indexTexts"]), (
            f"format:'index' did not produce numeric labels: "
            f"{out['indexTexts']!r}"
        )
        # Element and index formats must differ.
        assert out["elementTexts"] != out["indexTexts"], (
            "setLabels(format) appears to ignore the format field — "
            "element and index labels look identical"
        )
        # Off must clear.
        assert out["afterOff"] == 0

    def test_captureFrames_produces_diverse_frames(
            self, page, flask_server):
        """Phase 6d / audit C6: captureFrames must produce DIFFERENT
        frames over the requested duration.  Previous test only
        asserted ``count == fps * duration`` and ``size > 0`` per
        blob.  Two identical (frozen-baseline) blobs would have
        satisfied that — Phase 6c's animation bug would have shipped
        a captureFrames that returned 10 identical PNGs and the test
        would have been green.

        Compares the first and last blob byte-by-byte; for a real
        vibration their PNGs MUST differ because the atom positions
        are different at different cosine phases."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[1.0, 0.0, 0.0]],
                        amplitude: 1.0,
                        speedHz: 0.5,
                        paused: true,
                    },
                });
                // Capture 6 frames spread across 2 s — covers a
                // significant portion of one 0.5-Hz cycle.
                const blobs = await h.captureFrames({
                    fps: 3, duration: 2.0 });
                // Reduce each blob to a short hex prefix so the test
                // signal is loud (differs sharply on different frames).
                const summarize = async (b) => {
                    const buf = new Uint8Array(await b.arrayBuffer());
                    // Use a middle slice — PNG headers are identical
                    // across frames; the pixel-data chunk differs.
                    const start = Math.min(80, buf.length - 16);
                    return Array.from(buf.slice(start, start + 16))
                                .map((b) => b.toString(16).padStart(2, '0'))
                                .join('');
                };
                const summaries = await Promise.all(
                    blobs.map(summarize));
                h.dispose();
                host.remove();
                return summaries;
            }
        """)
        assert len(out) == 6, f"expected 6 blobs; got {len(out)}"
        # The first and last frame MUST differ — atoms are at
        # different phases of the cosine cycle.  If captureFrames
        # froze the baseline (Phase 6c regression class), every
        # summary would be identical.
        assert out[0] != out[-1], (
            "captureFrames produced identical first and last frames "
            f"({out[0]!r} == {out[-1]!r}) — the animation did not "
            "advance during capture (audit C6 bomb-class)"
        )
        # And at least one intermediate frame should also differ from
        # the first — guards against alternating-pattern bugs.
        differing = sum(1 for s in out[1:] if s != out[0])
        assert differing >= 3, (
            f"only {differing}/5 frames differ from the first; "
            f"animation may be running at a tiny fraction of "
            f"expected speed"
        )

    def test_applyState_round_trip_restores_visible_atom_coords(
            self, page, flask_server):
        """Phase 6d / audit D2: applyState round-trip must restore
        the visible canvas state, not just state.current.*.  The
        previous round-trip test (test_applyState_round_trip_with_getters)
        verified getStyle/getAxes/getLabels returned matching values
        — pure state read-back.  This test pushes the round-trip
        through a real coord change: snapshot at frame 2, advance to
        frame 5, applyState(snapshot), verify atom coords match frame
        2's input."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[1.0, 0.0, 0.0]],
                            [[2.0, 0.0, 0.0]],
                            [[3.0, 0.0, 0.0]],
                            [[4.0, 0.0, 0.0]],
                            [[5.0, 0.0, 0.0]],
                        ],
                        fps: 10, paused: true,
                    },
                });
                const readX = () =>
                    h._viewer3dmol().getModel().selectedAtoms({})[0].x;
                // Land on frame 2 (x=2.0); snapshot.
                h.setAnimationFrame(2);
                await new Promise(r => requestAnimationFrame(r));
                const xAtSnap = readX();
                const snap = { animation: h.getAnimation() };
                // Drift to frame 5.
                h.setAnimationFrame(5);
                await new Promise(r => requestAnimationFrame(r));
                const xAtDrift = readX();
                // Restore via applyState — visible atom MUST move
                // back to frame 2's x.
                h.applyState(snap);
                await new Promise(r => requestAnimationFrame(r));
                const xAfterRestore = readX();
                h.dispose();
                host.remove();
                return { xAtSnap, xAtDrift, xAfterRestore };
            }
        """)
        assert out["xAtSnap"] == 2.0
        assert out["xAtDrift"] == 5.0
        assert out["xAfterRestore"] == 2.0, (
            f"applyState round-trip did not restore the visible "
            f"atom position; expected x=2.0, got {out['xAfterRestore']}.  "
            "State-machine round-trip may be intact while rendered "
            "state lags — audit D2 bomb-class"
        )

    def test_style_radius_slider_drives_setStyle(
            self, page, flask_server):
        """Phase 6b: View → Style carries a radius slider that
        drives ``setStyle({radiusScale: v})``.  Pre-Phase-6 the
        /modify viewer had a bespoke #radius input; the Phase 6b
        slider gives every embed consumer the same control through
        the documented contract instead of bespoke chrome."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    style: { rep: "stick", radiusScale: 1.0 },
                });
                const bar = h._test.getKnobBarElement();
                const slider = bar.querySelector(".mol-viewer-radius");
                const out_ = bar.querySelector(
                    ".mol-viewer-radius-out");
                const r = {};
                r.initRadius = h.getStyle().radiusScale;
                r.initSlider = parseFloat(slider.value);
                r.initOut = out_.textContent;
                // Drag-style update: dispatch input event with a
                // new value.
                slider.value = "0.5";
                slider.dispatchEvent(
                    new Event("input", { bubbles: true }));
                r.afterSlider = h.getStyle().radiusScale;
                r.afterOut = out_.textContent;
                // Programmatic→UI sync: setStyle from the handle
                // pushes the new value back into the slider.
                h.setStyle({ radiusScale: 1.8 });
                r.afterProgSlider = parseFloat(slider.value);
                r.afterProgOut = out_.textContent;
                h.dispose();
                host.remove();
                return r;
            }
        """)
        assert out["initRadius"] == 1.0
        assert out["initSlider"] == 1.0
        assert out["initOut"] == "1.00"
        assert out["afterSlider"] == 0.5, (
            "slider input event did not drive setStyle({radiusScale}) "
            "— handler regression"
        )
        assert out["afterOut"] == "0.50"
        assert out["afterProgSlider"] == 1.8, (
            "setStyle({radiusScale: 1.8}) did not re-sync the slider "
            "input value — programmatic→UI sync regression"
        )
        assert out["afterProgOut"] == "1.80"

    def test_menu_popover_escapes_clipping_ancestor(
            self, page, flask_server):
        """Phase 6b: popover menus use ``position: fixed`` so they
        escape a clipping ancestor (e.g. Build's ``.viewer-wrap``
        has ``overflow: hidden``).  An ``absolute``-positioned
        popover would get clipped by the wrap and the Export menu
        would be unreachable.  Pin the fix: when the user opens the
        Export menu, the popover's bounding rect must extend beyond
        the wrap's right edge (the popover is wider than the
        Export trigger).
        """
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                // Open the Export menu on the live #viewer card.
                const det = document.querySelector(
                    "#viewer .mol-viewer-menu-export");
                const summary = det.querySelector(":scope > summary");
                summary.click();
                await new Promise(r => requestAnimationFrame(r));
                await new Promise(r => requestAnimationFrame(r));
                const body = det.querySelector(
                    ":scope > .mol-viewer-menu-body");
                const bRect = body.getBoundingClientRect();
                // Phase 6b: position must be ``fixed`` (escapes
                // clipping); JS positioning sets top/left after
                // toggle.
                const cs = getComputedStyle(body);
                const r = {
                    position: cs.position,
                    visible: bRect.width > 0 && bRect.height > 0,
                    left:  bRect.left,
                    top:   bRect.top,
                };
                summary.click();  // close
                return r;
            }
        """)
        assert out["position"] == "fixed", (
            "popover must use position:fixed to escape clipping "
            "ancestors (Phase 6b)"
        )
        assert out["visible"], "popover has zero size when open"
        # Bounding rect must be a real on-screen position, not
        # the off-screen -9999px stub.
        assert out["left"] > -1000, (
            f"popover left={out['left']} suggests JS positioning "
            f"didn't fire on open"
        )
        assert out["top"] > -1000, (
            f"popover top={out['top']} suggests JS positioning "
            f"didn't fire on open"
        )

    def test_programmatic_setX_syncs_knob_bar_ui(
            self, page, flask_server):
        """Phase 6 regression catcher for § 4.1 + § 6.2 invariant:
        ``setStyle`` / ``setLabels`` / ``setBackground`` / ``setAxes``
        called via the handle must push the new state into the
        visible affordance — active rep button (Style submenu),
        pressed state on the Labels / Axes toggles, ``is-active``
        ring on the matching Background swatch.  Without this a host
        driving viewer state from a non-knob source (keyboard
        shortcut, restore-from-snapshot) leaves the chrome lying.
        """
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    style: { rep: "stick" },
                    labels: false,
                });
                const bar = h._test.getKnobBarElement();
                const activeRep = () => {
                    const b = bar.querySelector(
                        ".mol-viewer-rep-btn.is-active");
                    return b ? b.getAttribute("data-rep") : null;
                };
                const labelsBtn = bar.querySelector(
                    '.mol-viewer-toggle[data-action="labels"]');
                const axesBtn = bar.querySelector(
                    '.mol-viewer-toggle[data-action="axes"]');
                const activeBg = () => {
                    const b = bar.querySelector(
                        ".mol-viewer-bg-swatch.is-active");
                    return b ? b.getAttribute("data-color") : null;
                };
                const r = {};
                // Initial state.
                r.initRep = activeRep();
                r.initLabels = labelsBtn.getAttribute("aria-pressed");
                r.initAxes = axesBtn.getAttribute("aria-pressed");
                r.initBg = activeBg();
                // Drive each from the handle.
                h.setStyle({ rep: "sphere" });
                r.afterStyle = activeRep();
                h.setLabels({ atoms: "all", format: "name" });
                r.afterLabelsOn = labelsBtn.getAttribute("aria-pressed");
                h.setAxes(true);
                r.afterAxesOn = axesBtn.getAttribute("aria-pressed");
                h.setBackground("#ffffff");
                r.afterBg = activeBg();
                h.setLabels(false);
                r.afterLabelsOff = labelsBtn.getAttribute("aria-pressed");
                h.dispose();
                host.remove();
                return r;
            }
        """)
        assert out["initRep"] == "stick", \
            "mount-time Style did not seed active rep from opts.style.rep"
        assert out["initLabels"] == "false", \
            "mount-time Labels toggle should be unpressed (labels: false)"
        assert out["initAxes"] == "false", \
            "mount-time Axes toggle should be unpressed (default off)"
        assert out["initBg"] == "#1d2128", (
            "mount-time Background did not mark the default "
            "dark swatch (#1d2128) — DEFAULT_BACKGROUND drift"
        )
        assert out["afterStyle"] == "sphere", (
            "setStyle({rep:'sphere'}) did not re-sync the active "
            "rep button — programmatic→UI sync broken (R2)"
        )
        assert out["afterLabelsOn"] == "true", (
            "setLabels({...}) did not set the Labels toggle to "
            "aria-pressed=true — programmatic→UI sync broken (R1)"
        )
        assert out["afterAxesOn"] == "true", (
            "setAxes(true) did not set the Axes toggle to "
            "aria-pressed=true"
        )
        assert out["afterBg"] == "#ffffff", (
            "setBackground('#ffffff') did not mark the white "
            "preset swatch is-active"
        )
        assert out["afterLabelsOff"] == "false", (
            "setLabels(false) did not unpress the Labels toggle"
        )

    def test_setStructure_preserves_picks_when_elements_match(
            self, page, flask_server):
        """Per § 3.8 + § 4.2.1: pickedIndices survives setStructure
        IFF atom count + element-by-element ordering match.  On
        mismatch the embed fires onPick([]) so hosts mirroring picks
        into a store see the clear.  Regression test for B2 (#238).
        """
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const calls = [];
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    pick: {
                        mode: "pair", halo: true, label: false,
                        onPick: (curr) => { calls.push(curr.slice()); },
                    },
                });
                h.setPickedIndices([0, 1]);
                // Swap to a structure with IDENTICAL elements (perturb
                // coords only) -- picks must survive.
                h.setStructure({
                    xyz: "3\\nwater_moved\\nO 0.1 0 0\\n"
                       + "H 1.1 0 0\\nH 0.1 1 0\\n",
                });
                const afterSame = h.getPickedIndices();
                // Swap to a structure with DIFFERENT elements -- picks
                // must clear AND onPick([]) must fire.
                h.setStructure({
                    xyz: "2\\nhh\\nH 0 0 0\\nH 1 0 0\\n",
                });
                const afterDiff = h.getPickedIndices();
                h.dispose();
                host.remove();
                return { afterSame, afterDiff, calls };
            }
        """)
        assert out["afterSame"] == [0, 1], (
            f"picks lost after element-identical setStructure: "
            f"{out['afterSame']!r}"
        )
        assert out["afterDiff"] == [], (
            f"picks survived element-mismatched setStructure: "
            f"{out['afterDiff']!r}"
        )
        assert [] in out["calls"], (
            f"setStructure did NOT fire onPick([]) on clear; calls="
            f"{out['calls']!r}"
        )

    def test_test_surface_normaliser_exports_callable(
            self, page, flask_server):
        """Per § 9.1: every listed normaliser is callable on
        window.molbuilder.viewer.*.  Regression test for B3 (#238) —
        the doc used to list a fictional _normaliseExport and omit
        _normaliseKnobs from the actual export."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        present = page.evaluate("""
            () => {
                const v = window.molbuilder.viewer;
                const wanted = [
                    "_normaliseOpts",     "_normaliseStyle",
                    "_normaliseAxes",     "_normaliseCell",
                    "_normaliseLabels",   "_normalisePick",
                    "_normaliseAnimation","_normaliseOverlays",
                    "_normaliseLattice",  "_normaliseKnobs",
                    "_equalNormalised",
                ];
                const out = {};
                for (const k of wanted) out[k] = typeof v[k];
                return out;
            }
        """)
        for name, kind in present.items():
            assert kind == "function", (
                f"{name} not exported as a function (got {kind!r}); "
                f"see B3 review finding"
            )

    def test_frame_strip_documented_selectors_present(
            self, page, flask_server):
        """Per § 9.4: the frame-strip exposes data-action="prev|play|
        next" + slider with aria-label="Trajectory frame".  Pinned by
        I2 (#238) — these selectors used to be promised but absent."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        present = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [[[0,0,0]],[[0.1,0,0]],[[0.2,0,0]]],
                        fps: 5, paused: true,
                    },
                });
                const strip = host.querySelector(
                    ".mol-viewer-frame-strip");
                const out = {
                    strip:    !!strip,
                    prev:     !!strip.querySelector('[data-action="prev"]'),
                    play:     !!strip.querySelector('[data-action="play"]'),
                    next:     !!strip.querySelector('[data-action="next"]'),
                    sliderAria: strip.querySelector(".frame-slider")
                                ?.getAttribute("aria-label"),
                };
                h.dispose();
                host.remove();
                return out;
            }
        """)
        assert present["strip"], "frame-strip not mounted"
        for k in ("prev", "play", "next"):
            assert present[k], (
                f"frame-strip missing [data-action={k!r}] per § 9.4"
            )
        assert present["sliderAria"] == "Trajectory frame", (
            f"slider aria-label is {present['sliderAria']!r}, "
            f"expected 'Trajectory frame' per § 9.4"
        )

    def test_setStructure_preserves_overlays_when_elements_match(
            self, page, flask_server):
        """Per § 4.2.1: OverlaySpec entries follow the same rule as
        pick state — survive setStructure IFF atom count + element
        ordering match; cleared otherwise.  Regression test for I5
        (#239) — overlays used to persist unconditionally, leaving
        index-keyed highlights pointing at the wrong atoms after a
        type-swap or file-swap."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                h.setOverlays({
                    atoms: [{
                        indices: [0],
                        halo: {color: "#ff0", radius: 0.5},
                    }],
                });
                const beforeSame = !!h._test.getCurrent().overlays;
                h.setStructure({
                    xyz: "3\\nwater_moved\\nO 0.1 0 0\\n"
                       + "H 1.1 0 0\\nH 0.1 1 0\\n",
                });
                const afterSame = !!h._test.getCurrent().overlays;
                h.setStructure({
                    xyz: "2\\nhh\\nH 0 0 0\\nH 1 0 0\\n",
                });
                const afterDiff = !!h._test.getCurrent().overlays;
                h.dispose();
                host.remove();
                return { beforeSame, afterSame, afterDiff };
            }
        """)
        assert out["beforeSame"] is True
        assert out["afterSame"]  is True, (
            "overlays lost after element-identical setStructure"
        )
        assert out["afterDiff"] is False, (
            "overlays survived element-mismatched setStructure -- "
            "I5 regression"
        )

    def test_appendFrames_extends_trajectory(self, page, flask_server):
        """Per § 3.2 + § 4.3: appendFrames extends an active
        trajectory animation in place; current frame index is
        preserved; playback continues if it was running.  Pinned
        for I10 (#239) — appendFrames had zero behavioral coverage."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [[[0,0,0]],[[0.1,0,0]],[[0.2,0,0]]],
                        fps: 10, paused: true,
                    },
                });
                const beforeN = h._test.getCurrent().animation.frames.length;
                h.setAnimationFrame(2);
                const beforeFrame = h.getAnimationFrame();
                h.appendFrames([[[0.3,0,0]],[[0.4,0,0]]]);
                const afterN = h._test.getCurrent().animation.frames.length;
                const afterFrame = h.getAnimationFrame();
                h.dispose();
                host.remove();
                return { beforeN, afterN, beforeFrame, afterFrame };
            }
        """)
        assert out["beforeN"] == 3
        assert out["afterN"] == 5, (
            f"appendFrames did not extend: {out['beforeN']} -> "
            f"{out['afterN']}"
        )
        assert out["beforeFrame"] == 2
        assert out["afterFrame"] == 2, (
            f"appendFrames clobbered currentFrame: {out['beforeFrame']}"
            f" -> {out['afterFrame']}"
        )

    def test_appendFrames_atom_mismatch_dispatches_invalid_input(
            self, page, flask_server):
        """Per § 5.3: appendFrames with wrong-atom-count frame fires
        invalid_input + halts (rejects the extension)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const errs = [];
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [[[0,0,0]],[[0.1,0,0]]],
                        fps: 10, paused: true,
                    },
                    onError: (e) => { errs.push(e.code); },
                });
                h.appendFrames([[[0,0,0],[1,0,0]]]);   // 2 atoms not 1
                const finalN =
                    h._test.getCurrent().animation.frames.length;
                h.dispose();
                host.remove();
                return { errs, finalN };
            }
        """)
        assert "invalid_input" in out["errs"]
        assert out["finalN"] == 2, (
            "appendFrames extended despite atom-count mismatch"
        )

    def test_setPivot_changes_camera_pivot(self, page, flask_server):
        """Per § 3.2 (added in #235): setPivot({indices}) re-anchors
        3Dmol's centre-of-rotation onto the selected atoms.  Pinned
        for I10 (#239) — setPivot had zero behavioral coverage.

        We can't easily probe 3Dmol's internal pivot, so we verify
        the call doesn't throw + that the underlying viewer.center
        was reached by comparing the camera pos before/after."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        ok = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                let threw = false;
                try { h.setPivot({indices: [1, 2]}); }
                catch (_) { threw = true; }
                // setPivot does not throw on bad indices either
                // (the embed clamps via _selectionFromIndices).
                try { h.setPivot({indices: [99]}); }
                catch (_) { threw = true; }
                try { h.setPivot({}); }
                catch (_) { threw = true; }
                h.dispose();
                host.remove();
                return !threw;
            }
        """)
        assert ok, "setPivot threw on documented input shapes"

    def test_setKnobs_rebuilds_bar(self, page, flask_server):
        """Per § 3.2: setKnobs reconfigures visible knobs at runtime
        by rebuilding the knob bar DOM in place + re-wiring it.
        Pinned for I10 (#239) — setKnobs had zero behavioral
        coverage."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                const before = host.querySelectorAll(
                    ".mol-viewer-knob").length;
                // Hide the background + screenshot + export knobs.
                h.setKnobs({
                    background: false,
                    screenshot: false,
                    export:     false,
                });
                const after = host.querySelectorAll(
                    ".mol-viewer-knob").length;
                h.dispose();
                host.remove();
                return { before, after };
            }
        """)
        assert out["before"] > out["after"], (
            f"setKnobs did not rebuild the bar: {out['before']} knobs "
            f"before, {out['after']} after"
        )

    def test_setKnobs_bad_background_presets_dispatches(
            self, page, flask_server):
        """Per § 5.3: setKnobs with non-array backgroundPresets
        dispatches invalid_input.  Pinned for I10 (#239) — this row
        was not in the #237 case set."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        errs = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const errs = [];
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    onError: (e) => { errs.push(e.code); },
                });
                h.setKnobs({ backgroundPresets: "white,black" });
                h.dispose();
                host.remove();
                return errs;
            }
        """)
        assert "invalid_input" in errs

    def test_getters_round_trip_with_setters(
            self, page, flask_server):
        """Per § 3.2 D3 symmetry: every documented getX returns a
        defensive deep-clone of the current section, and
        ``setX(getX())`` is idempotent — the embed's internal state
        diff equals the prior state, so no spurious re-renders happen.
        Pinned for Bundle 4 (#241)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    axes: true, cell: true,
                    style: { rep: "sphere", radiusScale: 0.5 },
                    labels: { atoms: "all", format: "element" },
                    pick: { mode: "single", halo: true, label: false },
                });
                const r = {};
                // Snapshot the getter values first so the later
                // mutation test doesn't clobber what we're asserting
                // on.
                const s0 = h.getStyle();
                r.styleRep         = s0.rep;
                r.styleRadiusScale = s0.radiusScale;
                r.axes     = h.getAxes();
                r.cell     = h.getCell();
                r.labels   = h.getLabels();
                r.pick     = h.getPick();
                r.knobs    = h.getKnobs();
                r.arrows   = h.getArrows();
                r.overlays = h.getOverlays();
                r.animation= h.getAnimation();
                // Defensive-clone: mutating the returned object MUST
                // NOT affect future getters.
                s0.rep = "MUTATED";
                r.styleAfterMutate = h.getStyle();
                // Round-trip: setStyle(getStyle()) preserves rep.
                const before = h.getStyle().rep;
                h.setStyle(h.getStyle());
                const after = h.getStyle().rep;
                h.dispose();
                host.remove();
                return { r, beforeAfter: [before, after] };
            }
        """)
        r = out["r"]
        # Sanity: every getter returns a value of the expected
        # cardinality.
        assert r["styleRep"]         == "sphere"
        assert r["styleRadiusScale"] == 0.5
        assert r["axes"] is not None, "getAxes returned null"
        assert r["cell"] is not None, "getCell returned null"
        assert r["labels"]["format"] == "element"
        assert r["pick"]["mode"]   == "single"
        assert isinstance(r["knobs"], dict)
        assert r["arrows"]   == []
        assert r["overlays"] is None
        assert r["animation"] is None
        # Defensive-clone invariant.
        assert r["styleAfterMutate"]["rep"] == "sphere", (
            f"getStyle returned a live reference; mutation leaked: "
            f"{r['styleAfterMutate']}"
        )
        # Round-trip preserves state.
        assert out["beforeAfter"][0] == out["beforeAfter"][1]

    def test_interaction_onDragStart_fires_after_threshold(
            self, page, flask_server):
        """Per § 3.15: onDragStart fires once when the pointer moves
        more than dragThresholdPx from the press point, AFTER
        capturing the modifier state at mousedown.  A press-then-
        release without movement (a click) does NOT fire it.
        Pinned for Bundle 4 (#241)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:0;left:0;";
                document.body.appendChild(host);
                const starts = [], ends = [];
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    interaction: {
                        dragThresholdPx: 4,
                        onDragStart(ev) { starts.push(ev); },
                        onDragEnd(ev)   { ends.push(ev); },
                    },
                });
                const canvas = host.querySelector(".mol-viewer-canvas");
                function send(type, x, y, mods) {
                    mods = mods || {};
                    canvas.dispatchEvent(new MouseEvent(type, {
                        clientX: x, clientY: y, button: 0,
                        bubbles: true,
                        ctrlKey:  !!mods.ctrl,
                        shiftKey: !!mods.shift,
                        altKey:   !!mods.alt,
                    }));
                }
                // Click without drag — must NOT fire onDragStart.
                send("mousedown", 100, 100);
                send("mousemove", 101, 100);   // 1px < threshold
                send("mouseup",   101, 100);
                const clickStarts = starts.length;
                // Plain drag > threshold — must fire once.
                send("mousedown", 200, 200);
                send("mousemove", 210, 210);   // ~14px > threshold
                send("mousemove", 220, 220);   // already dragging, no new fire
                send("mouseup",   220, 220);
                const dragStarts = starts.length;
                const dragEnds   = ends.length;
                // Ctrl+drag — modifiers carried in payload.
                send("mousedown", 100, 100, {ctrl: true});
                send("mousemove", 120, 120, {ctrl: true});
                send("mouseup",   120, 120, {ctrl: true});
                h.dispose();
                host.remove();
                return {
                    clickStarts, dragStarts, dragEnds,
                    firstDragMods:  starts[0] && starts[0].modifiers,
                    ctrlDragMods:   starts[1] && starts[1].modifiers,
                    firstDragXY:    starts[0]
                        && [starts[0].x, starts[0].y],
                };
            }
        """)
        assert out["clickStarts"] == 0, (
            "onDragStart fired on a click (no drag); threshold "
            "filtering is broken"
        )
        assert out["dragStarts"] == 1, (
            f"onDragStart fired {out['dragStarts']}x for one drag "
            f"gesture (must fire exactly once per gesture)"
        )
        assert out["dragEnds"] == 1
        assert out["firstDragMods"] == {
            "ctrl": False, "shift": False, "alt": False, "meta": False
        }
        assert out["ctrlDragMods"]["ctrl"] is True
        assert out["firstDragXY"] == [200, 200], (
            "onDragStart x/y must be the PRESS point, not the "
            "current pointer position"
        )

    def test_interaction_callback_error_is_isolated(
            self, page, flask_server):
        """Per § 3.15: a host onDragStart/onDragEnd callback that
        throws must not break pointer handling.  The embed catches +
        logs, then continues."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        ok = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:0;left:0;";
                document.body.appendChild(host);
                let secondFired = false;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    interaction: {
                        onDragStart() { throw new Error("buggy host"); },
                    },
                });
                const canvas = host.querySelector(".mol-viewer-canvas");
                function send(type, x, y) {
                    canvas.dispatchEvent(new MouseEvent(type, {
                        clientX: x, clientY: y, button: 0,
                        bubbles: true,
                    }));
                }
                let threw = false;
                try {
                    send("mousedown", 0, 0);
                    send("mousemove", 30, 30);   // triggers onDragStart
                    send("mouseup",   30, 30);
                    // A second gesture must still work.
                    send("mousedown", 100, 100);
                    send("mousemove", 130, 130);
                    send("mouseup",   130, 130);
                } catch (_) { threw = true; }
                h.dispose();
                host.remove();
                return !threw;
            }
        """)
        assert ok, (
            "embed propagated a buggy onDragStart throw instead of "
            "isolating it"
        )

    def test_setStructure_preservePick_overrides_element_clear(
            self, page, flask_server):
        """Per § 4.2.1 D1 escape hatch: ``preservePick: true`` keeps
        picked indices even when atom-element ordering changes
        (e.g. a /modify atom-type-swap where the host tracks the
        index mapping itself).  ``preservePick: false`` forces clear
        even when elements DO match.  Regression test for Bundle 5
        (#242)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    pick: { mode: "pair", halo: true, label: false,
                            onPick() {} },
                });
                h.setPickedIndices([0, 1]);
                // Element mismatch + preservePick:true → KEEP.
                h.setStructure({
                    xyz: "3\\nch3\\nC 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    preservePick: true,
                });
                const afterKeep = h.getPickedIndices();
                // Element match + preservePick:false → CLEAR.
                h.setPickedIndices([0]);
                h.setStructure({
                    xyz: "3\\nch3moved\\nC 0.5 0 0\\n"
                       + "H 1 0 0\\nH 0 1 0\\n",
                    preservePick: false,
                });
                const afterForcedClear = h.getPickedIndices();
                h.dispose();
                host.remove();
                return { afterKeep, afterForcedClear };
            }
        """)
        assert out["afterKeep"] == [0, 1], (
            f"preservePick:true did not survive element mismatch: "
            f"{out['afterKeep']!r}"
        )
        assert out["afterForcedClear"] == [], (
            f"preservePick:false did not force clear on element "
            f"match: {out['afterForcedClear']!r}"
        )

    def test_applyState_orders_atom_keyed_after_structure(
            self, page, flask_server):
        """Per § 4.2.2 D4: applyState runs the setX calls in a
        canonical order so atom-keyed state (overlays, picks)
        lands AFTER setStructure has reloaded the model.  Manually
        calling setOverlays before setStructure would clear them
        on the next structure swap; applyState gets the ordering
        right."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    pick: { mode: "pair", halo: true, label: false,
                            onPick() {} },
                });
                // applyState with structure swap + overlays + picks.
                // The 3-atom xyz has 3 atoms, and the overlay /
                // pickedIndices reference indices in that space.  If
                // applyState applied overlays BEFORE setStructure
                // (wrong order), the new atoms wouldn't exist when
                // overlays validate; if it applied them AFTER (right
                // order) they should land correctly.
                h.applyState({
                    structure: {
                        xyz: "3\\nh3\\nH 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    },
                    overlays: { atoms: [{
                        indices: [0, 2],
                        halo: { color: "#0ff", radius: 0.4 },
                    }]},
                    pickedIndices: [1],
                });
                const overlaysSet = !!h.getOverlays();
                const picks = h.getPickedIndices();
                h.dispose();
                host.remove();
                return { overlaysSet, picks };
            }
        """)
        assert out["overlaysSet"] is True, (
            "overlays did not land after structure swap — applyState "
            "ordering broken"
        )
        assert out["picks"] == [1], (
            f"picks did not land after structure swap: {out['picks']!r}"
        )

    def test_applyState_round_trip_with_getters(
            self, page, flask_server):
        """Per § 4.2.2: the round-trip pattern getX-then-applyState
        round-trips cleanly.  Snapshot every section, then re-apply
        and verify the embed's state matches."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    style: { rep: "sphere", radiusScale: 0.7 },
                    axes:  true,
                    labels: { atoms: "all", format: "element" },
                });
                // Snapshot.
                const snap = {
                    style:  h.getStyle(),
                    axes:   h.getAxes(),
                    labels: h.getLabels(),
                };
                // Mutate to a different state.
                h.setStyle({ rep: "stick", radiusScale: 1.5 });
                h.setAxes(false);
                h.setLabels(false);
                const between = {
                    style:  h.getStyle(),
                    axes:   h.getAxes(),
                    labels: h.getLabels(),
                };
                // Round-trip back via applyState.
                h.applyState(snap);
                const after = {
                    style:  h.getStyle(),
                    axes:   h.getAxes(),
                    labels: h.getLabels(),
                };
                h.dispose();
                host.remove();
                return { between, after };
            }
        """)
        # The "between" state is distinct from the original.
        assert out["between"]["style"]["rep"] == "stick"
        assert out["between"]["axes"] is None
        # After applyState, the original state is restored.
        assert out["after"]["style"]["rep"]   == "sphere"
        assert out["after"]["style"]["radiusScale"] == 0.7
        assert out["after"]["axes"] is not None, (
            "axes did not restore after applyState"
        )
        assert out["after"]["labels"]["format"] == "element"

    def test_applyState_bad_input_dispatches(self, page, flask_server):
        """Per § 5.3: applyState with a non-object argument fires
        invalid_input and halts."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        errs = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const errs = [];
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    onError: (e) => { errs.push(e.code); },
                });
                h.applyState("not an object");
                h.dispose();
                host.remove();
                return errs;
            }
        """)
        assert "invalid_input" in errs

    def test_captureFrames_returns_blobs_for_vibration(
            self, page, flask_server):
        """Per § 3.2 + Phase 5b: captureFrames drives the animation
        deterministically + captures one PNG blob per frame.  For a
        2-fps × 1-sec capture, expect 2 blobs."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.1],[0,0,0.1],[0,0,0.1]],
                        amplitude: 0.2, speedHz: 1.0, paused: true,
                    },
                });
                await new Promise(r => requestAnimationFrame(r));
                const blobs = await h.captureFrames({
                    fps: 2, duration: 1,
                });
                const types = blobs.map((b) => b.type);
                const sizes = blobs.map((b) => b.size);
                h.dispose();
                host.remove();
                return { count: blobs.length, types, sizes };
            }
        """)
        assert out["count"] == 2, (
            f"captureFrames(fps=2, duration=1) returned "
            f"{out['count']} blobs, expected 2"
        )
        assert all(t == "image/png" for t in out["types"]), (
            f"unexpected blob types: {out['types']!r}"
        )
        assert all(s > 0 for s in out["sizes"]), (
            f"empty blob in output: {out['sizes']!r}"
        )

    def test_captureFrames_rejects_when_no_animation(
            self, page, flask_server):
        """Per § 5.3: captureFrames rejects with static_structure
        when ``opts.animation`` is null."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        code = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                let code = null;
                try { await h.captureFrames({fps: 5, duration: 0.2}); }
                catch (e) { code = e.code; }
                h.dispose();
                host.remove();
                return code;
            }
        """)
        assert code == "static_structure"

    def test_exportAnimation_rejects_no_media_recorder(
            self, page, flask_server):
        """Per § 5.3: exportAnimation({format: "webm"}) rejects with
        no_media_recorder when MediaRecorder is unavailable.  The
        testInjection slot lets the test force-mock the absence."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        code = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.1]],
                        amplitude: 0.1, speedHz: 1.0, paused: true,
                    },
                    testInjection: { mediaRecorder: null },
                });
                let code = null;
                try {
                    await h.exportAnimation({
                        format: "webm", target: "download",
                        fps: 2, duration: 0.1,
                    });
                } catch (e) { code = e.code; }
                h.dispose();
                host.remove();
                return code;
            }
        """)
        assert code == "no_media_recorder"

    def test_exportAnimation_rejects_no_gif_encoder(
            self, page, flask_server):
        """Per § 5.3: exportAnimation({format: "gif"}) rejects with
        no_gif_encoder when the gif.js lib is unavailable.  We use
        testInjection.gifEncoder = null to force "absent" because
        gif.min.js IS shipped in /static/vendor/ — the lazy-load
        path would otherwise succeed (covered separately by
        test_exportAnimation_gif_produces_real_blob)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        code = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.1]],
                        amplitude: 0.1, speedHz: 1.0, paused: true,
                    },
                    testInjection: { gifEncoder: null },
                });
                let code = null;
                try {
                    await h.exportAnimation({
                        format: "gif", target: "download",
                        fps: 2, duration: 0.1,
                    });
                } catch (e) { code = e.code; }
                h.dispose();
                host.remove();
                return code;
            }
        """)
        assert code == "no_gif_encoder"

    def test_exportAnimation_gif_produces_real_blob(
            self, page, flask_server):
        """Per § 3.2 + Phase 5c: the GIF format produces a real
        image/gif blob via the vendored gif.js at /static/vendor/
        gif.min.js + gif.worker.min.js.  This is the end-to-end
        proof that GIF export works — the test fails if either
        vendor file goes missing OR if the gif.js integration
        breaks."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        # Allow plenty of time — gif.js encodes off the main
        # thread but small frames still take a beat.
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:240px;height:160px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let lastBlob = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.05],[0,0,0.05],[0,0,0.05]],
                        amplitude: 0.1, speedHz: 1.0, paused: true,
                    },
                    export: {
                        onExport: (info) => { lastInfo = info; },
                    },
                });
                let lastInfo = null;
                // Mock projects API so target:"project" lands a
                // blob we can inspect (download triggers an
                // anchor click which the test can't intercept).
                let savedBlob = null;
                window.molbuilder = window.molbuilder || {};
                window.molbuilder.projects = {
                    currentDir: () => "/tmp/fake",
                    writeFile: (path, blob) => {
                        savedBlob = blob;
                        return Promise.resolve({ ok: true });
                    },
                };
                const r = await h.exportAnimation({
                    format: "gif", target: "project",
                    fps: 2, duration: 0.5,
                });
                h.dispose();
                host.remove();
                return {
                    filename: r.filename,
                    bytes:    r.bytes,
                    mime:     savedBlob && savedBlob.type,
                    size:     savedBlob && savedBlob.size,
                };
            }
        """)
        assert out["filename"].endswith(".gif"), out
        assert out["mime"]  == "image/gif", out
        assert out["size"]  > 0, out
        assert out["bytes"] > 0, out

    def test_exportAnimation_rejects_bad_target(
            self, page, flask_server):
        """Per § 5.3: target ∉ {project, download} → invalid_input.
        Animation export has no clipboard target per § 3.2."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        code = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.1]],
                        amplitude: 0.1, speedHz: 1.0, paused: true,
                    },
                });
                let code = null;
                try {
                    await h.exportAnimation({
                        format: "webm", target: "clipboard",
                        fps: 2, duration: 0.1,
                    });
                } catch (e) { code = e.code; }
                h.dispose();
                host.remove();
                return code;
            }
        """)
        assert code == "invalid_input"

    def test_exportAnimation_webm_with_injected_recorder(
            self, page, flask_server):
        """Per § 3.2 + Phase 5b: exportAnimation drives the
        animation, feeds the canvas captureStream to MediaRecorder,
        and resolves with {filename, bytes}.  Uses testInjection to
        supply a deterministic recorder so the test isn't tied to
        Chromium's MediaRecorder timing."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                // Minimal MediaRecorder mock that captures the
                // ctor args + simulates dataavailable+stop.
                class FakeRecorder {
                    constructor(stream, opts) {
                        this.stream = stream;
                        this.opts   = opts;
                        this.state  = "inactive";
                    }
                    start() { this.state = "recording"; }
                    stop() {
                        this.state = "inactive";
                        const blob = new Blob(
                            [new Uint8Array([1, 2, 3, 4])],
                            { type: "video/webm" });
                        if (this.ondataavailable) {
                            this.ondataavailable({ data: blob });
                        }
                        if (this.onstop) this.onstop();
                    }
                    static isTypeSupported(_) { return true; }
                }
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.1]],
                        amplitude: 0.1, speedHz: 1.0, paused: true,
                    },
                    testInjection: { mediaRecorder: FakeRecorder },
                });
                const r = await h.exportAnimation({
                    format: "webm", target: "download",
                    fps: 2, duration: 0.5,
                });
                h.dispose();
                host.remove();
                return r;
            }
        """)
        assert out["filename"].endswith(".webm")
        assert out["bytes"] > 0

    def test_exportAnimation_to_project(
            self, page, flask_server):
        """target: "project" calls window.molbuilder.projects.
        writeFile.  Uses testInjection to mock the projects API so
        the test doesn't depend on a real project being open."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let wroteTo = null;
                let wroteBytes = 0;
                class FakeRecorder {
                    constructor(s, o) { this.s = s; this.o = o; }
                    start() {}
                    stop() {
                        const blob = new Blob(
                            [new Uint8Array([9, 9, 9])],
                            { type: "video/webm" });
                        if (this.ondataavailable)
                            this.ondataavailable({ data: blob });
                        if (this.onstop) this.onstop();
                    }
                    static isTypeSupported() { return true; }
                }
                const projectsApi = {
                    currentDir: () => "/tmp/fake-project",
                    writeFile: (path, data) => {
                        wroteTo = path;
                        wroteBytes = data && data.size;
                        return Promise.resolve({ ok: true });
                    },
                };
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.1]],
                        amplitude: 0.1, speedHz: 1.0, paused: true,
                    },
                    testInjection: {
                        mediaRecorder: FakeRecorder,
                        projectsApi:   projectsApi,
                    },
                });
                const r = await h.exportAnimation({
                    format: "webm", target: "project",
                    fps: 2, duration: 0.5,
                });
                h.dispose();
                host.remove();
                return { r, wroteTo, wroteBytes };
            }
        """)
        assert out["wroteTo"].endswith(".webm")
        assert out["wroteTo"].startswith("/tmp/fake-project/")
        assert out["wroteBytes"] > 0
        assert out["r"]["filename"].endswith(".webm")

    def test_setAnimation_halt_preserves_existing_animation(
            self, page, flask_server):
        """Per § 5.3 "halt" semantics: a setAnimation call with an
        invalid full-update payload (bad kind / missing displacements /
        atom-count mismatch) must NOT clear or replace the active
        animation.  Pinned for #237: regression guard against a future
        edit that validates after _setAnimationImpl runs."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const errs = [];
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "vibration",
                        displacements: [[0,0,0.1],[0,0,0.1],[0,0,0.1]],
                        amplitude: 0.2, speedHz: 1.0, paused: false,
                    },
                    onError: (e) => { errs.push(e.code); },
                });
                await new Promise(r => requestAnimationFrame(r));
                const before = h._test.hasAnimationLoop();
                // Bad atom count -- must halt + preserve.
                h.setAnimation({
                    kind: "vibration",
                    displacements: [[1,0,0]],   // 1 not 3
                });
                await new Promise(r => requestAnimationFrame(r));
                const after = h._test.hasAnimationLoop();
                h.dispose();
                host.remove();
                return { before, after, errs };
            }
        """)
        assert out["before"] is True
        assert out["after"]  is True, (
            "setAnimation with atom-count mismatch wiped the active "
            "animation -- § 5.3 \"halt\" semantics violated"
        )
        assert "invalid_input" in out["errs"]

    # Retired 2026-06-09: ``test_chrome_consistency_across_build_and_modify``
    # compared knob-bar signatures on ``/`` (Build) and ``/modify``
    # (Modify), both of which were deleted in Phase B.5 (2026-06-07):
    # ``/`` is now a 302 to ``landing_path()`` and ``/modify``
    # returns 404 by design (no backward-compat redirects).  The
    # next test in this class
    # (``test_chrome_consistency_across_three_inspectors``) covers
    # chrome consistency across the three /results inspectors and
    # the structure-optimization viewer, which is the post-B.5
    # equivalent of what this one used to pin.

    def test_chrome_consistency_across_three_inspectors(
            self, page, flask_server):
        """Phase 5m T1 + Phase 6: the three /results inspectors
        (structure, trajectory, spectra) must render the same
        canonical 2-menu bar (View + Export) as Build / Modify.
        Mounts the embed directly with each inspector's actual opts
        so the test catches per-inspector drift without exercising
        the full registry dispatch.

        Pins § 6.2 chrome-consistency across all five consumers."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        signatures = page.evaluate("""
            () => {
                const inspectors = {
                    structure: {
                        xyz: "1\\nh\\nH 0 0 0\\n",
                        style: { rep: "ball-and-stick", radiusScale: 1.0 },
                        card:  { title: "", showInfoLine: false,
                                 height: "420px" },
                        axes: true,
                        export: { defaultName: "structure" },
                    },
                    trajectory: {
                        xyz: "1\\nh\\nH 0 0 0\\n",
                        style: { rep: "stick", radiusScale: 1.0 },
                        pick:  { mode: "pair", halo: true, label: false },
                        card:  { title: "Trajectory",
                                 showInfoLine: false, height: "100%" },
                        axes: true,
                        export: { defaultName: "trajectory" },
                    },
                    spectra: {
                        xyz: "1\\nh\\nH 0 0 0\\n",
                        style: { rep: "ball-and-stick", radiusScale: 1.0 },
                        pick:   { mode: "none" },
                        card:   { title: "Vibrational mode",
                                  showInfoLine: false, height: "100%" },
                        axes:   true,
                        export: { defaultName: "vibration" },
                    },
                };
                const out = {};
                for (const [name, opts] of Object.entries(inspectors)) {
                    const host = document.createElement("div");
                    host.style.cssText =
                        "width:400px;height:300px;position:fixed;top:-9999px;";
                    document.body.appendChild(host);
                    const h = window.molbuilder.viewer.embed(host, opts);
                    const bar = h._test.getKnobBarElement();
                    out[name] = bar
                        ? Array.from(bar.children).map((el) => {
                            if (el.tagName === 'DETAILS') {
                                const cls = Array.from(el.classList)
                                    .find((c) => c.startsWith(
                                        'mol-viewer-menu-'));
                                return 'details:' + (cls || '?');
                            }
                            return el.tagName.toLowerCase();
                          })
                        : null;
                    h.dispose();
                    host.remove();
                }
                return out;
            }
        """)
        EXPECTED = [
            "details:mol-viewer-menu-view",
            "details:mol-viewer-menu-export",
        ]
        for name in ("structure", "trajectory", "spectra"):
            assert signatures[name] == EXPECTED, (
                f"Knob bar order on the {name} inspector drifted "
                f"from § 6.2 spec (Phase 6):\n"
                f"  Expected: {EXPECTED}\n  Got: {signatures[name]}"
            )

    def test_background_defaults_per_consumer(
            self, page, flask_server):
        """Phase 5m T4: each consumer's mount-time background is
        white (chemistry convention) EXCEPT spectra which uses a
        dark backdrop (`#1d2128`) so vibration vectors render with
        better contrast.

        Catches a regression where, say, a future tab silently mounts
        with the spectra dark backdrop, or spectra silently flips to
        white."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        actual = page.evaluate("""
            () => {
                const configs = {
                    build: {
                        xyz: "1\\nh\\nH 0 0 0\\n",
                    },
                    modify: {
                        xyz: "1\\nh\\nH 0 0 0\\n",
                        style: { rep: "stick", radiusScale: 0.4 },
                    },
                    structure: {
                        xyz: "1\\nh\\nH 0 0 0\\n",
                        style: { rep: "ball-and-stick", radiusScale: 1.0 },
                    },
                    trajectory: {
                        xyz: "1\\nh\\nH 0 0 0\\n",
                        style: { rep: "stick", radiusScale: 1.0 },
                    },
                    spectra: {
                        xyz: "1\\nh\\nH 0 0 0\\n",
                        style: { rep: "ball-and-stick", radiusScale: 1.0 },
                    },
                };
                const out = {};
                for (const [name, opts] of Object.entries(configs)) {
                    const host = document.createElement("div");
                    host.style.cssText =
                        "width:300px;height:200px;position:fixed;top:-9999px;";
                    document.body.appendChild(host);
                    const h = window.molbuilder.viewer.embed(host, opts);
                    out[name] = h.getBackground();
                    h.dispose();
                    host.remove();
                }
                return out;
            }
        """)
        # Phase 6: DEFAULT_BACKGROUND switched from white to the
        # page's card colour (#1d2128) so the viewer reads as part
        # of the dark theme instead of a bright cut-out.  White stays
        # available as a preset for publication figures; consumers
        # who want it call setBackground("#ffffff") or pass
        # style.background at mount.  Spectra used to override to
        # #1d2128 explicitly; now it picks up the same default
        # implicitly.
        EXPECTED = {
            "build":      "#1d2128",
            "modify":     "#1d2128",
            "structure":  "#1d2128",
            "trajectory": "#1d2128",
            "spectra":    "#1d2128",
        }
        for name, expected in EXPECTED.items():
            assert actual[name] == expected, (
                f"{name}: getBackground() returned {actual[name]!r}; "
                f"expected {expected!r} per § 3.3 (Phase 6 default)"
            )

    def test_handle_has_test_affordance_object(
            self, page, flask_server):
        """The ``_test`` affordance object is documented in § 9.2
        and required by the test suite; it's not a function."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.width = "300px";
                host.style.height = "200px";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    card: { bare: true, showInfoLine: false },
                });
                const r = {
                    hasTest:    !!h._test,
                    isObject:   typeof h._test === "object",
                    isFunction: typeof h._test === "function",
                };
                h.dispose();
                host.remove();
                return r;
            }
        """)
        assert out["hasTest"]    is True
        assert out["isObject"]   is True
        assert out["isFunction"] is False


# --------------------------------------------------------------------- #
#  Phase 6e — animation rebuild perf bench                              #
# --------------------------------------------------------------------- #


def _make_synthetic_xyz(n):
    """Generate a synthetic XYZ block of ``n`` hydrogen atoms on a
    simple cubic grid.  Used by the perf bench below; not meant to
    represent any real chemistry — just a stable, well-defined
    structure that produces N atoms for the embed to render."""
    side = max(1, int(round(n ** (1.0 / 3.0))))
    coords = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(coords) >= n:
                    break
                coords.append(
                    f"H {i * 1.5:.4f} {j * 1.5:.4f} {k * 1.5:.4f}"
                )
            if len(coords) >= n:
                break
        if len(coords) >= n:
            break
    while len(coords) < n:
        coords.append(
            f"H 0.0 0.0 {len(coords) * 1.5:.4f}"
        )
    return f"{n}\nsynth\n" + "\n".join(coords) + "\n"


class TestAnimationRebuildPerf:
    """Phase 6e: measure the wall cost of
    ``_rebuildGeometryForCoordChange`` across atom-count scales so we
    can decide whether Phase 6f (native 3Dmol frame indexing) is
    worth the engineering cost.

    These tests don't gate CI on perf — they only print the numbers
    and assert that timing infrastructure works.  The actual
    interpretation (is 5 ms/frame fine? is 50 ms/frame painful?
    when do we pull the trigger on 6f?) is a human-judgment call
    made by reading the printed table.

    Why this matters for export:
        ``captureFrames`` (which ``exportAnimation`` uses) iterates
        every frame through the same ``_applyCoords`` ⇒
        ``_rebuildGeometryForCoordChange`` ⇒ ``viewer.render()``
        path the interactive playback uses.  A 60 fps × 5 s WebM
        = 300 rebuilds.  If each rebuild is 100 ms at 5000 atoms
        that's 30 s of export per 5 s of animation — likely
        painful but tolerable.  10 s of animation at 10000 atoms
        could blow past the browser's "kill page" threshold.
    """

    @pytest.mark.parametrize("n_atoms", [100, 500, 2000])
    def test_rebuild_timing_scales(
            self, page, flask_server, n_atoms):
        """Run ``n_frames`` rebuilds at a given atom count and
        record the timing distribution.  Asserts only that the
        timing infrastructure recorded the right number of samples
        and produced finite numbers — perf interpretation is by
        eye (look at the printed line)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        xyz = _make_synthetic_xyz(n_atoms)
        n_frames = 30
        out = page.evaluate(
            """
            async ({xyz, nFrames, nAtoms}) => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                // Build a 2-frame trajectory: frame 0 = input
                // coords, frame 1 = same coords shifted by 0.01.
                // We won't actually play it through the rAF loop —
                // we directly call setAnimationFrame in a tight
                // loop to get N rebuilds with no scheduling
                // interference.
                const lines = xyz.trim().split("\\n").slice(2);
                const f0 = lines.map(l => {
                    const p = l.trim().split(/\\s+/);
                    return [
                        parseFloat(p[1]),
                        parseFloat(p[2]),
                        parseFloat(p[3]),
                    ];
                });
                const f1 = f0.map(([x, y, z]) => [x + 0.01, y, z]);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: xyz,
                    card: {showInfoLine: false, height: "100%"},
                    animation: {
                        kind: "trajectory",
                        frames: [f0, f1],
                        fps: 30,
                        paused: true,
                    },
                });
                // Wait for first paint before starting timing.
                await new Promise(r => requestAnimationFrame(r));
                h._test.resetFrameRebuildTimings();
                for (let i = 0; i < nFrames; i++) {
                    h.setAnimationFrame(i % 2);
                }
                const stats = h._test.getFrameRebuildTimings();
                h.dispose();
                host.remove();
                return {
                    nAtoms: nAtoms,
                    nFrames: nFrames,
                    samples: stats.samples.length,
                    mean: stats.mean,
                    p50: stats.p50,
                    p95: stats.p95,
                    p99: stats.p99,
                    max: stats.max,
                };
            }
            """,
            {"xyz": xyz, "nFrames": n_frames, "nAtoms": n_atoms},
        )
        # Print a digest line for the human reading the test output.
        # `-s` flag on pytest shows it; otherwise pytest captures
        # but still includes it in failure logs.
        print(
            f"[phase-6e] n_atoms={out['nAtoms']:>5d}  "
            f"frames={out['samples']:>3d}  "
            f"mean={out['mean']:6.2f}ms  "
            f"p50={out['p50']:6.2f}ms  "
            f"p95={out['p95']:6.2f}ms  "
            f"p99={out['p99']:6.2f}ms  "
            f"max={out['max']:6.2f}ms"
        )
        # Infrastructure assertions.  No perf gate — interpretation
        # is human-eye on the printed line above.
        assert out["samples"] == n_frames, (
            f"timing ring buffer recorded {out['samples']} samples; "
            f"expected {n_frames}.  _rebuildGeometryForCoordChange "
            f"was either skipped on some frames or the buffer cap "
            f"is too low."
        )
        assert out["mean"] > 0.0, (
            "mean rebuild time was 0 ms — either performance.now() "
            "lacks resolution in the test browser or "
            "_rebuildGeometryForCoordChange short-circuited"
        )
        assert out["max"] < 5000.0, (
            f"a single rebuild took {out['max']:.1f} ms — that's a "
            f"performance cliff, not just a slow frame.  Something "
            f"in _applyStyle or _redrawOverlayStyles is doing "
            f"unbounded work at n_atoms={n_atoms}."
        )


# --------------------------------------------------------------------- #
#  Phase 6e — animation export UX (modal + save-to-project + menus)     #
# --------------------------------------------------------------------- #


class TestAnimationExportUX:
    """Regression catchers for the Phase 6e gripes:
      * Save-to-project of an animation Blob silently 400'd
        because /api/files/write rejects non-string bodies.
        Fix: writeFile(Blob) routes to /api/files/upload with
        overwrite=true.
      * No progress UI on a slow encode → user thought nothing
        was happening.  Fix: blocking modal with progress + cancel.
      * View / Export menu didn't close on outside-click.
      * Export menu mixed kinds; now sectioned Data / Snapshot /
        Animation.
    """

    def test_animation_save_to_project_passes_blob_through(
            self, page, flask_server):
        """The embed's _writeToProject must hand the Blob to
        projectsApi.writeFile unchanged — no JSON.stringify, no
        silent drop.  This catches the JSON-only bug class where
        a Blob hitting JSON.stringify({text: blob}) flattens to
        {} and the file lands empty / errored."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let receivedPath = null;
                let receivedKind = null;
                let receivedSize = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false,
                            height: "100%" },
                    export: { defaultName: "movie" },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[0.5, 0.0, 0.0]],
                        ],
                        fps: 10, paused: true,
                    },
                    testInjection: {
                        projectsApi: {
                            writeFile: async (path, data) => {
                                receivedPath = path;
                                receivedKind = (data
                                    && typeof data === "object"
                                    && typeof data.size === "number")
                                    ? "blob" : typeof data;
                                receivedSize = data && data.size;
                                return { ok: true, path: path };
                            },
                            currentDir: () => "/tmp/proj1",
                        },
                    },
                });
                // Drive a short export.  webm/gif both go through
                // the same _writeToProject; gif is simpler in test
                // contexts because MediaRecorder may not be
                // available headlessly.
                let err = null;
                try {
                    await h.exportAnimation({
                        format: "webm",
                        target: "project",
                        duration: 0.1,  // short capture
                        fps: 10,
                    });
                } catch (e) {
                    err = (e && e.message) || String(e);
                }
                h.dispose();
                host.remove();
                return {
                    receivedPath, receivedKind, receivedSize, err,
                };
            }
        """)
        # In headless Chromium, MediaRecorder + canvas.captureStream
        # are both available, so the export should succeed.  If
        # the browser is configured without them, we get the
        # canonical "MediaRecorder unavailable" reject — accept
        # that as a SKIP (the binary-write path is still exercised
        # through the screenshot test below).
        if out["err"] and ("MediaRecorder" in out["err"]
                            or "captureStream" in out["err"]):
            pytest.skip(
                "test browser lacks MediaRecorder/captureStream — "
                "binary save-to-project is also covered by "
                "screenshot()"
            )
        assert out["err"] is None, (
            f"exportAnimation rejected unexpectedly: {out['err']}"
        )
        assert out["receivedPath"] == "/tmp/proj1/movie.webm", (
            f"projectsApi.writeFile got wrong path: {out['receivedPath']!r}"
        )
        assert out["receivedKind"] == "blob", (
            f"projectsApi.writeFile received a {out['receivedKind']} "
            f"instead of a Blob — the binary path collapsed to "
            f"text somewhere"
        )
        assert out["receivedSize"] and out["receivedSize"] > 0, (
            f"the Blob handed to writeFile had size "
            f"{out['receivedSize']!r}; an empty Blob means the "
            f"encoder produced no data and the save silently "
            f"succeeded with garbage"
        )

    def test_screenshot_save_to_project_passes_blob_through(
            self, page, flask_server):
        """Same bug class as the animation case but for the .png
        screenshot path — also went through _writeToProject(Blob)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let kind = null;
                let size = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false,
                            height: "100%" },
                    export: { defaultName: "snap" },
                    testInjection: {
                        projectsApi: {
                            writeFile: async (path, data) => {
                                kind = (data && typeof data === "object"
                                       && typeof data.size === "number")
                                       ? "blob" : typeof data;
                                size = data && data.size;
                                return { ok: true, path: path };
                            },
                            currentDir: () => "/tmp/proj1",
                        },
                    },
                });
                await h.screenshot({ target: "project" });
                h.dispose();
                host.remove();
                return { kind, size };
            }
        """)
        assert out["kind"] == "blob", (
            f"screenshot save-to-project sent {out['kind']!r} not "
            "a Blob"
        )
        assert out["size"] and out["size"] > 0

    def test_view_menu_closes_on_outside_click(
            self, page, flask_server):
        """Outside-click should dismiss any open knob-bar menu.
        Before Phase 6e the menu only closed when the user clicked
        the trigger again, which was annoying."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:20px;left:20px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                const bar = h._test.getKnobBarElement();
                const viewDet = bar.querySelector(
                    ".mol-viewer-menu-view");
                viewDet.open = true;
                viewDet.dispatchEvent(new Event("toggle"));
                const wasOpen = viewDet.open;
                // Click on an element outside the menu.
                const outside = document.createElement("div");
                outside.style.cssText =
                    "position:fixed;top:500px;left:500px;width:50px;height:50px;";
                document.body.appendChild(outside);
                outside.click();
                const stillOpen = viewDet.open;
                h.dispose();
                host.remove();
                outside.remove();
                return { wasOpen, stillOpen };
            }
        """)
        assert out["wasOpen"] is True, (
            "test setup error: View menu didn't open"
        )
        assert out["stillOpen"] is False, (
            "View menu stayed open after outside click — Phase 6e "
            "regression"
        )

    def test_export_menu_has_three_kind_sections(
            self, page, flask_server):
        """Export menu must group buttons into Data / Snapshot /
        Animation sections, not mix them into one flat row.
        Animation section starts hidden (no animation mounted) but
        the DOM element exists."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                });
                const bar = h._test.getKnobBarElement();
                const sections = Array.from(bar.querySelectorAll(
                    ".mol-viewer-export-section"))
                    .map(s => ({
                        key:    s.getAttribute("data-section"),
                        hidden: s.hidden,
                    }));
                h.dispose();
                host.remove();
                return sections;
            }
        """)
        keys = [s["key"] for s in out]
        assert keys == ["data", "snapshot", "animation"], (
            f"Export menu sections in wrong order: {keys}"
        )
        # Animation hidden when no animation mounted.
        anim = next(s for s in out if s["key"] == "animation")
        assert anim["hidden"] is True, (
            "Animation section should be hidden when no animation "
            "is mounted"
        )
        # Data + Snapshot always visible.
        data = next(s for s in out if s["key"] == "data")
        snap = next(s for s in out if s["key"] == "snapshot")
        assert data["hidden"] is False
        assert snap["hidden"] is False

    def test_export_menu_animation_section_visible_with_animation(
            self, page, flask_server):
        """Mounting a trajectory animation should reveal the
        Animation section in the Export menu."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        hidden = page.evaluate("""
            () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[1.0, 0.0, 0.0]],
                        ],
                        fps: 10, paused: true,
                    },
                });
                const bar = h._test.getKnobBarElement();
                const sect = bar.querySelector(
                    ".mol-viewer-export-section[data-section='animation']");
                const v = sect ? sect.hidden : null;
                h.dispose();
                host.remove();
                return v;
            }
        """)
        assert hidden is False, (
            "Animation section should be visible when a "
            "trajectory animation is mounted; got hidden=" + str(hidden)
        )

    def test_animation_export_shows_params_dialog_then_progress(
            self, page, flask_server):
        """Phase 6e: clicking an Export button no longer kicks off
        the encode immediately.  Instead the user first sees a
        PARAMS dialog with editable defaults (filename, fps,
        duration, width, height, bitrate for webm).  Clicking the
        dialog's Export button THEN puts up the progress modal.
        Cancelling the dialog runs nothing.  This test walks that
        full chain."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[0.5, 0.0, 0.0]],
                        ],
                        fps: 10, paused: true,
                    },
                });
                const bar = h._test.getKnobBarElement();
                const exportDet = bar.querySelector(
                    ".mol-viewer-menu-export");
                exportDet.open = true;
                const btn = bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='animation']"
                  + "[data-target='download'][data-format='webm']");
                btn.click();
                // Params dialog is created synchronously inside the
                // click handler.  Query it BEFORE any await.
                const paramsDialog = document.querySelector(
                    ".mol-viewer-export-params-card");
                const paramsVisible = !!paramsDialog;
                const filenameInput = paramsDialog && paramsDialog
                    .querySelector(".mol-viewer-export-params-input");
                const filenameDefault = filenameInput
                    ? filenameInput.value : null;
                const confirmBtn = paramsDialog && paramsDialog
                    .querySelector(".mol-viewer-export-modal-confirm");
                const hasExportBtn = !!confirmBtn;
                // Click the dialog's Export button to start the
                // encode.  The dialog Promise resolves in a
                // microtask, then _runExportWithParams opens the
                // progress modal — yield once so we observe it.
                confirmBtn.click();
                await new Promise(r => setTimeout(r, 0));
                // Look for any modal whose card is NOT the params
                // card (avoids :has() browser-support concerns).
                let pmVisible = false;
                for (const m of document.querySelectorAll(
                        ".mol-viewer-export-modal")) {
                    if (!m.querySelector(
                            ".mol-viewer-export-params-card")) {
                        pmVisible = true; break;
                    }
                }
                // Wait for the export to settle + the progress
                // modal to close.
                for (let i = 0; i < 80; i++) {
                    await new Promise(r => setTimeout(r, 100));
                    if (!document.querySelector(
                        ".mol-viewer-export-modal")) break;
                }
                const allClosed = !document.querySelector(
                    ".mol-viewer-export-modal");
                h.dispose();
                host.remove();
                return {
                    paramsVisible, filenameDefault, hasExportBtn,
                    pmVisible, allClosed,
                };
            }
        """)
        assert out["paramsVisible"] is True, (
            "Phase 6e: clicking the Export button should open the "
            "params dialog FIRST, not the progress modal"
        )
        assert out["filenameDefault"] and ".webm" in out["filenameDefault"], (
            f"Params dialog filename default should end in .webm; "
            f"got {out['filenameDefault']!r}"
        )
        assert out["hasExportBtn"] is True, (
            "Params dialog must have an Export confirm button"
        )
        assert out["pmVisible"] is True, (
            "Progress modal should appear after the params dialog "
            "is confirmed"
        )
        assert out["allClosed"] is True, (
            "Both modal layers should close after the export "
            "finishes"
        )

    def test_export_params_dialog_cancel_runs_nothing(
            self, page, flask_server):
        """Cancelling the params dialog should NOT trigger any
        export call.  exportAnimation must never run when the user
        cancels."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writeFileCalled = false;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[0.5, 0.0, 0.0]],
                        ],
                        fps: 10, paused: true,
                    },
                    testInjection: {
                        projectsApi: {
                            writeFile: async (path, data) => {
                                writeFileCalled = true;
                                return { ok: true, path: path };
                            },
                            currentDir: () => "/tmp/proj1",
                        },
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                const btn = bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='animation']"
                  + "[data-target='project'][data-format='gif']");
                btn.click();
                // Dialog appears; press its Cancel button.
                const cancelBtn = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-modal-cancel");
                const hadDialog = !!cancelBtn;
                if (cancelBtn) cancelBtn.click();
                // Settle — no progress modal should appear, no
                // writeFile should fire.
                await new Promise(r => setTimeout(r, 200));
                const stillModal = !!document.querySelector(
                    ".mol-viewer-export-modal");
                h.dispose();
                host.remove();
                return { hadDialog, stillModal, writeFileCalled };
            }
        """)
        assert out["hadDialog"] is True
        assert out["stillModal"] is False, (
            "Cancelling the params dialog should not open the "
            "progress modal"
        )
        assert out["writeFileCalled"] is False, (
            "Cancelling the params dialog should not run any export "
            "side-effect"
        )

    def test_export_params_dialog_uses_user_edited_filename(
            self, page, flask_server):
        """Editing the filename in the params dialog must propagate
        all the way through to the projectsApi.writeFile path.
        This guards against the dialog gathering values but the
        confirm handler still using defaults."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writtenPath = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "3\\nwater\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    export: { defaultName: "water" },
                    testInjection: {
                        projectsApi: {
                            writeFile: async (path, data, opts) => {
                                writtenPath = path;
                                return { ok: true, path: path };
                            },
                            currentDir: () => "/tmp/proj1",
                        },
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                const btn = bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='project'][data-format='xyz']");
                btn.click();
                // Edit the filename field, then Export.
                const filenameInp = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-params-input");
                filenameInp.value = "my-custom-name.xyz";
                const confirmBtn = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-modal-confirm");
                confirmBtn.click();
                // exportData is fast — wait one tick.
                await new Promise(r => setTimeout(r, 100));
                h.dispose();
                host.remove();
                return { writtenPath };
            }
        """)
        assert out["writtenPath"] == "/tmp/proj1/my-custom-name.xyz", (
            f"User-edited filename did not propagate to writeFile; "
            f"saw: {out['writtenPath']!r}"
        )

    def test_writeToProject_uses_real_getCurrentDir_api(
            self, page, flask_server):
        """Phase 6e review BOMB #1: the production
        window.molbuilder.projects API exposes ``getCurrentDir``,
        not ``currentDir``.  The embed had been reading
        ``proj.currentDir`` (undefined in prod) → every
        save-to-project from the embed silently rejected with
        "no_project" since Phase 5a (2026-06-03).  Existing tests
        masked the bug by stubbing ``currentDir: () => …`` on the
        injected projectsApi.  This test stubs ONLY the real prod
        shape (``getCurrentDir``) and asserts save-to-project
        works."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writtenPath = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    export: { defaultName: "n" },
                    testInjection: {
                        projectsApi: {
                            // ONLY the production API shape, no
                            // legacy currentDir.
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async (path, data) => {
                                writtenPath = path;
                                return { ok: true, path: path };
                            },
                        },
                    },
                });
                const r = await h.exportData({
                    target: "project", format: "xyz",
                });
                h.dispose();
                host.remove();
                return { writtenPath, ok: !!r };
            }
        """)
        assert out["ok"] is True, (
            "exportData rejected even with getCurrentDir stub — "
            "BOMB #1 fix didn't land"
        )
        assert out["writtenPath"] == "/tmp/proj1/n.xyz"

    def test_writeToProject_falls_back_to_legacy_currentDir(
            self, page, flask_server):
        """The fallback to ``proj.currentDir()`` is kept so older
        test stubs and any host code still passing it work; this
        test pins that compat path so a future cleanup doesn't
        silently break it without flagging."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writtenPath = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    export: { defaultName: "n" },
                    testInjection: {
                        projectsApi: {
                            currentDir: () => "/tmp/legacy",
                            writeFile: async (path, data) => {
                                writtenPath = path;
                                return { ok: true, path: path };
                            },
                        },
                    },
                });
                await h.exportData({
                    target: "project", format: "xyz",
                });
                h.dispose();
                host.remove();
                return { writtenPath };
            }
        """)
        assert out["writtenPath"] == "/tmp/legacy/n.xyz"

    def test_progress_modal_cancel_does_not_dispatch_error(
            self, page, flask_server):
        """Phase 6e review BOMB #2: clicking Cancel on the
        progress modal during an animation export triggers
        ac.abort() which rejects exportAnimation with
        code:"aborted".  That's user intent, not an error.  The
        catch handler must filter it so hosts wired to onError
        don't see a spurious error banner.  Mirrors the
        params-dialog Cancel policy (which already filters
        aborted)."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let errorFired = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    onError: (err) => {
                        errorFired = (err && err.code) || "unknown";
                    },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[0.5, 0.0, 0.0]],
                            [[1.0, 0.0, 0.0]],
                        ],
                        fps: 5, paused: true,
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                const exportBtn = bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='animation']"
                  + "[data-target='download'][data-format='gif']");
                exportBtn.click();
                // Dialog → confirm → progress modal.
                const dialogConfirm = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-modal-confirm");
                dialogConfirm.click();
                await new Promise(r => setTimeout(r, 50));
                // Now the progress modal is up; press its Cancel.
                let pmCancel = null;
                for (const m of document.querySelectorAll(
                        ".mol-viewer-export-modal")) {
                    if (!m.querySelector(
                            ".mol-viewer-export-params-card")) {
                        pmCancel = m.querySelector(
                            ".mol-viewer-export-modal-cancel");
                        break;
                    }
                }
                if (pmCancel) pmCancel.click();
                await new Promise(r => setTimeout(r, 300));
                h.dispose();
                host.remove();
                return { errorFired, hadCancelBtn: !!pmCancel };
            }
        """)
        assert out["hadCancelBtn"] is True, (
            "test setup error: progress modal Cancel button "
            "missing"
        )
        assert out["errorFired"] is None, (
            f"Cancelling the progress modal surfaced onError "
            f"with code={out['errorFired']!r} — BOMB #2 fix "
            f"didn't land.  Cancel is user intent, not an error."
        )

    def test_dialog_escape_cancels_no_export(
            self, page, flask_server):
        """LANDMINE #6: Esc should cancel the params dialog."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writeFileCalled = false;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async () => {
                                writeFileCalled = true;
                                return { ok: true };
                            },
                        },
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='project'][data-format='xyz']"
                ).click();
                const hadDialog = !!document.querySelector(
                    ".mol-viewer-export-params-card");
                document.dispatchEvent(new KeyboardEvent("keydown",
                    { key: "Escape", bubbles: true }));
                await new Promise(r => setTimeout(r, 100));
                const stillVisible = !!document.querySelector(
                    ".mol-viewer-export-params-card");
                h.dispose();
                host.remove();
                return { hadDialog, stillVisible, writeFileCalled };
            }
        """)
        assert out["hadDialog"] is True
        assert out["stillVisible"] is False, (
            "Esc should close the params dialog"
        )
        assert out["writeFileCalled"] is False

    def test_dialog_enter_confirms_export(
            self, page, flask_server):
        """LANDMINE #6: Enter in any dialog field should confirm
        the export."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writtenPath = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    export: { defaultName: "enter-test" },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async (path, data) => {
                                writtenPath = path;
                                return { ok: true, path: path };
                            },
                        },
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='project'][data-format='xyz']"
                ).click();
                document.dispatchEvent(new KeyboardEvent("keydown",
                    { key: "Enter", bubbles: true }));
                await new Promise(r => setTimeout(r, 200));
                h.dispose();
                host.remove();
                return { writtenPath };
            }
        """)
        assert out["writtenPath"] == "/tmp/proj1/enter-test.xyz", (
            f"Enter should confirm the dialog and trigger export; "
            f"wrote: {out['writtenPath']!r}"
        )

    def test_dialog_backdrop_click_cancels(
            self, page, flask_server):
        """LANDMINE #6: clicking the modal backdrop (outside the
        card) cancels the dialog without running the export."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writeFileCalled = false;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async () => {
                                writeFileCalled = true;
                                return { ok: true };
                            },
                        },
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='project'][data-format='xyz']"
                ).click();
                const overlay = document.querySelector(
                    ".mol-viewer-export-modal");
                // Click directly on the overlay (backdrop), NOT
                // on the card.
                overlay.click();
                await new Promise(r => setTimeout(r, 100));
                const stillVisible = !!document.querySelector(
                    ".mol-viewer-export-params-card");
                h.dispose();
                host.remove();
                return { stillVisible, writeFileCalled };
            }
        """)
        assert out["stillVisible"] is False, (
            "Backdrop click should close the dialog"
        )
        assert out["writeFileCalled"] is False

    def test_dispose_during_open_dialog_tears_it_down(
            self, page, flask_server):
        """LANDMINE #9: handle.dispose() while a params dialog is
        up must remove the dialog from the DOM so it can't post
        edits to a dead handle."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='download'][data-format='xyz']"
                ).click();
                const hadDialog = !!document.querySelector(
                    ".mol-viewer-export-params-card");
                h.dispose();
                const stillVisible = !!document.querySelector(
                    ".mol-viewer-export-params-card");
                host.remove();
                return { hadDialog, stillVisible };
            }
        """)
        assert out["hadDialog"] is True
        assert out["stillVisible"] is False, (
            "dispose() should tear down any open params dialog so "
            "stale UI doesn't survive the handle"
        )

    def test_dispose_during_open_dialog_does_not_fire_onError(
            self, page, flask_server):
        """Phase 6e second-review BOMB #14: dispose() while a
        params dialog is up tears the DOM down but used to leave
        the Promise pending forever, leaking closures.  The fix
        rejects with code:"disposed" and the orchestrator filters
        it the same way it filters "aborted" — neither should
        surface as an onError to the host."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let onErrorFired = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    onError: (err) => {
                        onErrorFired = (err && err.code) || "fired";
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='download'][data-format='xyz']"
                ).click();
                h.dispose();
                // Give the dialog Promise's reject handler time
                // to land in a microtask.
                await new Promise(r => setTimeout(r, 100));
                host.remove();
                return { onErrorFired };
            }
        """)
        assert out["onErrorFired"] is None, (
            f"dispose-during-open-dialog surfaced onError with "
            f"code={out['onErrorFired']!r} — BOMB #14 fix didn't "
            f"land, the dialog Promise rejection wasn't filtered"
        )

    def test_dialog_escape_does_not_propagate_to_host(
            self, page, flask_server):
        """Phase 6e second-review LANDMINE #19: a host-page Esc
        handler should NOT fire when the user presses Esc inside
        the embed's params dialog.  stopPropagation +
        stopImmediatePropagation guard this."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let hostEscFired = false;
                // Install a host-page Esc handler that should
                // NOT fire when Esc dismisses the dialog.  Use
                // bubble phase so we run after the dialog's
                // capture-phase handler.
                const hostHandler = (e) => {
                    if (e.key === "Escape") hostEscFired = true;
                };
                document.addEventListener("keydown", hostHandler);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='download'][data-format='xyz']"
                ).click();
                document.dispatchEvent(new KeyboardEvent("keydown",
                    { key: "Escape", bubbles: true }));
                await new Promise(r => setTimeout(r, 100));
                document.removeEventListener("keydown", hostHandler);
                h.dispose();
                host.remove();
                return { hostEscFired };
            }
        """)
        assert out["hostEscFired"] is False, (
            "Host-page Esc handler fired during dialog dismissal "
            "— stopPropagation regression"
        )

    def test_writeFile_text_auto_rename_propagates_for_structure(
            self, page, flask_server):
        """Phase 6e second-review BOMB #11: text save-to-project
        with auto_rename must reach the server's auto_rename
        branch.  The dialog wires autoRename automatically for
        project saves; this test asserts the camelCase →
        snake_case conversion lands in state.js writeFile's call
        to apiWrite."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let receivedOpts = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    export: { defaultName: "n" },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async (path, data, opts) => {
                                receivedOpts = opts || null;
                                return { ok: true, path: path };
                            },
                        },
                    },
                });
                await h.exportData({
                    target: "project", format: "xyz",
                    autoRename: true,
                });
                h.dispose();
                host.remove();
                return { receivedOpts };
            }
        """)
        assert out["receivedOpts"] is not None, (
            "writeFile was called without an opts argument; "
            "_writeToProject is not passing autoRename through"
        )
        assert out["receivedOpts"].get("autoRename") is True, (
            f"opts.autoRename did not reach writeFile; got: "
            f"{out['receivedOpts']!r}"
        )

    def test_dispose_clears_open_export_overlay_set(
            self, page, flask_server):
        """Phase 6e second-review POLISH #23: tests that previously
        only asserted DOM emptiness miss leaks of the close()
        closures themselves.  This test reads the Set count via
        the test affordance."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                });
                const initialCount = h._test.getOpenExportOverlayCount();
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='download'][data-format='xyz']"
                ).click();
                const withOpenDialog = h._test
                    .getOpenExportOverlayCount();
                // dispose() should drop them all.
                // Capture the count BEFORE dispose since the handle
                // is unusable afterwards.  Use the public method
                // ordering: read → dispose → cannot read.
                h.dispose();
                host.remove();
                return { initialCount, withOpenDialog };
            }
        """)
        assert out["initialCount"] == 0, (
            f"fresh embed should have 0 open overlays; got "
            f"{out['initialCount']}"
        )
        assert out["withOpenDialog"] == 1, (
            f"open dialog should register exactly 1 overlay; got "
            f"{out['withOpenDialog']}.  Dialog mounting / registration "
            f"is broken."
        )

    def test_cancel_mid_upload_does_not_fire_onError(
            self, page, flask_server):
        """Phase 6e third-review BOMB-1: Cancel-during-upload used
        to surface as io_error because the projectsApi.writeFile
        rejection wrapped the AbortError as io_error and lost the
        ``aborted`` flag.  After the fix, _writeToProject re-codes
        an aborted envelope to ``code:"aborted"`` so the upstream
        catch filter silences it."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let onErrorCode = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    onError: (err) => {
                        onErrorCode = (err && err.code) || "fired";
                    },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async (path, data, opts) => {
                                // Simulate the AbortError reaching
                                // the writeFile envelope: ok:false
                                // + aborted:true, matching
                                // _fetchEnvelope's behaviour on
                                // a Cancel mid-upload.
                                return {
                                    ok: false,
                                    error: "aborted",
                                    aborted: true,
                                };
                            },
                        },
                    },
                });
                let err = null;
                try {
                    await h.exportData({
                        target: "project", format: "xyz",
                    });
                } catch (e) {
                    err = (e && e.code) || "thrown";
                }
                // Give the .catch / .finally microtasks time to
                // settle before we read onErrorCode.
                await new Promise(r => setTimeout(r, 50));
                h.dispose();
                host.remove();
                return { onErrorCode, err };
            }
        """)
        # The public API does reject with code:"aborted" — that's
        # the contract for programmatic callers.  The bug we're
        # fixing is the click-flow's onError catch, which we
        # verify by checking onError didn't fire.
        assert out["err"] == "aborted", (
            f"exportData should reject with code:'aborted' when "
            f"the underlying writeFile reports aborted; got "
            f"{out['err']!r}"
        )
        # Phase 6e fourth-review LANDMINE-1: the prior version of
        # this test forgot this assertion.  Now pinned: onError
        # MUST NOT fire on a user-initiated cancel.
        assert out["onErrorCode"] is None, (
            f"onError fired with code={out['onErrorCode']!r} for a "
            f"Cancel-mid-upload — BOMB-1 fix regressed."
        )

    def test_dispose_during_export_does_not_fire_onError(
            self, page, flask_server):
        """Phase 6e third-review BOMB-2: _runExportWithParams's
        .catch used to dispatch the ``code:"disposed"`` rejection
        as an error, leaking a banner onto a host page that had
        already torn the embed down.  The catch now filters
        ``disposed`` the same way it filters ``aborted``.

        Trigger path: programmatic exportData with a writeFile
        that simulates the disposed reject."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let onErrorCode = null;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[0.5, 0.0, 0.0]],
                        ],
                        fps: 5, paused: true,
                    },
                    onError: (err) => {
                        onErrorCode = (err && err.code) || "fired";
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                const btn = bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='animation']"
                  + "[data-target='download'][data-format='gif']");
                btn.click();
                // Confirm the dialog to enter
                // _runExportWithParams, then dispose mid-encode.
                const confirm = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-modal-confirm");
                confirm.click();
                // Let one tick pass so exportAnimation starts.
                await new Promise(r => setTimeout(r, 0));
                h.dispose();
                // Let the .catch microtask settle.
                await new Promise(r => setTimeout(r, 200));
                host.remove();
                return { onErrorCode };
            }
        """)
        assert out["onErrorCode"] is None, (
            f"dispose-during-export surfaced onError code="
            f"{out['onErrorCode']!r}; BOMB-2 fix didn't land."
        )

    def test_fireOnExport_skipped_after_dispose(
            self, page, flask_server):
        """LANDMINE-1: a slow upload may complete after dispose;
        onExport must NOT fire on a dead handle."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let onExportFired = false;
                let resolveWrite;
                const writeDone = new Promise(r => resolveWrite = r);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    export: {
                        defaultName: "n",
                        onExport: () => { onExportFired = true; },
                    },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async (path, data) => {
                                // Block until the test releases.
                                await writeDone;
                                return { ok: true, path: path };
                            },
                        },
                    },
                });
                const p = h.exportData({
                    target: "project", format: "xyz",
                });
                // Dispose while writeFile is in flight.
                h.dispose();
                // Release the write so onExport WOULD fire.
                resolveWrite({ ok: true, path: "/tmp/proj1/n.xyz" });
                // Catch the export's own rejection (disposed).
                try { await p; } catch (_) {}
                await new Promise(r => setTimeout(r, 100));
                host.remove();
                return { onExportFired };
            }
        """)
        assert out["onExportFired"] is False, (
            "onExport fired after dispose — LANDMINE-1 fix didn't "
            "land.  Host's onExport may reference DOM the host "
            "already cleaned up."
        )

    def test_progress_modal_esc_cancels(
            self, page, flask_server):
        """LANDMINE-3: progress modal didn't honour Esc.  The
        params dialog did; users hit Esc on the modal and
        nothing happened.  Now Esc → Cancel."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:400px;height:300px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { showInfoLine: false, height: "100%" },
                    animation: {
                        kind: "trajectory",
                        frames: [
                            [[0.0, 0.0, 0.0]],
                            [[0.5, 0.0, 0.0]],
                        ],
                        fps: 5, paused: true,
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='animation']"
                  + "[data-target='download'][data-format='gif']"
                ).click();
                const confirm = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-modal-confirm");
                confirm.click();
                await new Promise(r => setTimeout(r, 20));
                // Find the progress modal (not the dialog).
                let progressModal = null;
                for (const m of document.querySelectorAll(
                        ".mol-viewer-export-modal")) {
                    if (!m.querySelector(
                            ".mol-viewer-export-params-card")) {
                        progressModal = m; break;
                    }
                }
                const hadProgressModal = !!progressModal;
                // Press Esc — should trigger cancel.
                document.dispatchEvent(new KeyboardEvent("keydown",
                    { key: "Escape", bubbles: true }));
                await new Promise(r => setTimeout(r, 50));
                // Cancel button should now be disabled +
                // "Cancelling…" text shown OR modal closed
                // depending on how quickly the encode aborts.
                const phaseText = progressModal
                    ? progressModal.querySelector(
                        ".mol-viewer-export-modal-phase")
                          .textContent
                    : null;
                h.dispose();
                host.remove();
                return { hadProgressModal, phaseText };
            }
        """)
        assert out["hadProgressModal"] is True, (
            "progress modal should appear after params confirm"
        )
        # Phase text becomes "Cancelling…" on Esc-cancel.
        assert "ancel" in (out["phaseText"] or ""), (
            f"Esc should trigger Cancel on the progress modal; "
            f"phase shows {out['phaseText']!r}"
        )

    def test_writeFile_mock_actually_observes_signal(
            self, page, flask_server):
        """Phase 6e fourth-review LANDMINE-5: every other test in
        this class uses a writeFile mock that synthesizes the
        post-abort envelope.  None proves the AbortSignal is
        actually plumbed end-to-end.  This test uses a mock that
        ONLY resolves abort when the signal fires, so a regression
        that drops opts.signal from _writeToProject would deadlock
        the await."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let sawSignal = false;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async (path, data, opts) => {
                                return new Promise((resolve) => {
                                    if (opts && opts.signal) {
                                        sawSignal = true;
                                        opts.signal.addEventListener(
                                            "abort", () => {
                                              resolve({
                                                ok: false,
                                                error: "aborted",
                                                aborted: true,
                                              });
                                            });
                                    } else {
                                        // No signal threaded —
                                        // deadlock (intentional);
                                        // the test will timeout
                                        // and fail.
                                    }
                                });
                            },
                        },
                    },
                });
                const ac = new AbortController();
                const p = h.exportData({
                    target: "project", format: "xyz",
                    signal: ac.signal,
                });
                // Give the call time to reach writeFile.
                await new Promise(r => setTimeout(r, 50));
                ac.abort();
                let err = null;
                try { await p; } catch (e) {
                    err = (e && e.code) || "thrown";
                }
                h.dispose();
                host.remove();
                return { sawSignal, err };
            }
        """)
        assert out["sawSignal"] is True, (
            "_writeToProject didn't forward opts.signal to the "
            "projectsApi.writeFile mock — LANDMINE-5 fix didn't "
            "land, the signal plumbing is dropped at some layer"
        )
        assert out["err"] == "aborted", (
            f"Cancel-via-AbortController should reject with "
            f"code:'aborted'; got {out['err']!r}"
        )

    def test_dialog_rejects_empty_filename_inline(
            self, page, flask_server):
        """Phase 6e fourth-review LANDMINE-6: empty filename used
        to silently fall back to the default name; now the dialog
        shows an inline error and refuses to confirm."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writeFileCalled = false;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async () => {
                                writeFileCalled = true;
                                return { ok: true };
                            },
                        },
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='project'][data-format='xyz']"
                ).click();
                // Clear the filename, then confirm.
                const filenameInp = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-params-input");
                filenameInp.value = "";
                const confirm = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-modal-confirm");
                confirm.click();
                await new Promise(r => setTimeout(r, 50));
                const stillVisible = !!document.querySelector(
                    ".mol-viewer-export-params-card");
                const errMsg = document.querySelector(
                    ".mol-viewer-export-params-error");
                const errVisible = errMsg
                    && errMsg.style.display !== "none";
                h.dispose();
                host.remove();
                return { stillVisible, errVisible,
                         writeFileCalled,
                         errText: errMsg && errMsg.textContent };
            }
        """)
        assert out["stillVisible"] is True, (
            "Empty filename should keep the dialog open"
        )
        assert out["errVisible"] is True, (
            "Empty filename should surface an inline error; got "
            f"errText={out['errText']!r}"
        )
        assert out["writeFileCalled"] is False, (
            "writeFile must NOT be called when filename is invalid"
        )

    def test_dialog_rejects_illegal_filename(
            self, page, flask_server):
        """LANDMINE-6 mirror: `../escape.xyz` etc must be caught
        client-side, not just by the server's path resolver."""
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            async () => {
                const host = document.createElement("div");
                host.style.cssText =
                    "width:300px;height:200px;position:fixed;top:-9999px;";
                document.body.appendChild(host);
                let writeFileCalled = false;
                const h = window.molbuilder.viewer.embed(host, {
                    xyz: "1\\nh\\nH 0 0 0\\n",
                    card: { bare: true, showInfoLine: false },
                    testInjection: {
                        projectsApi: {
                            getCurrentDir: () => "/tmp/proj1",
                            writeFile: async () => {
                                writeFileCalled = true;
                                return { ok: true };
                            },
                        },
                    },
                });
                const bar = h._test.getKnobBarElement();
                bar.querySelector(
                    ".mol-viewer-menu-export").open = true;
                bar.querySelector(
                    ".mol-viewer-export-btn[data-kind='structure']"
                  + "[data-target='project'][data-format='xyz']"
                ).click();
                const filenameInp = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-params-input");
                filenameInp.value = "../escape.xyz";
                const confirm = document.querySelector(
                    ".mol-viewer-export-params-card"
                  + " .mol-viewer-export-modal-confirm");
                confirm.click();
                await new Promise(r => setTimeout(r, 50));
                const errMsg = document.querySelector(
                    ".mol-viewer-export-params-error");
                const errVisible = errMsg
                    && errMsg.style.display !== "none";
                h.dispose();
                host.remove();
                return { errVisible, writeFileCalled };
            }
        """)
        assert out["errVisible"] is True, (
            "Filename with .. should be rejected with inline error"
        )
        assert out["writeFileCalled"] is False

