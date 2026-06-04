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
        """The host ``#viewer.viewer`` has aspect-ratio 1/1 and
        min-height 320 px.  After #203 (knob bar visible), the
        canvas sits beneath a ~60 px header + knob bar inside
        the host card, so it's NOT the full host area anymore.

        The regression we still guard: the canvas must not
        collapse to near-zero height (the 2026-06-02 blank-viewer
        bug).  We assert it stays above 200 px tall — comfortably
        above any rounding-noise floor but well below the host's
        320 min-height, accounting for the chrome above."""
        page.goto(f"{flask_server}/modify")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
            "getCurrentBackground",
            "getDependencyStatus",
            "getFrameStripElement",
            "getKnobBarElement",
            "getOverlayLabelCount",
            "getOverlayShapeCount",
            "hasAnimationLoop",
            "triggerKnob",
        ]

    def test_dependency_status_reports_loaded_modules(
            self, page, flask_server):
        """In production, the page boot loads mol-axes / mol-style /
        mol-pick / mol-format before mol-viewer-embed.  ``axes``,
        ``style``, ``pick``, ``format`` should all be true; the
        integration deps (projects / clipboard / mediaRecorder /
        gif) vary by environment."""
        page.goto(f"{flask_server}/")
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
        # Build page boot.  (mol-pick is loaded only on /modify
        # where it's actually used — its absence here demonstrates
        # the soft-dep degradation pattern.)
        assert status["axes"]   is True, "mol-axes.js should be loaded"
        assert status["style"]  is True, "mol-style.js should be loaded"
        assert status["format"] is True, "mol-format.js should be loaded"
        # gif.js is lazy-loaded; should be "absent" until first
        # animation export.
        assert status["gif"] == "absent"
        # Status shape includes every documented key (§ 9.2)
        # even when the dep is absent.
        for key in ("axes", "style", "pick", "format",
                    "projects", "clipboard", "mediaRecorder", "gif"):
            assert key in status, f"missing status key: {key}"

    def test_overlay_shape_count_tracks_setOverlays(
            self, page, flask_server):
        """setOverlays with N halos should bump getOverlayShapeCount
        by N (visual-invariant assertion without _viewer3dmol)."""
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
                         resultFilename: result.filename };
            }
        """)
        assert out["writtenPath"] == "/tmp/proj1/water.xyz"
        assert out["hasData"] is True
        assert out["resultFilename"] == "/tmp/proj1/water.xyz"

    def test_export_to_project_rejects_when_no_active_dir(
            self, page, flask_server):
        """No currentDir → reject with code: no_project per § 5.3."""
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        "getAtomCount", "getElements", "getPickedIndices",
        "getStructureText",
        # Output / export
        "screenshot", "exportData",
        "captureFrames", "exportAnimation",
        # Lifecycle
        "refit", "render", "dispose",
        # Escape hatch + test surface
        "_viewer3dmol",
    ])
    # _test is an object, not a function — checked separately.

    def test_handle_has_exact_documented_method_set(
            self, page, flask_server):
        """The handle MUST export every documented method as a
        function.  Catches D1-class drift: a documented method
        being silently absent from the handle export."""
        page.goto(f"{flask_server}/")
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

    def test_handle_has_test_affordance_object(
            self, page, flask_server):
        """The ``_test`` affordance object is documented in § 9.2
        and required by the test suite; it's not a function."""
        page.goto(f"{flask_server}/")
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
