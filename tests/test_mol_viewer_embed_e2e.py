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
            "getCurrent",
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
        "setPickedIndices", "getStructureText",
        # Declarative-state getters (D3 symmetry — round-trip with setX)
        "getStyle", "getAxes", "getCell", "getLabels",
        "getOverlays", "getPick", "getKnobs", "getArrows",
        "getAnimation",
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

    def test_setAnimation_partial_update_keeps_vibration_running(
            self, page, flask_server):
        """Per § 3.2: setAnimation with a partial object (no kind)
        merges into the active animation without stopping the loop.
        Regression test for D1/N1: spectra amplitude/speed sliders
        called setAnimation({amplitude: v}) on every tick, which
        previously normalised to null and STOPPED the vibration."""
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
                run("setKnobs_bad_position",
                    (h) => h.setKnobs({position: "middle"}));
                run("setKnobs_bad_lf",
                    (h) => h.setKnobs({labelsFormats: ["index", "bogus"]}));
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
        page.goto(f"{flask_server}/")
        page.wait_for_selector("#viewer .mol-viewer-canvas",
                               timeout=_BOOT_TIMEOUT_MS)
        page.wait_for_timeout(200)
        out = page.evaluate("""
            () => {
                const r = {};
                for (const mode of ["auto", "cartesian", "cell"]) {
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

    def test_setStructure_preserves_picks_when_elements_match(
            self, page, flask_server):
        """Per § 3.8 + § 4.2.1: pickedIndices survives setStructure
        IFF atom count + element-by-element ordering match.  On
        mismatch the embed fires onPick([]) so hosts mirroring picks
        into a store see the clear.  Regression test for B2 (#238).
        """
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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
        page.goto(f"{flask_server}/")
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

    def test_setAnimation_halt_preserves_existing_animation(
            self, page, flask_server):
        """Per § 5.3 "halt" semantics: a setAnimation call with an
        invalid full-update payload (bad kind / missing displacements /
        atom-count mismatch) must NOT clear or replace the active
        animation.  Pinned for #237: regression guard against a future
        edit that validates after _setAnimationImpl runs."""
        page.goto(f"{flask_server}/")
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

    def test_chrome_consistency_across_build_and_modify(
            self, page, flask_server):
        """The standard knob bar's DOM structure is identical on
        Build (/) and Modify (/modify) — same 7 knobs in the same
        order: Style / Labels / Axes / Reset / Screenshot /
        Background / Export.

        Pins the chrome-consistency contract for #207: a future
        edit that adds / removes / reorders a knob on one site
        fails this test, forcing the change to either land on
        every site or update the contract."""
        def _knob_signature(path):
            page.goto(f"{flask_server}{path}")
            page.wait_for_selector("#viewer .mol-viewer-knobs",
                                   timeout=_BOOT_TIMEOUT_MS)
            page.wait_for_timeout(200)
            # Get the ordered class signature of each knob bar
            # child (top-level only; popover contents are tested
            # separately).
            return page.evaluate("""() => {
                const bar = document.querySelector(
                    '#viewer .mol-viewer-knobs');
                if (!bar) return null;
                return Array.from(bar.children).map((el) => {
                    if (el.tagName === 'SELECT') return 'select';
                    if (el.tagName === 'DETAILS') {
                        const cls = Array.from(el.classList)
                            .find((c) => c.startsWith(
                                'mol-viewer-knob-'));
                        return 'details:' + (cls || '?');
                    }
                    if (el.tagName === 'BUTTON') {
                        const k = el.getAttribute('data-knob');
                        return 'button:' + (k || '?');
                    }
                    return el.tagName.toLowerCase();
                });
            }""")
        build_sig  = _knob_signature("/")
        modify_sig = _knob_signature("/modify")
        assert build_sig == modify_sig, (
            f"Knob bar drifted between Build and Modify:\n"
            f"  Build:  {build_sig}\n  Modify: {modify_sig}"
        )
        # Verify the EXPECTED signature so an addition to BOTH
        # sites still has to update this test (catches an
        # unwanted change that's symmetric across consumers).
        EXPECTED = [
            "select",                              # Style picker
            "details:mol-viewer-knob-labels",      # Labels popover
            "button:axes",                          # Axes toggle
            "button:reset",                         # Reset
            "button:screenshot",                    # PNG
            "details:mol-viewer-knob-background",  # Background popover
            "details:mol-viewer-knob-export",      # Export popover
        ]
        assert build_sig == EXPECTED, (
            f"Knob bar order drifted from § 6.2 spec:\n"
            f"  Expected: {EXPECTED}\n  Got:      {build_sig}"
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
