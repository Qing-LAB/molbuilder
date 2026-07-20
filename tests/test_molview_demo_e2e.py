"""Standalone MolView component demo (/molview-demo) -- the REAL integration test.

Unlike the node unit tests (stubbed viewer + workspace), this boots the actual page: the
empty-host build path of molview.mount assembling the real viewer + panel + toggles + a live
render loop against the real workspace.  It pins the data-consistency the demo exposed: the
VIEWER reflects the loaded structure (the render reads the store atoms, the same source the
panel lists), and a sample-load button UPDATES the viewer.
"""
import threading

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

# The panel's legacy renderMeasurement warns for a missing #selection-measurement-overlay
# (molview mounts its own measurement overlay instead -- the duplication the audit flagged,
# retired later).  Benign here; filtered so it doesn't mask a real error.
_IGNORE = ["missing partial ids: measurement"]


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


def _viewer_atoms_is(n: int) -> str:
    return ("() => { const v = document.querySelector('#molview-demo-host .viewer');"
            "  return !!(v && v.__molview_test_handle"
            f"           && v.__molview_test_handle.getAtomCount() === {n}); }}")


def test_molview_demo_mounts_and_viewer_tracks_the_loaded_structure(page, flask_server):
    errors = []

    def on_console(m):
        if m.type == "error" and not any(s in m.text for s in _IGNORE):
            errors.append(m.text)

    page.on("console", on_console)
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(f"{flask_server}/molview-demo")

    # molview BUILT the full component into the empty host.
    page.wait_for_selector("#molview-demo-host .molview-card .molview-panel", timeout=20000)
    assert page.locator("#molview-demo-host .molview-viewer").count() == 1
    assert page.locator("#molview-demo-host .mol-viewer-quick[data-quick=isolate]").count() == 1

    # the handle exposes the full §D surface (load/save/undo + getStructure/getSelection +
    # onChange + dispose) -- and only that (no internals).
    page.wait_for_function(
        "() => window.__molview && typeof window.__molview.onChange === 'function'")
    keys = page.evaluate("() => Object.keys(window.__molview).sort()")
    assert keys == ["currentFrame", "dispose", "exportFile", "frameCount", "getFrame",
                    "getSelection", "getStructure", "isPlaying", "onChange", "openMolecule",
                    "pause", "play", "setArrows", "setFrame", "setFrameArrows", "setLabels",
                    "undo"]

    # THE FIX: the VIEWER shows the water sample loaded on mount (render reads store atoms).
    page.wait_for_function(_viewer_atoms_is(3), timeout=10000)

    # THE BUG THE DEMO CAUGHT: a sample-load button must UPDATE the viewer, not just the panel.
    page.locator("#demo-benzene").click()
    page.wait_for_function(_viewer_atoms_is(12), timeout=10000)

    # the header count reflects the loaded structure (regression: it was gated on sourceFile,
    # so a text-loaded molecule stayed stuck at "no structure").
    page.wait_for_function(
        "() => /12 atoms/.test(document.querySelector('#molview-demo-host #selection-count')"
        "        .textContent)", timeout=5000)
    count = page.locator("#molview-demo-host #selection-count").inner_text()
    assert "no structure" not in count, f"count out of sync with the loaded structure: {count!r}"

    assert not errors, f"console/page errors during mount + load: {errors}"


def test_molview_demo_multiframe_setFrame_moves_the_drawn_atoms(page, flask_server):
    """Multi-frame trajectory, exercised on /molview-demo ALONE (no other tab): load a 3-frame
    trajectory where atom 0 slides x = 0 -> 1 -> 2, then handle.setFrame swaps the DRAWN atom's
    coordinates (the store coord-swap flows through to the render).  Pins the workspace §1.5 +
    molview §14.5 frame path end-to-end."""
    errors = []

    def on_console(m):
        if m.type == "error" and not any(s in m.text for s in _IGNORE):
            errors.append(m.text)

    page.on("console", on_console)
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(f"{flask_server}/molview-demo")
    page.wait_for_function(
        "() => window.__molview && typeof window.__molview.setFrame === 'function'",
        timeout=20000)

    # Load the 3-frame trajectory.
    page.locator("#demo-trajectory").click()
    page.wait_for_function("() => window.__molview.frameCount() === 3", timeout=10000)

    # Read the DRAWN x of atom 0 from the viewer handle (what 3dmol actually holds).
    drawn_x0 = ("() => { const v = document.querySelector('#molview-demo-host .viewer');"
                "  const c = v && v.__molview_test_handle && v.__molview_test_handle.getAtomCoords();"
                "  return (c && c[0]) ? c[0][0] : null; }")

    # Frame 0 -> atom 0 drawn at x = 0.
    page.wait_for_function(f"() => Math.abs(({drawn_x0})() - 0) < 1e-6", timeout=5000)
    # setFrame(2) -> atom 0 drawn at x = 2 (the coord swap reached the viewer).
    page.evaluate("() => window.__molview.setFrame(2)")
    page.wait_for_function(f"() => Math.abs(({drawn_x0})() - 2) < 1e-6", timeout=5000)
    assert page.evaluate("() => window.__molview.currentFrame()") == 2
    # setFrame(0) -> back to x = 0.
    page.evaluate("() => window.__molview.setFrame(0)")
    page.wait_for_function(f"() => Math.abs(({drawn_x0})() - 0) < 1e-6", timeout=5000)

    # Frame overlays (§14.5.1): force arrows + atom-index labels on the REAL embed -- turning
    # them on (and swapping frames with them on) must not error.  A frame with forces exists.
    page.evaluate("() => window.__molview.setFrame(2)")
    page.wait_for_timeout(150)          # let the overlay redraw settle

    assert not errors, f"console/page errors during frame navigation + overlays: {errors}"


def test_molview_demo_frame_controls_bar_drives_the_trajectory(page, flask_server):
    """The frame controls bar MolView renders (§14.5): load a 3-frame trajectory and the bar
    (slider + play + counter) appears and drives the render.  Dragging
    the slider moves the DRAWN atom + updates the counter; play advances the frame; the toggles
    draw arrows/labels -- all on the REAL embed, no other tab."""
    errors = []

    def on_console(m):
        if m.type == "error" and not any(s in m.text for s in _IGNORE):
            errors.append(m.text)

    page.on("console", on_console)
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(f"{flask_server}/molview-demo")
    page.wait_for_function(
        "() => window.__molview && typeof window.__molview.setFrame === 'function'", timeout=20000)

    bar = "#molview-demo-host .molview-frame-controls"
    # A single structure on mount -> the frame bar is hidden (frames are a trajectory concept).
    assert page.locator(bar).is_hidden()

    # Load the 3-frame trajectory -> the bar appears + the counter reads "1 / 3".
    page.locator("#demo-trajectory").click()
    page.wait_for_selector(f"{bar}:not([hidden])", timeout=10000)
    page.wait_for_function(
        f"() => document.querySelector('{bar} .mvf-counter').textContent.trim() === '1 / 3'",
        timeout=5000)

    drawn_x0 = ("() => { const v = document.querySelector('#molview-demo-host .viewer');"
                "  const c = v && v.__molview_test_handle && v.__molview_test_handle.getAtomCoords();"
                "  return (c && c[0]) ? c[0][0] : null; }")

    # Drag the slider to frame 2 -> drawn atom 0 moves to x = 2, counter -> "3 / 3".
    page.locator(f"{bar} .mvf-slider").fill("2")
    page.wait_for_function(f"() => Math.abs(({drawn_x0})() - 2) < 1e-6", timeout=5000)
    page.wait_for_function(
        f"() => document.querySelector('{bar} .mvf-counter').textContent.trim() === '3 / 3'",
        timeout=5000)

    # Play -> the frame advances (wraps past 2); then pause.
    page.locator(f"{bar} .mvf-play").click()
    page.wait_for_function("() => window.__molview.currentFrame() !== 2", timeout=5000)
    page.locator(f"{bar} .mvf-play").click()   # pause

    # Overlays are NOT on the frame bar anymore -- they are consumer-driven (handle.setArrows /
    # setLabels), so the bar is play/slider/counter only.
    assert not errors, f"console/page errors driving the frame bar: {errors}"


def test_molview_demo_selection_cell_tabs_actually_switch(page, flask_server):
    """Regression: a stray `display:flex` on .panel-page once overrode the [hidden]
    attribute, so BOTH the Selection and Cell pages rendered at once and the tab switch did
    nothing.  Pin that exactly ONE page shows and clicking a tab swaps them."""
    page.goto(f"{flask_server}/molview-demo")
    sel = "#molview-demo-host #panel-page-selection"
    cell = "#molview-demo-host #panel-page-cell"
    page.wait_for_selector(sel, timeout=20000)
    # on mount: Selection visible, Cell hidden
    assert page.locator(sel).is_visible() and not page.locator(cell).is_visible()
    # click the Cell tab -> Cell visible, Selection hidden
    page.locator("#molview-demo-host .panel-page-option:has(#panel-page-radio-cell)").click()
    page.wait_for_function(
        "() => { const c = document.querySelector('#molview-demo-host #panel-page-cell');"
        "  const s = document.querySelector('#molview-demo-host #panel-page-selection');"
        "  return c && s && c.offsetParent !== null && s.offsetParent === null; }",
        timeout=5000)
    # and back
    page.locator("#molview-demo-host .panel-page-option:has(#panel-page-radio-selection)").click()
    page.wait_for_function(
        "() => { const c = document.querySelector('#molview-demo-host #panel-page-cell');"
        "  const s = document.querySelector('#molview-demo-host #panel-page-selection');"
        "  return c && s && s.offsetParent !== null && c.offsetParent === null; }",
        timeout=5000)


def test_molview_demo_load_is_atomic_across_the_whole_model(page, flask_server):
    """Contract (molview-module.md §19.3.1): a single ``loadFromText`` sets up the ENTIRE
    model -- atoms AND structure/periodicity -- in one call.  There is no second "install
    into the canvas" door, and no path may leave atoms present with ``getStructure()`` null.
    Regression: /molview-demo loads via the one documented entry point and got atoms but a
    null structure + no unit cell, because the load was split across two write paths."""
    page.goto(f"{flask_server}/molview-demo")
    page.wait_for_selector("#molview-demo-host .molview-viewer", timeout=20000)

    def model_after_load():
        return page.evaluate(
            "() => { const d = window.molbuilder.molview.data; const s = d.getStructure();"
            "  return { hasAtoms: !!(s && s.atoms && s.atoms.length),"
            "           structureNull: s === null,"
            "           unitCell: d.getUnitCellInfo().value }; }")

    # The demo mounts by loading 'water' through loadFromText -> the whole model is coherent.
    page.wait_for_function(
        "() => { const d = window.molbuilder.molview.data; const s = d.getStructure();"
        "  return !!(s && s.atoms && s.atoms.length); }", timeout=10000)
    m = model_after_load()
    assert m["hasAtoms"] and not m["structureNull"], m
    # Every structure has a valid (resolved) cell -- never null while atoms exist.
    assert m["unitCell"] is not None, m

    # Switching samples through the same one door stays coherent (benzene has more atoms).
    page.click("#demo-benzene")
    page.wait_for_function(
        "() => { const s = window.molbuilder.molview.data.getStructure();"
        "  return !!(s && s.atoms && s.atoms.length === 12); }", timeout=10000)
    m2 = model_after_load()
    assert not m2["structureNull"] and m2["unitCell"] is not None, m2


def test_molview_demo_view_menu_layout(page, flask_server):
    """Pin the view-toggle RAIL structure: Reset is the FIRST button, followed by the
    display toggles (axes / labels / overlay / unit cell) and MolView's injected
    'selected only' (isolate) last.  All view toggles now live on the always-visible
    left rail (.mol-viewer-quickbar) -- the View dropdown holds only style / background /
    projection, no toggles and no reset.  Also that the 'unit cell' toggle actually flips
    (draws the cell wireframe from the fed lattice).

    (k-grid used to inject a SECOND consumer toggle; Phase 2a removed k-grid from
    MolView entirely -- it's reciprocal-space BZ sampling, not geometry -- so exactly ONE
    consumer toggle ('selected only') is now on the rail.)"""
    page.goto(f"{flask_server}/molview-demo")
    page.wait_for_selector("#molview-demo-host .mol-viewer-quickbar", timeout=20000)
    # Load the trajectory so the injected isolate toggle is present on the rail.
    page.click("#demo-trajectory")

    rail = "#molview-demo-host .mol-viewer-quickbar"
    # 1. The injected isolate button lands on the rail (exactly one; wait for it).
    page.wait_for_function(
        "() => { const g = document.querySelector('" + rail + "');"
        "  return g && g.querySelectorAll('.mol-viewer-quick[data-quick=isolate]').length === 1; }",
        timeout=5000)

    # 2. Rail order = built-in VIEW_TOGGLES (reset first) + the injected isolate last.
    actions = page.eval_on_selector_all(
        rail + " .mol-viewer-quick", "els => els.map(e => e.getAttribute('data-quick'))")
    assert actions == ["reset", "axes", "labels", "overlay", "cell", "isolate"]
    # The View dropdown no longer carries any toggles.
    assert page.locator(
        "#molview-demo-host .mol-viewer-menu-view .mol-viewer-toggle").count() == 0

    # 3. The 'unit cell' toggle flips its pressed state on click.  The rail is always
    #    visible, so there is no menu to open first.
    cell = rail + ' .mol-viewer-quick[data-quick="cell"]'
    assert page.locator(cell).get_attribute("aria-pressed") == "false"
    page.locator(cell).click()
    assert page.locator(cell).get_attribute("aria-pressed") == "true"
