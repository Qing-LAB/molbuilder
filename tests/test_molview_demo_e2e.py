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
    assert page.locator("#molview-demo-host .viewer-toggles .vc-isolate").count() == 1
    assert page.locator("#molview-demo-host .viewer-toggles .vc-kgrid").count() == 1

    # the handle exposes the full §D surface (load/save/undo + getStructure/getSelection +
    # onChange + dispose) -- and only that (no internals).
    page.wait_for_function(
        "() => window.__molview && typeof window.__molview.onChange === 'function'")
    keys = page.evaluate("() => Object.keys(window.__molview).sort()")
    assert keys == ["dispose", "getSelection", "getStructure",
                    "load", "onChange", "save", "undo"]

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
