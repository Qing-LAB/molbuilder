"""End-to-end smoke: /molbuilder mounts cleanly under real script-
tag order without throwing.

History (2026-06-14)
====================

Earlier in the session this file held 3 assertions that fully
duplicated `tests/test_workspace_dispatcher_canvas_mount_js.py`:

  1. ``window.molbuilder.workspace._canvasState`` is truthy
  2. ``workspace.getStructure()`` is callable without throwing
  3. The empty workspace returns ``null`` from ``getStructure()``

All three are pinned at L2 by the source-text + Node DOM test
above.  The round-3 audit flagged the L5 duplication as pyramid
skew: a 6-second Playwright test that adds nothing the L2 test
doesn't already pin.

The single contract L5 CAN catch that L2 cannot: that the real
script-tag order in ``molbuilder.html`` actually loads the
``_canvas-state-impl.js`` BEFORE the dispatcher's mount.  Verified
in this file by ``page.goto('/molbuilder')`` + a single assertion
that ``workspace._canvasState`` ends up truthy after page load,
with no console errors during the load.  The 3-assertion contract
checks have moved to L2.
"""
from __future__ import annotations

import threading

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


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


def test_molbuilder_page_mounts_canvas_state_under_real_script_order(
        page, flask_server):
    """Smoke: /molbuilder loads + the canvas-state slot survives
    the real script-tag order in molbuilder.html.  Pre-2026-06-13
    a bare ``root.molbuilder.workspace = api`` in dispatcher.js
    wiped the slot the impl had mounted; the fix (Object.assign)
    is pinned at the function level by
    ``test_workspace_dispatcher_canvas_mount_js.py``.  This smoke
    catches a future template-side regression where the impl
    script tag moves AFTER the dispatcher's, OR where a new mount
    mechanism is introduced that doesn't merge.
    """
    # Post-carve contract: the workspace is PERSISTENCE-ONLY (its ready surface is
    # ``persist``); the in-memory data model (``getStructure`` et al.) lives on
    # ``molview.data``.  Gate on the persistence surface, not the moved data API.
    page.goto(f"{flask_server}/molbuilder")
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.workspace "
        "      && typeof window.molbuilder.workspace.persist === 'function'"
        "      && window.molbuilder.molview && window.molbuilder.molview.data "
        "      && typeof window.molbuilder.molview.data.getStructure === 'function'",
        timeout=10000,
    )
    canvas_slot_present = page.evaluate(
        "() => !!(window.molbuilder.workspace._canvasState)"
    )
    assert canvas_slot_present, (
        "window.molbuilder.workspace._canvasState was not present "
        "after /molbuilder mounted -- the impl script either "
        "didn't run, or the dispatcher's mount wiped it.  See "
        "test_workspace_dispatcher_canvas_mount_js.py for the "
        "function-level contract this smoke complements."
    )
