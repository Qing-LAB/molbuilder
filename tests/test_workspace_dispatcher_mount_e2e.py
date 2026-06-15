"""End-to-end: the workspace dispatcher MUST preserve sibling slots
on ``window.molbuilder.workspace`` when it mounts.

2026-06-13 BLOCKER (3 commits to fully ship):
=============================================

Pre-fix ``lib/workspace/dispatcher.js`` ended with the bare
assignment

    root.molbuilder.workspace = api;

That replaced the entire ``workspace`` object with the dispatcher's
``api`` surface, WIPING OUT the ``workspace._canvasState`` slot
that ``lib/workspace/_canvas-state-impl.js`` had already mounted.
The ``_canvas()`` helper that every store-accessing function uses
returned ``null`` → every later ``getAtoms()``, ``setStructure()``,
etc., threw ``workspace dispatcher: canvas store not available on
this page``.

The fix at dispatcher.js:1101 is

    root.molbuilder.workspace = Object.assign(
        root.molbuilder.workspace || {}, api);

preserving prior slots while merging in the dispatcher's API.

What the L2 source-text guard catches:
  * ``test_workspace_dispatcher_canvas_mount_js.py`` greps the
    file for the merge pattern.  Brittle to refactors that move
    the line OR rephrase the assignment.

What an L2 guard CANNOT catch:
  * Whether the script-tag order in ``molbuilder.html`` actually
    loads ``_canvas-state-impl.js`` BEFORE ``dispatcher.js`` (a
    template typo could swap them and an L2 test would still
    pass).
  * Whether the merge survives a future refactor that adds a
    different mount mechanism (e.g., a class with side effects in
    its constructor that races the canvas mount).

This e2e pins both: it loads the real /molbuilder page in a
browser and asserts BOTH halves of the contract:

  1. ``window.molbuilder.workspace._canvasState`` exists after
     full page mount (the slot survives).
  2. ``window.molbuilder.workspace.getStructure()`` is callable
     without throwing (the integrated stores are reachable).
  3. Empty-page contract: ``getStructure()`` returns ``null``
     because the workspace starts empty -- if the slot were
     wiped, the call would THROW instead of returning null
     (per ``_missing()`` in dispatcher.js).

A future regression that swaps script order, drops the
Object.assign, or otherwise breaks the slot survival fails this
test loudly.
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


def test_workspace_dispatcher_preserves_canvas_slot_on_molbuilder(
        page, flask_server):
    """The dispatcher's mount at dispatcher.js:1101 MUST merge with
    the existing ``workspace`` object (Object.assign), not replace
    it.  Bare ``root.molbuilder.workspace = api`` wipes the
    ``_canvasState`` slot mounted by ``_canvas-state-impl.js`` and
    every downstream store-accessing helper throws.
    """
    page.goto(f"{flask_server}/molbuilder")
    # Wait for the dispatcher API to be wired up — the simplest
    # observable is the public ``getStructure`` function appearing
    # on the workspace object.
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.workspace "
        "      && typeof window.molbuilder.workspace.getStructure "
        "         === 'function'",
        timeout=10000,
    )
    # Contract assertion 1: the canvas-state slot survived the
    # dispatcher mount.  Pre-fix this was null (because the bare
    # assignment wiped it).
    canvas_slot_present = page.evaluate(
        "() => !!(window.molbuilder.workspace._canvasState)"
    )
    assert canvas_slot_present, (
        "window.molbuilder.workspace._canvasState was wiped by "
        "the dispatcher mount -- regression of the 2026-06-13 "
        "BLOCKER (bare ``workspace = api`` instead of "
        "``Object.assign(workspace || {}, api)``).  Restore the "
        "Object.assign merge at dispatcher.js:1101 or fix the "
        "script-tag order in molbuilder.html so the impl mounts "
        "AFTER the dispatcher (whichever the refactor intended)."
    )
    # Contract assertion 2: the integrated API is reachable
    # without throwing.  Empty page → getStructure() returns null.
    # Pre-fix this THREW ``canvas store not available on this
    # page`` because _canvas() returned null.
    structure_call_returns = page.evaluate(
        "() => {"
        "  try {"
        "    const s = window.molbuilder.workspace.getStructure();"
        "    return { ok: true, value: s };"
        "  } catch (e) {"
        "    return { ok: false, error: String(e && e.message || e) };"
        "  }"
        "}"
    )
    assert structure_call_returns["ok"], (
        f"workspace.getStructure() threw on /molbuilder mount: "
        f"{structure_call_returns.get('error')!r}.  This is the "
        f"2026-06-13 BLOCKER class: the canvas store slot was "
        f"unreachable so _canvas() returned null and the dispatcher "
        f"raised via _missing()."
    )
    # Empty page → null structure.  Anything else would indicate
    # the page bootstrapped with a structure (test fixture leak).
    assert structure_call_returns["value"] is None, (
        f"Expected getStructure() to return null on a freshly-"
        f"loaded /molbuilder (no structure), got: "
        f"{structure_call_returns['value']!r}.  Did a test fixture "
        f"leak structure state?"
    )
