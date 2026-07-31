"""Node test: the canvas-state mount survives the full load order.

The MolView data model (``lib/molview/data-model.js``) reads the canvas-state
store through the private ``molview._canvasState`` slot that
``_canvas-state-impl.js`` mounts earlier in the script-tag order.  The slot lives
on the ``molview`` namespace (the 2026-07 carve made ``workspace``
persistence-only, so a data store must NOT hang off ``workspace.*``).

Descended from the 2026-06-14 BLOCKER (then the dispatcher's
``root.molbuilder.workspace = api`` clobbered the slot when it lived on
``workspace``).  The invariant now: every module that assigns a shared namespace
MERGES it (``root.molbuilder.<ns> = root.molbuilder.<ns> || {}`` /
``Object.assign``), so data-model.js mounting ``molview.data`` must not clobber
``molview._canvasState``.  This test loads the workspace/molview JS files in
production order through the shared ES-module harness (tests/_node_esm) and asserts:

  (a) the private ``_canvasState`` slot survives the full load, and
  (b) a canvas write is READABLE end-to-end through the data model's
      public ``molview.data`` surface afterwards.

If either assertion fails, a mount step has regressed — fix the module, NOT this
test.

The molecule I/O surface is the unified ``molview.data.load()`` /
``save()`` (molview-module.md §19.3-19.4); the pre-carve ``installStructure``
door is gone, so (b) seeds the canvas through the internal store slot the
mount must have preserved and reads it back through the public API.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from _node_esm import run_node

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"

# Production script-tag order: the two internal store impls, the state-timeline
# factory, then the MolView data model, then the workspace persistence dispatcher.
# data-model.js is now an ES module (the MolView ESM migration), so it loads through
# the shared ES-module harness (tests/_node_esm.run_node) rather than a classic vm
# script -- the ``export`` it now carries is a SyntaxError under a sloppy-script vm run.
WORKSPACE_FILES = [
    STATIC / "lib/molview/_selection-store-impl.js",
    STATIC / "lib/molview/_canvas-state-impl.js",
    STATIC / "lib/molview/_state-timeline-impl.js",
    STATIC / "lib/molview/data-model.js",
    STATIC / "lib/workspace/dispatcher.js",
]

# Minimal browser stubs the dispatcher/data-model touch at load (pagehide listener,
# document, sessionStorage, runtime registry).  Injected before the module imports.
_GLOBALS = """
    global.window.addEventListener = () => {};
    global.document = {
        readyState: "complete",
        addEventListener: () => {},
        getElementById:  () => null,
    };
    const _storage = {};
    global.sessionStorage = {
        getItem:    (k) => (_storage[k] == null ? null : _storage[k]),
        setItem:    (k, v) => { _storage[k] = String(v); },
        removeItem: (k) => { delete _storage[k]; },
    };
    global.molbuilder.runtime = {
        register:  () => {},
        whenReady: () => new Promise(() => {}),
    };
"""

# Resolve the public surfaces from the globals the modules publish, so the snippet
# bodies read ``ws`` / ``molview`` / ``data`` exactly as before.
_PREAMBLE = """
    const ws = window.molbuilder && window.molbuilder.workspace;
    const molview = window.molbuilder && window.molbuilder.molview;
    const data = molview && molview.data;
"""


def _run_node(snippet: str) -> dict:
    return run_node(WORKSPACE_FILES, _PREAMBLE + snippet, globals_js=_GLOBALS)


def test_canvas_state_mount_survives_dispatcher_load():
    """After all four files load, the canvas-state IIFE's private
    ``_canvasState`` slot MUST still be on ``molview`` (data-model.js merges
    ``molview`` with ``|| {}``, so mounting ``molview.data`` must not clobber it),
    and the data model + its public surface MUST be mounted."""
    out = _run_node(dedent("""
        console.log(JSON.stringify({
            workspace_present:    !!ws,
            molview_present:      !!molview,
            canvas_state_present: !!(molview && molview._canvasState),
            data_present:         !!data,
            load_present:         typeof (data && data.load),
            save_present:         typeof (data && data.save),
            getStructure_present: typeof (data && data.getStructure),
            isEmpty_present:      typeof (data && data.isEmpty),
        }));
    """))
    assert out["workspace_present"] is True
    assert out["canvas_state_present"] is True, (
        "molview._canvasState was clobbered -- data-model.js must MERGE the "
        "molview namespace (root.molbuilder.molview = root.molbuilder.molview "
        "|| {}), not replace it.  See the module docstring."
    )
    assert out["data_present"] is True
    assert out["load_present"] == "function"
    assert out["save_present"] == "function"
    assert out["getStructure_present"] == "function"
    assert out["isEmpty_present"] == "function"


def test_canvas_write_is_readable_through_the_data_model():
    """End-to-end shape check: with the ``_canvasState`` slot preserved,
    a canvas write is readable through the PUBLIC ``molview.data``
    surface (getStructure / isEmpty).  The pre-fix dispatcher wiped the
    slot, so the data model's ``_canvas()`` resolved null and every read
    came back empty."""
    out = _run_node(dedent("""
        try {
            // Seed the canvas through the internal slot the mount must
            // have preserved (the public door is load(), which needs a
            // server; this test is about the MOUNT, not the load path).
            molview._canvasState.setStructure(
                { source_format: "xyz", text: "1\\n\\nH 0 0 0\\n" },
                { kind: "smiles" }
            );
            const s = data.getStructure();
            console.log(JSON.stringify({
                ok:             true,
                is_empty_after: data.isEmpty(),
                source_format:  s && s.source_format,
                text_prefix:    s && s.text && s.text.slice(0, 1),
            }));
        } catch (e) {
            console.log(JSON.stringify({
                ok: false, error: e && e.message ? e.message : String(e),
            }));
        }
    """))
    assert out["ok"] is True, f"data model read threw: {out.get('error')}"
    assert out["is_empty_after"] is False
    assert out["source_format"] == "xyz"
    assert out["text_prefix"] == "1"


def test_dispatcher_mount_is_merge_not_replace():
    """Lock the dispatcher's mount to ``Object.assign`` semantics.  A
    reader who sees the comment + this test gets the answer to "why is
    Object.assign load-bearing here" without re-deriving the BLOCKER.

    Tightened guard (2026-06-14 follow-up): REQUIRE the merge form to
    read EXACTLY ``Object.assign(\\n        root.molbuilder.workspace ||
    {}, api)`` (the literal fix shipped), and REJECT any whole-object-
    replacement form — including ``Object.assign({}, workspace, api)``
    (new object on the RHS) and the bare ``workspace = api`` replace —
    because those re-introduce the BLOCKER class.
    """
    src = (STATIC / "lib/workspace/dispatcher.js").read_text(encoding="utf-8")

    required = (
        "Object.assign(\n        root.molbuilder.workspace"
        " || {}, api)"
    )
    assert required in src, (
        "dispatcher.js MUST mount via\n"
        f"    {required}\n"
        "to preserve the _canvasState slot the canvas-state-impl set "
        "earlier in the script-tag order.  See this module's docstring "
        "for the BLOCKER history."
    )

    forbidden_patterns = [
        "root.molbuilder.workspace = api;",
        "root.molbuilder.workspace = api\n",
        "Object.assign({}, root.molbuilder.workspace",
        "root.molbuilder.workspace = Object.assign(api,",
    ]
    for bad in forbidden_patterns:
        assert bad not in src, (
            f"dispatcher.js contains the forbidden pattern\n    {bad!r}\n"
            "which re-introduces the 2026-06-14 BLOCKER class "
            "(whole-object replacement of workspace)."
        )
