"""Node test: dispatcher.js MUST preserve the canvas-state mount.

Regression pin for the 2026-06-14 BLOCKER.  The MolView data model
(``lib/molview/data-model.js``) reads the canvas-state store through the
private ``workspace._canvasState`` slot that ``_canvas-state-impl.js``
mounts earlier in the script-tag order.  The workspace dispatcher
(``lib/workspace/dispatcher.js``) originally mounted itself with
``root.molbuilder.workspace = api``, which silently CLOBBERED that slot;
after the clobber every data-model read that resolves the canvas via
``_canvas()`` got ``null`` and the Generate / Load buttons on
``/molbuilder`` failed after a successful build.

The fix is mechanical: ``Object.assign(root.molbuilder.workspace || {},
api)``.  This test loads the four workspace/molview JS files in their
production order in a vm sandbox (browser-like: no ``module`` global) and
asserts:

  (a) the private ``_canvasState`` slot survives the dispatcher load, and
  (b) a canvas write is READABLE end-to-end through the data model's
      public ``molview.data`` surface afterwards.

If either assertion fails, the dispatcher's mount step has regressed —
fix the dispatcher, NOT this test.

The molecule I/O surface is the unified ``molview.data.load()`` /
``save()`` (molview-module.md §19.3-19.4); the pre-carve ``installStructure``
door is gone, so (b) seeds the canvas through the internal store slot the
mount must have preserved and reads it back through the public API.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"

# Production script-tag order: the two internal store impls, then the
# MolView data model, then the workspace persistence dispatcher.
WORKSPACE_FILES = [
    STATIC / "lib/molview/_selection-store-impl.js",
    STATIC / "lib/molview/_canvas-state-impl.js",
    STATIC / "lib/molview/data-model.js",
    STATIC / "lib/workspace/dispatcher.js",
]


def _run_node(script: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", script],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


def _bootstrap_js() -> str:
    """Load the workspace/molview files in production order into a vm
    sandbox that mimics the browser (no ``module`` global, so each IIFE
    takes its browser mount branch).  Exposes ``ws`` (the workspace
    persistence surface) and ``data`` (the MolView data model)."""
    paths_js = json.dumps([str(p) for p in WORKSPACE_FILES])
    return dedent(f"""
        const fs = require("fs");
        const vm = require("vm");
        const sandbox = {{ window: {{}}, console }};
        sandbox.globalThis = sandbox;
        const ctx = vm.createContext(sandbox);
        for (const f of {paths_js}) {{
            vm.runInContext(
                fs.readFileSync(f, "utf8"),
                ctx,
                {{ filename: f }}
            );
        }}
        const ws = sandbox.window.molbuilder
                && sandbox.window.molbuilder.workspace;
        const data = sandbox.window.molbuilder
                && sandbox.window.molbuilder.molview
                && sandbox.window.molbuilder.molview.data;
    """)


def test_canvas_state_mount_survives_dispatcher_load():
    """The BLOCKER fix: after the dispatcher loads, the canvas-state
    IIFE's private ``_canvasState`` slot MUST still be on ``workspace``,
    and the data model + its public surface MUST be mounted."""
    out = _run_node(_bootstrap_js() + dedent("""
        console.log(JSON.stringify({
            workspace_present:    !!ws,
            canvas_state_present: !!(ws && ws._canvasState),
            data_present:         !!data,
            load_present:         typeof (data && data.load),
            save_present:         typeof (data && data.save),
            getStructure_present: typeof (data && data.getStructure),
            isEmpty_present:      typeof (data && data.isEmpty),
        }));
    """))
    assert out["workspace_present"] is True
    assert out["canvas_state_present"] is True, (
        "dispatcher.js clobbered workspace._canvasState.  See the module "
        "docstring for the BLOCKER this test pins."
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
    out = _run_node(_bootstrap_js() + dedent("""
        try {
            // Seed the canvas through the internal slot the mount must
            // have preserved (the public door is load(), which needs a
            // server; this test is about the MOUNT, not the load path).
            ws._canvasState.setStructure(
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
