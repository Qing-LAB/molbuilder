"""L2 Node test: dispatcher.js MUST preserve the canvas-state mount.

Regression pin for the 2026-06-14 BLOCKER: dispatcher.js originally
mounted itself with ``root.molbuilder.workspace = api`` which silently
clobbered the ``workspace._canvasState`` slot the canvas-state IIFE
had already populated.  After the clobber, every dispatcher consumer
(``ws.installStructure``, ``ws.markDirty``, ``ws.markSaved``, ``ws.
isEmpty``) reaching for the canvas store via ``_canvas()`` got
``null`` and threw the now-infamous ::

    workspace dispatcher: canvas store not available on this page.
    Phase 4 wires the dispatcher on /molbuilder; ...

The symptom was: on /molbuilder, the Generate (SMILES / DNA / RNA /
peptide / name) and Load buttons silently failed after the structure
generation succeeded — the build-fetch resolved, then ``structurePage.
loadIntoCanvas → ws.installStructure → _canvas() === null → throw``
fired inside the SMILES/etc. promise chain's ``.catch``, surfacing as
"Could not reach /api/build/molecule: workspace dispatcher: canvas
store not available on this page".  Reload behaviour was identical:
canvas not mounted, isEmpty() returned the default ``true``, no
workspace state ever installed.

The fix is mechanical: ``Object.assign(workspace || {}, api)``.  This
test loads the three workspace JS files in their production order in
a vm sandbox and asserts (a) the private ``_canvasState`` slot survives
the dispatcher load, and (b) ``ws.installStructure`` actually drives
the canvas state to non-empty afterwards.

If either assertion fails, the dispatcher's mount step has regressed —
fix the dispatcher, NOT this test.
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
    """Load the three workspace files in production order into a vm
    sandbox that mimics the browser (no ``module`` global).  Returns
    JS that exposes ``window`` in the sandbox so callers can probe
    the final mount state.
    """
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
    """Pinning the BLOCKER fix: after the dispatcher loads, the
    canvas-state IIFE's private ``_canvasState`` slot MUST still be
    on ``workspace``.  Pre-fix dispatcher.js did ``workspace = api``
    which wiped it; this test failed (verified manually) on the
    pre-fix dispatcher and passes on the Object.assign version."""
    out = _run_node(_bootstrap_js() + dedent("""
        console.log(JSON.stringify({
            workspace_present:   !!ws,
            canvas_state_present: !!(ws && ws._canvasState),
            install_present:     typeof (data && data.installStructure),
            isEmpty_present:     typeof (data && data.isEmpty),
        }));
    """))
    assert out["workspace_present"] is True
    assert out["canvas_state_present"] is True, (
        "dispatcher.js clobbered workspace._canvasState.  See the "
        "module docstring for the BLOCKER this test pins."
    )
    assert out["install_present"] == "function"
    assert out["isEmpty_present"] == "function"


def test_install_structure_actually_drives_canvas_state():
    """End-to-end shape check: after installStructure runs, isEmpty
    must be false AND getStructure must return the bytes we passed
    in.  This is the round-trip a Generate / Load button completes
    after the user clicks; the pre-fix dispatcher threw at line one
    of installStructure."""
    out = _run_node(_bootstrap_js() + dedent("""
        try {
            data.installStructure(
                { source_format: "xyz",
                  text: "1\\n\\nH 0 0 0\\n" },
                { kind: "smiles" }
            );
            const s = data.getStructure();
            console.log(JSON.stringify({
                ok: true,
                is_empty_after: data.isEmpty(),
                source_format:  s && s.source_format,
                text_prefix:    s && s.text && s.text.slice(0, 1),
            }));
        } catch (e) {
            console.log(JSON.stringify({
                ok: false,
                error: e && e.message ? e.message : String(e),
            }));
        }
    """))
    assert out["ok"] is True, (
        f"installStructure threw: {out.get('error')}"
    )
    assert out["is_empty_after"] is False
    assert out["source_format"] == "xyz"
    assert out["text_prefix"] == "1"


def test_dispatcher_mount_is_merge_not_replace():
    """Lock the dispatcher's mount to ``Object.assign`` semantics.
    A reader who sees the comment + this test gets the answer to
    "why is Object.assign load-bearing here" without re-deriving
    the BLOCKER.

    2026-06-14 follow-up: the original guard was TOO LOOSE.  It
    accepted ANY ``Object.assign(root.molbuilder.workspace, …)``
    construct.  But ``Object.assign({}, root.molbuilder.workspace,
    api)`` (with an empty seed object) creates a NEW object,
    copies workspace's slots into it (preserving _canvasState
    temporarily), THEN assigns that new object back to
    ``workspace``.  The _canvasState slot survives one merge but
    a future caller would lose any AFTER-mount additions.  AND if
    the same line later changes from
    ``root.molbuilder.workspace = Object.assign({}, workspace, api)``
    to ``= Object.assign({}, api)`` (dropping the workspace arg),
    _canvasState dies and the test still passes.

    Tightened guard: REQUIRE the merge form to read EXACTLY
    ``Object.assign(\\n        root.molbuilder.workspace || {}, api)``
    (the literal fix shipped).  Reject any other form including:
      * ``Object.assign({}, root.molbuilder.workspace, api)`` (new
        object on the RHS -- _canvasState reachable today but the
        whole-object-replacement pattern is the bug pattern).
      * ``root.molbuilder.workspace = api`` (bare replace).
    """
    src = (STATIC / "lib/workspace/dispatcher.js").read_text(
        encoding="utf-8")

    # Required: the exact merge expression that preserves the
    # in-place workspace object's _canvasState slot.
    required = (
        "Object.assign(\n        root.molbuilder.workspace"
        " || {}, api)"
    )
    assert required in src, (
        "dispatcher.js MUST mount via\n"
        f"    {required}\n"
        "to preserve the _canvasState slot the canvas-state-impl "
        "set earlier in the script-tag order.  See "
        "tests/test_workspace_dispatcher_canvas_mount_js.py "
        "docstring for the BLOCKER history."
    )

    # Forbidden: any form that creates a NEW object on the RHS or
    # bare replace.  These all silently re-introduce the BLOCKER
    # class (whole-object replacement of workspace).
    forbidden_patterns = [
        "root.molbuilder.workspace = api;",
        "root.molbuilder.workspace = api\n",
        "Object.assign({}, root.molbuilder.workspace",
        # Future refactor red-flag: any RHS that drops the workspace
        # argument entirely.
        "root.molbuilder.workspace = Object.assign(api,",
    ]
    for bad in forbidden_patterns:
        assert bad not in src, (
            f"dispatcher.js contains the forbidden pattern\n"
            f"    {bad!r}\n"
            f"which re-introduces the 2026-06-14 BLOCKER class "
            f"(whole-object replacement of workspace).  Replace "
            f"with:\n    {required}"
        )
