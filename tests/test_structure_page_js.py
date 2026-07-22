"""Unit tests for the Molbuilder-tab page orchestrator.

Pins the public API of ``molbuilder/web/static/modify/structure/
page.js`` — the gate every Sources panel routes Load / Generate
calls through.  The orchestrator's one load-bearing job is the
unsaved-modifications check:

  * empty canvas → set directly
  * clean canvas → set directly
  * dirty canvas → ask via warning-modal; set only on Discard

These tests use a fake canvas-state + warning-modal API so the
state machine can be exercised without the real DOM / sessionStorage
machinery (which has its own dedicated test files).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/structure/page.js"


# Fake canvas-state + warning-modal — minimum surface the page
# orchestrator pokes.  Kept inline in the bootstrap so each test
# starts from a fresh state.
_FAKES = r"""
// page.js binds against MolView's unified DATA model
// (window.molbuilder.molview.data), NOT the workspace.  The gate
// reads isEmpty/isDirty and routes through the ONE atomic load
// door: molview.data.load({text, filename, source, periodicity,
// annotations}).  Helper preserved as _mkFakeCanvas for
// test-name continuity.
function _mkFakeCanvas(initial) {
    initial = initial || {};
    let state = {
        empty:      initial.empty !== undefined ? initial.empty : true,
        dirty:      !!initial.dirty,
        structure:  initial.structure || null,
        source:     initial.source || {kind: "blank", file: null,
                                       generator_input: null},
        lastSaveTo: initial.lastSaveTo || null,
    };
    const calls = [];
    return {
        // Public surface page.js uses (molview.data names).
        isEmpty:        () => state.empty,
        isDirty:        () => state.dirty,
        getStructure:   () => state.structure,
        getSource:      () => state.source,
        getLastSavedTo: () => state.lastSaveTo,
        // The ONE atomic whole-model load door (molview.data model primitive).
        // installMolecule forwards {text, filename, source, periodicity,
        // annotations}; record the full arg so tests can assert the payload,
        // and mirror text/source into the state so getStructure() reflects the load.
        installMolecule: (arg) => {
            state.empty     = false;
            state.dirty     = false;
            state.structure = { text: arg.text };
            state.source    = arg.source;
            calls.push({fn: "installMolecule", arg: arg,
                        structure: { text: arg.text },
                        source: arg.source});
            return Promise.resolve();
        },
        markDirty: () => {
            state.dirty = true;
            calls.push({fn: "markDirty"});
        },
        markSaved: (p) => {
            state.dirty      = false;
            state.lastSaveTo = p;
            calls.push({fn: "markSaved", path: p});
        },
        subscribe: (cb) => {
            calls.push({fn: "onChange"});
            return () => {};
        },
        // Test inspection helpers.
        _state: () => ({...state}),
        _calls: () => calls.slice(),
    };
}
function _mkFakeModal(verdict) {
    // ``verdict`` is the boolean the modal will resolve to.
    const calls = [];
    return {
        confirmDiscardUnsaved: () => {
            calls.push("confirmDiscardUnsaved");
            return Promise.resolve(verdict);
        },
        _calls: () => calls.slice(),
    };
}
"""


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        {_FAKES}
        const page = require({json.dumps(str(module_path))});
        try {{
            (async () => {{
                {snippet}
            }})().catch(err => {{
                console.log(JSON.stringify({{
                    __test_unexpected_throw: true,
                    message: err && err.message ? err.message : String(err),
                    stack:   err && err.stack ? err.stack : null,
                }}));
            }});
        }} catch (err) {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
                stack:   err && err.stack ? err.stack : null,
            }}));
        }}
    """
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    last_line = proc.stdout.strip().splitlines()[-1]
    out = json.loads(last_line)
    if isinstance(out, dict):
        assert "__test_unexpected_throw" not in out, (
            "module threw: " + str(out)
        )
    return out


# ----- Surface presence ------------------------------------------ #


class TestSurfacePresence:

    def test_methods_callable(self):
        out = _run_node('''
            console.log(JSON.stringify({
                _bind:                      typeof page._bind,
                loadIntoCanvas:             typeof page.loadIntoCanvas,
                markDirtyAfterModification: typeof page.markDirtyAfterModification,
                markSavedTo:                typeof page.markSavedTo,
                getCanvasSnapshot:          typeof page.getCanvasSnapshot,
                onCanvasChange:             typeof page.onCanvasChange,
            }));
        ''')
        for fn in ("_bind", "loadIntoCanvas",
                   "markDirtyAfterModification", "markSavedTo",
                   "getCanvasSnapshot", "onCanvasChange"):
            assert out[fn] == "function", f"missing: {fn}"


# ----- _bind ------------------------------------------------------ #


class TestBind:

    def test_rejects_missing_canvas(self):
        out = _run_node('''
            const modal = _mkFakeModal(true);
            let threw = false;
            try { page._bind(null, modal); }
            catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is True

    def test_rejects_missing_modal(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas();
            let threw = false;
            try { page._bind(canvas, null); }
            catch (e) { threw = true; }
            console.log(JSON.stringify(threw));
        ''')
        assert out is True


# ----- loadIntoCanvas gate --------------------------------------- #


class TestLoadGateEmptyCanvas:

    def test_empty_canvas_sets_directly_no_modal(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas({ empty: true });
            const modal  = _mkFakeModal(false);  // would say Cancel
            page._bind(canvas, modal);
            const r = await page.loadIntoCanvas(
                { source_format: "xyz", text: "1\\nC\\nC 0 0 0\\n" },
                { kind: "smiles", generator_input: { smiles: "CCO" } }
            );
            console.log(JSON.stringify({
                envelope:   r,
                canvasCalls: canvas._calls(),
                modalCalls:  modal._calls(),
            }));
        ''')
        assert out["envelope"] == {"ok": True}
        # Routed through the ONE atomic load door with the text +
        # generator source riding on the load payload.
        loads = [c for c in out["canvasCalls"] if c["fn"] == "installMolecule"]
        assert len(loads) == 1
        assert loads[0]["arg"]["text"] == "1\nC\nC 0 0 0\n"
        assert loads[0]["arg"]["source"]["kind"] == "smiles"
        # Modal NOT consulted — the empty canvas gate didn't ask.
        assert out["modalCalls"] == []


class TestLoadGateCleanCanvas:

    def test_clean_canvas_sets_directly_no_modal(self):
        """A loaded-but-not-modified canvas is overwriteable without
        prompting — the user has saved (or just loaded) it; nothing
        will be lost."""
        out = _run_node('''
            const canvas = _mkFakeCanvas({
                empty: false, dirty: false,
                structure: { source_format: "xyz", text: "old" },
                source: { kind: "file", file: "/p/a.xyz" },
            });
            const modal = _mkFakeModal(false);
            page._bind(canvas, modal);
            const r = await page.loadIntoCanvas(
                { source_format: "xyz", text: "new" },
                { kind: "smiles" }
            );
            console.log(JSON.stringify({
                envelope:   r,
                modalCalls: modal._calls(),
                lastSet:    canvas._calls().filter(
                                c => c.fn === "installMolecule"
                            ).pop(),
            }));
        ''')
        assert out["envelope"] == {"ok": True}
        assert out["modalCalls"] == []  # not consulted
        assert out["lastSet"]["structure"]["text"] == "new"


class TestLoadGateDirtyCanvas:

    def test_dirty_canvas_proceed_overwrites(self):
        """Dirty canvas + user picks Discard → canvas is overwritten."""
        out = _run_node('''
            const canvas = _mkFakeCanvas({
                empty: false, dirty: true,
                structure: { source_format: "xyz", text: "edited" },
                source: { kind: "smiles" },
            });
            const modal = _mkFakeModal(true);  // user clicks Discard
            page._bind(canvas, modal);
            const r = await page.loadIntoCanvas(
                { source_format: "xyz", text: "new" },
                { kind: "file", file: "/p/b.xyz" }
            );
            console.log(JSON.stringify({
                envelope:   r,
                modalCalls: modal._calls(),
                lastSet:    canvas._calls().filter(
                                c => c.fn === "installMolecule"
                            ).pop(),
            }));
        ''')
        assert out["envelope"] == {"ok": True}
        assert out["modalCalls"] == ["confirmDiscardUnsaved"]
        # The overwrite landed.
        assert out["lastSet"]["structure"]["text"] == "new"
        assert out["lastSet"]["source"]["file"] == "/p/b.xyz"

    def test_dirty_canvas_cancel_does_not_overwrite(self):
        """Dirty canvas + user picks Cancel → canvas untouched +
        envelope reports cancelled."""
        out = _run_node('''
            const canvas = _mkFakeCanvas({
                empty: false, dirty: true,
                structure: { source_format: "xyz", text: "edited" },
                source: { kind: "smiles" },
            });
            const modal = _mkFakeModal(false);  // user clicks Cancel
            page._bind(canvas, modal);
            const r = await page.loadIntoCanvas(
                { source_format: "xyz", text: "new" },
                { kind: "file", file: "/p/b.xyz" }
            );
            const setCalls = canvas._calls().filter(
                c => c.fn === "installMolecule");
            console.log(JSON.stringify({
                envelope:   r,
                modalCalls: modal._calls(),
                setCount:   setCalls.length,
                stillDirty: canvas.isDirty(),
                stillText:  canvas.getStructure().text,
            }));
        ''')
        assert out["envelope"] == {"ok": False, "cancelled": True}
        assert out["modalCalls"] == ["confirmDiscardUnsaved"]
        # No load call landed — canvas is intact.
        assert out["setCount"] == 0
        assert out["stillDirty"] is True
        assert out["stillText"] == "edited"


# ----- modifier helpers ------------------------------------------ #


class TestModifierHelpers:

    def test_mark_dirty_after_modification_delegates(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas({ empty: false, dirty: false });
            page._bind(canvas, _mkFakeModal(false));
            page.markDirtyAfterModification();
            console.log(JSON.stringify({
                isDirty: canvas.isDirty(),
                calls:   canvas._calls().map(c => c.fn),
            }));
        ''')
        assert out["isDirty"] is True
        assert "markDirty" in out["calls"]

    def test_mark_saved_to_delegates(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas({ empty: false, dirty: true });
            page._bind(canvas, _mkFakeModal(false));
            page.markSavedTo("/p/saved.xyz");
            console.log(JSON.stringify({
                isDirty:    canvas.isDirty(),
                lastSavedTo: canvas.getLastSavedTo(),
            }));
        ''')
        assert out["isDirty"] is False
        assert out["lastSavedTo"] == "/p/saved.xyz"


# ----- snapshot + onChange --------------------------------------- #


class TestSnapshot:

    def test_getCanvasSnapshot_returns_full_state(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas({
                empty: false, dirty: true,
                structure: { source_format: "pdb", text: "HETATM" },
                source: { kind: "smiles",
                          generator_input: { smiles: "CCO" } },
                lastSaveTo: null,
            });
            page._bind(canvas, _mkFakeModal(false));
            console.log(JSON.stringify(page.getCanvasSnapshot()));
        ''')
        assert out["isEmpty"] is False
        assert out["isDirty"] is True
        assert out["structure"] == {
            "source_format": "pdb", "text": "HETATM"}
        assert out["source"]["kind"] == "smiles"
        assert out["lastSaveTo"] is None

    def test_onCanvasChange_is_a_passthrough(self):
        out = _run_node('''
            const canvas = _mkFakeCanvas();
            page._bind(canvas, _mkFakeModal(false));
            const unsub = page.onCanvasChange(() => {});
            console.log(JSON.stringify({
                canvasOnChangeCalled: canvas._calls()
                                            .some(c => c.fn === "onChange"),
                unsubIsFn:            typeof unsub === "function",
            }));
        ''')
        assert out["canvasOnChangeCalled"] is True
        assert out["unsubIsFn"] is True


# ----- unbound error handling ------------------------------------ #


class TestUnboundErrors:
    """Each method must surface a clear error if called before
    _bind() — production code wires the bind once at module load,
    but tests + future module orders shouldn't crash mysteriously."""

    def test_loadIntoCanvas_rejects_when_unbound(self):
        out = _run_node('''
            // Don't call _bind.
            const p = page.loadIntoCanvas(
                { source_format: "xyz", text: "x" },
                { kind: "file" }
            );
            let rejected = false;
            let msg = "";
            try { await p; }
            catch (e) { rejected = true; msg = e.message; }
            console.log(JSON.stringify({
                rejected: rejected, msg: msg,
            }));
        ''')
        assert out["rejected"] is True
        assert "not bound" in out["msg"]

    def test_synchronous_helpers_throw_when_unbound(self):
        out = _run_node('''
            const r = {};
            for (const m of ["markDirtyAfterModification",
                             "markSavedTo",
                             "getCanvasSnapshot",
                             "onCanvasChange"]) {
                let threw = false;
                try {
                    if (m === "markSavedTo") page[m]("/p");
                    else if (m === "onCanvasChange") page[m](() => {});
                    else page[m]();
                } catch (e) { threw = true; }
                r[m] = threw;
            }
            console.log(JSON.stringify(r));
        ''')
        for m in ("markDirtyAfterModification", "markSavedTo",
                  "getCanvasSnapshot", "onCanvasChange"):
            assert out[m] is True
