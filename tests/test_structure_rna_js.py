"""Unit tests for the RNA generator panel module.

Mirrors test_structure_dna_js.py — the modules are structural
twins.  Differences pinned here:
  * Alphabet: ACGU (uracil) instead of ACGT (thymine).
  * Default helix form: A (canonical RNA) instead of B (DNA).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/structure/rna.js"
#: The panel reads its dependency slots from the shared wiring
#: (`panel-deps.js`), which the page loads before it.  A harness that
#: skips it is testing a module the browser never runs.
PANEL_DEPS = ROOT / "molbuilder/web/static/modify/structure/panel-deps.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        require({json.dumps(str(PANEL_DEPS.resolve()))});
        const rna = require({json.dumps(str(module_path))});
        try {{
            (async () => {{
                {snippet}
            }})().catch(err => {{
                console.log(JSON.stringify({{
                    __test_unexpected_throw: true,
                    message: err && err.message ? err.message : String(err),
                }}));
            }});
        }} catch (err) {{
            console.log(JSON.stringify({{
                __test_unexpected_throw: true,
                message: err && err.message ? err.message : String(err),
            }}));
        }}
    """
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", bootstrap],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if isinstance(out, dict):
        assert "__test_unexpected_throw" not in out, (
            "module threw: " + str(out))
    return out


class TestSurfacePresence:

    def test_methods_callable(self):
        out = _run_node('''
            console.log(JSON.stringify({
                configure:   typeof rna.configure,
                generate:    typeof rna.generate,
                wirePanel:   typeof rna.wirePanel,
                BUILD_URL:   rna.BUILD_URL,
                VALID_FORMS: rna.VALID_FORMS,
            }));
        ''')
        assert out["configure"] == "function"
        assert out["generate"]  == "function"
        assert out["wirePanel"] == "function"
        assert out["BUILD_URL"] == "/api/build/molecule"
        # A first because that's the canonical RNA helix form.
        assert out["VALID_FORMS"] == ["A", "B", "Z"]


class TestInputValidation:

    def test_empty_sequence_rejected(self):
        out = _run_node('''
            let fetchCalls = 0;
            rna.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await rna.generate("");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "RNA sequence" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0

    def test_dna_thymine_rejected_in_rna_panel(self):
        """T is DNA's thymine; RNA uses U.  Catching this client
        side gives a clearer error than the backend tleap failure."""
        out = _run_node('''
            let fetchCalls = 0;
            rna.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            // ACGT is DNA — should be rejected.
            const r = await rna.generate("ACGT");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "ACGU" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0

    def test_invalid_form_rejected_client_side(self):
        out = _run_node('''
            let fetchCalls = 0;
            rna.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await rna.generate("ACGU", {form: "Q"});
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "A, B, Z" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0


class TestHappyPath:

    def test_default_form_is_A(self):
        """A is the canonical RNA helix form — the default must
        match what tleap / 3DNA expect for an unspecified helix."""
        out = _run_node('''
            let capturedBody = null;
            rna.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "20\\nrna\\nC 0 0 0\\n",
                            n_atoms: 20,
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await rna.generate("ACGU");
            console.log(JSON.stringify(capturedBody));
        ''')
        assert out == {"kind": "rna", "input": "ACGU",
                       "form": "A", "backend": "auto"}

    def test_explicit_form_forwarded(self):
        out = _run_node('''
            let capturedBody = null;
            rna.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "5\\nx\\nC 0 0 0\\n",
                            n_atoms: 5,
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await rna.generate("ACGU", {form: "B"});
            console.log(JSON.stringify(capturedBody));
        ''')
        assert out["form"] == "B"

    def test_canvas_source_provenance_records_sequence_and_form(self):
        out = _run_node('''
            let canvasArgs = null;
            rna.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "5\\nx\\nC 0 0 0\\n",
                        n_atoms: 5,
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async (struct, src) => {
                        canvasArgs = {struct, src};
                        return {ok: true};
                    },
                },
            });
            await rna.generate("ACGU", {form: "Z"});
            console.log(JSON.stringify(canvasArgs));
        ''')
        assert out["src"]["kind"] == "rna"
        assert out["src"]["generator_input"]["sequence"] == "ACGU"
        assert out["src"]["generator_input"]["form"] == "Z"


class TestErrorPaths:

    def test_backend_failure_surfaces_message(self):
        out = _run_node('''
            rna.configure({
                fetch: async () => ({
                    ok: false,
                    json: async () => ({
                        ok: false,
                        error: "tleap: residue RA5 unknown",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await rna.generate("ACGU");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "tleap" in out["error"]

    def test_canvas_cancel_passes_through_as_cancelled(self):
        """User cancels the dirty-canvas warning modal (load door
        returns cancelled) → envelope carries cancelled, called
        through the single load door exactly once."""
        out = _run_node('''
            let loadCalls = 0;
            rna.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "1\\nx\\nC 0 0 0\\n",
                        n_atoms: 1,
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => {
                        loadCalls++;
                        return { ok: false, cancelled: true };
                    },
                },
            });
            const r = await rna.generate("A");
            console.log(JSON.stringify({envelope: r, loadCalls}));
        ''')
        assert out["envelope"] == {"ok": False, "cancelled": True}
        assert out["loadCalls"] == 1
