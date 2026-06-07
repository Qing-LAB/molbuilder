"""Unit tests for the SMILES generator panel module.

Pins the public API of ``molbuilder/web/static/lib/structure/
smiles.js`` — the Sources-card panel that POSTs to
``/api/build/molecule`` and routes the generated XYZ through the
canvas-state gate.

These tests use a fake fetch + structurePage + viewer-loader so
the state machine can be exercised without a real DOM or HTTP
roundtrip.  The DOM wiring (``wirePanel`` against actual buttons +
inputs) is exercised by an e2e test in test_modify_e2e.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/structure/smiles.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        const smiles = require({json.dumps(str(module_path))});
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
                configure: typeof smiles.configure,
                generate:  typeof smiles.generate,
                wirePanel: typeof smiles.wirePanel,
                BUILD_URL: smiles.BUILD_URL,
            }));
        ''')
        assert out["configure"] == "function"
        assert out["generate"]  == "function"
        assert out["wirePanel"] == "function"
        assert out["BUILD_URL"] == "/api/build/molecule"


# ----- Input validation ------------------------------------------ #


class TestInputValidation:

    def test_empty_smiles_rejected_without_fetch(self):
        """An empty input must NOT hit the network — saves a
        roundtrip + gives the user a clear inline error."""
        out = _run_node('''
            let fetchCalls = 0;
            smiles.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await smiles.generate("");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "Enter a SMILES" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0

    def test_whitespace_only_rejected(self):
        out = _run_node('''
            let fetchCalls = 0;
            smiles.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await smiles.generate("   \\n\\t  ");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert out["fetchCalls"] == 0

    def test_non_string_rejected(self):
        out = _run_node('''
            smiles.configure({
                fetch: async () => ({}),
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await smiles.generate(null);
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False


# ----- Happy path ------------------------------------------------ #


class TestHappyPath:

    def test_successful_generate_calls_loadIntoCanvas_with_xyz(self):
        """Pins the canvas-state contract: a successful build POST
        routes the XYZ + source provenance through
        ``structurePage.loadIntoCanvas``."""
        out = _run_node('''
            let capturedUrl = null;
            let capturedBody = null;
            let canvasArgs = null;
            let viewerArgs = null;
            smiles.configure({
                fetch: async (url, init) => {
                    capturedUrl = url;
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true,
                            xyz: "3\\nethanol\\nC 0 0 0\\nC 1 0 0\\nO 2 0 0\\n",
                            n_atoms: 3,
                            title: "ethanol",
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async (struct, src) => {
                        canvasArgs = {struct, src};
                        return {ok: true};
                    },
                },
                viewerLoader: (text, fname) => {
                    viewerArgs = {text, fname};
                    return Promise.resolve();
                },
            });
            const r = await smiles.generate("CCO");
            console.log(JSON.stringify({
                envelope: r,
                url:      capturedUrl,
                body:     capturedBody,
                canvas:   canvasArgs,
                viewer:   viewerArgs,
            }));
        ''')
        assert out["envelope"] == {"ok": True, "n_atoms": 3}
        assert out["url"] == "/api/build/molecule"
        assert out["body"] == {"kind": "smiles", "input": "CCO"}
        # Canvas gate receives the XYZ + source provenance.
        assert out["canvas"]["struct"]["source_format"] == "xyz"
        assert out["canvas"]["struct"]["text"].startswith("3\n")
        assert out["canvas"]["src"]["kind"] == "smiles"
        assert out["canvas"]["src"]["generator_input"]["smiles"] == "CCO"
        # Viewer loader was called with the XYZ text + a sane
        # filename derived from the SMILES.
        assert out["viewer"]["text"].startswith("3\n")
        assert "smiles-CCO" in out["viewer"]["fname"]

    def test_smiles_input_is_trimmed_before_post(self):
        """Leading/trailing whitespace is stripped — the server
        side rejects empty input, and a SMILES with surrounding
        space is the same molecule."""
        out = _run_node('''
            let capturedBody = null;
            smiles.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "1\\nC\\nC 0 0 0\\n",
                            n_atoms: 1,
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await smiles.generate("  C  ");
            console.log(JSON.stringify(capturedBody));
        ''')
        assert out["input"] == "C"


# ----- Error paths ----------------------------------------------- #


class TestErrorPaths:

    def test_http_error_surfaces_message(self):
        out = _run_node('''
            smiles.configure({
                fetch: async () => ({
                    ok: false,
                    json: async () => ({
                        ok: false,
                        error: "RDKit could not parse SMILES",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await smiles.generate("not-a-real-smiles");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "RDKit" in out["error"]

    def test_network_failure_returns_envelope_not_throws(self):
        out = _run_node('''
            smiles.configure({
                fetch: async () => {
                    throw new TypeError("Failed to fetch");
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await smiles.generate("CCO");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "/api/build/molecule" in out["error"]
        assert "Failed to fetch" in out["error"]

    def test_server_returns_ok_false_envelope(self):
        """Server reports ok:false in the body (validation error,
        unknown kind, etc.) even when the HTTP status is 200 — the
        module must surface the error text."""
        out = _run_node('''
            smiles.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: false,
                        error: "empty input",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await smiles.generate("CCO");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert out["error"] == "empty input"


# ----- Canvas-gate integration ----------------------------------- #


class TestCanvasGate:

    def test_canvas_cancel_passes_through_and_skips_viewer(self):
        """If the user cancels the warning modal, the result envelope
        carries ``cancelled: true`` and the viewer loader is NOT
        called (no surprise re-render of an unchanged canvas)."""
        out = _run_node('''
            let viewerCalls = 0;
            smiles.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "1\\nC\\nC 0 0 0\\n", n_atoms: 1,
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({
                        ok: false, cancelled: true,
                    }),
                },
                viewerLoader: () => {
                    viewerCalls++;
                    return Promise.resolve();
                },
            });
            const r = await smiles.generate("C");
            console.log(JSON.stringify({
                envelope:    r,
                viewerCalls: viewerCalls,
            }));
        ''')
        assert out["envelope"] == {"ok": False, "cancelled": True}
        assert out["viewerCalls"] == 0

    def test_no_viewer_loader_still_returns_ok_when_canvas_accepts(self):
        """In test contexts the viewer loader may be unset; the
        canvas gate accepting the XYZ is enough — the module
        returns ok:true regardless of whether the viewer renders."""
        out = _run_node('''
            smiles.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "1\\nC\\nC 0 0 0\\n", n_atoms: 1,
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
                viewerLoader: null,
            });
            const r = await smiles.generate("C");
            console.log(JSON.stringify(r));
        ''')
        assert out == {"ok": True, "n_atoms": 1}


# ----- Configuration error surface ------------------------------- #


class TestConfigurationErrors:

    def test_generate_rejects_when_fetch_unconfigured(self):
        out = _run_node('''
            // Fresh smiles module never configured.
            const p = smiles.generate("CCO");
            let rejected = false, msg = "";
            try { await p; }
            catch (e) { rejected = true; msg = e.message; }
            console.log(JSON.stringify({rejected, msg}));
        ''')
        assert out["rejected"] is True
        assert "fetch" in out["msg"]

    def test_generate_rejects_when_structurePage_unconfigured(self):
        out = _run_node('''
            smiles.configure({
                fetch: async () => ({ok: true, json: async () => ({})}),
                // structurePage NOT configured
            });
            const p = smiles.generate("CCO");
            let rejected = false, msg = "";
            try { await p; }
            catch (e) { rejected = true; msg = e.message; }
            console.log(JSON.stringify({rejected, msg}));
        ''')
        assert out["rejected"] is True
        assert "structurePage" in out["msg"]
