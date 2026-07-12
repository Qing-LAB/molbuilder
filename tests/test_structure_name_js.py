"""Unit tests for the name-lookup generator panel module.

Pins the public API of ``molbuilder/web/static/lib/structure/
name.js`` — the Sources-card panel that POSTs to
``/api/build/molecule`` with kind="name" and routes the
generated XYZ through the canvas-state gate.

Mirrors the test shape of test_structure_smiles_js.py since the
two generators are structural twins (only the request kind +
field IDs differ).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/structure/name.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        const nm = require({json.dumps(str(module_path))});
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
                configure: typeof nm.configure,
                generate:  typeof nm.generate,
                wirePanel: typeof nm.wirePanel,
                BUILD_URL: nm.BUILD_URL,
            }));
        ''')
        assert out["configure"] == "function"
        assert out["generate"]  == "function"
        assert out["wirePanel"] == "function"
        assert out["BUILD_URL"] == "/api/build/molecule"


class TestInputValidation:

    def test_empty_name_rejected_without_fetch(self):
        out = _run_node('''
            let fetchCalls = 0;
            nm.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await nm.generate("");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "Enter a name" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0

    def test_whitespace_rejected(self):
        out = _run_node('''
            let fetchCalls = 0;
            nm.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await nm.generate("   \\n   ");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert out["fetchCalls"] == 0


class TestHappyPath:

    def test_successful_generate_routes_through_canvas(self):
        out = _run_node('''
            let capturedBody = null;
            let canvasArgs = null;
            nm.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true,
                            xyz: "9\\nethanol\\nC 0 0 0\\n...\\n",
                            n_atoms: 9,
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async (struct, src) => {
                        canvasArgs = {struct, src};
                        return {ok: true};
                    },
                },
            });
            const r = await nm.generate("ethanol");
            console.log(JSON.stringify({
                envelope: r,
                body:     capturedBody,
                canvas:   canvasArgs,
            }));
        ''')
        assert out["envelope"] == {"ok": True, "n_atoms": 9}
        # POST body uses kind="name", NOT kind="smiles".
        assert out["body"] == {"kind": "name", "input": "ethanol"}
        # Canvas source provenance carries kind="name" + the
        # original query.
        assert out["canvas"]["src"]["kind"] == "name"
        assert out["canvas"]["src"]["generator_input"]["name"] == "ethanol"

    def test_name_trimmed_before_post(self):
        out = _run_node('''
            let capturedBody = null;
            nm.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "1\\nx\\nC 0 0 0\\n",
                            n_atoms: 1,
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await nm.generate("  water  ");
            console.log(JSON.stringify(capturedBody));
        ''')
        assert out["input"] == "water"


class TestErrorPaths:

    def test_pubchem_lookup_failure_surfaces_message(self):
        """Backend returns ok:false when PubChem doesn't recognise
        the name; the module passes the error through verbatim
        so the user sees the actual reason."""
        out = _run_node('''
            nm.configure({
                fetch: async () => ({
                    ok: false,
                    json: async () => ({
                        ok: false,
                        error: "PubChem: no match for 'qwerty'",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await nm.generate("qwerty");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "PubChem" in out["error"]

    def test_canvas_cancel_passes_through_as_cancelled(self):
        """User cancels the dirty-canvas warning modal (load door
        returns cancelled) → envelope carries cancelled, called
        through the single load door exactly once."""
        out = _run_node('''
            let loadCalls = 0;
            nm.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "1\\nx\\nC 0 0 0\\n", n_atoms: 1,
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => {
                        loadCalls++;
                        return { ok: false, cancelled: true };
                    },
                },
            });
            const r = await nm.generate("benzene");
            console.log(JSON.stringify({
                envelope:  r,
                loadCalls: loadCalls,
            }));
        ''')
        assert out["envelope"] == {"ok": False, "cancelled": True}
        assert out["loadCalls"] == 1

    def test_network_failure_returns_envelope(self):
        out = _run_node('''
            nm.configure({
                fetch: async () => {
                    throw new TypeError("Failed to fetch");
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await nm.generate("ethanol");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "Failed to fetch" in out["error"]
