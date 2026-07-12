"""Unit tests for the file-upload generator panel module.

Pins the public API of ``molbuilder/web/static/lib/structure/
file.js`` — the Sources-card panel that loads an XYZ / PDB file
from the user's disk via FileReader + /api/build/load and routes
the result through the canvas-state gate.

Mirrors the test shape of test_structure_smiles_js.py since
file.js follows the same structural twin pattern (different
endpoint + source kind only).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/structure/file.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        const fileMod = require({json.dumps(str(module_path))});
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
                configure: typeof fileMod.configure,
                loadText:  typeof fileMod.loadText,
                wirePanel: typeof fileMod.wirePanel,
                LOAD_URL:  fileMod.LOAD_URL,
            }));
        ''')
        assert out["configure"] == "function"
        assert out["loadText"]  == "function"
        assert out["wirePanel"] == "function"
        assert out["LOAD_URL"]  == "/api/build/load"


class TestInputValidation:

    def test_empty_text_rejected_without_fetch(self):
        out = _run_node('''
            let fetchCalls = 0;
            fileMod.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await fileMod.loadText("", "water.xyz");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "empty" in out["envelope"]["error"].lower()
        assert out["fetchCalls"] == 0


class TestHappyPath:

    def test_xyz_upload_routes_through_canvas(self):
        """Backend reports source_format=xyz; we route body.xyz
        through loadIntoCanvas with kind="file"."""
        out = _run_node('''
            let capturedBody = null;
            let canvasArgs = null;
            fileMod.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true,
                            xyz: "3\\nH2O\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                            n_atoms: 3,
                            source_format: "xyz",
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
            const r = await fileMod.loadText(
                "3\\nH2O\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n",
                "water.xyz");
            console.log(JSON.stringify({
                envelope: r,
                body:     capturedBody,
                canvas:   canvasArgs,
            }));
        ''')
        assert out["envelope"] == {"ok": True, "n_atoms": 3}
        assert out["body"]["format"] == "auto"
        assert out["body"]["filename"] == "water.xyz"
        assert out["canvas"]["src"]["kind"] == "file"
        assert out["canvas"]["src"]["file"] == "water.xyz"
        assert out["canvas"]["struct"]["source_format"] == "xyz"

    def test_pdb_upload_routes_through_canvas_with_pdb_format(self):
        """Backend reports source_format=pdb + body.pdb populated;
        we route body.pdb (not body.xyz) through loadIntoCanvas."""
        out = _run_node('''
            fileMod.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true,
                        xyz: "5\\nMOL\\nC 0 0 0\\n...\\n",
                        pdb: "HEADER PDB\\nATOM ...\\nEND\\n",
                        n_atoms: 5,
                        source_format: "pdb",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async (struct, src) => ({ok: true,
                        captured: {struct, src}}),
                },
            });
            let canvasArgs = null;
            fileMod.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true,
                        xyz: "5\\nMOL\\nC 0 0 0\\n...\\n",
                        pdb: "HEADER PDB\\nATOM ...\\nEND\\n",
                        n_atoms: 5,
                        source_format: "pdb",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async (struct, src) => {
                        canvasArgs = {struct, src};
                        return {ok: true};
                    },
                },
            });
            await fileMod.loadText("HEADER\\nATOM...", "thing.pdb");
            console.log(JSON.stringify(canvasArgs));
        ''')
        assert out["struct"]["source_format"] == "pdb"
        assert "HEADER" in out["struct"]["text"]


class TestErrorPaths:

    def test_backend_failure_surfaces_message(self):
        out = _run_node('''
            fileMod.configure({
                fetch: async () => ({
                    ok: false,
                    json: async () => ({
                        ok: false,
                        error: "Could not parse as XYZ or PDB",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await fileMod.loadText("garbage", "broken.txt");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "parse" in out["error"]

    def test_canvas_cancel_passes_through_as_cancelled(self):
        """User cancels the dirty-canvas warning modal (load door
        returns cancelled) → envelope carries cancelled, called
        through the single load door exactly once."""
        out = _run_node('''
            let loadCalls = 0;
            fileMod.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "1\\nx\\nC 0 0 0\\n",
                        n_atoms: 1, source_format: "xyz",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => {
                        loadCalls++;
                        return { ok: false, cancelled: true };
                    },
                },
            });
            const r = await fileMod.loadText("1\\nx\\nC 0 0 0\\n", "x.xyz");
            console.log(JSON.stringify({
                envelope:  r,
                loadCalls: loadCalls,
            }));
        ''')
        assert out["envelope"] == {"ok": False, "cancelled": True}
        assert out["loadCalls"] == 1

    def test_network_failure_returns_envelope(self):
        out = _run_node('''
            fileMod.configure({
                fetch: async () => {
                    throw new TypeError("Failed to fetch");
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await fileMod.loadText("1\\nx\\nC 0 0 0\\n", "x.xyz");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "Failed to fetch" in out["error"]
