"""Unit tests for the file-upload generator panel module.

Pins the public API of ``molbuilder/web/static/modify/structure/
file.js`` — the Sources-card panel that loads an XYZ / PDB file
from the user's disk via FileReader and installs it through the
canvas-state load door (molview.data.installMolecule) -- ONE parse,
no pre-POST to /api/build/load.

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
MODULE = ROOT / "molbuilder/web/static/modify/structure/file.js"


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
            }));
        ''')
        assert out["configure"] == "function"
        assert out["loadText"]  == "function"
        assert out["wirePanel"] == "function"


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

    def test_xyz_upload_routes_raw_text_through_canvas(self):
        """The RAW uploaded text is handed to loadIntoCanvas ONCE (no pre-POST
        to /api/build/load); n_atoms is read off the loaded model."""
        out = _run_node('''
            let canvasArgs = null;
            globalThis.molbuilder = { molview: { data: {
                getStructure: () => ({ atoms: [{}, {}, {}] }),
            } } };
            fileMod.configure({
                structurePage: {
                    loadIntoCanvas: async (struct, src) => {
                        canvasArgs = {struct, src};
                        return {ok: true};
                    },
                },
            });
            const RAW = "3\\nH2O\\nO 0 0 0\\nH 1 0 0\\nH 0 1 0\\n";
            const r = await fileMod.loadText(RAW, "water.xyz");
            console.log(JSON.stringify({
                envelope: r, canvas: canvasArgs, raw: RAW,
            }));
        ''')
        assert out["envelope"] == {"ok": True, "n_atoms": 3}
        # RAW text handed straight to the load door -- ONE parse, no canonicalised copy.
        assert out["canvas"]["struct"]["text"] == out["raw"]
        assert out["canvas"]["src"]["kind"] == "file"
        assert out["canvas"]["src"]["file"] == "water.xyz"

    def test_pdb_upload_routes_raw_text_through_canvas(self):
        """A .pdb upload hands its RAW text to loadIntoCanvas (the server sniffs
        PDB from the filename); no body.pdb canonicalisation step."""
        out = _run_node('''
            let canvasArgs = null;
            globalThis.molbuilder = { molview: { data: {
                getStructure: () => ({ atoms: [{}, {}, {}, {}, {}] }),
            } } };
            fileMod.configure({
                structurePage: {
                    loadIntoCanvas: async (struct, src) => {
                        canvasArgs = {struct, src};
                        return {ok: true};
                    },
                },
            });
            const RAW = "HEADER PDB\\nATOM ...\\nEND\\n";
            const r = await fileMod.loadText(RAW, "thing.pdb");
            console.log(JSON.stringify({ envelope: r, canvas: canvasArgs }));
        ''')
        assert out["envelope"]["ok"] is True
        assert out["canvas"]["src"]["file"] == "thing.pdb"
        assert "HEADER" in out["canvas"]["struct"]["text"]


class TestErrorPaths:

    def test_parse_failure_surfaces_message(self):
        """A parse failure now comes from installMolecule (via loadIntoCanvas)
        rejecting; loadText's .catch turns it into an {ok:false, error} envelope."""
        out = _run_node('''
            fileMod.configure({
                structurePage: {
                    loadIntoCanvas: async () => {
                        throw new Error("Could not parse as XYZ or PDB");
                    },
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

    def test_load_door_rejection_returns_envelope(self):
        """A network / other rejection from the load door (installMolecule) is
        caught into the {ok:false, error} envelope, not left unhandled."""
        out = _run_node('''
            fileMod.configure({
                structurePage: {
                    loadIntoCanvas: async () => {
                        throw new TypeError("Failed to fetch");
                    },
                },
            });
            const r = await fileMod.loadText("1\\nx\\nC 0 0 0\\n", "x.xyz");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "Failed to fetch" in out["error"]
