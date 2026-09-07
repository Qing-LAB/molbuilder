"""Unit tests for the peptide generator panel module.

Pins the public API of ``molbuilder/web/static/modify/structure/
peptide.js`` — the Sources-card panel that POSTs to
``/api/build/molecule`` with kind="peptide" and routes the
generated XYZ through the canvas-state gate.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/structure/peptide.js"
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
        const pep = require({json.dumps(str(module_path))});
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
                configure: typeof pep.configure,
                generate:  typeof pep.generate,
                wirePanel: typeof pep.wirePanel,
                BUILD_URL: pep.BUILD_URL,
            }));
        ''')
        assert out["configure"] == "function"
        assert out["generate"]  == "function"
        assert out["wirePanel"] == "function"
        assert out["BUILD_URL"] == "/api/build/molecule"


class TestInputValidation:

    def test_empty_sequence_rejected_without_fetch(self):
        out = _run_node('''
            let fetchCalls = 0;
            pep.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await pep.generate("");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "sequence" in out["envelope"]["error"].lower()
        assert out["fetchCalls"] == 0

    def test_illegal_codes_rejected_client_side(self):
        """One-letter codes outside the 20 canonical amino acids
        (B, J, O, U, X, Z plus non-alpha chars) MUST be rejected
        before the request hits the network — the backend's
        tleap call would fail with a less actionable error."""
        out = _run_node('''
            let fetchCalls = 0;
            pep.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await pep.generate("ACBDEF");  // B is illegal
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "one-letter amino-acid" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0

    def test_lowercase_sequence_accepted_and_uppercased(self):
        out = _run_node('''
            let capturedBody = null;
            pep.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "1\\nA\\nC 0 0 0\\n",
                            n_atoms: 1,
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await pep.generate("aaaa");
            console.log(JSON.stringify(capturedBody));
        ''')
        assert out["input"] == "AAAA"


class TestHappyPath:

    def test_successful_generate_routes_through_canvas(self):
        """The peptide is handed over WHOLE.

        This is the panel where the loss was measured: installing the
        ``xyz`` string beside the envelope put every residue through a
        format with no residue column, so ``build_peptide("AG")`` --
        ALA x5, GLY x4 -- arrived as nineteen residues all named MOL,
        with CA/CB collapsed to C.  ``by_residue_name "ALA"`` then
        matched nothing on a peptide the user had just generated."""
        out = _run_node('''
            let capturedBody = null;
            let canvasArgs = null;
            pep.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true,
                            structure: {
                                title: "ACD peptide",
                                elements: ["N", "C", "C"],
                                positions: [[0,0,0], [1,0,0], [2,0,0]],
                                residue_names: ["ALA", "CYS", "ASP"],
                                atom_names: ["N", "CA", "CB"],
                            },
                            // The flattened rendering rides along and is
                            // deliberately NOT what gets installed.
                            xyz: "23\\nACD peptide\\nN 0 0 0\\n...\\n",
                            n_atoms: 23,
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
            const r = await pep.generate("ACD");
            console.log(JSON.stringify({
                envelope: r,
                body:     capturedBody,
                canvas:   canvasArgs,
            }));
        ''')
        assert out["envelope"] == {"ok": True, "n_atoms": 23}
        assert out["body"] == {"kind": "peptide", "input": "ACD"}
        # The residues survive, because an envelope has a slot for them
        # and a coordinate document does not.
        env = out["canvas"]["struct"]["structure"]
        assert env["residue_names"] == ["ALA", "CYS", "ASP"]
        assert env["atom_names"] == ["N", "CA", "CB"]
        assert "text" not in out["canvas"]["struct"]
        assert out["canvas"]["src"]["kind"] == "peptide"
        assert out["canvas"]["src"]["generator_input"]["sequence"] == "ACD"


class TestErrorPaths:

    def test_backend_failure_surfaces_message(self):
        out = _run_node('''
            pep.configure({
                fetch: async () => ({
                    ok: false,
                    json: async () => ({
                        ok: false,
                        error: "tleap failed: residue ACE not parametrized",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await pep.generate("AAA");
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
            pep.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "1\\nA\\nC 0 0 0\\n", n_atoms: 1,
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => {
                        loadCalls++;
                        return { ok: false, cancelled: true };
                    },
                },
            });
            const r = await pep.generate("A");
            console.log(JSON.stringify({
                envelope:  r,
                loadCalls: loadCalls,
            }));
        ''')
        assert out["envelope"] == {"ok": False, "cancelled": True}
        assert out["loadCalls"] == 1
