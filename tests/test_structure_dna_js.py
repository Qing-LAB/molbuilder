"""Unit tests for the DNA generator panel module.

Pins the public API of ``molbuilder/web/static/lib/structure/
dna.js``: validates ACGT-only sequences, accepts a helix form
(B/A/Z), POSTs the right body to /api/build/molecule, routes
the result through structurePage.loadIntoCanvas.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/structure/dna.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    module_path = MODULE.resolve()
    bootstrap = f"""
        const dna = require({json.dumps(str(module_path))});
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
                configure:   typeof dna.configure,
                generate:    typeof dna.generate,
                wirePanel:   typeof dna.wirePanel,
                BUILD_URL:   dna.BUILD_URL,
                VALID_FORMS: dna.VALID_FORMS,
            }));
        ''')
        assert out["configure"] == "function"
        assert out["generate"]  == "function"
        assert out["wirePanel"] == "function"
        assert out["BUILD_URL"] == "/api/build/molecule"
        assert out["VALID_FORMS"] == ["B", "A", "Z"]


class TestInputValidation:

    def test_empty_sequence_rejected(self):
        out = _run_node('''
            let fetchCalls = 0;
            dna.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await dna.generate("");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "DNA sequence" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0

    def test_illegal_bases_rejected_client_side(self):
        """RNA / amino acids / random text MUST be rejected before
        the request hits the network."""
        out = _run_node('''
            let fetchCalls = 0;
            dna.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            // U is RNA, not DNA.
            const r = await dna.generate("AUGCAUGC");
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "ACGT" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0

    def test_invalid_form_rejected_client_side(self):
        out = _run_node('''
            let fetchCalls = 0;
            dna.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await dna.generate("ACGT", {form: "Q"});
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "B, A, Z" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0


class TestHappyPath:

    def test_default_form_is_B(self):
        out = _run_node('''
            let capturedBody = null;
            dna.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "20\\ndna\\nC 0 0 0\\n",
                            n_atoms: 20,
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await dna.generate("ACGT");
            console.log(JSON.stringify(capturedBody));
        ''')
        assert out == {"kind": "dna", "input": "ACGT",
                       "form": "B", "backend": "auto"}

    def test_explicit_form_forwarded(self):
        out = _run_node('''
            let capturedBody = null;
            dna.configure({
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
            // Use GCGC for the Z-form payload check: the client-
            // side Z-only-poly-(GC) guard added 2026-06-07 rejects
            // non-(GC)n sequences with form=Z + auto/threedna
            // backend BEFORE the fetch, so ACGT would have skipped
            // the network call this test is asserting on.  Forcing
            // backend=rdkit also bypasses the guard, but GCGC is
            // the most direct way to exercise the existing assertion.
            await dna.generate("GCGC", {form: "Z"});
            console.log(JSON.stringify(capturedBody));
        ''')
        assert out["form"] == "Z"

    def test_canvas_source_provenance_records_sequence_and_form(self):
        out = _run_node('''
            let canvasArgs = null;
            dna.configure({
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
            await dna.generate("ACGT", {form: "A"});
            console.log(JSON.stringify(canvasArgs));
        ''')
        assert out["src"]["kind"] == "dna"
        assert out["src"]["generator_input"]["sequence"] == "ACGT"
        assert out["src"]["generator_input"]["form"] == "A"


class TestBackendSelection:
    """Backend selector for DNA generator panel (restored 2026-06-07
    after user reported it was missing from the new Sources card
    UI).  Legacy index.html had a #backend select inside
    #nucleic-options offering auto/threedna/amber/rdkit; the new
    DNA panel was migrated without it, defaulting silently to
    auto.  The selector is back + plumbed through generate()."""

    def test_default_backend_is_auto(self):
        out = _run_node('''
            let capturedBody = null;
            dna.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "20\\ndna\\nC 0 0 0\\n",
                            n_atoms: 20, backend_used: "rdkit",
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await dna.generate("ACGT");
            console.log(JSON.stringify(capturedBody.backend));
        ''')
        assert out == "auto"

    def test_explicit_backend_forwarded(self):
        out = _run_node('''
            let capturedBody = null;
            dna.configure({
                fetch: async (url, init) => {
                    capturedBody = JSON.parse(init.body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "5\\nx\\nC 0 0 0\\n",
                            n_atoms: 5, backend_used: "rdkit",
                        }),
                    };
                },
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await dna.generate("ACGT", {backend: "rdkit"});
            console.log(JSON.stringify(capturedBody.backend));
        ''')
        assert out == "rdkit"

    def test_unknown_backend_rejected_client_side(self):
        out = _run_node('''
            let fetchCalls = 0;
            dna.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await dna.generate("ACGT", {backend: "bogus"});
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "Backend must be one of" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0

    def test_backend_used_propagated_through_envelope(self):
        """If the server resolves auto to a different name, the
        client envelope must surface backend_used so the status
        text can show which backend actually ran."""
        out = _run_node('''
            dna.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "5\\nx\\nC 0 0 0\\n",
                        n_atoms: 5, backend_used: "threedna",
                    }),
                }),
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await dna.generate("GCGC", {backend: "auto"});
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is True
        assert out["backend_used"] == "threedna"


class TestZFormGuard:
    """Bug #2 fix (2026-06-07): Z-DNA via 3DNA's ``fiber`` only
    supports alternating poly-d(GC) sequences (GCGC, CGCGCG, ...).
    Other sequences cause fiber to fall into an interactive
    'Number of repeats' prompt that stalls reading from stdin
    until the 60-second subprocess timeout — perceived as a 'hang'
    by the user.  Catch upfront."""

    def test_z_form_with_non_gc_sequence_rejected_client_side(self):
        out = _run_node('''
            let fetchCalls = 0;
            dna.configure({
                fetch: async () => { fetchCalls++; return {}; },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await dna.generate("ATGC", {form: "Z"});
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is False
        assert "Z-DNA" in out["envelope"]["error"]
        assert out["fetchCalls"] == 0, (
            "Z + non-(GC)n must not even reach the server"
        )

    def test_z_form_with_gcgc_passes_guard(self):
        out = _run_node('''
            let fetchCalls = 0;
            dna.configure({
                fetch: async (url, init) => {
                    fetchCalls++;
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "5\\nx\\nC 0 0 0\\n",
                            n_atoms: 5, backend_used: "threedna",
                        }),
                    };
                },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await dna.generate("GCGCGC", {form: "Z"});
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is True
        assert out["fetchCalls"] == 1

    def test_z_form_with_rdkit_backend_skips_guard(self):
        """The Z-DNA guard only kicks in for the threedna/auto
        backends — RDKit ignores form anyway, so blocking Z would
        force the user to pick A or B for a request that's already
        producing a folded conformer."""
        out = _run_node('''
            let fetchCalls = 0;
            dna.configure({
                fetch: async () => {
                    fetchCalls++;
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true, xyz: "5\\nx\\nC 0 0 0\\n",
                            n_atoms: 5, backend_used: "rdkit",
                        }),
                    };
                },
                structurePage: { loadIntoCanvas: async () => ({ok: true}) },
            });
            const r = await dna.generate("ATGC", {
                form: "Z", backend: "rdkit",
            });
            console.log(JSON.stringify({envelope: r, fetchCalls}));
        ''')
        assert out["envelope"]["ok"] is True
        assert out["fetchCalls"] == 1


class TestErrorPaths:

    def test_backend_failure_surfaces_message(self):
        out = _run_node('''
            dna.configure({
                fetch: async () => ({
                    ok: false,
                    json: async () => ({
                        ok: false,
                        error: "3DNA not installed: rebuild backend",
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await dna.generate("ACGT");
            console.log(JSON.stringify(r));
        ''')
        assert out["ok"] is False
        assert "3DNA" in out["error"]

    def test_canvas_cancel_skips_viewer(self):
        out = _run_node('''
            let viewerCalls = 0;
            dna.configure({
                fetch: async () => ({
                    ok: true,
                    json: async () => ({
                        ok: true, xyz: "1\\nx\\nC 0 0 0\\n",
                        n_atoms: 1,
                    }),
                }),
                structurePage: {
                    loadIntoCanvas: async () => ({
                        ok: false, cancelled: true,
                    }),
                },
                viewerLoader: () => { viewerCalls++; return Promise.resolve(); },
            });
            const r = await dna.generate("A");
            console.log(JSON.stringify({envelope: r, viewerCalls}));
        ''')
        assert out["envelope"] == {"ok": False, "cancelled": True}
        assert out["viewerCalls"] == 0
