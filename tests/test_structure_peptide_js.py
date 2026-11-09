"""Unit tests for the peptide generator panel module.

Pins the public API of ``molbuilder/web/static/modify/structure/
peptide.js`` — the Sources-card panel that POSTs to
``/api/build/molecule`` with kind="peptide" and routes the
generated structure through the canvas-state gate.

The panel runs under node (`_node_esm.run_node`), wired as the page wires
it, its request answered by the app's own `/api/build/molecule`
(`_node_esm.served`).  What the generate chain does with the load door's
answer is `panel-deps.js`'s, tested through SMILES
(`test_structure_smiles_js.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node, served


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/modify/structure/peptide.js"
#: The panel reads its dependency slots from the shared wiring
#: (`panel-deps.js`), which the page loads before it, and asks through the
#: page's one reader (`lib/server.js`, in the shared header).  A harness that
#: skips either is testing a module the browser never runs.
PANEL_DEPS = ROOT / "molbuilder/web/static/modify/structure/panel-deps.js"
SERVER = ROOT / "molbuilder/web/static/lib/server.js"

#: The panel wired as the page wires it -- its findings list named
#: (`wirePanel`), the one renderer a recorder (`drawn`) -- its builds asked
#: of the server and handed to a load door that takes them.
WIRED = """
    const list = {id: "peptide-findings"};
    pep.wirePanel({doc: {getElementById: (id) =>
        id === "peptide-findings" ? list
            : {addEventListener() {}, value: "", checked: false}}});
    const drawn = [];
    pep.configure({
        structurePage: { loadIntoCanvas: async () => ({ok: true}) },
        findings: {render: (rows, opts) => drawn.push(
            {rows: rows.map(r => r.where), under: opts.panel === list})},
    });
"""


def _run(snippet: str, app_at: str) -> object:
    """``snippet`` against the panel (``pep``), wired as the page wires it
    (:data:`WIRED`), the page's ``fetch`` answered by the app serving at
    ``app_at`` (`_node_esm.served`: ``__calls`` each request as it left,
    ``__answered`` each answer, read)."""
    return run_node(
        [SERVER, PANEL_DEPS],
        f"const pep = (await import({json.dumps(MODULE.as_uri())})).default;\n"
        f"{WIRED}\n{snippet}",
        globals_js=served(app_at))


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the panel (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


class TestInputValidation:

    @pytest.mark.parametrize("seq, said", [
        ("", "peptide sequence"),
        ("ACBDEF", "one-letter amino-acid"),
    ], ids=["no sequence", "B is no amino acid's letter"])
    def test_what_cannot_be_built_is_refused_before_any_request(
            self, app_at, seq, said):
        """One-letter codes outside the 20 canonical amino acids
        (B, J, O, U, X, Z plus non-alpha chars) MUST be rejected
        before the request hits the network — the backend's
        tleap call would fail with a less actionable error."""
        out = _run(f'''
            const r = await pep.generate({json.dumps(seq)});
            console.log(JSON.stringify({{r, calls: __calls.length}}));
        ''', app_at)
        assert out["r"]["ok"] is False
        assert said in out["r"]["error"]
        assert out["calls"] == 0

    def test_lowercase_sequence_accepted_and_uppercased(self, app_at):
        out = _run('''
            await pep.generate("aaaa");
            console.log(JSON.stringify({body: __calls[0].body,
                                        answer: __answered[0]}));
        ''', app_at)
        assert out["body"]["input"] == "AAAA"
        assert out["answer"]["ok"] is True, out["answer"]


class TestHappyPath:

    def test_successful_generate_routes_through_canvas(self, app_at):
        """The peptide is handed over WHOLE.

        This is the panel where the loss was measured: installing the
        ``xyz`` string beside the envelope put every residue through a
        format with no residue column, so ``build_peptide("AG")`` --
        ALA x5, GLY x4 -- arrived as nineteen residues all named MOL,
        with CA/CB collapsed to C.  ``by_residue_name "ALA"`` then
        matched nothing on a peptide the user had just generated.  The
        server's build of ACD names its residues; the load door receives
        that envelope as it came, the document beside it left out."""
        out = _run('''
            let canvasArgs = null;
            pep.configure({
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
                body:     __calls[0].body,
                canvas:   canvasArgs,
                answer:   __answered[0],
            }));
        ''', app_at)
        answer = out["answer"]
        assert answer["ok"] is True and "text" in answer, answer
        assert set(answer["structure"]["residue_names"]) >= {"ALA", "CYS",
                                                             "ASP"}
        assert out["envelope"] == {"ok": True, "n_atoms": answer["n_atoms"],
                                   "backend_used": answer["backend_used"]}
        assert out["body"] == {"kind": "peptide", "input": "ACD"}
        # The residues survive, because an envelope has a slot for them
        # and a coordinate document does not.
        assert out["canvas"]["struct"]["structure"] == answer["structure"]
        assert "text" not in out["canvas"]["struct"]
        assert out["canvas"]["src"]["kind"] == "peptide"
        assert out["canvas"]["src"]["generator_input"]["sequence"] == "ACD"
