"""Unit tests for the DNA generator panel module.

Pins the DNA panel, ``molbuilder/web/static/modify/structure/dna.js``:
what it refuses before any request (no sequence, a letter not ACGT, Z-DNA
on a sequence 3DNA cannot build it on), the form and backend it sends the
server, and what the build's answer leaves in its envelope and under the
panel.

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
MODULE = ROOT / "molbuilder/web/static/modify/structure/dna.js"
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
    const list = {id: "dna-findings"};
    dna.wirePanel({doc: {getElementById: (id) =>
        id === "dna-findings" ? list
            : {addEventListener() {}, value: "", checked: false}}});
    const drawn = [];
    dna.configure({
        structurePage: { loadIntoCanvas: async () => ({ok: true}) },
        findings: {render: (rows, opts) => drawn.push(
            {rows: rows.map(r => r.where), under: opts.panel === list})},
    });
"""


def _run(snippet: str, app_at: str) -> object:
    """``snippet`` against the panel (``dna``), wired as the page wires it
    (:data:`WIRED`), the page's ``fetch`` answered by the app serving at
    ``app_at`` (`_node_esm.served`: ``__calls`` each request as it left,
    ``__answered`` each answer, read)."""
    return run_node(
        [SERVER, PANEL_DEPS],
        f"const dna = (await import({json.dumps(MODULE.as_uri())})).default;\n"
        f"{WIRED}\n{snippet}",
        globals_js=served(app_at))


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the panel (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


@pytest.mark.parametrize("seq, opts, said", [
    ("", {}, "DNA sequence"),
    ("AUGCAUGC", {}, "ACGT"),
    ("ATGC", {"form": "Z"}, "Z-DNA"),
], ids=["no sequence", "U is RNA's, not DNA's",
        "Z-DNA on a sequence not poly-(GC)"])
def test_what_cannot_be_built_is_refused_before_any_request(app_at, seq,
                                                            opts, said):
    """The panel's own checks, each in its words, before the request
    leaves -- the backend would fail with a less actionable error.

    Z-DNA (bug #2, 2026-06-07): 3DNA's ``fiber`` builds Z only for
    alternating poly-d(GC) (GCGC, CGCGCG, ...); on any other sequence it
    fell into an interactive 'Number of repeats' prompt that stalled reading
    stdin until the 60-second subprocess timeout -- a 'hang' to the user.
    Caught upfront: it must not even reach the server."""
    out = _run(f'''
        const r = await dna.generate({json.dumps(seq)}, {json.dumps(opts)});
        console.log(JSON.stringify({{r, calls: __calls.length}}));
    ''', app_at)
    assert out["r"]["ok"] is False
    assert said in out["r"]["error"]
    assert out["calls"] == 0


class TestHappyPath:

    def test_explicit_form_forwarded(self, app_at):
        """GCGC for the Z-form payload: the client-side Z-only-poly-(GC)
        guard rejects other sequences with form=Z and the auto/threedna
        backend before the request leaves."""
        out = _run('''
            await dna.generate("GCGC", {form: "Z"});
            console.log(JSON.stringify({body: __calls[0].body,
                                        answer: __answered[0]}));
        ''', app_at)
        assert out["body"]["form"] == "Z"
        assert out["answer"]["ok"] is True, out["answer"]


class TestBackendSelection:
    """Backend selector for the DNA generator panel, plumbed through
    generate(), each build the server's."""

    @pytest.mark.parametrize("opts, sent", [
        ({}, "auto"), ({"backend": "rdkit"}, "rdkit"),
    ], ids=["the default is auto", "an explicit backend is forwarded"])
    def test_the_backend_asked_is_sent(self, app_at, opts, sent):
        out = _run(f'''
            await dna.generate("ACGT", {json.dumps(opts)});
            console.log(JSON.stringify({{body: __calls[0].body,
                                         answer: __answered[0]}}));
        ''', app_at)
        assert out["body"]["backend"] == sent
        assert out["answer"]["ok"] is True, out["answer"]

    def test_backend_used_propagated_through_envelope(self, app_at):
        """When the server resolves auto to a backend, the client envelope
        must surface backend_used so the status text can show which backend
        actually ran -- and what the build found (its notices) is drawn
        under the panel through the one renderer, the panel's earlier
        findings cleared first (`science/validation.md` § 4.1 R5;
        `web/tabs.md`, *One chain for the five generators*).  Wired as the
        page wires it (`wirePanel`), the renderer a recorder through the
        test door."""
        out = _run('''
            const r = await dna.generate("GCGC", {backend: "auto"});
            console.log(JSON.stringify({r, drawn, answer: __answered[0]}));
        ''', app_at)
        answer = out["answer"]
        assert answer["ok"] is True and answer["notices"], answer
        assert out["r"] == {"ok": True,
                            "n_atoms": len(answer["structure"]["elements"]),
                            "backend_used": answer["backend_used"]}, out["r"]
        assert out["drawn"] == [
            {"rows": [], "under": True},
            {"rows": [n["where"] for n in answer["notices"]],
             "under": True}], out["drawn"]


@pytest.mark.parametrize("seq, opts", [
    ("GCGCGC", {"form": "Z"}),
    ("ATGC", {"form": "Z", "backend": "rdkit"}),
], ids=["poly-(GC) passes the guard",
        "RDKit, which ignores the form, skips it"])
def test_what_the_z_guard_lets_through_is_built(app_at, seq, opts):
    """The Z-DNA guard only kicks in for the threedna/auto backends --
    RDKit ignores form anyway, so blocking Z would force the user to
    pick A or B for a request that's already producing a folded
    conformer.  What passes reaches the server once, and is built."""
    out = _run(f'''
        const r = await dna.generate({json.dumps(seq)}, {json.dumps(opts)});
        console.log(JSON.stringify({{envelope: r, calls: __calls.length}}));
    ''', app_at)
    assert out["envelope"]["ok"] is True, out["envelope"]
    assert out["calls"] == 1
