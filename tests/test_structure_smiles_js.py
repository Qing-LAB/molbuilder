"""Unit tests for the SMILES generator panel module.

Pins the public API of ``molbuilder/web/static/modify/structure/
smiles.js`` — the Sources-card panel that POSTs to
``/api/build/molecule`` and routes the generated structure through the
canvas-state gate.

The panel runs under node (`_node_esm.run_node`), loaded after the shared
wiring as the page loads it; its request is answered by the app's own
`/api/build/molecule` (`_node_esm.served`) -- the real RDKit and 3DNA builds.
The load door is a recorder handed in through ``configure``, so the state
machine runs without a canvas.  The DOM wiring (``wirePanel`` against actual
buttons + inputs) is exercised by an e2e test in test_molbuilder_e2e.py.

The one generate chain is `panel-deps.js`'s, tested here through SMILES.  What
each other panel asks the server is its own: the RNA panel's are the rows of
the table at the end of this file, DNA's and the peptide's their own files'
(`test_structure_dna_js.py`, `test_structure_peptide_js.py`).  The name panel
has no row: its build asks PubChem, which nothing in this suite may reach
(`process/testing.md` § 2b), and what it sends is `generateText`'s, the chain
SMILES is tested through.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node, served


ROOT = Path(__file__).resolve().parents[1]
#: The panel reads its dependency slots from the shared wiring
#: (`panel-deps.js`), which the page loads before it, and asks through the
#: page's one reader (`lib/server.js`, in the shared header).  A harness that
#: skips either is testing a module the browser never runs.
PANEL_DEPS = ROOT / "molbuilder/web/static/modify/structure/panel-deps.js"
SERVER = ROOT / "molbuilder/web/static/lib/server.js"


def _run(snippet: str, app_at: "str | None" = None, *,
         panel: str = "smiles"):
    """``snippet`` against the builder panel ``panel`` -- its exports under
    its own name -- loaded after the shared wiring; with ``app_at``, the
    page's ``fetch`` answered by the app serving there (`_node_esm.served`:
    ``__calls`` each request as it left, ``__answered`` each answer,
    read)."""
    module = (PANEL_DEPS.parent / f"{panel}.js").as_uri()
    return run_node(
        [SERVER, PANEL_DEPS],
        f"const {panel} = (await import({json.dumps(module)})).default;\n"
        f"{snippet}",
        globals_js=served(app_at) if app_at else "")


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the panels (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


class TestLazyResolve:
    """LANDMINE-2 fix (2026-06-07): smiles.js (and the other four
    generator modules) used to capture window.molbuilder.* at IIFE
    eval time.  A future template change that loaded smiles.js
    before page.js / mol-viewer.js would have left the slots null
    forever — first generate() click would reject with "not
    configured".  Pin the lazy-resolve so a later registration is
    picked up."""

    def test_generate_picks_up_slots_registered_after_it_loaded(self,
                                                                app_at):
        """Nothing configured: the load door and the findings renderer are
        registered on the page AFTER the panel loaded, as a template that
        loads smiles.js before page.js leaves them, and the build -- the
        server's -- reaches both."""
        out = _run('''
            const drawn = [];
            smiles.wirePanel({ doc: { getElementById: (id) =>
                (id === "smiles-findings" ? { id } : null) } });
            globalThis.molbuilder.structurePage = {
                loadIntoCanvas: async () => ({ ok: true }) };
            globalThis.molbuilder.validationFindings = {
                render: (rows) => drawn.push(rows) };
            const r = await smiles.generate("C");
            console.log(JSON.stringify({ r, drawn, answer: __answered[0] }));
        ''', app_at)
        answer = out["answer"]
        assert answer["ok"] is True, answer
        # Without the lazy-resolve fix generate() rejects: "smiles:
        # structurePage not configured".
        assert out["r"] == {"ok": True,
                            "n_atoms": len(answer["structure"]["elements"]),
                            "backend_used": answer["backend_used"]}
        assert out["drawn"] == [[], answer["notices"]]


class TestInputValidation:

    def test_nothing_to_build_is_refused_without_a_request(self, app_at):
        """Blank input is refused in the panel's words before anything is
        sent -- a round trip saved and an inline error the person can act
        on."""
        out = _run('''
            const r = await smiles.generate("   \\n\\t  ");
            console.log(JSON.stringify({ r, calls: __calls.length }));
        ''', app_at)
        assert out["r"]["ok"] is False
        assert "Enter a SMILES" in out["r"]["error"]
        assert out["calls"] == 0


class TestHappyPath:

    def test_successful_generate_hands_over_the_envelope(self, app_at):
        """The build's answer reaches the load door WHOLE.

        ``/api/build/molecule`` already built a Structure; it returns it
        under ``structure``, once (`web/web-api.md` § 1).  This panel used
        to install an ``xyz`` rendering of the same atoms, sent beside it
        then, so the server parsed back a flattened copy of what it had
        just built -- and XYZ has no slots for ``residue_names`` or
        ``atom_names``, so a generated peptide arrived with every residue
        named MOL and CA/CB collapsed to C.  The load door must receive
        the server's envelope as it came, and no document: identity
        columns survive an envelope, never a document."""
        out = _run('''
            let canvasArgs = null;
            let loadCalls = 0;
            const drawn = [];
            // The panel wired as the page wires it: its findings list named.
            smiles.wirePanel({ doc: { getElementById: (id) =>
                (id === "smiles-findings" ? { id } : null) } });
            smiles.configure({
                findings: { render: (rows) => drawn.push(rows) },
                structurePage: {
                    loadIntoCanvas: async (struct, src) => {
                        loadCalls++;
                        canvasArgs = {struct, src};
                        return {ok: true};
                    },
                },
            });
            const r = await smiles.generate("CCO");
            console.log(JSON.stringify({
                drawn,
                envelope:  r,
                call:      __calls[0],
                answer:    __answered[0],
                canvas:    canvasArgs,
                loadCalls: loadCalls,
            }));
        ''', app_at)
        answer = out["answer"]
        assert answer["ok"] is True, answer
        assert out["envelope"] == {"ok": True,
                                   "n_atoms": len(answer["structure"]["elements"]),
                                   "backend_used": answer["backend_used"]}
        # the panel's earlier findings cleared as the build starts, then what
        # the build said of what it built, drawn under the panel
        assert out["drawn"] == [[], answer["notices"]]
        assert out["call"]["url"] == "/api/build/molecule"
        assert out["call"]["body"] == {"kind": "smiles", "input": "CCO"}
        # The load door receives the ENVELOPE, whole, as the server sent it
        assert out["canvas"]["struct"]["structure"] == answer["structure"]
        # ...and no coordinate document beside it.
        assert "text" not in out["canvas"]["struct"]
        assert out["canvas"]["src"]["kind"] == "smiles"
        assert out["canvas"]["src"]["generator_input"]["smiles"] == "CCO"
        # ...exactly once — no redundant second load.
        assert out["loadCalls"] == 1

    def test_smiles_input_is_trimmed_before_post(self, app_at):
        """Leading/trailing whitespace is stripped — the server
        side rejects empty input, and a SMILES with surrounding
        space is the same molecule."""
        out = _run('''
            smiles.configure({
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            await smiles.generate("  C  ");
            console.log(JSON.stringify(__calls[0].body));
        ''', app_at)
        assert out["input"] == "C"


class TestErrorPaths:

    def test_http_error_surfaces_message(self, app_at):
        """A SMILES RDKit cannot read: the server's refusal, in its words."""
        out = _run('''
            smiles.configure({
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await smiles.generate("not-a-real-smiles");
            console.log(JSON.stringify({r, answer: __answered[0]}));
        ''', app_at)
        assert out["answer"]["ok"] is False, out
        assert out["r"] == {"ok": False, "error": out["answer"]["error"]}

    def test_network_failure_returns_envelope_not_throws(self):
        """The browser's own failure to reach the server -- no answer --
        said in the one reader's words (`ui-contract.md` § 12)."""
        out = _run('''
            globalThis.fetch = async () => {
                throw new TypeError("Failed to fetch");
            };
            smiles.configure({
                structurePage: {
                    loadIntoCanvas: async () => ({ok: true}),
                },
            });
            const r = await smiles.generate("CCO");
            const read = await globalThis.molbuilder.server.ask(
                smiles.BUILD_URL, { method: "POST", body: {} });
            console.log(JSON.stringify({ r, read }));
        ''')
        assert out["r"]["ok"] is False
        assert out["r"]["error"] == out["read"]["error"]
        # ...plainly, never the browser's own words (molview.md § 6.9).
        assert "could not reach the server" in out["r"]["error"]
        assert "Failed to fetch" not in out["r"]["error"]


class TestCanvasGate:

    def test_canvas_cancel_passes_through_as_cancelled(self, app_at):
        """If the load door returns ``cancelled: true``, the result envelope
        carries ``cancelled: true`` — no ``ok`` success -- and the door is
        asked exactly once.  The chain is `panel-deps.js`'s, shared by the
        five panels: this is its one test."""
        out = _run('''
            let loadCalls = 0;
            smiles.configure({
                structurePage: {
                    loadIntoCanvas: async () => {
                        loadCalls++;
                        return { ok: false, cancelled: true };
                    },
                },
            });
            const r = await smiles.generate("C");
            console.log(JSON.stringify({ r, loadCalls }));
        ''', app_at)
        assert out["r"] == {"ok": False, "cancelled": True}
        assert out["loadCalls"] == 1

    def test_the_build_runs_under_the_fence_and_its_cancel_is_an_answer(
            self, app_at):
        """`tabs.md`, *One chain for the five generators*: the build runs
        under the page's fence with a Cancel that aborts the request, and a
        Cancel pressed is answered ``cancelled`` -- nothing on the canvas,
        no error -- never "Could not reach" a server that was reached."""
        out = _run('''
            const claims = [];
            globalThis.molbuilder.pageBusy = {
                run: async (reason, cancelers, fn) => {
                    claims.push({ reason, cancelers: cancelers.length });
                    const answer = fn();
                    cancelers.forEach((c) => c());     // Cancel, pressed
                    return await answer;
                },
            };
            let loads = 0;
            smiles.configure({
                structurePage: { loadIntoCanvas: async () => {
                    loads += 1; return { ok: true }; } },
            });
            const r = await smiles.generate("C");
            console.log(JSON.stringify({ claims, r, loads }));
        ''', app_at)
        assert out["claims"] == [{
            "reason": "Generating 3-D structure from SMILES…",
            "cancelers": 1}], out
        assert out["r"] == {"ok": False, "cancelled": True}, out
        assert out["loads"] == 0, out


# ----- What each panel asks the server --------------------------- #


def _ask_panel(panel: str, typed: str, opts: dict, app_at) -> dict:
    """What ``panel`` answers for ``typed``, and what it sent and recorded --
    its request answered by the app serving at ``app_at``."""
    return _run(f"""
        let made = null;
        {panel}.configure({{
            structurePage: {{
                loadIntoCanvas: async (s, g) => {{
                    made = g;
                    return {{ ok: true }};
                }},
            }},
        }});
        const said = await {panel}.generate({json.dumps(typed)},
                                            {json.dumps(opts)});
        const sent = __calls[0];
        console.log(JSON.stringify({{
            said, made, body: sent ? sent.body : null,
            answer: __answered[0] || null }}));
    """, app_at, panel=panel)


@pytest.mark.parametrize("panel,typed,opts,asked,record", [
    ("rna", "ACGU", {"form": "A", "backend": "threedna"},
     {"kind": "rna", "input": "ACGU", "form": "A", "backend": "threedna"},
     {"sequence": "ACGU", "form": "A", "backend": "threedna"}),
    ("rna", "5'-ACGU-3'", {},
     {"kind": "rna", "input": "5'-ACGU-3'", "backend": "auto"},
     {"sequence": "5'-ACGU-3'", "backend": "auto"}),
    ("rna", "3'-UGCA", {},
     {"kind": "rna", "input": "3'-UGCA", "backend": "auto"}, None),
], ids=["RNA with its form and backend", "a strand labelled with its ends",
        "a strand missing an end"])
def test_what_a_panel_asks_the_server(panel, typed, opts, asked, record,
                                      app_at):
    """Each panel's whole request, and the record of what made the structure
    (`web/tabs.md` § 2) where the server built it.  THE SILENT FAILURES: a
    chosen form or backend dropped, and the server builds its default with no
    word; a strand's letters sent in place of what was typed: `3'-UGCA`,
    which the server refuses for its missing end -- its refusal said in its
    words and nothing built -- built as if unlabelled, the strand read the
    other way, nothing said."""
    got = _ask_panel(panel, typed, opts, app_at)
    assert got["body"] == asked
    if record is None:
        assert got["answer"]["ok"] is False, got
        assert got["said"] == {"ok": False, "error": got["answer"]["error"]}
        assert got["made"] is None
    else:
        assert got["answer"]["ok"] is True, got
        assert got["made"] == {"kind": panel, "generator_input": record}


def test_dna_thymine_rejected_in_rna_panel(app_at):
    """T is DNA's thymine; RNA uses U.  The letter is refused, naming the
    alphabet, and nothing is sent to be built."""
    got = _ask_panel("rna", "ACGT", {}, app_at)
    assert got["said"]["ok"] is False and "ACGU" in got["said"]["error"]
    assert got["body"] is None
