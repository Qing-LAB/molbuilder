"""``lib/chemistry.js`` and ``lib/detection-chip.js`` -- the chemistry card
and each form's chip, under node (`_node_esm.run_node`), each request
answered by the app's own ``/api/structure/analyze`` (`_node_esm.served`),
each answer the card was handed read back from ``__answered``.

Three things are pinned here that no browser test pins well:

  * the SUPERSEDE protocol.  "An answer that arrives after the person loaded
    a different structure, or edited a field again, must not touch the page"
    is a race, and a race is exactly what an end-to-end test cannot schedule.
    What each real request comes to -- the server's answer, or the runtime's
    AbortError once the card let it go -- is HELD until the test releases
    it, so the interleaving is chosen rather than raced for; the answers
    stay the server's, the aborts the runtime's.
  * that an answer for ANOTHER structure never stays on screen: with no
    structure, or no answer, the card and the chips go;
  * what the card and the chip SAY for each form's answer
    (``science/chemistry-correctness.md`` § 2a.5): each engine's own lines,
    each value with the server's own reason, a free moment and a closed
    shell in words.

The card following a typed field, and filling nothing in, is the browser's
to show: ``test_chemistry_card_e2e.py``.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tests._node_esm import run_node, served

REPO = pathlib.Path(__file__).resolve().parent.parent
STATIC = REPO / "molbuilder/web/static/lib"
#: Loaded as the page has them: the card asks through the one reader
#: (server.js) and delegates the chips to detection-chip.js -- a stub that
#: fakes either would test the stub.
MODULES = [STATIC / "server.js", STATIC / "dom.js",
           STATIC / "detection-chip.js", STATIC / "chemistry.js"]

#: A DOM stub with just enough semantics for the modules, and a fetch whose
#: answers -- the server's -- are released BY THE TEST, so request
#: interleaving is chosen rather than raced for.
_STUB = r"""
function El(tag) {
  this.tagName = tag; this.children = []; this.className = "";
  this._text = ""; this.hidden = false;
}
// As in a real DOM, setting the text replaces the children.
Object.defineProperty(El.prototype, "textContent", {
  get: function () { return this._text; },
  set: function (v) { this._text = String(v); this.children = []; },
});
El.prototype.appendChild = function (c) { this.children.push(c); return c; };
El.prototype.addEventListener = function () {};
function text(e) {
  return e.textContent + e.children.map(text).join("");
}

var els = {};
["chemistry-panel", "chemistry-state", "chemistry-metals-box",
 "chemistry-metals", "chemistry-status"].forEach(function (id) {
  els[id] = new El("div"); els[id].hidden = true;
});
globalThis.document = {
  getElementById: function (id) { return els[id] || null; },
  createElement: function (t) { return new El(t); },
};

// Every request goes to the server at once, with the card's own signal; what
// it comes to -- the server's answer, or the runtime's AbortError when the
// card let it go -- parks here until the test releases it, in the order the
// test wants.  ``settled``: what each request came to on the wire.
var pending = [];
var __served = globalThis.fetch;
globalThis.fetch = function (url, opts) {
  var real = __served(url, opts);
  real.catch(function () {});     // parked: handled when released
  return new Promise(function (resolve, reject) {
    pending.push({ url: url, opts: opts, real: real,
                   resolve: resolve, reject: reject });
  });
};
var settled = [];
async function release(i) {
  var p = pending[i];
  try { var r = await p.real; settled[i] = "answered"; p.resolve(r); }
  catch (e) { settled[i] = e.name; p.reject(e); }
}
function out(v) { console.log(JSON.stringify(v)); }
"""


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the card (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


def _run(app_at, script: str):
    """``script`` against the card (``C``) and the chip (``chip``), loaded in
    the page's order onto the stub (:data:`_STUB`), the card's requests
    answered by the app serving at ``app_at``."""
    return run_node(MODULES,
                    "var C = globalThis.molbuilder.chemistry;\n"
                    "var chip = globalThis.molbuilder.detectionChip;\n"
                    + script,
                    globals_js=served(app_at) + _STUB)


#: A repeating cell holding an iron atom and two oxygens -- the structure a
#: page would hand over, in the envelope (`support.envelope`).
def _iron_cell() -> dict:
    import numpy as np

    from support.envelope import envelope
    return envelope(["Fe", "O", "O"],
                    [[2.5, 2.5, 2.5], [4.1, 2.5, 2.5], [0.9, 2.5, 2.5]],
                    cell=np.diag([5.0, 5.0, 5.0]),
                    axis_kind=("periodic",) * 3)


_S = json.dumps(_iron_cell())
#: Two forms saying different things about it: SIESTA's blank -- work it
#: out -- and PySCF's stating a charge of -1, restricted, Hartree-Fock.
_FORMS = {"siesta": {},
          "pyscf": {"net_charge": -1, "spin_treatment": "restricted",
                    "method": "HF"}}
_ASK = json.dumps({"kind": "optimization", "forms": _FORMS})


class TestTheCard:

    def test_each_form_gets_its_own_lines_with_their_reasons(self, app_at):
        """Each engine's block, headed by its name as the server says it:
        the charge with the server's reason, the spin in words -- a closed
        shell said so, a free moment said so -- with each reason, the method
        only where someone chose it, and the electrons of a repeating cell
        said not to be a spin."""
        got = _run(app_at, f"""
            var p = C.analyze({_S}, {_ASK});
            await release(0);
            C.renderPanel((await p).body, {{}});
            out({{hidden: els["chemistry-panel"].hidden,
                  blocks: els["chemistry-state"].children.map(text),
                  answer: __answered[0]}});
        """)
        resp = got["answer"]
        siesta, pyscf = resp["state"]["siesta"], resp["state"]["pyscf"]
        assert siesta["unpaired_electrons"]["value"] == "free", siesta
        assert siesta["method"]["source"] == "rule", siesta
        assert pyscf["spin_treatment"]["value"] == "restricted", pyscf
        assert got["hidden"] is False
        words = resp["engine_words"]
        assert len(got["blocks"]) == len(resp["state"]) == 2, got["blocks"]
        blocks = {engine: next(b for b in got["blocks"]
                               if b.startswith(words[engine]))
                  for engine in resp["state"]}
        for engine, block in blocks.items():
            st = resp["state"][engine]
            assert st["net_charge"]["said"] in block, (engine, block)
            assert st["spin_treatment"]["said"] in block, (engine, block)
        assert "unrestricted, the moment floats (free)" in blocks["siesta"]
        assert "a repeating cell" in blocks["siesta"]
        # a rule is not a choice anyone made: SIESTA's method is not shown
        assert "Method" not in blocks["siesta"]
        assert "-1 — " + pyscf["net_charge"]["said"] in blocks["pyscf"]
        assert "restricted (closed shell, 2S = 0)" in blocks["pyscf"]
        assert ("HF — " + pyscf["method"]["said"]) in blocks["pyscf"]

    def test_the_metals_box_shows_only_when_there_is_a_metal(self, app_at):
        """An empty box left visible is an empty box on the page: the iron
        cell's answer names iron and its usual spins; water's names no
        metal."""
        water = json.dumps(__import__("support.envelope", fromlist=["x"])
                           .envelope(["O", "H", "H"],
                                     [[0, 0, 0], [0.96, 0, 0],
                                      [-0.24, 0.93, 0]]))
        got = _run(app_at, f"""
            var p = C.analyze({_S}, {_ASK});
            await release(0);
            C.renderPanel((await p).body, {{}});
            var a = {{hidden: els["chemistry-metals-box"].hidden,
                     lines: els["chemistry-metals"].children.map(text)}};
            var q = C.analyze({water}, {{kind: "optimization"}});
            await release(1);
            C.renderPanel((await q).body, {{}});
            out({{with: a, without: els["chemistry-metals-box"].hidden,
                  answer: __answered[0]}});
        """)
        hints = got["answer"]["metal_hints"]
        assert [h["element"] for h in hints] == ["Fe"], hints
        assert got["with"]["hidden"] is False
        assert got["with"]["lines"][0] == "Fe"
        assert got["with"]["lines"][1:] == [
            f"2S = {c['spin']} — {c['label']}"
            for c in hints[0]["common_spins"]]
        assert got["without"] is True


class TestNothingStaleStays:

    def test_with_no_structure_the_card_hides_and_asks_nobody(self, app_at):
        got = _run(app_at, f"""
            var p = C.analyze({_S}, {_ASK});
            await release(0);
            C.renderPanel((await p).body, {{}});
            var card = C.attach({{structure: function () {{ return null; }}}});
            await card.refresh();
            out({{hidden: els["chemistry-panel"].hidden,
                  lines: els["chemistry-state"].children.length,
                  calls: pending.length}});
        """)
        assert got == {"hidden": True, "lines": 0, "calls": 1}

    def test_a_failed_answer_takes_the_last_one_away(self, app_at):
        """The server could not answer for THIS structure -- an element it
        cannot read: the card must not go on showing the previous
        structure's answer under the error, which is the server's."""
        bad = json.loads(_S)
        bad["elements"] = ["Xx", "O", "O"]
        got = _run(app_at, f"""
            var which = {_S};
            var card = C.attach({{structure: function () {{ return which; }},
                                 kind: "optimization"}});
            var first = card.refresh();
            await release(0);
            await first;
            var shown = !els["chemistry-panel"].hidden;
            which = {json.dumps(bad)};
            var second = card.refresh();
            await release(1);
            await second;
            out({{shown: shown, hidden: els["chemistry-panel"].hidden,
                  lines: els["chemistry-state"].children.length,
                  status: els["chemistry-status"].textContent,
                  refused: __answered[1]}});
        """)
        refused = got.pop("refused")
        assert refused["ok"] is False, refused
        assert got == {"shown": True, "hidden": True, "lines": 0,
                       "status": refused["error"]}


class TestTheChip:

    def test_each_form_reads_its_own_answer(self, app_at):
        """One form, one chip, from that form's own answer: the two forms
        say different things, so their chips differ.  A zero charge is not
        worth a word; a closed shell is said in words, not as a count."""
        got = _run(app_at, f"""
            var p = C.analyze({_S}, {_ASK});
            await release(0);
            var r = (await p).body;
            out({{siesta: chip.buildText(r, "siesta").profile,
                 pyscf:  chip.buildText(r, "pyscf").profile,
                 answer: __answered[0]}});
        """)
        resp = got.pop("answer")
        assert resp["state"]["siesta"]["net_charge"]["value"] == 0
        assert got == {"siesta": "3 atoms · Fe · unrestricted, moment free",
                       "pyscf": "3 atoms · Fe · closed shell · charge -1"}


class TestTheAnalyzeProtocol:

    def test_a_newer_analyze_supersedes_the_older_one(self, app_at):
        """Two edits: the newer request lets the older one go ON THE WIRE --
        the card's signal reaches the request, and the runtime fails it with
        its own AbortError -- and the older one comes back as superseded,
        never as an error: a scary message flashed on an ordinary second
        edit, or the card repainted with the older answer.  The newer one
        answers first, then the older one's end is let through.

        WHAT NO TEST HERE SCHEDULES: the counter's gate on an answer that
        arrives anyway -- the older body read whole before the abort, its
        resolution not yet run.  A browser queues that resolution as a task,
        so an edit can land between; node resolves it in the same turn."""
        got = _run(app_at, f"""
            var first  = C.analyze({_S}, {_ASK});
            var second = C.analyze({_S}, {_ASK});
            await release(1);     // the newer one answers first
            await release(0);     // ...then the older one's end
            var a = await first, b = await second;
            out({{first: a, second_ok: b.ok, wire: settled}});
        """)
        assert got["first"] == {"ok": False, "superseded": True}
        assert got["second_ok"] is True
        # the older request was killed on the wire, not merely ignored
        assert got["wire"] == ["AbortError", "answered"], got["wire"]
