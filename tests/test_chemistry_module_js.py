"""``lib/chemistry.js`` and ``lib/detection-chip.js`` -- the chemistry card
and each form's chip, under node.

Three things are pinned here that no browser test pins well:

  * the SUPERSEDE protocol.  "An answer that arrives after the person loaded
    a different structure, or edited a field again, must not touch the page"
    is a race, and a race is exactly what an end-to-end test cannot schedule.
    Driving ``fetch`` by hand makes the interleaving deterministic.
  * that an answer for ANOTHER structure never stays on screen: with no
    structure, or no answer, the card and the chips go;
  * what the card and the chip SAY for each form's answer
    (``science/chemistry-correctness.md`` § 2a.5): each engine's own lines,
    each value with the server's own reason, a free moment and a closed
    shell in words.

The card following a typed field, and filling nothing in, is the browser's
to show: ``test_chemistry_card_e2e.py``.  (This file replaced
``test_auto_detect_module_js.py`` on 2026-09-28, with the module it tested.)
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
STATIC = REPO / "molbuilder/web/static/lib"
#: Loaded in the page's order: the card delegates the failed-fetch sentence to
#: fetch-error.js and the chips to detection-chip.js -- a stub that fakes
#: either would test the stub.
MODULES = [STATIC / "fetch-error.js", STATIC / "dom.js",
           STATIC / "detection-chip.js", STATIC / "chemistry.js"]

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")

#: A DOM stub with just enough semantics for the modules, plus a fetch whose
#: responses are resolved BY THE TEST so request interleaving is chosen
#: rather than raced for.
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

// Every fetch parks here; the test resolves them in the order it wants.
var pending = [];
globalThis.fetch = function (url, opts) {
  return new Promise(function (resolve, reject) {
    pending.push({ url: url, opts: opts, resolve: resolve, reject: reject });
  });
};
function reply(i, status, body) {
  pending[i].resolve({
    ok: status >= 200 && status < 300,
    status: status,
    json: function () {
      return (body === "__nonjson__")
        ? Promise.reject(Object.assign(new SyntaxError("Unexpected token <"),
                                       { name: "SyntaxError" }))
        : Promise.resolve(body);
    },
  });
}
var aborted = [];
globalThis.AbortController = function () {
  var self = this;
  this.signal = { id: aborted.length };
  this.abort = function () { aborted.push(self.signal.id); };
};
function out(v) { console.log(JSON.stringify(v)); }
"""


def _run(script: str):
    src = "\n".join(m.read_text(encoding="utf-8") for m in MODULES)
    prog = (f"{_STUB}\n{src}\n"
            f"var C = globalThis.molbuilder.chemistry;\n"
            f"var chip = globalThis.molbuilder.detectionChip;\n"
            f"(async function () {{\n{script}\n}})();")
    res = subprocess.run(["node", "-e", prog], capture_output=True,
                         text=True, timeout=60)
    assert res.returncode == 0, f"node failed:\n{res.stderr}"
    return json.loads(res.stdout.strip().splitlines()[-1])


def _item(value, source, why=None):
    """One item as the server sends it -- the class's own serialisation, so
    the fixture cannot drift from the route's shape."""
    from molbuilder.electronic_state import Resolved
    return Resolved(value, source, why or source).as_dict()


#: The structure a page would hand over -- any envelope; the server is faked.
_S = "{elements: ['Fe'], positions: [[0, 0, 0]], metadata: {}}"


#: Two forms saying different things about one repeating cell with an iron
#: atom: SIESTA's form is blank (a floating moment, detected), PySCF's says
#: charge -1 and closed shell.  The same shape `/api/structure/analyze`
#: returns (`ElectronicState.as_dict`).
_RESP = json.dumps({
    "ok": True, "n_atoms": 3, "metals": ["Fe"],
    "metal_hints": [{"element": "Fe",
                     "common_spins": [{"spin": 4, "label": "high-spin"}]}],
    "state": {
        "siesta": {
            "net_charge": _item(0, "detected",
                                "no deprotonated phosphate groups"),
            "spin_treatment": _item("unrestricted", "detected",
                                    "Fe is an open-d metal in a repeating "
                                    "cell"),
            "unpaired_electrons": _item("free", "detected",
                                        "Fe is an open-d metal in a "
                                        "repeating cell"),
            "method": _item("DFT", "rule",
                            "SIESTA is a density-functional code"),
            "n_electrons": 42, "finite": False,
        },
        "pyscf": {
            "net_charge": _item(-1, "stated"),
            "spin_treatment": _item("restricted", "stated"),
            "unpaired_electrons": _item(0, "implied",
                                        "restricted: every electron paired"),
            "method": _item("HF", "stated"),
            "n_electrons": 43, "finite": True,
        },
    },
})


class TestTheCard:

    def test_each_form_gets_its_own_lines_with_their_reasons(self):
        got = _run(f"""
            C.renderPanel({_RESP}, {{}});
            var blocks = els["chemistry-state"].children.map(text);
            out({{hidden: els["chemistry-panel"].hidden, blocks: blocks}});
        """)
        assert got["hidden"] is False
        siesta, pyscf = got["blocks"]
        assert siesta.startswith("SIESTA") and pyscf.startswith("PySCF")
        assert "0 — detected: no deprotonated phosphate groups" in siesta
        assert "unrestricted, the moment floats (free)" in siesta
        assert "a repeating cell" in siesta
        # a rule is not a choice anyone made: SIESTA's method is not shown
        assert "Method" not in siesta
        assert "-1 — stated" in pyscf
        assert "restricted (closed shell, 2S = 0)" in pyscf
        # the treatment was stated and the count implied: both reasons
        assert ("stated; the count: implied: restricted: every electron "
                "paired") in pyscf
        assert "HF — stated" in pyscf

    def test_the_metals_box_shows_only_when_there_is_a_metal(self):
        """An empty box left visible is an empty box on the page."""
        got = _run(f"""
            C.renderPanel({_RESP}, {{}});
            var a = {{hidden: els["chemistry-metals-box"].hidden,
                     lines: els["chemistry-metals"].children.map(text)}};
            var none = {_RESP};
            none.metal_hints = [];
            C.renderPanel(none, {{}});
            out({{with: a, without: els["chemistry-metals-box"].hidden}});
        """)
        assert got["with"] == {"hidden": False,
                               "lines": ["Fe", "2S = 4 — high-spin"]}
        assert got["without"] is True


class TestNothingStaleStays:

    def test_with_no_structure_the_card_hides_and_asks_nobody(self):
        got = _run(f"""
            C.renderPanel({_RESP}, {{}});
            var card = C.attach({{structure: function () {{ return null; }}}});
            await card.refresh();
            out({{hidden: els["chemistry-panel"].hidden,
                  lines: els["chemistry-state"].children.length,
                  calls: pending.length}});
        """)
        assert got == {"hidden": True, "lines": 0, "calls": 0}

    def test_a_failed_answer_takes_the_last_one_away(self):
        """The server could not answer for THIS structure: the card must
        not go on showing the previous structure's answer under the error."""
        got = _run(f"""
            var card = C.attach({{structure: function () {{ return {_S}; }}}});
            var first = card.refresh();
            reply(0, 200, {_RESP});
            await first;
            var shown = !els["chemistry-panel"].hidden;
            var second = card.refresh();
            reply(1, 400, {{ok: false, error: "unreadable element symbol 'Xx'"}});
            await second;
            out({{shown: shown, hidden: els["chemistry-panel"].hidden,
                  lines: els["chemistry-state"].children.length,
                  status: els["chemistry-status"].textContent}});
        """)
        assert got == {"shown": True, "hidden": True, "lines": 0,
                       "status": "unreadable element symbol 'Xx'"}


class TestTheChip:

    def test_each_form_reads_its_own_answer(self):
        """One form, one chip, from that form's own answer: the two forms
        say different things, so their chips differ.  A zero charge is not
        worth a word; a closed shell is said in words, not as a count."""
        got = _run(f"""
            out({{siesta: chip.buildText({_RESP}, "siesta").profile,
                 pyscf:  chip.buildText({_RESP}, "pyscf").profile}});
        """)
        assert got == {"siesta": "3 atoms · Fe · unrestricted, moment free",
                       "pyscf": "3 atoms · Fe · closed shell · charge -1"}


class TestTheAnalyzeProtocol:

    def test_the_forms_and_the_kind_travel_with_the_structure(self):
        got = _run(f"""
            var p = C.analyze({_S}, {{kind: "vibration",
                                     forms: {{pyscf: {{net_charge: -1}}}}}});
            reply(0, 200, {{ok: true, state: {{}}}});
            var res = await p;
            out({{ok: res.ok, sent: JSON.parse(pending[0].opts.body)}});
        """)
        assert got == {"ok": True,
                       "sent": {"structure": {"elements": ["Fe"],
                                              "positions": [[0, 0, 0]],
                                              "metadata": {}},
                                "kind": "vibration",
                                "forms": {"pyscf": {"net_charge": -1}}}}

    def test_a_newer_analyze_supersedes_the_older_one(self):
        """The race the counter exists for: two edits, and the FIRST server
        answer arrives last.  Without the gate it would repaint the card
        with the superseded answer."""
        got = _run(f"""
            var first  = C.analyze({_S});
            var second = C.analyze({_S});
            reply(1, 200, {{ok: true}});     // the newer one answers first
            reply(0, 200, {{ok: true}});     // ...then the stale one
            var a = await first, b = await second;
            out({{first: a, second_ok: b.ok, aborted: aborted.length}});
        """)
        assert got["first"] == {"ok": False, "superseded": True}
        assert got["second_ok"] is True
        # the older request was killed on the wire, not merely ignored
        assert got["aborted"] == 1

    def test_the_servers_own_message_is_what_the_person_gets(self):
        got = _run(f"""
            var p = C.analyze({_S});
            reply(0, 400, {{ok: false, error: "unreadable element symbol 'Xx'"}});
            out(await p);
        """)
        assert got == {"ok": False,
                       "error": "unreadable element symbol 'Xx'"}

    def test_an_error_page_says_so_instead_of_unexpected_token(self):
        """A 5xx HTML page parses as neither JSON nor a server message.
        "Unexpected token <" tells a chemist nothing about what broke."""
        got = _run(f"""
            var p = C.analyze({_S});
            reply(0, 500, "__nonjson__");
            out(await p);
        """)
        assert got["ok"] is False
        assert "non-JSON" in got["error"] and "server log" in got["error"]

    def test_an_abort_is_a_supersede_not_a_failure(self):
        """AbortError means a newer request took over -- reporting it as an
        error would flash a scary message on an ordinary second edit."""
        got = _run(f"""
            var p = C.analyze({_S});
            pending[0].reject(Object.assign(new Error("aborted"),
                                            {{name: "AbortError"}}));
            out(await p);
        """)
        assert got == {"ok": False, "superseded": True}

    def test_no_structure_is_answered_without_calling_the_server(self):
        got = _run("""
            var res = await C.analyze(null);
            out({ok: res.ok, calls: pending.length});
        """)
        assert got["ok"] is False and got["calls"] == 0
