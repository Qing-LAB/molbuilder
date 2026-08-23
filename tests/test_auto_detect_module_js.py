"""``lib/auto-detect.js`` -- the one auto-detect surface, under test.

Why this file exists
====================

Auto-detect used to be hand-pasted per tab: ``_renderAutoDetectPanel``
three times (byte-identical for 40 lines) over five hand-rolled
``POST /api/structure/analyze`` call sites, each with its own
``AbortController`` and sequence counter -- ``docs/web/audit-2026-08-05
-tab-ui.md`` §§ C1, C2.  The cost was not the duplication itself but
what it hid: the Spectrum tab rendered the rationale panel and no
detection chip (§ A2), because the chip fix had been applied to a
different copy.

Two things are pinned here that no browser test pins well:

  * the SUPERSEDE protocol.  "A response that arrives after the user
    loaded a different structure must not touch the DOM" is a race,
    and a race is exactly what an end-to-end test cannot schedule.
    Driving ``fetch`` by hand makes the interleaving deterministic.
  * the chip pass living INSIDE ``renderPanel``.  A caller cannot
    forget what it does not call, which is what makes § A2
    unrepeatable rather than merely fixed.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
MODULE = REPO / "molbuilder/web/static/lib/auto-detect.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")

#: A DOM stub with just enough semantics for the module, plus a fetch
#: whose responses are resolved BY THE TEST so request interleaving is
#: chosen rather than raced for.
_STUB = r"""
function El(tag) {
  this.tagName = tag; this.children = []; this.attrs = {};
  this.className = ""; this.textContent = ""; this.hidden = false;
  this.open = false;
}
El.prototype.appendChild = function (c) { this.children.push(c); return c; };
El.prototype.setAttribute = function (k, v) { this.attrs[k] = String(v); };

var els = {};
["auto-detect-panel", "auto-detect-rationale", "auto-detect-warnings",
 "auto-detect-metals"].forEach(function (id) {
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
function reply(i, status, body, opts) {
  var p = pending[i];
  p.resolve({
    ok: (opts && "ok" in opts) ? opts.ok : (status >= 200 && status < 300),
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

var chipCalls = [];
globalThis.molbuilder = { detectionChip: {
  render: function (resp) { chipCalls.push(resp); return 1; },
} };
function out(v) { console.log(JSON.stringify(v)); }
"""


def _run(script: str):
    src = MODULE.read_text(encoding="utf-8")
    prog = (f"{_STUB}\n{src}\n"
            f"var A = globalThis.molbuilder.autoDetect;\n"
            f"(async function () {{\n{script}\n}})();")
    res = subprocess.run(["node", "-e", prog], capture_output=True,
                         text=True, timeout=60)
    assert res.returncode == 0, f"node failed:\n{res.stderr}"
    return json.loads(res.stdout.strip().splitlines()[-1])


_RESP = """{
  ok: true,
  warnings: ["open-shell metal"],
  metal_hints: [{element: "Fe", common_spins: [{spin: 4, label: "high-spin"}]}],
  suggested: {pyscf: {rationale: "Fe(II) is open-shell", net_charge: 0}}
}"""


class TestThePanel:

    def test_it_fills_every_part_and_opens(self):
        got = _run(f"""
            A.renderPanel({_RESP});
            out({{
              hidden: els["auto-detect-panel"].hidden,
              open:   els["auto-detect-panel"].open,
              rationale: els["auto-detect-rationale"].textContent,
              warnings: els["auto-detect-warnings"].children
                          .map(function (li) {{ return li.textContent; }}),
              metals: els["auto-detect-metals"].children
                          .map(function (n) {{ return n.textContent; }}),
            }});
        """)
        assert got["hidden"] is False and got["open"] is True
        assert got["rationale"] == "Fe(II) is open-shell"
        assert got["warnings"] == ["open-shell metal"]
        # dt (the element) then dd (each candidate spin)
        assert got["metals"] == ["Fe", "spin=4 — high-spin"]

    def test_a_part_with_nothing_to_say_is_hidden(self):
        """An empty <ul>/<dl> left visible is an empty box on the page."""
        got = _run("""
            A.renderPanel({ok: true, warnings: [], metal_hints: [],
                           suggested: {pyscf: {rationale: "closed shell"}}});
            out({w: els["auto-detect-warnings"].hidden,
                 m: els["auto-detect-metals"].hidden});
        """)
        assert got == {"w": True, "m": True}

    def test_the_rationale_falls_back_to_the_other_engine(self):
        """The rationale is engine-agnostic -- the analyzer writes it once
        and every adapter echoes it.  A SIESTA-only response still has
        reasoning to show, which the `.pyscf`-only reads did not."""
        got = _run("""
            A.renderPanel({ok: true,
                           suggested: {siesta: {rationale: "from siesta"}}});
            out(els["auto-detect-rationale"].textContent);
        """)
        assert got == "from siesta"

    def test_rendering_the_panel_renders_the_chips(self):
        """§ A2, made unrepeatable: the chip pass is not the caller's to
        remember.  Spectrum showed a rationale and no chip for weeks
        because its copy of the renderer simply lacked this line."""
        got = _run(f"A.renderPanel({_RESP}); out(chipCalls.length);")
        assert got == 1

    def test_it_reports_when_there_is_no_card_to_draw_on(self):
        got = _run("""
            els["auto-detect-panel"] = null;
            out(A.renderPanel({ok: true}));
        """)
        assert got is False


class TestTheAnalyzeProtocol:

    def test_a_good_answer_comes_back_whole(self):
        got = _run(f"""
            var p = A.analyze("/s.xyz");
            reply(0, 200, {_RESP});
            var res = await p;
            out({{ok: res.ok, rationale: res.body.suggested.pyscf.rationale,
                 sent: JSON.parse(pending[0].opts.body).structure_path}});
        """)
        assert got == {"ok": True, "rationale": "Fe(II) is open-shell",
                       "sent": "/s.xyz"}

    def test_a_newer_analyze_supersedes_the_older_one(self):
        """The race the counter exists for: two clicks, and the FIRST
        server answer arrives last.  Without the gate it would repaint
        the panel with the superseded structure's chemistry."""
        got = _run(f"""
            var first  = A.analyze("/old.xyz");
            var second = A.analyze("/new.xyz");
            reply(1, 200, {_RESP});          // the newer one answers first
            reply(0, 200, {_RESP});          // ...then the stale one
            var a = await first, b = await second;
            out({{first: a, second_ok: b.ok, aborted: aborted.length}});
        """)
        assert got["first"] == {"ok": False, "superseded": True}
        assert got["second_ok"] is True
        # the older request was killed on the wire, not merely ignored
        assert got["aborted"] == 1

    def test_a_structure_loaded_mid_flight_supersedes_it_too(self):
        """The module cannot see a structure load -- only the page can,
        which is what isStale reports.  Re-checked AFTER the await,
        because that is when the answer would land on the wrong form."""
        got = _run(f"""
            var stale = false;
            var p = A.analyze("/s.xyz", {{isStale: function () {{ return stale; }}}});
            stale = true;                    // user loads another structure
            reply(0, 200, {_RESP});
            out(await p);
        """)
        assert got == {"ok": False, "superseded": True}

    def test_the_servers_own_message_is_what_the_user_gets(self):
        got = _run("""
            var p = A.analyze("/s.xyz");
            reply(0, 400, {ok: false, error: "unreadable element symbol 'Xx'"});
            out(await p);
        """)
        assert got == {"ok": False,
                       "error": "unreadable element symbol 'Xx'"}

    def test_an_error_page_says_so_instead_of_unexpected_token(self):
        """A 5xx HTML page parses as neither JSON nor a server message.
        "Unexpected token <" tells a chemist nothing about what broke."""
        got = _run("""
            var p = A.analyze("/s.xyz");
            reply(0, 500, "__nonjson__");
            out(await p);
        """)
        assert got["ok"] is False
        assert "non-JSON" in got["error"] and "server log" in got["error"]

    def test_an_abort_is_a_supersede_not_a_failure(self):
        """AbortError means a newer request took over -- reporting it as
        an error would flash a scary message on an ordinary second click."""
        got = _run("""
            var p = A.analyze("/s.xyz");
            pending[0].reject(Object.assign(new Error("aborted"),
                                            {name: "AbortError"}));
            out(await p);
        """)
        assert got == {"ok": False, "superseded": True}

    def test_no_path_is_answered_without_calling_the_server(self):
        got = _run("""
            var res = await A.analyze("");
            out({ok: res.ok, calls: pending.length});
        """)
        assert got["ok"] is False and got["calls"] == 0
