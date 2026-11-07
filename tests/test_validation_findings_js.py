"""The findings-presentation contract (docs/science/validation.md § 4.1, R2-R4).

lib/validation-findings.js is the ONE module that puts a scientific finding on
screen.  These tests execute it under node against a minimal DOM stub and assert
the behaviours the three per-tab renderers it replaced got wrong:

  * an issue whose ``card`` names a card the form schema did NOT
    render was silently dropped by all three (they iterated the card panels, so
    a bucket with no panel was built and never read) -- R3;
  * server order was re-sorted by the spectra copy -- R4;
  * a missing residual panel made the transport copy return BEFORE clearing the
    per-card panels, so stale findings survived a re-render.

The findings are what `/api/build/preflight` answers of a structure and a
form, and the cards a form draws are the ones its served schema names for its
fields (each field names the card it is drawn in, the rule a finding is placed
by too).  The spectra copy also dropped a severity it did not know (R4); no
route sends one -- the server's are error, warn and info -- so it is not
asked.

Whether each page that shows findings draws them is asked of the page:
`test_build_e2e.py`'s findings beside their field on the optimization tab,
`test_spectrum_form_e2e.py` on the Spectrum tab, and
`test_transport_on_a_real_junction_e2e.py` on the Transport tab.
"""
from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from support.road import CO2, H2
from tests._node_esm import run_node

REPO = pathlib.Path(__file__).resolve().parent.parent
MODULE = REPO / "molbuilder/web/static/lib/validation-findings.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")

# A DOM stub: just enough element/list semantics for the module under test.
_STUB = r"""
function El(tag) {
  this.tagName = tag; this.children = []; this.attrs = {}; this.className = "";
  this.textContent = ""; this.hidden = false; this.ownerDocument = doc;
}
El.prototype.appendChild = function (c) { this.children.push(c); return c; };
El.prototype.removeChild = function (c) {
  this.children = this.children.filter(function (x) { return x !== c; });
};
El.prototype.setAttribute = function (k, v) { this.attrs[k] = String(v); };
El.prototype.getAttribute = function (k) {
  return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null;
};
Object.defineProperty(El.prototype, "firstChild", {
  get: function () { return this.children.length ? this.children[0] : null; }
});
var doc = { createElement: function (t) { return new El(t); } };

// A form scope whose querySelectorAll returns the card panels it was given.
function FormScope(cards) { this._cards = cards; }
FormScope.prototype.querySelectorAll = function () { return this._cards; };

function card(name) {
  var ul = new El("ul");
  ul.setAttribute("data-card", name);
  ul.hidden = true;
  return ul;
}
globalThis.document = doc;
"""


def _run(script: str) -> dict:
    """Import the REAL module, as a page does, against the stub DOM."""
    return run_node(
        [], f"const F = await import({json.dumps(MODULE.as_uri())});\n{script}",
        globals_js=_STUB, timeout=60)


def _rows(panel_expr: str) -> str:
    """JS snippet: rows of a panel as [{severity, message}]."""
    return (f"{panel_expr}.children.map(function (li) {{ return "
            f"{{severity: li.getAttribute('data-severity'), "
            f"cls: li.className, "
            f"message: li.children.length ? li.children[0].textContent : ''}}; }})")


def _answer(structure, engine, calculation, params):
    """What `/api/build/preflight` answers of ``structure`` and a form
    holding ``params``: its findings, in its order."""
    from molbuilder.web.app import create_app
    from support.road import road_structure
    r = create_app(config={}).test_client().post("/api/build/preflight", json={
        "structure": road_structure(structure).to_dict(), "engine": engine,
        "calculation": calculation, "params": params})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["issues"]


def _drawn_cards(url):
    """The cards the form the served schema at ``url`` draws: each one a
    field of it names."""
    from molbuilder.web.app import create_app
    schema = create_app(config={}).test_client().get(url).get_json()["schema"]
    return sorted({f["card"] for sec in schema["sections"]
                   for f in sec.get("fields", []) if f.get("card")})


#: A SIESTA optimization of the road's H2, its form asking a mixing weight,
#: a mesh and a k-grid the checks speak of: findings on three cards and one
#: on none.
_H2_FORM = (H2, "siesta", "optimization",
            {"mixing_weight": 0.9, "mesh_cutoff": 30.0, "kgrid": [4, 4, 4]})


def _cards_js(names) -> str:
    return "[" + ", ".join(f"card({json.dumps(n)})" for n in names) + "]"


class TestNothingIsDropped:
    """PINS: docs/science/validation.md § 4.1 clause R3 (nothing is dropped),
    with docs/web/ui-contract.md § 5.1 for the display side.

    INVARIANT: every finding the server sends appears somewhere.  A finding whose
    ``card`` names a form card the schema DID render goes on that card;
    anything else — no card, or a card the form did not draw — goes to the
    residual panel.  ``render`` returns the counts it actually wrote, and
    ``total`` always equals ``issues.length``.

    PREVENTS: the bug all three per-tab renderers shared.  They iterated the card
    PANELS and wrote only buckets that matched one, so a finding tagged with a
    role the schema had not rendered was built into a bucket that was never read
    — it appeared nowhere at all.  The shared module iterates the FINDINGS; that
    inversion is the fix.
    """

    def test_every_finding_lands_on_its_card_or_in_the_residual_panel(self):
        issues = _answer(*_H2_FORM)
        cards = _drawn_cards("/api/build/schema/siesta?calculation=optimization")
        assert {i.get("card") for i in issues} - {None} <= set(cards), issues
        assert any(i.get("card") is None for i in issues), issues
        got = _run(f"""
        var panel = new El("ul");
        var scope = new FormScope({_cards_js(cards)});
        var s = F.render({json.dumps(issues)},
                         {{panel: panel, formScope: scope}});
        var onCards = {{}};
        scope._cards.forEach(function (c) {{
            onCards[c.getAttribute("data-card")] = {_rows("c")};
        }});
        console.log(JSON.stringify({{summary: s, residual: {_rows("panel")},
                                     onCards: onCards}}));
        """)
        assert got["summary"]["total"] == len(issues)
        counts = {sev: sum(1 for i in issues if i["severity"] == sev)
                  for sev in ("error", "warn", "info")}
        assert got["summary"]["counts"] == counts
        for card in cards:
            assert [r["message"] for r in got["onCards"][card]] == [
                i["message"] for i in issues if i.get("card") == card], card
        assert [r["message"] for r in got["residual"]] == [
            i["message"] for i in issues if i.get("card") is None]

    def test_a_form_drawn_in_parts_finds_the_card_in_any_part(self):
        """`form-schema.md` § 1.3: the Spectrum tab draws one engine's form
        in two parts -- the settings both engines share, and the engine's
        own -- and passes both as the scope.  A finding lands on its card in
        whichever part draws it; none whose card a part draws falls to the
        residual panel.
        MUTATION THIS MUST FAIL AGAINST: a list scope read as one element."""
        issues = _answer(CO2, "siesta", "vibration", {"compute_raman": True})
        common = _drawn_cards("/api/build/schema/common?calculation=vibration")
        own = _drawn_cards(
            "/api/build/schema/siesta?calculation=vibration&surface=own")
        drawn = set(common) | set(own)
        placed = [i for i in issues if i.get("card") in drawn]
        assert {i["card"] for i in placed} & set(common), (issues, common)
        assert {i["card"] for i in placed} & set(own) - set(common), (
            issues, own)
        got = _run(f"""
        var panel = new El("ul");
        var common = new FormScope({_cards_js(common)});
        var own = new FormScope({_cards_js(own)});
        var s = F.render({json.dumps(issues)},
                         {{panel: panel, formScope: [common, own]}});
        var onCards = 0;
        common._cards.concat(own._cards).forEach(function (c) {{
            onCards += c.children.length; }});
        console.log(JSON.stringify({{residual: s.residual, onCards: onCards}}));
        """)
        assert got["onCards"] == len(placed)
        assert got["residual"] == len(issues) - len(placed)

    def test_no_form_scope_sends_everything_to_residual_in_server_order(self):
        """A page with no form draws every finding in its residual panel, in
        the order the server sent them (`validate()` emits geometry, then
        config, then engine checks -- a documented deterministic order)."""
        issues = _answer(*_H2_FORM)
        # THE ORDER ASKS SOMETHING: the server's answer is in no severity
        # rank, either way, so a renderer that sorts by severity reorders it.
        rank = {"error": 0, "warn": 1, "info": 2}
        sent = [i["severity"] for i in issues]
        assert sent != sorted(sent, key=rank.get), sent
        assert sent != sorted(sent, key=rank.get, reverse=True), sent
        got = _run(f"""
        var panel = new El("ul");
        var s = F.render({json.dumps(issues)}, {{panel: panel}});
        console.log(JSON.stringify({{summary: s, rows: {_rows("panel")}}}));
        """)
        assert got["summary"]["residual"] == len(issues)
        assert [(r["severity"], r["message"]) for r in got["rows"]] == [
            (i["severity"], i["message"]) for i in issues]


class TestSeverityIsUniform:
    """PINS: docs/science/validation.md § 4.1 clause R4 (severity means the same
    everywhere) + docs/web/ui-contract.md § 5.1 (one row shape, styled once).

    PREVENTS: the spectra copy's divergence — it kept a SECOND row vocabulary for
    its residual list, so the same finding looked and ranked differently
    depending on which tab you were standing on.
    """

    def test_one_row_shape(self):
        issue = _answer(*_H2_FORM)[0]
        got = _run(f"""
        var panel = new El("ul");
        F.render([{json.dumps(issue)}], {{panel: panel}});
        console.log(JSON.stringify({{
            cls: panel.children[0].className,
            sev: panel.children[0].getAttribute("data-severity"),
            kids: panel.children[0].children.map(function (c) {{
                return c.className + ":" + c.textContent; }}),
            where: panel.children[0].getAttribute("data-where")}}));
        """)
        assert got["cls"] == "issue-item"
        assert got["sev"] == issue["severity"]
        assert got["kids"] == ["issue-msg:" + issue["message"]]
        assert got["where"] == issue["where"]


class TestStaleStateCannotSurvive:
    """PINS: docs/science/validation.md § 4.1 clause R2 — the ONE renderer owns
    the panel's whole lifecycle, clearing included.

    INVARIANT: a render REPLACES what is on screen; findings from a previous
    Generate can never survive into the next one, and the per-card panels are
    cleared unconditionally even when the residual panel is absent.  An empty
    result hides the panel, unless the caller supplied ``emptyText`` (the spectra
    Issues panel prefers a word to disappearing).

    PREVENTS: transport's copy returned early when the residual markup was
    missing, which skipped clearing the CARDS too, so stale findings stayed up
    after a re-Generate.
    """

    def test_cards_are_cleared_even_with_no_residual_panel(self):
        """The transport copy returned before clearing the cards when the
        residual markup was missing, so a stale finding stayed up.  The
        second render is the page's own clearing -- nothing to show."""
        issues = _answer(*_H2_FORM)
        card = next(i["card"] for i in issues if i.get("card"))
        got = _run(f"""
        var scope = new FormScope([card({json.dumps(card)})]);
        F.render({json.dumps(issues)},
                 {{panel: new El("ul"), formScope: scope}});
        var before = scope._cards[0].children.length;
        F.render([], {{formScope: scope}});          // no panel at all
        console.log(JSON.stringify({{before: before,
            after: scope._cards[0].children.length,
            hidden: scope._cards[0].hidden}}));
        """)
        assert got["before"] == sum(1 for i in issues if i.get("card") == card)
        assert got["after"] == 0 and got["hidden"] is True

    def test_a_second_render_replaces_rather_than_appends(self):
        """The form changed and the server answered again: its second
        answer, not both."""
        first = _answer(*_H2_FORM)
        second = _answer(*_H2_FORM[:3], {})
        assert first != second
        got = _run(f"""
        var panel = new El("ul");
        F.render({json.dumps(first)}, {{panel: panel}});
        F.render({json.dumps(second)}, {{panel: panel}});
        console.log(JSON.stringify({{rows: {_rows("panel")}}}));
        """)
        assert [r["message"] for r in got["rows"]] == [
            i["message"] for i in second]

    def test_empty_hides_the_panel_unless_empty_text_is_given(self):
        """The page's own clearing: nothing to show hides the panel, or says
        the caller's word for it."""
        got = _run("""
        var a = new El("ul"); var b = new El("ul");
        F.render([], {panel: a});
        F.render([], {panel: b, emptyText: "No issues."});
        console.log(JSON.stringify({
            aHidden: a.hidden, aRows: a.children.length,
            bHidden: b.hidden, bText: b.children.length
                ? b.children[0].textContent : null}));
        """)
        assert got["aHidden"] is True and got["aRows"] == 0
        assert got["bHidden"] is False and got["bText"] == "No issues."
