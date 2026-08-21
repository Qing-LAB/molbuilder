"""The findings-presentation contract (docs/science/validation.md § 4.1, R2-R4).

lib/validation-findings.js is the ONE module that puts a scientific finding on
screen.  These tests execute it under node against a minimal DOM stub and assert
the behaviours the three per-tab renderers it replaced got wrong:

  * an issue whose ``workflow_group`` names a card the form schema did NOT
    render was silently dropped by all three (they iterated the card panels, so
    a bucket with no panel was built and never read) -- R3;
  * an unrecognised or missing severity was dropped by the spectra copy -- R4;
  * server order was re-sorted by the spectra copy -- R4;
  * a missing residual panel made the transport copy return BEFORE clearing the
    per-card panels, so stale findings survived a re-render.

Plus the structural guard that no page re-grows its own renderer.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

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

function card(role) {
  var ul = new El("ul");
  ul.setAttribute("data-workflow-group", role);
  ul.hidden = true;
  return ul;
}
var window = { document: doc };
"""


def _run(script: str) -> dict:
    src = MODULE.read_text(encoding="utf-8")
    prog = f"{_STUB}\n{src}\nvar F = window.molbuilder.validationFindings;\n{script}"
    out = subprocess.run(["node", "-e", prog], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, f"node failed:\n{out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def _rows(panel_expr: str) -> str:
    """JS snippet: rows of a panel as [{severity, message}]."""
    return (f"{panel_expr}.children.map(function (li) {{ return "
            f"{{severity: li.getAttribute('data-severity'), "
            f"cls: li.className, "
            f"message: li.children.length ? li.children[0].textContent : ''}}; }})")


class TestNothingIsDropped:
    """PINS: docs/science/validation.md § 4.1 clause R3 (nothing is dropped),
    with docs/web/ui-contract.md § 5.1 for the display side.

    INVARIANT: every finding the server sends appears somewhere.  A finding whose
    ``workflow_group`` names a form card the schema DID render goes on that card;
    anything else — no group, or a group with no rendered card — goes to the
    residual panel.  ``render`` returns the counts it actually wrote, and
    ``total`` always equals ``issues.length``.

    PREVENTS: the bug all three per-tab renderers shared.  They iterated the card
    PANELS and wrote only buckets that matched one, so a finding tagged with a
    role the schema had not rendered was built into a bucket that was never read
    — it appeared nowhere at all.  The shared module iterates the FINDINGS; that
    inversion is the fix.
    """

    def test_group_with_no_rendered_card_lands_in_residual(self):
        got = _run(f"""
        var panel = new El("ul");
        var scope = new FormScope([card("profile")]);   // NO "budget" card
        var s = F.render([
            {{severity: "warn",  message: "tagged for a card that exists",
              where: "config.a", workflow_group: "profile"}},
            {{severity: "error", message: "tagged for a card that does NOT exist",
              where: "config.b", workflow_group: "budget"}},
        ], {{panel: panel, formScope: scope}});
        console.log(JSON.stringify({{summary: s,
            residual: {_rows("panel")},
            card: {_rows("scope._cards[0]")}}}));
        """)
        # Both findings are on screen: one on its card, the unroutable one in
        # the residual panel -- previously it vanished entirely.
        assert got["summary"]["total"] == 2
        assert got["summary"]["residual"] == 1
        assert [r["message"] for r in got["residual"]] == [
            "tagged for a card that does NOT exist"]
        assert len(got["card"]) == 1
        assert got["summary"]["counts"] == {"error": 1, "warn": 1, "info": 0}

    def test_rendered_count_equals_received_count(self):
        got = _run(f"""
        var panel = new El("ul");
        var scope = new FormScope([card("stage")]);
        var issues = [];
        for (var i = 0; i < 7; i++) {{
            issues.push({{severity: "info", message: "m" + i,
                          workflow_group: (i % 2) ? "stage" : "nope"}});
        }}
        var s = F.render(issues, {{panel: panel, formScope: scope}});
        console.log(JSON.stringify({{summary: s,
            onCard: scope._cards[0].children.length,
            inResidual: panel.children.length}}));
        """)
        assert got["summary"]["total"] == 7
        assert got["onCard"] + got["inResidual"] == 7

    def test_no_form_scope_sends_everything_to_residual(self):
        got = _run(f"""
        var panel = new El("ul");
        var s = F.render([
            {{severity: "warn", message: "grouped but no form on this page",
              workflow_group: "profile"}},
        ], {{panel: panel}});
        console.log(JSON.stringify({{summary: s, rows: {_rows("panel")}}}));
        """)
        assert got["summary"]["residual"] == 1
        assert len(got["rows"]) == 1


class TestSeverityIsUniform:
    """PINS: docs/science/validation.md § 4.1 clause R4 (severity means the same
    everywhere) + docs/web/ui-contract.md § 5.1 (one row shape, styled once).

    INVARIANT: one row vocabulary (``li.issue-item[data-severity]``, styled once
    in lib/form-components.css); an unrecognised or missing severity renders as
    ``info`` and is NEVER dropped; server order is preserved (``validate()``
    emits geometry, then config, then engine checks — a documented deterministic
    order).

    PREVENTS: the spectra copy's divergence — it kept a SECOND row vocabulary for
    its residual list, dropped any severity outside error/warn/info, and
    re-sorted the rest, so the same finding looked and ranked differently
    depending on which tab you were standing on.
    """

    @pytest.mark.parametrize("sent", ["critical", "", None, "WARN"])
    def test_unrecognised_severity_renders_as_info_never_dropped(self, sent):
        payload = "null" if sent is None else json.dumps(sent)
        got = _run(f"""
        var panel = new El("ul");
        var s = F.render([{{severity: {payload}, message: "kept"}}],
                         {{panel: panel}});
        console.log(JSON.stringify({{summary: s, rows: {_rows("panel")}}}));
        """)
        assert got["summary"]["total"] == 1
        assert len(got["rows"]) == 1, "an odd severity must never drop a finding"
        assert got["rows"][0]["severity"] == "info"
        assert got["summary"]["counts"]["info"] == 1

    def test_server_order_is_preserved(self):
        got = _run(f"""
        var panel = new El("ul");
        F.render([
            {{severity: "info",  message: "first"}},
            {{severity: "error", message: "second"}},
            {{severity: "warn",  message: "third"}},
        ], {{panel: panel}});
        console.log(JSON.stringify({{rows: {_rows("panel")}}}));
        """)
        assert [r["message"] for r in got["rows"]] == ["first", "second", "third"]

    def test_one_row_shape(self):
        got = _run(f"""
        var panel = new El("ul");
        F.render([{{severity: "error", message: "m", where: "cell.x"}}],
                 {{panel: panel}});
        console.log(JSON.stringify({{
            cls: panel.children[0].className,
            sev: panel.children[0].getAttribute("data-severity"),
            kids: panel.children[0].children.map(function (c) {{
                return c.className + ":" + c.textContent; }})}}));
        """)
        assert got["cls"] == "issue-item"
        assert got["sev"] == "error"
        assert got["kids"] == ["issue-msg:m", "issue-where:cell.x"]


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
        residual markup was missing, so a stale finding stayed up."""
        got = _run("""
        var scope = new FormScope([card("profile")]);
        F.render([{severity: "warn", message: "old", workflow_group: "profile"}],
                 {panel: new El("ul"), formScope: scope});
        var before = scope._cards[0].children.length;
        F.render([], {formScope: scope});          // no panel at all
        console.log(JSON.stringify({before: before,
            after: scope._cards[0].children.length,
            hidden: scope._cards[0].hidden}));
        """)
        assert got["before"] == 1
        assert got["after"] == 0 and got["hidden"] is True

    def test_a_second_render_replaces_rather_than_appends(self):
        got = _run("""
        var panel = new El("ul");
        F.render([{severity: "warn", message: "a"}], {panel: panel});
        F.render([{severity: "warn", message: "b"}], {panel: panel});
        console.log(JSON.stringify({n: panel.children.length,
            msg: panel.children[0].children[0].textContent}));
        """)
        assert got["n"] == 1 and got["msg"] == "b"

    def test_empty_hides_the_panel_unless_empty_text_is_given(self):
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


class TestOneRendererOnly:
    """PINS: docs/science/validation.md § 4.1 clause R2 (one channel into the
    UI) — enforced STRUCTURALLY, by source inspection, because behaviour tests
    cannot catch a fourth renderer appearing next to the third.

    INVARIANT: the three consumer surfaces delegate to
    ``molbuilder.validationFindings`` and build no finding rows of their own; the
    second row vocabulary and its stylesheet are gone; and every page that shows
    findings actually loads the module (a page that forgets it would silently
    render nothing).
    """

    _CONSUMERS = (
        "molbuilder/web/static/structure-optimization/viewer.js",
        "molbuilder/web/static/lib/transport/core.js",
        # lib/spectra/core.js left this list at P3 and RETURNED the
        # same day: the live-preflight panel (gate ① for the
        # vibration kind) renders through the shared module.
        "molbuilder/web/static/lib/spectra/core.js",
    )

    def test_consumers_delegate_and_do_not_build_issue_rows(self):
        for rel in self._CONSUMERS:
            src = (REPO / rel).read_text(encoding="utf-8")
            assert "validationFindings" in src, f"{rel} does not use the module"
            # The tell-tales of a private renderer: creating the row element or
            # tagging severity itself.
            assert 'className = "issue-item"' not in src, (
                f"{rel} builds its own finding row again")
            assert '"data-severity"' not in src, (
                f"{rel} maps severity itself again")

    def test_no_second_row_vocabulary_survives(self):
        """The spectra copy's div.issue/.badge markup and its competing
        .issues-panel declaration are gone from the page sheet."""
        import re
        css = (REPO / "molbuilder/web/static/spectra/style.css").read_text(
            encoding="utf-8")
        # Strip comments first: the replacement comment NAMES the removed
        # declarations, so a raw substring search would match its own prose.
        rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        for dead in (".issue .badge", ".issue.error", ".issue.warn",
                     ".issue .where", ".issues-panel {"):
            assert dead not in rules, (
                f"{dead!r} is still declared in the spectra sheet — the page "
                f"has re-grown its own findings vocabulary")

    def test_every_page_that_renders_findings_loads_the_module(self):
        tpl = REPO / "molbuilder/web/templates"
        for name in ("index.html", "spectra.html",
                     "transport_calculation.html", "results.html"):
            html = (tpl / name).read_text(encoding="utf-8")
            assert "filename='lib/validation-findings.js'" in html, (
                f"{name} renders findings but never loads the module")
