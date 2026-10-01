"""Tests for ``window.molbuilder.formSchema.setValues`` — the
public helper a tab writes values into a rendered form through (the
Recommended panel's reset, a restored form).  It was added for the
Auto-detect button's "apply suggestion" path, retired 2026-09-28.  See
``molbuilder/web/static/lib/form-schema.js`` for the helper, and
``docs/web/form-schema.md`` § 3.0a for what it guarantees.

It cited ``science/validation.md`` § 5.1 until 2026-09-02 -- a
section that document has never had, and a document this does not
belong in: validation.md is the SCIENTIFIC machinery (analyzer,
adapters, gates), and a helper that writes values into DOM inputs
is the form module's own contract.  form-schema.md has carried
``setValues`` in its API table all along; § 3.0a now states the
two things it guarantees, which are the two a new field kind gets
wrong by omission.

These run as Playwright tests against a minimal HTML page that
mounts a schema with one field of each kind (checkbox, int,
number, text, select, tri-select, int-triple) — so a regression
in any field-kind branch surfaces here without needing the full
``/structure-optimization`` page.

The setValues helper has four behavioural promises:

* Set ``input[type=checkbox].checked`` from a boolean
* Set ``input.value`` from int / number / select / tri-select / text
* Fill the three sub-inputs of an int-triple from an array
* Dispatch ``input`` and ``change`` events so dirty-trackers observe

All four are pinned.  Edge cases: undefined values, missing fields
in the schema, malformed values — covered too.
"""
from __future__ import annotations

import threading
import textwrap

import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture
def flask_server():
    """Tiny Flask app that serves the form-schema.js bundle + an
    inline HTML host page.  No need for the full molbuilder app —
    we only exercise the JS helper.
    """
    from werkzeug.serving import make_server
    from flask import Flask, send_from_directory
    from pathlib import Path

    static_root = Path(__file__).resolve().parents[1] \
        / "molbuilder" / "web" / "static"

    app = Flask(__name__)

    @app.route("/lib/<path:p>")
    def lib(p):
        return send_from_directory(static_root / "lib", p)

    @app.route("/")
    def root():
        # Minimal page with a container; the test will inject the
        # schema and call setValues directly.
        return textwrap.dedent("""
            <!doctype html><html><body>
              <div id="container"></div>
              <script src="/lib/form-schema.js"></script>
            </body></html>
        """)

    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# A schema with one field of each kind — fixture for every test.
# Mirrors the shape ``form-schema.js`` expects (makeIntTriple needs
# ``labels`` for the three sub-input headings; makeSelect needs
# ``choices``).  Sub-input ids for the int-triple follow the
# convention ``f.id + "-" + label`` (e.g. "f-kgrid-x").  setValues
# uses fixed labels ["x", "y", "z"] for the writer side, so the
# schema must use the same.
ONE_OF_EACH_SCHEMA = {
    "sections": [{
        "id":     "main",
        "title":  "Main",
        "fields": [
            {"name": "flag",   "id": "f-flag",   "kind": "checkbox",
             "label": "Flag",  "default": False},
            {"name": "count",  "id": "f-count",  "kind": "int",
             "label": "Count", "default": 0},
            {"name": "ratio",  "id": "f-ratio",  "kind": "number",
             "label": "Ratio", "default": 0.0},
            {"name": "method", "id": "f-method", "kind": "select",
             "label": "Method","default": "A",
             "choices": ["A", "B", "C"]},
            {"name": "label",  "id": "f-label",  "kind": "text",
             "label": "Label", "default": ""},
            {"name": "kgrid",  "id": "f-kgrid",  "kind": "int-triple",
             "label": "K-grid","default": [1, 1, 1],
             "labels": ["x", "y", "z"]},
        ],
    }],
}


def _mount(page, base_url, schema, **opts):
    """Goto the host page, render the schema (with ``renderForm``'s
    ``opts``), return when ready."""
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.formSchema "
        "      && typeof window.molbuilder.formSchema.setValues "
        "             === 'function'",
        timeout=5000,
    )
    page.evaluate(
        "([schema, opts]) => {"
        "  const c = document.getElementById('container');"
        "  window.molbuilder.formSchema.renderForm(c, schema, opts);"
        "  window.__schema_for_test = schema;"
        "}",
        [schema, opts],
    )


def _collect(page):
    """Read the form's current values via the existing collectForm
    API — the canonical roundtrip target for setValues."""
    return page.evaluate(
        "() => {"
        "  const c = document.getElementById('container');"
        "  return window.molbuilder.formSchema.collectForm("
        "    c, window.__schema_for_test);"
        "}"
    )


# --------------------------------------------------------------------- #
#  Per-kind round trip                                                  #
# --------------------------------------------------------------------- #


def test_setvalues_sets_checkbox_from_boolean(page, flask_server):
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {flag: true})"
    )
    vals = _collect(page)
    assert vals["flag"] is True


def test_setvalues_sets_int_from_number(page, flask_server):
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {count: 42})"
    )
    vals = _collect(page)
    assert vals["count"] == 42


def test_setvalues_sets_number_from_float(page, flask_server):
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {ratio: 1.5})"
    )
    vals = _collect(page)
    assert vals["ratio"] == 1.5


def test_setvalues_sets_select(page, flask_server):
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {method: 'B'})"
    )
    vals = _collect(page)
    assert vals["method"] == "B"


def test_setvalues_sets_text(page, flask_server):
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {label: 'hello'})"
    )
    vals = _collect(page)
    assert vals["label"] == "hello"


def test_setvalues_sets_int_triple_from_array(page, flask_server):
    """int-triple branch — three sub-inputs each get one element."""
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {kgrid: [4, 8, 4]})"
    )
    vals = _collect(page)
    assert vals["kgrid"] == [4, 8, 4]


# --------------------------------------------------------------------- #
#  Multi-field one call                                                 #
# --------------------------------------------------------------------- #


def test_setvalues_applies_multiple_fields_in_one_call(page, flask_server):
    """The Recommended reset passes a dict of every ticked field
    (`structure-optimization/viewer.js`) — pin that multi-field sets work
    in one call."""
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test,"
        "  {flag: true, count: 7, ratio: 2.5, method: 'C'})"
    )
    vals = _collect(page)
    assert vals["flag"]   is True
    assert vals["count"]  == 7
    assert vals["ratio"]  == 2.5
    assert vals["method"] == "C"


# --------------------------------------------------------------------- #
#  Edge cases                                                           #
# --------------------------------------------------------------------- #


def test_setvalues_ignores_unknown_field_name(page, flask_server):
    """A value for a field not in the schema is silently skipped —
    no error, no DOM mutation.  A saved form restored after a field
    was renamed carries a name the schema no longer has, and that must
    not break the restore."""
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test,"
        "  {count: 99, totally_unknown: 'whatever'})"
    )
    vals = _collect(page)
    assert vals["count"] == 99   # the known field still landed


def test_setvalues_with_undefined_input_is_noop(page, flask_server):
    """Passing undefined / null as the values arg is a no-op."""
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    before = _collect(page)
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, undefined)"
    )
    after = _collect(page)
    assert before == after


def test_setvalues_with_null_value_clears_text(page, flask_server):
    """``null`` blanks a field -- not chosen -- and ``collectForm`` reads a
    blank back as ``null``: the inverse operation (form-schema.md § 1.1)."""
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    # Seed with content, then set to null.
    page.locator("#f-label").fill("seed")
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {label: null})"
    )
    assert _collect(page)["label"] is None


def test_setvalues_int_triple_rejects_wrong_arity(page, flask_server):
    """An int-triple value with the wrong length is silently
    skipped (not crashing the call).  No caller sends a wrong arity,
    but a stale saved form could; pin the defensive behaviour."""
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    # Seed kgrid with [1,1,1] (the default after render).
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {kgrid: [1, 1, 1]})"
    )
    # Wrong arity — should be a no-op.
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {kgrid: [4, 8]})"
    )
    assert _collect(page)["kgrid"] == [1, 1, 1]


# --------------------------------------------------------------------- #
#  Event dispatch (dirty tracker observes programmatic edits)           #
# --------------------------------------------------------------------- #


def test_setvalues_dispatches_input_event_on_changed_fields(page, flask_server):
    """A programmatic set -- the Recommended reset -- must trigger the
    listeners a typed edit does (the dirty tracker, the preflight, the
    chemistry card), or the page goes on describing the values it had.
    Pin that ``input`` events fire on changed controls.
    """
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    # Install a counter that increments on every input event.
    page.evaluate(
        "() => {"
        "  window.__input_count = 0;"
        "  document.getElementById('container').addEventListener("
        "    'input', () => window.__input_count++, true);"
        "}"
    )
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test,"
        "  {flag: true, count: 1, ratio: 1, method: 'B', label: 'x'})"
    )
    n = page.evaluate("() => window.__input_count")
    # Five fields changed → at least 5 input events.
    assert n >= 5, f"expected ≥ 5 input events, got {n}"


def test_a_value_the_kind_does_not_offer_is_shown_as_itself(page,
                                                            flask_server):
    """`engines/template.md` § 6.3a: a menu narrowed to what the kind
    offers still shows a value the template holds outside it as itself,
    marked -- a select with no option for it shows another, and the form
    would say a value nobody holds (the K2 review).

    MUTATION THIS MUST FAIL AGAINST: `makeSelect` without the branch (the
    menu reads blank, not chosen)."""
    schema = {"sections": [{"id": "main", "title": "Main", "fields": [
        {"name": "method", "id": "f-method", "kind": "select",
         "label": "Method", "default": "A", "choices": ["A", "B"],
         "value": "Z", "source": "person"}]}]}
    _mount(page, flask_server, schema)
    got = page.evaluate(
        "() => { const s = document.getElementById('f-method');"
        "  return {value: s.value,"
        "          text: s.options[s.selectedIndex].text}; }")
    assert got == {"value": "Z", "text": "Z (not offered here)"}, got


# --------------------------------------------------------------------- #
#  One field state (form-schema.md § 1.1, plan § 5w K7)                 #
# --------------------------------------------------------------------- #

from molbuilder.template import SOURCE_WORDS   # noqa: E402 -- the one vocabulary


def _said(page, field_id):
    """The caption under a field: whose its value is."""
    return page.evaluate(
        "(id) => document.getElementById(id).closest('.schema-field')"
        "  .querySelector('.schema-source').textContent", field_id)


def _set(page, values):
    page.evaluate(
        "(v) => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, v)", values)


def test_a_field_nobody_chose_is_blank_and_reads_as_not_chosen(
        page, flask_server):
    """A new calculation's form holds nothing until the person gives it
    something: every field reads as ``null`` -- not chosen -- and shows the
    kind's default as its hint.  Never a list's first choice, never a zero
    in a triple, never an unticked box (the M11 review's T-F25: a select
    showed and sent its first choice, a triple 0 0 0).

    MUTATIONS THIS MUST FAIL AGAINST: `makeSelect` without its blank option
    (reads "A"); `makeTriple` drawing the default as cells (reads [1, 1, 1]);
    `makeCheckbox` drawing an unticked box (reads false)."""
    _mount(page, flask_server,
           dict(ONE_OF_EACH_SCHEMA, source_words=dict(SOURCE_WORDS)))
    assert _collect(page) == {name: None for name in (
        "flag", "count", "ratio", "method", "label", "kgrid")}
    got = page.evaluate("""() => {
        const s = document.getElementById("f-method");
        const shown = s.options[s.selectedIndex];
        return {
            box: document.getElementById("f-flag").indeterminate,
            menu: shown ? shown.text : null,
            hints: ["x", "y", "z"].map(
                (l) => document.getElementById("f-kgrid-" + l).placeholder),
        };
    }""")
    assert got == {"box": True,
                   "menu": "(not chosen \u00b7 recommended A)",
                   "hints": ["1", "1", "1"]}, got
    assert _said(page, "f-kgrid") == "not chosen \u00b7 recommended 1, 1, 1"
    # The first click answers the box, and its caption follows.
    page.click("#f-flag")
    assert _collect(page)["flag"] is True
    assert _said(page, "f-flag") == "you set this"


def test_a_template_value_is_held_or_shown_as_what_a_blank_runs(
        page, flask_server):
    """A surface that EDITS the template holds its value, named by its
    source; an edit is the person's, and going back is the source's again.
    A surface of OVERRIDES over the template (a transport rung's tab) holds
    the rung's own values and shows the template's as what a blank field
    runs -- and a value the person gives is read whatever it equals: an
    explicit 1 1 1 transmission grid could not be sent (T-F1), nor an
    optional field set on a rung (T-F24).  A field the rung fixes is shown
    at its answer and never read."""
    grid = {"name": "tbt_k_grid", "id": "f-tk", "kind": "int-triple",
            "label": "Transmission grid", "default": [1, 1, 1],
            "labels": ["x", "y", "z"], "value": [4, 4, 1], "source": "cited"}
    must = {"name": "must", "id": "f-must", "kind": "tri-select",
            "label": "Must converge", "default": None, "optional": True,
            "choices": ["auto", "true", "false"]}
    solver = {"name": "solver", "id": "f-solver", "kind": "select",
              "label": "Solver", "choices": ["diagon", "transiesta"],
              "locked": {"value": "transiesta", "why": "the rung's own"}}
    schema = {"source_words": dict(SOURCE_WORDS), "sections": [
        {"id": "m", "title": "M", "fields": [grid, must, solver]}]}

    _mount(page, flask_server, schema)
    assert _collect(page) == {"tbt_k_grid": [4, 4, 1], "must": None}
    assert _said(page, "f-tk") == "from the run you cited"
    _set(page, {"tbt_k_grid": [1, 1, 1]})
    assert _said(page, "f-tk") == "you set this"
    _set(page, {"tbt_k_grid": [4, 4, 1]})
    assert _said(page, "f-tk") == "from the run you cited"
    assert page.evaluate(
        "() => { const s = document.getElementById('f-solver');"
        "        return [s.disabled, s.value]; }") == [True, "transiesta"]

    _mount(page, flask_server, schema, holds="overrides")
    assert _collect(page) == {"tbt_k_grid": None, "must": None}
    assert _said(page, "f-tk") == ("not chosen \u00b7 4, 4, 1 "
                                   "from the run you cited")
    _set(page, {"tbt_k_grid": [1, 1, 1], "must": True})
    assert _collect(page) == {"tbt_k_grid": [1, 1, 1], "must": True}


def test_a_value_that_will_not_read_is_refused_beside_its_field(
        page, flask_server):
    """A value that will not read as its type is refused, naming its field,
    and the field's own caption says why: a fractional count is never
    rounded (`parseInt` read 4.5 as 4 -- the K3 review), and a triple
    holding only some of its components is not a mesh.

    MUTATION THIS MUST FAIL AGAINST: `readNumber` without its whole-number
    check (reads 4)."""
    _mount(page, flask_server,
           dict(ONE_OF_EACH_SCHEMA, source_words=dict(SOURCE_WORDS)))

    def refusal():
        return page.evaluate("""() => {
            try {
                window.molbuilder.formSchema.collectForm(
                    document.getElementById("container"),
                    window.__schema_for_test);
                return null;
            } catch (e) { return {field: e.field, message: e.message}; }
        }""")

    page.fill("#f-count", "4.5")
    got = refusal()
    assert got and got["field"] == "count" and "4.5" in got["message"], got
    assert "whole number" in _said(page, "f-count")
    page.fill("#f-count", "4")
    assert refusal() is None

    page.fill("#f-kgrid-x", "4")
    got = refusal()
    assert got and got["field"] == "kgrid", got
    page.fill("#f-kgrid-y", "4")
    page.fill("#f-kgrid-z", "1.5")
    got = refusal()
    assert got and got["field"] == "kgrid" and "1.5" in got["message"], got
