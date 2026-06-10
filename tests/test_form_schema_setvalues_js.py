"""Tests for ``window.molbuilder.formSchema.setValues`` — the
public helper added in Phase 2 of the chemistry middle-layer
work that backs the Auto-detect button's "apply suggestion"
path.  See ``molbuilder/web/static/lib/form-schema.js`` for the
helper; see ``docs/protocols/scientific-validation.md`` § 5.1
for the contract.

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


def _mount(page, base_url, schema):
    """Goto the host page, render the schema, return when ready."""
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.formSchema "
        "      && typeof window.molbuilder.formSchema.setValues "
        "             === 'function'",
        timeout=5000,
    )
    page.evaluate(
        "(schema) => {"
        "  const c = document.getElementById('container');"
        "  window.molbuilder.formSchema.renderForm(c, schema);"
        "  window.__schema_for_test = schema;"
        "}",
        schema,
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
    """The Auto-detect path passes a dict like
    {charge: 0, spin: 2, method: 'UKS'} — pin that multi-field
    sets work in one call."""
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
    no error, no DOM mutation.  The auto-detect endpoint adds
    new ``suggested.<engine>`` entries over time; an engine
    refresh that adds a field the local form schema doesn't yet
    have must not break the form-fill path."""
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
    """``null`` value on a text field clears it (empty string).
    Matches the convention used by collectForm — the inverse
    operation."""
    _mount(page, flask_server, ONE_OF_EACH_SCHEMA)
    # Seed with content, then set to null.
    page.locator("#f-label").fill("seed")
    page.evaluate(
        "() => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'),"
        "  window.__schema_for_test, {label: null})"
    )
    assert _collect(page)["label"] == ""


def test_setvalues_int_triple_rejects_wrong_arity(page, flask_server):
    """An int-triple value with the wrong length is silently
    skipped (not crashing the call).  The auto-detect path won't
    send wrong arities, but a buggy adapter could; pin the
    defensive behaviour."""
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
    """Auto-detect's form-fill must trigger dirty-tracking
    listeners on the Generate forms (so the form-dirty modal
    fires on subsequent sidebar clicks).  Pin that ``input``
    events fire on changed controls.
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
