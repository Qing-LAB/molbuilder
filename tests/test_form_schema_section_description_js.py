"""A section's description has to REACH THE SCREEN, and only where it is true.

`_form_section_descriptions` is a paragraph per section explaining what the
knobs under that legend are for -- the transport config carries twelve of them.
Until 2026-09-15 the renderer appended them on ONE of its two paths: the bare
path, for a section whose fields carry no ``workflow_group``.  Every field on
the transport tab carries one, so every section rendered inside a card, and all
twelve paragraphs were written, reviewed, and displayed nowhere.

**And a test already "covered" them.**
`test_transport_config.py::test_every_section_has_a_description` asserts each
declared section has an entry in the dict.  It passed the whole time.  That is
the shape this file exists to answer: a presence check on the DATA cannot see
that the RENDERER drops it, so the assertion has to be on the rendered DOM.

**The second case is the interesting one.**  A section may straddle workflow
groups -- SIESTA's "Compute & budget" has fields in four -- and then each card
holds a SUBSET.  A paragraph about the whole section is partly false over a
subset ("Runtime ... memory budget, CPU thread count, log verbosity" above a
card holding only verbosity), and repeating it once per card says it several
times and is wrong every time.  So the rule is: show it when the card holds
the whole section, and not otherwise.  Both halves are asserted, because an
implementation that always shows it passes the first one alone.

Runs the real module through Playwright, the harness
`test_form_schema_diff_js.py` set up.
"""
from __future__ import annotations

import textwrap
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder" / "web" / "static"

#: The paragraph under the legend, and the class the stylesheet answers.
DESC_SELECTOR = "fieldset.schema-section > p.schema-section-desc"

INTACT_DESC = "Everything in this section is tagged profile."
SPLIT_DESC = "This section has one field in each of two cards."


@pytest.fixture
def form_server():
    from flask import Flask, send_from_directory
    from werkzeug.serving import make_server

    app = Flask(__name__)

    @app.route("/lib/<path:p>")
    def lib(p):
        return send_from_directory(STATIC / "lib", p)

    @app.route("/")
    def root():
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


def _field(name, group):
    return {"name": name, "id": "f-" + name, "kind": "number",
            "label": name, "default": 1.0, "workflow_group": group}


#: Two sections, both entirely inside workflow-group cards -- which is the
#: case the old renderer dropped.  "Intact" lives in one card; "Split" has
#: one field in each of two.
SCHEMA = {
    "sections": [
        {"id": "intact", "name": "Intact", "title": "Intact",
         "description": INTACT_DESC,
         "fields": [_field("a", "profile"), _field("b", "profile")]},
        {"id": "split", "name": "Split", "title": "Split",
         "description": SPLIT_DESC,
         "fields": [_field("c", "profile"), _field("d", "stage")]},
    ],
}


def _render(page, base_url):
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.formSchema"
        "      && typeof window.molbuilder.formSchema.renderForm"
        "             === 'function'",
        timeout=5000,
    )
    page.evaluate(
        "(schema) => window.molbuilder.formSchema.renderForm("
        "  document.getElementById('container'), schema)",
        SCHEMA,
    )


def test_a_section_inside_a_card_shows_its_description(page, form_server):
    """THE ASSERTION THE TWELVE INVISIBLE PARAGRAPHS WOULD HAVE TRIPPED."""
    _render(page, form_server)
    shown = page.eval_on_selector_all(
        DESC_SELECTOR, "els => els.map(e => e.textContent)")
    assert INTACT_DESC in shown, (
        "a section whose fields all sit in one workflow-group card renders "
        "inside that card, and its description has to render with it -- it "
        "reached the screen on the BARE path only until 2026-09-15, so a "
        "form where every field is tagged showed none of them:\n"
        f"  rendered descriptions: {shown}")


def test_a_section_split_across_cards_shows_no_description(page, form_server):
    """...and only where it is true.  Without this, "always append" passes
    the test above while saying a whole-section paragraph over a subset,
    once per card."""
    _render(page, form_server)
    shown = page.eval_on_selector_all(
        DESC_SELECTOR, "els => els.map(e => e.textContent)")
    assert SPLIT_DESC not in shown, (
        "this section has one field in the profile card and one in the "
        "stage card, so each card holds a SUBSET -- a description of the "
        "whole section is partly false above either one, and printing it "
        "in both says it twice:\n"
        f"  rendered descriptions: {shown}")
    # AND the fieldsets themselves are there, or the assertion above is
    # vacuous: nothing rendered also contains no description.
    legends = page.eval_on_selector_all(
        "fieldset.schema-section > legend", "els => els.map(e => e.textContent)")
    assert legends.count("Split") == 2, (
        f"expected the split section to render in two cards, got {legends}")
