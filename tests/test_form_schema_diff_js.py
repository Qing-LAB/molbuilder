"""``formSchema.diffFromDefaults`` — which parameters are not at the
catalogue's recommended value, and what each would go back to.

**Why the tab does not do this itself.**  The comparison needs both
halves that ``form-schema.js`` already owns: what the DOM currently
holds (``collectForm``) and what the schema says.  A tab comparing them
would need its own reader for every field kind this module handles —
checkbox, int, number, select, tri-select, text, int-triple — which is
the drift a shared module exists to prevent.

**Why a per-parameter list rather than one Reset button.**  A 4×4×1
k-grid and a stale mixing weight sit in the same form and look the same
to a button.  The real folder that prompted this work
(``BDT-Au/optimization/HPC-BDT-Au111``) had three deliberate
differences and two that came along from an older session; a single
reset would have discarded the surface k-grid to fix the mixing weight.

Behavioural tests run through Playwright against the real module — the
same harness ``test_form_schema_setvalues_js.py`` uses — so a
regression in any field-kind branch surfaces here.
"""
from __future__ import annotations

import re
import threading
import textwrap
from pathlib import Path

import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder" / "web" / "static"
FORM_JS = STATIC / "lib" / "form-schema.js"
FORM_CSS = STATIC / "lib" / "form-schema.css"
TOKENS = STATIC / "lib" / "tokens.css"
VIEWER = STATIC / "structure-optimization" / "viewer.js"
INDEX = ROOT / "molbuilder" / "web" / "templates" / "index.html"


@pytest.fixture
def flask_server():
    from werkzeug.serving import make_server
    from flask import Flask, send_from_directory

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


SCHEMA = {
    "sections": [{
        "id": "main", "title": "Main",
        "fields": [
            {"name": "flag", "id": "f-flag", "kind": "checkbox",
             "label": "Flag", "default": False},
            {"name": "mesh_cutoff", "id": "f-mesh", "kind": "number",
             "label": "Mesh cutoff", "default": 300.0, "unit": "Ry"},
            {"name": "mixing_weight", "id": "f-mix", "kind": "number",
             "label": "Mixing weight", "default": 0.02},
            {"name": "relax_type", "id": "f-relax", "kind": "select",
             "label": "Relax type", "default": "CG",
             "choices": ["CG", "Broyden"]},
            {"name": "kgrid", "id": "f-kgrid", "kind": "int-triple",
             "label": "K-grid", "default": [1, 1, 1],
             "labels": ["x", "y", "z"]},
            # No default: there is nothing to recommend, so it must never
            # be offered — resetting it would blank a value on the user's
            # behalf.
            {"name": "note", "id": "f-note", "kind": "text",
             "label": "Note", "default": None},
        ],
    }],
}


def _mount(page, base_url):
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.formSchema"
        "      && typeof window.molbuilder.formSchema.diffFromDefaults"
        "             === 'function'",
        timeout=5000,
    )
    page.evaluate(
        "(schema) => {"
        "  const c = document.getElementById('container');"
        "  window.molbuilder.formSchema.renderForm(c, schema);"
        "  window.__schema = schema;"
        "}",
        SCHEMA,
    )


def _diff(page):
    return page.evaluate(
        "() => window.molbuilder.formSchema.diffFromDefaults("
        "  document.getElementById('container'), window.__schema)"
    )


def _set(page, values):
    page.evaluate(
        "(v) => window.molbuilder.formSchema.setValues("
        "  document.getElementById('container'), window.__schema, v)",
        values,
    )


# ------------------------------------------------------------------ #
# behaviour
# ------------------------------------------------------------------ #

def test_a_freshly_rendered_form_differs_in_nothing(page, flask_server):
    """renderForm seeds every field from its default, so the panel must
    start empty.  A form that reports differences before anybody has
    touched it is the failure that makes the whole feature noise."""
    _mount(page, flask_server)
    assert _diff(page) == []


def test_it_names_the_changed_parameter_and_what_it_goes_back_to(page,
                                                                flask_server):
    _mount(page, flask_server)
    _set(page, {"mesh_cutoff": 450.0})
    diffs = _diff(page)
    assert [d["name"] for d in diffs] == ["mesh_cutoff"]
    d = diffs[0]
    assert d["current"] == 450.0
    assert d["recommended"] == 300.0
    assert d["label"] == "Mesh cutoff" and d["unit"] == "Ry"


def test_the_composite_kinds_are_compared_as_wholes(page, flask_server):
    """A k-grid is one decision, not three.  Comparing element-wise would
    report `kgrid` three times and reset it a third at a time."""
    _mount(page, flask_server)
    _set(page, {"kgrid": [4, 4, 1]})
    diffs = _diff(page)
    assert [d["name"] for d in diffs] == ["kgrid"]
    assert diffs[0]["current"] == [4, 4, 1]
    assert diffs[0]["recommended"] == [1, 1, 1]


def test_a_number_typed_as_text_is_not_a_difference(page, flask_server):
    """An input reads back as a string.  Comparing JSON alone would make
    "300" ≠ 300 and flag every numeric field the moment it is focused —
    a panel that cries wolf is one nobody reads."""
    _mount(page, flask_server)
    page.evaluate(
        "() => { const el = document.getElementById('f-mesh');"
        "        el.value = '300';"
        "        el.dispatchEvent(new Event('change', {bubbles: true})); }"
    )
    assert [d["name"] for d in _diff(page)] == []


def test_a_field_with_no_default_is_never_offered(page, flask_server):
    """There is nothing to recommend, so 'reset' would mean blanking it."""
    _mount(page, flask_server)
    _set(page, {"note": "anything at all"})
    assert "note" not in [d["name"] for d in _diff(page)]


def test_resetting_one_leaves_the_others_alone(page, flask_server):
    """The whole reason this is a list: a deliberate k-grid survives while
    a stale mixing weight goes back.  This is the HPC-BDT-Au111 case."""
    _mount(page, flask_server)
    _set(page, {"kgrid": [4, 4, 1], "mixing_weight": 0.1})
    assert sorted(d["name"] for d in _diff(page)) == ["kgrid", "mixing_weight"]

    # what the panel's Reset does, through the same public verb
    _set(page, {"mixing_weight": 0.02})
    diffs = _diff(page)
    assert [d["name"] for d in diffs] == ["kgrid"], (
        "resetting one parameter disturbed another")
    assert diffs[0]["current"] == [4, 4, 1]


def test_booleans_compare_without_a_number_detour(page, flask_server):
    """`false` and `0` are not the same answer, and Number(false) === 0
    would quietly merge them."""
    _mount(page, flask_server)
    _set(page, {"flag": True})
    assert [d["name"] for d in _diff(page)] == ["flag"]
    assert _diff(page)[0]["current"] is True


# ------------------------------------------------------------------ #
# the panel, and where its parts live
# ------------------------------------------------------------------ #

def test_the_panel_resets_through_setvalues(page=None):
    """The tab must not write the inputs itself — every field kind is
    already handled by `setValues`, and a second writer is the drift the
    module exists to prevent."""
    src = VIEWER.read_text()
    body = src.split("function mountRecommended", 1)[1].split("\n    async ", 1)[0]
    assert "diffFromDefaults" in body, "the panel compares by hand"

    # The reset handler ALONE — the panel's own checkboxes are its to
    # build, but the form's inputs are `setValues`' to write.
    handler = body.split('resetEl.addEventListener("click"', 1)[1]
    handler = handler.split("\n        });", 1)[0]
    assert "fs.setValues(" in handler, "reset writes the form by hand"
    assert "container.querySelector" not in handler, (
        "reset reaches into the form's inputs instead of going through "
        "setValues, which handles every field kind")


def test_both_engine_forms_carry_the_panel():
    html = INDEX.read_text()
    for engine in ("siesta", "pyscf"):
        assert f'id="{engine}-recommend"' in html, (
            f"{engine}'s form has no recommended-value panel")
    src = VIEWER.read_text()
    mount = src.split("mountRecommended(engine, host", 1)
    assert len(mount) == 2, "nothing mounts the panels"
    for host in ("siesta-form-container", "pyscf-form-container"):
        assert host in mount[0].rsplit("for (const [engine, hostId]", 1)[-1], (
            f"{host} is not in the mount loop")


def test_the_panel_is_mounted_AFTER_the_session_restore():
    """The ordering IS the feature.  `restoreFormState` assigns `el.value`
    directly and dispatches nothing, so a panel mounted before it takes its
    first reading from a form still at its defaults — reports no
    differences — and then never hears the saved values arrive.  That is
    exactly how this shipped broken the first time."""
    src = VIEWER.read_text()
    body = src.split("async function initFormsFromSchema", 1)[1]
    body = body.split("\n    // ----- Sidebar-driven", 1)[0]
    assert "restoreFormState();" in body and "mountRecommended(" in body
    assert body.index("restoreFormState();") < body.index("mountRecommended("), (
        "the panel is mounted before the restore, so its first reading is "
        "the defaults and the restored values never reach it")


def test_the_panel_follows_the_form_rather_than_only_typing():
    """The compatibility engine and session restore both write the form
    without a keystroke; a panel that only listened to typing would show
    a stale count exactly when it matters."""
    src = VIEWER.read_text()
    body = src.split("function mountRecommended", 1)[1].split("\n    async ", 1)[0]
    assert 'container.addEventListener("change", refresh)' in body
    assert 'container.addEventListener("input", refresh)' in body


def test_the_panel_sheet_has_one_owner_and_no_literals():
    """`ui-contract.md` § 1: a shared element has exactly one owner.  The
    widget is generic — it works off `diffFromDefaults` — so it lives in
    the forms module sheet, not a tab sheet."""
    assert ".rec-diff" in FORM_CSS.read_text(), (
        "the panel's styles are not in the module that owns the widget")
    for sheet in (STATIC / "structure-optimization" / "style.css",):
        assert ".rec-diff-list" not in sheet.read_text(), (
            "a tab sheet is redefining the shared widget")

    block = FORM_CSS.read_text().split("recommended-value panel", 1)[1]
    block = re.sub(r"/\*.*?\*/", "", block, flags=re.S)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block), (
        "raw colour in the panel sheet; every colour is a --rec-* token")
    assert not re.search(r"rgba?\(", block), "raw colour function in the sheet"


def test_the_flag_tone_is_passive_not_a_call_to_action():
    """A parameter away from its recommendation is a FLAG, not an error —
    a 4x4x1 k-grid is a correct answer that differs.  `tokens.css` keeps
    --warning for a call to action and --warn-soft for a passive one."""
    toks = TOKENS.read_text()
    m = re.search(r"--rec-flag:\s*var\((--[a-z-]+)\)", toks)
    assert m, "--rec-flag is not defined as a token reference"
    assert m.group(1) == "--warn-soft", (
        f"--rec-flag borrows {m.group(1)}; a difference is not an error")


def test_a_commit_carrying_a_bare_name_still_finds_the_file():
    """`publishCommit`'s second argument is a full PATH — `list.js` passes
    `fullPath` — but it was called `file` for a long time, which reads as a
    name.  A caller that passed a name got it resolved against the server
    process's own directory and refused as "outside every configured root",
    with nothing in the message naming the caller or the argument.

    The commit already carries the directory, so the tab completes a bare name
    rather than failing on it."""
    src = VIEWER.read_text()
    body = src.split("async function _commitStructure", 1)[1].split("\n        }", 1)[0]
    assert 'sel.dir' in body, (
        "a bare name is still resolved against whatever the process's cwd is")
    assert '!f.includes("/")' in body, "nothing detects a bare name"


def test_the_contract_says_the_argument_is_a_path():
    doc = (ROOT / "docs/web/projects.md").read_text()
    assert "publishCommit(dir, path)" in doc, (
        "the contract still calls it `file`, which is what caused the misuse")
    assert "FULL path" in doc
