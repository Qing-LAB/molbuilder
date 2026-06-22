"""Playwright tests for the ``stage-table`` form-schema widget
(task #534 commit 3 — the JS half of the staged-opt UI).

Contract the JS pins:

* ``renderForm`` mounts a per-stage table with one row per
  stage_field (name / enabled / conv_tol / gmax / grms / dmax /
  drms / etol / max_steps) × one column per stage (Stage 1 /
  Stage 2 / Stage 3 by default), pre-populated from each stage's
  default values.
* Each cell input id is ``<f.id>-stage<index>-<param_name>``.
* The "Stage strategy" preset dropdown rewrites the per-stage
  ``enabled`` checkboxes (and only them — convergence knobs stay
  the user's).
* ``collectForm`` rebuilds the list-of-dicts payload, ready for
  the server-side ``coerce_to_field_type`` ``List[<dataclass>]``
  branch to turn back into ``List[StageSpec]``.

Runs against a minimal Flask host that just serves
``form-schema.js`` + the CSS — no need for the full molbuilder
app for these JS-only assertions.
"""
from __future__ import annotations

import textwrap
import threading

import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# Stage-table fixture mirroring the shape ``_shared.py``'s
# stage-table branch emits for PySCFConfig.stages.  Three stages,
# nine per-stage fields, defaults from the publication-guide.
STAGE_TABLE_SCHEMA = {
    "config":    "PySCFConfig",
    "id_prefix": "p",
    "sections": [{
        "name": "Compute & budget",
        "fields": [{
            "name":            "stages",
            "id":              "p-stages",
            "kind":            "stage-table",
            "label":           "Optimization stages",
            "help":            "Per-stage knobs.",
            "tier":            "advanced",
            "workflow_group":  "stage",
            "default": [
                {"name": "stage1", "enabled": True,  "conv_tol": 1e-7,
                 "gmax": 2e-3, "grms": 1.3e-3, "dmax": 7.2e-3,
                 "drms": 4.8e-3, "etol": 1e-5, "max_steps": 50},
                {"name": "stage2", "enabled": True,  "conv_tol": 1e-9,
                 "gmax": 4.5e-4, "grms": 3e-4, "dmax": 1.8e-3,
                 "drms": 1.2e-3, "etol": 1e-6, "max_steps": 200},
                {"name": "stage3", "enabled": False, "conv_tol": 1e-10,
                 "gmax": 1.5e-5, "grms": 1e-5, "dmax": 6e-5,
                 "drms": 4e-5, "etol": 1e-6, "max_steps": 100},
            ],
            "stage_fields": [
                {"name": "name",      "label": "Name",     "kind": "text",
                 "default": "stage1", "pattern": "^[A-Za-z0-9_]+$"},
                {"name": "enabled",   "label": "Run",      "kind": "checkbox",
                 "default": True},
                {"name": "conv_tol",  "label": "SCF tol",  "kind": "number",
                 "unit": "Hartree",  "step": "any", "default": 1e-9},
                {"name": "gmax",      "label": "‖F‖∞",     "kind": "number",
                 "unit": "Ha/Bohr",  "step": "any", "default": 4.5e-4},
                {"name": "grms",      "label": "‖F‖RMS",   "kind": "number",
                 "unit": "Ha/Bohr",  "step": "any", "default": 3e-4},
                {"name": "dmax",      "label": "Δx max",   "kind": "number",
                 "unit": "Å",        "step": "any", "default": 1.8e-3},
                {"name": "drms",      "label": "Δx RMS",   "kind": "number",
                 "unit": "Å",        "step": "any", "default": 1.2e-3},
                {"name": "etol",      "label": "ΔE tol",   "kind": "number",
                 "unit": "Hartree",  "step": "any", "default": 1e-6},
                {"name": "max_steps", "label": "Max steps", "kind": "int",
                 "min": 1, "max": 10000, "default": 200},
            ],
        }],
    }],
}


@pytest.fixture
def flask_server():
    """Tiny Flask host that serves the form-schema bundle + an
    inline page with a container.  Same pattern as
    test_form_schema_setvalues_js.py."""
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
        return textwrap.dedent("""
            <!doctype html><html><body>
              <div id="container"></div>
              <link rel="stylesheet" href="/lib/form-schema.css">
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


def _mount(page, base_url, schema):
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.formSchema "
        "      && typeof window.molbuilder.formSchema.renderForm "
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
    return page.evaluate(
        "() => {"
        "  const c = document.getElementById('container');"
        "  return window.molbuilder.formSchema.collectForm("
        "    c, window.__schema_for_test);"
        "}"
    )


# --------------------------------------------------------------------- #
#  Rendering                                                            #
# --------------------------------------------------------------------- #


def test_stage_table_renders_three_columns(page, flask_server):
    """One <th> per stage in the header row."""
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    col_headers = page.locator(
        ".schema-stage-table thead th.stage-col-header"
    ).all_text_contents()
    assert col_headers == ["Stage 1", "Stage 2", "Stage 3"]


def test_stage_table_renders_one_row_per_param(page, flask_server):
    """One <tr> per stage_field (9 rows)."""
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    rows = page.locator(".schema-stage-table tbody tr").count()
    assert rows == 9


def test_stage_table_cells_pre_populated_from_defaults(page, flask_server):
    """Per-stage default values land in the per-cell inputs."""
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    # Stage 2 SCF tol = 1e-9
    val = page.locator("#p-stages-stage1-conv_tol").input_value()
    assert float(val) == pytest.approx(1e-9)
    # Stage 3 enabled = False
    checked = page.locator("#p-stages-stage2-enabled").is_checked()
    assert checked is False


# --------------------------------------------------------------------- #
#  Collect round-trip                                                   #
# --------------------------------------------------------------------- #


def test_stage_table_collect_round_trips_defaults(page, flask_server):
    """No edits: collectForm returns the 3-stage default list-of-
    dicts the schema declared."""
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    vals = _collect(page)
    assert "stages" in vals
    assert len(vals["stages"]) == 3
    assert vals["stages"][0]["name"] == "stage1"
    assert vals["stages"][0]["enabled"] is True
    assert vals["stages"][1]["conv_tol"] == pytest.approx(1e-9)
    assert vals["stages"][2]["enabled"] is False
    assert vals["stages"][2]["max_steps"] == 100


def test_stage_table_collect_observes_user_edits(page, flask_server):
    """User-typed values flow into the collected payload."""
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    # Tighten Stage 1's SCF tolerance.
    page.fill("#p-stages-stage0-conv_tol", "1e-8")
    # Toggle Stage 3 on.
    page.check("#p-stages-stage2-enabled")
    vals = _collect(page)
    assert vals["stages"][0]["conv_tol"] == pytest.approx(1e-8)
    assert vals["stages"][2]["enabled"] is True


# --------------------------------------------------------------------- #
#  Stage strategy preset                                                #
# --------------------------------------------------------------------- #


def test_stage_preset_publishable_disables_stage3(page, flask_server):
    """The 'Publishable' preset enables stages 1+2, disables 3.
    Defaults already match this, so toggle stage 3 ON first to
    check the preset overrides."""
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    page.check("#p-stages-stage2-enabled")
    # Switch the preset
    page.select_option("#p-stages-preset", "publishable")
    assert page.locator("#p-stages-stage0-enabled").is_checked()
    assert page.locator("#p-stages-stage1-enabled").is_checked()
    assert not page.locator("#p-stages-stage2-enabled").is_checked()


def test_stage_preset_loose_only_keeps_stage1(page, flask_server):
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    page.select_option("#p-stages-preset", "loose-only")
    assert page.locator("#p-stages-stage0-enabled").is_checked()
    assert not page.locator("#p-stages-stage1-enabled").is_checked()
    assert not page.locator("#p-stages-stage2-enabled").is_checked()


def test_stage_preset_vib_quality_enables_all_three(page, flask_server):
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    page.select_option("#p-stages-preset", "vib-quality")
    for i in range(3):
        assert page.locator(
            f"#p-stages-stage{i}-enabled").is_checked()


def test_stage_preset_does_not_touch_convergence_knobs(page, flask_server):
    """Switching a preset only flips ``enabled`` flags.  The user's
    custom conv_tol / gmax / etc. survive — that's the contract
    that makes preset switching safe to experiment with."""
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    page.fill("#p-stages-stage0-conv_tol", "5e-8")
    page.fill("#p-stages-stage1-gmax", "8e-4")
    page.select_option("#p-stages-preset", "vib-quality")
    assert float(page.locator(
        "#p-stages-stage0-conv_tol").input_value()) == pytest.approx(5e-8)
    assert float(page.locator(
        "#p-stages-stage1-gmax").input_value()) == pytest.approx(8e-4)


def test_stage_preset_custom_is_no_op(page, flask_server):
    """Selecting 'Custom (manual)' is the explicit no-op so a user
    can return to manual editing without losing their state."""
    _mount(page, flask_server, STAGE_TABLE_SCHEMA)
    # Toggle stage 3 ON manually.
    page.check("#p-stages-stage2-enabled")
    page.select_option("#p-stages-preset", "custom")
    assert page.locator("#p-stages-stage2-enabled").is_checked()
