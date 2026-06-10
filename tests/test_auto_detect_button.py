"""Tests for the Auto-detect button on the Optimization tab
(``/structure-optimization``).  Phase 2 of the chemistry middle-
layer work; see ``docs/protocols/scientific-validation.md`` § 2.5
for the scientific-guard role.

The Markup test runs as a plain Flask test_client request; the
end-to-end test drives the browser via Playwright (load a Fe-
bearing fixture, click Auto-detect, assert form fields get
populated + rationale appears).
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


# --------------------------------------------------------------------- #
#  Markup test — pin the new template scaffolding                       #
# --------------------------------------------------------------------- #


def test_auto_detect_markup_present():
    """The Auto-detect button + status line + rationale/warnings/
    metals panel are rendered into /structure-optimization at page-
    serve time.  Pin so a future template refactor that drops the
    scaffolding silently disabling the feature would surface."""
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    c = create_app(config={}).test_client()
    r = c.get("/structure-optimization")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    for needle in (
        'id="auto-detect-btn"',
        'id="auto-detect-status"',
        'id="auto-detect-panel"',
        'id="auto-detect-rationale"',
        'id="auto-detect-warnings"',
        'id="auto-detect-metals"',
    ):
        assert needle in body, (
            f"{needle} missing from /structure-optimization markup — "
            f"Auto-detect button cannot mount"
        )
    # Initial-state contract: button starts disabled (no structure
    # loaded yet); panel hidden.
    btn_attrs = body.split('id="auto-detect-btn"', 1)[1].split(">", 1)[0]
    assert "disabled" in btn_attrs, (
        "Auto-detect must start disabled — enabled only after a "
        "structure is loaded"
    )
    assert 'id="auto-detect-panel"' in body
    panel_attrs = body.split('id="auto-detect-panel"', 1)[1].split(">", 1)[0]
    assert "hidden" in panel_attrs, (
        "Auto-detect panel must start hidden — opens only after a "
        "successful analyze"
    )


# --------------------------------------------------------------------- #
#  E2E — Fe fixture, click Auto-detect, assert form-fill                #
# --------------------------------------------------------------------- #


pytest.importorskip("playwright.sync_api")


@pytest.fixture
def flask_server():
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)


def test_auto_detect_button_populates_both_engine_forms(
        page, flask_server, tmp_path, monkeypatch):
    """End-to-end: a Fe-bearing structure loads into the viewer,
    user clicks Auto-detect, the response from
    ``/api/structure/analyze`` populates (charge, spin, method) on
    BOTH the SIESTA and PySCF sub-forms, and the rationale panel
    surfaces the chemistry conclusion.

    Pins the load-bearing claim of the Phase-1 refactor: one
    analyzer + per-engine adapters = one click pre-fills every
    engine's chemistry-driven defaults.

    Geometry: Fe atom alone.  Z=26 (even), default 2S=2 (even) —
    parity matches → no warning, just the suggestion.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "auto_detect_proj"
    proj.mkdir()
    target = proj / "fe.xyz"
    target.write_text("1\nFe atom for auto-detect test\nFe 0 0 0\n")

    base = flask_server
    page.goto(f"{base}/structure-optimization",
              wait_until="domcontentloaded")
    # Wait for the runtime + projects API to be available before we
    # programmatically set the picker selection.
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.projects "
        "      && typeof window.molbuilder.projects.setShared "
        "             === 'function'",
        timeout=10000,
    )
    # Sidebar single-click semantics: setShared is the preview path.
    # The Load button reads from the candidate; we then click Load.
    page.evaluate(
        "(p) => window.molbuilder.projects.setShared("
        "  p.substring(0, p.lastIndexOf('/')), p)",
        str(target),
    )
    # Load button becomes enabled once a loadable file is picked.
    page.wait_for_function(
        "() => !document.getElementById('load-from-sidebar-btn').disabled",
        timeout=5000,
    )
    page.locator("#load-from-sidebar-btn").click()
    # Wait for the load to complete — load-status flips to ok.
    page.wait_for_function(
        "() => /Loaded \\d+-atom/.test("
        "  document.getElementById('load-status').textContent)",
        timeout=15000,
    )
    # Auto-detect button enables after the load completes.
    page.wait_for_function(
        "() => !document.getElementById('auto-detect-btn').disabled",
        timeout=5000,
    )

    page.locator("#auto-detect-btn").click()
    page.wait_for_function(
        "() => /Applied to both forms/.test("
        "  document.getElementById('auto-detect-status').textContent)",
        timeout=10000,
    )

    # SIESTA sub-form populated.  The schema field names are
    # ``net_charge`` / ``spin_polarized`` / ``spin_total`` (matching
    # the SiestaSuggestedParams dataclass field names).  The schema's
    # ``f.id`` is the DOM input id; we look up by name via the
    # form-container so the test doesn't depend on the id naming
    # scheme.
    def _field_value(container_id: str, field_name: str):
        """Read a field's current value via the JS form-schema API."""
        return page.evaluate(
            "(args) => {"
            "  const container = document.getElementById(args.cid);"
            "  if (!container) return null;"
            "  for (const sect of args.schema.sections) {"
            "    for (const f of sect.fields) {"
            "      if (f.name !== args.fname) continue;"
            "      const elx = container.querySelector('#' + CSS.escape(f.id));"
            "      if (!elx) return null;"
            "      if (f.kind === 'checkbox') return elx.checked;"
            "      return elx.value;"
            "    }"
            "  }"
            "  return null;"
            "}",
            {
                "cid":    container_id,
                "fname":  field_name,
                "schema": page.evaluate(
                    "(eng) => window.__molbuilder_form_schemas_for_test "
                    "        ? window.__molbuilder_form_schemas_for_test[eng] "
                    "        : null",
                    container_id.split("-")[0],
                ),
            },
        )

    # Fall back to a simpler check: querySelector for inputs by
    # name attribute is NOT reliable here (form-schema renders by
    # id, not name), so we use known schema field ids by walking
    # the live schemas via window.molbuilder.formSchema.collectForm.
    siesta_collected = page.evaluate(
        "() => {"
        "  const fs = (window.molbuilder || {}).formSchema;"
        "  const c = document.getElementById('siesta-form-container');"
        "  const sch = window.__molbuilder_siesta_schema_for_test;"
        "  return fs && sch && c ? fs.collectForm(c, sch) : null;"
        "}"
    )
    # The test hook __molbuilder_siesta_schema_for_test may not be
    # exposed by viewer.js (it isn't today); collectForm above
    # returns null if not.  Use the DOM-level checks below instead.

    # SIESTA fields by DOM input name (the schema renders an
    # ``input[name="<field-name>"]`` for each field):
    # Note: name attributes may not be set by makeNumber/makeCheckbox;
    # walk by id instead, which is the schema's f.id.  The IDs follow
    # the convention of the schema's ``id`` field — we read them
    # back through the running schema via formSchema.collectForm.

    # Simpler robust approach: trigger collectForm on each sub-form
    # via the existing public API, then assert the values.
    siesta_vals = page.evaluate(
        "async () => {"
        "  const fs = (window.molbuilder || {}).formSchema;"
        "  const sch = await fs.fetchSchema('siesta');"
        "  const c = document.getElementById('siesta-form-container');"
        "  return fs.collectForm(c, sch);"
        "}"
    )
    pyscf_vals = page.evaluate(
        "async () => {"
        "  const fs = (window.molbuilder || {}).formSchema;"
        "  const sch = await fs.fetchSchema('pyscf');"
        "  const c = document.getElementById('pyscf-form-container');"
        "  return fs.collectForm(c, sch);"
        "}"
    )

    # PySCF: charge=0, spin=2 (Fe(II) intermediate), method=UKS
    assert pyscf_vals.get("charge") == 0
    assert pyscf_vals.get("spin")   == 2
    assert pyscf_vals.get("method") == "UKS"

    # SIESTA: net_charge=0, spin_polarized=True, spin_total=2.0
    assert siesta_vals.get("net_charge")     == 0
    assert siesta_vals.get("spin_polarized") is True
    assert siesta_vals.get("spin_total")     == 2.0

    # Rationale panel: visible (not hidden) and carries
    # engine-agnostic "open-shell" + the metal symbol.
    is_hidden = page.evaluate(
        "() => document.getElementById('auto-detect-panel').hidden"
    )
    assert is_hidden is False, (
        "auto-detect-panel did not become visible after the click"
    )
    rationale = page.locator("#auto-detect-rationale").inner_text()
    assert "Fe" in rationale
    assert "open-shell" in rationale.lower()


def test_auto_detect_button_disabled_without_loaded_structure(
        page, flask_server, tmp_path, monkeypatch):
    """Auto-detect MUST start disabled and stay disabled until a
    structure loads.  Without the guard, the user would post an
    empty body and get a 400 — confusing UX.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    page.goto(f"{flask_server}/structure-optimization",
              wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.getElementById('auto-detect-btn') !== null",
        timeout=5000,
    )
    assert page.locator("#auto-detect-btn").is_disabled()
