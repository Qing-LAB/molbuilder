"""Transport-calculation blueprint tests (Phase D form skeleton).

Pins the contract of the schema endpoint + the page template +
the static JS module the page depends on.  Engine-backed
rendering tests follow when transiesta_engine.py / pyscf_negf_
engine.py land; today the surface is just the form.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def web():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


class TestTransportSchemaEndpoint:

    def test_returns_ok_envelope(self, web):
        r = web.get("/api/transport/schema")
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is True
        assert "schema" in body

    def test_schema_carries_TransportConfig_sections(self, web):
        """Section order MUST come from
        ``TransportConfig._form_section_order`` so the workflow-
        order (System → Geometry → Electrodes → Transmission →
        NEGF → Runtime) is stable independent of field-declaration
        order in the dataclass."""
        body = web.get("/api/transport/schema").get_json()
        sections = body["schema"]["sections"]
        names = [s["name"] for s in sections]
        assert names == [
            "System", "Geometry", "Electrodes",
            "Transmission", "NEGF", "Runtime",
        ]

    def test_schema_includes_engine_field_with_choices(self, web):
        """The engine selector is the load-bearing System field —
        future engine backends register through it.  Pin its
        presence + that it has at least the two planned choices."""
        body = web.get("/api/transport/schema").get_json()
        engine_field = None
        for s in body["schema"]["sections"]:
            for f in s.get("fields", []):
                if f.get("name") == "engine":
                    engine_field = f
                    break
            if engine_field:
                break
        assert engine_field is not None, (
            "schema missing 'engine' field"
        )
        choices = engine_field.get("choices") or []
        choice_values = [c["value"] if isinstance(c, dict) else c
                         for c in choices]
        assert "transiesta" in choice_values
        assert "pyscf-negf" in choice_values

    def test_schema_carries_field_metadata_for_render(self, web):
        """Every field must carry the metadata form-schema.js needs
        to render (kind + label).  A bare ``{name: ...}`` blob
        would crash renderForm at runtime."""
        body = web.get("/api/transport/schema").get_json()
        missing = []
        for s in body["schema"]["sections"]:
            for f in s.get("fields", []):
                if not f.get("kind"):
                    missing.append((s["name"], f.get("name", "?")))
                if not f.get("label"):
                    missing.append((s["name"], f.get("name", "?")))
        assert not missing, (
            f"fields missing render metadata: {missing}"
        )


class TestTransportPageRendering:

    def test_page_loads_at_canonical_route(self, web):
        r = web.get("/transport-calculation")
        assert r.status_code == 200

    def test_page_includes_form_container(self, web):
        body = web.get("/transport-calculation").data.decode()
        assert 'id="transport-form-container"' in body
        assert 'id="transport-generate-btn"' in body

    def test_page_loads_form_schema_helper_then_core(self, web):
        """form-schema.js MUST load BEFORE lib/transport/core.js so
        ``window.molbuilder.formSchema`` is defined when core.js's
        IIFE runs."""
        body = web.get("/transport-calculation").data.decode()
        assert "lib/form-schema.js" in body
        assert "lib/transport/core.js" in body
        assert body.index("lib/form-schema.js") \
               < body.index("lib/transport/core.js")

    def test_generate_button_is_disabled_with_hint(self, web):
        """Engines aren't wired; the button MUST stay disabled so
        users don't click expecting output.  The hint span tells
        them why."""
        body = web.get("/transport-calculation").data.decode()
        # Button starts disabled (HTML attribute).
        assert (
            'id="transport-generate-btn"' in body
            and 'disabled' in body.split('id="transport-generate-btn"')[1][:200]
        ), "Generate button must start disabled"
        # Hint text mentions the planned engines so users know what's coming.
        assert "TranSIESTA" in body or "PySCF-NEGF" in body

    def test_active_tab_marker_set(self, web):
        """``active_tab`` must equal ``transport-calculation`` so
        the shared header partial marks the right tab is-active."""
        body = web.get("/transport-calculation").data.decode()
        # The is-active marker should be on the
        # /transport-calculation tab link only.
        import re
        m = re.search(
            r'<a[^>]*href="/transport-calculation"[^>]*class="[^"]*is-active[^"]*"',
            body,
        )
        assert m, "Transport tab must mark itself active in the nav"


class TestTransportCoreJsServed:
    """The static JS module that drives the form is served + carries
    the contract the page depends on (schema fetch URL, render
    container id, persistence key)."""

    def test_core_js_served(self, web):
        r = web.get("/static/lib/transport/core.js")
        assert r.status_code == 200

    def test_core_js_targets_schema_endpoint(self, web):
        js = web.get("/static/lib/transport/core.js").data.decode()
        assert "/api/transport/schema" in js

    def test_core_js_renders_into_known_container(self, web):
        js = web.get("/static/lib/transport/core.js").data.decode()
        assert 'transport-form-container' in js

    def test_core_js_subscribes_to_commit_channel_only(self, web):
        """Per the universal interaction model (B.5.2): sidebar
        dblclick = commit.  Pin that this tab subscribes to
        ``onCommit`` and DOES NOT fall back to ``onChange``.

        BOMB-3 fix (2026-06-07): the original code fell back to
        ``onChange`` if ``onCommit`` was absent.  ``onChange``
        fires on every preview click + fires-on-subscribe — using
        it for transport would clobber the status line on every
        browse-click and (once engines wire in) re-build the
        geometry on every browse-click.  Build + Spectra tolerate
        the fallback because they have form-dirty + warning-modal
        gates upstream; Transport has neither, so the fallback was
        actively wrong."""
        js = web.get("/static/lib/transport/core.js").data.decode()
        assert "onCommit" in js
        # The onChange fallback path is intentionally absent.  The
        # comment block documenting the BOMB-3 rationale mentions
        # ``onChange`` in prose; pin against the actual call site
        # using a fragment that wouldn't appear in a comment.
        assert ".onChange(" not in js, (
            "transport/core.js must NOT call proj.onChange — "
            "BOMB-3 fix removed the fallback"
        )
        assert "proj.onChange.bind" not in js
