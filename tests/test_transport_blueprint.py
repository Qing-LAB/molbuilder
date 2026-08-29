"""Transport-calculation blueprint tests.

Pins the contract of the schema endpoint + the page template + the
static JS module the page depends on.  The page is the COMPOSITE's
describe surface since P7b (cite -> bias -> send); the engine-backed
render endpoint stays as the registry's validation surface.
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

    def test_every_field_carries_engine_key_metadata(self):
        """Per the 2026-05-26 decision (web-api.md § 4) + the
        2026-06-10 post-ship review: every form field MUST declare
        an ``engine_key`` in its metadata so users see exactly
        which keyword the field writes into the generated script.
        SiestaConfig and PySCFConfig already pin
        this; TransportConfig was the post-review gap.

        Pin so a future field addition that forgets ``engine_key``
        surfaces at test time instead of as a silent UX hole.
        """
        from dataclasses import fields
        from molbuilder.config.transport import TransportConfig
        missing = [
            f.name for f in fields(TransportConfig)
            if "engine_key" not in f.metadata
        ]
        assert not missing, (
            f"TransportConfig fields missing engine_key metadata: "
            f"{missing}.  Add engine_key to the field declaration "
            f"in molbuilder/config/transport.py — see the existing "
            f"fields for the convention "
            f"('(molbuilder: ...)' for selector/path fields, "
            f"the actual engine keyword string otherwise)."
        )

    def test_schema_serves_only_the_override_lane(self, web):
        """The tab's form is the OVERRIDE lane: the electronic contract
        is the citation's to say, and a field the describe door refuses
        BY NAME must not be offered as an input (found rendered
        2026-08-29 — ten sealed fields as editable inputs, the bias
        asked twice).  The filter empties System and Electrodes whole,
        so the served sections are the three that carry transport-only
        knobs, still in ``_form_section_order`` order."""
        from molbuilder.transport.stages import SEALED_TRANSPORT_FIELDS
        body = web.get("/api/transport/schema").get_json()
        sections = body["schema"]["sections"]
        assert [s["name"] for s in sections] == [
            "Transmission", "NEGF", "Runtime",
        ]
        offered = {f["name"] for s in sections for f in s["fields"]}
        leaked = offered & SEALED_TRANSPORT_FIELDS
        assert not leaked, (
            f"sealed fields served as form inputs: {sorted(leaked)} — "
            f"the describe door refuses these by name, so offering "
            f"them is a guaranteed 400"
        )

    def test_engine_choices_are_registered_engines(self):
        """Every engine the form could offer must be one the registry
        answers for — a choice ``get_engine`` refuses
        (``UnknownEngineError``) is a trap, not an option.  Until
        2026-08-29 the metadata offered ``pyscf-negf``, which no
        backend ever registered."""
        from dataclasses import fields as _fields
        import molbuilder.transport  # noqa: F401 -- registration side-effect
        from molbuilder.config.transport import TransportConfig
        from molbuilder.transport.engine_base import registered_engines
        engine_field = next(f for f in _fields(TransportConfig)
                            if f.name == "engine")
        choices = engine_field.metadata["choices"]
        registered = set(registered_engines())
        unknown = [c for c in choices if c not in registered]
        assert not unknown, (
            f"engine choices offer unregistered backends: {unknown} "
            f"(registered: {sorted(registered)}).  A backend that "
            f"registers itself adds its choice back in the same commit."
        )
        assert "transiesta" in choices

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

    def test_builder_kinds_for_sequence_fields(self):
        """The BUILDER's branch pins, kept at the builder (the served
        schema filters these sealed fields out, but the branches they
        regression-pin are shared by every config form):

        * ``Sequence[float]`` → ``comma-floats`` with the factory
          default serialized as a comma-string (2026-06-11: it fell
          through to ``text`` with a blank input);
        * ``Tuple[int, int, int]`` → ``int-triple`` with three
          labelled spinners (same review: it was a free-text field).
        """
        from molbuilder.web.blueprints._shared import (
            dataclass_to_form_schema)
        from molbuilder.config.transport import TransportConfig
        schema = dataclass_to_form_schema(TransportConfig, "t")
        by_name = {f["name"]: f
                   for s in schema["sections"] for f in s["fields"]}
        bias = by_name["bias_voltages_v"]
        assert bias["kind"] == "comma-floats"
        assert bias["default"] == "0.0"
        kmesh = by_name["k_mesh_transverse"]
        assert kmesh["kind"] == "int-triple"
        assert kmesh["default"] == [1, 1, 1]
        assert kmesh["labels"] == ["x", "y", "z"]


class TestTransportPageRendering:

    def test_page_loads_at_canonical_route(self, web):
        r = web.get("/transport-calculation")
        assert r.status_code == 200

    def test_page_includes_form_container(self, web):
        body = web.get("/transport-calculation").data.decode()
        assert 'id="transport-form-container"' in body
        # the composite card (P7b): cite + bias + send -- the Generate
        # button retired with the bundle road
        assert 'id="transport-junction-btn"' in body
        assert 'id="transport-send-btn"' in body
        assert 'transport-generate-btn' not in body

    def test_page_loads_form_schema_helper_then_core(self, web):
        """form-schema.js MUST load BEFORE lib/transport/core.js so
        ``window.molbuilder.formSchema`` is defined when core.js's
        IIFE runs."""
        body = web.get("/transport-calculation").data.decode()
        assert "lib/form-schema.js" in body
        assert "lib/transport/core.js" in body
        assert body.index("lib/form-schema.js") \
               < body.index("lib/transport/core.js")

    def test_send_button_is_disabled_until_a_junction_is_cited(self, web):
        """The composite's one hard requirement is the citation
        (transport-design.md 4.1); the button says so and starts
        disabled -- core.js enables it when a junction is picked."""
        body = web.get("/transport-calculation").data.decode()
        assert (
            'id="transport-send-btn"' in body
            and 'disabled' in body.split('id="transport-send-btn"')[1][:200]
        ), "Send must start disabled"
        assert "Task setup" in body

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

    def test_core_js_reads_no_sidebar_structure_channel(self, web):
        """P7b review (user, 2026-08-29): the CITATION is the tab's one
        driver -- the viewer shows the cited junction's structure, so a
        sidebar commit channel would be a second source for the
        composite's one fact (molview.md 9.3a, one level up).  The tab
        subscribes to NEITHER commit nor change."""
        import re
        js = web.get("/static/lib/transport/core.js").data.decode()
        code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
        code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
        assert "onCommit" not in code, (
            "the sidebar commit channel is back -- the citation drives")
        assert "onChange" not in code
        assert "_adoptCitation" in code, (
            "the cite flow is the one structure door")
