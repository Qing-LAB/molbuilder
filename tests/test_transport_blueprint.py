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

    def test_no_lane_offers_the_contract_fields(self, web):
        """The describe door refuses the electronic-contract fields
        UNCONDITIONALLY, so no value of ``?contract=`` may offer them.

        This asserted the opposite until 2026-09-16 — that ``open``
        (a form-B citation: a labeled pair has no deck) offers them —
        which was the SEALED reading `engines/transport.md` § 2a.7
        reversed.  Under the ruling the cited run DEFAULTS these values
        into the calculation's TEMPLATE, which every form gets: form A
        from the cited deck, form B from the catalogue
        (``citation_defaults.siesta_config_from_citation``).  They are
        editable there and are never a per-stage override for anyone,
        so a lane that rendered them served seven controls the door was
        guaranteed to 400 — the exact trap this filter exists to stop.
        """
        from molbuilder.transport.stages import (CONTRACT_FIELDS,
                                                 SEALED_ALWAYS)
        for lane in ("", "?contract=open", "?contract=cited"):
            body = web.get(f"/api/transport/schema{lane}").get_json()
            offered = {f["name"] for s in body["schema"]["sections"]
                       for f in s["fields"]}
            assert not (offered & CONTRACT_FIELDS), (
                f"lane {lane or '(default)'!r} offers contract fields "
                f"{sorted(offered & CONTRACT_FIELDS)} -- the describe "
                f"door refuses them by name whatever the citation form")
            assert not (offered & SEALED_ALWAYS)

    def test_schema_serves_only_the_override_lane(self, web):
        """The tab's form is the OVERRIDE lane: the electronic contract
        is the citation's to say, and a field the describe door refuses
        BY NAME must not be offered as an input (found rendered
        2026-08-29 — ten sealed fields as editable inputs, the bias
        asked twice).  The filter empties System and Electrodes whole,
        so the served sections are the ones that carry transport-only
        knobs, still in ``_form_section_order`` order.

        **The list grew on 2026-09-15** and the order is now a scientific
        one (`engines/transport.md` § 3.3): what is computed, how sharply,
        what is written, then the machinery.  "NEGF" is gone as a heading --
        it had collected both the density contour AND the five electronic-
        contract fields, which are not NEGF parameters at all."""
        from molbuilder.transport.stages import SEALED_TRANSPORT_FIELDS
        body = web.get("/api/transport/schema").get_json()
        sections = body["schema"]["sections"]
        assert [s["name"] for s in sections] == [
            "Transmission", "Transmission k-sampling", "Spin channel",
            "Broadening", "Outputs", "NEGF density contour", "Leads",
        ], (
            "the override lane's sections, in the order a person decides "
            "in.  System and Electrodes are emptied by the seal filter, "
            "and 'Electronic contract' appears in NO lane: § 2a.7 makes "
            "those the template's to answer, never a per-stage override.  "
            "Runtime and Logging went on 2026-09-16: they held only "
            "`max_memory_mb`, `num_threads` and `log_level`, which `prep` "
            "cannot resolve -- three controls whose use guaranteed that "
            "every later prep refused the whole calculation")
        offered = {f["name"] for s in sections for f in s["fields"]}
        from molbuilder.transport.stages import resolvable_override_names
        unresolvable = offered - resolvable_override_names()
        assert not unresolvable, (
            f"the form offers {sorted(unresolvable)}, which `prep` refuses "
            f"by name -- a control the door is guaranteed to reject is not "
            f"a control")
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
        r = web.get("/transport-calculation")
        assert r.status_code == 200
        body = r.data.decode()
        assert 'id="transport-form-container"' in body
        # the composite card (P7b): cite + bias + send -- the Generate
        # button retired with the bundle road
        assert 'id="transport-junction-btn"' in body
        assert 'id="transport-send-btn"' in body
        assert 'transport-generate-btn' not in body


    def test_send_button_is_disabled_until_a_junction_is_cited(self, web):
        """The composite's one hard requirement is the citation
        (archive/2026-09-01-transport-design.md 4.1); the button says so and starts
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
