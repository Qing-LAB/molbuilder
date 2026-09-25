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

    def test_a_rungs_tab_offers_its_own_items_and_what_any_rung_may_set(self, web):
        """`engines/transport.md` § 3.8.2a: the per-rung form is a tab per
        rung.  `?surface=rung&rung=<name>` answers that tab -- the items
        whose `stages` names the rung, plus the items declaring no rung --
        and never another rung's; every field carries its `stages` so the
        tab can fold the cards that hold none of the rung's own.  The
        un-narrowed answer carries the rung list the strip is built from:
        name, one-based ladder index, one-line note."""
        from molbuilder.transport.stages import RUNG_NOTES, TRANSPORT_STAGES
        body = web.get("/api/transport/schema?surface=rung").get_json()
        assert [r["name"] for r in body["rungs"]] == list(TRANSPORT_STAGES)
        assert [r["index"] for r in body["rungs"]] == [1, 2, 3, 4, 5]
        assert all(r["note"] == RUNG_NOTES[r["name"]] and r["note"]
                   for r in body["rungs"])
        seed = web.get("/api/transport/schema?surface=rung&rung=seed").get_json()
        assert seed["ok"] and seed["rung"] == "seed"
        fields = [f for s in seed["schema"]["sections"] for f in s["fields"]]
        assert fields and all("stages" in f for f in fields)
        assert all(not f["stages"] or "seed" in f["stages"] for f in fields), (
            [f["name"] for f in fields if f["stages"] and "seed" not in f["stages"]])
        names = {f["name"] for f in fields}
        assert "transmission_n_points" not in names and "dm_tolerance" in names
        tr = web.get("/api/transport/schema?surface=rung&rung=transmission").get_json()
        tr_names = {f["name"] for s in tr["schema"]["sections"] for f in s["fields"]}
        assert {"transmission_n_points", "tbt_k_grid", "dm_tolerance"} <= tr_names
        assert "electrode_kz" not in tr_names
        r = web.get("/api/transport/schema?surface=rung&rung=lead")
        assert r.status_code == 400 and "rung must name" in r.get_json()["error"]

    def test_the_rung_surface_never_offers_a_shared_value(self, web):
        """`engines/transport.md` § 3.8.2: the per-rung form edits a rung's
        override bag and never offers a value that binds every rung -- the
        catalogue's `shared` marker (§ 3.8.6, decided 2026-09-24), nor a
        role-fixed one, nor the machine's.  Offering one is a control the
        describe door refuses by name (the trap of 2026-08-29 and the
        revert of 2026-09-23)."""
        from molbuilder.template import catalogue, select
        cat = catalogue()
        shared = {it.name for it in select(cat, engine="siesta", shared=True)
                  if "transport" in it.shared}
        role = {it.name for it in select(cat, engine="siesta", role=True)
                if "transport" in it.role}
        machine = {it.name for it in select(cat, engine="siesta",
                                             allocation=True)}
        for lane in ("", "?surface=rung"):
            body = web.get(f"/api/transport/schema{lane}").get_json()
            assert body["ok"] is True and body["surface"] == "rung"
            offered = {f["name"] for s in body["schema"]["sections"]
                       for f in s["fields"]}
            assert offered, "the per-rung form offers nothing"
            assert not (offered & shared), sorted(offered & shared)
            assert not (offered & role), sorted(offered & role)
            assert not (offered & machine), sorted(offered & machine)
            from molbuilder.transport.stages import resolvable_override_names
            unresolvable = offered - resolvable_override_names()
            assert not unresolvable, (
                f"the form offers {sorted(unresolvable)}, which `prep` "
                f"refuses by name -- a control the door is guaranteed to "
                f"reject is not a control")

    def test_the_shared_surface_offers_every_shared_value_outside_setup(self, web):
        """The other surface: every item the catalogue marks `shared` for
        transport, except the `setup` group -- the identity the description
        derives and the pseudopotential directory the citation supplies
        (`engines/transport.md` § 3.8.2)."""
        from molbuilder.template import catalogue, select
        cat = catalogue()
        expected = {it.name for it in select(cat, engine="siesta", shared=True)
                    if "transport" in it.shared and it.group != "setup"}
        body = web.get("/api/transport/schema?surface=shared").get_json()
        assert body["ok"] is True and body["surface"] == "shared"
        offered = {f["name"] for s in body["schema"]["sections"]
                   for f in s["fields"]}
        assert offered == expected, (sorted(offered ^ expected))
        assert body["source"] == {"kind": "none", "name": ""}

    def test_an_unknown_surface_is_refused(self, web):
        r = web.get("/api/transport/schema?surface=lane")
        assert r.status_code == 400

    # `test_engine_choices_are_registered_engines` deleted 2026-09-17 with the engine registry.

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
