"""Transport-calculation blueprint tests.

Pins the contract of the schema endpoint + the page template + the
static JS module the page depends on.  The page is the COMPOSITE's
describe surface (cite -> bias -> send).
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
        tr_fields = [f for s in tr["schema"]["sections"] for f in s["fields"]]
        tr_names = {f["name"] for f in tr_fields}
        # tbtrans runs no SCF (`transport.md` § 6.1b): its tab offers none
        # of the SCF settings.  (Its temperature is the shared value, on
        # the shared panel -- never a rung's control.)
        assert {"transmission_n_points", "tbt_k_grid"} <= tr_names
        assert "dm_tolerance" not in tr_names and "electrode_kz" not in tr_names
        # ONE ID PER CONTROL ON THE PAGE (`transport.md` § 3.8.2a): the five
        # tabs' copies of one item never share an id.
        ids = {}
        for rung in TRANSPORT_STAGES:
            got = web.get(f"/api/transport/schema?surface=rung&rung={rung}").get_json()
            ids[rung] = {f["id"] for s in got["schema"]["sections"] for f in s["fields"]}
        for a in TRANSPORT_STAGES:
            for b in TRANSPORT_STAGES:
                if a < b:
                    assert ids[a].isdisjoint(ids[b]), (a, b, ids[a] & ids[b])
        r = web.get("/api/transport/schema?surface=rung&rung=lead")
        assert r.status_code == 400 and "rung must name" in r.get_json()["error"]

    def test_the_rung_surface_never_offers_a_shared_value(self, web):
        """`engines/transport.md` § 3.8.2: the per-rung form edits a rung's
        override bag and never offers a value that binds every rung -- the
        catalogue's `shared` marker (§ 3.8.6, decided 2026-09-24), nor a
        role-fixed one, nor the machine's.  Offering one is a control the
        describe door refuses by name (the trap of 2026-08-29 and the
        revert of 2026-09-23).  A role-fixed one is SHOWN, read-only at the
        rung's answer (`locked`, K7: § 6.6 obligation 3) -- never a
        control."""
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
            fields = [f for s in body["schema"]["sections"]
                      for f in s["fields"]]
            offered = {f["name"] for f in fields if "locked" not in f}
            echoed = {f["name"] for f in fields if "locked" in f}
            assert offered, "the per-rung form offers nothing"
            assert not (offered & shared), sorted(offered & shared)
            assert not (offered & role), sorted(offered & role)
            assert not (offered & machine), sorted(offered & machine)
            assert echoed <= role, sorted(echoed - role)
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
        r = web.get("/transport-calculation")
        assert r.status_code == 200
        body = r.data.decode()
        assert 'id="transport-form-container"' in body
        assert 'id="transport-junction-btn"' in body
        assert 'id="transport-send-btn"' in body
        assert 'transport-generate-btn' not in body


    def test_send_button_is_disabled_until_a_junction_is_cited(self, web):
        """The composite's one hard requirement is the citation
        (engines/transport.md § 3.1); the button says so and starts
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


def test_core_js_served(web):
    """The tab's script is served at the path the page loads it from."""
    assert web.get("/static/lib/transport/core.js").status_code == 200


def test_a_rung_tab_offers_no_run_setting(web):
    """A run setting is the rung's run card's, never a rung override
    (`stages.md` § 6.8d, plan § 5w K5): the rung tabs offered the solver,
    `block_size` and `parallel_over_k` -- each then refused at the save
    with *"put it on the run card"* (the K5 review's A2, 2026-09-30)."""
    from molbuilder.template import run_settings
    body = web.get("/api/transport/schema?surface=rung").get_json()
    names = {f["name"] for s in body["schema"]["sections"]
             for f in s["fields"]}
    assert names, "the rung surface offered nothing"
    assert "diag_algorithm" in run_settings("siesta")
    assert not (names & run_settings("siesta")), sorted(
        names & run_settings("siesta"))
