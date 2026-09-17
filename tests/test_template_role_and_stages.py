"""`role` and `stages` — the third answerer, and which rung may own an item.

Two declarations added 2026-09-16 (`engines/template.md` § 6.4;
`engines/transport.md` § 2a.3), both siblings of `allocation`:

* **`role`** — *the stage's role answers this*.  A list of KINDS, like
  `citation`, because `solution_method` is the rung's business for transport
  and an ordinary choice for an optimization.
* **`stages`** — *which rungs may carry their own value*.  `calculations` one
  level down.  Absent means any rung may, which is the optimization ladder's
  behaviour and must stay that way.

What is asserted here is the BEHAVIOUR each buys, not that the field exists:
a role item is not offered and carries no value, and an item's stage list
routes it to the rung that owns it.  `tests/test_catalogue_agreement.py`
covers the rows themselves.
"""
from __future__ import annotations

import pytest

from molbuilder import template as T


@pytest.fixture(scope="module")
def cat():
    return T.catalogue()


class TestRole:
    def test_a_role_item_carries_no_value_for_that_kind(self, cat):
        """The point of the marker: no description can claim to have set it.

        A transport template must not answer `solution_method` -- the rung
        does.  An OPTIMIZATION template must still answer it, which is the
        half that makes this per-kind rather than a boolean.
        """
        from molbuilder.config.siesta import SiestaConfig
        cfg = SiestaConfig(system_label="x")
        t_tr = T.read_template(T.template_with_values(
            cfg, engine="siesta", calculation="transport"))
        t_op = T.read_template(T.template_with_values(
            cfg, engine="siesta", calculation="optimization"))
        assert T.one(t_tr, "solution_method").value is None, (
            "transport: the rung decides the solver, so the template must "
            "not carry a value for it")
        assert T.one(t_op, "solution_method").value is not None, (
            "optimization: nothing else decides it, so the person does -- "
            "if this is None the marker was made a boolean by mistake")

    def test_the_declared_role_items_are_the_ones_the_map_names(self, cat):
        """Both directions, so neither drift is silent.

        `engines/transport.md` § 2a.13 lists what the stage's role fixes.  A
        row that quietly gains `role` disappears from every surface with no
        other symptom; one that loses it starts offering a choice with one
        correct answer.
        """
        assert {i.name for i in T.select(cat, engine="siesta", role=True)} == {
            "solution_method", "wrap_into_cell", "ts_hs_save"}

    def test_a_role_item_is_not_offered_as_a_column(self, cat):
        """The reader that makes the declaration bite.

        Without this the marker would be a control that does nothing -- the
        defect `electrode_kz` shipped with.
        """
        from molbuilder.web.blueprints.build import api_build_schema  # noqa
        import molbuilder.web as _w
        from flask import Flask
        from molbuilder.web.blueprints.build import bp
        app = Flask(__name__)
        app.register_blueprint(bp)
        with app.test_client() as c:
            tr = c.get("/api/task-setup/columns?engine=siesta"
                       "&calculation=transport").get_json()
            op = c.get("/api/task-setup/columns?engine=siesta"
                       "&calculation=optimization").get_json()
        tr_names = {i["name"] for i in tr["items"]}
        op_names = {i["name"] for i in op["items"]}
        assert "solution_method" not in tr_names, (
            "a role item must not be offered as a stage-table column")
        assert "solution_method" in op_names, (
            "...and must still be offered where it is a real choice -- "
            "without this half the test passes on an empty column list")


class TestStages:
    def test_absent_means_any_rung_may_own_it(self, cat):
        """The ordinary case, and the one that must not regress.

        Every optimization item is unconstrained, so the ladder keeps
        behaving as `engines/stages.md` § 6.2 says: any promoted field may
        vary per rung.
        """
        unconstrained = [i for i in T.select(cat, engine="siesta")
                         if not i.stages]
        assert len(unconstrained) > 40, (
            "most items declare no stage list; if this collapsed, something "
            "constrained the optimization ladder by accident")
        assert all(i.name in {x.name for x in
                              T.select(cat, engine="siesta", stages="coarse")}
                   for i in unconstrained), (
            "an item with no stage list must match every rung asked about")

    def test_each_transport_item_is_routed_to_the_rung_that_owns_it(self, cat):
        """§ 2a.13's map, as data rather than prose.

        This is the basis on which an override reaches the right deck -- the
        thing whose absence sent every transport override to the `device`
        rung, so a value the transmission owned never reached the
        transmission's deck.
        """
        owned = {rung: {i.name for i in
                        T.select(cat, engine="siesta", stages=rung) if i.stages}
                 for rung in ("seed", "electrode_L", "electrode_R",
                              "device", "transmission")}
        assert owned["electrode_L"] == owned["electrode_R"] == {
            "electrode_kz", "ts_hs_save"}
        assert owned["device"] == {"negf_eq_pole_ev", "negf_neq_eta_ev",
                                   "elecs_bulk", "bias_voltage_v"}
        assert owned["seed"] == set(), (
            "the seed owns nothing of its own -- it only obeys")
        assert "transmission_emin_ev" in owned["transmission"]
        assert "transmission_emin_ev" not in owned["device"], (
            "THE LIVE DEFECT, as a test: the T(E) window is the "
            "transmission's, and routing it to the device is what made a "
            "person's setting reach the deck tbtrans never reads")

    def test_the_declarations_survive_a_round_trip(self, cat):
        """A marker that does not survive read/write is not a declaration.

        Through the door a calculation actually uses -- `template_with_values`
        narrows to one engine and one kind, and both markers must come back,
        because a reader of a transport template that sees `solution_method`
        carrying no value needs the file itself to say why.
        """
        from molbuilder.config.siesta import SiestaConfig
        text = T.template_with_values(SiestaConfig(system_label="x"),
                                      engine="siesta",
                                      calculation="transport")
        back = T.read_template(text)
        assert T.one(back, "solution_method").role == ("transport",)
        assert T.one(back, "transmission_emin_ev").stages == ("transmission",)
        assert T.one(back, "electrode_kz").stages == ("electrode_L",
                                                      "electrode_R")
