"""The Build form's schema comes from the CATALOGUE (`web/form-schema.md` § 1).

**The presentation does not change.** The JS renderer already takes whatever
schema it is handed and knows nothing about any individual parameter, so this
is one server function pointed at a different source — not a new UI.

What changes is where the data comes from, and one visible consequence: the six
`category` panels are SHARED, so SIESTA and PySCF show the same headings in the
same order for the first time. `section` — free text chosen per engine, so
*"Basis & grid"* and *"Method"* were unrelated words — is gone from this path.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from molbuilder import template as T
from molbuilder.web.blueprints._shared import catalogue_to_form_schema


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_the_panels_are_the_shared_six_in_reading_order(engine):
    """§ 6.2's whole purpose: one panel set serving every engine.

    Order is the vocabulary's own declaration order — the *reading* order — not
    alphabetical and not the order items happen to sit in the file.
    """
    sch = catalogue_to_form_schema(engine)
    names = [s["name"] for s in sch["sections"]]
    assert names == [c for c in T.CATEGORIES if c in names]
    assert names, "no panels at all"


def test_both_engines_draw_their_panels_from_the_same_vocabulary():
    """The claim that could not be made while `section` was per-engine.

    THE CLAIM IS THE VOCABULARY, NOT AN IDENTICAL LIST.  This asserted
    ``a == b`` until 2026-08-15, which held only while both engines happened
    to carry an item in all six categories.  It stopped holding the moment
    PySCF's ``threads`` and ``use_gpu`` moved to the staging surface: they
    were its only ``execution`` items, so PySCF draws five panels and SIESTA
    six.  Neither is wrong -- a panel exists because the engine HAS a
    parameter answering that question, and an engine with none should not be
    shown an empty heading.

    What `template.md` § 6.2 actually buys is that the words are shared and
    ordered, so *"accuracy"* means the same thing and sits in the same place
    on both forms.  That is what is asserted here.
    """
    from molbuilder.template import CATEGORIES
    panels = {e: [s["name"] for s in catalogue_to_form_schema(e)["sections"]]
              for e in ("siesta", "pyscf")}
    for engine, names in panels.items():
        assert set(names) <= set(CATEGORIES), (engine, names)
        # Same relative order as the shared vocabulary -- a panel may be
        # absent, never out of sequence.
        assert names == [c for c in CATEGORIES if c in names], (engine, names)
    # And they are not two disjoint worlds: the engines share most panels.
    assert set(panels["siesta"]) & set(panels["pyscf"])


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_every_item_this_engine_has_reaches_the_form(engine):
    """Membership is TOTAL (§ 7): a parameter the catalogue carries and the
    form omits is a control no user can reach.

    This is what makes the six previously-unreachable SIESTA parameters appear
    — `species_order`, `write_forces`, `write_coor_step`, `write_molwatch_log`,
    `copy_psml`, `kgrid_displacement` — none of which had a `section`.

    **The one exclusion is declared by the item, not listed here** (2026-08-15):
    ``group = "staging"`` says *this parameter is answered by the staging
    surface*. The stage token is the only one today — it is derived from which
    stage is running, so it is not something a person types into the physics
    form at all. Filtering on the declaration rather than on a name means a
    second such parameter needs no edit to this test.
    """
    items = [i for i in T.select(T.read_template(T.load_catalogue()),
                                 engine=engine) if i.group != "staging"]

    # Membership is total PER KIND since the vibration items landed
    # (template.md § 6.3): an item carrying `calculations` reaches the
    # form of exactly those kinds; an item without the key reaches every
    # kind.  The kinds to check come from the items themselves, so a new
    # kind needs no edit here.
    kinds = {"optimization"} | {k for i in items for k in i.calculations}
    for kind in sorted(kinds):
        expect = {i.name for i in items
                  if not i.calculations or kind in i.calculations}
        got = {f["name"]
               for sec in catalogue_to_form_schema(
                   engine, calculation=kind)["sections"]
               for f in sec["fields"]}
        assert got == expect, (engine, kind)

    # And the no-argument call IS the optimization form -- the default
    # spelled out, so the two spellings cannot drift.
    default_fields = {f["name"]
                      for sec in catalogue_to_form_schema(engine)["sections"]
                      for f in sec["fields"]}
    assert default_fields == {i.name for i in items if not i.calculations
                              or "optimization" in i.calculations}


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_a_staging_parameter_is_not_on_the_physics_form(engine):
    """The other half, asserted rather than implied.

    A test that only checks *"everything except staging is present"* passes
    just as happily if the filter stops working and staging appears too — the
    two sets would simply both grow. This asks the question directly.
    """
    fields = {f["name"] for s in catalogue_to_form_schema(engine)["sections"]
              for f in s["fields"]}
    staged = {i.name for i in T.select(T.read_template(T.load_catalogue()),
                                       engine=engine) if i.group == "staging"}
    assert not (fields & staged), (
        f"{sorted(fields & staged)} is set by the staging surface and must "
        f"not appear on the parameter form (user, 2026-08-15).")


def test_the_displacement_gets_a_control_that_can_carry_its_value():
    """§ 57.3, the defect this job had to fix before the field could appear.

    Every tuple used to dispatch to the integer control, which renders
    ``step="1"`` and reads back with ``parseInt``.  ``kgrid_displacement``'s
    useful value is **0.5** — the classic Monkhorst-Pack shift — and that
    control turns it into 0 silently, which is the Gamma-centred grid the user
    was moving off.
    """
    f = {x["name"]: x for s in catalogue_to_form_schema("siesta")["sections"]
         for x in s["fields"]}
    assert f["kgrid"]["kind"] == "int-triple"
    assert f["kgrid_displacement"]["kind"] == "float-triple"
    assert f["kgrid_displacement"]["default"] == [0.0, 0.0, 0.0]


def test_no_control_kind_the_renderer_does_not_know():
    """The schema may not invent widgets.  Every kind it emits must appear in
    the renderer's dispatch, or a field silently falls through to a text box.
    """
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "molbuilder" / "web" /
          "static" / "lib" / "form-schema.js").read_text()
    kinds = {f["kind"] for e in ("siesta", "pyscf")
             for s in catalogue_to_form_schema(e)["sections"]
             for f in s["fields"]}
    missing = [k for k in kinds if f'case "{k}"' not in js]
    assert not missing, f"the schema emits kinds the renderer cannot draw: {missing}"


def test_the_two_grouping_axes_both_survive():
    """§ 1.3: `group` is the OUTER card (when do I set this), `category` the
    inner legend (what question is this).  The outer cards are load-bearing —
    they exist because the stage selector once silently rewrote budget and
    system fields — so every item that had one keeps it.
    """
    fields = [f for s in catalogue_to_form_schema("siesta")["sections"]
              for f in s["fields"]]
    grouped = [f for f in fields if f.get("workflow_group")]
    assert grouped, "no field carries a workflow_group — the outer cards would vanish"
    # EVERY field, not merely every field that happens to have one: an item
    # with no card renders loose below the form and its findings fall to the
    # residual panel.  Fifteen were in that state until 2026-08-15.
    assert len(grouped) == len(fields), (
        f"field(s) with no card: "
        f"{sorted(f['name'] for f in fields if not f.get('workflow_group'))}")
    # The renderer draws a card per name here; a value it does not know
    # renders nothing, so the vocabulary is asserted against the renderer's
    # own list rather than a copy of it.
    assert {f["workflow_group"] for f in grouped} <= set(T.GROUPS)


# RETIRED 2026-09-03 — test_the_renderer_knows_every_card_the_form_actually
# _asks_for.  It regex-extracted WORKFLOW_GROUP_ORDER from form-schema.js and
# compared it to the catalogue's groups as sets, having said in its own
# docstring that "only a browser would show it" (`process/testing.md` § 3a.1).
# Replaced by test_build_e2e.py::TestFormSchemasRender::
# test_no_field_renders_loose_outside_a_card, which asks the rendered DOM
# whether any field group sits outside a card -- the outcome a person sees,
# and one that also catches the two ways the set comparison passed while the
# page was wrong (a role in ORDER but not in META, and a card skipped for an
# empty section map).  Mutation-verified.
def test_an_optional_item_offers_its_unset_state():
    """§ 1.2: `optional` is written to the catalogue precisely so a surface can
    offer *(auto)* / *(no cap)*.  It cannot be inferred from `null_label`."""
    f = {x["name"]: x for s in catalogue_to_form_schema("siesta")["sections"]
         for x in s["fields"]}
    # ``block_size``, not ``mpi_np``: the rank count moved to the
    # staging surface on 2026-08-15 (it is a bench axis prep measures, not a
    # parameter typed here), so it is no longer on this form to check.  The
    # block size is the same shape of claim -- optional, auto-resolved, and
    # its *(auto)* state is the whole reason `optional` is written down.
    assert f["block_size"]["optional"] is True
    assert f["block_size"]["null_option"] is True
    assert f["block_size"]["null_label"]
    assert f["mesh_cutoff"]["optional"] is False


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_a_tri_select_carries_the_three_states_it_walks(engine):
    """An ``Optional[bool]`` renders as auto / true / false, and the renderer
    walks ``f.choices`` to build them.

    **They are the CONTROL's vocabulary, not the item's** — § 5's `choices` is
    an enum's members, and a bool has none — so the catalogue does not carry
    them and the schema must supply them.  It did not, and the form died on
    arrival with ``TypeError: f.choices is not iterable``.  Nine passing tests
    missed it because none of them RENDERED the schema; found by drawing the
    form through the real form-schema.js (2026-08-14).
    """
    for s in catalogue_to_form_schema(engine)["sections"]:
        for f in s["fields"]:
            if f["kind"] == "tri-select":
                assert f.get("choices") == ["auto", "true", "false"], f["name"]


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_every_select_has_something_to_select_from(engine):
    """The general form of the same defect: any control the renderer builds by
    walking ``choices`` must have them, or it throws and the whole form is
    blank rather than one field being wrong."""
    for s in catalogue_to_form_schema(engine)["sections"]:
        for f in s["fields"]:
            if f["kind"] in ("select", "tri-select"):
                assert f.get("choices"), f"{f['name']}: {f['kind']} with no choices"
