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


def test_both_engines_show_the_same_panels():
    """The claim that could not be made while `section` was per-engine."""
    a = [s["name"] for s in catalogue_to_form_schema("siesta")["sections"]]
    b = [s["name"] for s in catalogue_to_form_schema("pyscf")["sections"]]
    assert a == b


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_every_item_this_engine_has_reaches_the_form(engine):
    """Membership is TOTAL (§ 7): a parameter the catalogue carries and the
    form omits is a control no user can reach.

    This is what makes the six previously-unreachable SIESTA parameters appear
    — `species_order`, `write_forces`, `write_coor_step`, `write_molwatch_log`,
    `copy_psml`, `kgrid_displacement` — none of which had a `section`.
    """
    items = {i.name for i in T.select(T.read_template(T.load_catalogue()),
                                      engine=engine)}
    fields = {f["name"] for s in catalogue_to_form_schema(engine)["sections"]
              for f in s["fields"]}
    assert fields == items


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
    assert {f["workflow_group"] for f in grouped} <= {"profile", "stage", "budget"}


def test_an_optional_item_offers_its_unset_state():
    """§ 1.2: `optional` is written to the catalogue precisely so a surface can
    offer *(auto)* / *(no cap)*.  It cannot be inferred from `null_label`."""
    f = {x["name"]: x for s in catalogue_to_form_schema("siesta")["sections"]
         for x in s["fields"]}
    assert f["mpi_np"]["optional"] is True
    assert f["mpi_np"]["null_option"] is True
    assert f["mpi_np"]["null_label"]
    assert f["mesh_cutoff"]["optional"] is False
