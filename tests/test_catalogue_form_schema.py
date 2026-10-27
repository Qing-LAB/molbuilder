"""The Build form's schema comes from the CATALOGUE (`web/form-schema.md` § 1).

The JS renderer takes whatever schema it is handed and knows nothing about
any individual parameter.  The `category` panels are SHARED, so SIESTA and
PySCF show the same headings in the same order.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from molbuilder import template as T
from molbuilder.web.blueprints._shared import catalogue_to_form_schema


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_every_item_this_engine_has_reaches_the_form(engine):
    """Membership is TOTAL (§ 7): a parameter the catalogue carries and the
    form omits is a control no user can reach.

    **The exclusions are declared by the ITEM, never listed here**, and there
    are two.

    ``group = "staging"`` (2026-08-15) says *this parameter is answered by the
    staging surface*. The stage token is the only one today — derived from
    which stage is running, so not something a person types into a physics
    form.

    ``role`` (2026-09-16, `template.md` § 6.4) says *the stage's own role
    answers it*: **"a `role` item is not offered as a form field, is not
    offered as a stage-table column, and carries no value in a template of
    that kind"**. Offering one presents a choice with exactly one correct
    answer — a transport device solves with NEGF and a bulk lead does not —
    and a person who changed it would not be tuning the run, they would be
    stopping it being the run it is.

    **AND `role` IS PER KIND, which is why it is a membership test and not a
    boolean.** `solution_method` is the rung's business for *transport* and an
    ordinary person-answered choice for an *optimization*, where nothing else
    decides it. Excluding every item that has a role at all would take a
    legitimate control off the Build form.
    """
    items = [i for i in T.select(T.read_template(T.load_catalogue()),
                                 engine=engine) if i.group != "staging"]

    # Membership is total PER KIND (template.md § 6.3): an item carrying `calculations` reaches the
    # form of exactly those kinds; an item without the key reaches every
    # kind.  The kinds to check come from the items themselves, so a new
    # kind needs no edit here.
    kinds = {"optimization"} | {k for i in items for k in i.calculations}
    for kind in sorted(kinds):
        expect = {i.name for i in items
                  if (not i.calculations or kind in i.calculations)
                  and kind not in (getattr(i, "role", None) or ())}
        fields = [f for sec in catalogue_to_form_schema(
                      engine, calculation=kind)["sections"]
                  for f in sec["fields"]]
        got = {f["name"] for f in fields if "locked" not in f}
        assert got == expect, (engine, kind)
        # ...and a role item the rungs answer alike is SHOWN, read-only at
        # that answer -- never a control (§ 6.6 obligation 3; K7).
        echoed = {f["name"]: f["locked"]["value"] for f in fields
                  if "locked" in f}
        assert echoed == T.fixed_on_every_rung(engine, kind), (engine, kind)

    # And the no-argument call IS the optimization form -- the default
    # spelled out, so the two spellings cannot drift.
    def names(**kw):
        return {f["name"] for sec in catalogue_to_form_schema(
                    engine, **kw)["sections"] for f in sec["fields"]}
    assert names() == names(calculation="optimization")


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_the_vibration_form_in_its_parts_holds_each_setting_once(engine):
    """`web/form-schema.md` § 1.3, `engines/template.md` § 6.2a: on the
    Spectrum tab the settings every engine shares are drawn once, before the
    engine -- their electronic state in the chemistry card, the rest under
    *The calculation* -- and each engine's form holds its own: together
    exactly the engine's whole vibration form, no setting in two parts and
    none in none, each shared one offered every choice the engine takes,
    the electronic state's part holding the charge and spin alone.

    THE SILENT FAILURE: a setting the person can no longer reach (in no
    part), or reached twice with two values the Send must pick between (in
    two).  MUTATIONS THIS MUST FAIL AGAINST: the own part left whole (the
    shared settings drawn twice); the shared part filtered by one engine (a
    setting the other engine has, in none); the charge and spin left in
    the common part as well."""
    from molbuilder.template import STATE_ITEMS

    def _fields(schema):
        return {f["name"]: f for s in schema["sections"] for f in s["fields"]}
    whole = _fields(catalogue_to_form_schema(engine, "p",
                                             calculation="vibration"))
    parts = {
        surface: _fields(catalogue_to_form_schema(
            None if surface != "own" else engine, "p",
            calculation="vibration", surface=surface))
        for surface in ("state", "common", "own")}
    names = [set(p) for p in parts.values()]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not a & b, a & b
    assert set().union(*names) == set(whole), set(whole) ^ set().union(*names)
    assert parts["state"] and set(parts["state"]) <= set(STATE_ITEMS), \
        parts["state"]
    for name, f in {**parts["state"], **parts["common"]}.items():
        assert set(whole[name].get("choices") or ()) <= set(
            f.get("choices") or ()), name


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


def test_an_optional_item_offers_its_unset_state():
    """§ 1.2: `optional` is written to the catalogue precisely so a surface can
    offer *(auto)* / *(no cap)*.  It cannot be inferred from `null_label`."""
    f = {x["name"]: x for s in catalogue_to_form_schema("siesta")["sections"]
         for x in s["fields"]}
    # ``net_charge``: optional, worked out from the structure when unset,
    # and its *(auto)* state is the whole reason `optional` is written down.
    assert f["net_charge"]["optional"] is True
    assert f["net_charge"]["null_option"] is True
    assert f["net_charge"]["null_label"]
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
