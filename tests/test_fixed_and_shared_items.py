"""What the rung fixes and what the calculation shares, through the road:
describe -> ``jobset prep`` -> the deck, or prep's refusal.

PINS: ``docs/engines/template.md`` § 6.4 (`role` -- nobody is asked: the rung
fixes the value, and no template value, stage override, pin or sweep axis
may set it) and § 5 (`shared` -- the calculation's identity binds every
stage, on every SIESTA kind); ``docs/plans/plan.md`` § 5w K1.

PREVENTS, each read in the code before 2026-09-29:

* a vibration told not to write its forces or coordinates: SIESTA prints
  neither unless asked (``LongOutput``), so the finish, which reads the
  reference step's, failed after the whole force-constant run (SS-C1);
* one rung of a ladder with its own species order or label: the ``.XV`` a
  rung reads numbers each atom by the species table of the rung that wrote
  it, and the warm files are found by the label (SO-C2).

Nothing here launches an engine: prep writes the deck and stops.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.pyscf.stages import vibration_stages
from molbuilder.siesta.stages import default_siesta_stages

from test_electronic_state import WATER
from test_engine_offset_reaches_every_deck import _prep


def _ladder(kind):
    return (default_siesta_stages("publishable") if kind == "optimization"
            else vibration_stages("siesta", already_relaxed=False))


def _siesta(root, *, kind="optimization", before_prep=None, refused=False):
    return _prep(root, WATER(), SiestaConfig(system_label="JOB"),
                 _ladder(kind), "siesta", calculation=kind,
                 before_prep=before_prep, refused=refused)


def _template_says(dest, name, value):
    """The calculation's template with ``name`` answered ``value`` -- a hand
    edit, or a template written before the item became fixed -- through the
    product's own reader and writer."""
    from molbuilder.task import read_task
    from molbuilder.template import _emit, find_template, read_template
    tmpl = find_template(dest, read_task(dest / "task.json").label)
    parsed = read_template(tmpl.read_text())
    tmpl.write_text(_emit(
        [dataclasses.replace(i, value=value) if i.name == name else i
         for i in parsed.items], engines=parsed.engines))


def _first_rung_overrides(dest, name, value):
    """The first rung's own value for ``name``, promoted (`varies`) as a
    stage override must be (`engines/stages.md` § 6.2), so the refusal met
    is the item's own."""
    task = json.loads((dest / "task.json").read_text())
    task["varies"] = sorted(set(task.get("varies") or []) | {name})
    task["stages"][0].setdefault("overrides", {})[name] = value
    (dest / "task.json").write_text(json.dumps(task))


def _setting(deck: str, keyword: str):
    """``(value, the note lines above it)`` for the one line setting
    ``keyword``."""
    lines = deck.splitlines()
    at = [k for k, ln in enumerate(lines)
          if ln.split()[:1] == [keyword]]
    assert len(at) == 1, (keyword, [lines[k] for k in at])
    return " ".join(lines[at[0]].split()[1:]), lines[max(0, at[0] - 3):at[0]]


# ------------------------------------------ what the rung fixes (§ 6.4)

@pytest.mark.parametrize("stated, refused", [(True, False), (False, True)])
def test_a_template_states_the_fixed_answer_or_is_refused(
        isolated_projects_root, stated, refused):
    """Every SIESTA template written before 2026-09-29 carries
    ``write_forces = true`` -- the answer, read as it.  ``false`` is a hand
    edit the rung would lay ``true`` over without a word, so prep refuses
    it by name.

    MUTATIONS THIS MUST FAIL AGAINST: the template door left open (the
    ``false`` preps); every stated value refused (the old templates stop
    prepping); the walk skipping a fixed item (the deck's line, or its
    note, goes).  (The answer not being laid on is the transport device
    test's to catch: here the class default is already the answer.)"""
    _d, _s, said = _siesta(
        isolated_projects_root,
        before_prep=lambda dest: _template_says(dest, "write_forces", stated),
        refused=refused)
    if refused:
        assert "'write_forces'" in said and "the rung fixes" in said, said
        return
    for keyword in ("WriteForces", "WriteCoorStep"):
        value, note = _setting(said, keyword)
        assert value == ".true.", (keyword, value)
        assert any("Fixed by this rung" in ln for ln in note), (keyword, note)


@pytest.mark.parametrize("kind", ["optimization", "vibration"])
def test_no_rung_turns_the_forces_or_coordinates_off(
        isolated_projects_root, kind):
    """SS-C1: on EVERY SIESTA kind -- a vibration's force-constant finish
    reads the reference step's forces and coordinates, an optimization's
    record its forces.  A rung's own ``false`` is refused, naming why.

    MUTATION THIS MUST FAIL AGAINST: the kind missing from the item's
    `role` list (the override is honoured and the deck says ``.false.``)."""
    _d, _s, said = _siesta(
        isolated_projects_root, kind=kind,
        before_prep=lambda dest: _first_rung_overrides(
            dest, "write_coor_step", False),
        refused=True)
    assert "'write_coor_step'" in said and "the rung fixes" in said, said
    assert "reads them back" in said, said


# ----------------------------------- what the calculation shares (§ 5)

@pytest.mark.parametrize("name, value", [
    ("species_order", ["H", "O"]),
    ("system_label", "OTHER"),
    ("psml_lib", "pseudopotential"),
])
def test_no_rung_renames_or_renumbers_the_calculation(
        isolated_projects_root, name, value):
    """SO-C2: one name, one species table, one pseudopotential set per
    calculation -- each rung reads what the one before it wrote under that
    name, numbered by that table.  A rung's own value is refused, saying so.

    MUTATION THIS MUST FAIL AGAINST: the kind missing from the item's
    `shared` list (the override is honoured)."""
    _d, _s, said = _siesta(
        isolated_projects_root,
        before_prep=lambda dest: _first_rung_overrides(dest, name, value),
        refused=True)
    assert f"'{name}'" in said and "shared by every stage" in said, said
    assert "one species table" in said, said
