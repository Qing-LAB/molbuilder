"""A recommended range and a hard limit -- one severity each, on every surface
-- through the road: the Task-setup save, `molbuilder describe`, `jobset prep`.

PINS: ``docs/engines/template.md`` § 5.3 (`range` warned and never refused;
`above` refused everywhere with one message, through ``template.why_not``;
the bias list held to its item, in ``tests/test_transport_decks_e2e.py``, a
transport calculation needing a real relaxation); ``docs/engines/stages.md`` § 6.6's rows;
``docs/workflow.md`` § 9's gate ③ at every describe; ``docs/plans/plan.md``
§ 5w K3 (SS-C5, PS-C22, SO-N4).

PREVENTS, each read in the code before 2026-09-30:

* ``fc_displacement = 0`` (SIESTA divides by it) and a thermochemistry
  temperature or pressure of 0 (PySCF takes logarithms of kT/P) reaching a
  deck with at most a warning (SS-C5, PS-C22);
* one range, two severities: a stage's value outside it refused where the
  description is saved while the same value in the template was warned
  (SO-N4).

Nothing here launches an engine: prep writes the deck and stops.
"""
from __future__ import annotations

import pytest

from support.road import (_description_with_overrides, describe_calculation,
                          read_request, save_both, stage_file)
from test_electronic_state import WATER

pytestmark = pytest.mark.usefixtures("in_a_folder_of_its_own")


@pytest.mark.parametrize("engine, name, reason", [
    ("siesta", "fc_displacement",
     "SIESTA divides each force difference by it (ofc.f90)"),
    ("pyscf", "temperature_K",
     "PySCF's ideal-gas thermochemistry divides by kT"),
    ("pyscf", "pressure_atm",
     "PySCF's ideal-gas entropy takes a logarithm of kT/P"),
])
def test_a_value_the_engine_cannot_take_is_refused_with_its_reason(
        prepared, engine, name, reason):
    """A vibration's template stating 0 for a value the engine divides by or
    takes a logarithm of is refused at prep, naming where it came from and
    why -- the item's own declared limit, the one clause every door gives.

    MUTATION THIS MUST FAIL AGAINST: the item's `above` removed (the deck is
    written, and the run writes inf or divides by zero)."""
    _d, _s, said = prepared(
        WATER(), engine, calculation="vibration",
        template={name: 0.0, **({"already_relaxed": True}
                                if engine == "pyscf" else {})},
        refused=True)
    assert f"the template sets {name} = 0.0: it must be greater than 0" \
        in said, said
    assert reason in said, said


def test_a_value_outside_its_recommended_range_is_warned_where_it_is_saved(
        web_client, tmp_path, monkeypatch):
    """A stage's value outside the item's recommended range is a choice a
    person may make -- a coarse test run -- so the Task-setup save WARNS and
    saves, as the settings gate warns about the same value in a template; a
    triple per component, as the gate does.

    MUTATIONS THIS MUST FAIL AGAINST: the stage branch back at "error" (the
    save refuses what the template would only warn about); the save checking
    scalars alone (the triple says nothing, the K3 review)."""
    d = describe_calculation(tmp_path, monkeypatch)
    r = save_both(d, _description_with_overrides(
        d, {"coarse": {"mesh_cutoff": 2000.0, "kgrid": [100, 1, 1]}}),
        read_request(d), web_client)
    assert r.status_code == 200, r.get_json()
    for where, said in (("config.mesh_cutoff", "mesh_cutoff = 2000.0"),
                        ("config.kgrid", "kgrid[0] = 100")):
        found = [f for f in r.get_json()["findings"] if f["where"] == where]
        assert [f["severity"] for f in found] == ["warn"], found
        assert f"{said}, outside the recommended range" in \
            found[0]["message"], found
        assert "a recommendation, not a limit" in found[0]["message"], found


def test_what_the_save_refuses_it_refuses_once(web_client, tmp_path,
                                               monkeypatch):
    """The description's own check at the Task-setup save, where a person's
    hand-edited description lands:

    * a run card's value and a bench's point -- the job request's -- become
      run settings and sweep points, so the one per-value door is asked
      about them as about a stage's value -- ``block_size = 0`` is refused
      here, not first by `prep bench` on the cluster;
    * one value draws one refusal (`engines/template.md` § 5.3): a quoted
      component is the type check's alone, never the limit's beside it; and
      a WHOLE float is the count it names, as `resolve` will see it
      (``template.as_declared``), so ``0.0`` is refused as the count 0.

    MUTATIONS THIS MUST FAIL AGAINST: the check not asking about execution
    values and bench points (both save); `template.why_not` without its type
    guard (the quoted component crashes the save: a text compared with a
    number); the door asked about the raw value (``0.0`` for a count
    saves); a run card judged without the setting door (it saves -- the
    2.27.0 review)."""
    d = describe_calculation(tmp_path, monkeypatch)
    was = (d / "task.json").read_text()

    def save(**change):
        request = read_request(d)
        overrides = change.pop("overrides", {})
        request.update(change)
        r = save_both(d, _description_with_overrides(d, {"coarse": overrides}),
                      request, web_client)
        assert r.status_code == 400, r.get_json()
        return r.get_json()

    for change, said in (
            ({"execution": {"block_size": 0}},
             "execution sets block_size = 0: it must be greater than 0"),
            ({"bench": {"block_size": [0, 32]}},
             "bench declares block_size = 0: it must be greater than 0"),
            ({"overrides": {"kgrid": [4, 4, "0"]}},
             "which is not three whole numbers"),
            ({"overrides": {"kgrid": [0.0, 4, 4]}},
             "each component must be greater than 0"),
            # A RUN CARD'S VALUE IS ITS ITEM'S, judged by the setting door
            # (`engines/template.md` § 5.4) -- never first at prep.
            ({"execution": {"mpi_np": "8"}},
             "execution sets mpi_np = '8', which is not a whole number")):
        body = save(**change)
        assert said in body["error"], body["error"]
        errors = [f for f in body["findings"] if f["severity"] == "error"]
        assert len(errors) == 1, [f["message"] for f in errors]
    assert (d / "task.json").read_text() == was


def test_a_triples_range_is_warned_at_prep_too(prepared):
    """The settings gate warns a triple per component, as the save does: a
    ``kgrid`` of 100 along an axis is outside the recommended range, and
    the deck's validation report says so.  Its range was warned nowhere
    between the day `kgrid`'s callable lost its bounds and the K3 review.

    MUTATION THIS MUST FAIL AGAINST: the metadata pass standing aside for a
    field with a `validate` callable (the report says nothing)."""
    dest, stage, _deck = prepared(WATER(), "siesta",
                                  overrides={"kgrid": [100, 1, 1]})
    report = stage_file(dest, stage, ".validation.txt").read_text()
    assert "[0] = 100 is outside the recommended range [1, 64]" in report, \
        report


