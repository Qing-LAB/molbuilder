"""A recommended range and a hard limit -- one severity each, on every surface
-- through the road: the Task-setup save, the Transport tab's Send and
`jobset init`, `jobset prep`.

PINS: ``docs/engines/template.md`` § 5.3 (`range` warned and never refused;
`above` refused everywhere with one message, through ``template.why_not``;
the bias list held to its item); ``docs/engines/stages.md`` § 6.6's rows;
``docs/workflow.md`` § 9's gate ③ at every describe; ``docs/plans/plan.md``
§ 5w K3 (SS-C5, PS-C22, T-F15, SO-N4).

PREVENTS, each read in the code before 2026-09-30:

* ``fc_displacement = 0`` (SIESTA divides by it) and a thermochemistry
  temperature or pressure of 0 (PySCF takes logarithms of kT/P) reaching a
  deck with at most a warning (SS-C5, PS-C22);
* one range, two severities: a stage's value outside it refused where the
  description is saved while the same value in the template was warned
  (SO-N4);
* a bias list with a repeated point -- two runs of one voltage in one
  folder -- or a point outside the item's range, said nowhere (T-F15); and
  the Transport tab's Send and `jobset init`'s transport arm running only
  the codec, never the description's own check.

Nothing here launches an engine: prep writes the deck and stops.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from molbuilder.config.pyscf import PySCFConfig

from test_fixed_and_shared_items_e2e import _template_says
from test_transport_prep import _CITE, _isolated, _junction_struct, \
    _write_junction  # noqa: F401
from test_what_a_kind_offers_e2e import _pyscf_vibration, _siesta


@pytest.mark.parametrize("engine, name, reason", [
    ("siesta", "fc_displacement",
     "SIESTA divides each force difference by it (ofc.f90)"),
    ("pyscf", "temperature_K",
     "PySCF's ideal-gas thermochemistry divides by kT"),
    ("pyscf", "pressure_atm",
     "PySCF's ideal-gas entropy takes a logarithm of kT/P"),
])
def test_a_value_the_engine_cannot_take_is_refused_with_its_reason(
        isolated_projects_root, engine, name, reason):
    """A vibration's template stating 0 for a value the engine divides by or
    takes a logarithm of is refused at prep, naming where it came from and
    why -- the item's own declared limit, the one clause every door gives.

    MUTATION THIS MUST FAIL AGAINST: the item's `above` removed (the deck is
    written, and the run writes inf or divides by zero)."""
    put = lambda dest: _template_says(dest, name, 0.0)      # noqa: E731
    if engine == "siesta":
        _d, _s, said = _siesta(isolated_projects_root, "vibration",
                               before_prep=put, refused=True)
    else:
        from test_engine_offset_reaches_every_deck import _prep
        from molbuilder.pyscf.stages import vibration_stages
        from test_electronic_state import WATER
        _d, _s, said = _prep(
            isolated_projects_root, WATER(),
            PySCFConfig(job_name="JOB", already_relaxed=True),
            vibration_stages("pyscf", already_relaxed=True), "pyscf",
            calculation="vibration", before_prep=put, refused=True)
    assert f"the template sets {name} = 0.0: it must be greater than 0" \
        in said, said
    assert reason in said, said


def test_a_value_outside_its_recommended_range_is_warned_where_it_is_saved(
        web_client, isolated_projects_root):
    """A stage's value outside the item's recommended range is a choice a
    person may make -- a coarse test run -- so the Task-setup save WARNS and
    saves, as the settings gate warns about the same value in a template; a
    triple per component, as the gate does.

    MUTATIONS THIS MUST FAIL AGAINST: the stage branch back at "error" (the
    save refuses what the template would only warn about); the save checking
    scalars alone (the triple says nothing, the K3 review)."""
    from molbuilder.identity import run_id
    from test_task_setup_tab import _fresh_calc_dir
    d = _fresh_calc_dir(isolated_projects_root)
    described = {
        "schema": "molbuilder/task@1", "engine": {"name": "siesta"},
        "shape": "hierarchical",
        "run": {"name": "x", "id": run_id("x", "H2"),
                "created": "2026-09-30T00:00:00-07:00"},
        "structure": {"source": "s.xyz", "formula": "H2", "atoms": 2},
        "varies": ["kgrid", "mesh_cutoff"],
        "stages": [{"name": "coarse", "enabled": True,
                    "overrides": {"mesh_cutoff": 2000.0,
                                  "kgrid": [100, 1, 1]}}]}
    r = web_client.post("/api/task-setup/save", json={
        "dest": str(d), "text": json.dumps(described)})
    assert r.status_code == 200, r.get_json()
    assert (d / "task.json").is_file()
    for where, said in (("config.mesh_cutoff", "mesh_cutoff = 2000.0"),
                        ("config.kgrid", "kgrid[0] = 100")):
        found = [f for f in r.get_json()["findings"] if f["where"] == where]
        assert [f["severity"] for f in found] == ["warn"], found
        assert f"{said}, outside the recommended range" in \
            found[0]["message"], found
        assert "a recommendation, not a limit" in found[0]["message"], found


def test_the_bias_list_is_held_to_its_item(web_client,
                                           isolated_projects_root):
    """The bias list is the points `bias_voltage_v` takes (`engines/
    transport.md` § 2a.10): a repeated point is two runs of one voltage in
    one folder and is refused where the description is read -- on the
    Transport tab's Send and at `jobset init` alike -- and a point outside
    the item's recommended range describes, and says so among the Send's
    notices.

    MUTATIONS THIS MUST FAIL AGAINST: the codec's repeat check removed (the
    repeat describes); points compared by value rather than by folder (the
    near pair describes into one folder); the description's own check not
    run at the Transport tab's Send, or its warnings not said by `jobset
    init` (the out-of-range point says nothing)."""
    from molbuilder.jobset._cli import jobset_group
    _write_junction(isolated_projects_root, _junction_struct())

    def send(bias):
        return web_client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T", junction=_CITE, bias=bias))

    repeat = send([0.0, 0.5, 0.5])
    assert repeat.status_code == 400, repeat.get_json()
    assert "the bias list repeats 0.5 V" in repeat.get_json()["error"]
    # ...and two voltages whose folder names are one: v0.1/
    near = send([0.0, 0.1, 0.1000001])
    assert near.status_code == 400, near.get_json()
    assert "0.1 V and 0.1000001 V in one folder (v0.1/)" in \
        near.get_json()["error"], near.get_json()["error"]

    def init(bias, bundle):
        return CliRunner().invoke(jobset_group, [
            "init", "--calculation", "transport", "--shape", "hierarchical",
            "--bundle", bundle, "--slot", f"junction={_CITE}",
            "--bias", bias])
    cli = init("0.0,0.5,0.5", "J/transport/T")
    assert cli.exit_code != 0, cli.output
    assert "the bias list repeats 0.5 V" in cli.output, cli.output

    far = send([0.0, 6.0])
    assert far.status_code == 200, far.get_json()
    notices = [n["message"] for n in far.get_json()["notices"]]
    assert any("the bias list holds 6 V, outside the recommended range "
               "[-5, 5] V" in m for m in notices), notices
    cli = init("0.0,6.0", "J/transport/Tfar")
    assert cli.exit_code == 0, cli.output
    assert ("note: the bias list holds 6 V, outside the recommended range"
            in cli.output), cli.output


def test_what_the_save_refuses_it_refuses_once(web_client,
                                                isolated_projects_root):
    """The description's own check at the Task-setup save, where a person's
    hand-edited description lands:

    * an ``execution`` block's value and a bench's point become pins and
      sweep points, so the one per-value door is asked about them as about a
      stage's value -- ``block_size = 0`` is refused here, not first by
      `prep bench` on the cluster;
    * one value draws one refusal (`engines/template.md` § 5.3): a quoted
      component is the type check's alone, never the limit's beside it; and
      a WHOLE float is the count it names, as `resolve` will see it
      (``template.as_declared``), so ``0.0`` is refused as the count 0.

    MUTATIONS THIS MUST FAIL AGAINST: the check not asking about execution
    values and bench points (both save); `template.why_not` without its type
    guard (the quoted component crashes the save: a text compared with a
    number); the door asked about the raw value (``0.0`` for a count
    saves)."""
    from molbuilder.identity import run_id
    from test_task_setup_tab import _fresh_calc_dir
    d = _fresh_calc_dir(isolated_projects_root)
    base = {
        "schema": "molbuilder/task@1", "engine": {"name": "siesta"},
        "shape": "hierarchical",
        "run": {"name": "x", "id": run_id("x", "H2"),
                "created": "2026-09-30T00:00:00-07:00"},
        "structure": {"source": "s.xyz", "formula": "H2", "atoms": 2},
        "varies": [],
        "stages": [{"name": "coarse", "enabled": True, "overrides": {}}]}

    def save(**change):
        described = json.loads(json.dumps(base))
        for key, value in change.items():
            if key == "overrides":
                described["varies"] = sorted(value)
                described["stages"][0]["overrides"] = value
            else:
                described[key] = value
        r = web_client.post("/api/task-setup/save", json={
            "dest": str(d), "text": json.dumps(described)})
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
             "each component must be greater than 0")):
        body = save(**change)
        assert said in body["error"], body["error"]
        errors = [f for f in body["findings"] if f["severity"] == "error"]
        assert len(errors) == 1, [f["message"] for f in errors]
    assert not (d / "task.json").exists()


def test_a_triples_range_is_warned_at_prep_too(isolated_projects_root):
    """The settings gate warns a triple per component, as the save does: a
    ``kgrid`` of 100 along an axis is outside the recommended range, and
    the deck's validation report says so.  Its range was warned nowhere
    between the day `kgrid`'s callable lost its bounds and the K3 review.

    MUTATION THIS MUST FAIL AGAINST: the metadata pass standing aside for a
    field with a `validate` callable (the report says nothing)."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.stages import default_siesta_stages
    from molbuilder.task import Stage
    from test_electronic_state import WATER
    from test_engine_offset_reaches_every_deck import _prep
    ladder = default_siesta_stages("publishable")
    first = ladder[0]
    ladder = (Stage(name=first.name, enabled=first.enabled,
                    overrides={**dict(first.overrides or {}),
                               "kgrid": (100, 1, 1)}),) + tuple(ladder[1:])
    dest, stage, _deck = _prep(isolated_projects_root, WATER(),
                               SiestaConfig(system_label="JOB"), ladder,
                               "siesta")
    report = next((dest / p).read_text()
                  for p in (x.relative_to(dest)
                            for x in dest.rglob("*.validation.txt")))
    assert "[0] = 100 is outside the recommended range [1, 64]" in report, \
        report
