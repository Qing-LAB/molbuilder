"""Where an item binds -- each rung reads what its ROLE reads, decided by
the catalogue alone -- through the road: the Task-setup save, the Transport
tab's Send, `jobset prep`, and the stage table in the browser.

PINS: ``docs/engines/template.md`` § 6.4 (``stages`` names rung roles; one
door refuses an override a rung does not read, or of an item the kind does
not carry) and § 5.3 (a rung that relaxes takes a step);
``docs/engines/stages.md`` § 4 R3 (the ladder check reads ``tightens`` on
every engine and compares rungs of one role); ``docs/engines/vibration.md``
§ 5.2a and § 5.5 (each stage's settings are its role's; the finish judges by
the relaxation rung's own criterion); ``docs/web/task-setup.md`` § 5 and § 9;
``docs/plans/plan.md`` § 5w K4.

PREVENTS, each read in the code before 2026-09-30 (the M11 review):

* a vibration's force-constant rung offered, and carrying, the relaxation's
  settings, its copy of ``relax_force_tol`` becoming the finish's criterion
  -- a preset that relaxed at 0.05 eV/A judged at 0.01 (SS-C6);
* an override of an item the kind does not carry, inert and said nowhere
  (the K3 review);
* the ladder check knowing SIESTA's four fields alone (PO-C14), and reading
  transport's per-rung tolerance as a loosening (T-F14);
* a relaxation rung of 0 steps, which relaxes nothing (SS-C5).

Nothing here launches an engine.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from support.junction import _isolated  # noqa: F401 -- its sandbox, autouse


def _described(engine, stages, calculation="optimization"):
    """A description as a person's editor leaves it: ``stages`` the rows,
    ``varies`` what they override."""
    from molbuilder.identity import run_id
    out = {"schema": "molbuilder/task@1", "engine": {"name": engine},
           "calculation": "optimization",
           "shape": "hierarchical",
           "run": {"name": "x", "id": run_id("x", "H2"),
                   "created": "2026-09-30T00:00:00-07:00"},
           "structure": {"source": "s.xyz", "formula": "H2", "atoms": 2},
           "varies": sorted({k for s in stages for k in s["overrides"]}),
           # EACH ROW ITS OWN GROUP, as the page numbers a stage it adds.
           "stages": [dict(s, group=s.get("group", i + 1))
                      for i, s in enumerate(stages)]}
    if calculation != "optimization":
        out["calculation"] = calculation
    return out


def _save(web_client, dest, described):
    return web_client.post("/api/task-setup/save", json={
        "dest": str(dest), "text": json.dumps(described)})


#: A transport calculation's five rungs, in their default groups, and the
#: relaxation it cites.
_RUNGS = ("seed", "electrode_L", "electrode_R", "device", "transmission")
_GROUPS = (1, 1, 1, 2, 3)
_CITE = "J/optimization/relax/01_coarse/run-0"


def _transport(*, execution=None, on=None, rung_execution=None):
    """A transport description as Task setup's Save takes it: ``execution``
    the calculation's run card, ``rung_execution`` the card of rung ``on``."""
    from molbuilder.identity import run_id
    out = {"schema": "molbuilder/task@1", "engine": {"name": "siesta"},
           "calculation": "transport", "shape": "hierarchical",
           "run": {"name": "t", "id": run_id("t", _CITE, stage_names=_RUNGS),
                   "created": "2026-10-12T00:00:00-07:00"},
           "slots": {"junction": _CITE},
           "bias": {"voltages_v": [0.0, 0.2],
                    "low_bias_approximation": False},
           "varies": [],
           "stages": [dict({"name": n, "group": g, "overrides": {}},
                           **({"execution": rung_execution} if n == on
                              else {}))
                      for n, g in zip(_RUNGS, _GROUPS)]}
    if execution is not None:
        out["execution"] = execution
    return out


def _template_beside(dest, engine, calculation, cfg):
    """The template the hand-over leaves beside the description, which the
    save reads for the checks that compare RESOLVED rungs."""
    from molbuilder.template import template_with_values
    from molbuilder.template_file import template_path
    template_path(dest, "x").write_text(template_with_values(
        cfg, engine=engine, calculation=calculation), encoding="utf-8")


def test_an_override_its_rung_does_not_read_is_refused_where_it_is_saved(
        web_client, isolated_projects_root):
    """A vibration's relaxation rung reads the relaxation's settings and its
    force-constant rung the displacement, whatever the stages are called:
    each on the other rung is refused by name, naming the rung that reads
    it -- as an item the kind does not carry is, on any rung, and a name no
    field has, on a ladder whose groups are checked -- and nothing is saved.

    MUTATIONS THIS MUST FAIL AGAINST: the description's check not asking
    which rung reads an override (both save); the group check asked of a
    ladder that does not resolve (the save crashes, 500)."""
    from test_task_setup_tab import _fresh_calc_dir
    d = _fresh_calc_dir(isolated_projects_root)
    r = _save(web_client, d, _described("siesta", [
        {"name": "relax", "overrides": {"fc_displacement": 0.02}},
        {"name": "freq", "overrides": {"relax_force_tol": 0.05}}],
        calculation="vibration"))
    assert r.status_code == 400, r.get_json()
    refused = {f["stage"]: f["message"] for f in r.get_json()["findings"]
               if f["severity"] == "error"}
    assert ("only the force_constants rung of a vibration reads, not "
            "'relax' (its relaxation rung)") in refused["relax"], refused
    assert ("only the relaxation rung of a vibration reads, not 'freq' "
            "(its force_constants rung)") in refused["freq"], refused

    r = _save(web_client, d, _described("siesta", [
        {"name": "coarse", "overrides": {"fc_displacement": 0.02}}]))
    assert r.status_code == 400, r.get_json()
    assert "which an optimization does not carry" in r.get_json()["error"]
    assert not (d / "task.json").exists()

    # A NAME NO FIELD HAS, on a ladder whose groups the save checks against
    # the template beside it: the stages' links are read of a ladder that
    # resolves, so the answer is this refusal, never a crash.
    from molbuilder.config.siesta import SiestaConfig
    _template_beside(d, "siesta", "optimization",
                     SiestaConfig(system_label="x"))
    r = _save(web_client, d, _described("siesta", [
        {"name": "coarse", "overrides": {"no_such_item": 1}},
        {"name": "medium", "overrides": {}}]))
    assert r.status_code == 400, r.get_json()
    assert "no_such_item" in r.get_json()["error"], r.get_json()["error"]
    assert not (d / "task.json").exists()


def test_a_run_card_value_its_rung_does_not_read_is_refused_where_it_is_saved(
        web_client, isolated_projects_root):
    """`web/task-setup.md` § 6.2b: a run-card value is read by the rungs the
    item's ``stages`` names, as an override is -- a GPU count on a transport
    ladder's transmission, which runs on no GPU, is refused by name, saying
    to remove it, and nothing is saved; on the device, which reads it, it
    is not.  Without it the count reached prep's G5 refusal, which named a
    GPU switch the transmission's card does not have (measured 2026-10-12).

    MUTATIONS THIS MUST FAIL AGAINST: the description's check not asking
    which rung reads a run-card value (it saves); `gpu_count` read by every
    rung (the catalogue's `stages` dropped -- it saves)."""
    d = isolated_projects_root / "J" / "transport" / "t"
    d.mkdir(parents=True)
    r = _save(web_client, d, _transport(on="transmission",
                                        rung_execution={"gpu_count": 1}))
    assert r.status_code == 400, r.get_json()
    assert ("stage 'transmission' states gpu_count = 1 on its run card, "
            "which only the seed / electrode_L / electrode_R / device / "
            "relaxation / force_constants rungs of a transport calculation "
            "read, not 'transmission'") in r.get_json()["error"], \
        r.get_json()["error"]
    assert ("not 'transmission' -- remove it from that run card -- on Task "
            "setup, the × beside it") in r.get_json()["error"], \
        r.get_json()["error"]
    assert not (d / "task.json").exists()

    r = _save(web_client, d, _transport(on="device", rung_execution={
        "gpu_count": 1, "use_gpu": True}))
    assert r.status_code == 200, r.get_json()


def test_a_calculation_wide_value_is_taken_by_the_rungs_that_read_it(
        web_client, isolated_projects_root):
    """`engines/stages.md` § 6.8d, `execution/gpu.md` § 1.1: a calculation-
    wide value reaches the rungs the item's ``stages`` names.  A transport
    calculation run on the GPU is saved, and each SCF rung's prep takes the
    GPU and its count, while the transmission's -- `tbtrans` runs on no GPU
    -- takes neither, its other values kept.  Read through what a rung's
    prep takes from the description (`prep_inputs.run_inputs`: the launch
    shape and the deck's pins), not prep's files: a transport rung is
    prepared on a finished relaxation, which only the engine makes, and the
    real junction's e2e runs on the CPU.  Without it the transmission was a
    GPU job, holding a device it never uses (measured 2026-10-17).

    MUTATION THIS MUST FAIL AGAINST: the calculation's card handed whole to
    every rung (`Task.run_condition`)."""
    from molbuilder.jobset.prep_inputs import run_inputs
    from molbuilder.task import FILENAME, of as task_of
    d = isolated_projects_root / "J" / "transport" / "t"
    d.mkdir(parents=True)
    r = _save(web_client, d, _transport(execution={
        "use_gpu": True, "gpu_count": 1, "mpi_np": 4, "omp_threads": 1}))
    assert r.status_code == 200, r.get_json()
    task = task_of(d)
    for rung in _RUNGS[:-1]:
        shape, pins = run_inputs(d, task, rung)
        assert (shape.get("gres"), pins.get("use_gpu")) == ("gpu:1", True), \
            (rung, shape, pins)
    shape, pins = run_inputs(d, task, "transmission")
    assert "gres" not in shape and "use_gpu" not in pins, (shape, pins)
    assert (shape.get("mpi_np"), shape.get("cpus_per_task")) == (4, 1), shape


def test_the_templates_gpu_is_taken_by_the_rungs_that_read_it(
        web_client, isolated_projects_root):
    """`engines/stages.md` § 6.8d, `execution/gpu.md` § 1.1: the GPU said in
    the calculation's own template, its count on the calculation's card,
    reaches the rungs that read it -- each SCF rung resolves to a run on one
    GPU, and the transmission to a run on none, so prep's G5 has nothing to
    refuse there.  Read through prep's two steps for a rung
    (`prep_inputs.run_inputs`, then `resolve`), not prep's files, for the
    reason the test above gives.  Without it the transmission resolved to a
    GPU run with no count, refused with no way out: the calculation's card
    does not reach it, and a count on its own card is refused (measured
    2026-10-17).

    MUTATION THIS MUST FAIL AGAINST: `resolve` carrying the template's
    `use_gpu` onto every rung."""
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset.model import Resources, gpu_request
    from molbuilder.jobset.prep_inputs import run_inputs
    from molbuilder.resolve import resolve
    from molbuilder.task import FILENAME, of as task_of
    from molbuilder.template import template_with_values
    from molbuilder.template_file import template_path
    d = isolated_projects_root / "J" / "transport" / "t"
    d.mkdir(parents=True)
    text = template_with_values(
        dataclasses.replace(SiestaConfig(), use_gpu=True), engine="siesta",
        calculation="transport")
    template_path(d, "t").write_text(text, encoding="utf-8")
    r = _save(web_client, d, _transport(execution={
        "gpu_count": 1, "mpi_np": 4, "omp_threads": 1}))
    assert r.status_code == 200, r.get_json()
    task = task_of(d)
    asked = {}
    for rung in _RUNGS:
        shape, pins = run_inputs(d, task, rung)
        ps = resolve(text, task, SiestaConfig, allocation=Resources(**shape),
                     stage=rung, pins=pins)
        g = gpu_request(ps.elements[0].resources)
        asked[rung] = (g.uses, g.count)
    assert asked == {**{rung: (True, 1) for rung in _RUNGS[:-1]},
                     "transmission": (False, None)}, asked


def test_a_relaxation_that_takes_no_step_is_refused(web_client,
                                                    isolated_projects_root):
    """``relax_steps = 0`` on a rung that relaxes is a single point -- the
    rung relaxes nothing, and a vibration's force constants would be taken
    at a geometry nobody relaxed -- so the description is refused where it
    is saved.  (The same value is `prep bench`'s own measurement pin, which
    no description carries; its tests run the bench.)  One value, one
    refusal: the value's range warning stands aside.

    MUTATIONS THIS MUST FAIL AGAINST: the step rule not asked (it saves);
    the range check not told of the refusal (a warning beside it)."""
    from molbuilder.config.siesta import SiestaConfig
    from test_task_setup_tab import _fresh_calc_dir
    d = _fresh_calc_dir(isolated_projects_root)
    _template_beside(d, "siesta", "vibration", SiestaConfig(system_label="x"))
    r = _save(web_client, d, _described("siesta", [
        {"name": "relax", "overrides": {"relax_steps": 0}},
        {"name": "freq", "overrides": {}}], calculation="vibration"))
    assert r.status_code == 400, r.get_json()
    assert "stage 'relax' relaxes" in r.get_json()["error"], \
        r.get_json()["error"]
    assert "relaxes nothing" in r.get_json()["error"]
    found = [f["severity"] for f in r.get_json()["findings"]
             if f["where"] == "config.relax_steps"]
    assert found == ["error"], r.get_json()["findings"]


# --------------------------------------------------------------------- #
#  The stage table, in the browser                                      #
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


@pytest.mark.e2e
def test_the_stage_table_offers_each_rung_what_it_reads(
        page, flask_server, isolated_projects_root):
    """A vibration ladder in Task setup: the `relax` row's displacement cell
    and the `freq` row's relaxation-tolerance cell are drawn disabled --
    each rung's deck does not read them -- and a relaxation preset applied
    to `freq` fills nothing there, while on `relax` it fills its tier.

    A person renames the `relax` row `Relax`: in any case it is one name
    (`stages.md` § 2), so the row is still the relaxation (plan § 5w K12).

    MUTATIONS THIS MUST FAIL AGAINST: the cell asked by the stage's name
    rather than its role (the `relax` row's own tolerance is disabled); the
    columns sent without the kind's role rule (no cell is); a preset filling
    every row (`freq` gains the relaxation's columns); the role matched by
    the exact name (the renamed row reads as a force-constant rung)."""
    pytest.importorskip("playwright.sync_api")
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.structure import Structure
    from molbuilder.task import Stage
    from molbuilder.workingcopy_structure import StructureCodec
    from test_task_setup_one_folder_e2e import _open

    root = isolated_projects_root / "proj"
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[5.0, 5.0, 5.0],
                                           [5.0, 5.0, 5.741]]),
                       regions={"frozen_atoms": [0]},
                       cell=np.diag([10.0, 10.0, 10.0]),
                       axis_kind=("isolated",) * 3)
    (root / "structure").mkdir(parents=True)
    StructureCodec().write(struct, root / "structure" / "h2.xyz")
    calc = root / "vibration" / "h2-vib"
    D.write_description(D.build_description(
        struct, SiestaConfig(system_label="h2"),
        (Stage(name="relax", overrides={"relax_force_tol": 0.05}),
         Stage(name="freq", overrides={"fc_displacement": 0.02})),
        engine="siesta", shape="hierarchical", name="h2",
        calculation="vibration", source=str(root / "structure" / "h2.xyz")),
        calc, struct=struct)

    _open(page, flask_server, calc)
    rows = "#ts-stage-table tbody tr"
    page.wait_for_selector(rows, timeout=20000)
    page.wait_for_function(
        "() => document.querySelectorAll("
        "  '#ts-stage-table tbody td[data-foreign]').length > 0",
        timeout=20000)

    def foreign():
        return page.evaluate(
            "(sel) => [...document.querySelectorAll(sel)].map(tr =>"
            # the name's cell and the group's, then a cell per column
            "  [...tr.querySelectorAll(':scope > td')].slice(2)"
            "   .map(td => td.getAttribute('data-foreign') === 'yes'))", rows)

    def columns():
        return page.evaluate(
            "() => document.querySelectorAll("
            "  '#ts-stage-table thead th').length - 2")

    # the columns are the description's `varies`, in its order
    varies = json.loads((calc / "task.json").read_text())["varies"]
    unread = {("relax", "fc_displacement"), ("freq", "relax_force_tol")}
    assert foreign() == [[(row, col) in unread for col in varies]
                         for row in ("relax", "freq")], (varies, foreign())
    box = f"{rows}:nth-child(1) input.ts-cell-name"
    # the box typed into is marked, so only the REDRAWN row -- the one the
    # rename's role is read for -- satisfies the wait
    page.evaluate("(sel) => { document.querySelector(sel).dataset.typed = '1'; }",
                  box)
    name_box = page.locator(box)
    name_box.fill("Relax")
    name_box.dispatch_event("change")
    page.wait_for_function(
        "(sel) => { const b = document.querySelector(sel);"
        "           return b && !b.dataset.typed && b.value === 'Relax'; }",
        arg=box, timeout=10000)
    assert foreign() == [[(row, col) in unread for col in varies]
                         for row in ("relax", "freq")], (
        "the row renamed `Relax` lost its role", varies, foreign())
    before = columns()
    page.locator(f"{rows}:nth-child(2) select.ts-preset").select_option(
        index=1)
    assert columns() == before, "a preset filled the freq rung's columns"
    page.locator(f"{rows}:nth-child(1) select.ts-preset").select_option(
        index=1)
    page.wait_for_function(f"() => document.querySelectorAll("
                           f"'#ts-stage-table thead th').length - 2 > {before}",
                           timeout=10000)
