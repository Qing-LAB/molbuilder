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
from test_transport_prep import _isolated  # noqa: F401 -- its sandbox, autouse


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
           "stages": [dict(s, enabled=True) for s in stages]}
    if calculation != "optimization":
        out["calculation"] = calculation
    return out


def _save(web_client, dest, described):
    return web_client.post("/api/task-setup/save", json={
        "dest": str(dest), "text": json.dumps(described)})


def _template_beside(dest, engine, calculation, cfg):
    """The template the hand-over leaves beside the description, which the
    save reads for the checks that compare RESOLVED rungs."""
    from molbuilder.template import template_path, template_with_values
    template_path(dest, "x").write_text(template_with_values(
        cfg, engine=engine, calculation=calculation), encoding="utf-8")


def test_an_override_its_rung_does_not_read_is_refused_where_it_is_saved(
        web_client, isolated_projects_root):
    """A vibration's relaxation rung reads the relaxation's settings and its
    force-constant rung the displacement, whatever the stages are called:
    each on the other rung is refused by name, naming the rung that reads
    it -- as an item the kind does not carry is, on any rung -- and nothing
    is saved.

    MUTATION THIS MUST FAIL AGAINST: the description's check not asking
    which rung reads an override (both save)."""
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
        (Stage(name="relax", enabled=True,
               overrides={"relax_force_tol": 0.05}),
         Stage(name="freq", enabled=True,
               overrides={"fc_displacement": 0.02})),
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
            "  [...tr.querySelectorAll(':scope > td')].slice(1)"
            "   .map(td => td.getAttribute('data-foreign') === 'yes'))", rows)

    def columns():
        return page.evaluate(
            "() => document.querySelectorAll("
            "  '#ts-stage-table thead th').length - 1")

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
                           f"'#ts-stage-table thead th').length - 1 > {before}",
                           timeout=10000)
