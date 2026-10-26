"""Where an item binds -- each rung reads what its ROLE reads, decided by
the catalogue alone -- through the road: the Task-setup save, the Transport
tab's Send and `jobset prep`.

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


from support.junction import _isolated  # noqa: F401 -- its sandbox, autouse


def _described(engine, stages, calculation="optimization"):
    """A description as a person's editor leaves it, and its job request:
    ``stages`` the rows, ``varies`` what they override -- each row its own
    group, as the page numbers a stage it adds."""
    from molbuilder.identity import run_id
    from molbuilder.job_request import SCHEMA as REQUEST_SCHEMA
    from molbuilder.task import SCHEMA as TASK_SCHEMA
    task = {"schema": TASK_SCHEMA, "engine": {"name": engine},
            "calculation": calculation,
            "shape": "hierarchical",
            "run": {"name": "x", "id": run_id("x", "H2"),
                    "created": "2026-09-30T00:00:00-07:00"},
            "structure": {"source": "s.xyz", "formula": "H2", "atoms": 2},
            "varies": sorted({k for s in stages for k in s["overrides"]}),
            "stages": stages}
    request = {"schema": REQUEST_SCHEMA,
               "stages": {s["name"]: {"group": i + 1}
                          for i, s in enumerate(stages)}}
    return task, request


def _save(web_client, dest, described):
    """``described`` -- a description and its job request -- through Task
    setup's Save, which writes both."""
    from support.road import save_both
    return save_both(dest, *described, client=web_client)


#: A transport calculation's five rungs, in their default groups, and the
#: relaxation it cites.
_RUNGS = ("seed", "electrode_L", "electrode_R", "device", "transmission")
_GROUPS = (1, 1, 1, 2, 3)
_CITE = "J/optimization/relax/01_coarse/run-0"


def _transport(*, execution=None, on=None, rung_execution=None):
    """A transport description and its job request as Task setup's Save
    takes them: ``execution`` the calculation's run card, ``rung_execution``
    the card of rung ``on``."""
    from molbuilder.identity import run_id
    from molbuilder.job_request import SCHEMA as REQUEST_SCHEMA
    from molbuilder.task import SCHEMA as TASK_SCHEMA
    task = {"schema": TASK_SCHEMA, "engine": {"name": "siesta"},
            "calculation": "transport", "shape": "hierarchical",
            "run": {"name": "t", "id": run_id("t", _CITE, stage_names=_RUNGS),
                    "created": "2026-10-12T00:00:00-07:00"},
            "slots": {"junction": _CITE},
            "bias": {"voltages_v": [0.0, 0.2],
                     "low_bias_approximation": False},
            "varies": [],
            "stages": [{"name": n, "overrides": {}} for n in _RUNGS]}
    request = {"schema": REQUEST_SCHEMA,
               "stages": {n: dict({"group": g},
                                  **({"execution": rung_execution}
                                     if n == on else {}))
                          for n, g in zip(_RUNGS, _GROUPS)}}
    if execution is not None:
        request["execution"] = execution
    return task, request


def _template_beside(dest, engine, calculation, cfg):
    """The template the hand-over leaves beside the description, which the
    save reads for the checks that compare RESOLVED rungs."""
    from molbuilder.template import template_with_values
    from molbuilder.template_file import template_path
    template_path(dest, "x").write_text(template_with_values(
        cfg, engine=engine, calculation=calculation), encoding="utf-8")


def test_an_override_its_rung_does_not_read_is_refused_where_it_is_saved(
        web_client, isolated_projects_root):
    """A transport ladder's transmission rung reads the transmission's
    settings, whatever the rungs are called: one on the seed is refused by
    name, naming the rung that reads it -- as an item the kind does not
    carry is, on any rung, and a name no field has, on a ladder whose
    groups are checked -- and nothing is saved.

    MUTATIONS THIS MUST FAIL AGAINST: the description's check not asking
    which rung reads an override (it saves); the group check asked of a
    ladder that does not resolve (the save crashes, 500)."""
    from test_task_setup_tab import _fresh_calc_dir
    d = isolated_projects_root / "J" / "transport" / "t"
    d.mkdir(parents=True)
    task, request = _transport()
    task["stages"][0]["overrides"] = {"transmission_emin_ev": -1.0}
    task["varies"] = ["transmission_emin_ev"]
    r = _save(web_client, d, (task, request))
    assert r.status_code == 400, r.get_json()
    refused = {f["stage"]: f["message"] for f in r.get_json()["findings"]
               if f["severity"] == "error"}
    assert ("only the transmission rung of a transport calculation reads, "
            "not 'seed'") in refused["seed"], refused
    assert not (d / "task.json").exists()

    d = _fresh_calc_dir(isolated_projects_root)
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
    assert ("stage 'transmission' execution sets gpu_count = 1, which only "
            "the seed / electrode_L / electrode_R / device rungs of a "
            "transport calculation read, not 'transmission'") \
        in r.get_json()["error"], r.get_json()["error"]
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
    calculation run on the GPU is saved, and each SCF rung resolves to a run
    on one GPU, while the transmission -- `tbtrans` runs on no GPU --
    resolves to a run on none, its other values kept, so prep's G5 has
    nothing to refuse there.  Read through prep's two steps for a rung
    (`prep_inputs.run_inputs`, then `resolve`), not prep's files: a
    transport rung is prepared on a finished relaxation, which only the
    engine makes, and the real junction's e2e runs on the CPU.  Without it
    the transmission was a GPU job, holding a device it never uses
    (measured 2026-10-17).

    MUTATION THIS MUST FAIL AGAINST: the calculation's card handed whole to
    every rung (`JobRequest.run_condition`)."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset.model import Resources, gpu_request
    from molbuilder.jobset.prep_inputs import run_inputs
    from molbuilder.resolve import resolve
    from molbuilder.template import template_with_values
    from molbuilder.template_file import template_path
    d = isolated_projects_root / "J" / "transport" / "t"
    d.mkdir(parents=True)
    text = template_with_values(SiestaConfig(), engine="siesta",
                                calculation="transport")
    template_path(d, "t").write_text(text, encoding="utf-8")
    r = _save(web_client, d, _transport(execution={
        "use_gpu": True, "gpu_count": 1, "mpi_np": 4, "omp_threads": 1}))
    assert r.status_code == 200, r.get_json()
    calc = open_calculation(d)
    asked = {}
    for rung in _RUNGS:
        shape, settings = run_inputs(calc.request, rung)
        assert (shape.get("mpi_np"), shape.get("cpus_per_task")) == (4, 1), \
            (rung, shape)
        ps = resolve(text, calc.task, SiestaConfig,
                     allocation=Resources(**shape), stage=rung,
                     run_settings=settings)
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
        {"name": "freq", "overrides": {"relax_steps": 0}}],
        calculation="vibration"))
    assert r.status_code == 400, r.get_json()
    assert "stage 'freq' relaxes" in r.get_json()["error"], \
        r.get_json()["error"]
    assert "relaxes nothing" in r.get_json()["error"]
    found = [f["severity"] for f in r.get_json()["findings"]
             if f["where"] == "config.relax_steps"]
    assert found == ["error"], r.get_json()["findings"]
