"""What a calculation kind may take -- through the road: describe -> ``jobset
prep`` -> the deck, or prep's refusal; and the two surfaces a person picks a
value on, through their own routes.

PINS: ``docs/engines/template.md`` § 6.3a (`offered`: the choices a kind may
take, one door read by the form, the stage table's cells, ``resolve`` and the
settings gate, refusing by name with the reason ``template.why_not_offered``
gives); ``docs/engines/vibration.md`` § 4.6 (PCM reaches one route);
``docs/plans/plan.md`` § 5w K2.

PREVENTS, each read in the code before 2026-09-30:

* a SIESTA vibration's relaxation run as dynamics (`Verlet`, `Nose`) or not
  at all (`none`) -- offered on the form and the stage table (SS-C4);
* an optimization asked to solve with `transiesta`, which dies after the
  queue wait (SO-C13);
* a PySCF vibration run unrestricted, whose spectrum record carries one spin
  channel (PS-C2's refusal half);
* PCM on the routes built without the solvent's response (PS-C3).

Nothing here launches an engine: prep writes the deck and stops.
"""
from __future__ import annotations

import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.pyscf.stages import vibration_stages
from molbuilder.siesta.stages import default_siesta_stages

from test_electronic_state import WATER
from test_engine_offset_reaches_every_deck import _prep
from test_fixed_and_shared_items_e2e import (_first_rung_overrides,
                                             _template_says)


def _siesta(root, kind, *, before_prep=None, refused=False):
    stages = (default_siesta_stages("publishable") if kind == "optimization"
              else vibration_stages("siesta", already_relaxed=False))
    return _prep(root, WATER(), SiestaConfig(system_label="JOB"), stages,
                 "siesta", calculation=kind, before_prep=before_prep,
                 refused=refused)


def _pyscf_vibration(root, cfg, *, refused=False):
    return _prep(root, WATER(), cfg,
                 vibration_stages("pyscf", already_relaxed=True), "pyscf",
                 calculation="vibration", refused=refused)


# ------------------------------------------------- refused by name at prep

@pytest.mark.parametrize("door", ["template", "stage"])
def test_a_vibration_relaxes_with_a_relaxer(isolated_projects_root, door):
    """SS-C4, at both doors a value arrives by: the template (`resolve`
    refuses it, naming the template) and the relax rung's own override (the
    description's own check, gate ③, refuses it first, naming the rung).
    Each says why a vibration does not take it, and what it does take.

    MUTATION THIS MUST FAIL AGAINST: the vibration's set undeclared (the
    deck is written with `MD.TypeOfRun Verlet`)."""
    def put(dest):
        if door == "template":
            _template_says(dest, "relax_type", "Verlet")
        else:
            _first_rung_overrides(dest, "relax_type", "Verlet")
    _d, _s, said = _siesta(isolated_projects_root, "vibration",
                           before_prep=put, refused=True)
    assert "a vibration does not offer" in said, said
    assert "molecular dynamics" in said, said
    assert "CG, Broyden, FIRE" in said, said
    assert ("the template sets relax_type = 'Verlet'" if door == "template"
            else "stage 'relax' sets relax_type = 'Verlet'") in said, said


def test_an_optimization_offers_no_transport_solver(isolated_projects_root):
    """SO-C13.  MUTATION THIS MUST FAIL AGAINST: the optimization's set
    undeclared (the deck says `SolutionMethod transiesta` and the run dies
    after the queue wait)."""
    _d, _s, said = _siesta(
        isolated_projects_root, "optimization",
        before_prep=lambda dest: _template_says(dest, "solution_method",
                                                "transiesta"),
        refused=True)
    assert "solution_method = 'transiesta'" in said, said
    assert "does not offer" in said, said
    assert "transport device's solver" in said, said


def test_a_pyscf_vibration_is_restricted(isolated_projects_root):
    """PS-C2's refusal half: the spectrum record carries one spin channel.
    MUTATION THIS MUST FAIL AGAINST: the PySCF vibration's spin set
    undeclared (the deck renders a UKS vibration)."""
    _d, _s, said = _pyscf_vibration(
        isolated_projects_root,
        PySCFConfig(job_name="JOB", already_relaxed=True,
                    spin_treatment="unrestricted", unpaired_electrons=0),
        refused=True)
    assert "spin_treatment = unrestricted (stated)" in said, said
    assert "one spin channel" in said and "offers restricted" in said, said


def test_pcm_reaches_one_route(isolated_projects_root):
    """vibration.md § 4.6: with a solvent, the frequencies of a structure
    with no atoms held, IR and Raman off -- the one route whose Hessian
    carries the solvent's response.  Raman is on by default, so a solvent
    alone is refused, naming the route and what turns it off.

    MUTATION THIS MUST FAIL AGAINST: the route rule removed (the Raman loop
    runs without the solvent's response, under an info saying it has it)."""
    _d, _s, said = _pyscf_vibration(
        isolated_projects_root,
        PySCFConfig(job_name="JOB", already_relaxed=True, solvent="water"),
        refused=True)
    assert "PCM solvation (water) reaches one route" in said, said
    assert "Raman -- its polarizability loop" in said, said
    assert "Turn Raman off" in said, said
    _d, _s, deck = _pyscf_vibration(
        isolated_projects_root,
        PySCFConfig(job_name="JOB", already_relaxed=True, solvent="water",
                    compute_raman=False, compute_ir=False))
    assert "78.3553" in deck and ("mf = mf.PCM()" in deck
                                  or "_mb_apply_solvent" in deck), (
        "the frequencies-only route with a solvent preps, solvated")


def test_an_unknown_solvent_is_one_refusal(isolated_projects_root):
    """One fact, one finding: a solvent the dielectric table does not know
    is the engine's refusal alone -- the route rule speaks of the solvents
    PCM can apply (the K2 review).  MUTATION THIS MUST FAIL AGAINST: the
    route rule asked of any name (two refusals on one field)."""
    _d, _s, said = _pyscf_vibration(
        isolated_projects_root,
        PySCFConfig(job_name="JOB", already_relaxed=True, solvent="xylol"),
        refused=True)
    assert "unknown solvent 'xylol'" in said, said
    assert "reaches one route" not in said, said


# ------------------------------------ what the two picking surfaces offer

def _client():
    from flask import Flask
    from molbuilder.web.blueprints.build import bp
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app.test_client()


def _form_choices(c, engine, kind, name):
    sch = c.get(f"/api/build/schema/{engine}?calculation={kind}"
                ).get_json()["schema"]
    return next(f["choices"] for s in sch["sections"] for f in s["fields"]
                if f["name"] == name)


def _cell_choices(c, engine, kind, name):
    cols = c.get(f"/api/task-setup/columns?engine={engine}"
                 f"&calculation={kind}").get_json()
    return next(i["choices"] for i in cols["items"] if i["name"] == name)


def test_the_form_and_the_stage_table_offer_the_kinds_set():
    """The form and the stage table's cells offer what the kind may take --
    the set `prep` holds a value to, so neither offers what prep would
    refuse (`template.md` § 6.3a).  The Build form's own route and the Task
    setup columns route, as the browser asks them.

    MUTATION THIS MUST FAIL AGAINST: either surface reading the item's whole
    `choices` (a vibration's relaxation cell offering `Verlet`)."""
    with _client() as c:
        for choices in (_form_choices(c, "siesta", "vibration", "relax_type"),
                        _cell_choices(c, "siesta", "vibration", "relax_type")):
            assert choices == ["CG", "Broyden", "FIRE"], choices
        assert "Verlet" in _cell_choices(c, "siesta", "optimization",
                                         "relax_type")
        assert _form_choices(c, "pyscf", "vibration",
                             "spin_treatment") == ["restricted"]
        assert "transiesta" not in _form_choices(c, "siesta", "optimization",
                                                 "solution_method")
    # ...and the Transport tab's shared panel, the one surface a transport
    # calculation's spin is picked on: two treatments, and no fixed count
    # but restricted's 0 (`engines/transport.md` § 3.1's spin note).
    from molbuilder.web.app import create_app
    body = create_app(config={}).test_client().get(
        "/api/transport/schema?surface=shared").get_json()
    shared = {f["name"]: f.get("choices") for s in body["schema"]["sections"]
              for f in s["fields"]}
    assert shared["spin_treatment"] == ["restricted", "unrestricted"], shared
    assert shared["unpaired_electrons"] == [0, "free"], shared


def test_a_description_refuses_what_its_kind_does_not_offer():
    """Gate ③ (`validation/task.py`): a rung's value the kind does not offer
    is refused when the description is WRITTEN -- the describe door
    `jobset init` and the web hand-over share -- never first at prep, on the
    machine that runs it (`engines/template.md` § 6.3a).

    MUTATION THIS MUST FAIL AGAINST: the check removed from gate ③ (the
    description is written, and only prep refuses it)."""
    from molbuilder import describe as D
    from molbuilder.task import Stage
    ladder = vibration_stages("siesta", already_relaxed=False)
    relax = ladder[0]
    ladder = (Stage(name=relax.name, enabled=relax.enabled,
                    overrides={**dict(relax.overrides or {}),
                               "relax_type": "Verlet"}),) + tuple(ladder[1:])
    with pytest.raises(D.DescribeError) as refused:
        D.build_description(WATER(), SiestaConfig(system_label="JOB"), ladder,
                            engine="siesta", shape="hierarchical", name="JOB",
                            calculation="vibration", source="water.xyz")
    said = str(refused.value)
    assert "relax_type = 'Verlet'" in said, said
    assert "a vibration does not offer" in said, said
    assert "CG, Broyden, FIRE" in said, said
