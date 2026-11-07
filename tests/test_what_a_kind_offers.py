"""What a calculation kind may take -- through the road: describe -> ``jobset
prep`` -> the deck, or prep's refusal; and the two surfaces a person picks a
value on, through their own routes.

PINS: ``docs/engines/template.md`` § 6.3a (`offered`: the choices a kind may
take, one door read by the form, the stage table's cells, ``resolve`` and the
settings gate, refusing by name with the reason ``template.why_not_offered``
gives); ``docs/engines/vibration.md`` § 4.6 (PCM reaches one route);
``docs/plans/plan.md`` § 5w K2.

PREVENTS, each read in the code before 2026-09-30:

* a SIESTA vibration's relaxation not run at all (`none`) -- offered on the
  form and the stage table (SS-C4);
* an optimization asked to solve with `transiesta`, which dies after the
  queue wait (SO-C13);
* a PySCF vibration run unrestricted, whose spectrum record carries one spin
  channel (PS-C2's refusal half);
* PCM on the routes built without the solvent's response (PS-C3).

Nothing here launches an engine: prep writes the deck and stops.
"""
from __future__ import annotations

import pytest

from test_electronic_state import WATER


# ------------------------------------------------- refused by name at prep

@pytest.mark.parametrize("door", ["template", "stage"])
def test_a_vibration_relaxes_with_a_relaxer(prepared, door):
    """SS-C4, at both doors a value arrives by: the template (`resolve`
    refuses it, naming the template) and the one stage's own override (the
    description's own check, gate ③, refuses it first, naming the stage).
    Each says why a vibration does not take it, and what it does take.

    MUTATION THIS MUST FAIL AGAINST: the vibration's set undeclared (the
    relaxation's deck is written with no relaxation)."""
    said_by = ({"template": {"relax_type": "none"}} if door == "template"
               else {"overrides": {"relax_type": "none"}})
    _d, _s, said = prepared(WATER(), "siesta", calculation="vibration",
                            refused=True, **said_by)
    assert "a vibration does not offer" in said, said
    assert "none relaxes nothing" in said, said
    assert "CG, Broyden, FIRE" in said, said
    assert ("the template sets relax_type = 'none'" if door == "template"
            else "stage 'freq' sets relax_type = 'none'") in said, said


def test_an_optimization_offers_no_transport_solver(prepared):
    """SO-C13.  MUTATION THIS MUST FAIL AGAINST: the optimization's set
    undeclared (the deck says `SolutionMethod transiesta` and the run dies
    after the queue wait)."""
    _d, _s, said = prepared(WATER(), "siesta",
                            template={"solution_method": "transiesta"},
                            refused=True)
    assert "solution_method = 'transiesta'" in said, said
    assert "does not offer" in said, said
    assert "transport device's solver" in said, said


def test_a_pyscf_vibration_is_restricted(prepared):
    """PS-C2's refusal half: the spectrum record carries one spin channel.
    MUTATION THIS MUST FAIL AGAINST: the PySCF vibration's spin set
    undeclared (the deck renders a UKS vibration)."""
    _d, _s, said = prepared(WATER(), "pyscf", calculation="vibration", template=dict(already_relaxed=True, spin_treatment="unrestricted", unpaired_electrons=0), refused=True)
    assert "spin_treatment = unrestricted (stated)" in said, said
    assert "one spin channel" in said and "offers restricted" in said, said


def test_pcm_reaches_one_route(prepared):
    """vibration.md § 4.6: with a solvent, the frequencies of a structure
    with no atoms held, IR and Raman off -- the one route whose Hessian
    carries the solvent's response.  Infrared is on by default, so a
    solvent alone is refused, naming the route and what turns it off.

    MUTATION THIS MUST FAIL AGAINST: the route rule removed (the dipole
    route runs without the solvent's response, under an info saying it has
    it)."""
    _d, _s, said = prepared(WATER(), "pyscf", calculation="vibration", template=dict(already_relaxed=True, solvent="water"), refused=True)
    assert "PCM solvation (water) reaches one route" in said, said
    assert "IR -- its dipole-derivative route" in said, said
    assert "Turn IR off" in said, said
    _d, _s, deck = prepared(WATER(), "pyscf", calculation="vibration", template=dict(already_relaxed=True, solvent="water", compute_raman=False, compute_ir=False))
    assert "78.3553" in deck and ("mf = mf.PCM()" in deck
                                  or "_mb_apply_solvent" in deck), (
        "the frequencies-only route with a solvent preps, solvated")


def test_an_unknown_solvent_is_one_refusal(prepared):
    """One fact, one finding: a solvent the dielectric table does not know
    is the engine's refusal alone -- the route rule speaks of the solvents
    PCM can apply (the K2 review).  MUTATION THIS MUST FAIL AGAINST: the
    route rule asked of any name (two refusals on one field)."""
    _d, _s, said = prepared(WATER(), "pyscf", calculation="vibration", template=dict(already_relaxed=True, solvent="xylol"), refused=True)
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
    `choices` (a vibration's relaxation cell offering `none`)."""
    with _client() as c:
        for choices in (_form_choices(c, "siesta", "vibration", "relax_type"),
                        _cell_choices(c, "siesta", "vibration", "relax_type")):
            assert choices == ["CG", "Broyden", "FIRE"], choices
        assert "none" in _cell_choices(c, "siesta", "optimization",
                                       "relax_type")
        assert _form_choices(c, "pyscf", "vibration",
                             "spin_treatment") == ["restricted"]
        assert "transiesta" not in _form_choices(c, "siesta", "optimization",
                                                 "solution_method")
    # ...and the Transport tab's shared panel, the one surface a transport
    # calculation's spin is picked on: two treatments, and every count --
    # a stated one is written as stated, TranSIESTA's stop said beforehand,
    # never refused (`engines/transport.md` § 3.1's spin note;
    # `science/chemistry-correctness.md` ES6, the user, 2026-11-02).
    from molbuilder.web.app import create_app
    body = create_app(config={}).test_client().get(
        "/api/transport/schema?engine=siesta&surface=shared").get_json()
    shared = {f["name"]: f.get("choices") for s in body["schema"]["sections"]
              for f in s["fields"]}
    assert shared["spin_treatment"] == ["restricted", "unrestricted"], shared
    assert shared["unpaired_electrons"] == list(range(11)) + ["free"], \
        shared
