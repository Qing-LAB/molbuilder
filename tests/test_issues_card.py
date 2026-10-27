"""A finding lands on the card its field is drawn in -- one rule for the form
and for the finding (`template.card_of`; `engines/template.md` § 6.2a,
`web/ui-contract.md` § 5.1).

The silent failure: a warning shown on another card than its field's.  A
vibration draws its Raman settings in the Raman card and its relaxation in
*Frequencies and mode shapes*; a finding placed by any other rule would sit
cards away from the control it is about, and on the Spectrum tab -- a form
drawn in parts -- in the wrong part.  Asked of the producer the pages read
(`issues_to_json`) against the schema the pages draw, field by field, and of
the routes that send findings.
"""
from __future__ import annotations

import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.issues import Issue
from molbuilder.web.blueprints._shared import (catalogue_to_form_schema,
                                                issues_to_json)

_CLS = {"siesta": SiestaConfig, "pyscf": PySCFConfig}

#: Every form part a page draws: the Build tab's two engine forms, and the
#: Spectrum tab's shared parts -- the electronic state and the rest -- and
#: each engine's own (`form-schema.md` § 1.3).  A shared part is checked
#: against both engines' configs: its findings come from whichever engine
#: is sent.
_PARTS = [("siesta", "optimization", None), ("pyscf", "optimization", None),
          ("siesta", "vibration", "own"), ("pyscf", "vibration", "own"),
          ("siesta", "vibration", "common"), ("pyscf", "vibration", "common"),
          ("siesta", "vibration", "state"), ("pyscf", "vibration", "state")]


def _card_of(where: str, engine: str, kind: str):
    out = issues_to_json([Issue("warn", "x", where)], cfg=_CLS[engine],
                         calculation=kind)
    return out[0].get("card")


@pytest.mark.parametrize("engine,kind,surface", _PARTS)
def test_a_finding_about_a_field_lands_on_the_card_it_is_drawn_in(
        engine, kind, surface):
    """Field by field, the card the form draws it in is the card a finding
    about it carries.  MUTATION THIS MUST FAIL AGAINST: the finding placed
    by the field's `group` alone (every Raman, infrared and relaxation
    setting of a vibration on a group card the form does not draw them in).
    """
    schema = catalogue_to_form_schema(
        None if surface in ("common", "state") else engine, "p",
        calculation=kind,
        surface=surface)
    fields = [f for s in schema["sections"] for f in s["fields"]]
    assert fields, (engine, kind, surface)
    wrong = {f["name"]: (f["card"], _card_of(f"config.{f['name']}",
                                             engine, kind))
             for f in fields
             if _card_of(f"config.{f['name']}", engine, kind) != f["card"]}
    assert not wrong, wrong


#: (where, card) -- a finding not about a field of the form goes to the
#: residual panel; a sub-path goes through its root field.
_ROWS = [
    ("geometry.min_distance", None),
    ("cell.volume", None),
    ("config.no_such_field", None),
    ("", None),
    ("config.psml_lib.Au", "setup"),
    ("config.net_charge.makov_payne", "profile"),
]


@pytest.mark.parametrize("where,card", _ROWS)
def test_what_is_not_a_field_goes_to_the_residual_panel(where, card):
    assert _card_of(where, "siesta", "optimization") == card


def test_a_config_is_placed_by_its_kind_and_none_places_nothing():
    """The card depends on the kind -- `relax_type` is a group's card on an
    optimization and a result's on a vibration -- so a config sent without
    its kind is refused rather than placed by the group; with no config the
    finding carries no card and lands in the residual panel."""
    issue = [Issue("warn", "x", "config.relax_type")]
    assert _card_of("config.relax_type", "siesta", "optimization") == "stage"
    assert _card_of("config.relax_type", "siesta", "vibration") == "modes"
    with pytest.raises(ValueError):
        issues_to_json(issue, cfg=SiestaConfig)
    assert "card" not in issues_to_json(issue)[0]


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_the_preflight_sends_the_findings_card(web_client, engine):
    """The live check's route places a finding by the same rule: Raman for
    a range stated nowhere is refused on the Raman card, on both engines.
    MUTATION THIS MUST FAIL AGAINST: the route serialising its findings
    without the kind (no card on the wire, the finding in the residual
    panel)."""
    from support.envelope import envelope
    r = web_client.post("/api/build/preflight", json={
        "structure": envelope(["H", "H"], [[0, 0, 0], [0, 0, 0.74]]),
        "engine": engine, "calculation": "vibration",
        "params": {"compute_raman": True, "raman_modes": "range"}})
    body = r.get_json()
    ours = [i for i in body.get("issues") or []
            if i.get("where") == "config.raman_modes"]
    assert ours and all(i.get("card") == "raman" for i in ours), body


def test_the_task_setup_save_findings_carry_their_card(
        web_client, isolated_projects_root):
    """A `config.<key>` finding from the task-setup SAVE route reaches the
    page with the card its field is drawn in (`web/ui-contract.md` § 5.1):
    a stage overriding an integer setting with text."""
    import dataclasses
    import json as _json
    from molbuilder.template import card_of, items_of
    from support.envelope import envelope
    from support.road import save_both
    items = items_of(SiestaConfig)
    key = next(f.name for f in dataclasses.fields(SiestaConfig)
               if f.name in items and items[f.name].group == "stage"
               and f.type in ("int", int))
    d = isolated_projects_root / "wg" / "optimization" / "probe"
    d.mkdir(parents=True, exist_ok=True)
    over = web_client.post("/api/task-setup/handover", json=dict(
        structure=envelope(["H", "H"], [[0, 0, 0], [0, 0, 0.74]]),
        engine="siesta", calculation="optimization", name="probe",
        params={"system_label": "probe"})).get_json()
    assert over and over.get("ok"), over
    (d / over["handover_name"]).write_text(over["handover_text"])
    h = _json.loads(over["handover_text"])
    proposed = {"schema": h["task_schema"], "engine": h["engine"],
                "calculation": h["calculation"],
                "shape": "flat", "run": h["run"], "structure": h["structure"],
                "varies": [key],
                "stages": [{"name": "coarse",
                            "overrides": {key: "not-an-int"}}]}
    r = save_both(d, proposed, {"schema": h["job_request_schema"],
                                "stages": {"coarse": {"group": 1}}},
                  web_client)
    ours = [f for f in (r.get_json().get("findings") or [])
            if f.get("where") == f"config.{key}"]
    assert ours, r.get_json()
    assert all(f.get("card") == card_of(items[key], "optimization")
               for f in ours), ours
