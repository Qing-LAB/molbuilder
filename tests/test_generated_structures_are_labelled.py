"""A generated structure arrives carrying a label naming what generated it.

WHY: a molecule built from a SMILES string, a PubChem name or a sequence had
no handle on it.  The text that produced it lived in a status line that the
next load erased, and "select the thing I just built" meant drawing a box
round it.  One region over every atom, named for the input, makes it a click
-- and because labels persist in the `.molstruct.json` beside the geometry,
the provenance survives the save.

THE TRAILING `#` MARKS IT MOLBUILDER'S (`model/structure-annotations.md`
§ 5.1): a machine-written label stays told apart from a hand-written one, and
whatever was typed, the label is never one of the two lead names.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


@pytest.mark.parametrize("kind,text", [
    ("peptide", "AG"),
    ("smiles",  "CCO"),
    ("dna",     "ATCG"),
])
def test_the_generator_signs_its_work(client, kind, text):
    r = client.post("/api/build/molecule", json={"kind": kind, "input": text})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"], body
    env = body["structure"]
    regions = (env.get("metadata") or {}).get("regions") or {}
    label = f"{text}#"
    assert label in regions, (
        f"a {kind} built from {text!r} carries no label naming its source; "
        f"regions={list(regions)}")
    # It covers the WHOLE molecule -- the point is selecting it as a unit.
    n_atoms = len(env["elements"])
    assert sorted(regions[label]) == list(range(n_atoms))
