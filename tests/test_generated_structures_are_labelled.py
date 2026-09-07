"""A generated structure arrives carrying a label naming what generated it.

WHY: a molecule built from a SMILES string, a PubChem name or a sequence had
no handle on it.  The text that produced it lived in a status line that the
next load erased, and "select the thing I just built" meant drawing a box
round it.  One region over every atom, named for the input, makes it a click
-- and because labels persist in the `.molstruct.json` beside the geometry,
the provenance survives the save.

THE TRAILING `#` IS LOAD-BEARING, not decoration.  Region labels are ONE
namespace: the user's own labels, and the transport vocabulary where any label
ending `-electrode` IS a semi-infinite lead
(`config.transport.is_electrode_label`).  The name generator takes whatever a
person types into it, so a PubChem search for `gold-electrode` would have
labelled the whole molecule a TranSIESTA electrode -- silently, at HTTP 200,
and the next transport run would have believed it.

The marker settles that without touching the electrode matcher in either
direction, which is the point: TranSIESTA imposes no naming rule of its own
(names are free strings in `%block TS.Elecs`, and molbuilder strips its suffix
before the deck is written), so the suffix is a molbuilder convention and the
right fix was to stay off it rather than widen or narrow it.
"""
from __future__ import annotations

import pytest

from molbuilder.config.transport import is_electrode_label


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


@pytest.mark.parametrize("kind,text", [
    ("peptide", "AG"),
    ("smiles",  "CCO"),
    ("dna",     "ATCG"),
])
def test_a_generated_label_is_never_read_as_an_electrode(client, kind, text):
    r = client.post("/api/build/molecule", json={"kind": kind, "input": text})
    assert r.status_code == 200
    regions = (r.get_json()["structure"].get("metadata") or {}).get("regions") or {}
    for label in regions:
        assert not is_electrode_label(label), (
            f"{label!r} would be emitted as a TranSIESTA lead")


def test_the_marker_is_what_makes_that_true():
    """The guard above passes trivially on `CCO`; this is the case it is for.

    A person can type anything into the name lookup.  Without the marker the
    label IS the query, and the query can read as a lead.
    """
    assert is_electrode_label("gold-electrode") is True
    assert is_electrode_label("gold-electrode#") is False
    assert is_electrode_label("electrode#") is False
