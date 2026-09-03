"""**The measured bond lengths are a REFERENCE now, not a default.**

*(User ruling, 2026-09-03: "we abandoned the way how a junction is constructed
by using bond distances… we can give this information as a reference though.
When we measure two points, if we detect the combination is inside a known
list, we can give some reference value somewhere.")*

`data/contact_distance.json` holds six metal–anchor distances with their
literature sources.  They used to feed `default_contact_distance`, which
supplied the gap when a builder placed an electrode.  That builder
(`add_electrode_slab`) was deleted 2026-09-01 — metal is added by hand and a
slab is placed by an absolute z offset — so the lookup answered a question
nobody asks, and it went.

What the numbers are good for is answering *"what is this bond usually"* at
the moment you are looking at one.  So MolView's measurement readout shows the
literature value when the two atoms you picked are a known pair.

**Two homes, one guarded.**  The numbers live in the JSON with their sources;
`molview/ui.js` carries a copy, because fetching six constants per measurement
is not worth a round trip.  This file is what makes the copy safe — the same
arrangement `script_emit.SIESTA_BENCH_FIELDS` has with the catalogue, where a
hand-written list is matched against its source and a disagreement is refused.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "molbuilder" / "data" / "contact_distance.json"
UI_JS = REPO / "molbuilder" / "web" / "static" / "lib" / "molview" / "ui.js"

_ENTRY = re.compile(r'"([A-Z][a-z]?)\|([A-Z][a-z]?)"\s*:\s*([0-9.]+)')


def _from_json() -> dict:
    data = json.loads(TABLE.read_text(encoding="utf-8"))["metals"]
    return {f"{sym}|{e['anchor']}": float(e["d"]) for sym, e in data.items()}


def _from_js() -> dict:
    text = UI_JS.read_text(encoding="utf-8")
    m = re.search(r"const CONTACT_REFERENCE = \{(.*?)\};", text, re.S)
    assert m, "CONTACT_REFERENCE is not in molview/ui.js any more"
    return {f"{m2.group(1)}|{m2.group(2)}": float(m2.group(3))
            for m2 in _ENTRY.finditer(m.group(1))}


def test_the_browsers_copy_matches_the_measured_table():
    """The whole reason a copy is allowed."""
    src, copy = _from_json(), _from_js()
    assert copy == src, (
        "molview/ui.js's CONTACT_REFERENCE disagrees with "
        "data/contact_distance.json.  The JSON is the source -- it carries the "
        "literature citations -- so fix the copy, not the table.\n"
        f"  json: {sorted(src.items())}\n  js:   {sorted(copy.items())}"
    )


def test_both_are_non_empty_and_keyed_by_the_PAIR():
    """Guards the comparison above from passing on two empty dicts, and pins
    the shape the readout depends on: the anchor element is part of the key,
    because Pt–N (2.05) and Pt–S (2.30) are different bonds."""
    src = _from_json()
    assert len(src) >= 6, f"only {len(src)} entries parsed from the table"
    assert "Au|S" in src and src["Au|S"] == 2.40
    assert "Pt|N" in src and src["Pt|N"] == 2.05, (
        "Pt's anchor is nitrogen, not sulfur -- keying by the metal alone is "
        "what the retired `default_contact_distance` did, and it is why the "
        "anchor was unreachable")


def test_every_entry_carries_a_source_note():
    """The numbers are physics, so the file must say where each came from.
    A value with no provenance is a number somebody typed."""
    raw = json.loads(TABLE.read_text(encoding="utf-8"))
    assert raw.get("_sources"), "the table states no literature sources"
    for sym, entry in raw["metals"].items():
        assert entry.get("note"), f"{sym} has no note saying what bond it is"
        assert sym in raw["_sources"] or entry["anchor"] in raw["_sources"], (
            f"{sym} is not mentioned in the table's _sources line")


def test_the_default_supplier_is_gone():
    """`default_contact_distance` handed out one number per METAL, dropping
    the anchor and the note.  It is retired with its last caller; importing it
    must fail rather than quietly work."""
    import molbuilder.modify as modify
    for name in ("default_contact_distance", "_load_contact_distance",
                 "_get_contact_distance"):
        assert not hasattr(modify, name), (
            f"modify.{name} is back -- the contact table is a reference now, "
            "not a source of defaults")
