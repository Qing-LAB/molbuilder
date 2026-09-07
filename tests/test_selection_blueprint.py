"""`/api/selection/eval` — the one selection door, and what it evaluates.

**The browser sends the atoms it is looking at.** MolView holds the structure;
a filter is answered by handing the server that atom list plus the rule
(`lib/molview/model-jobs.js:962`, the single call site). No file is opened,
because the file on disk may not be what is on screen.

**WHAT STOOD HERE UNTIL 2026-09-07.** This file had 38 tests. One drove the
door above; the other 37 drove `/api/selection/atoms` and a `structure_path`
branch of `eval` that read a structure from disk. Both were deleted with the
code they tested, and the code was deleted because of `71729ff9` — the commit
that introduced this door, titled *"filter evaluates against in-memory
workspace atoms, not disk (BUG FIX)"*:

> *Fixes the user bug: assign "L-electrode" in the panel (in-memory), filter
> by it → "no region labelled L-electrode". The filter read the stale saved
> file; now it reads the workspace (memory).*

That commit kept the disk path beside its own replacement, for a case
("Results viewing a saved file") that Results never used — it loads through
MolView like every other tab. So the wrong code outlived its fix by four
months and drifted further: its private reader applied only the sidecar's
`regions`, dropping the identity columns, so `by_residue_name`, `by_chain_id`
and `by_atom_name` returned nothing through it. Measured, not inferred.

The rule operators themselves are pure and live in `molbuilder/selection.py`;
they are covered at that layer by `tests/test_atom_selection.py`. What this
file covers is the DOOR: the body it accepts, what it refuses, and the
envelope it answers with.
"""
from __future__ import annotations

import pytest

from molbuilder.web.app import create_app


@pytest.fixture
def web():
    app = create_app(config={"rate_limit": {"enabled": False}})
    return app.test_client()


def _au_bdt():
    """Two gold atoms labelled in the panel, one sulphur labelled on disk."""
    return [
        {"index": 0, "element": "Au", "labels": ["L-electrode"]},
        {"index": 1, "element": "Au", "labels": ["L-electrode"]},
        {"index": 2, "element": "S",  "labels": ["BDT"]},
    ]


class TestTheDoor:
    """What the browser sends, and what comes back."""

    def test_a_label_assigned_in_the_panel_is_found_before_it_is_saved(self, web):
        """The bug `71729ff9` fixed, and the reason the door takes atoms.

        Assign "L-electrode" in the panel, filter by it, and it is found —
        even though nothing has been written to disk. Reading a file here
        would answer about a structure the person is not looking at.
        """
        r = web.post("/api/selection/eval", json={
            "atoms": _au_bdt(),
            "rule": {"op": "by_region", "name": "L-electrode"}})
        assert r.status_code == 200, r.get_json()
        j = r.get_json()
        assert j["selected_indices"] == [0, 1]
        assert j["n_atoms_total"] == 3

    def test_the_envelope_is_the_uniform_one(self, web):
        r = web.post("/api/selection/eval", json={
            "atoms": _au_bdt(), "rule": {"op": "by_element", "elements": ["Au"]}})
        j = r.get_json()
        assert j["ok"] is True
        assert set(j) >= {"ok", "selected_indices", "count", "n_atoms_total"}
        assert j["count"] == len(j["selected_indices"]) == 2

    @pytest.mark.parametrize("rule,expected", [
        ({"op": "by_element", "elements": ["Au"]},        [0, 1]),
        ({"op": "by_region",  "name": "BDT"},            [2]),
        ({"op": "by_index_range", "expression": "1-2"},  [1, 2]),
        ({"op": "not", "rule": {"op": "by_element", "elements": ["Au"]}}, [2]),
    ])
    def test_the_operators_the_panel_can_build_reach_the_evaluator(
            self, web, rule, expected):
        """The four row kinds the filter UI can produce, through the door.

        Not a re-test of the operators — `tests/test_atom_selection.py` owns
        those as pure functions. This asserts the WIRING: that a rule the
        panel builds arrives intact and its answer comes back as indices.
        """
        r = web.post("/api/selection/eval",
                     json={"atoms": _au_bdt(), "rule": rule})
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["selected_indices"] == expected


class TestWhatItRefuses:
    """The refusals, which had NO coverage at all before 2026-09-07.

    Every one of these lines was live code with zero tests: the only door the
    browser uses was covered by a single test, while 37 covered the two that
    nothing called.
    """

    def test_a_body_with_neither_atoms_nor_a_rule_is_refused(self, web):
        r = web.post("/api/selection/eval", json={})
        assert r.status_code == 400
        assert r.get_json()["ok"] is False

    def test_atoms_must_be_a_list(self, web):
        r = web.post("/api/selection/eval",
                     json={"atoms": "all of them",
                           "rule": {"op": "by_element", "elements": ["Au"]}})
        assert r.status_code == 400
        assert "atoms" in r.get_json()["error"]

    def test_an_atom_must_be_an_object(self, web):
        r = web.post("/api/selection/eval",
                     json={"atoms": ["Au", "Au"],
                           "rule": {"op": "by_element", "elements": ["Au"]}})
        assert r.status_code == 400
        assert "atoms" in r.get_json()["error"]

    def test_a_rule_the_evaluator_does_not_know_is_refused(self, web):
        r = web.post("/api/selection/eval",
                     json={"atoms": _au_bdt(), "rule": {"op": "by_vibes"}})
        assert r.status_code == 400
        assert r.get_json()["ok"] is False

    def test_a_region_no_atom_carries_is_refused_by_name(self, web):
        """`SelectionError` reaches the caller as a 400 that names the label."""
        r = web.post("/api/selection/eval",
                     json={"atoms": _au_bdt(),
                           "rule": {"op": "by_region", "name": "nope"}})
        assert r.status_code == 400
        assert "nope" in r.get_json()["error"]

    def test_a_body_that_is_not_json_is_refused(self, web):
        r = web.post("/api/selection/eval", data="not json",
                     content_type="text/plain")
        assert r.status_code == 400
        assert "json" in r.get_json()["error"].lower()

    def test_a_top_level_array_is_refused(self, web):
        r = web.post("/api/selection/eval", json=[1, 2, 3])
        assert r.status_code == 400
        assert r.get_json()["ok"] is False
