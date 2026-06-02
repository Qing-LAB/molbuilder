"""Tests for /api/selection/eval and /api/selection/toggle (L2 of
the selection system).

Endpoint contract is pinned: shape of response, error responses on
bad input, and the click-toggle bookkeeping semantics:

  * Clicking an unselected atom adds it to the rule (via Or with a
    ByClick clause).
  * Clicking an already-selected atom deselects it (via Minus or by
    removing it from an existing ByClick clause).
  * The endpoints are stateless: the rule travels in the request /
    response body, the server does not store it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder import diagnostics


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


@pytest.fixture
def selection_root(tmp_path: Path):
    """tmp dir wired in as the file-picker root with a small mixed-
    element XYZ ready to use as the structure for selection tests."""
    (tmp_path / "junction.xyz").write_text(
        "11\njunction\n"
        "Au 0 0 0\n" "Au 1 0 0\n" "Au 2 0 0\n" "Au 3 0 0\n"
        "C  4 0 0\n" "C  5 0 0\n" "C  6 0 0\n"
        "Au 7 0 0\n" "Au 8 0 0\n" "Au 9 0 0\n" "Au 10 0 0\n"
    )
    caps = diagnostics.Capabilities(
        runtime_config={},
        conda_binary=None,
        conda_envs=frozenset(),
    )

    def _only_tmp_roots(self):
        return ((tmp_path.resolve(), "projects"),)

    cls = type(caps)
    old = cls.file_picker_roots
    cls.file_picker_roots = _only_tmp_roots
    diagnostics.set_capabilities(caps)
    try:
        yield tmp_path
    finally:
        cls.file_picker_roots = old
        diagnostics.reset_capabilities()


@pytest.fixture
def web(selection_root):
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    app = create_app(config={})
    return app.test_client()


def _path(root, name="junction.xyz"):
    return str((root / name).resolve())


# --------------------------------------------------------------------- #
#  /api/selection/eval                                                  #
# --------------------------------------------------------------------- #


class TestEval:
    def test_by_element_au(self, web, selection_root):
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_element", "elements": ["Au"]},
        })
        assert r.status_code == 200
        j = r.get_json()
        assert j["selected_indices"] == [0, 1, 2, 3, 7, 8, 9, 10]
        assert j["count"] == 8
        assert j["n_atoms_total"] == 11

    def test_by_index_range(self, web, selection_root):
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_index_range", "expression": "4-6"},
        })
        assert r.status_code == 200
        assert r.get_json()["selected_indices"] == [4, 5, 6]

    def test_first_n_au(self, web, selection_root):
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {
                "op": "first_n", "n": 4,
                "rule": {"op": "by_element", "elements": ["Au"]},
            },
        })
        assert r.get_json()["selected_indices"] == [0, 1, 2, 3]

    def test_missing_path_returns_400(self, web):
        r = web.post("/api/selection/eval", json={
            "rule": {"op": "all"},
        })
        assert r.status_code == 400
        assert "structure_path" in r.get_json()["error"]

    def test_missing_rule_returns_400(self, web, selection_root):
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
        })
        assert r.status_code == 400
        assert "rule" in r.get_json()["error"]

    def test_invalid_rule_returns_400(self, web, selection_root):
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "not_a_real_op"},
        })
        assert r.status_code == 400
        assert "invalid rule" in r.get_json()["error"]

    def test_path_traversal_rejected(self, web, selection_root):
        # ".." in raw input is the defense-in-depth reject -- before
        # the resolution step even runs.
        r = web.post("/api/selection/eval", json={
            "structure_path": "../etc/passwd",
            "rule": {"op": "all"},
        })
        assert r.status_code == 400

    def test_pdb_structure_accepted(self, web, selection_root):
        """/api/selection/eval accepts ``.pdb`` files via
        ``Structure.from_pdb`` (broadened 2026-05-21 from XYZ-only
        so /modify can load PDB picks from the sidebar).  A simple
        rule against a tiny PDB returns the expected atom set."""
        (selection_root / "tiny.pdb").write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  N   ALA A   1       0.000   1.500   0.000  1.00  0.00           N\n"
        )
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root, "tiny.pdb"),
            "rule": {"op": "by_element", "elements": ["C"]},
        })
        assert r.status_code == 200, r.data
        j = r.get_json()
        assert j["selected_indices"] == [0, 1]
        assert j["n_atoms_total"] == 3

    def test_non_structure_extension_rejected(self, web, selection_root):
        """The selection blueprint accepts ``.xyz`` and ``.pdb``
        only.  A pick of ``.json`` / ``.log`` / etc. is rejected
        with a clear error naming the supported list (so an end
        user with a sidebar pick of a log file gets useful
        feedback, not a stack trace)."""
        (selection_root / "junk.json").write_text("{}")
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root, "junk.json"),
            "rule": {"op": "all"},
        })
        assert r.status_code == 400
        err = r.get_json()["error"]
        assert "unsupported structure extension" in err
        assert ".xyz" in err and ".pdb" in err

    def test_unknown_region_returns_400(self, web, selection_root):
        # No sidecar exists, so ByRegion can't resolve any name.
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_region", "name": "L-electrode"},
        })
        assert r.status_code == 400
        assert "evaluation failed" in r.get_json()["error"]

    def test_by_region_picks_atoms_saved_to_label(self, web, selection_root):
        """End-to-end: save → reload → eval cycle for the by-label
        filter the panel uses.  Repro for the "by-label filter
        returns no atoms" bug.
        """
        # 1. Save: tag atoms 0..3 as L-electrode.
        save = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, 1, 2, 3],
        })
        assert save.status_code == 200
        # 2. Eval: ask the server to return atoms with that label.
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_region", "name": "L-electrode"},
        })
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["selected_indices"] == [0, 1, 2, 3]


# --------------------------------------------------------------------- #
#  /api/selection/toggle -- bookkeeping semantics                        #
# --------------------------------------------------------------------- #


class TestToggleSemantics:
    def test_click_unselected_with_empty_rule(self, web, selection_root):
        # Starting with ByClick([]), clicking index 5 should add it.
        r = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_click", "indices": []},
            "index": 5,
        })
        assert r.status_code == 200
        j = r.get_json()
        assert j["rule"]["op"] == "by_click"
        assert j["rule"]["indices"] == [5]
        assert j["selected_indices"] == [5]

    def test_click_selected_with_byclick_removes_it(self, web, selection_root):
        # Starting with ByClick([5, 8]), clicking 5 should remove it.
        r = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_click", "indices": [5, 8]},
            "index": 5,
        })
        j = r.get_json()
        assert j["rule"]["op"] == "by_click"
        assert j["rule"]["indices"] == [8]
        assert j["selected_indices"] == [8]

    def test_click_unselected_with_algorithmic_rule_wraps_in_or(
        self, web, selection_root,
    ):
        # ByElement('C') selects {4,5,6}.  Clicking 9 (an Au atom)
        # should add 9 to the selection by wrapping in Or.
        r = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_element", "elements": ["C"]},
            "index": 9,
        })
        j = r.get_json()
        assert j["rule"]["op"] == "or"
        assert j["selected_indices"] == [4, 5, 6, 9]

    def test_click_selected_with_algorithmic_rule_wraps_in_minus(
        self, web, selection_root,
    ):
        # ByElement('C') selects {4,5,6}.  Clicking 5 (already
        # selected) should deselect it via Minus.
        r = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_element", "elements": ["C"]},
            "index": 5,
        })
        j = r.get_json()
        assert j["rule"]["op"] == "minus"
        assert j["selected_indices"] == [4, 6]

    def test_or_with_byclick_edits_in_place(self, web, selection_root):
        # Or(ByElement('C'), ByClick([8])) selects {4,5,6,8}.
        # Clicking 9 should ADD 9 to the existing ByClick clause,
        # not nest in another Or.
        r = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule": {
                "op": "or",
                "operands": [
                    {"op": "by_element", "elements": ["C"]},
                    {"op": "by_click",   "indices":  [8]},
                ],
            },
            "index": 9,
        })
        j = r.get_json()
        # Top-level still Or, ByClick now [8, 9]:
        assert j["rule"]["op"] == "or"
        byclick = [op for op in j["rule"]["operands"]
                   if op["op"] == "by_click"][0]
        assert byclick["indices"] == [8, 9]
        assert j["selected_indices"] == [4, 5, 6, 8, 9]

    def test_round_trip_two_clicks(self, web, selection_root):
        """Click index 5, then click 5 again -- back to nothing."""
        first = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_click", "indices": []},
            "index": 5,
        }).get_json()
        assert first["selected_indices"] == [5]

        second = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule": first["rule"],
            "index": 5,
        }).get_json()
        assert second["selected_indices"] == []

    def test_index_out_of_range_returns_400(self, web, selection_root):
        r = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule":           {"op": "by_click", "indices": []},
            "index":          99,
        })
        assert r.status_code == 400
        assert "out of range" in r.get_json()["error"]

    def test_non_integer_index_returns_400(self, web, selection_root):
        r = web.post("/api/selection/toggle", json={
            "structure_path": _path(selection_root),
            "rule":           {"op": "by_click", "indices": []},
            "index":          "five",
        })
        assert r.status_code == 400

    def test_bool_index_rejected(self, web, selection_root):
        """``isinstance(True, int)`` is True in Python -- without an
        explicit bool guard, ``{"index": true}`` would toggle index 1
        (and ``{"index": false}`` would toggle index 0).  Pin the
        rejection so a future refactor that drops the guard fails
        loudly.
        """
        for bad in (True, False):
            r = web.post("/api/selection/toggle", json={
                "structure_path": _path(selection_root),
                "rule":           {"op": "by_click", "indices": []},
                "index":          bad,
            })
            assert r.status_code == 400, (
                f"bool index {bad!r} should be rejected; got "
                f"{r.status_code} with body {r.get_json()!r}"
            )


# --------------------------------------------------------------------- #
#  Stateless contract                                                   #
# --------------------------------------------------------------------- #


class TestAtomsEndpoint:
    def test_returns_full_atom_list(self, web, selection_root):
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        assert r.status_code == 200
        j = r.get_json()
        assert j["n_atoms"] == 11
        assert len(j["atoms"]) == 11

    def test_element_present_on_each(self, web, selection_root):
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        atoms = r.get_json()["atoms"]
        assert atoms[0]["element"] == "Au"
        assert atoms[4]["element"] == "C"

    def test_indices_are_zero_based_and_dense(self, web, selection_root):
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        idxs = [a["index"] for a in r.get_json()["atoms"]]
        assert idxs == list(range(11))

    def test_pdb_metadata_defaults_for_plain_xyz(self, web, selection_root):
        """Plain XYZ files have no PDB-level atom_name / residue_name /
        chain_id; Structure's dataclass fills them with canonical
        defaults (element / "MOL" / "A").  Pin those defaults so a
        future change to Structure (e.g. ``residue_name = ""``)
        doesn't silently flow through to the panel's tag column as a
        blank row -- the panel relies on these fields being non-empty
        strings to render the cell.

        Earlier shape of this assertion was ``if "residue_name" in a:
        assert a["residue_name"]`` which silently passed both when
        the key was absent AND when it carried the canonical default,
        because the guard skipped on absence (the very case the test
        thought it was validating).  The 2026-05-20 coverage review
        flagged this; aligning the test to the actual server contract
        (defaults are populated, NOT omitted) is the right fix.
        """
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        atoms = r.get_json()["atoms"]
        # All rows must carry the canonical defaults.
        for a in atoms:
            assert a.get("atom_name") == a.get("element"), (
                f"plain-XYZ row's atom_name should equal element; "
                f"got {a!r}"
            )
            assert a.get("residue_name") == "MOL", (
                f"plain-XYZ row's residue_name should be the default "
                f"'MOL'; got {a!r}"
            )
            assert a.get("chain_id") == "A", (
                f"plain-XYZ row's chain_id should be the default 'A'; "
                f"got {a!r}"
            )

    def test_regions_propagate_from_sidecar(self, web, selection_root):
        """When a .molstruct.json sidecar is next to the XYZ, its
        region labels appear in the atoms' ``regions`` field."""
        # Write a sidecar that labels atoms 0..3 as L-electrode.
        import hashlib
        xyz_bytes = (selection_root / "junction.xyz").read_bytes()
        struct_hash = hashlib.sha256(xyz_bytes).hexdigest()
        sidecar = selection_root / "junction.molstruct.json"
        import json as _json
        sidecar.write_text(_json.dumps({
            "schema_version": 3,
            "n_atoms_total":  11,
            "structure_hash": struct_hash,
            "regions":        {"L-electrode": [0, 1, 2, 3]},
            "frozen_atoms":   [10],
            "selection_rules": {},
        }))
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        atoms = r.get_json()["atoms"]
        assert atoms[0]["regions"] == ["L-electrode"]
        assert atoms[4]["regions"] == []
        assert atoms[10]["is_frozen"] is True
        assert atoms[0]["is_frozen"] is False

    def test_missing_path_returns_400(self, web):
        r = web.post("/api/selection/atoms", json={})
        assert r.status_code == 400


class TestSaveEndpoint:
    """Pin /api/selection/save: writes the materialised selection
    into the .molstruct.json sidecar, removing the assigned indices
    from every other region (mutual-exclusion invariant)."""

    def _sidecar_path(self, root):
        return root / "junction.molstruct.json"

    def test_creates_sidecar_when_absent(self, web, selection_root):
        assert not self._sidecar_path(selection_root).exists()
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, 1, 2, 3],
        })
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["regions"]["L-electrode"] == [0, 1, 2, 3]
        assert self._sidecar_path(selection_root).exists()

    def test_assigning_region_does_not_remove_from_other_regions(
        self, web, selection_root,
    ):
        """Multi-label model: assigning to one region does NOT prune
        the atom from other regions.  The user must explicitly
        remove a label (e.g. via the per-tag × button)."""
        # Seed: bridge has atoms 4-6.
        web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "bridge",
            "indices":        [4, 5, 6],
        })
        # Now ALSO assign 5 to L-electrode -- atom 5 should end up
        # in BOTH regions.
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [5],
        })
        j = r.get_json()
        assert j["regions"]["L-electrode"] == [5]
        assert j["regions"]["bridge"] == [4, 5, 6]  # unchanged

    def test_frozen_atoms_independent_of_regions(
        self, web, selection_root,
    ):
        # Atoms can be in a region AND frozen (e.g. electrode buffer
        # layers); frozen_atoms shouldn't reshape the regions map.
        web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, 1],
        })
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "frozen_atoms",
            "indices":        [0, 1],
        })
        j = r.get_json()
        assert j["regions"]["L-electrode"] == [0, 1]
        assert j["frozen_atoms"] == [0, 1]

    def test_empty_indices_removes_region(self, web, selection_root):
        web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, 1],
        })
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [],
        })
        j = r.get_json()
        assert "L-electrode" not in j["regions"]

    def test_rule_persisted_in_selection_rules(
        self, web, selection_root,
    ):
        rule = {"op": "by_element", "elements": ["C"]}
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "bridge",
            "indices":        [4, 5, 6],
            "rule":           rule,
        })
        j = r.get_json()
        assert j["selection_rules"]["bridge"]["op"] == "by_element"
        assert j["selection_rules"]["bridge"]["elements"] == ["C"]

    def test_out_of_range_index_returns_400(self, web, selection_root):
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [99],
        })
        assert r.status_code == 400

    def test_missing_target_returns_400(self, web, selection_root):
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "indices":        [0],
        })
        assert r.status_code == 400

    def test_float_indices_rejected_not_truncated(
        self, web, selection_root,
    ):
        """``int(1.5) -> 1`` is a silent-truncation bug.  Float
        indices in the request body must be rejected with a 400, not
        quietly turned into integers -- otherwise a JS-side bug that
        sent floats would land bogus indices in the sidecar.

        Bool also slips through ``isinstance(x, int)`` in Python
        (``True == 1``), so the endpoint also rejects bool indices
        explicitly.
        """
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, 1.5, 2],
        })
        assert r.status_code == 400, (
            f"float index should be rejected; got {r.status_code} "
            f"with body {r.get_json()!r}"
        )
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, True, 2],
        })
        assert r.status_code == 400, (
            f"bool index should be rejected; got {r.status_code} "
            f"with body {r.get_json()!r}"
        )
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        ["1", "2"],
        })
        assert r.status_code == 400, (
            "numeric string index should be rejected; "
            "got " + str(r.status_code)
        )

    def test_corrupt_sidecar_rejects_save_without_destroying_data(
        self, web, selection_root,
    ):
        """A corrupt sidecar carries user work the server can neither
        read nor safely overwrite.  Writing a fresh sidecar from the
        current save's target would silently destroy every OTHER
        region / frozen_atoms / rule the user had set previously.
        The endpoint must refuse the save (409 Conflict) so the user
        can rename / recover the file -- their action fails loudly
        instead of erasing data.

        Pinned against the 2026-05-20 regression where the endpoint
        silently caught MolstructJsonError, started with empty
        defaults, and overwrote the corrupt sidecar with a fresh one
        containing only the current target.
        """
        sidecar = self._sidecar_path(selection_root)
        sidecar.write_text("{not valid json")
        r = web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, 1, 2],
        })
        assert r.status_code == 409, (
            f"expected 409 Conflict on corrupt sidecar; got {r.status_code} "
            f"with body {r.get_json()!r}"
        )
        j = r.get_json()
        assert j["ok"] is False
        assert "sidecar" in j["error"].lower(), (
            f"error message should mention the sidecar; got {j['error']!r}"
        )
        # The corrupt file is left in place for the user to inspect
        # (NOT silently overwritten).
        assert sidecar.read_text() == "{not valid json"

    def test_multi_label_round_trip_via_atoms(
        self, web, selection_root,
    ):
        """Atoms endpoint surfaces ALL region labels an atom belongs
        to, not just the most recently assigned."""
        web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, 1, 2, 3],
        })
        web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "interface",
            "indices":        [3, 4],
        })
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        atoms = r.get_json()["atoms"]
        # Atom 3 should carry BOTH labels (order is insertion order):
        assert set(atoms[3]["regions"]) == {"L-electrode", "interface"}
        # Atoms 0-2 only L-electrode:
        assert atoms[0]["regions"] == ["L-electrode"]
        # Atom 4 only interface:
        assert atoms[4]["regions"] == ["interface"]

    def test_atoms_endpoint_reflects_saved_regions(
        self, web, selection_root,
    ):
        """Round-trip: save a region, then GET /api/selection/atoms
        and verify the atoms list carries the new label."""
        web.post("/api/selection/save", json={
            "structure_path": _path(selection_root),
            "target":         "L-electrode",
            "indices":        [0, 1, 2, 3],
        })
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        atoms = r.get_json()["atoms"]
        assert atoms[0]["regions"] == ["L-electrode"]
        assert atoms[4]["regions"] == []   # bridge atoms still bare


# --------------------------------------------------------------------- #
#  /api/selection/save  -- concurrency (task #148)                      #
# --------------------------------------------------------------------- #


class TestSaveConcurrency:
    """Integration test for the sidecar lock added 2026-06-02.

    Two concurrent saves on different region keys MUST both land
    in the sidecar -- the lock around the read-modify-write window
    in ``api_selection_save`` serialises them so the second writer
    sees the first writer's update.  Without the lock, the
    second-arriving response clobbers the first.
    """

    def test_two_parallel_saves_both_persist(
            self, selection_root, tmp_path):
        """Spin up a real WSGI server (the Flask test_client is not
        designed for true concurrent use across threads; using a
        werkzeug server matches the production wire-up).  Fire two
        POSTs to /api/selection/save in parallel from worker threads,
        each tagging a different region.  After both complete, the
        sidecar must hold both."""
        pytest.importorskip("flask")
        import threading
        import urllib.request
        import urllib.parse
        import json as _json
        from werkzeug.serving import make_server
        from molbuilder.web.app import create_app

        app = create_app(config={})
        server = make_server("127.0.0.1", 0, app, threaded=True)
        port = server.server_port
        srv_thread = threading.Thread(
            target=server.serve_forever, daemon=True)
        srv_thread.start()

        try:
            base = f"http://127.0.0.1:{port}"

            def _save(region_name, indices, out):
                """Worker: POST one save with a small artificial
                slow-down to widen the race window between the
                client-side read and write so an unsynchronised
                server-side RMW would reliably lose one of them."""
                body = _json.dumps({
                    "structure_path": _path(selection_root),
                    "target":         region_name,
                    "indices":        indices,
                }).encode("utf-8")
                req = urllib.request.Request(
                    base + "/api/selection/save",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    out.append(_json.loads(r.read()))

            results_a, results_b = [], []
            t1 = threading.Thread(target=_save,
                                  args=("L-electrode", [0, 1], results_a))
            t2 = threading.Thread(target=_save,
                                  args=("R-electrode", [7, 8], results_b))
            t1.start()
            t2.start()
            t1.join(timeout=15)
            t2.join(timeout=15)

            assert not t1.is_alive(), "save #1 still running"
            assert not t2.is_alive(), "save #2 still running"
            assert results_a and results_a[0].get("ok") is True, (
                f"save #1 response: {results_a!r}")
            assert results_b and results_b[0].get("ok") is True, (
                f"save #2 response: {results_b!r}")

            # The DEFINITIVE check: read the sidecar back via the
            # atoms endpoint and verify BOTH regions landed.  If the
            # lock wasn't in place, only the later writer's region
            # would survive on disk and atoms[0]['regions'] would NOT
            # include "L-electrode".
            req = urllib.request.Request(
                base + "/api/selection/atoms",
                data=_json.dumps({
                    "structure_path": _path(selection_root),
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                body = _json.loads(r.read())
            atoms = body["atoms"]

            assert "L-electrode" in atoms[0]["regions"], (
                f"L-electrode lost from sidecar after concurrent save; "
                f"atoms[0]={atoms[0]!r}"
            )
            assert "R-electrode" in atoms[7]["regions"], (
                f"R-electrode lost from sidecar after concurrent save; "
                f"atoms[7]={atoms[7]!r}"
            )
        finally:
            server.shutdown()
            srv_thread.join(timeout=5)


class TestStateless:
    def test_same_request_yields_same_response(self, web, selection_root):
        """No server-side state between identical requests."""
        body = {
            "structure_path": _path(selection_root),
            "rule": {"op": "by_element", "elements": ["Au"]},
        }
        first  = web.post("/api/selection/eval", json=body).get_json()
        second = web.post("/api/selection/eval", json=body).get_json()
        assert first == second
