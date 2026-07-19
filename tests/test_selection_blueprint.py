"""Tests for /api/selection/eval (L2 of
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


def _seed_sidecar(root, *, n_atoms, regions=None, frozen=None,
                  name="junction.xyz"):
    """Write a ``.molstruct.json`` sidecar next to ``name`` DIRECTLY via
    the codec.  This is the seed path formerly done with a
    ``POST /api/selection/save-sidecar`` before that endpoint was removed
    (the save is now the projects.parser door → ``/api/files/write``).  ``save-sidecar``
    was REPLACE-all, so a single ``to_dict`` mirrors one endpoint call."""
    from molbuilder.sidecars import molstruct as _msj
    xyz = root / name
    payload = _msj.to_dict(
        n_atoms_total=n_atoms,
        structure_hash=_msj.sha256_of_file(xyz),
        regions=regions or {},
        frozen_atoms=frozen or [],
    )
    _msj.save(_msj.sidecar_path_for(xyz), payload)


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

    def test_eval_against_in_memory_atoms_reflects_unsaved_labels(self, web):
        """A5b: eval against the workspace's IN-MEMORY atoms (the store), NOT disk,
        so a filter finds a label assigned in the panel but never saved.  Direct
        repro of the user bug: 'L-electrode' assigned in memory is found here even
        though disk has only 'BDT'.  No structure_path -- pure in-memory."""
        atoms = [
            {"index": 0, "element": "Au", "labels": ["L-electrode"], "is_frozen": True},
            {"index": 1, "element": "Au", "labels": ["L-electrode"], "is_frozen": True},
            {"index": 2, "element": "S",  "labels": ["BDT"], "is_frozen": False},
        ]
        r = web.post("/api/selection/eval", json={
            "atoms": atoms,
            "rule": {"op": "by_region", "name": "L-electrode"},
        })
        assert r.status_code == 200, r.get_json()
        j = r.get_json()
        assert j["selected_indices"] == [0, 1]      # the in-memory-labelled Au
        assert j["n_atoms_total"] == 3

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
        _seed_sidecar(selection_root, n_atoms=11,
                      regions={"L-electrode": [0, 1, 2, 3]})
        # 2. Eval: ask the server to return atoms with that label.
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_region", "name": "L-electrode"},
        })
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["selected_indices"] == [0, 1, 2, 3]

    def test_by_region_frozen_atoms_resolves_via_synthetic(
            self, web, selection_root):
        """2026-06-12: the ``By label`` filter dropdown shows the
        ``frozen`` tag (from atoms' ``is_frozen`` flag) as an option.
        Picking it should resolve to the frozen-atoms set even though
        ``frozen_atoms`` is stored as a separate sidecar field, not
        as a region.

        The fix exposes ``struct.frozen_atoms`` as a synthetic region
        named ``frozen_atoms`` ONLY during rule resolution (eval +
        toggle endpoints).  The synthetic does NOT leak into
        ``/api/selection/atoms`` (no duplicate ``frozen_atoms`` tag
        on the per-atom regions list) and does NOT leak into the
        sidecar on save (the on-disk file's ``regions`` block stays
        user-defined).
        """
        # Save frozen_atoms via the whole-sidecar writer.
        _seed_sidecar(selection_root, n_atoms=11, frozen=[5, 6, 7])
        # Eval ``frozen_atoms`` via the standard by_region rule.
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_region", "name": "frozen_atoms"},
        })
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["selected_indices"] == [5, 6, 7]

    def test_by_region_frozen_does_not_leak_into_atoms_list(
            self, web, selection_root):
        """Companion of the test above: the synthetic ``frozen_atoms``
        region must NOT appear in ``/api/selection/atoms``'s per-atom
        ``regions`` field — otherwise frozen atoms would carry BOTH
        the (correct) ``is_frozen`` flag AND a (duplicate)
        ``frozen_atoms`` tag in the panel's atom rows.
        """
        _seed_sidecar(selection_root, n_atoms=11, frozen=[5, 6, 7])
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        atoms = r.get_json()["atoms"]
        assert atoms[5]["is_frozen"] is True
        assert atoms[6]["is_frozen"] is True
        assert "frozen_atoms" not in atoms[5]["regions"], (
            "synthetic frozen_atoms region must not appear in per-atom "
            "regions; got " + repr(atoms[5]["regions"])
        )
        assert "frozen_atoms" not in atoms[6]["regions"]



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

    def test_cell_from_sidecar_surfaces_in_atoms_response(
            self, web, selection_root):
        """Phase 1 (structure-periodicity.md): the sidecar's 3x3 `cell` is
        returned by /api/selection/atoms so the viewer can draw the unit-cell
        box + tile k-grid.  The viewer never parses -- the host reads it here."""
        import hashlib
        import json as _json
        xyz_bytes = (selection_root / "junction.xyz").read_bytes()
        struct_hash = hashlib.sha256(xyz_bytes).hexdigest()
        cell = [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 22.0]]
        (selection_root / "junction.molstruct.json").write_text(_json.dumps({
            "schema_version": 3,
            "n_atoms_total":  11,
            "structure_hash": struct_hash,
            "regions":        {},
            "frozen_atoms":   [],
            "cell":           cell,
            "selection_rules": {},
        }))
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        assert r.get_json()["cell"] == cell

    def test_periodicity_from_sidecar_for_path_based_load(
            self, web, selection_root):
        """structure-periodicity.md: /api/selection/atoms returns the full
        `periodicity` {cell, axis_kind, vacuum} so the Modify path-based load
        restores a reopened structure's periodicity (the .json sits next to the
        .xyz on the server).  A legacy `kgrid` key in the sidecar is ignored --
        k-grid is a sampling knob on SiestaConfig, not geometry."""
        import hashlib
        import json as _json
        xyz_bytes = (selection_root / "junction.xyz").read_bytes()
        struct_hash = hashlib.sha256(xyz_bytes).hexdigest()
        cell = [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 22.0]]
        (selection_root / "junction.molstruct.json").write_text(_json.dumps({
            "schema_version": 4, "n_atoms_total": 11,
            "structure_hash": struct_hash, "regions": {}, "frozen_atoms": [],
            "cell": cell, "axis_kind": ["periodic", "periodic", "transport"],
            "kgrid": [4, 4, 1], "selection_rules": {},   # legacy key -> ignored
        }))
        per = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        }).get_json()["periodicity"]
        assert per["cell"] == cell
        assert per["axis_kind"] == ["periodic", "periodic", "transport"]
        assert "kgrid" not in per   # k-grid is not part of the geometry payload

    def test_cell_null_without_sidecar(self, web, selection_root):
        """No sidecar next to the .xyz -> `cell` + `periodicity` null, not error."""
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        body = r.get_json()
        assert body["ok"] is True and body["cell"] is None
        assert body["periodicity"] is None

    def test_missing_path_returns_400(self, web):
        r = web.post("/api/selection/atoms", json={})
        assert r.status_code == 400

    def test_sidecar_ahead_of_xyz_loads_surviving_labels(
            self, web, selection_root):
        """2026-06-12 regression: sidecar/XYZ desync after a
        ``writeLabel`` succeeded but the subsequent Save failed
        (or the user closed the browser without saving).

        The sidecar's ``n_atoms_total`` is ahead of the XYZ's
        actual atom count; some region/frozen indices reference
        atoms that never persisted.  Before the fix,
        ``apply_to_structure``'s strict mismatch check raised,
        ``_load_structure`` silently swallowed it, and ALL labels
        disappeared — including the ones still in range for the
        XYZ.  After the fix, in-range indices apply normally;
        out-of-range indices are dropped silently (the orphan
        atoms they reference don't exist on disk).
        """
        import hashlib
        import json as _json
        xyz_bytes = (selection_root / "junction.xyz").read_bytes()
        struct_hash = hashlib.sha256(xyz_bytes).hexdigest()
        sidecar = selection_root / "junction.molstruct.json"
        # Sidecar claims 20 atoms but the on-disk XYZ has 11.
        # Index 5 + 10 are in-range, 12 + 19 are orphaned.
        sidecar.write_text(_json.dumps({
            "schema_version": 3,
            "n_atoms_total":  20,
            "structure_hash": struct_hash,
            "regions": {
                "bridge":      [5, 12],     # 12 orphaned
                "L-electrode": [0, 1, 19],  # 19 orphaned
                "ghosts":      [15, 19],    # all orphaned → drop region
            },
            "frozen_atoms":   [0, 10, 18],  # 18 orphaned
            "selection_rules": {},
        }))
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["n_atoms"] == 11   # XYZ's actual count
        atoms = body["atoms"]
        # In-range labels apply.
        assert atoms[0]["regions"] == ["L-electrode"]
        assert atoms[1]["regions"] == ["L-electrode"]
        assert atoms[5]["regions"] == ["bridge"]
        # Empty regions after filter ("ghosts" — all indices orphaned)
        # are dropped entirely: no atom carries it.
        for a in atoms:
            assert "ghosts" not in a["regions"]
        # In-range frozen flags apply, orphans dropped.
        assert atoms[0]["is_frozen"] is True
        assert atoms[10]["is_frozen"] is True
        assert atoms[5]["is_frozen"] is False  # not in frozen list

    def test_sidecar_with_all_orphan_indices_loads_clean(
            self, web, selection_root):
        """Edge case of the desync-tolerance: every region index is
        orphaned by an XYZ shrink.  Expectation: atoms load cleanly
        with no labels (instead of silent total failure).
        """
        import hashlib
        import json as _json
        xyz_bytes = (selection_root / "junction.xyz").read_bytes()
        struct_hash = hashlib.sha256(xyz_bytes).hexdigest()
        sidecar = selection_root / "junction.molstruct.json"
        sidecar.write_text(_json.dumps({
            "schema_version": 3,
            "n_atoms_total":  20,
            "structure_hash": struct_hash,
            "regions":        {"ghost": [11, 12, 13]},
            "frozen_atoms":   [15, 16],
            "selection_rules": {},
        }))
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        assert r.status_code == 200
        atoms = r.get_json()["atoms"]
        for a in atoms:
            assert a["regions"] == []
            assert a["is_frozen"] is False


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


# --------------------------------------------------------------------- #
#  Error-path coverage gaps (task #147)                                 #
#                                                                       #
#  Each endpoint has 3-5 error branches.  Pre-existing test classes    #
#  (TestEval, TestAtomsEndpoint)                                       #
#  cover the happy path + a few of the most common bad inputs.  This   #
#  class fills the gaps so every error branch in the blueprint has    #
#  at least one test pinning its status code + message shape.          #
# --------------------------------------------------------------------- #


class TestAtomsErrorPaths:
    """Error branches in ``api_selection_atoms``."""

    def test_path_traversal_rejected(self, web, selection_root):
        """``_resolve_within_roots`` rejects paths that escape the
        picker-root set with a 400 / 403, depending on the kind of
        failure.  /atoms must surface this -- the path-validation
        layer is shared with /eval, but each route is responsible
        for handling _PickerError."""
        r = web.post("/api/selection/atoms", json={
            "structure_path": "/etc/passwd",
        })
        assert r.status_code in (400, 403), (
            f"expected 4xx for path-traversal; got {r.status_code} "
            f"with body {r.get_json()!r}"
        )

    def test_non_structure_extension_rejected(
            self, web, selection_root):
        """A real file that exists but isn't a supported structure
        type must be refused, not parsed (parsing a .out file as XYZ
        would yield arbitrary garbage)."""
        bad = selection_root / "junk.out"
        bad.write_text("not a structure\n")
        r = web.post("/api/selection/atoms", json={
            "structure_path": str(bad),
        })
        assert r.status_code in (400, 415), (
            f"expected 4xx for unsupported extension; got "
            f"{r.status_code}"
        )

    def test_file_not_found_returns_404(self, web, selection_root):
        """A picker-root path to a file that doesn't exist must
        surface as 404 (the user / client gave a stale path)."""
        r = web.post("/api/selection/atoms", json={
            "structure_path": str(selection_root / "no-such.xyz"),
        })
        assert r.status_code == 404

    def test_structure_path_must_be_string(self, web):
        """A null or non-string ``structure_path`` is a 400, not a
        500 -- the type check runs before path resolution."""
        for bad in (None, 42, [1, 2, 3], {"x": 1}):
            r = web.post("/api/selection/atoms", json={
                "structure_path": bad,
            })
            assert r.status_code == 400, (
                f"got {r.status_code} for structure_path={bad!r}"
            )

    def test_pdb_metadata_propagates(self, selection_root, web):
        """PDB-derived atom_name / residue_name / chain_id fields
        appear in the response when the loaded structure carries
        them (covered indirectly elsewhere; this pins the explicit
        contract)."""
        pdb = selection_root / "tiny.pdb"
        pdb.write_text(
            "ATOM      1 CA   ALA A   1       0.000   0.000   0.000\n"
            "ATOM      2 CB   ALA A   1       1.000   0.000   0.000\n"
            "END\n"
        )
        r = web.post("/api/selection/atoms",
                     json={"structure_path": str(pdb)})
        body = r.get_json()
        assert body["n_atoms"] == 2
        a0 = body["atoms"][0]
        assert a0.get("residue_name") == "ALA"
        assert a0.get("chain_id")     == "A"


class TestEvalErrorPaths:
    """Eval branches not covered by ``TestEval``."""

    def test_file_not_found_returns_404(self, web, selection_root):
        r = web.post("/api/selection/eval", json={
            "structure_path": str(selection_root / "no-such.xyz"),
            "rule":           {"op": "by_element", "elements": ["Au"]},
        })
        assert r.status_code == 404

    def test_structure_path_must_be_string(self, web):
        for bad in (None, 42, [1, 2, 3]):
            r = web.post("/api/selection/eval", json={
                "structure_path": bad,
                "rule":           {"op": "by_element",
                                   "elements": ["Au"]},
            })
            assert r.status_code == 400


class TestCrossCutting:
    """Body-validation paths shared by every endpoint."""

    @pytest.mark.parametrize("endpoint", [
        "/api/selection/atoms",
        "/api/selection/eval",
    ])
    def test_non_json_body_returns_400(self, web, endpoint):
        """Hitting any endpoint without ``Content-Type:
        application/json`` returns a 400 with a clear message --
        not a 500 from a downstream parse failure."""
        r = web.post(endpoint, data="not json", content_type="text/plain")
        assert r.status_code == 400
        # The message must mention JSON so a client trying to debug
        # a malformed request has a starting point.
        body = r.get_json(silent=True) or {}
        assert "json" in (body.get("error", "")).lower()

    @pytest.mark.parametrize("endpoint", [
        "/api/selection/atoms",
        "/api/selection/eval",
    ])
    def test_top_level_not_object_returns_400(self, web, endpoint):
        """JSON body must be an OBJECT, not an array / string / null."""
        r = web.post(endpoint, json=[1, 2, 3])
        assert r.status_code == 400


class TestUniformEnvelope:
    """Audit task #187 (2026-06-02) -- pin the uniform ``{ok, ...}`` /
    ``{ok: false, error}`` envelope contract on every /api/selection/*
    endpoint.  Pre-audit the four endpoints returned bare bodies
    (``{n_atoms, atoms}`` / ``{selected_indices, count, ...}``) and the
    ``_bad_request`` helper returned ``{error}`` without ``ok: false``;
    JS branched on HTTP status and the drift was invisible.  These
    tests close the contract so a future bare-body return regresses
    loudly.
    """

    def test_atoms_success_has_ok_true(self, web, selection_root):
        r = web.post("/api/selection/atoms", json={
            "structure_path": _path(selection_root),
        })
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("ok") is True
        # Existing payload still present.
        assert body["n_atoms"] == 11
        assert len(body["atoms"]) == 11

    def test_eval_success_has_ok_true(self, web, selection_root):
        r = web.post("/api/selection/eval", json={
            "structure_path": _path(selection_root),
            "rule": {"op": "by_element", "elements": ["C"]},
        })
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("ok") is True
        assert body["selected_indices"] == [4, 5, 6]
        assert body["count"] == 3
        assert body["n_atoms_total"] == 11

    @pytest.mark.parametrize("endpoint", [
        "/api/selection/atoms",
        "/api/selection/eval",
    ])
    def test_error_envelope_has_ok_false(self, web, endpoint):
        """Every error path through ``_bad_request`` must include
        ``ok: false`` alongside ``error``."""
        # Both endpoints reject a non-JSON body with a 400.
        r = web.post(endpoint, data="not json", content_type="text/plain")
        assert r.status_code == 400
        body = r.get_json(silent=True) or {}
        assert body.get("ok") is False
        assert isinstance(body.get("error"), str) and body["error"]
