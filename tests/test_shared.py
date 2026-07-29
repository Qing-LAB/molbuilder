"""Tests for ``molbuilder/web/blueprints/_shared.py``.

The unifier helpers (``atoms_list``, ``structure_to_dict``,
``ok_structure_response``, and the 2026-06-07 ``workspace_payload``
addition) are the single source of truth for "how a Structure
becomes JSON" across every Flask blueprint.  Pin the canonical
shape here so a future refactor that touches one endpoint can't
silently drift the schema.

Workspace-state protocol:
``docs/web/workspace.md`` § 4.4 + § 5.

Migration phase tracking: this file is gated against Phase 1 of
the workspace-state migration plan (§ 6).  Phase 2 will pin the
endpoint-level migration (build/load + build/molecule + modify/*
emit ``workspace_payload`` directly).  (Phase 3's ``selection_remap``
was later retired: the client clears the selection on any atom-count
change, molview-module.md §19.3.2.)  Later phases extend this file
rather than replacing it.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.web.blueprints._shared import (
    atoms_list,
    ok_structure_response,
    structure_to_dict,
    workspace_payload,
)


def _h2o() -> Structure:
    """Plain water Structure — three atoms, no PDB metadata, no regions."""
    return Structure(
        elements  = ["O", "H", "H"],
        positions = np.array([
            [0.000,  0.000, 0.000],
            [0.957,  0.000, 0.000],
            [-0.239, 0.927, 0.000],
        ]),
        title="water",
    )


def _h2o_with_regions() -> Structure:
    """Water + an L-electrode region tag on the O + a frozen H."""
    s = _h2o()
    return Structure(
        elements      = list(s.elements),
        positions     = s.positions.copy(),
        atom_names    = list(s.atom_names),
        residue_ids   = list(s.residue_ids),
        residue_names = list(s.residue_names),
        chain_ids     = list(s.chain_ids),
        regions       = {"L-electrode": [0]},
        frozen_atoms  = [1],
        title         = s.title,
    )


# --------------------------------------------------------------------- #
#  workspace_payload — Phase 1 canonical-shape pins                     #
# --------------------------------------------------------------------- #


class TestWorkspacePayloadCanonicalKeys:
    """Pin the exact key set + types of the canonical shape.

    Any future field addition must update both the protocol doc
    (§ 4.4) AND this test class.  Removals require the migration
    table in protocol § 6 to advance to a phase that retires the
    field.
    """

    def test_has_canonical_keys(self):
        payload = workspace_payload(_h2o())
        canonical = {
            "text", "source_format", "title", "n_atoms",
            "atoms", "lattice", "issues", "extra",
        }
        assert set(payload.keys()) == canonical, (
            f"workspace_payload key set drifted from canonical;\n"
            f"  expected: {sorted(canonical)}\n"
            f"  got:      {sorted(payload.keys())}\n"
            f"Update docs/web/workspace.md "
            f"before extending."
        )

    def test_text_is_xyz_bytes(self):
        payload = workspace_payload(_h2o())
        assert isinstance(payload["text"], str)
        # XYZ header is the atom count line.
        first = payload["text"].splitlines()[0].strip()
        assert first == "3"

    def test_source_format_defaults_to_xyz(self):
        """Phase 1 default — PDB callers override via extra (Phase 2)."""
        assert workspace_payload(_h2o())["source_format"] == "xyz"

    def test_title_falls_back_to_empty_string_not_none(self):
        s = _h2o()
        no_title = Structure(
            elements=list(s.elements), positions=s.positions.copy())
        assert workspace_payload(no_title)["title"] == ""

    def test_n_atoms_matches_structure(self):
        assert workspace_payload(_h2o())["n_atoms"] == 3

    def test_atoms_is_a_list_of_dicts_with_per_atom_shape(self):
        payload = workspace_payload(_h2o())
        atoms = payload["atoms"]
        assert isinstance(atoms, list)
        assert len(atoms) == 3
        for row in atoms:
            assert "index" in row
            assert "element" in row
            assert "regions" in row
            assert "is_frozen" in row

    def test_atoms_carry_coordinates_on_the_atom(self):
        """workspace-contract.md §1.2.1 (MANDATORY): each atom row carries its own
        x/y/z as numbers -- the atom is the geometric truth, not a re-parsed xyz
        string."""
        s = _h2o()
        atoms = workspace_payload(s)["atoms"]
        for i, row in enumerate(atoms):
            assert isinstance(row["x"], float)
            assert isinstance(row["y"], float)
            assert isinstance(row["z"], float)
            assert (row["x"], row["y"], row["z"]) == (
                float(s.positions[i][0]), float(s.positions[i][1]),
                float(s.positions[i][2]))

    def test_atoms_matches_atoms_list_helper(self):
        """One source of truth — workspace_payload routes through
        atoms_list rather than building per-atom rows itself."""
        s = _h2o_with_regions()
        assert workspace_payload(s)["atoms"] == atoms_list(s)

    def test_lattice_is_none_for_non_periodic_structure(self):
        """Structure carries no lattice today; helper returns None.

        When Structure grows a periodic-cell field (currently lives
        on Frame / Trajectory instead), this assertion becomes a
        positive shape check.  Pinning the current contract makes
        the future extension visible in the diff."""
        assert workspace_payload(_h2o())["lattice"] is None

    def test_issues_array_is_present_and_a_list(self):
        """validate_geometry runs at serialisation time; the array
        is always present (possibly empty) so consumers don't have
        to feature-detect."""
        issues = workspace_payload(_h2o())["issues"]
        assert isinstance(issues, list)

    def test_extra_defaults_to_empty_dict_not_none(self):
        """``extra`` is the open-set bag for endpoint-specific
        keys.  Defaulting to ``{}`` (not ``None``) keeps consumers
        from feature-detecting on the wrong shape."""
        assert workspace_payload(_h2o())["extra"] == {}

    def test_extra_passes_through_keyword_dict(self):
        payload = workspace_payload(
            _h2o(),
            extra={"backend_used": "rdkit",
                   "add_hydrogens_mode": "auto"},
        )
        assert payload["extra"] == {
            "backend_used": "rdkit",
            "add_hydrogens_mode": "auto",
        }

    def test_extra_is_a_copy_not_an_alias(self):
        """A caller mutating the dict it passed in must not bleed
        into the payload (defensive copy at the boundary)."""
        ext = {"backend_used": "rdkit"}
        payload = workspace_payload(_h2o(), extra=ext)
        ext["backend_used"] = "amber"   # mutate caller-side
        assert payload["extra"]["backend_used"] == "rdkit"


class TestWorkspacePayloadRegionsAndFrozen:
    """The atoms list inside the payload must surface regions +
    is_frozen the same way the selection store expects (atom-selection
    protocol § 2 / Atom schema)."""

    def test_region_tag_lands_on_the_right_atom(self):
        atoms = workspace_payload(_h2o_with_regions())["atoms"]
        assert atoms[0]["regions"] == ["L-electrode"]
        assert atoms[1]["regions"] == []
        assert atoms[2]["regions"] == []

    def test_frozen_atom_is_marked(self):
        atoms = workspace_payload(_h2o_with_regions())["atoms"]
        assert atoms[0]["is_frozen"] is False
        assert atoms[1]["is_frozen"] is True
        assert atoms[2]["is_frozen"] is False


# --------------------------------------------------------------------- #
#  structure_to_dict — legacy shim contract                             #
# --------------------------------------------------------------------- #


class TestStructureToDictLegacyShim:
    """Phase 1 of the migration retains structure_to_dict as a
    legacy shim — same wire shape as before, now routed through
    workspace_payload internally.  Pin that:

      1. Existing legacy keys still emit (no breakage for the
         modify-tab front-end).
      2. The canonical keys are ALSO emitted, so a Phase-2-ready
         consumer can read them today without waiting for the
         endpoint-level migration.
      3. ``atoms`` is the same list workspace_payload exposes
         (single source of truth across both helpers).
    """

    def test_emits_every_legacy_key(self):
        d = structure_to_dict(_h2o())
        legacy = {
            "xyz", "elements", "atom_names", "residue_ids",
            "residue_names", "chain_ids", "n_atoms", "n_residues",
            "title", "atoms",
        }
        missing = legacy - set(d.keys())
        assert not missing, (
            f"structure_to_dict dropped legacy keys "
            f"the modify-tab front-end reads: {sorted(missing)}"
        )

    def test_emits_canonical_keys_alongside_legacy(self):
        d = structure_to_dict(_h2o())
        canonical_subset = {
            "text", "source_format", "lattice",
        }
        missing = canonical_subset - set(d.keys())
        assert not missing, (
            f"structure_to_dict is missing canonical keys; "
            f"Phase-2-ready consumers won't be able to read them: "
            f"{sorted(missing)}"
        )

    def test_xyz_alias_equals_canonical_text(self):
        d = structure_to_dict(_h2o())
        assert d["xyz"] == d["text"]

    def test_atoms_matches_workspace_payload_atoms(self):
        s = _h2o_with_regions()
        assert structure_to_dict(s)["atoms"] \
            == workspace_payload(s)["atoms"]


# --------------------------------------------------------------------- #
#  Phase 2: structure_to_dict accepts extra + threads it everywhere     #
# --------------------------------------------------------------------- #


class TestStructureToDictExtraThreading:
    """Phase 2 (2026-06-07) — ``structure_to_dict`` accepts an
    optional ``extra`` dict for endpoint-specific keys.  The
    helper threads each key BOTH into the top level of the
    returned dict (so existing JS consumers reading off the
    response root keep working) AND into the canonical ``extra``
    sub-dict (where the Phase 4+ workspace dispatcher reads
    them).
    """

    def test_extras_appear_at_top_level(self):
        d = structure_to_dict(_h2o(), extra={
            "pdb":     "PDB\n",
            "summary": "1 mol",
        })
        assert d["pdb"]     == "PDB\n"
        assert d["summary"] == "1 mol"

    def test_extras_also_appear_in_canonical_extra_subdict(self):
        d = structure_to_dict(_h2o(), extra={
            "backend_used":      "rdkit",
            "add_hydrogens_mode": "auto",
        })
        assert d["extra"] == {
            "backend_used":      "rdkit",
            "add_hydrogens_mode": "auto",
        }

    def test_extra_can_override_canonical_source_format(self):
        """The /api/build/load endpoint sets source_format=fmt from
        the parsed shape.  The helper must let that override the
        canonical XYZ default at both the top level and inside
        the canonical extra dict."""
        d = structure_to_dict(_h2o(), extra={"source_format": "pdb"})
        assert d["source_format"]      == "pdb"
        assert d["extra"]["source_format"] == "pdb"

    def test_no_extra_yields_empty_extra_dict_not_none(self):
        d = structure_to_dict(_h2o())
        assert d["extra"] == {}

    def test_issues_array_is_present_when_no_extra(self):
        """structure_to_dict's issues array is sourced from
        workspace_payload — same validate-pass, same shape."""
        d = structure_to_dict(_h2o())
        assert "issues" in d
        assert isinstance(d["issues"], list)


# --------------------------------------------------------------------- #
#  ok_structure_response — modify endpoints + issues array              #
# --------------------------------------------------------------------- #


class TestOkStructureResponse:
    """Pin the response Flask emits for ``/api/modify/*``.

    ``ok: True`` envelope, every legacy + canonical key from
    structure_to_dict, plus a top-level ``issues`` array — the
    shape every modifier-op caller depends on.
    """

    def _payload(self, struct: Structure):
        from flask import Flask
        app = Flask(__name__)
        with app.app_context():
            resp = ok_structure_response(struct)
            return resp.get_json()

    def test_ok_envelope(self):
        body = self._payload(_h2o())
        assert body["ok"] is True

    def test_carries_canonical_shape_keys(self):
        body = self._payload(_h2o())
        for k in ("text", "source_format", "title",
                  "n_atoms", "atoms", "lattice", "issues"):
            assert k in body, f"missing canonical key {k!r}"

    def test_carries_legacy_shape_keys(self):
        body = self._payload(_h2o())
        for k in ("xyz", "elements", "atom_names", "residue_ids",
                  "residue_names", "chain_ids", "n_residues"):
            assert k in body, f"missing legacy key {k!r}"

    def test_issues_array_present(self):
        body = self._payload(_h2o())
        assert isinstance(body["issues"], list)

    def test_phase_2_extras_at_top_level_and_in_extra_subdict(self):
        """Phase 2 contract — endpoint extras land at BOTH the
        response root AND the canonical ``extra`` sub-dict."""
        from flask import Flask
        app = Flask(__name__)
        with app.app_context():
            resp = ok_structure_response(_h2o(), extra={
                "pdb":               "PDB...\n",
                "summary":           "water (1 mol)",
                "backend_used":      "rdkit",
                "add_hydrogens_mode": "auto",
            })
            body = resp.get_json()
        for k in ("pdb", "summary", "backend_used",
                  "add_hydrogens_mode"):
            assert k in body, (
                f"{k!r} missing from top level — Phase 1-3 client "
                f"back-compat broken"
            )
            assert k in body["extra"], (
                f"{k!r} missing from extra sub-dict — Phase 4+ "
                f"workspace-dispatcher reader broken"
            )


# --------------------------------------------------------------------- #
#  Sanity: every helper that returns a Structure shape carries atoms    #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  apply_companion_labels_if_present + companion-wins-over-sidecar      #
#  (bundle-contract.md § 4.2 + § 5.3)                                   #
# --------------------------------------------------------------------- #


def _h2o_xyz_text():
    return "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n"


def _h2o_structure_path(tmp_path):
    """Write a small XYZ to tmp_path; return its absolute path."""
    p = tmp_path / "water.xyz"
    p.write_text(_h2o_xyz_text())
    return p


def _fdf_with_atom_metadata(regions, frozen, n_atoms_total):
    """Synthesize a minimal .fdf body carrying an ATOM-METADATA
    block via the canonical emitter so the test text always tracks
    the emitter's format."""
    from molbuilder import script_emit as sc
    block = sc.emit_atom_metadata(
        regions=regions,
        frozen_atoms=frozen,
        n_atoms_total=n_atoms_total,
    )
    assert block is not None
    return f"SystemLabel test\nBlockSize 64\n\n{block}\n"


def test_apply_companion_labels_returns_none_when_no_companion(tmp_path):
    """No .fdf / .py next to the .xyz -> companion path is a no-op."""
    from molbuilder.structure import Structure
    from molbuilder.web.blueprints._shared import apply_companion_labels_if_present
    p = _h2o_structure_path(tmp_path)
    struct = Structure.from_xyz(_h2o_xyz_text())
    assert apply_companion_labels_if_present(struct, p) is None
    assert struct.regions == {}
    assert struct.frozen_atoms == []


def test_apply_companion_labels_picks_up_sibling_fdf(tmp_path):
    """.fdf with ATOM-METADATA next to the .xyz applies its labels."""
    from molbuilder.structure import Structure
    from molbuilder.web.blueprints._shared import apply_companion_labels_if_present
    p = _h2o_structure_path(tmp_path)
    (tmp_path / "water.fdf").write_text(
        _fdf_with_atom_metadata(
            regions={"L-electrode": [0, 1]},
            frozen=[2],
            n_atoms_total=3,
        )
    )
    struct = Structure.from_xyz(_h2o_xyz_text())
    marker = apply_companion_labels_if_present(struct, p)
    assert marker == "applied:fdf"
    assert struct.regions == {"L-electrode": [0, 1]}
    assert struct.frozen_atoms == [2]


def test_apply_companion_labels_picks_up_sibling_py_when_no_fdf(tmp_path):
    """Falls through to .py when .fdf absent."""
    from molbuilder.structure import Structure
    from molbuilder.web.blueprints._shared import apply_companion_labels_if_present
    p = _h2o_structure_path(tmp_path)
    (tmp_path / "water.py").write_text(
        _fdf_with_atom_metadata(
            regions={"R-electrode": [1, 2]},
            frozen=[],
            n_atoms_total=3,
        )
    )
    struct = Structure.from_xyz(_h2o_xyz_text())
    marker = apply_companion_labels_if_present(struct, p)
    assert marker == "applied:py"
    assert struct.regions == {"R-electrode": [1, 2]}


def test_apply_companion_labels_fdf_wins_when_both_present(tmp_path):
    """When both .fdf and .py exist, .fdf is checked first
    (transport/SIESTA being the canonical continuation workflow)."""
    from molbuilder.structure import Structure
    from molbuilder.web.blueprints._shared import apply_companion_labels_if_present
    p = _h2o_structure_path(tmp_path)
    (tmp_path / "water.fdf").write_text(
        _fdf_with_atom_metadata(regions={"from_fdf": [0]}, frozen=[],
                                n_atoms_total=3)
    )
    (tmp_path / "water.py").write_text(
        _fdf_with_atom_metadata(regions={"from_py": [1]}, frozen=[],
                                n_atoms_total=3)
    )
    struct = Structure.from_xyz(_h2o_xyz_text())
    marker = apply_companion_labels_if_present(struct, p)
    assert marker == "applied:fdf"
    assert struct.regions == {"from_fdf": [0]}
    assert "from_py" not in struct.regions


def test_apply_sidecar_if_possible_companion_wins_over_sidecar(tmp_path, monkeypatch):
    """End-to-end of the priority rule (bundle-contract.md § 4.2):
    when a companion .fdf + a .molstruct.json BOTH sit next to the
    .xyz, the .fdf is the authoritative label source.  The sidecar's
    regions are NOT applied on top."""
    import hashlib, json
    from molbuilder.structure import Structure
    from molbuilder.web.blueprints._shared import apply_sidecar_if_possible
    from molbuilder import diagnostics

    # Pin tmp_path as a picker root so apply_sidecar_if_possible's
    # _resolve_within_roots accepts the .xyz path.
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)

    xyz_text = _h2o_xyz_text()
    xyz = tmp_path / "water.xyz"
    xyz.write_text(xyz_text)

    # .molstruct.json carries one region label.
    sidecar = tmp_path / "water.molstruct.json"
    sidecar.write_text(json.dumps({
        "schema_version": 3,
        "n_atoms_total":  3,
        "structure_hash": hashlib.sha256(xyz_text.encode("utf-8")).hexdigest(),
        "frozen_atoms":   [],
        "regions":        {"from_sidecar": [0]},
        "created_by":     "test",
        "created_at":     "2026-06-17T00:00:00Z",
    }))

    # Companion .fdf carries a DIFFERENT region label.
    (tmp_path / "water.fdf").write_text(
        _fdf_with_atom_metadata(
            regions={"from_fdf": [1, 2]},
            frozen=[1],
            n_atoms_total=3,
        )
    )

    struct = Structure.from_xyz(xyz_text)
    notice = apply_sidecar_if_possible(struct, str(xyz))
    assert notice is None
    # In-body wins: the companion .fdf's labels are present;
    # the sidecar's are NOT layered in or stacked.
    assert struct.regions == {"from_fdf": [1, 2]}
    assert struct.frozen_atoms == [1]
    assert "from_sidecar" not in struct.regions


def test_apply_sidecar_if_possible_falls_through_when_no_companion(tmp_path, monkeypatch):
    """Smoke check: when only the .molstruct.json exists (no
    companion .fdf/.py), apply_sidecar_if_possible still applies
    the sidecar as before.  Pins backward compatibility."""
    import hashlib, json
    from molbuilder.structure import Structure
    from molbuilder.web.blueprints._shared import apply_sidecar_if_possible
    from molbuilder import diagnostics

    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)

    xyz_text = _h2o_xyz_text()
    xyz = tmp_path / "water.xyz"
    xyz.write_text(xyz_text)
    sidecar = tmp_path / "water.molstruct.json"
    sidecar.write_text(json.dumps({
        "schema_version": 3,
        "n_atoms_total":  3,
        "structure_hash": hashlib.sha256(xyz_text.encode("utf-8")).hexdigest(),
        "frozen_atoms":   [0],
        "regions":        {"from_sidecar": [0, 1]},
        "created_by":     "test",
        "created_at":     "2026-06-17T00:00:00Z",
    }))

    struct = Structure.from_xyz(xyz_text)
    assert apply_sidecar_if_possible(struct, str(xyz)) is None
    assert struct.regions == {"from_sidecar": [0, 1]}
    assert struct.frozen_atoms == [0]


def test_every_helper_carries_atoms_for_three_atom_water():
    """Regression for the 2026-06-07 audit's root cause: the
    ``atoms`` key was missing from ``/api/build/load`` because the
    endpoint had its own hand-rolled jsonify that didn't route
    through the helper.  This test pins that every helper in
    _shared keeps the canonical atoms list present and accurate."""
    s = _h2o()
    wp = workspace_payload(s)["atoms"]
    sd = structure_to_dict(s)["atoms"]
    al = atoms_list(s)
    assert len(wp) == 3
    assert len(sd) == 3
    assert len(al) == 3
    assert wp == sd == al


# --------------------------------------------------------------------- #
#  Annotations round-trip through the modify pipeline (§19.3.2)          #
# --------------------------------------------------------------------- #


def test_annotations_survive_a_delete_op_index_aligned():
    """§19.3.2 completeness: v4 per-atom annotation channels ride the modify round-trip
    (client body -> apply_labels_to_struct -> op reindexes -> structure_to_dict) so a
    delete keeps them ALIGNED to the surviving atoms -- the field the pipeline used to drop."""
    from molbuilder.web.blueprints._shared import (
        struct_from_body, apply_labels_to_struct)
    from molbuilder.modify import delete_atoms
    body = {
        "xyz": "4\ndiag\nC 0 0 0\nC 1 1 0\nC 2 2 0\nC 3 3 0\n",
        "frozen_atoms": [], "regions": {},
        "annotations": {"spin": {"kind": "tag", "data": [2, 3]}},
    }
    struct = struct_from_body(body)
    assert apply_labels_to_struct(struct, body) is None      # applied, no notice
    assert struct.annotations["spin"].data == [2, 3]         # read off the body
    # Delete atom 1: old [0,2,3] -> new [0,1,2]; the tag on old {2,3} remaps to {1,2}.
    out = structure_to_dict(delete_atoms(struct, [1]))
    assert out["n_atoms"] == 3
    spin = out["annotations"]["spin"]
    assert spin["kind"] == "tag" and spin["data"] == [1, 2], out["annotations"]


def test_annotations_out_of_range_index_is_rejected():
    """apply_labels_to_struct validates channel indices (like regions/frozen) -- an
    out-of-range annotation index returns a user-facing notice, never a crash."""
    from molbuilder.web.blueprints._shared import (
        struct_from_body, apply_labels_to_struct)
    body = {
        "xyz": "2\nx\nH 0 0 0\nH 1 0 0\n",
        "frozen_atoms": [], "regions": {},
        "annotations": {"bad": {"kind": "tag", "data": [9]}},   # index 9 on a 2-atom struct
    }
    struct = struct_from_body(body)
    notice = apply_labels_to_struct(struct, body)
    assert notice is not None and "annotations" in notice and "9" in notice
