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
change, web/molview.md § 11.)  Later phases extend this file
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


    def test_atoms_carry_coordinates_on_the_atom(self):
        """web/workspace.md § 5 (MANDATORY): each atom row carries its own
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
#  (execution/handoff-bundle.md § 4 + § 5.3)                                   #
# --------------------------------------------------------------------- #


def _h2o_xyz_text():
    return "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n"


def _h2o_structure_path(tmp_path):
    """Write a small XYZ to tmp_path; return its absolute path."""
    p = tmp_path / "water.xyz"
    p.write_text(_h2o_xyz_text())
    return p



# (test_apply_companion_labels_returns_none_when_no_companion retired
#  2026-08-21 with its subject: the companion/sidecar-file family lost
#  its last production caller when the emitting doors moved to the
#  envelope -- C-shared.  Pattern B's re-homed check is pinned in
#  tests/validation/.)





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
    """§19.3.2 completeness: per-atom annotation channels ride the modify
    round-trip (envelope -> from_dict -> op reindexes -> structure_to_dict) so a
    delete keeps them ALIGNED to the surviving atoms -- the field the pipeline
    used to drop.

    The channels arrive inside the structure and `Structure.from_dict` applies
    them."""
    from molbuilder.web.blueprints._shared import struct_from_body
    from molbuilder.modify import delete_atoms
    body = {"structure": {
        "elements":  ["C", "C", "C", "C"],
        "positions": [[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0]],
        "metadata":  {"annotations": {"spin": {"kind": "tag",
                                               "data": [2, 3]}}},
    }}
    struct = struct_from_body(body)
    assert struct.annotations["spin"].data == [2, 3]         # read off the body
    # Delete atom 1: old [0,2,3] -> new [0,1,2]; the tag on old {2,3} remaps to {1,2}.
    out = structure_to_dict(delete_atoms(struct, [1]))
    assert out["n_atoms"] == 3
    spin = out["annotations"]["spin"]
    assert spin["kind"] == "tag" and spin["data"] == [1, 2], out["annotations"]


def test_annotations_out_of_range_index_is_refused():
    """An annotation channel naming an atom that isn't there is REFUSED, by the
    same validator that refuses a malformed Structure anywhere -- and the
    message names the channel and the index.

    The channels ride inside the structure, so a bad index cannot produce a
    half-built structure to warn about -- the structure is simply not built."""
    import pytest
    from molbuilder.web.blueprints._shared import struct_from_body
    body = {"structure": {
        "elements":  ["H", "H"],
        "positions": [[0, 0, 0], [1, 0, 0]],
        # index 9 on a 2-atom struct
        "metadata":  {"annotations": {"bad": {"kind": "tag", "data": [9]}}},
    }}
    with pytest.raises(ValueError) as exc:
        struct_from_body(body)
    assert "annotations" in str(exc.value) and "9" in str(exc.value)
