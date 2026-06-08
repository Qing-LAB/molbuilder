"""Tests for ``molbuilder/web/blueprints/_shared.py``.

The unifier helpers (``atoms_list``, ``structure_to_dict``,
``ok_structure_response``, and the 2026-06-07 ``workspace_payload``
addition) are the single source of truth for "how a Structure
becomes JSON" across every Flask blueprint.  Pin the canonical
shape here so a future refactor that touches one endpoint can't
silently drift the schema.

Workspace-state protocol:
``docs/protocols/workspace-state.md`` § 4.4 + § 5.

Migration phase tracking: this file is gated against Phase 1 of
the workspace-state migration plan (§ 6).  Phase 2 will pin the
endpoint-level migration (build/load + build/molecule + modify/*
emit ``workspace_payload`` directly) and Phase 3 adds
``selection_remap``.  Both extend this file rather than replacing
it.
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
            f"Update docs/protocols/workspace-state.md § 4.4 "
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


# --------------------------------------------------------------------- #
#  Sanity: every helper that returns a Structure shape carries atoms    #
# --------------------------------------------------------------------- #


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
