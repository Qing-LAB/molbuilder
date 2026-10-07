"""Validator-issue-to-card attachment contract (task #373).

Per docs/web/ui-contract.md Rule 2, validator findings
should attach to the workflow-group card whose fields they concern.
The attachment is computed from the Issue's ``where`` field (e.g.
``"config.mesh_cutoff"``) by looking up the corresponding dataclass
field's ``metadata["workflow_group"]``.

These tests pin:

  * ``resolve_workflow_group(where, cfg)`` returns the expected
    role for every config field that carries ``workflow_group``
    metadata.
  * Returns ``None`` for non-config wheres (geometry, cell, polymer)
    and for fields that lack the metadata (legacy untagged).
  * ``issues_to_json(issues, cfg=cfg)`` enriches each output dict
    with the resolved ``workflow_group`` key (only when the
    resolver returns a value — keep the wire schema lean).
"""

from __future__ import annotations

import pytest

from molbuilder.issues import Issue
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.web.blueprints._shared import (issues_to_json,
                                                resolve_workflow_group)


# --------------------------------------------------------------------- #
#  resolve_workflow_group                                                #
# --------------------------------------------------------------------- #


class TestResolveWorkflowGroup:

    def test_siesta_mesh_cutoff_resolves_to_stage(self):
        """mesh_cutoff carries ``workflow_group="stage"`` in
        SiestaConfig — validator findings on this field should
        attach to the Stage card."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.mesh_cutoff", cfg) == "stage"

    def test_siesta_spin_treatment_resolves_to_profile(self):
        """spin_treatment is a Run-profile decision."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.spin_treatment", cfg) == "profile"

    def test_siesta_max_scf_iter_resolves_to_budget(self):
        """max_scf_iter is a compute-budget knob."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.max_scf_iter", cfg) == "budget"

    def test_pyscf_method_resolves_to_profile(self):
        """PySCF method (DFT / HF) is profile-level."""
        cfg = PySCFConfig()
        assert resolve_workflow_group(
            "config.method", cfg) == "profile"

    def test_pyscf_scf_max_cycle_resolves_to_budget(self):
        cfg = PySCFConfig()
        assert resolve_workflow_group(
            "config.scf_max_cycle", cfg) == "budget"

    @pytest.mark.parametrize("field", [
        "scf_conv_tol", "geom_gmax", "geom_grms", "geom_dmax",
        "geom_drms", "geom_etol", "geom_max_steps",
    ])
    def test_pyscf_per_rung_knobs_resolve_to_stage(self, field):
        """The knobs a rung varies land in the Stage card.

        `stages.md` § 1.1a: the ladder lives in ``task.json`` and each rung
        is its own deck, so what the config carries is THIS rung's flat
        values, and it is each of them that must be tagged."""
        cfg = PySCFConfig()
        assert resolve_workflow_group(f"config.{field}", cfg) == "stage"

    # ---- Non-config wheres (geometry / cell / polymer) -------------- #

    @pytest.mark.parametrize("where", [
        "geometry.min_distance",
        "geometry.h_ratio",
        "geometry.dipole",
        "cell.determinant",
        "cell.volume",
        "cell.image_distance",
        "polymer.orientation",
    ])
    def test_non_config_wheres_resolve_to_none(self, where):
        """Structure / cell / polymer findings have no workflow-
        group binding; they render in the residual panel."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(where, cfg) is None

    def test_dotted_subfield_strips_to_root(self):
        """``config.net_charge.makov_payne`` (a sub-field tag used by
        SIESTA's Makov-Payne notice) should resolve via the root
        ``net_charge`` field's metadata."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.net_charge.makov_payne", cfg) == "profile"

    def test_psml_lib_per_element_subfield(self):
        """``config.psml_lib.Au`` (one-per-element pseudo coverage
        finding) resolves via root ``psml_lib``'s metadata.

        The VALUE (``setup``) is incidental;
        what this pins is that a per-element sub-path still resolves through
        its ROOT field, so a coverage finding about one element lands on the
        card holding the control -- wherever that control has been moved to.
        """
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.psml_lib.Au", cfg) == "setup"

    def test_empty_where_returns_none(self):
        cfg = SiestaConfig()
        assert resolve_workflow_group("", cfg) is None

    def test_non_dataclass_cfg_returns_none(self):
        """Defensive: a non-dataclass cfg (None, dict, etc.) should
        return None rather than crashing."""
        assert resolve_workflow_group(
            "config.mesh_cutoff", None) is None
        assert resolve_workflow_group(
            "config.mesh_cutoff", {"mesh_cutoff": 300}) is None

    def test_unknown_field_returns_none(self):
        """A config.<unknown> path resolves to None — defensive
        against typo'd wheres rather than a KeyError."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.no_such_field", cfg) is None


# --------------------------------------------------------------------- #
#  issues_to_json enrichment                                             #
# --------------------------------------------------------------------- #


class TestIssuesToJsonEnrichment:

    def test_enriches_each_dict_with_workflow_group(self):
        """When cfg is provided, every Issue with a config.<X>
        ``where`` that maps to a tagged field gets a
        ``workflow_group`` key in its output dict."""
        cfg = SiestaConfig()
        issues = [
            Issue("warn", "mesh_cutoff too low", "config.mesh_cutoff"),
            Issue("warn", "spin needs setting", "config.spin_treatment"),
            Issue("warn", "iteration cap low", "config.max_scf_iter"),
        ]
        out = issues_to_json(issues, cfg=cfg)
        assert [d.get("workflow_group") for d in out] == [
            "stage", "profile", "budget"]

    def test_omits_workflow_group_key_for_unmapped_issues(self):
        """Issues whose ``where`` doesn't resolve to a tagged config
        field should NOT carry a workflow_group key (keep the JSON
        wire lean; the frontend treats absent === None)."""
        cfg = SiestaConfig()
        issues = [
            Issue("warn", "atom too close", "geometry.min_distance"),
            Issue("warn", "cell too tight", "cell.volume"),
        ]
        out = issues_to_json(issues, cfg=cfg)
        assert out, "nothing serialized; the loop below checks nothing"
        for d in out:
            assert "workflow_group" not in d

    def test_no_cfg_means_no_enrichment(self):
        """When cfg is None, the serialiser doesn't try to resolve
        groups — it just emits severity / message / where."""
        issues = [Issue("warn", "mesh_cutoff", "config.mesh_cutoff")]
        out = issues_to_json(issues)
        assert "workflow_group" not in out[0]
        assert out[0] == {"severity": "warn",
                          "message": "mesh_cutoff",
                          "where":   "config.mesh_cutoff"}

    def test_issue_pretagged_group_wins_over_resolver(self):
        """An Issue may explicitly pre-tag its workflow_group at
        construction time (e.g. for a finding that doesn't have a
        clean field mapping).  When that field is set, the
        resolver doesn't override it."""
        cfg = SiestaConfig()
        # A geometry issue (resolver returns None) that the caller
        # decided to pin to the profile card.
        issues = [Issue("warn", "structural quirk",
                        "geometry.min_distance",
                        workflow_group="profile")]
        out = issues_to_json(issues, cfg=cfg)
        assert out[0]["workflow_group"] == "profile"
