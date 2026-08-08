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
  * End-to-end: ``validate(struct, cfg)`` produces Issues whose
    ``where`` fields all resolve to documented roles (or None for
    geometry).
"""

from __future__ import annotations

import dataclasses

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

    def test_siesta_spin_polarized_resolves_to_profile(self):
        """spin_polarized is a Run-profile decision."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.spin_polarized", cfg) == "profile"

    def test_siesta_max_scf_iter_resolves_to_budget(self):
        """max_scf_iter is a compute-budget knob."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.max_scf_iter", cfg) == "budget"

    def test_pyscf_method_resolves_to_profile(self):
        """PySCF method (RKS / UKS) is profile-level."""
        cfg = PySCFConfig()
        assert resolve_workflow_group(
            "config.method", cfg) == "profile"

    def test_pyscf_scf_max_cycle_resolves_to_budget(self):
        cfg = PySCFConfig()
        assert resolve_workflow_group(
            "config.scf_max_cycle", cfg) == "budget"

    def test_pyscf_stages_resolves_to_stage(self):
        """Post-#534 commit 4b the per-stage convergence ladder is the
        ``stages`` field (a stage-table widget) rather than the flat
        geom_conv_* scalars.  The workflow-group resolver tags it
        ``stage`` so it lands in the Stage card."""
        cfg = PySCFConfig()
        assert resolve_workflow_group(
            "config.stages", cfg) == "stage"

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
        finding) resolves via root ``psml_lib``'s metadata."""
        cfg = SiestaConfig()
        assert resolve_workflow_group(
            "config.psml_lib.Au", cfg) == "profile"

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
            Issue("warn", "spin needs setting", "config.spin_polarized"),
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
        for d in out:
            assert "workflow_group" not in d

    def test_no_cfg_means_no_enrichment(self):
        """When cfg is None, the serialiser doesn't try to resolve
        groups — it just emits severity / message / where (legacy
        behaviour preserved)."""
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


# --------------------------------------------------------------------- #
#  End-to-end: validate(struct, cfg) → issues with correct groups        #
# --------------------------------------------------------------------- #


class TestValidateEndToEndAttaches:
    """Every where-field the live validators emit must either map to
    a documented workflow_group or be a legitimately group-less
    (geometry / cell / polymer) finding.  Catches the regression
    where a new _check_ uses a where like ``config.fooble`` that no
    field's metadata covers."""

    # Wheres known to be group-less by design (not config.* fields):
    _GEOMETRIC_WHERES = {
        "geometry.min_distance", "geometry.h_ratio", "geometry.dipole",
        "cell.determinant", "cell.volume", "cell.image_distance",
        "polymer.orientation",
    }

    def test_every_siesta_check_emits_resolvable_where(self):
        """Sweep every ``where=`` string literal in
        ``molbuilder/validation/`` and verify each ``config.*``
        prefix resolves to a documented workflow_group (one of
        profile / stage / budget) under SiestaConfig."""
        import re
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        cfg = SiestaConfig()
        wheres: set[str] = set()
        for src in (root / "molbuilder/validation").rglob("*.py"):
            text = src.read_text(encoding="utf-8")
            wheres.update(re.findall(
                r'["\'](config\.[a-z_][a-z_0-9.]*)["\']', text))
        # Every collected config.* where must resolve under at
        # least one of the two engine configs.  (Some wheres are
        # SIESTA-only; some are PySCF-only.)
        pyscf = PySCFConfig()
        unresolved = []
        for where in sorted(wheres):
            if (resolve_workflow_group(where, cfg) is not None
                    or resolve_workflow_group(where, pyscf) is not None):
                continue
            # Allowed exceptions — wheres that are intentionally
            # group-less because the underlying concept isn't owned
            # by a single workflow-group card:
            #   * ``config`` — the "bad parameters" sentinel from
            #     the preflight endpoint when params don't parse
            #     (no specific field to highlight).
            #   * ``config.frozen_atoms`` — describes the user's
            #     sidecar-derived constraint set being absorbed but
            #     ignored by the engine.  The fix is in a different
            #     field (``relax_type`` for SIESTA, ``optimize`` /
            #     ``optimizer`` for PySCF), so attaching to one
            #     card would mis-direct the user.  Renders in the
            #     residual panel.
            if where in ("config", "config.frozen_atoms"):
                continue
            unresolved.append(where)
        assert not unresolved, (
            f"Validators emit ``where=`` strings that don't map to "
            f"any tagged dataclass field: {sorted(unresolved)}.  "
            f"Either tag the corresponding config field with "
            f"``workflow_group=...`` or document the where as "
            f"intentionally group-less.")


# ===================================================================== #
#  Integration completeness — a field is in the form, or it is not      #
# ===================================================================== #

class TestEveryExposedFieldIsTagged:
    """A field exposed in the form MUST carry a ``workflow_group``.

    User rule, 2026-08-07: *"all template fields should have a tag that
    either enriches the information/validation or gives very minimum brief
    information — if any field does not have that, someone added a new field
    and did not finish the integration to the data system."*

    Two pieces of metadata decide a field's place in the surface, and they
    answer different questions:

      ``section``         is this field in the form at all?  A field without
                          one is deliberately internal (``use_save_dm``,
                          ``species_order``, ``copy_psml``) and the form
                          never shows it.
      ``workflow_group``  WHICH card, and therefore where a finding about it
                          appears (``web/ui-contract.md`` Rule 2).

    **So they must move together.**  A field with a ``section`` and no
    ``workflow_group`` renders bare after the three cards and its findings
    fall to a residual panel instead of sitting beside the field they
    concern — a half-integrated field, which is exactly the state this test
    exists to catch.  The reverse, a tagged field with no ``section``, is a
    tag nothing can ever read.

    **It holds today, on both engines** — checked 2026-08-07, zero offenders
    in either direction.  It has simply never been written down or guarded,
    which is why a new field can quietly break it.
    """

    @pytest.mark.parametrize("cfg_cls", [SiestaConfig, PySCFConfig])
    def test_section_and_workflow_group_move_together(self, cfg_cls):
        exposed_untagged, tagged_hidden = [], []
        for f in dataclasses.fields(cfg_cls):
            has_section = bool(f.metadata.get("section"))
            has_group = bool(f.metadata.get("workflow_group"))
            if has_section and not has_group:
                exposed_untagged.append(f.name)
            if has_group and not has_section:
                tagged_hidden.append(f.name)

        assert not exposed_untagged, (
            f"{cfg_cls.__name__}: field(s) exposed in the form with no "
            f"workflow_group: {exposed_untagged}.\n"
            "The form will render them bare after the three cards, and any "
            "validation finding about them lands in the residual panel "
            "instead of beside the field.  Add a workflow_group "
            "(profile / stage / budget), or drop the `section` if the field "
            "is meant to stay internal."
        )
        assert not tagged_hidden, (
            f"{cfg_cls.__name__}: field(s) carrying a workflow_group but no "
            f"section: {tagged_hidden}.\n"
            "Nothing can read that tag — the form never renders the field.  "
            "Either expose it with a `section` or drop the tag."
        )
