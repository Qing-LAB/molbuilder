"""Tests for the StageSpec dataclass + PySCFConfig.stages field
(task #534 commit 1, data layer only).

The generator + form UI consume ``stages`` in commit-family follow-
ups; these tests pin the data-layer contract so the follow-ups have
a stable foundation:

  * StageSpec default values match the publication-guide tier 2
    (GAU/publishable) middle case.
  * ``_default_stages()`` returns 3 stages with the publication-
    guide loose/publishable/TIGHT defaults; stage 3 is opt-in.
  * ``validate_stages()`` flags empty list, all-disabled, duplicate
    names, bogus name chars, non-positive numeric knobs, non-int /
    non-positive ``max_steps``.
  * ``PySCFConfig().stages`` returns a fresh list per instance (no
    shared mutable default — the classic dataclass foot-gun).
  * The new field is INVISIBLE in the form schema (commit 1 is
    data-only; UI lands in commit 2).
"""
from __future__ import annotations

import pytest

from molbuilder.config.pyscf import (
    PySCFConfig,
    StageSpec,
    _default_stages,
    validate_stages,
)


# --------------------------------------------------------------------- #
#  StageSpec defaults                                                   #
# --------------------------------------------------------------------- #


def test_stagespec_defaults_match_publication_guide_gau():
    """Bare ``StageSpec()`` is the publication-guide GAU tier
    (publishable, tier 2 of the 3-stage default)."""
    s = StageSpec()
    assert s.name == "stage1"
    assert s.enabled is True
    assert s.conv_tol  == pytest.approx(1.0e-9)
    assert s.gmax      == pytest.approx(4.5e-4)
    assert s.grms      == pytest.approx(3.0e-4)
    assert s.dmax      == pytest.approx(1.8e-3)
    assert s.drms      == pytest.approx(1.2e-3)
    assert s.etol      == pytest.approx(1.0e-6)
    assert s.max_steps == 200


# --------------------------------------------------------------------- #
#  _default_stages — 3-stage publication-guide default                  #
# --------------------------------------------------------------------- #


def test_default_stages_returns_three_stages():
    stages = _default_stages()
    assert len(stages) == 3


def test_default_stages_stage1_loose_preopt_enabled():
    s1 = _default_stages()[0]
    assert s1.name == "stage1"
    assert s1.enabled is True
    assert s1.conv_tol  == pytest.approx(1.0e-7)
    assert s1.gmax      == pytest.approx(2.0e-3)
    assert s1.max_steps == 50


def test_default_stages_stage2_publishable_enabled():
    s2 = _default_stages()[1]
    assert s2.name == "stage2"
    assert s2.enabled is True
    assert s2.conv_tol  == pytest.approx(1.0e-9)
    assert s2.gmax      == pytest.approx(4.5e-4)
    assert s2.max_steps == 200


def test_default_stages_stage3_tight_disabled_by_default():
    """Publication-guide TIGHT tier is opt-in; most users tick 1+2,
    leave 3 off.  This pins that default so a future change is loud."""
    s3 = _default_stages()[2]
    assert s3.name == "stage3"
    assert s3.enabled is False
    assert s3.conv_tol  == pytest.approx(1.0e-10)
    assert s3.gmax      == pytest.approx(1.5e-5)
    assert s3.max_steps == 100


def test_default_stages_returns_fresh_list_per_call():
    """Mutating the returned list must NOT affect the next call —
    the classic dataclass-default-factory foot-gun."""
    a = _default_stages()
    a[0].enabled = False
    b = _default_stages()
    assert b[0].enabled is True


# --------------------------------------------------------------------- #
#  validate_stages — error branches                                     #
# --------------------------------------------------------------------- #


def test_validate_stages_empty_list_is_an_error():
    errs = validate_stages([])
    assert errs and "empty" in errs[0]


def test_validate_stages_all_disabled_is_an_error():
    stages = _default_stages()
    for s in stages:
        s.enabled = False
    errs = validate_stages(stages)
    assert any("no stage is enabled" in e for e in errs)


def test_validate_stages_default_is_valid():
    """Sanity: the publication-guide default must validate."""
    assert validate_stages(_default_stages()) == []


def test_validate_stages_duplicate_names_collide():
    stages = [StageSpec(name="same"), StageSpec(name="same")]
    errs = validate_stages(stages)
    assert any("duplicate" in e for e in errs)


def test_validate_stages_bad_name_charset_rejected():
    stages = [StageSpec(name="bad name with spaces")]
    errs = validate_stages(stages)
    assert any("[A-Za-z0-9_]+" in e for e in errs)


def test_validate_stages_negative_knob_rejected():
    s = StageSpec(name="stage_bad", gmax=-1.0)
    errs = validate_stages([s])
    assert any("gmax" in e and "positive" in e for e in errs)


def test_validate_stages_zero_max_steps_rejected():
    s = StageSpec(name="stage_zero", max_steps=0)
    errs = validate_stages([s])
    assert any("max_steps" in e for e in errs)


def test_validate_stages_non_int_max_steps_rejected():
    s = StageSpec(name="stage_float", max_steps=12.5)  # type: ignore[arg-type]
    errs = validate_stages([s])
    assert any("max_steps" in e for e in errs)


# --------------------------------------------------------------------- #
#  PySCFConfig.stages — wiring                                          #
# --------------------------------------------------------------------- #


def test_pyscfconfig_stages_default_is_publication_guide_three_stages():
    cfg = PySCFConfig()
    assert len(cfg.stages) == 3
    assert cfg.stages[0].name == "stage1"
    assert cfg.stages[1].name == "stage2"
    assert cfg.stages[2].name == "stage3"
    assert cfg.stages[2].enabled is False


def test_pyscfconfig_stages_is_per_instance_not_shared():
    """``default_factory`` (not ``default``) means each PySCFConfig
    gets its own list — mutating one's stages must not bleed into
    a second instance's."""
    a = PySCFConfig()
    a.stages[0].enabled = False
    b = PySCFConfig()
    assert b.stages[0].enabled is True


def test_pyscfconfig_stages_invisible_in_form_schema_until_commit_2():
    """The data-layer field carries no ``section`` metadata so the
    form-schema generator skips it.  Pins the commit-1 contract:
    the field exists in Python land + the CLI + the JSON envelope,
    but the web form doesn't render per-stage rows yet (commit 2)."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    schema = dataclass_to_form_schema(PySCFConfig, id_prefix="test")
    every_field = [
        f["name"]
        for section in schema["sections"]
        for f in section["fields"]
    ]
    assert "stages" not in every_field
