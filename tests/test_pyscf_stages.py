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


def test_pyscfconfig_stages_visible_in_form_schema():
    """Commit-2 contract flip: the ``stages`` field IS in the form
    schema now, with ``kind: "stage-table"`` so the JS renderer
    (commit 3) emits a per-stage row table.  Lives in the
    ``Compute & budget`` section (alongside the flat geom_conv_*
    knobs it'll replace in commit 4)."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    schema = dataclass_to_form_schema(PySCFConfig, id_prefix="test")
    found = None
    for section in schema["sections"]:
        for f in section["fields"]:
            if f["name"] == "stages":
                found = (section["name"], f)
                break
        if found:
            break
    assert found, "stages field is missing from PySCFConfig schema"
    section_name, f = found
    assert section_name == "Compute & budget"
    assert f["kind"] == "stage-table"
    assert f["workflow_group"] == "stage"
    assert f["id"] == "test-stages"
    assert f["tier"] == "advanced"


def test_stages_schema_carries_default_three_stage_list_of_dicts():
    """``default`` is the asdict-serialised publication-guide
    default — JSON-friendly so the form can pre-populate without a
    second round-trip."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    schema = dataclass_to_form_schema(PySCFConfig, id_prefix="test")
    f = _find_field(schema, "stages")
    default = f["default"]
    assert isinstance(default, list) and len(default) == 3
    assert default[0]["name"]    == "stage1"
    assert default[0]["enabled"] is True
    assert default[0]["conv_tol"] == pytest.approx(1.0e-7)
    assert default[1]["name"]    == "stage2"
    assert default[1]["conv_tol"] == pytest.approx(1.0e-9)
    assert default[2]["name"]    == "stage3"
    assert default[2]["enabled"] is False


def test_stages_schema_carries_per_row_field_descriptors():
    """``stage_fields`` is the per-column shape the JS table
    renderer uses to label columns + pick widgets per cell.  9
    fields in declaration order; types map to ``kind``."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    schema = dataclass_to_form_schema(PySCFConfig, id_prefix="test")
    f = _find_field(schema, "stages")
    stage_fields = f["stage_fields"]
    assert [sf["name"] for sf in stage_fields] == [
        "name", "enabled", "conv_tol", "gmax", "grms",
        "dmax", "drms", "etol", "max_steps",
    ]
    kinds = {sf["name"]: sf["kind"] for sf in stage_fields}
    assert kinds["name"]      == "text"
    assert kinds["enabled"]   == "checkbox"
    assert kinds["conv_tol"]  == "number"
    assert kinds["gmax"]      == "number"
    assert kinds["max_steps"] == "int"
    units = {sf["name"]: sf.get("unit") for sf in stage_fields}
    assert units["conv_tol"] == "Hartree"
    assert units["gmax"]     == "Ha/Bohr"
    assert units["dmax"]     == "Å"


def test_stages_coerce_round_trips_list_of_dicts_back_to_stagespec():
    """Form payload arrives as list-of-dicts; coerce rebuilds
    each row as a StageSpec."""
    from dataclasses import fields as dc_fields
    from molbuilder.web.blueprints._shared import (
        coerce_to_field_type,
    )
    import typing
    f = next(f for f in dc_fields(PySCFConfig) if f.name == "stages")
    hints = typing.get_type_hints(PySCFConfig)
    payload = [
        {"name": "stage1", "enabled": True,  "conv_tol": 1.0e-7,
         "gmax": 2.0e-3, "grms": 1.3e-3, "dmax": 7.2e-3,
         "drms": 4.8e-3, "etol": 1.0e-5, "max_steps": 50},
        {"name": "stage2", "enabled": False, "conv_tol": 1.0e-9,
         "gmax": 4.5e-4, "grms": 3.0e-4, "dmax": 1.8e-3,
         "drms": 1.2e-3, "etol": 1.0e-6, "max_steps": 200},
    ]
    coerced = coerce_to_field_type(f, payload, hints)
    assert len(coerced) == 2
    assert all(isinstance(s, StageSpec) for s in coerced)
    assert coerced[0].name == "stage1"
    assert coerced[0].enabled is True
    assert coerced[1].enabled is False
    assert coerced[1].max_steps == 200


def test_stages_coerce_handles_string_typed_payload():
    """Non-browser HTTP clients may send numerics as strings;
    coerce must promote them per inner-field type."""
    from dataclasses import fields as dc_fields
    from molbuilder.web.blueprints._shared import (
        coerce_to_field_type,
    )
    import typing
    f = next(f for f in dc_fields(PySCFConfig) if f.name == "stages")
    hints = typing.get_type_hints(PySCFConfig)
    payload = [
        {"name": "stage1", "enabled": "true", "conv_tol": "1e-7",
         "gmax": "2e-3", "max_steps": "50"},
    ]
    coerced = coerce_to_field_type(f, payload, hints)
    assert coerced[0].enabled is True
    assert coerced[0].conv_tol == pytest.approx(1.0e-7)
    assert coerced[0].gmax     == pytest.approx(2.0e-3)
    assert coerced[0].max_steps == 50


def test_stages_coerce_ignores_unknown_keys():
    """Forward-compat: a payload with extra keys (older client +
    newer server, or vice versa) doesn't TypeError on
    ``StageSpec(**kwargs)``."""
    from dataclasses import fields as dc_fields
    from molbuilder.web.blueprints._shared import (
        coerce_to_field_type,
    )
    import typing
    f = next(f for f in dc_fields(PySCFConfig) if f.name == "stages")
    hints = typing.get_type_hints(PySCFConfig)
    payload = [
        {"name": "stage1", "enabled": True, "future_knob": 42},
    ]
    coerced = coerce_to_field_type(f, payload, hints)
    assert coerced[0].name == "stage1"
    # Defaults kick in for unspecified knobs (no TypeError).
    assert coerced[0].max_steps == 200


def test_stages_coerce_passes_through_already_typed_items():
    """If the caller already constructed StageSpec instances (Python
    test path, or a non-web caller), the coerce path is a no-op."""
    from dataclasses import fields as dc_fields
    from molbuilder.web.blueprints._shared import (
        coerce_to_field_type,
    )
    import typing
    f = next(f for f in dc_fields(PySCFConfig) if f.name == "stages")
    hints = typing.get_type_hints(PySCFConfig)
    payload = [StageSpec(name="custom", enabled=False)]
    coerced = coerce_to_field_type(f, payload, hints)
    assert len(coerced) == 1
    assert coerced[0] is payload[0]  # identity-preserving


def _find_field(schema, name):
    """Test helper: walk a form schema for the named field."""
    for section in schema["sections"]:
        for f in section["fields"]:
            if f["name"] == name:
                return f
    raise AssertionError(f"field {name!r} not in schema")
