"""Tests for the SiestaStageSpec dataclass + SiestaConfig.stages field
+ SIESTA_STAGE_STRATEGY_PRESETS (task #542, commit 1).

Pins the cross-engine consistent surface:
  * stage names are stage1 / stage2 / stage3 (matches PySCF)
  * defaults match SIESTA_STAGE_PRESETS (the --stage {1,2,3} CLI overlay
    that ships today) so single-stage and multi-stage paths produce
    identical per-stage .fdf content
  * strategy preset names match PySCF (publishable / loose-only /
    vib-quality)
  * round-trip via dict survives the schema gate (CLI's --stages-json
    consumes the same shape that the form-schema emitter produces)
"""
from __future__ import annotations

import dataclasses
import pytest

from molbuilder.config.siesta import (
    SiestaConfig,
    SiestaStageSpec,
    SIESTA_STAGE_PRESETS,
    SIESTA_STAGE_STRATEGY_PRESETS,
    _default_siesta_stages,
    apply_siesta_stage_strategy,
    siesta_stages_from_dicts,
    validate_siesta_stages,
)


# --------------------------------------------------------------------- #
#  SiestaStageSpec defaults                                              #
# --------------------------------------------------------------------- #


def test_stagespec_default_is_stage1_loose_warmup():
    """Bare ``SiestaStageSpec()`` is the stage1 loose-preopt tier:
    CG warm-up, 600 steps, 0.05 eV/Å, 0.20 Å displacement cap.  Matches
    SIESTA_STAGE_PRESETS[1] value-for-value."""
    s = SiestaStageSpec()
    assert s.name == "stage1"
    assert s.enabled is True
    assert s.relax_type == "CG"
    assert s.relax_steps == 600
    assert s.relax_force_tol == pytest.approx(0.05)
    assert s.relax_max_displ == pytest.approx(0.20)
    assert s.on_nonconvergence == "halt"  # field default; per-stage
                                          # overrides happen in _default_siesta_stages


# --------------------------------------------------------------------- #
#  _default_siesta_stages() — three-tier ladder                          #
# --------------------------------------------------------------------- #


def test_default_stages_returns_three_named_stages():
    stages = _default_siesta_stages()
    assert len(stages) == 3
    assert [s.name for s in stages] == ["stage1", "stage2", "stage3"]


def test_default_stages_enabled_pattern_matches_pyscf():
    """stage1 + stage2 enabled by default; stage3 OFF (vib/IR opt-in).
    Same enabled-mask as PySCF's _default_stages()."""
    stages = _default_siesta_stages()
    assert [s.enabled for s in stages] == [True, True, False]


@pytest.mark.parametrize("idx,stage_name", [(0, 1), (1, 2), (2, 3)])
def test_default_stages_match_existing_cli_overlay(idx, stage_name):
    """Each _default_siesta_stages() entry matches SIESTA_STAGE_PRESETS
    for the same stage number -- so ``--stage 2`` (single-stage CLI)
    and ``stages[1]`` (multi-stage pipeline) produce identical .fdfs."""
    stages = _default_siesta_stages()
    overlay = SIESTA_STAGE_PRESETS[stage_name]
    assert stages[idx].relax_type == overlay["relax_type"]
    assert stages[idx].relax_steps == overlay["relax_steps"]
    assert stages[idx].relax_force_tol == pytest.approx(
        overlay["relax_force_tol"])
    assert stages[idx].relax_max_displ == pytest.approx(
        overlay["relax_max_displ"])


def test_default_stages_on_nonconvergence_policy_matches_pyscf():
    """stage1 = proceed (loose warm-up; hand off "good enough" to next
    tier); stage2 + stage3 = halt (publishable / tight; failure is real
    signal).  Matches PySCF's _default_stages() policy ladder."""
    stages = _default_siesta_stages()
    assert stages[0].on_nonconvergence == "proceed"
    assert stages[1].on_nonconvergence == "halt"
    assert stages[2].on_nonconvergence == "halt"


# --------------------------------------------------------------------- #
#  validate_siesta_stages()                                              #
# --------------------------------------------------------------------- #


def test_validate_passes_defaults():
    assert validate_siesta_stages(_default_siesta_stages()) == []


def test_validate_catches_bad_name():
    bad = [SiestaStageSpec(name="not a valid name!")]
    errs = validate_siesta_stages(bad)
    assert any("name" in e for e in errs)


def test_validate_catches_unknown_relax_type():
    bad = [dataclasses.replace(_default_siesta_stages()[0],
                                relax_type="NotARelaxAlgorithm")]
    errs = validate_siesta_stages(bad)
    assert any("relax_type" in e for e in errs)


def test_validate_catches_nonpositive_force_tol():
    bad = [dataclasses.replace(_default_siesta_stages()[0],
                                relax_force_tol=0.0)]
    errs = validate_siesta_stages(bad)
    assert any("relax_force_tol" in e for e in errs)


def test_validate_catches_unknown_policy():
    bad = [dataclasses.replace(_default_siesta_stages()[0],
                                on_nonconvergence="bail-and-cry")]
    errs = validate_siesta_stages(bad)
    assert any("on_nonconvergence" in e for e in errs)


# --------------------------------------------------------------------- #
#  Strategy presets                                                      #
# --------------------------------------------------------------------- #


def test_strategy_preset_names_match_pyscf():
    """Same three preset names as the PySCF surface.  This keeps the
    UI's preset dropdown semantically aligned -- "publishable" means
    the same thing across engines (stage1 loose + stage2 publishable;
    skip the vib/IR-tight tier)."""
    from molbuilder.config.pyscf import STAGE_STRATEGY_PRESETS as _PYSCF
    assert (set(SIESTA_STAGE_STRATEGY_PRESETS.keys())
            == set(_PYSCF.keys()))


@pytest.mark.parametrize("strategy,expected", [
    ("publishable", (True, True, False)),
    ("loose-only", (True, False, False)),
    ("vib-quality", (True, True, True)),
])
def test_strategy_preset_enabled_masks(strategy, expected):
    """Each preset's enabled-mask matches its semantic role."""
    stages = _default_siesta_stages()
    applied = apply_siesta_stage_strategy(stages, strategy)
    assert tuple(s.enabled for s in applied) == expected


def test_strategy_preset_preserves_per_stage_values():
    """Preset only flips ``enabled``; relax_type / steps / force_tol /
    max_displ / on_nonconvergence are preserved.  User overrides on
    those fields survive the strategy-apply call."""
    stages = _default_siesta_stages()
    customised = list(stages)
    customised[1] = dataclasses.replace(stages[1], relax_force_tol=0.123)
    applied = apply_siesta_stage_strategy(customised, "loose-only")
    assert applied[1].relax_force_tol == pytest.approx(0.123)


def test_strategy_preset_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown SIESTA stage strategy"):
        apply_siesta_stage_strategy(
            _default_siesta_stages(), "no-such-strategy")


# --------------------------------------------------------------------- #
#  siesta_stages_from_dicts -- round-trip via JSON payload               #
# --------------------------------------------------------------------- #


def test_from_dicts_round_trips_defaults():
    """``siesta_stages_from_dicts([... defaults as dicts ...])`` returns
    a list identical (field-for-field) to ``_default_siesta_stages()``.
    The CLI's --stages-json path depends on this."""
    stages = _default_siesta_stages()
    payload = [dataclasses.asdict(s) for s in stages]
    restored = siesta_stages_from_dicts(payload)
    for original, copy in zip(stages, restored):
        assert dataclasses.asdict(original) == dataclasses.asdict(copy)


def test_from_dicts_silently_ignores_unknown_keys():
    """Forward-compat: a payload from a future SiestaStageSpec with
    extra fields shouldn't break the current parser."""
    payload = [{
        "name": "stage1", "enabled": True,
        "relax_type": "CG", "relax_steps": 600,
        "relax_force_tol": 0.05, "relax_max_displ": 0.20,
        "future_field": "future_value",  # unknown -> silently dropped
    }]
    restored = siesta_stages_from_dicts(payload)
    assert restored[0].relax_type == "CG"
    assert not hasattr(restored[0], "future_field")


def test_from_dicts_coerces_string_booleans():
    """JSON-from-shell payloads may carry ``"true"`` / ``"false"``
    strings instead of bool literals; the parser coerces both."""
    payload = [{"name": "stage1", "enabled": "true",
                "relax_type": "CG", "relax_steps": 100,
                "relax_force_tol": 0.05, "relax_max_displ": 0.20}]
    restored = siesta_stages_from_dicts(payload)
    assert restored[0].enabled is True


# --------------------------------------------------------------------- #
#  SiestaConfig.stages field                                             #
# --------------------------------------------------------------------- #


def test_siesta_config_has_stages_field():
    cfg = SiestaConfig()
    assert hasattr(cfg, "stages")
    assert isinstance(cfg.stages, list)


def test_siesta_config_stages_defaults_to_three_stage_ladder():
    cfg = SiestaConfig()
    assert len(cfg.stages) == 3
    assert [s.name for s in cfg.stages] == ["stage1", "stage2", "stage3"]


def test_siesta_config_stages_default_is_independent_per_instance():
    """The default_factory pattern means two SiestaConfig()s get
    distinct list objects -- mutating one's stages doesn't leak into
    the other.  Pins the field's use of ``default_factory=lambda: ...``
    over a mutable default."""
    cfg1 = SiestaConfig()
    cfg2 = SiestaConfig()
    cfg1.stages[0].relax_force_tol = 0.999
    assert cfg2.stages[0].relax_force_tol != 0.999
