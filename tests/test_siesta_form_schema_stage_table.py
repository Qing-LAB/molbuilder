"""Pin the SIESTA form schema's stage-table widget (task #542 / C1.5).

The Python form-schema infrastructure (``_field_to_schema`` +
``_stagespec_to_field_schemas`` + ``coerce_to_field_type``) is
type-driven and generic over ``List[<dataclass>]``.  Since
:class:`SiestaConfig.stages` is annotated ``List[SiestaStageSpec]``,
the entire pipeline already produces the right stage-table widget
without SIESTA-specific code.

These tests pin the resulting SHAPE so a future change to the schema
helper, the ``SiestaStageSpec`` dataclass, or the rendering pipeline
can't silently break the multi-stage form -- e.g. drop a per-stage
field, change a choice list, or break the round-trip from form
payload to ``cfg.stages``.

Companion to:
  * tests/test_siesta_stages.py (the dataclass + defaults)
  * tests/test_siesta_stages_emit.py (multi-stage fdf emission)
  * tests/test_cli_siesta_stages.py (CLI surface)
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from molbuilder.config.siesta import (
    SIESTA_STAGE_STRATEGY_PRESETS,
    SiestaConfig,
    SiestaStageSpec,
    _default_siesta_stages,
)
from molbuilder.web.blueprints._shared import (
    coerce_to_field_type,
    dataclass_to_form_schema,
)


@pytest.fixture
def schema() -> Dict[str, Any]:
    return dataclass_to_form_schema(SiestaConfig, "p")


def _find_field(schema: Dict[str, Any], name: str) -> Dict[str, Any]:
    for section in schema.get("sections", []):
        for f in section.get("fields", []):
            if f.get("name") == name:
                return f
    raise AssertionError(f"field {name!r} not found in schema")


# --------------------------------------------------------------------- #
#  Top-level stages field                                                #
# --------------------------------------------------------------------- #


def test_stages_field_is_emitted_as_stage_table(schema):
    """SiestaConfig.stages must surface as a ``stage-table`` widget --
    the kind the JS form-schema renderer recognises."""
    f = _find_field(schema, "stages")
    assert f["kind"] == "stage-table"


def test_stages_field_carries_correct_id_prefix(schema):
    """The form-schema id prefix for SIESTA is ``p`` (the SIESTA panel
    in the Build UI prefixes every input with ``p-``).  The stage-table
    renderer composes per-cell ids as ``{f.id}-stage{i}-{field_name}``,
    so the id chain depends on this prefix being correct."""
    f = _find_field(schema, "stages")
    assert f["id"] == "p-stages"


def test_stages_default_matches_default_siesta_stages_value_for_value(
        schema):
    """The schema's serialized default must equal
    ``_default_siesta_stages()`` -- so the form opens pre-populated
    with the canonical 3-stage ladder and a user who clicks Generate
    without editing gets exactly the documented presets."""
    f = _find_field(schema, "stages")
    expected = [
        {"name": s.name, "enabled": s.enabled,
         "relax_type": s.relax_type, "relax_steps": s.relax_steps,
         "relax_force_tol": s.relax_force_tol,
         "relax_max_displ": s.relax_max_displ,
         "on_nonconvergence": s.on_nonconvergence,
         "continue_retries": s.continue_retries}
        for s in _default_siesta_stages()
    ]
    assert f["default"] == expected


def test_stages_field_lives_in_optimization_section(schema):
    """Per the SiestaStageSpec field metadata, stages belongs in the
    Optimization section / workflow_group=budget.  Pinning the
    section assignment ensures the field shows up where the user
    expects it (next to the other relaxation knobs)."""
    for section in schema.get("sections", []):
        names = [f.get("name") for f in section.get("fields", [])]
        if "stages" in names:
            assert section.get("name") == "Optimization", (
                f"stages landed in section {section.get('name')!r}; "
                f"expected Optimization"
            )
            return
    raise AssertionError("stages field missing entirely from schema")


# --------------------------------------------------------------------- #
#  Per-stage field schemas (the inner stage_fields list)                #
# --------------------------------------------------------------------- #


def test_stage_fields_cover_all_stagespec_fields(schema):
    """Each SiestaStageSpec dataclass field must appear in
    ``stage_fields``.  Missing one means the form silently strips
    that knob from every per-stage row -- documented as a contract
    in ``_stagespec_to_field_schemas``'s docstring."""
    f = _find_field(schema, "stages")
    stage_field_names = [sf["name"] for sf in f["stage_fields"]]
    import dataclasses as _dc
    expected = [df.name for df in _dc.fields(SiestaStageSpec)]
    assert stage_field_names == expected


def test_relax_type_renders_as_choice_with_cg_broyden_fire(schema):
    f = _find_field(schema, "stages")
    relax_type = next(sf for sf in f["stage_fields"]
                       if sf["name"] == "relax_type")
    assert relax_type["kind"] == "choice"
    assert relax_type["choices"] == ["CG", "Broyden", "FIRE"]


def test_on_nonconvergence_renders_as_choice_with_three_policies(schema):
    f = _find_field(schema, "stages")
    onc = next(sf for sf in f["stage_fields"]
                if sf["name"] == "on_nonconvergence")
    assert onc["kind"] == "choice"
    assert onc["choices"] == ["proceed", "continue", "halt"]


def test_enabled_renders_as_checkbox(schema):
    f = _find_field(schema, "stages")
    en = next(sf for sf in f["stage_fields"] if sf["name"] == "enabled")
    assert en["kind"] == "checkbox"


def test_relax_steps_renders_as_int_with_range(schema):
    f = _find_field(schema, "stages")
    rs = next(sf for sf in f["stage_fields"] if sf["name"] == "relax_steps")
    assert rs["kind"] == "int"
    assert rs.get("min") == 1
    assert rs.get("max") == 10000


def test_relax_force_tol_renders_as_number_with_unit(schema):
    f = _find_field(schema, "stages")
    rft = next(sf for sf in f["stage_fields"]
                if sf["name"] == "relax_force_tol")
    assert rft["kind"] == "number"
    assert rft.get("unit") == "eV/Å"


def test_stagespec_name_field_carries_pattern(schema):
    """The ``name`` field is a string with a regex pattern (must match
    ``[A-Za-z0-9_]+``) -- the JS renderer surfaces this as the input's
    HTML ``pattern`` attribute so the browser validates it client-side
    before the user clicks Generate."""
    f = _find_field(schema, "stages")
    n = next(sf for sf in f["stage_fields"] if sf["name"] == "name")
    assert n["kind"] == "text"
    assert n.get("pattern") == r"^[A-Za-z0-9_]+$"


# --------------------------------------------------------------------- #
#  Round-trip: form payload -> SiestaStageSpec list                     #
# --------------------------------------------------------------------- #


def test_form_payload_round_trips_into_siesta_stages():
    """A form submission payload (list-of-dicts) must coerce back into
    ``List[SiestaStageSpec]`` correctly.  This is the path between the
    JS ``collectField`` for ``stage-table`` and the SIESTA generator's
    ``cfg.stages`` consumption."""
    import dataclasses as _dc
    payload = [
        {"name": "warmup",   "enabled": True,  "relax_type": "CG",
         "relax_steps": 100, "relax_force_tol": 0.10,
         "relax_max_displ": 0.30, "on_nonconvergence": "proceed",
         "continue_retries": 1},
        {"name": "final",    "enabled": True,  "relax_type": "Broyden",
         "relax_steps": 50,  "relax_force_tol": 0.02,
         "relax_max_displ": 0.05, "on_nonconvergence": "halt",
         "continue_retries": 1},
    ]
    stages_field = next(
        f for f in _dc.fields(SiestaConfig) if f.name == "stages"
    )
    import typing
    hints = typing.get_type_hints(SiestaConfig)
    out = coerce_to_field_type(stages_field, payload, hints)
    assert isinstance(out, list)
    assert len(out) == 2
    for row in out:
        assert isinstance(row, SiestaStageSpec)
    assert out[0].name == "warmup"
    assert out[0].relax_type == "CG"
    assert out[1].relax_force_tol == pytest.approx(0.02)
    assert out[1].on_nonconvergence == "halt"


def test_form_payload_with_partial_row_uses_dataclass_defaults():
    """A row carrying only a subset of keys falls back to per-field
    SiestaStageSpec defaults for the missing knobs.  This lets the
    JS send only what changed (debounced live preflight) without
    blanking the unchanged knobs."""
    import dataclasses as _dc
    import typing
    payload = [
        # Only ``enabled`` -- everything else should default.
        {"name": "stage1", "enabled": False},
    ]
    stages_field = next(
        f for f in _dc.fields(SiestaConfig) if f.name == "stages"
    )
    hints = typing.get_type_hints(SiestaConfig)
    out = coerce_to_field_type(stages_field, payload, hints)
    assert out[0].name == "stage1"
    assert out[0].enabled is False
    # Other fields fall back to SiestaStageSpec defaults.
    assert out[0].relax_type == "CG"
    assert out[0].relax_steps == 600


def test_form_payload_rejects_non_list():
    """A dict on the stages slot is a wire-format error -- must raise
    TypeError at the boundary (not silently crash later in
    ``validate_siesta_stages`` with an AttributeError)."""
    import dataclasses as _dc
    import typing
    stages_field = next(
        f for f in _dc.fields(SiestaConfig) if f.name == "stages"
    )
    hints = typing.get_type_hints(SiestaConfig)
    with pytest.raises(TypeError, match="expected list"):
        coerce_to_field_type(stages_field, {"stages": []}, hints)


# --------------------------------------------------------------------- #
#  Strategy-preset cross-check                                           #
# --------------------------------------------------------------------- #


def test_strategy_preset_names_match_pyscf_for_drift_gate():
    """The JS ``STAGE_STRATEGY_PRESETS`` in form-schema.js hardcodes
    the preset names + enable patterns; the Python side has the same
    names per engine.  Both engines (SIESTA + PySCF) must use the
    same set of names so the JS dropdown surfaces the same options
    regardless of which engine's stage-table is rendering.

    The deeper JS<->Python drift gate (per-engine enable patterns +
    JS dropdown options) is C1.6's job; this test pins just the
    name set so a future SIESTA-only preset rename doesn't
    desynchronize the engines."""
    from molbuilder.config.pyscf import STAGE_STRATEGY_PRESETS as PYSCF
    assert (set(SIESTA_STAGE_STRATEGY_PRESETS.keys())
            == set(PYSCF.keys())), (
        "SIESTA + PySCF strategy preset names diverged -- the JS "
        "dropdown shows ONE list, so the names must match across "
        "engines.  Update molbuilder/config/siesta.py "
        "SIESTA_STAGE_STRATEGY_PRESETS to match PySCF's keys (or "
        "vice versa)."
    )
