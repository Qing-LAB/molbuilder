"""Resolution — what a staged render actually uses when two homes disagree.

Contract: ``docs/engines/stages.md`` § 4 — *effective config = the template's
values ⊕ that stage's ``overrides``*, one object validated **and** rendered
(R1), validated as a resolved whole and never as a diff (R2).

The stage's value beats the shared one, and a field the stage says nothing
about keeps the shared value; a stage names any field of the shared schema
through ``overrides``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.task import Stage


@pytest.fixture
def h2() -> Structure:
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        vacuum=(12.0, 12.0, 12.0),
    )


# --------------------------------------------------------------------- #
#  What a ladder must refuse                                            #
# --------------------------------------------------------------------- #


def test_two_stages_with_one_name_are_refused_where_the_ladder_is_read():
    """Every per-stage artifact is keyed ``<label>_<name>.fdf``, so a name
    collision does not fail -- it silently overwrites a stage.  The refusal
    lives where the ladder is READ (task.py), and it is
    case-insensitive because the names become filenames.  Driven through
    ``read_task`` — the codec's live route."""
    import json as _json
    import tempfile
    from pathlib import Path as _P

    from molbuilder.task import (Stage, StructureRef, Task, derive_run,
                                 read_task, write_task)
    with tempfile.TemporaryDirectory() as d:
        p = _P(d) / "task.json"
        write_task(p, Task(
            engine="siesta", shape="flat", calculation="optimization",
            run=derive_run("j", stage_names=("tight", "medium")),
            structure=StructureRef(source="h2.xyz"),
            varies=("mesh_cutoff",),
            stages=(Stage(name="tight", overrides={"mesh_cutoff": 200.0}),
                    Stage(name="medium", overrides={"mesh_cutoff": 300.0}))))
        # the codec wrote a valid file; a hand edit introduces the clash
        raw = _json.loads(p.read_text())
        raw["stages"][1]["name"] = "Tight"
        p.write_text(_json.dumps(raw))
        with pytest.raises(ValueError, match="two stages named"):
            read_task(p)


def test_a_stage_name_that_is_not_a_filename_is_refused():
    """The rule holds for the OBJECT, not only for a description parsed
    from disk: a stage name becomes a filename and an unquoted bash word."""
    with pytest.raises(ValueError, match=r"\[A-Za-z0-9_\]"):
        Stage(name="rm -rf /")


# --------------------------------------------------------------------- #
#  effective_config, the one place a stage is resolved                  #
# --------------------------------------------------------------------- #
#
# engines/stages.md § 4:
#
#   effective config = the template's values ⊕ that stage's `overrides`
#
#   R1  one object is validated AND rendered — what was checked and what
#       was written cannot come apart.
#   R2  a stage is validated as a resolved WHOLE, never as a diff.
#
# § 6.2's subset rule decides the fallback: a stage may omit a varied key,
# and omitting it means "use the template's value".

from molbuilder.resolve import effective_config                # noqa: E402


def _template() -> SiestaConfig:
    """The backbone: every field, with values.  What the generating tab
    wrote and what a stage overlays."""
    return SiestaConfig(system_label="JOB", mesh_cutoff=150.0,
                        relax_type="CG", relax_force_tol=0.05,
                        restart="clean")


def test_an_override_wins_over_the_template():
    cfg = effective_config(_template(), {"mesh_cutoff": 300.0})
    assert cfg.mesh_cutoff == 300.0


def test_a_field_the_stage_does_not_name_keeps_the_templates_value():
    """§ 6.2's subset rule: absent means 'use the template's value'."""
    cfg = effective_config(_template(), {"mesh_cutoff": 300.0})
    assert cfg.relax_type == "CG"
    assert cfg.relax_force_tol == 0.05


def test_it_returns_an_ordinary_config_of_the_engines_own_type():
    """§ 4: 'an ordinary instance of the engine's config dataclass — a
    SiestaConfig, not a new type'.  Which is what lets the SHIPPED
    validator and the SHIPPED emitter take it unchanged."""
    cfg = effective_config(_template(), {})
    assert type(cfg) is SiestaConfig


def test_the_template_is_not_mutated():
    """R1's precondition.  Resolving stage 2 must not disturb what stage 3
    resolves against, or the ladder depends on the order it was resolved
    in — a bug that would only appear with three stages."""
    tpl = _template()
    before = dataclasses.asdict(tpl)
    effective_config(tpl, {"mesh_cutoff": 300.0})
    assert dataclasses.asdict(tpl) == before


def test_two_stages_resolve_independently():
    tpl = _template()
    coarse = effective_config(tpl, {"mesh_cutoff": 150.0})
    tight = effective_config(tpl, {"mesh_cutoff": 300.0})
    assert (coarse.mesh_cutoff, tight.mesh_cutoff) == (150.0, 300.0)


def test_a_stage_may_override_ANY_field_not_a_privileged_four():
    """**The gate.**  `mesh_cutoff`, `basis_size` and `kgrid` are fields of
    the shared schema like any other (a run setting excepted: it is the
    rung's run card's, `stages.md` § 1.2)."""
    cfg = effective_config(_template(), {
        "mesh_cutoff": 400.0, "basis_size": "TZP", "kgrid": (2, 2, 2),
        "relax_type": "Broyden"})
    assert cfg.mesh_cutoff == 400.0
    assert cfg.basis_size == "TZP"
    assert tuple(cfg.kgrid) == (2, 2, 2)
    assert cfg.relax_type == "Broyden"


def test_an_unknown_field_is_refused_by_name():
    """The task codec has no schema; the operator does, and § 6.6 says the
    refusal names the field.

    **And it names WHO supplied the override.** The operator takes a mapping,
    not a stage, so the caller passes the label — *"stage 'tight'
    overrides a field that does not exist"* is findable and *"an override does
    not exist"* is not.
    """
    with pytest.raises(ValueError) as e:
        effective_config(_template(), {"mesh_cutof": 300.0},
                         where="stage 'tight'")
    assert "mesh_cutof" in str(e.value)
    assert "tight" in str(e.value)


def test_a_stage_field_name_in_overrides_is_refused():
    """§ 2: an override may not redefine a stage field.  It would also not
    be a schema field, so the message must still be the useful one."""
    with pytest.raises(ValueError) as e:
        effective_config(_template(), {"overrides": {}})
    assert "overrides" in str(e.value)


def test_an_OPTIONAL_float_is_widened_like_any_other():
    """§ 25.3: the DECLARED type decides, not the annotation.

    JSON has one number, so a stage override arrives as an int where the field
    wants a float — ``{"mesh_cutoff": 150}`` renders ``MeshCutoff 150 Ry``
    where ``150.0`` renders ``MeshCutoff 150.0 Ry``: the same number, a
    different deck.  The operator widens int → float to close that.

    Under ``from __future__ import annotations`` a field's ``type`` is the
    source text, so ``Optional[float]`` is not ``"float"`` and a string match
    on the annotation never widens ``md_target_temperature``.  The catalogue
    declares it ``float``, because a declared type is exactly *"what a parser
    cannot know"* (`template.md` § 5).
    """
    cfg = _template()
    assert isinstance(effective_config(cfg, {"mesh_cutoff": 300}).mesh_cutoff, float)
    for name in ("md_target_temperature",):
        got = getattr(effective_config(cfg, {name: 2}), name)
        assert isinstance(got, float), (
            f"{name} is declared float in the catalogue but an int override "
            f"came back {type(got).__name__} -- the operator is reading the "
            f"annotation again")


def test_a_list_override_becomes_the_tuple_the_field_declares():
    """JSON HAS ONE SEQUENCE — the widening argument above, one shape up.

    ``kgrid`` declares ``Tuple[int, int, int]`` and a description can only
    spell it ``[4, 4, 1]``.  Left a LIST, an overridden field would differ
    from the template's TUPLE: one field with two shapes, decided by whether
    a cell happened to be filled in.
    """
    got = effective_config(_template(), {"kgrid": [4, 4, 1]}).kgrid
    assert isinstance(got, tuple) and got == (4, 4, 1), (
        f"kgrid came back {got!r} ({type(got).__name__}) where the field "
        f"declares Tuple[int, int, int]")
    disp = effective_config(
        _template(),
        {"kgrid_displacement": [0.5, 0.5, 0.0]}).kgrid_displacement
    assert isinstance(disp, tuple), (
        f"kgrid_displacement came back {type(disp).__name__}")


def test_nothing_but_a_lossless_respelling_is_coerced():
    """The other half, and the reason both coercions are safe.

    ``int -> float`` and ``list -> tuple`` are the SAME value written the
    only other way the source format allows, so doing them silently is
    honest.  ``float -> int`` is not: it would truncate ``relax_steps:
    100.7`` to 100.  Nor is parsing a string, which would make a quoting
    slip invisible — the slip that produced ``"kgrid": "4,4,1"``.  Both are
    the caller's mistake and are refused BY NAME in the preflight, which is
    where a wrong value belongs.
    """
    assert effective_config(_template(), {"relax_steps": 100}).relax_steps == 100
    assert isinstance(
        effective_config(_template(), {"relax_steps": 100}).relax_steps, int)
    kept = effective_config(_template(), {"basis_size": "TZP"}).basis_size
    assert kept == "TZP"
    kept_text = effective_config(_template(), {"kgrid": "4,4,1"}).kgrid
    assert kept_text == "4,4,1", (
        "the ⊕ operator parsed a string -- a quoting slip must stay visible "
        "so the preflight can name it")
