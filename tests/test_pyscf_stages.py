"""The shipped PySCF ladder: where it comes from, and what it is made of.

Contract: ``docs/engines/stages.md`` § 1.1 (**no engine config carries a
stage list**) and § 1.1a (a PySCF ladder is N decks and N jobs, declared in
``task.json``, exactly as SIESTA's is), § 2 (a stage has three fields),
§ 4 (``template ⊕ overrides``).

**This file replaced one that tested ``StageSpec``, ``_default_stages`` and
``validate_stages``.**  Those are deleted, so tests naming that class's ten
fields, its per-field defaults and its validator went with them -- a test
that pins a mechanism the contract removed is not coverage, it is a second
copy of the old design that fails the next time the right thing is done.

What survived is every rule that was about the *ladder* rather than about
that class, restated against the mechanism that carries it now.  **This file
is deliberately the twin of ``test_siesta_stages.py``**: the two engines
differ in which parameters a rung varies and in nothing else, so a rule that
holds for one and is untested on the other is where they drift apart.

The per-tier NUMBERS are not asserted here.  They are asserted against
``tuning.md`` § 2.4 / § 2.5 -- the table that is their authority -- by
``test_doc_claims.py::test_a_tier_table_stated_in_prose_is_the_table_the_code_ships``.
Restating them here would make this file a third copy and the check circular.
"""
from __future__ import annotations

import dataclasses

import pytest

from molbuilder.config.pyscf import (
    PySCFConfig,
    PYSCF_STAGE_PRESETS,
    STAGE_STRATEGY_PRESETS,
)
from molbuilder.config.siesta import SIESTA_STAGE_NAMES
from molbuilder.pyscf.stages import default_pyscf_stages
from molbuilder.task import Stage


# --------------------------------------------------------------------- #
#  § 1.1 — an engine config carries no stage list                       #
# --------------------------------------------------------------------- #

def test_the_config_has_no_stages_field():
    """**The deletion, asserted directly.**

    ``PySCFConfig.stages`` outlived ``SiestaConfig.stages`` on the strength
    of one reader: the in-script ``for STAGE in STAGES:`` loop, which made
    the list engine BEHAVIOUR rather than a rival declaration.  § 1.1a
    retired that loop, and the field went with the premise that kept it."""
    names = {f.name for f in dataclasses.fields(PySCFConfig)}
    assert "stages" not in names


def test_no_field_of_the_config_is_a_list_of_dataclasses():
    """The stronger form: not merely *this* name, but the SHAPE.  A
    ``List[<dataclass>]`` on an engine config is what the form-schema
    generator used to turn into a stage-table, so a differently-named
    ladder would reopen § 1.2 just as wide."""
    import typing
    hints = typing.get_type_hints(PySCFConfig)
    for f in dataclasses.fields(PySCFConfig):
        ann = hints.get(f.name)
        args = typing.get_args(ann)
        assert not (typing.get_origin(ann) in (list, tuple)
                    and args and dataclasses.is_dataclass(args[0])), (
            f"PySCFConfig.{f.name} is a list of dataclasses -- the form "
            f"generator would emit a stage-table for it (stages.md § 1.2)")


def test_the_pyscf_form_schema_emits_no_stage_table():
    """The consequence at the surface that had the bug.  The generator
    answers *what settings exist and how is each drawn*; it must never meet
    a stage (web/form-schema.md § 1's callout)."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    sch = catalogue_to_form_schema("pyscf", "py")
    kinds = [f["kind"] for s in sch["sections"] for f in s["fields"]]
    assert "stage-table" not in kinds


# --------------------------------------------------------------------- #
#  The shipped ladder                                                   #
# --------------------------------------------------------------------- #

def test_default_ladder_is_three_named_stages():
    """The names are the SHARED vocabulary, not PySCF's own.  A rung is a
    rung in both engines, so ``coarse`` means the same thing in a PySCF
    description and a SIESTA one (`tuning.md` § 4)."""
    stages = default_pyscf_stages()
    assert [s.name for s in stages] == [SIESTA_STAGE_NAMES[i]
                                        for i in (1, 2, 3)]
    assert all(isinstance(s, Stage) for s in stages)


def test_default_ladder_enabled_pattern_matches_siesta():
    """publishable = coarse + medium; tight is opt-in.  The same shape as
    SIESTA's default, so the two engines read alike."""
    from molbuilder.siesta.stages import default_siesta_stages
    assert ([s.enabled for s in default_pyscf_stages()]
            == [s.enabled for s in default_siesta_stages()]
            == [True, True, False])


@pytest.mark.parametrize("tier", sorted(PYSCF_STAGE_PRESETS))
def test_each_stages_overrides_are_exactly_that_tiers_preset(tier):
    """**The one that keeps the science in one place.**  A rung's overrides
    are that tier's preset row and nothing else -- so a tier value can be
    changed in exactly one place, and that place is checked against
    `tuning.md` § 2.4.

    Nothing is added to that row.  ``restart`` was spliced in here
    positionally until 2026-08-18; it is a property of neither the tier nor
    the rung's index -- the folder answers it, at run time
    (`run-identity.md` § 4 rule 3)."""
    stage = default_pyscf_stages("vib-quality")[tier - 1]
    assert stage.overrides == PYSCF_STAGE_PRESETS[tier]


def test_no_shipped_ladder_says_anything_about_restart():
    """Not the presets table, and not the rungs built from it.

    ``restart`` in the table would make it a property of the TIER; on a rung
    it would make it a property of the POSITION.  It is neither: a run
    continues from what is in the folder, and a person who wants otherwise
    says so (`run-identity.md` § 4 rule 3)."""
    for tier, row in PYSCF_STAGE_PRESETS.items():
        assert "restart" not in row, tier
    for strategy in sorted(STAGE_STRATEGY_PRESETS):
        for st in default_pyscf_stages(strategy):
            assert "restart" not in st.overrides, (strategy, st.name)


def test_a_stage_has_exactly_the_three_fields_of_section_2():
    """§ 2 -- the same three for both engines, because a ``Stage`` is not
    an engine's type."""
    assert [f.name for f in dataclasses.fields(Stage)] == [
        "name", "enabled", "overrides"]


def test_every_field_the_shipped_ladder_varies_exists_in_the_schema():
    """The preflight's rule, applied to what molbuilder itself ships: a
    ladder naming a field the schema does not have is refused, so the
    default must not be one."""
    known = {f.name for f in dataclasses.fields(PySCFConfig)}
    varied = {k for s in default_pyscf_stages("vib-quality")
              for k in s.overrides}
    assert varied <= known
    assert varied == {"scf_conv_tol", "geom_gmax", "geom_grms", "geom_dmax",
                      "geom_drms", "geom_etol", "geom_max_steps"}


#: The catalogue, through its one door (`template.md` § 8.0).  A test that
#: parsed the TOML itself would be a second reader of the file the whole
#: contract exists to give one.
def _catalogue_item(name: str):
    from molbuilder import template as _T
    return _T.one(_T.read_template(_T.load_catalogue()), name, engine="pyscf")


_VARIED = ("scf_conv_tol", "geom_gmax", "geom_grms", "geom_dmax",
           "geom_drms", "geom_etol", "geom_max_steps")


@pytest.mark.parametrize("key", _VARIED)
def test_every_varied_field_is_a_catalogue_item_with_a_bound(key):
    """§ 1.1a's derivation depended on these being catalogue items, and the
    per-rung numeric checks the deleted ``validate_stages`` made now rest on
    the ``range`` each item declares (`validation/task.py` checks every
    override against it).  An item without one is a rung that can carry any
    number at all."""
    item = _catalogue_item(key)
    assert item is not None, f"{key} does not apply to pyscf"
    assert item.range is not None, key


@pytest.mark.parametrize("tier", sorted(PYSCF_STAGE_PRESETS))
def test_every_shipped_tier_value_is_inside_its_items_bound(tier):
    """The shipped ladder must pass the check every user ladder passes.
    A default outside its own declared bound would be refused by
    `jobset describe` -- of the ladder molbuilder itself ships."""
    for key, value in PYSCF_STAGE_PRESETS[tier].items():
        lo, hi = _catalogue_item(key).range
        assert lo <= value <= hi, f"{key}={value} outside [{lo}, {hi}]"


# --------------------------------------------------------------------- #
#  The strategy presets choose which tiers run, and nothing else        #
# --------------------------------------------------------------------- #

def test_strategy_preset_names_match_siesta():
    from molbuilder.config.siesta import SIESTA_STAGE_STRATEGY_PRESETS
    assert (set(STAGE_STRATEGY_PRESETS)
            == set(SIESTA_STAGE_STRATEGY_PRESETS))


@pytest.mark.parametrize("strategy,expected", [
    ("publishable", [True, True, False]),
    ("loose-only",  [True, False, False]),
    ("vib-quality", [True, True, True]),
])
def test_strategy_preset_enabled_masks(strategy, expected):
    assert [s.enabled for s in default_pyscf_stages(strategy)] == expected


def test_strategy_preset_changes_only_the_enable_flags():
    """A preset says which tiers run; it never retunes one.  If it did,
    picking 'loose-only' would silently change what the coarse rung
    computes."""
    a = default_pyscf_stages("loose-only")
    b = default_pyscf_stages("vib-quality")
    assert [s.overrides for s in a] == [s.overrides for s in b]
    assert [s.name for s in a] == [s.name for s in b]


def test_strategy_preset_rejects_unknown_name():
    """It REFUSES rather than falling back.  A silent fallback answers a
    question nobody asked: a misspelled strategy would run a ladder the
    caller did not name, and `jobset describe --stage-strategy` is the one
    door a ladder is authored through."""
    with pytest.raises(ValueError, match="unknown PySCF stage strategy"):
        default_pyscf_stages("no-such-preset")


def test_each_call_returns_independent_stages():
    """A caller that disables a stage must not disturb the next caller's
    ladder -- the mutable-default bug the old factory existed to avoid,
    asserted rather than assumed."""
    a, b = default_pyscf_stages(), default_pyscf_stages()
    assert a is not b
    assert a[0].overrides is not b[0].overrides
    a[0].overrides["basis"] = "def2-QZVP"
    assert "basis" not in b[0].overrides


# --------------------------------------------------------------------- #
#  The two engines build a ladder the same way                          #
# --------------------------------------------------------------------- #

def test_both_engines_build_a_ladder_through_the_same_shape():
    """§ 1.1a's whole claim, asserted directly: same call signature, same
    return type, same field names, same stage names.  The difference is
    which parameters the overrides carry.

    This is the test that fails when one engine is 'fixed' and the other is
    not -- the drift the user named as the thing to prevent."""
    from molbuilder.siesta.stages import default_siesta_stages
    for strategy in sorted(STAGE_STRATEGY_PRESETS):
        p = default_pyscf_stages(strategy)
        s = default_siesta_stages(strategy)
        assert [x.name for x in p] == [x.name for x in s]
        assert [x.enabled for x in p] == [x.enabled for x in s]
        # And neither says anything about restarting: that is the folder's
        # answer at run time, not the ladder's (`run-identity.md` § 4 rule 3).
        assert not any("restart" in x.overrides for x in p + s)
