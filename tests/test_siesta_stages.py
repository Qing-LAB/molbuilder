"""The shipped SIESTA ladder: where it comes from, and what it is made of.

Contract: ``docs/engines/stages.md`` § 1.1 (**no engine config carries a
stage list**), § 1.3 (the default *selection*), § 2 (a stage has three
fields), § 4 (``template ⊕ overrides``).
"""
from __future__ import annotations

import dataclasses

import pytest

from molbuilder.config.siesta import SiestaConfig, SIESTA_STAGE_PRESETS
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.task import Stage


# --------------------------------------------------------------------- #
#  § 1.1 — an engine config carries no stage list                       #
# --------------------------------------------------------------------- #

def test_the_config_has_no_stages_field():
    """A config that contains a ladder cannot be the ordinary single
    parameter set § 4 resolves *to*, and the form generator would publish
    its fields as the columns a user may vary (§ 1.2)."""
    names = {f.name for f in dataclasses.fields(SiestaConfig)}
    assert "stages" not in names


def test_no_field_of_the_config_is_a_list_of_dataclasses():
    """The stronger form: not merely *this* name, but the SHAPE.  A
    ``List[<dataclass>]`` on an engine config is what the form-schema
    generator turns into a stage-table, so a differently-named ladder
    would reopen § 1.2 just as wide."""
    import typing
    hints = typing.get_type_hints(SiestaConfig)
    for f in dataclasses.fields(SiestaConfig):
        ann = hints.get(f.name)
        args = typing.get_args(ann)
        assert not (typing.get_origin(ann) in (list, tuple)
                    and args and dataclasses.is_dataclass(args[0])), (
            f"SiestaConfig.{f.name} is a list of dataclasses -- the form "
            f"generator would emit a stage-table for it (stages.md § 1.2)")


def test_the_siesta_form_schema_emits_no_stage_table():
    """The consequence, at the surface that had the bug.  The generator
    answers *what settings exist and how is each drawn*; it must never
    meet a stage (web/form-schema.md § 1's callout).

    ``catalogue_to_form_schema`` is the builder the tab is served by.  Both
    engines are checked, because the rule is the tab's and not SIESTA's.
    """
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    for engine, prefix in (("siesta", "p"), ("pyscf", "py")):
        sch = catalogue_to_form_schema(engine, prefix)
        kinds = [f["kind"] for s in sch["sections"] for f in s["fields"]]
        assert "stage-table" not in kinds, engine


# --------------------------------------------------------------------- #
#  The shipped ladder                                                   #
# --------------------------------------------------------------------- #

def test_default_ladder_is_three_named_stages():
    stages = default_siesta_stages()
    # The tiers' NAMES, from SIESTA_STAGE_NAMES: decision 27 puts the ordinal
    # in the artifact token (``01_coarse``), so a name that is itself a
    # position would say the number twice and the science none.
    assert [s.name for s in stages] == ["coarse", "medium", "tight"]
    assert all(isinstance(s, Stage) for s in stages)


def test_default_ladder_enabled_pattern_matches_pyscf():
    """publishable = coarse + medium; tight is opt-in.  Same
    shape as PySCF's default so the two engines read alike."""
    assert [s.enabled for s in default_siesta_stages()] == [True, True, False]


@pytest.mark.parametrize("tier", sorted(SIESTA_STAGE_PRESETS))
def test_each_stages_overrides_are_exactly_that_tiers_preset(tier):
    """Stage N of the ladder overrides with ``SIESTA_STAGE_PRESETS[N]`` --
    one table, so a tier value can be changed in exactly one place."""
    stage = default_siesta_stages("vib-quality")[tier - 1]
    # The rung's overrides ARE that tier's row, with nothing added: `restart`
    # is a property of neither the tier nor the rung's index -- the folder
    # answers it, at run time (`run-identity.md` § 4 rule 3).
    assert stage.overrides == SIESTA_STAGE_PRESETS[tier]


def test_a_stage_has_exactly_the_four_fields_of_section_2():
    """§ 2.  Anything else a stage seemed to need turned out to belong to
    the shared schema or to a producer's input (§ 3).

    `execution` is the fourth (§ 6.8d): WHAT THIS RUNG RUNS AT, where
    `overrides` is what it *is*.  Two maps and not one
    because different things read them -- `overrides` reaches the deck
    through `varies`, `execution` reaches the launch through the grid
    enumerator -- and a field that changes the answer is refused from
    `execution` by name."""
    # This is a TRIPWIRE ON A DECISION, not a shape check: `task.STAGE_FIELDS`
    # is derived from these fields, so the two cannot drift, but "four and
    # no others" is a design ruling and no type says it.  A fifth field should
    # cost a conversation.
    assert [f.name for f in dataclasses.fields(Stage)] == [
        "name", "enabled", "overrides", "execution"]


def test_every_field_the_shipped_ladder_varies_exists_in_the_schema():
    """The preflight's rule, applied to what molbuilder itself ships: a
    ladder naming a field the schema does not have is refused, so the
    default must not be one.

    ``varies`` is DERIVED from the overrides, here as everywhere -- § 6.2's
    subset rule is then checked against the thing it was computed from, so
    the two cannot disagree."""
    known = {f.name for f in dataclasses.fields(SiestaConfig)}
    varied = {k for s in default_siesta_stages("vib-quality")
              for k in s.overrides}
    assert varied <= known
    assert varied == {"relax_type", "relax_steps",
                      "relax_force_tol", "relax_max_displ"}


# --------------------------------------------------------------------- #
#  The strategy presets choose which tiers run, and nothing else        #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy,expected", [
    ("publishable", [True, True, False]),
    ("loose-only",  [True, False, False]),
    ("vib-quality", [True, True, True]),
])
def test_strategy_preset_enabled_masks(strategy, expected):
    assert [s.enabled for s in default_siesta_stages(strategy)] == expected


def test_strategy_preset_changes_only_the_enable_flags():
    """A preset says which tiers run; it never retunes one.  If it did,
    picking 'loose-only' would silently change what stage1 computes."""
    a = default_siesta_stages("loose-only")
    b = default_siesta_stages("vib-quality")
    assert [s.overrides for s in a] == [s.overrides for s in b]
    assert [s.name for s in a] == [s.name for s in b]


def test_strategy_preset_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown SIESTA stage strategy"):
        default_siesta_stages("no-such-preset")


def test_each_call_returns_independent_stages():
    """A caller that disables a stage must not disturb the next caller's
    ladder -- the mutable-default bug, asserted rather than assumed."""
    a, b = default_siesta_stages(), default_siesta_stages()
    assert a is not b
    assert a[0].overrides is not b[0].overrides
    a[0].overrides["mesh_cutoff"] = 999
    assert "mesh_cutoff" not in b[0].overrides


# --------------------------------------------------------------------- #
#  § 3 — the non-convergence policy is the producer's input             #
# --------------------------------------------------------------------- #

def test_no_stage_carries_a_nonconvergence_policy():
    """§ 3: it fails question 1 -- without a scheduler there is nothing
    for it to mean -- so it is not a stage field.  And it is not a
    shared-schema field either, or ``overrides`` would readmit it."""
    assert "on_nonconvergence" not in {f.name for f in dataclasses.fields(Stage)}
    assert "on_nonconvergence" not in {f.name
                                       for f in dataclasses.fields(SiestaConfig)}


def test_the_nonconvergence_policy_left_with_the_edges_it_was():
    """§ 3: *"its entire effect is the edge between one attempt and the next."*

    Reinstating a per-stage policy means giving it a home in the description
    **and** a reader that does something with it.
    """
    import molbuilder.siesta.stages as stages_mod

    assert not hasattr(stages_mod, "stages_to_jobset")
    assert not hasattr(stages_mod, "build_siesta_stage_bundle")


def test_continue_retries_is_a_shared_field_not_a_stage_one():
    """§ 3's other half: it passes both questions, so it is ordinary.
    What made it look special is only where it lands (the wrapper)."""
    assert "continue_retries" in {f.name
                                  for f in dataclasses.fields(SiestaConfig)}
    assert "continue_retries" not in {f.name for f in dataclasses.fields(Stage)}
