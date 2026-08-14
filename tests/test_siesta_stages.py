"""The shipped SIESTA ladder: where it comes from, and what it is made of.

Contract: ``docs/engines/stages.md`` § 1.1 (**no engine config carries a
stage list**), § 1.3 (the default *selection*), § 2 (a stage has three
fields), § 4 (``template ⊕ overrides``).

**This file replaced one that tested ``SiestaStageSpec`` (2026-08-07, P2
unit 2).**  That class is deleted, so tests naming its eight fields, its
per-field defaults and its validator went with it — a test that pins a
mechanism the contract removed is not coverage, it is a second copy of the
old design that fails the next time the right thing is done.

What survived is every rule that was about the *ladder* rather than about
that class, restated against the mechanism that carries it now:

  * three stages named stage1 / stage2 / stage3 (matches PySCF — the user
    sees one mental model across engines)
  * their values ARE ``SIESTA_STAGE_PRESETS``, so ``--stage {1,2,3}``'s
    one-shot deck and stage N of the ladder cannot drift apart
  * the strategy presets choose which tiers run and nothing else
  * the ladder is a ``task.Stage`` list, and the config has no ``stages``

The checks the old validator made that were genuinely structural — an
empty / all-disabled ladder, duplicate names, a name that is not a
filename — moved to the layers that own them and are asserted in
``tests/test_stage_resolution.py``.
"""
from __future__ import annotations

import dataclasses

import pytest

from molbuilder.config.siesta import (
    SiestaConfig,
    SIESTA_STAGE_PRESETS,
    SIESTA_STAGE_STRATEGY_PRESETS,
    apply_siesta_stage,
)
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.task import Stage


# --------------------------------------------------------------------- #
#  § 1.1 — an engine config carries no stage list                       #
# --------------------------------------------------------------------- #

def test_the_config_has_no_stages_field():
    """**The deletion, asserted directly.**

    ``SiestaConfig.stages`` was the fault, not a detail of it: a config
    that contains a ladder cannot be the ordinary single parameter set
    § 4 resolves *to*, and its presence is what made the form generator
    publish a Python class's fields as the columns a user may vary
    (§ 1.2).  A field re-added here would reopen both."""
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
    meet a stage (web/form-schema.md § 1's callout)."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    sch = dataclass_to_form_schema(SiestaConfig, id_prefix="p")
    kinds = [f["kind"] for s in sch["sections"] for f in s["fields"]]
    assert "stage-table" not in kinds


# --------------------------------------------------------------------- #
#  The shipped ladder                                                   #
# --------------------------------------------------------------------- #

def test_default_ladder_is_three_named_stages():
    stages = default_siesta_stages()
    # The tiers' NAMES, from SIESTA_STAGE_NAMES.  This read
    # ``["stage1", "stage2", "stage3"]`` until 2026-08-10: decision 27 puts the
    # ordinal in the artifact token (``01_coarse``), so a name that is itself a
    # position would say the number twice and the science none.
    assert [s.name for s in stages] == ["coarse", "medium", "tight"]
    assert all(isinstance(s, Stage) for s in stages)


def test_default_ladder_enabled_pattern_matches_pyscf():
    """publishable = stage1 + stage2; stage3 (tight) is opt-in.  Same
    shape as PySCF's default so the two engines read alike."""
    assert [s.enabled for s in default_siesta_stages()] == [True, True, False]


@pytest.mark.parametrize("tier", sorted(SIESTA_STAGE_PRESETS))
def test_each_stages_overrides_are_exactly_that_tiers_preset(tier):
    """**The one that keeps the two paths aligned.**  ``--stage 2``
    overlays ``SIESTA_STAGE_PRESETS[2]`` onto a config; stage2 of the
    ladder overrides with the same dict.  One table, two readers — so a
    tier value can be changed in exactly one place."""
    stage = default_siesta_stages("vib-quality")[tier - 1]
    # Split deliberately: the tier's SCIENCE comes from the shared table and
    # must match it exactly, while `restart` is POSITIONAL and belongs to no
    # tier -- stage2 is 'continue' because something precedes it, not because
    # tier 2 means anything about restarting (run-identity.md § 4 rule 3).
    # Folding them into one dict would let a tier value drift in under cover
    # of the positional one.
    science = {k: v for k, v in stage.overrides.items() if k != "restart"}
    assert science == SIESTA_STAGE_PRESETS[tier]
    assert stage.overrides["restart"] == ("clean" if tier == 1 else "continue")


def test_the_one_shot_overlay_and_the_ladder_agree_field_for_field():
    """The same claim from the other side, through both real code paths
    rather than through the table they share."""
    from molbuilder.resolve import effective_config
    template = SiestaConfig(system_label="JOB")
    for tier, stage in enumerate(default_siesta_stages("vib-quality"), 1):
        overlaid = dataclasses.asdict(apply_siesta_stage(template, tier))
        resolved = dataclasses.asdict(effective_config(
            template, getattr(stage, "overrides", None),
            where=f"stage {stage.name!r}"))
        # `restart` is the ONE field they may differ in, and the difference is
        # the point rather than a leak: `--stage 2` is a single run with
        # nothing before it, so it stays on the template's value, while
        # stage2 OF A LADDER has stage1 in front of it and continues.
        # Asserted as an exact set so a second divergence cannot hide behind
        # this one.
        differing = {k for k in overlaid if overlaid[k] != resolved[k]}
        assert differing <= {"restart"}
        if tier > 1:
            assert differing == {"restart"}
            assert resolved["restart"] == "continue"


def test_a_stage_has_exactly_the_three_fields_of_section_2():
    """§ 2.  Anything else a stage seemed to need turned out to belong to
    the shared schema or to a producer's input (§ 3)."""
    assert [f.name for f in dataclasses.fields(Stage)] == [
        "name", "enabled", "overrides"]


def test_every_field_the_shipped_ladder_varies_exists_in_the_schema():
    """The preflight's rule, applied to what molbuilder itself ships: a
    ladder naming a field the schema does not have is refused, so the
    default must not be one.

    ``varies`` is DERIVED from the overrides, here as everywhere -- § 6.2's
    subset rule is then checked against the thing it was computed from, so
    the two cannot disagree.  There is deliberately no shipped helper that
    computes it: nothing in the backend needs one, and the surface that
    writes ``task.json`` (P10) will know better what shape it wants."""
    known = {f.name for f in dataclasses.fields(SiestaConfig)}
    varied = {k for s in default_siesta_stages("vib-quality")
              for k in s.overrides}
    assert varied <= known
    assert varied == {"relax_type", "relax_steps",
                      "relax_force_tol", "relax_max_displ",
                      # P3 unit 4: the shipped ladder now SAYS it continues
                      # rather than continuing because three booleans
                      # happened to default True (run-identity.md § 4).
                      "restart"}


# --------------------------------------------------------------------- #
#  The strategy presets choose which tiers run, and nothing else        #
# --------------------------------------------------------------------- #

def test_strategy_preset_names_match_pyscf():
    from molbuilder.config.pyscf import STAGE_STRATEGY_PRESETS
    assert (set(SIESTA_STAGE_STRATEGY_PRESETS)
            == set(STAGE_STRATEGY_PRESETS))


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
    ladder -- the mutable-default bug the old ``_default_siesta_stages``
    factory existed to avoid, asserted rather than assumed."""
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

    P7 unit 2 removed those edges, so the policy had no effect left — and it
    was never reachable by a user in the first place, being absent from
    `task.json`'s three stage fields (asserted above). A parameter accepted,
    resolved and then dropped is the *"present but not honoured"* shape this
    phase keeps deleting, so it was subtracted rather than left inert.

    Reinstating a per-stage policy means giving it a home in the description
    **and** a reader that does something with it. This test is what fails if
    half of that lands.
    """
    import molbuilder.siesta.stages as stages_mod

    # The producers this pinned by signature -- stages_to_jobset,
    # build_siesta_stage_bundle -- were DELETED 2026-08-12 (u5): their
    # ABSENCE is now the assertion, and reinstating a per-stage policy
    # means giving it a home in the description first.
    assert not hasattr(stages_mod, "stages_to_jobset")
    assert not hasattr(stages_mod, "build_siesta_stage_bundle")


def test_continue_retries_is_a_shared_field_not_a_stage_one():
    """§ 3's other half: it passes both questions, so it is ordinary.
    What made it look special is only where it lands (the wrapper)."""
    assert "continue_retries" in {f.name
                                  for f in dataclasses.fields(SiestaConfig)}
    assert "continue_retries" not in {f.name for f in dataclasses.fields(Stage)}
