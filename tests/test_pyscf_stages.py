"""The shipped PySCF ladder: where it comes from, and what it is made of.

Contract: ``docs/engines/stages.md`` § 1.1a (a PySCF ladder is N decks and N
jobs, declared in ``task.json``, exactly as SIESTA's is), § 2 (a stage has four
fields), § 4 (``template ⊕ overrides``).

Every rule here is about the *ladder*, stated against the mechanism that
carries it.  **This file is deliberately the twin of ``test_siesta_stages.py``**: the two engines
differ in which parameters a rung varies and in nothing else, so a rule that
holds for one and is untested on the other is where they drift apart.

The per-tier NUMBERS are not asserted here: each rung's deck states them,
asserted by ``tests/data/the_deck.toml``'s rows -- the numbers ``tuning.md``
§ 2.4 / § 2.5 state as their authority.
"""
from __future__ import annotations

import dataclasses

import pytest

from molbuilder.config.pyscf import PySCFConfig, PYSCF_STAGE_PRESETS
from molbuilder.config.stages import STAGE_STRATEGY_PRESETS
from molbuilder.jobset.ladder import engine_ladder


# --------------------------------------------------------------------- #
#  The shipped ladder                                                   #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("tier", sorted(PYSCF_STAGE_PRESETS))
def test_each_stages_overrides_are_exactly_that_tiers_preset(tier):
    """**The one that keeps the science in one place.**  A rung's overrides
    are that tier's preset row and nothing else -- so a tier value can be
    changed in exactly one place, and that place is checked against
    `tuning.md` § 2.4.

    Nothing is added to that row.  ``restart`` is a property of neither the
    tier nor the rung's index -- the folder answers it, at run time
    (`run-identity.md` § 4 rule 3)."""
    stage = engine_ladder("pyscf", "vib-quality")[tier - 1]
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
        for st in engine_ladder("pyscf", strategy):
            assert "restart" not in st.overrides, (strategy, st.name)


def test_every_field_the_shipped_ladder_varies_exists_in_the_schema():
    """The preflight's rule, applied to what molbuilder itself ships: a
    ladder naming a field the schema does not have is refused, so the
    default must not be one."""
    known = {f.name for f in dataclasses.fields(PySCFConfig)}
    varied = {k for s in engine_ladder("pyscf", "vib-quality")
              for k in s.overrides}
    assert varied <= known
    # THE ONLY GUARD against a shipped tier gaining a parameter: a level shift
    # in a rung would change what every run of it computes, and no deck row
    # asks for what a rung does not state.
    assert varied == set(_VARIED)


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
    """§ 1.1a's derivation depends on these being catalogue items, and the
    per-rung numeric checks rest on the ``range`` each item declares (`validation/task.py` checks every
    override against it).  An item without one is a rung that can carry any
    number at all."""
    item = _catalogue_item(key)
    assert item is not None, f"{key} does not apply to pyscf"
    assert item.range is not None, key


@pytest.mark.parametrize("tier", sorted(PYSCF_STAGE_PRESETS))
def test_every_shipped_tier_value_is_inside_its_items_bound(tier):
    """The shipped ladder must pass the check every user ladder passes.
    A default outside its own declared bound would be refused by
    `molbuilder describe` -- of the ladder molbuilder itself ships."""
    for key, value in PYSCF_STAGE_PRESETS[tier].items():
        lo, hi = _catalogue_item(key).range
        assert lo <= value <= hi, f"{key}={value} outside [{lo}, {hi}]"


# --------------------------------------------------------------------- #
#  The strategy presets choose which tiers run, and nothing else        #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy,expected", [
    ("publishable", ["coarse", "medium"]),
    ("loose-only",  ["coarse"]),
    ("vib-quality", ["coarse", "medium", "tight"]),
])
def test_strategy_preset_writes_the_stages_it_runs(strategy, expected):
    assert [s.name for s in engine_ladder("pyscf", strategy)] == expected


def test_strategy_preset_changes_only_which_tiers_run():
    """A preset says which tiers run; it never retunes one.  If it did,
    picking 'loose-only' would silently change what the coarse rung
    computes."""
    a = engine_ladder("pyscf", "loose-only")
    b = engine_ladder("pyscf", "vib-quality")
    assert [s.overrides for s in a] == [s.overrides for s in b][:len(a)]
    assert [s.name for s in a] == [s.name for s in b][:len(a)]


def test_strategy_preset_rejects_unknown_name():
    """It REFUSES rather than falling back.  A silent fallback answers a
    question nobody asked: a misspelled strategy would run a ladder the
    caller did not name, and `molbuilder describe --stage-strategy` is the one
    door a ladder is authored through."""
    with pytest.raises(ValueError, match="unknown PySCF stage strategy"):
        engine_ladder("pyscf", "no-such-preset")


def test_each_call_returns_independent_stages():
    """A caller that disables a stage must not disturb the next caller's
    ladder -- the mutable-default bug the old factory existed to avoid,
    asserted rather than assumed."""
    a, b = engine_ladder("pyscf"), engine_ladder("pyscf")
    assert a is not b
    assert a[0].overrides is not b[0].overrides
    a[0].overrides["basis"] = "def2-QZVP"
    assert "basis" not in b[0].overrides



_UNCHANGED = lambda moved: None


def _an_older_release_numbered():
    import molbuilder
    from molbuilder.deck_record import BUILD_NAME
    return (lambda moved: moved.setattr(molbuilder, "__version__", "0.0.0"),
            {}, f"{BUILD_NAME} = '0.0.0'")


def _an_older_release_recorded_each_settings_own_default():
    from molbuilder import template as T
    from molbuilder.pyscf.script_input import PARAMS_NAME
    gmax = T.one(T.catalogue(), "geom_gmax", engine="pyscf")
    return (lambda moved: moved.setattr(T, "default_of",
                                        lambda item, kind: item.default),
            {}, f"{PARAMS_NAME}['geom_gmax'] = ({gmax.default!r}, ")


def _another_release_worded_the_methods():
    from molbuilder.spectra import methods
    worded = methods.render_methods_md
    return (lambda moved: moved.setattr(
                methods, "render_methods_md",
                lambda *a, **k: worded(*a, **k) + "\n\nWorded otherwise."),
            {}, "Worded otherwise.")


def _another_release_printed_otherwise():
    from molbuilder import script_emit
    return (lambda moved: moved.setattr(script_emit, "PARAMETER_COLUMNS_LINE",
                                        "#   printed otherwise"),
            {}, "#   printed otherwise")


def _another_release_worded_why_the_charge():
    import dataclasses
    from molbuilder import electronic_state
    resolved = electronic_state._charge
    return (lambda moved: moved.setattr(
                electronic_state, "_charge",
                lambda *a, **k: dataclasses.replace(
                    resolved(*a, **k), why="worded otherwise")),
            {}, "# detected: worded otherwise")


def _another_job_request():
    from support.road import RUN_CARD
    threads = RUN_CARD["pyscf"]["threads"] + 1
    return (_UNCHANGED, {"execution": {**RUN_CARD["pyscf"],
                                       "threads": threads}},
            f"'threads': {threads},")


#: WHAT A PySCF DECK SAYS BESIDE WHAT IT COMPUTES (`execution/job-request.md`
#: § 5–6), each the way another release or another job request writes it:
#: the release that wrote it (`deck_record.BUILD_NAME`); what it only reports
#: (the user, 2026-11-12: "ignore the report-only text") -- its parameters
#: record (`model/parse.md` § 5d.3a), whose default column held each
#: setting's own default until 2.69.43 and holds the kind's since (for a
#: vibration, `geom_gmax` 0.00045 then, 0.0002 now), its Methods text, which
#: its docstring holds too, a message it prints, and a comment beside a
#: statement -- the charge's reason, which a release worded otherwise before;
#: and the job request, which its settings copy echoes (§ 5: a launched
#: stage's request may change).  Each row: how the other release is made
#: while prep writes the deck, the request the other calculation is
#: described with, and the words its deck holds that say so.
_BESIDE_THE_CALCULATION = [
    pytest.param(_an_older_release_numbered, id="the-release"),
    pytest.param(_an_older_release_recorded_each_settings_own_default,
                 id="the-parameters-record"),
    pytest.param(_another_release_worded_the_methods, id="the-methods-text"),
    pytest.param(_another_release_printed_otherwise, id="a-printed-message"),
    pytest.param(_another_release_worded_why_the_charge,
                 id="a-comment-beside-a-statement"),
    pytest.param(_another_job_request, id="the-job-request"),
]


@pytest.mark.parametrize("other", _BESIDE_THE_CALCULATION)
def test_what_a_deck_says_beside_the_calculation_is_no_difference(
        tmp_path, monkeypatch, other):
    """`execution/job-request.md` § 5–6: *the test is the engine's input*.  A
    PySCF vibration described in two trees, one deck made as the row says and
    the other as this release makes it with the road's request -- two trees,
    since a prep keeps a deck that differs from the one it holds in stamps
    alone: the two decks differ in what the row names, and are the same
    calculation.  THE FAILURE: a launched PySCF stage refused when prepared
    again for its job request -- at every version move, over the release's
    name; since 2.69.43, over the record of each setting's default; at any
    rewording of what the deck reports; at any change of the request, which
    its settings copy echoes -- over lines that change nothing PySCF
    computes."""
    from molbuilder.deck_record import same_calculation
    from molbuilder.runfiles import deck_role
    from support.road import described, prep_groups, road_tree, stage_file

    def deck(where, make=_UNCHANGED, request=None):
        bundle = described(
            road_tree(tmp_path / where, monkeypatch, engine="pyscf"),
            calculation="vibration", engine="pyscf", **(request or {}))
        with monkeypatch.context() as moved:
            make(moved)
            prep_groups(bundle, "1")
        return stage_file(bundle, "freq", deck_role("pyscf")).read_text()
    make, request, said = other()
    ran, now = deck("then", make, request), deck("now")
    assert said in ran and said not in now, said
    assert same_calculation("pyscf", ran, now) == []
