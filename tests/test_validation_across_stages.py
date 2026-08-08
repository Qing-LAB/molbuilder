"""Validation across a ladder — the members (R2) and the sequence (R3).

Contract: ``docs/engines/stages.md`` § 4 — R2 (*a stage is validated as a
resolved **whole**, never as a diff*; the label travels beside ``where``, never
inside it) and R3 (*the sequence is checked as well as its members*, and a
finding about it carries **no** stage label) · ``docs/science/validation.md``
§ 4.1 (one result type, ``where`` is the stable id, severity means the same
everywhere) · ``docs/engines/tuning.md`` § 2 (**which** parameters must not go
backwards — that document's to say, not this one's).

P2 unit 6.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.issues import Issue
from molbuilder.structure import Structure
from molbuilder.task import Stage
from molbuilder.validation.stages import (
    check_identical_stages,
    check_ladder_does_not_loosen,
    validate_ladder,
)


@pytest.fixture
def h2() -> Structure:
    return Structure(elements=["H", "H"],
                     positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                     vacuum=(12.0, 12.0, 12.0))


@pytest.fixture
def tpl() -> SiestaConfig:
    return SiestaConfig(system_label="JOB", mesh_cutoff=300.0)


def _ladder(*pairs):
    """``_ladder(("a", {...}), ("b", {...}))`` -> stages, in order."""
    return [Stage(name=n, overrides=o) for n, o in pairs]


def _seq(issues):
    return [i for i in issues if i.stage is None]


def _per_stage(issues):
    return [i for i in issues if i.stage is not None]


# --------------------------------------------------------------------- #
#  R2 — each stage, as a resolved whole                                 #
# --------------------------------------------------------------------- #

def test_a_per_stage_finding_carries_the_stage_beside_where(h2, tpl):
    """The label rides in its own field.  Folding it into ``where`` would give
    one check as many ids as the ladder has stages, and the UI binds behaviour
    to the id (`validation.md § 4.1` R1)."""
    issues = validate_ladder(h2, tpl, _ladder(("coarse", {}), ("tight", {})))
    assert _per_stage(issues), "no per-stage findings at all -- test is blind"
    for i in _per_stage(issues):
        assert i.stage in ("coarse", "tight")
        assert "coarse" not in i.where and "tight" not in i.where


def test_each_stage_is_judged_on_its_OWN_resolved_values(h2):
    """R2's substance.  A stage that overrides a value into trouble must be the
    one reported — if the template were validated instead, a ladder could put
    any stage anywhere and be told nothing.

    The perturbation is ``mesh_cutoff`` because it fires for **this**
    structure.  The first version of this test used ``spin_polarized``, which
    keys on open-shell metals and produces nothing at all for H₂ — a test that
    could only ever have failed, and did."""
    tpl = SiestaConfig(system_label="JOB", mesh_cutoff=300.0)
    issues = validate_ladder(h2, tpl, _ladder(
        ("sane", {}),
        ("starved", {"mesh_cutoff": 50.0})))
    starved = {i.where for i in issues if i.stage == "starved"}
    sane = {i.where for i in issues if i.stage == "sane"}
    assert "config.mesh_cutoff" in starved
    assert "config.mesh_cutoff" not in sane


def test_the_shipped_validator_is_what_runs(h2, tpl):
    """Not a parallel per-stage copy.  The old one drifted into a weaker rule
    set than a plain calculation got, which is why it was deleted with
    ``SiestaStageSpec``; this asserts a finding only the shipped validator
    produces reaches a stage."""
    from molbuilder.validation import validate
    single = {i.where for i in validate(h2, tpl)}
    staged = {i.where for i in validate_ladder(h2, tpl, _ladder(("s", {})))}
    assert single and single <= staged


def test_a_disabled_stage_is_not_validated(h2, tpl):
    issues = validate_ladder(h2, tpl, [
        Stage(name="on", enabled=True),
        Stage(name="off", enabled=False, overrides={"mesh_cutoff": 1.0}),
    ])
    assert {i.stage for i in _per_stage(issues)} == {"on"}


def test_the_validators_own_issues_are_not_mutated(h2, tpl):
    """``Issue`` is frozen and the stamp makes a new one, so a caller holding
    the validator's list cannot find stage labels appearing in it."""
    issues = validate_ladder(h2, tpl, _ladder(("a", {}), ("b", {})))
    assert len({id(i) for i in issues}) == len(issues)


# --------------------------------------------------------------------- #
#  R3 — the sequence                                                    #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("field,loose,tight", [
    ("relax_force_tol", 0.05, 0.01),     # tighter is SMALLER
    ("relax_max_displ", 0.30, 0.02),     # tighter is SMALLER
    ("dm_tolerance",    1e-3, 1e-5),     # tighter is SMALLER
    ("mesh_cutoff",     150.0, 500.0),   # tighter is LARGER
])
def test_a_ladder_that_loosens_is_reported(field, loose, tight):
    """The four parameters ``tuning.md`` § 2 gives an explicit tier table for,
    each in its own direction.  A later stage that is coarser discards what the
    earlier one paid for."""
    backwards = [("first", SiestaConfig(**{field: tight})),
                 ("second", SiestaConfig(**{field: loose}))]
    found = check_ladder_does_not_loosen(backwards)
    assert [i.where for i in found] == [f"stages.loosens.{field}"]


@pytest.mark.parametrize("field,loose,tight", [
    ("relax_force_tol", 0.05, 0.01),
    ("relax_max_displ", 0.30, 0.02),
    ("dm_tolerance",    1e-3, 1e-5),
    ("mesh_cutoff",     150.0, 500.0),
])
def test_a_ladder_that_tightens_is_silent(field, loose, tight):
    """The half that stops this being a check that always fires."""
    forwards = [("first", SiestaConfig(**{field: loose})),
                ("second", SiestaConfig(**{field: tight}))]
    assert check_ladder_does_not_loosen(forwards) == []


def test_a_ladder_that_holds_a_parameter_steady_is_silent():
    """Ordinary: most stages change one or two things.  And a value that has
    round-tripped through a template must not read as a step backwards."""
    same = [("a", SiestaConfig(mesh_cutoff=300.0)),
            ("b", SiestaConfig(mesh_cutoff=300.0))]
    assert check_ladder_does_not_loosen(same) == []


def test_a_sequence_finding_carries_NO_stage_label():
    """**R3's whole point.**  It is a fact about the description, not about a
    member of it — naming one of the two stages would blame a member for a
    property of the pair."""
    found = check_ladder_does_not_loosen([
        ("first", SiestaConfig(mesh_cutoff=500.0)),
        ("second", SiestaConfig(mesh_cutoff=150.0))])
    assert found and all(i.stage is None for i in found)


def test_a_loosening_ladder_warns_and_never_blocks():
    """`validation.md § 4.1`: adequacy is advisory, representability is
    blocking.  A loosening ladder runs perfectly well — what is in question is
    whether it is what the user meant, and deliberately loosening is
    legitimate (a cheap wide scan after an expensive refinement)."""
    found = check_ladder_does_not_loosen([
        ("first", SiestaConfig(mesh_cutoff=500.0)),
        ("second", SiestaConfig(mesh_cutoff=150.0))])
    assert found and all(i.severity == "warn" for i in found)


def test_one_finding_per_parameter_not_one_per_adjacent_pair():
    """A three-stage ladder built entirely backwards is ONE mistake.  Reporting
    each adjacent pair would read as two problems and bury the message."""
    found = check_ladder_does_not_loosen([
        ("a", SiestaConfig(mesh_cutoff=500.0)),
        ("b", SiestaConfig(mesh_cutoff=350.0)),
        ("c", SiestaConfig(mesh_cutoff=150.0))])
    assert len(found) == 1


def test_each_loosening_parameter_gets_its_own_id():
    """`validation.md § 4.1` R1: ``where`` is the stable machine-readable id,
    and two faults with two different repairs must not share one — the lesson
    of ``cell.determinant`` being split into ``cell.no_volume`` and
    ``cell.left_handed``."""
    found = check_ladder_does_not_loosen([
        ("a", SiestaConfig(mesh_cutoff=500.0, relax_force_tol=0.01)),
        ("b", SiestaConfig(mesh_cutoff=150.0, relax_force_tol=0.05))])
    wheres = [i.where for i in found]
    assert len(wheres) == len(set(wheres)) == 2


def test_the_message_names_both_stages_and_both_values():
    """A finding a user can act on: which two stages, which way round, and the
    tier ladder it is being measured against."""
    found = check_ladder_does_not_loosen([
        ("warmup", SiestaConfig(mesh_cutoff=500.0)),
        ("final", SiestaConfig(mesh_cutoff=150.0))])
    msg = found[0].message
    for bit in ("warmup", "final", "500", "150", "mesh_cutoff", "tuning.md"):
        assert bit in msg, (bit, msg)


def test_a_parameter_tuning_md_gives_no_tier_ladder_for_is_NOT_checked():
    """**Deliberate, and the reason matters.**  ``tuning.md`` has tier tables
    for four parameters and none for ``basis_size`` or ``pao_energy_shift``.
    A monotonic direction invented here would be a scientific claim with no
    source — and a false "your ladder loosens" is worse than a missing one,
    because the first teaches users to ignore the check."""
    found = check_ladder_does_not_loosen([
        ("a", SiestaConfig(basis_size="TZP", pao_energy_shift=0.001)),
        ("b", SiestaConfig(basis_size="SZ", pao_energy_shift=0.05))])
    assert found == []


def test_the_sequence_is_checked_in_ladder_order_not_sorted():
    """The order IS the thing being checked, so the caller's order is used as
    given.  Sorting by name would call a correctly-tightening ladder broken
    whenever its stage names happened to sort the other way."""
    tighten = [("z_first", SiestaConfig(mesh_cutoff=150.0)),
               ("a_second", SiestaConfig(mesh_cutoff=500.0))]
    assert check_ladder_does_not_loosen(tighten) == []


# --------------------------------------------------------------------- #
#  The wire                                                             #
# --------------------------------------------------------------------- #

def test_the_stage_reaches_the_json_beside_where():
    from molbuilder.web.blueprints._shared import issues_to_json
    [d] = issues_to_json([Issue("warn", "m", "config.mesh_cutoff",
                                stage="tight")])
    assert d["stage"] == "tight" and d["where"] == "config.mesh_cutoff"


def test_a_single_run_response_is_unchanged():
    """A finding with no stage must serialise exactly as it did before ladders
    existed — every existing wire-shape test pins these keys literally."""
    [d] = issues_to_json_single()
    assert "stage" not in d
    assert set(d) == {"severity", "message", "where"}


def issues_to_json_single():
    from molbuilder.web.blueprints._shared import issues_to_json
    return issues_to_json([Issue("warn", "m", "geometry.min_distance")])


def test_issue_still_defaults_its_stage_to_none():
    assert Issue("info", "m").stage is None
    assert "stage" in {f.name for f in dataclasses.fields(Issue)}


# --------------------------------------------------------------------- #
#  § 6.6a — two stages that resolve to the same thing                   #
# --------------------------------------------------------------------- #

def _pair(a_restart, b_restart, **b_over):
    return [("first", SiestaConfig(restart=a_restart)),
            ("second", SiestaConfig(restart=b_restart, **b_over))]


def test_identical_stages_are_allowed_when_the_later_one_continues():
    """**The case the rule exists to protect.**  `tight` then `tight` where
    the second continues is *more steps at these settings* — the honest way to
    say keep going after a stage ran out of its step budget.  Refusing it
    would make someone invent a token difference to get past the check, which
    is worse than the thing being prevented."""
    assert check_identical_stages(_pair("clean", "continue")) == []


def test_identical_stages_warn_when_the_later_one_starts_clean():
    """It recomputes what the stage before just produced and throws that
    result away — always a mistake, and an expensive one."""
    [issue] = check_identical_stages(_pair("clean", "clean"))
    assert issue.severity == "warn"
    assert issue.where == "stages.recomputes_previous"


def test_it_warns_when_an_earlier_stage_continued_and_the_later_cleans():
    """``restart`` is the DISCRIMINATOR, so it is not part of the equality
    test.  Reading § 6.6a the other way — equality including ``restart`` —
    would make its second clause redundant and would miss this, which is a
    real recompute: the second stage redoes the first from scratch."""
    assert check_identical_stages(_pair("continue", "clean"))


def test_stages_that_differ_in_any_setting_do_not_warn():
    assert check_identical_stages(_pair("clean", "clean",
                                        mesh_cutoff=999.0)) == []


def test_the_comparison_is_over_the_RESOLVED_pair_not_the_overrides(h2, tpl):
    """Comparing overrides would flag the legitimate case and miss nothing —
    which is how a warning becomes noise people learn to click through.  Here
    the two stages carry DIFFERENT override maps and resolve to the same
    thing, so an overrides-based test would stay silent where this one
    speaks."""
    issues = validate_ladder(h2, SiestaConfig(system_label="JOB",
                                              mesh_cutoff=300.0,
                                              restart="clean"), _ladder(
        ("a", {"mesh_cutoff": 300.0}),      # restates the template's value
        ("b", {"restart": "clean"})))       # a different map, same result
    assert "stages.recomputes_previous" in [i.where for i in _seq(issues)]


def test_only_ADJACENT_stages_are_compared():
    """*"the stage before it"*.  Two identical stages with a different one
    between them do not recompute each other's output."""
    assert check_identical_stages([
        ("a", SiestaConfig(restart="clean")),
        ("b", SiestaConfig(restart="clean", mesh_cutoff=999.0)),
        ("c", SiestaConfig(restart="clean")),
    ]) == []


def test_the_warning_carries_no_stage_label():
    """A fact about a PAIR is not a fact about a member of it — the same rule
    § 4 R3 gives the loosening check."""
    [issue] = check_identical_stages(_pair("clean", "clean"))
    assert issue.stage is None


def test_the_warning_says_what_to_do_about_it():
    """It is *"this is probably not what you meant"*, and the repair is one
    field, so the message names it."""
    [issue] = check_identical_stages(_pair("clean", "clean"))
    assert "continue" in issue.message
    assert "first" in issue.message and "second" in issue.message


def test_it_never_refuses():
    """§ 6.6a: *"a warning, not a preflight row"* — § 6.6's table is refusals
    before anything is written; this one proceeds if it was meant."""
    assert all(i.severity != "error"
               for i in check_identical_stages(_pair("clean", "clean")))
