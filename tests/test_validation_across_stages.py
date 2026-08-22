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
)


def _ladder(*pairs):
    """``_ladder(("a", {...}), ("b", {...}))`` -> stages, in order."""
    return [Stage(name=n, overrides=o) for n, o in pairs]


# --------------------------------------------------------------------- #
#  R2 — each stage, as a resolved whole                                 #
# --------------------------------------------------------------------- #
# The per-stage door is the RENDER gate: every rung's deck runs the shipped
# validator (with the calculation kind) at render_deck step 3.3, one rung at
# a time.  The batch aggregator `validate_ladder` that this section tested
# retired 2026-08-21 (G-1a): no production caller, and its validate() call
# omitted `calculation`.  The gate's own behaviour is pinned where the gate
# is: tests/test_prep_calculation.py and the engines' render tests.


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


def test_the_comparison_is_over_the_RESOLVED_pair_not_the_overrides():
    """Comparing overrides would flag the legitimate case and miss nothing —
    which is how a warning becomes noise people learn to click through.  Here
    the two stages carry DIFFERENT override maps and resolve to the same
    thing, so an overrides-based test would stay silent where this one
    speaks.  Resolution runs through the production primitive
    (`resolve.effective_config` — the same one `resolved_ladder` applies on
    the preflight route) before the comparison sees the pair."""
    from molbuilder.resolve import effective_config
    base = SiestaConfig(system_label="JOB", mesh_cutoff=300.0,
                        restart="clean")
    resolved = [
        ("a", effective_config(base, {"mesh_cutoff": 300.0})),  # restates
        ("b", effective_config(base, {"restart": "clean"}))]    # same result
    issues = check_identical_stages(resolved)
    assert "stages.recomputes_previous" in [i.where for i in issues]


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


# --------------------------------------------------------------------- #
#  A-8 — the sequence findings are REACHABLE, not merely implemented     #
# --------------------------------------------------------------------- #

def test_the_sequence_warnings_reach_the_prep_surface(tmp_path):
    """A-8 (final review, 2026-08-13): `stages.md` :884 said § 6.6a was
    "implemented at validation/stages.py::check_identical_stages" while NO
    production surface called it — `validate_ladder`'s only callers were
    tests, so the warning was unreachable end-to-end.  The sequence checks
    now ride the § 6.6 preflight when the template is in hand, and `prep`
    is the surface that has both halves.  Pinned through the CLI: the note
    must reach the person about to pay for the recompute."""
    import json

    from click.testing import CliRunner

    from molbuilder import describe as D
    from molbuilder.jobset._cli import jobset_group
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(
            struct, SiestaConfig(system_label="JOB"),
            # Stage b resolves identically AND starts clean -- the one case
            # § 6.6a warns about, and since 2026-08-18 the only way to reach
            # it: `continue` is the default, and two identical rungs where the
            # second continues are simply *more steps at these settings*.
            _ladder(("a", {"mesh_cutoff": 200.0}),
                    ("b", {"mesh_cutoff": 200.0, "restart": "clean"})),
            engine="siesta", shape="hierarchical",
            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, ["H"])
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "a", "--bundle", str(dest),
                                  "--no-sbatch"])
    assert res.exit_code == 0, res.output
    assert "recomputes" in res.output, res.output
