"""§ 6.6's preflight — the half that needs a field schema.

Contract: ``docs/engines/stages.md`` § 6.6 (eight checks, *"in order, and all
of it before anything is written"*, each naming what it refused) · § 6.7
(`shape` is required and never inferred) · ``docs/science/validation.md`` § 4.1.

The structural rows are ``task.py``'s and are tested in
``test_task_description.py``.  That the codec imports no engine config -- the
reason ``preflight`` takes ``config_cls`` as a parameter -- is `stages.md`
§ 6's rule and review's to hold; what runs here is the preflight answering
for either engine's class.
"""
from __future__ import annotations

import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.issues import ValidationError
from molbuilder.task import Stage, StructureRef, Task, derive_run
from molbuilder.validation.task import preflight, refuse_on_error


def _task(**kw) -> Task:
    # ``derive_run``, not a hand-typed id: ``run.id`` is derived.
    base = dict(engine="siesta", shape="flat", calculation="optimization",
                run=derive_run("opt"),
                structure=StructureRef(source="h2.xyz"))
    base.update(kw)
    return Task(**base)


def _staged(overrides, *, varies=None, **kw) -> Task:
    return _task(varies=tuple(varies if varies is not None else overrides),
                 stages=(Stage(name="tight", overrides=dict(overrides)),),
                 **kw)


def _wheres(issues):
    return [i.where for i in issues]


def _errors(issues):
    return [i for i in issues if i.severity == "error"]


def _override_findings(issues):
    """Only the rows about a stage's ``overrides``.

    ``_staged`` derives ``varies`` from the overrides keys, which is the
    REALISTIC bad input: ``task.py`` enforces ``overrides ⊆ varies`` without a
    schema, so a name misspelt in one place is misspelt in both or the codec
    refuses it first.  A misspelt override with a correct ``varies`` is a state
    no reader can produce, so testing it would be testing an impossible file.
    """
    return [i for i in issues if i.where == "task.stages.overrides"]


# --------------------------------------------------------------------- #
#  A description with nothing wrong                                     #
# --------------------------------------------------------------------- #

def test_a_clean_description_produces_nothing():
    """The half that stops every other test here being vacuous."""
    assert preflight(_staged(
        {"mesh_cutoff": 300.0},
        )) == []


# --------------------------------------------------------------------- #
#  Row 1 — the engine has a generator                                   #
# --------------------------------------------------------------------- #

def test_an_engine_with_no_generator_is_refused_naming_what_there_is():
    [issue] = preflight(_task(engine="vasp"))
    assert issue.severity == "error" and issue.where == "task.engine"
    assert "vasp" in issue.message
    assert "siesta" in issue.message and "pyscf" in issue.message


def test_an_unknown_engine_stops_there_rather_than_cascading():
    """Every later row asks a question about *that engine's schema*.  Running
    them against a guess would bury the one finding that matters under a list
    of consequences of it."""
    issues = preflight(_task(engine="vasp", varies=("nonsense",),
                             stages=(Stage(name="a",
                                           overrides={"nonsense": 1}),)))
    assert len(issues) == 1


def test_the_generator_table_is_an_argument_so_a_test_need_not_invent_one():
    assert preflight(_task(engine="siesta"), generators={}) != []
    assert preflight(_task(engine="siesta"),
                     generators={"siesta": SiestaConfig}) == []


# --------------------------------------------------------------------- #
#  § 6.7 — every engine offers both shapes                              #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("engine", ("siesta", "pyscf"))
@pytest.mark.parametrize("shape", ("flat", "hierarchical"))
def test_both_shapes_are_offered_by_both_engines(engine, shape):
    """`stages.md` § 6.7: how a calculation's files are kept apart is a question
    about the CALCULATION, not about the engine -- and the layer that answers it
    names no engine.

    A PySCF ladder is N decks and N jobs (§ 1.1a), so there is a directory
    per rung to lay out, exactly as there is for SIESTA.
    """
    assert preflight(_task(engine=engine, shape=shape)) == []


# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #


def test_an_overrides_key_that_is_no_field_is_refused_by_name():
    [issue] = _override_findings(preflight(_staged({"mesh_cutof": 300.0})))
    assert issue.severity == "error"
    assert "mesh_cutof" in issue.message
    assert issue.stage == "tight"


def test_the_refusal_suggests_the_field_they_probably_meant():
    """A description is JSON people edit by hand, so a refusal owes them the
    near miss when there is an obvious one."""
    [issue] = _override_findings(preflight(_staged({"mesh_cutof": 300.0})))
    assert "mesh_cutoff" in issue.message


def test_varies_is_checked_too_not_only_the_overrides():
    """**The case only this half can catch.**  ``task.py`` enforces
    ``overrides ⊆ varies`` with no schema, so a name misspelt *the same way in
    both* satisfies the subset and passes the codec cleanly — both are wrong,
    and consistently so."""
    issues = preflight(_staged({"mesh_cutof": 300.0}))
    assert "task.varies" in _wheres(issues)


def test_a_varies_name_that_no_stage_overrides_is_still_checked():
    """``varies`` is what the surface draws a column for.  A misspelt one with
    no override yet is a column that can never be filled."""
    issues = preflight(_task(varies=("mesh_cutof",),
                             stages=(Stage(name="a"),)))
    assert _wheres(issues) == ["task.varies"]


def test_a_field_level_finding_carries_the_stage_it_came_from():
    [issue] = _override_findings(preflight(_staged({"nonsense_field": 1})))
    assert issue.stage == "tight"


# --------------------------------------------------------------------- #
#  Row 4 — every value is inside the schema's bounds                    #
# --------------------------------------------------------------------- #

# A value outside the recommended range is warned where the description is
# saved: `test_hard_limits.py`, on the Task-setup save.

def test_a_value_at_either_bound_is_accepted():
    """Inclusive, as § 3.3's ``range=[a,b]`` says.  An off-by-one here would
    refuse the exact value the schema advertises as legal."""
    assert preflight(_staged({"mesh_cutoff": 100.0})) == []
    assert preflight(_staged({"mesh_cutoff": 1000.0})) == []


def test_a_value_outside_an_enums_choices_is_refused_naming_them():
    [issue] = preflight(_staged({"basis_size": "NOPE"}))
    assert issue.severity == "error" and issue.where == "config.basis_size"
    assert "DZP" in issue.message and "TZP" in issue.message


def test_a_legal_enum_value_is_accepted():
    assert preflight(_staged({"basis_size": "TZP"})) == []


def test_a_field_with_no_declared_bound_is_not_checked():
    """The schema is the authority on what a bound is.  Inventing one here
    would refuse a description for breaking a rule nobody wrote."""
    assert preflight(_staged({"system_label": "anything_at_all"})) == []


def test_a_bad_value_is_not_reported_twice_for_a_bad_name():
    """A key that is not a field cannot also be out of bounds — reporting both
    would give one mistake two findings and two repairs."""
    assert len(preflight(_staged({"mesh_cutof": 99999.0}))) == 2  # name x2


# --------------------------------------------------------------------- #
#  Refusing                                                             #
# --------------------------------------------------------------------- #

def test_refuse_on_error_raises_for_an_error():
    with pytest.raises(ValidationError):
        refuse_on_error(preflight(_staged({"kgrid": [0, 4, 4]})))


def test_refuse_on_error_passes_warnings_through():
    """The preflight reports; it does not stop a produce — the one row that
    proceeds."""
    issues = preflight(_task())
    assert refuse_on_error(issues) == issues


def test_refuse_on_error_carries_every_error_not_just_the_first():
    """§ 6.6 lists the checks *in order* and runs all of them: a person fixing
    a description by hand should see the whole list, not one round trip per
    mistake."""
    issues = preflight(_staged({"kgrid": [0, 4, 4], "basis_size": "NOPE"}))
    with pytest.raises(ValidationError) as e:
        refuse_on_error(issues)
    assert len(e.value.issues) == 2


# --------------------------------------------------------------------- #
#  The split                                                            #
# --------------------------------------------------------------------- #

def test_pyscf_gets_the_same_treatment_as_siesta():
    """The preflight is engine-agnostic: it asks the config class, and both
    ship one.  A check that only ever ran for SIESTA would be a third place
    the two engines diverge."""
    issues = preflight(_task(engine="pyscf", varies=("not_a_pyscf_field",),
                             stages=(Stage(name="a"),)),
                       config_cls=PySCFConfig)
    assert _wheres(issues) == ["task.varies"]


# --------------------------------------------------------------------- #
#  Row 4a — a value the field can actually hold                         #
#  (found by the M2 seam walk, 2026-08-07)                              #
# --------------------------------------------------------------------- #

def test_a_fractional_value_for_a_counting_field_is_refused():
    """**The defect the seam walk found.**  ``relax_steps`` is declared
    ``int`` and 100.7 is inside its range, so the bounds row passed it and it
    would reach the deck as ``MD.Steps 100.7``.  Neither side could see it:
    ``task.py`` has no schema, and ``effective_config`` had no reason to think
    a value might not be one the field can hold."""
    [issue] = preflight(_staged({"relax_steps": 100.7}))
    assert issue.severity == "error" and issue.where == "config.relax_steps"
    assert "100.7" in issue.message and "whole number" in issue.message


def test_a_whole_valued_float_is_accepted_for_a_counting_field():
    """``100.0`` IS an integer.  Refusing it would reject a description JSON
    round-tripping had every right to produce."""
    assert preflight(_staged({"relax_steps": 100.0})) == []


def test_an_int_is_accepted_for_a_float_field():
    """JSON has one number, so ``150`` for a float field is the same value
    written differently -- widened on resolve, never refused."""
    assert preflight(_staged({"mesh_cutoff": 150})) == []


def test_a_string_is_refused_for_a_number():
    """A quoting slip is a mistake, not a value: coercing it would make the
    slip invisible."""
    assert _errors(preflight(_staged({"mesh_cutoff": "300"})))


def test_a_bool_is_refused_for_a_number():
    """``bool`` is a subclass of ``int`` in Python, so ``True`` would slide
    into a numeric field as 1 without this."""
    assert _errors(preflight(_staged({"relax_steps": True})))


def test_a_number_is_refused_for_a_boolean_field():
    assert _errors(preflight(_staged({"copy_psml": 1})))


def test_a_legal_boolean_is_accepted():
    assert preflight(_staged({"copy_psml": True})) == []



# --------------------------------------------------------------------- #
#  Row 4a, the SEQUENCE half -- added 2026-08-25 after it shipped broken #
# --------------------------------------------------------------------- #
#
#  The row above checked three declared types: ``int``, ``float``, ``bool``.
#  It did it by comparing ``f.type`` -- the SOURCE TEXT, under
#  ``from __future__ import annotations`` -- against those three strings, so
#  every ``Optional[...]`` and every sequence field matched nothing and was
#  waved through.  A description carrying ``"kgrid": "4,4,1"`` (which is what
#  the Task-setup stage table wrote for any non-bool cell) saved cleanly here,
#  resolved into a config holding a str where ``Tuple[int, int, int]`` is
#  declared, and failed at ``prep`` inside the metadata range check as *"this
#  is a programmer bug"* -- naming neither the stage nor the key.  Reported
#  live against Au-BDT-Au, 2026-08-25.


def test_the_comma_text_a_person_types_is_refused_for_a_triple():
    """THE LIVE DEFECT.  ``4,4,1`` is the right thing to TYPE -- it is one of
    the three spellings ``--kgrid`` itself takes -- and the wrong thing to
    STORE, because JSON has no tuple and a description spells a triple as a
    list.  So the refusal must name the list to write, or it sends someone
    who typed the correct value away with nothing to change."""
    [issue] = preflight(_staged({"kgrid": "4,4,1"}))
    assert issue.severity == "error" and issue.where == "config.kgrid"
    assert issue.stage == "tight"
    assert "'4,4,1'" in issue.message
    assert "three whole numbers" in issue.message
    assert "[4, 4, 1]" in issue.message


def test_a_list_is_accepted_for_a_triple():
    """JSON's only spelling of a triple.  Refusing it would refuse every
    description the browser and the CLI can write."""
    assert preflight(_staged({"kgrid": [4, 4, 1]})) == []


def test_a_triple_of_the_wrong_length_is_refused_by_its_length():
    """Two counts is not a k-grid, and saying *"it has 2, not 3"* is the
    difference between a fixable message and a puzzle."""
    [issue] = preflight(_staged({"kgrid": [4, 4]}))
    assert "2, not 3" in issue.message


def test_a_fractional_component_is_refused_for_an_int_triple():
    """The counting argument from ``relax_steps``, one shape up: a k-grid
    axis is a COUNT of points, so 1.5 of one is not a sampling anybody
    described."""
    [issue] = preflight(_staged({"kgrid": [4, 4, 1.5]}))
    assert "1.5" in issue.message


def test_a_float_triple_takes_ints_per_component():
    """TOML's ``0`` and ``0.0`` are different types to a parser and the same
    displacement to SIESTA -- the same argument the int -> float widening
    makes for a scalar."""
    assert preflight(_staged({"kgrid_displacement": [0, 0, 0]})) == []


def test_comma_text_is_refused_for_a_list_field():
    """``species_order`` had the same hole for the same reason: a str IS a
    sequence, so every check that asked only *"is it a sequence"* said yes."""
    [issue] = preflight(_staged({"species_order": "Au,C,H,S"}))
    assert issue.where == "config.species_order"
    assert "['Au', 'C', 'H', 'S']" in issue.message


def test_a_run_card_value_the_kind_does_not_carry_is_refused():
    """A run card's value is read by the kind's decks or by nothing
    (`template.md` § 6.3), as an override is (K4): `restart` is an
    optimization's, and on a vibration's card it was pinned into a deck that
    ignores it (the K5 review's C2, 2026-09-30)."""
    issues = _errors(preflight(_task(calculation="vibration",
                                     execution={"restart": "clean"})))
    assert any("'restart'" in i.message and "does not carry" in i.message
               for i in issues), issues


def test_an_optional_field_still_checks_the_type_inside_it():
    """The other half of the source-text bug.  ``Optional[float]`` is not
    the string ``"float"``, so an optional field accepted anything at all."""
    assert _errors(preflight(_staged({"md_target_temperature": "300"})))


def test_none_is_a_legal_value_for_an_optional_field():
    """An optional field unset is a real state the config declares.
    Checking the inner type must not cost it."""
    assert preflight(_staged({"md_target_temperature": None})) == []


# --------------------------------------------------------------------- #
#  A-9 — a machine fact refuses with § 7's story, not the typo story     #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("name, value", [("mpi_np", 8), ("use_gpu", True)])
def test_a_run_setting_as_a_column_is_refused_naming_the_run_card(name,
                                                                 value):
    """A run setting lives on the rung's run card (`stages.md` § 6.8d, plan
    § 5w K5) -- the machine's answer (A-9, 2026-08-13: a description varying
    ``mpi_np`` rendered a deck for a rank count the allocation never
    granted) and a person's alike (SO-C1, 2026-09-30: a rung's ``use_gpu``
    reached its deck and not the scheduler's device ask).  ``varies`` and
    the stages travel together (§ 6.5), so both rows are refused, each
    naming the card -- the person typing it is mid-mistake about where it
    goes."""
    issues = _errors(preflight(_staged({name: value})))
    for where in ("task.varies", "task.stages.overrides"):
        assert any(i.where == where and "run card" in i.message
                   and repr(name) in i.message for i in issues), (where,
                                                                  issues)
