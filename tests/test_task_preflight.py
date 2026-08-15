"""§ 6.6's preflight — the half that needs a field schema.

Contract: ``docs/engines/stages.md`` § 6.6 (eight checks, *"in order, and all
of it before anything is written"*, each naming what it refused) · § 6.7
(`shape` is required and never inferred) · ``docs/science/validation.md`` § 4.1.

P2 unit 7. The other four rows are ``task.py``'s and are tested in
``test_task_description.py``; the split itself is asserted at the foot of this
file, because two halves of one list are exactly the thing that drifts.
"""
from __future__ import annotations

import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.issues import ValidationError
from molbuilder.task import Stage, StructureRef, Task, derive_run
from molbuilder.validation.task import preflight, refuse_on_error


def _task(**kw) -> Task:
    # ``derive_run``, not a hand-typed id.  This read
    # ``Run(name="opt", id="opt-0001")`` until 2026-08-09 -- an id nothing
    # derives, which was legal only while ``run.id`` was a free string.
    base = dict(engine="siesta", shape="flat",
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
#  Row 1a — § 6.7, a one-process ladder has no hierarchy                #
# --------------------------------------------------------------------- #

def test_hierarchical_on_a_one_process_engine_is_refused_naming_it():
    """PySCF runs its whole ladder inside a single process, so a directory per
    stage describes a layout that never exists on disk."""
    [issue] = preflight(_task(engine="pyscf", shape="hierarchical"))
    assert issue.severity == "error" and issue.where == "task.shape"
    assert "pyscf" in issue.message and "flat" in issue.message


def test_hierarchical_is_fine_on_an_engine_whose_stages_are_real_jobs():
    assert preflight(_task(engine="siesta", shape="hierarchical")) == []


def test_flat_is_fine_on_a_one_process_engine():
    assert preflight(_task(engine="pyscf", shape="flat")) == []


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

def test_a_value_outside_a_numeric_range_is_refused_naming_both_bounds():
    [issue] = preflight(_staged({"mesh_cutoff": 99999.0}))
    assert issue.severity == "error" and issue.where == "config.mesh_cutoff"
    assert "99999" in issue.message
    assert "100" in issue.message and "1000" in issue.message


def test_a_value_below_the_range_is_refused_too():
    assert _errors(preflight(_staged({"mesh_cutoff": 1.0})))


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


def test_a_boolean_is_not_range_checked_as_a_number():
    """``bool`` is a subclass of ``int``, so a naive numeric check reads
    ``True`` as 1 and refuses it against whatever range the field carries."""
    assert preflight(_staged({"copy_psml": True})) == []


def test_a_bad_value_is_not_reported_twice_for_a_bad_name():
    """A key that is not a field cannot also be out of bounds — reporting both
    would give one mistake two findings and two repairs."""
    assert len(preflight(_staged({"mesh_cutof": 99999.0}))) == 2  # name x2


# --------------------------------------------------------------------- #
#  Refusing                                                             #
# --------------------------------------------------------------------- #

def test_refuse_on_error_raises_for_an_error():
    with pytest.raises(ValidationError):
        refuse_on_error(preflight(_staged({"mesh_cutoff": 99999.0})))


def test_refuse_on_error_passes_warnings_through():
    """The preflight reports; it does not stop a produce — the one row that
    proceeds."""
    issues = preflight(_task())
    assert refuse_on_error(issues) == issues


def test_refuse_on_error_carries_every_error_not_just_the_first():
    """§ 6.6 lists the checks *in order* and runs all of them: a person fixing
    a description by hand should see the whole list, not one round trip per
    mistake."""
    issues = preflight(_staged({"mesh_cutoff": 99999.0, "basis_size": "NOPE"}))
    with pytest.raises(ValidationError) as e:
        refuse_on_error(issues)
    assert len(e.value.issues) == 2


# --------------------------------------------------------------------- #
#  The split                                                            #
# --------------------------------------------------------------------- #

def test_the_codec_still_does_not_import_an_engine():
    """L1.  If ``task.py`` ever reaches for a config class, this half stops
    having a reason to exist and ``tests/test_layering.py`` stops protecting
    anything."""
    import molbuilder.task as t
    src = open(t.__file__).read()
    for banned in ("config.siesta", "config.pyscf", "SiestaConfig",
                   "PySCFConfig"):
        assert banned not in src, banned


def test_both_halves_of_the_split_are_written_down_in_task_py():
    """``task.py``'s docstring carries the same split, *so the two halves
    cannot quietly diverge* — the plan's words.  This asserts the rows this
    module implements are the ones it says belong to P2."""
    import molbuilder.task as t
    # The docstring is hard-wrapped, so compare on collapsed whitespace --
    # otherwise this passes or fails on where a line happened to break.
    doc = " ".join((t.__doc__ or "").split())
    for row in ("the engine has a generator",
                "every named field exists in the schema",
                "every value is inside its bounds"):
        assert row in doc, row


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
    reached the deck as ``MD.NumCGsteps 100.7``.  Neither side could see it:
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
#  A-9 — a machine fact refuses with § 7's story, not the typo story     #
# --------------------------------------------------------------------- #

def test_an_override_naming_a_machine_fact_is_refused_with_the_story():
    """A-9 (final review, 2026-08-13): ``mpi_np`` IS a schema field --
    tagged ``allocation: True``, `engines/template.md` § 7's forbidden
    machine fact -- so the existence check admitted it and a description
    varying it rendered a deck for a rank count the allocation never
    granted.  The refusal names the § 7 road, because the person typing
    it is mid-mistake about exactly that."""
    issues = _errors(preflight(_staged({"mpi_np": 8})))
    assert issues, "a machine-fact override passed the preflight"
    assert any("machine fact" in i.message and "ALLOCATION" in i.message
               for i in issues)


def test_varies_naming_a_machine_fact_is_refused_too():
    # varies and stages travel together (§ 6.5), so the realistic bad
    # input carries the name in both -- and both rows must say so.
    issues = _errors(preflight(_staged({"omp_threads": 4})))
    assert any(i.where == "task.varies"
               and "machine fact" in i.message for i in issues)
    assert any(i.where == "task.stages.overrides"
               and "machine fact" in i.message for i in issues)
