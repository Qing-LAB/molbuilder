"""Tests for molbuilder.validation.__init__.

Per docs/process/testing.md (test layout mirrors source
layout).
"""

from __future__ import annotations

import io

import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.validation import report


def test_issue_severity_accepts_error_warn_info():
    """Severity is restricted to error / warn / info.  Info is for
    advisory hints (e.g. 'Fe with 2S = 4: high-spin Fe(II)') that don't
    add to the warn count."""
    Issue("error", "fine")
    Issue("warn",  "fine")
    Issue("info",  "fine")
    with pytest.raises(ValueError, match="severity"):
        Issue("debug", "not allowed")


def test_validation_error_carries_issues():
    issues = [
        Issue("warn", "minor", "x"),
        Issue("error", "fatal", "y"),
    ]
    with pytest.raises(ValidationError) as exc:
        raise ValidationError(issues)
    assert exc.value.issues == issues
    # Message lists the error but not the warn.
    assert "fatal" in str(exc.value)
    # The warning should NOT be in the formatted error message;
    # warnings get their own stderr path via report().
    assert "minor" not in str(exc.value)


def test_validation_error_rejects_empty_or_warn_only():
    with pytest.raises(ValueError, match="error-severity"):
        ValidationError([])
    with pytest.raises(ValueError, match="error-severity"):
        ValidationError([Issue("warn", "just a warning")])


# --------------------------------------------------------------------- #
#  report() helper: warnings to stderr, raise on errors                 #
# --------------------------------------------------------------------- #


def test_report_prints_warnings_to_stream():
    buf = io.StringIO()
    report(
        [Issue("warn", "watch out", "test.case")],
        raise_on_error=False, stream=buf,
    )
    out = buf.getvalue()
    assert "watch out" in out
    assert "[test.case]" in out


def test_report_raises_on_error_by_default():
    with pytest.raises(ValidationError):
        report([Issue("error", "fatal", "x")])


def test_report_emits_warnings_even_when_also_raising():
    """A run with both warnings and errors should surface BOTH -- the
    user wants to see all the warnings even if the error blocks
    emission."""
    buf = io.StringIO()
    issues = [
        Issue("warn", "minor first", "a"),
        Issue("warn", "minor second", "b"),
        Issue("error", "fatal", "c"),
    ]
    with pytest.raises(ValidationError):
        report(issues, stream=buf)
    out = buf.getvalue()
    assert "minor first" in out
    assert "minor second" in out


# --------------------------------------------------------------------- #
#  ONE validation gate per engine and per kind (V1/V2 --                #
#  backend-architecture.md)                                             #
# --------------------------------------------------------------------- #


def test_the_two_registries_hold_what_the_contract_says():
    """`science/validation.md` § 7: an engine's checks are the part its
    record in the engine table names (``validator``), a kind's science the
    part its kind record names -- a vibration's science and a junction's
    are the KIND's.

    Asserted by EQUALITY, not containment, so a row cannot be added and
    never noticed.
    """
    from molbuilder.engine_table import engine, engines
    from molbuilder.kind_table import kind, kinds, part
    for e in engines():
        assert callable(part(engine(e).validator, record=f"engine {e}",
                             key="validator")), (
            f"engine {e}'s validator does not resolve -- every engine's "
            f"config is checked by the part its record names")
    named = {k for k in kinds() if kind(k).validator}
    assert named == {"vibration", "transport"}, (
        "a calculation kind lost its science -- this is the road a transport "
        "or vibration prep actually travels")
    for k in named:
        assert callable(part(kind(k).validator, record=k, key="validator"))


#: (engine's config, calculation, the setting given, what it waits on) -- a
#: value given to a setting whose `active_when` the other settings do not
#: meet (`engines/template.md` § 5): a PySCF solvent model with no solvent
#: (`active_when = { solvent = "set" }`), a SIESTA step count with no
#: relaxation (`relax_type = "none"`).
INACTIVE_ROWS = [
    pytest.param("pyscf", "vibration", {"solvent_method": "C-PCM"},
                 "solvent_method", "solvent", 1,
                 id="solvent-model-without-a-solvent"),
    pytest.param("siesta", "optimization",
                 {"relax_type": "none", "relax_steps": 77}, "relax_steps",
                 "relax_type", 1, id="steps-without-a-relaxation"),
    pytest.param("pyscf", "vibration", {"method": "HF", "functional": "PBE0"},
                 "functional", "method", 1, id="a-functional-under-hartree-fock"),
    pytest.param("pyscf", "optimization", {"method": "HF", "grid_level": 3},
                 "grid_level", "method", 1, id="a-grid-under-hartree-fock"),
    # ...and a value at the kind's default is nobody's statement: nothing.
    pytest.param("pyscf", "vibration", {}, "solvent_method", "solvent", 0,
                 id="solvent-model-at-its-default-says-nothing"),
]


@pytest.mark.parametrize("engine, calculation, given, name, waits_on, said",
                         INACTIVE_ROWS)
def test_a_setting_waiting_on_another_is_said_once_never_refused(
        water_structure, engine, calculation, given, name, waits_on, said):
    """The settings gate says once that the value has no effect here, in
    the catalogue's words for what it waits on, and refuses nothing for
    it -- the form greys the same field (`form-schema.md` § 1.3).

    THE SILENT FAILURE: a page lock or an engine's own warning standing in
    for the catalogue's rule -- a second sentence, or none, for the same
    fact (until 2.65.0 the PySCF solvent model was warned by the spectra
    checks and the SIESTA steps were greyed by the page alone)."""
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.validation import validate
    cfg = (PySCFConfig if engine == "pyscf" else SiestaConfig)(**given)
    issues = validate(water_structure, cfg, calculation=calculation)
    about = [i for i in issues if i.where == f"config.{name}"]
    assert [i.severity for i in about] == ["warn"] * said, about
    # ...naming the setting it waits on, in the why after the value.
    assert all(waits_on in i.message.split("has no effect here:", 1)[1]
               for i in about), about



#: (engine, kind, the values given, the value refused, a setting waiting
#: on it): a value refused draws that refusal alone (`engines/template.md`
#: § 5.3) -- one error -- and a setting that waits on it is not said to
#: enter nothing, since what it waits on is not what will run.
REFUSED_ROWS = [
    pytest.param("siesta", "vibration",
                 {"relax_type": "none", "relax_steps": 77}, "relax_type",
                 "relax_steps", id="a-relaxer-a-vibration-does-not-offer"),
]


@pytest.mark.parametrize("engine, calculation, given, refused, waiting",
                         REFUSED_ROWS)
def test_a_refused_value_draws_its_refusal_alone(
        water_structure, engine, calculation, given, refused, waiting):
    """THE SILENT FAILURE (until 2.65.0): the gate said a refused value
    entered nothing too, and so the settings waiting on it -- three
    findings for one statement, two of them about a run that will not
    happen."""
    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.validation import validate
    cfg = (PySCFConfig if engine == "pyscf" else SiestaConfig)(**given)
    issues = validate(water_structure, cfg, calculation=calculation)
    assert [i.severity for i in issues
            if i.where == f"config.{refused}"] == ["error"], issues
    assert not [i for i in issues if i.where == f"config.{waiting}"
                and "has no effect here" in i.message], issues