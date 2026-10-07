"""Tests for molbuilder.validation.__init__.

Per docs/process/testing.md (test layout mirrors source
layout).  Shared fixtures live in tests/validation/conftest.py.
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
    """`science/validation.md`: an ENGINE row keys on a config class, a KIND
    row on `task.calculation` -- and which registry a science belongs in is
    decided by whether production constructs that class.

    TWO engine rows: a vibration's science and a junction's are the KIND's.

    Asserted by EQUALITY, not containment, so a row cannot be added and
    never noticed.
    """
    from molbuilder.validation import _ENGINE_VALIDATORS, _KIND_VALIDATORS
    assert {c.__name__ for c in _ENGINE_VALIDATORS} == {
        "SiestaConfig", "PySCFConfig"}, (
        "the engine registry no longer matches `science/validation.md`; a row "
        "added here must name a config class production actually validates")
    assert set(_KIND_VALIDATORS) == {"vibration", "transport"}, (
        "a calculation kind lost its science -- this is the road a transport "
        "or vibration prep actually travels")
