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
