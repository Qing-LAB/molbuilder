"""Unit tests for the section-rule primitives in
``molbuilder/parsers/_rules.py``.

The primitives are tiny pure functions; these tests pin the
case-insensitivity guarantee + the OR-combinator semantics + the
SectionRule dataclass shape.  Engine-level tests (siesta /
pyscf parser regression) live in their own files.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.engines._section_rules import (
    CONTINUE, END_SECTION, END_BUBBLE,
    SectionRule, any_of, contains_ci, starts_with_ci,
)


class TestStartsWithCi:
    def test_exact_case(self):
        m = starts_with_ci("outcoor:")
        assert m("outcoor: Atomic coordinates (Ang):")

    def test_uppercase(self):
        """The user's verbatim concern: a SIESTA build that capitalises
        section headers must still parse."""
        m = starts_with_ci("outcoor:")
        assert m("OUTCOOR: Atomic coordinates (Ang):")

    def test_mixed_case(self):
        m = starts_with_ci("outcoor:")
        assert m("Outcoor: foo")

    def test_leading_whitespace_stripped(self):
        """Indented section headers (the SCF block lives indented in
        v5 output) still match."""
        m = starts_with_ci("iscf")
        assert m("   iscf  Eharris  E_KS")

    def test_no_match_on_non_prefix(self):
        m = starts_with_ci("outcoor:")
        assert not m("# outcoor: this is a comment")
        assert not m("the outcoor: line came earlier")

    def test_empty_line(self):
        m = starts_with_ci("outcoor:")
        assert not m("")
        assert not m("   ")


class TestContainsCi:
    def test_substring_match(self):
        m = contains_ci("E_KS(eV)")
        assert m("siesta: E_KS(eV) =      -1234.567")

    def test_case_insensitive_substring(self):
        m = contains_ci("e_ks(ev)")
        assert m("siesta: E_KS(eV) =      -1234.567")

    def test_no_match(self):
        m = contains_ci("FreeEnergy")
        assert not m("siesta: E_KS(eV) = -42")


class TestAnyOf:
    def test_first_matcher_wins(self):
        m = any_of(starts_with_ci("outcoor:"),
                   starts_with_ci("OUTPUT COORDS:"))
        assert m("outcoor: foo")
        assert m("OUTPUT COORDS: bar")

    def test_none_matches(self):
        m = any_of(starts_with_ci("a:"),
                   starts_with_ci("b:"))
        assert not m("c: nope")

    def test_empty_combine_raises(self):
        """Defensive guard against constructing an always-false matcher
        by accident -- callers usually mean to pass at least one."""
        with pytest.raises(ValueError):
            any_of()


class TestSectionRule:
    """Shape contract of the dataclass.  No driver logic here -- the
    parser's regression tests are the integration check."""

    def test_minimal_rule_only_needs_name_and_start(self):
        rule = SectionRule(name="end", start=starts_with_ci(">> End of run"))
        assert rule.name == "end"
        assert rule.aliases == []
        assert rule.on_start is None
        assert rule.consume is None

    def test_full_rule(self):
        calls = []

        def on_start(line, ln): calls.append(("start", line, ln))

        def consume(line, ln):
            calls.append(("consume", line, ln))
            return END_SECTION

        rule = SectionRule(
            name="coords",
            start=any_of(
                starts_with_ci("outcoor:"),
                starts_with_ci("OUTPUT COORDS:"),
            ),
            on_start=on_start,
            consume=consume,
            aliases=["outcoor:", "OUTPUT COORDS:"],
        )
        assert rule.aliases == ["outcoor:", "OUTPUT COORDS:"]
        # The matcher itself works as expected.
        assert rule.start("outcoor: foo")
        assert rule.start("OUTPUT COORDS: bar")
        # And the callables fire.
        rule.on_start("outcoor: foo", 7)
        assert rule.consume("blah", 8) == END_SECTION
        assert calls == [("start", "outcoor: foo", 7),
                         ("consume", "blah", 8)]


class TestSentinels:
    """The sentinels are string constants; pin their values so the
    driver loop's identity-compare stays semantically meaningful even
    if a future contributor mis-types them."""

    def test_distinct_values(self):
        assert CONTINUE != END_SECTION != END_BUBBLE != CONTINUE

    def test_are_strings(self):
        for s in (CONTINUE, END_SECTION, END_BUBBLE):
            assert isinstance(s, str)
