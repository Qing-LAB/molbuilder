"""Unit tests for the _PatternMatcher + matches_regex_ci abstraction
(introduced 2026-05-31, Commit 1 of the combined-regex dispatch
refactor).

This file covers the SETUP commit: matcher helpers return
``_PatternMatcher`` instances; SectionRule accepts both pattern-
backed matchers and plain callables; ``compile_rules`` builds a
:class:`CompiledRules` dispatch table.

End-to-end parser equivalence (the second commit's work) lives in
``test_combined_dispatch.py``.
"""
from __future__ import annotations

import re

import pytest

from molbuilder.parse.engines._section_rules import (
    SectionRule, _PatternMatcher,
    any_of, compile_rules, contains_ci, matches_regex_ci,
    starts_with_ci,
)


# --------------------------------------------------------------------
# Helper shape: each builder returns a _PatternMatcher
# --------------------------------------------------------------------


class TestStartsWithCiShape:

    def test_returns_pattern_matcher(self):
        m = starts_with_ci("outcoor:")
        assert isinstance(m, _PatternMatcher)

    def test_pattern_includes_lstrip_anchor(self):
        m = starts_with_ci("outcoor:")
        # Anchored at start of line, allows leading [ \t].
        assert m.pattern.startswith(r"^[ \t]*")

    def test_pattern_escapes_metachars(self):
        m = starts_with_ci("E_KS(eV)")
        # ``(`` and ``)`` MUST be escaped (otherwise they form a
        # capture group in the combined regex).
        assert r"\(" in m.pattern
        assert r"\)" in m.pattern

    def test_pattern_has_no_inline_ignorecase_flag(self):
        """The combined-regex driver applies re.IGNORECASE once.
        An inline ``(?i)`` in a sub-pattern would double-apply and
        was the source of subtle behaviour drift in early drafts."""
        m = starts_with_ci("OutCoor:")
        assert "(?i)" not in m.pattern

    def test_matcher_still_callable_directly(self):
        """Old code calling ``rule.start(line)`` keeps working."""
        m = starts_with_ci("outcoor:")
        assert m("OUTCOOR: Atomic coords") is True
        assert m("# outcoor: comment") is False


class TestContainsCiShape:

    def test_returns_pattern_matcher(self):
        m = contains_ci("siesta: E_KS(eV)")
        assert isinstance(m, _PatternMatcher)

    def test_pattern_has_no_start_anchor(self):
        m = contains_ci("siesta: error")
        assert not m.pattern.startswith("^")

    def test_pattern_escapes_metachars(self):
        m = contains_ci("siesta: E_KS(eV) =")
        assert r"\(" in m.pattern
        assert r"\)" in m.pattern

    def test_matcher_still_callable_directly(self):
        m = contains_ci("siesta: error")
        assert m("foo siesta: ERROR bar") is True
        assert m("foo SIESTA prefix bar") is False


class TestMatchesRegexCiShape:

    def test_returns_pattern_matcher(self):
        m = matches_regex_ci(r"^\s*iscf\s+\S")
        assert isinstance(m, _PatternMatcher)

    def test_pattern_is_raw_passthrough(self):
        m = matches_regex_ci(r"^\s*scf:\s*\d+")
        assert m.pattern == r"^\s*scf:\s*\d+"

    def test_matcher_callable_directly(self):
        m = matches_regex_ci(r"^\s*iscf\s+\S")
        assert m("   iscf Eharris(eV) E_KS(eV)") is True
        assert m("scf:    1   -100") is False

    def test_case_insensitive_via_compile_flag(self):
        """The fallback compiles with re.IGNORECASE so the matcher
        behaves the same as the combined regex would."""
        m = matches_regex_ci(r"^\s*iscf\s+\S")
        assert m("   ISCF Eharris(eV)") is True
        assert m("   IsCf Eharris(eV)") is True


# --------------------------------------------------------------------
# Helper equivalence: pattern + callable agree on a crafted suite
# --------------------------------------------------------------------


class TestHelperEquivalence:
    """For each helper, the regex pattern and the callable fallback
    must give the SAME answer on every line in a hand-crafted suite.
    Catches escape / anchor / lookahead drift."""

    @staticmethod
    def _check(matcher, line, expected):
        """Call the matcher AND run the pattern as a standalone
        compiled regex; both must agree with ``expected``."""
        callable_result = matcher(line)
        regex_result = (
            re.compile(matcher.pattern, re.IGNORECASE).search(line)
            is not None
        )
        assert callable_result == expected, (
            f"callable disagreed with expected={expected!r} on "
            f"line={line!r}: got {callable_result!r}"
        )
        assert regex_result == expected, (
            f"pattern {matcher.pattern!r} disagreed with "
            f"expected={expected!r} on line={line!r}: got {regex_result!r}"
        )

    def test_starts_with_ci_equivalence_suite(self):
        m = starts_with_ci("outcoor:")
        cases = [
            ("outcoor: atoms", True),
            ("OUTCOOR: atoms", True),
            ("  outcoor: atoms", True),    # leading whitespace OK
            ("\toutcoor: atoms", True),    # leading tab OK
            ("# outcoor: comment", False), # leading hash breaks anchor
            ("the outcoor: appears mid", False),
            ("", False),
            ("outcoor", False),            # missing colon
        ]
        for line, exp in cases:
            self._check(m, line, exp)

    def test_contains_ci_equivalence_suite(self):
        m = contains_ci("siesta: E_KS(eV)")
        cases = [
            ("siesta: E_KS(eV) = -123", True),
            ("SIESTA: E_KS(EV) = -123", True),
            ("   siesta: e_ks(ev) = -42", True),
            ("blah siesta: E_KS(eV) blah", True),
            ("siesta: e_ks(au) = -42", False),  # different unit
            ("E_KS appears without prefix", False),
            ("", False),
        ]
        for line, exp in cases:
            self._check(m, line, exp)

    def test_metachar_heavy_substring(self):
        """The substring contains BOTH regex meta-chars (parens, dots,
        plus) AND ordinary chars.  Catches re.escape application gaps."""
        m = contains_ci("a.b+c (d)")
        cases = [
            ("foo a.b+c (d) bar", True),
            ("foo aXb+c (d) bar", False),       # the dot is literal
            ("foo a.b c (d) bar", False),       # the plus is literal
            ("foo a.b+c d bar", False),         # the parens are literal
        ]
        for line, exp in cases:
            self._check(m, line, exp)


# --------------------------------------------------------------------
# any_of pattern composition
# --------------------------------------------------------------------


class TestAnyOfShape:

    def test_all_pattern_matchers_yields_pattern_matcher(self):
        m = any_of(starts_with_ci("outcoor:"), starts_with_ci("outcell:"))
        assert isinstance(m, _PatternMatcher)
        # Combined pattern uses ``|`` for alternation.
        assert "|" in m.pattern

    def test_mixed_yields_plain_callable(self):
        """If any sub-matcher is a plain callable, the combined
        matcher is also a plain callable (combined-regex can't
        include the callable rule)."""
        plain = lambda line: line == "exact"
        m = any_of(starts_with_ci("outcoor:"), plain)
        assert not isinstance(m, _PatternMatcher)
        # Still callable.
        assert callable(m)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            any_of()

    def test_short_circuit_semantics(self):
        """First sub-matcher to return True ends the OR.  Subsequent
        sub-matchers are NOT consulted (their side effects don't fire)."""
        calls = []
        def m1(line):
            calls.append("m1")
            return True
        def m2(line):
            calls.append("m2")
            return True
        combined = any_of(m1, m2)
        combined("anything")
        assert calls == ["m1"]   # m2 not called


# --------------------------------------------------------------------
# SectionRule validation
# --------------------------------------------------------------------


class TestSectionRuleValidation:

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            SectionRule(name="", start=lambda line: False)

    def test_non_string_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            SectionRule(name=None, start=lambda line: False)  # type: ignore

    def test_non_callable_start_raises(self):
        with pytest.raises(ValueError, match="callable"):
            SectionRule(name="x", start="not callable")  # type: ignore

    def test_pattern_matcher_start_accepted(self):
        r = SectionRule(name="x", start=starts_with_ci("foo"))
        assert isinstance(r.start, _PatternMatcher)

    def test_plain_callable_start_accepted(self):
        r = SectionRule(name="x", start=lambda line: line == "foo")
        assert callable(r.start)
        assert not isinstance(r.start, _PatternMatcher)


# --------------------------------------------------------------------
# compile_rules + CompiledRules.find_match
# --------------------------------------------------------------------


class TestCompileRules:

    def test_empty_rule_list_compiles(self):
        cr = compile_rules([])
        assert cr.combined_regex is None
        assert cr.individual_patterns == []
        # find_match on empty -> None.
        assert cr.find_match("anything") is None

    def test_all_predicate_only_no_combined_regex(self):
        r1 = SectionRule(name="a", start=lambda line: line == "a")
        r2 = SectionRule(name="b", start=lambda line: line == "b")
        cr = compile_rules([r1, r2])
        # No regex-capable rules -> no combined regex.
        assert cr.combined_regex is None
        # Predicate-only paths still work.
        assert cr.find_match("a") is r1
        assert cr.find_match("b") is r2
        assert cr.find_match("c") is None

    def test_mixed_rules_combined_includes_only_pattern_rules(self):
        r1 = SectionRule(name="pred", start=lambda line: "PRED" in line)
        r2 = SectionRule(name="patt", start=starts_with_ci("foo:"))
        cr = compile_rules([r1, r2])
        # Combined regex only has the pattern-rule contribution.
        assert cr.combined_regex is not None
        # Individual patterns: None for predicate, compiled for pattern.
        assert cr.individual_patterns[0] is None
        assert cr.individual_patterns[1] is not None

    def test_duplicate_name_raises(self):
        r1 = SectionRule(name="x", start=starts_with_ci("a"))
        r2 = SectionRule(name="x", start=starts_with_ci("b"))
        with pytest.raises(ValueError, match="duplicate"):
            compile_rules([r1, r2])


class TestFindMatchDispatchOrder:
    """The trickiest part: when multiple rules could match the same
    line, registration order wins (NOT regex's leftmost-position-wins).
    Tests cover all four combinations of (regex|predicate) before/after.
    """

    def test_two_regex_first_wins(self):
        r1 = SectionRule(name="r1", start=contains_ci("foo"))
        r2 = SectionRule(name="r2", start=contains_ci("foo"))
        cr = compile_rules([r1, r2])
        # Line matches both; first in registration order wins.
        assert cr.find_match("foo") is r1

    def test_predicate_before_regex(self):
        """A predicate-only rule registered before a regex rule wins
        when both could match.  Easy to get wrong if the combined
        regex is consulted FIRST and the regex rule is reported as
        the winner."""
        pred_calls = []
        def pred(line):
            pred_calls.append(line)
            return True   # always matches
        r1 = SectionRule(name="r1", start=pred)
        r2 = SectionRule(name="r2", start=contains_ci("anything"))
        cr = compile_rules([r1, r2])
        assert cr.find_match("anything goes") is r1
        # Predicate WAS called; regex rule was NOT reached.
        assert pred_calls == ["anything goes"]

    def test_regex_before_predicate(self):
        pred_called = []
        def pred(line):
            pred_called.append(line)
            return True
        r1 = SectionRule(name="r1", start=contains_ci("foo"))
        r2 = SectionRule(name="r2", start=pred)
        cr = compile_rules([r1, r2])
        assert cr.find_match("foo") is r1
        # Regex won; predicate NOT consulted.
        assert pred_called == []

    def test_three_rules_first_registration_wins(self):
        r1 = SectionRule(name="r1", start=contains_ci("siesta:"))
        r2 = SectionRule(name="r2", start=contains_ci("E_KS"))
        r3 = SectionRule(name="r3", start=lambda line: True)
        cr = compile_rules([r1, r2, r3])
        # All three would match the line; first wins.
        assert cr.find_match("siesta: E_KS = -42") is r1

    def test_no_rule_matches_returns_none(self):
        r1 = SectionRule(name="r1", start=contains_ci("foo"))
        r2 = SectionRule(name="r2", start=contains_ci("bar"))
        cr = compile_rules([r1, r2])
        assert cr.find_match("baz") is None

    def test_predicate_exception_isolated(self):
        """A predicate that raises is caught + dispatch continues to
        the next rule per § 6 error isolation."""
        def broken(line):
            raise RuntimeError("intentional")
        r1 = SectionRule(name="broken", start=broken)
        r2 = SectionRule(name="ok", start=contains_ci("foo"))
        cr = compile_rules([r1, r2])
        # Broken rule's exception swallowed; r2 still fires.
        assert cr.find_match("foo line") is r2


class TestFindMatchFastFilter:
    """The combined regex is supposed to ELIMINATE per-rule iteration
    when no regex rule could fire.  These tests pin that contract."""

    def test_fast_filter_skips_regex_rules_when_no_match(self):
        """When the combined regex doesn't match, the per-rule
        individual regex is NOT consulted (we'd waste regex calls
        otherwise).  Verified by substituting the individual_patterns
        with spy objects -- re.Pattern instances don't let us
        monkey-patch their .search method directly."""
        calls = []

        class _SpyPattern:
            def __init__(self, name, real):
                self.name = name
                self.real = real
            def search(self, line):
                calls.append((self.name, line))
                return self.real.search(line)

        r1 = SectionRule(name="r1", start=contains_ci("zzz"))
        r2 = SectionRule(name="r2", start=contains_ci("zzz"))
        cr = compile_rules([r1, r2])
        cr.individual_patterns[0] = _SpyPattern("r1", cr.individual_patterns[0])
        cr.individual_patterns[1] = _SpyPattern("r2", cr.individual_patterns[1])
        # Line doesn't match anything -> combined regex misses ->
        # no individual search runs at all.
        result = cr.find_match("this line has none of those substrings")
        assert result is None
        assert calls == [], f"individual searches were called: {calls!r}"

    def test_predicate_rules_still_run_when_fast_filter_misses(self):
        """When the combined regex doesn't match, predicate-only rules
        still get their chance -- they could match anything, the
        combined regex doesn't constrain them."""
        pred_calls = []
        def pred(line):
            pred_calls.append(line)
            return line == "match-me"
        r1 = SectionRule(name="r1", start=contains_ci("zzz"))
        r2 = SectionRule(name="pred", start=pred)
        cr = compile_rules([r1, r2])
        result = cr.find_match("match-me")
        assert result is r2
        assert pred_calls == ["match-me"]


# --------------------------------------------------------------------
# Regression: ALL existing matcher behaviour preserved
# --------------------------------------------------------------------


class TestExistingTestsStillWork:
    """The existing test_section_rule.py + test_siesta_parser*.py
    suites should keep passing after this commit's API change.  This
    file just adds explicit checks that the OLD matcher calling
    pattern still works (calling matcher(line) directly)."""

    def test_old_style_match_call_works(self):
        m = starts_with_ci("outcoor:")
        # Old code: ``if rule.start(line): ...`` -- this is the
        # same thing.
        assert m("OUTCOOR: foo") is True
