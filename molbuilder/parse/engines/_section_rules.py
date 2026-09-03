"""Section-rule primitives for free-form text-output parsers.

Absorbed from the legacy ``molbuilder.parsers._rules``, deleted with
that package on 2026-06-21 -- this is the only copy (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

The model: a parser is a list of :class:`SectionRule` objects + a
tiny state-machine driver.  In the SCAN state, each rule's ``start``
matcher gets a chance to claim the current line; when one matches,
the parser optionally runs ``on_start`` and (if ``consume`` is set)
enters that rule's section state.  While in a section the rule's
``consume`` receives each subsequent line and returns a sentinel
deciding whether to stay, leave, or leave-and-re-feed.

Scope (locked 2026-05-29):

  * **Case-insensitive matching only** -- no Levenshtein fuzz.
  * **Per-rule alias list** via :func:`any_of`.
  * **SIESTA + molwatch** consume this module; ``pyscf.py`` uses
    its own hand-rolled scanner (no win from the abstraction
    given its JSON-like log format).

Module layout:

  * Matcher builders (:func:`starts_with_ci`, :func:`contains_ci`,
    :func:`matches_regex_ci`, :func:`any_of`).
  * :class:`SectionRule` dataclass.
  * :class:`CompiledRules` + :func:`compile_rules`.
  * Sentinels: :data:`CONTINUE`, :data:`END_SECTION`, :data:`END_BUBBLE`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Pattern


# ---------------------------------------------------------------------------
# Sentinels returned by SectionRule.consume to drive the state machine.
# ---------------------------------------------------------------------------

CONTINUE = "continue"
"""Stay in this section; feed the next line to ``consume`` too."""

END_SECTION = "end"
"""Leave the section.  The line that triggered the end is discarded
(its content was either a section-terminator like a blank line, or
already malformed).  Subsequent lines go to scan mode."""

END_BUBBLE = "end_bubble"
"""Leave the section AND re-process the current line in scan mode.
Used when the section-end signal IS the start of the next section."""


# ---------------------------------------------------------------------------
# Matcher builders.
# ---------------------------------------------------------------------------


class _PatternMatcher:
    """A matcher that exposes both a regex pattern + a callable.

    The pattern is a raw string fragment WITHOUT a ``(?i)`` inline
    flag — :class:`CompiledRules` compiles the alternation with
    ``re.IGNORECASE`` once.  Including the flag inline would
    double-apply when the matcher is combined.
    """

    __slots__ = ("pattern", "_callable")

    def __init__(self, pattern: str, fn: Callable[[str], bool]) -> None:
        self.pattern = pattern
        self._callable = fn

    def __call__(self, line: str) -> bool:
        return self._callable(line)

    def __repr__(self) -> str:
        return f"_PatternMatcher(pattern={self.pattern!r})"


def starts_with_ci(prefix: str) -> "_PatternMatcher":
    """Build a case-insensitive leading-prefix matcher.

    The matcher strips leading whitespace then compares against
    ``prefix.lower()``.
    """
    p = prefix.lower()
    pattern = r"^[ \t]*" + re.escape(prefix)

    def _fallback(line: str) -> bool:
        return line.lstrip().lower().startswith(p)

    return _PatternMatcher(pattern, _fallback)


def contains_ci(substr: str) -> "_PatternMatcher":
    """Build a case-insensitive substring matcher."""
    s = substr.lower()
    pattern = re.escape(substr)

    def _fallback(line: str) -> bool:
        return s in line.lower()

    return _PatternMatcher(pattern, _fallback)


def matches_regex_ci(pattern: str) -> "_PatternMatcher":
    """Build a matcher backed by a raw regex pattern (case-insensitive)."""
    compiled = re.compile(pattern, re.IGNORECASE)

    def _fallback(line: str) -> bool:
        return compiled.search(line) is not None

    return _PatternMatcher(pattern, _fallback)


def any_of(*matchers):
    """Combine matchers with OR semantics.

    Type promotion: if EVERY sub-matcher is a :class:`_PatternMatcher`,
    the return is also a ``_PatternMatcher`` whose pattern is the
    alternation of the sub-patterns and whose fallback is the OR
    short-circuit.
    """
    if not matchers:
        raise ValueError("any_of requires at least one matcher")

    def _fallback(line: str) -> bool:
        for m in matchers:
            if m(line):
                return True
        return False

    if all(isinstance(m, _PatternMatcher) for m in matchers):
        combined_pattern = "|".join(f"(?:{m.pattern})" for m in matchers)
        return _PatternMatcher(combined_pattern, _fallback)

    return _fallback


# ---------------------------------------------------------------------------
# SectionRule.
# ---------------------------------------------------------------------------


@dataclass
class SectionRule:
    """One section's start condition + per-line consumer.

    Fields
    ------
    name:
        Canonical section name (``"coords"``, ``"cell"``, ...).
    start:
        Matcher built via :func:`starts_with_ci` / :func:`contains_ci`
        / :func:`any_of`, or a custom ``(line -> bool)`` callable.
    on_start:
        Optional ``(line, line_no) -> None`` hook fired on the line
        that triggered the match.  Must not raise.
    consume:
        Optional ``(line, line_no) -> sentinel`` callable invoked
        for each subsequent line while the section is active.
        Returns :data:`CONTINUE`, :data:`END_SECTION`, or
        :data:`END_BUBBLE`.  Omit (None) to make this a single-line
        rule.  Must not raise.
    aliases:
        Human-readable copies of the accepted headers (debugging /
        docs only; matching is driven by ``start``).
    """
    name: str
    start: Callable[[str], bool]
    on_start: Optional[Callable[[str, int], None]] = None
    consume: Optional[Callable[[str, int], str]] = None
    aliases: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SectionRule.name must be a non-empty string")
        if not callable(self.start):
            raise ValueError(
                f"SectionRule(name={self.name!r}).start must be a "
                f"_PatternMatcher or a plain callable, got "
                f"{type(self.start).__name__}"
            )


# ---------------------------------------------------------------------------
# CompiledRules dispatch table.
# ---------------------------------------------------------------------------


class CompiledRules:
    """Pre-compiled rule dispatch table for the driver loop.

    :meth:`find_match` is the dispatch entry-point.  Per scan-state
    line, runs the combined regex as a fast pre-filter then iterates
    rules in registration order — first hit wins.  Predicate-only
    matcher exceptions are caught + ignored (error isolation).
    """

    __slots__ = ("rules", "individual_patterns", "combined_regex")

    def __init__(self, rules: List["SectionRule"]) -> None:
        self.rules: List["SectionRule"] = list(rules)
        self.individual_patterns: List[Optional[Pattern[str]]] = []
        regex_parts: List[str] = []
        for r in self.rules:
            if isinstance(r.start, _PatternMatcher):
                self.individual_patterns.append(
                    re.compile(r.start.pattern, re.IGNORECASE)
                )
                regex_parts.append(r.start.pattern)
            else:
                self.individual_patterns.append(None)
        if regex_parts:
            combined_pattern = "|".join(f"(?:{p})" for p in regex_parts)
            self.combined_regex: Optional[Pattern[str]] = re.compile(
                combined_pattern, re.IGNORECASE
            )
        else:
            self.combined_regex = None

    def find_match(self, line: str) -> Optional["SectionRule"]:
        any_regex_could_fire = (
            self.combined_regex is not None
            and self.combined_regex.search(line) is not None
        )
        for i, rule in enumerate(self.rules):
            ip = self.individual_patterns[i]
            if ip is not None:
                if not any_regex_could_fire:
                    continue
                if ip.search(line) is not None:
                    return rule
            else:
                try:
                    if rule.start(line):
                        return rule
                except Exception:
                    continue
        return None


def compile_rules(rules: List["SectionRule"]) -> CompiledRules:
    """Compile a list of :class:`SectionRule` into a dispatch table.
    Rule names must be unique."""
    seen = set()
    for r in rules:
        if r.name in seen:
            raise ValueError(
                f"duplicate SectionRule name: {r.name!r}.  Rule names "
                f"must be unique in a rule list."
            )
        seen.add(r.name)
    return CompiledRules(rules)
