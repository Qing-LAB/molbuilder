"""Declarative section-rule primitives for free-form text-output parsers.

The model: a parser is a list of :class:`SectionRule` objects + a tiny
state-machine driver.  In the SCAN state, each rule's ``start``
matcher gets a chance to claim the current line; when one matches,
the parser optionally runs ``on_start`` and (if ``consume`` is set)
enters that rule's section state.  While in a section the rule's
``consume`` receives each subsequent line and returns a sentinel
deciding whether to stay, leave, or leave-and-re-feed.

This is the answer to the user's 2026-05-28 directive:

  "detection of when a section starts, what that section is about
   and how the data are printed should be basic capability.  the
   detection of names should be immune to capitalization, small
   spelling differences etc.  we should have a smart parser that
   will be defined by rules."

Scope (locked 2026-05-29):

  * **Case-insensitive matching only** -- no Levenshtein fuzz.  The
    realistic SIESTA version-skew case is capitalisation; a parser
    that goes one step further and accepts typos would invite false
    positives on comment lines.
  * **Per-rule alias list**: a rule can accept multiple equivalent
    section headers ("outcoor:", "OUTPUT COORDS:", "final
    geometry:") by wiring an OR-combined matcher.  Future ORCA /
    NWChem / Gaussian text parsers reuse the same primitives.
  * **SIESTA-only port in this pass**.  ``molwatch_log.py`` /
    ``spectra_json.py`` / ``pyscf.py`` keep their existing
    implementations (their inputs are JSONL or markered, not
    free-form text -- no win from the abstraction).

Module layout:

  * Matcher builders (:func:`starts_with_ci`, :func:`contains_ci`,
    :func:`any_of`) -- pure functions, no parser state.
  * :class:`SectionRule` dataclass -- name + matcher + optional
    on_start hook + optional per-line consumer + alias list (docs).
  * Sentinels: :data:`CONTINUE`, :data:`END_SECTION`,
    :data:`END_BUBBLE`.  Returned by ``consume`` to drive transitions.

The state-machine *driver* lives in the engine-specific parser (here,
:mod:`molbuilder.parsers.siesta`).  The driver only needs ~30 lines
of dispatch logic; the engine-specific concerns live in the rule
callbacks (which close over the parser's locals).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Pattern, Union


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
Used when the section-end signal IS the start of the next section --
e.g. a SIESTA outcoor: block ending with the next "siesta: ..." line
that itself should match another rule.  Prevents losing one line."""


# ---------------------------------------------------------------------------
# Matcher builders.  Tiny pure helpers; each returns a ``(line -> bool)``
# closure suitable as a ``SectionRule.start`` field.
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------
# Pattern-matcher abstraction (introduced 2026-05-31).
#
# A ``_PatternMatcher`` is what ``starts_with_ci`` / ``contains_ci`` /
# ``matches_regex_ci`` / ``any_of`` return.  It carries BOTH:
#
#   * a regex pattern fragment (string) suitable for inclusion in a
#     combined-alternation regex compiled by :class:`CompiledRules`;
#   * a callable fallback so the matcher still works as
#     ``(line: str) -> bool`` when invoked outside the driver
#     (tests, ad-hoc usage).
#
# The driver's :class:`CompiledRules` partitions rules into
# regex-capable (``start`` is a ``_PatternMatcher``) and predicate-
# only (``start`` is a plain callable).  Only regex-capable rules
# contribute to the combined regex; predicate-only rules are invoked
# individually on each scan-state line.
#
# Registration-order semantics are preserved: the driver iterates
# rules in registration order; for each regex-capable rule it checks
# the rule's INDIVIDUAL pre-compiled regex (not just the combined-
# group result, which would tie-break by position instead of by
# registration).  The combined regex acts as a fast pre-filter:
# when it returns no match, no regex-capable rule can fire on this
# line and the driver skips straight to predicate-only rules.
# --------------------------------------------------------------------


class _PatternMatcher:
    """A matcher that exposes both a regex pattern + a callable.

    Returned by :func:`starts_with_ci`, :func:`contains_ci`,
    :func:`matches_regex_ci`, and :func:`any_of` (when all sub-
    matchers are themselves ``_PatternMatcher`` instances).

    The pattern is a raw string fragment WITHOUT a ``(?i)`` inline
    flag — :class:`CompiledRules` compiles the alternation with
    ``re.IGNORECASE`` once.  Including the flag inline would
    double-apply when the matcher is combined.

    The callable signature is ``(line: str) -> bool``.  Calling a
    ``_PatternMatcher`` directly delegates to that callable (so old
    code that did ``rule.start(line)`` keeps working).
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
    ``prefix.lower()``.  Use this for section headers that always
    sit at column 0 (modulo indentation), e.g. ``outcoor:``.

    Returned matcher signature: ``(line: str) -> bool``.

    >>> m = starts_with_ci("outcoor:")
    >>> m("OUTCOOR: Atomic coordinates")
    True
    >>> m("  outcoor: foo")
    True
    >>> m("# outcoor: comment")
    False
    """
    p = prefix.lower()
    # Regex form: anchor at start, allow leading whitespace, then the
    # literal prefix (escaped).  ``re.IGNORECASE`` is applied by
    # ``CompiledRules`` when assembling the combined regex; we don't
    # inline ``(?i)`` so the flag doesn't multiply when combined.
    pattern = r"^[ \t]*" + re.escape(prefix)

    def _fallback(line: str) -> bool:
        return line.lstrip().lower().startswith(p)

    return _PatternMatcher(pattern, _fallback)


def contains_ci(substr: str) -> "_PatternMatcher":
    """Build a case-insensitive substring matcher.

    Returned matcher signature: ``(line: str) -> bool``.

    Use this for markers embedded in a longer line, e.g.
    ``siesta: E_KS(eV)`` which sits inside ``siesta: E_KS(eV) =
    -1234.567``.  Slightly looser than :func:`starts_with_ci`; reach
    for the prefix variant when you can.
    """
    s = substr.lower()
    # Regex form: literal substring (escaped), no anchor.
    pattern = re.escape(substr)

    def _fallback(line: str) -> bool:
        return s in line.lower()

    return _PatternMatcher(pattern, _fallback)


def matches_regex_ci(pattern: str) -> "_PatternMatcher":
    """Build a matcher backed by a raw regex pattern.

    Use this when :func:`starts_with_ci` / :func:`contains_ci` aren't
    expressive enough -- e.g. the SCF data line ``"   scf:    1 ..."``
    needs a regex like ``r"^\\s*scf:\\s*\\d+\\s+"`` to discriminate
    against arbitrary lines containing ``"scf:"``.

    The pattern is matched case-insensitively (via ``re.IGNORECASE``
    on the compiled regex; do NOT include ``(?i)`` in your pattern).
    The matcher's fallback compiles the pattern once at call site +
    uses ``re.search`` so the matcher behaves identically inside and
    outside the driver loop.
    """
    compiled = re.compile(pattern, re.IGNORECASE)

    def _fallback(line: str) -> bool:
        return compiled.search(line) is not None

    return _PatternMatcher(pattern, _fallback)


def any_of(*matchers):
    """Combine matchers with OR semantics.

    Used to wire a per-rule alias list: build one matcher per accepted
    section header, then OR them together.  Returning early on the
    first hit keeps the per-line cost ~O(aliases) in the worst case
    but usually O(1) (the first alias matches, the rest are skipped).

    Type promotion: if EVERY sub-matcher is a :class:`_PatternMatcher`,
    the return is also a ``_PatternMatcher`` whose pattern is the
    alternation of the sub-patterns and whose fallback is the OR
    short-circuit.  This keeps ``any_of(...)`` chains regex-capable
    so the combined-regex driver path covers them.

    If any sub-matcher is a plain callable (e.g. a closure with
    state), the return is a plain callable -- the combined-regex
    optimisation can't include this rule, but correctness is
    preserved via the predicate-only path.
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
# SectionRule -- the dataclass each engine parser populates.
# ---------------------------------------------------------------------------


@dataclass
class SectionRule:
    """One section's start condition + per-line consumer.

    The driver loop iterates the list of rules in registration order
    on every scan-state line.  Order matters: a more specific rule
    must come before a more general rule that would also match (e.g.
    ``"outcell: Unit cell vectors"`` before any rule keyed on
    ``"outcell:"`` alone).

    Fields
    ------
    name:
        Canonical section name (``"coords"``, ``"cell"``,
        ``"forces"``, ...).  Used as the state label while in this
        section; also surfaced in :class:`ParseWarning` for debugging.
    start:
        Matcher built via :func:`starts_with_ci` / :func:`contains_ci`
        / :func:`any_of`, or a custom ``(line -> bool)`` callable.
        Returning True on a scan-state line causes the driver to
        run ``on_start`` (if any) and -- if ``consume`` is set --
        transition into this section's state.
    on_start:
        Optional ``(line, line_no) -> None`` hook fired on the line
        that triggered the match.  Use it to flush previously
        accumulated state or to capture a value from the header
        line itself.  Should handle its own value-extraction errors
        (the driver does NOT wrap on_start in try/except; an
        unhandled exception aborts the whole parse).  The SIESTA
        callbacks shipping with this module follow the pattern:
        ``try: ... except (ValueError, IndexError) as exc: _warn(...)``.
    consume:
        Optional ``(line, line_no) -> sentinel`` callable invoked
        for each subsequent line while the section is active.
        Returns :data:`CONTINUE`, :data:`END_SECTION`, or
        :data:`END_BUBBLE`.  Omit (None) to make this a single-line
        rule -- ``on_start`` fires and the driver stays in scan mode.
        Same error contract as ``on_start``: consume must not raise.
    aliases:
        Human-readable copies of the section headers this rule
        accepts.  Stored for debugging / docs / introspection only;
        the actual matching is driven by ``start``.  Mirror the
        :func:`starts_with_ci` arguments here.
    """
    name: str
    start: Callable[[str], bool]
    on_start: Optional[Callable[[str, int], None]] = None
    consume: Optional[Callable[[str, int], str]] = None
    aliases: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # name must be non-empty and a valid identifier-ish slug.
        # Used to disambiguate matches in the combined-regex's named
        # groups (see :class:`CompiledRules`).
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SectionRule.name must be a non-empty string")
        # start must be callable (either a _PatternMatcher or a plain
        # callable / closure).  We can't check pattern shape here
        # because plain callables are legal.
        if not callable(self.start):
            raise ValueError(
                f"SectionRule(name={self.name!r}).start must be a "
                f"_PatternMatcher or a plain callable, got "
                f"{type(self.start).__name__}"
            )


# ---------------------------------------------------------------------------
# CompiledRules -- compile a rule list into a combined-regex pre-filter
# + per-rule fallback so the driver can dispatch with one regex search
# in the common case.
# ---------------------------------------------------------------------------


class CompiledRules:
    """Pre-compiled rule dispatch table for the driver loop.

    Built once at parse-init via :func:`compile_rules`.  Carries:

      * the original rule list (in registration order);
      * a per-rule pre-compiled regex for ``_PatternMatcher`` rules
        (None for plain-callable rules);
      * a combined regex over ALL ``_PatternMatcher`` rules, used as
        a fast pre-filter so the driver can skip per-rule iteration
        when no regex-capable rule can possibly match the line.

    :meth:`find_match` is the dispatch entry-point.  Per scan-state
    line:

      1. Run the combined regex once.  If it returns no match, no
         regex-capable rule can fire on this line.  Skip to step 3
         (predicate-only rules might still match).
      2. If the combined regex matched, the line contains SOMETHING
         a regex rule wants.  Iterate the rule list in registration
         order; for each regex-capable rule, check its individual
         pre-compiled regex (the combined regex's leftmost-position
         tie-break can disagree with registration order if multiple
         rules match at different positions).  First rule that fires
         wins.
      3. Iterate the rule list in registration order; for each
         predicate-only rule, call its callable.  First rule that
         fires wins.

    Steps 2 and 3 are interleaved -- the actual implementation
    iterates the rule list ONCE and dispatches each rule by type.
    Per-rule errors are caught + ignored to preserve § 6 isolation.
    """

    __slots__ = ("rules", "individual_patterns", "combined_regex")

    def __init__(self, rules: List["SectionRule"]) -> None:
        self.rules: List["SectionRule"] = list(rules)
        self.individual_patterns: List[Optional[Pattern[str]]] = []
        regex_parts: List[str] = []
        for r in self.rules:
            if isinstance(r.start, _PatternMatcher):
                # Pre-compile each rule's pattern individually so
                # ``find_match`` can verify per-rule matches without
                # re-running the combined regex.
                self.individual_patterns.append(
                    re.compile(r.start.pattern, re.IGNORECASE)
                )
                regex_parts.append(r.start.pattern)
            else:
                self.individual_patterns.append(None)
        # Combined regex: one DFA scan tests every literal-pattern rule
        # at once.  When this returns no match on a scan-state line,
        # the driver knows NO regex-capable rule can fire -- skip
        # straight to predicate-only iteration.
        if regex_parts:
            combined_pattern = "|".join(f"(?:{p})" for p in regex_parts)
            self.combined_regex: Optional[Pattern[str]] = re.compile(
                combined_pattern, re.IGNORECASE
            )
        else:
            self.combined_regex = None

    def find_match(self, line: str) -> Optional["SectionRule"]:
        """Return the first rule (in registration order) whose
        matcher fires on ``line``, or None if none fire.

        Exceptions raised by predicate-only matchers are caught and
        ignored per § 6 error isolation -- the same line goes on to
        be tested against subsequent rules.
        """
        # Step 1: fast pre-filter.  If no regex-capable rule could
        # fire, we still need to iterate predicate-only rules.
        any_regex_could_fire = (
            self.combined_regex is not None
            and self.combined_regex.search(line) is not None
        )
        # Step 2 + 3 (interleaved): iterate rules in order.
        for i, rule in enumerate(self.rules):
            ip = self.individual_patterns[i]
            if ip is not None:
                # Regex-capable rule.  Combined-regex fast path:
                # if no regex rule could fire on this line, skip
                # the per-rule regex search.
                if not any_regex_could_fire:
                    continue
                if ip.search(line) is not None:
                    return rule
            else:
                # Predicate-only rule.
                try:
                    if rule.start(line):
                        return rule
                except Exception:
                    # § 6 error isolation: one bad matcher must not
                    # poison dispatch for the rest of the line.
                    continue
        return None


def compile_rules(rules: List["SectionRule"]) -> CompiledRules:
    """Compile a list of :class:`SectionRule` into a
    :class:`CompiledRules` dispatch table.  Call once at parse-init;
    the returned object is reusable across lines."""
    # Validate uniqueness of rule.name (used for diagnostics + future
    # named-group dispatch).
    seen = set()
    for r in rules:
        if r.name in seen:
            raise ValueError(
                f"duplicate SectionRule name: {r.name!r}.  Rule names "
                f"must be unique in a rule list."
            )
        seen.add(r.name)
    return CompiledRules(rules)
