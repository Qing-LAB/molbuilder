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

from dataclasses import dataclass, field
from typing import Callable, List, Optional


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


def starts_with_ci(prefix: str) -> Callable[[str], bool]:
    """Build a case-insensitive leading-prefix matcher.

    The matcher strips leading whitespace then compares against
    ``prefix.lower()``.  Use this for section headers that always
    sit at column 0 (modulo indentation), e.g. ``outcoor:``.

    Returned matcher signature: ``(line: str) -> bool``.

    Performance note (2026-05-31): the driver loop in siesta.py
    pre-computes ``line.lower()`` and ``line.lstrip().lower()`` once
    per iteration and rebinds them on closure-captured module
    globals before invoking matchers, so the matcher itself avoids
    per-call ``.lower()`` / ``.lstrip()`` overhead.  See
    :func:`_set_iteration_cache` for the driver hook.

    >>> m = starts_with_ci("outcoor:")
    >>> _set_iteration_cache("OUTCOOR: Atomic coordinates")
    >>> m("OUTCOOR: Atomic coordinates")
    True
    >>> _set_iteration_cache("  outcoor: foo")
    >>> m("  outcoor: foo")
    True
    >>> _set_iteration_cache("# outcoor: comment")
    >>> m("# outcoor: comment")
    False
    """
    p = prefix.lower()

    def _match(line: str) -> bool:
        # _ITER_LSTRIP_LOWER is set by _set_iteration_cache.  Fallback
        # to a one-shot compute when called outside the driver (tests,
        # ad-hoc usage) so matchers stay drop-in safe.
        cached = _ITER_LSTRIP_LOWER
        if cached is None:
            cached = line.lstrip().lower()
        return cached.startswith(p)

    return _match


def contains_ci(substr: str) -> Callable[[str], bool]:
    """Build a case-insensitive substring matcher.

    Returned matcher signature: ``(line: str) -> bool``.  Performance
    note: same as :func:`starts_with_ci` — driver pre-computes
    ``line.lower()`` and matchers read it from a module-global rebound
    once per iteration.

    Use this for markers embedded in a longer line, e.g.
    ``siesta: E_KS(eV)`` which sits inside ``siesta: E_KS(eV) =
    -1234.567``.  Slightly looser than :func:`starts_with_ci`; reach
    for the prefix variant when you can.
    """
    s = substr.lower()

    def _match(line: str) -> bool:
        cached = _ITER_LOWER
        if cached is None:
            cached = line.lower()
        return s in cached

    return _match


# --------------------------------------------------------------------
# Per-iteration matcher cache (perf optimisation, 2026-05-31).
#
# Pre-this-commit: every ``starts_with_ci`` / ``contains_ci`` matcher
# did its own ``line.lower()`` (and ``starts_with_ci`` also did
# ``line.lstrip()``).  With 14 rules per scan-state line and SIESTA
# .out files routinely > 26k lines, the same line was lower-cased
# ~10x per iteration.
#
# The fix: driver pre-computes both ``line.lower()`` and
# ``line.lstrip().lower()`` ONCE per iteration and stores them on
# module-level globals before invoking matchers.  Matchers read
# from the globals directly -- no per-call computation, no dict
# lookup, no ``id()`` check overhead.
#
# The matcher API ``(line: str) -> bool`` is preserved.  Matchers
# called WITHOUT a prior :func:`_set_iteration_cache` call (tests,
# ad-hoc usage) fall back to computing the lowered form themselves
# so behaviour is identical -- just slower per call.
#
# Module-private + single-threaded (the parser processes one file
# at a time on one thread).
# --------------------------------------------------------------------

_ITER_LOWER:        Optional[str] = None
_ITER_LSTRIP_LOWER: Optional[str] = None


def _set_iteration_cache(line: str) -> None:
    """Driver hook: call once per loop iteration BEFORE invoking any
    matcher.  Pre-computes ``line.lower()`` and ``line.lstrip().lower()``
    so matchers can read them via the module-global directly."""
    global _ITER_LOWER, _ITER_LSTRIP_LOWER
    _ITER_LOWER = line.lower()
    # Skip a redundant lstrip().lower() when the line has no leading
    # whitespace -- common for SIESTA output lines that DO have
    # leading whitespace are SCF data + force rows, both of which
    # use regex matchers, not starts_with_ci.
    if line and line[0] not in (" ", "\t"):
        _ITER_LSTRIP_LOWER = _ITER_LOWER
    else:
        _ITER_LSTRIP_LOWER = line.lstrip().lower()


def _clear_iteration_cache() -> None:
    """Driver hook: call when leaving the matcher-dispatch loop so
    out-of-loop ``starts_with_ci`` / ``contains_ci`` calls fall back
    to the one-shot compute path."""
    global _ITER_LOWER, _ITER_LSTRIP_LOWER
    _ITER_LOWER = None
    _ITER_LSTRIP_LOWER = None


def any_of(*matchers: Callable[[str], bool]) -> Callable[[str], bool]:
    """Combine matchers with OR semantics.

    Used to wire a per-rule alias list: build one matcher per accepted
    section header, then OR them together.  Returning early on the
    first hit keeps the per-line cost ~O(aliases) in the worst case
    but usually O(1) (the first alias matches, the rest are skipped).
    """
    if not matchers:
        raise ValueError("any_of requires at least one matcher")

    def _match(line: str) -> bool:
        for m in matchers:
            if m(line):
                return True
        return False

    return _match


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
