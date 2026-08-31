"""L2 source-text lint: a negative token value is written with ``calc()``.

Why
===

``-var(--space-md)`` looks like it negates a token.  It does not.  The
minus sign does not bind to a ``var()`` function, so the value is
invalid — and an invalid value does not fall back to something
reasonable, it makes the browser **discard the whole declaration**.

That is the part worth remembering: the loss is not the negation, it is
everything else on the same line.  This shipped in
``modify/style.css``::

    .modify-op-tabs {
        margin: var(--space-sm) -var(--space-md) var(--space-sm);  /* bleed */
    }

Computed ``margin: 0px``.  The bleed the comment describes never
happened, *and* the two vertical margins the same line asked for were
thrown away with it.  Nothing errored.  Nothing logged.  The tab strip
simply sat 16 px inside every other edge on the card, and the comment
went on describing a layout that had never rendered, for months.

The spelling that works::

    margin: var(--space-sm) calc(-1 * var(--space-md)) var(--space-sm);

See ``docs/web/ui-contract.md`` § 9 for the rule.

What this test pins
===================

No ``.css`` file under ``molbuilder/web/static/`` contains a minus sign
immediately followed by ``var(``.  Module sheets are included: this is
a syntax fact, not a style opinion, so the module boundary
(``css-system-plan.md`` § 1) does not apply — a guard may *read* a
module's sheet, it just may not tell it how to look.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "molbuilder/web/static"

#: A ``-`` glued to ``var(``, with no operator or digit in between.
#: ``calc(-1 * var(--x))`` is fine — the minus there precedes a number.
#: ``foo-var(...)`` cannot occur (no CSS function ends in ``-``), and a
#: custom-property NAME containing ``-var(`` is not expressible.
_NEGATED_VAR = re.compile(r"(?<![\w)])-var\s*\(")

#: A CSS block comment, replaced by an equal number of newlines so the
#: line numbers a violation reports stay the file's own.
_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _declarations_only(text: str) -> str:
    """The sheet with its comments blanked out.

    A comment is prose, and prose about this very bug quotes it -- the
    fix in ``modify/style.css`` explains what the invalid line was, and
    a lint reading raw text flagged its own explanation.  A rule that
    cannot be *written about* is a rule people stop writing about.
    """
    return _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def _iter_css_files():
    """Every stylesheet the app ships, vendor bundles excluded."""
    for path in sorted(_STATIC.rglob("*.css")):
        if "vendor" in path.parts:
            continue
        yield path


def test_css_negated_var_is_calc():
    """A negative token is ``calc(-1 * var(--t))``, never ``-var(--t)``."""
    bad = []
    for path in _iter_css_files():
        text = path.read_text(encoding="utf-8")
        # Count the line in the BLANKED text, not the original: blanking a
        # comment shortens its lines, so a character offset taken from one
        # and counted in the other names an innocent line.  (It named line
        # 124 for a violation on 380, first time this ran.)  The newlines
        # themselves are preserved one-for-one, so the ordinal is right.
        blanked = _declarations_only(text)
        for m in _NEGATED_VAR.finditer(blanked):
            line_no = blanked.count("\n", 0, m.start()) + 1
            line = text.splitlines()[line_no - 1].strip()
            bad.append(f"{path.relative_to(_ROOT)}:{line_no}: {line}")
    assert not bad, (
        "`-var(--token)` is invalid CSS and the browser DROPS THE WHOLE "
        "DECLARATION -- every other value on the line goes with it. Write "
        "`calc(-1 * var(--token))`:\n  " + "\n  ".join(bad))


def test_the_pattern_catches_the_bug_that_shipped():
    """The exact line that shipped, so the pin cannot pass vacuously.

    A lint whose pattern never matched anything would stay green over a
    regression forever; this asserts it fires on the real case.
    """
    shipped = "margin: var(--space-sm) -var(--space-md) var(--space-sm);"
    assert _NEGATED_VAR.search(shipped)


def test_a_comment_may_quote_the_bug_it_explains():
    """The fix records the invalid line it replaced; that is not a violation.

    Without this the guard fails on the commit that fixes the bug, which
    is the surest way to teach someone to delete the guard.
    """
    sheet = ("/* What stood here was\n"
             "   `margin: var(--a) -var(--b) var(--a)` -- invalid. */\n"
             "p { margin: calc(-1 * var(--b)); }\n")
    assert not _NEGATED_VAR.search(_declarations_only(sheet))
    # ...and the blanking keeps the line numbering honest.
    assert _declarations_only(sheet).count("\n") == sheet.count("\n")


def test_the_pattern_accepts_the_spellings_that_work():
    """calc() negation, subtraction, and a plain var() are all legal."""
    for ok in ("margin: calc(-1 * var(--space-md));",
               "width: calc(100% - var(--space-md));",
               "padding: var(--space-md);",
               "top: calc(0px - var(--space-sm));"):
        assert not _NEGATED_VAR.search(ok), ok


def test_the_lint_actually_reads_stylesheets():
    """A guard over an empty file list is a guard over nothing."""
    assert len(list(_iter_css_files())) > 10
