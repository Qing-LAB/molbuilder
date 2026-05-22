"""CSP invariant: no inline ``<script>`` blocks in any served HTML.

molbuilder's Content-Security-Policy sets ``script-src 'self'`` --
meaning the browser ONLY runs JS loaded from same-origin
``<script src="...">`` tags.  Inline ``<script>...</script>`` blocks
get silently blocked at execution time; the browser shows ONE
CSP-violation notice on the console at page load and nothing else.
Page wiring that lived inside the inline block (event handlers,
button-enable logic, DOMContentLoaded hooks) silently disappears.

A 2026-05-18 bug surfaced exactly this: three tabs (/watch, /modify,
/spectra) each carried an inline ``<script>`` block at the bottom
of their template that wired the "Load from current selection"
button against the projects sidebar.  CSP blocked all three; the
buttons stayed disabled / non-functional; no test caught the
regression because no test pinned the invariant.

This test pins the invariant.  Every Jinja template under
``molbuilder/web/templates/`` is scanned; any opening ``<script>``
tag that lacks an ``src=`` attribute fails the test, naming the
file and line so a developer fixes it before pushing.

How to add NEW page-level JavaScript:

    DO:    move the code into an external file under
           ``molbuilder/web/static/<scope>/<name>.js`` and
           reference it with ``<script src="...">``.
    DON'T: write a ``<script>...</script>`` block in the template,
           even for "just a one-liner".  Patches like that are how
           the original bug landed.

Allowed shapes (these all PASS the test):

    <script src="..."></script>
    <script type="module" src="..."></script>
    <script defer src="..."></script>

Forbidden shapes (these all FAIL the test):

    <script>doSomething();</script>
    <script type="text/javascript">window.onload = ...;</script>
    <script>const x = 1;</script>
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


TEMPLATES_ROOT = (
    Path(__file__).resolve().parent.parent
    / "molbuilder" / "web" / "templates"
)


def _all_html_templates() -> list[Path]:
    """Recursively find every ``.html`` under templates/."""
    return sorted(TEMPLATES_ROOT.rglob("*.html"))


# Match an opening <script ...> tag.  Capture the attribute list
# so we can check it for ``src=``.  Multi-line tolerant (attributes
# can wrap).
_SCRIPT_OPEN_TAG = re.compile(r"<script\b([^>]*)>", re.IGNORECASE | re.DOTALL)

# Spans we elide before searching for <script>: an inline <style>
# block's CSS is text, not markup, so a CSS comment that mentions
# the literal characters ``<script>`` (e.g. to document the
# no-inline-JS rule) must not trigger a false positive.  HTML
# comments and Jinja comments are similarly text-only.  We
# replace these spans with spaces of the same length so line
# numbers in the offender report stay accurate.
_NON_MARKUP_SPANS = re.compile(
    r"<style\b[^>]*>.*?</style>"        # inline CSS
    r"|<!--.*?-->"                       # HTML comments
    r"|\{#.*?#\}",                       # Jinja comments
    re.IGNORECASE | re.DOTALL,
)


def _strip_non_markup(src: str) -> str:
    """Replace text-only spans with same-length whitespace.

    Keeps byte offsets stable so line-number reporting in
    ``_find_inline_script_blocks`` matches the original source.
    """
    def _blank(m):
        # Preserve newlines so line numbers don't shift.
        return "".join(ch if ch == "\n" else " " for ch in m.group(0))
    return _NON_MARKUP_SPANS.sub(_blank, src)


def _find_inline_script_blocks(src: str) -> list[tuple[int, str]]:
    """Return (line_number, attrs) for every inline ``<script>`` tag.

    An inline tag is one whose opening tag has no ``src=`` attribute.
    Non-markup spans (inline <style> CSS, HTML/Jinja comments) are
    elided before matching, so a CSS comment that documents the
    no-inline-script rule by quoting ``<script>`` literally is
    correctly ignored.
    """
    src_searchable = _strip_non_markup(src)
    offenders = []
    for m in _SCRIPT_OPEN_TAG.finditer(src_searchable):
        attrs = m.group(1)
        # The negative invariant: tags without ``src=`` are inline.
        # Spaces around ``=`` are tolerated (HTML5 allows ``src =``).
        if not re.search(r"\bsrc\s*=", attrs, re.IGNORECASE):
            line_no = src_searchable[: m.start()].count("\n") + 1
            offenders.append((line_no, f"<script{attrs}>"))
    return offenders


class TestNoInlineScripts:
    """The CSP-compliance contract: zero inline ``<script>`` blocks
    across every Jinja template molbuilder ever renders."""

    @pytest.mark.parametrize(
        "template_path",
        _all_html_templates(),
        ids=lambda p: str(p.relative_to(TEMPLATES_ROOT)),
    )
    def test_template_has_no_inline_script_blocks(self, template_path):
        src = template_path.read_text()
        offenders = _find_inline_script_blocks(src)
        if not offenders:
            return  # clean
        rel = template_path.relative_to(TEMPLATES_ROOT)
        lines = "\n".join(
            f"  line {ln}: {tag.strip()[:140]}"
            for ln, tag in offenders
        )
        pytest.fail(
            f"{rel}: {len(offenders)} inline <script> block(s) violate "
            f"the CSP ``script-src 'self'`` contract.  Inline scripts "
            f"are silently blocked in production; the wiring inside "
            f"the block never runs.\n\n"
            f"Offending sites:\n{lines}\n\n"
            f"Fix: move the JS into an external file under "
            f"``molbuilder/web/static/<scope>/<name>.js`` and "
            f"reference it via ``<script src=\"...\"></script>``."
        )


class TestInlineScriptDetectionHeuristic:
    """Self-tests for the helper that finds inline ``<script>`` tags
    -- pin the boundary cases so a future regex tweak doesn't
    accidentally widen or narrow the contract.
    """

    @pytest.mark.parametrize("snippet,expected_count", [
        # Plain inline -- the canonical offender.
        ("<script>doIt();</script>", 1),
        # Inline with type attribute (still no src).
        ('<script type="text/javascript">x</script>', 1),
        # Inline with multi-line attrs.
        ('<script\n  type="text/javascript"\n>x</script>', 1),
        # External -- passes.
        ('<script src="/static/x.js"></script>', 0),
        # External, multi-attr -- passes.
        ('<script type="module" src="/x.js" defer></script>', 0),
        # External, attribute-order-swapped -- passes.
        ('<script src="/x.js" type="module"></script>', 0),
        # Edge: spaces around the equals in src= are tolerated.
        ('<script src ="/x.js"></script>', 0),
        # Self-closing-ish on the OPENING tag isn't legal HTML but
        # accept it as external (defensive).
        ('<script src="/x.js"/>', 0),
        # ``script`` substring of another tag must NOT match.
        ('<scriptlike>x</scriptlike>', 0),
        # <script> mentioned inside a CSS comment in <style>: text,
        # not markup; must NOT count as an offender.
        ('<style>/* the <script> tag is forbidden */</style>', 0),
        # <script> mentioned inside an HTML comment: same -- text only.
        ('<!-- inline <script> blocks violate CSP -->', 0),
        # <script> mentioned inside a Jinja comment: same.
        ('{#- inline <script> is forbidden by CSP -#}', 0),
        # An ACTUAL inline <script> still gets caught when it sits
        # alongside the harmless text mentions above.
        ('<style>/* <script> ok */</style><script>real();</script>', 1),
    ])
    def test_detection(self, snippet, expected_count):
        assert len(_find_inline_script_blocks(snippet)) == expected_count
