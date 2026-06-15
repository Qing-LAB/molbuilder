"""L2 source-text lint: every ``repeat(auto-fit, minmax(...))`` in
molbuilder CSS uses the ``min(<N>px, 100%)`` floor.

Why
===

A grid declared as

    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));

demands a 360-px MINIMUM track width.  When the container's content
area is narrower than 360 px (e.g. a phone viewport whose body+main
padding leaves ~316 px usable, or a workspace panel slotted into a
narrow column), the track can't shrink — the parent overflows and
a horizontal scrollbar appears.

The fix pattern is

    grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr));

The ``min(360px, 100%)`` floor clamps the lower bound to whichever
is smaller — the declared minimum OR the container's actual width.
When the container is wider, behaviour is identical; when narrower,
the track shrinks with it.

See ``docs/protocols/mobile-layout.md`` for the rule + rationale.

What this test pins
===================

For every ``.css`` file under ``molbuilder/web/static/``:

* Find each ``minmax(`` call inside a ``repeat(auto-fit, ...)`` /
  ``repeat(auto-fill, ...)`` declaration.
* Assert the lower bound is either a bare unit-less keyword OR a
  ``min(...)`` expression.  A literal ``<N>px`` lower bound is
  forbidden.

A future grid declared with a literal floor fails this test loudly.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "molbuilder/web/static"


# Match a ``repeat(<auto-keyword>, minmax(<floor>, ...))`` pattern.
# Group 1: the floor expression (everything between the first
# ``minmax(`` and the first comma at the same paren depth).
_PATTERN = re.compile(
    r"""
    repeat\s*\(
      \s* auto-(?:fit|fill) \s* ,
      \s* minmax\s*\(
        \s* (?P<floor> [^,()]* (?:\([^)]*\) [^,()]* )* )
        \s* ,
    """,
    re.VERBOSE,
)


def _iter_css_files():
    yield from _STATIC.rglob("*.css")


def _violations_in_file(path: Path):
    """Yield (lineno, snippet, floor) for every literal-<N>px
    floor in this file's ``repeat(auto-*, minmax(...))`` calls.
    """
    text = path.read_text(encoding="utf-8")
    for m in _PATTERN.finditer(text):
        floor = m.group("floor").strip()
        # Acceptable: ``min(<N>px, 100%)``, ``min(<N>px, …)``, or
        # any non-numeric keyword (e.g., ``auto``, ``max-content``).
        if floor.startswith("min("):
            continue
        if not re.match(r"^\d", floor):
            # Non-numeric floor (auto, max-content, ...) — accept.
            continue
        # A literal numeric floor (``320px``, ``20rem``, ...).
        # Find the line number of the match start.
        lineno = text.count("\n", 0, m.start()) + 1
        snippet = text[m.start():m.start() + 100].replace("\n", " ")
        yield (lineno, snippet, floor)


def test_every_auto_fit_grid_uses_min_floor():
    """Every ``repeat(auto-fit, minmax(...))`` in molbuilder CSS
    must use the ``min(<N>px, 100%)`` floor.  A bare numeric floor
    forces overflow on any container narrower than the floor."""
    offenders = []
    for css_path in _iter_css_files():
        for lineno, snippet, floor in _violations_in_file(css_path):
            rel = css_path.relative_to(_ROOT)
            offenders.append((str(rel), lineno, floor, snippet))

    if offenders:
        # Format each offender on its own line for readable output.
        lines = [""]
        for rel, lineno, floor, snippet in offenders:
            lines.append(f"  {rel}:{lineno} -- floor {floor!r}")
            lines.append(f"      {snippet[:120]}")
        msg = "\n".join(lines)
        raise AssertionError(
            "Found CSS grid declarations using a bare numeric floor "
            "in ``repeat(auto-fit, minmax(...))``.  Replace with "
            "``minmax(min(<N>px, 100%), 1fr)``.  See "
            "``docs/protocols/mobile-layout.md`` for the pattern.\n"
            + msg
        )


def test_self_check_pattern_catches_violation():
    """Sanity: the regex actually catches the antipattern.  If this
    breaks the test above is silently green and a real bug ships."""
    sample = """
    .x {
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }
    """
    floors = []
    for m in _PATTERN.finditer(sample):
        floors.append(m.group("floor").strip())
    assert floors == ["320px"], (
        f"regex should match bare '320px' floor; got {floors!r}"
    )


def test_self_check_pattern_accepts_min_floor():
    """Sanity: the min(...) form is accepted as compliant."""
    sample = """
    .x {
        grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr));
    }
    """
    offenders = [
        (m.group("floor").strip())
        for m in _PATTERN.finditer(sample)
        if re.match(r"^\d", m.group("floor").strip())
        and not m.group("floor").strip().startswith("min(")
    ]
    assert offenders == [], (
        f"min(...) form should be accepted; got offenders {offenders!r}"
    )
