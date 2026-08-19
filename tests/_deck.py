"""Reading a generated deck the way its ENGINE reads it.

**Why this exists.** Tests asserted on exact substrings of the emitted text --
``assert "DM.Tolerance      1e-05" in fdf`` -- which pins the COLUMN PADDING
into the assertion.  Widening one field's padding by a single space, a change
libfdf cannot even perceive, failed 45 tests across eight files (measured
2026-08-19).  None of them was testing spacing on purpose; each meant *this
deck sets this keyword to this value*, and said it in a way that breaks on any
reformatting of the emitter.  That is a tax on exactly the refactoring this
layer keeps needing.

**How fdf itself compares.** ``fdf_get`` normalises through ``labeleq``: case
is ignored, and so are ``.``, ``_`` and ``-``.  So ``MD.MaxForceTol``,
``md_maxforcetol`` and ``MDMaxForceTol`` are one keyword to SIESTA, and a test
that insists on one spelling reports a deck the engine reads perfectly well.
"""
from __future__ import annotations

from typing import Optional

__all__ = ["fdf_value", "fdf_sets", "assert_fdf"]


def _norm(label: str) -> str:
    return label.lower().replace(".", "").replace("_", "").replace("-", "")


def fdf_value(text: str, keyword: str) -> Optional[str]:
    """The value a ``.fdf`` gives ``keyword``, or ``None`` if it sets none.

    Comments are stripped; the FIRST setting wins, which is what ``fdf_locate``
    does -- it walks from the top and stops at the first match, so a later
    duplicate is the line SIESTA ignores.
    """
    want = _norm(keyword)
    for line in text.splitlines():
        code = line.split("#", 1)[0].strip()
        if not code or code.startswith("%"):
            continue
        parts = code.split(None, 1)
        if _norm(parts[0]) == want:
            return parts[1].strip() if len(parts) > 1 else ""
    return None


def fdf_sets(text: str, keyword: str) -> bool:
    """Whether the deck sets ``keyword`` at all."""
    return fdf_value(text, keyword) is not None


def assert_fdf(text: str, keyword: str, value: str) -> None:
    """The deck sets ``keyword`` to ``value``, whatever the spacing."""
    got = fdf_value(text, keyword)
    assert got is not None, (
        f"the deck does not set {keyword!r} at all")
    assert got == value, (
        f"{keyword}: deck says {got!r}, expected {value!r}")
