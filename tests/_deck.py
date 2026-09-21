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

__all__ = ["fdf_value", "fdf_sets", "assert_fdf",
           "fdf_block", "fdf_block_rows", "fdf_energy_window"]


def _norm(label: str) -> str:
    """fdf's own `labeleq` normalisation, from the parser that owns it."""
    from molbuilder.parse.fdf import _norm as _fdf_norm
    return _fdf_norm(label)


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


# --------------------------------------------------------------------- #
#  Blocks                                                              #
# --------------------------------------------------------------------- #
#
# The scalar helpers above cannot see inside a `%block`, so these hand
# the block over as the parser read it.  Assert on these, never on the
# deck text: a substring only notices a number that disappeared, not one
# that moved to another block, flipped sign, or changed unit.
#
# `molbuilder.parse.fdf._parse_fdf` does the reading -- the same reader
# the product uses on a cited deck.


def fdf_block(text: str, name: str):
    """The rows of ``%block <name>``, tokenised, or ``None`` if absent."""
    from molbuilder.parse.fdf import _parse_fdf
    _scalars, blocks = _parse_fdf(text)
    return blocks.get(_norm(name))


def fdf_block_rows(text: str, name: str) -> dict:
    """``%block <name>`` as ``{first token (lowered): remaining tokens}``."""
    rows = fdf_block(text, name)
    return {} if rows is None else {r[0].lower(): r[1:] for r in rows if r}


def _energy_ev(value: str, unit: str) -> float:
    from molbuilder.constants import HARTREE_EV, RYDBERG_EV
    u = (unit or "ev").lower()
    if u == "ev":
        return float(value)
    if u in ("ry", "ryd", "rydberg"):
        return float(value) * RYDBERG_EV
    if u in ("ha", "hartree"):
        return float(value) * HARTREE_EV
    raise AssertionError(f"unknown energy unit in the deck: {unit!r}")


def fdf_energy_window(text: str, block: str):
    """``from V [unit] to V [unit]`` inside *block*, as ``(from_eV, to_eV)``.

    A tbtrans contour row puts both bounds on ONE row with ``to`` between
    them, so `fdf_block_rows` alone leaves the second value at index 2 of
    the ``from`` row.  Returns ``None`` when the block or the row is
    absent, so the caller asserts on that rather than this inventing a
    default.  Converted to eV, so a deck that switched unit while keeping
    the same figures is caught instead of silently reinterpreted.
    """
    rows = fdf_block_rows(text, block)
    toks = rows.get("from")
    if not toks:
        return None
    lowered = [t.lower() for t in toks]
    if "to" not in lowered:
        raise AssertionError(
            f"%block {block} has a `from` row with no `to`: {toks!r}")
    cut = lowered.index("to")
    lo, hi = toks[:cut], toks[cut + 1:]
    if not lo or not hi:
        raise AssertionError(
            f"%block {block} `from ... to ...` is missing a bound: {toks!r}")
    return (_energy_ev(lo[0], lo[1] if len(lo) > 1 else ""),
            _energy_ev(hi[0], hi[1] if len(hi) > 1 else ""))
