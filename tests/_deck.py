"""A tbtrans contour window as a test reads it from a generated deck: read by
the same reader the product uses on a cited deck, its bounds in eV, its units
converted, so a test asserts on the window and never on the deck's text.
"""
from __future__ import annotations

# fdf's own `labeleq` normalisation, from the parser that owns it.
from molbuilder.parse.fdf import fdf_key

__all__ = ["fdf_energy_window"]


# `molbuilder.parse.fdf.read_fdf` does the reading -- the same reader
# the product uses on a cited deck.


def fdf_block(text: str, name: str):
    """The rows of ``%block <name>``, tokenised, or ``None`` if absent."""
    from molbuilder.parse.fdf import read_fdf
    _scalars, blocks = read_fdf(text)
    return blocks.get(fdf_key(name))


def fdf_block_rows(text: str, name: str) -> dict:
    """``%block <name>`` as ``{first token (lowered): remaining tokens}``.

    Only for blocks keyed by their first token.  Two rows sharing one
    leading token collapse to the last -- use :func:`fdf_block` for a
    block whose rows repeat (``TS.Elecs``, coordinates, lattice vectors).
    """
    rows = fdf_block(text, name)
    return {} if rows is None else {r[0].lower(): r[1:] for r in rows if r}


def _energy_ev(value: str, unit: str, where: str) -> float:
    """``value unit`` in eV, through `molbuilder.units`.

    No default: fdf's own rule for a bare energy is Ry and this block is
    written in eV, so either guess is wrong by 13.6x for the other.
    """
    from molbuilder.units import energy_ev
    return energy_ev(value, unit, what=f"energy {value!r}", source=where)


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
    where = f"%block {block}"
    return (_energy_ev(lo[0], lo[1] if len(lo) > 1 else "", where),
            _energy_ev(hi[0], hi[1] if len(hi) > 1 else "", where))
