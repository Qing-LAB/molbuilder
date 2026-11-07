"""The window reader in ``tests/_deck.py``, which nothing else checks.

`test_transport_decks_e2e.py` asserts through it, so an error here
is an error there -- and the dangerous direction is silent: a unit factor that
is wrong, or a bound read off the wrong end, makes the caller agree with the
deck for the wrong reason and nothing goes red.
"""
from __future__ import annotations

import pytest

from molbuilder.units import UnknownUnit

from _deck import fdf_energy_window

_WINDOW = """\
SystemLabel probe
%block TBT.Contour.window
  part line
   from -2.00000 eV to 2.00000 eV
    points 401
     method mid-rule
%endblock TBT.Contour.window
"""


def test_both_bounds_come_off_one_row():
    """`to` sits inside the `from` row, which is the whole reason this
    reader exists rather than callers slicing tokens."""
    assert fdf_energy_window(_WINDOW, "TBT.Contour.window") == (-2.0, 2.0)


def test_the_bounds_are_not_interchangeable():
    """A window that runs backwards must not read the same as one that
    does not — the mutant that passed 79 tests under a substring check."""
    backwards = _WINDOW.replace("from -2.00000 eV to 2.00000 eV",
                                "from 2.00000 eV to -2.00000 eV")
    assert fdf_energy_window(backwards, "TBT.Contour.window") == (2.0, -2.0)


@pytest.mark.parametrize("unit,factor", [
    ("eV", 1.0), ("Ry", 13.605693122994), ("Ha", 27.211386245988),
    ("rydberg", 13.605693122994), ("HARTREE", 27.211386245988),
])
def test_the_unit_is_converted_not_ignored(unit, factor):
    """The silent direction: a deck that switches unit keeping the same
    figures moves the window 13.6x, and comparing bare numbers misses it."""
    text = _WINDOW.replace("from -2.00000 eV to 2.00000 eV",
                           f"from -2.00000 {unit} to 2.00000 {unit}")
    lo, hi = fdf_energy_window(text, "TBT.Contour.window")
    assert lo == pytest.approx(-2.0 * factor)
    assert hi == pytest.approx(2.0 * factor)


def test_a_bare_energy_is_REFUSED_rather_than_assumed():
    """fdf's own default for a unitless energy is Ry; this block is
    written in eV.  Either guess is wrong by 13.6x for the other, so a
    deck that states no unit gets a refusal, not a reading."""
    text = _WINDOW.replace("from -2.00000 eV to 2.00000 eV",
                           "from -2.00000 to 2.00000")
    with pytest.raises(UnknownUnit, match="states no unit"):
        fdf_energy_window(text, "TBT.Contour.window")


def test_an_absent_window_is_None_not_a_default():
    assert fdf_energy_window("SystemLabel probe\n", "TBT.Contour.window") is None
