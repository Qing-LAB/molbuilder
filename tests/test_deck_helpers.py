"""The deck readers in ``tests/_deck.py``, which nothing else checks.

Six test files assert through these, so an error here is an error in all
of them — and the dangerous direction is silent: a unit factor that is
wrong, or a bound read off the wrong end, makes every caller agree with
the deck for the wrong reason and nothing goes red.

Only the logic is covered. The thin delegations to
``molbuilder.parse.fdf._parse_fdf`` are that module's to test.
"""
from __future__ import annotations

import pytest

from molbuilder.units import UnknownUnit

from _deck import (assert_fdf, fdf_block, fdf_block_rows, fdf_energy_window,
                   fdf_sets, fdf_value)

_WINDOW = """\
SystemLabel probe
%block TBT.Contour.window
  part line
   from -2.00000 eV to 2.00000 eV
    points 401
     method mid-rule
%endblock TBT.Contour.window
"""


# --------------------------------------------------------------------- #
#  scalars                                                              #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("spelling", [
    "SystemLabel", "system_label", "System.Label", "SYSTEMLABEL",
    "system-label",
])
def test_a_keyword_is_found_by_any_fdf_spelling(spelling):
    """fdf matches ignoring case, ``.``, ``-`` and ``_``, so a test that
    insists on one spelling reports a deck the engine reads fine."""
    assert fdf_value(_WINDOW, spelling) == "probe"


def test_a_keyword_the_deck_does_not_set_reads_as_None():
    assert fdf_value(_WINDOW, "MeshCutoff") is None
    assert fdf_sets(_WINDOW, "MeshCutoff") is False


def test_assert_fdf_says_which_of_the_two_failures_it_is():
    """Absent and wrong are different repairs, so they read differently."""
    with pytest.raises(AssertionError, match="does not set"):
        assert_fdf(_WINDOW, "MeshCutoff", "250")
    with pytest.raises(AssertionError, match="deck says"):
        assert_fdf(_WINDOW, "SystemLabel", "other")


def test_spacing_is_not_part_of_the_value():
    """The reason this module exists: padding changed, 45 tests failed,
    none of them meant to be testing padding."""
    assert fdf_value("SystemLabel      probe\n", "SystemLabel") == "probe"


# --------------------------------------------------------------------- #
#  blocks                                                               #
# --------------------------------------------------------------------- #

def test_a_block_comes_back_tokenised_row_by_row():
    assert fdf_block(_WINDOW, "TBT.Contour.window")[0] == ["part", "line"]


def test_an_absent_block_is_None_and_its_rows_are_empty():
    assert fdf_block(_WINDOW, "TS.Elecs") is None
    assert fdf_block_rows(_WINDOW, "TS.Elecs") == {}


def test_block_rows_key_on_the_first_token():
    rows = fdf_block_rows(_WINDOW, "TBT.Contour.window")
    assert rows["points"] == ["401"]
    assert rows["part"] == ["line"]


def test_repeated_leading_tokens_collapse_and_that_is_documented():
    """THE TRAP IN `fdf_block_rows`, pinned so it cannot surprise: a
    block whose rows repeat a leading token keeps only the last, which is
    why coordinates and lattice vectors must use `fdf_block`."""
    text = "%block X\n a 1\n a 2\n%endblock X\n"
    assert fdf_block_rows(text, "X") == {"a": ["2"]}
    assert fdf_block(text, "X") == [["a", "1"], ["a", "2"]]


# --------------------------------------------------------------------- #
#  the from ... to ... row                                              #
# --------------------------------------------------------------------- #

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


@pytest.mark.parametrize("row,match,exc", [
    ("from -2.0 eV 2.0 eV", "no `to`",          AssertionError),
    ("from to 2.0 eV",      "missing a bound",  AssertionError),
    ("from -2.0 eV to",     "missing a bound",  AssertionError),
    ("from x eV to 2.0 eV", "not a number",     UnknownUnit),
    ("from -2.0 furlongs to 2.0 furlongs", "does not know how to convert",
     UnknownUnit),
])
def test_a_malformed_row_names_the_block_and_the_problem(row, match, exc):
    """The row GRAMMAR is this module's; the unit vocabulary is
    `molbuilder.units`'.  Both refusals must still name the block."""
    text = _WINDOW.replace("from -2.00000 eV to 2.00000 eV", row)
    with pytest.raises(exc, match=match) as e:
        fdf_energy_window(text, "TBT.Contour.window")
    assert "TBT.Contour.window" in str(e.value), (
        "a refusal must say which block it is about")
