"""The dialects a physical quantity is written in.

Every reader of an engine file goes through these tables, so a missing
word here is a wrong number everywhere — and wrong in the silent
direction: the value comes back, in the wrong unit, off by a fixed
ratio. That is the failure this module exists to end, so the tests are
mostly about words that must be PRESENT and words that must be REFUSED.
"""
from __future__ import annotations

import pytest

from molbuilder.constants import (BOHR_ANGSTROM, BOLTZMANN_EV_K, HARTREE_EV,
                                  RYDBERG_EV)
from molbuilder.units import (ENERGY_EV, LENGTH_ANGSTROM, TEMPERATURE_K,
                              UnknownUnit, convert, energy_ev, energy_ry,
                              length_ang, temperature_k)


# --------------------------------------------------------------------- #
#  the vocabularies                                                     #
# --------------------------------------------------------------------- #

#: SIESTA 5.4.2's own energy dimension, read out of the shipped binary's
#: libfdf table.  A word the engine accepts and this table lacks is not an
#: error at read time -- it is a refusal where a number was expected.
_SIESTA_ENERGY_WORDS = (
    "eV", "meV", "Ry", "mRy", "Ha", "mHa", "Hartree", "mHartree",
    "K", "Kelvin", "J", "kJ", "erg", "kcal/mol", "kJ/mol",
    "Hz", "THz", "cm-1", "cm^-1", "cm**-1",
)

_SIESTA_LENGTH_WORDS = ("m", "cm", "nm", "Ang", "Bohr", "pm")


@pytest.mark.parametrize("word", _SIESTA_ENERGY_WORDS)
def test_every_energy_word_siesta_accepts_is_known(word):
    """`eV` is the one that mattered: without it, a deck written in eV
    read as Ry and every number after was 13.6x.  The rest are here
    because the same hole is the same hole for any of them."""
    assert energy_ev(1.0, word) > 0.0


@pytest.mark.parametrize("word", _SIESTA_LENGTH_WORDS)
def test_every_length_word_siesta_accepts_is_known(word):
    assert length_ang(1.0, word) > 0.0


@pytest.mark.parametrize("word,ev", [
    ("eV", 1.0), ("meV", 1e-3),
    ("Ry", RYDBERG_EV), ("Ryd", RYDBERG_EV), ("rydberg", RYDBERG_EV),
    ("Ha", HARTREE_EV), ("HARTREE", HARTREE_EV),
])
def test_the_common_energy_words_carry_the_right_factor(word, ev):
    assert energy_ev(1.0, word) == pytest.approx(ev)


@pytest.mark.parametrize("word,ev,why", [
    ("cm-1",     1.239841984e-4, "the wavenumber the vibrational decks speak"),
    ("kcal/mol", 0.0433641,      "the unit a chemist quotes a barrier in"),
    ("K",        8.617333262e-5, "kelvin IS an energy word to SIESTA"),
])
def test_the_derived_energy_words_are_physically_right(word, ev, why):
    """Checked against published values, not against our own arithmetic."""
    assert energy_ev(1.0, word) == pytest.approx(ev, rel=1e-6), why


@pytest.mark.parametrize("word,ang", [
    ("Ang", 1.0), ("angstrom", 1.0), ("Bohr", BOHR_ANGSTROM), ("nm", 10.0),
    ("pm", 1e-2), ("m", 1e10),
])
def test_the_length_words_carry_the_right_factor(word, ang):
    assert length_ang(1.0, word) == pytest.approx(ang)


def test_temperature_accepts_a_temperature_OR_an_energy():
    """SIESTA's `ElectronicTemperature` takes either and means the same
    physics by both, which is the one place the two vocabularies meet."""
    assert temperature_k(300.0, "K") == pytest.approx(300.0)
    for word, value in (("eV", 300.0 * BOLTZMANN_EV_K),
                        ("meV", 300.0 * BOLTZMANN_EV_K * 1e3),
                        ("Ry", 300.0 * BOLTZMANN_EV_K / RYDBERG_EV)):
        assert temperature_k(value, word) == pytest.approx(300.0), word


def test_the_energy_words_all_reach_the_temperature_table():
    """Derived, not retyped — so a word added to one cannot go missing
    from the other."""
    assert set(ENERGY_EV) <= set(TEMPERATURE_K)
    assert {"k", "kelvin"} <= set(TEMPERATURE_K)


def test_the_tables_are_keyed_lowercase_and_matched_case_insensitively():
    assert all(w == w.lower() for w in ENERGY_EV), sorted(ENERGY_EV)
    assert all(w == w.lower() for w in LENGTH_ANGSTROM)
    assert energy_ev(1.0, "  RyDbErG  ") == pytest.approx(RYDBERG_EV)


# --------------------------------------------------------------------- #
#  policy: the caller's, not the table's                                #
# --------------------------------------------------------------------- #

def test_a_bare_value_takes_the_CALLERS_default():
    """The format's rule, not the quantity's: a bare energy in an .fdf is
    Ry, and one in a tbtrans contour block is eV."""
    assert energy_ev(2.0, None, default="ry") == pytest.approx(2 * RYDBERG_EV)
    assert energy_ev(2.0, None, default="ev") == pytest.approx(2.0)


def test_a_bare_value_with_NO_default_is_refused():
    """A reader that has no rule for a bare value must not be given one
    by this module."""
    with pytest.raises(UnknownUnit, match="states no unit"):
        energy_ev(2.0, None)


def test_an_unknown_word_is_refused_and_lists_what_is_known():
    with pytest.raises(UnknownUnit) as e:
        energy_ev(2.0, "furlongs", what="MeshCutoff", source="probe.fdf")
    msg = str(e.value)
    assert "furlongs" in msg and "MeshCutoff" in msg and "probe.fdf" in msg
    assert "ev" in msg and "hartree" in msg, (
        f"a refusal must say what it DOES know, or the person cannot fix "
        f"it: {msg}")


def test_a_value_that_is_not_a_number_names_its_field_too():
    with pytest.raises(UnknownUnit, match="not a number"):
        energy_ev("x", "eV", what="MeshCutoff", source="probe.fdf")


def test_the_refusal_says_why_it_refuses_rather_than_guessing():
    """The sentence is the point: a wrong factor is invisible in the
    result, so silence would be worse than a stop."""
    with pytest.raises(UnknownUnit, match="fixed ratio"):
        energy_ev(2.0, "furlongs")


# --------------------------------------------------------------------- #
#  the Ry door                                                          #
# --------------------------------------------------------------------- #

def test_energy_ry_returns_Ry_exactly():
    """SIESTA's scalars are stored in Ry, and `MeshCutoff 200 Ry` must
    come back as 200.0 -- not 200.00000000000003, which is what
    multiplying to eV and dividing back gives, and which `siesta/input`
    writes into a deck verbatim."""
    from molbuilder.parse.fdf import parse_fdf_params
    for v in (100.0, 200.0, 250.0, 350.0, 400.0):
        assert parse_fdf_params(f"MeshCutoff {v:g} Ry\n").mesh_cutoff_ry == v
    assert energy_ry(1.0, "Ry") == pytest.approx(1.0)
    assert energy_ry(1.0, "Ha") == pytest.approx(2.0)
    assert energy_ry(RYDBERG_EV, "eV") == pytest.approx(1.0)


def test_the_13_6x_case_end_to_end():
    """A mesh cutoff written in eV, which is what a published electrode
    deck may state."""
    assert energy_ry(4080.0, "eV", default="ry") == pytest.approx(
        4080.0 / RYDBERG_EV)
    assert energy_ry(4080.0, "eV", default="ry") == pytest.approx(299.874,
                                                                  abs=1e-3)


# --------------------------------------------------------------------- #
#  convert() itself                                                     #
# --------------------------------------------------------------------- #

def test_convert_is_the_one_door_the_helpers_go_through():
    assert convert(2.0, "Bohr", LENGTH_ANGSTROM, what="x", source="y") == \
        pytest.approx(2 * BOHR_ANGSTROM)


def test_no_table_silently_returns_the_input():
    """The defect this module replaces: an unknown word returning the
    number unchanged.  No table may do that for any word it lacks."""
    for table, door in ((ENERGY_EV, energy_ev),
                        (LENGTH_ANGSTROM, length_ang),
                        (TEMPERATURE_K, temperature_k)):
        with pytest.raises(UnknownUnit):
            door(7.0, "notaunit")
        assert "notaunit" not in table


def test_an_ARRAY_converts_elementwise():
    """A netCDF reader hands whole coordinate arrays through this door,
    so coercing to a scalar breaks it — which it did, taking 12 tests of the
    `.MD.nc` reader with it."""
    import numpy as np
    out = length_ang(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]), "Bohr")
    assert out.shape == (2, 3)
    assert out[1][2] == pytest.approx(2 * BOHR_ANGSTROM)
