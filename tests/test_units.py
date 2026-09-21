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

@pytest.mark.parametrize("word,ev", [
    ("eV", 1.0), ("meV", 1e-3),
    ("Ry", RYDBERG_EV), ("Ryd", RYDBERG_EV), ("rydberg", RYDBERG_EV),
    ("Ha", HARTREE_EV), ("HARTREE", HARTREE_EV),
])
def test_every_energy_word_siesta_accepts_is_known(word, ev):
    """`eV` is the one that mattered: without it, a deck written in eV
    read as Ry and every number after was 13.6x."""
    assert energy_ev(1.0, word) == pytest.approx(ev)


@pytest.mark.parametrize("word,ang", [
    ("Ang", 1.0), ("angstrom", 1.0), ("Bohr", BOHR_ANGSTROM), ("nm", 10.0),
])
def test_every_length_word_is_known(word, ang):
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

def test_energy_ry_is_the_eV_door_divided_once():
    """SIESTA's scalars are stored in Ry, so this is the shape `.fdf`
    reads through; it must not be a second table."""
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


# --------------------------------------------------------------------- #
#  the copies the contract allows, pinned to the one home               #
# --------------------------------------------------------------------- #
#
# `architecture.md` § 3 permits a second spelling in three mechanisms, and
# the value still comes from `constants`.  A copy that cannot import --
# a class body carried by `inspect.getsource` -- is pinned here instead,
# because nothing else can notice it drifting.

def test_the_getsource_copied_emitter_matches_the_one_home():
    """`trajectory_log.emitter`'s conversion class is copied into the
    generated wrapper by `inspect.getsource`, so a module-level import
    would not travel with it and the literals must stay literals.  They
    are still facts about the universe, and this is what keeps them
    equal to the ones every other reader uses."""
    from molbuilder.constants import HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_EV
    from molbuilder.trajectory_log import emitter as em

    holder = next(
        (obj for obj in vars(em).values()
         if isinstance(obj, type) and hasattr(obj, "HARTREE_TO_EV")), None)
    assert holder is not None, (
        "the emitter no longer carries HARTREE_TO_EV -- if it now imports "
        "it, delete this test rather than loosening it")
    assert holder.HARTREE_TO_EV == pytest.approx(HARTREE_EV, rel=0, abs=0)
    force = getattr(holder, "HARTREE_BOHR_TO_EV_ANG", None)
    if force is not None:
        assert force == pytest.approx(HARTREE_BOHR_EV_ANGSTROM_ASE,
                                      rel=0, abs=0), (
            "the emitter's force factor and the one every other reader "
            "uses must be the SAME number, or a force read back does not "
            "equal the force emitted")


def test_an_ARRAY_converts_elementwise():
    """A netCDF reader hands whole coordinate arrays through this door,
    so coercing to a scalar breaks it — which it did, taking 12 tests in
    `parse/test_siesta_mdnc.py` with it."""
    import numpy as np
    out = length_ang(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]), "Bohr")
    assert out.shape == (2, 3)
    assert out[1][2] == pytest.approx(2 * BOHR_ANGSTROM)
