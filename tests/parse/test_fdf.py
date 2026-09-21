"""`parse/fdf.py` — the one reader of SIESTA's input format.

**Why this file exists.** Until 2026-09-17 there were eight readers of deck
content across the tree and only one implemented fdf's actual keyword rule.
The others were hand-rolled regexes and awk that each got part of it, and two
of them disagreed about the same file.  The survivors all ask this module now,
so its rules are worth stating once, here.

The two cases below came FROM the wrapper's tests: they pinned an awk that read
`SystemLabel` out of a deck at launch.  That awk is deleted — the wrapper is
told its label (`gpu.md` G7) — but the behaviours it was pinned for are real
and still needed, because `web/blueprints/watch.py` reads a directory a person
points at, where there is no description to ask and the deck is all there is.
So the coverage moved rather than went.
"""
from __future__ import annotations

import pytest

from molbuilder.constants import BOHR_ANGSTROM as _B
from molbuilder.parse.fdf import (_norm, _parse_fdf, parse_fdf_params,
                                  system_label)


class TestTheKeywordRuleIsTheFormatS:
    """fdf matches a keyword ignoring case AND `.`, `-`, `_`.

    All four spellings below are one keyword and SIESTA accepts each.  A regex
    anchored on the literal word matches the first and last only, which is what
    every hand-rolled reader did — `web/blueprints/watch.py` resolved a deck
    spelled `System.Label` to nothing, so the Results tab found no trajectory.
    """

    @pytest.mark.parametrize("spelling", [
        "SystemLabel", "systemlabel", "SYSTEMLABEL",
        "System.Label", "system_label", "System-Label",
    ])
    def test_every_legal_spelling_is_the_same_keyword(self, spelling):
        assert system_label(f"{spelling}  bdt\n") == "bdt"

    def test_norm_is_that_rule_and_nothing_else(self):
        assert _norm("System.Label") == _norm("system_label") == "systemlabel"


class TestAQuotedValueReadsAsSIESTAReadsIt:
    """`SystemLabel "foo"` is legal fdf and SIESTA then writes `foo.DM`.

    A reader that keeps the quotes looks for files that do not exist.  Measured
    when this landed: the wrapper's cold sweep missed the warm files entirely
    and would have overwritten them without a word.
    """

    @pytest.mark.parametrize("line,expect", [
        ('SystemLabel "foo"', "foo"),
        ("SystemLabel 'foo'", "foo"),
        ("SystemLabel foo", "foo"),
    ])
    def test_a_quote_pair_is_stripped(self, line, expect):
        assert system_label(line + "\n") == expect

    def test_an_unmatched_quote_is_not_stripped(self):
        """Only a PAIR is syntax.  A lone quote is a character in the value —
        and one the basename validator would have refused, so leaving it means
        the caller's own charset check sees it and falls back."""
        assert system_label('SystemLabel "foo\n') == '"foo'


class TestTheAbsentCases:
    def test_a_deck_that_states_none_answers_none(self):
        assert system_label("MeshCutoff 300 Ry\n") is None

    def test_an_empty_value_is_not_a_label(self):
        """`None`, not `""` — its sibling `pyscf.input.job_name` answers the
        same question the same way, so a caller cannot need two idioms."""
        assert system_label('SystemLabel ""\n') is None


class TestRepeatedKeywords:
    def test_the_first_wins_because_libfdf_takes_the_first(self):
        """`siesta/layout.py::check_rules` states the rule and cites the
        lookup: *"libfdf takes the FIRST match and ignores the rest
        (`fdf_locate` walks from the top and stops)"*.

        molbuilder never emits one — the deck gate refuses a keyword written
        twice with different values — so this only arises in a deck someone
        hand-edited or another tool wrote, which is exactly what this reader is
        pointed at.
        """
        assert _parse_fdf("MeshCutoff 300 Ry\nMeshCutoff 500 Ry\n")[0][
            "meshcutoff"] == ["300", "Ry"]

    def test_a_comment_is_not_a_value(self):
        assert system_label("SystemLabel bdt  # the junction\n") == "bdt"


class TestTheFermiLevelIsKept:
    """SIESTA prints `Ef(eV)` in its SCF table, and the parser dropped it.

    `_SCF_COLUMN_KEYS` mapped `ef` to ``None`` -- *"valid bookkeeping column
    we don't extract"* -- since the parser was written.  For an ordinary run
    that was right: nothing plotted it.

    **For a transport LEAD it is the one number that matters.**  An electrode
    is a periodic bulk run and `engines/transport.md` says what its E_F is
    for: *"its E_F is the reference energy"* -- what T(E) is measured relative
    to, and where `G = G0 * T(E_F)` is evaluated.  `electrode_kz` defaults to
    40 because that is "the Fermi-level resolution", and moving it invalidates
    both lead stages *"because the lead's Fermi level moved"*.

    Kept 2026-09-18 so the Results tab can show it per electrode.
    """

    def test_the_closed_shell_column_lands_in_the_cycle(self):
        from molbuilder.parse.engines.siesta import (
            _build_cycle_dict_from_header, _parse_scf_header)
        head = _parse_scf_header(
            "   iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     "
            "dDmax     Ef(eV) dHmax(eV)")
        cyc = _build_cycle_dict_from_header(
            3, [-1.0, -2.0, -3.0, 0.01, -4.83, 0.02], head)
        assert cyc["ef"] == -4.83, cyc

    def test_the_spin_polarised_columns_land_too(self):
        """A collinear run prints Ef_up and Ef_dn instead of one Ef."""
        from molbuilder.parse.engines.siesta import (
            _build_cycle_dict_from_header, _parse_scf_header)
        head = _parse_scf_header(
            "   iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     "
            "dDmax     Ef_up Ef_dn(eV) dHmax(eV)")
        cyc = _build_cycle_dict_from_header(
            2, [-1.0, -2.0, -3.0, 0.01, -4.8, -4.9, 0.02], head)
        assert cyc["ef_up"] == -4.8 and cyc["ef_dn"] == -4.9, cyc

    def test_the_energy_still_lands_where_it_did(self):
        """THE DISCRIMINATING HALF: keeping a column must not move another.
        `energy` is what every plot reads."""
        from molbuilder.parse.engines.siesta import (
            _build_cycle_dict_from_header, _parse_scf_header)
        head = _parse_scf_header(
            "   iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     "
            "dDmax     Ef(eV) dHmax(eV)")
        cyc = _build_cycle_dict_from_header(
            3, [-1.0, -2.0, -3.0, 0.01, -4.83, 0.02], head)
        assert cyc["energy"] == -2.0 and cyc["cycle"] == 3


class TestTheCoordinateFormatsAreConverted:
    """``coords_ang`` is the frozen gate's baseline.

    `transport/compose.py` hands it to the extraction as the geometry the
    relaxation STARTED from, and every electrode atom is compared against
    the ``.XV`` at 1e-3 A.  A wrong conversion here does not look like a
    parse error -- it looks like a lead that moved, so a correct junction
    is refused and the person is told to re-relax something that is fine.

    All six formats SIESTA accepts here, plus the refusal: nothing
    referenced ``coords_ang`` from a test before this.
    """

    CELL = ("%block LatticeVectors\n 10.0 0 0\n 0 10.0 0\n 0 0 20.0\n"
            "%endblock LatticeVectors\n")

    def _deck(self, fmt: str, z: float) -> str:
        return (self.CELL + f"AtomicCoordinatesFormat {fmt}\n"
                "%block AtomicCoordinatesAndAtomicSpecies\n"
                " 0.0 0.0 0.0 1\n"
                f" 0.0 0.0 {z} 1\n"
                "%endblock AtomicCoordinatesAndAtomicSpecies\n")

    @pytest.mark.parametrize("fmt,z_in", [
        ("Ang",                    2.5),
        ("NotScaledCartesianAng",  2.5),
        ("Bohr",                   2.5 / _B),
        ("NotScaledCartesianBohr", 2.5 / _B),
        ("Fractional",             0.125),      # of c = 20 A
        ("ScaledByLatticeVectors", 0.125),
    ])
    def test_every_accepted_format_lands_in_angstrom(self, fmt, z_in):
        """Same physical atom, six spellings, one answer."""
        p = parse_fdf_params(self._deck(fmt, z_in))
        assert p.coords_ang is not None, f"{fmt} produced no coordinates"
        assert p.coords_ang[1][2] == pytest.approx(2.5, abs=1e-9), (
            f"{fmt}: 2.5 A came back as {p.coords_ang[1][2]}")
        assert p.atom_z_span_ang == pytest.approx(2.5, abs=1e-9)

    def test_a_format_that_cannot_be_converted_is_REFUSED_not_guessed(self):
        """`ScaledCartesian` scales by the lattice CONSTANT, which this
        does not read -- so it answers None, and `compose` refuses the
        citation rather than comparing against invented coordinates."""
        p = parse_fdf_params(self._deck("ScaledCartesian", 2.5))
        assert p.coords_ang is None

    def test_fractional_without_a_cell_is_refused(self):
        """Fractional coordinates mean nothing without the vectors that
        scale them."""
        text = ("AtomicCoordinatesFormat Fractional\n"
                "%block AtomicCoordinatesAndAtomicSpecies\n"
                " 0.0 0.0 0.0 1\n 0.0 0.0 0.125 1\n"
                "%endblock AtomicCoordinatesAndAtomicSpecies\n")
        assert parse_fdf_params(text).coords_ang is None

    def test_a_row_that_is_not_numbers_takes_the_whole_block_down(self):
        """Half a geometry is worse than none: the gate must not compare
        against a partly-read structure."""
        text = (self.CELL + "AtomicCoordinatesFormat Ang\n"
                "%block AtomicCoordinatesAndAtomicSpecies\n"
                " 0.0 0.0 0.0 1\n x y z 1\n"
                "%endblock AtomicCoordinatesAndAtomicSpecies\n")
        assert parse_fdf_params(text).coords_ang is None
