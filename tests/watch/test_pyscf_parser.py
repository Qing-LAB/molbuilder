"""What the PySCF / geomeTRIC trajectory parser claims: an XYZ whose frames
are structures, whatever its comment lines say, and never a file that only
starts with a number.

Its readings of a geomeTRIC trajectory -- energies, forces, SCF history --
need a trajectory a real run wrote; the tests that read one typed by hand
were retired 2026-10-04 (`process/testing.md` § 6).
"""

from __future__ import annotations


# THE SAME DIALECT THE PARSER NAMES, not a second derivation of it.
# `constants.py` keeps two spellings of Hartree/Bohr -> eV/A on purpose --
# the ASE/NIST value and `HARTREE_EV / BOHR_ANGSTROM`, 0.36 ppm apart --
# and says a call site NAMES the one it needs, because a force and the
# threshold drawn over it must share one.  This file composed the second
# while `parse/engines/pyscf.py` imports the first, so the assertion was
# measuring the gap between the two dialects rather than the conversion.
# The gap is on the seventh significant figure: 257.11043095 against
# 257.11033738 (measured 2026-09-21, which is how it surfaced).

from molbuilder.parse.engines.pyscf import PySCFParser


def test_can_parse_rejects_non_xyz(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("just some random text\nhello world\n")
    assert PySCFParser.can_parse(str(p)) is False


def test_can_parse_accepts_plain_xyz_without_iteration_marker(tmp_path):
    """A plain XYZ with any comment text -- not just geomeTRIC's
    `Iteration K Energy E` -- must be accepted.  The detector is
    structural, not banner-based: it should not be tied to one
    specific tool's comment-line format.  ``parse()`` already
    handles the no-energy case (energy=None)."""
    p = tmp_path / "regular.xyz"
    p.write_text("3\nwater\nO 0 0 0\nH 0 0 1\nH 1 0 0\n")
    assert PySCFParser.can_parse(str(p)) is True


def test_can_parse_rejects_text_with_just_a_number(tmp_path):
    """An integer first line is not enough -- a CSV with a count
    header, or any random text starting with a number, must be
    rejected.  The atom-line check (element + 3 floats) is what
    keeps the detector honest."""
    p = tmp_path / "fake.xyz"
    p.write_text("42\nsome header text\nname,age,occupation\n"
                 "alice,30,engineer\nbob,25,scientist\n")
    assert PySCFParser.can_parse(str(p)) is False


def test_can_parse_rejects_zero_or_negative_atom_count(tmp_path):
    """Atom count must be a positive integer.  0 isn't a structure;
    a negative number can't even appear (`isdigit()` rejects it)."""
    p = tmp_path / "zero.xyz"
    p.write_text("0\nempty\n")
    assert PySCFParser.can_parse(str(p)) is False


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 13 tests here parsed a geomeTRIC trajectory, a .qdata.txt or a
# PySCF log typed by hand (`process/testing.md` § 6).
