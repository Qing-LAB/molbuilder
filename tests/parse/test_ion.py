"""``parse/ion.py`` — the basis reach an ``.ion`` file records.

The principal-layer gate (`transport/compose.py`) reads each element's
largest orbital cutoff from the run's own ``<El>.ion``; these pin the
anchor (the ``#orbital`` header), the unit (Bohr in, Å out), and the
honest ``None`` for anything unreadable.
"""
from __future__ import annotations

from molbuilder.parse.ion import max_orbital_rc_ang
# THE CONVERSION THE PARSER USES, imported rather than retyped: a copy here
# would make the test compare two Bohr radii instead of checking the parse.
# It did -- these two lines carried `0.529177` and failed by 1.3e-6 A the day
# the tree stopped having three different values for it.
from molbuilder.constants import BOHR_ANGSTROM

_HEADER = "  0  6  1  0  1.000000   #orbital l, n, z, is_polarized, population\n"


def test_the_largest_orbital_cutoff_wins_in_angstrom(tmp_path):
    f = tmp_path / "Au.ion"
    f.write_text(
        _HEADER + " 500  0.48E-02  4.000000  # npts, delta, cutoff\n"
        + _HEADER + " 500  0.48E-02  6.130000  # npts, delta, cutoff\n")
    rc = max_orbital_rc_ang(f)
    assert abs(rc - 6.13 * BOHR_ANGSTROM) < 1e-6


def test_only_orbital_blocks_count(tmp_path):
    """KB projectors and other sections carry cutoffs too; nothing
    without the ``#orbital`` anchor may contribute."""
    f = tmp_path / "Au.ion"
    f.write_text(
        "  0  1  1.5  # KBs: l, n, energy\n"
        " 500  0.48E-02  99.000000  # npts, delta, cutoff\n"
        + _HEADER + " 500  0.48E-02  6.130000  # npts, delta, cutoff\n")
    rc = max_orbital_rc_ang(f)
    assert abs(rc - 6.13 * BOHR_ANGSTROM) < 1e-6


def test_missing_and_garbage_answer_none(tmp_path):
    assert max_orbital_rc_ang(tmp_path / "absent.ion") is None
    f = tmp_path / "junk.ion"
    f.write_text("nothing orbital about this\n")
    assert max_orbital_rc_ang(f) is None
