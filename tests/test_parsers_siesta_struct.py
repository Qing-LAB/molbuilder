"""L1 tests for molbuilder.parsers.siesta_struct.

Pins the .XV and .fdf-initial-coords readers' contracts.  See
``docs/execution/job-contracts.md`` for source priority.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.parse.coords.siesta_xv import (
    SiestaFdfStructureError,
    SiestaXVError,
    read_fdf_initial_coords,
    read_xv,
)


# --------------------------------------------------------------------- #
#  .XV reader                                                           #
# --------------------------------------------------------------------- #


_BOHR = 0.5291772108  # Å per Bohr


def _h2_xv() -> str:
    """Minimal valid .XV: 2 atoms, cell, three rows of cell+vel,
    then atom count + atom rows."""
    # Cell rows: 3 floats (cell vector in Bohr) + 3 floats (velocity).
    cell = "  10.0   0.0   0.0   0.0   0.0   0.0\n" \
           "   0.0  10.0   0.0   0.0   0.0   0.0\n" \
           "   0.0   0.0  10.0   0.0   0.0   0.0\n"
    n = "  2\n"
    # ispec iza  x        y       z       vx vy vz   (Bohr / Bohr·fs⁻¹)
    a1 = "  1   1   0.000   0.000   0.000   0.0 0.0 0.0\n"
    a2 = "  1   1   1.500   0.000   0.000   0.0 0.0 0.0\n"
    return cell + n + a1 + a2


def test_read_xv_parses_minimal_h2(tmp_path):
    p = tmp_path / "h2.XV"
    p.write_text(_h2_xv())
    s = read_xv(p)
    assert s.elements == ["H", "H"]
    assert s.positions.shape == (2, 3)
    # Bohr -> Å conversion.  Position 1 was at x=1.5 Bohr.
    np.testing.assert_allclose(s.positions[1, 0], 1.5 * _BOHR, atol=1e-9)
    # Title carries the file stem so downstream re-render labels it.
    assert s.title == "h2"


def test_read_xv_rejects_short_file(tmp_path):
    p = tmp_path / "broken.XV"
    p.write_text("0.0 0.0 0.0\n")
    with pytest.raises(SiestaXVError) as exc:
        read_xv(p)
    assert "too short" in str(exc.value).lower()


def test_read_xv_rejects_atom_count_mismatch(tmp_path):
    """N declared but fewer atom lines."""
    p = tmp_path / "mismatch.XV"
    p.write_text(
        "  10.0 0 0 0 0 0\n"
        "  0 10.0 0 0 0 0\n"
        "  0 0 10.0 0 0 0\n"
        "  3\n"
        "  1   1 0 0 0 0 0 0\n"
    )
    with pytest.raises(SiestaXVError) as exc:
        read_xv(p)
    assert "declares 3" in str(exc.value)


def test_read_xv_rejects_out_of_range_z(tmp_path):
    p = tmp_path / "weirdz.XV"
    p.write_text(
        "  10.0 0 0 0 0 0\n"
        "  0 10.0 0 0 0 0\n"
        "  0 0 10.0 0 0 0\n"
        "  1\n"
        "  1   500 0 0 0 0 0 0\n"
    )
    with pytest.raises(SiestaXVError) as exc:
        read_xv(p)
    assert "outside the element table" in str(exc.value)


# --------------------------------------------------------------------- #
#  .fdf initial-coords reader                                           #
# --------------------------------------------------------------------- #


def _h2_fdf_ang() -> str:
    """Tiny .fdf with all the blocks the reader needs."""
    return (
        "SystemLabel h2\n"
        "NumberOfAtoms 2\n"
        "NumberOfSpecies 1\n"
        "%block ChemicalSpeciesLabel\n"
        "    1    1    H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "AtomicCoordinatesFormat Ang\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        "    0.000   0.000   0.000   1\n"
        "    0.740   0.000   0.000   1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )


def test_read_fdf_initial_coords_ang_units():
    s = read_fdf_initial_coords(_h2_fdf_ang())
    assert s.elements == ["H", "H"]
    np.testing.assert_allclose(s.positions[1], [0.74, 0.0, 0.0])


def test_read_fdf_initial_coords_bohr_units():
    text = _h2_fdf_ang().replace(
        "AtomicCoordinatesFormat Ang",
        "AtomicCoordinatesFormat Bohr",
    )
    s = read_fdf_initial_coords(text)
    # 0.74 Bohr -> 0.74 * 0.529... Å
    np.testing.assert_allclose(s.positions[1, 0], 0.74 * _BOHR, atol=1e-9)


def test_read_fdf_initial_coords_handles_path(tmp_path):
    p = tmp_path / "h2.fdf"
    p.write_text(_h2_fdf_ang())
    s = read_fdf_initial_coords(p)
    assert s.elements == ["H", "H"]


def test_read_fdf_initial_coords_missing_species_block():
    text = (
        "AtomicCoordinatesFormat Ang\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        "    0.0 0.0 0.0    1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    with pytest.raises(SiestaFdfStructureError) as exc:
        read_fdf_initial_coords(text)
    assert "ChemicalSpeciesLabel" in str(exc.value)


def test_read_fdf_initial_coords_missing_coords_block():
    text = (
        "%block ChemicalSpeciesLabel\n"
        "    1    1    H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "AtomicCoordinatesFormat Ang\n"
    )
    with pytest.raises(SiestaFdfStructureError) as exc:
        read_fdf_initial_coords(text)
    assert "AtomicCoordinatesAndAtomicSpecies" in str(exc.value)


def test_read_fdf_initial_coords_unknown_format():
    text = _h2_fdf_ang().replace(
        "AtomicCoordinatesFormat Ang",
        "AtomicCoordinatesFormat WeirdFormat",
    )
    with pytest.raises(SiestaFdfStructureError) as exc:
        read_fdf_initial_coords(text)
    assert "unsupported" in str(exc.value).lower()


def test_read_fdf_initial_coords_lattice_constant_default_unit_is_bohr():
    """Per the SIESTA manual, ``LatticeConstant`` without an explicit
    unit means Bohr.  Pre-fix the regex required a unit and silently
    failed to match ``LatticeConstant 10.0``; cell scale was then
    raw Å, off by 0.529 ×.  Audit BLOCKER 1."""
    text = (
        "%block ChemicalSpeciesLabel\n"
        "    1    1    H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "LatticeConstant  10.0\n"
        "%block LatticeVectors\n"
        "  1.0 0.0 0.0\n"
        "  0.0 1.0 0.0\n"
        "  0.0 0.0 1.0\n"
        "%endblock LatticeVectors\n"
        "AtomicCoordinatesFormat Fractional\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        "  0.5 0.0 0.0    1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    s = read_fdf_initial_coords(text)
    # 0.5 fractional in a 10 Bohr cube → 5 Bohr = 5 × 0.5291772108 Å.
    np.testing.assert_allclose(s.positions[0, 0], 5.0 * _BOHR, atol=1e-9)


def test_extract_system_label_finds_canonical_directive():
    from molbuilder.parse.coords.siesta_xv import extract_system_label
    assert extract_system_label("SystemLabel h2\nBlockSize 64\n") == "h2"


def test_extract_system_label_handles_indented_and_mixed_case():
    from molbuilder.parse.coords.siesta_xv import extract_system_label
    assert extract_system_label("   systemlabel  my-job\n") == "my-job"


def test_extract_system_label_returns_none_when_absent():
    from molbuilder.parse.coords.siesta_xv import extract_system_label
    assert extract_system_label("# no SystemLabel here\n") is None


# --------------------------------------------------------------------- #
#  check_xv_handedness + check_fdf_handedness                            #
# --------------------------------------------------------------------- #


def test_check_xv_handedness_returns_none_for_right_handed_cell(tmp_path):
    """Identity cell has det = +1; no warning."""
    from molbuilder.parse.coords.siesta_xv import check_xv_handedness
    p = tmp_path / "right.XV"
    p.write_text(_h2_xv())
    assert check_xv_handedness(p) is None


def test_check_xv_handedness_warns_on_left_handed_cell(tmp_path):
    """Flip the third cell vector → det = -1."""
    from molbuilder.parse.coords.siesta_xv import check_xv_handedness
    p = tmp_path / "left.XV"
    # Cell row 3 has z = -10 (negated) → det = -1000.
    p.write_text(
        "  10.0   0.0   0.0   0.0 0.0 0.0\n"
        "   0.0  10.0   0.0   0.0 0.0 0.0\n"
        "   0.0   0.0 -10.0   0.0 0.0 0.0\n"
        "  1\n"
        "  1   1   0.000   0.000   0.000   0.0 0.0 0.0\n"
    )
    warn = check_xv_handedness(p)
    assert warn is not None
    assert "LEFT-HANDED" in warn
    assert "chirality" in warn.lower()
    assert "left.XV" in warn


def test_check_xv_handedness_returns_none_on_unreadable(tmp_path):
    from molbuilder.parse.coords.siesta_xv import check_xv_handedness
    # Path doesn't exist.
    assert check_xv_handedness(tmp_path / "nope.XV") is None


def test_check_fdf_handedness_warns_on_left_handed_lattice():
    from molbuilder.parse.coords.siesta_xv import check_fdf_handedness
    text = (
        "%block LatticeVectors\n"
        "  1.0 0.0 0.0\n"
        "  0.0 1.0 0.0\n"
        "  0.0 0.0 -1.0\n"
        "%endblock LatticeVectors\n"
    )
    warn = check_fdf_handedness(text)
    assert warn is not None
    assert "LEFT-HANDED" in warn
    assert "chirality" in warn.lower()


def test_check_fdf_handedness_returns_none_when_no_lattice_block():
    """No LatticeVectors block -> nothing to check.  Fractional coords
    would fail in read_fdf_initial_coords for a separate reason."""
    from molbuilder.parse.coords.siesta_xv import check_fdf_handedness
    assert check_fdf_handedness("SystemLabel x\n") is None


def test_check_fdf_handedness_returns_none_for_right_handed():
    from molbuilder.parse.coords.siesta_xv import check_fdf_handedness
    text = (
        "%block LatticeVectors\n"
        "  1.0 0.0 0.0\n"
        "  0.0 1.0 0.0\n"
        "  0.0 0.0 1.0\n"
        "%endblock LatticeVectors\n"
    )
    assert check_fdf_handedness(text) is None


def test_read_fdf_initial_coords_fractional_uses_lattice():
    """Fractional needs LatticeVectors + projection.  Atom at frac
    [0.5, 0, 0] in a 10×10×10 Å box → 5 Å."""
    text = (
        "%block ChemicalSpeciesLabel\n"
        "    1    1    H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "LatticeConstant  10.0  Ang\n"
        "%block LatticeVectors\n"
        "  1.0 0.0 0.0\n"
        "  0.0 1.0 0.0\n"
        "  0.0 0.0 1.0\n"
        "%endblock LatticeVectors\n"
        "AtomicCoordinatesFormat Fractional\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        "  0.5 0.0 0.0    1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    s = read_fdf_initial_coords(text)
    np.testing.assert_allclose(s.positions[0], [5.0, 0.0, 0.0])
