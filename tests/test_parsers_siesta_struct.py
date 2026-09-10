"""L1 tests for ``molbuilder.parse.coords.siesta_xv``.

Pins the .XV and .fdf-initial-coords readers' contracts.  See
``docs/execution/job-contracts.md`` for source priority.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.parse.coords.siesta_xv import (
    SiestaXVError,
    read_xv,
)


# --------------------------------------------------------------------- #
#  .XV reader                                                           #
# --------------------------------------------------------------------- #


# THE one Bohr->Angstrom value (`molbuilder/constants.py`).  A literal
# here would be a second home: three of them had already drifted to
# THREE different values by 2026-09-09 (0.5291772108, 0.529177249 --
# CODATA 1986 -- and 0.529177).  Importing is not circular: every use
# below WRITES a fixture in Bohr, and the assertion is on the Angstrom
# value that comes back.
from molbuilder.constants import BOHR_ANGSTROM as _BOHR


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






















# --------------------------------------------------------------------- #
#  check_xv_handedness + check_fdf_handedness                            #
# --------------------------------------------------------------------- #














