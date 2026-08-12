"""Tests for the convenient .XV extraction API/CLI
(``molbuilder.parse.coords.xv_to_xyz`` + ``molbuilder xv2xyz``).

The key contract: a SIESTA ``.XV`` carries the periodic cell, and the
translation must PRESERVE it (as an ASE extended-XYZ ``Lattice=`` header) so
a downstream describe + prep keeps the real cell, not a vacuum box.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from molbuilder import cli
from molbuilder.parse.coords import read_xv, read_xv_cell, xv_to_xyz

_ANG = 0.5291772108

# Minimal 3-atom .XV: cubic 10-Bohr cell + C, H, Au (Z=6,1,79); coords Bohr.
_XV = (
    "   10.0 0.0 0.0   0.0 0.0 0.0\n"
    "   0.0 10.0 0.0   0.0 0.0 0.0\n"
    "   0.0 0.0 10.0   0.0 0.0 0.0\n"
    "   3\n"
    "  1   6   0.0 0.0 0.0   0.0 0.0 0.0\n"
    "  2   1   1.0 0.0 0.0   0.0 0.0 0.0\n"
    "  3  79   0.0 2.0 0.0   0.0 0.0 0.0\n"
)


@pytest.fixture
def xv(tmp_path) -> Path:
    p = tmp_path / "j.XV"
    p.write_text(_XV)
    return p


def test_read_xv_elements_and_units(xv):
    s = read_xv(xv)
    assert s.elements == ["C", "H", "Au"]      # keyed off Z
    assert s.n_atoms == 3
    # coords converted Bohr -> Å
    assert s.positions[1][0] == pytest.approx(1.0 * _ANG, rel=1e-6)


def test_read_xv_cell_in_angstrom(xv):
    cell = read_xv_cell(xv)
    assert cell is not None
    assert cell[0][0] == pytest.approx(10.0 * _ANG, rel=1e-6)


def test_xv_to_xyz_preserves_cell(xv, tmp_path):
    out = tmp_path / "j.xyz"
    text = xv_to_xyz(xv, out)
    assert out.is_file()
    lines = text.splitlines()
    assert lines[0].strip() == "3"                 # atom count
    assert 'Lattice="' in lines[1]                 # cell on comment line
    assert "Properties=species:S:1:pos:R:3" in lines[1]
    # the 10-Bohr cell -> ~5.2918 Å appears in the Lattice header
    assert f"{10.0 * _ANG:.8f}" in lines[1]


def test_xv_to_xyz_roundtrips_through_struct_reader(xv, tmp_path):
    out = tmp_path / "j.xyz"
    xv_to_xyz(xv, out)
    from molbuilder.siesta.input import _struct_from_file
    s, cell = _struct_from_file(str(out))
    assert s.n_atoms == 3
    assert cell is not None
    assert cell[2][2] == pytest.approx(10.0 * _ANG, rel=1e-6)


def test_cli_xv2xyz(xv, tmp_path):
    out = tmp_path / "out.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(xv), str(out)])
    assert res.exit_code == 0, res.output
    assert out.is_file()
    # The artifact says 3 atoms where an XYZ says it -- its first line.
    # (Replaced `"444" not in output`, a negative on an arbitrary literal
    # that could never fail; found 2026-08-12.)
    assert out.read_text().splitlines()[0].strip() == "3"
    assert 'Lattice="' in out.read_text().splitlines()[1]
