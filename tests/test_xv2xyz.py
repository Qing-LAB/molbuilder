"""Tests for the .XV extraction CLI (``molbuilder xv2xyz``) and the one
``.XV`` reader beneath it (``molbuilder.parse.coords.read_xv_with_cell``).

The contract, restated 2026-09-22: a SIESTA ``.XV`` carries the periodic
cell, and the translation must PRESERVE it -- but through the PAIR, not
through a hand-built ``Lattice=`` comment. The pair is what
``siesta/input.py::_struct_from_file`` reopens, so the cell survives into a
describe + prep and the geometry does not arrive as a molecule in a vacuum
box. What the ``.XV`` cannot carry -- which atoms were held -- comes from the
run's siblings, and only when ``--from-run`` says to go looking.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from molbuilder import cli
from molbuilder.parse.coords import read_xv, read_xv_cell, read_xv_with_cell

# The one home (`molbuilder/constants.py`); this file carried a stale
# `0.5291772108` until 2026-09-09.
from molbuilder.constants import BOHR_ANGSTROM as _ANG

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

# The deck that drove the run, holding the constraint block `--from-run`
# must recover. 1-based on disk (SIESTA's convention); {0, 2} once routed
# through the engine index API.
_FDF = (
    "SystemLabel j\n"
    "%block Geometry.Constraints\n"
    "  position 1\n"
    "  position 3\n"
    "%endblock Geometry.Constraints\n"
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


def test_read_xv_with_cell_is_one_pass(xv):
    struct, cell = read_xv_with_cell(xv)
    assert struct.n_atoms == 3
    assert cell[2][2] == pytest.approx(10.0 * _ANG, rel=1e-6)


def test_xv2xyz_writes_the_pair(xv, tmp_path):
    out = tmp_path / "j.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(xv), str(out)])
    assert res.exit_code == 0, res.output
    # BOTH halves, or the cell had nowhere to go.
    assert out.is_file()
    assert (tmp_path / "j.molstruct.json").is_file()
    assert out.read_text().splitlines()[0].strip() == "3"


def test_xv2xyz_cell_reaches_the_siesta_reader(xv, tmp_path):
    """The round trip that matters: what the deck generator reopens.

    This is the assertion the old `Lattice=` header existed to satisfy. It
    still holds with the header gone, because `_struct_from_file` reads the
    PAIR -- which is why deleting the header was safe rather than lucky.
    """
    out = tmp_path / "j.xyz"
    assert CliRunner().invoke(
        cli.cli, ["xv2xyz", str(xv), str(out)]).exit_code == 0

    from molbuilder.siesta.input import _struct_from_file
    s, cell = _struct_from_file(str(out))
    assert s.n_atoms == 3
    assert cell is not None
    assert cell[2][2] == pytest.approx(10.0 * _ANG, rel=1e-6)
    # A lattice implies periodicity -- stated once, in `Structure`.
    assert s.axis_kind == ("periodic", "periodic", "periodic")


def test_xv2xyz_leaves_frozen_atoms_alone_without_the_flag(xv, tmp_path):
    """The flag is the whole point: the deck is RIGHT THERE and unread."""
    (tmp_path / "j.fdf").write_text(_FDF)
    out = tmp_path / "j.xyz"
    assert CliRunner().invoke(
        cli.cli, ["xv2xyz", str(xv), str(out)]).exit_code == 0

    from molbuilder.workingcopy_structure import StructureCodec
    assert StructureCodec().load(out).frozen_atoms == []


def test_xv2xyz_from_run_recovers_the_declared_frozen_atoms(xv, tmp_path):
    (tmp_path / "j.fdf").write_text(_FDF)
    out = tmp_path / "j.xyz"
    res = CliRunner().invoke(
        cli.cli, ["xv2xyz", str(xv), str(out), "--from-run"])
    assert res.exit_code == 0, res.output

    from molbuilder.workingcopy_structure import StructureCodec
    # 1-based `position 1` / `position 3` in the deck -> 0-based on the
    # Structure, through `engine_atom_index`, never a bare `n - 1`.
    assert StructureCodec().load(out).frozen_atoms == [0, 2]


def test_xv2xyz_from_run_with_nothing_to_find_is_not_an_error(xv, tmp_path):
    out = tmp_path / "j.xyz"
    res = CliRunner().invoke(
        cli.cli, ["xv2xyz", str(xv), str(out), "--from-run"])
    assert res.exit_code == 0, res.output
    assert "no frozen atoms found" in res.output


def test_xv2xyz_from_run_reads_the_sidecar_over_the_deck(xv, tmp_path):
    """Precedence, shared with the SIESTA `.out` parser: sidecar beats
    `.fdf`, because the sidecar is this structure's own declaration and the
    deck is only what was on disk at some point before the run."""
    (tmp_path / "j.fdf").write_text(_FDF)
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    # THROUGH THE DOOR, in the test too -- the sidecar beside `j.XV` is
    # written by the codec, not hand-packed here.
    StructureCodec().write(
        Structure(elements=["C", "H", "Au"],
                  positions=[[0, 0, 0], [1, 0, 0], [0, 2, 0]],
                  frozen_atoms=[1]),
        tmp_path / "j.xyz")

    out = tmp_path / "out.xyz"
    assert CliRunner().invoke(
        cli.cli, ["xv2xyz", str(xv), str(out), "--from-run"]).exit_code == 0

    assert StructureCodec().load(out).frozen_atoms == [1]
