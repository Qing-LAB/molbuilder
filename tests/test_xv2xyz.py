"""Tests for the .XV extraction CLI (``molbuilder xv2xyz``) and the one
``.XV`` reader beneath it (``molbuilder.parse.coords.read_xv_with_cell``).

The contract, restated 2026-09-22: a SIESTA ``.XV`` carries the periodic
cell, and the translation must PRESERVE it -- but through the PAIR, not
through a hand-built ``Lattice=`` comment. The pair is what
``siesta/input.py::_struct_from_file`` reopens, so the cell survives into a
describe + prep and the geometry does not arrive as a molecule in a vacuum
box.

TWO MODES. A bare ``.XV`` states the geometry and the lattice, and
everything else is written at `Structure`'s default -- applied by
`Structure`, never restated by the verb. ``--from-run`` says the ``.XV``
is a run of ours: what that run declared -- its labels, held atoms and
axis kinds -- comes from its own deck, through the run door
(`runs.declared`; plan B12, D19), and a ``.XV`` no run of ours holds is
refused.

The axis kinds are the half that matters most and the half a ``.XV`` cannot
state: ``periodic``, ``isolated`` and ``transport`` are three different
physics (`model/structure-periodicity.md` § 2), and only a metadata source
knows which. Pinned below so the verb cannot go back to guessing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_read_xv_with_cell_puts_the_cell_on_the_structure(xv):
    """A file that states a lattice yields a structure that HAS one.

    It did not until 2026-09-22, and that omission was what made `xv2xyz`
    restate the axis kinds by hand: a cell-less Structure defaults to
    `isolated` on every axis, `replace` carries that forward, and attaching
    a cell afterwards does not re-derive it.
    """
    struct, cell = read_xv_with_cell(xv)
    assert struct.n_atoms == 3
    assert cell[2][2] == pytest.approx(10.0 * _ANG, rel=1e-6)
    assert struct.cell is not None
    assert struct.cell[2][2] == pytest.approx(10.0 * _ANG, rel=1e-6)
    # and the kinds come from `Structure`, not from this reader
    assert struct.axis_kind == ("periodic", "periodic", "periodic")


def test_xv2xyz_writes_the_pair(xv, tmp_path):
    out = tmp_path / "j.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(xv), str(out)])
    assert res.exit_code == 0, res.output
    # BOTH halves, or the cell had nowhere to go.
    assert out.is_file()
    assert (tmp_path / "j.molstruct.json").is_file()
    assert out.read_text().splitlines()[0].strip() == "3"


def test_xv2xyz_cell_and_origin_reach_the_siesta_reader(xv, tmp_path):
    """The round trip that matters: what the deck generator reopens.

    This is the assertion the old `Lattice=` header existed to satisfy. It
    still holds with the header gone, because `_struct_from_file` reads the
    PAIR -- which is why deleting the header was safe rather than lucky.

    THE ORIGIN TRAVELS WITH THE CELL. A `.XV` is SIESTA's own frame, the cell's
    corner at (0,0,0), so its structure states an offset of 0
    (`model/structure-periodicity.md` § 6.0, *A stated offset*), and the next
    deck hands the engine the coordinates the run wrote -- not a re-centred
    copy. The coordinates and their origin are set together.
    """
    out = tmp_path / "j.xyz"
    assert CliRunner().invoke(
        cli.cli, ["xv2xyz", str(xv), str(out)]).exit_code == 0

    from molbuilder.cell import to_engine
    from molbuilder.siesta.input import _struct_from_file
    s, cell = _struct_from_file(str(out))
    assert s.n_atoms == 3
    assert cell is not None
    assert cell[2][2] == pytest.approx(10.0 * _ANG, rel=1e-6)
    # A lattice implies periodicity -- stated once, in `Structure`.
    assert s.axis_kind == ("periodic", "periodic", "periodic")
    frame = to_engine(s)
    assert frame.stated, "the pair lost the .XV's origin: the next deck re-centres it"
    np.testing.assert_allclose(frame.positions, read_xv(xv).positions, atol=1e-6,
                               err_msg="the engine would not get the run's coordinates")


# `test_xv2xyz_leaves_frozen_atoms_alone_without_the_flag` retired
# 2026-10-04 (W56 review): it wrote a deck beside the `.XV` to show the
# plain conversion leaves it unread -- and no mode reads a deck beside a
# `.XV` since 4a, so it pinned what `test_xv2xyz_writes_the_pair` does.


# `test_xv2xyz_from_run_recovers_the_declared_frozen_atoms`,
# `test_xv2xyz_from_run_with_nothing_to_find_is_not_an_error` and
# `test_xv2xyz_from_run_applies_the_sidecar_whole` retired 2026-10-04 (W56
# 4a; plan B12, D19): each laid a folder by hand -- a `.XV`, a deck and a
# sidecar written beside it by the test -- and pinned the lookup by name that
# `--from-run` no longer makes.  What a run declared comes from its own deck,
# through the run door: the two tests below, on a measured run of ours and on
# a `.XV` no run holds.

#: A flat calculation of ours, measured through the road (its README): H2,
#: isolated on every axis, the first atom held.
_FLAT_H2 = Path(__file__).parent / "fixtures" / "siesta_flat_h2"


def test_from_run_takes_what_the_run_declared_from_its_own_deck(tmp_path):
    """D19's measured failure, turned round: the flat H2 run's ``H2.XV``,
    converted with ``--from-run``, keeps the atom its deck holds and the axes
    its deck records -- isolated, an H2 in a 10 Å box -- on the ``.XV``'s own
    cell and the engine's origin.  Its folder holds two decks; the run's is
    the one the run door names (`runs.run_of`, `runs.declared`).

    WHY API-LEVEL: a measured fixture, read where it was measured -- the
    road that produced it is in its README."""
    out = tmp_path / "h2.xyz"
    res = CliRunner().invoke(
        cli.cli, ["xv2xyz", str(_FLAT_H2 / "H2.XV"), str(out), "--from-run"])
    assert res.exit_code == 0, res.output
    assert "H2_01_coarse.fdf" in res.output, res.output

    from molbuilder.workingcopy_structure import StructureCodec
    got = StructureCodec().load(out)
    assert got.frozen_atoms == [0]
    assert got.axis_kind == ("isolated", "isolated", "isolated")
    np.testing.assert_allclose(np.asarray(got.cell), np.eye(3) * 10.0,
                               atol=1e-4)
    np.testing.assert_allclose(got.engine_offset, np.zeros(3))


def test_from_run_refuses_a_xv_no_run_of_ours_holds(xv, tmp_path):
    """The flag says a run is there.  A ``.XV`` in a folder no calculation
    marks has nothing declaring its atoms, so it is refused by name -- never
    written at the defaults as if the run had said so.

    WHY API-LEVEL: a refusal the road cannot reach -- every ``.XV`` a road
    run leaves is in a run of ours."""
    out = tmp_path / "j.xyz"
    res = CliRunner().invoke(
        cli.cli, ["xv2xyz", str(xv), str(out), "--from-run"])
    assert res.exit_code != 0
    assert "no run of ours holds" in res.output, res.output
    assert not out.exists()

