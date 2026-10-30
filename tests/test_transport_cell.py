"""Cell/boundary preservation in the transport emitter + sidecar
round-trip (the hex-Au(111) fix)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder.sidecars import molstruct as msj
from molbuilder.structure import Structure
from molbuilder.parse.fdf import parse_fdf_params
from molbuilder.transport.transiesta import _emit_geometry

# A hexagonal Au(111)-like cell: a,b at 60 deg ~17.3 A; c (transport) 40 A.
HEX = np.array([[17.30, 0.0, 0.0],
                [8.65, 14.98, 0.0],
                [0.0, 0.0, 40.0]])


def _hex_device():
    d = 2.40
    elems, pos, L, B, R = [], [], [], [], []
    i = 0
    for k in range(4):
        elems.append("Au"); pos.append([1.0, 1.0, k * d]); L.append(i); i += 1
    z0 = 4 * d
    for j in range(2):
        elems.append("S"); pos.append([1.0, 1.0, z0 + 1.5 + j * 1.8]); B.append(i); i += 1
    z1 = z0 + 1.5 + 2 * 1.8 + 1.5
    for k in range(4):
        elems.append("Au"); pos.append([1.0, 1.0, z1 + k * d]); R.append(i); i += 1
    return Structure(elements=elems, positions=np.asarray(pos, float), cell=HEX,
                     regions={"L-electrode": L, "bridge": B, "R-electrode": R})


# ------------------------------------------------------------------ #
#  Structure + sidecar carry the cell                                #
# ------------------------------------------------------------------ #


def test_structure_cell_and_periodicity_defaults():
    """No cell means a vacuum box; a stated cell means a lattice.  The
    boolean view follows from the kind rather than being stored beside it."""
    s = Structure(elements=["C"], positions=np.zeros((1, 3)))
    assert s.cell is None
    assert s.axis_kind == ("isolated",) * 3 and s.pbc() == (False,) * 3
    s2 = Structure(elements=["C"], positions=np.zeros((1, 3)), cell=HEX)
    assert s2.cell.shape == (3, 3)
    assert s2.axis_kind == ("periodic",) * 3 and s2.pbc() == (True,) * 3


def test_structure_cell_validation():
    with pytest.raises(ValueError):
        Structure(elements=["C"], positions=np.zeros((1, 3)),
                  cell=np.zeros((2, 3)))


def test_a_degenerate_cell_is_reported_rather_than_unopenable():
    """Zero volume (two parallel vectors) must be said out loud -- but by the
    checker, not by refusing to build the object.

    § 8.2 says reading does not judge: a model that cannot HOLD a bad box
    cannot show one on the Cell page either, so it could not be fixed.
    """
    from molbuilder.cell import resolve_and_check
    bad = np.array([[10.0, 0, 0], [10.0, 0, 0], [0, 0, 10.0]])
    s = Structure(elements=["C"], positions=np.zeros((1, 3)), cell=bad)
    assert s.cell is not None, "the box the user has to fix was dropped"
    _rc, issues = resolve_and_check(s)
    assert [i.where for i in issues] == ["cell.no_volume"]


def test_copy_translated_preserve_cell_and_axis_kind():
    s = Structure(elements=["C", "N"], positions=np.array([[0, 0, 0], [1.0, 0, 0]]),
                  cell=HEX, axis_kind=("periodic", "periodic", "isolated"))
    for clone in (s.copy(), s.translated([1.0, 2.0, 3.0]), s.centered()):
        assert clone.cell is not None and np.allclose(clone.cell, HEX)
        assert clone.axis_kind == ("periodic", "periodic", "isolated")
        assert clone.pbc() == (True, True, False)
    # copy is independent (no shared array)
    s.copy().cell[0, 0] = 999.0
    assert s.cell[0, 0] == pytest.approx(17.30)


def test_sidecar_cell_round_trip(tmp_path):
    """The cell and the axis kinds, through the pair: written by the codec,
    read back by its one door -- and `pbc`, retired, is not written again."""
    from molbuilder.workingcopy_structure import StructureCodec
    p = StructureCodec().write(
        Structure(elements=["C"], positions=np.zeros((1, 3)), cell=HEX,
                  axis_kind=("periodic", "periodic", "isolated")),
        tmp_path / "x.xyz")
    assert "pbc" not in json.loads(msj.sidecar_path_for(p).read_text()), (
        "the retired duplicate must not be written again")
    s = StructureCodec().load(p)
    assert np.allclose(s.cell, HEX)
    assert s.axis_kind == ("periodic", "periodic", "isolated")
    assert s.pbc() == (True, True, False)


def test_a_pair_without_a_cell_is_nonperiodic(tmp_path):
    """A plain molecule's pair -- its sidecar written too, since v13 -- reads
    back with no cell, never one invented on the way."""
    from molbuilder.workingcopy_structure import StructureCodec
    p = StructureCodec().write(
        Structure(elements=["C"], positions=np.zeros((1, 3))),
        tmp_path / "x.xyz")
    s = StructureCodec().load(p)
    assert s.cell is None
    assert s.pbc() == (False, False, False)


# ------------------------------------------------------------------ #
#  Emitter preserves the explicit cell verbatim                      #
# ------------------------------------------------------------------ #


def test_emitter_preserves_hex_cell_verbatim():
    fdf = "\n".join(_emit_geometry(_hex_device()))
    p = parse_fdf_params(fdf)
    assert np.allclose(p.cell_ang[0], [17.30, 0.0, 0.0])
    assert np.allclose(p.cell_ang[1], [8.65, 14.98, 0.0])   # NOT squared off

