"""Structure periodicity fields — axis_kind / vacuum / kgrid + resolve_cell
(structure-periodicity.md, Phase 2a data model).
"""
import numpy as np
import pytest

from molbuilder.structure import Structure


def _s(**kw):
    return Structure(elements=["C", "H"], positions=[[0, 0, 0], [2, 0, 0]], **kw)


class TestAxisKindReconciliation:
    def test_molecule_no_cell_derives_isolated(self):
        s = _s()
        assert s.axis_kind == ("isolated", "isolated", "isolated")
        assert s.pbc == (False, False, False)

    def test_cell_present_derives_periodic(self):
        s = _s(cell=np.eye(3) * 5)
        assert s.axis_kind == ("periodic", "periodic", "periodic")
        assert s.pbc == (True, True, True)

    def test_explicit_axis_kind_derives_pbc(self):
        # transport -> ASE pbc True (periodic box, Γ-sampled); isolated -> False.
        s = _s(cell=np.eye(3) * 5,
               axis_kind=("periodic", "periodic", "transport"))
        assert s.axis_kind == ("periodic", "periodic", "transport")
        assert s.pbc == (True, True, True)

    def test_isolated_axis_kind_derives_pbc_false(self):
        s = _s(cell=np.eye(3) * 5,
               axis_kind=("periodic", "periodic", "isolated"))
        assert s.pbc == (True, True, False)

    def test_axis_kind_wins_over_pbc(self):
        # axis_kind is authoritative: a conflicting pbc is overwritten.
        s = _s(cell=np.eye(3) * 5, pbc=(False, False, False),
               axis_kind=("periodic", "periodic", "periodic"))
        assert s.pbc == (True, True, True)

    def test_invalid_axis_kind_raises(self):
        with pytest.raises(ValueError, match="axis_kind"):
            _s(axis_kind=("periodic", "bogus", "isolated"))

    def test_vacuum_kgrid_defaults(self):
        s = _s()
        assert s.vacuum == (0.0, 0.0, 0.0)
        assert s.kgrid == (1, 1, 1)
        assert _s(kgrid=(4, 4, 0)).kgrid == (4, 4, 1)  # clamped >= 1


class TestResolveCell:
    def test_explicit_cell_wins(self):
        c = np.diag([3.0, 4.0, 5.0])
        assert np.allclose(_s(cell=c).resolve_cell(), c)

    def test_molecule_bbox_plus_vacuum(self):
        # x extent 2 + vacuum 5 = 7; y,z extent 0 + 5 = 5.
        s = _s(vacuum=(5, 5, 5))
        assert np.allclose(s.resolve_cell(), np.diag([7.0, 5.0, 5.0]))

    def test_transport_axis_bbox_no_vacuum(self):
        # a transport axis ignores vacuum (matched device length); needs a cell
        # on the periodic axes, so give one and clear it to force derivation of z.
        s = Structure(
            elements=["Au", "Au"], positions=[[0, 0, 0], [0, 0, 6]],
            axis_kind=("isolated", "isolated", "transport"),
            vacuum=(5, 5, 5),
        )
        cell = s.resolve_cell()
        assert cell[2, 2] == pytest.approx(6.0)   # z extent, NO vacuum
        assert cell[0, 0] == pytest.approx(0.0 + 5.0)  # isolated x gets vacuum

    def test_periodic_axis_without_cell_raises(self):
        s = _s(axis_kind=("periodic", "isolated", "isolated"))
        with pytest.raises(ValueError, match="periodic"):
            s.resolve_cell()

    def test_empty_structure_returns_none(self):
        assert Structure(elements=[], positions=np.zeros((0, 3))).resolve_cell() is None
