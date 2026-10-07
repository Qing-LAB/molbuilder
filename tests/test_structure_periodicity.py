"""Structure periodicity fields — axis_kind / vacuum + resolve_cell
(structure-periodicity.md, Phase 2a data model).  k-grid is deliberately NOT
here: it's a reciprocal-space sampling knob on SiestaConfig.
"""
import numpy as np
import pytest

from molbuilder import cell as cellmod
from molbuilder.structure import Structure


def _two_slabs(struct, element, plane, size, *, gap=8.0, **kw):
    """A junction: two slabs, one per side, each starting at ``gap/2``.

    What these tests are about -- the CELL a build captures -- is
    `_finish_slab`'s.  `sequence="ACB"` on the `-z` side continues the
    crystal rather than mirroring it.
    """
    from molbuilder.modify import add_slab
    out = add_slab(struct, element, plane, size,
                   start_z=gap / 2.0, grow="+z", **kw)
    return add_slab(out, element, plane, size,
                    start_z=-gap / 2.0, grow="-z", sequence="ACB", **kw)


def _s(**kw):
    return Structure(elements=["C", "H"], positions=[[0, 0, 0], [2, 0, 0]], **kw)


class TestAxisKindIsTheOnePeriodicityField:
    """`axis_kind` is the only periodicity state a Structure holds.

    A stored boolean view could not hold a fact `axis_kind` does not (the
    mapping flattens `periodic` and `transport` onto the same True) -- user,
    2026-09-22: *"why the fuck need pbc when axis_kind fully contains this
    information and more"*.

    The boolean survives as the `pbc()` ACCESSOR, for the two outside
    formats that need one — see `TestPbcIsAnInteropAccessor`.
    """

    def test_molecule_no_cell_is_isolated(self):
        assert _s().axis_kind == ("isolated", "isolated", "isolated")

    def test_cell_present_is_periodic(self):
        s = _s(cell=np.eye(3) * 5)
        assert s.axis_kind == ("periodic", "periodic", "periodic")

    def test_transport_is_never_guessed(self):
        """A cell alone says "there is a lattice", not "this is a device".
        Only a builder states `transport`, which is precisely the fact the
        boolean view cannot carry."""
        assert "transport" not in _s(cell=np.eye(3) * 5).axis_kind

    def test_a_stated_kind_is_kept_verbatim(self):
        s = _s(cell=np.eye(3) * 5,
               axis_kind=("periodic", "periodic", "transport"))
        assert s.axis_kind == ("periodic", "periodic", "transport")

    def test_the_boolean_view_is_no_longer_a_field(self):
        """Constructing with it is an error, not a silent second opinion."""
        import dataclasses as _dc
        from molbuilder.structure import Structure as _S
        assert "pbc" not in {f.name for f in _dc.fields(_S)}
        with pytest.raises(TypeError):
            _s(cell=np.eye(3) * 5, pbc=(False, False, False))

    def test_invalid_axis_kind_raises(self):
        with pytest.raises(ValueError, match="axis_kind"):
            _s(axis_kind=("periodic", "bogus", "isolated"))

    def test_vacuum_default(self):
        """UNSET, not zero (2026-08-03).  `None` is the state that says "nobody
        chose a vacuum", which is what earns an isolated axis the 3 A default
        gap; a stored (0,0,0) would mean "I deliberately want none" and would
        give a flat molecule a box with no volume."""
        s = _s()
        assert s.vacuum is None
        assert s.effective_vacuum() == (3.0, 3.0, 3.0)
        assert s.defaulted_vacuum_axes() == [0, 1, 2]


class TestResolveCell:
    def test_explicit_cell_wins(self):
        c = np.diag([3.0, 4.0, 5.0])
        assert np.allclose(_s(cell=c).resolve_cell(), c)

    def test_molecule_bbox_plus_vacuum(self):
        # vacuum is a PER-SIDE gap: cell = bbox + 2*vacuum.
        # x extent 2 + 2*5 = 12; y,z extent 0 + 2*5 = 10.
        s = _s(vacuum=(5, 5, 5))
        assert np.allclose(s.resolve_cell(), np.diag([12.0, 10.0, 10.0]))

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
        assert cell[0, 0] == pytest.approx(0.0 + 2 * 5.0)  # isolated x gets 2*vacuum

    def test_periodic_axis_without_cell_raises(self):
        s = _s(axis_kind=("periodic", "isolated", "isolated"))
        with pytest.raises(ValueError, match="periodic"):
            s.resolve_cell()

    def test_empty_structure_returns_none(self):
        assert Structure(elements=[], positions=np.zeros((0, 3))).resolve_cell() is None


class TestSidecarRoundTrip:
    """axis_kind / vacuum persist through the .molstruct.json sidecar
    (structure-periodicity.md § 7).  k-grid is NOT a geometry field, so it's
    neither written nor read."""

    def test_round_trip_through_to_dict_and_apply(self):
        from molbuilder.sidecars import molstruct as ms
        d = ms.to_dict(
            {"cell": [[5, 0, 0], [0, 5, 0], [0, 0, 10]],
             "axis_kind": ["periodic", "periodic", "transport"],
             "vacuum": [0.0, 0.0, 0.0]},
            n_atoms_total=2, structure_hash="a" * 64,
        )
        assert d["axis_kind"] == ["periodic", "periodic", "transport"]
        # WHAT WENT IN COMES BACK.  [0,0,0] is a deliberate zero and survives
        # as one; `null` is "nobody chose" and survives as that.
        assert d["vacuum"] == [0.0, 0.0, 0.0]
        assert "kgrid" not in d   # k-grid is not geometry -> not in the sidecar

        s = Structure(elements=["C", "H"], positions=[[0, 0, 0], [1, 0, 0]])
        ms.apply_to_structure(s, d)
        assert s.axis_kind == ("periodic", "periodic", "transport")
        assert not hasattr(s, "kgrid")
        assert s.pbc() == (True, True, True)  # the accessor: transport -> True


    def test_invalid_axis_kind_rejected_at_build(self):
        from molbuilder.sidecars import molstruct as ms
        with pytest.raises(ms.MolstructJsonError, match="axis_kind"):
            ms.to_dict({"axis_kind": ["periodic", "nope", "isolated"]},
                       n_atoms_total=1, structure_hash="a" * 64)


class TestElectrodeCaptureCell:
    """The slab builder captures its ASE cell + sets axis_kind
    (structure-periodicity.md § 4)."""

    def test_a_built_slab_captures_cell_and_axis_kind(self):
        from molbuilder.modify import add_slab
        # Built onto a structure with a typed box and an ASSIGNED origin, so
        # that "the new box drops it" is something this test can see.
        dev = Structure(elements=["S"], positions=[[0.0, 0.0, 0.0]],
                        cell=np.eye(3) * 20.0, axis_kind=("isolated",) * 3,
                        engine_offset=np.array([1.0, 2.0, 3.0]))
        out = add_slab(dev, "Au", "111", (2, 2, 3), start_z=2.4)
        assert out.cell is not None, "electrode cell must be captured, not discarded"
        assert out.axis_kind == ("periodic", "periodic", "transport")
        assert out.pbc() == (True, True, True)        # accessor: transport -> True
        assert out.cell[2, 2] > 0.0
        # in-plane vectors are non-degenerate (hexagonal for fcc111)
        assert abs(float(np.linalg.det(out.cell))) > 1e-6
        # § 6.0: the builder states no origin, and its new box drops the one
        # stated against the old -- the rule places the atoms, and the
        # hand-off accepts them.
        assert out.engine_offset is None, "the new box kept an origin set for the old one"
        cellmod.require_placed(cellmod.to_engine(out), out.axis_kind)

    @staticmethod
    def _seam(out):
        """Shortest metal-metal distance ACROSS the periodic z boundary."""
        import itertools
        C = np.asarray(out.resolve_cell(), dtype=float)
        P = np.asarray(out.positions, dtype=float)
        au = P[[e == "Au" for e in out.elements]]
        top = au[au[:, 2] > au[:, 2].max() - 1e-3]
        bot = au[au[:, 2] < au[:, 2].min() + 1e-3]
        best = float("inf")
        for i, j in itertools.product((-1, 0, 1), repeat=2):
            img = bot.copy()
            img[:, :2] += i * C[0][:2] + j * C[1][:2]
            img[:, 2] += C[2, 2]
            best = min(best, float(np.min(
                np.linalg.norm(top[:, None, :] - img[None, :, :], axis=-1))))
        return best

    def test_c_is_the_atoms_extent_and_the_collision_is_visible(self):
        """**`c` IS MEASURED AND SET, NEVER INVENTED** (junction-cell.md § 6,
        rewritten 2026-08-31 on the user's decision).

        A freshly built slab gets `c = z_extent` verbatim, so the top layer
        sits on its own periodic image and the seam is zero. That is the
        point, not a bug: the missing step is a decision about the
        calculation, it is visible in the tab you are already in, and
        `classify_seam` names it on every build.
        """
        a = 4.0782
        dev = Structure(elements=["S"], positions=[[0.0, 0.0, 0.0]])
        out = _two_slabs(dev, "Au", "111", (2, 2, 3), gap=8.0,
                         lattice_constant=a)
        extent = float(out.positions[:, 2].max() - out.positions[:, 2].min())
        assert out.cell[2, 2] == pytest.approx(extent, abs=1e-9)

        # AND WHAT THE FACES DO WHEN THEY MEET, which is a different rule
        # (junction-cell.md § 3.3).  `c == extent` puts the two faces at Δz = 0
        # across the boundary; MIRRORED slabs would then be eclipsed, atom on
        # atom, at distance 0.  These CONTINUE the crystal, so the faces sit
        # one registry step apart -- a/√6 for fcc(111) -- and the boundary is
        # a stacking fault rather than a collision.
        assert self._seam(out) == pytest.approx(a / np.sqrt(6), abs=1e-6)
