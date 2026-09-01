"""Structure periodicity fields — axis_kind / vacuum + resolve_cell
(structure-periodicity.md, Phase 2a data model).  k-grid is deliberately NOT
here: it's a reciprocal-space sampling knob on SiestaConfig / TransportConfig.
"""
import numpy as np
import pytest

from molbuilder.structure import Structure


def _two_slabs(struct, element, plane, size, *, gap=8.0, **kw):
    """A junction: two slabs, one per side, each starting at ``gap/2``.

    Repointed twice, and each time the builder under it was deleted rather
    than the rule.  `add_symmetric_electrodes` did this in one call and went
    with the Junction panel (2026-08-31); `add_electrode_slab` did it in two
    and went on 2026-09-01, its anchor-relative placement folded into the
    callers that wanted it.  What these tests are about -- the CELL a build
    captures -- is `_finish_slab`'s, and that has not moved.

    `stacking="continue"` on the `-z` side is the redesign's answer to the
    mirror the old builder applied unconditionally.
    """
    from molbuilder.modify import add_slab
    out = add_slab(struct, element, plane, size,
                   start_z=gap / 2.0, grow="+z", **kw)
    return add_slab(out, element, plane, size,
                    start_z=-gap / 2.0, grow="-z", stacking="continue", **kw)




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

    def test_vacuum_default(self):
        """UNSET, not zero (2026-08-03).  `None` is the state that says "nobody
        chose a vacuum", which is what earns an isolated axis the 3 A default
        gap; a stored (0,0,0) would mean "I deliberately want none" and would
        give a flat molecule a box with no volume."""
        s = _s()
        assert s.vacuum is None
        assert s.effective_vacuum() == (3.0, 3.0, 3.0)
        assert s.defaulted_vacuum_axes() == [0, 1, 2]

    def test_kgrid_is_not_a_structure_field(self):
        # k-grid is a reciprocal-space SAMPLING knob (SiestaConfig /
        # TransportConfig), not geometry -- structure-periodicity.md.
        assert not hasattr(_s(), "kgrid")
        with pytest.raises(TypeError):
            _s(kgrid=(4, 4, 1))


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
    (structure-periodicity.md § 7; additive @ schema v4).  k-grid is NOT a
    geometry field, so it's neither written nor read (dropped @ schema v5)."""

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
        # as one; `null` is "nobody chose" and survives as that.  A brief
        # legacy rule folded the first into the second -- removed 2026-08-03.
        assert d["vacuum"] == [0.0, 0.0, 0.0]
        assert "kgrid" not in d   # k-grid is not geometry -> not in the sidecar

        s = Structure(elements=["C", "H"], positions=[[0, 0, 0], [1, 0, 0]])
        ms.apply_to_structure(s, d)
        assert s.axis_kind == ("periodic", "periodic", "transport")
        assert not hasattr(s, "kgrid")
        assert s.pbc == (True, True, True)   # derived: transport -> True



    def test_invalid_axis_kind_rejected_at_build(self):
        from molbuilder.sidecars import molstruct as ms
        with pytest.raises(ms.MolstructJsonError, match="axis_kind"):
            ms.to_dict({"axis_kind": ["periodic", "nope", "isolated"]},
                       n_atoms_total=1, structure_hash="a" * 64)


class TestElectrodeCaptureCell:
    """The slab builder captures its ASE cell + sets axis_kind
    (structure-periodicity.md § 4 -- fixes the modify.py:955 discard)."""

    def test_a_built_slab_captures_cell_and_axis_kind(self):
        from molbuilder.modify import add_slab
        dev = Structure(elements=["S"], positions=[[0.0, 0.0, 0.0]])
        out = add_slab(dev, "Au", "111", (2, 2, 3), start_z=2.4)
        assert out.cell is not None, "electrode cell must be captured, not discarded"
        assert out.axis_kind == ("periodic", "periodic", "transport")
        assert out.pbc == (True, True, True)          # transport -> True
        assert out.cell[2, 2] > 0.0
        # in-plane vectors are non-degenerate (hexagonal for fcc111)
        assert abs(float(np.linalg.det(out.cell))) > 1e-6
        # § 3c: the captured cell WRAPS the atoms -- cell_origin = structure low corner.
        assert out.cell_origin is not None
        assert np.allclose(out.cell_origin, out.positions.min(axis=0))

    # ---- the z length: extent + ONE interlayer spacing ---------------- #
    #  science/junction-cell.md § 1.  This used to be the bare extent, and
    #  `assert cell[2,2] > 0` above was the only thing watching it -- which
    #  a box that collides with its own image passes.  The seam distance is
    #  the assertion that cannot be satisfied by the bug.

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

        *Three tests here pinned the opposite until 2026-09-01 -- extent plus
        one layer spacing, an `inter_layer_offset` override, and a monolayer
        falling back to the crystal spacing. They went with the padding they
        described, which had survived in `add_electrode_slab` alone for a
        month after the rule that retired it.*
        """
        a = 4.0782
        dev = Structure(elements=["S"], positions=[[0.0, 0.0, 0.0]])
        out = _two_slabs(dev, "Au", "111", (2, 2, 3), gap=8.0,
                         lattice_constant=a)
        extent = float(out.positions[:, 2].max() - out.positions[:, 2].min())
        assert out.cell[2, 2] == pytest.approx(extent, abs=1e-9)

        # AND WHAT THE FACES DO WHEN THEY MEET, which is a different rule
        # (junction-cell.md § 3.3) and worth pinning here because it is what
        # the redesign changed.  `c == extent` puts the two faces at Δz = 0
        # across the boundary; MIRRORED slabs would then be eclipsed, atom on
        # atom, at distance 0.  These CONTINUE the crystal, so the faces sit
        # one registry step apart -- a/√6 for fcc(111) -- and the boundary is
        # a stacking fault rather than a collision.
        assert self._seam(out) == pytest.approx(a / np.sqrt(6), abs=1e-6)


def _parse_fdf_cell_coords(fdf):
    """Pull the LatticeVectors (3x3, rows) + the atom Cartesian coords (Ang) out of a
    rendered FDF, so a test can check where every atom sits relative to the cell."""
    import re
    lv = re.search(r"%block\s+LatticeVectors\s*\n(.*?)%endblock\s+LatticeVectors",
                   fdf, re.IGNORECASE | re.DOTALL)
    cell = np.array([[float(x) for x in ln.split()[:3]]
                     for ln in lv.group(1).strip().splitlines() if ln.split()])
    ac = re.search(r"%block\s+AtomicCoordinatesAndAtomicSpecies\s*\n(.*?)%endblock",
                   fdf, re.IGNORECASE | re.DOTALL)
    coords = []
    for ln in ac.group(1).strip().splitlines():
        p = ln.split()
        if len(p) >= 3 and not ln.lstrip().startswith("#"):
            coords.append([float(p[0]), float(p[1]), float(p[2])])
    return cell, np.array(coords)


class TestCellOriginAndCalibration:
    """§ 3c: an explicit cell wraps off-origin atoms via cell_origin; the viewer box
    and render_fdf stay consistent; calibrate bakes the SIESTA frame."""

    def _junction(self):
        """Molecule pinned at the origin (2 anchors on z) + symmetric Au(111)
        electrodes -- atoms straddle the origin, cell captured with cell_origin."""
        mol = Structure(elements=["S", "S"],
                        positions=[[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]])
        # `center_indices=[0, 1]` used to say "midpoint of the two S atoms",
        # which IS the origin here -- so the absolute placement below is the
        # same junction, said outright instead of derived.
        return _two_slabs(mol, "Au", "111", (2, 2, 3), gap=6.0)

    def test_cell_origin_wraps_the_atoms(self):
        j = self._junction()
        assert j.positions[:, 2].min() < 0.0, "junction should straddle the origin"
        o = j.resolve_cell_origin()
        assert o is not None and np.allclose(o, j.positions.min(axis=0))
        # The box [o, o+cell] contains every atom on the transport (z) axis exactly.
        cz = j.resolve_cell()[2, 2]
        assert j.positions[:, 2].max() <= o[2] + cz + 1e-6

    def test_render_fdf_puts_device_inside_the_transport_cell(self):
        """The viewer ≡ SIESTA invariant: render_fdf emits the atoms translated into
        [0, cell) -- every device atom sits inside the transport (z) cell [0, Lz]."""
        from molbuilder.siesta import render_fdf
        from molbuilder.config.siesta import SiestaConfig
        j = self._junction()
        fdf = render_fdf(j, SiestaConfig(system_label="jx"))
        cell, coords = _parse_fdf_cell_coords(fdf)
        # The emitted lattice IS the resolved (explicit) cell -- viewer & FDF agree.
        assert np.allclose(cell, j.resolve_cell(), atol=1e-4)
        # Transport z is orthogonal ([0,0,Lz]); every atom's z is inside [0, Lz].
        Lz = cell[2, 2]
        assert coords[:, 2].min() >= -1e-4, "device atom below the transport cell"
        assert coords[:, 2].max() <= Lz + 1e-4, "device atom above the transport cell"

    def test_calibrate_bakes_the_shift_idempotent(self):
        from molbuilder.modify import calibrate_to_cell
        j = self._junction()
        c = calibrate_to_cell(j)
        assert c.cell_origin is None                      # now anchored at (0,0,0)
        assert c.resolve_cell_origin() is None
        assert np.allclose(c.positions[:, 2].min(), 0.0, atol=1e-6)  # atoms in [0, Lz]
        assert np.allclose(c.resolve_cell(), j.resolve_cell(), atol=1e-6)  # same cell
        # Idempotent: a second calibrate is a no-op.
        c2 = calibrate_to_cell(c)
        assert np.allclose(c.positions, c2.positions)

    def test_imported_crystal_no_shift(self):
        """An explicit cell with NO cell_origin (imported crystal, atoms already in
        [0,cell)) -> resolve_cell_origin None -> render_fdf does not shift."""
        s = Structure(elements=["H"], positions=[[0.5, 0.5, 0.5]],
                      cell=(np.eye(3) * 3).tolist())
        assert s.cell_origin is None
        assert s.resolve_cell_origin() is None
