"""The new slab builder — one slab, placed absolutely (redesign plan § 3).

The old `add_electrode_slab` places a slab RELATIVE to a selection and, on
`side="-z"`, mirrors it unconditionally — which reverses the layer order and
breaks the seam (`junction-cell.md` § 3.2).  This one is told everything:
which registry the layer at `start_z` sits on, what z that is, which way the
rest grow, and whether growing down continues the crystal or mirrors it.

**Build tall, trim, then move as one piece** (user, 2026-08-30).  Superset
layer `j` already sits on registry `j mod period` because ASE put it there,
so the registry is *which slice you take* and nothing moves sideways at all.
The result is a contiguous slice of a real crystal by construction, and every
later operation is rigid — a rigid motion of a crystal is a crystal.

**Why not a lateral shift**, which two earlier attempts used: there is no such
thing as "the" step from one layer to the next.  Measured on ASE's own
untouched Au(111) slab, consecutive layer centroids walk by three DIFFERENT
vectors repeating with period 3, because each layer is wrapped into the cell.
That measurement is `test_ase_itself_has_no_single_layer_step` below — it is
kept because it is the fact that refutes the whole shifting approach, and a
future reader will otherwise re-derive it the slow way.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.cell import STACKING_PERIOD
from molbuilder.modify import _build_ase_slab, add_slab
from molbuilder.structure import Structure

#: Stated outright and passed to the builder, never only to the assertions:
#: the packaged table says Au = 4.0782, and a test that retypes 4.078 and
#: compares against it is measuring the two constants, not the behaviour.
A_AU = 4.078
SURFACES = (("111", False), ("100", True), ("110", True))


def _base():
    return Structure(elements=["C"], positions=np.zeros((1, 3)))


def _slab(struct=None, plane="111", size=(2, 2, 3), **kw):
    kw.setdefault("lattice_constant", A_AU)
    return add_slab(struct if struct is not None else _base(),
                    "Au", plane, size, **kw)


def _metal(out):
    """The appended slab atoms — everything after the one base atom."""
    return np.asarray(out.positions, dtype=float)[1:]


def _by_layer(pos, tol=1e-6):
    """[(z, the layer's atoms as a sorted xy array)], low z first."""
    out = []
    for z in sorted({round(float(v), 6) for v in pos[:, 2]}):
        lay = pos[np.abs(pos[:, 2] - z) < tol][:, :2]
        out.append((z, lay[np.lexsort((lay[:, 1], lay[:, 0]))]))
    return out


def _shapes(pos):
    """Each layer's atoms relative to THAT LAYER's own centroid — its shape.

    Use for "is this a layer ASE built".  Deliberately blind to where the
    layer sits, so it cannot be used to ask about registry.
    """
    return [np.round(xy - xy.mean(axis=0), 6) for _, xy in _by_layer(pos)]


def _offsets(pos):
    """Each layer's centroid relative to THE WHOLE SLAB's centroid.

    Use for registry and stacking.  A registry choice is a lateral
    displacement, so dividing out each layer's own centroid — which `_shapes`
    does — removes precisely the thing being asked about: on fcc(100) every
    registry then looked identical and a passing test said the control did
    nothing.  Dividing out the SLAB's centroid instead removes the absolute
    placement and keeps the layer-to-layer relationships.
    """
    whole = pos[:, :2].mean(axis=0)
    return [np.round(xy.mean(axis=0) - whole, 6) for _, xy in _by_layer(pos)]


class TestTheFactThatDecidedTheDesign:

    @pytest.mark.parametrize("size", [(1, 1, 6), (2, 2, 6), (3, 3, 6)])
    def test_ase_itself_has_no_single_layer_step(self, size):
        """There is no "the" lateral step between layers, so no amount of
        care in measuring one would have made shifting by it correct.

        ASE's own untouched slab walks by three different centroid vectors
        repeating with period 3 — each layer is wrapped into the cell.  Kept
        as a test because it is the evidence that refutes the approach, and
        because if ASE ever stops wrapping, the builder's whole rationale
        should be re-read rather than silently still passing.
        """
        pos = np.asarray(_build_ase_slab("Au", "111", size, False,
                                         A_AU).positions, float)
        cents = [xy.mean(axis=0) for _, xy in _by_layer(pos)]
        walk = [np.round(cents[i + 1] - cents[i], 4) for i in range(len(cents) - 1)]
        assert not all(np.allclose(w, walk[0], atol=1e-6) for w in walk), (
            "ASE now gives a uniform layer step; the builder's reason for "
            "trimming instead of shifting has changed and should be re-read")
        assert np.allclose(walk[0], walk[3], atol=1e-6), (
            "the walk should still repeat with the stacking period")


class TestItIsASliceOfARealCrystal:

    @pytest.mark.parametrize("plane,orth", SURFACES)
    @pytest.mark.parametrize("k0", (0, 1, 2))
    def test_every_layer_matches_a_layer_of_the_ase_superset(
            self, plane, orth, k0):
        """The claim the whole design rests on.  Each layer's own lateral
        pattern must be one ASE built — not a shifted approximation of one."""
        n = 3
        period = STACKING_PERIOD[plane]
        tall = _build_ase_slab("Au", plane, (2, 2, n + period - 1), orth, A_AU)
        want = _shapes(np.asarray(tall.positions, float))
        got = _shapes(_metal(_slab(plane=plane, size=(2, 2, n),
                                   orthogonal=orth, start_registry=k0)))
        assert len(got) == n
        for layer in got:
            assert any(layer.shape == w.shape and np.allclose(layer, w, atol=1e-6)
                       for w in want), "a layer is not one ASE built"

    @pytest.mark.parametrize("plane,orth", SURFACES)
    def test_the_registry_control_actually_changes_the_slab(self, plane, orth):
        """A control that does nothing is worse than none.  An earlier
        version shifted laterally and then re-centred, which subtracted
        exactly the same vector — every registry gave an identical slab."""
        seen = [_offsets(_metal(_slab(plane=plane, orthogonal=orth,
                                      start_registry=k)))
                for k in range(STACKING_PERIOD[plane])]
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                assert not all(np.allclose(x, y, atol=1e-6)
                               for x, y in zip(seen[i], seen[j])), (
                    f"registry {i} and {j} produced the same slab")

    @pytest.mark.parametrize("plane,orth", SURFACES)
    def test_the_registry_wraps_at_the_period(self, plane, orth):
        """'A, B, or C **if available**' falls out of the period rather than
        needing a table: asking for registry `period` is asking for A."""
        p = STACKING_PERIOD[plane]
        a = _metal(_slab(plane=plane, orthogonal=orth, start_registry=0))
        b = _metal(_slab(plane=plane, orthogonal=orth, start_registry=p))
        assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0), atol=1e-6)

    @pytest.mark.parametrize("plane,orth", SURFACES)
    def test_the_layer_count_is_what_was_asked_for(self, plane, orth):
        """The superset is taller than the ask; the trim is what makes it
        the ask.  A superset that leaked through would be a silently thicker
        electrode."""
        for n in (1, 2, 5):
            out = _metal(_slab(plane=plane, size=(2, 2, n), orthogonal=orth))
            assert len(_by_layer(out)) == n


class TestPlacementIsAbsolute:

    @pytest.mark.parametrize("grow,side", [("+z", 1), ("-z", -1)])
    def test_the_starting_layer_lands_on_start_z(self, grow, side):
        zs = [z for z, _ in _by_layer(_metal(_slab(start_z=5.0, grow=grow)))]
        assert 5.0 in [round(z, 6) for z in zs], zs
        others = [z for z in zs if abs(z - 5.0) > 1e-6]
        assert all((z - 5.0) * side > 0 for z in others), zs

    def test_dx_dy_are_from_the_world_origin_not_a_selection(self):
        """§ 3: the panel reads no selection at all.  A base atom parked far
        from the origin must not drag the slab with it."""
        far = Structure(elements=["C"], positions=np.array([[40.0, -25.0, 12.0]]))
        a = _metal(_slab(size=(2, 2, 2), offset=(3.0, 1.0)))
        b = _metal(_slab(far, size=(2, 2, 2), offset=(3.0, 1.0)))
        assert np.allclose(a[:, :2].mean(axis=0), [3.0, 1.0], atol=1e-6)
        assert np.allclose(a[:, :2], b[:, :2], atol=1e-9)


class TestTheStackingSwitch:

    def test_growing_up_the_switch_changes_nothing(self):
        """The choice only bites downward, and saying so is half of what
        makes it understandable."""
        kw = dict(start_z=2.0, grow="+z", start_registry=1)
        cont = _metal(_slab(stacking="continue", **kw))
        mirr = _metal(_slab(stacking="mirror", **kw))
        assert np.allclose(np.sort(cont, axis=0), np.sort(mirr, axis=0),
                           atol=1e-9)

    def test_growing_down_they_differ(self):
        kw = dict(start_z=0.0, grow="-z", start_registry=0)
        cont = _offsets(_metal(_slab(stacking="continue", **kw)))
        mirr = _offsets(_metal(_slab(stacking="mirror", **kw)))
        assert not all(np.allclose(c, m, atol=1e-6)
                       for c, m in zip(cont, mirr)), (
            "the stacking switch changed nothing growing down, which is the "
            "only direction it is for")

    def test_continue_is_a_translation_and_mirror_is_a_reflection(self):
        """The two readings are two RIGID motions, and this is what tells
        them apart: read from the starting surface outward, `mirror` gives
        the same layer sequence an upward slab does, and `continue` gives
        the reverse — because it never reflected anything."""
        up = _offsets(_metal(_slab(start_z=0.0, grow="+z")))
        mirr = _offsets(_metal(_slab(start_z=0.0, grow="-z",
                                     stacking="mirror")))
        cont = _offsets(_metal(_slab(start_z=0.0, grow="-z",
                                     stacking="continue")))
        # `_offsets` is low-z-first, so reverse for "outward from start_z".
        assert all(np.allclose(a, b, atol=1e-6)
                   for a, b in zip(up, mirr[::-1])), "mirror is not a reflection"
        assert not all(np.allclose(a, b, atol=1e-6)
                       for a, b in zip(up, cont[::-1])), (
            "continue behaved like a reflection")


class TestTheBoxItCaptures:
    """`junction-cell.md` § 1: an unpadded box puts the bottom atom's periodic
    image exactly on the top atom, at zero distance, and SIESTA stops.  The
    padding is one interlayer spacing.

    Pinned at ONE layer as well as several, because that is the case with no
    spacing of its own to measure — and the case whose handling moved during
    the API review, from a probe inside the shared helper to the caller that
    owns the build recipe.
    """

    @pytest.mark.parametrize("n_layers", (1, 2, 3))
    def test_the_box_is_padded_by_exactly_one_interlayer_spacing(self, n_layers):
        d_au111 = 2.3544                       # Au(111), a = 4.078
        out = _slab(size=(2, 2, n_layers), start_z=2.4)
        pos = np.asarray(out.positions, dtype=float)
        span = float(pos[:, 2].max() - pos[:, 2].min())
        assert out.cell is not None, "a slab with z extent must capture a box"
        assert abs((out.cell[2][2] - span) - d_au111) < 1e-3, (
            f"{n_layers} layer(s): padded by {out.cell[2][2] - span}, "
            f"expected one spacing ({d_au111})")

    def test_a_degenerate_z_extent_captures_no_box(self):
        """A monolayer landing exactly on a flat molecule has no z extent, and
        a box with none is singular.  Answered with `cell=None` — no box —
        rather than with a box no engine can use."""
        out = _slab(size=(2, 2, 1), start_z=0.0)
        assert out.cell is None


class TestWhatItRefuses:

    @pytest.mark.parametrize("bad", ["z", "up", "+Z", ""])
    def test_a_growth_direction_it_does_not_know(self, bad):
        with pytest.raises(ValueError, match="grow must be"):
            _slab(size=(2, 2, 2), grow=bad)

    def test_a_stacking_it_does_not_know(self):
        with pytest.raises(ValueError, match="stacking must be"):
            _slab(size=(2, 2, 2), stacking="flip")

    def test_zero_layers_is_a_no_op_not_an_error(self):
        assert _slab(size=(2, 2, 0)).n_atoms == 1
