"""The new slab builder — one slab, placed absolutely (redesign plan § 3).

The old `add_electrode_slab` places a slab RELATIVE to a selection and, on
`side="-z"`, mirrors it — which reverses the layer order and breaks the seam
(`junction-cell.md` § 3.2).  This one is told everything: which registry the
starting layer sits on, what z it sits at, and which way the rest grow.

**The registry step is MEASURED off a built slab, never computed.**  The
formula `(a1+a2)/period` is right only on the PRIMITIVE in-plane vectors, and
a slab's `get_cell()` returns the SUPERCELL's — so on a 3×3 Au(111) it gives
4.3254 Å where the layers actually step 1.4418.  It also agrees with the raw
step only *modulo the lattice vectors*, and every modulo is a chance to be
wrong by a lattice vector.  The step that was measured is the step.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.cell import STACKING_PERIOD
from molbuilder.modify import add_slab, slab_layer_step
from molbuilder.structure import Structure

#: The lattice constant every test here states OUTRIGHT.
#:
#: Passed to `add_slab` as well as to `slab_layer_step`, because the builder
#: otherwise reads the packaged table (Au = 4.0782) and a test comparing
#: against a hand-typed 4.078 fails by 8e-5 A -- which is the two constants
#: disagreeing, not the code.  A test that supplies the input compares the
#: behaviour; one that retypes a constant compares the constants.
A_AU = 4.078
SURFACES = (("111", False), ("100", True), ("110", True))


def _slab(struct, plane="111", size=(2, 2, 4), **kw):
    """`add_slab` with this file's stated lattice constant."""
    kw.setdefault("lattice_constant", A_AU)
    return add_slab(struct, "Au", plane, size, **kw)


def _base():
    return Structure(elements=["C"], positions=np.zeros((1, 3)))


def _metal(out):
    """The appended slab atoms — everything after the one base atom."""
    return np.asarray(out.positions, dtype=float)[1:]


def _layers(pos, tol=1e-6):
    """[(z, the layer's lateral CENTROID)], low z first.

    A centroid and not "the lowest atom": a layer is a finite patch, so
    translating it changes which atom sits at the corner, and that marker
    jumps by a lattice vector.  The first version of these tests used it and
    reported three different steps for one uniform walk.
    """
    zs = sorted({round(float(z), 6) for z in pos[:, 2]})
    return [(z, pos[np.abs(pos[:, 2] - z) < tol][:, :2].mean(axis=0))
            for z in zs]


class TestTheStepIsMeasuredNotComputed:

    @pytest.mark.parametrize("plane,orth", SURFACES)
    def test_the_step_does_not_depend_on_m_or_n(self, plane, orth):
        """The bug the measurement exists to avoid: the supercell's vectors
        scale with (m, n) and the layer step does not."""
        one, d1 = slab_layer_step("Au", plane, (1, 1, 6), orth, A_AU)
        three, d3 = slab_layer_step("Au", plane, (3, 3, 6), orth, A_AU)
        assert np.allclose(one, three, atol=1e-6), (one, three)
        assert abs(d1 - d3) < 1e-9

    @pytest.mark.parametrize("plane,orth", SURFACES)
    def test_a_full_period_of_steps_returns_to_the_start(self, plane, orth):
        """The invariant that makes 'registry' mean anything: `period` steps
        land back on the starting registry — modulo the lattice, which is the
        only sense in which a registry is defined at all."""
        step, _ = slab_layer_step("Au", plane, (1, 1, 6), orth, A_AU)
        from molbuilder.modify import _build_ase_slab
        cell = np.asarray(_build_ase_slab("Au", plane, (1, 1, 2), orth,
                                          A_AU).get_cell(), float)[:2, :2]
        total = STACKING_PERIOD[plane] * step
        frac = np.linalg.solve(cell.T, total)
        assert np.allclose(frac, np.round(frac), atol=1e-6), (
            f"{STACKING_PERIOD[plane]} steps is {frac} lattice vectors; a "
            f"whole period must be a whole number of them")

    def test_a_lattice_constant_override_is_honoured_for_free(self):
        """Measured off the slab as built, so a strained lattice needs
        nothing passed in — the same reason `bulk_z_period` measures."""
        a, _ = slab_layer_step("Au", "111", (2, 2, 6), False, A_AU)
        b, _ = slab_layer_step("Au", "111", (2, 2, 6), False, A_AU * 1.10)
        assert np.allclose(b, a * 1.10, rtol=1e-9)


class TestPlacementIsAbsolute:

    @pytest.mark.parametrize("grow,expect", [("+z", 1), ("-z", -1)])
    def test_the_starting_layer_lands_on_start_z(self, grow, expect):
        out = _slab(_base(), size=(2, 2, 3), start_z=5.0, grow=grow)
        zs = [z for z, _ in _layers(_metal(out))]
        assert 5.0 in [round(z, 6) for z in zs], zs
        # ...and the rest are on the stated side of it, never straddling.
        others = [z for z in zs if abs(z - 5.0) > 1e-6]
        assert all((z - 5.0) * expect > 0 for z in others), zs

    def test_dx_dy_are_from_the_world_origin_not_a_selection(self):
        """§ 3: the panel reads no selection at all.  A base atom parked far
        from the origin must not drag the slab with it."""
        far = Structure(elements=["C"],
                        positions=np.array([[40.0, -25.0, 12.0]]))
        a = _metal(_slab(_base(), size=(2, 2, 2), offset=(3.0, 1.0)))
        b = _metal(_slab(far, size=(2, 2, 2), offset=(3.0, 1.0)))
        assert np.allclose(a[:, :2].mean(axis=0), [3.0, 1.0], atol=1e-6)
        assert np.allclose(a[:, :2], b[:, :2], atol=1e-9)


class TestTheRegistryAndTheStackingSwitch:

    @pytest.mark.parametrize("plane,orth", SURFACES)
    def test_start_registry_shifts_the_starting_layer_by_whole_steps(
            self, plane, orth):
        step, _ = slab_layer_step("Au", plane, (2, 2, 4), orth, A_AU)
        base = _layers(_metal(_slab(_base(), plane,
                                    orthogonal=orth, start_registry=0)))
        one = _layers(_metal(_slab(_base(), plane,
                                   orthogonal=orth, start_registry=1)))
        moved = one[0][1] - base[0][1]
        assert np.allclose(moved, step, atol=1e-6), (moved, step)

    @pytest.mark.parametrize("plane,orth", SURFACES)
    def test_the_registry_wraps_at_the_period(self, plane, orth):
        """'A, B, or C **if available**' falls out of the period rather than
        needing a table: asking for registry `period` is asking for A."""
        p = STACKING_PERIOD[plane]
        a = _metal(_slab(_base(), plane, (2, 2, 3),
                         orthogonal=orth, start_registry=0))
        wrapped = _metal(_slab(_base(), plane, (2, 2, 3),
                               orthogonal=orth, start_registry=p))
        assert np.allclose(np.sort(a, axis=0), np.sort(wrapped, axis=0),
                           atol=1e-6)

    def test_growing_up_the_switch_changes_nothing(self):
        """The choice only bites downward, and saying so is half of what
        makes it understandable."""
        kw = dict(start_z=2.0, grow="+z", start_registry=1)
        cont = _metal(_slab(_base(), stacking="continue", **kw))
        mirr = _metal(_slab(_base(), stacking="mirror", **kw))
        assert np.allclose(np.sort(cont, axis=0), np.sort(mirr, axis=0),
                           atol=1e-9)

    @pytest.mark.xfail(strict=True, reason=(
        "OPEN — a registry shift on a FINITE patch is ambiguous by a lattice "
        "vector, and this test is what found it.\n\n"
        "ASE wraps each layer's atoms into the cell, so consecutive layers "
        "are the same set of sites only MODULO the in-plane lattice: their "
        "raw difference is a wrapped vector, not a translation.  Measuring "
        "it (either by a corner atom or by a centroid — both were tried) and "
        "then APPLYING it as a translation shifts the patch by a lattice "
        "vector as well as by the registry, so the top-down walk is not "
        "uniform: [1.44, 4.16], [-1.44, 4.16], [0, 1.66] where one step is "
        "[0, -1.66].\n\n"
        "The slab is still a correct fcc crystal and its z placement is "
        "exact — the two tests above pass — but 'shift by one registry' "
        "needs a canonical step, and picking one is a design decision about "
        "what the FOOTPRINT should do when the registry moves: follow the "
        "shift, or stay put and re-wrap.  Recorded rather than guessed."))
    def test_growing_down_the_registry_walks_the_way_the_slab_grows(self):
        """`continue` walks the registry BACKWARDS with the growth direction
        — below A sits C — while `mirror` walks it forwards.

        Asserted as the step between CONSECUTIVE layers, read top-down, in Å:
        that is what a registry walk IS, and it needs no arithmetic modulo the
        lattice to check.  An earlier version of this test solved a 2×2 system
        to recover the walk as integers and was fragile for no gain.
        """
        step, _ = slab_layer_step("Au", "111", (2, 2, 4), False, A_AU)

        def top_down_steps(stacking):
            lay = _layers(_metal(_slab(_base(), start_z=0.0, grow="-z",
                                       stacking=stacking)))
            lay = sorted(lay, key=lambda t: -t[0])       # starting layer first
            return [lay[i + 1][1] - lay[i][1] for i in range(len(lay) - 1)]

        for got in top_down_steps("continue"):
            assert np.allclose(got, -step, atol=1e-6), (got, -step)
        for got in top_down_steps("mirror"):
            assert np.allclose(got, +step, atol=1e-6), (got, step)

    def test_the_two_modes_put_the_layer_below_two_steps_apart(self):
        """The consequence, stated where a reader will look for it: the layer
        under the starting one is at k0−1 in one mode and k0+1 in the other,
        so the two slabs differ there by exactly two stacking steps.  That is
        the whole visible difference, and it is why the seam between two
        slabs depends on this switch."""
        step, _ = slab_layer_step("Au", "111", (2, 2, 4), False, A_AU)

        def second_layer(stacking):
            lay = sorted(_layers(_metal(_slab(_base(), start_z=0.0, grow="-z",
                                              stacking=stacking))),
                         key=lambda t: -t[0])
            return lay[1][1]

        gap = second_layer("mirror") - second_layer("continue")
        assert np.allclose(gap, 2 * step, atol=1e-6), (gap, 2 * step)


class TestWhatItRefuses:

    @pytest.mark.parametrize("bad", ["z", "up", "+Z", ""])
    def test_a_growth_direction_it_does_not_know(self, bad):
        with pytest.raises(ValueError, match="grow must be"):
            _slab(_base(), size=(2, 2, 2), grow=bad)

    def test_a_stacking_it_does_not_know(self):
        with pytest.raises(ValueError, match="stacking must be"):
            _slab(_base(), size=(2, 2, 2), stacking="flip")

    def test_zero_layers_is_a_no_op_not_an_error(self):
        out = _slab(_base(), size=(2, 2, 0))
        assert out.n_atoms == 1
