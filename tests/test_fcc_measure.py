"""Reading a lattice constant back out of a relaxed bulk result.

`archive/2026-09-01-modify-redesign-plan.md` § 3.3, user's own design: *"we should have a
backend to extract this from a .xyz or .XV result where one single periodic
lattice is correctly optimized with the same pseudopotential/basis etc., but
that is at user's hand … and the backend just extracts the lattice from that
result."*

**It measures the ATOMS, not the cell, and that is the load-bearing choice.**
A relaxed result's cell may be conventional cubic (edge `a`), primitive
rhombohedral (`a/√2`), or the user's own m×n×N lead cell (`m·a/√2`) — three
relations to `a`, and the file does not say which.  So the tests below hand the
same crystal to the measurement in all three shapes and demand **one answer**:
that is the whole claim, and it is why reading `cell[0][0]` would not do.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.cell import measure_fcc

A = 4.078                     # Å — gold, near enough
D_NN = A / np.sqrt(2.0)       # 2.8836 Å


def _conventional(a=A):
    """The 4-atom cubic cell: corner + three face centres."""
    box = np.eye(3) * a
    pos = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]) * a
    return pos, box


def _supercell(nx, ny, nz, a=A):
    pos, _ = _conventional(a)
    blocks = [pos + np.array([i, j, k]) * a
              for i in range(nx) for j in range(ny) for k in range(nz)]
    return np.vstack(blocks), np.diag([nx * a, ny * a, nz * a])


def _primitive(a=A):
    """One atom, rhombohedral cell — every neighbour is its own image."""
    return np.zeros((1, 3)), np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a


class TestOneAnswerFromThreeConventions:

    @pytest.mark.parametrize("name,make", [
        ("conventional cubic", lambda: _conventional()),
        ("primitive rhombohedral", lambda: _primitive()),
        ("3x3x2 supercell", lambda: _supercell(3, 3, 2)),
        ("1x1x6 layered lead cell", lambda: _supercell(1, 1, 6)),
    ])
    def test_the_same_crystal_measures_the_same_whatever_the_cell(self, name, make):
        """The point of the whole design: the box shape does not enter the
        answer.  Each of these has a different `cell[0][0]` — a, a/√2·√2,
        3a, a — and reading it would have given three different lattice
        constants for one crystal."""
        pos, box = make()
        m = measure_fcc(pos, box)
        assert abs(m.a - A) < 1e-9, f"{name}: a={m.a}"
        assert abs(m.d_nn - D_NN) < 1e-9, f"{name}: d_nn={m.d_nn}"

    def test_one_atom_is_enough(self):
        """A primitive cell holds exactly one, and its twelve neighbours are
        its own periodic images.

        This is the case the first version got wrong twice over: it collapsed
        each atom's images to their nearest before dropping the self-distance,
        which threw away every own-image; and it then refused fewer than two
        atoms, which hid the first mistake behind a plausible message.
        """
        m = measure_fcc(*_primitive())
        assert m.n_atoms == 1
        assert abs(m.a - A) < 1e-9
        assert m.coordination == 12


class TestTheTwoChecksItReports:

    def test_bulk_fcc_has_twelve_neighbours_and_a_root_two_second_shell(self):
        """The two signatures § 3.3 asks for, and both are exact in a perfect
        crystal: twelve at `d`, then six at `√2·d`."""
        m = measure_fcc(*_supercell(3, 3, 2))
        assert m.coordination == 12
        assert m.second_shell is not None
        assert abs(m.second_shell / m.d_nn - np.sqrt(2.0)) < 1e-9

    def test_a_molecule_is_not_a_crystal_and_the_count_says_so(self):
        """The check earns its keep here: a file that is not the bulk crystal
        the user meant to point at.  Two atoms in a big box have ONE neighbour
        each, and `a` computed from them is meaningless — which is exactly why
        the number is reported for the user to look at rather than swallowed."""
        pos = np.array([[0.0, 0, 0], [D_NN, 0, 0]])
        m = measure_fcc(pos, np.eye(3) * 40.0)
        assert m.coordination == 1, m.coordination
        # The arithmetic still runs — it is a NOTE, not a refusal (§ 3.3).
        assert abs(m.a - A) < 1e-9

    def test_the_median_ignores_a_surface(self):
        """A thick slab's interior is still bulk, so the median atom still has
        twelve — and its `a` is still right.  A mean would have been dragged
        down by the two open faces and reported a defect in a good file."""
        pos, box = _supercell(2, 2, 3)
        big = np.diag([box[0, 0], box[1, 1], box[2, 2] + 20.0])   # vacuum on z
        m = measure_fcc(pos, big)
        assert m.coordination == 12
        assert abs(m.a - A) < 1e-9


class TestWhatItRefuses:

    def test_a_cell_with_no_volume(self):
        """No box means no periodic images, and on a small cell the measured
        minimum is then simply wrong rather than absent — so this refuses
        instead of answering (§ 3.3)."""
        pos, _ = _conventional()
        with pytest.raises(ValueError, match="no volume"):
            measure_fcc(pos, np.zeros((3, 3)))

    def test_no_atoms(self):
        with pytest.raises(ValueError, match="no atoms"):
            measure_fcc(np.zeros((0, 3)), np.eye(3) * A)


class TestTheMistakeAnyoneActuallyMakes:

    def test_a_second_shell_pair_is_a_factor_root_two_out(self):
        """§ 3.3: the one mistake anyone makes is picking a SECOND-shell pair,
        which in fcc sits at exactly `a` — a factor 1.414 out, landing 41%
        from both literature references, where the derived line says so
        immediately.

        Asserted as arithmetic rather than trusted as prose: a crystal built
        with `d_nn` set to what is really the second-shell distance reports an
        `a` that is √2 too large.
        """
        pos, box = _conventional(A * np.sqrt(2.0))
        m = measure_fcc(pos, box)
        assert abs(m.a - A * np.sqrt(2.0)) < 1e-9
        assert abs(m.a / A - np.sqrt(2.0)) < 1e-9, (
            "the 41% error the derived line is there to expose")
