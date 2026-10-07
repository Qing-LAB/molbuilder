"""The rank rule: how many whole-body motions survive holding atoms still.

``science/normal-modes.md`` § 7 says the rules are checked through the
results.  Tier 1 here needs no quantum chemistry: positions in, a count
and a set of patterns out.  Every row of § 7's table is a system where
a table of cases gets the answer wrong somewhere -- the two collinear
traps (CO2 with both O held, acetylene with both C held) and the
water-dimer over-removal guard are the argument for computing a rank
instead of tabulating one.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.spectra.normal_modes import rigid_motions, vibrational_modes

ISO = ("isolated",) * 3
SLAB = ("periodic", "periodic", "isolated")
WIRE = ("isolated", "isolated", "periodic")
CELL = np.array([[8.0, 0.0, 0.0], [4.0, 6.93, 0.0], [0.0, 0.0, 12.0]])

WATER = np.array([[0.0, 0.0, 0.119],
                  [0.0, 0.757, -0.477],
                  [0.0, -0.757, -0.477]])
CO2 = np.array([[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
ARGON = np.array([[0.3, -0.2, 1.0]])
ACETYLENE = np.array([[-1.66, 0.0, 0.0], [-0.60, 0.0, 0.0],
                      [0.60, 0.0, 0.0], [1.66, 0.0, 0.0]])
NH3 = np.array([[0.0, 0.0, 0.11],
                [0.94, 0.0, -0.26],
                [-0.47, 0.81, -0.26],
                [-0.47, -0.81, -0.26]])
SLAB_ATOMS = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.7, 0.0],
                       [0.0, 0.0, 2.4], [2.0, 0.0, 2.4], [1.0, 1.7, 2.4]])
ZIGZAG = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 1.5], [0.0, 0.5, 3.0],
                   [0.0, -0.5, 4.5]])
STRAIGHT = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 3.0]])
DIMER = np.vstack([WATER, WATER + np.array([20.0, 0.0, 0.0])])


# system, held, axis_kind, cell, n_rigid -- science/normal-modes.md § 7's table
TABLE = [
    pytest.param(WATER, [], ISO, None, 6, id="water-free"),
    pytest.param(CO2, [], ISO, None, 5, id="co2-straight"),
    pytest.param(ARGON, [], ISO, None, 3, id="lone-atom"),
    pytest.param(WATER, [0], ISO, None, 3, id="water-O-held"),
    pytest.param(WATER, [0, 1], ISO, None, 1, id="water-O-and-H-held"),
    pytest.param(CO2, [0, 2], ISO, None, 0, id="co2-both-O-held-trap"),
    pytest.param(ACETYLENE, [1, 2], ISO, None, 0, id="acetylene-both-C-held-trap"),
    pytest.param(NH3, [1, 2, 3], ISO, None, 0, id="nh3-three-H-held"),
    pytest.param(SLAB_ATOMS, [], SLAB, CELL, 3, id="slab-free"),
    pytest.param(SLAB_ATOMS, [0], SLAB, CELL, 0, id="slab-one-held"),
    pytest.param(ZIGZAG, [], WIRE, CELL, 4, id="wire-keeps-its-own-turn"),
    pytest.param(STRAIGHT, [], WIRE, CELL, 3, id="straight-chain-on-its-axis"),
    pytest.param(DIMER, [0, 1, 2], ISO, None, 0, id="water-dimer-guard"),
]


@pytest.mark.parametrize("positions, held, kinds, cell, expected", TABLE)
def test_the_count_is_the_rank(positions, held, kinds, cell, expected):
    patterns = rigid_motions(positions, held, kinds, cell=cell)
    n_free = len(positions) - len(held)
    assert patterns.shape == (expected, n_free, 3)


@pytest.mark.parametrize("positions, held, kinds, cell, expected", TABLE)
def test_every_pattern_is_rigid_and_moves_no_held_atom(positions, held, kinds,
                                                       cell, expected):
    """The physical property, not the shape of the implementation: a
    returned motion changes no interatomic distance to first order --
    between free atoms, and between a free atom and a held one (which
    does not move).  That is what "whole-body" means, and it is what
    projecting the pattern out of a Hessian relies on."""
    patterns = rigid_motions(positions, held, kinds, cell=cell)
    n = len(positions)
    free = [i for i in range(n) if i not in set(held)]
    for pat in patterns:
        u = np.zeros((n, 3))
        u[free] = pat
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[i] - positions[j]
                assert abs(float(np.dot(d, u[i] - u[j]))) < 1e-9, (
                    f"atoms {i},{j}: the pattern stretches their distance")
    # Orthonormal over the free atoms, so a projector can be built from them.
    flat = patterns.reshape(len(patterns), 3 * len(free))
    assert np.allclose(flat @ flat.T, np.eye(len(patterns)), atol=1e-12)


def test_a_repeating_axis_needs_its_lattice_vector():
    with pytest.raises(ValueError, match="lattice vectors are needed"):
        rigid_motions(ZIGZAG, [], WIRE)
    with pytest.raises(ValueError, match="outside the structure"):
        rigid_motions(WATER, [3], ISO)


def _spring_hessian(positions, k=1.0):
    """Pair springs between every pair, at rest at this geometry, so the
    energy is exactly invariant under every whole-body motion the
    geometry permits and the Hessian is otherwise positive."""
    n = len(positions)
    H = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            r = d / np.linalg.norm(d)
            blk = k * np.outer(r, r)
            H[i, j] -= blk
            H[j, i] -= blk
            H[i, i] += blk
            H[j, j] += blk
    return H


@pytest.mark.parametrize("positions, held, kinds, cell, n_rigid", [
    p for p in TABLE if p.id not in ("lone-atom",)])
def test_what_is_reported_is_what_is_left(positions, held, kinds, cell,
                                          n_rigid):
    """R2, R3, R4 and § 7's acceptance test on one Hessian: exactly
    3 n_free - n_rigid modes come out, none of them is a removed motion
    (zero component along every pattern, in the mass metric), no
    spurious negative curvature appears, and each mode carries the
    canonical normalisation sum m |L|^2 = 1."""
    masses = np.linspace(1.0, 16.0, len(positions))
    H = _spring_hessian(positions)
    lam, modes, patterns = vibrational_modes(H, masses, positions, held,
                                             kinds, cell=cell)
    n_free = len(positions) - len(held)
    assert len(patterns) == n_rigid
    assert lam.shape == (3 * n_free - n_rigid,)
    assert modes.shape == (3 * n_free - n_rigid, n_free, 3)
    assert lam.min() > -1e-10
    free = [i for i in range(len(positions)) if i not in set(held)]
    sqm = np.sqrt(masses[free])
    for L in modes:
        assert abs(float((masses[free] * (L ** 2).sum(axis=1)).sum()) - 1.0) < 1e-10
        for pat in patterns:
            v = (pat * sqm[:, None]).reshape(-1)
            v /= np.linalg.norm(v)
            assert abs(float(np.dot((L * sqm[:, None]).reshape(-1), v))) < 1e-9


def test_the_free_case_is_the_held_case_with_nothing_held():
    """One path, not two: holding no atom must reproduce the free
    calculation exactly -- same count, same eigenvalues."""
    masses = np.array([15.999, 1.008, 1.008])
    H = _spring_hessian(WATER)
    lam_free, _, pats_free = vibrational_modes(H, masses, WATER, [], ISO)
    lam_none, _, pats_none = vibrational_modes(H, masses, WATER, (), ISO)
    assert len(pats_free) == len(pats_none) == 6
    assert np.allclose(lam_free, lam_none, atol=0, rtol=0)
