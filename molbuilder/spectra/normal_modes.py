"""The whole-body motions a vibration must remove, and the one path that removes them.

The science is ``docs/science/normal-modes.md``: § 3.1 is the rank rule
computed here, § 3.1a says which whole-body motions a lattice permits, § 6
R1-R4 are the rules these two functions hold, and § 7 is the test they are
checked by.  This module is the ONE derivation of ``n_rigid`` (R1): no
caller tabulates the count, branches on how many atoms are held, or asks
whether a molecule is straight.  Those answers fall out of a rank.

WHY A RANK AND NOT A TABLE.  The five-row table of § 3 (none held: six
motions; one held: three; two held: one; ...) is a consequence of one
statement -- *the whole-body motions that cost this system nothing and
leave every held atom in place* -- and writing the table into code is
writing five branches, each a chance to be wrong.  The table itself is
wrong twice: it says one leftover motion for CO2 with both oxygens held,
and the free carbon sits ON the O...O line, so the surviving turn moves
nothing and the answer is zero.  The rank gets that right without being
told.

WHY BOTH FUNCTIONS ARE SELF-CONTAINED.  The projection happens INSIDE the
generated deck, at run time, on a machine where this package is not
importable, so both functions travel into the deck as source text, the
way ``pyscf.vibration_emitters.dipole_derivatives`` does.  Nothing here
may read a module-level name: a constant referenced from module scope is
a ``NameError`` in the deck.  Each function imports numpy under its own
roof and takes everything else as an argument.  ``vibrational_modes``
calls ``rigid_motions`` by name, so a deck splices the two in this order.
"""
from __future__ import annotations


def rigid_motions(positions, held, axis_kind, cell=None, tol_ang=1e-3):
    """The whole-body motions that survive holding ``held`` still, over the free atoms.

    ``positions`` are every atom's coordinates, shape (n_atoms, 3), in
    Angstrom.  ``held`` is the set of 0-based atom indices held in place
    (may be empty).  ``axis_kind`` is the structure's own per-axis kind --
    three of ``"periodic"``, ``"isolated"``, ``"transport"`` -- and
    ``cell`` its lattice vectors as ROWS, required whenever an axis is
    not isolated: which turns survive depends on the vectors, not only on
    how many there are.

    Returns an array of shape (n_rigid, n_free, 3): an orthonormal basis
    (plain Euclidean, over the free atoms' Cartesian coordinates) of the
    motions that cost this system no energy and leave every held atom
    exactly where it is.  Its length is ``n_rigid(F)`` of
    science/normal-modes.md § 3.1.

    The rule, with no cases:

    1. The energy is invariant under three translations, and under the
       rotations whose generator maps every lattice vector onto itself --
       an antisymmetric ``A`` with ``A.a = 0`` for every lattice vector
       ``a`` of a non-isolated axis.  With the axial vector ``w`` of ``A``
       that reads ``w x a = 0``, so ``w`` lies in the null space of the
       stacked cross-product matrices of those vectors: three such ``w``
       with no lattice, one for a wire, none for a slab or a crystal.
       An axis that continues (``"transport"``) binds a turn exactly as a
       repeating one does.
    2. Of the combinations of those motions, keep the ones that move no
       held atom: the null space of the held atoms' rows.
    3. Of what survives, keep what actually moves a free atom: the rank of
       the free atoms' rows restricted to the survivors.  A straight
       molecule's turn about its own axis, a lone atom's turns, and CO2's
       turn about a held O...O line with the carbon on it all vanish here,
       unasked.

    ``tol_ang`` decides steps 2 and 3, in Angstrom per unit motion: a
    combination whose root-sum-square displacement of the atoms in
    question is below it moves nothing the geometry can resolve.  A
    millionth of an Angstrom would make a molecule bent by numerical noise
    "not straight" and hand a real bend to the projection; a tenth would
    call a genuinely bent molecule straight.  PySCF's own linearity test
    sits at a moment of inertia of 5e-6 amu.A^2, which for a light atom is
    this order of off-axis distance.
    """
    import numpy as _np

    R = _np.asarray(positions, dtype=float).reshape(-1, 3)
    n = R.shape[0]
    held_idx = sorted({int(i) for i in held})
    for i in held_idx:
        if not 0 <= i < n:
            raise ValueError(
                f"held atom index {i} is outside the structure "
                f"({n} atoms, indices 0..{n - 1})")
    held_set = set(held_idx)
    free_idx = [i for i in range(n) if i not in held_set]
    kinds = tuple(str(k) for k in axis_kind)
    if len(kinds) != 3:
        raise ValueError(f"axis_kind must name three axes; got {kinds!r}")

    def _nullspace(M, tol):
        # The columns spanning {x : M x = 0}, to the given tolerance on
        # the singular values.
        _, s, vt = _np.linalg.svd(_np.asarray(M, dtype=float),
                                  full_matrices=True)
        rank = int((s > tol).sum())
        return vt[rank:].T

    # 1. The motions the energy is invariant under.
    lattice_axes = [i for i, k in enumerate(kinds) if k != "isolated"]
    if lattice_axes:
        if cell is None:
            raise ValueError(
                f"axis_kind {kinds!r} has a non-isolated axis, so the "
                f"lattice vectors are needed to say which turns survive; "
                f"pass cell=")
        C = _np.asarray(cell, dtype=float).reshape(3, 3)
        cross = []
        for i in lattice_axes:
            a = C[i]
            cross.append(_np.array([[0.0, -a[2], a[1]],
                                    [a[2], 0.0, -a[0]],
                                    [-a[1], a[0], 0.0]]))
        M = _np.vstack(cross)
        w_basis = _nullspace(M, 1e-8 * max(1.0, float(_np.abs(M).max())))
    else:
        w_basis = _np.eye(3)
    centre = R.mean(axis=0)
    columns = [_np.tile(e, (n, 1)) for e in _np.eye(3)]
    columns += [_np.cross(w, R - centre) for w in w_basis.T]
    G = _np.stack([c.reshape(-1) for c in columns], axis=1)   # (3n, m)
    G3 = G.reshape(n, 3, G.shape[1])

    # 2. Keep the combinations that leave every held atom in place.
    if held_idx:
        N = _nullspace(G3[held_idx].reshape(-1, G.shape[1]), tol_ang)
    else:
        N = _np.eye(G.shape[1])
    if N.shape[1] == 0 or not free_idx:
        return _np.zeros((0, len(free_idx), 3))

    # 3. Keep what actually moves a free atom.
    S = G3[free_idx].reshape(-1, G.shape[1]) @ N               # (3 n_free, k)
    U, s, _ = _np.linalg.svd(S, full_matrices=False)
    keep = int((s > tol_ang).sum())
    return U[:, :keep].T.reshape(keep, len(free_idx), 3)


def vibrational_modes(hessian, masses, positions, held, axis_kind, cell=None):
    """Every vibration of the system, and nothing that is not one (R2-R4).

    ``hessian`` is the FULL second-derivative table over all atoms, shape
    (n_atoms, n_atoms, 3, 3), in any energy/length^2 unit; ``masses`` one
    per atom, in any mass unit; the rest as for ``rigid_motions``.

    Returns ``(eigenvalues, modes, patterns)``:

    * ``eigenvalues`` -- shape (n_vib,), ascending, in hessian units per
      mass unit; a negative one is an imaginary mode.  ``n_vib`` is
      exactly ``3 * n_free - n_rigid`` (R2), by construction: the
      mass-weighted Hessian is diagonalised in the complement of the
      removed motions, so nothing removed can come back as a mode.
    * ``modes`` -- shape (n_vib, n_free, 3), Cartesian, in the canonical
      mass-weighted normalisation ``sum_k m_k |L_k|^2 = 1`` with the
      masses as given.
    * ``patterns`` -- what was removed, from ``rigid_motions``, shape
      (n_rigid, n_free, 3), so the report can say what it was (R7).

    The held atoms enter in two places only: which block of the Hessian
    is kept -- the free-free block of the TRUE Hessian, with the held
    atoms present in the energy (normal-modes.md § 3) -- and which motions
    survive holding them.  With nothing held this is the free-molecule
    calculation; there is no second path (R3).
    """
    import numpy as _np

    H = _np.asarray(hessian, dtype=float)
    m = _np.asarray(masses, dtype=float).reshape(-1)
    n = m.shape[0]
    if H.shape != (n, n, 3, 3):
        raise ValueError(
            f"hessian must have shape (n_atoms, n_atoms, 3, 3) = "
            f"({n}, {n}, 3, 3) for {n} masses; got {H.shape}")
    if (m <= 0).any():
        raise ValueError("every mass must be positive")
    held_idx = sorted({int(i) for i in held})
    held_set = set(held_idx)
    free_idx = [i for i in range(n) if i not in held_set]
    nf = len(free_idx)
    patterns = rigid_motions(positions, held_idx, axis_kind, cell)

    # The free-free block, mass-weighted: H_ij / sqrt(m_i m_j) per 3x3.
    Hf = H[free_idx][:, free_idx]
    h2 = Hf.transpose(0, 2, 1, 3).reshape(3 * nf, 3 * nf)
    sqm = _np.sqrt(m[free_idx])
    wgt = _np.repeat(1.0 / sqm, 3)
    hmw = h2 * _np.outer(wgt, wgt)
    hmw = 0.5 * (hmw + hmw.T)

    # The complement of the removed motions, in the mass-weighted metric:
    # a Cartesian motion u is the mass-weighted vector sqrt(m) u.
    n_rigid = patterns.shape[0]
    if n_rigid:
        V = (patterns * sqm[None, :, None]).reshape(n_rigid, -1).T
        U, _, _ = _np.linalg.svd(V, full_matrices=True)
        B = U[:, n_rigid:]
    else:
        B = _np.eye(3 * nf)
    hred = B.T @ hmw @ B
    hred = 0.5 * (hred + hred.T)
    lam, modes_red = _np.linalg.eigh(hred)
    L_mw = B @ modes_red                                        # (3nf, n_vib)
    L_cart = L_mw.T.reshape(-1, nf, 3) / sqm[None, :, None]
    return lam, L_cart, patterns


__all__ = ["rigid_motions", "vibrational_modes"]
