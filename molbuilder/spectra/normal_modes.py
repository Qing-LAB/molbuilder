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

TRAVELS AS ITSELF, in both bundles: the PySCF vibration script imports it
from ``mb_pyscf.pyz`` and runs the projection inside its run
(`runwrap.PYSCF_COMPANIONS`, `engines/pyscf.md` § 3), and a SIESTA
force-constant job's finish runs it from ``mb_vibration.pyz``
(`runwrap.VIBRATION_COMPANIONS`, `engines/vibration.md` § 5.5) -- both where
this package is not installed, so it imports nothing of molbuilder.  Until
2026-10-05 the PySCF deck carried its functions as source text, which is why
each still imports numpy under its own roof and takes everything else as an
argument.
"""
from __future__ import annotations

# TWO WAYS, because this module travels (above).
try:                                        # inside molbuilder
    from ..constants import (BOLTZMANN_HARTREE_K,
                             CM1_PER_SQRT_HARTREE_BOHR2_AMU, HARTREE_CM1)
except ImportError:                         # beside a job, in either bundle
    from constants import (BOLTZMANN_HARTREE_K,
                           CM1_PER_SQRT_HARTREE_BOHR2_AMU, HARTREE_CM1)


def rigid_motions(positions, held, axis_kind, cell=None, tol_ang=1e-3):
    """The whole-body motions that survive holding ``held`` still, over the free atoms.

    ``positions`` are every atom's coordinates, shape (n_atoms, 3), in
    Angstrom.  ``held`` is the set of 0-based atom indices held in place
    (may be empty).  ``axis_kind`` is the per-axis kind the ENGINE computes
    on (``cell.engine_axis_kinds``: the structure's in a cell, a cluster's
    in free space) -- three of ``"periodic"``, ``"isolated"``,
    ``"transport"`` -- and
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


def signed_omega(eigenvalues):
    """``sign(lambda) * sqrt(|lambda|)`` per mode, in atomic units: a
    negative eigenvalue of the mass-weighted Hessian is an imaginary mode,
    carried as a negative number.  The eigenvalues are
    :func:`vibrational_modes`' own, in Eh/(Bohr^2 amu)."""
    import numpy as _np
    lam = _np.asarray(eigenvalues, dtype=float)
    return _np.sign(lam) * _np.sqrt(_np.abs(lam))


def frequencies_cm1(eigenvalues):
    """Each mode's wavenumber, cm-1 -- :func:`signed_omega` through the one
    constant (``constants.CM1_PER_SQRT_HARTREE_BOHR2_AMU``), an imaginary
    mode negative.  Both routes report their modes through it."""
    return signed_omega(eigenvalues) * CM1_PER_SQRT_HARTREE_BOHR2_AMU


def display_form(canonical):
    """Each mode rescaled so its largest component is 1 -- the form the
    viewer animates and the per-mode probe displaces along; never for a
    physical amplitude, which the canonical form carries.  A mode of zeros
    stays zeros.  ``canonical`` is the modes' array, one mode per row."""
    import numpy as _np
    L = _np.asarray(canonical, dtype=float)
    out = _np.zeros_like(L)
    for k in range(L.shape[0]):
        peak = float(_np.max(_np.abs(L[k]))) if L[k].size else 0.0
        if peak > 0:
            out[k] = L[k] / peak
    return out


def thermo_temperatures(temperature_K):
    """The temperatures the viewer's curves run over -- :data:`THERMO_GRID_K`
    with the headline temperature added, sorted, so the curve passes through
    the headline number and the two cannot disagree."""
    return sorted(set(float(t) for t in THERMO_GRID_K)
                  | {float(temperature_K)})


def vibrational_thermo(freqs_cm1, temperature_K):
    """The harmonic vibrational sums at one temperature: ``(zpe, u_vib, s_vib)``
    in Hartree, Hartree and Hartree/K.

    Over the frequencies given -- every one a vibration, none imaginary:
    the caller has already removed the whole-body motions (R3) and left
    out an imaginary mode, which has no partition function.  The constants
    are the one home's (``constants.py``).  No frequencies give three
    zeros; ``T <= 0`` gives the T -> 0 limit -- the zero-point energy,
    which no temperature removes, with no thermal energy and no entropy.
    """
    import numpy as _np
    w = _np.asarray(freqs_cm1, dtype=float).reshape(-1) * (1.0 / HARTREE_CM1)
    T = float(temperature_K)
    if w.size == 0:
        return 0.0, 0.0, 0.0
    zpe = float(0.5 * w.sum())
    if T <= 0.0:
        return zpe, 0.0, 0.0
    x = _np.clip(w / (BOLTZMANN_HARTREE_K * T), 1e-12, 700.0)
    u = float((w / (_np.exp(x) - 1.0)).sum())
    s = float(BOLTZMANN_HARTREE_K * ((x / (_np.exp(x) - 1.0)
                                      - _np.log1p(-_np.exp(-x))).sum()))
    return zpe, u, s


def vibrational_thermo_grid(freqs_cm1, temperatures_K, e_ref_eh):
    """The vibrational-only curves a viewer draws, at each temperature:
    ``{temperatures_K, zpe_eh, u_vib_eh, h_eh, s_eh_k, g_eh}`` as lists.

    ``h = e_ref + zpe + u_vib`` and ``g = h - T s``: the harmonic sums above
    a reference energy (the electronic energy when one is reported, zero
    when it is not), with NO ``kT`` term -- that is the ideal gas's ``pV``,
    which a system with atoms held has no claim to.  One home for the
    assembly, so the deck's grid and the host-side derivation cannot
    differ.
    """
    grid = {"temperatures_K": [], "zpe_eh": [], "u_vib_eh": [],
            "h_eh": [], "s_eh_k": [], "g_eh": []}
    for T in temperatures_K:
        T = float(T)
        z, u, s = vibrational_thermo(freqs_cm1, T)
        h = float(e_ref_eh) + z + u
        grid["temperatures_K"].append(T)
        grid["zpe_eh"].append(z)
        grid["u_vib_eh"].append(u)
        grid["h_eh"].append(h)
        grid["s_eh_k"].append(s)
        grid["g_eh"].append(h - T * s)
    return grid


#: The temperatures the viewer's curves run over -- a documented
#: presentation default, not a scientific knob (the headline temperature
#: is the knob): 50-1500 K in 30 points, wide enough to show the trend and
#: free to compute.  Both writers take it from here and add the headline
#: temperature to it, so the curve passes through the headline number.
THERMO_GRID_K = tuple(float(x) for x in
                      __import__("numpy").linspace(50.0, 1500.0, 30))


__all__ = ["rigid_motions", "vibrational_modes", "signed_omega",
           "frequencies_cm1", "display_form", "thermo_temperatures",
           "vibrational_thermo", "vibrational_thermo_grid", "THERMO_GRID_K"]
