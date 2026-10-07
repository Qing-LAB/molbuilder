"""The PySCF vibration's run-time rules -- what the vibration script calls
inside its run.

The script imports them from ``mb_pyscf.pyz`` beside it
(``runwrap.PYSCF_COMPANIONS``, ``engines/pyscf.md`` § 3), where molbuilder is
not installed: this file travels as itself, so it imports the standard library,
numpy and ``constants`` at load -- the last the two ways a travelling module
does -- and PySCF only inside the functions that drive it, under the job's own
env.  The SIESTA route's counterpart is
``spectra/siesta_vibration.py``.
"""
from __future__ import annotations

try:                                        # inside molbuilder
    from ..constants import DEBYE_E_ANGSTROM
except ImportError:                         # beside a job, in mb_pyscf.pyz
    from constants import DEBYE_E_ANGSTROM


#: The HOMO rule, as a REAL FUNCTION so it can be called and tested.
#:
#: It has a BRANCH: PySCF gives `mo_occ` as a 1-D array for RHF/RKS and a 2-D (alpha, beta) array
#: for UHF/UKS, which must be summed before the highest occupied level can be
#: found.  Get that wrong and every OPEN-SHELL calculation reports the wrong
#: HOMO -- and `web/spectra.md` § 3.1 records the level diagram, the gap and the
#: gap shift all reading from this index, with shifts of ~0.018 meV, so a wrong
#: index looks like a different answer rather than an error.
#:
#: The script imports THIS function from `mb_pyscf.pyz`, so there is one
#: implementation and the tests exercise the one that runs.  That is the trigger
#: stated in `siesta/makov_payne.py`: copy a branchless formula if you must, but
#: make it a callable that travels as its module once it has a branch.
def homo_index(mo_occ) -> int:
    """Index of the highest occupied molecular orbital.

    ``mo_occ`` is PySCF's occupation array: 1-D for a restricted reference,
    2-D ``(alpha, beta)`` for an unrestricted one, which is summed to a total
    occupancy first.  "Occupied" is occupancy above 0.5 -- half an electron --
    which separates a filled level (1.0 unrestricted, 2.0 restricted) from an
    empty one without assuming integer occupations.
    """
    import numpy as _np
    occ = _np.asarray(mo_occ, dtype=float)
    total = occ.sum(axis=0) if occ.ndim == 2 else occ
    filled = _np.where(total > 0.5)[0]
    if filled.size == 0:
        raise ValueError(
            "no molecular orbital has occupancy above 0.5: the reference has "
            "no occupied levels, so there is no HOMO to index")
    return int(_np.max(filled))


#: Imported by the deck from `mb_pyscf.pyz`, for the reason
#: `docs/engines/vibration.md` § 6.4 states: a derivation with a BRANCH is a
#: callable, so one implementation runs and the tests exercise the one that runs.
#: The branch here is unavoidable and RUNTIME, not emit-time -- the deck is
#: generated on the host and executed inside `molbuilder-pySCF`, so whether the
#: analytic route exists is a property of the env it lands in, not of the
#: machine that wrote it.
def dipole_derivatives(mf, free_atom_idxs, want_ir):
    """The Hessian, and -- when asked and available -- dmu/dR with it.

    Returns ``(hessian, dmu_dr, route)``:

    * ``hessian`` -- Hartree/Bohr^2, shape (n_atoms, n_atoms, 3, 3).
      **With atoms held, only the free atoms' blocks are computed** (see
      below); the held atoms' rows are zero and are never read.  The
      SAME Hessian the no-IR path produces -- asking for IR must not
      move the frequencies.  That is established by CONSTRUCTION, not
      by sampling: ``Hessian.kernel()`` is ``hess_elec + hess_nuc``
      plus a dispersion term when the functional has one, and the two
      corrections below make this path compute that same sum with the
      same object.  Verified to ~1e-12 with and without dispersion;
    * ``dmu_dr``  -- Debye/Angstrom, shape (n_free, 3, 3) indexed
      (free atom, Cartesian displacement, dipole component), or ``None``
      when the caller must fall back to finite differences;
    * ``route``   -- ``"analytic"``, ``"finite-difference"`` or
      ``"none"``, recorded in the results so a reader can tell which
      one produced the numbers.

    WHY THE ANALYTIC ROUTE IS NEARLY FREE.  An analytic Hessian's
    dominant cost is solving the CPHF equations for nuclear
    displacement, and dmu/dR is those same solutions contracted with
    dipole integrals.  ``pyscf.prop.infrared`` computes the Hessian as
    a by-product of that contraction, so asking it for both costs one
    CPHF solve, not two -- measured +14% over the Hessian alone, versus
    +486% for the 6N extra SCFs a finite-difference dipole sweep needs
    (NH3/PBE0/6-31G, 2026-09-11).  The gap widens with atom count: the
    sweep grows as 6N full SCFs, this does not.

    WHY IT CAN BE ABSENT.  ``pyscf.prop.infrared`` has never been
    released to PyPI -- it exists only on the project's master branch
    (see `docs/ops/installation.md` § 3.1).  An env installed from the
    index has no analytic route, and that is a supported state: the
    finite-difference path produces the SAME intensities (the two
    dmu/dR tensors agree to 0.02%), just slowly.  So absence returns
    ``None`` rather than raising.

    The unit conversion is the one thing a reader cannot check by
    eye: upstream returns d(mu)/d(R) in atomic units per Bohr, and the
    deck's projection wants Debye per Angstrom.

    THE REDUCED CALCULATION, which is what holding atoms is FOR.  With
    atoms held, second derivatives are computed for the free atoms only:
    PySCF's ``atmlst`` reaches the coupled-perturbed solve -- the expensive
    step -- so its cost scales with the free atoms rather than with all
    of them, while the held atoms still shape the energy through the
    self-consistent field (the block is the free-free block of the TRUE
    Hessian, Besley's partial Hessian).  ``kernel(atmlst=...)`` cannot be
    used for it: the dispersion term it adds is full-size, so the three
    pieces are summed here and the dispersion block is cut to the free
    atoms.  The result is numbered by position in the list passed, so it
    is placed back into a full-size table by index.  The analytic
    dipole-derivative route takes no atom list, so infrared with held
    atoms is by finite differences over the free atoms -- the same
    numbers, more SCFs.  **The mean field handed in must not be density
    fitted**: PySCF's density-fitted Hessian class takes no atom list (its
    three-centre contraction fails on one), so the deck builds a plain
    mean field for this route and the run states that the Hessian ran
    without density fitting.  Measured against compute-everything-and-slice
    (engines/vibration.md § 4.4): Hartree-Fock blocks agree to 1e-8 Hartree/Bohr^2; DFT
    blocks to 1.5e-5, the held atoms' grid-weight response that the partial
    list omits -- about 0.05 cm^-1 on a stretch.
    """
    import numpy as _np

    # The elementary charge expressed in D/A -- the nuclear term
    # de[a] = Z_a * I is a point charge, so the factor must be e.
    _AU_BOHR_TO_DEBYE_ANG = DEBYE_E_ANGSTROM

    _n_atoms = int(mf.mol.natm)
    _free = [int(i) for i in free_atom_idxs]
    if len(_free) < _n_atoms:
        _hobj = mf.Hessian()
        _block = (_np.asarray(_hobj.hess_elec(atmlst=_free))
                  + _np.asarray(_hobj.hess_nuc(mf.mol, atmlst=_free)))
        if mf.do_disp():
            _block = _block + _np.asarray(_hobj.get_dispersion())[_free][:, _free]
        _full = _np.zeros((_n_atoms, _n_atoms, 3, 3))
        _full[_np.ix_(_free, _free)] = _block
        return _full, None, ("finite-difference" if want_ir else "none")

    def _infrared_class(infrared, mf_):
        """Upstream splits by reference, and so must we.

        Unrestricted references carry a 2-D occupancy; a Kohn-Sham
        object carries ``xc``.  Density fitting is NOT a split -- a
        DF-RKS goes through the rks class and agrees with the non-DF
        answer to 0.004 km/mol (measured 2026-09-11), unlike the
        polarizability module, which has no DF implementation at all.
        """
        occ = _np.asarray(mf_.mo_occ)
        unrestricted = occ.ndim == 2
        is_ks = hasattr(mf_, "xc")
        if unrestricted:
            return infrared.uks.Infrared if is_ks else infrared.uhf.Infrared
        return infrared.rks.Infrared if is_ks else infrared.rhf.Infrared

    if want_ir:
        try:
            from pyscf.prop import infrared as _infrared
            _cls = _infrared_class(_infrared, mf)
            _mf_ir = _cls(mf)
            # HAND IT THE SCF'S OWN HESSIAN OBJECT.  Upstream's
            # ``hess_cls`` is hardcoded to the NON-DF class, so on a
            # density-fitted SCF -- molbuilder's default -- it builds a
            # non-DF Hessian of a DF density.  That is a real mismatch,
            # not a rounding artifact: measured 7.2e-5 Hartree/Bohr^2
            # against the SCF's own Hessian, shifting frequencies by
            # 0.11 cm^-1.  Small, but it would mean asking for IR
            # silently changed the frequencies, which is not a trade a
            # user agreed to.  Injecting ``mf.Hessian()`` makes
            # ``proc_hessian_`` solve CPHF with the SCF's own machinery:
            # the Hessian then matches the no-IR path to 3.6e-12 -- and
            # it is FASTER, because there is still only one solve
            # (14.8 s versus 14.4 s for the Hessian alone).
            _hobj = mf.Hessian()
            _mf_ir.mf_hess = _hobj
            _mf_ir.kernel_dipderiv()          # Hessian + dmu/dR, one CPHF solve
            _hess = _np.asarray(_mf_ir.mf_hess.de)
            # RESTORE THE DISPERSION TERM.  ``Hessian.kernel()`` is
            #     hess_elec + hess_nuc + get_dispersion() if base.do_disp()
            # while upstream's ``proc_hessian_`` computes only the first
            # two and overwrites ``.de`` with them.  The paths therefore
            # differ by EXACTLY the dispersion Hessian -- an omission,
            # not a rounding difference, and invisible on any functional
            # that carries no dispersion correction.  Measured on
            # B3LYP-D3BJ: 7.2e-4 Hartree/Bohr^2, a 3.7 cm^-1 shift on
            # every frequency; adding the term back reproduces
            # ``kernel()`` to 5.7e-12.  molbuilder ships
            # ``pyscf-dispersion``, so -D functionals are ordinary here.
            if mf.do_disp():
                _hess = _hess + _np.asarray(_hobj.get_dispersion())
            _de = _np.asarray(_mf_ir.de)[list(free_atom_idxs)]
            return _hess, _de * _AU_BOHR_TO_DEBYE_ANG, "analytic"
        except ImportError:
            print("  pyscf.prop.infrared not installed -- IR falls back to "
                  "finite-difference dipoles (same intensities, 6N extra "
                  "SCFs).  Install it: bash scripts/install-env.sh repair "
                  "molbuilder-pySCF --include-optional")
        except Exception as _exc:          # noqa: BLE001 -- see below
            # Deliberately broad, and deliberately LOUD.  The analytic
            # route is an optimisation over a working fallback, so no
            # failure of it may cost the user their run -- but a silent
            # swallow would hide a real defect behind a slow success,
            # so the reason is printed and lands in the job log.
            print(f"  analytic dmu/dR unavailable ({type(_exc).__name__}: "
                  f"{_exc}) -- falling back to finite-difference dipoles")
    return (_np.asarray(mf.Hessian().kernel()), None,
            "finite-difference" if want_ir else "none")


def mo_window(mo_energy, homo, n_below, n_above):
    """The orbital energies the per-mode probe records (`engines/vibration.md`
    § 4.8): the HOMO and ``n_below`` levels under it, the LUMO and ``n_above``
    over it -- symmetric by contract, so the slice ends at
    ``homo + 2 + n_above``, one past the last kept -- and the HOMO's place in
    it.  Returns ``(energies, homo_in_window)``."""
    import numpy as np
    e = np.asarray(mo_energy)
    lo = max(0, int(homo) - int(n_below))
    hi = min(len(e), int(homo) + 2 + int(n_above))
    return e[lo:hi].copy(), int(homo) - lo


def as_numpy(x):
    """A CuPy or NumPy array -- or a list, or a scalar -- as NumPy.

    On a GPU run (gpu4pyscf) ``mf.mo_energy``, ``mf.mo_occ`` and
    ``mf.Hessian().kernel()`` come back as CuPy arrays, which refuse an
    implicit conversion; everything downstream -- ``pyscf.hessian.thermo``,
    ``np.linalg.eigh``, the result's JSON -- is CPU.  This is the one
    crossing: an explicit ``.get()`` for CuPy, found by its module's name so
    a CPU env never imports cupy, and ``np.asarray`` for anything else.
    """
    import numpy as np
    if type(x).__module__.startswith("cupy"):
        return x.get()
    return np.asarray(x)


__all__ = ["homo_index", "mo_window", "dipole_derivatives", "as_numpy"]
