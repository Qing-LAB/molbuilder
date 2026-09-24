"""The SIESTA vibration's artifact, derived on the host from a force-constant run.

SIESTA's part of a vibration is the force-constant run: it nudges the free
atoms and writes ``<SystemLabel>.FC`` (`parse.engines.siesta_fc`).  Every
step after that is the same as PySCF's and runs here, through the same
functions the PySCF deck carries as source: the free-free block is
mass-weighted and diagonalised in the complement of the whole-body motions
the held geometry permits (`spectra.normal_modes.vibrational_modes`), the
vibrational thermochemistry is summed (`vibrational_thermo`), and the
result is the one artifact both engines share (`web/spectra.md` § 9b.3).
Two things this route cannot produce are absent, never zero: intensities
(`science/normal-modes.md` § 4a.6) and the molecular-orbital block.

THE ORDER THE PERSON SEES IS THE INPUT ORDER (`model/overview.md` § 2.2).
The deck was written from a sorted copy -- held atoms first, free atoms
last, so the FC range is one run -- and the permutation was recorded
beside the calculation.  This module takes that record back: the free
block's rows come out of the file in the sorted order and are put back in
the input's, so ``free_atom_idxs``, every eigenvector row and every removed
pattern speak the person's numbering.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..chemistry import atomic_mass
from ..constants import (BOLTZMANN_HARTREE_K, CM1_PER_SQRT_HARTREE_BOHR2_AMU,
                         HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_CM1)
from ..parse.engines.siesta_fc import (fc_block_asymmetry, hessian_from_fc,
                                        read_fc)
from ..structure import Structure
from ..transport.sort import Permutation
from .methods import extract_citation_keys
from .normal_modes import (THERMO_GRID_K, vibrational_modes,
                           vibrational_thermo, vibrational_thermo_grid)
from .results import (PHASE_COMPLETE, SCHEMA_VERSION, ModeData,
                      SpectraResults)

#: The temperature and pressure the thermochemistry is summed at when the
#: description states none -- the same headline PySCF's items default to.
DEFAULT_TEMPERATURE_K = 298.15
DEFAULT_PRESSURE_ATM = 1.0


def siesta_methods_text(*, displacement_bohr: float, n_free: int,
                        n_held: int, n_rigid: int, siesta_version: str) -> str:
    """The Methods paragraph for the force-constant route, with the same
    citation keys the science contract carries."""
    held = (f" {n_held} atom(s) were held fixed and no force constant was "
            f"taken with respect to them -- partial Hessian vibrational "
            f"analysis [Head1997, LiJensen2002], the block taken from the "
            f"forces of the full system [Besley2008]." if n_held else "")
    removed = (f" The {n_rigid} whole-body motion(s) the held geometry "
               f"permits were projected out before diagonalisation "
               f"[Ghysels2008]." if n_rigid else "")
    ver = f" (SIESTA {siesta_version})" if siesta_version else ""
    return (
        "## Methods\n\n"
        f"Harmonic force constants were obtained by central finite "
        f"differences of the analytic forces{ver}: each of the {n_free} "
        f"free atoms was displaced by ±{displacement_bohr:g} Bohr along "
        f"x, y and z (`MD.TypeOfRun FC`).{held} The resulting partial "
        f"Hessian was mass-weighted with isotope-averaged masses and "
        f"diagonalised.{removed} Infrared and Raman intensities are not "
        f"computed on this route."
    )


def spectra_results_from_fc(struct: Structure, sorted_struct: Structure,
                            permutation: Permutation, fc_path,
                            *, label: str,
                            displacement_bohr: Optional[float] = None,
                            temperature_K: float = DEFAULT_TEMPERATURE_K,
                            pressure_atm: float = DEFAULT_PRESSURE_ATM,
                            engine_version: str = "",
                            molbuilder_version: str = "",
                            config: Optional[dict] = None,
                            timestamp: Optional[str] = None,
                            reference_forces_ev_ang: Optional[np.ndarray] = None,
                            force_criterion_ev_ang: Optional[float] = None,
                            already_relaxed: bool = False) -> SpectraResults:
    """The artifact for a finished force-constant run.

    ``struct`` is the structure the calculation is OF, in INPUT order;
    ``sorted_struct`` the copy the deck was written from and
    ``permutation`` the record read back beside the calculation
    (`transport.sort.read_permutation`); ``fc_path`` the ``.FC`` the run
    left.  ``reference_forces_ev_ang`` are the forces SIESTA evaluated at
    the undisplaced geometry (its FC step 0, `siesta_fc.reference_forces_from_out`),
    in the SORTED order like everything the run wrote; with them and
    ``force_criterion_ev_ang`` (the relaxation's own criterion) the
    artifact's ``relaxation`` block judges stationarity over the free
    atoms (R5 on this route, `engines/vibration.md` § 5.5).
    ``already_relaxed`` is the person's assertion, carried as such.
    """
    n = len(sorted_struct.elements)
    if len(struct.elements) != n or permutation.n_atoms != n:
        raise ValueError(
            f"the recorded permutation does not fit this structure: "
            f"{len(struct.elements)} atoms in the input, {n} in the sorted "
            f"copy, {permutation.n_atoms} in the record")
    held_s = sorted(int(i) for i in (sorted_struct.frozen_atoms or []))
    free_s = [i for i in range(n) if i not in set(held_s)]
    if free_s != list(range(n - len(free_s), n)):
        raise ValueError(
            "the sorted copy does not hold the free atoms as one trailing "
            "run -- the deck cannot have been written from it "
            "(model/overview.md § 2.2)")
    if not free_s:
        raise ValueError("every atom is held: there is nothing to vibrate")

    fc = read_fc(fc_path)
    if fc.n_atoms != n:
        raise ValueError(
            f"{Path(fc.path).name} describes {fc.n_atoms} atoms; the "
            f"structure has {n}")
    H = hessian_from_fc(fc, free_s)
    # The block's own numerics, before the symmetrisation hides them
    # (`engines/vibration.md` § 5.5): read from the same two-sided mean.
    fc_asymmetry = fc_block_asymmetry(fc, free_s)
    masses = np.array([atomic_mass(e) for e in sorted_struct.elements],
                      dtype=float)
    lam, L_sorted, patterns_sorted = vibrational_modes(
        H, masses, sorted_struct.positions, held_s,
        sorted_struct.axis_kind or ("isolated",) * 3,
        cell=sorted_struct.cell)
    omega = np.sign(lam) * np.sqrt(np.abs(lam))
    freqs_cm1 = omega * CM1_PER_SQRT_HARTREE_BOHR2_AMU

    # Back to the input order through the framework's one inversion:
    # every per-free-atom row follows the free atoms' INPUT indices.
    L, free_atom_idxs = permutation.rows_to_input_order(L_sorted, free_s)
    patterns, _ = permutation.rows_to_input_order(patterns_sorted, free_s)
    frozen_atom_idxs = sorted(permutation.original_of(held_s))

    modes = []
    for k, f in enumerate(freqs_cm1):
        canon = L[k]
        peak = float(np.max(np.abs(canon))) if canon.size else 0.0
        modes.append(ModeData(
            index_1based=k + 1,
            frequency_cm1=float(f),
            raman_activity_a4_amu=None,
            ir_intensity_km_mol=None,
            eigenvector_canonical=canon,
            eigenvector_display=(canon / peak if peak > 0 else canon.copy()),
            has_imag=bool(f < 0),
        ))

    kept = np.array([m.frequency_cm1 for m in modes if not m.has_imag])
    n_imag = int(sum(1 for m in modes if m.has_imag))
    eh_per_cm1 = 1.0 / HARTREE_CM1
    zpe, u0, s0 = vibrational_thermo(kept, temperature_K,
                                     BOLTZMANN_HARTREE_K, eh_per_cm1)
    # No total energy is reported on this route, so the curves are the
    # vibrational contributions above the electronic minimum (e_ref = 0),
    # through the one grid home both writers share; the headline
    # temperature is on the grid, so the headline IS a grid row.
    temps = sorted(set(THERMO_GRID_K) | {float(temperature_K)})
    grid = vibrational_thermo_grid(kept, temps, 0.0, BOLTZMANN_HARTREE_K,
                                   eh_per_cm1)
    k = grid["temperatures_K"].index(float(temperature_K))
    n_rigid = int(len(patterns))
    thermo = {
        "regime": "vibrational-only",
        "temperature_K": float(temperature_K),
        "pressure_atm": float(pressure_atm),
        "zpe_eh": zpe,
        "h_eh": grid["h_eh"][k],
        "s_eh_k": grid["s_eh_k"][k],
        "g_eh": grid["g_eh"][k],
        "n_modes": len(modes),
        "n_imag_excluded": n_imag,
        "n_rigid_removed": n_rigid,
        "note": ("VIBRATIONAL contributions only, above the electronic "
                 "minimum, the headline and the grid alike: the "
                 "force-constant route reports no total energy and has no "
                 "gas-phase translational or rotational partition function "
                 "to add; the whole-body motions of the free atoms "
                 "(n_rigid_removed) were removed before diagonalising"),
        "grid": grid,
    }

    # STATIONARITY, JUDGED HERE (R5 on this route; `engines/vibration.md`
    # § 5.5).  The forces are the run's, in the sorted order; the judged
    # number is the largest component over the FREE atoms, the way the
    # PySCF deck judges its own (§ 4.3), converted with the force-unit
    # constant every force number in this codebase uses.  `already_relaxed`
    # is the person's assertion and is carried as such: on this route it is
    # the precondition itself, and the numbers below say whether it held.
    warning = ("the force-constant route does not relax: the input geometry "
               "is taken as the stationary point")
    max_free_eh = max_all_eh = None
    converged = None
    if reference_forces_ev_ang is not None:
        f_ref = np.asarray(reference_forces_ev_ang, dtype=float)
        if f_ref.shape != (n, 3):
            raise ValueError(
                f"the reference forces describe "
                f"{f_ref.shape[0] if f_ref.ndim else '?'} atoms; the "
                f"structure has {n}")
        max_free_ev = float(np.max(np.abs(f_ref[free_s])))
        max_all_ev = float(np.max(np.abs(f_ref)))
        max_free_eh = max_free_ev / HARTREE_BOHR_EV_ANGSTROM_ASE
        max_all_eh = max_all_ev / HARTREE_BOHR_EV_ANGSTROM_ASE
        if force_criterion_ev_ang is not None:
            converged = bool(max_free_ev <= float(force_criterion_ev_ang))
            if converged:
                warning += (f".  The forces at the reference geometry were "
                            f"read back: the largest on the free atoms is "
                            f"{max_free_ev:.4f} eV/Å, within the relaxation "
                            f"criterion of {float(force_criterion_ev_ang):g} eV/Å")
            else:
                warning = (f"the reference geometry is not a stationary point "
                           f"at this level of theory: the largest force on the "
                           f"free atoms is {max_free_ev:.4f} eV/Å against the "
                           f"relaxation criterion of "
                           f"{float(force_criterion_ev_ang):g} eV/Å.  The "
                           f"frequencies are the curvature at this point, not "
                           f"at the minimum.  Relax first (an optimization "
                           f"calculation holding the same atoms, its final "
                           f"frame exported as a pair), or keep this run "
                           f"knowing that")
    relaxation = {"enabled": False, "already_relaxed": bool(already_relaxed),
                  "n_steps": 0, "max_force_eh_bohr": max_free_eh,
                  "max_force_all_atoms_eh_bohr": max_all_eh,
                  "converged": converged, "warning": warning}

    from ..sidecars.spectra import structure_hash_text
    methods = siesta_methods_text(
        displacement_bohr=(float(displacement_bohr) if displacement_bohr
                           is not None else fc.displacement_ang / 0.529177210903),
        n_free=len(free_s), n_held=len(held_s), n_rigid=n_rigid,
        siesta_version=engine_version)
    return SpectraResults(
        schema_version=SCHEMA_VERSION,
        engine="siesta",
        engine_version=str(engine_version),
        molbuilder_version=str(molbuilder_version),
        timestamp=(timestamp or datetime.now(timezone.utc).isoformat()
                   .replace("+00:00", "Z")),
        structure_hash=structure_hash_text(n, label, struct.elements,
                                           struct.positions),
        n_atoms_total=n,
        free_atom_idxs=free_atom_idxs,
        frozen_atom_idxs=frozen_atom_idxs,
        equilibrium_scf_eh=None,
        equilibrium_mo_energies_eh=None,
        equilibrium_homo_idx=None,
        modes=modes,
        selected_mode_idxs_1based=[],
        config=dict(config or {}),
        methods_text=methods,
        bibliography_keys=extract_citation_keys(methods),
        phase_frequencies=PHASE_COMPLETE,
        phase_raman=PHASE_COMPLETE,      # not requested; nothing owed
        phase_es=PHASE_COMPLETE,
        phase_relaxation=PHASE_COMPLETE,
        relaxation=relaxation,
        thermo=thermo,
        removed_motions={"count": n_rigid,
                         "patterns": [p.tolist() for p in patterns]},
        hessian_scope=("free" if held_s else "all"),
        n_atoms_in_hessian=len(free_s),
        hessian_density_fit=None,
        ir_route="none",
        raman_route="none",
        equilibrium_elements=list(struct.elements),
        equilibrium_positions_ang=np.asarray(struct.positions, dtype=float),
        engine_metadata={"fc_file": Path(fc.path).name,
                         "fc_displacement_ang": fc.displacement_ang,
                         "fc_range_1based": [free_s[0] + 1, free_s[-1] + 1],
                         "fc_asymmetry_max_ev_ang2": fc_asymmetry,
                         "reference_force_criterion_ev_ang": (
                             None if force_criterion_ev_ang is None
                             else float(force_criterion_ev_ang))},
    )


__all__ = ["spectra_results_from_fc", "siesta_methods_text",
           "DEFAULT_TEMPERATURE_K", "DEFAULT_PRESSURE_ATM"]
