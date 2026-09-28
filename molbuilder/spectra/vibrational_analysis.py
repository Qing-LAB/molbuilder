"""A vibration's result from its second-derivative block -- the analysis, engine-neutral.

MODULE  spectra.vibrational_analysis (L2; numpy, the standard library and its
        siblings -- nothing that needs the package installed)
ROLE    the harmonic analysis's RESULT: a second-derivative block over the free
        atoms, the masses and the geometry with its held set and frame ->
        :class:`~molbuilder.spectra.results.SpectraResults`
USED-BY spectra/siesta_vibration.py (the SIESTA job's finish); any route that
        holds a block
TRAVELS in ``mb_vibration.pyz`` beside a SIESTA force-constant job
        (`runwrap.VIBRATION_COMPANIONS`)

The science is `engines/vibration.md` § 4.5 and `science/normal-modes.md`:
the free-free block of the TRUE second derivatives, mass-weighted and
diagonalised in the complement of the whole-body motions the held geometry
permits -- one harmonic path, `normal_modes.vibrational_modes` (R1-R4), which
the PySCF deck carries as source and every other route calls here.  What this
module adds is the rest of the ONE result both engines write (§ 6): the
wavenumbers and both eigenvector forms, the removed motions, the vibrational
thermochemistry, the stationarity verdict at the geometry the block belongs
to, and every per-atom row in the person's order.

WHAT A CALLER HANDS IN, AND IN WHICH ORDER.  Everything per-atom -- the
block, the masses, the positions, the elements, the held set, the reference
forces -- is in the order the block was computed in: a sorted copy's, when
the deck was written from one (held atoms first for a SIESTA force-constant
run, § 5.2).  ``permutation`` is that copy's record, and every row this
module writes goes back to the input order through it, once
(`atom_permutation.Permutation`, I7); without one the block's order IS the
input's.

WHAT A CALLER SAYS FOR ITS ROUTE: the engine and its version, the Methods
paragraph, the route's own ``engine_metadata`` and ``config``, and one
sentence on how the geometry was reached (``geometry_note``) -- a
force-constant run relaxes nothing, and its result says so.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import numpy as np

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from ..atom_permutation import Permutation
    from ..constants import (BOLTZMANN_HARTREE_K,
                             CM1_PER_SQRT_HARTREE_BOHR2_AMU,
                             HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_CM1)
    from ..sidecars.spectra import structure_hash_text
    from .methods import extract_citation_keys
    from .normal_modes import (THERMO_GRID_K, vibrational_modes,
                               vibrational_thermo, vibrational_thermo_grid)
    from .results import (PHASE_COMPLETE, PHASE_NOT_REQUESTED,
                          SCHEMA_VERSION, ModeData, SpectraResults)
except ImportError:                         # beside a job, in mb_vibration.pyz
    from atom_permutation import Permutation
    from constants import (BOLTZMANN_HARTREE_K,
                           CM1_PER_SQRT_HARTREE_BOHR2_AMU,
                           HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_CM1)
    from spectra_sidecar import structure_hash_text
    from methods import extract_citation_keys
    from normal_modes import (THERMO_GRID_K, vibrational_modes,
                              vibrational_thermo, vibrational_thermo_grid)
    from results import (PHASE_COMPLETE, PHASE_NOT_REQUESTED,
                         SCHEMA_VERSION, ModeData, SpectraResults)

#: The temperature and pressure the thermochemistry is summed at when the
#: route states none -- the headline PySCF's items default to
#: (`engines/vibration.md` § 5.5: reachable on SIESTA is owed).
DEFAULT_TEMPERATURE_K = 298.15
DEFAULT_PRESSURE_ATM = 1.0


def vibrational_analysis(hessian, masses_amu: Sequence[float],
                         positions_ang, elements: Sequence[str],
                         held: Sequence[int], *,
                         axis_kind: Sequence[str],
                         cell=None,
                         permutation: Optional[Permutation] = None,
                         label: str,
                         engine: str,
                         engine_version: str = "",
                         molbuilder_version: str = "",
                         methods_text: str = "",
                         geometry_note: str = "",
                         reference_forces_ev_ang=None,
                         force_criterion_ev_ang: Optional[float] = None,
                         already_relaxed: bool = False,
                         ladder_relaxation: Optional[Mapping[str, Any]] = None,
                         temperature_K: float = DEFAULT_TEMPERATURE_K,
                         pressure_atm: float = DEFAULT_PRESSURE_ATM,
                         config: Optional[Mapping[str, Any]] = None,
                         engine_metadata: Optional[Mapping[str, Any]] = None,
                         timestamp: Optional[str] = None) -> SpectraResults:
    """The result of a harmonic analysis over ``hessian``.

    ``hessian`` is the second-derivative table, shape ``(n, n, 3, 3)`` in
    Hartree/Bohr², filled at least over the free atoms (the one path slices
    that block); ``masses_amu``, ``positions_ang`` (the geometry the block was
    taken at), ``elements`` and ``held`` (0-based) are per atom, all in the
    block's order.  ``axis_kind`` and ``cell`` say which whole-body turns
    survive (R3).

    ``reference_forces_ev_ang`` are the forces at that geometry, in the same
    order; with ``force_criterion_ev_ang`` -- the description's own
    `relax_force_tol` -- stationarity is judged as the largest absolute
    Cartesian component over the free atoms (R5, `engines/vibration.md`
    § 5.5).  ``already_relaxed`` is the person's statement, carried as made.
    ``ladder_relaxation`` is the relaxation record of the stage that relaxed
    first (`parse.contract.relaxation_of_output`): the result then says the
    relaxation ran, how many steps it took, and ``phase_relaxation`` is
    complete; without it the phase is `not requested`.

    Raises ``ValueError`` naming the mismatch -- every atom held, a
    permutation or a force table that does not fit the atoms.
    """
    elements = [str(e) for e in elements]
    n = len(elements)
    R = np.asarray(positions_ang, dtype=float).reshape(-1, 3)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    if R.shape[0] != n or masses.shape[0] != n:
        raise ValueError(
            f"{n} elements, {R.shape[0]} positions and {masses.shape[0]} "
            f"masses: every per-atom input describes the same atoms")
    if permutation is None:
        permutation = Permutation(tuple(range(n)), tuple(range(n)))
    if permutation.n_atoms != n:
        raise ValueError(
            f"the recorded permutation describes {permutation.n_atoms} atoms; "
            f"the block has {n}")
    held_b = sorted({int(i) for i in held})
    free_b = [i for i in range(n) if i not in set(held_b)]
    if not free_b:
        raise ValueError("every atom is held: there is nothing to vibrate")

    lam, L_b, patterns_b = vibrational_modes(
        hessian, masses, R, held_b, tuple(axis_kind), cell=cell)
    omega = np.sign(lam) * np.sqrt(np.abs(lam))
    freqs_cm1 = omega * CM1_PER_SQRT_HARTREE_BOHR2_AMU

    # BACK TO THE INPUT ORDER, through the record's one inversion (I7): every
    # per-atom row this result carries follows the person's numbering.
    L, free_atom_idxs = permutation.rows_to_input_order(L_b, free_b)
    patterns, _ = permutation.rows_to_input_order(patterns_b, free_b)
    frozen_atom_idxs = sorted(permutation.original_of(held_b))
    everyone = list(range(n))
    positions_in, _ = permutation.rows_to_input_order(R, everyone)
    elements_in, _ = permutation.rows_to_input_order(
        np.asarray(elements, dtype=object), everyone)
    elements_in = [str(e) for e in elements_in]

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
    zpe, _u0, _s0 = vibrational_thermo(kept, temperature_K,
                                       BOLTZMANN_HARTREE_K, eh_per_cm1)
    # No total energy is reported by a route that hands in only a block, so
    # the curves are the vibrational contributions above the electronic
    # minimum (e_ref = 0), through the one grid home both writers share; the
    # headline temperature is on the grid, so the headline IS a grid row.
    temps = sorted(set(THERMO_GRID_K) | {float(temperature_K)})
    grid = vibrational_thermo_grid(kept, temps, 0.0, BOLTZMANN_HARTREE_K,
                                   eh_per_cm1)
    k_head = grid["temperatures_K"].index(float(temperature_K))
    n_rigid = int(len(patterns))
    thermo = {
        "regime": "vibrational-only",
        "temperature_K": float(temperature_K),
        "pressure_atm": float(pressure_atm),
        "zpe_eh": zpe,
        "h_eh": grid["h_eh"][k_head],
        "s_eh_k": grid["s_eh_k"][k_head],
        "g_eh": grid["g_eh"][k_head],
        "n_modes": len(modes),
        "n_imag_excluded": n_imag,
        "n_rigid_removed": n_rigid,
        "note": ("VIBRATIONAL contributions only, above the electronic "
                 "minimum, the headline and the grid alike: this route "
                 "reports no total energy and has no gas-phase "
                 "translational or rotational partition function to add; "
                 "the whole-body motions of the free atoms "
                 "(n_rigid_removed) were removed before diagonalising"),
        "grid": grid,
    }

    relaxation = _stationarity(reference_forces_ev_ang, n, free_b,
                               force_criterion_ev_ang, geometry_note,
                               already_relaxed, ladder_relaxation)
    return SpectraResults(
        schema_version=SCHEMA_VERSION,
        engine=str(engine),
        engine_version=str(engine_version),
        molbuilder_version=str(molbuilder_version),
        timestamp=(timestamp or datetime.now(timezone.utc).isoformat()
                   .replace("+00:00", "Z")),
        structure_hash=structure_hash_text(n, label, elements_in,
                                           positions_in),
        n_atoms_total=n,
        free_atom_idxs=free_atom_idxs,
        frozen_atom_idxs=frozen_atom_idxs,
        equilibrium_scf_eh=None,
        equilibrium_mo_energies_eh=None,
        equilibrium_homo_idx=None,
        modes=modes,
        selected_mode_idxs_1based=[],
        config=dict(config or {}),
        methods_text=methods_text,
        bibliography_keys=extract_citation_keys(methods_text),
        # The frequencies are this analysis's whole answer; the strengths
        # and the probe were never asked of it, and the flags say so
        # (`engines/vibration.md` § 4.9) rather than reporting them done.
        # The relaxation was asked of the ladder when the box was unticked --
        # the relaxing stage's record is what says so here.
        phase_frequencies=PHASE_COMPLETE,
        phase_raman=PHASE_NOT_REQUESTED,
        phase_es=PHASE_NOT_REQUESTED,
        phase_relaxation=(PHASE_COMPLETE if ladder_relaxation is not None
                          else PHASE_NOT_REQUESTED),
        relaxation=relaxation,
        thermo=thermo,
        removed_motions={"count": n_rigid,
                         "patterns": [p.tolist() for p in patterns]},
        hessian_scope=("free" if held_b else "all"),
        n_atoms_in_hessian=len(free_b),
        hessian_density_fit=None,
        ir_route="none",
        raman_route="none",
        equilibrium_elements=elements_in,
        equilibrium_positions_ang=np.asarray(positions_in, dtype=float),
        engine_metadata=dict(engine_metadata or {}),
    )


def _stationarity(reference_forces_ev_ang, n: int, free: Sequence[int],
                  criterion_ev_ang: Optional[float], geometry_note: str,
                  already_relaxed: bool,
                  ladder_relaxation: Optional[Mapping[str, Any]]) -> dict:
    """The result's ``relaxation`` block: the forces at the block's geometry
    judged against the criterion (R5 -- the largest absolute Cartesian
    COMPONENT over the free atoms, the convention the PySCF deck judges its
    own by, `engines/vibration.md` § 4.3), the person's statement as made,
    and the relaxing stage's step count when one ran first."""
    warning = geometry_note or ("the second derivatives are taken at the "
                                "geometry given, which is taken as the "
                                "stationary point")
    max_free_eh = max_all_eh = None
    converged = None
    if reference_forces_ev_ang is not None:
        f_ref = np.asarray(reference_forces_ev_ang, dtype=float)
        if f_ref.shape != (n, 3):
            raise ValueError(
                f"the reference forces describe "
                f"{f_ref.shape[0] if f_ref.ndim else '?'} atoms; the "
                f"structure has {n}")
        max_free_ev = float(np.max(np.abs(f_ref[list(free)])))
        max_all_ev = float(np.max(np.abs(f_ref)))
        max_free_eh = max_free_ev / HARTREE_BOHR_EV_ANGSTROM_ASE
        max_all_eh = max_all_ev / HARTREE_BOHR_EV_ANGSTROM_ASE
        if criterion_ev_ang is not None:
            converged = bool(max_free_ev <= float(criterion_ev_ang))
            if converged:
                warning += (f".  The forces at the reference geometry were "
                            f"read back: the largest on the free atoms is "
                            f"{max_free_ev:.4f} eV/Å, within this "
                            f"calculation's relaxation tolerance "
                            f"(relax_force_tol) of "
                            f"{float(criterion_ev_ang):g} eV/Å")
            else:
                warning = (f"the reference geometry is not a stationary point "
                           f"at this level of theory: the largest force on the "
                           f"free atoms is {max_free_ev:.4f} eV/Å against this "
                           f"calculation's relaxation tolerance "
                           f"(relax_force_tol) of "
                           f"{float(criterion_ev_ang):g} eV/Å.  The "
                           f"frequencies are the curvature at this point, not "
                           f"at the minimum, and will be off.  Relax first -- "
                           f"untick `already_relaxed` so the calculation "
                           f"relaxes first, or relax elsewhere at this level of "
                           f"theory and hand the result over -- or keep this "
                           f"run knowing that")
    ladder = ladder_relaxation if isinstance(ladder_relaxation, Mapping) else None
    return {"enabled": ladder is not None,
            "already_relaxed": bool(already_relaxed),
            "n_steps": (int(ladder.get("n_steps") or 0) if ladder else 0),
            "max_force_eh_bohr": max_free_eh,
            "max_force_all_atoms_eh_bohr": max_all_eh,
            "converged": converged, "warning": warning}


__all__ = ["vibrational_analysis", "DEFAULT_TEMPERATURE_K",
           "DEFAULT_PRESSURE_ATM"]
