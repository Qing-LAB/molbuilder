"""Shared helpers for the spectra test suite.

These build cheap, in-memory L1 fixtures (no PySCF / SCF anywhere) so
tests across types / config / selection / methods / engine / script
emission can construct realistic-shaped ``SpectraResults`` /
``ModeData`` / ``Structure`` instances without each file re-defining
its own builders.

Plain module — import directly:

    from tests.spectra._helpers import (
        _make_es, _make_mode, _make_results,
        _modes_fixture, _struct_water,
    )

(Not a pytest conftest fixture because the call sites use these as
factory functions with optional arguments — fixture indirection would
add noise without value.)
"""

from __future__ import annotations

import numpy as np

from molbuilder.spectra import ModeData, ModeElectronicStructure, SpectraResults
from molbuilder.spectra.results import PHASE_COMPLETE, PHASE_EMPTY, SCHEMA_VERSION
from molbuilder.structure import Structure


def _make_es(amplitude: float = 0.1) -> ModeElectronicStructure:
    return ModeElectronicStructure(
        amplitude_ang        = amplitude,
        mo_energies_eq_eh    = np.array([-1.0, -0.5, -0.2, 0.1, 0.3]),
        mo_energies_minus_eh = np.array([-1.01, -0.51, -0.21, 0.09, 0.29]),
        mo_energies_plus_eh  = np.array([-0.99, -0.49, -0.19, 0.11, 0.31]),
        homo_index_in_window = 2,
        scf_energy_eq_eh     = -76.4123,
        scf_energy_minus_eh  = -76.4115,
        scf_energy_plus_eh   = -76.4119,
    )


def _make_mode(index: int = 1,
               freq: float = 412.3,
               with_es: bool = False,
               raman: float = 12.5) -> ModeData:
    # Tests don't exercise the canonical-vs-display distinction (no
    # Placzek formula evaluation here), so we feed the same toy array
    # into both eigenvector slots.  In a real run from
    # render_spectra_script these would differ by a per-mode scaling.
    _ev = np.array([[0.7, 0.0, 0.0],
                    [-0.7, 0.0, 0.0]])
    return ModeData(
        index_1based          = index,
        frequency_cm1         = freq,
        raman_activity_a4_amu = raman,
        ir_intensity_km_mol   = None,
        eigenvector_canonical = _ev,
        eigenvector_display   = _ev,
        has_imag              = (freq < 0),
        electronic_structure  = _make_es() if with_es else None,
    )


def _make_results(complete: bool = True) -> SpectraResults:
    """2-atom molecule, three vibrational modes; the middle mode has
    electronic-structure data populated, the others don't.  Mirrors
    the structural shape of a real PySCF run on H2O (toy numerics).

    ``complete`` controls the phase_* fields:
      * True  -> all three layers PHASE_COMPLETE (full run done).
      * False -> phase_frequencies done, phase_raman + phase_es
                 empty (a "L2 only" intermediate state).
    """
    phases = ((PHASE_COMPLETE, PHASE_COMPLETE, PHASE_COMPLETE) if complete
              else (PHASE_COMPLETE, PHASE_EMPTY, PHASE_EMPTY))
    return SpectraResults(
        schema_version             = SCHEMA_VERSION,
        engine                     = "pyscf",
        engine_version             = "2.6.0",
        molbuilder_version         = "1.2.0",
        timestamp                  = "2026-05-11T12:00:00Z",
        structure_hash             = "sha256:abc123",
        n_atoms_total              = 2,
        free_atom_idxs             = [0, 1],
        fixed_atom_idxs            = [],
        equilibrium_scf_eh         = -76.4123,
        equilibrium_mo_energies_eh = np.array([-1.0, -0.5, -0.2, 0.1, 0.3]),
        equilibrium_homo_idx       = 2,
        modes                      = [
            _make_mode(index=1, freq=412.3, with_es=False),
            _make_mode(index=2, freq=1023.4, with_es=True),
            _make_mode(index=3, freq=3656.0, with_es=False),
        ],
        selected_mode_idxs_1based  = [2],
        config                     = {"engine": "pyscf", "compute_raman": True},
        methods_text               = "Harmonic vibrational analysis ...",
        bibliography_keys          = ["Sun2020", "Becke1993"],
        phase_frequencies          = phases[0],
        phase_raman                = phases[1],
        phase_es                   = phases[2],
    )


def _modes_fixture() -> list[ModeData]:
    """6-mode fixture with varied frequencies + Raman activities for
    exercising top_n / threshold / window / explicit selectors.

      idx  freq (cm⁻¹)  raman_activity_a4_amu
      ---  -----------  ---------------------
        1       412.3                  3.2
        2       745.0                  12.5    <- bright
        3      1023.4                  87.2    <- brightest
        4      1612.0                  None    (raman not computed)
        5      2956.0                  45.0    <- bright, C-H stretch
        6      3656.0                  18.5    (high-freq O-H)
    """
    def _m(idx, freq, raman):
        _ev = np.zeros((2, 3))
        return ModeData(
            index_1based          = idx,
            frequency_cm1         = freq,
            raman_activity_a4_amu = raman,
            ir_intensity_km_mol   = None,
            eigenvector_canonical = _ev,
            eigenvector_display   = _ev,
            has_imag              = False,
        )
    return [
        _m(1,  412.3,  3.2),
        _m(2,  745.0, 12.5),
        _m(3, 1023.4, 87.2),
        _m(4, 1612.0, None),
        _m(5, 2956.0, 45.0),
        _m(6, 3656.0, 18.5),
    ]


def _struct_water() -> Structure:
    """Cheap real Structure for water -- used by engine preflight +
    script-template tests that need a Structure (not just a config)."""
    return Structure(
        elements  = ["O", "H", "H"],
        positions = np.array([[0., 0., 0.],
                              [0.96, 0., 0.],
                              [-0.24, 0.93, 0.]]),
    )
