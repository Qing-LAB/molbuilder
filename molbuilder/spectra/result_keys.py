"""THE VIBRATION RESULT'S KEYS, each block's -- the rows of
`engines/vibration.md` § 6.2-6.4, and nothing else.

MODULE  spectra.result_keys (L2; it imports three words -- the atom index
        map's key, the derived mode keys and the modes' thermal block's key
        -- and nothing else, two ways, for it travels beside the job in both
        bundles, `runwrap.members`)
ROLE    the keys of each block the reader reads key by key
        (:data:`RESULT_KEYS`, :data:`EQUILIBRIUM_KEYS`, :data:`MODE_KEYS`,
        :data:`ES_KEYS`, :data:`PARTIAL_CHARGES_KEYS`,
        :data:`CHARGE_STEP_KEYS`, :data:`OPTICAL_KEYS`,
        :data:`OPTICAL_MODE_KEYS`), and the four the data file lays out key
        by key (:data:`WRITTEN_KEYS`); the
        `relaxation`, `thermo` and `modes_thermal` blocks are their
        builders' (`results.relaxation_block`, `normal_modes.thermo_block`,
        `derived.thermal_block`), which the data file's tables state
USED-BY the result's reader (`spectra.results`), which refuses a key
        outside them BY NAME; the data file (`spectra.datafile`), which
        states each key a writer states and checks so as it loads

A format of ours keeps its words in one module (`model/parse.md` § 4a): a key
that starts being written is a row in § 6.2-6.4 first, then here -- and then a
row of the data file, which refuses to load without one.  Its own imports
load nothing of the result's (`results`, which loads the parse package for
the SIESTA grammar), so the data file can ask it as it loads, whichever
module loads first, half-loaded `results` or not.
"""

from __future__ import annotations

# TWO WAYS, because this module travels beside the job in both bundles.
try:                                        # inside molbuilder
    from ..atom_permutation import MAP_KEY
    from .derived import DERIVED_MODE_KEYS, THERMAL_KEY
except ImportError:                         # beside a job, in a bundle
    from atom_permutation import MAP_KEY  # type: ignore[no-redef]
    from derived import DERIVED_MODE_KEYS, THERMAL_KEY  # type: ignore[no-redef]


ES_KEYS = frozenset({
    "amplitude_ang", "mo_energies_eq_eh", "mo_energies_minus_eh",
    "mo_energies_plus_eh", "homo_index_in_window", "scf_energy_eq_eh",
    "scf_energy_minus_eh", "scf_energy_plus_eh",
})
MODE_KEYS = frozenset({
    "index_1based", "frequency_cm1", "raman_activity_a4_amu",
    "ir_intensity_km_mol", "eigenvector_canonical", "eigenvector_display",
    "has_imag", "electronic_structure", "dipole_derivative_d_per_ang_amu12",
    # DERIVED (`spectra.derived`, `engines/vibration.md` § 6.6) -- stated by
    # every writer, checked on read.
    *DERIVED_MODE_KEYS,
})
EQUILIBRIUM_KEYS = frozenset({
    "scf_energy_eh", "mo_energies_eh", "homo_idx", "masses_amu",
})
RESULT_KEYS = frozenset({
    "schema_version", "engine", "engine_version", "molbuilder_version",
    "timestamp", "n_atoms_total", "free_atom_idxs",
    "frozen_atom_idxs", "structure", "equilibrium", "modes",
    "selected_mode_idxs_1based",
    "config", "methods_text", "bibliography_keys", "phase_frequencies",
    "phase_relaxation", "relaxation", "thermo", "removed_motions",
    "hessian_scope", "n_atoms_in_hessian", "hessian_density_fit", "ir_route",
    "ir_fd_step_ang", "raman_route", "raman_fd_step_ang", "phase_raman",
    "phase_ir", "phase_es", "engine_metadata", "runtime_info",
    MAP_KEY, "partial_charges", "optical",
    # DERIVED: each mode's spread and mean quanta at the modes' own
    # temperatures (`engines/vibration.md` § 6.2).
    THERMAL_KEY,
})
#: The force-constant run's charges (v11, `engines/vibration.md` § 6.2)...
PARTIAL_CHARGES_KEYS = frozenset({"displacement_ang", "steps", "charges_e"})
#: ...and each of its steps'.
CHARGE_STEP_KEYS = frozenset({"atom", "axis", "side"})
#: The optical citation's keys (`engines/vibration.md` § 6.2).
OPTICAL_KEYS = frozenset({"data_file", "data_sha256", "direction",
                          "displacement_ang", "selection", "reference",
                          "modes"})
#: ...and each of its mode rows'.
OPTICAL_MODE_KEYS = frozenset({"mode", "frequency_cm1", "dq_amu12_ang",
                               "plus", "minus"})

#: WHAT A WRITER STATES OF EACH BLOCK -- every key the reader takes.  The
#: data file states each, and checks so as it loads (`spectra.datafile`), so
#: a key a writer starts stating is a row there too, never dropped from the
#: ``.nc``.
WRITTEN_KEYS = {
    "result": RESULT_KEYS,
    "equilibrium": EQUILIBRIUM_KEYS,
    "mode": MODE_KEYS,
    "electronic_structure": ES_KEYS,
}


__all__ = ["ES_KEYS", "MODE_KEYS", "EQUILIBRIUM_KEYS", "RESULT_KEYS",
           "PARTIAL_CHARGES_KEYS", "CHARGE_STEP_KEYS", "OPTICAL_KEYS",
           "OPTICAL_MODE_KEYS", "WRITTEN_KEYS"]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.atom_permutation",
                "molbuilder.spectra.derived")
