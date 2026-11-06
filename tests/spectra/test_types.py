"""What a vibration's result will not hold: which atoms it is about, and what
its reader refuses (`engines/vibration.md` § 6.7).

THE SILENT FAILURES: a result whose free and held atoms do not partition the
structure, or a mode whose rows are not the free atoms, animates the wrong
atoms; a complex number cast to a float loses its imaginary part; a misspelled
key serves *not computed* with every number thrown away.

Each number coming back from its own key is asked of the file, in
``test_parsers_json.py``.  Runtime-cheap: no PySCF, no SCF anywhere.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder.spectra import (
    ModeData,
    ModeElectronicStructure,
    SpectraResults,
)
from molbuilder.spectra.results import SCHEMA_VERSION

from tests.spectra._helpers import _make_mode


# --------------------------------------------------------------------- #
#  ModeData                                                             #
# --------------------------------------------------------------------- #


class TestModeData:

    def test_imaginary_mode(self):
        """Imaginary modes carry a negative frequency and has_imag=True."""
        m = _make_mode(freq=-123.4)
        assert m.has_imag is True
        m2 = ModeData.from_dict(m.to_dict())
        assert m2.frequency_cm1 == pytest.approx(-123.4)
        assert m2.has_imag is True


# --------------------------------------------------------------------- #
#  SpectraResults                                                       #
# --------------------------------------------------------------------- #


class TestSpectraResults:

    def test_from_dict_rejects_missing_or_wrong_schema_version(self):
        """STRICT: the decoder self-enforces the schema version -- a
        missing or unreadable schema_version raises rather than
        reconstituting at whatever version the payload claims."""
        base = {
            "engine": "pyscf", "engine_version": "2.6.0",
            "molbuilder_version": "1.2.0", "timestamp": "t",
            "structure_hash": "h", "n_atoms_total": 1,
            "free_atom_idxs": [0], "frozen_atom_idxs": [],
            "equilibrium": {"scf_energy_eh": -1.0,
                            "mo_energies_eh": [-0.5, 0.5], "homo_idx": 0},
            "modes": [], "config": {},
        }
        with pytest.raises(ValueError, match="not supported"):
            SpectraResults.from_dict(dict(base))            # no version key
        with pytest.raises(ValueError, match="not supported"):
            SpectraResults.from_dict({**base, "schema_version": 999})


# --------------------------------------------------------------------- #
#  Type / shape normalisation in __post_init__                          #
# --------------------------------------------------------------------- #


class TestPostInitValidation:
    """Strict types from the start: every numpy field is coerced to
    dtype=float and shape-validated at construction.  Failures
    raise at the construction site, not 100 lines downstream when
    a caller assumes a float array but got ints.
    """

    # ----- ModeElectronicStructure -----

    def test_mo_arrays_must_match_size(self):
        """All three MO arrays must have the same window size."""
        with pytest.raises(ValueError, match="must share the same shape"):
            ModeElectronicStructure(
                amplitude_ang        = 0.1,
                mo_energies_eq_eh    = np.zeros(5),
                mo_energies_minus_eh = np.zeros(4),       # wrong size
                mo_energies_plus_eh  = np.zeros(5),
                homo_index_in_window = 0,
                scf_energy_eq_eh     = -1.0,
                scf_energy_minus_eh  = -1.0,
                scf_energy_plus_eh   = -1.0,
            )

    # ----- SpectraResults -----

    def test_free_fixed_partition_disjoint(self):
        """An atom can be in either free or fixed, never both."""
        with pytest.raises(ValueError, match="overlap"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                n_atoms_total              = 3,
                free_atom_idxs             = [0, 1, 2],
                frozen_atom_idxs            = [1, 2],            # overlaps with free
                equilibrium_scf_eh         = 0.0,
                equilibrium_mo_energies_eh = np.zeros(3),
                equilibrium_homo_idx       = 0,
                modes                      = [],
                selected_mode_idxs_1based  = [],
                config                     = {},
                methods_text               = "",
                bibliography_keys          = [],
            )

    def test_free_fixed_partition_complete(self):
        """Free + fixed must cover all atoms exactly once."""
        with pytest.raises(ValueError, match="partition"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                n_atoms_total              = 10,                # claim 10
                free_atom_idxs             = [0, 1],            # only 2 free
                frozen_atom_idxs            = [2, 3],            # only 2 fixed
                equilibrium_scf_eh         = 0.0,               # total = 4, not 10
                equilibrium_mo_energies_eh = np.zeros(3),
                equilibrium_homo_idx       = 0,
                modes                      = [],
                selected_mode_idxs_1based  = [],
                config                     = {},
                methods_text               = "",
                bibliography_keys          = [],
            )

    def test_cross_mode_eigenvector_shape_consistency(self):
        """Every mode's eigenvectors (canonical + display) must agree on n_free
        (= len(free_atom_idxs)).  Catches "I shipped a wrong-shape
        mode in the middle of the list" bugs."""
        good_mode = _make_mode()                                 # n_free=2
        # Construct a mode with n_free=3 -- should fail validation
        # at SpectraResults construction because the rest of the
        # results say n_free=2 via free_atom_idxs.
        bad_mode = ModeData(
            index_1based          = 99,
            frequency_cm1         = 50.0,
            raman_activity_a4_amu = 1.0,
            ir_intensity_km_mol   = None,
            eigenvector_canonical = np.zeros((3, 3)),  # 3 atoms, not 2
            eigenvector_display   = np.zeros((3, 3)),
            has_imag              = False,
        )
        with pytest.raises(ValueError, match=r"eigenvector shape.*expected"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                n_atoms_total              = 2,
                free_atom_idxs             = [0, 1],             # 2 free
                frozen_atom_idxs            = [],
                equilibrium_scf_eh         = 0.0,
                equilibrium_mo_energies_eh = np.zeros(3),
                equilibrium_homo_idx       = 0,
                modes                      = [good_mode, bad_mode],
                selected_mode_idxs_1based  = [],
                config                     = {},
                methods_text               = "",
                bibliography_keys          = [],
            )

    # ----- Complex-value rejection across all three dataclasses ----- #
    #
    # numpy's asarray(complex_input, dtype=float) silently discards the
    # imaginary part with only a warning -- a quiet correctness hole
    # for an all-real-valued wire format.  All three dataclasses must
    # reject complex inputs LOUDLY at __post_init__.

    def test_es_mo_array_rejects_complex(self):
        with pytest.raises(TypeError, match="complex"):
            ModeElectronicStructure(
                amplitude_ang        = 0.1,
                mo_energies_eq_eh    = np.array([1+0j, 2+0j, 3+0j]),
                mo_energies_minus_eh = np.zeros(3),
                mo_energies_plus_eh  = np.zeros(3),
                homo_index_in_window = 0,
                scf_energy_eq_eh     = -1.0,
                scf_energy_minus_eh  = -1.0,
                scf_energy_plus_eh   = -1.0,
            )

    def test_mode_eigenvector_rejects_complex(self):
        with pytest.raises(TypeError, match="complex"):
            ModeData(
                index_1based          = 1,
                frequency_cm1         = 100.0,
                raman_activity_a4_amu = None,
                ir_intensity_km_mol   = None,
                eigenvector_canonical = np.array([[1+0j, 0, 0],
                                                                       [-1+0j, 0, 0]]),
                eigenvector_display   = np.array([[1+0j, 0, 0],
                                                                       [-1+0j, 0, 0]]),
                has_imag              = False,
            )

    def test_equilibrium_mos_reject_complex(self):
        with pytest.raises(TypeError, match="complex"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                n_atoms_total              = 1,
                free_atom_idxs             = [0],
                frozen_atom_idxs            = [],
                equilibrium_scf_eh         = -1.0,
                equilibrium_mo_energies_eh = np.array([1+0j, 0, -1+0j]),
                equilibrium_homo_idx       = 1,
                modes                      = [],
                selected_mode_idxs_1based  = [],
                config                     = {},
                methods_text               = "",
                bibliography_keys          = [],
            )

    def test_zero_imaginary_part_complex_still_rejected(self):
        """Even when the imaginary part is exactly 0, we reject -- the
        dtype carries the type information regardless of value.  This
        means a user who computed a complex array and forgot to take
        ``.real`` gets a clear error instead of silent data loss."""
        with pytest.raises(TypeError, match="complex"):
            ModeData(
                index_1based          = 1,
                frequency_cm1         = 100.0,
                raman_activity_a4_amu = None,
                ir_intensity_km_mol   = None,
                eigenvector_canonical = np.zeros((2, 3), dtype=complex),
                eigenvector_display   = np.zeros((2, 3), dtype=complex),
                has_imag              = False,
            )

# --------------------------------------------------------------------- #
#  Phase status                                                         #
# --------------------------------------------------------------------- #


class TestPhaseStatus:
    """The per-phase status fields are the engine + UI's
    coordination point (`engines/vibration.md` § 4.9).  Each phase reading
    from its own key is asked of the file, in ``test_parsers_json.py``."""

    def test_invalid_phase_status_rejected(self):
        """Phase values are constrained to the controlled vocabulary;
        a stray string is a programmer error, not a forward-compat
        opportunity."""
        with pytest.raises(ValueError, match="is not a valid phase status"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                n_atoms_total              = 1,
                free_atom_idxs             = [0],
                frozen_atom_idxs            = [],
                equilibrium_scf_eh         = 0.0,
                equilibrium_mo_energies_eh = np.zeros(3),
                equilibrium_homo_idx       = 0,
                modes                      = [],
                selected_mode_idxs_1based  = [],
                config                     = {},
                methods_text               = "",
                bibliography_keys          = [],
                phase_frequencies          = "kindof",  # not in vocab
            )


def test_the_reader_refuses_a_key_it_does_not_know_by_name():
    """A misspelled key is refused, naming the key, at every block -- dropped,
    it serves a chart titled 'not computed' with every number thrown away;
    and a file claiming 1e12 atoms is refused without a range that size
    being built."""
    from molbuilder.spectra.results import SpectraResults
    from tests.spectra._helpers import _make_results
    good = _make_results().to_dict()
    SpectraResults.from_dict(good)                      # the control

    d = json.loads(json.dumps(good))
    d["modes"][0]["ir_intesity_km_mol"] = d["modes"][0].pop("ir_intensity_km_mol")
    with pytest.raises(ValueError, match="ir_intesity_km_mol"):
        SpectraResults.from_dict(d)

    d = json.loads(json.dumps(good))
    d["hessian_scop"] = "free"
    with pytest.raises(ValueError, match="hessian_scop"):
        SpectraResults.from_dict(d)

    d = json.loads(json.dumps(good))
    d["equilibrium"]["homo_index"] = 0
    with pytest.raises(ValueError, match="homo_index"):
        SpectraResults.from_dict(d)

    # The per-mode electronic-structure block too.
    d = json.loads(json.dumps(good))
    es = next(m for m in d["modes"] if m.get("electronic_structure"))
    es["electronic_structure"]["future_field"] = 1.0
    with pytest.raises(ValueError, match="future_field"):
        SpectraResults.from_dict(d)

    d = json.loads(json.dumps(good))
    d["n_atoms_total"] = 10 ** 12
    with pytest.raises(ValueError, match="partition range"):
        SpectraResults.from_dict(d)
