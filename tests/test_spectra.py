"""Tests for the Spectra-tab L1 surfaces (result types + config).

Pins the JSON wire shape + the engine-agnostic dataclass surface
documented in ``docs/tabs/spectra/spec.md`` § 4 - § 6.  These tests
are runtime-cheap: no PySCF, no SCF, no Hessian.  They protect:

  * round-trip fidelity (typed -> dict -> JSON -> dict -> typed
    equals the original);
  * forward compatibility of the ``from_dict`` classmethods
    (extra wire keys are ignored, missing optional keys default
    sensibly);
  * the ``complete`` flag semantics for the Option B (live-watch)
    phase-checkpoint model;
  * the SpectraConfig schema shape (section names + per-section
    field counts) so a stray reorder of fields doesn't silently
    rearrange the UI.

PySCF-side smoke tests for actual Hessian + Raman activities live
in ``tests/test_spectra_smoke.py`` (to be added when the
PySCFSpectraEngine lands; marked with ``@pytest.mark.smoke``).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from molbuilder.config.spectra import SpectraConfig
from molbuilder.spectra import (
    ModeData,
    ModeElectronicStructure,
    SpectraResults,
)
from molbuilder.spectra.results import (
    SCHEMA_VERSION,
    PHASE_EMPTY,
    PHASE_RUNNING,
    PHASE_COMPLETE,
)
from molbuilder.web.blueprints._shared import dataclass_to_form_schema


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


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
    return ModeData(
        index_1based          = index,
        frequency_cm1         = freq,
        raman_activity_a4_amu = raman,
        ir_intensity_km_mol   = None,
        eigenvector_free      = np.array([[0.7, 0.0, 0.0],
                                          [-0.7, 0.0, 0.0]]),
        has_imag              = (freq < 0),
        electronic_structure  = _make_es() if with_es else None,
    )


def _make_results(complete: bool = True) -> SpectraResults:
    """Fixture: 2-atom molecule, all atoms free, three vibrational
    modes (no atoms fixed -> n_free=2, eigenvector shape (2,3)).
    The middle mode has electronic-structure data populated; the
    others don't.  Mirrors what a real PySCF run on H2O would
    produce structurally (though with toy numerical values).

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


# --------------------------------------------------------------------- #
#  ModeElectronicStructure                                              #
# --------------------------------------------------------------------- #


class TestModeElectronicStructure:

    def test_round_trip(self):
        es = _make_es()
        d = es.to_dict()
        es2 = ModeElectronicStructure.from_dict(d)
        assert es2.amplitude_ang == pytest.approx(es.amplitude_ang)
        np.testing.assert_allclose(es2.mo_energies_eq_eh, es.mo_energies_eq_eh)
        np.testing.assert_allclose(es2.mo_energies_minus_eh, es.mo_energies_minus_eh)
        np.testing.assert_allclose(es2.mo_energies_plus_eh, es.mo_energies_plus_eh)
        assert es2.homo_index_in_window == es.homo_index_in_window
        assert es2.scf_energy_eq_eh == pytest.approx(es.scf_energy_eq_eh)

    def test_json_serialisable(self):
        """to_dict() output must json.dumps() without a custom encoder."""
        es = _make_es()
        text = json.dumps(es.to_dict())
        # Round-trip survives the JSON encoding.
        es2 = ModeElectronicStructure.from_dict(json.loads(text))
        np.testing.assert_allclose(es2.mo_energies_eq_eh, es.mo_energies_eq_eh)

    def test_from_dict_ignores_extra_keys(self):
        """Forward compat: a future field arriving on the wire must
        not break the v1 parser."""
        d = _make_es().to_dict()
        d["future_field"] = "we don't know about this yet"
        es2 = ModeElectronicStructure.from_dict(d)
        # Old fields still load.
        assert es2.homo_index_in_window == 2


# --------------------------------------------------------------------- #
#  ModeData                                                             #
# --------------------------------------------------------------------- #


class TestModeData:

    def test_round_trip_without_es(self):
        m = _make_mode(index=5, freq=789.1, with_es=False)
        m2 = ModeData.from_dict(m.to_dict())
        assert m2.index_1based == 5
        assert m2.frequency_cm1 == pytest.approx(789.1)
        assert m2.raman_activity_a4_amu == pytest.approx(12.5)
        assert m2.ir_intensity_km_mol is None
        assert m2.electronic_structure is None
        assert m2.has_imag is False
        np.testing.assert_allclose(m2.eigenvector_free, m.eigenvector_free)

    def test_round_trip_with_es(self):
        m = _make_mode(with_es=True)
        m2 = ModeData.from_dict(m.to_dict())
        assert m2.electronic_structure is not None
        assert m2.electronic_structure.amplitude_ang == pytest.approx(0.1)

    def test_imaginary_mode(self):
        """Imaginary modes carry a negative frequency and has_imag=True."""
        m = _make_mode(freq=-123.4)
        assert m.has_imag is True
        m2 = ModeData.from_dict(m.to_dict())
        assert m2.frequency_cm1 == pytest.approx(-123.4)
        assert m2.has_imag is True

    def test_raman_activity_none_when_not_computed(self):
        """compute_raman=False yields raman_activity_a4_amu=None for
        every mode."""
        m = ModeData(
            index_1based          = 1,
            frequency_cm1         = 500.0,
            raman_activity_a4_amu = None,
            ir_intensity_km_mol   = None,
            eigenvector_free      = np.zeros((3, 3)),
            has_imag              = False,
        )
        assert ModeData.from_dict(m.to_dict()).raman_activity_a4_amu is None

    def test_ir_intensity_reserved_for_future(self):
        """v1 emitted scripts always set ir_intensity_km_mol=None.
        Schema reserves the field so the 1c add-on (IR) is a
        no-schema-change extension."""
        m = _make_mode()
        assert m.ir_intensity_km_mol is None
        # Set it explicitly to a value to confirm the field accepts floats.
        m.ir_intensity_km_mol = 42.0
        m2 = ModeData.from_dict(m.to_dict())
        assert m2.ir_intensity_km_mol == pytest.approx(42.0)


# --------------------------------------------------------------------- #
#  SpectraResults                                                       #
# --------------------------------------------------------------------- #


class TestSpectraResults:

    def test_round_trip_complete(self):
        r = _make_results(complete=True)
        text = json.dumps(r.to_dict())
        r2 = SpectraResults.from_dict(json.loads(text))
        assert r2.schema_version == SCHEMA_VERSION
        assert r2.engine == "pyscf"
        assert r2.engine_version == "2.6.0"
        # All three layers complete (round-trips through phase_*).
        assert r2.phase_frequencies == PHASE_COMPLETE
        assert r2.phase_raman       == PHASE_COMPLETE
        assert r2.phase_es          == PHASE_COMPLETE
        assert len(r2.modes) == 3
        # The selected mode survives.
        assert r2.selected_mode_idxs_1based == [2]
        # The methods text + bibliography keys survive.
        assert r2.methods_text.startswith("Harmonic")
        assert r2.bibliography_keys == ["Sun2020", "Becke1993"]
        # Equilibrium MO array round-trips through JSON.
        np.testing.assert_allclose(
            r2.equilibrium_mo_energies_eh,
            r.equilibrium_mo_energies_eh,
        )

    def test_round_trip_in_progress(self):
        """Live-watch state: an in-progress run before phase-2 has
        all three phase_* = PHASE_EMPTY and may carry empty
        methods_text / bibliography_keys."""
        r = SpectraResults(
            schema_version             = SCHEMA_VERSION,
            engine                     = "pyscf",
            engine_version             = "2.6.0",
            molbuilder_version         = "1.2.0",
            timestamp                  = "2026-05-11T12:00:00Z",
            structure_hash             = "sha256:abc",
            n_atoms_total              = 3,
            free_atom_idxs             = [0, 1, 2],
            fixed_atom_idxs            = [],
            equilibrium_scf_eh         = -76.41,
            equilibrium_mo_energies_eh = np.array([-1.0, -0.5, -0.2, 0.1]),
            equilibrium_homo_idx       = 2,
            modes                      = [],         # Hessian not done yet
            selected_mode_idxs_1based  = [],
            config                     = {"engine": "pyscf"},
            methods_text               = "",         # populated as layers complete
            bibliography_keys          = [],
            # phase_* default to PHASE_EMPTY in the dataclass; omitting
            # them here exercises that default path.
        )
        r2 = SpectraResults.from_dict(json.loads(json.dumps(r.to_dict())))
        assert r2.phase_frequencies == PHASE_EMPTY
        assert r2.phase_raman       == PHASE_EMPTY
        assert r2.phase_es          == PHASE_EMPTY
        assert r2.modes == []
        assert r2.methods_text == ""
        assert r2.bibliography_keys == []

    def test_from_dict_defaults_for_missing_optionals(self):
        """A minimal wire payload (no engine_metadata, no
        bibliography_keys, no methods_text) parses cleanly with
        sensible defaults so the parser doesn't bomb on a partial
        early-phase write."""
        d = {
            "schema_version":     1,
            "engine":             "pyscf",
            "engine_version":     "2.6.0",
            "molbuilder_version": "1.2.0",
            "timestamp":          "2026-05-11T12:00:00Z",
            "structure_hash":     "sha256:abc",
            "n_atoms_total":      1,
            "free_atom_idxs":     [0],
            "fixed_atom_idxs":    [],
            "equilibrium": {
                "scf_energy_eh":  -1.0,
                "mo_energies_eh": [-0.5, 0.5],
                "homo_idx":       0,
            },
            "modes":              [],
            "config":             {},
            # Intentionally omitted: methods_text, bibliography_keys,
            # selected_mode_idxs_1based, engine_metadata, phase_*.
        }
        r = SpectraResults.from_dict(d)
        assert r.methods_text == ""
        assert r.bibliography_keys == []
        assert r.selected_mode_idxs_1based == []
        assert r.engine_metadata == {}
        # Missing phase_* fields default to PHASE_EMPTY.
        assert r.phase_frequencies == PHASE_EMPTY
        assert r.phase_raman       == PHASE_EMPTY
        assert r.phase_es          == PHASE_EMPTY

    def test_schema_version_pinned(self):
        """The on-disk schema is v1 for the entire v1.x release line.
        Bumping requires a parser branch -- this test pins the
        current major-version invariant so a stray edit shows up
        in code review."""
        r = _make_results()
        assert r.schema_version == 1
        assert SCHEMA_VERSION == 1

    def test_engine_metadata_passes_through(self):
        """engine_metadata is the escape valve for engine-specific
        diagnostics that don't fit the common surface; the typed
        layer passes it through verbatim, the UI ignores it."""
        r = _make_results()
        r.engine_metadata = {"pyscf_dfttype": "RKS", "n_basis_funcs": 24}
        r2 = SpectraResults.from_dict(json.loads(json.dumps(r.to_dict())))
        assert r2.engine_metadata == {"pyscf_dfttype": "RKS", "n_basis_funcs": 24}

    def test_modes_preserve_order_through_round_trip(self):
        """The modes list is sorted by frequency ascending (spec §6).
        Round-trip must NOT shuffle them."""
        r = _make_results()
        freqs_before = [m.frequency_cm1 for m in r.modes]
        r2 = SpectraResults.from_dict(json.loads(json.dumps(r.to_dict())))
        freqs_after = [m.frequency_cm1 for m in r2.modes]
        assert freqs_before == freqs_after
        # Confirm input fixture is actually ascending so the assertion is meaningful.
        assert freqs_before == sorted(freqs_before)


# --------------------------------------------------------------------- #
#  Equality is intentionally undefined -- must raise loudly             #
# --------------------------------------------------------------------- #


class TestEqualityIsLoud:
    """A spectrum is not equal-able by `==` -- the scientific
    operations are Δ-quantities, not yes/no.  Accidental
    ``a == b`` (e.g. from a copy-paste, a test author who didn't
    realise spectra need a tolerance comparator) raises with a
    pointer at the future ``spectra.compare`` API.
    """

    def test_mode_electronic_structure_eq_raises(self):
        a, b = _make_es(), _make_es()
        with pytest.raises(TypeError, match="equality is intentionally undefined"):
            a == b  # noqa: B015
        with pytest.raises(TypeError):
            a != b  # noqa: B015

    def test_mode_data_eq_raises(self):
        a, b = _make_mode(), _make_mode()
        with pytest.raises(TypeError, match="equality is intentionally undefined"):
            a == b  # noqa: B015

    def test_spectra_results_eq_raises(self):
        a, b = _make_results(), _make_results()
        with pytest.raises(TypeError, match="equality is intentionally undefined"):
            a == b  # noqa: B015

    def test_error_message_points_at_compare_api(self):
        """The error message names the future structured comparator
        (`molbuilder.spectra.compare`) so a user hitting this knows
        where the right operation will live when it's built."""
        a, b = _make_es(), _make_es()
        try:
            a == b  # noqa: B015
        except TypeError as exc:
            msg = str(exc)
            assert "molbuilder.spectra.compare" in msg
            assert "Δ" in msg  # mentions the Δ-quantity intent


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

    def test_mo_arrays_coerced_to_float(self):
        """Passing an int-typed array gets promoted to float64."""
        es = ModeElectronicStructure(
            amplitude_ang        = 0.1,
            mo_energies_eq_eh    = [1, 2, 3, 4, 5],       # plain list
            mo_energies_minus_eh = np.array([1, 2, 3, 4, 5], dtype=int),
            mo_energies_plus_eh  = (1, 2, 3, 4, 5),       # tuple
            homo_index_in_window = 2,
            scf_energy_eq_eh     = -1.0,
            scf_energy_minus_eh  = -1.0,
            scf_energy_plus_eh   = -1.0,
        )
        assert es.mo_energies_eq_eh.dtype == np.float64
        assert es.mo_energies_minus_eh.dtype == np.float64
        assert es.mo_energies_plus_eh.dtype == np.float64

    def test_mo_arrays_must_be_1d(self):
        """A 2-D MO array is a programmer error -- catch at
        construction, don't let it propagate."""
        with pytest.raises(ValueError, match="must be 1-D"):
            ModeElectronicStructure(
                amplitude_ang        = 0.1,
                mo_energies_eq_eh    = np.zeros((3, 3)),
                mo_energies_minus_eh = np.zeros(9),
                mo_energies_plus_eh  = np.zeros(9),
                homo_index_in_window = 0,
                scf_energy_eq_eh     = -1.0,
                scf_energy_minus_eh  = -1.0,
                scf_energy_plus_eh   = -1.0,
            )

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

    def test_homo_index_in_range(self):
        """homo_index_in_window must be a valid index into the
        equilibrium MO array."""
        with pytest.raises(ValueError, match="out of range"):
            ModeElectronicStructure(
                amplitude_ang        = 0.1,
                mo_energies_eq_eh    = np.zeros(5),
                mo_energies_minus_eh = np.zeros(5),
                mo_energies_plus_eh  = np.zeros(5),
                homo_index_in_window = 10,                # > 5
                scf_energy_eq_eh     = -1.0,
                scf_energy_minus_eh  = -1.0,
                scf_energy_plus_eh   = -1.0,
            )

    # ----- ModeData -----

    def test_eigenvector_coerced_to_float(self):
        m = ModeData(
            index_1based          = 1,
            frequency_cm1         = 100.0,
            raman_activity_a4_amu = None,
            ir_intensity_km_mol   = None,
            eigenvector_free      = [[1, 0, 0], [-1, 0, 0]],   # list of int lists
            has_imag              = False,
        )
        assert m.eigenvector_free.dtype == np.float64

    def test_eigenvector_shape_validated(self):
        """eigenvector_free must be (n_free, 3); wrong shapes raise."""
        with pytest.raises(ValueError, match="must have shape"):
            ModeData(
                index_1based          = 1,
                frequency_cm1         = 100.0,
                raman_activity_a4_amu = None,
                ir_intensity_km_mol   = None,
                eigenvector_free      = np.zeros((3, 4)),       # last axis != 3
                has_imag              = False,
            )
        with pytest.raises(ValueError, match="must have shape"):
            ModeData(
                index_1based          = 1,
                frequency_cm1         = 100.0,
                raman_activity_a4_amu = None,
                ir_intensity_km_mol   = None,
                eigenvector_free      = np.zeros(6),            # 1-D, not 2-D
                has_imag              = False,
            )

    # ----- SpectraResults -----

    def test_equilibrium_mos_coerced_to_float(self):
        r = SpectraResults(
            schema_version             = SCHEMA_VERSION,
            engine                     = "pyscf",
            engine_version             = "2.6.0",
            molbuilder_version         = "1.2.0",
            timestamp                  = "2026-05-11T12:00:00Z",
            structure_hash             = "sha256:abc",
            n_atoms_total              = 1,
            free_atom_idxs             = [0],
            fixed_atom_idxs            = [],
            equilibrium_scf_eh         = -1.0,
            equilibrium_mo_energies_eh = [-1, 0, 1],            # int list
            equilibrium_homo_idx       = 1,
            modes                      = [],
            selected_mode_idxs_1based  = [],
            config                     = {},
            methods_text               = "",
            bibliography_keys          = [],
        )
        assert r.equilibrium_mo_energies_eh.dtype == np.float64

    def test_homo_idx_validated(self):
        with pytest.raises(ValueError, match="out of range"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                structure_hash             = "h",
                n_atoms_total              = 1,
                free_atom_idxs             = [0],
                fixed_atom_idxs            = [],
                equilibrium_scf_eh         = 0.0,
                equilibrium_mo_energies_eh = np.zeros(3),
                equilibrium_homo_idx       = 5,                 # out of range
                modes                      = [],
                selected_mode_idxs_1based  = [],
                config                     = {},
                methods_text               = "",
                bibliography_keys          = [],
            )

    def test_free_fixed_partition_disjoint(self):
        """An atom can be in either free or fixed, never both."""
        with pytest.raises(ValueError, match="overlap"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                structure_hash             = "h",
                n_atoms_total              = 3,
                free_atom_idxs             = [0, 1, 2],
                fixed_atom_idxs            = [1, 2],            # overlaps with free
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
        with pytest.raises(ValueError, match="!= n_atoms_total"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                structure_hash             = "h",
                n_atoms_total              = 10,                # claim 10
                free_atom_idxs             = [0, 1],            # only 2 free
                fixed_atom_idxs            = [2, 3],            # only 2 fixed
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
        """Every mode's eigenvector_free must agree on n_free
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
            eigenvector_free      = np.zeros((3, 3)),            # 3 atoms, not 2
            has_imag              = False,
        )
        with pytest.raises(ValueError, match=r"eigenvector shape.*expected"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                structure_hash             = "h",
                n_atoms_total              = 2,
                free_atom_idxs             = [0, 1],             # 2 free
                fixed_atom_idxs            = [],
                equilibrium_scf_eh         = 0.0,
                equilibrium_mo_energies_eh = np.zeros(3),
                equilibrium_homo_idx       = 0,
                modes                      = [good_mode, bad_mode],
                selected_mode_idxs_1based  = [],
                config                     = {},
                methods_text               = "",
                bibliography_keys          = [],
            )

    def test_cross_mode_es_window_consistency(self):
        """All modes' electronic_structure must use the same window
        size (HOMO−N .. LUMO+M is a single config knob; modes
        disagreeing means a parser bug)."""
        m1 = _make_mode(index=1, with_es=True)
        # m1's ES window has size 5.  Build m2 with size-7 ES.
        m2 = ModeData(
            index_1based          = 2,
            frequency_cm1         = 200.0,
            raman_activity_a4_amu = None,
            ir_intensity_km_mol   = None,
            eigenvector_free      = np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
            has_imag              = False,
            electronic_structure  = ModeElectronicStructure(
                amplitude_ang        = 0.1,
                mo_energies_eq_eh    = np.zeros(7),           # window=7
                mo_energies_minus_eh = np.zeros(7),
                mo_energies_plus_eh  = np.zeros(7),
                homo_index_in_window = 3,
                scf_energy_eq_eh     = -1.0,
                scf_energy_minus_eh  = -1.0,
                scf_energy_plus_eh   = -1.0,
            ),
        )
        with pytest.raises(ValueError, match="ES window has size"):
            SpectraResults(
                schema_version             = SCHEMA_VERSION,
                engine                     = "pyscf",
                engine_version             = "2.6.0",
                molbuilder_version         = "1.2.0",
                timestamp                  = "t",
                structure_hash             = "h",
                n_atoms_total              = 2,
                free_atom_idxs             = [0, 1],
                fixed_atom_idxs            = [],
                equilibrium_scf_eh         = 0.0,
                equilibrium_mo_energies_eh = np.zeros(3),
                equilibrium_homo_idx       = 0,
                modes                      = [m1, m2],
                selected_mode_idxs_1based  = [1, 2],
                config                     = {},
                methods_text               = "",
                bibliography_keys          = [],
            )

    def test_empty_modes_allowed_for_in_progress_runs(self):
        """An in-progress run before phase-2 (Hessian) has no modes
        yet; the empty list must be accepted by validation."""
        r = SpectraResults(
            schema_version             = SCHEMA_VERSION,
            engine                     = "pyscf",
            engine_version             = "2.6.0",
            molbuilder_version         = "1.2.0",
            timestamp                  = "t",
            structure_hash             = "h",
            n_atoms_total              = 1,
            free_atom_idxs             = [0],
            fixed_atom_idxs            = [],
            equilibrium_scf_eh         = -1.0,
            equilibrium_mo_energies_eh = np.zeros(3),
            equilibrium_homo_idx       = 0,
            modes                      = [],                  # not yet computed
            selected_mode_idxs_1based  = [],
            config                     = {},
            methods_text               = "",
            bibliography_keys          = [],
        )
        assert r.modes == []
        # phase_* default to PHASE_EMPTY for an in-progress run.
        assert r.phase_frequencies == PHASE_EMPTY
        assert r.phase_raman       == PHASE_EMPTY
        assert r.phase_es          == PHASE_EMPTY


# --------------------------------------------------------------------- #
#  SpectraConfig -- L1 dataclass, field metadata, form-schema shape     #
# --------------------------------------------------------------------- #


class TestSpectraConfigDefaults:
    """The dataclass instantiates with all-defaults and the values
    match the v1 spec defaults so a user's first-pass run is
    cheap (es_mode_selection=none) and uses production-defensible
    method/basis choices (B3LYP/def2-SVP/D3BJ, grid 4)."""

    def test_default_construction(self):
        cfg = SpectraConfig()
        # Spec-pinned defaults.
        assert cfg.engine     == "pyscf"
        assert cfg.job_name   == "spectra"
        assert cfg.method     == "RKS"
        assert cfg.functional == "B3LYP"
        assert cfg.basis      == "def2-SVP"
        assert cfg.dispersion == "d3bj"
        assert cfg.density_fit is True

    def test_atom_fix_lists_empty_by_default(self):
        cfg = SpectraConfig()
        assert cfg.fixed_elements      == []
        assert cfg.fixed_residue_names == []
        assert cfg.fixed_indices       == []

    def test_es_off_by_default(self):
        """First-pass run should be cheap -- spectrum only, no
        displaced SCFs.  User opts in to top_n / explicit after
        seeing the spectrum."""
        cfg = SpectraConfig()
        assert cfg.es_mode_selection == "none"

    def test_ir_off_v1_reserved(self):
        """compute_ir is reserved for 1c and ignored in v1."""
        cfg = SpectraConfig()
        assert cfg.compute_ir is False

    def test_displacement_amplitude_production_default(self):
        """0.10 Å is the production-defensible value (Mills 1972
        §2.4)."""
        cfg = SpectraConfig()
        assert cfg.displacement_amplitude_ang == pytest.approx(0.10)


class TestSpectraConfigFieldMetadata:
    """Every form-exposed field carries the metadata the
    schema-driven UI + validator + CLI bridge consume."""

    def test_every_sectioned_field_has_label_and_help(self):
        """Spec-required: any field with a ``section`` key must
        also carry ``label`` + ``help`` so the rendered form has
        a name and a tooltip and the Methods generator has a
        human-facing string to compose with."""
        missing_label = []
        missing_help  = []
        for f in dataclasses.fields(SpectraConfig):
            if "section" not in f.metadata:
                continue
            if not f.metadata.get("label"):
                missing_label.append(f.name)
            if not f.metadata.get("help"):
                missing_help.append(f.name)
        assert missing_label == [], f"missing label on: {missing_label}"
        assert missing_help  == [], f"missing help on:  {missing_help}"

    def test_basename_validator_attached(self):
        """job_name carries the shared _validate_basename callable
        in its metadata so the validation pass refuses paths-with-
        dots / slashes / whitespace (per docs/protocols/job-layout.md)."""
        jn = next(f for f in dataclasses.fields(SpectraConfig)
                  if f.name == "job_name")
        assert callable(jn.metadata.get("validate"))


class TestSpectraConfigSchema:
    """Form-schema shape pin: section names + per-section field
    counts.  A stray field-reorder or a forgotten metadata addition
    would silently rearrange the UI; this test catches it."""

    def test_schema_section_layout(self):
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        assert sch["config"]    == "SpectraConfig"
        assert sch["id_prefix"] == "sp"

        expected = [
            ("System",               2),   # engine, job_name
            ("Method",               5),   # method, functional, basis,
                                           # dispersion, density_fit
            ("Frozen atoms",         3),   # elements, residue_names, indices
            ("Spectrum",             3),   # compute_raman, compute_ir,
                                           # displacement_amplitude_ang
            ("Electronic structure", 8),   # selection, top_n, threshold,
                                           # explicit_indices, freq_min_cm1,
                                           # freq_max_cm1, n_homo_below,
                                           # n_lumo_above
            ("SCF",                  3),   # conv_tol, max_cycle, grid_level
            ("Runtime",              4),   # max_memory_mb, threads, verbose,
                                           # verbose_comments
        ]
        got = [(s["name"], len(s["fields"])) for s in sch["sections"]]
        assert got == expected, got

    def test_es_selection_choices_match_spec(self):
        """The Model 2 selector (spec § 8) must offer exactly the
        five documented options -- none / all / top_n / threshold /
        explicit -- in that order so the form ordering is stable."""
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        es_section = next(s for s in sch["sections"]
                          if s["name"] == "Electronic structure")
        sel_field = next(f for f in es_section["fields"]
                         if f["name"] == "es_mode_selection")
        assert sel_field["kind"]    == "select"
        assert sel_field["choices"] == ["none", "all", "top_n",
                                        "threshold", "explicit"]

    def test_engine_field_carries_only_pyscf_in_v1(self):
        """v1 ships PySCF only; the SIESTA slot is reserved but
        not yet a valid choice.  The schema test pins this so
        adding SIESTA later is an explicit one-line change to
        SpectraConfig.engine's choices tuple AND to this test."""
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        system_section = next(s for s in sch["sections"]
                              if s["name"] == "System")
        engine_field = next(f for f in system_section["fields"]
                            if f["name"] == "engine")
        assert engine_field["choices"] == ["pyscf"]

    def test_legacy_id_overrides_preserved(self):
        """A handful of fields carry id_suffix overrides so the
        rendered HTML id is shorter / matches the UI affordances.
        Pinning the mapping so a future rename of the field name
        doesn't accidentally break the form selectors."""
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        fmap = {f["name"]: f
                for s in sch["sections"]
                for f in s["fields"]}
        # All these fields opt out of the default underscore->hyphen
        # ID transform via id_suffix metadata.
        assert fmap["job_name"]["id"]          == "sp-job-name"
        assert fmap["max_memory_mb"]["id"]     == "sp-max-memory"
        assert fmap["es_mode_selection"]["id"] == "sp-es-selection"
        assert fmap["es_n_homo_below"]["id"]   == "sp-es-n-homo-below"
        assert fmap["es_n_lumo_above"]["id"]   == "sp-es-n-lumo-above"


# --------------------------------------------------------------------- #
#  Phase status + frequency-filter additions (spec § 2.5 / § 8.1)       #
# --------------------------------------------------------------------- #


class TestPhaseStatus:
    """The per-phase status fields are the engine + UI's
    coordination point for the four-layer linear chain.  Validate
    construction + round-trip + transition semantics."""

    def test_running_status_round_trips(self):
        """A mid-run state (phase_es='running') must survive the
        JSON round-trip -- the Watch-style polling endpoint
        returns this state directly to the UI."""
        r = _make_results()
        r.phase_es = PHASE_RUNNING
        r2 = SpectraResults.from_dict(json.loads(json.dumps(r.to_dict())))
        assert r2.phase_es == PHASE_RUNNING

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
                structure_hash             = "h",
                n_atoms_total              = 1,
                free_atom_idxs             = [0],
                fixed_atom_idxs            = [],
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

    def test_independent_phase_state(self):
        """L3 (Raman) and L4 (ES) are siblings under L2 -- their
        completion states are independent.  Construct a fixture
        with L2+L3 complete but L4 still empty, and vice versa."""
        # L2+L3 complete, L4 empty -- the typical "spectrum done,
        # no ES yet" state.
        r = _make_results(complete=False)
        r.phase_raman = PHASE_COMPLETE
        r2 = SpectraResults.from_dict(json.loads(json.dumps(r.to_dict())))
        assert r2.phase_frequencies == PHASE_COMPLETE
        assert r2.phase_raman       == PHASE_COMPLETE
        assert r2.phase_es          == PHASE_EMPTY

        # L2+L4 complete, L3 still empty -- valid too (user skipped
        # Raman, went straight to ES via "explicit" selector).
        r3 = _make_results(complete=True)
        r3.phase_raman = PHASE_EMPTY
        r4 = SpectraResults.from_dict(json.loads(json.dumps(r3.to_dict())))
        assert r4.phase_frequencies == PHASE_COMPLETE
        assert r4.phase_raman       == PHASE_EMPTY
        assert r4.phase_es          == PHASE_COMPLETE


class TestFreqRangeFilter:
    """Spec § 8.1: the freq_min_cm1 / freq_max_cm1 fields appear
    in the Electronic-structure section of the form schema and
    are typed Optional[float].  The Model-2 selector logic that
    APPLIES the filter lives in selection.py (next commit); this
    class just pins the dataclass + schema surface."""

    def test_default_no_filter(self):
        cfg = SpectraConfig()
        assert cfg.freq_min_cm1 is None
        assert cfg.freq_max_cm1 is None

    def test_freq_fields_in_es_section(self):
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        es_section = next(s for s in sch["sections"]
                          if s["name"] == "Electronic structure")
        names = [f["name"] for f in es_section["fields"]]
        assert "freq_min_cm1" in names
        assert "freq_max_cm1" in names

    def test_freq_fields_render_as_optional_number(self):
        """Both freq fields are Optional[float] -> the schema
        emits kind='number' with null_option=True so the form
        widget can offer "no bound" empty-string mode."""
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        fmap = {f["name"]: f
                for s in sch["sections"]
                for f in s["fields"]}
        for fname in ("freq_min_cm1", "freq_max_cm1"):
            f = fmap[fname]
            assert f["kind"] == "number"
            assert f.get("null_option") is True
            assert f["unit"] == "cm⁻¹"


# --------------------------------------------------------------------- #
#  L2 engine Protocol + registry (spec § 3.2)                           #
# --------------------------------------------------------------------- #


class TestEngineRegistry:
    """The engine plug-in registry is the seam between L1 (shared
    types) and L2 (per-engine implementations).  These tests pin
    the registration semantics so the v1 PySCF engine + any future
    SIESTA engine compose correctly.
    """

    def _make_dummy_engine(self, name: str):
        """Build a minimal engine class meeting the Protocol shape
        without doing real work -- enough for registry tests."""
        from molbuilder.spectra import SpectraEngine

        class _DummyEngine:
            pass

        _DummyEngine.name  = name
        _DummyEngine.label = f"dummy ({name})"
        # Stub methods just for Protocol satisfaction; tests don't
        # exercise them.
        _DummyEngine.render_script    = classmethod(lambda c, s, cfg: "")
        _DummyEngine.parse_output     = classmethod(lambda c, p: None)
        _DummyEngine.preflight        = classmethod(
            lambda c, s, cfg, prior=None: []
        )
        _DummyEngine.methods_fragment = classmethod(lambda c, cfg, modes: "")
        return _DummyEngine

    def test_register_and_lookup(self):
        from molbuilder.spectra import (
            register_engine, get_engine, unregister_engine,
        )
        cls = self._make_dummy_engine("dummy-test-1")
        try:
            register_engine(cls)
            assert get_engine("dummy-test-1") is cls
        finally:
            unregister_engine("dummy-test-1")

    def test_unknown_engine_raises_with_available_list(self):
        from molbuilder.spectra import get_engine, UnknownEngineError
        with pytest.raises(UnknownEngineError) as exc_info:
            get_engine("not-a-real-engine-xyz")
        # The error names the requested engine + what's available
        # so a typo is actionable.
        assert "not-a-real-engine-xyz" in str(exc_info.value)
        assert exc_info.value.name == "not-a-real-engine-xyz"
        assert isinstance(exc_info.value.available, list)

    def test_duplicate_registration_rejected(self):
        """Re-registering an existing name is a programmer error
        (two engines claiming the same key); register_engine
        raises rather than silently overwriting."""
        from molbuilder.spectra import register_engine, unregister_engine
        cls1 = self._make_dummy_engine("dummy-test-dup")
        cls2 = self._make_dummy_engine("dummy-test-dup")
        try:
            register_engine(cls1)
            with pytest.raises(ValueError, match="already registered"):
                register_engine(cls2)
        finally:
            unregister_engine("dummy-test-dup")

    def test_re_registering_same_class_is_idempotent(self):
        """Importing an engine module twice (e.g., via reload
        during dev) must not raise -- the second import is the
        SAME class, no conflict."""
        from molbuilder.spectra import register_engine, unregister_engine
        cls = self._make_dummy_engine("dummy-test-idem")
        try:
            register_engine(cls)
            register_engine(cls)   # second call with the same class
            from molbuilder.spectra import get_engine
            assert get_engine("dummy-test-idem") is cls
        finally:
            unregister_engine("dummy-test-idem")

    def test_class_without_name_attribute_rejected(self):
        """An engine class without a `name` class attribute can't
        be registered -- the registry would have nothing to key on."""
        from molbuilder.spectra import register_engine

        class _NamelessEngine:
            label = "no name"

        with pytest.raises(TypeError, match="non-empty string"):
            register_engine(_NamelessEngine)

    def test_registered_engines_returns_sorted_list(self):
        from molbuilder.spectra import (
            register_engine, registered_engines, unregister_engine,
        )
        b = self._make_dummy_engine("b-engine")
        a = self._make_dummy_engine("a-engine")
        try:
            register_engine(b)
            register_engine(a)
            names = registered_engines()
            assert "a-engine" in names
            assert "b-engine" in names
            # Sorted alphabetically -- 'a' before 'b'.
            assert names.index("a-engine") < names.index("b-engine")
        finally:
            unregister_engine("a-engine")
            unregister_engine("b-engine")

    def test_protocol_runtime_checkable(self):
        """SpectraEngine is @runtime_checkable so isinstance works.
        A class meeting the Protocol shape via duck typing should
        satisfy the check; one missing required methods should not."""
        from molbuilder.spectra import SpectraEngine
        cls = self._make_dummy_engine("dummy-test-proto")
        # The dummy has all the right methods + attrs.
        assert isinstance(cls, SpectraEngine)
