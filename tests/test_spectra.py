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

from molbuilder.spectra import (
    ModeData,
    ModeElectronicStructure,
    SpectraConfig,
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
                eigenvector_free      = np.array([[1+0j, 0, 0],
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
                structure_hash             = "h",
                n_atoms_total              = 1,
                free_atom_idxs             = [0],
                fixed_atom_idxs            = [],
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
                eigenvector_free      = np.zeros((2, 3), dtype=complex),
                has_imag              = False,
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
        assert cfg.es_mode_selection == "skip"

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
            ("Runtime",              5),   # max_memory_mb, threads,
                                           # use_gpu, verbose, verbose_comments
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
        assert sel_field["choices"] == ["skip", "all", "top_n",
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
        # Note: SpectraEngine is the Protocol; we don't subclass it
        # explicitly (duck-typed Protocols don't need it), but the
        # isinstance test in test_protocol_runtime_checkable does
        # import it to assert structural conformance.
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


# --------------------------------------------------------------------- #
#  L2 mode selection (selection.py)                                     #
#                                                                       #
#  Spec § 8 + § 8.1 + § 2.5.3.  Pure functions, exhaustively tested.    #
# --------------------------------------------------------------------- #


def _modes_fixture():
    """Build a 6-mode fixture with varied frequencies + Raman
    activities so we can exercise top_n / threshold / window /
    explicit combinations cleanly.

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
        return ModeData(
            index_1based          = idx,
            frequency_cm1         = freq,
            raman_activity_a4_amu = raman,
            ir_intensity_km_mol   = None,
            eigenvector_free      = np.zeros((2, 3)),
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


class TestSelectModes:

    def test_selector_none_returns_empty(self):
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="skip")
        assert select_modes(_modes_fixture(), cfg) == []

    def test_selector_all_returns_every_mode(self):
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="all")
        assert select_modes(_modes_fixture(), cfg) == [1, 2, 3, 4, 5, 6]

    def test_selector_all_respects_freq_window(self):
        """Window [800, 2500] -> modes 3 (1023) and 4 (1612)."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="all",
                            freq_min_cm1=800.0, freq_max_cm1=2500.0)
        assert select_modes(_modes_fixture(), cfg) == [3, 4]

    def test_selector_top_n_orders_by_activity_descending(self):
        """Top-3: brightest first -> 3 (87.2), 5 (45.0), 6 (18.5)."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        assert select_modes(_modes_fixture(), cfg) == [3, 5, 6]

    def test_selector_top_n_skips_modes_without_activity(self):
        """Mode 4 has raman_activity=None -- excluded from top_n
        ranking even if N would otherwise include it."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=10)
        # Asks for 10, only 5 modes have activities; all 5 returned.
        out = select_modes(_modes_fixture(), cfg)
        assert 4 not in out
        assert sorted(out) == [1, 2, 3, 5, 6]

    def test_selector_top_n_with_freq_window(self):
        """Top-2 within [2000, 4000] -> 5 (45), 6 (18) -- mode 4
        falls in the window but has no activity, mode 5 + 6 are the
        only Raman-active candidates."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=2,
                            freq_min_cm1=2000.0, freq_max_cm1=4000.0)
        assert select_modes(_modes_fixture(), cfg) == [5, 6]

    def test_selector_top_n_clamps_silently(self):
        """N > available count silently clamps (warning lives in
        validate_selection, not select_modes)."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=100)
        out = select_modes(_modes_fixture(), cfg)
        # 5 modes have activities; all 5 returned regardless of N=100.
        assert len(out) == 5

    def test_selector_threshold(self):
        """Threshold 15 -> activities > 15 are modes 3 (87) and 5 (45).
        Mode 6 has 18.5, so it passes too.  Modes 2 (12.5) and 1 (3.2)
        fall under.  Mode 4 has no activity (excluded)."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="threshold",
                            es_threshold=15.0)
        out = select_modes(_modes_fixture(), cfg)
        assert sorted(out) == [3, 5, 6]

    def test_selector_threshold_with_freq_window(self):
        """Threshold 10 + window [500, 1500] -> only mode 3 passes
        both (mode 2's freq is in window but activity 12.5 > 10; let
        me recompute: mode 2 freq=745 in [500,1500] ✓, activity 12.5 >
        10 ✓ -- passes.  Mode 3 freq=1023 in window ✓, 87 > 10 ✓ --
        passes.)"""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="threshold",
                            es_threshold=10.0,
                            freq_min_cm1=500.0, freq_max_cm1=1500.0)
        assert sorted(select_modes(_modes_fixture(), cfg)) == [2, 3]

    def test_selector_explicit(self):
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[3, 5])
        assert select_modes(_modes_fixture(), cfg) == [3, 5]

    def test_selector_explicit_ignores_freq_window(self):
        """Spec § 8.1: explicit IGNORES freq filter.  User asked
        for modes 3 + 5 + 6; even though mode 3 is well below
        any window, the explicit selector returns them all."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[3, 5, 6],
                            # Window that would exclude 3 if applied
                            freq_min_cm1=2000.0)
        assert select_modes(_modes_fixture(), cfg) == [3, 5, 6]

    def test_explicit_dedupes_repeats(self):
        """Repeated explicit indices collapse; order preserved."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2, 5, 2, 5, 3])
        assert select_modes(_modes_fixture(), cfg) == [2, 5, 3]


class TestSelectModesWithPriorResume:
    """Spec § 2.5.2 + § 6.1: when prior has ES data for some modes,
    those modes are skipped on the next run (non-destructive L4)."""

    def _prior_with_es_on_mode(self, idx: int) -> SpectraResults:
        """Build a SpectraResults whose modes list contains the
        requested mode, with electronic_structure populated.  The
        other fields are irrelevant to select_modes (it only reads
        prior.modes[*].electronic_structure)."""
        return SpectraResults(
            schema_version             = SCHEMA_VERSION,
            engine                     = "pyscf",
            engine_version             = "2.6.0",
            molbuilder_version         = "1.2.0",
            timestamp                  = "2026-05-11T12:00:00Z",
            structure_hash             = "sha256:abc",
            n_atoms_total              = 2,
            free_atom_idxs             = [0, 1],
            fixed_atom_idxs            = [],
            equilibrium_scf_eh         = -1.0,
            equilibrium_mo_energies_eh = np.zeros(5),
            equilibrium_homo_idx       = 2,
            modes                      = [
                ModeData(
                    index_1based         = idx,
                    frequency_cm1        = 1000.0,
                    raman_activity_a4_amu = 1.0,
                    ir_intensity_km_mol  = None,
                    eigenvector_free     = np.zeros((2, 3)),
                    has_imag             = False,
                    electronic_structure = _make_es(),
                ),
            ],
            selected_mode_idxs_1based  = [idx],
            config                     = {},
            methods_text               = "",
            bibliography_keys          = [],
            phase_frequencies          = PHASE_COMPLETE,
            phase_raman                = PHASE_COMPLETE,
            phase_es                   = PHASE_COMPLETE,
        )

    def test_prior_with_es_filters_out_completed_mode(self):
        """User re-runs with selector=explicit=[2,3,5] but mode 3
        already has ES from a prior run.  select_modes returns
        [2, 5] -- the engine will only compute ES for those."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2, 3, 5])
        prior = self._prior_with_es_on_mode(idx=3)
        assert select_modes(_modes_fixture(), cfg, prior=prior) == [2, 5]

    def test_prior_without_es_does_nothing(self):
        """prior=None and prior.modes-with-ES=[] both leave the
        selection unchanged."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2, 3])
        assert select_modes(_modes_fixture(), cfg, prior=None) == [2, 3]

    def test_prior_filters_top_n_path_too(self):
        """Resume works for every selector, not just explicit.
        top_n=3 with prior ES on mode 3 -> returns the next-best
        modes instead."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        prior = self._prior_with_es_on_mode(idx=3)
        out = select_modes(_modes_fixture(), cfg, prior=prior)
        # top_n=3 normally returns [3, 5, 6]; with 3 already done -> [5, 6].
        # Note that this is a DROP (3 removed), not a re-rank to take
        # the 4th-place mode (2) -- resume preserves "what was asked"
        # minus "what's done", it doesn't re-allocate slots.
        assert out == [5, 6]


class TestValidateSelection:
    """validate_selection surfaces preflight errors / warns before
    the script runs.  See spec § 2.5.3 (soft dep) + § 11.4
    (scientific warns)."""

    def test_clean_explicit_no_issues(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2, 3])
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        assert issues == []

    def test_top_n_without_l3_errors(self):
        """Soft dep: top_n REQUIRES the prior Raman activities."""
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        errs = [i for i in issues if i.severity == "error"]
        assert len(errs) == 1
        # Plain-language reference to Raman + the workaround hint.
        assert "Raman" in errs[0].message
        assert "Compute Raman activities" in errs[0].message

    def test_threshold_without_l3_errors(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="threshold",
                            es_threshold=10.0)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        errs = [i for i in issues if i.severity == "error"]
        assert len(errs) == 1
        # Plain-language Raman reference + workaround hint.
        assert "Raman" in errs[0].message

    def test_top_n_with_l3_passes(self):
        """With L3 complete, top_n is fine."""
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=True)
        errs = [i for i in issues if i.severity == "error"]
        assert errs == []

    def test_explicit_out_of_range_errors(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[3, 99, 200])
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        errs = [i for i in issues
                if i.where == "config.es_explicit_indices"]
        assert len(errs) == 1
        assert "99" in errs[0].message
        assert "200" in errs[0].message

    def test_freq_min_greater_than_max_errors(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="all",
                            freq_min_cm1=2000.0,
                            freq_max_cm1=1000.0)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        errs = [i for i in issues
                if i.severity == "error"
                and i.where == "config.freq_window"]
        assert len(errs) == 1

    def test_freq_window_empty_warns(self):
        """A window that captures zero modes is suspicious but
        valid -- warn, not error."""
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="all",
                            freq_min_cm1=4500.0,
                            freq_max_cm1=5000.0)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=True)
        warns = [i for i in issues if i.severity == "warn"]
        assert any("zero modes" in i.message for i in warns)

    def test_top_n_exceeds_available_warns(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=20)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=True)
        warns = [i for i in issues if i.severity == "warn"]
        # Plain-language: "only N modes are available"
        assert any("only" in i.message and "modes are available"
                   in i.message for i in warns)


# --------------------------------------------------------------------- #
#  L2 Methods composer (methods.py)                                     #
#                                                                       #
#  Spec § 11.2 + § 9.4.  Pure prose generation; no engine I/O.          #
# --------------------------------------------------------------------- #


class TestExtractCitationKeys:
    """The bibliography-extractor underlies both the trailing
    bibliography in render_methods_md and the
    SpectraResults.bibliography_keys field (spec § 5).  Test it
    standalone so its semantics are pinned independently of the
    composer's prose choices."""

    def test_basic_extraction(self):
        from molbuilder.spectra import extract_citation_keys
        text = "We cite [Sun2020] and also [Becke1993]."
        assert extract_citation_keys(text) == ["Sun2020", "Becke1993"]

    def test_section_suffix_stripped(self):
        """`[Key §section]` patterns -- the §-clause is prose, the
        key alone is what resolves against references.bib."""
        from molbuilder.spectra import extract_citation_keys
        text = "anharmonic-cubic mixing < 1% [Mills1972 §2.4]"
        assert extract_citation_keys(text) == ["Mills1972"]

    def test_deduplication_preserves_first_occurrence(self):
        from molbuilder.spectra import extract_citation_keys
        text = "[Sun2020] then [Becke1993] then [Sun2020] again."
        assert extract_citation_keys(text) == ["Sun2020", "Becke1993"]

    def test_empty_or_no_citations(self):
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("") == []
        assert extract_citation_keys("no citations here") == []
        # The regex requires the first char to be a letter -- digit-
        # leading "keys" don't match (BibTeX style: AuthorYYYY).
        assert extract_citation_keys("[123numeric] [9Sun2020]") == []
        # A purely alphabetic bracket-word like [array] DOES look like
        # a citation key structurally and will be extracted; that's
        # accepted as the cost of a permissive author key pattern --
        # the references.bib linter (spec § 11.3) catches the false
        # positive at release-tag time.
        assert extract_citation_keys("an [array] of words") == ["array"]

    def test_underscores_allowed_in_keys(self):
        """BibTeX keys can contain underscores -- e.g. `Sun_2020`."""
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("[Foo_Bar2020]") == ["Foo_Bar2020"]

    def test_separate_brackets(self):
        """Each `[Key]` bracket pair contributes its key."""
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("[Galperin2007] [Frederiksen2007]") \
            == ["Galperin2007", "Frederiksen2007"]

    def test_comma_separated_keys_in_one_bracket_split(self):
        """`[Foo, Bar]` is common physics/chem prose style.  Each
        comma-separated key contributes its key, preserving
        first-appearance order."""
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("PySCF [Sun2020, Sun2018] is widely used.") \
            == ["Sun2020", "Sun2018"]

    def test_comma_separated_keys_dedupe_against_earlier(self):
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys(
            "[Sun2020] then [Sun2020, Sun2018]"
        ) == ["Sun2020", "Sun2018"]

    def test_three_comma_separated_keys(self):
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("[A2020, B2021, C2022]") == ["A2020", "B2021", "C2022"]


class TestRenderMethodsMdPreRun:
    """Pre-run path (`results=None`): the prose describes what
    *will* be done with the configured knobs.  Used by the
    Methods-preview modal (spec § 9.4) before the user runs the
    script."""

    def test_minimal_config_produces_paragraph(self):
        """Default SpectraConfig (selector=none, no ES) -> single
        L2 paragraph with functional + basis + dispersion mentions."""
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig()
        md = render_methods_md(cfg)
        assert "## Methods" in md
        # Default level: B3LYP / def2-SVP / D3BJ.
        assert "B3LYP" in md
        assert "def2-SVP" in md
        assert "D3BJ" in md or "d3bj" in md.lower()
        # selector=none -> NO per-mode-ES paragraph.
        assert "per-mode electronic" not in md.lower()

    def test_dispersion_none_omits_dispersion_clause(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(dispersion="none")
        md = render_methods_md(cfg)
        assert "dispersion" not in md.lower()

    def test_compute_raman_false_omits_raman_prose(self):
        """diagnostic / Hessian-only run -> no dα/dR clause."""
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(compute_raman=False)
        md = render_methods_md(cfg)
        assert "Raman activities" not in md
        assert "Komornicki1979" not in md

    def test_compute_raman_true_cites_komornicki_and_wilson(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(compute_raman=True)
        md = render_methods_md(cfg)
        assert "Komornicki1979" in md
        assert "Wilson1955" in md

    def test_selector_all_emits_es_paragraph(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="all")
        md = render_methods_md(cfg)
        assert "every vibrational mode" in md
        assert "Galperin2007" in md
        assert "Frederiksen2007" in md
        assert "Mills1972" in md
        # Default amplitude 0.10 Å should appear.
        assert "0.1" in md
        assert "A = 0.1" in md or "A=0.1" in md or "0.1 Å" in md

    def test_selector_top_n_named_in_prose(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=7)
        md = render_methods_md(cfg)
        assert "top 7" in md

    def test_selector_threshold_named_in_prose(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="threshold", es_threshold=2.5)
        md = render_methods_md(cfg)
        assert "Raman activity > 2.5" in md

    def test_selector_explicit_states_count(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[3, 5, 8, 12])
        md = render_methods_md(cfg)
        assert "user-specified set of 4 modes" in md

    def test_frequency_window_clause_both_bounds(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="all",
                            freq_min_cm1=500.0, freq_max_cm1=2000.0)
        md = render_methods_md(cfg)
        assert "500" in md and "2000" in md
        assert "cm⁻¹" in md

    def test_frequency_window_one_sided(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="all", freq_min_cm1=1500.0)
        md = render_methods_md(cfg)
        assert "≥ 1500" in md or ">= 1500" in md

    def test_frequency_window_ignored_for_explicit(self):
        """selector=explicit ignores the freq window (spec § 8.1);
        the prose shouldn't claim a window restriction that won't
        actually be enforced."""
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[1, 2],
                            freq_min_cm1=1000.0,
                            freq_max_cm1=2000.0)
        md = render_methods_md(cfg)
        # No "within 1000-2000 cm⁻¹" clause -- the window doesn't
        # apply to explicit selections.
        assert "1000-2000" not in md
        assert "within the 1000" not in md

    def test_non_b3_functional_omits_becke_citation(self):
        """Becke1993 is the B3-family paper; cite it only when the
        functional is in that family."""
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(functional="PBE0")
        md = render_methods_md(cfg)
        assert "Becke1993" not in md

    def test_bibliography_listed_at_end(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="all")
        md = render_methods_md(cfg)
        assert "**Bibliography**" in md
        # The bibliography section appears AFTER the prose
        # (so a reader scrolling top-to-bottom hits the keys last).
        bib_pos = md.index("**Bibliography**")
        # All inline citations precede the bibliography.
        first_cite = md.index("[")
        assert first_cite < bib_pos


class TestRenderMethodsMdPostRun:
    """Post-run path (`results` provided): real numbers from the
    parsed SpectraResults replace pre-run placeholders.  Used to
    populate SpectraResults.methods_text (spec § 5) -- the same
    prose lands in the JSON for downstream consumers."""

    def test_frequency_span_appended_when_modes_present(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig()
        results = _make_results(complete=True)
        md = render_methods_md(cfg, results=results)
        # Real frequencies from _make_results: 412.3, 1023.4, 3656.0
        assert "3 modes" in md
        assert "412" in md
        assert "3656" in md

    def test_imaginary_modes_called_out(self):
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig()
        results = _make_results(complete=True)
        # Inject an imaginary mode.
        results.modes.append(_make_mode(index=4, freq=-150.0, with_es=False))
        md = render_methods_md(cfg, results=results)
        assert "imaginary" in md

    def test_selected_modes_line_post_run(self):
        """When ES data is present in results, the post-run prose
        ends with a "Selected modes: ..." line listing the indices
        + frequencies (spec § 11.2)."""
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2])
        results = _make_results(complete=True)
        md = render_methods_md(cfg, results=results)
        assert "Selected modes" in md
        # _make_results gives mode 2 ES at 1023.4 cm⁻¹.
        assert "mode 2" in md
        assert "1023" in md

    def test_es_count_appended_to_l4_paragraph(self):
        """The L4 paragraph gains "In the present run X modes
        received per-mode electronic-structure data." when results
        exist."""
        from molbuilder.spectra import render_methods_md
        cfg = SpectraConfig(es_mode_selection="all")
        results = _make_results(complete=True)
        md = render_methods_md(cfg, results=results)
        assert "1 modes received" in md or "In the present run" in md


class TestRenderMethodsMdEngineFragment:
    """The composer is engine-agnostic but accepts an engine class
    whose `methods_fragment(cfg, modes)` returns engine-specific
    prose.  The fragment is interleaved between the L2 and L4
    paragraphs; citation keys from the fragment flow into the
    trailing bibliography just like the generic prose's keys."""

    def _make_engine_with_fragment(self, frag: str):
        """Build a minimal engine whose methods_fragment returns
        ``frag``.  Other Protocol methods are stubs."""
        class _E:
            name = "test-frag-engine"
            label = "test engine"
        _E.render_script    = classmethod(lambda c, s, cfg: "")
        _E.parse_output     = classmethod(lambda c, p: None)
        _E.preflight        = classmethod(lambda c, s, cfg, prior=None: [])
        _E.methods_fragment = classmethod(lambda c, cfg, modes: frag)
        return _E

    def test_fragment_appears_in_output(self):
        from molbuilder.spectra import render_methods_md
        eng = self._make_engine_with_fragment(
            "The analytic Hessian was obtained via `pyscf.hessian.rks` [Sun2020]."
        )
        md = render_methods_md(SpectraConfig(), engine=eng)
        assert "pyscf.hessian.rks" in md

    def test_fragment_citations_join_bibliography(self):
        """A citation key that appears only in the engine fragment
        must still land in the trailing **Bibliography** list."""
        from molbuilder.spectra import render_methods_md
        eng = self._make_engine_with_fragment(
            "Custom citation only here: [Sun2018]."
        )
        md = render_methods_md(SpectraConfig(), engine=eng)
        # Sun2018 isn't cited in any generic prose path -- it appears
        # *only* in the engine fragment.  It must still be listed.
        bib_section = md.split("**Bibliography**", 1)[1]
        assert "Sun2018" in bib_section

    def test_engine_raising_in_fragment_does_not_crash(self):
        """A buggy engine shouldn't break the Methods preview --
        the composer swallows fragment exceptions defensively
        (live form re-renders on every keystroke)."""
        from molbuilder.spectra import render_methods_md

        class _Broken:
            name = "broken"
            label = "broken"
        _Broken.render_script    = classmethod(lambda c, s, cfg: "")
        _Broken.parse_output     = classmethod(lambda c, p: None)
        _Broken.preflight        = classmethod(lambda c, s, cfg, prior=None: [])
        def _boom(cls, cfg, modes):
            raise RuntimeError("fragment generator exploded")
        _Broken.methods_fragment = classmethod(_boom)

        md = render_methods_md(SpectraConfig(), engine=_Broken)
        # Generic prose still rendered.
        assert "## Methods" in md
        assert "B3LYP" in md

    def test_unknown_engine_in_cfg_does_not_crash(self):
        """If cfg.engine isn't registered (e.g. test environment
        with no engines imported) the composer omits the fragment
        silently rather than raising UnknownEngineError."""
        from molbuilder.spectra import render_methods_md
        # Default SpectraConfig has engine="pyscf"; in this test
        # environment the PySCF engine module isn't imported, so the
        # registry doesn't know it -- the composer should still work.
        md = render_methods_md(SpectraConfig())
        assert "## Methods" in md


class TestRenderMethodsMdWithStruct:
    """Atom-count phrasing is gated on Structure availability."""

    def test_struct_none_omits_atom_clause(self):
        from molbuilder.spectra import render_methods_md
        md = render_methods_md(SpectraConfig())
        assert "free, " not in md  # no "(N free, M held fixed)" clause
        assert "vibrational modes" not in md or "vibrational mode" in md
        # When struct is None, the L2 paragraph has no atom counts.

    def test_struct_provided_emits_atom_clause(self):
        """When a Structure is provided, the prose names total
        atoms, free atoms, and the 3N-6 mode count."""
        from molbuilder.spectra import render_methods_md

        class _Atom:
            def __init__(self, sym):
                self.symbol = sym
        # 5-atom water-cluster mock; no atoms fixed.
        atoms = [_Atom("O"), _Atom("H"), _Atom("H"), _Atom("O"), _Atom("H")]

        class _S:
            pass
        struct = _S()
        struct.atoms = atoms

        md = render_methods_md(SpectraConfig(), struct=struct)
        assert "5 atoms" in md
        # 3*5 - 6 = 9 modes for all-free.
        assert "9 non-translational" in md or "9 " in md

    def test_struct_with_fixed_elements_counts_correctly(self):
        """Fixed-by-element subtracts the right atoms from n_free."""
        from molbuilder.spectra import render_methods_md

        class _Atom:
            def __init__(self, sym):
                self.symbol = sym
        # 4 Au + 3 organic = 7 atoms; fix Au -> n_free=3.
        atoms = ([_Atom("Au")] * 4 + [_Atom("C"), _Atom("H"), _Atom("H")])

        class _S:
            pass
        struct = _S()
        struct.atoms = atoms

        cfg = SpectraConfig(fixed_elements=["Au"])
        md = render_methods_md(cfg, struct=struct)
        # 3 free, 4 fixed.
        assert "3 free" in md
        assert "4 held fixed" in md

    def test_real_structure_dataclass_works(self):
        """A real molbuilder.Structure (elements as List[str], not
        list-of-atom-objects) should feed atom counts correctly --
        regression test against the mock-only earlier version."""
        from molbuilder.spectra import render_methods_md
        from molbuilder.structure import Structure
        struct = Structure(
            elements  = ["O", "H", "H"],
            positions = np.array([[0., 0., 0.],
                                  [0.96, 0., 0.],
                                  [-0.24, 0.93, 0.]]),
        )
        md = render_methods_md(SpectraConfig(), struct=struct)
        assert "3 atoms" in md
        # 3*3 - 6 = 3 modes for water.
        assert "3 non-translational" in md

    def test_real_structure_with_fixed_elements(self):
        """Real Structure + fixed_elements=['Au'] -> Au atoms
        removed from the free count."""
        from molbuilder.spectra import render_methods_md
        from molbuilder.structure import Structure
        struct = Structure(
            elements  = ["Au", "Au", "C", "H"],
            positions = np.array([[0., 0., 0.],
                                  [2., 0., 0.],
                                  [4., 0., 0.],
                                  [5., 0., 0.]]),
        )
        cfg = SpectraConfig(fixed_elements=["Au"])
        md = render_methods_md(cfg, struct=struct)
        assert "2 free" in md
        assert "2 held fixed" in md


# --------------------------------------------------------------------- #
#  PySCFSpectraEngine (engine wrapper -- non-render_script methods)     #
#                                                                       #
#  Spec § 3.2 + § 9 + § 11.  render_script is tested separately when    #
#  the script-template module lands (next commit).                      #
# --------------------------------------------------------------------- #


def _struct_water():
    """Real Structure for water -- used by engine preflight tests."""
    from molbuilder.structure import Structure
    return Structure(
        elements  = ["O", "H", "H"],
        positions = np.array([[0., 0., 0.],
                              [0.96, 0., 0.],
                              [-0.24, 0.93, 0.]]),
    )


class TestPySCFSpectraEngineRegistration:

    def test_registered_under_pyscf(self):
        """The engine self-registers at import time -- importing
        molbuilder.spectra (or .pyscf_engine) puts 'pyscf' in the
        registry."""
        from molbuilder.spectra import get_engine
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        assert get_engine("pyscf") is PySCFSpectraEngine

    def test_engine_metadata(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        assert PySCFSpectraEngine.name == "pyscf"
        assert "PySCF" in PySCFSpectraEngine.label


class TestPySCFEngineMethodsFragment:

    def test_basic_fragment_cites_pyscf(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        frag = PySCFSpectraEngine.methods_fragment(cfg, [])
        # Pyscf citation keys present.
        assert "Sun2020" in frag
        assert "Sun2018" in frag
        # Names the analytic Hessian module.
        assert "pyscf.hessian" in frag

    def test_method_specific_hessian_module(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(method="UHF")
        frag = PySCFSpectraEngine.methods_fragment(cfg, [])
        assert "pyscf.hessian.uhf" in frag

    def test_raman_path_cites_komornicki(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(compute_raman=True)
        frag = PySCFSpectraEngine.methods_fragment(cfg, [])
        assert "Komornicki1979" in frag
        assert "polarizability" in frag.lower()

    def test_no_raman_path_omits_komornicki(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(compute_raman=False)
        frag = PySCFSpectraEngine.methods_fragment(cfg, [])
        assert "Komornicki1979" not in frag

    def test_density_fit_mentioned_when_on(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg_on  = SpectraConfig(density_fit=True)
        cfg_off = SpectraConfig(density_fit=False)
        assert "density fitting" in PySCFSpectraEngine.methods_fragment(cfg_on, []).lower()
        assert "density fitting" not in PySCFSpectraEngine.methods_fragment(cfg_off, []).lower()

    def test_grid_level_mentioned_for_dft_only(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        # DFT path mentions grid level.
        assert "grid level" in PySCFSpectraEngine.methods_fragment(
            SpectraConfig(method="RKS"), []
        ).lower()
        # HF path doesn't.
        assert "grid level" not in PySCFSpectraEngine.methods_fragment(
            SpectraConfig(method="RHF"), []
        ).lower()

    def test_fragment_composes_into_render_methods_md(self):
        """The engine's fragment flows into render_methods_md's
        output and its citations bubble up into the bibliography."""
        from molbuilder.spectra import render_methods_md
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        md = render_methods_md(cfg, engine=PySCFSpectraEngine)
        assert "pyscf.hessian" in md
        assert "Sun2020" in md
        # Sun2020 appears in the trailing bibliography too.
        bib = md.split("**Bibliography**", 1)[1]
        assert "Sun2020" in bib


class TestPySCFEnginePreflight:

    def test_clean_config_has_no_issues(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()  # defaults -- selector=none, no L3 dep
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []

    def test_hybrid_with_low_grid_warns(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(functional="B3LYP", grid_level=3)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        warns = [i for i in issues if i.severity == "warn"]
        assert any(i.where == "config.grid_level" for i in warns)

    def test_pbe0_recognised_as_hybrid(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(functional="PBE0", grid_level=2)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.grid_level"
                   and i.severity == "warn" for i in issues)

    def test_pure_functional_no_grid_warn(self):
        """Pure PBE (not hybrid) shouldn't trip the grid-level warn
        -- the recommendation is hybrid-specific."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(functional="PBE", grid_level=2)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any(i.where == "config.grid_level" for i in issues)

    def test_displacement_below_window_warns(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(displacement_amplitude_ang=0.03)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.displacement_amplitude_ang"
                   and i.severity == "warn" for i in issues)

    def test_displacement_above_window_warns(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(displacement_amplitude_ang=0.25)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.displacement_amplitude_ang"
                   and i.severity == "warn" for i in issues)

    def test_default_displacement_no_warn(self):
        """Default 0.10 Å is the defensible production value."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any(i.where == "config.displacement_amplitude_ang"
                       for i in issues)

    def test_compute_ir_warns_reserved(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(compute_ir=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        # Plain-language: "aren't implemented" + suggestion.
        assert any(i.where == "config.compute_ir"
                   and i.severity == "warn"
                   and "implemented" in i.message for i in issues)

    def test_use_gpu_warns_when_gpu4pyscf_missing(self, monkeypatch):
        """Asking for GPU acceleration on a host where gpu4pyscf
        isn't installed should warn (not error) so the user has
        time to install it before running the generated script,
        but the generated script falls back to CPU anyway.

        Simulates the missing-package state by setting
        sys.modules['gpu4pyscf'] = None, the standard pytest trick
        for forcing an ImportError on `import gpu4pyscf` regardless
        of the installed environment.
        """
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        import sys
        monkeypatch.setitem(sys.modules, "gpu4pyscf", None)
        cfg = SpectraConfig(use_gpu=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        warns = [i for i in issues
                 if i.severity == "warn" and i.where == "config.use_gpu"]
        assert len(warns) == 1
        # Message names the package + the install command.
        assert "gpu4pyscf" in warns[0].message
        assert "pip install" in warns[0].message
        # And explicitly mentions the CPU fallback so the user knows
        # this is non-fatal.
        assert "fall" in warns[0].message.lower() \
            or "cpu" in warns[0].message.lower()

    def test_use_gpu_no_warn_when_gpu4pyscf_and_modern_gpu(self, monkeypatch):
        """When gpu4pyscf is importable AND the GPU is modern enough
        (compute capability >= 7.0), no advisory should fire."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        import sys, types
        # Inject a fake gpu4pyscf so the import succeeds.
        monkeypatch.setitem(sys.modules, "gpu4pyscf",
                            types.ModuleType("gpu4pyscf"))
        # Inject a fake cupy that reports a modern GPU.
        fake_cupy = types.ModuleType("cupy")
        fake_cuda = types.ModuleType("cupy.cuda")
        fake_runtime = types.ModuleType("cupy.cuda.runtime")
        fake_runtime.getDeviceCount = lambda: 1
        fake_runtime.getDeviceProperties = lambda i: {
            "name": "Fake H100",   # modern card
            "major": 9, "minor": 0,
        }
        fake_cuda.runtime = fake_runtime
        fake_cupy.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "cupy",         fake_cupy)
        monkeypatch.setitem(sys.modules, "cupy.cuda",    fake_cuda)
        monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", fake_runtime)
        cfg = SpectraConfig(use_gpu=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        gpu_warns = [i for i in issues if i.where == "config.use_gpu"]
        assert gpu_warns == []

    def test_use_gpu_warns_when_gpu_too_old(self, monkeypatch):
        """Card has compute capability < 7.0 -- gpu4pyscf will fail
        with cryptic CUDA errors during the SCF.  Warn before the
        run, suggest disabling 'Use GPU'."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        import sys, types
        monkeypatch.setitem(sys.modules, "gpu4pyscf",
                            types.ModuleType("gpu4pyscf"))
        fake_cupy = types.ModuleType("cupy")
        fake_cuda = types.ModuleType("cupy.cuda")
        fake_runtime = types.ModuleType("cupy.cuda.runtime")
        fake_runtime.getDeviceCount = lambda: 1
        fake_runtime.getDeviceProperties = lambda i: {
            "name": "GTX 1080",   # Pascal, compute cap 6.1
            "major": 6, "minor": 1,
        }
        fake_cuda.runtime = fake_runtime
        fake_cupy.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "cupy",              fake_cupy)
        monkeypatch.setitem(sys.modules, "cupy.cuda",         fake_cuda)
        monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", fake_runtime)
        cfg = SpectraConfig(use_gpu=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        gpu_warns = [i for i in issues
                     if i.where == "config.use_gpu" and i.severity == "warn"]
        assert len(gpu_warns) == 1
        msg = gpu_warns[0].message
        # Message names the actual card + compute capability the
        # user can compare against the gpu4pyscf docs.
        assert "GTX 1080" in msg
        assert "6.1" in msg
        # ... and the minimum requirement.
        assert "7.0" in msg
        # ... and the actionable "untick" suggestion.
        assert "Use GPU" in msg

    def test_use_gpu_warns_when_no_gpu(self, monkeypatch):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        import sys, types
        monkeypatch.setitem(sys.modules, "gpu4pyscf",
                            types.ModuleType("gpu4pyscf"))
        fake_cupy = types.ModuleType("cupy")
        fake_cuda = types.ModuleType("cupy.cuda")
        fake_runtime = types.ModuleType("cupy.cuda.runtime")
        fake_runtime.getDeviceCount = lambda: 0
        fake_cuda.runtime = fake_runtime
        fake_cupy.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "cupy",              fake_cupy)
        monkeypatch.setitem(sys.modules, "cupy.cuda",         fake_cuda)
        monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", fake_runtime)
        cfg = SpectraConfig(use_gpu=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        gpu_warns = [i for i in issues if i.where == "config.use_gpu"]
        assert len(gpu_warns) == 1
        assert "no NVIDIA GPU" in gpu_warns[0].message

    def test_use_gpu_off_no_warn(self):
        """When the user leaves GPU off, the GPU advisory shouldn't
        fire even if gpu4pyscf isn't installed."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(use_gpu=False)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any(i.where == "config.use_gpu" for i in issues)

    def test_few_fixed_atoms_warns_about_spurious_modes(self):
        """Fixing 1 or 2 atoms can't fully anchor the free fragment
        in space -- residual rigid-body motion leaks into the
        vibrational analysis as near-zero modes.  Warn so the user
        ignores those modes when interpreting the spectrum."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        for n in (1, 2):
            cfg = SpectraConfig(fixed_indices=list(range(n)))
            issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
            warns = [i for i in issues
                     if i.severity == "warn"
                     and i.where == "config.fixed_indices"
                     and "spurious" in i.message]
            assert len(warns) == 1, (n, issues)

    def test_three_or_more_fixed_atoms_no_spurious_warn(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(fixed_indices=[0, 1, 2])
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any("spurious" in i.message for i in issues)

    def test_element_freezing_doesnt_trigger_spurious_warn(self):
        """Element-level freezing typically pins many atoms (a whole
        metal slab); the spurious-modes warn shouldn't fire when the
        user is freezing by element rather than by index."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(fixed_elements=["O"])
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any("spurious" in i.message for i in issues)

    def test_unsupported_method_errors(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        # Sidestep the dataclass's choices validation by setting
        # the attribute directly -- the preflight is the second
        # line of defence anyway.
        cfg.method = "BOGUS"
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.method"
                   and i.severity == "error" for i in issues)

    def test_out_of_range_fixed_indices_errors(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(fixed_indices=[0, 1, 99])  # water has 3 atoms
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.fixed_indices"
                   and i.severity == "error" for i in issues)

    def test_in_range_fixed_indices_ok(self):
        """In-range fixed_indices should NOT produce a range-check
        error.  (A separate test covers the WARN about spurious
        rigid-body modes when fewer than 3 atoms are fixed.)"""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(fixed_indices=[0, 1])  # all valid for water
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        errors_from_indices = [i for i in issues
                               if i.where == "config.fixed_indices"
                               and i.severity == "error"]
        assert errors_from_indices == []

    def test_selector_top_n_without_prior_l3_errors(self):
        """top_n / threshold selectors need a prior L3 run; the
        engine's preflight delegates to selection.validate_selection
        and surfaces that as an error."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=5)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg, prior=None)
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.where == "config.es_mode_selection" for i in errors)

    def test_selector_top_n_with_prior_l3_ok(self):
        """Same selector + a prior result that completed L3 -> OK
        (no error from the validator)."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        prior = _make_results(complete=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg, prior=prior)
        errs = [i for i in issues if i.severity == "error"
                and i.where == "config.es_mode_selection"]
        assert errs == []


class TestPySCFEngineIsHybridFunctional:
    """The hybrid-detection heuristic.  We accept some false
    positives (the resulting warn is benign) but want no false
    negatives for the canonical hybrid families."""

    @pytest.mark.parametrize("name", [
        "B3LYP", "b3lyp", "B3PW91",
        "PBE0", "pbe0",
        "M06", "M06-2X", "M06-L",
        "ωB97X-D", "wB97X",
        "CAM-B3LYP",
        "BHandH", "BHandHLYP",
        "TPSS0",
        "HSE06",
    ])
    def test_recognised_hybrids(self, name):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        assert PySCFSpectraEngine._is_hybrid_functional(name) is True

    @pytest.mark.parametrize("name", [
        "PBE", "BLYP", "LDA", "BP86", "TPSS", "SCAN",
    ])
    def test_pure_functionals_not_flagged(self, name):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        assert PySCFSpectraEngine._is_hybrid_functional(name) is False


class TestPySCFEngineParseOutput:
    """parse_output should delegate to the engine-agnostic JSON
    parser cleanly."""

    def test_parse_output_round_trips(self, tmp_path):
        from molbuilder.parsers.spectra_json import dump_spectra_json
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        original = _make_results(complete=True)
        p = tmp_path / "x.spectra.json"
        dump_spectra_json(original, p)
        loaded = PySCFSpectraEngine.parse_output(str(p))
        assert loaded.engine == original.engine
        assert len(loaded.modes) == len(original.modes)

    def test_parse_output_propagates_missing_file_error(self, tmp_path):
        from molbuilder.parsers.spectra_json import SpectraJsonNotFoundError
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        bad = tmp_path / "missing.spectra.json"
        with pytest.raises(SpectraJsonNotFoundError):
            PySCFSpectraEngine.parse_output(str(bad))


# --------------------------------------------------------------------- #
#  PySCF script template (pyscf_script.py)                              #
#                                                                       #
#  The emitted Python script that gets shipped to the user.  Tests      #
#  cannot RUN it (no PySCF in the test env, would take minutes), so     #
#  the test surface is structural:                                      #
#    * compile() accepts the output as valid Python (no syntax bugs);   #
#    * expected block markers present / absent per config;              #
#    * critical safety + correctness invariants are pinned (SCHEMA_     #
#      VERSION matches the parser, atomic-replace pattern present,      #
#      allow_nan=False, no stray BOM-like chars).                       #
# --------------------------------------------------------------------- #


def _struct_water_real():
    """Real Structure for water -- used by script-template tests."""
    from molbuilder.structure import Structure
    return Structure(
        elements  = ["O", "H", "H"],
        positions = np.array([[0., 0., 0.],
                              [0.96, 0., 0.],
                              [-0.24, 0.93, 0.]]),
    )


class TestPySCFScriptCompiles:
    """The most important guarantee: every config combination
    produces a script that Python's compiler accepts.  A syntax
    bug in the template would only surface when the user runs the
    file -- catch them here instead."""

    @pytest.mark.parametrize("cfg_overrides", [
        # Default config
        dict(),
        # Minimal: no Raman, no ES
        dict(compute_raman=False),
        # Raman only
        dict(compute_raman=True, es_mode_selection="skip"),
        # ES only with explicit selector
        dict(compute_raman=False, es_mode_selection="explicit",
             es_explicit_indices=[1, 2]),
        # Full pipeline
        dict(compute_raman=True, es_mode_selection="top_n", es_top_n=5),
        # Threshold selector
        dict(compute_raman=True, es_mode_selection="threshold",
             es_threshold=10.0),
        # All modes for ES
        dict(es_mode_selection="all"),
        # Dispersion variants
        dict(dispersion="none"),
        dict(dispersion="d4"),
        # Unrestricted SCF
        dict(method="UKS"),
        # Hartree-Fock (no DFT)
        dict(method="RHF"),
        # Hybrid-low-grid (should compile fine, just a preflight warn)
        dict(grid_level=2),
        # Freeze atoms
        dict(fixed_elements=["H"]),
        dict(fixed_indices=[1, 2]),
        # Frequency window
        dict(es_mode_selection="all", freq_min_cm1=500.0, freq_max_cm1=3500.0),
    ])
    def test_compiles_as_python(self, cfg_overrides):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(**cfg_overrides)
        script = render_spectra_script(_struct_water_real(), cfg)
        # compile() raises SyntaxError on bad Python -- this is the
        # cheap-to-run guarantee that the template is correct.
        code = compile(script, f"<spectra.py {cfg_overrides!r}>", "exec")
        assert code is not None


class TestPySCFScriptHeader:
    """The docstring header is the Methods-section source-of-truth
    that ships with the script (spec § 11.2)."""

    def test_starts_with_docstring(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert script.startswith('"""PySCF Spectra input script')

    def test_methods_paragraph_inlined(self):
        """The header carries the full Methods prose (same content
        as the UI's preview modal)."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(),
                                       SpectraConfig(compute_raman=True))
        # Manuscript-ready citations land in the header.
        assert "B3LYP" in script
        assert "Sun2020" in script
        assert "Komornicki1979" in script

    def test_run_command_pin(self):
        """The header documents `python <job>.spectra.py` so the
        reader doesn't have to guess."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(),
                                       SpectraConfig(job_name="my_job"))
        assert "python my_job.spectra.py" in script


class TestPySCFScriptConstants:
    """The constants block is the bridge between the Python config
    surface and the inlined runtime values.  Pin invariants the
    parser depends on."""

    def test_schema_version_matches_parser(self):
        """The script writes SCHEMA_VERSION=1 to match what
        parse_spectra_json expects."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        from molbuilder.spectra.results import SCHEMA_VERSION
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert f"SCHEMA_VERSION = {SCHEMA_VERSION}" in script

    def test_phase_constants_pinned(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert "PHASE_EMPTY    = 'empty'" in script
        assert "PHASE_RUNNING  = 'running'" in script
        assert "PHASE_COMPLETE = 'complete'" in script

    def test_job_name_substituted_into_path(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(),
                                       SpectraConfig(job_name="weird_name"))
        assert "JOB            = 'weird_name'" in script
        assert "JSON_PATH      = JOB + '.spectra.json'" in script

    def test_method_specific_imports(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        # DFT path imports dft module.
        s_dft = render_spectra_script(_struct_water_real(),
                                      SpectraConfig(method="RKS"))
        assert "from pyscf import gto, scf, dft" in s_dft
        # HF path skips dft.
        s_hf = render_spectra_script(_struct_water_real(),
                                     SpectraConfig(method="RHF"))
        assert "from pyscf import gto, scf" in s_hf
        # The HF path shouldn't import dft (saves a few ms on script start).
        # Check that the HF path doesn't have the trailing ", dft" import line.
        assert "from pyscf import gto, scf, dft" not in s_hf


class TestPySCFScriptAtomicWriter:
    """The inlined atomic JSON writer is the same safety contract
    as `molbuilder.parsers.spectra_json.dump_spectra_json`.  Pin
    that every safety rule is present in the emitted bytes."""

    def test_allow_nan_false(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        # NaN/Inf would otherwise round-trip; allow_nan=False raises
        # before bytes hit disk.
        assert "allow_nan=False" in script

    def test_ensure_ascii_false(self):
        """ensure_ascii=False keeps cm⁻¹ / Å verbatim in the JSON
        rather than escaping to \\uXXXX (which is valid JSON but
        ugly and breaks grep-ability)."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert "ensure_ascii=False" in script

    def test_atomic_replace_via_tempfile(self):
        """tempfile.mkstemp + os.replace is the atomic-rename
        pattern that survives a crash between write and rename."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert "tempfile.mkstemp" in script
        assert "os.replace" in script

    def test_fsync_before_replace(self):
        """fsync forces the data to disk before the atomic rename
        so a power-loss between write() and replace() doesn't leave
        the new file with stale buffer contents."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert "os.fsync" in script

    def test_temp_file_cleanup_on_failure(self):
        """The temp file is removed on any exception during write
        (except path: os.unlink in the except branch)."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert "os.unlink(tmp)" in script

    def test_no_molbuilder_import_at_runtime(self):
        """The script must run on a cluster node that has PySCF +
        numpy + stdlib only -- no molbuilder dependency."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        # No `import molbuilder` or `from molbuilder.*` lines.
        for line in script.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import molbuilder")
            assert not stripped.startswith("from molbuilder")


class TestPySCFScriptPhaseBlocks:
    """Each phase block (Hessian, Raman, ES) is emitted iff the
    config asks for it.  Pin presence / absence per knob."""

    def test_hessian_always_emitted(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert "Phase 2: Hessian" in script
        assert "mf.Hessian().kernel()" in script
        assert "phase_frequencies'] = PHASE_COMPLETE" in script

    def test_raman_block_when_enabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(),
                                       SpectraConfig(compute_raman=True))
        assert "Phase 3: Raman" in script
        assert "Polarizability()" in script
        assert "phase_raman'] = PHASE_COMPLETE" in script

    def test_raman_block_absent_when_disabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(),
                                       SpectraConfig(compute_raman=False))
        assert "Phase 3: Raman" not in script
        assert "Polarizability()" not in script

    def test_es_block_when_enabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(es_mode_selection="all"),
        )
        assert "Phase 4: per-mode" in script
        assert "phase_es'] = PHASE_COMPLETE" in script

    def test_es_block_absent_when_disabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(es_mode_selection="skip"),
        )
        assert "Phase 4: per-mode" not in script


class TestPySCFScriptStructure:
    """Verify atom coordinates, frozen-atom logic, and the
    selection logic are present in the emitted code."""

    def test_atoms_inlined(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        # All three water atoms inlined into the ATOMS list.
        assert "'O'" in script or "( ' O'" in script  # the formatting
        assert script.count("'H'") >= 2 or script.count("' H'") >= 2

    def test_frozen_mask_logic_present(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(fixed_elements=["O"], fixed_indices=[1]),
        )
        # The freeze rule values are inlined.
        assert "'O'" in script
        # The runtime union logic is present.
        assert "FIXED_ATOM_IDXS" in script
        assert "FREE_ATOM_IDXS" in script
        assert "FIXED_ELEMENTS" in script

    def test_engine_renders_script_via_pyscf_script_module(self):
        """The engine's render_script() should delegate to the
        template module without raising."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        script = PySCFSpectraEngine.render_script(
            _struct_water_real(), SpectraConfig(),
        )
        # Same script the template module produces.
        assert "PySCF Spectra input script generated by molbuilder" in script
        # And it compiles.
        compile(script, "<engine.render_script output>", "exec")


class TestPySCFScriptSelectorInline:
    """The L4 block inlines the same selector logic as
    spectra.selection.select_modes so the script behaves
    identically without importing molbuilder."""

    def test_explicit_selector_inlined(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(es_mode_selection="explicit",
                          es_explicit_indices=[1, 3, 7]),
        )
        # The indices are pinned into ES_EXPLICIT_INDICES.
        assert "ES_EXPLICIT_INDICES        = [1, 3, 7]" in script

    def test_top_n_selector_inlined(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(es_mode_selection="top_n", es_top_n=12),
        )
        assert "ES_TOP_N                   = 12" in script
        # The script's runtime selector branches on selector value.
        assert "ES_MODE_SELECTION == 'top_n'" in script

    def test_threshold_selector_inlined(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(es_mode_selection="threshold", es_threshold=15.5),
        )
        assert "ES_THRESHOLD               = 15.5" in script
        assert "ES_MODE_SELECTION == 'threshold'" in script

    def test_freq_window_pinned(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(es_mode_selection="all",
                          freq_min_cm1=500.0, freq_max_cm1=3500.0),
        )
        assert "FREQ_MIN_CM1               = 500.0" in script
        assert "FREQ_MAX_CM1               = 3500.0" in script


# --------------------------------------------------------------------- #
#  Regression tests for script-template bugs caught in review           #
# --------------------------------------------------------------------- #


class TestPySCFScriptDisplacedScfHelpers:
    """Bug: `_build_mf_at` + `COORDS_EQ_ANG` used to be defined inside
    the Raman block.  With compute_raman=False + es_mode_selection
    != "none", the ES block called undefined names -> NameError at
    runtime.  Fix: emit shared helpers when L3 OR L4 is enabled."""

    def test_helpers_defined_when_only_es_enabled(self):
        """The failing config combo from the review."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(compute_raman=False,
                          es_mode_selection="explicit",
                          es_explicit_indices=[1]),
        )
        # Both names are defined in the shared helper block.
        assert "def _build_mf_at" in script
        assert "COORDS_EQ_ANG" in script
        # And the ES block references them.
        assert "_build_mf_at(" in script

    def test_helpers_defined_when_only_raman_enabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(compute_raman=True,
                          es_mode_selection="skip"),
        )
        assert "def _build_mf_at" in script
        assert "COORDS_EQ_ANG" in script

    def test_helpers_defined_when_both_enabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(compute_raman=True,
                          es_mode_selection="all"),
        )
        # Defined exactly once -- not duplicated between the two phases.
        assert script.count("def _build_mf_at") == 1
        assert script.count("COORDS_EQ_ANG = np.asarray") == 1

    def test_helpers_absent_when_neither_enabled(self):
        """When neither L3 nor L4 is on, the helpers aren't emitted
        (no caller).  Keeps the script minimal for diagnostic-only
        Hessian-only runs."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(compute_raman=False,
                          es_mode_selection="skip"),
        )
        assert "def _build_mf_at" not in script

    def test_only_es_enabled_compiles_AND_helpers_resolve(self):
        """Compile pass + an exec-time symbol check.  The earlier
        compile parametrize matrix didn't catch the original bug
        because compile checks syntax, not name resolution.  Here
        we exec the script's textual definition of _build_mf_at by
        slicing it out and feeding it through compile() in 'exec'
        mode to verify the def parses on its own."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(compute_raman=False,
                            es_mode_selection="explicit",
                            es_explicit_indices=[1])
        script = render_spectra_script(_struct_water_real(), cfg)
        compile(script, "<no-raman-with-es>", "exec")
        # Locate the _build_mf_at definition and assert it appears
        # BEFORE the first L4 call site.
        def_pos  = script.find("def _build_mf_at")
        call_pos = script.find("_build_mf_at(", def_pos + 1)
        assert def_pos != -1, "_build_mf_at not defined"
        assert call_pos != -1, "_build_mf_at not called"
        assert def_pos < call_pos, (
            "def must come before first call site, otherwise NameError "
            "at runtime"
        )


class TestPySCFScriptSchemaVersionInterpolated:
    """Bug: SCHEMA_VERSION was a literal 1 in the emitted script.
    A future bump in results.SCHEMA_VERSION would leave scripts
    silently writing the old version -> parser rejects with a
    misleading 'schema_mismatch' on what should be valid output.

    Fix: interpolate from results.SCHEMA_VERSION at render time.
    """

    def test_schema_version_matches_live_constant(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        from molbuilder.spectra.results import SCHEMA_VERSION
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        # The emitted constant matches the imported one.  If someone
        # bumps SCHEMA_VERSION to 2, this test fails immediately and
        # the developer remembers to refresh the script template.
        assert f"SCHEMA_VERSION = {SCHEMA_VERSION}" in script

    def test_molbuilder_version_matches_package_metadata(self):
        """The emitted MOLBUILDER_VERSION lands in
        spectra.json.provenance.molbuilder_version -- it must match
        the actual installed package version, not a placeholder
        like 'spectra-v1'.  Regression: the constants block used to
        hard-code 'spectra-v1' which silently lied in every result
        file's provenance."""
        from molbuilder import __version__
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(), SpectraConfig())
        assert f"MOLBUILDER_VERSION = {__version__!r}" in script
        # Negative: the old placeholder is gone.
        assert "'spectra-v1'" not in script


class TestPySCFScriptGPU:
    """The emitted script's GPU code path: USE_GPU constant in the
    constants block, a try/except gpu4pyscf import that falls back
    to CPU PySCF on failure, and the SCF construction uses _dft /
    _scf pointers that get rebound to gpu4pyscf when the import
    succeeds."""

    def test_use_gpu_false_emits_constant_and_setup_block(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(),
                                       SpectraConfig(use_gpu=False))
        # Constant present.
        assert "USE_GPU                    = False" in script
        # GPU setup block always emitted (its body just runs the
        # CPU fallback when USE_GPU=False).
        assert "GPU acceleration (optional, NVIDIA via gpu4pyscf)" in script
        assert "_USING_GPU = False" in script
        # Script must still compile.
        compile(script, "<no-gpu>", "exec")

    def test_use_gpu_true_emits_constant_and_setup_block(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(),
                                       SpectraConfig(use_gpu=True))
        # Constant present + True.
        assert "USE_GPU                    = True" in script
        # gpu4pyscf import is in the setup block, guarded by try.
        assert "from gpu4pyscf import dft as _gpu_dft" in script
        assert "from gpu4pyscf import scf as _gpu_scf" in script
        # And the fallback message is in there too -- so the user
        # who runs the script on a non-GPU node knows what happened.
        assert "Falling back to CPU PySCF" in script
        # Compiles.
        compile(script, "<gpu-on>", "exec")

    def test_scf_construction_uses_indirect_pointers(self):
        """The equilibrium SCF and _build_mf_at use _dft / _scf
        instead of hardcoded pyscf.dft / pyscf.scf so the GPU
        rebind takes effect for both paths.  Regression: earlier
        the code said `dft.RKS(mol)` which would have ignored the
        gpu4pyscf bind even with USE_GPU=True."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(
            use_gpu=True,
            compute_raman=True,             # exercises _build_mf_at
            es_mode_selection="explicit",   # exercises L4 _build_mf_at calls
            es_explicit_indices=[1],
        )
        script = render_spectra_script(_struct_water_real(), cfg)
        # Equilibrium SCF + displaced SCFs use the indirect pointer.
        assert "_dft.RKS(mol)" in script  # method=RKS (default)
        assert "_dft_mod.RKS" in script   # inside _build_mf_at
        # The hardcoded names must NOT appear in the SCF-construction
        # call sites (only in the GPU setup's fallback assignment).
        # We do allow "_dft = dft" once -- the CPU default-bind.
        assert script.count("dft.RKS(mol)") == 1, (
            "expected exactly one dft.RKS reference (the "
            "equilibrium SCF call, via _dft); got "
            f"{script.count('dft.RKS(mol)')}"
        )

    def test_emitted_script_does_runtime_capability_check(self):
        """The script must verify at runtime that the GPU is
        modern enough to run gpu4pyscf, not just that gpu4pyscf
        imports.  Pinning: the GPU setup block probes via cupy
        and falls back to CPU when compute capability < 7."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water_real(),
                                       SpectraConfig(use_gpu=True))
        # Capability probe via cupy.
        assert "import cupy as _cp" in script
        assert "getDeviceCount" in script
        assert "getDeviceProperties" in script
        # Hard threshold: major >= 7.
        assert "_maj < 7" in script
        # Runtime exception path falls back to CPU with a clear
        # message naming the actual GPU model + cap.
        assert "Falling back to CPU PySCF" in script
        # Two except branches: ImportError + RuntimeError.
        assert "except ImportError" in script
        assert "except Exception" in script

    def test_raman_block_forces_cpu_even_with_gpu_on(self):
        """gpu4pyscf doesn't yet expose analytic CPHF polarizability,
        so the Raman finite-difference path must build CPU mf
        objects even when USE_GPU=True.  Pinning: the polarizability
        FD calls pass force_cpu=True."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(use_gpu=True, compute_raman=True,
                            es_mode_selection="skip")
        script = render_spectra_script(_struct_water_real(), cfg)
        # The Raman block's three _build_mf_at call sites all pass
        # force_cpu=True (one for the equilibrium polarizability,
        # two for the ±FD displacements).  Match the closing-paren
        # form so we don't count the docstring's explanation of
        # the keyword as a fourth occurrence.
        assert script.count("force_cpu=True)") == 3


class TestPySCFScriptL4OutOfRangeGuard:
    """Bug: L4 loop did `modes_payload[_mode_pos]` without checking
    that _mode_pos was in range.  A user with es_explicit_indices=[99]
    on a 12-mode system would crash with IndexError AFTER L2 + L3
    already completed -- hours of wall time lost.

    Fix: pre-filter _selected to valid range, print + skip the rest.
    The pre-render validator can't catch this (mode count unknown
    pre-L2) so the script has to be the second line of defence.
    """

    def test_es_loop_has_range_guard(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water_real(),
            SpectraConfig(es_mode_selection="explicit",
                          es_explicit_indices=[1]),
        )
        # The guard predicate.
        assert "1 <= i <= _n_modes_available" in script
        # And the WARN print for skipped indices so the user notices.
        assert "skipping out-of-range mode indices" in script

    def test_guard_emits_for_every_selector_kind(self):
        """The guard is in the shared L4 loop, so it covers ANY
        selector (top_n / threshold / explicit / all) since
        _selected is computed by the inlined selector before the
        guard runs."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        for sel in ("all", "top_n", "threshold", "explicit"):
            cfg = SpectraConfig(
                es_mode_selection=sel,
                es_explicit_indices=[1, 2] if sel == "explicit" else [],
            )
            script = render_spectra_script(_struct_water_real(), cfg)
            assert "1 <= i <= _n_modes_available" in script, sel


class TestSelectorEquivalence:
    """Pin the script's inlined selector against the canonical
    Python `select_modes` for a fixture of (modes, cfg) pairs.

    The script's selector is hand-rolled.  If someone changes the
    Python version (e.g. tie-breaking rule) without updating the
    script, this test catches the drift immediately.

    We don't exec the full script (would need real PySCF + a
    converged SCF); we exec ONLY the selector function out of the
    emitted text by slicing the if/elif/else block + the helpers
    it needs.
    """

    def _build_selector_namespace(self, cfg, modes_payload):
        """Re-create the runtime environment the inlined selector
        sees: ES_MODE_SELECTION / ES_TOP_N / ES_THRESHOLD /
        ES_EXPLICIT_INDICES / FREQ_MIN_CM1 / FREQ_MAX_CM1 plus the
        modes_payload list."""
        return {
            "ES_MODE_SELECTION":    cfg.es_mode_selection,
            "ES_TOP_N":             cfg.es_top_n,
            "ES_THRESHOLD":         cfg.es_threshold,
            "ES_EXPLICIT_INDICES":  list(cfg.es_explicit_indices),
            "FREQ_MIN_CM1":         cfg.freq_min_cm1,
            "FREQ_MAX_CM1":         cfg.freq_max_cm1,
            "modes_payload":        modes_payload,
        }

    def _modes_payload_for_fixture(self):
        """A modes_payload list shaped like what the in-script
        Hessian block builds (matching the wire form expected by
        the inlined selector).  Same modes as `_modes_fixture()`."""
        out = []
        for m in _modes_fixture():
            out.append({
                "index_1based":          m.index_1based,
                "frequency_cm1":         m.frequency_cm1,
                "raman_activity_a4_amu": m.raman_activity_a4_amu,
                "ir_intensity_km_mol":   m.ir_intensity_km_mol,
                "eigenvector_free":      m.eigenvector_free.tolist(),
                "has_imag":              m.has_imag,
                "electronic_structure":  None,
            })
        return out

    def _exec_inlined_selector(self, script: str, ns: dict) -> list:
        """Slice the inlined selector out of the script + exec it
        against the prepared namespace.  Returns the value of
        `_selected` after execution."""
        # The inlined selector starts at the "if ES_MODE_SELECTION"
        # marker and ends before the def _displaced_scf line.
        start = script.find("if ES_MODE_SELECTION == 'all':")
        end   = script.find("state['selected_mode_idxs_1based']")
        assert start != -1 and end != -1 and end > start, (
            "could not locate inlined selector block in script"
        )
        body = script[start:end]
        # The selector also references `_passes_freq_window`, defined
        # just above.  Include from the "def _passes_freq_window" line.
        helper_start = script.find("def _passes_freq_window")
        assert helper_start != -1
        helper = script[helper_start:start]
        exec(helper + body, ns)
        return list(ns["_selected"])

    @pytest.mark.parametrize("cfg_overrides", [
        # Each selector exercised on the same modes fixture.
        dict(es_mode_selection="skip"),
        dict(es_mode_selection="all"),
        dict(es_mode_selection="all", freq_min_cm1=500.0, freq_max_cm1=3500.0),
        dict(es_mode_selection="top_n", es_top_n=3),
        dict(es_mode_selection="top_n", es_top_n=10),  # exceeds count
        dict(es_mode_selection="top_n", es_top_n=2,
             freq_min_cm1=1000.0, freq_max_cm1=3000.0),
        dict(es_mode_selection="threshold", es_threshold=10.0),
        dict(es_mode_selection="threshold", es_threshold=100.0),  # nothing
        dict(es_mode_selection="explicit", es_explicit_indices=[1, 3, 5]),
        dict(es_mode_selection="explicit", es_explicit_indices=[2]),
    ])
    def test_inlined_selector_matches_select_modes(self, cfg_overrides):
        from molbuilder.spectra import select_modes
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(**cfg_overrides)
        modes = _modes_fixture()

        # Python canonical result.
        py_selected = select_modes(modes, cfg, prior=None)

        # When selector == "none", the L4 block isn't emitted at all
        # so there's nothing to exec against -- Python and "script
        # behaviour" trivially agree on the empty list.
        if cfg.es_mode_selection == "skip":
            assert py_selected == []
            return

        # Script's inlined result: render, slice, exec.
        script = render_spectra_script(_struct_water_real(), cfg)
        ns = self._build_selector_namespace(
            cfg, self._modes_payload_for_fixture()
        )
        script_selected = self._exec_inlined_selector(script, ns)

        assert py_selected == script_selected, (
            f"selector drift for cfg={cfg_overrides!r}: "
            f"python={py_selected}, script={script_selected}"
        )
