"""Tests for the Spectra-tab L1 result types.

Pins the JSON wire shape + the engine-agnostic dataclass surface
documented in ``docs/tabs/spectra/spec.md`` § 5 - § 6.  These tests
are runtime-cheap: no PySCF, no SCF, no Hessian.  They protect:

  * round-trip fidelity (typed -> dict -> JSON -> dict -> typed
    equals the original);
  * forward compatibility of the ``from_dict`` classmethods
    (extra wire keys are ignored, missing optional keys default
    sensibly);
  * the ``complete`` flag semantics for the Option B (live-watch)
    phase-checkpoint model.

PySCF-side smoke tests for actual Hessian + Raman activities live
in ``tests/test_spectra_smoke.py`` (to be added when the
PySCFSpectraEngine lands; marked with ``@pytest.mark.smoke``).
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
    return SpectraResults(
        schema_version             = SCHEMA_VERSION,
        engine                     = "pyscf",
        engine_version             = "2.6.0",
        molbuilder_version         = "1.2.0",
        timestamp                  = "2026-05-11T12:00:00Z",
        structure_hash             = "sha256:abc123",
        n_atoms_total              = 3,
        free_atom_idxs             = [0, 1, 2],
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
        complete                   = complete,
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
        assert r2.complete is True
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
        """Option B live-watch: an in-progress run has complete=False
        and may have empty methods_text / bibliography_keys."""
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
            methods_text               = "",         # populated only at complete=True
            bibliography_keys          = [],
            complete                   = False,
        )
        r2 = SpectraResults.from_dict(json.loads(json.dumps(r.to_dict())))
        assert r2.complete is False
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
            "complete":           False,
            # Intentionally omitted: methods_text, bibliography_keys,
            # selected_mode_idxs_1based, engine_metadata.
        }
        r = SpectraResults.from_dict(d)
        assert r.methods_text == ""
        assert r.bibliography_keys == []
        assert r.selected_mode_idxs_1based == []
        assert r.engine_metadata == {}

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
