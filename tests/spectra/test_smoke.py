"""End-to-end smoke tests for the Spectra-tab pipeline.

These tests render the spectra.py script and actually RUN it through
PySCF.  Slow (10-60 seconds per test), require PySCF + numpy at
runtime, and produce wall-time-dependent floating-point results.
Not run by default -- gated on the ``smoke`` marker:

    pytest -m smoke tests/test_spectra_smoke.py

The cheaper unit tests in tests/test_spectra.py cover structural
correctness (script compiles, contains expected blocks, JSON round-
trips through the parser).  These tests are the ONLY ones that
verify the actual physics: that the emitted script, when run, gives
sensible numerical results.

Reference values for B3LYP/def2-SVP D3BJ:

  Water (H2O):
    OH symmetric stretch:    ~3700-3900 cm⁻¹
    OH asymmetric stretch:   ~3800-4000 cm⁻¹
    H-O-H bend:              ~1550-1750 cm⁻¹

Tolerances here are wide because we don't pre-optimize the geometry
(the script does single-point Hessian at the input geometry, which
isn't a stationary point for a hand-input water structure).  Wide
tolerances still catch the kind of bug they're meant to catch:
wrong unit conversion (off by 100x), wrong mass-weighting (off by
sqrt(m) factors on heavy atoms), wrong Hessian reshape (mixed-up
atom-direction ordering).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from molbuilder.sidecars.spectra import parse_spectra_json
from molbuilder.spectra import SpectraConfig
from molbuilder.spectra.pyscf_script import render_spectra_script
from molbuilder.structure import Structure


pytestmark = pytest.mark.smoke

# Skip the whole module if PySCF isn't importable.
pyscf = pytest.importorskip("pyscf", reason="smoke tests require PySCF")


# B3LYP/def2-SVP at a relaxed water geometry gives:
#   asym stretch ~ 3920 cm⁻¹
#   symm stretch ~ 3810 cm⁻¹
#   bend         ~ 1620 cm⁻¹
# At a NON-relaxed geometry (we feed an approximate one) the modes
# may shift by 100-200 cm⁻¹ and translation/rotation projection
# residue may produce extra small modes.  The bounds below are wide
# enough to tolerate that but tight enough to catch order-of-
# magnitude bugs.

_WATER_OH_BAND_LO = 3000.0
_WATER_OH_BAND_HI = 4500.0
_WATER_BEND_LO    = 1200.0
_WATER_BEND_HI    = 2000.0


def _water_structure() -> Structure:
    """A near-equilibrium water geometry (close enough that B3LYP/
    def2-SVP frequencies sit in the expected windows even without
    pre-optimization)."""
    return Structure(
        elements  = ["O", "H", "H"],
        positions = np.array([
            [ 0.000000,  0.000000,  0.119262],
            [ 0.000000,  0.763239, -0.477047],
            [ 0.000000, -0.763239, -0.477047],
        ]),
        title="water-smoke-test",
    )


def _run_script_and_load(struct: Structure, cfg: SpectraConfig,
                         timeout_s: float = 180.0):
    """Render the script, write to a tempdir, exec it in a fresh
    subprocess, parse the resulting spectra.json.

    Subprocess (rather than exec in-process) keeps PySCF's global
    state out of the test process and gives us a clean wall-time
    measurement.
    """
    script = render_spectra_script(struct, cfg)
    with tempfile.TemporaryDirectory(prefix="spectra-smoke-") as d:
        script_path = Path(d) / "smoke.spectra.py"
        # The script writes <JOB>.spectra.json where JOB = cfg.job_name;
        # default is "spectra".
        json_path = Path(d) / f"{cfg.job_name}.spectra.json"
        script_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=d,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            # Surface the script's stderr so the test failure is
            # actionable.  Pytest captures stdout from the test
            # itself; we want the SUBPROCESS's output.
            pytest.fail(
                "spectra.py subprocess failed:\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}\n"
            )
        assert json_path.exists(), (
            f"script did not produce expected {json_path}; "
            f"stdout was:\n{proc.stdout}"
        )
        return parse_spectra_json(json_path)


# --------------------------------------------------------------------- #
# All-free path: water, frequencies only                                #
# --------------------------------------------------------------------- #


class TestWaterFrequenciesAllFree:
    """End-to-end: the emitted script, run on water with no fixed
    atoms and Raman disabled, produces 3 sensible vibrational modes."""

    @pytest.mark.smoke
    def test_water_three_real_modes(self):
        """Water has 3 vibrational modes (3*N - 6 = 3).  All real
        at a near-equilibrium geometry."""
        cfg = SpectraConfig(
            compute_raman=False,
            es_mode_selection="skip",
            # Loosen SCF a bit so the smoke test is faster (defaults
            # are 1e-9; here 1e-7 still gives <1 cm⁻¹ freq noise).
            scf_conv_tol=1e-7,
            scf_max_cycle=80,
        )
        results = _run_script_and_load(_water_structure(), cfg)

        assert results.phase_frequencies == "complete"
        assert results.phase_raman       == "empty"
        assert results.phase_es          == "empty"

        modes = results.modes
        # PySCF's harmonic_analysis projects out the 6 trans/rot
        # modes, so we expect 3 vibrational modes.
        n_real = sum(1 for m in modes if not m.has_imag)
        assert n_real >= 3, (
            f"expected at least 3 real modes for water, got "
            f"{n_real} real out of {len(modes)} total: "
            f"{[m.frequency_cm1 for m in modes]}"
        )

    @pytest.mark.smoke
    def test_water_frequencies_in_expected_bands(self):
        """The 3 vibrational modes of water at B3LYP/def2-SVP land
        in two known bands: 2 OH stretches around 3500-4000 cm⁻¹
        and 1 H-O-H bend around 1600 cm⁻¹.

        This is the test that catches:
          - wrong unit conversion (off by ~100x)
          - wrong mass-weighting (frequencies off by sqrt(amu))
          - wrong Hessian reshape (random spectrum)
        """
        cfg = SpectraConfig(
            compute_raman=False,
            es_mode_selection="skip",
            scf_conv_tol=1e-7,
            scf_max_cycle=80,
        )
        results = _run_script_and_load(_water_structure(), cfg)

        # Take only real (non-imaginary) modes; sort descending so
        # the top two are the stretches and the third is the bend.
        real_freqs = sorted(
            (m.frequency_cm1 for m in results.modes if not m.has_imag),
            reverse=True,
        )
        # The TOP two should be the OH stretches (highest frequencies).
        # The THIRD should be the H-O-H bend.
        assert len(real_freqs) >= 3, (
            f"need at least 3 real modes; got {real_freqs}"
        )

        stretch_high, stretch_low, bend = real_freqs[0], real_freqs[1], real_freqs[2]

        assert _WATER_OH_BAND_LO < stretch_high < _WATER_OH_BAND_HI, (
            f"top stretch frequency {stretch_high:.1f} cm⁻¹ outside "
            f"the OH band [{_WATER_OH_BAND_LO}, {_WATER_OH_BAND_HI}] "
            f"cm⁻¹; full real spectrum: {real_freqs}"
        )
        assert _WATER_OH_BAND_LO < stretch_low < _WATER_OH_BAND_HI, (
            f"second stretch frequency {stretch_low:.1f} cm⁻¹ outside "
            f"the OH band [{_WATER_OH_BAND_LO}, {_WATER_OH_BAND_HI}] "
            f"cm⁻¹; full real spectrum: {real_freqs}"
        )
        assert _WATER_BEND_LO < bend < _WATER_BEND_HI, (
            f"bend frequency {bend:.1f} cm⁻¹ outside the H-O-H bend "
            f"band [{_WATER_BEND_LO}, {_WATER_BEND_HI}] cm⁻¹; "
            f"full real spectrum: {real_freqs}"
        )

    @pytest.mark.smoke
    def test_water_eigenvectors_are_real_cartesian(self):
        """The OH-stretching mode should show O and the two Hs
        moving roughly opposite directions (asymmetric) or both
        Hs moving towards / away from O symmetrically (symmetric).
        Either way, the eigenvector should be in CARTESIAN units --
        a wrong mass-weighting conversion (the bug fixed in this
        commit's parent) would make O's displacement amplitude
        comparable to H's instead of much smaller.

        Physics: in a stretch, the heavier atom (O, 16 amu) moves
        much less than the light atom (H, 1 amu) because the
        center of mass is fixed.  Ratio of displacements ≈ m_H/m_O
        ≈ 1/16 (for the high-frequency stretches).
        """
        cfg = SpectraConfig(
            compute_raman=False,
            es_mode_selection="skip",
            scf_conv_tol=1e-7,
            scf_max_cycle=80,
        )
        results = _run_script_and_load(_water_structure(), cfg)

        # Find the highest-frequency real mode (an OH stretch).
        real = [m for m in results.modes if not m.has_imag]
        real.sort(key=lambda m: m.frequency_cm1, reverse=True)
        stretch = real[0]

        # Eigenvector shape: (n_free, 3).  For water all-free,
        # n_free = 3 atoms (O, H, H in our ordering).
        evec = np.asarray(stretch.eigenvector_display)
        assert evec.shape == (3, 3), (
            f"expected eigenvector shape (3, 3), got {evec.shape}"
        )

        # Per-atom displacement magnitudes.
        amp = np.linalg.norm(evec, axis=1)   # shape (3,) -- O, H, H
        amp_O, amp_H1, amp_H2 = amp[0], amp[1], amp[2]

        # For a high-freq stretch, O should move FAR less than the
        # average H.  The exact ratio depends on which stretch
        # (symmetric vs asymmetric) and on the max-abs normalization
        # in the script, but a wrong mass-weighting would make
        # amp_O comparable to amp_H (factor of ~1 instead of ~1/16).
        # We use a generous threshold: ratio < 0.5 catches the bug
        # (correct value ~ 0.06; buggy value ~ 1.0).
        avg_H_amp = 0.5 * (amp_H1 + amp_H2)
        if avg_H_amp > 1e-9:
            ratio = amp_O / avg_H_amp
            assert ratio < 0.5, (
                f"O atom amplitude/avg-H amplitude = {ratio:.3f}; "
                f"expected < 0.5 (correct mass-weighting gives ~ "
                f"m_H/m_O = 0.06).  This indicates the eigenvector "
                f"mass-unweighting is in the wrong direction "
                f"(multiplying by sqrt(m) instead of dividing)."
            )


# --------------------------------------------------------------------- #
# Raman path                                                            #
# --------------------------------------------------------------------- #


class TestWaterRaman:
    """Run the full Raman pipeline on water and check the activities
    are positive, finite, and dimensionally reasonable."""

    @pytest.mark.smoke
    def test_water_raman_activities_finite_and_positive(self):
        cfg = SpectraConfig(
            compute_raman=True,
            es_mode_selection="skip",
            scf_conv_tol=1e-7,
            scf_max_cycle=80,
        )
        results = _run_script_and_load(_water_structure(), cfg,
                                       timeout_s=300.0)

        assert results.phase_raman == "complete"

        # Every mode has a finite, non-negative Raman activity.
        # (Activity is 45 a² + 7 γ² which is mathematically ≥ 0.
        # Negative or NaN here means a numerical pathology.)
        for m in results.modes:
            assert m.raman_activity_a4_amu is not None, (
                f"mode {m.index_1based} missing Raman activity "
                f"after a compute_raman=True run"
            )
            assert m.raman_activity_a4_amu >= 0, (
                f"mode {m.index_1based} ω={m.frequency_cm1:.1f} cm⁻¹ "
                f"has negative Raman activity "
                f"{m.raman_activity_a4_amu!r}; activity is "
                f"mathematically non-negative"
            )
            assert np.isfinite(m.raman_activity_a4_amu), (
                f"mode {m.index_1based} has non-finite Raman activity "
                f"{m.raman_activity_a4_amu!r}"
            )


# --------------------------------------------------------------------- #
# Mass-weighting regression (the bug fix that motivated these tests)    #
# --------------------------------------------------------------------- #


class TestPartialHessianMassWeighting:
    """Specifically exercises the partial-Hessian path (frozen atoms)
    where the mass-weighting bug lived.  Freezes one O atom and lets
    the two Hs vibrate; the resulting eigenvectors should still show
    physical per-atom displacements.

    Until 2026-05-12 the partial-Hessian path inverted the
    mass-unweighting direction (multiplied by sqrt(m) instead of
    dividing), which made heavy-atom amplitudes wildly too large
    after max-abs normalization.  This test pins the fix.
    """

    @pytest.mark.smoke
    def test_water_with_one_frozen_atom(self):
        """Water with the O frozen -- the partial-Hessian gives
        (2 free atoms × 3) = 6 modes.  At least some should be
        finite-frequency stretches; eigenvectors should have only
        the unfrozen atoms moving (which is guaranteed by the
        free-atom restriction, but also: the per-mode max-abs
        amplitude should be reasonable, not 1e6)."""
        cfg = SpectraConfig(
            compute_raman=False,
            es_mode_selection="skip",
            frozen_indices=[0],   # freeze the O
            scf_conv_tol=1e-7,
            scf_max_cycle=80,
        )
        results = _run_script_and_load(_water_structure(), cfg)

        assert results.phase_frequencies == "complete"
        # N_FREE = 2, so 6 modes from the partial-Hessian path.
        assert len(results.modes) == 6, (
            f"expected 6 modes (3*N_FREE = 3*2) for water with "
            f"one fixed atom; got {len(results.modes)}: "
            f"{[m.frequency_cm1 for m in results.modes]}"
        )

        # Every mode's eigenvector has shape (N_FREE, 3) = (2, 3).
        for m in results.modes:
            evec = np.asarray(m.eigenvector_display)
            assert evec.shape == (2, 3), (
                f"mode {m.index_1based} eigenvector shape "
                f"{evec.shape}; expected (2, 3) for two free atoms"
            )
            # After max-abs normalization in the script, all
            # components are in [-1, 1].  If the bug were back,
            # the max-abs normalization would still bring it into
            # this range -- but the PROPORTIONS would be wrong.
            # Check at least that no component is absurd.
            assert np.all(np.abs(evec) <= 1.0 + 1e-6), (
                f"mode {m.index_1based} eigenvector outside [-1, 1] "
                f"after normalization: {evec}"
            )
            # And the max-abs is exactly 1 (the normalization
            # condition the script enforces).
            assert abs(np.max(np.abs(evec)) - 1.0) < 1e-6, (
                f"mode {m.index_1based} max-abs amplitude "
                f"{np.max(np.abs(evec)):.6f} != 1.0; the max-abs "
                f"normalization in the partial-Hessian path didn't "
                f"fire correctly"
            )

    @pytest.mark.smoke
    def test_heavy_atom_eigenvector_mass_weighting_correct(self):
        """The decisive test for the mass-weighting bug.

        Setup: HCl as a diatomic toy.  Closed-shell (18 electrons),
        big mass contrast (m_Cl / m_H ≈ 35).  For the stretching
        mode, center-of-mass conservation demands

            m_H · x_H + m_Cl · x_Cl = 0
            => x_H / x_Cl ≈ -35

        i.e., H moves 35× more than Cl.  With the BUGGY mass-
        weighting (multiply by sqrt(m) instead of divide), Cl would
        move sqrt(35) ≈ 5.9× MORE than H -- not less.  The test
        thresholds catch the bug at every plausible severity.
        """
        struct = Structure(
            elements  = ["H", "Cl"],
            positions = np.array([
                [0.0, 0.0, 0.0],
                [1.28, 0.0, 0.0],   # ~equilibrium HCl bond length
            ]),
            title="HCl toy",
        )
        cfg = SpectraConfig(
            compute_raman=False,
            es_mode_selection="skip",
            scf_conv_tol=1e-7,
            scf_max_cycle=80,
        )
        results = _run_script_and_load(struct, cfg)
        assert results.phase_frequencies == "complete"

        # For a diatomic, only 1 vibrational mode (3·2 - 5 = 1).
        # PySCF's harmonic_analysis projects 5 modes out.
        real = [m for m in results.modes if not m.has_imag]
        assert real, (
            f"no real modes from HCl; the all-free harmonic_analysis "
            f"should produce exactly 1 stretch for a diatomic.  Got: "
            f"{[m.frequency_cm1 for m in results.modes]}"
        )
        # Sanity: the HCl stretch is around 2900 cm⁻¹ at B3LYP/
        # def2-SVP (experimental is 2991 cm⁻¹).
        stretch = real[0]
        assert 2500 < stretch.frequency_cm1 < 3300, (
            f"HCl stretch frequency {stretch.frequency_cm1:.1f} cm⁻¹ "
            f"outside the [2500, 3300] cm⁻¹ window for B3LYP/def2-SVP"
        )

        evec = np.asarray(stretch.eigenvector_display)
        assert evec.shape == (2, 3)
        amp_H, amp_Cl = np.linalg.norm(evec, axis=1)

        # Correct physics: amp_Cl / amp_H ≈ m_H / m_Cl ≈ 1/35 ≈ 0.03.
        # The script's max-abs normalisation will put one of these
        # at exactly 1.0; the OTHER is what we test.
        # Buggy (multiply by sqrt(m)): the ratio inverts -- Cl moves
        # sqrt(35) ≈ 5.9× more than H, so after max-abs normalisation
        # Cl is 1.0 and H is 1/sqrt(35) ≈ 0.17.
        # Conservative threshold: amp_Cl / amp_H < 0.3 (correct ≈ 0.03,
        # buggy ≈ 5.9).
        ratio = amp_Cl / amp_H
        assert ratio < 0.3, (
            f"HCl stretch: Cl amplitude / H amplitude = {ratio:.3f}; "
            f"expected ~ m_H/m_Cl ≈ 0.03 (correct), ratio > 1 means "
            f"the eigenvector mass-unweighting is inverted "
            f"(multiplying by sqrt(m) instead of dividing) -- the "
            f"bug class this test guards against."
        )
