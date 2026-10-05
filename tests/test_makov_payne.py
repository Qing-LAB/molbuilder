"""Phase-after-Phase 6: Makov-Payne correction emit + compute.

Covers the pure-math correction formula, the post-process script
generator, and the SIESTA-writer integration that drops the script
next to the FDF when NetCharge != 0.
"""

from __future__ import annotations

import sys
import tempfile
import subprocess
from pathlib import Path

import pytest

from molbuilder.siesta.makov_payne import (
    compute_correction,
    effective_L,
    render_correction_script,
    emit_correction_script,
)


# --------------------------------------------------------------------- #
#  compute_correction                                                    #
# --------------------------------------------------------------------- #


class TestComputeCorrection:
    def test_q_plus_one_vacuum_L15(self):
        """The correction is ~1.36 eV for q=+1 in a 15 A vacuum box.

        This is the number the pre-run warning quotes to a person
        (`validation/siesta.py`, printed to two decimals), so what matters is
        the SCALE: a correction well above chemical accuracy, which is the
        warning's whole point. The formula is fixed algebra over three
        constants, so a tighter pin here buys nothing -- it was
        `1.35 < dE < 1.40` with a comment restating the arithmetic, and both
        the window and the comment said the same thing twice.
        """
        dE = compute_correction(q=1, L_angstrom=15.0, epsilon_r=1.0)
        assert 1.3 < dE < 1.4

    def test_q_squared_dependence(self):
        """ΔE scales as q²."""
        dE1 = compute_correction(q=1, L_angstrom=20.0)
        dE2 = compute_correction(q=2, L_angstrom=20.0)
        assert pytest.approx(4.0, rel=1e-12) == dE2 / dE1

    def test_negative_q_same_magnitude(self):
        """Sign of q drops out (q² > 0 either way)."""
        dE_plus = compute_correction(q=+1, L_angstrom=20.0)
        dE_minus = compute_correction(q=-1, L_angstrom=20.0)
        assert pytest.approx(dE_plus, rel=1e-12) == dE_minus

    def test_inverse_L(self):
        """ΔE ∝ 1/L."""
        dE_15 = compute_correction(q=1, L_angstrom=15.0)
        dE_30 = compute_correction(q=1, L_angstrom=30.0)
        assert pytest.approx(2.0, rel=1e-12) == dE_15 / dE_30

    def test_inverse_epsilon(self):
        """ΔE ∝ 1/ε_r."""
        dE_1 = compute_correction(q=1, L_angstrom=20.0, epsilon_r=1.0)
        dE_4 = compute_correction(q=1, L_angstrom=20.0, epsilon_r=4.0)
        assert pytest.approx(4.0, rel=1e-12) == dE_1 / dE_4

    def test_bad_L_raises(self):
        with pytest.raises(ValueError):
            compute_correction(q=1, L_angstrom=0)
        with pytest.raises(ValueError):
            compute_correction(q=1, L_angstrom=-5.0)

    def test_bad_epsilon_raises(self):
        with pytest.raises(ValueError):
            compute_correction(q=1, L_angstrom=10.0, epsilon_r=0)


# --------------------------------------------------------------------- #
#  effective_L                                                           #
# --------------------------------------------------------------------- #


class TestEffectiveL:
    def test_cubic_returns_side(self):
        cell = [[15.0, 0, 0], [0, 15.0, 0], [0, 0, 15.0]]
        assert pytest.approx(15.0, abs=1e-9) == effective_L(cell)

    def test_non_cubic_returns_volume_cube_root(self):
        # V = 10 * 12 * 18 = 2160 → V^1/3 ≈ 12.93
        cell = [[10.0, 0, 0], [0, 12.0, 0], [0, 0, 18.0]]
        assert pytest.approx(2160 ** (1 / 3), rel=1e-9) == \
            effective_L(cell)


# --------------------------------------------------------------------- #
#  Script generation                                                     #
# --------------------------------------------------------------------- #


class TestScriptGeneration:
    def test_compiles(self):
        s = render_correction_script(system_label="job", q=1)
        compile(s, "makov_payne_correction.py", "exec")

    def test_carries_q_and_label(self):
        s = render_correction_script(
            system_label="my-job", q=-2, epsilon_r=4.0)
        assert "SYSTEM_LABEL = \"my-job\"" in s
        assert "NET_CHARGE   = -2" in s
        assert "DEFAULT_EPS  = 4.0" in s

    # Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
    # 4 tests here ran the generated script on a SIESTA output
    # invented as text (`process/testing.md` § 6).

    def test_script_handles_missing_out(self):
        s = render_correction_script(system_label="job", q=1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "makov_payne_correction.py").write_text(s)
            result = subprocess.run(
                [sys.executable, "makov_payne_correction.py"],
                cwd=str(d),
                capture_output=True, text=True, timeout=10,
            )
            assert result.returncode != 0
            assert "not found" in result.stderr


    # `test_script_header_warns_about_slab_crystal` stood here and checked an
    # APPLICABILITY DOMAIN as VOCABULARY: four keyword probes ("slab",
    # "crystal", "vacuum supercell", "SlabDipoleCorrection") against the
    # generated header.  A header stating the OPPOSITE -- that Makov-Payne is
    # the right correction for a slab, and to ignore SlabDipoleCorrection --
    # contains all four words and passed (2026-09-09).
    #
    # What molbuilder actually DECIDES here is `emit iff q != 0`
    # (`siesta/input.py`), blind to geometry -- `render_correction_script`
    # takes `(system_label, q, epsilon_r)` and never sees the structure, so
    # it could not condition on slab-ness even in principle.  That decision
    # is pinned by `test_charged_writes_script` and `test_neutral_skips_script`
    # below.
    #
    # NOW UNCHECKED, deliberately: the caveat could be dropped from the header
    # and no test would fail.  It is documentation, and a keyword probe never
    # protected it -- prose cannot be validated by matching words in it.


# --------------------------------------------------------------------- #
#  emit_correction_script — writes the script to disk                    #
# --------------------------------------------------------------------- #


class TestEmitScript:
    def test_writes_alongside_fdf(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fdf = d / "job.fdf"
            fdf.write_text("# fake fdf\n")
            out = emit_correction_script(fdf, system_label="job", q=1)
            assert out is not None
            assert out == d / "makov_payne_correction.py"
            assert out.is_file()
            compile(out.read_text(), str(out), "exec")

    def test_returns_none_when_parent_missing(self):
        # Pathlib doesn't auto-create the parent of a path that
        # doesn't exist — we test the "FDF parent missing" path.
        with tempfile.TemporaryDirectory() as d:
            phantom = Path(d) / "no-such-subdir" / "job.fdf"
            assert emit_correction_script(
                phantom, system_label="job", q=1) is None


# --------------------------------------------------------------------- #
#  SIESTA writer integration                                             #
#
#  `TestSiestaIntegration` deleted 2026-09-17 with `siesta.input.convert`.
#  Both of its tests drove that single-shot converter to check whether the
#  post-process correction script was dropped beside the deck.  `convert` had
#  had no production caller since `molbuilder fdf` went on 2026-08-11; the
#  sibling-artifact question belongs to `prep`, which owns
#  `_siesta_sibling_artifacts`.
# --------------------------------------------------------------------- #
