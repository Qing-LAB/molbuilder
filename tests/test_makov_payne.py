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
    MADELUNG_CUBIC,
    BOHR_ANGSTROM,
    HARTREE_EV,
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
        """At q=+1, ε=1, L=15 Å the leading-term correction is
        ~1.36 eV.  This pins the formula constant in eV rather than
        Hartree."""
        dE = compute_correction(q=1, L_angstrom=15.0, epsilon_r=1.0)
        # α/(2 L_bohr) in Hartree = 2.8373 / (2 * 28.345) ≈ 0.05005
        # Times HARTREE_EV = 27.211 → ≈ 1.362 eV.
        assert 1.35 < dE < 1.40

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

    def test_script_runs_on_synthetic_out(self):
        """End-to-end: write a tiny fake SIESTA .out with a known
        E_KS + LatticeVectors block, run the script, parse its
        stdout, verify the corrected energy is E_raw - ΔE_MP."""
        s = render_correction_script(system_label="job", q=1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "makov_payne_correction.py").write_text(s)
            # Fake .out with the two lines the script needs.
            fake_out = (
                "siesta: ----- Final results -----\n"
                "siesta: E_KS(eV) =       -123.4567\n"
                "outcell: Unit cell vectors (Ang):\n"
                "       15.0000  0.0000  0.0000\n"
                "        0.0000 15.0000  0.0000\n"
                "        0.0000  0.0000 15.0000\n"
            )
            (d / "job.out").write_text(fake_out)
            result = subprocess.run(
                [sys.executable, "makov_payne_correction.py"],
                cwd=str(d),
                capture_output=True, text=True, timeout=10,
            )
            assert result.returncode == 0, (
                "script failed:\nstdout:\n" + result.stdout
                + "\nstderr:\n" + result.stderr
            )
            out = result.stdout
            assert "E_total (raw, eV)" in out
            assert "E_total (corrected, eV)" in out
            # Spot-check the numeric correction: V = 15³ → L = 15.
            # ΔE = 2.8373 / (2 × 15 / 0.529) × 27.211 → ≈ 1.362 eV.
            # E_corrected = -123.4567 - 1.362 → ≈ -124.82.
            lines = [ln for ln in out.splitlines()
                     if "E_total (corrected" in ln]
            assert lines, "expected an E_total (corrected) line"
            corrected = float(lines[0].split("=")[-1].strip())
            assert -125.0 < corrected < -124.5

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

    def test_script_handles_unparseable_out(self):
        s = render_correction_script(system_label="job", q=1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "makov_payne_correction.py").write_text(s)
            (d / "job.out").write_text("garbage no E_KS or outcell\n")
            result = subprocess.run(
                [sys.executable, "makov_payne_correction.py"],
                cwd=str(d),
                capture_output=True, text=True, timeout=10,
            )
            assert result.returncode != 0

    def test_outcell_parser_tolerates_blank_line(self):
        """6th-review A1: the outcell parser must skip a blank line
        between the marker and the lattice rows (some SIESTA builds
        print one).  Before the fix the script bailed out with
        'could not parse outcell' on a perfectly-good .out.
        """
        s = render_correction_script(system_label="job", q=1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "makov_payne_correction.py").write_text(s)
            (d / "job.out").write_text(
                "siesta: E_KS(eV) =       -100.0\n"
                "outcell: Unit cell vectors (Ang):\n"
                "\n"  # blank line — was the bug
                "       15.0 0.0 0.0\n"
                "        0.0 15.0 0.0\n"
                "        0.0  0.0 15.0\n"
            )
            r = subprocess.run(
                [sys.executable, "makov_payne_correction.py"],
                cwd=str(d), capture_output=True, text=True,
                timeout=10,
            )
            assert r.returncode == 0, r.stderr
            assert "E_total (corrected, eV)" in r.stdout

    def test_outcell_parser_case_insensitive(self):
        """6th-review A2: real SIESTA builds emit OUTCELL or
        outcell.  The script must match either."""
        s = render_correction_script(system_label="job", q=1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "makov_payne_correction.py").write_text(s)
            (d / "job.out").write_text(
                "siesta: E_KS(eV) =       -100.0\n"
                "OUTCELL: Unit cell vectors (Ang):\n"
                "       15.0 0.0 0.0\n"
                "        0.0 15.0 0.0\n"
                "        0.0  0.0 15.0\n"
            )
            r = subprocess.run(
                [sys.executable, "makov_payne_correction.py"],
                cwd=str(d), capture_output=True, text=True,
                timeout=10,
            )
            assert r.returncode == 0, r.stderr

    def test_script_header_warns_about_slab_crystal(self):
        """6th-review B3: the emit fires unconditionally on
        NetCharge != 0, but Makov-Payne is the WRONG correction for
        a charged crystal (3-D periodic with the unit cell carrying
        net charge) OR a charged slab (2-D + vacuum normal).  The
        script header must surface this applicability caveat so a
        user running on a slab doesn't paste the wrong number into
        a manuscript."""
        s = render_correction_script(system_label="job", q=1)
        assert "vacuum supercell" in s.lower()
        assert "slab" in s.lower()
        assert "crystal" in s.lower()
        # Mentions the 2-D dipole/quadrupole alternative for slabs.
        assert "SlabDipoleCorrection" in s


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
# --------------------------------------------------------------------- #


class TestSiestaIntegration:
    def test_charged_writes_script(self):
        """``write_siesta_input`` / ``convert`` drops the post-process
        script next to the FDF whenever the input deck carries a
        non-zero net charge."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.siesta.input import convert
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            xyz = d / "in.xyz"
            xyz.write_text(
                "5\nNH4+\nN 0 0 0\n"
                "H 0.5 0.5 0.5\nH -0.5 0.5 0.5\n"
                "H 0.5 -0.5 0.5\nH 0.5 0.5 -0.5\n"
            )
            fdf = d / "job.fdf"
            cfg = SiestaConfig(system_label="nh4plus", net_charge=1)
            summary = convert(str(xyz), str(fdf), config=cfg)
            assert "makov_payne_script" in summary
            mp = Path(summary["makov_payne_script"])
            assert mp.is_file()
            # The script's NET_CHARGE matches what the wrapper saw.
            text = mp.read_text()
            assert "NET_CHARGE   = 1" in text

    def test_neutral_skips_script(self):
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.siesta.input import convert
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            xyz = d / "in.xyz"
            xyz.write_text("1\nh\nH 0 0 0\n")
            fdf = d / "job.fdf"
            cfg = SiestaConfig(system_label="h", net_charge=0)
            summary = convert(str(xyz), str(fdf), config=cfg)
            assert "makov_payne_script" not in summary
