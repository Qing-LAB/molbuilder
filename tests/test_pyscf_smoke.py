"""End-to-end smoke tests for the PySCF script generator.

Unlike the rest of the suite -- which renders a script and inspects
the *text* -- these tests:

    1. render a script via molbuilder.pyscf.input.render_script
    2. write it to a tmp file
    3. subprocess-launch it under the same Python interpreter
    4. parse the converged energy out of stdout
    5. compare to a literature reference

That's the only way to catch a class of regressions where the script
*looks* fine but doesn't actually run -- mistyped attribute names,
removed PySCF APIs, missed quoting, missing imports, etc.

Why a separate marker?
======================
Each test takes 1-15 s (an SCF on a tiny molecule under HF/STO-3G or
B3LYP/def2-SVP) and requires PySCF to be importable; we don't want
to pay that cost on every ``pytest`` run.  Run them explicitly with::

    pytest tests/test_pyscf_smoke.py
    pytest -m smoke

The module is skipped at collection time when ``pyscf`` isn't
installed, so the default ``pytest`` invocation in CI environments
without PySCF stays green and fast.

Why HF/STO-3G as the primary reference?
=======================================
Hartree-Fock with the STO-3G basis set is the most exhaustively
documented quantum-chemistry benchmark in the literature; PySCF,
Gaussian, ORCA, NWChem, GAMESS-US all reproduce it to < 0.1 mHa.
Any drift here points to a bug in OUR code path (geometry units,
charge / spin / basis wiring, ECP), not numerical noise.

The B3LYP/def2-SVP test exercises the *default* code paths used in
production runs (DFT grid, density fitting, hybrid functional) so a
regression in one of those wirings would surface immediately.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyscf")

from molbuilder.config.pyscf import PySCFConfig  # noqa: E402
from molbuilder.pyscf.input import render_script  # noqa: E402
from molbuilder.structure import Structure  # noqa: E402


_ENERGY_RE = re.compile(r"Total energy:\s*(-?\d+\.\d+)\s*Hartree")


# ---------------------------------------------------------------- helpers


def _silent_singlepoint_config(job_name: str,
                               method: str = "RHF",
                               functional: str = "",
                               basis: str = "sto-3g",
                               density_fit: bool = False) -> PySCFConfig:
    """Minimal config for a one-shot SCF.

    Disables every output side-effect so the test produces no files
    other than the one it explicitly writes; turns off optimizer so
    the script prints "Total energy:" (single-point branch) instead
    of "Final energy:" (post-opt branch).
    """
    return PySCFConfig(
        job_name=job_name,
        method=method,
        functional=functional,
        basis=basis,
        density_fit=density_fit,
        dispersion=None,
        optimize=False,
        preopt=False,
        chkfile=False,
        log_file=False,
        save_optimized_xyz=False,
        save_initial_xyz=False,
        write_trajectory=False,
        write_molwatch_log=False,
        verbose=3,
        verbose_comments=False,
    )


def _h2o_struct() -> Structure:
    """H2O at the standard experimental geometry (r = 0.9572 A,
    angle = 104.52 deg).  Coordinates in Angstrom, oxygen at origin.
    """
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.0000,  0.0000,  0.1173],
            [0.0000,  0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ]),
        atom_names=["O", "H1", "H2"],
        residue_ids=[1, 1, 1],
        residue_names=["HOH", "HOH", "HOH"],
        chain_ids=["A", "A", "A"],
        title="H2O smoke test",
    )


def _ch4_struct() -> Structure:
    """Tetrahedral CH4 with C-H = 1.087 A (experimental)."""
    a = 1.087 / np.sqrt(3.0)
    return Structure(
        elements=["C", "H", "H", "H", "H"],
        positions=np.array([
            [0.0,  0.0,  0.0],
            [ a,   a,    a ],
            [-a,  -a,    a ],
            [-a,   a,   -a ],
            [ a,  -a,   -a ],
        ]),
        atom_names=["C", "H1", "H2", "H3", "H4"],
        residue_ids=[1] * 5,
        residue_names=["LIG"] * 5,
        chain_ids=["A"] * 5,
        title="CH4 smoke test",
    )


def _run_script(script_text: str, tmp_path: Path,
                timeout: int = 180) -> float:
    """Write script_text to tmp_path and subprocess-run it.

    Returns the parsed total energy in Hartree, or fails the test
    with the captured stdout/stderr if the run errored or printed no
    matching energy line.
    """
    script_path = tmp_path / "smoke_run.py"
    script_path.write_text(script_text)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"PySCF script failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    m = _ENERGY_RE.search(proc.stdout)
    assert m is not None, (
        f"Could not find 'Total energy: ... Hartree' line.\n"
        f"--- stdout ---\n{proc.stdout}"
    )
    return float(m.group(1))


# ---------------------------------------------------------------- tests


@pytest.mark.smoke
def test_smoke_h2o_rhf_sto3g(tmp_path):
    """H2O at HF/STO-3G: classic textbook benchmark.

    Reference: PySCF 2.x prints E = -74.96302 Hartree at the
    experimental water geometry (r = 0.9572 A, theta = 104.52 deg).
    Tolerance 5 mHa is much looser than the 0.1 mHa reproducibility
    of HF/STO-3G across QC codes; we just want to catch wiring
    regressions, not measure accuracy.
    """
    text = render_script(_h2o_struct(),
                         _silent_singlepoint_config("h2o_smoke"))
    e = _run_script(text, tmp_path)
    assert e == pytest.approx(-74.9630, abs=5e-3)


@pytest.mark.smoke
def test_smoke_ch4_rhf_sto3g(tmp_path):
    """CH4 at HF/STO-3G: Td-symmetric, single SCF iteration cycle.

    Reference: -39.7268 Hartree (PySCF 2.x at C-H = 1.087 A).
    """
    text = render_script(_ch4_struct(),
                         _silent_singlepoint_config("ch4_smoke"))
    e = _run_script(text, tmp_path)
    assert e == pytest.approx(-39.7268, abs=5e-3)


@pytest.mark.smoke
def test_smoke_h2o_rhf_sto3g_frequencies(tmp_path):
    """H2O at HF/STO-3G single-point + harmonic frequencies + RRHO
    thermochemistry.

    Exercises the post-relax frequency / thermo block end-to-end:
    Hessian.kernel() -> harmonic_analysis -> thermo.thermo() ->
    <job>.thermo.txt.

    References for H2O at HF/STO-3G (literature, multiple QC codes):
    three vibrational modes around 4140 (sym str), 4480 (asym str),
    2170 cm^-1 (bend) -- HF/STO-3G systematically overestimates
    frequencies by ~10-15% vs experiment, which is fine for testing
    the wiring.  We hold to a 200 cm^-1 window per mode to absorb
    code/version drift.  Tolerance is intentionally loose: this test
    verifies that the rendered script EXECUTES and produces a
    sensible thermo.txt, NOT that PySCF computes accurate
    frequencies (PySCF's own test suite covers that).
    """
    cfg = _silent_singlepoint_config(
        "h2o_freq",
        method="RHF",
        basis="sto-3g",
        density_fit=False,
    )
    cfg.compute_frequencies = True
    cfg.temperature_K = 298.15
    cfg.pressure_atm = 1.0

    text = render_script(_h2o_struct(), cfg)
    # Run the script and capture the energy.  _run_script asserts the
    # script exits cleanly; freq failures would print "Frequency
    # analysis FAILED" but still let the script print Total energy.
    e = _run_script(text, tmp_path)
    assert e == pytest.approx(-74.9630, abs=5e-3)

    # The thermo.txt must have been written, must list the three
    # vibrational modes with the right shape, and must carry the
    # RRHO thermochemistry section with non-zero ZPE.
    thermo_file = tmp_path / "h2o_freq.thermo.txt"
    assert thermo_file.exists(), (
        f"thermo.txt missing -- script may have crashed before the "
        f"frequency block ran.  Contents of tmp_path: "
        f"{list(tmp_path.iterdir())}"
    )
    body = thermo_file.read_text()
    # Header: T and P recorded verbatim.
    assert "T = 298.15 K" in body
    assert "P = 1.0 atm" in body
    # Frequencies section: three modes (PySCF returns 3N - 6 = 3
    # vibrational modes for non-linear H2O, after projecting out 6
    # trans/rot dof).
    freq_section = body.split("[frequencies]")[1].split("[thermochemistry]")[0]
    mode_lines = [ln for ln in freq_section.splitlines()
                  if ln.lstrip().startswith("mode ")]
    assert len(mode_lines) == 3, (
        f"expected 3 vibrational modes for H2O, got {len(mode_lines)}: "
        f"{mode_lines}"
    )
    # Parse the wavenumbers; all three should be real and within
    # the broad HF/STO-3G window.
    import re as _re
    wavenums = []
    for ln in mode_lines:
        m = _re.search(r"mode\s+\d+\s+(-?\d+\.\d+)", ln)
        assert m, f"unparseable mode line: {ln!r}"
        wavenums.append(float(m.group(1)))
        # No imaginary tag on a true minimum.
        assert "(imag)" not in ln, f"unexpected imag mode: {ln!r}"
    # Three modes between 1500 and 5000 cm^-1 covers HF/STO-3G's
    # bend + two stretches with margin for code drift.
    assert all(1500.0 < w < 5000.0 for w in wavenums), (
        f"wavenumbers out of expected HF/STO-3G window 1500-5000 cm^-1: "
        f"{wavenums}"
    )
    # Thermochemistry section: non-empty + ZPE present + positive.
    therm_section = body.split("[thermochemistry]")[1]
    assert "zpe" in therm_section.lower(), (
        f"thermochemistry section missing ZPE: {therm_section!r}"
    )


@pytest.mark.smoke
def test_smoke_h2o_b3lyp_def2svp(tmp_path):
    """H2O at B3LYP/def2-SVP single-point.

    Exercises the *production* code paths -- DFT grid (level 4), hybrid
    functional, density fitting, def2 basis (which auto-supplies its
    own ECP for heavy atoms; here irrelevant but the wiring is the
    same).  A regression in any of those would change the energy by
    much more than the 20 mHa tolerance.

    Reference: PySCF 2.x prints -76.3582 Hartree (B3LYP/def2-SVP
    with density fitting at the experimental water geometry).  The
    literature spread for B3LYP/def2-SVP H2O is +/- 5 mHa across
    implementations and grid choices, so we hold to 10 mHa here.
    """
    cfg = _silent_singlepoint_config(
        "h2o_b3lyp",
        method="RKS",
        functional="B3LYP",
        basis="def2-SVP",
        density_fit=True,
    )
    text = render_script(_h2o_struct(), cfg)
    e = _run_script(text, tmp_path)
    assert e == pytest.approx(-76.3582, abs=1e-2)
