"""Regression-prevention tests for the 10 known SIESTA / PySCF science gaps.

Each gap is documented in ``docs/design.md`` under "Known SIESTA /
PySCF science gaps".  Today the tests below are marked
``@pytest.mark.xfail`` because the fixes haven't landed -- the gap
list was confirmed unfixed in the 2026-05-01 audit.

When a fix for one of these gaps lands, the corresponding test
flips from xfail to pass; that's the signal that the fix worked.
If a fix later regresses, the test fails normally (the marker is
gone by then), which is the regression-prevention property this
file provides.

Each test asserts on the GENERATED OUTPUT, not on internal config
state -- that way a refactor of how the config is plumbed into the
generator doesn't false-pass the test.

Spec source: ``docs/design.md`` § "Known SIESTA / PySCF science gaps".
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from molbuilder.pyscf import PySCFConfig, render_script
from molbuilder.siesta import SiestaConfig, render_fdf
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Shared fixtures: tiny structures used across the gap tests          #
# --------------------------------------------------------------------- #


@pytest.fixture
def h2():
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        title="h2",
    )


@pytest.fixture
def methyl_radical():
    """CH3 radical -- canonical open-shell test case (1 unpaired electron)."""
    return Structure(
        elements=["C", "H", "H", "H"],
        positions=np.array([
            [ 0.000,  0.000, 0.000],
            [ 1.080,  0.000, 0.000],
            [-0.540,  0.935, 0.000],
            [-0.540, -0.935, 0.000],
        ]),
        title="ch3",
    )


# --------------------------------------------------------------------- #
#  Gap 1: SpinTotal keyword is not real SIESTA                          #
#                                                                        #
#  SIESTA uses "Spin.Total <v>" (with the dot), gated on "Spin.Fix      #
#  true".  The legacy emission writes "SpinTotal <v>" -- a single token #
#  that SIESTA's fdf parser silently ignores on a value mismatch.       #
# --------------------------------------------------------------------- #


def test_gap_1_siesta_emits_spin_total_with_dot(h2):
    """When spin_total is set, the FDF must contain `Spin.Total` (the
    real keyword) and `Spin.Fix true`, not the bogus single-token
    `SpinTotal`."""
    cfg = SiestaConfig(
        system_label="h2",
        spin_polarized=True,
        spin_total=1.0,
    )
    fdf = render_fdf(h2, cfg)
    # The real SIESTA keywords:
    assert re.search(r"^\s*Spin\.Total\s+1", fdf, re.MULTILINE), (
        "FDF must emit `Spin.Total <v>` (with the dot) -- "
        "see SIESTA manual Spin section."
    )
    assert re.search(r"^\s*Spin\.Fix\s+true", fdf, re.MULTILINE), (
        "FDF must emit `Spin.Fix true` to enable the total-spin pin."
    )
    # And the legacy bogus form must be GONE:
    assert "SpinTotal " not in fdf, "FDF still emits the bogus SpinTotal token"


# --------------------------------------------------------------------- #
#  Gap 2: SpinPolarized is v4-era; v5 wants `Spin polarized`            #
# --------------------------------------------------------------------- #


def test_gap_2_siesta_emits_v5_spin_block(h2):
    """A modern (v5) SIESTA prefers a single-line `Spin polarized`
    declaration over the v4-era `SpinPolarized true`.  The test
    asserts the v5 form is present; the fix should also document
    the targeted SIESTA version range."""
    cfg = SiestaConfig(system_label="h2", spin_polarized=True)
    fdf = render_fdf(h2, cfg)
    assert re.search(r"^\s*Spin\s+polarized\s*$", fdf, re.MULTILINE), (
        "FDF must emit the v5-compatible single-line `Spin polarized`."
    )


# --------------------------------------------------------------------- #
#  Gap 3: no SIESTA dispersion-correction template emitted              #
# --------------------------------------------------------------------- #


def test_gap_3_siesta_emits_dispersion_template_for_pbe(h2):
    """When the chosen XC is non-dispersive (default PBE), the
    generated FDF must contain a commented-out dispersion-correction
    template block that the user can uncomment.  Plain PBE
    underbinds organic / biomolecule systems without a vdW
    correction; making the template visible reduces the chance of
    silent under-binding."""
    cfg = SiestaConfig(
        system_label="h2",
        xc_functional="GGA",
        xc_authors="PBE",
    )
    fdf = render_fdf(h2, cfg)
    # The fix should add a commented `%block MM.Potentials` (D2/D3
    # empirical) template the user can uncomment.  Loose substring
    # matches like "dispersion" don't count -- the existing FDF
    # mentions VDW only in a comment about the XC functional CHOICE,
    # not as an actionable correction template.  Anchor on the
    # SIESTA-level template marker.
    assert (
        "%block MM.Potentials" in fdf
        or re.search(r"^\s*MM\.Potentials\s+", fdf, re.MULTILINE)
    ), "FDF needs a commented `%block MM.Potentials` D2/D3 template for non-vdW XC."


def test_gap_3_dispersion_template_suppressed_for_vdw_xc(h2):
    """The flip side: when the user already picked a vdW-aware XC
    (XC.functional VDW + DRSLL/KBM/...), the non-local correlation
    is in the functional itself.  An MM.Potentials block on top
    would double-count -- so the template MUST NOT appear."""
    cfg = SiestaConfig(
        system_label="h2",
        xc_functional="VDW",
        xc_authors="DRSLL",
    )
    fdf = render_fdf(h2, cfg)
    assert "MM.Potentials" not in fdf, (
        "vdW XC already includes dispersion; emitting an MM.Potentials "
        "template would double-count and confuse the user"
    )


# --------------------------------------------------------------------- #
#  Gap 4: mf.stability_analysis() not auto-emitted for UKS / UHF        #
# --------------------------------------------------------------------- #


def test_gap_4_pyscf_uks_emits_stability_analysis(methyl_radical):
    """UKS / UHF can converge to broken-symmetry saddles.  The
    generated script must call `mf.stability_analysis()` after the
    SCF so the user sees a warning when this happens.  The check is
    on the generated source -- not just a reference in a comment
    block; the call must be live code."""
    cfg = PySCFConfig(
        job_name="ch3",
        method="UKS",
        spin=1,                   # 2S = 1 (one unpaired electron)
        basis="STO-3G",
        preopt=False,
        density_fit=False,
        dispersion=None,
    )
    script = render_script(methyl_radical, cfg)
    # A *commented* mention in the troubleshooting block isn't enough;
    # the fix should emit live code.  Look for the call OUTSIDE comment
    # context.  We use a regex that matches the call at the start of
    # a non-comment line.
    matches = [
        ln for ln in script.splitlines()
        if "stability_analysis" in ln and not ln.lstrip().startswith("#")
    ]
    assert matches, (
        "UKS script must contain a non-commented "
        "`mf.stability_analysis()` call after SCF."
    )


# --------------------------------------------------------------------- #
#  Gap 5: PAO.EnergyShift default is too loose                          #
# --------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="design.md gap #5: SiestaConfig.pao_energy_shift defaults "
           "to 0.02 Ry; production work uses 0.005-0.01 Ry. Tighten "
           "default to 0.01 Ry.",
    strict=True,
)
def test_gap_5_siesta_pao_energy_shift_default_is_tight():
    """The default PAO.EnergyShift should be 0.01 Ry or tighter.
    0.02 Ry produces under-converged PAO basis tails for most
    production work."""
    assert SiestaConfig().pao_energy_shift <= 0.01, (
        "SiestaConfig.pao_energy_shift default is too loose for "
        "production work; should be <= 0.01 Ry."
    )


# --------------------------------------------------------------------- #
#  Gap 6: no post-processing block in either generator                  #
# --------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="design.md gap #6: no `# --- Post-processing hook ---` "
           "placeholder at end of FDF or PySCF script",
    strict=True,
)
def test_gap_6_siesta_emits_post_processing_hook(h2):
    """Generated FDF must end with a commented-out post-processing
    block (BandLines, PDOS, etc.) so a user knows where to add
    follow-up analysis."""
    fdf = render_fdf(h2, SiestaConfig(system_label="h2"))
    assert "Post-processing" in fdf or "BandLines" in fdf or "PDOS" in fdf, (
        "FDF needs a commented post-processing template "
        "(BandLines / PDOS) for follow-up analysis."
    )


@pytest.mark.xfail(
    reason="design.md gap #6: no post-processing hook in PySCF script",
    strict=True,
)
def test_gap_6_pyscf_emits_post_processing_hook(h2):
    """Generated PySCF script must include a commented-out
    post-processing block (analyze, mulliken_pop, dip_moment, etc.)
    so a user has a starting point for follow-ups."""
    script = render_script(
        h2,
        PySCFConfig(job_name="h2", preopt=False, density_fit=False,
                    dispersion=None),
    )
    assert (
        "Post-processing" in script
        or "mulliken_pop" in script
        or "dip_moment" in script
        or "analyze" in script
    ), "PySCF script needs a commented post-processing template."


# --------------------------------------------------------------------- #
#  Gap 7: no SIESTA minimum version pinned                              #
#                                                                        #
#  SIESTA isn't pip-installable (it's a Fortran binary), so the "pin"  #
#  is a documentation / requirements artifact rather than a runtime    #
#  check.  We test that requirements-runtime.txt mentions a SIESTA     #
#  version range (any spelling).                                       #
# --------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="design.md gap #7: requirements-runtime.txt doesn't "
           "document the SIESTA version range targeted by the "
           "emitted keywords",
    strict=True,
)
def test_gap_7_requirements_documents_siesta_version_range():
    """requirements-runtime.txt should mention the SIESTA version
    range targeted by the FDF emission -- emitted keywords like
    DM.Energy.Tolerance may be silently ignored on older builds."""
    from pathlib import Path
    repo_root = Path(__file__).parent.parent
    req = (repo_root / "requirements-runtime.txt").read_text().lower()
    assert "siesta" in req, (
        "requirements-runtime.txt should mention the SIESTA "
        "version range targeted by the generator."
    )
    # Must mention an actual version number (4.x or 5.x):
    assert re.search(r"siesta.*(4\.\d|5\.\d)", req) or re.search(
        r"(4\.\d|5\.\d).*siesta", req
    ), "Version range required (e.g. 'SIESTA >= 4.1' or 'SIESTA 5.x')."


# --------------------------------------------------------------------- #
#  Gap 8: no ECP support for non-def2 bases                             #
# --------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="design.md gap #8: PySCF script with cc-pVDZ + heavy atoms "
           "doesn't auto-emit an ECP block; users currently hand-edit",
    strict=True,
)
def test_gap_8_pyscf_emits_ecp_for_heavy_atoms_with_non_def2():
    """A structure containing transition metals (Pt here) on a basis
    that's NOT def2-* must auto-emit an ECP definition; otherwise
    PySCF crashes on missing core electrons / wrong basis."""
    pt_complex = Structure(
        elements=["Pt", "C", "C", "C", "C"],
        positions=np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
        ]),
        title="pt_complex",
    )
    cfg = PySCFConfig(
        job_name="pt",
        basis="cc-pVDZ",            # NOT def2-* -- needs explicit ECP
        preopt=False,
        density_fit=False,
        dispersion=None,
    )
    script = render_script(pt_complex, cfg)
    assert "ecp" in script.lower(), (
        "Heavy-atom (Pt) calculation on cc-pVDZ needs an ECP block."
    )


# --------------------------------------------------------------------- #
#  Gap 9: save_optimized_xyz writes mol_eq, mf.e_tot may not match     #
#                                                                        #
#  If the geom-opt didn't fully converge, mol_eq's geometry and        #
#  mf.e_tot (energy at the LAST inner SCF, possibly at a different     #
#  geometry) can disagree.  The fix is to recompute / re-evaluate mf   #
#  at mol_eq's geometry before reporting e_tot.  Today the script      #
#  prints mf.e_tot directly without that guard.                        #
# --------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="design.md gap #9: PySCF script reports mf.e_tot without "
           "a re-evaluation guard at mol_eq's geometry",
    strict=True,
)
def test_gap_9_pyscf_reevaluates_energy_at_optimized_geom(h2):
    """The generated script should re-evaluate mf at mol_eq's
    geometry before reporting the final energy -- otherwise a
    non-converged opt prints an energy that doesn't correspond to
    the saved coordinates."""
    cfg = PySCFConfig(
        job_name="h2", preopt=False, density_fit=False, dispersion=None,
    )
    script = render_script(h2, cfg)
    # The fix should run a single-point SCF at mol_eq's geometry
    # before the print -- check for the typical `mf_eq.kernel()`
    # / `mf.run(mol_eq)` pattern, or an explicit re-attach.
    assert re.search(
        r"mf.*=.*mol_eq|mol_eq.*kernel|re.?evaluate", script, re.IGNORECASE
    ), (
        "Script should re-evaluate mf at mol_eq's geometry before "
        "printing the final e_tot."
    )


# --------------------------------------------------------------------- #
#  Gap 10: no mf.diis_space / mf.damp in PySCFConfig                    #
# --------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="design.md gap #10: PySCFConfig has no diis_space / damp "
           "fields; hard-SCF troubleshooting requires script-editing",
    strict=True,
)
def test_gap_10_pyscf_config_exposes_diis_space_and_damp():
    """PySCFConfig should expose diis_space and damp as fields so
    users with hard-converging SCFs can tune them through the
    documented config surface, not by hand-editing the generated
    script."""
    from dataclasses import fields
    field_names = {f.name for f in fields(PySCFConfig)}
    assert "diis_space" in field_names, (
        "PySCFConfig.diis_space missing -- hard-SCF troubleshooting "
        "knob isn't exposed."
    )
    assert "damp" in field_names, (
        "PySCFConfig.damp missing -- hard-SCF troubleshooting knob "
        "isn't exposed."
    )
