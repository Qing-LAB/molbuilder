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
    real keyword) and `Spin.Fix .true.` (canonical SIESTA boolean),
    not the bogus single-token `SpinTotal`."""
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
    # Accept either bare `true` or canonical `.true.` -- SIESTA's
    # parser treats them as synonyms; we now emit the canonical form
    # to match the rest of the FDF (Diag.ParallelOverK, WriteForces, ...).
    assert re.search(r"^\s*Spin\.Fix\s+\.?true\.?", fdf, re.MULTILINE), (
        "FDF must emit `Spin.Fix .true.` to enable the total-spin pin."
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


def test_gap_6_siesta_emits_post_processing_hook(h2):
    """Generated FDF must end with a commented-out post-processing
    block (BandLines, PDOS, etc.) so a user knows where to add
    follow-up analysis."""
    fdf = render_fdf(h2, SiestaConfig(system_label="h2"))
    assert "Post-processing" in fdf or "BandLines" in fdf or "PDOS" in fdf, (
        "FDF needs a commented post-processing template "
        "(BandLines / PDOS) for follow-up analysis."
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


def test_gap_8_ecp_skipped_for_def2_basis():
    """def2-* basis families bundle their own ECP for Z > 36, so an
    extra `ecp = "lanl2dz"` would double-count.  Auto-emit must be
    suppressed when basis is def2-*."""
    pt = Structure(
        elements=["Pt", "C", "C"],
        positions=np.array([[0.0,0,0],[2,0,0],[-2,0,0]]),
        title="pt",
    )
    cfg = PySCFConfig(job_name="pt", basis="def2-SVP",
                      preopt=False, density_fit=False, dispersion=None)
    script = render_script(pt, cfg)
    # The token "ecp" appears in user-facing comments / docstrings;
    # what we want to suppress is the kwarg line `    ecp        = "..."`,
    # which lives inside the gto.M(...) call.
    assert not re.search(r"^\s*ecp\s*=", script, re.MULTILINE), (
        "def2-* basis bundles its own ECP; auto-emitting another "
        "would double-count"
    )


def test_gap_8_ecp_skipped_for_light_atoms_only():
    """No heavy atoms -> no ECP needed regardless of basis choice."""
    h2o = Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0,0,0],[0.96,0,0],[-0.24,0.93,0]]),
        title="h2o",
    )
    cfg = PySCFConfig(job_name="h2o", basis="cc-pVDZ",
                      preopt=False, density_fit=False, dispersion=None)
    script = render_script(h2o, cfg)
    assert not re.search(r"^\s*ecp\s*=", script, re.MULTILINE), (
        "Light-atom-only molecule on cc-pVDZ should not get an ECP"
    )


def test_gap_8_ecp_user_override_disables():
    """`cfg.ecp = ""` is the explicit opt-out for power users who
    want to provide their own ECP-and-basis block in a hand-edit
    of the script."""
    pt = Structure(
        elements=["Pt"],
        positions=np.array([[0.0,0,0]]),
        title="pt",
    )
    cfg = PySCFConfig(job_name="pt", basis="cc-pVDZ", ecp="",
                      preopt=False, density_fit=False, dispersion=None)
    script = render_script(pt, cfg)
    assert not re.search(r"^\s*ecp\s*=", script, re.MULTILINE), (
        "cfg.ecp = '' must suppress the auto-emit"
    )


@pytest.mark.parametrize("basis", [
    "def2-SVP", "def2_SVP", "def2svp",          # SVP variants
    "def2-TZVP", "def2_TZVP", "def2tzvp",        # TZVP variants
    "DEF2-SVP", "Def2-TZVP",                     # case variants
])
def test_gap_8_ecp_skipped_for_all_def2_spellings(basis):
    """All three PySCF-equivalent def2 spellings (hyphen / underscore /
    no separator) must skip the ECP auto-emit -- def2's own ECP is
    bundled with the basis, an extra `ecp = "lanl2dz"` would double-
    count.  Pre-fix the prefix check required the hyphen; the
    underscore and no-separator forms slipped through and emitted
    a spurious lanl2dz on top of def2's own ECP."""
    pt = Structure(
        elements=["Pt", "C", "C"],
        positions=np.array([[0.0,0,0],[2,0,0],[-2,0,0]]),
    )
    cfg = PySCFConfig(job_name="pt", basis=basis,
                      preopt=False, density_fit=False, dispersion=None)
    script = render_script(pt, cfg)
    assert not re.search(r"^\s*ecp\s*=", script, re.MULTILINE), (
        f"basis={basis!r} (a def2 family member) bundles its own ECP; "
        f"emitting an additional ecp= would double-count"
    )


def test_gap_8_dict_ecp_emits_as_python_dict_literal():
    """Per-element ECP control is a real use case for mixed light /
    heavy systems: the user passes ``cfg.ecp = {"Pt": "lanl2dz",
    "Au": "stuttgart"}``.  Pre-fix the f-string stuffed the dict's
    repr inside a string literal, producing
    ``ecp = "{'Pt': 'lanl2dz'}"`` -- which PySCF rejects as an
    unknown ECP NAME.  Post-fix the dict is emitted as a real
    Python dict literal so PySCF sees it as the per-element form."""
    pt = Structure(
        elements=["Pt", "Au", "C"],
        positions=np.array([[0.0,0,0],[2,0,0],[-2,0,0]]),
    )
    cfg = PySCFConfig(job_name="pt", basis="cc-pVDZ",
                      preopt=False, density_fit=False, dispersion=None)
    cfg.ecp = {"Pt": "lanl2dz", "Au": "stuttgart"}
    script = render_script(pt, cfg)

    # Find the ecp = ... line (only one).  Must open with `{`, not `"`.
    ecp_lines = [ln for ln in script.splitlines()
                 if re.match(r"\s*ecp\s*=", ln)]
    assert len(ecp_lines) == 1, f"expected exactly one ecp= line, got {ecp_lines}"
    ecp_line = ecp_lines[0]

    # Must be a dict literal: the value starts with `{` (Python dict),
    # not `"` (string).  Pre-fix the line was: ecp = "{'Pt': 'lanl2dz'}",
    assert re.search(r"ecp\s*=\s*\{", ecp_line), (
        f"dict ecp must emit as a Python dict literal; got: {ecp_line!r}"
    )
    assert not re.search(r'ecp\s*=\s*"', ecp_line), (
        f"dict ecp must NOT be wrapped in quotes; got: {ecp_line!r}"
    )
    # Both keys present, regardless of repr quote style.
    assert "Pt" in ecp_line and "lanl2dz" in ecp_line
    assert "Au" in ecp_line and "stuttgart" in ecp_line
    # Round-trip: the script must compile cleanly with the dict literal.
    compile(script, "<gen>", "exec")


# --------------------------------------------------------------------- #
#  Gap 9: save_optimized_xyz writes mol_eq, mf.e_tot may not match     #
#                                                                        #
#  If the geom-opt didn't fully converge, mol_eq's geometry and        #
#  mf.e_tot (energy at the LAST inner SCF, possibly at a different     #
#  geometry) can disagree.  The fix is to recompute / re-evaluate mf   #
#  at mol_eq's geometry before reporting e_tot.  Today the script      #
#  prints mf.e_tot directly without that guard.                        #
# --------------------------------------------------------------------- #


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


def test_gap_10_diis_damp_emitted_only_when_tuned(h2):
    """Defaults (diis_space=8, damp=0) should NOT appear in the
    generated script -- they match PySCF's own defaults and adding
    them is just noise.  Bumping either MUST surface as a
    `mf.diis_space = N` / `mf.damp = X` line."""
    def live_lines(text, needle):
        # Filter commented-out lines (the troubleshooting block at
        # end of script mentions both knobs as hints in `#` lines).
        return [ln for ln in text.splitlines()
                if needle in ln and not ln.lstrip().startswith("#")]

    cfg_default = PySCFConfig(job_name="h2", preopt=False,
                              density_fit=False, dispersion=None)
    s_default = render_script(h2, cfg_default)
    assert live_lines(s_default, "mf.diis_space") == []
    assert live_lines(s_default, "mf.damp")       == []

    # Hard-SCF case: both bumped to typical troubleshooting values
    cfg_hard = PySCFConfig(job_name="h2", preopt=False,
                           density_fit=False, dispersion=None,
                           diis_space=16, damp=0.4)
    s_hard = render_script(h2, cfg_hard)
    live_diis = live_lines(s_hard, "mf.diis_space = 16")
    live_damp = live_lines(s_hard, "mf.damp = 0.4")
    assert live_diis, "mf.diis_space = 16 must appear as live code"
    assert live_damp, "mf.damp = 0.4 must appear as live code"
