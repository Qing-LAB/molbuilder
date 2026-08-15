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
        title="h2", vacuum=(12.0, 12.0, 12.0))


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
        title="ch3", vacuum=(12.0, 12.0, 12.0))


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
        spin_treatment="polarized",
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


def test_gap_2_the_spin_mode_and_its_total_pin_reach_the_deck_together(h2):
    """The property, freed from a mechanism that turned out not to hold.

    HISTORY, because it is the whole point. This test was inverted on
    2026-05-24 to pin the v4 ``SpinPolarized .true.`` form, on the finding
    that the v5 ``Spin polarized`` path "does not read Spin.Fix / Spin.Total"
    and so aborted a hemeC-dithiol run at ``propor: ERROR: IMAX = 0``.

    That mechanism is NOT in SIESTA 5.4.2, verified against its source
    2026-08-15. ``spin_subs.F90`` reads the deprecated flags into ``opt_old``
    and then does ``opt = fdf_get('Spin', opt_old)`` — one variable, the new
    spelling merely winning. ``Spin.Fix`` / ``Spin.Total`` are read in a
    DIFFERENT file (``read_options.F90``), gated only on ``nspin == 2``, which
    both spellings produce identically. Whatever aborted that run in May, it
    was not this.

    So the mechanism is retired and the PROPERTY is kept, which is what the
    incident was really about: **asking for a fixed total spin must produce a
    deck that carries the mode AND the pin.** Losing either is what made that
    job fail, and this fails if either goes missing however it is spelled.
    """
    cfg = SiestaConfig(system_label="h2", spin_treatment="polarized",
                       spin_total=4.0)
    fdf = render_fdf(h2, cfg)
    assert re.search(r"^Spin\s+polarized\s*$", fdf, re.M), fdf
    assert re.search(r"^Spin\.Fix\s+\.true\.", fdf, re.M), fdf
    assert re.search(r"^Spin\.Total\s+4\.0", fdf, re.M), fdf
    # And the deprecated spelling is gone (SIESTA 5.4.2 deprecates all three
    # of SpinPolarized / NonCollinearSpin / SpinOrbit in favour of `Spin`).
    assert not re.search(r"^SpinPolarized\b", fdf, re.M), fdf

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
#  Gap 4: mf.stability() not auto-emitted for UKS / UHF                 #
# --------------------------------------------------------------------- #


def test_gap_4_pyscf_uks_emits_stability_analysis(methyl_radical):
    """UKS / UHF can converge to broken-symmetry saddles.  The
    generated script must call `mf.stability()` after the SCF so the
    user sees a warning when this happens.  The check is on the
    generated source -- not just a reference in a comment block; the
    call must be live code.

    Method name note: PySCF 2.x exposes the stability check as
    `mf.stability()` (no `_analysis` suffix); the older `stability_analysis`
    name does not exist on RKS / UKS objects in 2.x, so calling it
    raises AttributeError."""
    cfg = PySCFConfig(
        job_name="ch3",
        method="UKS",
        spin=1,                   # 2S = 1 (one unpaired electron)
        basis="STO-3G",
        density_fit=False,
        dispersion=None,
    )
    script = render_script(methyl_radical, cfg)
    # A *commented* mention in the troubleshooting block isn't enough;
    # the fix should emit live code.  Look for the call OUTSIDE comment
    # context.
    matches = [
        ln for ln in script.splitlines()
        if "mf.stability(" in ln and not ln.lstrip().startswith("#")
    ]
    assert matches, (
        "UKS script must contain a non-commented "
        "`mf.stability()` call after SCF."
    )
    # The retired API (`stability_analysis`) must not appear as live code.
    bad = [
        ln for ln in script.splitlines()
        if "stability_analysis" in ln and not ln.lstrip().startswith("#")
    ]
    assert not bad, (
        f"PySCF 2.x renamed the API to mf.stability(); "
        f"`stability_analysis` should not be emitted as live code: {bad}"
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
        PySCFConfig(job_name="h2", density_fit=False,
                    dispersion=None),
    )
    assert (
        "Post-processing" in script
        or "mulliken_pop" in script
        or "dip_moment" in script
        or "analyze" in script
    ), "PySCF script needs a commented post-processing template."


# --------------------------------------------------------------------- #
#  Gap 7: installation guide pins the supported SIESTA release           #
# --------------------------------------------------------------------- #


def test_gap_7_installation_documents_siesta_version():
    """The installation guide names the SIESTA release used by the recipe."""
    from pathlib import Path

    repo_root = Path(__file__).parent.parent
    installation = (repo_root / "docs" / "ops" / "installation.md").read_text().lower()
    assert "siesta" in installation
    assert re.search(r"siesta.*5\.4\.2", installation) or re.search(
        r"5\.4\.2.*siesta", installation
    ), "The installation guide must name the recipe SIESTA 5.4.2 release."


# --------------------------------------------------------------------- #
#  Gap 8: no ECP support for non-def2 bases                             #
# --------------------------------------------------------------------- #


# REDESIGNED 2026-08-13 -- gap #8 was "auto-emit an ECP for heavy atoms on
# a non-def2 basis", and six tests here pinned that rule plus its def2
# escape hatch (one of them parametrized over eight basis spellings).  The
# rule is retired on the user's ruling: *"there is no point to limit
# matching to heavy -- who defines heavy? there is no clear reasoning or
# standard"*, *"empty means empty"*, *"explicit is better than implicit"*.
#
# What was WORTH keeping out of that set is the emission bug the dict test
# caught -- an ECP map stuffed into a quoted string, which PySCF reads as
# an unknown ECP name.  Now that the resolver ALWAYS returns a map, that
# guard matters more than it did, so it is the one carried forward.


def _pt_complex():
    return Structure(
        elements=["Pt", "C", "C", "C", "C"],
        positions=np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
        ]),
        title="pt_complex", vacuum=(12.0, 12.0, 12.0))


def test_ecp_absent_when_the_user_declared_none():
    """A Pt complex on cc-pVDZ used to get ``lanl2dz`` added for it.  It
    no longer does: nothing is emitted that was not asked for."""
    cfg = PySCFConfig(job_name="pt", basis="cc-pVDZ",
                      density_fit=False, dispersion=None)
    script = render_script(_pt_complex(), cfg)
    assert not re.search(r"^\s*ecp\s*=", script, re.MULTILINE), (
        "no ecp was declared, so no ecp= kwarg may appear")


def test_ecp_emitted_as_a_dict_literal_not_a_quoted_string():
    """The one guard worth carrying over from the retired dict test.

    Pre-2026-05 the f-string stuffed the map's repr INSIDE a string
    literal -- ``ecp = "{'Pt': 'lanl2dz'}"`` -- which PySCF rejects as an
    unknown ECP name.  Every result is a map now, so every emission goes
    through this path.
    """
    cfg = PySCFConfig(job_name="pt", basis="cc-pVDZ",
                      ecp="lanl2dz", ecp_atoms=["Pt"],
                      density_fit=False, dispersion=None)
    script = render_script(_pt_complex(), cfg)

    ecp_lines = [ln for ln in script.splitlines()
                 if re.match(r"\s*ecp\s*=", ln)]
    assert len(ecp_lines) == 1, f"expected one ecp= line, got {ecp_lines}"
    line = ecp_lines[0]
    assert re.search(r"ecp\s*=\s*\{", line), (
        f"must emit a Python dict literal; got: {line!r}")
    assert not re.search(r'ecp\s*=\s*"', line), (
        f"must NOT be wrapped in quotes; got: {line!r}")
    assert "Pt" in line and "lanl2dz" in line
    compile(script, "<gen>", "exec")       # the literal must parse


def test_ecp_selector_reaches_only_the_named_element():
    """``["Pt"]`` in a Pt/C structure selects Pt and leaves C alone."""
    cfg = PySCFConfig(job_name="pt", basis="cc-pVDZ",
                      ecp="lanl2dz", ecp_atoms=["Pt"],
                      density_fit=False, dispersion=None)
    line = [ln for ln in render_script(_pt_complex(), cfg).splitlines()
            if re.match(r"\s*ecp\s*=", ln)][0]
    assert "'Pt'" in line and "'C'" not in line, line


def test_empty_means_empty():
    """Either half empty means no ECP -- never "pick one for me"."""
    for kw in ({"ecp": "", "ecp_atoms": ["*"]},
               {"ecp": "lanl2dz", "ecp_atoms": []},
               {"ecp": "", "ecp_atoms": []}):
        cfg = PySCFConfig(job_name="pt", basis="cc-pVDZ",
                          density_fit=False, dispersion=None, **kw)
        script = render_script(_pt_complex(), cfg)
        assert not re.search(r"^\s*ecp\s*=", script, re.MULTILINE), kw


def test_a_def2_basis_no_longer_suppresses_a_declared_ecp():
    """**A deliberate behaviour change.**  def2-* brings its own Stuttgart
    ECP, and the retired rule silently dropped any ECP the user named on a
    def2 basis -- across eight spellings of the basis name.  Silently
    discarding an explicit instruction is the implicit behaviour the whole
    rewrite removes: if you name one on def2, you get it, and whether that
    double-counts is a question for validation to raise, not for the
    emitter to decide by dropping your input.
    """
    cfg = PySCFConfig(job_name="pt", basis="def2-SVP",
                      ecp="stuttgart", ecp_atoms=["Pt"],
                      density_fit=False, dispersion=None)
    script = render_script(_pt_complex(), cfg)
    line = [ln for ln in script.splitlines()
            if re.match(r"\s*ecp\s*=", ln)]
    assert line and "stuttgart" in line[0], (
        f"a declared ECP must survive a def2 basis; got {line}")


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
        job_name="h2", density_fit=False, dispersion=None,
    )
    script = render_script(h2, cfg)
    # The fix should re-evaluate mf at mol_eq's geometry before
    # printing the final energy.  Post-#534 commit 4 the stages
    # loop's warm-start (mf.reset(mol_eq) followed by mf.kernel(...))
    # does this per iteration, so the last stage leaves mf converged
    # at mol_eq.  Accept any of the canonical patterns.
    assert re.search(
        r"mf.*=.*mol_eq|mol_eq.*kernel|re.?evaluate|mf\.reset\(mol_eq\)",
        script, re.IGNORECASE,
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

    cfg_default = PySCFConfig(job_name="h2",
                              density_fit=False, dispersion=None)
    s_default = render_script(h2, cfg_default)
    assert live_lines(s_default, "mf.diis_space") == []
    assert live_lines(s_default, "mf.damp")       == []

    # Hard-SCF case: both bumped to typical troubleshooting values
    cfg_hard = PySCFConfig(job_name="h2",
                           density_fit=False, dispersion=None,
                           diis_space=16, damp=0.4)
    s_hard = render_script(h2, cfg_hard)
    live_diis = live_lines(s_hard, "mf.diis_space = 16")
    live_damp = live_lines(s_hard, "mf.damp = 0.4")
    assert live_diis, "mf.diis_space = 16 must appear as live code"
    assert live_damp, "mf.damp = 0.4 must appear as live code"
