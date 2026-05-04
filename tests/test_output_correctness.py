"""Output-correctness invariants for SIESTA + PySCF generators.

These tests pin the *generated artefact* against what each engine
needs to produce a scientifically correct calculation.  They live
separately from test_pyscf_spec.py because they cover both engines
and they're the layer that should have caught the C1-C4 bugs.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.pyscf import PySCFConfig, render_script
from molbuilder.siesta import SiestaConfig, render_fdf
from molbuilder.structure import Structure


@pytest.fixture
def small_struct():
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0, 0, 0], [0.957, 0, 0], [-0.24, 0.927, 0]]),
        title="water",
    )


# --------------------------------------------------------------------- #
#  C1 — _initial.xyz captured BEFORE optimization mutates mol           #
# --------------------------------------------------------------------- #


def test_c1_initial_xyz_captured_before_preopt(small_struct):
    """Spec: 'Capture the user's actual input geometry NOW, before
    pre-opt has a chance to modify it.'  Without the fix, _initial.xyz
    saves the post-pre-opt geometry because mol gets reassigned later.
    """
    text = render_script(small_struct, PySCFConfig(preopt=True))

    # Find the position in the script where _initial.xyz is saved.
    save_pos = text.find('_save_xyz(mol, JOB + "_initial.xyz"')
    assert save_pos != -1, "no _initial.xyz save call found"

    # Find the position of the pre-opt block (its banner print).
    preopt_pos = text.find('"\\n=== Stage: pre-optimization ==="')
    assert preopt_pos != -1, "no pre-opt banner found"

    # The save MUST come before the pre-opt block.
    assert save_pos < preopt_pos, (
        "_initial.xyz is saved AFTER pre-opt runs; mol has been "
        "reassigned to mol_pre by then, so the file would contain "
        "the post-pre-opt geometry, not the user's input."
    )


def test_c1_initial_xyz_save_helper_defined_early(small_struct):
    """Corollary: _save_xyz must be defined before _initial.xyz is
    called, which means before the gto.M(mol = ...) line."""
    text = render_script(small_struct, PySCFConfig())
    helper_pos = text.find("def _save_xyz(")
    mol_pos    = text.find("mol = gto.M(")
    assert helper_pos != -1 and mol_pos != -1
    assert helper_pos < mol_pos, (
        "_save_xyz is defined AFTER mol is built; it can't be called "
        "to snapshot the input geometry."
    )


# --------------------------------------------------------------------- #
#  C2 — SIESTA spin polarisation                                        #
# --------------------------------------------------------------------- #


def test_c2_default_no_spin_block(small_struct):
    """Default closed-shell: no Spin block in the FDF."""
    fdf = render_fdf(small_struct, SiestaConfig(verbose_comments=False))
    # Don't match "Spin." (which is part of comments / other tokens) --
    # match the keyword line specifically.
    import re
    assert not re.search(r"^\s*Spin\s+polarized\s*$", fdf, re.MULTILINE)
    assert not re.search(r"^\s*Spin\.Fix",            fdf, re.MULTILINE)
    # And the legacy bogus tokens must NEVER appear (gap #1+#2).
    assert "SpinPolarized" not in fdf
    assert "SpinTotal " not in fdf


def test_c2_spin_polarized_emits_v5_form(small_struct):
    """Open-shell: emit `Spin polarized` (v5 single-line form), NOT
    the v4-era `SpinPolarized true` (gap #2)."""
    import re
    fdf = render_fdf(small_struct,
                     SiestaConfig(spin_polarized=True, verbose_comments=False))
    assert re.search(r"^\s*Spin\s+polarized\s*$", fdf, re.MULTILINE)
    # The legacy v4 form should be absent.
    assert "SpinPolarized" not in fdf


def test_c2_spin_total_emits_dotted_form_with_fix(small_struct):
    """SIESTA's total-spin pin is a TWO-line block (gap #1):
        Spin.Fix    true
        Spin.Total  <v>
    The bogus single-token `SpinTotal <v>` (silently ignored by SIESTA)
    must be gone, and Spin.Fix must always accompany Spin.Total or
    SIESTA ignores the constraint."""
    import re

    # spin_total set but spin_polarized off -> no Spin.Total emitted
    fdf = render_fdf(small_struct,
                     SiestaConfig(spin_total=1.0, verbose_comments=False))
    assert "Spin.Total" not in fdf
    assert "SpinTotal"  not in fdf      # legacy bogus form

    # both set -> emit Spin.Fix true + Spin.Total <v>
    fdf = render_fdf(small_struct,
                     SiestaConfig(spin_polarized=True,
                                  spin_total=2.0,
                                  verbose_comments=False))
    assert re.search(r"^\s*Spin\.Fix\s+true\s*$",  fdf, re.MULTILINE)
    assert re.search(r"^\s*Spin\.Total\s+2\.0\s*$", fdf, re.MULTILINE)
    # No SpinTotal anywhere (legacy form must be gone everywhere).
    assert "SpinTotal " not in fdf


# --------------------------------------------------------------------- #
#  C3 — pre-opt must set assert_convergence=False                       #
# --------------------------------------------------------------------- #


def test_c3_preopt_sets_assert_convergence_false(small_struct):
    """Spec: 'pre-opt's optimize() must pass assert_convergence=False
    so a partial pre-opt doesn't kill the production run.'"""
    text = render_script(small_struct, PySCFConfig(preopt=True))

    # Find the pre-opt's optimize() block.
    after_preopt_banner = text.split('"\\n=== Stage: pre-optimization ==="')[1]
    preopt_optimize_block = after_preopt_banner.split('"Pre-opt done')[0]

    assert "assert_convergence = False" in preopt_optimize_block, (
        "pre-opt's optimize() is missing assert_convergence=False; "
        "if pre-opt fails to converge in N steps, RuntimeError will "
        "kill the entire run before production starts."
    )


def test_c3_main_optimize_does_not_set_assert_convergence(small_struct):
    """Production-stage optimize() should NOT pass assert_convergence,
    so it defaults to True and we hear about real failures."""
    text = render_script(small_struct, PySCFConfig(preopt=False))
    main_block = text.split("=== Stage: production optimization")[1]
    main_optimize = main_block.split("Final energy")[0]
    assert "assert_convergence" not in main_optimize, (
        "production optimize() is suppressing convergence assertion -- "
        "that hides real production-run failures."
    )


# --------------------------------------------------------------------- #
#  C4 — RKS / RHF + nonzero spin must raise at generation               #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["RKS", "RHF"])
def test_c4_restricted_method_with_nonzero_spin_raises(small_struct, method):
    """Restricted methods assume mol.spin == 0.  Setting spin != 0
    is silently wrong physics -- we must catch it at generation."""
    with pytest.raises(ValueError, match="restricted.*incompatible"):
        render_script(small_struct, PySCFConfig(method=method, spin=1))


@pytest.mark.parametrize("method", ["UKS", "UHF"])
def test_c4_unrestricted_method_with_nonzero_spin_ok(small_struct, method):
    """UKS / UHF must accept any spin without raising."""
    text = render_script(small_struct, PySCFConfig(method=method, spin=1, charge=1))
    # Sanity: the method appears in the script.
    assert method in text


def test_c4_restricted_method_with_zero_spin_ok(small_struct):
    """Default RKS + spin=0 is the closed-shell case; must not raise."""
    text = render_script(small_struct, PySCFConfig(method="RKS", spin=0))
    assert "mf = dft.RKS(mol)" in text


# --------------------------------------------------------------------- #
#  Cosmetic: dump_input=False on rebuild paths                          #
# --------------------------------------------------------------------- #


def test_preopt_mol_pre_build_doesnt_double_dump(small_struct):
    """mol_pre.build() should pass dump_input=False -- otherwise the
    input file gets echoed into <JOB>.log a second time during
    pre-opt setup."""
    text = render_script(small_struct, PySCFConfig(preopt=True))
    # The pre-opt block builds mol_pre right after copying.
    pre_build = text.split("mol_pre = mol.copy()", 1)[1]
    pre_build = pre_build.split("mol_pre = optimize(", 1)[0]
    assert "mol_pre.build(dump_input=False)" in pre_build, (
        "mol_pre.build() should pass dump_input=False to avoid "
        "duplicating the input echo in the .log"
    )


# --------------------------------------------------------------------- #
#  Cross-engine charge consistency                                      #
# --------------------------------------------------------------------- #


def test_charge_propagates_consistently_across_engines(deprotonated_diester):
    """Same input structure -> SIESTA NetCharge and PySCF gto.M
    charge=... must agree.  Both should pick up -1 from the heuristic."""
    fdf  = render_fdf(deprotonated_diester,
                      SiestaConfig(verbose_comments=False))
    py   = render_script(deprotonated_diester,
                         PySCFConfig(verbose_comments=False))

    assert "NetCharge       -1" in fdf, "SIESTA didn't pick up the -1"
    assert "charge     = -1," in py, "PySCF didn't pick up the -1"


def test_user_charge_override_consistent_across_engines(deprotonated_diester):
    """Setting cfg.net_charge=-3 (SIESTA) and cfg.charge=-3 (PySCF)
    on the same structure must produce equally charged outputs."""
    fdf = render_fdf(deprotonated_diester,
                     SiestaConfig(net_charge=-3, verbose_comments=False))
    py  = render_script(deprotonated_diester,
                        PySCFConfig(charge=-3, verbose_comments=False))

    assert "NetCharge       -3" in fdf
    assert "charge     = -3," in py
