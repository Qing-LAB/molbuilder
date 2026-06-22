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


def test_c1_initial_xyz_captured_before_stages_loop(small_struct):
    """Spec: capture the user's actual input geometry NOW, before any
    stage in the optimization loop has a chance to modify it.
    Otherwise _initial.xyz would save the post-stage-N geometry because
    ``mol_eq`` shadows ``mol`` (and reset() rebinds mf.mol).
    """
    text = render_script(small_struct, PySCFConfig())

    # 2026-05-27: _initial.xyz path routes through _mb_outfile() so it
    # lands next to the script regardless of cwd at run time.
    save_pos = text.find('_save_xyz(mol, _mb_outfile(JOB + "_initial.xyz")')
    assert save_pos != -1, "no _initial.xyz save call found"

    # The stages driver loop comes after the SCF setup.
    loop_pos = text.find("for STAGE in STAGES:")
    assert loop_pos != -1, "no stages-loop header found"

    # The save MUST come before the loop.
    assert save_pos < loop_pos, (
        "_initial.xyz is saved AFTER the stages loop starts; mol "
        "may have been rebound to mol_eq via mf.reset() by then so "
        "the file would contain the post-relax geometry, not the "
        "user's input."
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


def test_c2_spin_polarized_emits_v4_form_for_aux_compat(small_struct):
    """Open-shell: emit ``SpinPolarized .true.`` (v4 form), NOT the
    v5 single-line ``Spin polarized``.

    Reversed 2026-05-24 against SIESTA 5.4.2: the v5 unified parser
    path does NOT subsequently read auxiliary ``Spin.Fix`` /
    ``Spin.Total`` keys -- a user's hemeC-dithiol run with
    ``Spin.Total 4.0`` aborted at ``propor: ERROR: IMAX = 0`` because
    Spin.Total never reached the initial-DM constructor.  Empirically
    diffed fdf.<timestamp>.log: v4 form -> aux keys honored;
    v5 form -> aux keys default.  See decisions log + the comment
    block above the emission in siesta/input.py."""
    fdf = render_fdf(small_struct,
                     SiestaConfig(spin_polarized=True, verbose_comments=False))
    assert "SpinPolarized .true." in fdf
    # The broken-in-5.4.2 v5 form must NOT be emitted.  Search at
    # line-start to avoid hitting verbose-comment mentions inside
    # the .fdf body.
    import re
    assert not re.search(r"^Spin\s+polarized\s*$", fdf, re.MULTILINE)


def test_c2_spin_total_emits_dotted_form_with_fix(small_struct):
    """SIESTA's total-spin pin is a TWO-line block (gap #1):
        Spin.Fix    .true.
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

    # both set -> emit Spin.Fix .true. + Spin.Total <v>.  Canonical
    # SIESTA boolean form (.true./.false.); the bare `true` synonym
    # was retired so the file matches the rest of the FDF.
    fdf = render_fdf(small_struct,
                     SiestaConfig(spin_polarized=True,
                                  spin_total=2.0,
                                  verbose_comments=False))
    assert re.search(r"^\s*Spin\.Fix\s+\.true\.\s*$", fdf, re.MULTILINE)
    assert re.search(r"^\s*Spin\.Total\s+2\.0\s*$", fdf, re.MULTILINE)
    # No SpinTotal anywhere (legacy form must be gone everywhere).
    assert "SpinTotal " not in fdf


# --------------------------------------------------------------------- #
#  C3 — stages loop must not silently swallow geomeTRIC errors          #
# --------------------------------------------------------------------- #


def test_c3_stages_loop_does_not_set_assert_convergence(small_struct):
    """The per-stage optimize() inside the stages loop should NOT
    pass ``assert_convergence=False``; we want geomeTRIC's default
    (True) so a stage that fails to converge in its max_steps raises
    instead of silently moving on.  Earlier preopt-era runs needed
    False on the cheap warm-up stage; the new design's looser early
    stages still want hard failure on a max-steps exhaustion, since
    that's the user's signal to add a looser stage or raise max_steps.
    """
    text = render_script(small_struct, PySCFConfig())
    main_block = text.split("for STAGE in STAGES:")[1]
    optimize_block = main_block.split("Final energy")[0]
    assert "assert_convergence" not in optimize_block, (
        "stages-loop optimize() is suppressing convergence assertion -- "
        "that hides real failures users need to see."
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
