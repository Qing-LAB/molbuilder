"""Tests for molbuilder.validation.pyscf.

Per docs/protocols/test-strategy.md § 3 (test layout mirrors source
layout).  Split from the pre-2026-06-13 flat tests/test_validation.py
on 2026-06-13; no test body was modified.  Shared fixtures
(``water_struct``, ``_vacuum_cell``) live in tests/validation/conftest.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import report, validate
from ._helpers import _vacuum_cell




# --------------------------------------------------------------------- #
#  PySCF: spin field validation                                         #
# --------------------------------------------------------------------- #


def test_pyscf_negative_spin_is_error(water_struct):
    cfg = PySCFConfig(spin=-1)
    issues = validate(water_struct, cfg)
    spin_issues = [i for i in issues if i.where == "config.spin"]
    # spin=-1 produces TWO issues now that the dataclass declares
    # range=(0, 10) for the schema-driven form:
    #   1. error: "spin = -1 is negative; ..." (explicit semantic check)
    #   2. warn:  "Spin (2S) = -1 is outside the recommended range [0, 10]"
    #             (auto from field range metadata)
    # The error must survive; the additional range warn is fine
    # (both convey "this is wrong" to the user, with the explicit
    # error carrying the actionable explanation).
    errs = [i for i in spin_issues if i.severity == "error"]
    assert len(errs) == 1
    assert "negative" in errs[0].message



def test_pyscf_open_shell_spin_with_rks_is_warn(water_struct):
    """Open-shell spin with RKS / RHF (closed-shell methods) is the
    "I forgot to switch to UKS" foot-gun."""
    cfg = PySCFConfig(spin=1, method="RKS")
    issues = validate(water_struct, cfg)
    warns = [i for i in issues if i.where == "config.method"
             and i.severity == "warn"]
    assert len(warns) == 1



def test_pyscf_open_shell_spin_with_uks_no_warn(water_struct):
    cfg = PySCFConfig(spin=1, method="UKS")
    issues = validate(water_struct, cfg)
    assert [i for i in issues if i.where == "config.method"] == []



def test_pyscf_grid_level_above_range_warns(water_struct):
    """grid_level has metadata range (0, 9).  Beyond that value isn't
    meaningful in PySCF; warn the user before they generate a script
    that PySCF will reject."""
    cfg = PySCFConfig(grid_level=20)
    issues = validate(water_struct, cfg)
    msgs = [i for i in issues if i.where == "config.grid_level"]
    assert len(msgs) == 1



# --------------------------------------------------------------------- #
#  Method / functional / grid_level cross-check rules                  #
# --------------------------------------------------------------------- #


def test_pyscf_uks_with_spin_zero_warns(water_struct):
    """UKS / UHF with spin = 0 runs the unrestricted formalism on a
    closed-shell system at ~2x the SCF cost.  Almost always a user
    mistake (default-of-RKS user flipped to UKS to "be safe").  Warn
    so the user knows."""
    cfg = PySCFConfig(method="UKS", spin=0)
    issues = validate(water_struct, cfg)
    method_warns = [i for i in issues
                    if i.severity == "warn" and i.where == "config.method"]
    assert len(method_warns) == 1
    assert "UKS" in method_warns[0].message
    assert "RKS" in method_warns[0].message



def test_pyscf_rks_with_spin_zero_no_warn(water_struct):
    """The flip side: RKS + spin=0 is the conventional closed-shell
    setup; no warning."""
    cfg = PySCFConfig(method="RKS", spin=0)
    issues = validate(water_struct, cfg)
    assert [i for i in issues if i.where == "config.method"] == []



def test_pyscf_uks_with_spin_nonzero_no_warn():
    """UKS + spin > 0 is correct open-shell; no warning."""
    s = Structure(elements=["C","H","H","H"],
                  positions=np.array([[0,0,0],[1.08,0,0],[-0.54,0.94,0],[-0.54,-0.94,0]]))
    cfg = PySCFConfig(method="UKS", spin=1)
    issues = validate(s, cfg)
    assert [i for i in issues if i.where == "config.method"] == []



def test_pyscf_grid_level_3_with_hybrid_warns(water_struct):
    """Hybrid functionals (B3LYP / PBE0 / M06-2X / wB97X) at grid_level
    < 4 give noisy forces.  The user can override but should know."""
    cfg = PySCFConfig(method="RKS", spin=0, functional="B3LYP", grid_level=3)
    issues = validate(water_struct, cfg)
    grid_warns = [i for i in issues if i.where == "config.grid_level"
                  and i.severity == "warn"
                  and "hybrid" in i.message.lower()]
    assert len(grid_warns) == 1



def test_pyscf_grid_level_3_with_pure_gga_no_warn(water_struct):
    """Pure LDA/GGAs (PBE / BLYP / BP86 / revPBE) are grid-robust at
    grid_level 3 — no τ-dependence — so the validator must not warn.
    (TPSS/SCAN are meta-GGAs, NOT pure GGAs; they DO warn — see the
    meta-GGA test below.)"""
    cfg = PySCFConfig(method="RKS", spin=0, functional="PBE", grid_level=3)
    issues = validate(water_struct, cfg)
    grid_warns = [i for i in issues if i.where == "config.grid_level"]
    assert grid_warns == []


@pytest.mark.parametrize("functional", ["SCAN", "TPSS", "M06-L", "r2SCAN"])
def test_pyscf_grid_level_3_with_meta_gga_warns(water_struct, functional):
    """SCIENTIFIC-AUDIT FIX (FN-1): the grid-sensitive class is META-GGA
    (τ-dependent XC — SCAN/TPSS/M06-L/…), NOT "hybrids" (whose HF
    exchange is analytic, off-grid).  A meta-GGA Hessian/opt at grid < 4
    must warn.  Pre-2026-07 the gate keyed on "hybrid" and SCAN/TPSS
    passed SILENTLY — the false-negative this fixes."""
    cfg = PySCFConfig(method="RKS", spin=0, functional=functional, grid_level=3)
    issues = validate(water_struct, cfg)
    grid_warns = [i for i in issues if i.where == "config.grid_level"
                  and i.severity == "warn"
                  and "meta-gga" in i.message.lower()]
    assert len(grid_warns) == 1, (
        f"{functional} (meta-GGA) at grid 3 should warn; got {grid_warns}")



def test_pyscf_default_grid_level_is_hybrid_safe():
    """Default grid_level should be >= 4 so the default
    `B3LYP + def2-SVP + density_fit + d3bj` recipe doesn't trip the
    hybrid-grid warning on its own defaults."""
    cfg = PySCFConfig()
    assert cfg.grid_level >= 4, (
        f"PySCFConfig.grid_level default = {cfg.grid_level}; should be "
        f">= 4 so the default hybrid recipe doesn't self-warn"
    )



# --------------------------------------------------------------------- #
#  R3: odd-electron count + RKS/RHF + spin=0 must warn                  #
# --------------------------------------------------------------------- #


def _ch3_radical_struct():
    """Methyl radical CH3 -- a textbook open-shell organic.
    n_e (neutral) = 6 (C) + 3 (H) = 9, odd -> doublet."""
    return Structure(
        elements=["C", "H", "H", "H"],
        positions=np.array([
            [ 0.000,  0.000, 0.000],
            [ 1.080,  0.000, 0.000],
            [-0.540,  0.935, 0.000],
            [-0.540, -0.935, 0.000],
        ]),
        title="ch3_radical",
    )


def test_pyscf_validator_warns_on_odd_electrons_with_rks_spin0():
    """R3: a system with an odd electron count run with method=RKS and
    spin=0 is silently the wrong electronic state -- the validator
    must emit a warn pointing at UKS / spin=1."""
    cfg = PySCFConfig(method="RKS", spin=0, charge=0)
    issues = validate(_ch3_radical_struct(), cfg)
    odd_warns = [i for i in issues
                 if i.severity == "warn" and "odd electron" in i.message]
    assert len(odd_warns) == 1
    assert "UKS" in odd_warns[0].message
    assert "UHF" in odd_warns[0].message



def test_pyscf_validator_silent_on_odd_electrons_with_uks_spin1():
    """The legitimate fix (UKS + spin=1 on the same odd-electron
    system) must NOT trip the new R3 warn."""
    cfg = PySCFConfig(method="UKS", spin=1, charge=0)
    issues = validate(_ch3_radical_struct(), cfg)
    odd_warns = [i for i in issues if "odd electron" in i.message]
    assert odd_warns == []



def test_pyscf_validator_silent_on_even_electrons_with_rks():
    """Closed-shell organic on RKS+spin=0: no R3 warn."""
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ]),
        title="water",
    )
    cfg = PySCFConfig(method="RKS", spin=0, charge=0)
    issues = validate(s, cfg)
    odd_warns = [i for i in issues if "odd electron" in i.message]
    assert odd_warns == []



def test_pyscf_validator_silent_when_user_overrode_charge_to_make_even():
    """If the user explicitly set cfg.charge to a value that makes the
    electron count even (e.g. methyl cation CH3+ with charge=+1, n_e=8,
    a closed-shell carbocation), R3 must NOT warn."""
    cfg = PySCFConfig(method="RKS", spin=0, charge=1)
    issues = validate(_ch3_radical_struct(), cfg)
    odd_warns = [i for i in issues if "odd electron" in i.message]
    assert odd_warns == []


def test_basis_adequacy_fires_for_closed_shell_metal():
    """SCIENTIFIC-AUDIT FOLLOW-UP: d-orbital basis coverage matters for
    CLOSED-shell d10 metals (Zn/Cd/Hg/Pd/Pt) too, not only open-shell ones
    -- the concern is orbital coverage, orthogonal to spin state.  A minimal
    basis on a Zn complex used to pass unflagged."""
    import numpy as np
    from molbuilder.structure import Structure
    zn = Structure(elements=["Zn", "O"],
                   positions=np.array([[0., 0, 0], [0, 0, 1.7]]))
    cfg = PySCFConfig(method="RKS", spin=0, basis="STO-3G")
    issues = validate(zn, cfg)
    assert any(i.where == "config.basis" and i.severity == "warn"
               and "Zn" in i.message for i in issues), (
        "STO-3G on a closed-shell Zn complex should warn on d-orbital "
        "basis inadequacy")
