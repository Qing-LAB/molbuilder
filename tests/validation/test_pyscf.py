"""Tests for molbuilder.validation.pyscf.

Per docs/process/testing.md (test layout mirrors source
layout).  Shared fixtures live in tests/validation/conftest.py.
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
#  PySCF: field range validation                                        #
#                                                                       #
#  The spin's findings (a restricted count, parity, a floating moment)  #
#  are the electronic state's one family, pinned through prep in        #
#  tests/test_electronic_state.py.                                      #
# --------------------------------------------------------------------- #


def test_pyscf_grid_level_above_range_warns(water_structure):
    """grid_level has metadata range (0, 9).  Beyond that value isn't
    meaningful in PySCF; warn the user before they generate a script
    that PySCF will reject."""
    cfg = PySCFConfig(grid_level=20)
    issues = validate(water_structure, cfg)
    msgs = [i for i in issues if i.where == "config.grid_level"]
    assert len(msgs) == 1


# --------------------------------------------------------------------- #
#  Method / functional / grid_level cross-check rules                  #
# --------------------------------------------------------------------- #


def test_pyscf_grid_level_3_with_hybrid_warns(water_structure):
    """Hybrid functionals (B3LYP / PBE0 / M06-2X / wB97X) at grid_level
    < 4 give noisy forces.  The user can override but should know."""
    cfg = PySCFConfig(functional="B3LYP", grid_level=3)
    issues = validate(water_structure, cfg)
    grid_warns = [i for i in issues if i.where == "config.grid_level"
                  and i.severity == "warn"
                  and "hybrid" in i.message.lower()]
    assert len(grid_warns) == 1


def test_pyscf_grid_level_3_with_pure_gga_no_warn(water_structure):
    """Pure LDA/GGAs (PBE / BLYP / BP86 / revPBE) are grid-robust at
    grid_level 3 — no τ-dependence — so the validator must not warn.
    (TPSS/SCAN are meta-GGAs, NOT pure GGAs; they DO warn — see the
    meta-GGA test below.)"""
    cfg = PySCFConfig(functional="PBE", grid_level=3)
    issues = validate(water_structure, cfg)
    grid_warns = [i for i in issues if i.where == "config.grid_level"]
    assert grid_warns == []


@pytest.mark.parametrize("functional", ["SCAN", "TPSS", "M06-L", "r2SCAN"])
def test_pyscf_grid_level_3_with_meta_gga_warns(water_structure, functional):
    """SCIENTIFIC-AUDIT FIX (FN-1): the grid-sensitive class is META-GGA
    (τ-dependent XC — SCAN/TPSS/M06-L/…), NOT "hybrids" (whose HF
    exchange is analytic, off-grid).  A meta-GGA Hessian/opt at grid < 4
    must warn.  Pre-2026-07 the gate keyed on "hybrid" and SCAN/TPSS
    passed SILENTLY — the false-negative this fixes."""
    cfg = PySCFConfig(functional=functional, grid_level=3)
    issues = validate(water_structure, cfg)
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


def test_basis_adequacy_fires_for_closed_shell_metal():
    """SCIENTIFIC-AUDIT FOLLOW-UP: d-orbital basis coverage matters for
    CLOSED-shell d10 metals (Zn/Cd/Hg/Pd/Pt) too, not only open-shell ones
    -- the concern is orbital coverage, orthogonal to spin state.  A minimal
    basis on a Zn complex used to pass unflagged."""
    import numpy as np
    from molbuilder.structure import Structure
    zn = Structure(elements=["Zn", "O"],
                   positions=np.array([[0., 0, 0], [0, 0, 1.7]]))
    cfg = PySCFConfig(basis="STO-3G")
    issues = validate(zn, cfg)
    assert any(i.where == "config.basis" and i.severity == "warn"
               and "Zn" in i.message for i in issues), (
        "STO-3G on a closed-shell Zn complex should warn on d-orbital "
        "basis inadequacy")
