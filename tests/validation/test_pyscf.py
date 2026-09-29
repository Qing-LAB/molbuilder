"""Tests for molbuilder.validation.pyscf.

Per docs/process/testing.md (test layout mirrors source
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
#  PySCF: field range validation                                        #
#                                                                       #
#  The spin's findings (a restricted count, parity, a floating moment)  #
#  are the electronic state's one family since 2026-09-28, pinned        #
#  through prep in tests/test_electronic_state.py.                      #
# --------------------------------------------------------------------- #


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


def test_pyscf_grid_level_3_with_hybrid_warns(water_struct):
    """Hybrid functionals (B3LYP / PBE0 / M06-2X / wB97X) at grid_level
    < 4 give noisy forces.  The user can override but should know."""
    cfg = PySCFConfig(functional="B3LYP", grid_level=3)
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
    cfg = PySCFConfig(functional="PBE", grid_level=3)
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
    cfg = PySCFConfig(functional=functional, grid_level=3)
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


def test_unconsumed_region_labels_are_named_on_every_deck_route(water_struct):
    """Pattern B, re-homed (validation.md § 5, C-shared 2026-08-21): the
    notice rode two deleted web endpoints and fired NOWHERE; it now runs
    inside the engine validators, so the CLI's prep gate says it too.
    The reserved frozen label is excluded -- the engines consume it."""
    from molbuilder.structure import FROZEN_LABEL
    s = water_struct
    s.regions["L-electrode"] = [0]
    s.regions[FROZEN_LABEL] = [1]
    issues = validate(s, PySCFConfig())
    named = [i for i in issues if i.where == "structure.regions"]
    assert len(named) == 1
    # The label LIST is what must exclude the consumed frozen label --
    # the trailing advice sentence always spells "frozen_atoms", so the
    # old prefix-split assert could not fail.
    listed = named[0].message.split("]", 1)[0]
    assert "L-electrode" in listed
    assert "frozen_atoms" not in listed, (
        "the consumed frozen label is listed as unconsumed")
    # SIESTA's validator runs the same body (same structure object on
    # purpose -- the labels set above are the fixture).
    from molbuilder.siesta import SiestaConfig
    issues2 = validate(s, SiestaConfig(system_label="JOB"))
    assert any(i.where == "structure.regions" for i in issues2)
