"""Regression tests for the 2026-06-19 review fixes on
parse/dirs/job.py (BLOCKERS B1-B3 + IMPORTANT I1-I4).

Each test pins the previously-broken behaviour against a focused
synthetic fixture so the fix can't silently revert.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse.dirs.job import (
    JobDirParser,
    _parse_engine_body_summary,
    _read_cell_from_fdf,
)


def test_b2_cell_scales_by_lattice_constant_ang(tmp_path: Path) -> None:
    """B2: %block LatticeVectors values must be multiplied by
    LatticeConstant.  Identity vectors + 5.43 Ang should give a
    5.43-Å cube, not a 1-Å cube."""
    fdf = tmp_path / "test.fdf"
    fdf.write_text(
        "SystemLabel test\n"
        "LatticeConstant 5.43 Ang\n"
        "%block LatticeVectors\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 1.0\n"
        "%endblock LatticeVectors\n"
    )
    cell = _read_cell_from_fdf(fdf)
    assert cell is not None
    assert abs(cell[0][0] - 5.43) < 1e-6
    assert abs(cell[1][1] - 5.43) < 1e-6
    assert abs(cell[2][2] - 5.43) < 1e-6


def test_b2_cell_scales_by_lattice_constant_bohr(tmp_path: Path) -> None:
    """B2: LatticeConstant in Bohr converts to Å (* 0.529...).
    SIESTA's default unit is Bohr if not specified."""
    fdf = tmp_path / "test.fdf"
    fdf.write_text(
        "SystemLabel test\n"
        "LatticeConstant 10.2625 Bohr\n"
        "%block LatticeVectors\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 1.0\n"
        "%endblock LatticeVectors\n"
    )
    cell = _read_cell_from_fdf(fdf)
    assert cell is not None
    # 10.2625 Bohr × (1/1.8897259886) = 5.430 Å (Si lattice).
    assert abs(cell[0][0] - 5.430) < 1e-3


def test_b2_default_unit_is_bohr_when_unspecified(tmp_path: Path) -> None:
    """B2: SIESTA defaults LatticeConstant to Bohr when no unit
    is given.  Catches the worst-case silent-bug pattern."""
    fdf = tmp_path / "test.fdf"
    fdf.write_text(
        "SystemLabel test\n"
        "LatticeConstant 10.2625\n"      # no unit -> Bohr
        "%block LatticeVectors\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 1.0\n"
        "%endblock LatticeVectors\n"
    )
    cell = _read_cell_from_fdf(fdf)
    assert cell is not None
    assert abs(cell[0][0] - 5.430) < 1e-3


def test_b3_can_parse_does_not_claim_py_only_dir(tmp_path: Path) -> None:
    """B3: JobDirParser must NOT claim a directory whose only
    engine script is a PySCF .py file — the decoder returns empty
    for it.  Re-add when PySCF DirParser actually exists."""
    (tmp_path / "calc.py").write_text("# PySCF calc\n")
    assert not JobDirParser.can_parse(tmp_path)


def test_b3_can_parse_claims_fdf_dir(tmp_path: Path) -> None:
    """B3 control: JobDirParser DOES claim a .fdf-only dir."""
    (tmp_path / "calc.fdf").write_text("SystemLabel test\n")
    assert JobDirParser.can_parse(tmp_path)


def test_i2_engine_body_summary_case_insensitive() -> None:
    """I2: SIESTA fdf keys are case-insensitive.  ``meshcutoff``
    must match ``MeshCutoff`` and produce the canonical-cased key
    in the output dict."""
    fdf_text = (
        "systemlabel test\n"
        "meshcutoff 350.0 Ry\n"
        "MD.TYPEofRun CG\n"
        "Xc.Functional GGA\n"
    )
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["SystemLabel"] == "test"
    assert summary["MeshCutoff"] == "350.0 Ry"
    assert summary["MD.TypeOfRun"] == "CG"
    assert summary["XC.functional"] == "GGA"


def test_i2_engine_body_summary_tab_separated() -> None:
    """I2: SIESTA accepts tab as separator; previous parser
    required a space.  Tab-separated MeshCutoff must parse."""
    fdf_text = "MeshCutoff\t350.0\tRy\n"
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["MeshCutoff"] == "350.0\tRy"


def test_i2_engine_body_summary_canonical_keys_unchanged() -> None:
    """I2: canonical-cased input still works after the I2 fix."""
    fdf_text = (
        "SystemLabel canonical\n"
        "MeshCutoff 200.0 Ry\n"
    )
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["SystemLabel"] == "canonical"
    assert summary["MeshCutoff"] == "200.0 Ry"
