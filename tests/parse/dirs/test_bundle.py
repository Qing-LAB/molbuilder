"""L3 tests for Phase G BundleDirParser.

Pins:
  * BundleDirParser is NOT in the auto-dispatch registry (would
    collide with JobDirParser on shared "directory with .fdf"
    claims; bundle vs job is a different user intent).
  * Explicit BundleDirParser.parse() returns a typed BundleResult
    with structure + regions + frozen_atoms + notes from the
    legacy assemble_from_run_dir round-trip.
  * BundleError from the legacy assembler (ambiguous engines,
    missing scripts, atom-count mismatch) propagates unchanged.
  * BundleResult is frozen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import BundleResult
from molbuilder.parse.dirs.bundle import BundleDirParser
from molbuilder.parse.registry import _registered_dir_parsers
from molbuilder.script_bundle import BundleError


REPO = Path(__file__).resolve().parents[3]
TJ_DIR  = REPO / "projects" / "BDT" / "optimization" / "TJ-BDT-Au111"
BDT_DIR = REPO / "projects" / "BDT" / "optimization" / "BDT-withAuJunction"


def _need(p: Path) -> Path:
    if not p.is_dir():
        pytest.skip(f"fixture dir absent: {p}")
    return p


# Dispatch isolation ----------------------------------------------- #


def test_bundle_dir_parser_not_in_dispatch_registry():
    """BundleDirParser is explicit-dispatch; it must NOT appear
    in the auto-registered DirParser pool (else parse_dir on any
    job directory would raise AmbiguousFormatError with
    JobDirParser as the competing claimant)."""
    names = {p.name for p in _registered_dir_parsers()}
    assert "bundle-dir" not in names, (
        "BundleDirParser must not auto-register; bundle vs job "
        "dispatch ambiguity would deadlock parse_dir()")


def test_bundle_can_parse_claims_fdf_dir(tmp_path: Path):
    """can_parse mirrors the legacy assembler's precondition
    (at least one .fdf or .py present)."""
    (tmp_path / "test.fdf").write_text("SystemLabel test\n")
    assert BundleDirParser.can_parse(tmp_path)


def test_bundle_can_parse_rejects_empty_dir(tmp_path: Path):
    assert not BundleDirParser.can_parse(tmp_path)


# Real-fixture parse ------------------------------------------------ #


def test_parse_tj_bdt_au111_returns_bundleresult():
    """End-to-end: a project dir with script-contract atom-metadata
    + a .XV yields a complete BundleResult."""
    result = BundleDirParser.parse(_need(TJ_DIR))
    assert isinstance(result, BundleResult)
    assert result.result_kind == "bundle"
    assert result.parser_name == "bundle-dir"
    # Structure populated from the .XV (Phase E coords-parser
    # work; legacy assembler uses the same code path).
    assert result.structure is not None
    assert len(result.structure.elements) == 444    # known fixture size
    # Regions + frozen_atoms came from the in-fdf ATOM-METADATA.
    assert "L-electrode" in result.regions
    assert "R-electrode" in result.regions
    assert len(result.frozen_atoms) > 0
    # Notes — non-fatal diagnostics from the assembler.
    assert isinstance(result.notes, list)
    # Source path resolved absolute, matching envelope convention.
    assert Path(result.source).is_absolute()


def test_parse_bdt_withaujunction_no_atom_metadata():
    """Pre-script-contract .fdf (no ATOM-METADATA block) gives an
    empty regions + frozen_atoms but still a valid BundleResult."""
    result = BundleDirParser.parse(_need(BDT_DIR))
    assert isinstance(result, BundleResult)
    # ATOM-METADATA absent → regions empty dict (NOT None).
    assert result.regions == {}
    assert result.frozen_atoms == []
    # Structure still populated (from .XV / .fdf-initial coords).
    assert result.structure is not None


# BundleError propagation ------------------------------------------ #


def test_bundle_error_propagates_on_empty_dir(tmp_path: Path):
    """Legacy assembler raises BundleError on a dir with no
    engine scripts; the DirParser wrapper must propagate
    unchanged (NOT swallow + return empty result)."""
    with pytest.raises(BundleError):
        BundleDirParser.parse(tmp_path)


def test_bundle_error_propagates_on_ambiguous_engines(tmp_path: Path):
    """Both .fdf and .py in one dir is ambiguous; assembler
    raises BundleError."""
    (tmp_path / "calc.fdf").write_text("SystemLabel test\n")
    (tmp_path / "calc.py").write_text("# pyscf\n")
    with pytest.raises(BundleError):
        BundleDirParser.parse(tmp_path)


# Frozen + envelope ---------------------------------------------- #


def test_bundleresult_is_frozen():
    from dataclasses import FrozenInstanceError
    result = BundleDirParser.parse(_need(TJ_DIR))
    with pytest.raises(FrozenInstanceError):
        result.notes = []   # noqa


def test_bundleresult_parser_name_is_slug():
    """Envelope convention: parser_name is the slug 'bundle-dir',
    not the literal classname."""
    result = BundleDirParser.parse(_need(TJ_DIR))
    assert result.parser_name == "bundle-dir"
    assert result.parser_name == BundleDirParser.name
