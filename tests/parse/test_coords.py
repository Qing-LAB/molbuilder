"""L2/L3 tests for Phase E coords FileParsers.

Pins:
  * SiestaXVFileParser claims .XV, returns StructureResult with
    BOTH structure AND cell (the field that was missing in
    Phase 1 — closing the legacy Structure-is-geometry-only gap).
  * Cell vectors are in Å (legacy reads Bohr internally then
    discards; the wrapper now converts + surfaces).
  * Cell matches the actual file's lattice vectors (round-trip
    against a synthetic .XV with a known cell).
  * PySCFGeomFileParser claims _optimized.xyz, cell stays None.
  * detect() dispatches files to the new parsers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molbuilder.parse import (
    StructureResult,
    detect,
    parse,
)
from molbuilder.parse.coords import (
    PySCFGeomFileParser,
    SiestaXVFileParser,
)
from molbuilder.parse.registry import _registered_file_parsers


REPO = Path(__file__).resolve().parents[2]
# BUILT, NOT FOUND.  The fixture was a real run under projects/, behind a
# `pytest.skip("fixture absent")` -- so it read the user's scientific record
# on this machine and SKIPPED (green, proving nothing) anywhere else.


def _xv(tmp_path):
    """A valid SIESTA .XV written from the junction defined in source."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from support.junction import xv_file
    return xv_file(tmp_path / "junction.XV")






# Registration --------------------------------------------------------- #


def test_coords_parsers_registered():
    names = {p.name for p in _registered_file_parsers()}
    assert "siesta-xv" in names
    assert "pyscf-geom" in names


# can_parse ---------------------------------------------------------- #


def test_siesta_xv_claims_uppercase_extension(tmp_path: Path):
    """Capital ``.XV`` is the SIESTA convention; lowercase isn't."""
    xv = tmp_path / "test.XV"
    xv.write_text("dummy\n")
    assert SiestaXVFileParser.can_parse(xv)
    xv_lower = tmp_path / "test.xv"
    xv_lower.write_text("dummy\n")
    assert not SiestaXVFileParser.can_parse(xv_lower)


def test_siesta_xv_doesnt_claim_xml(tmp_path: Path):
    """The ``.XV`` suffix-match must NOT false-claim ``.xml``."""
    xml = tmp_path / "config.xml"
    xml.write_text("<?xml?>\n")
    assert not SiestaXVFileParser.can_parse(xml)


def test_pyscf_geom_claims_optimized_xyz(tmp_path: Path):
    p = tmp_path / "job_optimized.xyz"
    p.write_text("0\n\n")
    assert PySCFGeomFileParser.can_parse(p)


def test_pyscf_geom_doesnt_claim_plain_xyz(tmp_path: Path):
    """Plain ``.xyz`` files (without ``_optimized``) are not claimed
    here — the trajectory parser at engines/ handles those."""
    p = tmp_path / "structure.xyz"
    p.write_text("0\n\n")
    assert not PySCFGeomFileParser.can_parse(p)


# Real-file parse + cell surface ---------------------------------- #


def test_xv_parse_returns_structureresult_with_cell(tmp_path):
    """End-to-end: parse a real .XV via the registry, get back a
    StructureResult with structure + cell + source_format set."""
    result = parse(_xv(tmp_path))
    assert isinstance(result, StructureResult)
    assert result.result_kind == "structure"
    assert result.parser_name == "siesta-xv"
    assert result.source_format == "siesta-xv"
    assert result.structure is not None
    assert len(result.structure.elements) > 0
    # Cell is the load-bearing fix this phase brings.
    assert result.cell is not None
    assert result.cell.shape == (3, 3)
    # Cell is in Å, not Bohr; a typical molbuilder Au junction has
    # cell entries 30-60 Å (vacuum-padded supercell).
    diag = (abs(result.cell[0, 0]),
            abs(result.cell[1, 1]),
            abs(result.cell[2, 2]))
    assert all(d > 1.0 and d < 1000.0 for d in diag), (
        f"cell diagonal {diag} looks suspicious; expected Å scale")


def test_xv_cell_round_trip_against_synthetic_file(tmp_path: Path):
    """Build a synthetic .XV with a known cell (5.43 Å cube,
    silicon's lattice constant), parse it, confirm we get 5.43 Å.

    The wrapper converts Bohr → Å internally; this test pins the
    conversion + the surface (so a future "I forgot to convert"
    regression fails loudly).
    """
    # 5.43 Å = 5.43 / 0.529177249 Bohr ≈ 10.2626 Bohr
    ang_per_bohr = 0.529177249
    a_bohr = 5.43 / ang_per_bohr
    xv = tmp_path / "si.XV"
    # Cell rows: 6 numbers each (3 vector + 3 velocity); we only
    # read the first 3 of each row.
    xv.write_text(
        f"{a_bohr} 0 0 0 0 0\n"
        f"0 {a_bohr} 0 0 0 0\n"
        f"0 0 {a_bohr} 0 0 0\n"
        "2\n"
        "1 14 0.0 0.0 0.0 0 0 0\n"
        "1 14 1.0 1.0 1.0 0 0 0\n"
    )
    result = parse(xv)
    assert result.cell is not None
    # The diagonal should be 5.43 Å (within float roundoff).
    assert abs(result.cell[0, 0] - 5.43) < 1e-3
    assert abs(result.cell[1, 1] - 5.43) < 1e-3
    assert abs(result.cell[2, 2] - 5.43) < 1e-3
    # Off-diagonals zero.
    assert abs(result.cell[0, 1]) < 1e-9
    assert abs(result.cell[1, 0]) < 1e-9


def test_xv_structure_elements_match_atomic_numbers(tmp_path: Path):
    """Z=79 → 'Au' via the ase mapping the legacy reader uses."""
    a = 10.0
    xv = tmp_path / "au.XV"
    xv.write_text(
        f"{a} 0 0 0 0 0\n"
        f"0 {a} 0 0 0 0\n"
        f"0 0 {a} 0 0 0\n"
        "1\n"
        "1 79 0.0 0.0 0.0 0 0 0\n"
    )
    result = parse(xv)
    assert result.structure.elements == ["Au"]


def test_detect_routes_xv_to_siesta_xv_parser(tmp_path):
    cls = detect(_xv(tmp_path))
    assert cls is SiestaXVFileParser


# Frozen invariant ------------------------------------------------- #


def test_structureresult_is_frozen(tmp_path):
    """StructureResult inherits the frozen invariant from
    ParseResult.  Catches an accidental drop of frozen=True on
    the subclass."""
    from dataclasses import FrozenInstanceError
    r = parse(_xv(tmp_path))
    with pytest.raises(FrozenInstanceError):
        r.source_format = "tampered"
