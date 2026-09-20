"""L2/L3 tests for Phase E coords FileParsers.

# Retired 2026-09-10: `@dataclass(frozen=True)` is the enforcement, and
# CPython refuses a non-frozen subclass of a frozen one on its own.
# A test that mutates an instance to watch Python raise tests Python.

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
    PdbFileParser,
    PySCFGeomFileParser,
    SiestaXVFileParser,
)
from molbuilder.parse.errors import UnknownFormatError
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
    assert "pdb" in names


# --------------------------------------------------------------------- #
#  .pdb -- a reader that existed for years with no row in the registry   #
# --------------------------------------------------------------------- #

_PDB = ("HEADER    TEST\n"
        "ATOM      1  N   MET A   1       0.000   0.000   0.000"
        "  1.00  0.00           N\n"
        "ATOM      2  CA  MET A   1       1.400   0.000   0.000"
        "  1.00  0.00           C\n"
        "END\n")


def test_a_pdb_is_read_by_a_parser_so_the_picker_can_offer_it(tmp_path: Path):
    """`structure.js` has claimed `.xyz` AND `.pdb` since it was written.

    Nothing in `parse/` claimed `.pdb`, and from 2026-09-18 the picker
    drops any file the server registry cannot read -- so half that
    presenter was unreachable from the Results tab.  The reader was
    already here (`Structure.from_pdb`); the gap was the registration.
    `StructureCodec.load` says so in its own comment: *"the file picker
    accepts .xyz AND .pdb, and each needs its own parser"*.
    """
    f = tmp_path / "mol.pdb"
    f.write_text(_PDB, encoding="utf-8")
    kind = detect(str(f))
    assert kind.name == "pdb"
    got = kind.parse(f)
    assert got.structure.elements == ["N", "C"]
    # No CRYST1 reader, so no cell -- stated, not invented.
    assert got.cell is None
    assert got.source_format == "pdb"


def test_a_long_header_does_not_hide_the_coordinates(tmp_path: Path):
    """An RCSB entry's preamble is not bounded, and ten real ones broke.

    `_looks_like_pdb` read a fixed 8 KB head for its first hour on
    2026-09-19, on the reasoning that a solvated system "reaches
    hundreds of megabytes".  That is true of the FILE and says nothing
    about the HEADER: measured over the 305 `.pdb` in this tree, the
    first coordinate record sits anywhere from byte 0 to 286011, and ten
    ordinary RCSB entries -- `1kx5` at 59940, `2acj` at 43821, the
    `2kei` NMR ensemble at 26406 -- carry REMARK/SEQRES/HELIX/SHEET past
    8 KB.  Each was refused by the registry while `StructureCodec.load`
    read it fine: two doors, two answers, which is the whole class of
    gap this parser was added to close.

    MUTATION THIS MUST FAIL AGAINST: read a fixed head instead of
    scanning to the first coordinate record.
    """
    f = tmp_path / "bulky.pdb"
    preamble = "".join(f"REMARK 999 {'x' * 60}\n" for _ in range(400))
    assert len(preamble) > 8192, "fixture must outgrow the old window"
    f.write_text(preamble + _PDB, encoding="utf-8")

    kind = detect(str(f))
    assert kind.name == "pdb"
    assert kind.parse(f).structure.elements == ["N", "C"]


def test_a_pdb_that_holds_no_coordinates_is_refused(tmp_path: Path):
    """THE CONTENT, not the suffix.

    A truncated download, a refusal page or somebody's notes saved with
    a `.pdb` name is not a structure, and claiming it puts an error card
    where a molecule should be.

    MUTATION THIS MUST FAIL AGAINST: drop the `_looks_like_pdb` call
    from `can_parse`, leaving the suffix test alone.
    """
    f = tmp_path / "notes.pdb"
    f.write_text("this is not a structure\njust some notes\n",
                 encoding="utf-8")
    with pytest.raises(UnknownFormatError):
        detect(str(f))


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
    # 5.43 Å = 5.43 / BOHR_ANGSTROM Bohr ≈ 10.2626 Bohr.  Imported, not
    # retyped: this WRITES the fixture, so it is input, and the assertion
    # below is on the 5.43 that comes back.
    from molbuilder.constants import BOHR_ANGSTROM as ang_per_bohr
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
    # 1e-3 Å is not a precision claim -- it discriminates the failure this
    # test names: a FORGOTTEN conversion returns 10.26 Å, a factor of 1.9.
    # Constant precision is not observable through a round trip at all
    # (the same value goes in and comes out); that needs an external
    # reference, and CODATA owns it, not us.
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


