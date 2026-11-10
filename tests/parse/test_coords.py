"""Tests for the coords FileParsers.


Pins:
  * SiestaXVFileParser claims .XV by its name, and not an XML file;
  * PySCFGeomFileParser claims _optimized.xyz, not a plain .xyz;
  * a PDB is read, its sidecar kept, one holding no coordinates refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import detect
from molbuilder.parse.coords import PySCFGeomFileParser, SiestaXVFileParser
from molbuilder.parse.errors import UnknownFormatError


# --------------------------------------------------------------------- #
#  .pdb                                                                 #
# --------------------------------------------------------------------- #

_PDB = ("HEADER    TEST\n"
        "ATOM      1  N   MET A   1       0.000   0.000   0.000"
        "  1.00  0.00           N\n"
        "ATOM      2  CA  MET A   1       1.400   0.000   0.000"
        "  1.00  0.00           C\n"
        "END\n")


def test_a_pdb_is_read_by_a_parser_so_the_picker_can_offer_it(tmp_path: Path):
    """`structure.js` claims `.xyz` AND `.pdb`, and the picker drops any
    file the server registry cannot read.

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
    """Capital ``.XV`` is the SIESTA convention; lowercase isn't.

    Asked of the REGISTRY, as the Results listing asks of each file.  THE
    SILENT FAILURE: the parser dropped from the registry, so a run's final
    coordinates are listed as nothing anyone can open -- the class's own
    ``can_parse`` still says yes.  `model/parse.md` § 3.
    """
    xv = tmp_path / "test.XV"
    xv.write_text("dummy\n")
    assert detect(str(xv)).name == "siesta-xv"
    xv_lower = tmp_path / "test.xv"
    xv_lower.write_text("dummy\n")
    with pytest.raises(UnknownFormatError):
        detect(str(xv_lower))


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
    here — `xyz`, the reader of any XYZ no other reader is for, reads those."""
    p = tmp_path / "structure.xyz"
    p.write_text("0\n\n")
    assert not PySCFGeomFileParser.can_parse(p)
