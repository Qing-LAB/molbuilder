"""One rule for the format of a text with no name, asked by the build page's
load route and by the command line's stdin: two copies disagreed on a leading
"0" line (``int(line) > 0`` -> pdb vs ``first.isdigit()`` -> xyz).
"""
from __future__ import annotations

from molbuilder.workingcopy_structure import StructureCodec


def test_leading_zero_line_is_pdb_not_xyz():
    # int("0") > 0 is False -> pdb.
    assert StructureCodec.format_of_text("0\n") == "pdb"


def test_pdb_header_before_first_atom_is_pdb():
    assert StructureCodec.format_of_text("HEADER    DNA\nTITLE     x\nATOM  ...\n") == "pdb"


def test_blank_leading_lines_skipped():
    assert StructureCodec.format_of_text("\n\n  \n2\nx\nH 0 0 0\nH 1 0 0\n") == "xyz"
