"""One content-format sniffer, used by BOTH /api/build/molecule and /api/build/load.

Both load paths delegate to the ONE helper, so the classification is identical
everywhere -- two sniffers disagreed on a leading "0" line (``int(line) > 0``
-> pdb vs ``first.strip().isdigit()`` -> xyz).
"""
from __future__ import annotations

from molbuilder.web.blueprints.build import _sniff_structure_format


def test_positive_count_is_xyz():
    assert _sniff_structure_format("3\nh2o\nO 0 0 0\nH 1 0 0\nH 0 1 0\n") == "xyz"


def test_leading_zero_line_is_pdb_not_xyz():
    # int("0") > 0 is False -> pdb.
    assert _sniff_structure_format("0\n") == "pdb"


def test_pdb_header_before_first_atom_is_pdb():
    assert _sniff_structure_format("HEADER    DNA\nTITLE     x\nATOM  ...\n") == "pdb"


def test_blank_leading_lines_skipped():
    assert _sniff_structure_format("\n\n  \n2\nx\nH 0 0 0\nH 1 0 0\n") == "xyz"
