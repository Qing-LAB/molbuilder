"""One content-format sniffer, used by BOTH /api/build/molecule and /api/build/load.

Regression: build.py used to carry two sniffers that disagreed on a leading "0"
line -- ``_sniff_structure_format`` (``int(line) > 0`` -> pdb) vs an inline
``first.strip().isdigit()`` (-> xyz).  Both load paths now delegate to the ONE
helper, so the classification is identical everywhere.
"""
from __future__ import annotations

from molbuilder.web.blueprints.build import _sniff_structure_format


def test_positive_count_is_xyz():
    assert _sniff_structure_format("3\nh2o\nO 0 0 0\nH 1 0 0\nH 0 1 0\n") == "xyz"


def test_leading_zero_line_is_pdb_not_xyz():
    # int("0") > 0 is False -> pdb.  The old inline `"0".isdigit()` said xyz; the
    # single helper removes that disagreement.
    assert _sniff_structure_format("0\n") == "pdb"


def test_pdb_header_before_first_atom_is_pdb():
    assert _sniff_structure_format("HEADER    DNA\nTITLE     x\nATOM  ...\n") == "pdb"


def test_blank_leading_lines_skipped():
    assert _sniff_structure_format("\n\n  \n2\nx\nH 0 0 0\nH 1 0 0\n") == "xyz"


def test_empty_is_pdb():
    assert _sniff_structure_format("") == "pdb"


def test_the_inline_load_sniffer_is_gone():
    # The /api/build/load path must NOT re-implement the sniff inline; it delegates
    # to _sniff_structure_format.  Guard against the duplication creeping back.
    import inspect
    import molbuilder.web.blueprints.build as build_mod
    src = inspect.getsource(build_mod)
    assert 'first.strip().isdigit()' not in src, (
        "the inline content sniffer is back -- both load paths must call "
        "_sniff_structure_format")
