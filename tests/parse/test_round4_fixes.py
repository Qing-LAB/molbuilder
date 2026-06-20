"""Round-4 review-fix regression tests.

Pins:
  * D.2 — UTF-8 BOM-prefixed .fdf files parse correctly.
    Pre-fix: ``read_text(encoding="utf-8")`` preserved the BOM,
    which then prevented engine_body_summary from matching the
    first-line key (``\\ufeffMeshCutoff`` != ``MeshCutoff``).
    SystemLabel regex + script-contract MARKER_RE would also
    silently fail on a BOM'd first line.
  * E.1 — `_KGRID_BLOCK_RE` extraction tolerates blank lines /
    comments within the %block.  Pre-fix: ``toks[len(rows)]``
    silently failed (IndexError swallowed) and the kgrid summary
    was None for a block with any non-row line.
"""

from __future__ import annotations

import codecs
from pathlib import Path

import pytest

from molbuilder.parse import parse_dir
from molbuilder.parse.dirs.job import (
    _parse_engine_body_summary,
    decode_run_dir,
)


# ---- D.2 UTF-8 BOM handling ---------------------------------------- #


def test_d2_bom_prefixed_fdf_parses_key():
    """A BOM-prefixed .fdf must NOT silently lose its first-line
    key.  This was the round-4 BLOCKER: read_text(encoding='utf-8')
    preserves the BOM; the key 'MeshCutoff' becomes '\\ufeffMeshCutoff'
    which doesn't match the curated key lookup."""
    bom_text = codecs.BOM_UTF8.decode("utf-8") + (
        "MeshCutoff 350.0 Ry\n"
        "PAO.BasisSize DZP\n"
    )
    summary = _parse_engine_body_summary(bom_text)
    assert summary["MeshCutoff"] == "350.0 Ry"
    assert summary["PAO.BasisSize"] == "DZP"


def test_d2_bom_round_trip_through_decode_run_dir(tmp_path: Path):
    """End-to-end: decode_run_dir on a project dir whose .fdf was
    saved with a BOM gives the same engine_input_summary as a
    BOM-less copy.  The utf-8-sig encoding switch on read_text
    must hold at the directory-level pipeline."""
    fdf = tmp_path / "test-stage1.fdf"
    bom_text = codecs.BOM_UTF8.decode("utf-8") + (
        "SystemLabel test\n"
        "MD.TypeOfRun CG\n"
        "MD.NumCGsteps 100\n"
        "MeshCutoff 350.0 Ry\n"
        "PAO.BasisSize DZP\n"
        "XC.functional GGA\n"
        "XC.authors PBE\n"
    )
    fdf.write_bytes(bom_text.encode("utf-8"))
    result = decode_run_dir(tmp_path)
    summary = next(iter(result.engine_input_by_stage.values()))["engine_body_summary"]
    # BOM must be stripped before key matching.
    assert summary["SystemLabel"] == "test"
    assert summary["MeshCutoff"] == "350.0 Ry"
    # system_label is also extracted via a separate regex on the
    # SAME text — must also work with the BOM stripped.
    assert result.system_label == "test"


# ---- E.1 kgrid block robustness ------------------------------------ #


def test_e1_kgrid_extracts_diagonal_simple():
    """Baseline: well-formed kgrid block extracts the diagonal."""
    fdf_text = (
        "%block kgrid_Monkhorst_Pack\n"
        "4 0 0 0.0\n"
        "0 5 0 0.0\n"
        "0 0 6 0.0\n"
        "%endblock kgrid_Monkhorst_Pack\n"
    )
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["kgrid_Monkhorst_Pack"] == "4x5x6"


def test_e1_kgrid_tolerates_blank_line_in_block():
    """Round-4 fix: a blank line inside the kgrid %block must NOT
    silently drop the summary.  Pre-fix: ``toks[len(rows)]`` would
    fail on the blank line; rows stays short; output is None."""
    fdf_text = (
        "%block kgrid_Monkhorst_Pack\n"
        "4 0 0 0.0\n"
        "\n"                              # blank line
        "0 5 0 0.0\n"
        "0 0 6 0.0\n"
        "%endblock kgrid_Monkhorst_Pack\n"
    )
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["kgrid_Monkhorst_Pack"] == "4x5x6"


def test_e1_kgrid_tolerates_comment_in_block():
    """Same for a comment line."""
    fdf_text = (
        "%block kgrid_Monkhorst_Pack\n"
        "# Monkhorst-Pack 4x5x6\n"
        "4 0 0 0.0\n"
        "0 5 0 0.0\n"
        "0 0 6 0.0\n"
        "%endblock kgrid_Monkhorst_Pack\n"
    )
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["kgrid_Monkhorst_Pack"] == "4x5x6"


def test_e1_kgrid_malformed_falls_to_none():
    """Round-4 fix preserves safe behavior: if the block doesn't
    have 3 valid rows, summary stays None — silent failure is
    appropriate when the .fdf itself is malformed."""
    fdf_text = (
        "%block kgrid_Monkhorst_Pack\n"
        "4 0\n"                            # too few tokens
        "%endblock kgrid_Monkhorst_Pack\n"
    )
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["kgrid_Monkhorst_Pack"] is None
