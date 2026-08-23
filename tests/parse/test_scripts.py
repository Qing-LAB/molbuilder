"""L2 tests for Phase F script-block TextParsers.

Pins:
  * Each per-block TextParser extracts its block + leaves other
    fields default-None.
  * Block-absent → field stays None (not empty dict / empty
    string) — the present-vs-absent distinction is load-bearing
    for the script-contract.md § 4.4 emission rule.
  * Umbrella ScriptSourceTextParser composes all 5 + merges
    schema_versions.
  * parse_text() dispatches via the TextParser ABC (no
    auto-detection — caller picks).
  * TextParsers do no I/O (lint-style assertion on imports).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import (
    ScriptResult,
    parse_text,
)
from molbuilder.parse.scripts import (
    AtomMetadataTextParser,
    BenchMarksTextParser,
    HeaderTextParser,
    ProvenanceTextParser,
    ScriptSourceTextParser,
    UserCustomTextParser,
)


# Synthetic fdf with EVERY block populated.  Round-trip target.
FULL_FDF = """\
# === molbuilder header BEGIN ===
# Run with mpirun -np 4 siesta < test.fdf > test.out
# === molbuilder header END ===

# === molbuilder provenance BEGIN ===
#   generator-version    git abc1234
#   generated-at         2026-06-19T12:00:00Z
#   resolved-defaults:
#     mpi_np            auto -> 4
#     use_gpu        true
# === molbuilder provenance END ===

# === molbuilder bench-marks BEGIN ===
#   version v1
#   n_atoms             5
#   gpu_mode            true
#
#   field MeshCutoff       anchor=MeshCutoff      type=float unit=Ry default=400.0
# === molbuilder bench-marks END ===

# === molbuilder atom-metadata BEGIN ===
# format: molstruct-json/v3
# {
#   "schema_version": 3,
#   "n_atoms_total":  5,
#   "regions":        {"bridge": [0, 1, 2, 3, 4]},
#   "frozen_atoms":   [],
#   "created_by":     "test",
#   "created_at":     "2026-06-19T12:00:00Z"
# }
# === molbuilder atom-metadata END ===

SystemLabel test
NumberOfAtoms 5

# === molbuilder user-custom BEGIN ===
# This is the user's territory
# Any text goes here
# === molbuilder user-custom END ===
"""


# Per-block extractors --------------------------------------------- #


def test_header_extracts_when_present():
    result = parse_text(FULL_FDF, parser=HeaderTextParser)
    assert isinstance(result, ScriptResult)
    assert result.result_kind == "script"
    assert result.parser_name == "fdf-header"
    assert result.header is not None
    assert "Run with mpirun" in result.header
    # Other fields stay default-None — this is per-block, not umbrella.
    assert result.provenance is None
    assert result.bench_marks is None


def test_header_returns_none_when_absent():
    """Block-absent → field stays None (NOT empty string).  The
    present-vs-absent distinction matters for script-contract.md
    § 4.4 emission rule."""
    minimal = "SystemLabel test\n"   # no HEADER block
    result = parse_text(minimal, parser=HeaderTextParser)
    assert result.header is None


def test_provenance_extracts_flat_dict():
    result = parse_text(FULL_FDF, parser=ProvenanceTextParser)
    assert result.provenance is not None
    assert result.provenance.get("generator-version") == "git abc1234"
    # resolved-defaults sub-block becomes "resolved-defaults.<key>"
    assert "resolved-defaults.mpi_np" in result.provenance


def test_bench_marks_extracts_version_and_fields():
    result = parse_text(FULL_FDF, parser=BenchMarksTextParser)
    assert result.bench_marks is not None
    assert result.bench_marks.get("version") == "v1"
    fields = result.bench_marks.get("fields", [])
    assert any(f.get("name") == "MeshCutoff" for f in fields)


def test_atom_metadata_extracts_v3_payload_and_surfaces_schema_version():
    result = parse_text(FULL_FDF, parser=AtomMetadataTextParser)
    assert result.atom_metadata is not None
    assert result.atom_metadata.get("schema_version") == 3
    assert result.atom_metadata.get("regions") == {"bridge": [0, 1, 2, 3, 4]}
    # block_schema_versions surfaces the per-block version for
    # downstream cross-block audit.
    assert result.block_schema_versions.get("atom-metadata") == 3


def test_user_custom_extracts_inner_lines():
    result = parse_text(FULL_FDF, parser=UserCustomTextParser)
    assert result.user_custom is not None
    assert isinstance(result.user_custom, list)
    # Lines are kept as-is (with the leading "# " from the .fdf
    # comment-prefix convention; legacy preserves byte-for-byte).
    assert any("user's territory" in line for line in result.user_custom)


# Umbrella ------------------------------------------------------- #


def test_script_source_composes_all_blocks():
    result = parse_text(FULL_FDF, parser=ScriptSourceTextParser)
    assert isinstance(result, ScriptResult)
    assert result.parser_name == "fdf-script-source"
    assert result.header is not None
    assert result.provenance is not None
    assert result.bench_marks is not None
    assert result.atom_metadata is not None
    assert result.user_custom is not None
    # block_schema_versions merges per-block (atom-metadata v3 +
    # bench-marks v1).
    assert result.block_schema_versions.get("atom-metadata") == 3
    assert result.block_schema_versions.get("bench-marks") == "v1"


def test_script_source_handles_no_blocks():
    """A vanilla .fdf with NO script-contract blocks gives every
    field as None / empty (graceful degradation)."""
    minimal = "SystemLabel test\nNumberOfAtoms 5\n"
    result = parse_text(minimal, parser=ScriptSourceTextParser)
    assert result.header is None
    assert result.provenance is None
    assert result.bench_marks is None
    assert result.atom_metadata is None
    assert result.user_custom is None
    assert result.block_schema_versions == {}


# Lint: TextParsers do no I/O ----------------------------------- #


def test_text_parsers_do_no_io():
    """Per parse-module.md § 9 forbidden #2: TextParsers do NO I/O.
    Lint by importing each module's source + asserting no
    open / read_text / Path() with .read in it."""
    from pathlib import Path as _Path
    base = _Path(__file__).resolve().parents[2] / "molbuilder" / "parse" / "scripts"
    forbidden = ("read_text", "open(", ".read()", "Path(")
    for mod in (
        "header.py", "provenance.py", "bench_marks.py",
        "atom_metadata.py", "user_custom.py", "source.py",
    ):
        text = (base / mod).read_text()
        for token in forbidden:
            assert token not in text, (
                f"parse/scripts/{mod} contains forbidden I/O token "
                f"{token!r} — TextParsers must operate on text in "
                f"memory only (parse-module.md § 9 forbidden #2)"
            )


# Frozen invariant ------------------------------------------------- #


def test_script_result_is_frozen():
    from dataclasses import FrozenInstanceError
    r = parse_text("", parser=ScriptSourceTextParser)
    with pytest.raises(FrozenInstanceError):
        r.header = "tampered"   # noqa
