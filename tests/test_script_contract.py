"""Unit tests for molbuilder.script_contract emitters.

Pins the block shape so a future format change is loud.  Tests are
pure: no rendering, no I/O.
"""
from __future__ import annotations

import json
import re

from molbuilder import script_contract as sc


# --------------------------------------------------------------------- #
#  Markers                                                              #
# --------------------------------------------------------------------- #


def test_begin_end_markers_are_literal():
    assert sc.begin_marker("provenance") == "# === molbuilder provenance BEGIN ==="
    assert sc.end_marker("provenance")   == "# === molbuilder provenance END ==="


def test_marker_re_matches_begin_and_end():
    m = sc.MARKER_RE.match("# === molbuilder provenance BEGIN ===")
    assert m and m.group(1) == "provenance" and m.group(2) == "BEGIN"
    m = sc.MARKER_RE.match("# === molbuilder bench-marks END ===")
    assert m and m.group(1) == "bench-marks" and m.group(2) == "END"


def test_marker_re_rejects_non_markers():
    for line in [
        "BlockSize 64",
        "# Just a comment",
        "# === something else BEGIN ===",
        "# === molbuilder ===",
    ]:
        assert sc.MARKER_RE.match(line) is None, line


# --------------------------------------------------------------------- #
#  emit_header                                                          #
# --------------------------------------------------------------------- #


def test_emit_header_wraps_given_lines():
    block = sc.emit_header(["# line one", "# line two"])
    assert block.splitlines()[0]  == "# === molbuilder header BEGIN ==="
    assert block.splitlines()[-1] == "# === molbuilder header END ==="
    assert "# line one" in block
    assert "# line two" in block


def test_emit_header_empty_list_produces_empty_block():
    block = sc.emit_header([])
    lines = block.splitlines()
    assert lines[0]  == "# === molbuilder header BEGIN ==="
    assert lines[-1] == "# === molbuilder header END ==="
    assert len(lines) == 2  # no body


# --------------------------------------------------------------------- #
#  emit_provenance                                                      #
# --------------------------------------------------------------------- #


def test_emit_provenance_includes_required_fields():
    block = sc.emit_provenance(
        generator_version="git abc1234",
        generated_at="2026-06-16T17:00:00-07:00",
        resolved_defaults={"BlockSize": "auto -> 256", "enable_gpu": "true"},
    )
    assert "git abc1234" in block
    assert "2026-06-16T17:00:00-07:00" in block
    assert "BlockSize" in block and "256" in block
    assert "enable_gpu" in block
    assert block.splitlines()[0] == "# === molbuilder provenance BEGIN ==="


def test_emit_provenance_without_defaults_still_emits_block():
    block = sc.emit_provenance(
        generator_version="git abc",
        generated_at="2026-06-16T17:00:00-07:00",
    )
    assert "# === molbuilder provenance BEGIN ===" in block
    assert "# === molbuilder provenance END ===" in block


# --------------------------------------------------------------------- #
#  emit_bench_marks                                                     #
# --------------------------------------------------------------------- #


def test_emit_bench_marks_uses_anchors_not_line_numbers():
    """v1 contract change: anchor= replaces line=N for layout-stability."""
    block = sc.emit_bench_marks(
        metadata={"n_atoms": 212, "gpu_mode": "true"},
        fields=sc.SIESTA_BENCH_FIELDS,
        defaults={"BlockSize": 256, "MaxSCFIterations": 500,
                  "MD.NumCGsteps": 200, "MeshCutoff": 400.0},
    )
    assert "anchor=BlockSize"        in block
    assert "anchor=MaxSCFIterations" in block
    assert "anchor=MD.NumCGsteps"    in block
    assert "anchor=MeshCutoff"       in block
    # No regression to line-number form.
    assert "line=" not in block


def test_emit_bench_marks_default_value_appears_per_field():
    block = sc.emit_bench_marks(
        metadata={"n_atoms": 212},
        fields=sc.SIESTA_BENCH_FIELDS,
        defaults={"BlockSize": 256, "MaxSCFIterations": 500,
                  "MD.NumCGsteps": 200, "MeshCutoff": 400.0},
    )
    assert "default=256"   in block
    assert "default=500"   in block
    assert "default=200"   in block
    assert "default=400.0" in block


def test_emit_bench_marks_carries_version_tag():
    block = sc.emit_bench_marks(metadata={}, fields=[], defaults={})
    assert "version v1" in block


# --------------------------------------------------------------------- #
#  emit_atom_metadata                                                   #
# --------------------------------------------------------------------- #


def test_emit_atom_metadata_returns_none_when_both_empty():
    """The contract's emission rule: absent block when no labels.
    Otherwise an empty in-body block would silently suppress a
    later .molstruct.json sidecar via the in-body-wins rule."""
    assert sc.emit_atom_metadata({}, [], n_atoms_total=100) is None
    assert sc.emit_atom_metadata(None, None, n_atoms_total=100) is None


def test_emit_atom_metadata_with_regions_only():
    block = sc.emit_atom_metadata(
        regions={"L-electrode": [0, 1, 2], "R-electrode": [10, 11]},
        frozen_atoms=[],
        n_atoms_total=20,
    )
    assert block is not None
    # Strip the leading "# " from each JSON body line and reparse.
    json_lines = [
        line[2:] if line.startswith("# ") else line[1:]
        for line in block.splitlines()
        if line.startswith("# ") or line == "#"
    ]
    # Find the JSON body (skip the marker + format header).
    json_text = "\n".join(line for line in json_lines if line.startswith(("{", " ", "}", '"')))
    payload = json.loads(json_text)
    assert payload["schema_version"] == 3
    assert payload["regions"] == {"L-electrode": [0, 1, 2], "R-electrode": [10, 11]}
    assert "frozen_atoms" not in payload  # was empty


def _atom_metadata_payload(block: str) -> dict:
    """Reverse the comment-prefix-per-line and reparse the JSON body."""
    body_lines = []
    in_json = False
    for raw in block.splitlines():
        if not raw.startswith("#"):
            continue
        line = raw[1:].lstrip() if raw.startswith("# ") else raw[1:]
        if line.strip() == "{":
            in_json = True
        if in_json:
            body_lines.append(line)
        if line.strip() == "}":
            in_json = False
    return json.loads("\n".join(body_lines))


def test_emit_atom_metadata_with_frozen_only_sorts_and_dedupes():
    block = sc.emit_atom_metadata(
        regions={},
        frozen_atoms=[5, 3, 1, 2, 1],  # unsorted + duplicate
        n_atoms_total=10,
    )
    assert block is not None
    payload = _atom_metadata_payload(block)
    assert payload["frozen_atoms"] == [1, 2, 3, 5]


def test_emit_atom_metadata_uses_zero_based_indices():
    """The contract pins 0-based atom indices (matches .molstruct.json
    schema v3 and Structure.regions in Python).  SIESTA's
    Geometry.Constraints in engine body remains 1-based by SIESTA
    convention -- the two coexist deliberately."""
    block = sc.emit_atom_metadata(
        regions={"single": [0]},  # 0-based
        frozen_atoms=[0],
        n_atoms_total=1,
    )
    payload = _atom_metadata_payload(block)
    # The JSON body must contain index 0 (would be 1 if anyone
    # accidentally added +1 shifting).
    assert payload["regions"]["single"] == [0]
    assert payload["frozen_atoms"] == [0]


def test_emit_atom_metadata_omits_structure_hash():
    """Per the contract: structure_hash is NOT emitted in-body because
    the metadata and coordinates are written by the same generator
    pass and cannot drift apart by construction."""
    block = sc.emit_atom_metadata(
        regions={"r": [0]}, frozen_atoms=[], n_atoms_total=1,
    )
    assert "structure_hash" not in block


def test_emit_atom_metadata_honors_created_by_and_created_at():
    """Audit fix 2026-06-16: callers (render_fdf / render_script)
    pass a specific ``created_by`` for traceability AND a real
    ``created_at`` timestamp.  Pin both."""
    block = sc.emit_atom_metadata(
        regions={"r": [0]},
        frozen_atoms=[],
        n_atoms_total=1,
        created_by="molbuilder render_fdf",
        created_at="2026-06-16T17:00:00-07:00",
    )
    payload = _atom_metadata_payload(block)
    assert payload["created_by"] == "molbuilder render_fdf"
    assert payload["created_at"] == "2026-06-16T17:00:00-07:00"


# --------------------------------------------------------------------- #
#  emit_user_custom_placeholder                                         #
# --------------------------------------------------------------------- #


def test_user_custom_placeholder_is_well_formed():
    block = sc.emit_user_custom_placeholder()
    lines = block.splitlines()
    assert lines[0]  == "# === molbuilder user-custom BEGIN ==="
    assert lines[-1] == "# === molbuilder user-custom END ==="


# --------------------------------------------------------------------- #
#  SIESTA bench fields registry                                         #
# --------------------------------------------------------------------- #


def test_siesta_bench_fields_cover_the_four_bench_knobs():
    names = {f.name for f in sc.SIESTA_BENCH_FIELDS}
    assert {"BlockSize", "MaxSCFIterations", "MD.NumCGsteps", "MeshCutoff"} <= names


def test_blocksize_field_constrains_pow2_and_range():
    bs = next(f for f in sc.SIESTA_BENCH_FIELDS if f.name == "BlockSize")
    assert bs.type_ == "pow2"
    assert bs.range_ == (16, 256)
