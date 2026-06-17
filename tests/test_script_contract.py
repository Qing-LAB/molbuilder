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
#  USER-CUSTOM round-trip preservation (Step 2b)                       #
# --------------------------------------------------------------------- #


_SAMPLE_WITH_USER_CUSTOM = """\
some engine body
more engine body
# === molbuilder user-custom BEGIN ===
# user line one
# user line two
SomeUserDirective foo
# === molbuilder user-custom END ===
"""


def test_extract_user_custom_inner_returns_inner_lines():
    inner = sc.extract_user_custom_inner(_SAMPLE_WITH_USER_CUSTOM)
    assert inner == [
        "# user line one",
        "# user line two",
        "SomeUserDirective foo",
    ]


def test_extract_user_custom_inner_returns_none_when_no_block():
    assert sc.extract_user_custom_inner("BlockSize 64\nSystemName foo\n") is None


def test_extract_user_custom_inner_returns_none_on_unbalanced_markers():
    # BEGIN without END.
    text = "engine\n# === molbuilder user-custom BEGIN ===\nstuff\n"
    assert sc.extract_user_custom_inner(text) is None


def test_replace_user_custom_inner_splices_new_inner():
    placeholder = sc.emit_user_custom_placeholder()
    full = "engine line\n\n" + placeholder + "\n"
    spliced = sc.replace_user_custom_inner(
        full, ["# MY EDIT", "MY_KEYWORD value"]
    )
    inner = sc.extract_user_custom_inner(spliced)
    assert inner == ["# MY EDIT", "MY_KEYWORD value"]


def test_replace_user_custom_inner_no_change_when_no_block():
    text = "no markers here\n"
    assert sc.replace_user_custom_inner(text, ["X"]) == text


def test_merge_user_custom_from_target_preserves_existing(tmp_path):
    """The canonical Step 2b case: regenerated text has the placeholder,
    on-disk target has the user's actual edits.  Result: new text
    with user's edits spliced in."""
    target = tmp_path / "siesta-stage3.fdf"
    target.write_text(_SAMPLE_WITH_USER_CUSTOM, encoding="utf-8")

    fresh_render = (
        "fresh engine body\n\n"
        + sc.emit_user_custom_placeholder() + "\n"
    )
    merged = sc.merge_user_custom_from_target(fresh_render, target)
    inner = sc.extract_user_custom_inner(merged)
    assert inner == [
        "# user line one",
        "# user line two",
        "SomeUserDirective foo",
    ]
    # Engine body from the fresh render is preserved.
    assert "fresh engine body" in merged
    assert "some engine body" not in merged  # was the old engine body


def test_merge_user_custom_from_target_missing_file(tmp_path):
    """Target doesn't exist (first-time generate): merge returns
    rendered text unchanged."""
    target = tmp_path / "new.fdf"
    fresh = "engine\n" + sc.emit_user_custom_placeholder() + "\n"
    assert sc.merge_user_custom_from_target(fresh, target) == fresh


def test_merge_user_custom_from_target_target_lacks_block(tmp_path):
    """Target file exists but has no USER-CUSTOM block (e.g.,
    pre-contract file): merge returns rendered text unchanged."""
    target = tmp_path / "legacy.fdf"
    target.write_text("BlockSize 64\nSystemName foo\n", encoding="utf-8")
    fresh = "engine\n" + sc.emit_user_custom_placeholder() + "\n"
    assert sc.merge_user_custom_from_target(fresh, target) == fresh


def test_merge_user_custom_from_target_rendered_lacks_placeholder(tmp_path):
    """Rendered text has no USER-CUSTOM block (e.g., a non-contract
    file type someone is writing through /api/files/write): merge
    leaves it alone."""
    target = tmp_path / "siesta-stage3.fdf"
    target.write_text(_SAMPLE_WITH_USER_CUSTOM, encoding="utf-8")
    fresh = "no placeholder in this content\n"
    assert sc.merge_user_custom_from_target(fresh, target) == fresh


# --------------------------------------------------------------------- #
#  ATOM-METADATA load path (Step 3)                                    #
# --------------------------------------------------------------------- #


def test_extract_atom_metadata_dict_round_trips_through_emit_then_parse():
    """Emit + extract should round-trip the same data."""
    emitted = sc.emit_atom_metadata(
        regions={"L-electrode": [0, 1, 2], "bridge": [3, 4]},
        frozen_atoms=[0, 4],
        n_atoms_total=5,
        created_by="molbuilder render_fdf",
        created_at="2026-06-16T17:00:00-07:00",
    )
    # Wrap in a "host" file so it looks like real .fdf content.
    text = "SystemLabel siesta\n\n" + emitted + "\n\nBlockSize 64\n"
    payload = sc.extract_atom_metadata_dict(text)
    assert payload is not None
    assert payload["regions"] == {"L-electrode": [0, 1, 2], "bridge": [3, 4]}
    assert payload["frozen_atoms"] == [0, 4]
    assert payload["created_by"] == "molbuilder render_fdf"


def test_extract_atom_metadata_dict_returns_none_when_block_missing():
    assert sc.extract_atom_metadata_dict("SystemLabel siesta\nBlockSize 64\n") is None


def test_extract_atom_metadata_dict_returns_none_on_malformed_json():
    text = (
        "# === molbuilder atom-metadata BEGIN ===\n"
        "# format: molstruct-json/v3\n"
        "# {this is not valid json\n"
        "# === molbuilder atom-metadata END ===\n"
    )
    assert sc.extract_atom_metadata_dict(text) is None


class _StructLike:
    """Duck-typed Structure for apply_inbody_atom_metadata tests."""
    def __init__(self):
        self.regions = {}
        self.frozen_atoms = []


def test_apply_inbody_atom_metadata_populates_regions_and_frozen():
    struct = _StructLike()
    emitted = sc.emit_atom_metadata(
        regions={"R-electrode": [10, 11, 12]},
        frozen_atoms=[10, 12],
        n_atoms_total=20,
    )
    text = "engine body\n" + emitted + "\nmore engine\n"
    assert sc.apply_inbody_atom_metadata(struct, text) is True
    assert struct.regions == {"R-electrode": [10, 11, 12]}
    assert struct.frozen_atoms == [10, 12]


def test_apply_inbody_atom_metadata_returns_false_when_no_block():
    struct = _StructLike()
    assert sc.apply_inbody_atom_metadata(struct, "SystemLabel siesta\n") is False
    assert struct.regions == {}
    assert struct.frozen_atoms == []


def test_apply_inbody_atom_metadata_normalises_indices():
    """Dedup + sort per region; coerce to int."""
    struct = _StructLike()
    text = (
        "# === molbuilder atom-metadata BEGIN ===\n"
        "# format: molstruct-json/v3\n"
        '# {"schema_version": 3, "n_atoms_total": 5,\n'
        '#  "regions": {"r": [3, 1, 3, 2]},\n'
        '#  "frozen_atoms": [2, 0, 2]}\n'
        "# === molbuilder atom-metadata END ===\n"
    )
    assert sc.apply_inbody_atom_metadata(struct, text) is True
    assert struct.regions == {"r": [1, 2, 3]}
    assert struct.frozen_atoms == [0, 2]


def test_merge_user_custom_round_trip_idempotent(tmp_path):
    """Generating, saving, then regenerating an unchanged form
    should produce a byte-identical user-custom block.  Pins the
    "no drift on no-op" property."""
    target = tmp_path / "siesta-stage3.fdf"
    fresh = "engine\n" + sc.emit_user_custom_placeholder() + "\n"
    # First write -- target doesn't exist.
    first = sc.merge_user_custom_from_target(fresh, target)
    target.write_text(first, encoding="utf-8")
    # Second write of identical fresh text -- should be unchanged
    # round-trip (placeholder in == placeholder out).
    second = sc.merge_user_custom_from_target(fresh, target)
    assert second == first


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


# --------------------------------------------------------------------- #
#  PROVENANCE extract (bundle-contract.md § 5.1)                        #
# --------------------------------------------------------------------- #


def test_extract_provenance_dict_round_trips_emit_output():
    """The extractor reads back what the emitter wrote."""
    block = sc.emit_provenance(
        generator_version="molbuilder git abc123",
        generated_at="2026-06-17T12:00:00-07:00",
        form_config_hash="hash-deadbeef",
        resolved_defaults={
            "BlockSize":   "auto -> 256 (10 * 212 atoms / mpi_np)",
            "MeshCutoff":  "350.0 Ry (default)",
        },
    )
    got = sc.extract_provenance_dict(block + "\nengine body line\n")
    assert got is not None
    assert got["generator-version"] == "molbuilder git abc123"
    assert got["generated-at"]      == "2026-06-17T12:00:00-07:00"
    assert got["form-config-hash"]  == "hash-deadbeef"
    assert got["resolved-defaults.BlockSize"]  == "auto -> 256 (10 * 212 atoms / mpi_np)"
    assert got["resolved-defaults.MeshCutoff"] == "350.0 Ry (default)"


def test_extract_provenance_dict_returns_none_when_block_missing():
    assert sc.extract_provenance_dict("SystemLabel siesta\n") is None


def test_extract_provenance_dict_handles_no_defaults_section():
    block = sc.emit_provenance(
        generator_version="vX",
        generated_at="2026-06-17T00:00:00Z",
    )
    got = sc.extract_provenance_dict(block + "\n")
    assert got == {
        "generator-version": "vX",
        "generated-at":      "2026-06-17T00:00:00Z",
    }


# --------------------------------------------------------------------- #
#  ScriptSource umbrella (bundle-contract.md § 5.1)                     #
# --------------------------------------------------------------------- #


def _composed_script(*, with_atom_md: bool = True,
                     with_user_custom: bool = True,
                     with_provenance: bool = True) -> str:
    parts = []
    if with_provenance:
        parts.append(sc.emit_provenance(
            generator_version="molbuilder git test",
            generated_at="2026-06-17T00:00:00Z",
        ))
    if with_atom_md:
        am = sc.emit_atom_metadata(
            regions={"L-electrode": [1, 2], "R-electrode": [10, 11]},
            frozen_atoms=[1, 11],
            n_atoms_total=12,
        )
        assert am is not None
        parts.append(am)
    parts.append("SystemLabel test\nBlockSize 64\n")
    if with_user_custom:
        parts.append(sc.emit_user_custom_placeholder())
    return "\n".join(parts) + "\n"


def test_extract_script_source_full_round_trip():
    text = _composed_script()
    src = sc.extract_script_source(text)
    assert src.regions == {"L-electrode": [1, 2], "R-electrode": [10, 11]}
    assert src.frozen_atoms == [1, 11]
    assert src.user_custom_lines is not None
    assert any("preserve" in line for line in src.user_custom_lines)
    assert src.provenance is not None
    assert src.provenance["generator-version"] == "molbuilder git test"
    assert src.schema_version == 3
    assert src.notes == []


def test_extract_script_source_no_atom_metadata():
    """Block absent -> regions / frozen are ``None`` (NOT empty)."""
    text = _composed_script(with_atom_md=False)
    src = sc.extract_script_source(text)
    assert src.regions is None
    assert src.frozen_atoms is None
    assert src.schema_version is None
    # Other fields still extracted.
    assert src.user_custom_lines is not None
    assert src.provenance is not None


def test_extract_script_source_returns_dataclass_with_notes_list():
    """``notes`` is never None (frozen dataclass invariant)."""
    src = sc.extract_script_source("SystemLabel only\n")
    assert isinstance(src.notes, list)
    assert src.regions is None
    assert src.frozen_atoms is None
    assert src.user_custom_lines is None
    assert src.provenance is None
    assert src.schema_version is None


def test_extract_script_source_notes_on_future_schema_version():
    """``schema_version > 3`` loads + notes; doesn't fail."""
    text = (
        "# === molbuilder atom-metadata BEGIN ===\n"
        "# format: molstruct-json/v4\n"
        '# {"schema_version": 4, "n_atoms_total": 3,\n'
        '#  "regions": {"r": [0]}, "frozen_atoms": [0]}\n'
        "# === molbuilder atom-metadata END ===\n"
    )
    src = sc.extract_script_source(text)
    assert src.schema_version == 4
    assert src.regions == {"r": [0]}
    assert src.frozen_atoms == [0]
    assert any("schema_version 4" in n for n in src.notes)


def test_extract_script_source_rejects_old_schema_version():
    """``schema_version < 3`` -> regions/frozen None + diagnostic note.
    Bundle layer raises BundleError on this state; the extractor
    itself is pure and only surfaces the note."""
    text = (
        "# === molbuilder atom-metadata BEGIN ===\n"
        "# format: molstruct-json/v2\n"
        '# {"schema_version": 2, "n_atoms_total": 3}\n'
        "# === molbuilder atom-metadata END ===\n"
    )
    src = sc.extract_script_source(text)
    assert src.schema_version == 2
    assert src.regions is None
    assert src.frozen_atoms is None
    assert any("older than v3" in n for n in src.notes)


def test_extract_script_source_empty_blocks_present_but_empty():
    """Present-but-empty regions distinct from missing.

    The emitter NEVER writes an empty atom-metadata block (per
    script-contract.md § 4.4 emission rule), so we hand-craft one
    here to pin the extractor's behavior if a future schema /
    third-party writer does emit empty arrays.  Distinct from the
    no-block case above where regions is ``None``."""
    text = (
        "# === molbuilder atom-metadata BEGIN ===\n"
        "# format: molstruct-json/v3\n"
        '# {"schema_version": 3, "n_atoms_total": 0,\n'
        '#  "regions": {}, "frozen_atoms": []}\n'
        "# === molbuilder atom-metadata END ===\n"
    )
    src = sc.extract_script_source(text)
    assert src.regions == {}        # present, empty
    assert src.frozen_atoms == []   # present, empty
    assert src.schema_version == 3
