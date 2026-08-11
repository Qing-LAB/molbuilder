"""Unit tests for molbuilder.script_emit emitters.

Pins the block shape so a future format change is loud.  Tests are
pure: no rendering, no I/O.
"""
from __future__ import annotations

import json
import re

from molbuilder import script_emit as sc


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
    assert sc.emit_atom_metadata({}, n_atoms_total=100) is None
    assert sc.emit_atom_metadata(None, n_atoms_total=100) is None


def test_emit_atom_metadata_with_regions_only():
    block = sc.emit_atom_metadata(
        regions={"L-electrode": [0, 1, 2], "R-electrode": [10, 11]},
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
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION
    assert payload["schema_version"] == SCHEMA_VERSION, (
        "the block must stamp the sidecar's version from the one constant -- "
        "a literal here was how it came to claim v4 while carrying v7 content")
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


def test_emit_atom_metadata_with_the_reserved_label_only_sorts_and_dedupes():
    """It is a label, so it normalises by the label rule and lands in the label
    store -- there is no second key for it to land in (schema 5)."""
    block = sc.emit_atom_metadata(
        regions={"frozen_atoms": [5, 3, 1, 2, 1]},   # unsorted + duplicate
        n_atoms_total=10,
    )
    assert block is not None
    payload = _atom_metadata_payload(block)
    assert payload["regions"]["frozen_atoms"] == [1, 2, 3, 5]
    assert "frozen_atoms" not in payload, "the fact is in the block twice"


def test_emit_atom_metadata_uses_zero_based_indices():
    """The contract pins 0-based atom indices (matches .molstruct.json
    schema v3 and Structure.regions in Python).  SIESTA's
    Geometry.Constraints in engine body remains 1-based by SIESTA
    convention -- the two coexist deliberately."""
    block = sc.emit_atom_metadata(
        regions={"single": [0], "frozen_atoms": [0]},  # 0-based
        n_atoms_total=1,
    )
    payload = _atom_metadata_payload(block)
    # The JSON body must contain index 0 (would be 1 if anyone
    # accidentally added +1 shifting).
    assert payload["regions"]["single"] == [0]
    assert payload["regions"]["frozen_atoms"] == [0]


def test_emit_atom_metadata_omits_structure_hash():
    """Per the contract: structure_hash is NOT emitted in-body because
    the metadata and coordinates are written by the same generator
    pass and cannot drift apart by construction."""
    block = sc.emit_atom_metadata(
        regions={"r": [0]}, n_atoms_total=1,
    )
    assert "structure_hash" not in block


def test_emit_atom_metadata_honors_created_by_and_created_at():
    """Audit fix 2026-06-16: callers (render_fdf / render_script)
    pass a specific ``created_by`` for traceability AND a real
    ``created_at`` timestamp.  Pin both."""
    block = sc.emit_atom_metadata(
        regions={"r": [0]},
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
        regions={"L-electrode": [0, 1, 2], "bridge": [3, 4],
                 "frozen_atoms": [0, 4]},
        n_atoms_total=5,
        created_by="molbuilder render_fdf",
        created_at="2026-06-16T17:00:00-07:00",
    )
    # Wrap in a "host" file so it looks like real .fdf content.
    text = "SystemLabel siesta\n\n" + emitted + "\n\nBlockSize 64\n"
    payload = sc.extract_atom_metadata_dict(text)
    assert payload is not None
    assert payload["regions"] == {"L-electrode": [0, 1, 2], "bridge": [3, 4],
                                  "frozen_atoms": [0, 4]}
    assert "frozen_atoms" not in payload, "the same fact written twice"
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


def _blank_struct(n=20):
    """A REAL Structure, not a duck-type: the reserved label's accessor is what
    makes `struct.frozen_atoms` mean anything, and a stand-in that reimplements
    it as a plain attribute would pass while the real thing broke (§ 13.1)."""
    from molbuilder.structure import Structure
    return Structure(elements=["C"] * n, positions=[[float(i), 0., 0.]
                                                    for i in range(n)])


def test_apply_inbody_atom_metadata_populates_the_label_store():
    struct = _blank_struct()
    emitted = sc.emit_atom_metadata(
        regions={"R-electrode": [10, 11, 12], "frozen_atoms": [10, 12]},
        n_atoms_total=20,
    )
    text = "engine body\n" + emitted + "\nmore engine\n"
    assert sc.apply_inbody_atom_metadata(struct, text) is True
    assert struct.regions == {"R-electrode": [10, 11, 12],
                              "frozen_atoms": [10, 12]}
    assert struct.frozen_atoms == [10, 12]



def test_apply_inbody_atom_metadata_returns_false_when_no_block():
    struct = _blank_struct()
    assert sc.apply_inbody_atom_metadata(struct, "SystemLabel siesta\n") is False
    assert struct.regions == {}
    assert struct.frozen_atoms == []


def test_apply_inbody_atom_metadata_normalises_indices():
    """Dedup + sort per region; coerce to int."""
    struct = _blank_struct()
    text = (
        "# === molbuilder atom-metadata BEGIN ===\n"
        "# format: molstruct-json/v4\n"
        '# {"schema_version": 7, "n_atoms_total": 5,\n'
        '#  "regions": {"r": [3, 1, 3, 2], "frozen_atoms": [2, 0, 2]}}\n'
        "# === molbuilder atom-metadata END ===\n"
    )
    assert sc.apply_inbody_atom_metadata(struct, text) is True
    assert struct.regions == {"r": [1, 2, 3], "frozen_atoms": [0, 2]}
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


def test_blocksize_field_constrains_pow2_and_leaves_the_range_to_the_deck():
    """Rewritten 2026-08-10.  This pinned ``range_ == (16, 256)`` -- a
    constant the emitted decks contradicted routinely, because the default
    beside it is derived from the rank count while the bound was not
    (``_auto_block_size(200, mpi_np=16)`` is 8, below the declared floor).

    The engine-wide list now declares the TYPE and leaves the window to the
    renderer that knows the launch (``siesta/input.py::_block_size_bounds``),
    so a forgotten range is an absent one rather than a wrong one.  The
    per-deck window is pinned in ``test_stage_resource_destinations.py``."""
    bs = next(f for f in sc.SIESTA_BENCH_FIELDS if f.name == "BlockSize")
    assert bs.type_ == "pow2"
    assert bs.range_ is None


# --------------------------------------------------------------------- #
#  PROVENANCE extract (execution/job-contracts.md § 3.2)                        #
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


def test_extract_provenance_dict_returns_empty_dict_for_present_but_empty():
    """Audit IMPORTANT 10: present-but-empty PROVENANCE returns ``{}``,
    NOT ``None``.  Pre-fix the function used ``out or None`` which
    collapsed "block present, no parseable k/v" into "block absent",
    contradicting execution/job-contracts.md § 3.2's None-vs-empty semantics."""
    text = (
        "# === molbuilder provenance BEGIN ===\n"
        "# === molbuilder provenance END ===\n"
        "SystemLabel anything\n"
    )
    got = sc.extract_provenance_dict(text)
    assert got == {}, f"expected empty dict, got {got!r}"
    assert got is not None


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
#  ScriptSource umbrella (execution/job-contracts.md § 3.2)                     #
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
            regions={"L-electrode": [1, 2], "R-electrode": [10, 11],
                     "frozen_atoms": [1, 11]},
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
    assert src["regions"] == {"L-electrode": [1, 2], "R-electrode": [10, 11],
                              "frozen_atoms": [1, 11]}
    assert src["frozen_atoms"] == [1, 11]   # the designated read, off the store
    assert src["user_custom_lines"] is not None
    assert any("preserve" in line for line in src["user_custom_lines"])
    assert src["provenance"] is not None
    assert src["provenance"]["generator-version"] == "molbuilder git test"
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION as _SV
    assert src["schema_version"] == _SV
    assert src["notes"] == []


def test_extract_script_source_no_atom_metadata():
    """Block absent -> regions / frozen are ``None`` (NOT empty)."""
    text = _composed_script(with_atom_md=False)
    src = sc.extract_script_source(text)
    assert src["regions"] is None
    assert src["frozen_atoms"] is None
    assert src["schema_version"] is None
    # Other fields still extracted.
    assert src["user_custom_lines"] is not None
    assert src["provenance"] is not None


def test_extract_script_source_returns_dataclass_with_notes_list():
    """``notes`` is never None (frozen dataclass invariant)."""
    src = sc.extract_script_source("SystemLabel only\n")
    assert isinstance(src["notes"], list)
    assert src["regions"] is None
    assert src["frozen_atoms"] is None
    assert src["user_custom_lines"] is None
    assert src["provenance"] is None
    assert src["schema_version"] is None


def test_a_block_at_any_other_schema_version_is_not_read():
    """One version, strictly (structure-molstruct.md § 2).

    RETIRED the pair this replaces -- `..._notes_on_future_schema_version` and
    `..._rejects_old_schema_version` -- which described a reader that LOADED an
    older or newer block and attached a note. That tolerance is gone, and it is
    what let a real junction's fifty frozen atoms come back as an empty list: a
    payload that looks complete and quietly is not.

    Older and newer are one case now, so they are one test. The block is not
    read; `regions` and `frozen_atoms` come back None with a note saying why.
    """
    for version in (2, 3, 4, 6, 99):
        text = (
            "# === molbuilder atom-metadata BEGIN ===\n"
            f'# {{"schema_version": {version}, "n_atoms_total": 3,\n'
            '#  "regions": {"r": [0]}}\n'
            "# === molbuilder atom-metadata END ===\n"
        )
        src = sc.extract_script_source(text)
        assert src["schema_version"] == version, "the version is still reported"
        assert src["regions"] is None, (
            f"v{version} was READ -- an older or newer block keeps the same "
            f"facts in different places, so reading it drops what it cannot map")
        assert src["frozen_atoms"] is None
        assert any("schema_version" in n for n in src["notes"]), (
            f"v{version} was refused without saying so")

def test_extract_script_source_empty_blocks_present_but_empty():
    """Present-but-empty regions distinct from missing.

    The emitter NEVER writes an empty atom-metadata block (per
    execution/job-contracts.md § 3.1 emission rule), so we hand-craft one
    here to pin the extractor's behavior if a future schema /
    third-party writer does emit empty arrays.  Distinct from the
    no-block case above where regions is ``None``."""
    text = (
        "# === molbuilder atom-metadata BEGIN ===\n"
        "# format: molstruct-json/v7\n"
        '# {"schema_version": 7, "n_atoms_total": 0,\n'
        '#  "regions": {}}\n'
        "# === molbuilder atom-metadata END ===\n"
    )
    src = sc.extract_script_source(text)
    assert src["regions"] == {}        # present, empty
    assert src["frozen_atoms"] == []   # nothing carries the label
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION as _SV
    assert src["schema_version"] == _SV


# --------------------------------------------------------------------- #
#  The version line on the in-body block is READ (2026-08-03)           #
# --------------------------------------------------------------------- #
#
# The block states the version that wrote it, and this reader took the contents
# at face value regardless.  So a block written in the older layout -- frozen
# atoms as a key BESIDE `regions` rather than a label inside it -- was applied,
# its frozen set dropped, and a run came back with nothing frozen and nothing
# said.  That is how real 50- and 216-atom frozen sets went missing.
#
# WARN AND TRANSLATE, not refuse (user decision).  Refusing would make a
# finished run unopenable, and the whole point of these notes is that a run
# directory explains itself.
#
# Built from a constructed junction, never a file found on disk: a fixture
# cannot go stale, and its relevance is not a guess.

def _block(doc: dict) -> str:
    import json
    body = json.dumps(doc, indent=2)
    return ("# === molbuilder atom-metadata BEGIN ===\n"
            f"# format: molstruct-json/v{doc['schema_version']}\n"
            + "\n".join("# " + l for l in body.splitlines())
            + "\n# === molbuilder atom-metadata END ===")


def _junction_parts():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from support.junction import build_junction
    from molbuilder.structure import FROZEN_LABEL
    s = build_junction()
    return s, list(s.regions[FROZEN_LABEL]), FROZEN_LABEL


def test_the_current_version_applies_without_a_word():
    from molbuilder import script_emit as sc
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION
    from molbuilder.structure import Structure
    s, frozen, FROZEN = _junction_parts()
    blk = _block({"schema_version": SCHEMA_VERSION, "n_atoms_total": s.n_atoms,
                  "regions": {k: list(v) for k, v in s.regions.items()},
                  "annotations": {}})
    back = Structure(elements=list(s.elements), positions=s.positions.copy())
    said = []
    assert sc.apply_inbody_atom_metadata(back, blk, notices=said) is True
    assert len(back.regions[FROZEN]) == len(frozen)
    assert said == [], f"a current block must not be complained about: {said}"


def test_an_older_layout_recovers_the_frozen_atoms_and_says_so():
    """The defect, inverted into a pin: the frozen set survives AND the user
    is told where it came from."""
    from molbuilder import script_emit as sc
    from molbuilder.structure import Structure
    s, frozen, FROZEN = _junction_parts()
    blk = _block({
        "schema_version": 4, "n_atoms_total": s.n_atoms,
        "regions": {k: list(v) for k, v in s.regions.items() if k != FROZEN},
        "frozen_atoms": frozen,          # the retired two-store shape
        "annotations": {},
    })
    back = Structure(elements=list(s.elements), positions=s.positions.copy())
    said = []
    assert sc.apply_inbody_atom_metadata(back, blk, notices=said) is True
    assert len(back.regions.get(FROZEN, [])) == len(frozen), (
        "the frozen set was dropped -- this is the loss the check exists for")
    assert said and said[0]["level"] == "warn"
    assert said[0]["where"] == "labels.atom_metadata_version"


def test_a_label_already_in_the_current_place_is_not_overwritten():
    """If both shapes are present the CURRENT one wins -- it is the newer
    truth, and a translation must never clobber it."""
    from molbuilder import script_emit as sc
    from molbuilder.structure import Structure
    s, frozen, FROZEN = _junction_parts()
    regions = {k: list(v) for k, v in s.regions.items()}
    regions[FROZEN] = [0]                                  # current, and short
    blk = _block({"schema_version": 4, "n_atoms_total": s.n_atoms,
                  "regions": regions, "frozen_atoms": frozen, "annotations": {}})
    back = Structure(elements=list(s.elements), positions=s.positions.copy())
    said = []
    sc.apply_inbody_atom_metadata(back, blk, notices=said)
    assert back.regions[FROZEN] == [0], "the old key overwrote the current one"
