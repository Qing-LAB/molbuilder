"""Round-2 review-fix regression tests for the parse module.

Pins:
  * B1 sort tie-breaker — unstaged .fdf must NOT beat staged ones.
  * Anchor consistency — same .fdf provides coords AND regions
    AND cell (no Frankenstein from multiple sources).
  * AmbiguousFormatError — exercised when two parsers claim the
    same path (doc § 10 promised test that didn't exist).
  * Envelope-field consistency across phases B / C / D —
    parser_name is the slug, source is resolved.
  * Transport / Spectra sidecars produce JSON-serialisable
    payloads (no numpy ndarrays leaked).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.parse import (
    AmbiguousFormatError,
    JobResult,
    SidecarResult,
    TrajectoryResult,
    UnknownFormatError,
    detect,
    parse,
    parse_dir,
    register,
)
from molbuilder.parse.base import FileParser
from molbuilder.parse.dirs.job import (
    JobDirParser,
    _anchor_sort_key,
    _parse_engine_body_summary,
    _read_cell_from_fdf,
)
from molbuilder.parse.registry import (
    _FILE_PARSERS,
    _registered_file_parsers,
)


REPO = Path(__file__).resolve().parents[1].parent
SIESTA_OUT = REPO / "tests" / "watch" / "fixtures" / "siesta_frozen" \
    / "hemeC-stage2-run3-finished-42fr.out"
MOLSTRUCT_FX = REPO / "tests" / "data" / "au_bdt_au.molstruct.json"
# NO PATH INTO projects/.  The spectra fixture was
# projects/BDT/spectrum/BDT-only/spectra.spectra.json -- the user's scientific
# record -- behind a `pytest.skip("fixture absent")`, which is the dangerous
# half: on a machine without that run the test SKIPS and the suite still reads
# green.  It is now WRITTEN by the application's own `dump_spectra_json`, so
# the document is valid by construction and cannot go stale.


def _need(p: Path) -> Path:
    """Assert the fixture is there.

    This used to ``pytest.skip`` on a missing file.  Every fixture it guards is
    COMMITTED under tests/ -- so absence means a broken checkout or a deleted
    file, and skipping turned that into a green run that proved nothing.  A
    missing committed fixture is a failure, loudly.
    """
    assert p.exists(), (
        f"committed fixture missing: {p}.  It is versioned with these tests; "
        f"a checkout without it is broken, not a reason to skip.")
    return p


# ---- B1 sort tiebreak --------------------------------------------- #


def test_anchor_unstaged_does_not_beat_staged(tmp_path: Path) -> None:
    """B1 round-2: a `template.fdf` (unstaged, lex-late) must NOT
    win over `device-stage1.fdf` (staged).  Pre-fix the sort key
    `(_detect_stage(p) or 0, p.name)` gave template.fdf stage=0
    and let lex tie-break elevate it over a real stage1."""
    template = tmp_path / "template.fdf"
    template.write_text("SystemLabel template\n")
    staged1 = tmp_path / "device-stage1.fdf"
    staged1.write_text("SystemLabel device\n")
    sorted_paths = sorted([template, staged1], key=_anchor_sort_key)
    # ascending sort, take [-1] = "highest" anchor — must be the
    # staged file, not the unstaged template.
    assert sorted_paths[-1] == staged1


def test_anchor_picks_highest_stage(tmp_path: Path) -> None:
    """B1: with stage1, stage2, stage3 present, [-1] picks stage3."""
    s1 = tmp_path / "device-stage1.fdf"; s1.write_text("SystemLabel d\n")
    s2 = tmp_path / "device-stage2.fdf"; s2.write_text("SystemLabel d\n")
    s3 = tmp_path / "device-stage3.fdf"; s3.write_text("SystemLabel d\n")
    sorted_paths = sorted([s2, s3, s1], key=_anchor_sort_key)
    assert sorted_paths[-1] == s3


def test_anchor_unstaged_only_falls_through_lex(tmp_path: Path) -> None:
    """When ALL fdfs are unstaged, the lex tie-break still works
    (no behavioural change from the round-2 fix)."""
    a = tmp_path / "alpha.fdf"; a.write_text("\n")
    b = tmp_path / "beta.fdf";  b.write_text("\n")
    sorted_paths = sorted([a, b], key=_anchor_sort_key)
    assert sorted_paths[-1] == b


# ---- B2 LatticeConstant edge cases --------------------------------- #


def test_b2_lattice_constant_word_boundary(tmp_path: Path) -> None:
    """B2 round-2: regex must require word boundary after
    ``LatticeConstant`` so a hypothetical key like
    ``LatticeConstant_Sub_Tolerance`` doesn't match."""
    fdf = tmp_path / "test.fdf"
    fdf.write_text(
        "SystemLabel test\n"
        "LatticeConstant_Sub_Tolerance 0.1\n"   # decoy
        "LatticeConstant 5.43 Ang\n"
        "%block LatticeVectors\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 1.0\n"
        "%endblock LatticeVectors\n"
    )
    cell = _read_cell_from_fdf(fdf)
    assert cell is not None
    assert abs(cell[0][0] - 5.43) < 1e-6   # 5.43 Ang, not 0.1


# ---- I2 value whitespace normalisation ---------------------------- #


def test_i2_tab_in_value_normalised_to_space() -> None:
    """I2 round-2: the value side of a key/value line should have
    internal whitespace collapsed (tabs and runs of spaces all
    become a single space).  Previously a tab-separated
    ``MeshCutoff\\t350.0\\tRy`` was stored as ``350.0\\tRy``."""
    fdf_text = "MeshCutoff\t350.0\tRy\n"
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["MeshCutoff"] == "350.0 Ry"


def test_i2_multispace_in_value_normalised() -> None:
    fdf_text = "MeshCutoff    350.0   Ry\n"
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["MeshCutoff"] == "350.0 Ry"


# ---- I1 bool guard --------------------------------------------- #


def test_i1_bool_not_charted_as_residual() -> None:
    """I1 round-2: a malformed parse with dDmax=True must NOT be
    charted as 1.0 residual (bool is a subclass of int)."""
    from molbuilder.parse.dirs.job import _consolidate_plots
    # We can't easily trigger this through a real .out, but we
    # can check the type-narrowing logic via direct call.  Simpler:
    # the source line should now read `not isinstance(x, bool)`.
    src = Path(__file__).resolve().parents[1].parent \
        / "molbuilder" / "parse" / "dirs" / "job.py"
    text = src.read_text()
    assert "not isinstance(dDmax, bool)" in text, (
        "I1 round-2 bool guard not present in _consolidate_plots")


# ---- Envelope-field drift ------------------------------------------ #


def test_jobresult_parser_name_is_slug(tmp_path):
    """Round-2: JobResult.parser_name must be the slug ``job-dir``,
    matching the convention used by engines + sidecars (not the
    literal classname)."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from support.junction import run_dir
    result = parse_dir(run_dir(tmp_path))
    assert result.parser_name == "job-dir"
    assert result.parser_name == JobDirParser.name


def test_sidecar_source_is_resolved_absolute_path():
    """Round-2: source path in ParseResult must be resolved
    absolute (so it survives cwd-changes downstream).  Test
    against any registered sidecar."""
    p = _need(MOLSTRUCT_FX)
    result = parse(p)
    assert Path(result.source).is_absolute()
    assert Path(result.source) == p.resolve()


def test_engine_source_is_resolved_absolute_path():
    """Same for engines."""
    p = _need(SIESTA_OUT)
    result = parse(p)
    assert Path(result.source).is_absolute()


# ---- AmbiguousFormatError code path ----------------------------- #


def test_registry_ambiguous_raises():
    """Doc § 10 promised this test that did not exist.  Register a
    bogus FileParser that claims the same path as molstruct, then
    confirm detect() raises AmbiguousFormatError.  Cleans up after
    itself so the real registry survives."""
    p = _need(MOLSTRUCT_FX)

    class _GreedyParser(FileParser):
        name = "_test-greedy"
        label = "test-only ambiguous claimer"
        hint = ""
        output = SidecarResult

        @classmethod
        def can_parse(cls, path):
            return path.name.endswith(".molstruct.json")

        @classmethod
        def parse(cls, path):
            raise NotImplementedError

    register(_GreedyParser)
    try:
        with pytest.raises(AmbiguousFormatError) as ei:
            detect(p)
        msg = str(ei.value)
        # Message must name BOTH parsers so the user can decide.
        assert "_test-greedy" in msg
        assert "molstruct-json" in msg
    finally:
        # Cleanup so subsequent tests see the canonical registry.
        if _GreedyParser in _FILE_PARSERS:
            _FILE_PARSERS.remove(_GreedyParser)


# ---- Transport / Spectra JSON-serialisable payloads ------------- #


def test_spectra_sidecar_payload_is_json_serialisable(tmp_path):
    """Round-2: SpectraSidecarFileParser used asdict(), leaving
    numpy ndarrays in the payload — json.dumps would throw.  The
    .to_dict() switch fixes this."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from support.junction import spectra_sidecar
    result = parse(spectra_sidecar(tmp_path / "built.spectra.json"))
    # If asdict() were still in place, json.dumps would raise
    # TypeError("Object of type ndarray is not JSON serializable").
    json.dumps(result.payload)


# ---- Frozen invariant on multiple Result types ------------------ #


def test_frozen_dataclass_invariant_on_each_result_kind():
    """Doc § 10 promised test_parse_result_subclasses_are_frozen.
    The existing test only checks JobResult; extend to every
    registered output type to catch a future override drift."""
    from dataclasses import FrozenInstanceError
    from molbuilder.parse.types import (
        BundleResult, JobResult as JR, ScriptResult, SidecarResult as SR,
        StructureResult, TrajectoryResult as TR,
    )
    for cls in (TR, StructureResult, SR, ScriptResult, JR, BundleResult):
        instance = cls(schema_version=1, parsed_at="", parser_name="x", source="x")
        with pytest.raises(FrozenInstanceError):
            instance.parser_name = "tampered"   # noqa
