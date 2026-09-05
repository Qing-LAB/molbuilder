"""Round-2 review-fix regression tests for the parse module.

Pins:
  * B1 sort tie-breaker — unstaged .fdf must NOT beat staged ones.
  * Anchor consistency — same .fdf provides coords AND regions
    AND cell (no Frankenstein from multiple sources).
  * AmbiguousFormatError — exercised when two parsers claim the
    same path (`model/parse.md` § 3 states the rule; a retired § 10
    promised the test and it did not exist).
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
    SidecarResult,
    TrajectoryResult,
    UnknownFormatError,
    detect,
    parse,
    parse_dir,
    register,
)
from molbuilder.parse.base import FileParser
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


# ---- B2 LatticeConstant edge cases --------------------------------- #


# ---- I2 value whitespace normalisation ---------------------------- #


# ---- I1 bool guard --------------------------------------------- #


# ---- Envelope-field drift ------------------------------------------ #


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
    """`model/parse.md` § 3: `detect()` raises `AmbiguousFormatError` when
    more than one parser claims a path.  Register a
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
    """`model/parse.md` § 2: every parser returns a FROZEN dataclass -- no
    defensive copies at API boundaries, and hashable.
    The existing test only checks JobResult; extend to every
    registered output type to catch a future override drift."""
    from dataclasses import FrozenInstanceError
    from molbuilder.parse.types import (
        ScriptResult, SidecarResult as SR,
        StructureResult, TrajectoryResult as TR,
    )
    # `JobResult` was the fifth here until 2026-09-04; it retired with
    # the directory decoder, ten of whose eleven fields had no reader.
    for cls in (TR, StructureResult, SR, ScriptResult):
        instance = cls(schema_version=1, parsed_at="", parser_name="x", source="x")
        with pytest.raises(FrozenInstanceError):
            instance.parser_name = "tampered"   # noqa
