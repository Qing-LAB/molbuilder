"""Regression tests for the parse module.

Pins:
  * AmbiguousFormatError — exercised when two parsers claim the
    same path (`model/parse.md` § 3 states the rule).
  * Spectra sidecars produce JSON-serialisable
    payloads (no numpy ndarrays leaked).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.parse import (
    AmbiguousFormatError,
    SidecarResult,
    detect,
    parse,
    register,
)
from molbuilder.parse.base import FileParser
from molbuilder.parse.registry import _FILE_PARSERS


REPO = Path(__file__).resolve().parents[1].parent
MOLSTRUCT_FX = REPO / "tests" / "data" / "au_bdt_au.molstruct.json"


def _need(p: Path) -> Path:
    """Assert the fixture is there.

    Every fixture it guards is COMMITTED under tests/ -- so absence means a
    broken checkout or a deleted file, and a missing committed fixture is a
    failure, loudly.
    """
    assert p.exists(), (
        f"committed fixture missing: {p}.  It is versioned with these tests; "
        f"a checkout without it is broken, not a reason to skip.")
    return p


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
    """SpectraSidecarFileParser's payload holds no numpy ndarrays, so
    json.dumps does not throw."""
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
    defensive copies at API boundaries, and hashable.  Checked on every
    registered output type."""
    from dataclasses import FrozenInstanceError
    from molbuilder.parse.types import ParseResult

    # DISCOVERED, not listed.  The docstring says "every registered output
    # type", and a hardcoded tuple cannot keep that promise: a kind added
    # tomorrow would simply not be checked, and nothing would say so.
    kinds = ParseResult.__subclasses__()
    assert len(kinds) >= 4, (
        f"only {len(kinds)} ParseResult subclasses found -- the scan is "
        "blind, so the assertion below would pass vacuously")

    for cls in kinds:
        instance = cls(schema_version=1, parsed_at="", parser_name="x", source="x")
        with pytest.raises(FrozenInstanceError):
            instance.parser_name = "tampered"   # noqa
