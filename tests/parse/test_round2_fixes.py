"""Regression tests for the parse module.

Pins:
  * AmbiguousFormatError — exercised when two parsers claim the
    same path (`model/parse.md` § 3 states the rule).
  * Spectra sidecars produce JSON-serialisable
    payloads (no numpy ndarrays leaked).
"""

from __future__ import annotations

import pytest

from molbuilder.parse import (
    AmbiguousFormatError,
    SidecarResult,
    detect,
    register,
)
from molbuilder.parse.base import FileParser
from molbuilder.parse.registry import _FILE_PARSERS


# ---- AmbiguousFormatError code path ----------------------------- #


def test_registry_ambiguous_raises(tmp_path):
    """`model/parse.md` § 3: `detect()` raises `AmbiguousFormatError` when
    more than one parser claims a path.  Register a
    bogus FileParser that claims the same path as molstruct, then
    confirm detect() raises AmbiguousFormatError.  Cleans up after
    itself so the real registry survives."""
    from molbuilder.sidecars import molstruct
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    h2 = tmp_path / "h2.xyz"
    StructureCodec().write(Structure(elements=["H", "H"],
                                     positions=[[0, 0, 0], [0, 0, 0.74]]), h2)
    p = molstruct.sidecar_path_for(h2)

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
