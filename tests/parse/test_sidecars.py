"""L2 tests for Phase D sidecar FileParsers.

Pins:
  * Each sidecar parser is registered + claims its filename
    suffix.
  * parse() returns a SidecarResult with the right schema tag.
  * detect() routes to the right sidecar parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import SidecarResult, detect, parse
from molbuilder.parse.sidecars import (
    MolstructSidecarFileParser,
    SpectraSidecarFileParser,
    TransportSidecarFileParser,
)
from molbuilder.parse.registry import _registered_file_parsers


REPO = Path(__file__).resolve().parents[2]
MOLSTRUCT_FX = REPO / "tests" / "data" / "au_bdt_au.molstruct.json"
SPECTRA_FX   = REPO / "projects" / "BDT" / "spectrum" / "BDT-only" / "spectra.spectra.json"


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"fixture absent: {p}")
    return p


# Registration --------------------------------------------------------- #


def test_sidecar_parsers_registered():
    names = {p.name for p in _registered_file_parsers()}
    assert "molstruct-json" in names
    assert "spectra-json"   in names
    assert "transport-json" in names


def test_molstruct_parser_claims_suffix():
    assert MolstructSidecarFileParser.can_parse(_need(MOLSTRUCT_FX))
    assert not MolstructSidecarFileParser.can_parse(REPO / "README.md")


def test_spectra_parser_claims_suffix():
    assert SpectraSidecarFileParser.can_parse(_need(SPECTRA_FX))


def test_detect_routes_to_molstruct_parser():
    cls = detect(_need(MOLSTRUCT_FX))
    assert cls is MolstructSidecarFileParser


# Parse + payload + schema -------------------------------------------- #


def test_parse_molstruct_returns_sidecarresult():
    result = parse(_need(MOLSTRUCT_FX))
    assert isinstance(result, SidecarResult)
    assert result.result_kind == "sidecar"
    assert result.schema.startswith("molstruct/v")
    assert result.parser_name == "molstruct-json"
    assert isinstance(result.payload, dict)
    # The schema tag matches what the payload declares
    sv_in_payload = result.payload.get("schema_version")
    if sv_in_payload is not None:
        assert result.schema == f"molstruct/v{sv_in_payload}"


def test_parse_spectra_returns_sidecarresult_with_payload():
    result = parse(_need(SPECTRA_FX))
    assert isinstance(result, SidecarResult)
    assert result.schema.startswith("spectra/v")
    assert "schema_version" in result.payload


def test_sidecar_result_is_frozen():
    result = parse(_need(MOLSTRUCT_FX))
    with pytest.raises(Exception):
        result.schema = "tampered/v0"   # noqa
