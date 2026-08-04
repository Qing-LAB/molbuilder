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


# Registration --------------------------------------------------------- #


def test_sidecar_parsers_registered():
    names = {p.name for p in _registered_file_parsers()}
    assert "molstruct-json" in names
    assert "spectra-json"   in names
    assert "transport-json" in names


def test_molstruct_parser_claims_suffix():
    assert MolstructSidecarFileParser.can_parse(_need(MOLSTRUCT_FX))
    assert not MolstructSidecarFileParser.can_parse(REPO / "README.md")


def test_spectra_parser_claims_suffix(tmp_path):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from support.junction import spectra_sidecar
    assert SpectraSidecarFileParser.can_parse(
        spectra_sidecar(tmp_path / "built.spectra.json"))


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


def test_parse_spectra_returns_sidecarresult_with_payload(tmp_path):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from support.junction import spectra_sidecar
    result = parse(spectra_sidecar(tmp_path / "built.spectra.json"))
    assert isinstance(result, SidecarResult)
    assert result.schema.startswith("spectra/v")
    assert "schema_version" in result.payload


def test_sidecar_result_is_frozen():
    result = parse(_need(MOLSTRUCT_FX))
    with pytest.raises(Exception):
        result.schema = "tampered/v0"   # noqa
