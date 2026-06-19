"""``.spectra.json`` sidecar FileParser.

Phase D of parse-module.md migration: thin wrapper over the
legacy :mod:`molbuilder.parsers.spectra_json` module.

Unlike the molstruct loader (which returns a raw dict), the
legacy ``parse_spectra_json`` returns a typed
:class:`molbuilder.spectra.results.SpectraResults` dataclass.
The wrapper unwraps it back to a dict for the SidecarResult
payload via :func:`dataclasses.asdict`, keeping the
SidecarResult shape uniform across kinds.  Callers that need the
typed object can ``from molbuilder.spectra.results import
SpectraResults; SpectraResults(**result.payload)`` (or use the
legacy import path directly — Phase H sorts the consumer side).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
from molbuilder.parsers.spectra_json import parse_spectra_json as _legacy_parse

from ._helpers import build_sidecar_result


class SpectraSidecarFileParser(FileParser):
    """Parse a ``<stem>.spectra.json`` sidecar — molbuilder
    spectrum-results envelope (peaks / dipole strengths / etc.).
    Returns a :class:`SidecarResult` with
    ``schema = "spectra/v<N>"``."""

    name   = "spectra-json"
    label  = "molbuilder .spectra.json sidecar"
    hint   = "files ending in .spectra.json"
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".spectra.json") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        results = _legacy_parse(path)
        payload = asdict(results)
        sv = payload.get("schema_version", 4)
        return build_sidecar_result(
            payload=payload,
            schema=f"spectra/v{sv}",
            parser_name=cls.name,
            source=path,
        )
