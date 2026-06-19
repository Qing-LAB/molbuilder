"""``.transport.json`` sidecar FileParser.

Phase D of parse-module.md migration: thin wrapper over the
legacy :mod:`molbuilder.parsers.transport_json` module.

Same unwrap-to-dict treatment as the spectra sidecar (legacy
loader returns a typed
:class:`molbuilder.transport.results.TransportResults`
dataclass).
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
from molbuilder.parsers.transport_json import parse_transport_json as _legacy_parse

from ._helpers import build_sidecar_result


class TransportSidecarFileParser(FileParser):
    """Parse a ``<stem>.transport.json`` sidecar — molbuilder
    transport-results envelope (T(E) curve / I-V / scattering
    region geometry).  Returns a :class:`SidecarResult` with
    ``schema = "transport/v<N>"``."""

    name   = "transport-json"
    label  = "molbuilder .transport.json sidecar"
    hint   = "files ending in .transport.json"
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".transport.json") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        results = _legacy_parse(path)
        # Use .to_dict() (not dataclasses.asdict) so numpy ndarrays
        # convert to JSON-trivially-serialisable lists.  asdict()
        # preserves ndarray objects, which then break json.dumps()
        # at webhook delivery + cache pickle (post-2026-06-19
        # round-2 fix).
        payload = results.to_dict()
        sv = payload.get("schema_version", "1")
        return build_sidecar_result(
            payload=payload,
            schema=f"transport/v{sv}",
            parser_name=cls.name,
            source=path,
        )
