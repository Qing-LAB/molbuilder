"""SIESTA/TranSIESTA transport results, read back — ``<label>.transport.json``.

Contract: `model/parse.md` § 5.5 (what a person can open is the REGISTRY's
question); the record's own shape is `transport/record.py`'s, which writes it.
`/api/results/dir` asks the registry what reads each file.

**IT CHECKS THE SCHEMA.**  `can_parse` reads the discriminator rather than
trusting the suffix: if the writer's schema moves, this refuses loudly
instead of claiming a file it will then misread.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..base import FileParser
from ..errors import ParseError
from ..types import SidecarResult
from ._helpers import build_sidecar_result


class TransportRecordError(ParseError):
    """A ``.transport.json`` that is not one, or is not this version."""


def _load(path: Path) -> Dict[str, Any]:
    """The record as a dict, or raise — the one reader of this file."""
    from molbuilder.persist import check_schema
    from molbuilder.transport.record import TRANSPORT_RESULT_SCHEMA
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TransportRecordError(f"{Path(path).name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TransportRecordError(
            f"{Path(path).name}: a transport record is a JSON object, "
            f"got {type(payload).__name__}")
    try:
        check_schema(str(payload.get("schema", "")), TRANSPORT_RESULT_SCHEMA,
                     label=Path(path).name)
    except ValueError as exc:
        raise TransportRecordError(str(exc)) from exc
    return payload


class TransportRecordFileParser(FileParser):
    """Parse a ``<label>.transport.json`` — the transport ladder's result:
    the I-V table, the per-bias points, the stage facts and the provenance
    slot.  Returns a :class:`SidecarResult` with ``schema = "transport/v1"``.
    """

    name   = "transport-json"
    label  = "molbuilder .transport.json record"
    hint   = ("a transport ladder's summarised result -- "
              "<label>.transport.json, written by `jobset summarize run`")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # THE SCHEMA, not the suffix alone.
        from molbuilder.runfiles import role_of
        if role_of(Path(path).name) != ".transport.json":
            return False
        try:
            _load(Path(path))
        except (TransportRecordError, OSError):
            return False
        return True

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        payload = _load(Path(path))
        return build_sidecar_result(
            payload=payload,
            schema="transport/v1",
            parser_name=cls.name,
            source=Path(path),
        )


__all__ = ["TransportRecordError", "TransportRecordFileParser"]
