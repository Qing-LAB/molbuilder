"""SIESTA/TranSIESTA transport results, read back — ``<label>.transport.json``.

Contract: `model/parse.md` § 5.5 (what a person can open is the REGISTRY's
question); the record's own shape is `transport/record.py`'s, which writes it.
`/api/results/dir` asks the registry what reads each file.

**IT CHECKS THE SCHEMA.**  `can_parse` reads the discriminator rather than
trusting the suffix: if the writer's schema moves, this refuses loudly
instead of claiming a file it will then misread.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..base import FileParser
from ..errors import ParseError
from ..types import SidecarResult
from ...runfiles import ROLE_TRANSPORT_RECORD, role_of
from ._helpers import build_sidecar_result, read_record


class TransportRecordError(ParseError):
    """A ``.transport.json`` that is not one, or is not this version."""


#: The record's schema (`execution/job-contracts.md` § 6.1): an older record is
#: refused by its version, and `summarize task` writes it again.
TRANSPORT_RESULT_SCHEMA = "molbuilder/transport-result@16"


def _load(path: Path) -> Dict[str, Any]:
    """The record as a dict, or raise — the one reader of this file."""
    return read_record(path, schema=TRANSPORT_RESULT_SCHEMA,
                       refused=TransportRecordError,
                       what="a transport record")


class TransportRecordFileParser(FileParser):
    """Parse a ``<label>.transport.json`` — the transport ladder's result:
    the I-V table, the per-bias points, the stage facts and the provenance
    slot.  Returns a :class:`SidecarResult` with ``schema =
    "transport/v<N>"``, ``N`` the major of the record's own ``schema``.
    """

    name   = "transport-json"
    label  = f"molbuilder {ROLE_TRANSPORT_RECORD} record"
    hint   = (f"a transport ladder's summarised result -- "
              f"<label>{ROLE_TRANSPORT_RECORD}, written by "
              f"`jobset summarize task`")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # THE SCHEMA, not the suffix alone.
        if role_of(Path(path).name) != ROLE_TRANSPORT_RECORD:
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
            kind="transport",
            parser_name=cls.name,
            source=Path(path),
        )


__all__ = ["TransportRecordError", "TransportRecordFileParser"]
