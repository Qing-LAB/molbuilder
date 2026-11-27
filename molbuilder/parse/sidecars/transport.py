"""SIESTA/TranSIESTA transport results, read back — ``<label>.transport.json``.

Contract: `model/parse.md` § 5.5 (what a person can open is the REGISTRY's
question); the record's own shape is `transport/record.py`'s, which writes it.
`/api/results/dir` asks the registry what reads each file.

**IT CLAIMS BY NAME AND REFUSES ON PARSE** (`model/parse.md` § 6): a
record of another version is listed as this parser's, and opened it is
refused in its reader's words -- what it says, what this molbuilder reads,
and the way on, `jobset summarize task` writing it again
(`execution/job-contracts.md` § 6.1) -- never read as this version's, and
never listed as no parser's.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..base import FileParser
from ..errors import FileUnreadable
from ..types import SidecarResult
from ...identity import as_typed
from ...runfiles import ROLE_TRANSPORT_RECORD, role_of
from ._helpers import build_sidecar_result, read_record


class TransportRecordError(FileUnreadable):
    """A ``.transport.json`` that is not one, or is not this version."""


#: The record's schema (`execution/job-contracts.md` § 6.1): an older record is
#: refused by its version, and `summarize task` writes it again
#: (:data:`WRITTEN_AGAIN`).
TRANSPORT_RESULT_SCHEMA = "molbuilder/transport-result@19"
#: THE WAY ON FOR A RESULT OF AN OLDER VERSION, its family's -- the command
#: that writes it again, as a person types it (`identity.as_typed`): said by
#: this record's reader, the displacement sweep's and the transport data
#: file's (`execution/job-contracts.md` § 6.1, the family table's results).
WRITTEN_AGAIN = f"{as_typed('jobset', 'summarize', 'task')} writes it again"


def _load(path: Path) -> Dict[str, Any]:
    """The record as a dict, or raise — the one reader of this file."""
    return read_record(path, schema=TRANSPORT_RESULT_SCHEMA,
                       refused=TransportRecordError,
                       what="a transport record", way=WRITTEN_AGAIN)


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
        # BY ITS ROLE: the version is the parse's to refuse.
        return (role_of(Path(path).name) == ROLE_TRANSPORT_RECORD
                and Path(path).is_file())

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
