"""A SIESTA vibration's displacement sweep, read back — ``<label>.fc-sweep.json``.

Contract: `model/parse.md` § 5.5 (what a person can open is the REGISTRY's
question); the record's own shape is `spectra/displacement_sweep.py`'s, which
writes it (`engines/vibration.md` § 5.9).

`/api/results/dir` asks the registry what reads each file and offers only what
something reads, so without this reader the sweep's record would sit at the
calculation root unoffered.  It is the transport record's twin
(`sidecars/transport.py`): it claims a file by its role in the catalogue and
refuses on parse (`model/parse.md` § 6) -- a record of another version is
listed as this parser's and, opened, refused in its reader's words, its way
on `jobset summarize task` writing it again, never misread.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..base import FileParser
from ..errors import FileUnreadable
from ..types import SidecarResult
from ...runfiles import ROLE_FC_SWEEP, role_of
from ._helpers import build_sidecar_result, read_record
from .transport import WRITTEN_AGAIN


class FcSweepRecordError(FileUnreadable):
    """A ``.fc-sweep.json`` that is not one, or is not this version."""


#: The record's schema (`execution/job-contracts.md` § 6.1).  ``@2``: each
#: stage's force-constant range is named as SIESTA's numbering,
#: ``fc_range_engine_1based`` (`model/overview.md` § 2.2a); a ``@1`` record is
#: refused by its version, and `summarize task` writes it again.
SWEEP_SCHEMA = "molbuilder/fc-displacement-sweep@2"


def _load(path: Path) -> Dict[str, Any]:
    """The record as a dict, or raise — the one reader of this file."""
    return read_record(path, schema=SWEEP_SCHEMA, refused=FcSweepRecordError,
                       what="a displacement sweep's record",
                       way=WRITTEN_AGAIN)


class FcSweepRecordFileParser(FileParser):
    """Parse a ``<label>.fc-sweep.json`` — the force-constant stages of a
    SIESTA vibration compared: each stage and what it varied, every mode's
    frequency per stage, the force-constant changes, the paths of each
    stage's files.  Returns a :class:`SidecarResult` with
    ``schema = "fc-sweep/v<N>"``, ``N`` the major of the record's own
    ``schema``."""

    name   = "fc-sweep-json"
    label  = f"molbuilder {ROLE_FC_SWEEP} record"
    hint   = (f"a SIESTA vibration's displacement sweep -- "
              f"<label>{ROLE_FC_SWEEP}, written by `jobset summarize task`")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # BY ITS ROLE: the version is the parse's to refuse.
        return role_of(Path(path).name) == ROLE_FC_SWEEP and Path(path).is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        payload = _load(Path(path))
        return build_sidecar_result(
            payload=payload,
            kind="fc-sweep",
            parser_name=cls.name,
            source=Path(path),
        )


__all__ = ["FcSweepRecordError", "FcSweepRecordFileParser"]
