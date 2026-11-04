"""A SIESTA vibration's displacement sweep, read back — ``<label>.fc-sweep.json``.

Contract: `model/parse.md` § 5.5 (what a person can open is the REGISTRY's
question); the record's own shape is `spectra/displacement_sweep.py`'s, which
writes it (`engines/vibration.md` § 5.9).

`/api/results/dir` asks the registry what reads each file and offers only what
something reads, so without this reader the sweep's record would sit at the
calculation root unoffered.  It is the transport record's twin (`sidecars/transport.py`): it claims a file by
its role in the catalogue AND by the schema its writer stamps, never by the
suffix alone, so a record of another version is refused rather than misread.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..base import FileParser
from ..errors import ParseError
from ..types import SidecarResult
from ...runfiles import ROLE_FC_SWEEP, role_of
from ._helpers import build_sidecar_result


class FcSweepRecordError(ParseError):
    """A ``.fc-sweep.json`` that is not one, or is not this version."""


def _load(path: Path) -> Dict[str, Any]:
    """The record as a dict, or raise — the one reader of this file."""
    from molbuilder.spectra.displacement_sweep import SWEEP_SCHEMA
    from ._helpers import read_record
    return read_record(path, schema=SWEEP_SCHEMA, refused=FcSweepRecordError,
                       what="a displacement sweep's record")


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
        if role_of(Path(path).name) != ROLE_FC_SWEEP:
            return False
        try:
            _load(Path(path))
        except (FcSweepRecordError, OSError):
            return False
        return True

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
