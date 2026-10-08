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

import json
from pathlib import Path
from typing import Any, Dict

from ..base import FileParser
from ..errors import ParseError
from ..types import SidecarResult
from ._helpers import build_sidecar_result


class FcSweepRecordError(ParseError):
    """A ``.fc-sweep.json`` that is not one, or is not this version."""


def _load(path: Path) -> Dict[str, Any]:
    """The record as a dict, or raise — the one reader of this file."""
    from molbuilder.persist import check_schema
    from molbuilder.spectra.displacement_sweep import SWEEP_SCHEMA
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FcSweepRecordError(f"{Path(path).name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FcSweepRecordError(
            f"{Path(path).name}: a displacement sweep's record is a JSON "
            f"object, got {type(payload).__name__}")
    try:
        check_schema(str(payload.get("schema", "")), SWEEP_SCHEMA,
                     label=Path(path).name)
    except ValueError as exc:
        raise FcSweepRecordError(str(exc)) from exc
    return payload


class FcSweepRecordFileParser(FileParser):
    """Parse a ``<label>.fc-sweep.json`` — the force-constant stages of a
    SIESTA vibration compared: each stage and what it varied, every mode's
    frequency per stage, the force-constant changes, the paths of each
    stage's files.  Returns a :class:`SidecarResult` with
    ``schema = "fc-sweep/v1"``."""

    name   = "fc-sweep-json"
    label  = "molbuilder .fc-sweep.json record"
    hint   = ("a SIESTA vibration's displacement sweep -- <label>.fc-sweep.json, "
              "written by `jobset summarize task`")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        from molbuilder.runfiles import role_of
        if role_of(Path(path).name) != ".fc-sweep.json":
            return False
        try:
            _load(Path(path))
        except (FcSweepRecordError, OSError):
            return False
        return True

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        return build_sidecar_result(
            payload=_load(Path(path)),
            schema="fc-sweep/v1",
            parser_name=cls.name,
            source=Path(path),
        )


__all__ = ["FcSweepRecordError", "FcSweepRecordFileParser"]
