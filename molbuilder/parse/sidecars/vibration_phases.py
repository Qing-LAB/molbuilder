"""A run in phases' index, read back — ``<label>_<stage>-run<N>.phases.json``.

Contract: `model/parse.md` § 5.5 (what a person can open is the REGISTRY's
question); the index's own shape is `siesta/vibration_phases.py`'s, which
writes and reads it (`engines/vibration.md` § 5.11 § 7).  `/api/results/dir`
asks the registry what reads each file and offers only what something
reads, so without this reader the index would sit in the run unoffered.  It
claims a file by its role in the catalogue AND by the schema its writer
stamps, never by the suffix alone.
"""
from __future__ import annotations

from pathlib import Path

from ..base import FileParser
from ..errors import ParseError
from ..types import SidecarResult
from ._helpers import build_sidecar_result


class VibrationPhasesError(ParseError):
    """A ``.phases.json`` that is not one, or is not this version."""


class VibrationPhasesFileParser(FileParser):
    """Parse a run in phases' index — the phases the run walks; every
    SIESTA call of the run, what each is, from what, how it ended; the
    finish's entry.  Returns a :class:`SidecarResult` with
    ``schema = "vibration-phases/v1"``."""

    name   = "vibration-phases-json"
    label  = "molbuilder .phases.json index"
    hint   = ("a SIESTA vibration run's index of its calls -- "
              "<label>_<stage>-run<N>.phases.json, written by the run")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        from molbuilder.runfiles import ROLE_PHASES, role_of
        from molbuilder.siesta.vibration_phases import read_index
        if role_of(Path(path).name) != ROLE_PHASES:
            return False
        try:
            read_index(Path(path))
        except (ValueError, OSError):
            return False
        return True

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        from molbuilder.siesta.vibration_phases import read_index
        try:
            payload = read_index(Path(path))
        except ValueError as exc:
            raise VibrationPhasesError(str(exc)) from exc
        return build_sidecar_result(
            payload=payload, schema="vibration-phases/v1",
            parser_name=cls.name, source=Path(path))


__all__ = ["VibrationPhasesError", "VibrationPhasesFileParser"]
