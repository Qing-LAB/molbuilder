"""SIESTA ``.out`` / ``.log`` FileParser.

Phase C of parse-module.md migration: thin wrapper over the
legacy :class:`molbuilder.parsers.siesta.SiestaParser`.  Phase H
absorbs the legacy parse logic directly and drops the wrapper.
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import TrajectoryResult
from molbuilder.parsers.siesta import SiestaParser as _LegacySiestaParser

from ._helpers import wrap_trajectory


class SiestaOutFileParser(FileParser):
    """Parse a SIESTA stdout-redirected output file (``.out`` /
    ``.log``).  Returns a :class:`TrajectoryResult` with one Frame
    per geometry step + per-step SCF history."""

    name   = "siesta-out"
    label  = _LegacySiestaParser.label
    hint   = _LegacySiestaParser.hint
    output = TrajectoryResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return _LegacySiestaParser.can_parse(str(path))

    @classmethod
    def parse(cls, path: Path) -> TrajectoryResult:
        traj = _LegacySiestaParser.parse(str(path))
        return wrap_trajectory(traj, cls.name, path)
