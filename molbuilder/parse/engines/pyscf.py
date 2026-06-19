"""PySCF / geomeTRIC ``.log`` / ``_geom_optim.xyz`` FileParser.

Phase C of parse-module.md migration: thin wrapper over the
legacy :class:`molbuilder.parsers.pyscf.PySCFParser`.  Phase H
absorbs the legacy parse logic directly and drops the wrapper.
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import TrajectoryResult
from molbuilder.parsers.pyscf import PySCFParser as _LegacyPySCFParser

from ._helpers import wrap_trajectory


class PySCFOutFileParser(FileParser):
    """Parse a PySCF + geomeTRIC trajectory file
    (``<job>_geom_optim.xyz``).  Returns a :class:`TrajectoryResult`
    with one Frame per geomeTRIC step + per-step SCF history
    (extracted from the companion ``.log`` when present)."""

    name   = "pyscf-out"
    label  = _LegacyPySCFParser.label
    hint   = _LegacyPySCFParser.hint
    output = TrajectoryResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return _LegacyPySCFParser.can_parse(str(path))

    @classmethod
    def parse(cls, path: Path) -> TrajectoryResult:
        traj = _LegacyPySCFParser.parse(str(path))
        return wrap_trajectory(traj, cls.name, path)
