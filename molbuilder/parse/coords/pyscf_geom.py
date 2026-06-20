"""PySCF / geomeTRIC final-geometry ``<job>_optimized.xyz`` FileParser.

Phase E of parse-module.md migration: wraps
:func:`molbuilder.parsers.pyscf_struct.read_optimized_xyz`.
PySCF's optimized geometry comes back as a plain .xyz with the
final coords; the legacy reader is a thin wrapper over
``Structure.from_xyz`` that adds a title from the file stem.

The .xyz format has no cell info — molecular calcs are typically
non-periodic.  StructureResult.cell stays None.
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import StructureResult
from molbuilder.parsers.pyscf_struct import read_optimized_xyz as _legacy_read

from ._helpers import build_structure_result


class PySCFGeomFileParser(FileParser):
    """Parse a PySCF + geomeTRIC ``<job>_optimized.xyz`` file.

    Returns :class:`StructureResult` with the optimized geometry.
    ``cell`` is None — PySCF optimizations are non-periodic by
    default; periodic-PySCF cell info lives elsewhere if needed.
    """

    name   = "pyscf-geom"
    label  = "PySCF / geomeTRIC final-geometry .xyz"
    hint   = "files matching <job>_optimized.xyz"
    output = StructureResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # Match the geomeTRIC convention: ``_optimized.xyz`` suffix.
        # Plain ``.xyz`` files (without the suffix) are intentionally
        # NOT claimed here; the PySCFOutFileParser registered under
        # engines/ handles ``_geom_optim.xyz`` trajectories.
        return path.name.endswith("_optimized.xyz") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> StructureResult:
        structure = _legacy_read(path)
        return build_structure_result(
            structure=structure,
            cell=None,
            parser_name=cls.name,
            source=path,
            source_format="pyscf-geom",
        )
