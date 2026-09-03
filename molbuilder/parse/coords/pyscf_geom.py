"""PySCF / geomeTRIC final-geometry ``<job>_optimized.xyz`` FileParser.

Absorbed from the legacy ``molbuilder.parsers.pyscf_struct``
``read_optimized_xyz``; that package was deleted 2026-06-21 and this is
the only reader (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

PySCF's optimized geometry comes back as a plain .xyz with the final
coords.  The reader is a thin wrapper over
:meth:`Structure.from_xyz` that adds a title from the file stem.

The .xyz format has no cell info — molecular calcs are typically
non-periodic.  StructureResult.cell stays None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import StructureResult
from molbuilder.structure import Structure

from ._helpers import build_structure_result


def _read_optimized_xyz(path: Union[str, Path]) -> Structure:
    """Read a ``<JOB>_optimized.xyz`` file and return a Structure.

    Thin wrapper over :meth:`Structure.from_xyz` so the parse layer
    has a single entry point for "final coords from the PySCF run".
    Errors surface as :class:`ValueError` from from_xyz.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    s = Structure.from_xyz(text)
    # XYZ comment line typically reads "Optimized geometry (PySCF)";
    # mirror the file stem onto Structure.title so downstream
    # re-render can label it.
    if not s.title:
        s.title = p.stem
    return s


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
        structure = _read_optimized_xyz(path)
        return build_structure_result(
            structure=structure,
            cell=None,
            parser_name=cls.name,
            source=path,
            source_format="pyscf-geom",
        )


# --------------------------------------------------------------------- #
#  Public-API re-exports                                                #
# --------------------------------------------------------------------- #
#
# H4a (test-cleanup): the .py-initial-coords + JOB-extract helpers
# live in ``parse/dirs/_assembler_helpers`` (the assembler composes
# them); re-export here for the natural per-file-type import path.
read_optimized_xyz = _read_optimized_xyz

from molbuilder.parse.dirs._assembler_helpers import (   # noqa: E402
    PyscfStructureError,
    extract_pyscf_job,
    read_py_initial_coords,
)


__all__ = [
    "PySCFGeomFileParser",
    "PyscfStructureError",
    "read_optimized_xyz",
    "read_py_initial_coords",
    "extract_pyscf_job",
]
