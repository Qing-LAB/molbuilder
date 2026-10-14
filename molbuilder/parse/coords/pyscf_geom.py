"""PySCF / geomeTRIC final-geometry ``<job>_optimized.xyz`` FileParser.

The run writes its optimized geometry as a pair, the ``.xyz`` and its
``.molstruct.json`` (``StructureCodec.write_moved``), so the reader is the
codec's one read door, which applies the labels and the cell the run was
given and compares the pin; a title from the file stem when the comment
line states none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from molbuilder.parse.base import FileParser
from molbuilder.parse.errors import ParseError
from molbuilder.parse.types import StructureResult
from molbuilder.structure import Structure

from ._helpers import build_structure_result


def _read_optimized_xyz(path: Union[str, Path]) -> Structure:
    """Read a ``<JOB>_optimized.xyz`` pair and return a Structure.

    THROUGH THE ONE DOOR, ``StructureCodec.load``: the run wrote the pair, and
    reading the ``.xyz`` alone would drop the labels and the cell its
    sidecar states (`model/structure.md` § 2.4).  Wrapped as the PDB parser
    wraps it: ``base.FileParser.parse`` requires the canonical
    :class:`ParseError` -- the file can vanish between ``can_parse`` and here,
    and the door refuses a pair whose atoms changed after it was written.
    """
    from molbuilder.workingcopy_structure import StructureCodec
    p = Path(path)
    try:
        s = StructureCodec().load(p)
    except (OSError, ValueError, TypeError) as exc:
        raise ParseError(f"{p.name}: {exc}") from exc
    # XYZ comment line typically reads "Optimized geometry (PySCF)";
    # mirror the file stem onto Structure.title so downstream
    # re-render can label it.
    if not s.title:
        s.title = p.stem
    return s


class PySCFGeomFileParser(FileParser):
    """Parse a PySCF + geomeTRIC ``<job>_optimized.xyz`` file.

    Returns :class:`StructureResult` with the optimized geometry, the cell
    its pair states on the structure.
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
read_optimized_xyz = _read_optimized_xyz

__all__ = [
    "PySCFGeomFileParser",
    "read_optimized_xyz",
]
