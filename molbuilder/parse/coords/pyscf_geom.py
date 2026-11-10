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
from molbuilder.parse.types import StructureResult
from molbuilder.runfiles import PYSCF_OPTIMIZED, ROLE_INITIAL, stem_before
from molbuilder.structure import Structure

from ._helpers import build_structure_result, read_through_the_codec


def _read_optimized_xyz(path: Union[str, Path]) -> Structure:
    """Read a ``<JOB>_optimized.xyz`` pair and return a Structure, through
    the codec's one read door (:func:`read_through_the_codec`): the run wrote
    the pair, and reading the ``.xyz`` alone would drop the labels and the
    cell its sidecar states (`model/structure.md` § 2.4)."""
    s = read_through_the_codec(path)
    # XYZ comment line typically reads "Optimized geometry (PySCF)";
    # mirror the file stem onto Structure.title so downstream
    # re-render can label it.
    if not s.title:
        s.title = Path(path).stem
    return s


class PySCFGeomFileParser(FileParser):
    """Parse a PySCF + geomeTRIC ``<job>_optimized.xyz`` file.

    Returns :class:`StructureResult` with the optimized geometry and the
    cell its pair states, on the structure and as the result's ``cell``.
    """

    name   = "pyscf-geom"
    label  = "PySCF / geomeTRIC geometry .xyz"
    hint   = (f"a PySCF deck's geometry: <job>{PYSCF_OPTIMIZED}, the one it "
              f"reached, or <job>{ROLE_INITIAL}, the one it started from")
    output = StructureResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # Match the geomeTRIC convention: ``_optimized.xyz`` suffix.
        # Plain ``.xyz`` files (without the suffix) are intentionally
        # NOT claimed here; the PySCFOutFileParser registered under
        # engines/ handles ``_geom_optim.xyz`` trajectories.
        # THE GEOMETRY REACHED, and the INPUT ECHOED BACK before relaxing
        # -- both single-frame pairs the PySCF deck writes, both this
        # reader's (the run's parts the trajectory inspector absorbs,
        # `web/trajectory.md`); the echo had no reader until 2026-10-19.
        return ((stem_before(path.name, PYSCF_OPTIMIZED) is not None
                 or stem_before(path.name, ROLE_INITIAL) is not None)
                and path.is_file())

    @classmethod
    def parse(cls, path: Path) -> StructureResult:
        structure = _read_optimized_xyz(path)
        return build_structure_result(
            structure=structure,
            cell=structure.cell,
            parser_name=cls.name,
            source=path,
            source_format="pyscf-geom",
        )


__all__ = [
    "PySCFGeomFileParser",
]
