"""A structure pair's ``.xyz`` FileParser -- a ``.xyz`` with its
``.molstruct.json`` beside it, read whole (`model/parse.md` § 4).

WHY IT EXISTS.  The pair is the file (`model/structure.md` § 2.4): a saved
structure, a frame set, a calculation's ``<label>.source.xyz``, a transport
calculation's ``junction.xyz`` -- each is its geometry AND its sidecar, and a
reader of the ``.xyz`` alone hands back less than is on disk: no cell, no
labels, no ``info``, no order record.  So a pair is read by the codec's one
read door, as the ``.pdb`` and PySCF geometry readers read theirs, every frame,
the box the pair states on the structure and as the result's ``cell``; the
``xyz`` reader keeps the XYZ files with no sidecar -- an engine's own, another
tool's.

WHAT IT DOES NOT CLAIM.  A file another reader is for by its name (a PySCF
deck's two geometries, geomeTRIC's trajectory): it and ``xyz`` ask one
question, `engines.xyz.unclaimed_by_name`, and split its files by the sidecar,
so ``detect`` -- exactly one or raise -- finds one reader for every XYZ.
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.engines.xyz import unclaimed_by_name
from molbuilder.parse.types import StructureResult
from molbuilder.sidecars.molstruct import paired_sidecar

from ._helpers import build_structure_result, read_through_the_codec


class StructurePairFileParser(FileParser):
    """Parse a structure pair into a :class:`StructureResult` -- the whole
    set of frames it holds, with what its sidecar states."""

    name   = "structure-pair"
    label  = "Structure pair (.xyz + .molstruct.json)"
    hint   = ("a .xyz with its <stem>.molstruct.json beside it -- a saved "
              "structure or frame set, a calculation's <label>.source.xyz")
    output = StructureResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return unclaimed_by_name(path) and paired_sidecar(path) is not None

    @classmethod
    def parse(cls, path: Path) -> StructureResult:
        # THROUGH THE ONE DOOR, every frame: the sidecar applied to the whole
        # set, its pin compared, its order record carried.
        structure = read_through_the_codec(path, frames=True)
        return build_structure_result(
            structure=structure,
            cell=structure.cell,
            parser_name=cls.name,
            source=Path(path),
            source_format="xyz",
        )


__all__ = ["StructurePairFileParser"]
