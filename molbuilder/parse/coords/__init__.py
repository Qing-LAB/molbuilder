"""Coords FileParsers — geometry-only file formats.

Each module here defines exactly one :class:`FileParser` subclass
returning a :class:`StructureResult`.  Adding a new geometry-file
format is two steps: new module here, import + ``register`` below.

These replaced the legacy ``parsers.{siesta,pyscf}_struct`` modules
(deleted 2026-06-21; provenance: `docs/archive/old_docs/protocols/parse-module.md` § 8).  What they add is that
:class:`StructureResult` carries
``cell`` (3x3 ndarray in Å) as a first-class field, fixing the
"Structure is geometry-only, cell got dropped" gap that
originally surfaced as the Phase 1 lattice-extraction bug in
``parse/dirs/job.py``.
"""

from molbuilder.parse.registry import register
from .pdb import PdbFileParser
from .pyscf_geom import PySCFGeomFileParser
from .siesta_xv import (
    SiestaXVFileParser,
    read_xv,
    read_xv_cell,
    read_xv_with_cell,
)


register(SiestaXVFileParser)
register(PySCFGeomFileParser)
register(PdbFileParser)

__all__ = [
    "SiestaXVFileParser",
    "PySCFGeomFileParser",
    "PdbFileParser",
    "read_xv",          # .XV -> Structure (Å)
    "read_xv_cell",     # .XV -> 3x3 cell (Å)
    "read_xv_with_cell",  # .XV -> (Structure, cell) in ONE pass -- the door
]
