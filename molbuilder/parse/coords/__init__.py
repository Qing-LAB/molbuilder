"""Coords FileParsers — geometry-only file formats.

Each module here defines exactly one :class:`FileParser` subclass
returning a :class:`StructureResult`.  Adding a new geometry-file
format is two steps: new module here, import + ``register`` below.

Phase E of parse-module.md migration: wraps the legacy
``parsers.{siesta,pyscf}_struct`` modules.  The key value
add over the legacy is that :class:`StructureResult` carries
``cell`` (3x3 ndarray in Å) as a first-class field, fixing the
"Structure is geometry-only, cell got dropped" gap that
originally surfaced as the Phase 1 lattice-extraction bug in
``parse/dirs/job.py``.
"""

from molbuilder.parse.registry import register
from .pyscf_geom import PySCFGeomFileParser
from .siesta_xv import SiestaXVFileParser


register(SiestaXVFileParser)
register(PySCFGeomFileParser)

__all__ = [
    "SiestaXVFileParser",
    "PySCFGeomFileParser",
]
