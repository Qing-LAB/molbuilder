"""Parse module — unified file/text/directory → ParseResult.

See ``docs/model/parse.md`` for the architectural
contract this package implements.

Public surface:

* ABCs — :class:`FileParser`, :class:`TextParser`, :class:`DirParser`.
* Result types — :class:`ParseResult` and its 5 frozen subclasses
  (trajectory / structure / sidecar / script / job — ``BundleResult``
  retired 2026-08-29 with calculation-to-calculation passing).
* Registry / dispatch — :func:`detect`, :func:`parse`,
  :func:`parse_text`, :func:`parse_dir`, :func:`register`.
* Exceptions — :exc:`UnknownFormatError`, :exc:`AmbiguousFormatError`.

The 2026-06 migration (phases A-H) is complete: the engine, coords,
sidecar, script and dir parsers all live here and register at import.
"""

from .base import DirParser, FileParser, TextParser
from .errors import AmbiguousFormatError, UnknownFormatError
from .registry import detect, parse, parse_dir, parse_text, register
from .types import (
    JobResult,
    ParseResult,
    ParseWarning,
    ScriptResult,
    SidecarResult,
    StructureResult,
    TrajectoryResult,
)

# Import sub-packages so their register() side-effects run.
from . import engines as _engines   # noqa: F401  -- side-effect import
from . import coords as _coords   # noqa: F401  -- side-effect import
from . import sidecars as _sidecars   # noqa: F401  -- side-effect import
from . import dirs as _dirs   # noqa: F401  -- side-effect import

# Convenience re-exports of the canonical entry-point classes
# from each sub-package.  Callers who know exactly which parser
# they want get them off the top-level namespace without
# having to know the sub-package layout.
from .scripts.source import ScriptSourceTextParser   # noqa: F401

__all__ = [
    # ABCs
    "FileParser", "TextParser", "DirParser",
    # Results
    "ParseResult", "TrajectoryResult", "StructureResult",
    "SidecarResult", "ScriptResult", "JobResult",
    "ParseWarning",
    # Registry / dispatch
    "detect", "parse", "parse_dir", "parse_text", "register",
    # Exceptions
    "UnknownFormatError", "AmbiguousFormatError",
    # Canonical entry-point classes (convenience re-exports)
    "ScriptSourceTextParser",     # umbrella TextParser (Phase F)
]
