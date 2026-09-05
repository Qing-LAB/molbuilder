"""Parse module — unified file/text/directory → ParseResult.

See ``docs/model/parse.md`` for the architectural
contract this package implements.

Public surface:

* ABCs — :class:`FileParser`, :class:`DirParser`.  *(`TextParser`
  retired 2026-09-05: its only implementations read molbuilder's OWN
  generated blocks, which need no detection — they moved to the
  module that writes them, `script_emit`.  `plans/plan.md` § 5d.)*
* Result types — :class:`ParseResult` and its 5 frozen subclasses
  (trajectory / structure / sidecar / script / instrument —
  ``BundleResult`` retired 2026-08-29 with calculation-to-calculation
  passing, ``JobResult`` 2026-09-04 with the run decoder).
* Registry / dispatch — :func:`detect`, :func:`parse`,
  :func:`parse_dir`, :func:`register`.
* Exceptions — :exc:`ParseError` and its two children
  :exc:`UnknownFormatError` / :exc:`AmbiguousFormatError`.  Detection
  raises BOTH, and they are SIBLINGS: a caller that catches only the
  first turns a registry overlap into an unhandled exception.

The 2026-06 migration (phases A-H) is complete: the engine, coords,
sidecar, script and dir parsers all live here and register at import.
"""

from .base import DirParser, FileParser
from .errors import (AmbiguousFormatError, ParseError,
                     UnknownFormatError)
from .registry import detect, parse, parse_dir, register
from .types import (
    ParseResult,
    ParseWarning,
    InstrumentResult,
    SidecarResult,
    StructureResult,
    TrajectoryResult,
)

# Import sub-packages so their register() side-effects run.
from . import engines as _engines   # noqa: F401  -- side-effect import
from . import coords as _coords   # noqa: F401  -- side-effect import
from . import sidecars as _sidecars   # noqa: F401  -- side-effect import
from . import instruments as _instruments   # noqa: F401  -- side-effect import
from . import dirs as _dirs   # noqa: F401  -- side-effect import

# Convenience re-exports of the canonical entry-point classes
# from each sub-package.  Callers who know exactly which parser
# they want get them off the top-level namespace without
# having to know the sub-package layout.

__all__ = [
    # ABCs
    "FileParser", "DirParser",
    # Results
    "ParseResult", "TrajectoryResult", "StructureResult",
    "SidecarResult", "InstrumentResult",
    "ParseWarning",
    # Registry / dispatch
    "detect", "parse", "parse_dir", "register",
    # Exceptions
    "ParseError", "UnknownFormatError", "AmbiguousFormatError",
    # Canonical entry-point classes (convenience re-exports)
]
