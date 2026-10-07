"""Parse module — unified file/text/directory → ParseResult.

See ``docs/model/parse.md`` for the architectural
contract this package implements.

Public surface:

* ABC — :class:`FileParser`.
* Result types — :class:`ParseResult` and its frozen subclasses
  (trajectory / structure / sidecar / instrument / engine parameters).
* Registry / dispatch — :func:`detect`, :func:`parse`, :func:`register`.
* Exceptions — :exc:`ParseError` and its two children
  :exc:`UnknownFormatError` / :exc:`AmbiguousFormatError`.  Detection
  raises BOTH, and they are SIBLINGS: a caller that catches only the
  first turns a registry overlap into an unhandled exception.
"""

from .base import FileParser
from .errors import (AmbiguousFormatError, ParseError,
                     UnknownFormatError)
from .registry import detect, parse, register
from .types import (
    ParseResult,
    ParseWarning,
    InstrumentResult,
    SidecarResult,
    StructureResult,
    TrajectoryResult,
    EngineParamsResult,
)

# Import sub-packages so their register() side-effects run.
from . import engines as _engines   # noqa: F401  -- side-effect import
from . import coords as _coords   # noqa: F401  -- side-effect import
from . import sidecars as _sidecars   # noqa: F401  -- side-effect import
from . import instruments as _instruments   # noqa: F401  -- side-effect import
from . import dirs as _dirs   # noqa: F401  -- side-effect import

__all__ = [
    # ABCs
    "FileParser",
    # Results
    "ParseResult", "TrajectoryResult", "StructureResult",
    "SidecarResult", "InstrumentResult", "EngineParamsResult",
    "ParseWarning",
    # Registry / dispatch
    "detect", "parse", "register",
    # Exceptions
    "ParseError", "UnknownFormatError", "AmbiguousFormatError",
]
