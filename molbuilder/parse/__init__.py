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

import importlib

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

# THE PARSERS REGISTER THEMSELVES when their sub-package is imported, so each
# is imported here for that alone -- in this order, and nothing bound.
for _sub in ("engines", "coords", "sidecars", "instruments", "dirs"):
    importlib.import_module(f"{__name__}.{_sub}")
del _sub

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
