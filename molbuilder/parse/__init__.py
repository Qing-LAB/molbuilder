"""Parse module — unified file/text/directory → ParseResult.

See ``docs/protocols/parse-module.md`` for the architectural
contract this package implements.

Public surface:

* ABCs — :class:`FileParser`, :class:`TextParser`, :class:`DirParser`.
* Result types — :class:`ParseResult` and its 7 frozen subclasses.
* Registry / dispatch — :func:`detect`, :func:`parse`,
  :func:`parse_text`, :func:`parse_dir`, :func:`register`.
* Exceptions — :exc:`UnknownFormatError`, :exc:`AmbiguousFormatError`.

Phase A (this commit) ships the skeleton only.  Phases B-H migrate
the existing parser/decoder code into this package incrementally.
"""

from .base import DirParser, FileParser, TextParser
from .errors import AmbiguousFormatError, UnknownFormatError
from .registry import detect, parse, parse_dir, parse_text, register
from .types import (
    BundleResult,
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
from . import sidecars as _sidecars   # noqa: F401  -- side-effect import
from . import dirs as _dirs   # noqa: F401  -- side-effect import

__all__ = [
    # ABCs
    "FileParser", "TextParser", "DirParser",
    # Results
    "ParseResult", "TrajectoryResult", "StructureResult",
    "SidecarResult", "ScriptResult", "JobResult", "BundleResult",
    "ParseWarning",
    # Registry / dispatch
    "detect", "parse", "parse_dir", "parse_text", "register",
    # Exceptions
    "UnknownFormatError", "AmbiguousFormatError",
]
