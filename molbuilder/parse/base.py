"""Abstract base classes for the parse module.

Per ``docs/model/parse.md`` § 1.  **One** ABC, :class:`FileParser`: a
parser turns one FILE into a result.

Forbidden by the doc:
* FileParsers do NO subprocess / network / threads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

# Imported here so subclasses can declare ``output = SomeResult``
# without circular imports: ``types.py`` does not import this module.
from .types import ParseResult


class FileParser(ABC):
    """One file path → one :class:`ParseResult`.

    Concrete subclasses set these class attributes:

    * ``name``   — short identifier (e.g. ``"siesta"``).
    * ``label``  — UI-friendly description.
    * ``hint``   — what file to point at when ``can_parse`` says no.
    * ``output`` — the concrete ``ParseResult`` subclass returned.
    """
    name:   str
    label:  str
    hint:   str
    output: Type[ParseResult]

    @classmethod
    def footgun_hint_for(cls, filename: str):
        """Optional per-parser foot-gun nudge for ``UnknownFormatError``.

        Called by the registry when no parser claims a file: each
        registered :class:`FileParser` gets a chance to return a
        short message of the form ``"X.log is the run-time log;
        point molbuilder at X_geom_optim.xyz"``.  Returning ``None``
        (the default) means "no nudge for this filename"; the
        registry only includes non-None replies in the error.

        Engine-specific knowledge (file-extension semantics,
        common mistakes) belongs HERE — not in the registry — so
        ``parse/registry.py`` stays engine-agnostic per
        model/parse.md § 7 #7.
        """
        return None

    @classmethod
    @abstractmethod
    def can_parse(cls, path: Path) -> bool:
        """Sniff the path's contents (cheaply) and return True iff
        this parser knows how to handle it.  Must not raise; a
        buggy sniffer must not take down the dispatch loop.
        """

    @classmethod
    @abstractmethod
    def parse(cls, path: Path) -> ParseResult:
        """Read + parse the file, return the typed
        ``ParseResult`` subclass declared in ``output``.
        Implementations should raise the canonical
        :exc:`molbuilder.parse.errors.ParseError` (or
        ``UnknownFormatError`` when reasonable) on malformed
        input rather than letting unstructured exceptions escape.
        """

