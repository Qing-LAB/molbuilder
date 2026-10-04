"""Abstract base classes for the parse module.

Per ``docs/model/parse.md`` § 1.  **One** ABC, :class:`FileParser`: a
parser turns one FILE into a result.  *(A ``DirParser`` stood beside it until
2026-10-04: its one subclass read the description, which this floor must not
-- the directory door moved up to the run door, `molbuilder.runs`, plan B11,
B14.  A ``TextParser`` retired 2026-09-05.)*

Forbidden by the doc:
* (TextParsers did NO I/O; the ABC retired 2026-09-05 -- see below.)
* FileParsers do NO subprocess / network / threads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

# Imported here so subclasses can declare ``output = SomeResult``
# without circular imports.  ``types.py`` only imports from this
# module (the ABCs), not the other way round.
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


# `TextParser` stood here until 2026-09-05.
#
# Its six implementations all read molbuilder's OWN generated blocks --
# HEADER / PROVENANCE / BENCH-MARKS / ATOM-METADATA / USER-CUSTOM -- and
# that is the one case in this package with nothing to detect: the caller
# always knows which block it wants, so the registry's whole purpose
# ("query it rather than knowing which parser to call") did not apply.
# Every class was a function in a costume: `ProvenanceTextParser.parse`
# built a ten-field `ScriptResult` to carry the one dict the extractor had
# already returned.
#
# The readers moved to the module that WRITES the blocks (`script_emit`),
# which also removed a circular import the split had forced.  `plans/plan.md` § 5d.


