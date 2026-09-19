"""Abstract base classes for the parse module.

Per ``docs/model/parse.md`` § 1.  **Two** ABCs, one discriminator per
scope (file / directory).  All concrete parsers subclass exactly one.
*(Three until 2026-09-05, when ``TextParser`` retired -- see below.  The
count here still said three until 2026-09-18.)*

Forbidden by the doc:
* (TextParsers did NO I/O; the ABC retired 2026-09-05 -- see below.)
* FileParsers do NO subprocess / network / threads.
* DirParsers MUST compose readers that own their formats; no inline
  file-level parsing.  WHICH reader is the question's: the REGISTRY answers
  *what typed result does this file hold* (`detect`+`parse`), and
  `engines/_run_ending.ending_of` answers *how did this run end* -- a
  substring scan, where the registry's answer costs a whole `Trajectory` to
  reach one string (`model/parse.md` § 5.4).  *(Said "compose registered
  FileParsers" until 2026-09-18, which the code stopped obeying that day for
  a measured reason; § 5.4 carries it.  Said "FileParsers + TextParsers"
  earlier the same day.)*
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


class DirParser(ABC):
    """One directory → one :class:`ParseResult`, composed from
    per-file parsers PLUS directory-level invariants
    (cross-file consistency, status state machine, stage ordering).

    Concrete subclasses MUST compose a reader that owns the format; they
    MUST NOT inline file-level parsing logic that duplicates one.  The
    REGISTRY is that reader for *what typed result does this file hold*
    (and is the only thing that may decide WHICH parser, so a registration
    change propagates); `engines/_run_ending.ending_of` is that reader for
    *how did this run end*, one per ROLE.  `model/parse.md` § 5.4 carries
    the split and why it exists.

    *(This licensed "(or directly invoke a known FileParser class)" until
    2026-09-18 -- which is exactly what ``registry.py``'s own header
    FORBIDS: "Callers MUST NOT import a parser class directly + call its
    methods -- go through the registry so registration changes propagate."
    Two files in one package, opposite rules, and the registry's is the one
    that has a reason attached, so the parenthetical went.)*
    """
    name:   str
    label:  str
    output: Type[ParseResult]

    @classmethod
    @abstractmethod
    def can_parse(cls, run_dir: Path) -> bool:
        """Cheap sniff of the directory's contents — typically
        "does it contain a .fdf + an .out?", or similar
        domain-specific check."""

    @classmethod
    @abstractmethod
    def parse(cls, run_dir: Path) -> ParseResult:
        """Walk the directory, compose per-file results, return
        the typed directory-level result."""
