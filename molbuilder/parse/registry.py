"""Parser registry + dispatch.

Per ``docs/model/parse.md`` § 4.  Three flat lists:
one for FileParsers, one for TextParsers (not auto-detected;
callers pick explicitly), one for DirParsers.

The dispatch functions :func:`detect`, :func:`parse`,
:func:`parse_text`, :func:`parse_dir` are the only public entry
points.  Callers MUST NOT import a parser class directly + call
its methods — go through the registry so registration changes
propagate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Type, Union

from .base import DirParser, FileParser, TextParser
from .errors import AmbiguousFormatError, UnknownFormatError
from .types import ParseResult


_FILE_PARSERS:  List[Type[FileParser]] = []
_TEXT_PARSERS:  List[Type[TextParser]] = []
_DIR_PARSERS:   List[Type[DirParser]]  = []


def register(parser: Type[Union[FileParser, TextParser, DirParser]]) -> None:
    """Add a parser to the registry.

    Module-init time only; not for runtime registration during
    request handling.  Idempotent — re-registration of the same
    class is a no-op.

    Order matters: the dispatcher tries registered parsers in
    insertion order, so more-specific parsers should be registered
    first.  Per-package ``__init__.py`` files own the registration
    order for their parsers.
    """
    if issubclass(parser, FileParser):
        if parser not in _FILE_PARSERS:
            _FILE_PARSERS.append(parser)
        return
    if issubclass(parser, TextParser):
        if parser not in _TEXT_PARSERS:
            _TEXT_PARSERS.append(parser)
        return
    if issubclass(parser, DirParser):
        if parser not in _DIR_PARSERS:
            _DIR_PARSERS.append(parser)
        return
    raise TypeError(
        f"register: {parser!r} is not a FileParser / TextParser / "
        f"DirParser subclass"
    )


def detect(path: Path) -> Union[Type[FileParser], Type[DirParser]]:
    """Return the ONE parser whose ``can_parse(path)`` is True.

    **Exactly one, not the first.**  ``_detect_one`` collects every
    match and raises :exc:`AmbiguousFormatError` when more than one
    parser claims the path -- registration order confers no
    precedence, and a parser added later cannot quietly shadow one
    added earlier.  (This said "the first parser" until 2026-09-04,
    which described a first-wins dispatch the code has never had;
    `model/parse.md` § 3's table had it right.)

    If ``path`` is a directory, only DirParsers are tried; if
    ``path`` is a file, only FileParsers are tried.  ``TextParser``
    has no detection by design (no path to inspect) -- callers
    pick TextParser implementations explicitly.

    Raises :exc:`UnknownFormatError` when no parser matches, with
    a tailored error message listing every registered parser of
    the appropriate kind.
    """
    path = Path(path)
    if path.is_dir():
        return _detect_one(path, _DIR_PARSERS, "directory")
    return _detect_one(path, _FILE_PARSERS, "file")


def _detect_one(path: Path,
                pool: List[Type[Union[FileParser, DirParser]]],
                kind_label: str
                ) -> Type[Union[FileParser, DirParser]]:
    matches: List[Type[Union[FileParser, DirParser]]] = []
    for cls in pool:
        try:
            if cls.can_parse(path):
                matches.append(cls)
        except Exception:
            # A buggy sniffer must not take down the dispatch loop.
            continue
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(c.name for c in matches)
        raise AmbiguousFormatError(
            f"Multiple {kind_label} parsers claim "
            f"{os.path.basename(str(path))!r}: {names}.  Check the "
            f"can_parse implementations for overlap."
        )
    # No matches.
    base = os.path.basename(str(path))
    lines = [f"No registered {kind_label} parser knows how to handle {base!r}."]
    if pool:
        lines.append(f"Supported {kind_label} formats:")
        for c in pool:
            hint = f" -- {getattr(c, 'hint', '')}" if getattr(c, "hint", "") else ""
            lines.append(f"  * {c.label}{hint}")
    else:
        lines.append(f"(no {kind_label} parsers registered)")
    # Foot-gun nudges: each registered parser exposes its own
    # ``footgun_hint_for(filename)`` (default returns ``None``),
    # so engine-specific knowledge (".fdf is INPUT not OUTPUT")
    # stays in the engine module per model/parse.md § 7 #7.
    nudges = []
    for c in pool:
        hint_fn = getattr(c, "footgun_hint_for", None)
        if hint_fn is None:
            continue
        try:
            nudge = hint_fn(base)
        except Exception:
            nudge = None
        if nudge:
            nudges.append(nudge)
    if nudges:
        for n in nudges:
            lines.append(f"\nHint: {n}")
    else:
        lines.append(
            "\nHint: see README / docs/model/parse.md for the "
            "list of recognised file types."
        )
    raise UnknownFormatError("\n".join(lines))


def parse(path: Path) -> ParseResult:
    """Detect + parse in one call.

    Convenience for code that doesn't need to know the parser
    class.  Equivalent to ``detect(path).parse(path)``.
    """
    return detect(path).parse(Path(path))


def parse_dir(path: Path) -> ParseResult:
    """Force-detect among DirParsers only.

    Used by JobMonitor, Results, and bundle handoff where the
    contract is "this is a project dir".  Raises
    :exc:`UnknownFormatError` if the path is not a directory.
    """
    path = Path(path)
    if not path.is_dir():
        raise UnknownFormatError(
            f"parse_dir: {path} is not a directory")
    return _detect_one(path, _DIR_PARSERS, "directory").parse(path)


def parse_text(text: str, parser: Type[TextParser]) -> ParseResult:
    """Parse a known text body.  Caller specifies the TextParser
    explicitly; no detection.

    Equivalent to ``parser.parse(text)``; routed through this
    function so tracking / instrumentation can be added at the
    registry layer without touching call sites.
    """
    return parser.parse(text)


# Test helpers --------------------------------------------------------- #


def _registered_file_parsers() -> List[Type[FileParser]]:
    """Snapshot of the registered FileParsers; for tests + audit."""
    return list(_FILE_PARSERS)


def _registered_text_parsers() -> List[Type[TextParser]]:
    return list(_TEXT_PARSERS)


def _registered_dir_parsers() -> List[Type[DirParser]]:
    return list(_DIR_PARSERS)
