"""Parser registry + dispatch.

Per ``docs/model/parse.md`` § 3.  ONE flat list, of FileParsers.  *(A
second held DirParsers until 2026-10-04, when the one directory door moved
up to the run door, `molbuilder.runs.folder_answer` -- it read the
description, which this floor must not (plan B11, B14); a third held
TextParsers until 2026-09-05.)*

The dispatch functions :func:`detect` and :func:`parse` are the only public
entry points.  Callers MUST NOT import a parser class directly + call
its methods — go through the registry so registration changes
propagate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Type

from .base import FileParser
from .errors import AmbiguousFormatError, UnknownFormatError
from .types import ParseResult


_FILE_PARSERS:  List[Type[FileParser]] = []


def register(parser: Type[FileParser]) -> None:
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
    raise TypeError(
        f"register: {parser!r} is not a FileParser subclass"
    )


def detect(path: Path) -> Type[FileParser]:
    """Return the ONE parser whose ``can_parse(path)`` is True.

    **Exactly one, not the first.**  ``_detect_one`` collects every
    match and raises :exc:`AmbiguousFormatError` when more than one
    parser claims the path -- registration order confers no
    precedence, and a parser added later cannot quietly shadow one
    added earlier.  (This said "the first parser" until 2026-09-04,
    which described a first-wins dispatch the code has never had;
    `model/parse.md` § 3's table had it right.)

    A DIRECTORY is not a file, and is refused by name: a run folder is the
    run door's (`molbuilder.runs.folder_answer`).  *(DirParsers answered one
    here until 2026-10-04, and a ``TextParser`` tier before 2026-09-05.)*

    Raises :exc:`UnknownFormatError` when no parser matches, with
    a tailored error message listing every registered parser of
    the appropriate kind.
    """
    path = Path(path)
    if path.is_dir():
        raise UnknownFormatError(
            f"{path.name!r} is a directory: a run folder is read through the "
            f"run door (`molbuilder.runs.folder_answer`), not a file parser.")
    return _detect_one(path, _FILE_PARSERS, "file")


def _detect_one(path: Path,
                pool: List[Type[FileParser]],
                kind_label: str
                ) -> Type[FileParser]:
    matches: List[Type[FileParser]] = []
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


# `parse_dir(path)` stood here until 2026-10-04: force-detect among
# DirParsers.  Its one parser, `JobDirParser`, read the description -- which
# floor 1 must not -- and moved up to the run door
# (`molbuilder.runs.folder_answer`, plan B11, B14).


# `parse_text(text, parser)` stood here until 2026-09-05, with the
# `TextParser` ABC it dispatched to.  It had no production caller:
# every use was a test or the docstring example.  See `base.py`.


# Test helpers --------------------------------------------------------- #


def _registered_file_parsers() -> List[Type[FileParser]]:
    """Snapshot of the registered FileParsers; for tests + audit."""
    return list(_FILE_PARSERS)
