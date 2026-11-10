"""Helpers shared across coords FileParser wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from molbuilder.parse.errors import ParseError
from molbuilder.parse.types import ParseResult, StructureResult
from molbuilder.structure import Structure


def read_through_the_codec(path, *, frames: bool = False) -> Structure:
    """A STRUCTURE FILE READ BY THE CODEC'S ONE READ DOOR
    (`StructureCodec.load`, `model/structure.md` § 2.4) -- a `.pdb`, a pair
    with its sidecar applied and its pin compared -- frame 0, or with
    ``frames`` every frame.  What the door refuses -- the file gone between
    ``can_parse`` and here, columns that do not read, a pair whose atoms
    changed after it was written, an engine's record whose order nothing
    states -- is the registry's :class:`ParseError`, naming the file, which
    ``base.FileParser.parse`` requires (an unwrapped one reaches Flask as an
    HTML 500 the browser cannot read)."""
    from molbuilder.workingcopy_structure import StructureCodec
    p = Path(path)
    try:
        return StructureCodec().load(p, frames=frames)
    except (OSError, ValueError, TypeError) as exc:
        raise ParseError(f"{p.name}: {exc}") from exc


def build_structure_result(structure: Structure,
                           cell: Optional[np.ndarray],
                           parser_name: str,
                           source: Path,
                           source_format: str = "unknown"
                           ) -> StructureResult:
    """Wrap a parsed Structure + cell in the typed StructureResult
    envelope.  Source path is resolved to an absolute path.
    """
    return StructureResult(
        **ParseResult.envelope(parser_name, source),
        structure=structure,
        cell=cell,
        source_format=source_format,
    )
