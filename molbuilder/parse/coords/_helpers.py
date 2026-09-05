"""Helpers shared across coords FileParser wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from molbuilder.parse.types import ParseResult, StructureResult
from molbuilder.structure import Structure


def build_structure_result(structure: Structure,
                           cell: Optional[np.ndarray],
                           parser_name: str,
                           source: Path,
                           source_format: str = "unknown"
                           ) -> StructureResult:
    """Wrap a parsed Structure + cell in the typed StructureResult
    envelope.  Source path is resolved to an absolute path for
    cross-phase envelope consistency.
    """
    return StructureResult(
        **ParseResult.envelope(parser_name, source),
        structure=structure,
        cell=cell,
        source_format=source_format,
    )
