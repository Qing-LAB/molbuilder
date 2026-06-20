"""Helpers shared across coords FileParser wrappers."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from molbuilder.parse.types import StructureResult
from molbuilder.structure import Structure


def _iso_z() -> str:
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


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
    try:
        source_str = str(Path(source).resolve())
    except OSError:
        source_str = str(source)
    return StructureResult(
        schema_version=1,
        parsed_at=_iso_z(),
        parser_name=parser_name,
        source=source_str,
        structure=structure,
        cell=cell,
        source_format=source_format,
    )
