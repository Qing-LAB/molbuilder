"""Helpers shared across sidecar FileParser wrappers.

Each sidecar wrapper builds a :class:`SidecarResult` from the
legacy loader's output.  This helper just standardises the
``parsed_at`` timestamp and the ParseResult envelope fields.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from molbuilder.parse.types import SidecarResult


def _iso_z() -> str:
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def build_sidecar_result(payload: Dict[str, Any], schema: str,
                         parser_name: str, source: Path) -> SidecarResult:
    """Wrap a loaded sidecar dict in the typed SidecarResult.

    ``schema`` is the discriminator (``"molstruct/v3"``,
    ``"spectra/v4"``, ``"transport/v1"``, etc.) consumers use to
    type-narrow further.
    """
    return SidecarResult(
        schema_version=1,
        parsed_at=_iso_z(),
        parser_name=parser_name,
        source=str(source),
        payload=payload,
        schema=schema,
    )
