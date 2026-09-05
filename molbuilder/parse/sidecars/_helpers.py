"""Helpers shared across sidecar FileParser wrappers.

Each sidecar wrapper builds a :class:`SidecarResult` from the
legacy loader's output.  This helper just standardises the
``parsed_at`` timestamp and the ParseResult envelope fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from molbuilder.parse.types import ParseResult, SidecarResult


def build_sidecar_result(payload: Dict[str, Any], schema: str,
                         parser_name: str, source: Path) -> SidecarResult:
    """Wrap a loaded sidecar dict in the typed SidecarResult.

    ``schema`` is the discriminator (``"molstruct/v3"``,
    ``"spectra/v4"``, ``"transport/v1"``, etc.) consumers use to
    type-narrow further.  Source path is resolved to an absolute
    path for envelope consistency across phases B/C/D
    (post-2026-06-19 round-2 fix).
    """
    return SidecarResult(
        **ParseResult.envelope(parser_name, source),
        payload=payload,
        schema=schema,
    )
