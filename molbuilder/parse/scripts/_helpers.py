"""Helpers shared across script-block TextParsers."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from molbuilder.parse.types import ScriptResult


def _iso_z() -> str:
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def empty_script_result(parser_name: str) -> ScriptResult:
    """Default-shape ScriptResult for per-block parsers — only the
    block they're responsible for gets populated; everything else
    stays at default-None.  Callers compose multiple blocks via the
    umbrella :class:`ScriptSourceTextParser`."""
    return ScriptResult(
        schema_version=1,
        parsed_at=_iso_z(),
        parser_name=parser_name,
        source="<text>",
    )
