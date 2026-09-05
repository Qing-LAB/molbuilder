"""The envelope every instrument parser fills — one place, like the
engines' and sidecars' helpers beside it."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from molbuilder.parse.types import InstrumentResult, ParseWarning

SCHEMA_VERSION = 1


def build_instrument_result(*, metrics: Dict[str, Any], parser_name: str,
                            source: Path,
                            parse_warnings: Optional[List[ParseWarning]] = None
                            ) -> InstrumentResult:
    return InstrumentResult(
        schema_version=SCHEMA_VERSION,
        parsed_at=datetime.now(timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z"),
        parser_name=parser_name,
        source=str(Path(source).resolve()),
        metrics=metrics,
        parse_warnings=list(parse_warnings or []),
    )
