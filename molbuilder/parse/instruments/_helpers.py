"""The envelope every instrument parser fills.

One line, because `ParseResult.envelope` is the one home for the four
fields every result carries -- see its docstring for why this file used
to build them by hand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from molbuilder.parse.types import InstrumentResult, ParseResult, ParseWarning


def build_instrument_result(*, metrics: Dict[str, Any], parser_name: str,
                            source: Path,
                            parse_warnings: Optional[List[ParseWarning]] = None
                            ) -> InstrumentResult:
    return InstrumentResult(
        **ParseResult.envelope(parser_name, source),
        metrics=metrics,
        parse_warnings=list(parse_warnings or []),
    )
