"""Helpers shared across script-block TextParsers."""

from __future__ import annotations


from molbuilder.parse.types import ParseResult, ScriptResult


def empty_script_result(parser_name: str) -> ScriptResult:
    """Default-shape ScriptResult for per-block parsers — only the
    block they're responsible for gets populated; everything else
    stays at default-None.  Callers compose multiple blocks via the
    umbrella :class:`ScriptSourceTextParser`."""
    return ScriptResult(**ParseResult.envelope(parser_name))
