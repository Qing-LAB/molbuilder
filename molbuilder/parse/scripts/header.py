"""HEADER block TextParser.

Phase F of parse-module.md migration: thin wrapper over the
legacy ``script_contract.extract_header_text``.  HEADER is
human-readable prose between the BEGIN/END markers — no
structure, just verbatim text.
"""

from __future__ import annotations

from dataclasses import replace

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult
from molbuilder.script_contract import extract_header_text as _legacy_extract

from ._helpers import empty_script_result


class HeaderTextParser(TextParser):
    """Extract the ``HEADER`` block content.  Returns a
    :class:`ScriptResult` with only the ``header`` field set;
    other block fields stay at their default-None."""

    name   = "fdf-header"
    label  = "molbuilder HEADER block"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        base = empty_script_result(cls.name)
        return replace(base, header=_legacy_extract(text))
