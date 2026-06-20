"""USER-CUSTOM block TextParser.

Phase F of parse-module.md migration: thin wrapper over the
legacy ``script_contract.extract_user_custom_inner``.  This is
the user-owned territory between the BEGIN/END markers;
molbuilder preserves it byte-for-byte across regenerations
(script-contract.md § 4.6).
"""

from __future__ import annotations

from dataclasses import replace

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult
from molbuilder.script_contract import (
    extract_user_custom_inner as _legacy_extract,
)

from ._helpers import empty_script_result


class UserCustomTextParser(TextParser):
    """Extract the ``USER-CUSTOM`` block inner lines.  Returns a
    :class:`ScriptResult` with only the ``user_custom`` field set
    (list of lines, NOT a single string — preserves line breaks
    for the byte-for-byte re-emission promise)."""

    name   = "fdf-user-custom"
    label  = "molbuilder USER-CUSTOM block"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        base = empty_script_result(cls.name)
        return replace(base, user_custom=_legacy_extract(text))
