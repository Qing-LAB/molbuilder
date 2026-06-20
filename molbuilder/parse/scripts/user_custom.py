"""USER-CUSTOM block TextParser.

H1 of parse-module.md migration (was Phase F wrapper around
``script_contract.extract_user_custom_inner``): absorbed the
extractor body directly so this module no longer imports from
``molbuilder.script_contract``.

USER-CUSTOM is user-owned territory between the BEGIN/END markers;
molbuilder preserves it byte-for-byte across regenerations
(script-contract.md § 4.6).  This module reads it; the write-side
helpers (``replace_user_custom_inner`` /
``merge_user_custom_from_target``) stay in ``molbuilder.script_contract``
until H2 rehomes them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult

from ._helpers import empty_script_result
from .markers import BLOCK_USER_CUSTOM, MARKER_RE


def _extract_user_custom_inner(text: str) -> Optional[List[str]]:
    """Return the inner lines of the USER-CUSTOM block in ``text``, or
    ``None`` if there is no well-formed USER-CUSTOM BEGIN/END pair.

    Inner lines are everything STRICTLY between the BEGIN and END
    markers (markers excluded).  Trailing/leading whitespace inside
    the block is preserved.
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1) != BLOCK_USER_CUSTOM:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        return None
    return lines[begin_idx + 1: end_idx]


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
        return replace(base, user_custom=_extract_user_custom_inner(text))
