"""HEADER block TextParser.

Absorbed from the retired
``molbuilder.script_contract.extract_header_text``, deleted with that
module on 2026-06-21 (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

HEADER is human-readable prose between the BEGIN/END markers — no
structure, just verbatim text with the leading ``# `` comment
prefix stripped from each line.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult

from ._helpers import empty_script_result
from .markers import BLOCK_HEADER, MARKER_RE


def _extract_header_text(text: str) -> Optional[str]:
    """Find the HEADER block and return its inner content as a single
    string (free-form prose, comment prefixes stripped).

    Returns ``None`` when no HEADER block is present.  The leading
    ``# `` (or ``#``) on each line is removed so the result is the
    raw prose the generator wrote; line ordering is preserved.
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m or m.group(1) != BLOCK_HEADER:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    out_lines: List[str] = []
    for raw in lines[begin_idx + 1: end_idx]:
        # Strip the comment prefix the generator emits ("# " or "#").
        if raw.startswith("# "):
            out_lines.append(raw[2:])
        elif raw.startswith("#"):
            out_lines.append(raw[1:])
        else:
            out_lines.append(raw)
    return "\n".join(out_lines)


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
        return replace(base, header=_extract_header_text(text))
