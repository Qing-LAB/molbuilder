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

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult

from ._helpers import empty_script_result

# The extractors MOVED to their format's owner on 2026-09-05
# (`script_emit`, `plan.md` § 5d).  Imported here, not copied:
# two implementations of one grammar is the defect this move
# exists to remove.
from molbuilder.script_emit import (  # noqa: E402
    _extract_header_text,
)


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
