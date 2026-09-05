"""USER-CUSTOM block TextParser.

Absorbed from the retired
``molbuilder.script_contract.extract_user_custom_inner``, deleted with that
module on 2026-06-21 (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

USER-CUSTOM is user-owned territory between the BEGIN/END markers;
molbuilder preserves it byte-for-byte across regenerations
(`execution/job-contracts.md` § 3.5).  This module reads it; the write-side
helpers (``replace_user_custom_inner`` /
``merge_user_custom_from_target``) stay in ``molbuilder.script_contract``
until H2 rehomes them.
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
    _extract_user_custom_inner,
)


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
