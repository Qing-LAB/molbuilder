"""PROVENANCE block TextParser.

Absorbed from the retired
``molbuilder.script_contract.extract_provenance_dict``, deleted with that
module on 2026-06-21 (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

PROVENANCE is a flat ``key value-or-list`` snapshot of generator
state at generation time (generator-version, generated-at,
resolved-defaults).  Format::

    # === molbuilder provenance BEGIN ===
    #   generator-version    <value>
    #   generated-at         <iso8601>
    #   form-config-hash     <hash>            (optional)
    #   resolved-defaults:
    #     <key>              <description>
    #     <key>              <description>
    # === molbuilder provenance END ===

:func:`_extract_provenance_dict` returns a flat ``{key -> value}``
where the ``resolved-defaults:`` sub-block expands into
``"resolved-defaults.<key>"`` entries.  Forward-compatible:
unknown top-level keys flow through.
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
    _extract_provenance_dict,
)


class ProvenanceTextParser(TextParser):
    """Extract the ``PROVENANCE`` block as a flat dict.  Returns
    a :class:`ScriptResult` with only the ``provenance`` field set."""

    name   = "fdf-provenance"
    label  = "molbuilder PROVENANCE block"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        base = empty_script_result(cls.name)
        return replace(base, provenance=_extract_provenance_dict(text))
