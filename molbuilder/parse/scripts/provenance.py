"""PROVENANCE block TextParser.

Phase F of parse-module.md migration: thin wrapper over the
legacy ``script_contract.extract_provenance_dict``.  PROVENANCE
is a flat ``key value-or-list`` snapshot of generator state at
generation time (generator-version, generated-at, resolved-
defaults).
"""

from __future__ import annotations

from dataclasses import replace

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult
from molbuilder.script_contract import (
    extract_provenance_dict as _legacy_extract,
)

from ._helpers import empty_script_result


class ProvenanceTextParser(TextParser):
    """Extract the ``PROVENANCE`` block as a flat dict.  Returns
    a :class:`ScriptResult` with only the ``provenance`` field set."""

    name   = "fdf-provenance"
    label  = "molbuilder PROVENANCE block"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        base = empty_script_result(cls.name)
        return replace(base, provenance=_legacy_extract(text))
