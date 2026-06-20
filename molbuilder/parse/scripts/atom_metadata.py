"""ATOM-METADATA block TextParser.

Phase F of parse-module.md migration: thin wrapper over the
legacy ``script_contract.extract_atom_metadata_dict``.
ATOM-METADATA carries the regions + frozen_atoms (the
``.molstruct.json`` schema v3 payload embedded in the .fdf).
"""

from __future__ import annotations

from dataclasses import replace

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult
from molbuilder.script_contract import (
    extract_atom_metadata_dict as _legacy_extract,
)

from ._helpers import empty_script_result


class AtomMetadataTextParser(TextParser):
    """Extract the ``ATOM-METADATA`` block as the embedded
    molstruct/v3 dict.  Returns a :class:`ScriptResult` with
    only the ``atom_metadata`` field set; the dict's
    ``schema_version`` is also surfaced into the result's
    ``block_schema_versions["atom-metadata"]`` for
    cross-block-version auditing."""

    name   = "fdf-atom-metadata"
    label  = "molbuilder ATOM-METADATA block"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        base = empty_script_result(cls.name)
        atom_md = _legacy_extract(text)
        schema_versions = {}
        if atom_md and isinstance(atom_md.get("schema_version"), int):
            schema_versions["atom-metadata"] = atom_md["schema_version"]
        return replace(
            base,
            atom_metadata=atom_md,
            block_schema_versions=schema_versions,
        )
