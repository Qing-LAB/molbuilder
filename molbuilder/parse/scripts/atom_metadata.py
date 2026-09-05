"""ATOM-METADATA block TextParser.

Absorbed from the retired
``molbuilder.script_contract.extract_atom_metadata_dict``, deleted with that
module on 2026-06-21 (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

ATOM-METADATA carries the regions + frozen_atoms (the
``.molstruct.json`` schema v3 payload embedded in the .fdf /
.py).  Inside the BEGIN/END markers each line is comment-prefixed
(``# `` or ``#``); the extractor strips the prefix and then walks
the brace-balance of the JSON to support both pretty-printed
multi-line and compact single-line payloads.
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
    _extract_atom_metadata_dict,
)


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
        atom_md = _extract_atom_metadata_dict(text)
        schema_versions = {}
        if atom_md and isinstance(atom_md.get("schema_version"), int):
            schema_versions["atom-metadata"] = atom_md["schema_version"]
        return replace(
            base,
            atom_metadata=atom_md,
            block_schema_versions=schema_versions,
        )
