"""ATOM-METADATA block TextParser.

H1 of parse-module.md migration (was Phase F wrapper around
``script_contract.extract_atom_metadata_dict``): absorbed the
extractor body directly so this module no longer imports from
``molbuilder.script_contract``.

ATOM-METADATA carries the regions + frozen_atoms (the
``.molstruct.json`` schema v3 payload embedded in the .fdf /
.py).  Inside the BEGIN/END markers each line is comment-prefixed
(``# `` or ``#``); the extractor strips the prefix and then walks
the brace-balance of the JSON to support both pretty-printed
multi-line and compact single-line payloads.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List, Optional

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult

from ._helpers import empty_script_result
from .markers import BLOCK_ATOM_METADATA, MARKER_RE


def _extract_atom_metadata_dict(text: str) -> Optional[Dict[str, Any]]:
    """Find the ATOM-METADATA block in ``text`` and return its JSON
    payload as a dict.

    Returns ``None`` when:
      * No ATOM-METADATA block is present.
      * The block markers are unbalanced.
      * The JSON between markers fails to parse.

    Comment-prefix-per-line is stripped before JSON parsing.
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1) != BLOCK_ATOM_METADATA:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    # Inner lines: strip leading "# " (or "#") to recover JSON.
    inner: List[str] = []
    for raw in lines[begin_idx + 1: end_idx]:
        if raw.startswith("# "):
            inner.append(raw[2:])
        elif raw.startswith("#"):
            inner.append(raw[1:])
        else:
            inner.append(raw)
    # Brace-balance walk so the extractor accepts BOTH pretty-printed
    # JSON (molbuilder's emit_atom_metadata via json.dumps indent=2)
    # AND compact / single-line JSON.  The contract on the wire is
    # "valid JSON inside the block"; how the writer formatted it isn't
    # load-bearing.
    json_lines: List[str] = []
    saw_open = False
    brace_depth = 0
    for line in inner:
        stripped = line.strip()
        if not saw_open:
            if not stripped or not stripped.startswith("{"):
                continue
            saw_open = True
        json_lines.append(line)
        brace_depth += stripped.count("{") - stripped.count("}")
        if brace_depth <= 0:
            break
    if not json_lines:
        return None
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None


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
