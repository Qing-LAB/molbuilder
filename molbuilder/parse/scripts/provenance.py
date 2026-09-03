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

import re
from dataclasses import replace
from typing import Dict, Optional

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult

from ._helpers import empty_script_result
from .markers import BLOCK_PROVENANCE, MARKER_RE


_PROVENANCE_KV_RE = re.compile(
    r"^#\s+(?P<key>[A-Za-z][A-Za-z0-9._-]*)\s{2,}(?P<val>.+?)\s*$")
_PROVENANCE_DEFAULTS_HDR = re.compile(
    r"^#\s+resolved-defaults\s*:\s*$")
_PROVENANCE_DEFAULTS_KV_RE = re.compile(
    r"^#\s{4,}(?P<key>[A-Za-z][A-Za-z0-9._-]*)\s{2,}(?P<val>.+?)\s*$")


def _extract_provenance_dict(text: str) -> Optional[Dict[str, str]]:
    """Find the PROVENANCE block and return its k/v payload as a flat
    dict.  Returns ``None`` when no well-formed PROVENANCE block is
    present.  Empty-but-present block returns ``{}`` (distinct from
    None — `model/parse.md`'s absent-vs-empty rule)."""
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1) != BLOCK_PROVENANCE:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    out: Dict[str, str] = {}
    in_defaults = False
    for raw in lines[begin_idx + 1: end_idx]:
        if _PROVENANCE_DEFAULTS_HDR.match(raw):
            in_defaults = True
            continue
        if in_defaults:
            dm = _PROVENANCE_DEFAULTS_KV_RE.match(raw)
            if dm:
                out[f"resolved-defaults.{dm.group('key')}"] = dm.group("val")
                continue
            # A non-defaults-shaped line ends the sub-block; the
            # top-level k/v matcher below may still pick it up.
            in_defaults = False
        m = _PROVENANCE_KV_RE.match(raw)
        if m:
            out[m.group("key")] = m.group("val")
    # Block present, no parseable k/v -> {} (NOT None).  Distinguished
    # from "block absent" -- `model/parse.md`'s None-vs-empty-dict
    # semantics.
    return out


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
