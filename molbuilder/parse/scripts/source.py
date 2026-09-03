"""Umbrella ScriptSource TextParser.

Composes the 5 per-block
TextParsers (HeaderTextParser, ProvenanceTextParser,
BenchMarksTextParser, AtomMetadataTextParser, UserCustomTextParser)
into a single :class:`ScriptResult` carrying every block.

This is the canonical entry point for callers that want "everything
the script-contract reserved blocks have" — same role the legacy
``molbuilder.script_contract.extract_script_source`` played, but
typed.  Single-pass over the text body; per-block extractors are
short regex scans so the cost is dominated by I/O at the caller.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult

from .atom_metadata import AtomMetadataTextParser
from .bench_marks import BenchMarksTextParser
from .header import HeaderTextParser
from ._helpers import empty_script_result
from .provenance import ProvenanceTextParser
from .user_custom import UserCustomTextParser


class ScriptSourceTextParser(TextParser):
    """Parse all 6 script-contract reserved blocks (every block
    except ENGINE BODY, which isn't a 'block') from a single
    ``.fdf`` / ``.py`` text body.

    Returns :class:`ScriptResult` with every per-block field
    populated.  Each field is None when its block is absent; an
    empty dict / empty list when the block is present-but-empty
    (matching the legacy ``ScriptSource`` semantics — the
    distinction matters for the ATOM-METADATA emission rule --
    `execution/job-contracts.md` § 3.1 + § 3.4).
    """

    name   = "fdf-script-source"
    label  = "molbuilder fdf script-source umbrella"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        header_r       = HeaderTextParser.parse(text)
        provenance_r   = ProvenanceTextParser.parse(text)
        bench_marks_r  = BenchMarksTextParser.parse(text)
        atom_md_r      = AtomMetadataTextParser.parse(text)
        user_custom_r  = UserCustomTextParser.parse(text)
        # Merge per-block schema_versions into one map.
        block_schema_versions = {}
        block_schema_versions.update(atom_md_r.block_schema_versions)
        # bench-marks ships with its own version tag ('version v1').
        if bench_marks_r.bench_marks:
            v = bench_marks_r.bench_marks.get("version")
            if isinstance(v, str):
                block_schema_versions["bench-marks"] = v
        base = empty_script_result(cls.name)
        return replace(
            base,
            header        = header_r.header,
            provenance    = provenance_r.provenance,
            bench_marks   = bench_marks_r.bench_marks,
            atom_metadata = atom_md_r.atom_metadata,
            user_custom   = user_custom_r.user_custom,
            block_schema_versions = block_schema_versions,
        )
