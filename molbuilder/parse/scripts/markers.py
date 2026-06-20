"""Re-export of the shared block markers + regex from the legacy
``molbuilder.script_contract`` module.

The script-contract reserved blocks (HEADER / PROVENANCE /
BENCH-MARKS / ATOM-METADATA / USER-CUSTOM) are bracketed by
literal marker lines:

    # === molbuilder <block-name> BEGIN ===
    ... content ...
    # === molbuilder <block-name> END ===

The marker regex and per-block name constants live in
``molbuilder.script_contract`` as the canonical source; Phase F
re-exports them so the new ``parse/scripts/*`` modules don't
have to depend on the legacy module path more than necessary.
Phase H absorbs the constants directly + drops the legacy
module.
"""

from molbuilder.script_contract import (
    BLOCK_ATOM_METADATA,
    BLOCK_BENCH_MARKS,
    BLOCK_HEADER,
    BLOCK_PROVENANCE,
    BLOCK_USER_CUSTOM,
    MARKER_RE,
)

__all__ = [
    "BLOCK_ATOM_METADATA",
    "BLOCK_BENCH_MARKS",
    "BLOCK_HEADER",
    "BLOCK_PROVENANCE",
    "BLOCK_USER_CUSTOM",
    "MARKER_RE",
]
