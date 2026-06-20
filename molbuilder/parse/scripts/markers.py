"""Shared block markers + regex for the script-contract reserved
blocks.

H2 of parse-module.md migration: re-exports the canonical
:data:`BLOCK_*` constants and :data:`MARKER_RE` from
:mod:`molbuilder.script_emit` so the read-side parsers stay in
lock-step with the write-side emitters.

The script-contract reserved blocks (HEADER / PROVENANCE /
BENCH-MARKS / ATOM-METADATA / USER-CUSTOM) are bracketed by literal
marker lines::

    # === molbuilder <block-name> BEGIN ===
    ... content ...
    # === molbuilder <block-name> END ===

:data:`MARKER_RE` matches either marker for any block.  Group 1:
block name (one of the :data:`BLOCK_*` constants); group 2: ``BEGIN``
or ``END``.
"""

from __future__ import annotations

from molbuilder.script_emit import (
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
