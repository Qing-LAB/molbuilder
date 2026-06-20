"""BENCH-MARKS block TextParser.

Phase F of parse-module.md migration: thin wrapper over the
legacy ``script_contract.extract_bench_marks_dict``.
BENCH-MARKS declares which fields in ENGINE BODY are safe for
bench tooling to override + their type/range/default.  Schema
``version v1`` per the script-contract doc.
"""

from __future__ import annotations

from dataclasses import replace

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult
from molbuilder.script_contract import (
    extract_bench_marks_dict as _legacy_extract,
)

from ._helpers import empty_script_result


class BenchMarksTextParser(TextParser):
    """Extract the ``BENCH-MARKS`` block.  Returns a
    :class:`ScriptResult` with only the ``bench_marks`` field set."""

    name   = "fdf-bench-marks"
    label  = "molbuilder BENCH-MARKS block"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        base = empty_script_result(cls.name)
        return replace(base, bench_marks=_legacy_extract(text))
