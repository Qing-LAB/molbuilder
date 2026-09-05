"""BENCH-MARKS block TextParser.

Absorbed from the retired
``molbuilder.script_contract.extract_bench_marks_dict``, deleted with that
module on 2026-06-21 (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

BENCH-MARKS declares which fields in ENGINE BODY are safe for bench
tooling to override + their type/range/default.  Schema ``version
v1`` per the script-contract doc.  Payload shape::

    {
      "version":         "v1",
      "n_atoms":         212,            # top-level scalars
      "n_orbitals_est":  2700,
      "gpu_mode":        true,
      "numa_pin":        "socket-0",
      "fields": [
        {"name": "BlockSize", "anchor": "BlockSize", "type": "pow2",
         "range": [16, 256], "default": 256, ...},
        ...
      ]
    }
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
    _extract_bench_marks_dict,
)


class BenchMarksTextParser(TextParser):
    """Extract the ``BENCH-MARKS`` block.  Returns a
    :class:`ScriptResult` with only the ``bench_marks`` field set."""

    name   = "fdf-bench-marks"
    label  = "molbuilder BENCH-MARKS block"
    output = ScriptResult

    @classmethod
    def parse(cls, text: str) -> ScriptResult:
        base = empty_script_result(cls.name)
        return replace(base, bench_marks=_extract_bench_marks_dict(text))
