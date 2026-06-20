"""BENCH-MARKS block TextParser.

H1 of parse-module.md migration (was Phase F wrapper around
``script_contract.extract_bench_marks_dict``): absorbed the
extractor body directly so this module no longer imports from
``molbuilder.script_contract``.

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
from typing import Any, Dict, Optional

from molbuilder.parse.base import TextParser
from molbuilder.parse.types import ScriptResult

from ._helpers import empty_script_result
from .markers import BLOCK_BENCH_MARKS, MARKER_RE


def _coerce_scalar(s: str) -> Any:
    """Best-effort numeric coercion for BENCH-MARKS scalar values.
    Returns the original string when neither int nor float parses."""
    s = s.strip()
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def _extract_bench_marks_dict(text: str) -> Optional[Dict[str, Any]]:
    """Find the BENCH-MARKS block and return its structured payload.

    Returns ``None`` when no BENCH-MARKS block is present.  See the
    module docstring for the payload shape.
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m or m.group(1) != BLOCK_BENCH_MARKS:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    out: Dict[str, Any] = {"fields": []}
    for raw in lines[begin_idx + 1: end_idx]:
        # Strip "#   " comment prefix.
        s = raw.lstrip("#").strip()
        if not s:
            continue
        # `field <name> anchor=<x> type=<y> ...` rows.
        if s.startswith("field "):
            tokens = s.split()
            field: Dict[str, Any] = {"name": tokens[1]}
            for tok in tokens[2:]:
                if "=" not in tok:
                    continue
                k, _, v = tok.partition("=")
                v = v.strip()
                # Coerce numeric range / default values.
                if k == "range" and v.startswith("[") and v.endswith("]"):
                    try:
                        a, b = v[1:-1].split(",")
                        field["range"] = [_coerce_scalar(a), _coerce_scalar(b)]
                    except ValueError:
                        field["range"] = v
                elif k == "default":
                    field["default"] = _coerce_scalar(v)
                else:
                    field[k] = v
            out["fields"].append(field)
            continue
        # Top-level `key value` scalars.
        if " " in s and not s.startswith("field "):
            k, _, v = s.partition(" ")
            v = v.strip()
            if k == "version":
                out["version"] = v
            elif v.lower() in ("true", ".true."):
                out[k] = True
            elif v.lower() in ("false", ".false."):
                out[k] = False
            else:
                out[k] = _coerce_scalar(v)
    return out


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
