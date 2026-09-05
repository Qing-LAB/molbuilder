"""``<base>-runN.scf-timing.log`` — seconds per SCF iteration.

The wrapper tees every ``scf:`` line into this file with an epoch stamp
in front (`running-a-job.md` § 4.1), so consecutive deltas ARE the
per-iteration durations.  Nothing else in the run states that number.

*(This logic lived in `bench/result.py::parse_scf_timing` until
2026-09-04, where it opened the file and read bytes directly.  It moves
here for `parse.md` § 5c's reason: being the wrapper's output rather
than the engine's is not a reason to read it a different way.)*
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import InstrumentResult

from ._helpers import build_instrument_result


def scf_timing_metrics(text: str) -> Dict[str, Any]:
    """``{"s_per_iter": float|None, "iters_measured": int}``.

    The FIRST delta is dropped when two or more are present: iteration
    1->2 can still carry warm-up.  Under the capped 3-iteration trial
    (2026-08-21) that leaves ONE clean sample -- iteration 3 -- which the
    user's measured experience says reads the scaling story as well as
    the older 5-iteration mean.  Records of either shape parse the same.
    """
    epochs: List[float] = []
    for line in text.splitlines():
        tok = line.split(None, 1)[0] if line.strip() else ""
        try:
            v = float(tok)
        except ValueError:
            continue
        if math.isfinite(v):                     # reject nan/inf tokens
            epochs.append(v)
    if len(epochs) < 2:
        return {"s_per_iter": None, "iters_measured": 0}
    # Only forward (positive) deltas are real iteration durations.
    deltas = [d for d in (epochs[i] - epochs[i - 1]
                          for i in range(1, len(epochs))) if d > 0]
    if not deltas:
        return {"s_per_iter": None, "iters_measured": 0}
    measured = deltas[1:] if len(deltas) >= 2 else deltas
    return {"s_per_iter": round(sum(measured) / len(measured), 1),
            "iters_measured": len(measured)}


class ScfTimingFileParser(FileParser):
    """The wrapper's SCF-timing tee."""

    name   = "scf-timing"
    label  = "wrapper SCF-timing log (.scf-timing.log)"
    hint   = "files ending in .scf-timing.log"
    output = InstrumentResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".scf-timing.log") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> InstrumentResult:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return build_instrument_result(metrics=scf_timing_metrics(text),
                                       parser_name=cls.name, source=path)
