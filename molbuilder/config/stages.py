"""The stage vocabulary both engines share (L1, no engine deps).

`SIESTA_STAGE_STRATEGY_PRESETS` and `STAGE_STRATEGY_PRESETS` were separate
tables with equal values, kept in step by a pair of drift-guard tests that
asserted one constant equal to the other in both directions.  A strategy is
not an engine's property -- it says WHICH TIERS RUN, which is the same
question for SIESTA and PySCF -- so it has one home now and the guards have
nothing left to compare.
"""
from __future__ import annotations

from typing import Dict, Tuple

#: Which of the three tiers a named strategy runs.  Pure enable-mask data: it
#: says nothing about how a stage is TUNED, only which are in the ladder.
#: Each engine's `default_*_stages` reads it with that engine's own
#: `*_STAGE_PRESETS` to build the shipped ladder.
STAGE_STRATEGY_PRESETS: Dict[str, Tuple[bool, ...]] = {
    "publishable": (True,  True,  False),   # stage1 loose + stage2 publishable
    "loose-only":  (True,  False, False),   # stage1 only (cheap warm-up)
    "vib-quality": (True,  True,  True),    # all three (TIGHT for vib/IR/NEB)
}

__all__ = ["STAGE_STRATEGY_PRESETS"]
