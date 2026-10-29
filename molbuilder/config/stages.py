"""The stage vocabulary both engines share (L1, no engine deps).

A strategy is not an engine's property -- it says WHICH TIERS RUN, which is
the same question for SIESTA and PySCF -- so it has one home.
"""
from __future__ import annotations

from typing import Dict, Tuple

#: THE TIERS' NAMES, engine-neutral (`engines/stages.md` § 2: a stage name is
#: the ladder's, not an engine's) -- read by the one ladder rule, by the
#: fresh one-stage start (`ladder.fresh_stages`) and by the Task setup page's
#: presets door.  It lived in `config/siesta.py` as `SIESTA_STAGE_NAMES`
#: until 2026-10-18, and PySCF's ladder imported it from there.
STAGE_NAMES = {1: "coarse", 2: "medium", 3: "tight"}

#: Which of the three tiers a named strategy runs.  Pure enable-mask data: it
#: says nothing about how a stage is TUNED, only which are in the ladder.
#: The one ladder rule (`jobset.ladder.engine_ladder`) reads it with an
#: engine's presets (the engine record's ``presets``) to build its ladder.
STAGE_STRATEGY_PRESETS: Dict[str, Tuple[bool, ...]] = {
    "publishable": (True,  True,  False),   # stage1 loose + stage2 publishable
    "loose-only":  (True,  False, False),   # stage1 only (cheap warm-up)
    "vib-quality": (True,  True,  True),    # all three (TIGHT for vib/IR/NEB)
}

__all__ = ["STAGE_NAMES", "STAGE_STRATEGY_PRESETS"]

