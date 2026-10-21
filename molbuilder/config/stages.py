"""The stage vocabulary both engines share (L1, no engine deps).

A strategy is not an engine's property -- it says WHICH TIERS RUN, which is
the same question for SIESTA and PySCF -- so it has one home.
"""
from __future__ import annotations

from typing import Dict, Tuple

#: Which of the three tiers a named strategy runs.  Pure enable-mask data: it
#: says nothing about how a stage is TUNED, only which are in the ladder.
#: Each engine's `default_*_stages` reads it with that engine's own
#: `*_STAGE_PRESETS` to build the shipped ladder.
#: THE TIERS' NAMES, engine-neutral (`engines/stages.md` § 2: a stage name is
#: the ladder's, not an engine's) -- read by both engines' ladders, by the
#: fresh one-stage start (`ladder.fresh_stages`) and by the Task setup page's
#: presets door.  It lived in `config/siesta.py` as `SIESTA_STAGE_NAMES`
#: until 2026-10-18, and PySCF's ladder imported it from there.
STAGE_NAMES = {1: "coarse", 2: "medium", 3: "tight"}

#: The vibration ladder's two stage names (`engines/vibration.md` § 2.2,
#: § 5.2a): the measurement, and the relaxation SIESTA runs before it when
#: the person has not stated the structure is relaxed.  Spelled here and
#: read everywhere else -- which deck a stage renders is asked of ``relax``
#: alone (`pyscf.stages.vibration_render_kind`), ``freq`` is the name the
#: proposed ladder gives its force-constant stage, and the Task setup tab
#: proposes both by the same words.
VIBRATION_FREQ_STAGE = "freq"
VIBRATION_RELAX_STAGE = "relax"

STAGE_STRATEGY_PRESETS: Dict[str, Tuple[bool, ...]] = {
    "publishable": (True,  True,  False),   # stage1 loose + stage2 publishable
    "loose-only":  (True,  False, False),   # stage1 only (cheap warm-up)
    "vib-quality": (True,  True,  True),    # all three (TIGHT for vib/IR/NEB)
}

__all__ = ["STAGE_NAMES", "STAGE_STRATEGY_PRESETS", "VIBRATION_FREQ_STAGE",
           "VIBRATION_RELAX_STAGE"]

