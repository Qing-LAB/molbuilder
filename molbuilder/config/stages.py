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


def refuse_values_outside_choices(cfg) -> None:
    """Every field that DECLARES its legal values must hold one of them.

    Eight fields on each engine config carry `metadata={"choices": (...)}` --
    the form renders them as a dropdown, the catalogue documents them, and
    until 2026-09-10 nothing checked them.  `SiestaConfig(spin_treatment=
    "banana")`, `relax_type="zzz"` and `PySCFConfig(restart="banana")` were
    all accepted, and the wrong word went straight into a deck.

    This repo runs no type checker, so the declaration alone refuses nothing;
    the pairing that works is `issues.Issue`'s -- a declared vocabulary plus a
    constructor that raises.  `None` passes: an optional field's absence is
    not a value outside its choices.
    """
    import dataclasses as _dc
    for f in _dc.fields(cfg):
        choices = f.metadata.get("choices")
        if not choices:
            continue
        value = getattr(cfg, f.name)
        if value is None or value in choices:
            continue
        raise ValueError(
            f"{type(cfg).__name__}.{f.name} must be one of "
            f"{tuple(choices)}; got {value!r}")
