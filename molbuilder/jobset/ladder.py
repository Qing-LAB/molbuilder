"""WHAT EACH STAGE OF A LADDER BUILDS ON, BY DEFAULT -- the ladder's rule,
floor 3 (`execution/architecture.md` § 2.1; `job-system.md` § 5.4).

A stage of an independent ladder continues from the stage before it, as the
description resolves it (an override ``restart: clean`` starts it from the
structure); a linked kind's stage builds on what its kind says -- a
transport rung takes its kind's gather, a SIESTA vibration's one stage is
one run whose calls build on each other inside the run
(`engines/vibration.md` § 5.11).  Read off the description and its template
-- which RUN of that stage is taken is the layout's
(`continuation.continuation_answer`).
"""
from __future__ import annotations

from typing import Optional


def source_stage(task, stage: str, template_text: Optional[str] = None
                  ) -> Optional[str]:
    """The stage whose run ``stage`` builds on by default: the stage before
    it in an independent ladder (:func:`stage_before`) -- ``None`` when it
    builds on none, a linked kind's stage among them."""
    return stage_before(task, stage, template_text)


def independent(task) -> bool:
    """An INDEPENDENT ladder, whose stages continue one from another: a kind
    without rung roles (the kind table).  A linked stage's input is
    its kind's: a transport rung's gather (`gather.gather_sources`); a
    SIESTA vibration's one run takes nothing from another stage."""
    from ..kind_table import kind
    return not kind(task.calculation).roles


def resolved_stages(task, template_text: Optional[str] = None) -> list:
    """``[(stage, resolved config)]`` of the described stages, as `prep`
    resolves them (`resolve.resolved_ladder`: template ⊕ stage overrides)
    -- from ``template_text`` as the verb read it (A16);
    ``[]`` with none."""
    from ..resolve import resolved_ladder
    from .engines import engine_seam
    if template_text is None:
        return []
    return resolved_ladder(template_text, task,
                           engine_seam(str(task.engine)).config_cls)


def stage_config(task, stage: str, template_text: Optional[str] = None):
    """``stage``'s resolved config (:func:`resolved_stages`), or ``None``."""
    from ..identity import stage_key
    return next((c for n, c in resolved_stages(task, template_text)
                 if stage_key(n) == stage_key(stage)), None)


def stage_before(task, stage: str,
                  template_text: Optional[str] = None) -> Optional[str]:
    """The stage ``stage`` continues from by default -- the one
    before it in the ladder, resolved as `prep` resolves it (template ⊕
    stage overrides, so an override's ``restart: clean`` is read) -- or
    ``None``: a linked kind, the first stage, one that starts clean,
    one the description does not hold (which prep refuses at its checkpoint
    2)."""
    from ..identity import continues
    if not independent(task):
        return None
    ladder = resolved_stages(task, template_text)
    names = [n for n, _c in ladder]
    if stage not in names:
        return None
    i = names.index(stage)
    if i == 0 or not continues(ladder[i][1]):
        return None
    return names[i - 1]


def fresh_stages(calculation: str, engine: str) -> tuple:
    """THE LADDER A NEW DESCRIPTION STARTS WITH, before it is numbered --
    the one rule, asked by `molbuilder describe` and by the hand-over the parameter
    tabs send (`web/handover-procedure.md`): a vibration's own one stage
    (`pyscf/stages.py::vibration_stages`, `engines/vibration.md` § 2.2,
    § 5.11); every other kind the ordinary
    start -- ONE stage carrying no overrides, the calculation that is just
    the template (`engines/stages.md` § 6.5), named by the shared ladder
    vocabulary.  A transport calculation proposes nothing here: its rungs
    are fixed by the kind (its part in the kind table)."""
    if calculation == "vibration":
        from ..pyscf.stages import vibration_stages
        return tuple(vibration_stages(engine))
    from ..config.stages import STAGE_NAMES
    from ..task import Stage
    return (Stage(name=STAGE_NAMES[1], overrides={}),)


__all__ = ["fresh_stages", "independent", "resolved_stages", "source_stage",
           "stage_before", "stage_config"]
