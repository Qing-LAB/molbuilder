"""WHAT EACH STAGE OF A LADDER BUILDS ON, BY DEFAULT -- the ladder's rule,
floor 3 (`execution/architecture.md` § 2.1; `job-system.md` § 5.4).

A stage of an independent ladder continues from the stage before it, as the
description resolves it (a run card's ``restart: clean`` starts it from the
structure); a SIESTA vibration's force-constant stage builds on its
``relax`` (`engines/vibration.md` § 5.2a); a transport rung takes its kind's
gather.  Read off the description and its template -- which RUN of that
stage is taken is the layout's (`continuation.continuation_answer`).
"""
from __future__ import annotations

from typing import Optional, Tuple


def force_constant_stage(task, stage: str) -> bool:
    """A force-constant stage of a SIESTA vibration -- `freq`, or any stage
    after `relax` in a displacement sweep, asked of its role and never of a
    name (`pyscf.stages.vibration_render_kind`, `engines/vibration.md`
    § 5.2a).  A PySCF vibration's one rung relaxes inside its deck."""
    if (str(getattr(task, "engine", "")) != "siesta"
            or getattr(task, "calculation", None) != "vibration"):
        return False
    from ..pyscf.stages import vibration_render_kind
    return vibration_render_kind(stage) == "vibration"


def relax_stage_of(task) -> Optional[str]:
    """The ladder's relaxation rung, which every force-constant stage
    builds on, or ``None`` -- the structure is then measured as given, and
    must be stated relaxed (`engines/vibration.md` § 5.2a).  By the one role
    rule, the name in any case (plan § 5w K12)."""
    from ..pyscf.stages import vibration_render_kind
    return next((s.name for s in task.stages
                 if vibration_render_kind(s.name) != "vibration"), None)


def source_stage(task, stage: str, template_text: Optional[str] = None
                  ) -> Tuple[Optional[str], bool]:
    """``(the stage whose run ``stage`` builds on by default, linked)``: the
    stage before it in an independent ladder (:func:`stage_before`), or the
    `relax` a force-constant stage builds on (linked) -- ``(None, ...)``
    when it builds on none."""
    if force_constant_stage(task, stage):
        return relax_stage_of(task), True
    return stage_before(task, stage, template_text), False


def independent(task) -> bool:
    """An INDEPENDENT ladder, whose stages continue one from another: a kind
    without rung roles (`paths.KIND_ROLES`).  A linked stage's input is
    its kind's: a transport rung's gather (`gather.gather_sources`),
    and a force-constant stage builds on `relax` (:func:`source_stage`)."""
    from ..paths import KIND_ROLES
    return task.calculation not in KIND_ROLES


def resolved_stages(task, template_text: Optional[str] = None) -> list:
    """``[(stage, resolved config)]`` of the described stages, as `prep`
    resolves them (`resolve.resolved_ladder`: template ⊕ stage overrides ⊕
    the run card) -- from ``template_text`` as the verb read it (A16);
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
    stage overrides ⊕ the run card, so a card's ``restart: clean`` is read)
    -- or ``None``: a linked kind, the first stage, one that starts clean,
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


def fresh_stages(calculation: str, engine: str, *,
                 already_relaxed: bool) -> tuple:
    """THE LADDER A NEW DESCRIPTION STARTS WITH, before it is numbered --
    the one rule, asked by `jobset init` and by the hand-over the parameter
    tabs send (`web/handover-procedure.md`): a vibration's own ladder, read
    from the person's one statement (`pyscf/stages.py::vibration_stages`,
    `engines/vibration.md` § 2.2, § 5.2a); every other kind the ordinary
    start -- ONE stage carrying no overrides, the calculation that is just
    the template (`engines/stages.md` § 6.5), named by the shared ladder
    vocabulary.  A transport calculation proposes nothing here: its rungs
    are fixed by the kind (`task.fixed_by_kind`)."""
    if calculation == "vibration":
        from ..pyscf.stages import vibration_stages
        return tuple(vibration_stages(engine,
                                      already_relaxed=bool(already_relaxed)))
    from ..config.stages import STAGE_NAMES
    from ..task import Stage
    return (Stage(name=STAGE_NAMES[1], overrides={}),)


__all__ = ["force_constant_stage", "fresh_stages", "independent",
           "relax_stage_of", "resolved_stages", "source_stage",
           "stage_before", "stage_config"]
