"""WHAT EACH STAGE OF A LADDER BUILDS ON, BY DEFAULT -- the ladder's rule,
floor 3 (`execution/architecture.md` § 2.1; `job-system.md` § 5.4).

Which ladder a kind has is read off its roles in the kind table
(`execution/architecture.md` § 3.3).  A stage of an independent ladder
continues from the stage before it, as the description resolves it (an
override ``restart: clean`` starts it from the structure); a linked kind's
stage builds on what its kind says -- the inputs its pair names, taken from
the rungs upstream (a transport rung's), or nothing of another stage (a
SIESTA vibration's one stage is one run whose calls build on each other
inside the run, `engines/vibration.md` § 5.11).  Read off the description
and its template -- which RUN of that stage is taken is the layout's
(`continuation.continuation_answer`).
"""
from __future__ import annotations

from typing import Callable, Optional, Set


def inputs_of(task) -> Optional[Callable]:
    """``(task, stage) -> [(stage upstream, file)]`` -- which rung hands
    which file to which -- for a linked kind whose rungs take inputs: the
    part its pair names in the kind table (``inputs``), resolved here, the
    floor that asks; ``None`` for every other ladder."""
    from ..kind_table import part_of
    return part_of(str(task.calculation), str(task.engine), "inputs")


def upstream_of(task, stage: str,
                template_text: Optional[str] = None) -> Set[str]:
    """The stages ``stage`` builds on, as the description states it: the
    rungs its inputs come from (:func:`inputs_of`); an independent stage's
    stage before it (none for one set `restart: clean`, or the first);
    nothing for any other.  What a job's members may not do to each other
    (`group`) asks it."""
    inputs = inputs_of(task)
    if inputs is not None:
        return {src for src, _f in inputs(task, stage)}
    src = stage_before(task, stage, template_text)
    return {src} if src else set()


def independent(task) -> bool:
    """An INDEPENDENT ladder, whose stages continue one from another -- a
    kind without roles in the kind table (`job-system.md` § 5.4).  A linked
    stage, its kind's rungs named by role, builds on what its kind says:
    its inputs (:func:`inputs_of`), or nothing of another stage."""
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


__all__ = ["fresh_stages", "independent", "inputs_of", "resolved_stages",
           "stage_before", "stage_config", "upstream_of"]
