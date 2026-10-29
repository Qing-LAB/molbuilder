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
    the one rule, asked by `molbuilder describe` and by the hand-over the
    parameter tabs send (`web/handover-procedure.md`): the pair's ladder,
    each stage with no overrides (its record's ``ladder`` -- a transport
    calculation's rungs; a vibration's one stage, `engines/vibration.md`
    § 2.2, § 5.11); else the ordinary start -- ONE stage carrying no
    overrides, the calculation that is just the template
    (`engines/stages.md` § 6.5), named by the shared ladder vocabulary."""
    from ..config.stages import STAGE_NAMES
    from ..kind_table import pair
    from ..task import Stage
    own = pair(calculation, engine)
    if own is not None and own.ladder:
        return tuple(Stage(name=n, overrides={}) for n in own.ladder)
    return (Stage(name=STAGE_NAMES[1], overrides={}),)


def engine_tiers(engine: str) -> tuple:
    """EVERY TIER OF ``engine``'s LADDER, ``(tier, name, values)`` -- its
    stage presets (the engine record's ``presets``) named by the shared
    ladder vocabulary, in order: what a ladder's stage is filled from."""
    from ..config.stages import STAGE_NAMES
    from ..engine_table import stage_presets
    presets = stage_presets(engine)
    return tuple((tier, STAGE_NAMES[tier], dict(presets[tier]))
                 for tier in sorted(presets))


def engine_ladder(engine: str, strategy: str = "publishable") -> list:
    """``engine``'s SHIPPED LADDER as ``Stage`` objects -- the one ladder rule
    for every engine (`engines/stages.md` § 1.1a, § 2): one stage per tier of
    its presets (:func:`engine_tiers`), each carrying that tier's values as
    its ``overrides``, the tiers the named strategy picks
    (`config.stages.STAGE_STRATEGY_PRESETS`).  No ``restart`` among them: a
    rung's position does not answer *is there anything to continue from* --
    the folder does, at run time (`run-identity.md` § 4 rule 3).
    :class:`ValueError` names an unknown strategy and the ones there are."""
    from ..config.stages import STAGE_STRATEGY_PRESETS
    from ..engine_table import engine as engine_record
    from ..task import Stage
    if strategy not in STAGE_STRATEGY_PRESETS:
        raise ValueError(
            f"unknown {engine_record(engine).words} stage strategy "
            f"{strategy!r}; choose from: "
            f"{', '.join(sorted(STAGE_STRATEGY_PRESETS))}")
    enables = STAGE_STRATEGY_PRESETS[strategy]
    return [Stage(name=name, overrides=values)
            for i, (_tier, name, values) in enumerate(engine_tiers(engine))
            if i < len(enables) and enables[i]]


__all__ = ["engine_ladder", "engine_tiers", "fresh_stages", "independent", "inputs_of", "resolved_stages",
           "stage_before", "stage_config", "upstream_of"]
