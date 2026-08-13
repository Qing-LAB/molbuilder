"""SIESTA's stage knowledge — the shipped ladder, and what a stage's
warm restart means (docs/execution/job-system.md).

Three things live here, all consumed by the engine seam
(`jobset/prep.py::_engine_seam`) or by `describe`:

  * :func:`default_siesta_stages` — the shipped ladder as ``Stage`` objects,
    the science stated once in the presets table;
  * :func:`_warm_declaration` — ``restart: continue`` means ``.XV``, ``.DM``
    and ``.CG``, the last conditioned on the optimizer (`run-identity.md`
    § 4);
  * :func:`_traits` — the opaque per-job facts a warm file is conditioned on.

**The producers that lived below these — ``stages_to_jobset``,
``StageBundle``, ``build_siesta_stage_bundle`` — were DELETED 2026-08-12
(plan step 6 u5).**  They had no production caller: the described route
renders per-element inside `jobset prep`, and floor 3 is derived from the
description, never emitted beside it (the one defect, closed at step 3).

**What went with the edges.**  ``on_nonconvergence`` and its
``DEFAULT_NONCONVERGENCE`` table: `engines/stages.md § 3` says *"its entire
effect is the edge between one attempt and the next"*, so with no edge it had
no effect -- and it was never reachable by a user anyway, being absent from
`task.json`'s three stage fields.  The derived ``Carry`` projection went too:
it existed only to render the declaration onto the one source a chain can know
in advance.

**Where the ladder comes from, and where it does not.**  An engine config
carries no stage list (``engines/stages.md`` § 1.1) — the ladder is the
user's decision about what varies, and it lives in the description.
Per-stage SCHEDULER resources are not stage fields either; they ride on
``Job.resources``, copied from each resolved element (`generator.md` § 5).
"""

from __future__ import annotations

from typing import Dict, List

from ..jobset.model import WarmFile
from ..task import Stage


def default_siesta_stages(strategy: str = "publishable") -> List[Stage]:
    """The shipped SIESTA ladder as ``Stage`` objects.

    One stage per tier of :data:`~molbuilder.config.siesta.SIESTA_STAGE_PRESETS`,
    each carrying that tier's four values as its ``overrides``, enabled
    according to the named strategy preset.  So the science is stated once,
    in the presets table, and this only decides *which* tiers run and *how*
    they are shaped for the description.

    Replaces the deleted ``_default_siesta_stages`` + ``apply_siesta_stage_strategy``
    pair: building the ladder and applying the enable-mask were never two
    steps, they were one function split across a mutable default.

    Raises ``ValueError`` on an unknown strategy.
    """
    from ..config.siesta import (SIESTA_STAGE_NAMES, SIESTA_STAGE_PRESETS,
                                 SIESTA_STAGE_STRATEGY_PRESETS)

    if strategy not in SIESTA_STAGE_STRATEGY_PRESETS:
        valid = ", ".join(sorted(SIESTA_STAGE_STRATEGY_PRESETS))
        raise ValueError(
            f"unknown SIESTA stage strategy {strategy!r}; "
            f"choose from: {valid}")
    enables = SIESTA_STAGE_STRATEGY_PRESETS[strategy]
    out: List[Stage] = []
    for i, tier in enumerate(sorted(SIESTA_STAGE_PRESETS)):
        overrides = dict(SIESTA_STAGE_PRESETS[tier])
        # § 4 rule 3: "A first stage is normally 'clean' and everything after
        # it 'continue'.  Nothing special is needed to say so."  Positional,
        # not tiered -- which is why it is set here and not in the presets
        # table, where every value is a property of that TIER's science.
        #
        # Keyed on POSITION IN THE TABLE, not on which stage is first
        # *enabled*.  Both readings are defensible and this one keeps an
        # invariant the ladder already had and the tests already assert: a
        # strategy preset says which tiers run and never retunes one.  Under
        # the other reading, `loose-only` (which disables stage2) and
        # `vib-quality` (which enables it) would hand stage2 different
        # overrides, so picking a strategy would silently change what a stage
        # computes.
        #
        # The case that reading protected -- stage1 off, so stage2 is first
        # and says 'continue' with nothing before it -- no shipped mask
        # produces, and it is not silent when it happens: the carry is empty
        # (the resolved ladder has no predecessor to take from) and § 5's
        # surface + wrapper banner both report the absence.
        #
        # Before P3 unit 4 the ladder carried no `restart` at all and
        # continued anyway, because the three use_save_* booleans defaulted
        # True -- continuation by accident rather than by description, which
        # is exactly what § 4 rule 2 exists to end.
        overrides["restart"] = "clean" if i == 0 else "continue"
        # The tier's NAME, from the one table -- not ``f"stage{tier}"``, which
        # this built until 2026-08-10.  Decision 27 puts the ordinal in the
        # artifact token, so a name that is itself a position says the number
        # twice (``<label>_01_stage1.fdf``) and the science none.
        out.append(Stage(name=SIESTA_STAGE_NAMES[tier],
                         enabled=bool(enables[i]) if i < len(enables) else False,
                         overrides=overrides))
    return out


#: The key a `.CG` carry is conditioned on.  Named once because two modules
#: must spell it identically for the comparison to mean anything, and a typo
#: on either side reads as *"the optimizers disagree"* -- which withholds the
#: file silently rather than failing.  ``JobSet.validate`` catches the
#: declaration half; this constant is why there is only one spelling to catch.
OPTIMIZER_TRAIT = "optimizer"


def _traits(eff) -> Dict[str, str]:
    """The opaque per-job facts a warm file can be conditioned on (§ 4 rule 1).

    One entry today: which relaxation algorithm this stage runs. The `jobset`
    layer compares it as a string and never learns what it means, which is what
    keeps SIESTA's restart group out of the engine-agnostic core.
    """
    # normalized at the ONE producer (R11, 2026-08-12): the jobset layer
    # compares this as a string, and "Broyden" vs "broyden" -- the same
    # optimizer, spelled by two hands -- compared unequal and silently
    # withheld the warm .CG the pair rule exists to carry
    return {OPTIMIZER_TRAIT:
            str(getattr(eff, "relax_type", "") or "").strip().lower()}


def _warm_declaration(label: str, eff,
                      calculation: str = "optimization") -> List[WarmFile]:
    """What a stage with this resolved config takes from a run it continues.

    THE ROWS COME FROM THE RULES FILE (`job-contracts.md` § 4.2a, U2
    2026-08-13): ``siesta/warm-files.toml``'s carry rows for this
    CALCULATION TYPE, gated -- as ever -- on the stage actually
    continuing: `run-identity.md` § 4 rule 2 makes ``restart`` the ONE
    field that says so, and ``_continues`` gates the deck's
    ``MD.UseSave*`` keywords from the same answer, so files placed for a
    ``clean`` stage would sit unread -- § 4's *"present but not
    honoured"*, the silent half of the pair.

    The pair conditions (``requires_same`` -- only `.CG` carries one
    today) are DECLARED here and EVALUATED by
    :func:`~molbuilder.jobset.model.warm_carry`, the one interpreter,
    once ``--from`` names the source and both stages are in hand.  The
    hard-coded XV/DM/CG list this replaces was one of the three copies
    § 4.2a's history records drifting.
    """
    from ..warmfiles import rules_for
    from .input import _continues
    if not _continues(eff):
        return []
    return [WarmFile(f"{label}{r.suffix}", requires_same=r.requires_same)
            for r in rules_for("siesta", calculation) if r.carry]


