"""SIESTA's stage knowledge — the shipped ladder, and what a stage's
warm restart means (docs/execution/job-system.md).

Two things live here, read by the engine seam
(`jobset/engines.py::engine_seam`) -- the shipped ladder is every engine's
one rule over its presets (`jobset.ladder.engine_ladder`):

  * :func:`warm_for` — ``restart: continue`` means ``.XV``, ``.DM``
    and ``.CG``, the last conditioned on the optimizer (`run-identity.md`
    § 4);
  * :func:`traits_for` — the opaque per-job facts a warm file is conditioned on.

**Where the ladder comes from, and where it does not.**  An engine config
carries no stage list (``engines/stages.md`` § 1.1) — the ladder is the
user's decision about what varies, and it lives in the description.
Per-stage SCHEDULER resources are not stage fields either; they ride on
``Job.resources``, copied from each resolved element (`generator.md` § 5).
"""

from __future__ import annotations

from typing import Dict, List

from ..jobset.model import WarmFile


#: The key a `.CG` carry is conditioned on -- re-exported from the one place
#: it is spelled (`jobset/model.py`), because the comparison it feeds is
#: engine-agnostic and both engines' producers must hand in the same string.
from ..jobset.model import OPTIMIZER_TRAIT


def traits_for(eff) -> Dict[str, str]:
    """The opaque per-job facts a warm file can be conditioned on (§ 4 rule 1).

    One entry today: which relaxation algorithm this stage runs. The `jobset`
    layer compares it as a string and never learns what it means, which is what
    keeps SIESTA's restart group out of the engine-agnostic core.
    """
    # normalized at the ONE producer: the jobset layer compares this as a
    # string, so "Broyden" and "broyden" -- the same optimizer -- must
    # compare equal, or the warm .CG the pair rule carries is withheld
    return {OPTIMIZER_TRAIT:
            str(getattr(eff, "relax_type", "") or "").strip().lower()}


def warm_for(label: str, eff,
             calculation: str = "optimization",
             base_dir=None) -> List[WarmFile]:
    """What a stage with this resolved config takes from a run it continues.

    THE ROWS COME FROM THE RULES FILE (`job-contracts.md` § 4.2a, U2
    2026-08-13): ``siesta/warm-files.toml``'s carry rows for this
    CALCULATION TYPE, gated -- as ever -- on the stage actually
    continuing: `run-identity.md` § 4 rule 2 makes ``restart`` the ONE
    field that says so, and ``identity.continues`` gates the deck's
    ``MD.UseSave*`` keywords from the same answer, so files placed for a
    ``clean`` stage would sit unread -- § 4's *"present but not
    honoured"*, the silent half of the pair.

    The pair conditions (``requires_same`` -- only `.CG` carries one
    today) are DECLARED here and EVALUATED by
    :func:`~molbuilder.jobset.model.warm_carry`, the one interpreter,
    once ``--from`` names the source and both stages are in hand.
    """
    from ..identity import continues
    from ..jobset.model import warm_declaration
    if not continues(eff):
        return []
    return warm_declaration("siesta", label, calculation, base_dir)


