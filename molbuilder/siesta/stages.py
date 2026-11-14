"""SIESTA's stage knowledge — the facts its warm files are conditioned on
(docs/execution/job-system.md).

One thing lives here, read by the engine seam
(`jobset/engines.py::engine_seam`) -- the shipped ladder is every engine's
one rule over its presets (`jobset.ladder.engine_ladder`), and what a stage
takes from a run it continues every engine's one door over its warm-files
table (`jobset.model.warm_for`; ``.XV``, ``.DM`` and ``.CG``, the last
conditioned on the optimizer, `run-identity.md` § 4):

  * :func:`traits_for` — the opaque per-job facts a warm file is conditioned on.

**Where the ladder comes from, and where it does not.**  An engine config
carries no stage list (``engines/stages.md`` § 1.1) — the ladder is the
user's decision about what varies, and it lives in the description.
Per-stage SCHEDULER resources are not stage fields either; they ride on
``Job.resources``, copied from each resolved element (`generator.md` § 5).
"""

from __future__ import annotations

from typing import Dict


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

