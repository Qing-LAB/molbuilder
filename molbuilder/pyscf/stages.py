"""PySCF's answers at the **declare** sub-step — `script-preparation.md` § 4.

Two of the seam's fifteen questions, and both are about the job rather than the
deck: *what may this run reuse from an earlier one*, and *what must the wrapper
route on*.  They live here beside `siesta/stages.py` for the same reason that
one does -- the engine knows, and the `jobset` layer must not.
"""
from __future__ import annotations

from typing import Dict, List

from ..jobset.model import WarmFile

#: The optimizer a stage runs, as an opaque string the `jobset` layer compares
#: and never interprets -- imported from the one place it is spelled, because
#: the comparison is engine-agnostic and two spellings of one idea is the drift
#: the warm-file pair rule exists to survive.
from ..jobset.model import OPTIMIZER_TRAIT


def _traits(eff) -> Dict[str, str]:
    """The per-job facts a warm file can be conditioned on.

    Normalised to lower case at this one producer: the pair rule compares
    strings, and ``geomeTRIC`` against ``geometric`` -- the same optimizer
    spelled by two hands -- would compare unequal and silently withhold a
    geometry the rule exists to carry.  That is the exact defect SIESTA's
    ``Broyden`` / ``broyden`` pair hit, and it is cheaper to not repeat it.
    """
    return {OPTIMIZER_TRAIT:
            str(getattr(eff, "optimizer", "") or "").strip().lower()}


def _warm_declaration(label: str, eff,
                      calculation: str = "optimization",
                      base_dir=None) -> List[WarmFile]:
    """What a PySCF stage takes from a run it continues.

    **The rows come from the rules file**, ``pyscf/warm-files.toml``
    (`job-contracts.md` § 4.2a) -- the checkpoint for every calculation, plus
    geomeTRIC's geometry and trajectory files for an optimization.  Nothing is
    listed here that the file does not say.

    **Gated on the stage actually continuing**, exactly as SIESTA's twin is:
    `run-identity.md` § 4 rule 2 makes ``restart`` the one field that says so,
    and the same answer gates the generated script's checkpoint and
    optimized-geometry reads.  Placing files for a ``clean`` stage would leave
    them sitting unread -- § 4's *"present but not honoured"*, the silent half
    of the pair.
    """
    from ..identity import continues
    from ..warmfiles import rules_for
    if not continues(eff):
        return []
    return [WarmFile(f"{label}{r.suffix}", requires_same=r.requires_same)
            for r in rules_for("pyscf", calculation, base_dir) if r.carry]


def default_pyscf_stages(strategy: str = "publishable") -> List["Stage"]:
    """The shipped PySCF ladder as ``Stage`` objects — SIESTA's shape exactly.

    **A ladder is declared the same way for both engines** (`stages.md`
    § 1.1a): a list of :class:`~molbuilder.task.Stage`, each carrying that
    rung's values as ``overrides`` on the shared schema.  The two engines
    differ in which parameters those overrides name and in nothing else --
    which is why this reads its per-tier science from a presets table and its
    enable-mask from a strategy table, the way
    ``siesta/stages.py::default_siesta_stages`` does, rather than carrying
    either inline.

    **The stage names are the shared ones** -- ``coarse`` / ``medium`` /
    ``tight``, which `tuning.md` § 4 fixes as the ladder's vocabulary.  They
    are not the quality TIER names (screening / loose preopt / publishable /
    tight, § 2): a tier is how good a number is, a stage is a rung of this
    particular ladder, and naming a rung after a tier would tie the two
    together for one engine only.

    Raises ``ValueError`` on an unknown strategy.  Falling back to the default
    mask instead would answer a question nobody asked -- and silently, since a
    misspelled strategy would run a ladder the caller did not name.
    """
    from ..config.pyscf import PYSCF_STAGE_PRESETS, STAGE_STRATEGY_PRESETS
    from ..config.siesta import SIESTA_STAGE_NAMES
    from ..task import Stage

    if strategy not in STAGE_STRATEGY_PRESETS:
        valid = ", ".join(sorted(STAGE_STRATEGY_PRESETS))
        raise ValueError(
            f"unknown PySCF stage strategy {strategy!r}; "
            f"choose from: {valid}")
    enables = STAGE_STRATEGY_PRESETS[strategy]
    out: List[Stage] = []
    for i, tier in enumerate(sorted(PYSCF_STAGE_PRESETS)):
        overrides = dict(PYSCF_STAGE_PRESETS[tier])
        # NO ``restart`` HERE -- SIESTA's twin says why, and it is the same
        # reason: a rung's POSITION does not answer *is there anything to
        # continue from*.  The folder answers that, at run time.
        out.append(Stage(name=SIESTA_STAGE_NAMES[tier],
                         enabled=bool(enables[i]) if i < len(enables) else False,
                         overrides=overrides))
    return out
