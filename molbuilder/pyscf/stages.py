"""PySCF's answers at the **declare** sub-step — `script-preparation.md` § 4.

Two of the seam's fifteen questions, and both are about the job rather than the
deck: *what may this run reuse from an earlier one*, and *what must the wrapper
route on*.  They live here beside `siesta/stages.py` for the same reason that
one does -- the engine knows, and the `jobset` layer must not.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # annotations only
    from ..task import Stage

from typing import Dict, List

from ..config.stages import VIBRATION_FREQ_STAGE
from ..jobset.model import WarmFile



def traits_for(eff) -> Dict[str, str]:
    """The per-job facts a warm file can be conditioned on -- none on PySCF.

    SIESTA conditions its ``.CG`` on the optimizer both rungs use
    (``requires_same = "optimizer"``).  PySCF has one optimizer, geomeTRIC
    (`engines/pyscf.md` § 3), and
    its rules file conditions nothing on a trait, so there is no fact to
    report.
    """
    return {}


def warm_for(label: str, eff,
             calculation: str = "optimization",
             base_dir=None) -> List[WarmFile]:
    """What a PySCF stage takes from a run it continues.

    **The rows come from the rules file**, ``pyscf/warm-files.toml``
    (`job-contracts.md` § 4.2a) -- the checkpoint for every calculation, plus
    the optimized geometry for an optimization.  Nothing is listed here that
    the file does not say.

    **Gated on the stage actually continuing**, exactly as SIESTA's twin is:
    `run-identity.md` § 4 rule 2 makes ``restart`` the one field that says so,
    and the same answer gates the generated script's checkpoint and
    optimized-geometry reads.  Placing files for a ``clean`` stage would leave
    them sitting unread -- § 4's *"present but not honoured"*, the silent half
    of the pair.
    """
    from ..identity import continues
    from ..jobset.model import warm_declaration
    if not continues(eff):
        return []
    return warm_declaration("pyscf", label, calculation, base_dir)


def default_pyscf_stages(strategy: str = "publishable") -> List["Stage"]:
    """The shipped PySCF ladder as ``Stage`` objects — SIESTA's shape exactly.

    **A ladder is declared the same way for both engines** (`stages.md`
    § 1.1a): a list of :class:`~molbuilder.task.Stage`, each carrying that
    rung's values as ``overrides`` on the shared schema.  The two engines
    differ in which parameters those overrides name and in nothing else --
    which is why this reads its per-tier science from a presets table and the
    tiers a strategy picks from a strategy table, the way
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
    from ..config.pyscf import PYSCF_STAGE_PRESETS
    from ..config.stages import STAGE_STRATEGY_PRESETS
    from ..config.stages import STAGE_NAMES
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
        # THE STAGES IT RUNS, and only those (`engines/stages.md` § 2): a
        # tier the strategy leaves out is not in the ladder.
        if i < len(enables) and enables[i]:
            out.append(Stage(name=STAGE_NAMES[tier],
                             overrides=overrides))
    return out


#: The engines a vibration runs on -- ONE home, read by the `molbuilder describe` refusal
#: (`engines/vibration.md` § 2.1): PySCF for the analytic Hessian and the
#: strengths, SIESTA for the force-constant run.
VIBRATION_ENGINES = ("pyscf", "siesta")


def vibration_stages(engine: str) -> List["Stage"]:
    """The vibration calculation's ladder: ONE stage, ``freq``, on both
    engines (`engines/vibration.md` § 2.2, § 5.11 § 1) -- PySCF's deck
    relaxes in-process (Phase 0) before the Hessian, and a SIESTA stage is
    one run in phases, its relaxation the run's first call, on when the
    structure is not stated relaxed (`already_relaxed`, the phases' switch,
    read at prep from the stage's config -- never from the ladder).  No
    tier overrides: the convergence settings are the template's own,
    recommended tight by the catalogue (`template.md` § 6.3a), and a
    person changes them there.
    """
    from ..task import Stage
    return [Stage(name=VIBRATION_FREQ_STAGE, overrides={})]
