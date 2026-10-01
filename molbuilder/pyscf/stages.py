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

from ..jobset.model import WarmFile



def _traits(eff) -> Dict[str, str]:
    """The per-job facts a warm file can be conditioned on -- none on PySCF.

    SIESTA conditions its ``.CG`` on the optimizer both rungs use
    (``requires_same = "optimizer"``).  PySCF has one optimizer, geomeTRIC
    (`engines/pyscf.md` § 3, the ``optimizer`` item retired 2026-09-29), and
    its rules file conditions nothing on a trait, so there is no fact to
    report.
    """
    return {}


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


#: The engines a vibration runs on -- ONE home, read by the `init` refusal
#: (`engines/vibration.md` § 2.1): PySCF for the analytic Hessian and the
#: strengths, SIESTA for the force-constant run.
VIBRATION_ENGINES = ("pyscf", "siesta")

#: The vibration ladder's two stage names (`engines/vibration.md` § 2.2,
#: § 5.2a): the measurement, and the relaxation SIESTA runs before it when
#: the person has not stated the structure is relaxed.  Spelled here and
#: read everywhere else -- which deck a stage renders is asked of ``relax``
#: alone (:func:`vibration_render_kind`), ``freq`` is the name the proposed
#: ladder gives its force-constant stage, and the Task setup tab proposes
#: both by the same words.
VIBRATION_FREQ_STAGE = "freq"
VIBRATION_RELAX_STAGE = "relax"


def vibration_stages(engine: str, *, already_relaxed: bool) -> List["Stage"]:
    """The vibration calculation's ladder, read from the person's one
    statement (`engines/vibration.md` § 2.2, § 5.2a).

    ``freq`` alone when the structure is stated relaxed, and always on
    PySCF, whose deck relaxes in-process (Phase 0) before the Hessian;
    ``relax`` then ``freq`` on SIESTA otherwise, because a SIESTA run is one
    ``MD.TypeOfRun`` and cannot relax and take force constants in one go.
    No tier overrides on either rung: the convergence settings are the
    template's own, recommended tight by the catalogue (`template.md`
    § 6.3a), and a person changes them there.
    """
    from ..task import Stage
    out: List[Stage] = []
    if str(engine) == "siesta" and not already_relaxed:
        out.append(Stage(name=VIBRATION_RELAX_STAGE, enabled=True, overrides={}))
    out.append(Stage(name=VIBRATION_FREQ_STAGE, enabled=True, overrides={}))
    return out


def vibration_render_kind(stage_name: str) -> str:
    """Which deck a vibration stage renders (`engines/vibration.md` § 5.2a):
    the ``relax`` stage is the ordinary relaxation deck -- nothing is
    invented for it -- and every other rung is the kind's own.  Read off the
    rung's ROLE on the engine whose vibration is two programs, SIESTA
    (`template.stage_role`, the one rule, plan § 5w K4)."""
    from ..template import stage_role
    return ("optimization"
            if stage_role("siesta", "vibration", stage_name) == "relaxation"
            else "vibration")


def force_constant_stages(task, *, include_disabled: bool = False
                          ) -> List[str]:
    """The stages of a vibration description that render the kind's own
    deck, in ladder order -- every stage but ``relax``
    (:func:`vibration_render_kind`), WHATEVER ITS NAME; the enabled ones
    unless ``include_disabled``.  On SIESTA each is a force-constant run,
    and two or more are a displacement sweep (`engines/vibration.md` § 5.9);
    each measures at the relax stage's geometry (§ 5.2a).

    ``include_disabled`` is the layout's question: `prep` preps a stage
    named on its command line whether or not it is enabled (plan W38 F5), so
    whether two force-constant stages would share a directory is asked of
    every one described."""
    return [s.name for s in task.stages
            if (include_disabled or getattr(s, "enabled", True))
            and vibration_render_kind(s.name) == "vibration"]
