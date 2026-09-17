"""PySCFConfig -- every parameter the PySCF script generator emits.

L1 dataclass.  Field metadata (label / unit / range / tier / help)
drives the CLI option list, the web form schema, and the validation
pass at ``molbuilder/validation.py``; the PySCF generator at
``molbuilder/pyscf/input.py:render_script`` is the only consumer of
the configured values themselves.

Defaults are tuned for "build a small/medium molecule and relax it":

    * B3LYP+D3BJ/def2-SVP  (modern hybrid, dispersion-corrected)
    * Density fitting on -- bare ``mf.density_fit()``, so PySCF auto-picks
      the *basis-matched* JK-fit set (``def2-svp-jkfit`` for this def2-SVP
      default, ``def2-tzvp-jkfit`` for def2-TZVP, ...).  NOT the single
      "def2-universal-jkfit" this docstring used to claim -- verified via
      ``mf.with_df.auxbasis`` on a real def2 hybrid.
    * geomeTRIC optimizer with maxsteps=200, grms=3e-4 Ha/Bohr
    * Closed-shell RKS (spin=0); change to UKS for radicals
    * NetCharge auto-detected from phosphate protonation state
    * Pre-optimization stage off by default; opt-in for systems
      where the builder geometry is rough (long ssDNA, large
      peptides) so PBE/def2-SVP can clean it up before B3LYP runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..identity import RestartGroup
from .siesta import _validate_basename     # shared with SiestaConfig


# --------------------------------------------------------------------- #
#  The ladder's per-tier science                                        #
# --------------------------------------------------------------------- #

#: What each rung of a PySCF ladder is tuned to, tier by tier.
#:
#: Read across from ``SIESTA_STAGE_PRESETS``: one table per engine, keyed by
#: the same three tiers, and ``<engine>/stages.py::default_<engine>_stages``
#: turns it into the shipped ladder of :class:`~molbuilder.task.Stage`
#: objects.  The two engines differ in which parameters a tier names and in
#: nothing else (`stages.md` § 1.1a).
#:
#: **Every number is `tuning.md` § 2.4's and § 2.5's**, column by column --
#: loose preopt, publishable, tight.  That table is what a reviewer is
#: pointed at, so it is what this states; a value here that disagrees with it
#: is a bug here rather than a second opinion.
#:
#: The keys are catalogue items, which is what makes them legal ``overrides``
#: on a stage (`stages.md` § 2).  ``restart`` is NOT among them: it follows
#: from a rung's POSITION rather than its tier (`run-identity.md` § 4 rule
#: 3), so ``default_pyscf_stages`` sets it -- exactly where SIESTA's twin
#: does.
#:
#: Units: ``geom_gmax`` / ``geom_grms`` in Ha/Bohr; ``geom_dmax`` /
#: ``geom_drms`` in Angstrom -- NOT Bohr, a long-standing geomeTRIC doc bug
#: whose source uses Angstrom; ``geom_etol`` and ``scf_conv_tol`` in Hartree.
PYSCF_STAGE_PRESETS: Dict[int, Dict[str, Any]] = {
    1: {   # loose preopt
        "scf_conv_tol":   1.0e-7,
        "geom_gmax":      2.0e-3,
        "geom_grms":      1.3e-3,
        "geom_dmax":      7.2e-3,
        "geom_drms":      4.8e-3,
        "geom_etol":      1.0e-5,
        "geom_max_steps": 50,
    },
    2: {   # publishable -- geomeTRIC's GAU preset
        "scf_conv_tol":   1.0e-9,
        "geom_gmax":      4.5e-4,
        "geom_grms":      3.0e-4,
        "geom_dmax":      1.8e-3,
        "geom_drms":      1.2e-3,
        "geom_etol":      1.0e-6,
        "geom_max_steps": 200,
    },
    3: {   # tight
        "scf_conv_tol":   1.0e-10,
        "geom_gmax":      2.0e-4,
        "geom_grms":      1.0e-4,
        "geom_dmax":      1.0e-3,
        "geom_drms":      5.0e-4,
        "geom_etol":      1.0e-6,
        "geom_max_steps": 100,
    },
}


#: § 4 rule 1 — PySCF's identity group.
#:
#: Read across from ``SIESTA_RESTART_GROUP`` and the contract's point is
#: visible: the identity is the same *idea* in both engines and a different
#: *mechanism*. SIESTA declares three keys; PySCF carries generated control
#: flow, so ``keys`` is empty and ``mechanism`` says what actually happens.
#: An empty tuple alone would read as "nothing is bound", which is the
#: opposite of true.
#:
#:
#: **Rule 2 is answered by the ``restart`` field below**, and by the same one
#: field SIESTA answers it with: two values, ``clean`` and ``continue``, and
#: a rerun that says ``clean`` writes its checkpoint without reading the one
#: already beside it. Before that field existed the resume branches were
#: gated on ``chkfile`` and ``save_optimized_xyz`` -- *write* flags doubling
#: as read gates -- so *"write a checkpoint but do not resume from one"* was
#: a sentence this engine could not say.
PYSCF_RESTART_GROUP = RestartGroup(
    literal="JOB",
    keys=(),
    mechanism="generated control flow: mf.chkfile + init_guess='chkfile' "
              "when the file exists, and <JOB>_optimized.xyz overriding the "
              "literal geometry",
    field="job_name",
)


from .stages import STAGE_STRATEGY_PRESETS   # THE shared table


@dataclass
class PySCFConfig:
    # Explicit form-section order for the schema-driven Build form.
    # PySCF runs in a natural reading order (system -> method -> SCF
    # -> opt -> solvent -> runtime -> post-relax analysis); the
    # dataclass field declaration order mostly matches but a few
    # field groups are out of place (Solvent declared next to Method,
    # Pre-opt declared between SCF and Optimization).  Pinning the
    # order here keeps the schema independent of those declaration
    # quirks.
    # 2026-06-15 restructure: merged "Optimization" + "Runtime & output"
    # into a single "Compute & budget" section, mirroring the SIESTA
    # form's same-day restructure.  Reasoning: both sections covered
    # "how the run proceeds" -- the optimization algorithm + its
    # convergence targets on one hand, the CPU/GPU compute budget +
    # I/O knobs on the other.  Keeping them split forced the user to
    # scroll past unrelated cards (Solvent, Frequencies) between two
    # semantically connected groups.  Merging keeps the physics axis
    # (System -> Method -> SCF -> Solvent -> Frequencies) compact and
    # gathers all the "execution strategy + resources" knobs in one
    # section at the end.  Workflow-group cards inside the new
    # section split the merged fields cleanly:
    #   * Profile card -- optimize toggle + optimizer choice
    #   * Stage card   -- THIS rung's convergence knobs (the fields
    #                     marked ``workflow_group = "stage"``).  The ladder
    #                     itself is not here: it is declared in task.json
    #                     and each rung is its own deck (`stages.md` § 1.1a)
    #   * Budget card  -- max_memory_mb + threads + use_gpu + verbose
    #                     + chkfile + log_file + verbose_comments
    _form_section_order = (
        "System",
        "Method",
        "SCF",
        "Solvent (optional)",
        "Frequencies / thermochemistry",
        "Compute & budget",
    )

    # ---------------- System ----------------
    job_name: str = field(default="pyscf_relax", metadata={
        "category": ("system", "procedure"),
        "workflow_group": "setup",
        "section":  "System",
        "item_kind":  "produce",
        "label":    "Job name",
        "engine_key":  '(molbuilder: filename + log-name basename)',
        "id_suffix": "job-name",
        "pattern":  r"^[A-Za-z0-9_\-]+$",
        "validate": _validate_basename("job_name"),
    })
    # RENAMED from ``charge`` 2026-08-19, when the catalogue merged this with
    # SIESTA's ``net_charge``: one question, one name.  ``net_charge`` is the
    # survivor because ``charge`` is overloaded in this codebase -- atomic
    # partial charges, formal charges, MolView's per-atom charge -- and reusing
    # it for the whole system's charge invites exactly the fusing-things-that-
    # sound-alike risk `template.md` § 6.3's merge gate exists to catch.
    net_charge: Optional[int] = field(default=None, metadata={
        "category": ("system",),
        "section": "System",
        # Run-profile identity — molecule's charge state.
        "workflow_group": "profile",
        "label":   "Net charge",
        "engine_key":  "NetCharge (SIESTA) | gto.M(charge=...) (PySCF)",
        "item_kind": "deck",
        "expands": ("NetCharge", "gto.M"),
        "null_label": "(auto-detect from phosphates)",
        "range": (-10, 10),
    })
    spin: int = field(default=0, metadata={
        "category": ("system",),
        "section": "System",
        # System characteristic — open-shell chemistry, not stage.
        "workflow_group": "profile",
        "label":   "Spin (2S)",
        "engine_key":  'gto.M(spin=...)  # 2S, # of unpaired electrons',
        "range":   (0, 10),
    })
    symmetry: bool = field(default=False, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "section": "System",
        "label":   "Use point-group symmetry",
        "engine_key":  'gto.M(symmetry=...)',
    })

    # ---------------- Engine ----------------
    #
    # NAMED, not implied.  Until now "which program runs this" was a
    # string literal in the browser (`engine: "pyscf"` in the tab's send
    # payload), which made a real choice look like a constant and gave a
    # second engine nowhere to land.  Declaring it here puts it on the
    # form, into the task payload, and into the results sidecar through
    # the same door every other parameter uses.
    #
    # One choice today.  That is the point: a single-option selector
    # says "this is a choice that has one answer right now", where a
    # hidden constant said "there is no choice".  A second engine adds
    # its own config class declaring its own identity -- SIESTA's
    # spectra path would carry engine="siesta" -- so this field is each
    # engine's statement of what it is, not a global registry.
    engine: str = field(default="pyscf", metadata={
        "category": ("method",),
        "section": "Engine",
        "workflow_group": "profile",
        "label":   "Calculation engine",
        "tier":    "basic",
        "choices": ("pyscf",),
        "engine_key":  '(molbuilder: selects the deck composer + the backend env)',
        # `produce`, and the contract forces the choice (template.md 6.0): an
        # `engine_key` that is a molbuilder NOTE derives no anchor, so the
        # default kind `engine` is refused -- "this item is not an engine
        # keyword, so tell me what it is."  It is not `deck`: it writes no
        # keyword and expands to none.  What it does is decide WHICH producer
        # writes the script, which is 6's definition of `produce` -- shaping
        # how the script is written without becoming a keyword itself.  Its
        # siblings here agree: `job_name`, `optimize` and `on_nonconvergence`
        # are all molbuilder-level selectors marked `produce`.
        #
        # Omitted when this field landed (3aaec645, 2026-09-11), which left
        # four tests failing on main.
        "item_kind": "produce",
    })

    # ---------------- Method (main run) ----------------
    method: str = field(default="RKS", metadata={
        "category": ("method",),
        "section": "Method",
        "workflow_group": "profile",
        "label":   "SCF method",
        "engine_key":  'RKS / UKS / RHF / UHF  (PySCF class selection)',
        "choices": ("RKS", "UKS", "RHF", "UHF"),
    })
    functional: str = field(default="B3LYP", metadata={
        "category": ("method",),
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Functional",
        "engine_key":  'mf.xc = ...',
    })
    basis: str = field(default="def2-SVP", metadata={
        "category": ("method", "accuracy"),
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Basis set",
        "engine_key":  'gto.M(basis=...)',
    })
    # auxbasis: Python-API knob; rarely set from the form (auto-pick
    # from density_fit() is the right default).  No section -> not on form.
    auxbasis: Optional[str] = field(default=None, metadata={
        "workflow_group": "profile",
        "category": ("method",),
            "engine_key":  'mf = mf.density_fit(auxbasis=...)',
    })
    density_fit: bool = field(default=True, metadata={
        "category": ("method", "execution"),
        "section": "Method",
        # Profile-level: method-family identity choice; the vibration
        # deck's density_fit is also profile.
        "workflow_group": "profile",
        "label":   "Density fitting",
        "engine_key":  'mf = mf.density_fit()',
    })
    dispersion: Optional[str] = field(default="d3bj", metadata={
        "category": ("method",),
        "section": "Method",
        # Profile-level: method-family choice; the vibration deck's
        # dispersion is also profile.  Setting once per project.
        "workflow_group": "profile",
        "label":   "Dispersion",
        # ``mf.disp = "d3bj"`` -- which is what the emitter has always
        # written.  The badge said ``mf = mf.add_dispersion(...)`` until
        # 2026-08-15, and PySCF has no such method: anyone who trusted the
        # badge and searched the docs for it found nothing.
        "engine_key":  'mf.disp = ...',
        # ``none`` is in the choices list so that the case-insensitive
        # click.Choice still accepts the disable spelling; cmd_pyscf
        # then normalises ``none`` -> None before constructing the
        # config.  (R4)
        #
        # ``d3`` WAS offered here and always crashed.  PySCF's own
        # ``pyscf/scf/dispersion.py`` accepts exactly d3bj, d3bjm, d3op,
        # d3zero, d3zerom and d4; anything else reaches
        # ``raise NotImplementedError(f'{method_lower} is not supported
        # yet.')``.  Confirmed against B3LYP, PBE and PBE0 on PySCF 2.13.
        # The zero-damping variant a user picking "d3" means is spelled
        # ``d3zero``, so the choice is renamed rather than dropped.
        "choices": ("d3bj", "d3zero", "d4", "none"),
    })
    # Effective Core Potential -- TWO plain fields, ONE format each.
    #
    # Rewritten 2026-08-13 (user).  It was ``str | dict | None`` where
    # ``""``, ``"none"`` and ``None`` all meant different things: the
    # first two disabled it, and ``None`` silently ADDED ``lanl2dz``
    # whenever any element had Z > 36 and the basis was not def2.  Three
    # spellings, a dict variant the CLI could not reach, and a hidden
    # default.  The rulings that replaced it:
    #
    #   * *"there is no point to limit matching to heavy -- who defines
    #     heavy? there is no clear reasoning or standard"* -- so no Z
    #     threshold decides anything.  ``["*"]`` means ALL atoms.
    #   * *"empty means empty"* -- an empty name or an empty list means
    #     no ECP.  It never means "pick one for me".
    #   * *"one choice, one explicit format"*, *"do not invent too many
    #     options/alias"* -- ``"none"`` is gone; so is the dict.
    #
    # Nothing is added behind the user's back.  ``validation`` still
    # HINTS when a structure looks like it wants an ECP and none was
    # declared -- a hint the user confirms, never a choice made for them.
    ecp: str = field(default="", metadata={
        "workflow_group": "profile",
        "category": ("method",),
        "label":      "Effective core potential",
        "null_label": "(none)",
        "engine_key": 'gto.M(ecp=...)',
    })
    ecp_atoms: List[str] = field(default_factory=list, metadata={
        "workflow_group": "profile",
        "category": ("method",),
        "label":      "ECP atoms",
        "null_label": "(none)",
        "engine_key": 'gto.M(ecp={<element>: ...})',
        # ``List[str]`` is past what ``add_dataclass_options`` generates
        # (P3 bails loudly rather than coercing), so ``cmd_pyscf`` rolls
        # ``--ecp-atoms`` by hand -- the same comma-separated shape as
        # ``--elements`` on ``pseudo check``, not a new spelling.
        "skip_cli":   True,
    })

    # ---------------- SCF ----------------
    scf_conv_tol: float = field(default=1e-9, metadata={
        "category": ("accuracy",),
        "section": "SCF",
        # Convergence target — tightens stage-to-stage.
        "workflow_group": "stage",
        "label": "scf.conv_tol", "unit": "Hartree",
        "engine_key":  'mf.conv_tol',
        "range": (1e-12, 1e-4),
        "tier":  "advanced",
    })
    scf_conv_tol_grad: float = field(default=0.0, metadata={
        "category": ("accuracy",),
        "section": "SCF",
        # Tightens stage-to-stage alongside scf_conv_tol: same reason,
        # a different (and for forces, the decisive) quantity.
        "workflow_group": "stage",
        "label": "scf.conv_tol_grad",
        "engine_key":  'mf.conv_tol_grad',
        "range": (0.0, 1e-2),
        "tier":  "advanced",
        # Verified against the installed PySCF 2.13.0 source
        # (``scf.hf.kernel``):
        #     if conv_tol_grad is None:
        #         conv_tol_grad = numpy.sqrt(conv_tol)
        # so the shipped default 1e-9 energy tolerance yields ~3.2e-5
        # for the gradient.  SCF declares convergence on the ENERGY
        # change and the orbital-gradient norm together, and it is the
        # gradient that sets how clean the forces are -- so tightening
        # only scf_conv_tol moves the criterion that matters for a
        # geometry optimization as a square root.
        #
        # Default 0.0 means "leave PySCF's derivation alone" rather
        # than a number of our choosing: picking one here would
        # silently re-tune every existing run's SCF.  The script
        # reports the effective value either way, so the parameter is
        # never merely implicit.
    })
    scf_soscf: bool = field(default=False, metadata={
        "category": ("convergence",),
        "section": "SCF",
        # Profile-level: an SCF-algorithm choice made with the system,
        # like level_shift -- not a per-stage tightening.
        "workflow_group": "profile",
        "label": "Second-order SCF (SOSCF)",
        "engine_key":  'mf.newton()',
        "tier":  "advanced",
    })
    scf_max_cycle: int = field(default=100, metadata={
        "category": ("convergence",),
        "section": "SCF",
        # Resource-budget cap — patience, not convergence definition.
        "workflow_group": "budget",
        "label": "scf.max_cycle",
        "engine_key":  'mf.max_cycle',
        "range": (10, 1000),
        "tier":  "advanced",
    })
    scf_init_guess: str = field(default="minao", metadata={
        "category": ("convergence",),
        "section": "SCF",
        # Profile-level: SCF initial-guess algorithm is a system-
        # character choice (chosen with the system, not tightened).
        "workflow_group": "profile",
        "label":  "scf.init_guess",
        "engine_key":  'mf.init_guess',
        "id_suffix": "init-guess",
        "choices": ("minao", "atom", "1e", "huckel"),
    })
    grid_level: int = field(default=4, metadata={
        "category": ("accuracy",),
        "section": "SCF",
        "workflow_group": "stage",
        "label": "DFT grid level",
        "engine_key":  'mf.grids.level',
        "range": (0, 9),
        "tier":  "advanced",
        # Default tightened from 3 -> 4: hybrid functionals (B3LYP /
        # PBE0 / M06-2X / wB97X-D, all our typical defaults) have
        # noisy forces at level 3.  Level 4 makes the SCF + force
        # noise floor low enough for tight geometry optimisation.
        # Loosen back to 3 for screening; tighten to 5 for vibrational
        # / phonon work.  Validator warns when level < 4 with hybrid.
    })
    level_shift: float = field(default=0.0, metadata={
        "category": ("convergence",),
        "section": "SCF",
        # Profile-level: SCF stability knob (mirrors SIESTA's
        # mixing_weight which is also profile) — set with the
        # system, doesn't tighten stage-to-stage.
        "workflow_group": "profile",
        "label": "Level shift", "unit": "Hartree",
        "engine_key":  'mf.level_shift',
        "range": (0.0, 1.0),
        "tier":  "advanced",
    })
    # Hard-SCF troubleshooting knobs.  No section -> not on form;
    # power users tweak via Python API.  Defaults preserve PySCF
    # behaviour for the easy-converge case.
    diis_space: int = field(default=8, metadata={
        "workflow_group": "profile",
        "category": ("convergence",),
        "label": "DIIS subspace size",
        "engine_key":  'mf.diis_space',
        "range": (4, 20),
        "tier":  "advanced",
    })
    damp: float = field(default=0.0, metadata={
        "workflow_group": "profile",
        "category": ("convergence",),
        "label": "SCF damping factor",
        "engine_key":  'mf.damp',
        "range": (0.0, 0.9),
        "tier":  "advanced",
    })

    # ---------------- Main optimization ----------------
    optimize: bool = field(default=True, metadata={
        "category": ("procedure",),
        "section": "Compute & budget",
        "item_kind":  "produce",
        # Profile-level: gates whether a relax happens at all --
        # run-shape identity (relax-or-single-point).
        "workflow_group": "profile",
        "label":   "Optimize geometry",
        "engine_key":  '(molbuilder: gates geomeTRIC opt() vs single-point)',
    })
    optimizer: str = field(default="geometric", metadata={
        "category": ("procedure",),
        "section": "Compute & budget",
        # Profile-level: optimizer family choice; parallel to SIESTA's
        # relax_type (also profile).
        "workflow_group": "profile",
        "label":   "Optimizer",
        "engine_key":  'geomeTRIC / berny  (driver selection)',
        "choices": ("geometric", "berny"),
    })
    # ---------------- geomeTRIC convergence (per-stage knobs) ----------
    #
    # **Flat, one value each** -- exactly as SIESTA's per-rung knobs
    # (``relax_force_tol``, ``relax_max_displ``, …) are.  An engine config
    # holds what ONE run does; `task.json`'s stages override it per rung, and
    # a ladder is N of these decks (`engines/stages.md` § 1.1, § 1.1a).  A
    # list-of-rungs field here would be a second place to declare the ladder,
    # free to disagree with the description -- which is what § 1.1 forbids.
    #
    # The SCF tolerance is not among them because it is already above:
    # ``scf_conv_tol`` declares ``mf.conv_tol``, and a per-stage twin of it
    # would be the same knob with two homes.
    geom_gmax: float = field(default=4.5e-4, metadata={
        "skip_cli": True,
        "category": ("accuracy",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "‖F‖∞", "unit": "Ha/Bohr", "step": "any",
        "engine_key": "geomeTRIC convergence_gmax",
        "range": (1.0e-6, 1.0e-1),
        "tier": "advanced",
    })
    geom_grms: float = field(default=3.0e-4, metadata={
        "skip_cli": True,
        "category": ("accuracy",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "‖F‖RMS", "unit": "Ha/Bohr", "step": "any",
        "engine_key": "geomeTRIC convergence_grms",
        "range": (1.0e-6, 1.0e-1),
        "tier": "advanced",
    })
    geom_dmax: float = field(default=1.8e-3, metadata={
        "skip_cli": True,
        "category": ("accuracy",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "Δx max", "unit": "Å", "step": "any",
        "engine_key": "geomeTRIC convergence_dmax",
        "range": (1.0e-5, 1.0),
        "tier": "advanced",
    })
    geom_drms: float = field(default=1.2e-3, metadata={
        "skip_cli": True,
        "category": ("accuracy",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "Δx RMS", "unit": "Å", "step": "any",
        "engine_key": "geomeTRIC convergence_drms",
        "range": (1.0e-5, 1.0),
        "tier": "advanced",
    })
    geom_etol: float = field(default=1.0e-6, metadata={
        "skip_cli": True,
        "category": ("accuracy",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "ΔE tol", "unit": "Hartree", "step": "any",
        "engine_key": "geomeTRIC convergence_energy",
        "range": (1.0e-12, 1.0e-2),
        "tier": "advanced",
    })
    geom_max_steps: int = field(default=200, metadata={
        "skip_cli": True,
        "category": ("procedure",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "Max steps",
        "engine_key": "geomeTRIC maxsteps",
        "range": (1, 10000),
        "tier": "advanced",
    })
    on_nonconvergence: str = field(default="halt", metadata={
        "skip_cli": True,
        "item_kind": "produce",
        "category": ("procedure",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "If max_steps runs out",
        "choices": ("proceed", "continue", "halt"),
        "engine_key": "(molbuilder: per-stage non-convergence policy)",
        "tier": "advanced",
    })
    geom_continue_retries: int = field(default=1, metadata={
        "skip_cli": True,
        "item_kind": "produce",
        "category": ("procedure",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "Continue retries",
        "engine_key": ("(molbuilder: max optimize() re-entries when "
                       "on_nonconvergence=continue)"),
        "range": (0, 5),
        "tier": "advanced",
    })

    # ---------------- Solvent (optional) ----------------
    solvent: Optional[str] = field(default=None, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "section": "Solvent (optional)",
        "label":   "Solvent",
        "engine_key":  'mf = mf.PCM()',
        "null_label": "(gas phase)",
    })
    solvent_method: str = field(default="IEF-PCM", metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "section": "Solvent (optional)",
        "label":   "PCM model",
        # The real attribute, and what the emitter writes.  ``pcm.method``
        # named no object that exists: the solvent handle only appears once
        # ``mf = mf.PCM()`` has run, and it is called ``with_solvent``.
        "engine_key":  'mf.with_solvent.method',
        "choices": ("IEF-PCM", "C-PCM", "COSMO"),
    })

    # ---------------- Runtime ----------------
    # UNLIMITED unless the user asks for a cap -- the typical memory limit is
    # all physical memory (user, ruled 2026-08-13 and again 2026-08-14).
    # ``default = 4000`` stood here until 2026-08-14 and was obsolete history,
    # not a competing default: it asserted a MACHINE FACT's value inside a
    # portable description, which is the one thing `engines/template.md` § 7
    # forbids floor 2 to do.  SIESTA's declaration had the right shape all
    # along -- Optional, valueless, resolved on the machine that runs it --
    # and this one never got the fix.  Now they are ONE item (§ 6.3).
    max_memory_mb: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        "allocation": True,
        "item_kind": "wrapper",
        "workflow_group": "staging",
        "section": "Compute & budget",
        "label": "Max memory", "unit": "MB",
        # NOT an engine keyword any more.  ``mol.max_memory`` is how PySCF
        # spells the answer, and § 6.3 is explicit that a merged item keeps no
        # anchor -- each engine's generator renders it its own way.
        "engine_key":  '(molbuilder: memory cap for the run -- ulimit -v in .run.sh / mol.max_memory)',
        "id_suffix": "max-memory",
        "range": (100, 1_000_000),
        "tier":  "advanced",
        "null_label": "(no cap)",
    })
    threads: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # THE MACHINE ANSWERS THIS, exactly as SIESTA's `omp_threads` does --
        # they are one fact, cores per task, under two engine spellings.  It
        # carried no such mark until 2026-08-18, so `template_fields` excluded
        # three machine facts for SIESTA and ONE for PySCF, and a portable
        # PySCF description could assert how many cores to use -- the one thing
        # `engines/template.md` § 7 forbids floor 2 to do.  Proven by the
        # asymmetry it produced: the identical stage override was REFUSED as a
        # machine fact for SIESTA and ACCEPTED for PySCF.
        "allocation": True,
        "workflow_group": "staging",
        "section": "Compute & budget",
        "label":      "CPU threads",
        "engine_key":  "lib.num_threads(N) + os.environ['OMP_NUM_THREADS']",
        "null_label": "(auto: physical cores)",
    })
    use_gpu: bool = field(default=False, metadata={
        "category": ("execution",),
        "workflow_group": "staging",
        "section": "Compute & budget",
        "label":     "Use GPU (NVIDIA)",
        "item_kind":   "deck",
        "engine_key":  "Diag.ELPA.GPU (SIESTA) | mf = mf.to_gpu() (PySCF)",
        "id_suffix": "use-gpu",
        # Help text intentionally references the recipe rather than
        # naming a specific cuda<N>x wheel tag: the project-wide
        # CUDA pin lives in ``MOLBUILDER_CUDA_VERSION`` /
        # ``molbuilder/envs/recipes.py`` and the right wheel is
        # auto-installed by ``molbuilder envs install molbuilder-pySCF``.
        # Quoting a specific tag here drifts the moment the toolkit
        # bumps; the recipe is the single source of truth.
    })
    verbose: int = field(default=4, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Compute & budget",
        "label": "PySCF verbose",
        "engine_key":  'mol.verbose',
        "range": (0, 9),
        "tier":  "advanced",
    })
    restart: str = field(default="continue", metadata={
        "category": ("convergence", "execution"),
        "section": "Compute & budget",
        # ``produce``, not ``deck``, and `template.md` § 8s own test decides
        # it: *does this item put keywords in the deck?*  On SIESTA yes --
        # three of them, which is why the shared catalogue row is ``deck``.
        # On PySCF no: it changes HOW the script is written, emitting the
        # branches that read the checkpoint and the optimized geometry, and
        # naming no keyword at all.  Same question, same field, same two
        # answers; the mechanism is the engine's (`stages.md` § 1.1a,
        # consequence 3).
        "item_kind":  "produce",
        "workflow_group": "staging",
        "label": "Start from",
        "choices": ("clean", "continue"),
        "id_suffix": "restart",
        "tier": "advanced",
        "engine_key": ("(molbuilder: one field, one mechanism per "
                       "engine -- SIESTA expands it to DM.UseSaveDM / "
                       "MD.UseSaveXV / MD.UseSaveCG; PySCF emits control "
                       "flow that reads <JOB>.chk and "
                       "<JOB>_optimized.xyz.  Not a single engine key on "
                       "either)"),
    })
    chkfile: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Compute & budget",
        "label":   "Write checkpoint (.chk)",
        "engine_key":  "mf.chkfile = '<path>'",
    })
    log_file: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Compute & budget",
        "label":   "Write PySCF log",
        "engine_key":  "gto.M(output='<job>_<stage>.log')",
    })
    # Always-on output knobs; unsectioned (no good reason to expose).
    save_optimized_xyz: bool = field(default=True, metadata={
        # molbuilder's own doing, not a PySCF keyword: it shapes
        # what the PRODUCER writes, so it is kind="produce" (§ 6).
        "item_kind": "produce",
        "workflow_group": "output",
        "category": ("procedure",),
            "engine_key":  '(molbuilder: writes <job>_optimized.xyz after the relaxation)',
    })
    save_initial_xyz: bool = field(default=True, metadata={
        # molbuilder's own doing, not a PySCF keyword: it shapes
        # what the PRODUCER writes, so it is kind="produce" (§ 6).
        "item_kind": "produce",
        "workflow_group": "output",
        "category": ("procedure",),
            "engine_key":  '(molbuilder: writes <job>_initial.xyz before the relaxation)',
    })
    write_trajectory: bool = field(default=True, metadata={
        # molbuilder's own doing, not a PySCF keyword: it shapes
        # what the PRODUCER writes, so it is kind="produce" (§ 6).
        "item_kind": "produce",
        "workflow_group": "output",
        "category": ("procedure",),
            "engine_key":  '(molbuilder: per-step .xyz from geomopt callback)',
    })
    # Match SiestaConfig's naming (``write_molwatch_log``) so the two
    # configs read the same way.  ``molwatch_log`` is kept as a
    # back-compat property below in __post_init__ for callers passing
    # the old kwarg.  Emission also requires ``optimize=True`` AND
    # ``optimizer="geometric"`` -- the molwatch hooks ride on the
    # SCF and geomeTRIC opt-step callbacks, so a single-point or
    # berny run has nowhere to attach.  See spec
    # docs/engines/pyscf.md L33 for the exact gate.
    write_molwatch_log: bool = field(default=True, metadata={
        # molbuilder's own doing, not a PySCF keyword: it shapes
        # what the PRODUCER writes, so it is kind="produce" (§ 6).
        "item_kind": "produce",
        "workflow_group": "output",
        "category": ("procedure",),
            "engine_key":  '(molbuilder: writes <basename>.molwatch.log for the live viewer)',
    })


    # ----- Vibrational spectroscopy (the vibration calculation kind; -----
    # ----- spectra-migration plan P0, 2026-08-20.  Carried from the -----
    # ----- the catalogue is the master. -----
    already_relaxed: bool = field(default=False, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "section": "Vibration",
        "label": 'Structure is already relaxed',
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('geomeTRIC optimize()', 'gradient check'),
        "engine_key": "(molbuilder: skips the deck's built-in relaxation)",
    })
    compute_raman: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "section": "Vibration",
        "label": 'Compute Raman activities',
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('finite-difference polarizability loop',),
        "engine_key": '(molbuilder: finite-diff polarizability path)',
    })
    compute_ir: bool = field(default=False, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "section": "Vibration",
        "label": 'Compute IR intensities',
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('analytic dipole derivatives (or a finite-difference dipole loop)',),
        "engine_key": '(molbuilder: pyscf.prop.infrared, else finite-diff dipoles)',
    })
    displacement_amplitude_ang: float = field(default=0.02, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "section": "Vibration",
        "label": 'Displacement amplitude',
        "unit": 'Å',
        "range": (0.02, 0.2),
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('mode displacement step',),
        "engine_key": '(molbuilder: finite-difference step amplitude)',
    })
    es_mode_selection: str = field(default='skip', metadata={
        "category": ("procedure",),
        "workflow_group": "stage",
        "section": "Vibration",
        "label": 'Mode selection',
        "choices": ('skip', 'all', 'top_n', 'threshold', 'explicit'),
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode electronic-structure selector)',
    })
    es_top_n: int = field(default=10, metadata={
        "category": ("procedure",),
        "workflow_group": "stage",
        "section": "Vibration",
        "label": 'Top-N modes',
        "range": (1, 1000),
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode selector parameter)',
    })
    es_threshold: float = field(default=1.0, metadata={
        "category": ("procedure",),
        "workflow_group": "stage",
        "section": "Vibration",
        "label": 'Raman-activity threshold',
        "unit": 'Å⁴/amu',
        "range": (0.0, 1000.0),
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode selector parameter)',
    })
    es_explicit_indices: str = field(default='', metadata={
        "category": ("procedure",),
        "workflow_group": "stage",
        "section": "Vibration",
        "label": 'Explicit modes',
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode selector parameter)',
    })
    freq_min_cm1: Optional[float] = field(default=None, metadata={
        "category": ("procedure",),
        "workflow_group": "stage",
        "section": "Vibration",
        "label": 'Min frequency',
        "unit": 'cm⁻¹',
        "null_label": '(no lower bound)',
        "optional": True,
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode frequency filter)',
    })
    freq_max_cm1: Optional[float] = field(default=None, metadata={
        "category": ("procedure",),
        "workflow_group": "stage",
        "section": "Vibration",
        "label": 'Max frequency',
        "unit": 'cm⁻¹',
        "null_label": '(no upper bound)',
        "optional": True,
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode frequency filter)',
    })
    es_n_homo_below: int = field(default=5, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Vibration",
        "label": 'Orbitals below HOMO to save',
        "range": (0, 50),
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('orbital-window record',),
        "engine_key": '(molbuilder: orbital-window record size)',
    })
    es_n_lumo_above: int = field(default=5, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Vibration",
        "label": 'Orbitals above LUMO to save',
        "range": (0, 50),
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('orbital-window record',),
        "engine_key": '(molbuilder: orbital-window record size)',
    })
    temperature_K: float = field(default=298.15, metadata={
        "category": ("procedure",),
        "section": "Frequencies / thermochemistry",
        # Profile-level: standard-state thermochemistry condition;
        # paired with ``pressure_atm`` (also profile).
        "workflow_group": "profile",
        "label": "Thermochemistry temperature", "unit": "K",
        "engine_key":  'thermo.thermo(temperature=...)',
        "id_suffix": "temperature",
        "range": (0.0, 5000.0),
        "tier":  "advanced",
    })
    pressure_atm: float = field(default=1.0, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "section": "Frequencies / thermochemistry",
        "label": "Thermochemistry pressure", "unit": "atm",
        "engine_key":  'thermo.thermo(pressure=...)',
        "id_suffix": "pressure",
        "range": (1.0e-6, 1.0e3),
        "tier":  "advanced",
    })

    # ---------------- Comments ----------------
    verbose_comments: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Compute & budget",
        "item_kind":  "produce",
        "label":   "Verbose inline comments",
        "engine_key":  '(molbuilder: comment-block control in the generated input)',
    })

    # Back-compat: the field was named ``molwatch_log`` before the
    # 2026-05-10 naming alignment with SiestaConfig.write_molwatch_log.
    # The property mirrors writes / reads to the canonical attribute
    # so existing user code passing ``molwatch_log=...`` still works.
    @property
    def molwatch_log(self) -> bool:                  # pragma: no cover
        return self.write_molwatch_log

    @molwatch_log.setter
    def molwatch_log(self, value: bool) -> None:     # pragma: no cover
        self.write_molwatch_log = value



__all__ = ["PySCFConfig"]
