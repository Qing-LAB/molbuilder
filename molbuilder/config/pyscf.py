"""PySCFConfig -- every parameter the PySCF script generator emits.

L1 dataclass.  Its field metadata is read by the validation pass
(``molbuilder/validation/``; the readers are `web/form-schema.md` § 1a's)
-- the forms are drawn from the catalogue; the PySCF deck's spec
(``molbuilder/pyscf/input.py:spec_for``, and ``pyscf/vibration_deck.py``
for a vibration) renders the configured values.

Defaults are tuned for "build a small/medium molecule and relax it":

    * B3LYP+D3BJ/def2-SVP  (modern hybrid, dispersion-corrected)
    * Density fitting on -- bare ``mf.density_fit()``, so PySCF auto-picks
      the *basis-matched* JK-fit set (``def2-svp-jkfit`` for this def2-SVP
      default, ``def2-tzvp-jkfit`` for def2-TZVP, ...), verified via
      ``mf.with_df.auxbasis`` on a real def2 hybrid.
    * geomeTRIC optimizer with maxsteps=200, grms=3e-4 Ha/Bohr
    * Kohn-Sham DFT, with the charge and the spin worked out from the
      structure when left blank (`electronic_state`,
      `science/chemistry-correctness.md` § 2a): the phosphate rule for the
      charge, then open or closed shell at that charge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..electronic_state import METHODS
from ..identity import RestartGroup
from ..selection import parse_index_list
from . import state as _state
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
#: on a stage (`stages.md` § 2).  ``restart`` is NOT among them: a rung's
#: position does not answer *is there anything to continue from*
#: (`run-identity.md` § 4 rule 3), so neither engine's ladder sets it.
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
#: already beside it.
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
    # ---------------- System ----------------
    job_name: str = field(default="pyscf_relax", metadata={
        "category": ("system", "procedure"),
        "workflow_group": "setup",
        "item_kind":  "produce",
        "label":    "Job name",
        "engine_key":  '(molbuilder: filename + log-name basename)',
        "pattern":  r"^[A-Za-z0-9_\-]+$",
        "validate": _validate_basename("job_name"),
    })
    # THE ELECTRONIC STATE (`science/chemistry-correctness.md` § 2a) -- the
    # three merged items, declared once in `config/state.py` for both
    # engines, and `method` below.  A blank means *work it out*;
    # `electronic_state` does, for the form, the checks and the deck alike.
    net_charge: Optional[int] = _state.net_charge()
    spin_treatment: Optional[str] = _state.spin_treatment()
    unpaired_electrons: Optional[Union[int, str]] = _state.unpaired_electrons()
    symmetry: bool = field(default=False, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "label":   "Use point-group symmetry",
        "engine_key":  'gto.M(symmetry=...)',
    })

    # ---------------- Method (main run) ----------------
    # WHICH THEORY -- Kohn-Sham DFT or Hartree-Fock -- and nothing else.  The
    # SCF class is COMPOSED from this and `spin_treatment` and written
    # explicitly (`dft.UKS`, `scf.ROHF`, ...; `pyscf/layout.scf_class`), never left
    # to PySCF to re-rule.
    method: str = field(default="DFT", metadata={
        "category": ("method",),
        "workflow_group": "profile",
        "label":   "Method",
        "engine_key":  '(molbuilder: the SCF class module -- dft.<class> for DFT, scf.<class> for HF)',
        "item_kind": "deck",
        "expands": ("dft", "scf"),
        "choices": METHODS,
    })
    functional: str = field(default="B3LYP", metadata={
        "category": ("method",),
        "workflow_group": "profile",
        "label":   "Functional",
        "engine_key":  'mf.xc = ...',
    })
    basis: str = field(default="def2-SVP", metadata={
        "category": ("method", "accuracy"),
        "workflow_group": "profile",
        "label":   "Basis set",
        "engine_key":  'gto.M(basis=...)',
    })
    # auxbasis: rarely set -- the auto-pick from density_fit() is the
    # right default.
    auxbasis: Optional[str] = field(default=None, metadata={
        "workflow_group": "profile",
        "category": ("method",),
            "engine_key":  'mf = mf.density_fit(auxbasis=...)',
    })
    density_fit: bool = field(default=True, metadata={
        "category": ("method", "execution"),
        # Profile-level: method-family identity choice; the vibration
        # deck's density_fit is also profile.
        "workflow_group": "profile",
        "label":   "Density fitting",
        "engine_key":  'mf = mf.density_fit()',
    })
    dispersion: str = field(default="d3bj", metadata={
        "category": ("method",),
        # Profile-level: method-family choice; the vibration deck's
        # dispersion is also profile.  Setting once per project.
        "workflow_group": "profile",
        "label":   "Dispersion",
        "engine_key":  'mf.disp = ...',
        # ``none`` IS THE VALUE for no correction -- stored, written into
        # the template, read back as itself -- because None is what an UNSET
        # item reads as, and an unset item takes the class default (d3bj).
        #
        # PySCF's own ``pyscf/scf/dispersion.py`` accepts exactly d3bj,
        # d3bjm, d3op, d3zero, d3zerom and d4; anything else raises
        # ``ValueError("Unknown dispersion version ...")`` (PySCF 2.14).
        "choices": ("d3bj", "d3zero", "d4", "none"),
    })
    # Effective Core Potential -- TWO plain fields, ONE format each.  The
    # user's rulings (2026-08-13):
    #
    #   * *"there is no point to limit matching to heavy -- who defines
    #     heavy? there is no clear reasoning or standard"* -- so no Z
    #     threshold decides anything.  ``["*"]`` means ALL atoms.
    #   * *"empty means empty"* -- an empty name or an empty list means
    #     no ECP.  It never means "pick one for me".
    #   * *"one choice, one explicit format"*, *"do not invent too many
    #     options/alias"*.
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
    })

    # ---------------- SCF ----------------
    scf_conv_tol: float = field(default=1e-9, metadata={
        "category": ("accuracy",),
        # Convergence target — tightens stage-to-stage.
        "workflow_group": "stage",
        "label": "scf.conv_tol", "unit": "Hartree",
        "engine_key":  'mf.conv_tol',
        "range": (1e-12, 1e-4),
        "tier":  "advanced",
    })
    scf_conv_tol_grad: float = field(default=0.0, metadata={
        "category": ("accuracy",),
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
        # Profile-level: an SCF-algorithm choice made with the system,
        # like level_shift -- not a per-stage tightening.
        "workflow_group": "profile",
        "label": "Second-order SCF (SOSCF)",
        "engine_key":  'mf.newton()',
        "tier":  "advanced",
    })
    scf_max_cycle: int = field(default=100, metadata={
        "category": ("convergence",),
        # Resource-budget cap — patience, not convergence definition.
        "workflow_group": "budget",
        "label": "scf.max_cycle",
        "engine_key":  'mf.max_cycle',
        "range": (10, 1000),
        "tier":  "advanced",
    })
    scf_init_guess: str = field(default="minao", metadata={
        "category": ("convergence",),
        # Profile-level: SCF initial-guess algorithm is a system-
        # character choice (chosen with the system, not tightened).
        "workflow_group": "profile",
        "label":  "scf.init_guess",
        "engine_key":  'mf.init_guess',
        "choices": ("minao", "atom", "1e", "huckel"),
    })
    grid_level: int = field(default=4, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "DFT grid level",
        "engine_key":  'mf.grids.level',
        "range": (0, 9),
        "tier":  "advanced",
        # Default 4: hybrid functionals (B3LYP /
        # PBE0 / M06-2X / wB97X-D, all our typical defaults) have
        # noisy forces at level 3.  Level 4 makes the SCF + force
        # noise floor low enough for tight geometry optimisation.
        # Loosen back to 3 for screening; tighten to 5 for vibrational
        # / phonon work.  Validator warns when level < 4 with hybrid.
    })
    level_shift: float = field(default=0.0, metadata={
        "category": ("convergence",),
        # Profile-level: SCF stability knob (mirrors SIESTA's
        # mixing_weight which is also profile) — set with the
        # system, doesn't tighten stage-to-stage.
        "workflow_group": "profile",
        "label": "Level shift", "unit": "Hartree",
        "engine_key":  'mf.level_shift',
        "range": (0.0, 1.0),
        "tier":  "advanced",
    })
    # Hard-SCF troubleshooting knobs.  Defaults preserve PySCF
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
        "item_kind":  "produce",
        # Profile-level: gates whether a relax happens at all --
        # run-shape identity (relax-or-single-point).
        "workflow_group": "profile",
        "label":   "Optimize geometry",
        "engine_key":  '(molbuilder: gates geomeTRIC opt() vs single-point)',
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
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "‖F‖∞", "unit": "Ha/Bohr",
        "engine_key": "geomeTRIC convergence_gmax",
        "range": (1.0e-6, 1.0e-1),
        "tier": "advanced",
    })
    geom_grms: float = field(default=3.0e-4, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "‖F‖RMS", "unit": "Ha/Bohr",
        "engine_key": "geomeTRIC convergence_grms",
        "range": (1.0e-6, 1.0e-1),
        "tier": "advanced",
    })
    geom_dmax: float = field(default=1.8e-3, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "Δx max", "unit": "Å",
        "engine_key": "geomeTRIC convergence_dmax",
        "range": (1.0e-5, 1.0),
        "tier": "advanced",
    })
    geom_drms: float = field(default=1.2e-3, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "Δx RMS", "unit": "Å",
        "engine_key": "geomeTRIC convergence_drms",
        "range": (1.0e-5, 1.0),
        "tier": "advanced",
    })
    geom_etol: float = field(default=1.0e-6, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "ΔE tol", "unit": "Hartree",
        "engine_key": "geomeTRIC convergence_energy",
        "range": (1.0e-12, 1.0e-2),
        "tier": "advanced",
    })
    geom_max_steps: int = field(default=200, metadata={
        "category": ("procedure",),
        "workflow_group": "stage",
        "label": "Max steps",
        "engine_key": "geomeTRIC maxsteps",
        "range": (1, 10000),
        "tier": "advanced",
    })
    on_nonconvergence: str = field(default="halt", metadata={
        "item_kind": "produce",
        "category": ("procedure",),
        "workflow_group": "stage",
        "label": "If max_steps runs out",
        "choices": ("proceed", "continue", "halt"),
        "engine_key": "(molbuilder: per-stage non-convergence policy)",
        "tier": "advanced",
    })
    geom_continue_retries: int = field(default=1, metadata={
        "item_kind": "produce",
        "category": ("procedure",),
        "workflow_group": "stage",
        "label": "Continue retries",
        "engine_key": ("(molbuilder: re-entries from the geometry reached "
                       "when on_nonconvergence=continue)"),
        "range": (0, 5),
        "tier": "advanced",
    })

    # ---------------- Solvent (optional) ----------------
    solvent: Optional[str] = field(default=None, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "label":   "Solvent",
        "engine_key":  'mf = mf.PCM()',
        "null_label": "(gas phase)",
    })
    solvent_method: str = field(default="IEF-PCM", metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "label":   "PCM model",
        # The real attribute, and what the emitter writes: the solvent
        # handle only appears once ``mf = mf.PCM()`` has run, and it is
        # called ``with_solvent``.
        "engine_key":  'mf.with_solvent.method',
        "choices": ("IEF-PCM", "C-PCM", "COSMO"),
    })

    # ---------------- Runtime ----------------
    # UNLIMITED unless the user asks for a cap -- the typical memory limit is
    # all physical memory (user, ruled 2026-08-13 and again 2026-08-14).
    max_memory_mb: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        "allocation": True,
        "item_kind": "wrapper",
        "workflow_group": "staging",
        "label": "Max memory", "unit": "MB",
        # Not an engine keyword: ``mol.max_memory`` is how PySCF
        # spells the answer, and § 6.3 is explicit that a merged item keeps no
        # anchor -- each engine's generator renders it its own way.
        "engine_key":  '(molbuilder: memory cap for the run -- ulimit -v in .run.sh / mol.max_memory)',
        "range": (100, 1_000_000),
        "tier":  "advanced",
        "null_label": "(no cap)",
    })
    threads: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # THE MACHINE ANSWERS THIS, exactly as SIESTA's `omp_threads` does --
        # they are one fact, cores per task, under two engine spellings.
        "allocation": True,
        "workflow_group": "staging",
        "label":      "CPU threads",
        "engine_key":  "lib.num_threads(N) + os.environ['OMP_NUM_THREADS']",
        "null_label": "(not stated: prep refuses)",
    })
    gpu_count: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # THE SAME ITEM AS SIESTA'S (`execution/gpu.md` § 1.1): how many
        # GPUs a run asks for, stated on its run card -- an allocation ask,
        # never a template value, and no default.
        "allocation": True,
        "item_kind":  "wrapper",
        "workflow_group": "staging",
        "label":      "GPUs (G)",
        "engine_key":  "(molbuilder: scheduler ``--gres=gpu:G``; "
                       "not in the deck)",
        "range":      (1, 16),
    })
    use_gpu: bool = field(default=False, metadata={
        "category": ("execution",),
        "workflow_group": "staging",
        "label":     "Use GPU (NVIDIA)",
        "item_kind":   "deck",
        "engine_key":  "Diag.ELPA.GPU (SIESTA) | mf = mf.to_gpu() (PySCF)",
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
        "label": "PySCF verbose",
        "engine_key":  'mol.verbose',
        "range": (0, 9),
        "tier":  "advanced",
    })
    restart: str = field(default="continue", metadata={
        "category": ("convergence", "execution"),
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
        "label":   "Write checkpoint (.chk)",
        "engine_key":  "mf.chkfile = '<path>'",
    })
    log_file: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "label":   "Write PySCF log",
        "engine_key":  "gto.M(output='<job>_<stage>.log')",
    })
    # Always-on output knobs.
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
    # configs read the same way.  Emission also requires ``optimize=True`` -- the
    # molwatch hooks ride on the SCF and geomeTRIC opt-step callbacks,
    # so a single-point run has nowhere to attach.
    write_molwatch_log: bool = field(default=True, metadata={
        # molbuilder's own doing, not a PySCF keyword: it shapes
        # what the PRODUCER writes, so it is kind="produce" (§ 6).
        "item_kind": "produce",
        "workflow_group": "output",
        "category": ("procedure",),
            "engine_key":  '(molbuilder: writes <basename>.molwatch.log for the live viewer)',
    })


    # ----- Vibrational spectroscopy (the vibration calculation kind) -----
    already_relaxed: bool = field(default=False, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": 'Structure is already relaxed',
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('geomeTRIC relaxation (relax_policy.relax)', 'gradient check'),
        "engine_key": "(molbuilder: no engine keyword -- the ladder or the deck decides)",
    })
    compute_raman: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": 'Compute Raman activities',
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('finite-difference polarizability loop',),
        "engine_key": '(molbuilder: finite-diff polarizability path)',
    })
    compute_ir: bool = field(default=False, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": 'Compute IR intensities',
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('analytic dipole derivatives (or a finite-difference dipole loop)',),
        "engine_key": '(molbuilder: pyscf.prop.infrared, else finite-diff dipoles)',
    })
    displacement_amplitude_ang: float = field(default=0.02, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
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
        "workflow_group": "profile",
        "label": 'Mode selection',
        "choices": ('skip', 'all', 'explicit'),
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode electronic-structure selector)',
    })
    es_explicit_indices: str = field(default='', metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": 'Explicit modes',
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode selector parameter)',
    })
    freq_min_cm1: Optional[float] = field(default=None, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": 'Min frequency',
        "unit": 'cm⁻¹',
        "null_label": '(no lower bound)',
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode frequency filter)',
    })
    freq_max_cm1: Optional[float] = field(default=None, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": 'Max frequency',
        "unit": 'cm⁻¹',
        "null_label": '(no upper bound)',
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('per-mode displaced-SCF loop',),
        "engine_key": '(molbuilder: per-mode frequency filter)',
    })
    es_n_homo_below: int = field(default=5, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
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
        "label": 'Orbitals above LUMO to save',
        "range": (0, 50),
        "tier": "advanced",
        "item_kind": "deck",
        "expands": ('orbital-window record',),
        "engine_key": '(molbuilder: orbital-window record size)',
    })
    temperature_K: float = field(default=298.15, metadata={
        "category": ("procedure",),
        # ONE MEANING ON BOTH ENGINES (`engines/vibration.md` § 3.1): the
        # thermochemistry's temperature.
        "workflow_group": "profile",
        "label": "Thermochemistry temperature", "unit": "K",
        "item_kind": "deck",
        "expands": ('thermo.thermo(temperature=)', 'the harmonic vibrational sums'),
        "engine_key":  "(molbuilder: the thermochemistry's temperature -- PySCF's thermo.thermo(temperature=) for a free molecule, the harmonic vibrational sums otherwise and in SIESTA's finish)",
        "range": (1.0, 5000.0),
        "tier":  "advanced",
    })
    pressure_atm: float = field(default=1.0, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": "Thermochemistry pressure", "unit": "atm",
        "engine_key":  'thermo.thermo(pressure=...)',
        "range": (1.0e-6, 1.0e3),
        "tier":  "advanced",
    })

    # ---------------- Comments ----------------
    verbose_comments: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "item_kind":  "produce",
        "label":   "Verbose inline comments",
        "engine_key":  '(molbuilder: comment-block control in the generated input)',
    })

    @property
    def is_dft(self) -> bool:
        """Whether the method is a density functional.  THE one answer every
        reader of the level of theory asks: the decks' SCF construction,
        headers, constants and line spellings, the grid advisory and the
        Methods paragraph (`engines/vibration.md` § 4.10).  Hartree-Fock has
        no functional and no integration grid, whatever those items hold.
        The dispersion correction is not asked of this answer: HF takes it
        like any method -- it has no correlation at all, so it misses
        dispersion entirely, D3 and D4 carry parameters fitted for it, and
        PySCF applies them through the energy, gradient and Hessian, reading
        the method as ``hf`` (`engines/pyscf.md` § 7a).  `method` is never
        blank, so this asks the field directly."""
        return self.method == "DFT"

    @property
    def explicit_modes(self) -> List[int]:
        """The modes ``es_explicit_indices`` names: 1-based, sorted, each once.
        THE one reading of that text -- the deck's constant (which the
        script's selector reads), the Methods paragraph and the kind's check
        all take it from here (`engines/vibration.md` § 4.8).  The grammar is the atom
        index list's (``selection.parse_index_list``): ``"3, 7, 12"``,
        ``"3-7, 12"``.  Raises ``SelectionError`` (a ``ValueError``) on text
        it cannot read; the kind's check turns that into a refusal at
        ``prep``, so no deck is written from it."""
        return parse_index_list(self.es_explicit_indices, first=1)

    # ``molwatch_log`` reads and writes ``write_molwatch_log``.
    @property
    def molwatch_log(self) -> bool:                  # pragma: no cover
        return self.write_molwatch_log

    @molwatch_log.setter
    def molwatch_log(self, value: bool) -> None:     # pragma: no cover
        self.write_molwatch_log = value


__all__ = ["PySCFConfig"]
