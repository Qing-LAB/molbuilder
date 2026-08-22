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


STAGE_STRATEGY_PRESETS: Dict[str, Tuple[bool, ...]] = {
    "publishable": (True,  True,  False),   # stage1 loose + stage2 publishable
    "loose-only":  (True,  False, False),   # stage1 only (cheap warm-up)
    "vib-quality": (True,  True,  True),    # all three (TIGHT for vib/IR/NEB)
}


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
        "help":     "job name; output files get this prefix.  Must match "
                    "[A-Za-z0-9_-]+ (job-layout v1; no dots).",
        # Same basename rule as SiestaConfig.system_label -- import the
        # shared helper so the regex has ONE home.
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
        "help": ("Net charge of the system, in units of |e|.  Default "
                 "(blank) auto-detects from phosphate protonation -- one "
                 "negative charge per backbone phosphate, which is right "
                 "for DNA/RNA from tleap.  For everything else (peptide, "
                 "SMILES, PDB load) auto resolves to 0, so set this "
                 "EXPLICITLY for a charged species: carboxylates "
                 "(Asp/Glu), protonated amines (Lys/Arg/His+), "
                 "sulfonates.  Sign convention: -1 = one extra electron; "
                 "+1 = one missing electron."),
    })
    spin: int = field(default=0, metadata={
        "category": ("system",),
        "section": "System",
        # System characteristic — open-shell chemistry, not stage.
        "workflow_group": "profile",
        "label":   "Spin (2S)",
        "engine_key":  'gto.M(spin=...)  # 2S, # of unpaired electrons',
        "range":   (0, 10),
        "help": "2S (NOT 2S+1); 0=closed shell, 1=doublet, 2=triplet, ...",
    })
    symmetry: bool = field(default=False, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "section": "System",
        "label":   "Use point-group symmetry",
        "engine_key":  'gto.M(symmetry=...)',
        "help": (
            "enable point-group symmetry; faster but rarely matches "
            "builder-output geometry exactly A symmetric molecule can run "
            "2-10x faster with this on, because PySCF skips the integrals "
            "symmetry makes redundant. It is unforgiving of numerical "
            "drift, though: builder output rarely places atoms on the "
            "symmetry elements to the precision PySCF checks, and a near "
            "miss is refused rather than approximated. Leave it off "
            "unless you know the geometry is exact."
        ),
    })

    # ---------------- Method (main run) ----------------
    method: str = field(default="RKS", metadata={
        "category": ("method",),
        "section": "Method",
        "workflow_group": "profile",
        "label":   "SCF method",
        "engine_key":  'RKS / UKS / RHF / UHF  (PySCF class selection)',
        "choices": ("RKS", "UKS", "RHF", "UHF"),
        "help": "RKS / UKS / RHF / UHF",
    })
    functional: str = field(default="B3LYP", metadata={
        "category": ("method",),
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Functional",
        "engine_key":  'mf.xc = ...',
        "help": (
            "XC functional, written as ``mf.xc`` (B3LYP / PBE / PBE0 / "
            "M06-2X / wB97X-D / ...). DEVIATION: PySCF's own default is "
            "'LDA,VWN'. That is the fallback ``dft.RKS`` carries when "
            "nothing has been set, not a recommendation -- setting mf.xc "
            "is expected of every real calculation. This catalogue starts "
            "at B3LYP, the most widely used hybrid in molecular chemistry "
            "(Becke, J. Chem. Phys. 98, 5648 (1993); Lee, Yang & Parr, "
            "Phys. Rev. B 37, 785 (1988)), and it pairs with the D3(BJ) "
            "dispersion this project also defaults to -- which is what "
            "repairs B3LYP's known weakness on non-covalent interactions. "
            "Pure GGAs (PBE, BLYP) run 2-3x faster than a hybrid because "
            "they need no exact-exchange build. What they lose with it is "
            "the partial cancellation of self-interaction error that "
            "exact exchange provides, so a pure functional suffers more "
            "of it -- which surfaces as underestimated gaps and "
            "over-delocalised anions. Worth taking for a "
            "pre-optimization, rarely for the number you publish."
        ),
    })
    basis: str = field(default="def2-SVP", metadata={
        "category": ("method", "accuracy"),
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Basis set",
        "engine_key":  'gto.M(basis=...)',
        "help": (
            "Gaussian basis set.  This is NOT a per-stage knob -- "
            "the convergence ladder varies tolerances, "
            "not the level of theory.  Pick once based on the chemistry.\n"
            "Per-tier:\n"
            "  • screening / loose preopt: def2-SVP (current default)\n"
            "  • publishable:               def2-TZVP "
            "(modern standard for organic chemistry; ECPs bundled to Rn)\n"
            "  • tight (vib/IR/energy):    def2-TZVPP or def2-QZVP\n"
            "WHAT def2-SVP IS AND IS NOT GOOD FOR.  It is a double-zeta "
            "set: fine for screening and preoptimisation, and usable for "
            "structures when paired with a dispersion correction (this "
            "project defaults to D3BJ).  Its real weakness is BASIS-SET "
            "SUPERPOSITION ERROR -- for double-zeta sets that can exceed "
            "40% of a binding energy, and it artificially SHORTENS "
            "intermolecular distances.  So treat def2-SVP interaction "
            "energies and complex geometries as unconverged, and move to "
            "def2-TZVP for anything published.  Covalent bond lengths are "
            "much less affected: DFT bond-length errors are typically "
            "0.01-0.07 A and are driven more by the functional than by the "
            "basis.\n"
            "NOTE ON WHAT THIS PROJECT CORRECTS: D3BJ repairs the missing "
            "dispersion, not the superposition error -- PySCF applies no "
            "geometric counterpoise (gCP).  The composite methods built "
            "for exactly this (PBEh-3c, r2SCAN-3c) pair a small basis with "
            "BOTH corrections.\n"
            "This help claimed a \"30% bond-length error in conjugated "
            "systems -- NEVER publish\" until 2026-08-15.  No source "
            "supports it; the figure appears to be a garbled import of the "
            "40%-of-binding-energy BSSE result, which is about noncovalent "
            "interactions rather than covalent bonds, and is not specific "
            "to conjugation.\n"
            "Sources: Weigend & Ahlrichs, Phys. Chem. Chem. Phys. 7, 3297 "
            "(2005) for the basis sets themselves; Sure & Grimme, J. "
            "Comput. Chem. 34, 1672 (2013) for the small-basis error "
            "analysis; Bursch et al., Angew. Chem. Int. Ed. 61, e202205735 "
            "(2022) for the best-practice recommendation.  See "
            "docs/engines/tuning.md § 2.8."
        ),
    })
    # auxbasis: Python-API knob; rarely set from the form (auto-pick
    # from density_fit() is the right default).  No section -> not on form.
    auxbasis: Optional[str] = field(default=None, metadata={
        "workflow_group": "profile",
        "category": ("method",),
        "help": "auxiliary fitting basis; None lets density_fit() auto-pick",
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
        "help": (
            "Approximate the four-centre electron-repulsion integrals by "
            "three-centre ones over an auxiliary basis (also called "
            "resolution of the identity), which is what makes the Coulomb "
            "and exchange build cheap. DEVIATION: PySCF applies NO "
            "density fitting unless ``mf.density_fit()`` is called; we "
            "default it on. The speed-up is large and the error it "
            "introduces sits well below the error of the orbital basis "
            "itself when the auxiliary set matches the basis -- which is "
            "why it is standard practice rather than a shortcut. Turn it "
            "off when you need exact-integral reference numbers, or when "
            "comparing against a published value that did not use it. In "
            "practice the SCF iteration cost drops by roughly 5-10x, and "
            "for organic systems the total-energy error it introduces "
            "stays under about 0.1 kcal/mol."
        ),
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
        "help": (
            "Grimme dispersion correction, written as ``mf.disp``. d3bj "
            "(default) is D3 with Becke-Johnson damping; d3zero is D3 "
            "with the original zero damping; d4 is the newer, "
            "charge-dependent D4. 'none' disables the correction. BJ "
            "damping is the usual recommendation for D3 -- it does not go "
            "to zero at short range, which is where zero damping tends to "
            "under-bind. Use d3zero only to reproduce a published number "
            "that used it. SOURCE: the accepted spellings are PySCF "
            "2.13's own (pyscf/scf/dispersion.py); a value outside that "
            "set raises NotImplementedError at run time, not at setup. It "
            "matters most for biomolecules and anything weakly bound: "
            "pi-stacking, van der Waals contacts and hydrogen-bond "
            "geometries are all under-bound without a correction, and the "
            "cost of adding one is negligible."
        ),
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
        "help": ("effective core potential name, e.g. 'lanl2dz'.  Empty = no "
                 "ECP.  Applies to the atoms named by --ecp-atoms; with no "
                 "atoms selected it does nothing."),
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
        "help": ("which elements get the ECP, as element patterns: empty = "
                 "none; '*' = every element in the structure; 'Au' = that "
                 "element; 'A*' = every element whose symbol starts with A. "
                 "Several may be given."),
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
        "help": (
            "SCF convergence tolerance on the energy (Hartree) Tighten it "
            "(1e-10) when forces look noisy, or when DFT energies drift "
            "between geometry steps. Both are symptoms of an SCF that "
            "stopped while the density was still moving, which shows up "
            "in the derivative long before it shows up in the energy."
        ),
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
        "help":  ("orbital-gradient convergence threshold; 0 = PySCF's "
                  "own sqrt(conv_tol) (~3e-5 at conv_tol=1e-9).  The "
                  "gradient is what the forces come from -- tighten "
                  "this (1e-6, 1e-7) when forces look noisy"),
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
        "help":  ("switch the SCF to a second-order Newton solver.  "
                  "Slower per iteration and needs more memory, but "
                  "converges cases where DIIS oscillates forever "
                  "(open-shell metals, near-degenerate frontier "
                  "orbitals).  The usual escalation order is DIIS -> "
                  "level shift / damping -> SOSCF"),
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
        "help":  "The most SCF cycles PySCF will run for one single-point "
                 "before giving up.\n"
                 "DEVIATION: PySCF's own default is 50; this catalogue starts at 100.  This "
                 "is a RUNAWAY GUARD, not a target -- the two failure modes "
                 "are not symmetric.  Too high wastes some CPU on a run that "
                 "was not going to converge anyway; too low stops a "
                 "converging SCF at the cap and throws away that geometry "
                 "step, which then repeats at every step of a relaxation.  A "
                 "well-behaved closed-shell molecule converges in 10-30.",
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
        "help": "SCF initial guess: minao / atom / 1e / huckel",
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
        "help":  "Density of the numerical grid the exchange-correlation "
                 "energy is integrated on: 0 = coarse, 3 = screening, "
                 "4 = production, 5 = tight, 9 = ultra.\n"
                 "DEVIATION: PySCF's own default is 3; this catalogue starts at 4.  Level 3 "
                 "is fine for an energy, but the quadrature error does not "
                 "cancel in derivatives -- it shows up in forces and, most "
                 "visibly, in vibrational frequencies, where grid noise "
                 "produces spurious low-frequency modes.  Since this project "
                 "exists to run geometry optimisations and Hessians, the "
                 "grid that supports them is the right default.  Drop to 3 "
                 "for single-point screening.",
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
        "help":  "0.1-0.3 helps hard SCFs; 0 if SCF converges cleanly",
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
        "help":  "DIIS subspace size; bump to 12-20 for oscillating SCFs",
    })
    damp: float = field(default=0.0, metadata={
        "workflow_group": "profile",
        "category": ("convergence",),
        "label": "SCF damping factor",
        "engine_key":  'mf.damp',
        "range": (0.0, 0.9),
        "tier":  "advanced",
        "help":  "Roothaan damping factor; 0.3-0.5 helps when DIIS alone isn't enough",
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
        "help": "run geometry optimization; --no-optimize for single-point only",
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
        "help": (
            "Geometry optimizer driver.\n"
            "  • geometric (default)  — translation-rotation-invariant "
            "internal coords (Wang & Song JCP 2016); BFGS quasi-Newton "
            "under the hood.  Robust on large flexible molecules, "
            "transition states, surface-anchored systems.  REQUIRED "
            "for the staged-opt loop (#534) — only geometric accepts "
            "the per-stage convergence_drms / convergence_dmax kwargs.\n"
            "  • berny  — Cartesian + redundant internals; ships with "
            "PySCF (no extra dep).  Less robust on biomolecules; "
            "DOES NOT work with the staged-opt loop (incompatible "
            "kwarg set).  Use only for single-stage runs.\n"
            "Both are quasi-Newton; both converge tightly near a "
            "minimum.  See docs/engines/tuning.md § 2.1."
        ),
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
        "help": "max-gradient convergence (Ha/Bohr)",
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
        "help": "RMS-gradient convergence (Ha/Bohr)",
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
        "help": "max-displacement convergence (Å)",
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
        "help": "RMS-displacement convergence (Å)",
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
        "help": "energy-step convergence (Hartree)",
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
        "help": "max geomeTRIC iterations in this stage",
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
        "help": (
            "what to do if geomeTRIC's 5 criteria aren't all met when "
            "max_steps runs out: proceed (accept the partial geometry and "
            "let the next rung start from it), continue (extend this rung "
            "with more iterations), halt (raise rather than hand on a "
            "geometry nobody accepted). This is THIS rung's answer: a "
            "ladder is N decks and N jobs, so each deck carries its own "
            "policy and none of them can see the others."
        ),
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
        "range": (1, 5),
        "tier": "advanced",
        "help": ("only meaningful when on_nonconvergence='continue': how "
                 "many additional max_steps batches to spend before falling "
                 "through to halt.  Total step budget = max_steps * "
                 "(1 + continue_retries).  Named geom_* to stay distinct "
                 "from Resources.continue_retries, which is the WRAPPER's "
                 "warm-restart count and a different thing entirely."),
    })

    # ---------------- Solvent (optional) ----------------
    solvent: Optional[str] = field(default=None, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "section": "Solvent (optional)",
        "label":   "Solvent",
        "engine_key":  'mf = mf.PCM()',
        "null_label": "(gas phase)",
        "help": "solvent (water / methanol / dmso / chloroform / ...)",
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
        "help": "Which polarisable-continuum model represents the solvent.\n"
                "DEVIATION: PySCF's own default is C-PCM "
                "(pyscf.solvent.pcm.PCM.method); this catalogue starts at IEF-PCM.  IEF-PCM "
                "solves the full integral-equation formalism, so it is the "
                "more general of the two and is what most quantum-chemistry "
                "packages default to; C-PCM is the conductor-like "
                "approximation, cheaper and very close to IEF-PCM for polar "
                "solvents but less reliable as the dielectric constant falls.  "
                "COSMO is the original conductor-like model.\n"
                "For water the three agree closely; the choice matters in "
                "low-dielectric solvents.",
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
        "help":  (
        "How much memory this run may use. Left blank -- the normal "
        "state -- it is the machine's maximum, resolved at prep on the "
        "node that granted it; set a number only when you need a "
        "ceiling. Each engine applies it its own way: SIESTA emits a "
        "SystemMemory hint into the deck and caps the wrapper, PySCF "
        "passes it to mol.max_memory, which is what it consults to "
        "choose in-core versus out-of-core."),
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
        "help":       "how many CPU threads PySCF uses.  Default "
                      "(blank) asks what this job was GIVEN before it "
                      "asks how big the machine is: OMP_NUM_THREADS, "
                      "then the scheduler's allocation "
                      "(SLURM_CPUS_PER_TASK / PBS_NCPUS / NSLOTS), and "
                      "only as a last resort this node's PHYSICAL cores "
                      "(not logical/HT) -- hyperthreading rarely helps "
                      "QC kernels and can hurt cache locality.  Asking "
                      "the node first is how a job holding 8 cores of a "
                      "128-core machine started 128 threads and had "
                      "them time-sliced onto its 8.  The "
                      "emitted script pins BLAS to 1 thread per "
                      "worker (OPENBLAS_NUM_THREADS=1, "
                      "MKL_NUM_THREADS=1) so PySCF threads * BLAS "
                      "threads don't multiply -- the canonical "
                      "anti-oversubscription recipe.  Set explicitly "
                      "to bench, or to leave cores free for other jobs.",
    })
    use_gpu: bool = field(default=False, metadata={
        "category": ("execution",),
        "workflow_group": "staging",
        "section": "Compute & budget",
        "label":     "Use GPU (NVIDIA, via gpu4pyscf)",
        "engine_key":  'gpu4pyscf: mf = mf.to_gpu()',
        "id_suffix": "use-gpu",
        # Help text intentionally references the recipe rather than
        # naming a specific cuda<N>x wheel tag: the project-wide
        # CUDA pin lives in ``MOLBUILDER_CUDA_VERSION`` /
        # ``molbuilder/envs/recipes.py`` and the right wheel is
        # auto-installed by ``molbuilder envs install molbuilder-pySCF``.
        # Quoting a specific tag here drifts the moment the toolkit
        # bumps; the recipe is the single source of truth.
        "help":      "run the SCF (and geom-opt forces) on an NVIDIA "
                     "GPU via the gpu4pyscf extension.  The recipe "
                     "for ``molbuilder-pySCF`` installs the matching "
                     "``cupy-cudaNx[ctk]`` + ``gpu4pyscf-cudaNx`` "
                     "wheels for the project's pinned CUDA toolkit "
                     "(see ``molbuilder envs doctor molbuilder-pySCF``); "
                     "the script probes gpu4pyscf at run start and "
                     "STOPS with an actionable message if the "
                     "package isn't importable or the GPU is "
                     "missing / too old (compute capability "
                     "< 7.0) -- there is no silent CPU fallback: "
                     "a run that changed where it executed would "
                     "report a CPU time under a GPU label.",
    })
    verbose: int = field(default=4, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Compute & budget",
        "label": "PySCF verbose",
        "engine_key":  'mol.verbose',
        "range": (0, 9),
        "tier":  "advanced",
        "help":  "How much PySCF writes to the log: 0 silent, 3 note, "
                 "4 info, 5 debug.\n"
                 "DEVIATION: PySCF's own default is 3; this catalogue starts at 4, because "
                 "level 4 is where the per-cycle SCF convergence table "
                 "appears -- and that table is what makes a run that failed "
                 "diagnosable from the log alone, without repeating it.",
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
        "help": (
            'Whether this run starts from what is already in the folder.\n'
            '  continue  -- read it (the default).  Nothing there is not an '
            "error: the engine starts from the deck's own coordinates.\n"
            '  clean     -- ignore it and start over, OVERWRITING what is there '
            'as the run proceeds.\n'
            "Which files that means is the engine's own: SIESTA reads .XV / .DM "
            '/ .CG, PySCF reads <JOB>_optimized.xyz and <JOB>.chk.\n'
            'Continuing is the default because a run you start in a folder that '
            'already holds a result is a run you started after looking at that '
            'result.  To keep the old state, save it first with `molbuilder '
            'checkpoint save` -- the launcher warns before a clean run overwrites '
            'anything and stops unless you pass --force.'
        ),
    })
    chkfile: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Compute & budget",
        "label":   "Write checkpoint (.chk)",
        "engine_key":  "mf.chkfile = '<path>'",
        "help": "write <job>.chk (DM, mol, energies for restart)",
    })
    log_file: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Compute & budget",
        "label":   "Write PySCF log",
        "engine_key":  "gto.M(output='<job>_<stage>.log')",
        "help": "write the PySCF text log to <job>.log",
    })
    # Always-on output knobs; unsectioned (no good reason to expose).
    save_optimized_xyz: bool = field(default=True, metadata={
        # molbuilder's own doing, not a PySCF keyword: it shapes
        # what the PRODUCER writes, so it is kind="produce" (§ 6).
        "item_kind": "produce",
        "workflow_group": "output",
        "category": ("procedure",),
        "help": "snapshot the relaxed geometry to <job>_optimized.xyz",
            "engine_key":  '(molbuilder: writes <job>_opt.xyz post-relax)',
    })
    save_initial_xyz: bool = field(default=True, metadata={
        # molbuilder's own doing, not a PySCF keyword: it shapes
        # what the PRODUCER writes, so it is kind="produce" (§ 6).
        "item_kind": "produce",
        "workflow_group": "output",
        "category": ("procedure",),
        "help": "snapshot the input geometry to <job>_initial.xyz",
            "engine_key":  '(molbuilder: writes <job>_init.xyz pre-relax)',
    })
    write_trajectory: bool = field(default=True, metadata={
        # molbuilder's own doing, not a PySCF keyword: it shapes
        # what the PRODUCER writes, so it is kind="produce" (§ 6).
        "item_kind": "produce",
        "workflow_group": "output",
        "category": ("procedure",),
        "help": ("stream geomeTRIC's <job>_geom_optim.xyz so molwatch can "
                 "watch it live"),
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
        "help": ("Write <job>.molwatch.log alongside the run: per-step "
                 "coordinates, energy and forces, in one additive file the "
                 "Watch tab reads. It exists so a trajectory can be followed "
                 "while the engine is still running."),
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
        "help": "YOUR ASSERTION that this structure sits at a stationary point, so the vibration deck skips its built-in relaxation and goes straight to the Hessian.  A harmonic analysis is only valid at a stationary point -- off one, frequencies shift and spurious imaginary modes appear -- which is why relaxation is the deck's mandatory first act unless you state this.  When set, the deck still checks the gradient at the input geometry and WARNS with the max-force number (never refuses): the statement is yours to make.  The check's number also appears on the viewer's relaxation phase chip.",
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
        "help": 'compute Raman scattering intensity for every mode.  Cost: roughly 6 x (number of free atoms) extra response calculations (one polarizability per +/- finite-difference displacement in each Cartesian direction) -- the expensive optional.  Turn off if you only want frequencies or IR (IR is far cheaper; the two are independent toggles over one shared displacement loop).',
    })
    compute_ir: bool = field(default=False, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "section": "Vibration",
        "label": 'Compute IR intensities',
        "tier": "basic",
        "item_kind": "deck",
        "expands": ('finite-difference dipole loop',),
        "engine_key": '(molbuilder: finite-diff dipole-moment derivative path)',
        "help": "compute IR absorption intensities (km/mol) from finite-difference dipole-moment derivatives.  Independent of Raman since 2026-08-20 (one Hessian, one mode set, one shared displacement loop computing whichever properties are ticked) -- and far cheaper: a dipole read per displacement versus Raman's response calculation.  VALIDATED at the band level 2026-08-20 (the vibration E2E holds water at B3LYP/def2-SVP to its literature windows: bend ~55 km/mol > asym ~27 > sym ~5, pattern and magnitudes both); an external cross-code digit match would harden it further and is welcome, not owed.",
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
        "help": 'how far atoms are pushed along each mode eigenvector when probing how the orbitals shift (only used by the per-mode electronic-structure step).  The default 0.02 A sits inside the linear-response regime where the orbital-energy change is proportional to the displacement, so the slope you extract is the physically meaningful number rather than a finite-difference-amplitude artefact.  Trade-off: orbital-energy differences shrink to ~meV at small amplitude and need a tight SCF tolerance (the default scf_conv_tol=1e-9 is sufficient).  Above ~0.10 A anharmonic mixing starts to contaminate the response ([Mills1972] section 2.4); above ~0.20 A the linear-response assumption breaks outright.',
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
        "help": "which vibrational modes get the displaced-geometry orbital-energy probe.  Each chosen mode costs two extra SCF calculations (one at +A, one at -A along the mode), so this is the most expensive part of the run and its cost scales linearly with how many modes you pick.\n    skip      -- don't run this step at all; you get a spectrum but no per-mode HOMO/LUMO data.  Use this for first-pass exploration.\n    all       -- every mode (cost ~ 2 N modes).\n    top_n     -- the N modes with the strongest Raman activity.\n    threshold -- every mode whose Raman activity exceeds your cutoff.\n    explicit  -- you list specific mode numbers.\nCaveat: top_n and threshold rank by Raman brightness, which is NOT the same as electron-phonon coupling strength -- a mode that's transport-critical can be Raman-weak ([Galperin2007]).  When in doubt, use explicit (after looking at the spectrum) or all.",
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
        "help": '(only used when selector = top_n) how many of the brightest Raman-active modes to compute orbital-energy data for.  Cost grows linearly: N modes = 2 N SCF calculations.',
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
        "help": '(only used when selector = threshold) Raman activity cutoff in A^4/amu; every mode brighter than this gets orbital-energy data.  Final mode count is unpredictable -- depends on how many modes happen to be above your cutoff.',
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
        "help": '(only used when selector = explicit) comma-separated list of 1-based mode numbers to compute orbital-energy data for, e.g. "3, 7, 12".  Ranges supported: "3-7, 12".  ONE string format, parsed at the emitter (the T9 precedent: one format, no aliases).  Typical workflow: run with selector = skip first, look at the spectrum, then re-run with selector = explicit and the modes you care about.',
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
        "help": 'restrict orbital-energy data to modes at or above this frequency.  Useful for skipping low-frequency rocking / librational modes that are often noisy and rarely matter for transport.  Caveat: filtering may skip modes whose strong electron-phonon coupling lies outside your chosen window ([Galperin2007]).  Ignored when selector = explicit.',
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
        "help": 'restrict orbital-energy data to modes at or below this frequency.  Combine with Min frequency to target a specific spectral window (e.g. 2800-3200 cm^-1 for C-H stretches).  Ignored when selector = explicit.',
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
        "help": "how many frontier orbitals BELOW the HOMO to record at each displaced geometry.  Five is enough to study HOMO/LUMO behaviour for transport; raise it to see a richer slice of the orbital landscape (e.g. for density-of-states plots).  Doesn't change cost.",
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
        "help": "how many frontier orbitals ABOVE the LUMO to record at each displaced geometry.  Five matches the HOMO setting for a symmetric window around the gap.  Doesn't change cost.",
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
        "help":  "RRHO temperature for thermo.thermo() (standard: 298.15 K)"
            " Re-homed to the vibration calculation kind (spectra-migration plan D2, 2026-08-20): sets the headline (T, P) for the .spectra.json thermo block; the viewer's G/H/S curves run over a documented default temperature grid beside it.",
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
        "help":  "RRHO pressure for thermo.thermo() (standard: 1 atm = 101325 Pa)"
            " Re-homed to the vibration calculation kind (spectra-migration plan D2, 2026-08-20): sets the headline (T, P) for the .spectra.json thermo block; the viewer's G/H/S curves run over a documented default temperature grid beside it.",
    })

    # ---------------- Comments ----------------
    verbose_comments: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "section": "Compute & budget",
        "item_kind":  "produce",
        "label":   "Verbose inline comments",
        "engine_key":  '(molbuilder: comment-block control in the generated input)',
        "help": (
        "Emit inline tuning hints and a Troubleshooting block in the "
        "generated script, in whatever comment syntax that engine uses."),
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
