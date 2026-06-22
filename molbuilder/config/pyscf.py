"""PySCFConfig -- every parameter the PySCF script generator emits.

L1 dataclass.  Field metadata (label / unit / range / tier / help)
drives the CLI option list, the web form schema, and the validation
pass at ``molbuilder/validation.py``; the PySCF generator at
``molbuilder/pyscf/input.py:render_script`` is the only consumer of
the configured values themselves.

Defaults are tuned for "build a small/medium molecule and relax it":

    * B3LYP+D3BJ/def2-SVP  (modern hybrid, dispersion-corrected)
    * Density fitting on (def2-universal-jkfit auto-selected)
    * geomeTRIC optimizer with maxsteps=200, grms=3e-4 Ha/Bohr
    * Closed-shell RKS (spin=0); change to UKS for radicals
    * NetCharge auto-detected from phosphate protonation state
    * Pre-optimization stage off by default; opt-in for systems
      where the builder geometry is rough (long ssDNA, large
      peptides) so PBE/def2-SVP can clean it up before B3LYP runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .siesta import _validate_basename     # shared with SiestaConfig


# --------------------------------------------------------------------- #
#  Staged optimization (task #534)                                      #
# --------------------------------------------------------------------- #
#
# Per ``docs/engines/pyscf-publication-guide.md`` § "In-script staged
# optimization", every PySCF script runs up to N stages internally
# (default 3).  Each stage carries its own SCF tolerance + 5 geomeTRIC
# convergence knobs + max-step cap; the generated script's main loop
# walks ``STAGES`` and skips disabled entries, carrying ``mol``
# forward between enabled stages.  This commit lands the DATA layer
# only — the generator + form UI consume it in commit-family
# follow-ups (parse-module H4 is done; #534 commit 2 = form schema,
# commit 3 = generator rewrite).
#
# Naming convention: stages are addressed by their ``name`` field
# (default ``stage1`` / ``stage2`` / ``stage3``) so per-stage output
# files end up ``<JOB>_stage1_geom_optim.xyz``, etc.  Keep names
# ``[A-Za-z0-9_]+`` (filesystem-safe; no dots, no whitespace).
#
# Units recap (geomeTRIC source-of-truth):
#   * ``gmax`` / ``grms``  — Hartree / Bohr
#   * ``dmax`` / ``drms``  — Angstrom (NOT Bohr — long-standing
#                            geomeTRIC doc bug; the source uses Å)
#   * ``etol``             — Hartree
#   * ``conv_tol``         — Hartree (SCF energy convergence —
#                            ``mf.conv_tol``, separate from geomeTRIC)


@dataclass
class StageSpec:
    """One stage of the in-script PySCF optimization loop.

    Defaults are the geomeTRIC GAU set (publication-quality middle
    tier per the publication-guide table).  Override on construction
    to build loose / TIGHT / etc. tiers; the publication-guide tier
    table is the canonical reference.

    Per-field ``metadata`` (label / unit / help / step / range) is
    walked by ``web/blueprints/_shared.dataclass_to_form_schema``
    when it emits the ``stage-table`` schema for
    ``PySCFConfig.stages`` — the JS renderer (commit 3) uses it to
    label table columns and pick input widget kinds.
    """
    name:      str = field(default="stage1", metadata={
        "label":      "Name",
        "help":       "filename suffix (``<JOB>_<name>_geom_optim.xyz``); "
                      "must match [A-Za-z0-9_]+",
        "pattern":    r"^[A-Za-z0-9_]+$",
    })
    enabled:   bool = field(default=True, metadata={
        "label":      "Run",
        "help":       "run this stage; uncheck to skip + carry the prior "
                      "stage's geometry forward",
    })
    conv_tol:  float = field(default=1.0e-9, metadata={
        "label":      "SCF tol", "unit": "Hartree", "step": "any",
        "engine_key": "mf.conv_tol",
        "help":       "SCF energy convergence (Hartree)",
    })
    gmax:      float = field(default=4.5e-4, metadata={
        "label":      "‖F‖∞", "unit": "Ha/Bohr", "step": "any",
        "engine_key": "geomeTRIC convergence_gmax",
        "help":       "max-gradient convergence (Ha/Bohr)",
    })
    grms:      float = field(default=3.0e-4, metadata={
        "label":      "‖F‖RMS", "unit": "Ha/Bohr", "step": "any",
        "engine_key": "geomeTRIC convergence_grms",
        "help":       "RMS-gradient convergence (Ha/Bohr)",
    })
    dmax:      float = field(default=1.8e-3, metadata={
        "label":      "Δx max", "unit": "Å", "step": "any",
        "engine_key": "geomeTRIC convergence_dmax",
        "help":       "max-displacement convergence (Å)",
    })
    drms:      float = field(default=1.2e-3, metadata={
        "label":      "Δx RMS", "unit": "Å", "step": "any",
        "engine_key": "geomeTRIC convergence_drms",
        "help":       "RMS-displacement convergence (Å)",
    })
    etol:      float = field(default=1.0e-6, metadata={
        "label":      "ΔE tol", "unit": "Hartree", "step": "any",
        "engine_key": "geomeTRIC convergence_energy",
        "help":       "energy-step convergence (Hartree)",
    })
    max_steps: int = field(default=200, metadata={
        "label":      "Max steps", "range": (1, 10000),
        "engine_key": "geomeTRIC maxsteps",
        "help":       "max geomeTRIC iterations in this stage",
    })


def _default_stages() -> List[StageSpec]:
    """Three-stage publication-guide default.

    Tier 1 = loose pre-opt (enabled; cheap structure cleanup).
    Tier 2 = publishable GAU (enabled; what papers cite).
    Tier 3 = TIGHT GAU_TIGHT (DISABLED by default; opt-in for vib/
             IR/NEB where tight Hessians matter).

    Most users tick 1 + 2 and leave 3 off.  Power users override any
    knob per stage via the form (lands in #534 commit 2)."""
    return [
        StageSpec(name="stage1", enabled=True,
                  conv_tol=1.0e-7,
                  gmax=2.0e-3, grms=1.3e-3,
                  dmax=7.2e-3, drms=4.8e-3,
                  etol=1.0e-5,
                  max_steps=50),
        StageSpec(name="stage2", enabled=True,
                  conv_tol=1.0e-9,
                  gmax=4.5e-4, grms=3.0e-4,
                  dmax=1.8e-3, drms=1.2e-3,
                  etol=1.0e-6,
                  max_steps=200),
        StageSpec(name="stage3", enabled=False,
                  conv_tol=1.0e-10,
                  gmax=1.5e-5, grms=1.0e-5,
                  dmax=6.0e-5, drms=4.0e-5,
                  etol=1.0e-6,
                  max_steps=100),
    ]


def validate_stages(stages: List[StageSpec]) -> List[str]:
    """Return a list of error strings; empty == OK.

    The validator is intentionally permissive on per-knob ranges
    (publication-quality tiers span 5 orders of magnitude; clamping
    locks out legitimate power-user choices).  It only rejects what
    would corrupt the loop or produce a file-name collision:

      * Must contain at least one enabled stage (an all-disabled
        list is indistinguishable from ``optimize=False`` and would
        emit a no-op script).
      * Stage names must be filesystem-safe (``[A-Za-z0-9_]+``) and
        unique within the list (otherwise per-stage output files
        collide).
      * Every numeric knob must be strictly positive (zero or
        negative breaks geomeTRIC's convergence test).
      * ``max_steps`` must be a positive integer.
    """
    import re as _re
    errors: List[str] = []
    if not stages:
        errors.append("stages: list is empty; need at least 1 stage")
        return errors
    if not any(s.enabled for s in stages):
        errors.append(
            "stages: no stage is enabled; either enable one or set "
            "optimize=False to skip the optimization loop entirely")
    seen_names = set()
    name_re = _re.compile(r"^[A-Za-z0-9_]+$")
    for i, s in enumerate(stages):
        prefix = f"stages[{i}]"
        if not isinstance(s.name, str) or not name_re.match(s.name or ""):
            errors.append(
                f"{prefix}.name = {s.name!r}: must match [A-Za-z0-9_]+ "
                f"(used as filename suffix)")
        elif s.name in seen_names:
            errors.append(
                f"{prefix}.name = {s.name!r}: duplicate; per-stage "
                f"output files would collide")
        else:
            seen_names.add(s.name)
        for knob in ("conv_tol", "gmax", "grms", "dmax", "drms", "etol"):
            v = getattr(s, knob)
            if not isinstance(v, (int, float)) or v <= 0:
                errors.append(
                    f"{prefix}.{knob} = {v!r}: must be a positive number")
        if not isinstance(s.max_steps, int) or s.max_steps <= 0:
            errors.append(
                f"{prefix}.max_steps = {s.max_steps!r}: "
                f"must be a positive integer")
    return errors


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
    #   * Stage card   -- per-stage convergence ladder (cfg.stages,
    #                     rendered as a stage-table widget)
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
        "workflow_group": "profile",
        "section":  "System",
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
    charge: Optional[int] = field(default=None, metadata={
        "section": "System",
        # Run-profile identity — molecule's charge state.
        "workflow_group": "profile",
        "label":   "Net charge",
        "engine_key":  'gto.M(charge=...)',
        "null_label": "(auto)",
        "help": ("net charge.  Default (auto) only deduces a value "
                 "for DNA / RNA — one negative charge per backbone "
                 "phosphate.  For everything else (peptide, SMILES, "
                 "PDB load) auto resolves to 0; set this explicitly "
                 "when working with a charged species."),
    })
    spin: int = field(default=0, metadata={
        "section": "System",
        # System characteristic — open-shell chemistry, not stage.
        "workflow_group": "profile",
        "label":   "Spin (2S)",
        "engine_key":  'gto.M(spin=...)  # 2S, # of unpaired electrons',
        "range":   (0, 10),
        "help": "2S (NOT 2S+1); 0=closed shell, 1=doublet, 2=triplet, ...",
    })
    symmetry: bool = field(default=False, metadata={
        "workflow_group": "profile",
        "section": "System",
        "label":   "Use point-group symmetry",
        "engine_key":  'gto.M(symmetry=...)',
        "help": "enable point-group symmetry; faster but rarely matches "
                "builder-output geometry exactly",
    })

    # ---------------- Method (main run) ----------------
    method: str = field(default="RKS", metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "SCF method",
        "engine_key":  'RKS / UKS / RHF / UHF  (PySCF class selection)',
        "choices": ("RKS", "UKS", "RHF", "UHF"),
        "help": "RKS / UKS / RHF / UHF",
    })
    functional: str = field(default="B3LYP", metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Functional",
        "engine_key":  'mf.xc = ...',
        "help": "XC functional (e.g. B3LYP / PBE / PBE0 / M06-2X / wB97X-D)",
    })
    basis: str = field(default="def2-SVP", metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Basis set",
        "engine_key":  'gto.M(basis=...)',
        "help": "Gaussian basis set (e.g. def2-SVP / def2-TZVP / cc-pVDZ)",
    })
    # auxbasis: Python-API knob; rarely set from the form (auto-pick
    # from density_fit() is the right default).  No section -> not on form.
    auxbasis: Optional[str] = field(default=None, metadata={
        "help": "auxiliary fitting basis; None lets density_fit() auto-pick",
            "engine_key":  'df.auxbasis = ...',
    })
    density_fit: bool = field(default=True, metadata={
        "section": "Method",
        # Profile-level: method-family identity choice; SpectraConfig's
        # density_fit (config/spectra.py) is also profile.
        "workflow_group": "profile",
        "label":   "Density fitting",
        "engine_key":  'mf = mf.density_fit()',
        "help": "use density fitting (faster Coulomb/exchange evaluation)",
    })
    dispersion: Optional[str] = field(default="d3bj", metadata={
        "section": "Method",
        # Profile-level: method-family choice; SpectraConfig's
        # dispersion is also profile.  Setting once per project.
        "workflow_group": "profile",
        "label":   "Dispersion",
        "engine_key":  'mf = mf.add_dispersion(...)',
        # ``none`` is in the choices list so that the case-insensitive
        # click.Choice still accepts the disable spelling; cmd_pyscf
        # then normalises ``none`` -> None before constructing the
        # config.  (R4)
        "choices": ("d3", "d3bj", "d4", "none"),
        "help": "dispersion correction: d3 / d3bj / d4 / 'none' to disable",
    })
    # Effective Core Potential (gap #8).  None = auto: emit
    # ecp="lanl2dz" when heavy atoms (Z > 36) are present AND the
    # basis is not in the def2 family (def2-SVP / def2_SVP / def2svp
    # all bundle their own ECP).  Set to a name string ("lanl2dz",
    # "stuttgart", "def2", ...) to force a specific ECP; pass a dict
    # ({"Pt": "lanl2dz", "Au": "stuttgart"}) for per-element control.
    # Set to "" to disable auto-emit.  Per-element dicts aren't
    # accessible from the CLI; use the Python API for that.
    ecp: "str | dict | None" = field(default=None, metadata={
        "help": ("effective core potential (e.g. 'lanl2dz'); default = auto "
                 "for heavy atoms on non-def2 bases; pass 'none' to disable"),
        # The dict variant is Python-API-only (per-element ECPs); the
        # CLI surface is only the str case, plus the "none"/"" coercion
        # to "" (= explicitly disable auto-emit).  add_dataclass_options
        # would otherwise reject the str|dict|None union; cmd_pyscf
        # hand-rolls --ecp instead.
        "skip_cli": True,
            "engine_key":  'gto.M(ecp=...)',
    })

    # ---------------- SCF ----------------
    scf_conv_tol: float = field(default=1e-9, metadata={
        "section": "SCF",
        # Convergence target — tightens stage-to-stage.
        "workflow_group": "stage",
        "label": "scf.conv_tol", "unit": "Hartree",
        "engine_key":  'mf.conv_tol',
        "range": (1e-12, 1e-4),
        "tier":  "advanced",
        "help":  "SCF convergence tolerance on the energy (Hartree)",
    })
    scf_max_cycle: int = field(default=100, metadata={
        "section": "SCF",
        # Resource-budget cap — patience, not convergence definition.
        "workflow_group": "budget",
        "label": "scf.max_cycle",
        "engine_key":  'mf.max_cycle',
        "range": (10, 1000),
        "tier":  "advanced",
        "help":  "max SCF cycles per single-point",
    })
    scf_init_guess: str = field(default="minao", metadata={
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
        "help":  "0=coarse, 3=screening, 4=default (hybrid-friendly), 5=tight, 9=ultra",
    })
    level_shift: float = field(default=0.0, metadata={
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
        "label": "mf.diis_space",
        "engine_key":  'mf.diis_space',
        "range": (4, 20),
        "tier":  "advanced",
        "help":  "DIIS subspace size; bump to 12-20 for oscillating SCFs",
    })
    damp: float = field(default=0.0, metadata={
        "label": "mf.damp",
        "engine_key":  'mf.damp',
        "range": (0.0, 0.9),
        "tier":  "advanced",
        "help":  "Roothaan damping factor; 0.3-0.5 helps when DIIS alone isn't enough",
    })

    # ---------------- Main optimization ----------------
    optimize: bool = field(default=True, metadata={
        "section": "Compute & budget",
        # Profile-level: gates whether a relax happens at all --
        # run-shape identity (relax-or-single-point).
        "workflow_group": "profile",
        "label":   "Optimize geometry",
        "engine_key":  '(molbuilder: gates geomeTRIC opt() vs single-point)',
        "help": "run geometry optimization; --no-optimize for single-point only",
    })
    optimizer: str = field(default="geometric", metadata={
        "section": "Compute & budget",
        # Profile-level: optimizer family choice; parallel to SIESTA's
        # relax_type (also profile).
        "workflow_group": "profile",
        "label":   "Optimizer",
        "engine_key":  'geomeTRIC / berny  (driver selection)',
        "choices": ("geometric", "berny"),
        "help": "geomeTRIC or berny",
    })
    # 3-stage in-script optimization (task #534).  Per-stage SCF +
    # geomeTRIC convergence ladder lives here; the generator's
    # ``_emit_stages_loop`` walks the enabled stages, applies each
    # row's knobs to ``mf.conv_tol`` + the six geomeTRIC convergence
    # kwargs (gmax/grms/dmax/drms/etol/max_steps), and warm-starts
    # the next iter at the relaxed geometry.  Defaults match the
    # publication-guide tier table (stage 1 loose, stage 2
    # publishable, stage 3 TIGHT opt-in).
    stages: List[StageSpec] = field(
        default_factory=_default_stages,
        metadata={
            # ``skip_cli`` keeps ``molbuilder.cli.add_dataclass_options``
            # from trying to auto-generate a ``--stages`` Click option
            # for a List[StageSpec] (cli.py:198 bails loudly on
            # unsupported types).  CLI exposure (likely a JSON-string
            # ``--stages-json`` plus a ``--stage-strategy`` preset)
            # lands later in the #534 family.
            "skip_cli":   True,
            # Form layer (commit 2): visible in the schema as a
            # ``kind: "stage-table"`` field.  Backend round-trips a
            # list-of-dicts payload back to ``List[StageSpec]`` via
            # ``coerce_to_field_type``'s dataclass-list branch.  JS
            # renderer for the table + ``Stage strategy`` preset
            # dropdown lands in commit 3.
            "section":        "Compute & budget",
            "workflow_group": "stage",
            "id_suffix":      "stages",
            "label":          "Optimization stages",
            "tier":           "advanced",
            "engine_key":     "STAGES = [...]; for stage in STAGES: optimize(...)",
            "help": (
                "Per-stage optimization knobs.  Each row carries a "
                "name (used as the per-stage output-file suffix), an "
                "enable checkbox, an SCF tolerance, 5 geomeTRIC "
                "convergence targets (max + RMS gradient, max + RMS "
                "displacement, energy step), and a max-step cap.  "
                "Generated-script loop walks the list and skips "
                "disabled stages; ``mol`` carries forward between "
                "enabled stages.  Defaults match the publication-"
                "guide tier table (Stage 1 loose, Stage 2 "
                "publishable, Stage 3 TIGHT-opt-in)."),
        },
    )

    # ---------------- Solvent (optional) ----------------
    solvent: Optional[str] = field(default=None, metadata={
        "workflow_group": "profile",
        "section": "Solvent (optional)",
        "label":   "Solvent",
        "engine_key":  'mf = mf.PCM() / mf.SMD()',
        "null_label": "(gas phase)",
        "help": "solvent (water / methanol / dmso / chloroform / ...)",
    })
    solvent_method: str = field(default="IEF-PCM", metadata={
        "workflow_group": "profile",
        "section": "Solvent (optional)",
        "label":   "PCM model",
        "engine_key":  'pcm.method',
        "choices": ("IEF-PCM", "C-PCM", "COSMO"),
        "help": "PCM model: IEF-PCM / C-PCM / COSMO",
    })

    # ---------------- Runtime ----------------
    max_memory_mb: int = field(default=4000, metadata={
        "workflow_group": "budget",
        "section": "Compute & budget",
        "label": "max_memory", "unit": "MB",
        "engine_key":  'mol.max_memory',
        "id_suffix": "max-memory",
        "range": (100, 1_000_000),
        "tier":  "advanced",
        "help":  "MB hint for PySCF's max_memory",
    })
    threads: Optional[int] = field(default=None, metadata={
        "workflow_group": "budget",
        "section": "Compute & budget",
        "label":      "CPU threads",
        "engine_key":  "lib.num_threads(N) + os.environ['OMP_NUM_THREADS']",
        "null_label": "(auto: physical cores)",
        "help":       "how many CPU threads PySCF uses.  Default "
                      "(blank) auto-detects PHYSICAL cores (not "
                      "logical/HT) -- hyperthreading rarely helps "
                      "QC kernels and can hurt cache locality.  The "
                      "emitted script pins BLAS to 1 thread per "
                      "worker (OPENBLAS_NUM_THREADS=1, "
                      "MKL_NUM_THREADS=1) so PySCF threads * BLAS "
                      "threads don't multiply -- the canonical "
                      "anti-oversubscription recipe.  Set explicitly "
                      "to bench, or to leave cores free for other jobs.",
    })
    use_gpu: bool = field(default=False, metadata={
        "workflow_group": "budget",
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
                     "the script probes gpu4pyscf at runtime and "
                     "falls back to CPU if the package isn't "
                     "importable or the GPU is missing / too old "
                     "(compute capability < 7.0).",
    })
    verbose: int = field(default=4, metadata={
        "workflow_group": "profile",
        "section": "Compute & budget",
        "label": "PySCF verbose",
        "engine_key":  'mol.verbose',
        "range": (0, 9),
        "tier":  "advanced",
        "help":  "PySCF verbosity: 0 silent, 4 info, 5 debug",
    })
    chkfile: bool = field(default=True, metadata={
        "workflow_group": "profile",
        "section": "Compute & budget",
        "label":   "Write checkpoint (.chk)",
        "engine_key":  "mf.chkfile = '<path>'",
        "help": "write <job>.chk (DM, mol, energies for restart)",
    })
    log_file: bool = field(default=True, metadata={
        "workflow_group": "profile",
        "section": "Compute & budget",
        "label":   "Write PySCF log",
        "engine_key":  "mol.stdout = open('<path>','w')",
        "help": "write the PySCF text log to <job>.log",
    })
    # Always-on output knobs; unsectioned (no good reason to expose).
    save_optimized_xyz: bool = field(default=True, metadata={
        "help": "snapshot the relaxed geometry to <job>_optimized.xyz",
            "engine_key":  '(molbuilder: writes <job>_opt.xyz post-relax)',
    })
    save_initial_xyz: bool = field(default=True, metadata={
        "help": "snapshot the input geometry to <job>_initial.xyz",
            "engine_key":  '(molbuilder: writes <job>_init.xyz pre-relax)',
    })
    write_trajectory: bool = field(default=True, metadata={
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
        "help": ("write the additive <job>.molwatch.log (self-contained "
                 "per-step coords / energy / forces; the Watch tab's "
                 "preferred input).  Requires --optimize and "
                 "--optimizer geometric"),
            "engine_key":  '(molbuilder: writes .molwatch.log for live viewer)',
    })

    # ---------------- Frequencies / thermochemistry (post-relax) ----------------
    # Opt-in: when True, the script computes the analytic Hessian at
    # the relaxed (or, for single-point runs, the input) geometry,
    # runs PySCF's harmonic_analysis to get wavenumbers in cm^-1, and
    # passes the result through ``thermo.thermo`` for RRHO ZPE / U /
    # H / G / S / Cv / Cp at (temperature_K, pressure_atm).  The
    # summary is written to ``<job>.thermo.txt`` so the file lives
    # alongside the converged log / chkfile.
    #
    # Cost: one analytic Hessian -- typically 5-15x a single SCF for
    # small molecules, more for larger ones.  Default off so the
    # extra cost is explicit.  Imaginary modes are reported but the
    # script does not auto-perturb; the user decides whether to
    # restart the optimization along the imaginary coordinate.
    compute_frequencies: bool = field(default=False, metadata={
        "workflow_group": "profile",
        "section": "Frequencies / thermochemistry",
        "label": "Post-relax frequencies + thermochemistry",
        "engine_key":  'pyscf.hessian + thermo.thermo()',
        "tier":  "advanced",
        "help":  "compute analytic Hessian + RRHO thermochemistry "
                 "(ZPE, H, G, S, Cv, Cp) at temperature_K / pressure_atm",
    })
    temperature_K: float = field(default=298.15, metadata={
        "section": "Frequencies / thermochemistry",
        # Profile-level: standard-state thermochemistry condition;
        # paired with ``pressure_atm`` (also profile).
        "workflow_group": "profile",
        "label": "Thermochemistry temperature", "unit": "K",
        "engine_key":  'thermo.thermo(temperature=...)',
        "id_suffix": "temperature",
        "range": (0.0, 5000.0),
        "tier":  "advanced",
        "help":  "RRHO temperature for thermo.thermo() (standard: 298.15 K)",
    })
    pressure_atm: float = field(default=1.0, metadata={
        "workflow_group": "profile",
        "section": "Frequencies / thermochemistry",
        "label": "Thermochemistry pressure", "unit": "atm",
        "engine_key":  'thermo.thermo(pressure=...)',
        "id_suffix": "pressure",
        "range": (1.0e-6, 1.0e3),
        "tier":  "advanced",
        "help":  "RRHO pressure for thermo.thermo() (standard: 1 atm = 101325 Pa)",
    })

    # ---------------- Comments ----------------
    verbose_comments: bool = field(default=True, metadata={
        "workflow_group": "profile",
        "section": "Compute & budget",
        "label":   "Verbose comments in script",
        "engine_key":  '(molbuilder: script comment-block control)',
        "help": "emit inline tuning hints + troubleshooting block in the script",
    })

    # ---------------- Staged relaxation (job-layout v1) ----------------
    # When 1, 2, or 3, the inlined ``MolwatchEmitter`` writes to
    # ``<job>-stage<N>.molwatch.log`` instead of ``<job>.molwatch.log``,
    # so the three stages of a coarse->medium->tight relaxation in one
    # directory each get their own log and the Watch tab's multi-stage
    # merge picks them up automatically.  ``None`` (default) keeps the
    # unsuffixed name for single-run workflows.  ``job_name`` (the
    # protocol basename) stays identical across stages.
    stage: Optional[int] = field(default=None, metadata={
        "label": "Relaxation stage",
        "engine_key":  '(molbuilder: filename suffix + log naming)',
        "help":  "stage marker (1/2/3) for the .molwatch.log filename; "
                 "None keeps the unsuffixed name",
        "range": (1, 3),
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
