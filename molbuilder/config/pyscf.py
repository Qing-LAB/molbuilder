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
from typing import Optional

from .siesta import _validate_basename     # shared with SiestaConfig


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
    # (System -> Method -> SCF -> Pre-opt -> Solvent -> Frequencies)
    # compact and gathers all the "execution strategy + resources"
    # knobs in one section at the end.  Workflow-group cards inside
    # the new section split the merged fields cleanly:
    #   * Profile card -- optimize toggle + optimizer choice
    #   * Stage card   -- geom_conv_energy + geom_conv_grms +
    #                     geom_conv_gmax (convergence targets)
    #   * Budget card  -- geom_max_steps + max_memory_mb + threads +
    #                     use_gpu + verbose + chkfile + log_file +
    #                     verbose_comments
    _form_section_order = (
        "System",
        "Method",
        "SCF",
        "Pre-optimization (optional)",
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
        "label":   "Density fitting",
        "engine_key":  'mf = mf.density_fit()',
        "help": "use density fitting (faster Coulomb/exchange evaluation)",
    })
    dispersion: Optional[str] = field(default="d3bj", metadata={
        "section": "Method",
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

    # ---------------- Pre-optimization (optional warm-up) ----------------
    preopt: bool = field(default=False, metadata={
        "section": "Pre-optimization (optional)",
        "label":   "Enable pre-optimization",
        "engine_key":  '(molbuilder: two-stage relax workflow)',
        "help": "run a cheap PBE/def2-SVP pre-opt before main run",
    })
    preopt_functional: str = field(default="PBE", metadata={
        "section": "Pre-optimization (optional)",
        "label":   "Pre-opt functional",
        "engine_key":  'mf.xc  (in pre-opt stage)',
        "help": "XC functional for the pre-opt stage",
    })
    preopt_basis: str = field(default="def2-SVP", metadata={
        "section": "Pre-optimization (optional)",
        "label":   "Pre-opt basis",
        "engine_key":  'gto.M(basis=...)  (in pre-opt stage)',
        "help": "Gaussian basis for the pre-opt stage",
    })
    # preopt_density_fit / preopt_dispersion: kept off the form to
    # avoid clutter; they default sensibly and power users tweak via API.
    preopt_density_fit: bool = field(default=True, metadata={
        "help": "density fitting on the pre-opt SCF",
            "engine_key":  'mf.density_fit()  (in pre-opt stage)',
    })
    preopt_dispersion: Optional[str] = field(default=None, metadata={
        # Same choice list as ``dispersion``; cmd_pyscf normalises
        # ``none`` -> None.  (R4)
        "choices": ("d3", "d3bj", "d4", "none"),
        "help": "dispersion correction on pre-opt mf (d3 / d3bj / d4); default off",
            "engine_key":  'mf.add_dispersion()  (in pre-opt stage)',
    })
    preopt_max_steps: int = field(default=50, metadata={
        "section": "Pre-optimization (optional)",
        "label":   "Pre-opt max steps",
        "engine_key":  'geomeTRIC max_steps  (in pre-opt stage)',
        "range":   (1, 1000),
        "tier":    "advanced",
        "help": "max geomeTRIC steps in the pre-opt stage",
    })
    preopt_grms: float = field(default=1.0e-3, metadata={
        "section": "Pre-optimization (optional)",
        "workflow_group": "stage",
        "label":   "Pre-opt grms", "unit": "Ha/Bohr",
        "engine_key":  'geomeTRIC convergence_grms  (in pre-opt stage)',
        "tier":    "advanced",
        "help": "pre-opt grms convergence (Ha/Bohr); 3x looser than main",
    })

    # ---------------- Main optimization ----------------
    optimize: bool = field(default=True, metadata={
        "section": "Compute & budget",
        "label":   "Optimize geometry",
        "engine_key":  '(molbuilder: gates geomeTRIC opt() vs single-point)',
        "help": "run geometry optimization; --no-optimize for single-point only",
    })
    optimizer: str = field(default="geometric", metadata={
        "section": "Compute & budget",
        "label":   "Optimizer",
        "engine_key":  'geomeTRIC / berny  (driver selection)',
        "choices": ("geometric", "berny"),
        "help": "geomeTRIC or berny",
    })
    geom_max_steps: int = field(default=200, metadata={
        "section": "Compute & budget",
        # Resource-budget cap — same logic as SIESTA's relax_steps.
        # Scale with system size, not stage.
        "workflow_group": "budget",
        "label": "geom max steps",
        "engine_key":  'optimizer.max_steps',
        "range": (1, 10000),
        "tier":  "advanced",
        "help":  "max optimization steps",
    })
    geom_conv_energy: float = field(default=1.0e-6, metadata={
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label":   "geom_conv_energy", "unit": "Hartree",
        "engine_key":  'geomeTRIC convergence_energy',
        "tier":    "advanced",
        "help": "geomeTRIC energy convergence (Hartree)",
    })
    geom_conv_grms: float = field(default=3.0e-4, metadata={
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label":   "geom_conv_grms", "unit": "Ha/Bohr",
        "engine_key":  'geomeTRIC convergence_grms',
        "tier":    "advanced",
        "help": "geomeTRIC RMS gradient convergence (Ha/Bohr)",
    })
    geom_conv_gmax: float = field(default=4.5e-4, metadata={
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label":   "geom_conv_gmax", "unit": "Ha/Bohr",
        "engine_key":  'geomeTRIC convergence_gmax',
        "tier":    "advanced",
        "help": "geomeTRIC max-gradient convergence (Ha/Bohr)",
    })

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
        "help":      "run the SCF (and geom-opt forces) on an NVIDIA "
                     "GPU via the gpu4pyscf extension.  Install: "
                     "``pip install gpu4pyscf-cuda12x`` (or "
                     "-cuda13x to match your driver).  The script "
                     "probes gpu4pyscf at runtime and falls back to "
                     "CPU if the package isn't installed or the GPU "
                     "is missing / too old (compute capability < 7.0).",
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
