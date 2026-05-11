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
    # ---------------- System ----------------
    job_name: str = field(default="pyscf_relax", metadata={
        "help":     "job name; output files get this prefix.  Must match "
                    "[A-Za-z0-9_-]+ (job-layout v1; no dots).",
        # Same basename rule as SiestaConfig.system_label -- import the
        # shared helper so the regex has ONE home.
        "validate": _validate_basename("job_name"),
    })
    charge: Optional[int] = field(default=None, metadata={
        "help": "net charge (default: auto-detect from phosphates)",
    })
    spin: int = field(default=0, metadata={
        "help": "2S (NOT 2S+1); 0=closed shell, 1=doublet, 2=triplet, ...",
    })
    symmetry: bool = field(default=False, metadata={
        "help": "enable point-group symmetry; faster but rarely matches "
                "builder-output geometry exactly",
    })

    # ---------------- Method (main run) ----------------
    method: str = field(default="RKS", metadata={
        "choices": ("RKS", "UKS", "RHF", "UHF"),
        "help": "RKS / UKS / RHF / UHF",
    })
    functional: str = field(default="B3LYP", metadata={
        "help": "XC functional (e.g. B3LYP / PBE / PBE0 / M06-2X / wB97X-D)",
    })
    basis: str = field(default="def2-SVP", metadata={
        "help": "Gaussian basis set (e.g. def2-SVP / def2-TZVP / cc-pVDZ)",
    })
    auxbasis: Optional[str] = field(default=None, metadata={
        "help": "auxiliary fitting basis; None lets density_fit() auto-pick",
    })
    density_fit: bool = field(default=True, metadata={
        "help": "use density fitting (faster Coulomb/exchange evaluation)",
    })
    dispersion: Optional[str] = field(default="d3bj", metadata={
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
    })

    # ---------------- Solvent (optional) ----------------
    solvent: Optional[str] = field(default=None, metadata={
        "help": "solvent (water / methanol / dmso / chloroform / ...)",
    })
    solvent_method: str = field(default="IEF-PCM", metadata={
        "help": "PCM model: IEF-PCM / C-PCM / COSMO",
    })

    # ---------------- SCF ----------------
    scf_conv_tol: float = field(default=1e-9, metadata={
        "label": "scf.conv_tol", "unit": "Hartree",
        "range": (1e-12, 1e-4),
        "tier":  "advanced",
        "help":  "SCF convergence tolerance on the energy (Hartree)",
    })
    scf_max_cycle: int = field(default=100, metadata={
        "label": "scf.max_cycle",
        "range": (10, 1000),
        "tier":  "advanced",
        "help":  "max SCF cycles per single-point",
    })
    scf_init_guess: str = field(default="minao", metadata={
        "choices": ("minao", "atom", "1e", "huckel"),
        "help": "SCF initial guess: minao / atom / 1e / huckel",
    })
    grid_level: int = field(default=4, metadata={
        "label": "DFT grid level",
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
        "label": "Level shift", "unit": "Hartree",
        "range": (0.0, 1.0),
        "tier":  "advanced",
        "help":  "0.1-0.3 helps hard SCFs; 0 if SCF converges cleanly",
    })
    # Hard-SCF troubleshooting knobs (gap #10).  Both default to
    # PySCF's own defaults so behaviour is unchanged for the
    # easy-converge case; bump them when SCF oscillates.
    diis_space: int = field(default=8, metadata={
        "label": "mf.diis_space",
        "range": (4, 20),
        "tier":  "advanced",
        "help":  "DIIS subspace size; bump to 12-20 for oscillating SCFs",
    })
    damp: float = field(default=0.0, metadata={
        "label": "mf.damp",
        "range": (0.0, 0.9),
        "tier":  "advanced",
        "help":  "Roothaan damping factor; 0.3-0.5 helps when DIIS alone isn't enough",
    })

    # ---------------- Pre-optimization (optional warm-up) ----------------
    preopt: bool = field(default=False, metadata={
        "help": "run a cheap PBE/def2-SVP pre-opt before main run",
    })
    preopt_functional: str = field(default="PBE", metadata={
        "help": "XC functional for the pre-opt stage",
    })
    preopt_basis: str = field(default="def2-SVP", metadata={
        "help": "Gaussian basis for the pre-opt stage",
    })
    preopt_density_fit: bool = field(default=True, metadata={
        "help": "density fitting on the pre-opt SCF",
    })
    preopt_dispersion: Optional[str] = field(default=None, metadata={
        # Same choice list as ``dispersion``; cmd_pyscf normalises
        # ``none`` -> None.  (R4)
        "choices": ("d3", "d3bj", "d4", "none"),
        "help": "dispersion correction on pre-opt mf (d3 / d3bj / d4); default off",
    })
    preopt_max_steps: int = field(default=50, metadata={
        "help": "max geomeTRIC steps in the pre-opt stage",
    })
    preopt_grms: float = field(default=1.0e-3, metadata={
        "help": "pre-opt grms convergence (Ha/Bohr); 3x looser than main",
    })

    # ---------------- Main optimization ----------------
    optimize: bool = field(default=True, metadata={
        "help": "run geometry optimization; --no-optimize for single-point only",
    })
    optimizer: str = field(default="geometric", metadata={
        "choices": ("geometric", "berny"),
        "help": "geomeTRIC or berny",
    })
    geom_max_steps: int = field(default=200, metadata={
        "label": "geom max steps",
        "range": (1, 10000),
        "tier":  "advanced",
        "help":  "max optimization steps",
    })
    geom_conv_energy: float = field(default=1.0e-6, metadata={
        "help": "geomeTRIC energy convergence (Hartree)",
    })
    geom_conv_grms: float = field(default=3.0e-4, metadata={
        "help": "geomeTRIC RMS gradient convergence (Ha/Bohr)",
    })
    geom_conv_gmax: float = field(default=4.5e-4, metadata={
        "help": "geomeTRIC max-gradient convergence (Ha/Bohr)",
    })

    # ---------------- Output ----------------
    chkfile: bool = field(default=True, metadata={
        "help": "write <job>.chk (DM, mol, energies for restart)",
    })
    log_file: bool = field(default=True, metadata={
        "help": "write the PySCF text log to <job>.log",
    })
    save_optimized_xyz: bool = field(default=True, metadata={
        "help": "snapshot the relaxed geometry to <job>_optimized.xyz",
    })
    save_initial_xyz: bool = field(default=True, metadata={
        "help": "snapshot the input geometry to <job>_initial.xyz",
    })
    write_trajectory: bool = field(default=True, metadata={
        "help": ("stream geomeTRIC's <job>_geom_optim.xyz so molwatch can "
                 "watch it live"),
    })
    # Match SiestaConfig's naming (``write_molwatch_log``) so the two
    # configs read the same way.  ``molwatch_log`` is kept as a
    # back-compat property below in __post_init__ for callers passing
    # the old kwarg.  Emission also requires ``optimize=True`` AND
    # ``optimizer="geometric"`` -- the molwatch hooks ride on the
    # SCF and geomeTRIC opt-step callbacks, so a single-point or
    # berny run has nowhere to attach.  See spec
    # docs/spec/pyscf-script.md L33 for the exact gate.
    write_molwatch_log: bool = field(default=True, metadata={
        "help": ("write the additive <job>.molwatch.log (self-contained "
                 "per-step coords / energy / forces; the Watch tab's "
                 "preferred input).  Requires --optimize and "
                 "--optimizer geometric"),
    })

    # ---------------- Runtime ----------------
    max_memory_mb: int = field(default=4000, metadata={
        "label": "max_memory", "unit": "MB",
        "range": (100, 1_000_000),
        "tier":  "advanced",
        "help":  "MB hint for PySCF's max_memory",
    })
    threads: Optional[int] = field(default=None, metadata={
        "help": "OMP_NUM_THREADS pin; default = inherit env",
    })
    verbose: int = field(default=4, metadata={
        "label": "PySCF verbose",
        "range": (0, 9),
        "tier":  "advanced",
        "help":  "PySCF verbosity: 0 silent, 4 info, 5 debug",
    })

    # ---------------- Comments ----------------
    verbose_comments: bool = field(default=True, metadata={
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
