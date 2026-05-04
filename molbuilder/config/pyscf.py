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


@dataclass
class PySCFConfig:
    # ---------------- System ----------------
    job_name: str = "pyscf_relax"
    charge: Optional[int] = None        # None -> auto from phosphates
    spin: int = 0                       # PySCF convention: 2S, NOT 2S+1
    symmetry: bool = False              # True is faster but rarely matches
                                        # builder-output geometry exactly

    # ---------------- Method (main run) ----------------
    method: str = "RKS"                 # "RKS" / "UKS" / "RHF" / "UHF"
    functional: str = "B3LYP"
    basis: str = "def2-SVP"
    auxbasis: Optional[str] = None      # None -> let density_fit() pick
    density_fit: bool = True
    dispersion: Optional[str] = "d3bj"  # None / "d3" / "d3bj" / "d4"
    # Effective Core Potential (gap #8).  None = auto: emit
    # ecp="lanl2dz" when heavy atoms (Z > 36) are present AND the
    # basis is non-def2 (def2-* families bundle their own ECP).
    # Set to a name string ("lanl2dz", "stuttgart", "def2", ...) to
    # force a specific ECP; set to "" to disable auto-emit.
    ecp: Optional[str] = None

    # ---------------- Solvent (optional) ----------------
    solvent: Optional[str] = None       # None or one of _SOLVENTS keys
    solvent_method: str = "IEF-PCM"     # "IEF-PCM" / "C-PCM" / "COSMO"

    # ---------------- SCF ----------------
    scf_conv_tol: float = field(default=1e-9, metadata={
        "label": "scf.conv_tol", "unit": "Hartree",
        "range": (1e-12, 1e-4),
        "tier":  "advanced",
    })
    scf_max_cycle: int = field(default=100, metadata={
        "label": "scf.max_cycle",
        "range": (10, 1000),
        "tier":  "advanced",
    })
    scf_init_guess: str = "minao"       # "minao" / "atom" / "1e" / "huckel"
    grid_level: int = field(default=3, metadata={
        "label": "DFT grid level",
        "range": (0, 9),
        "tier":  "advanced",
        "help":  "0=coarse (rapid testing), 3=default, 5=tight, 9=ultra",
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
    preopt: bool = False
    preopt_functional: str = "PBE"
    preopt_basis: str = "def2-SVP"
    preopt_density_fit: bool = True
    preopt_dispersion: Optional[str] = None
    preopt_max_steps: int = 50
    preopt_grms: float = 1.0e-3         # Ha/Bohr; 3x looser than main

    # ---------------- Main optimization ----------------
    optimize: bool = True
    optimizer: str = "geometric"        # "geometric" or "berny"
    geom_max_steps: int = field(default=200, metadata={
        "label": "geom max steps",
        "range": (1, 10000),
        "tier":  "advanced",
    })
    geom_conv_energy: float = 1.0e-6    # Hartree
    geom_conv_grms: float = 3.0e-4      # Ha/Bohr
    geom_conv_gmax: float = 4.5e-4      # Ha/Bohr

    # ---------------- Output ----------------
    chkfile: bool = True                # write <job>.chk (DM, mol, energies)
    log_file: bool = True               # write <job>.log
    save_optimized_xyz: bool = True
    save_initial_xyz: bool = True
    write_trajectory: bool = True       # stream geomeTRIC's <job>_geom_optim.xyz
                                        # so molwatch can watch it live
    molwatch_log: bool = True           # additive: write <job>.molwatch.log with
                                        # one self-contained, marker-delimited
                                        # block per opt step (coords + energy +
                                        # forces + per-cycle SCF residuals).
                                        # Standard outputs are still emitted;
                                        # this is purely an additional file.

    # ---------------- Runtime ----------------
    max_memory_mb: int = field(default=4000, metadata={
        "label": "max_memory", "unit": "MB",
        "range": (100, 1_000_000),
        "tier":  "advanced",
    })
    threads: Optional[int] = None       # None -> inherit OMP_NUM_THREADS
    verbose: int = field(default=4, metadata={
        "label": "PySCF verbose",
        "range": (0, 9),
        "tier":  "advanced",
        "help":  "0 silent, 4 info, 5 debug",
    })

    # ---------------- Comments ----------------
    verbose_comments: bool = True       # inline tuning hints + troubleshooting


__all__ = ["PySCFConfig"]
