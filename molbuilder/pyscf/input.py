"""molbuilder.pyscf_input -- generate a runnable PySCF script for
molecule relaxation / single-point work.

Mirrors the molbuilder.siesta module:

    PySCFConfig      -- dataclass holding every parameter
    render_script    -- format an in-memory Structure as a Python script
    convert          -- read XYZ/PDB, write .py, return summary

The generated script is fully self-contained: build mole -> SCF ->
(optional) pre-optimization -> main optimization -> save outputs.
The user runs it with `python <script>.py`.

We default to B3LYP+D3BJ/def2-SVP with density fitting -- the modern
production default for organic chemistry / biomolecule work in PySCF.
The optional pre-optimization stage uses the cheaper PBE/def2-SVP to
fix bad bond lengths before the hybrid functional sees them.

Module name: this lives at ``molbuilder/pyscf_input.py`` so an
``import pyscf`` inside the generated user script is unambiguous (the
file name avoids any possibility that ``pyscf`` resolves to our local
module instead of the actual PySCF library).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..structure import Structure


# --------------------------------------------------------------------- #
#  Solvent presets (dielectric constants at 25 deg C)                  #
# --------------------------------------------------------------------- #


_SOLVENTS = {
    "water":      78.3553,
    "methanol":   32.613,
    "ethanol":    24.852,
    "acetone":    20.493,
    "dmso":       46.826,
    "thf":        7.4257,
    "chloroform": 4.7113,
    "toluene":    2.3741,
    "hexane":     1.8819,
}


# --------------------------------------------------------------------- #
#  Config                                                               #
# --------------------------------------------------------------------- #


@dataclass
class PySCFConfig:
    """All PySCF parameters in one place.

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


# --------------------------------------------------------------------- #
#  Renderer                                                             #
# --------------------------------------------------------------------- #


def _atoms_block(struct: Structure, indent: str = "    ") -> str:
    """Format atoms as PySCF's multi-line `atom=` string (Angstrom)."""
    lines = []
    for el, (x, y, z) in zip(struct.elements, struct.positions):
        lines.append(f"{indent}{el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    return "\n".join(lines)


def _resolve_charge(struct: Structure, cfg: PySCFConfig) -> int:
    """Resolve the molecule's net charge.

    Order of precedence:
      1. cfg.charge explicit override (including 0, which disables
         auto-detection).
      2. phosphate-protonation heuristic via
         :func:`molbuilder.chemistry.formal_charge_from_phosphates`.

    The heuristic only counts deprotonated phosphate non-bridging
    oxygens; charged side chains (Asp, Glu, Lys, Arg, His) are NOT
    detected and the user must override with cfg.charge for those.
    """
    if cfg.charge is not None:
        return int(cfg.charge)
    from ..chemistry import formal_charge_from_phosphates
    return formal_charge_from_phosphates(struct)


def render_script(struct: Structure,
                  config: Optional[PySCFConfig] = None) -> str:
    """Format a Structure as a runnable PySCF script (Python text).

    The result is what you'd write by hand if you knew exactly what
    every PySCF knob does -- with verbose comments turned on by
    default so you can read the file as documentation of the choices.
    """
    cfg = config or PySCFConfig()
    charge = _resolve_charge(struct, cfg)
    method_class = cfg.method.upper()
    if method_class not in ("RKS", "UKS", "RHF", "UHF"):
        raise ValueError(
            f"unsupported method {cfg.method!r}; "
            f"expected RKS/UKS/RHF/UHF"
        )
    # PySCF's RKS / RHF assume closed-shell (mol.spin == 0).  Setting
    # spin != 0 with a restricted method is silently wrong physics:
    # PySCF will raise at SCF-time, but only after the user has
    # invoked Python.  Catch it at script-generation time instead.
    if method_class in ("RKS", "RHF") and cfg.spin != 0:
        raise ValueError(
            f"method={cfg.method!r} (restricted) is incompatible with "
            f"spin={cfg.spin} (which is 2S, the # unpaired electrons). "
            f"For an open-shell system, switch to method='UKS' "
            f"(or 'UHF') and keep your spin value."
        )
    is_dft = method_class.endswith("KS")
    label = cfg.job_name
    v = cfg.verbose_comments

    # ---------- pre-emission validation (Phase 2.6) ----------
    # PySCF doesn't have a meaningful cell here (the script builds a
    # gas-phase or PCM-solvent molecule), so we skip the cell-side
    # checks and run only the structure / config-side validators.
    # Warnings print to stderr; errors raise ValidationError before
    # any script text is emitted.
    from ..validation import validate, report
    report(validate(struct, cfg))

    out: List[str] = []
    # ------------------------------------------------------------- header
    summary_line = (f"{struct.n_atoms} atoms, charge={charge:+d}, "
                    f"spin={cfg.spin} (2S)")
    out.append('"""PySCF input script generated by molbuilder.')
    out.append("")
    out.append(f"System    : {struct.title or 'untitled'}")
    out.append(f"Atoms     : {summary_line}")
    out.append(f"Method    : {method_class} / "
               f"{cfg.functional if is_dft else 'HF'}")
    out.append(f"Basis     : {cfg.basis}")
    if cfg.preopt:
        out.append(f"Pre-opt   : {cfg.preopt_functional} / {cfg.preopt_basis} "
                   f"({cfg.preopt_max_steps} steps max, looser tol)")
    if cfg.optimize:
        out.append(f"Optimizer : {cfg.optimizer} "
                   f"(maxsteps={cfg.geom_max_steps}, "
                   f"grms={cfg.geom_conv_grms:.0e} Ha/Bohr)")
    if cfg.solvent:
        out.append(f"Solvent   : {cfg.solvent} ({cfg.solvent_method}, "
                   f"eps={_SOLVENTS.get(cfg.solvent, '?')})")
    out.append("")
    out.append("Run with:")
    out.append(f"    python {label}.py")
    out.append("")
    out.append("Outputs:")
    if cfg.log_file:
        out.append(f"    {label}.log              -- pyscf verbose log")
    if cfg.chkfile:
        out.append(f"    {label}.chk              -- checkpoint (DM, mol)")
    if cfg.save_initial_xyz:
        out.append(f"    {label}_initial.xyz      -- input coordinates")
    if cfg.save_optimized_xyz and cfg.optimize:
        out.append(f"    {label}_optimized.xyz    -- final relaxed coords")
    if cfg.optimize and cfg.write_trajectory and cfg.optimizer == "geometric":
        if cfg.preopt:
            out.append(f"    {label}_preopt_optim.xyz -- pre-opt streaming trajectory")
            out.append(f"    {label}_preopt.log       -- geomeTRIC's pre-opt log")
        out.append(f"    {label}_geom_optim.xyz   -- main streaming trajectory")
        out.append("                                  (multi-frame XYZ, one frame")
        out.append("                                   per step; readable live by")
        out.append("                                   molwatch).")
        out.append(f"    {label}_geom.log         -- geomeTRIC's main-opt log")
    if cfg.optimize and cfg.molwatch_log and cfg.optimizer == "geometric":
        out.append(f"    {label}.molwatch.log     -- unified per-step log: marker-")
        out.append("                                  delimited blocks containing")
        out.append("                                  coords, energy (eV), forces")
        out.append("                                  (eV/Ang), and SCF cycle history.")
        out.append("                                  Single-file input for molwatch.")
    out.append("")
    out.append("Dependencies:")
    out.append("    pip install pyscf")
    if cfg.optimize and cfg.optimizer == "geometric":
        out.append("    pip install geometric           # optimizer")
    if cfg.optimize and cfg.optimizer == "berny":
        out.append("    pip install pyberny             # optimizer")
    if cfg.dispersion or (cfg.preopt and cfg.preopt_dispersion):
        out.append("    pip install pyscf-dispersion    # D3/D3BJ/D4 corrections")
    out.append("")
    out.append("Or in one shot (full molbuilder runtime stack):")
    out.append("    pip install -r requirements-runtime.txt")
    out.append("    # or:  pip install 'molbuilder[runtime]'")
    out.append('"""')
    out.append("")

    # ------------------------------------------------------------- imports
    out.append("import os")
    out.append("import time")
    out.append("")
    if cfg.threads is not None:
        if v:
            out.append("# Pin BLAS thread count BEFORE importing pyscf so")
            out.append("# the BLAS/OpenMP runtime sees this preference at init.")
            out.append("# Set in os.environ rather than via pyscf.lib.num_threads()")
            out.append("# because some BLAS libraries (MKL, OpenBLAS) only honour")
            out.append("# OMP_NUM_THREADS when read at process start.")
        out.append(f'os.environ.setdefault("OMP_NUM_THREADS", "{cfg.threads}")')
        out.append(f'os.environ.setdefault("MKL_NUM_THREADS", "{cfg.threads}")')
        out.append("")
    out.append("from pyscf import gto, scf, dft")
    if cfg.optimize:
        if cfg.optimizer == "geometric":
            opt_pkg = "geometric"
            opt_module = "pyscf.geomopt.geometric_solver"
        elif cfg.optimizer == "berny":
            opt_pkg = "pyberny"
            opt_module = "pyscf.geomopt.berny_solver"
        else:
            raise ValueError(
                f"unknown optimizer {cfg.optimizer!r}; "
                f"expected 'geometric' or 'berny'"
            )
        # Wrap the optimizer import in a try/except so missing-dep gives
        # a one-line actionable message instead of a 6-frame traceback.
        out.append("try:")
        out.append(f"    from {opt_module} import optimize")
        out.append("except ImportError as _exc:")
        out.append("    raise SystemExit(")
        out.append(f'        "molbuilder PySCF script needs the {opt_pkg} '
                   'optimizer package.\\n"')
        out.append(f'        "Install with:  pip install {opt_pkg}\\n"')
        out.append('        "Or:  pip install -r requirements-runtime.txt\\n"')
        out.append('        f"(import error: {_exc})"')
        out.append("    )")
    if cfg.solvent:
        out.append("from pyscf.solvent import pcm")
    out.append("")
    out.append("t0 = time.time()")
    out.append(f'JOB = "{label}"')
    out.append("")

    # ---- _save_xyz helper, defined EARLY so _initial.xyz can be
    # captured *before* any optimization mutates `mol` ------------
    if cfg.save_initial_xyz or cfg.save_optimized_xyz:
        out += _emit_save_helper(v)

    # ------------------------------------------------------------- molecule
    if v:
        out.append("# ============================================================")
        out.append("#  1. Build the molecule")
        out.append("# ============================================================")
        out.append("# spin = 2S, NOT 2S+1.  Closed shell -> spin = 0.  Doublet ->")
        out.append("# spin = 1, triplet -> spin = 2 (with method='UKS').")
        out.append("# charge is auto-detected from phosphate protonation when")
        out.append("# molbuilder builds DNA/RNA; override by setting cfg.charge.")
        out.append("# symmetry=True can give a 2-10x speedup for symmetric systems")
        out.append("# but is sensitive to numerical drift in builder geometry --")
        out.append("# leave False unless you know your atoms sit on the symmetry")
        out.append("# elements exactly.")
        out.append("# max_memory is a soft hint to PySCF in MB; raise for big jobs.")
    out.append("mol = gto.M(")
    out.append("    atom = '''")
    out.append(_atoms_block(struct))
    out.append("    ''',")
    out.append(f'    basis      = "{cfg.basis}",')
    out.append(f"    charge     = {charge},")
    out.append(f"    spin       = {cfg.spin},")
    out.append(f"    symmetry   = {cfg.symmetry},")
    out.append(f"    verbose    = {cfg.verbose},")
    if cfg.log_file:
        out.append('    output     = JOB + ".log",')
    out.append(f"    max_memory = {cfg.max_memory_mb},   # MB")
    out.append("    unit       = 'Ang',")
    out.append(")")
    out.append('print(f"Built mol: {mol.natm} atoms, {mol.nelectron} electrons, '
               f'charge={charge:+d}")')
    # Capture the user's actual input geometry NOW, before pre-opt
    # has a chance to modify it.  Otherwise _initial.xyz would end up
    # being the post-pre-opt geometry (since later we set mol = mol_pre).
    if cfg.save_initial_xyz:
        if v:
            out.append("# Snapshot the input geometry before any optimization runs.")
        out.append('_save_xyz(mol, JOB + "_initial.xyz", "Initial geometry (input)")')
    out.append("")

    # ------------------------------------------------------------- preopt
    if cfg.preopt and cfg.optimize:
        out += _emit_preopt_block(cfg, charge, v)

    # ------------------------------------------------------------- main scf
    out.append("# ============================================================")
    if cfg.preopt and cfg.optimize:
        out.append("#  3. Main run -- production functional / basis")
    else:
        out.append("#  2. SCF setup")
    out.append("# ============================================================")
    if v:
        out.append("# DFT functional: B3LYP is the modern default for organic /")
        out.append("# biomolecule chemistry.  PBE0 is a faster hybrid alternative;")
        out.append("# wB97X-D / wB97M-V are more accurate range-separated hybrids.")
        out.append("# Pure GGAs (PBE, BLYP) are 2-3x faster than hybrids but miss")
        out.append("# self-interaction-error effects (band gaps, anion binding).")
        out.append("#")
        out.append("# Density fitting (RIJK / RIJ) is essentially free accuracy:")
        out.append("# usually the SCF iteration cost drops 5-10x with negligible")
        out.append("# total-energy error (< 0.1 kcal/mol for organic systems).")
        out.append("#")
        out.append("# Dispersion (D3BJ / D4) is also nearly free and matters")
        out.append("# greatly for biomolecules (stacking, vdW, hydrogen bonds).")
    out.append(f"mf = {('dft' if is_dft else 'scf')}.{method_class}(mol)")
    if is_dft:
        out.append(f'mf.xc = "{cfg.functional}"')
        out.append(f"mf.grids.level = {cfg.grid_level}")
    if cfg.density_fit:
        if cfg.auxbasis:
            out.append(f'mf = mf.density_fit(auxbasis="{cfg.auxbasis}")')
        else:
            out.append("mf = mf.density_fit()")
    if cfg.dispersion and is_dft:
        out.append(f'mf.disp = "{cfg.dispersion}"')
    if cfg.solvent:
        eps = _SOLVENTS.get(cfg.solvent.lower())
        if eps is None:
            raise ValueError(
                f"unknown solvent {cfg.solvent!r}; "
                f"valid: {sorted(_SOLVENTS)}"
            )
        out.append("# PCM solvation -- continuum model (cheaper than ddCOSMO).")
        out.append("mf = pcm.PCM(mf)")
        out.append(f'mf.with_solvent.method = "{cfg.solvent_method}"')
        out.append(f"mf.with_solvent.eps = {eps}    "
                   f"# {cfg.solvent} dielectric")
    out.append("")
    if v:
        out.append("# SCF convergence parameters.  Tighten if forces look noisy")
        out.append("# (1e-10) or DFT energies drift across geometry steps.")
        out.append("# level_shift > 0 helps for hard cases (open-shell metals,")
        out.append("# diffuse anions); 0.1-0.3 is typical when needed.")
    out.append(f"mf.conv_tol  = {cfg.scf_conv_tol:.0e}")
    out.append(f"mf.max_cycle = {cfg.scf_max_cycle}")
    out.append(f'mf.init_guess = "{cfg.scf_init_guess}"')
    if cfg.level_shift:
        out.append(f"mf.level_shift = {cfg.level_shift}")
    if cfg.chkfile:
        out.append('mf.chkfile = JOB + ".chk"')
    out.append("")

    # ---------------- Unified molwatch log emitter (optional, additive) ---
    # Defined here so its hooks are wired BEFORE optimize() runs but AFTER
    # mf has its final form (post-pre-opt rebind, density fitting, etc.).
    if cfg.optimize and cfg.molwatch_log and cfg.optimizer == "geometric":
        out += _emit_molwatch_emitter(v)

    # ------------------------------------------------------------- run
    if cfg.optimize:
        if v:
            out.append("# ============================================================")
            out.append("#  4. Geometry optimization")
            out.append("# ============================================================")
            out.append("# geomeTRIC is the recommended optimizer (translation-")
            out.append("# rotation-invariant internal coords, robust on large")
            out.append("# steps).  Berny is built into PySCF, fewer dependencies,")
            out.append("# but less robust on flexible biomolecules.")
            out.append("#")
            out.append("# Convergence thresholds (Ha, Ha/Bohr):")
            out.append("#   energy 1e-6     standard")
            out.append("#   grms   3e-4     Ha/Bohr  (~ 0.015 eV/Ang)")
            out.append("#   gmax   4.5e-4   Ha/Bohr  (~ 0.023 eV/Ang)")
            out.append("# Loosen by 3-10x for screening; tighten 10x for phonons.")
        out.append('print("\\n=== Stage: production optimization ===")')
        out.append("mol_eq = optimize(")
        out.append("    mf,")
        out.append(f"    maxsteps              = {cfg.geom_max_steps},")
        out.append(f"    convergence_energy    = {cfg.geom_conv_energy:.1e},")
        out.append(f"    convergence_grms      = {cfg.geom_conv_grms:.1e},")
        out.append(f"    convergence_gmax      = {cfg.geom_conv_gmax:.1e},")
        if cfg.write_trajectory and cfg.optimizer == "geometric":
            if v:
                out.append("    # geomeTRIC writes a multi-frame XYZ to")
                out.append("    #     <JOB>_geom_optim.xyz")
                out.append("    # with one frame per accepted step.  molwatch")
                out.append("    # tails this file live, frame-by-frame.")
            out.append('    prefix                = JOB + "_geom",')
        if cfg.molwatch_log and cfg.optimizer == "geometric":
            if v:
                out.append("    # callback fires once per accepted opt step;")
                out.append("    # _molwatch.opt_step_hook flushes one block to")
                out.append("    # <JOB>.molwatch.log with coords/energy/forces +")
                out.append("    # the SCF cycles captured since the previous step.")
            out.append("    callback              = _molwatch.opt_step_hook,")
        out.append(")")
        out.append('print(f"Final energy: {mf.e_tot:.8f} Hartree")')
    else:
        if v:
            out.append("# ============================================================")
            out.append("#  4. Single-point SCF (no optimization)")
            out.append("# ============================================================")
        out.append("e = mf.kernel()")
        out.append('print(f"Total energy: {e:.8f} Hartree")')
        out.append("mol_eq = mol")
    out.append("")

    # ------------------------------------------------------------- save
    # _save_xyz is defined early in the script (before mol is built),
    # and _initial.xyz was captured immediately after gto.M().  Here
    # we only write the FINAL geometry.
    if cfg.save_optimized_xyz and cfg.optimize:
        out.append("")
        out.append('_save_xyz(mol_eq, JOB + "_optimized.xyz", '
                   '"Optimized geometry (PySCF)")')
    out.append("")
    out.append('print(f"\\nJob complete in {time.time() - t0:.1f} s")')

    # ------------------------------------------------------------- hints
    if v:
        out += _emit_troubleshooting_block(cfg)

    return "\n".join(out) + "\n"


def _emit_preopt_block(cfg: PySCFConfig, charge: int, v: bool) -> List[str]:
    """Stage 1: cheap geometry warm-up before the production functional."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  2. Pre-optimization  (cheap warm-up: PBE/def2-SVP)")
    out.append("# ============================================================")
    if v:
        out.append("# Why pre-optimize: hybrid functionals (B3LYP, M06-2X) are")
        out.append("# much more sensitive to bad starting geometry than pure GGAs")
        out.append("# (PBE).  A handful of cheap PBE/def2-SVP geometry steps fix")
        out.append("# the worst bond-length errors from the builder, so the main")
        out.append("# stage starts from a clean structure.")
        out.append("# This typically costs 5-15% of the production-stage time.")
        out.append("#")
        out.append("# We don't fully converge here -- the looser grms tolerance")
        out.append("# (1e-3 Ha/Bohr vs 3e-4 in main) means we stop as soon as the")
        out.append("# geometry is reasonable.")
    out.append('print("\\n=== Stage: pre-optimization ===")')
    out.append("mol_pre = mol.copy()")
    out.append(f'mol_pre.basis = "{cfg.preopt_basis}"')
    # dump_input=False: the original mol.build() already echoed the
    # input file into <JOB>.log; we don't need a second copy.
    out.append("mol_pre.build(dump_input=False)")
    # Pre-opt always uses DFT (a hybrid + cheap basis is the point of
    # having a warm-up).  Mirror RKS/UKS choice from the production run.
    out.append(f'mf1 = dft.{cfg.method.upper().replace("HF", "KS")}(mol_pre)')
    out.append(f'mf1.xc = "{cfg.preopt_functional}"')
    if cfg.preopt_density_fit:
        out.append("mf1 = mf1.density_fit()")
    if cfg.preopt_dispersion:
        out.append(f'mf1.disp = "{cfg.preopt_dispersion}"')
    out.append(f"mf1.conv_tol  = {cfg.scf_conv_tol:.0e}")
    out.append(f"mf1.max_cycle = {cfg.scf_max_cycle}")
    out.append("")
    out.append("mol_pre = optimize(")
    out.append("    mf1,")
    out.append(f"    maxsteps          = {cfg.preopt_max_steps},")
    out.append(f"    convergence_grms  = {cfg.preopt_grms:.1e},")
    if v:
        out.append("    # assert_convergence=False so a partial pre-opt (which is")
        out.append("    # GOOD ENOUGH by design) doesn't kill the production run.")
        out.append("    # Production-stage optimize() keeps assert_convergence=True")
        out.append("    # because there we DO want to know if the run failed.")
    out.append("    assert_convergence = False,")
    if cfg.write_trajectory and cfg.optimizer == "geometric":
        if v:
            out.append("    # Pre-opt has its own trajectory file:")
            out.append("    #   <JOB>_preopt_optim.xyz")
            out.append("    # so molwatch can watch either stage live.")
        out.append('    prefix            = JOB + "_preopt",')
    out.append(")")
    out.append('print("Pre-opt done; carrying optimised geometry into the main run.")')
    out.append("")
    if v:
        out.append("# Reuse mol_pre as the production-stage molecule.  This is")
        out.append("# important: a fresh `gto.M(..., output=JOB+'.log')` call")
        out.append("# would open the .log in 'w' mode and TRUNCATE the pre-opt")
        out.append("# log entries we just wrote.  By reusing mol_pre we keep")
        out.append("# the same open file handle (mol_pre.stdout) so production-")
        out.append("# stage SCFs append cleanly to the existing log.")
    out.append("mol = mol_pre")
    if cfg.basis != cfg.preopt_basis:
        if v:
            out.append("# Production basis differs from pre-opt; rebuild internals.")
            out.append("# dump_input=False so we don't echo the input file a 3rd time.")
        out.append(f'mol.basis = "{cfg.basis}"')
        out.append("mol.build(dump_input=False)")
    out.append("")
    return out


def _emit_molwatch_emitter(v: bool) -> List[str]:
    """Inline streaming writer for ``<JOB>.molwatch.log``.

    The emitter is wired in two places:

      * ``mf.callback = _molwatch.scf_cycle_hook``  (per SCF cycle)
      * ``optimize(..., callback=_molwatch.opt_step_hook)`` (per opt step)

    Each opt step it flushes one block to the unified log.  Block layout
    is heavily marker-based so a downstream parser can locate every
    field by string match -- no positional fragility:

        ==== molwatch step <N> begin ====
        step_index: <N>
        n_atoms:    <K>
        coordinates (Ang):
           <element>   <x>   <y>   <z>
           ...
        energy (eV): <E>
        forces (eV/Ang):
           <element>  <fx>  <fy>  <fz>
           ...
        max_force (eV/Ang): <Fmax>
        scf_history begin
        #  cycle    energy(eV)    delta_E(eV)    gnorm(eV/Ang)    ddm
              ...
        scf_history end
        ==== molwatch step <N> end ====

    Tolerant to truncation: a torn final block (begin without end) is
    dropped on parse, so molwatch can tail a still-running job.

    Standard PySCF / geomeTRIC outputs (<JOB>.log, <JOB>_geom_optim.xyz,
    <JOB>_geom.qdata.txt) are still written -- this file is purely
    additive.  If the user prefers not to have it, set
    ``cfg.molwatch_log = False`` at script-generation time.
    """
    out: List[str] = []
    out.append("")
    out.append("# ============================================================")
    out.append("#  Unified molwatch log emitter (additive, single-file view)")
    out.append("# ============================================================")
    if v:
        out.append("# This block defines a small helper that writes one self-")
        out.append("# contained, marker-delimited record per accepted opt step")
        out.append("# to <JOB>.molwatch.log.  molwatch reads this file directly")
        out.append("# (no sibling-file discovery needed) and shows trajectory +")
        out.append("# energy + force + per-cycle SCF residual plots.")
        out.append("#")
        out.append("# All standard PySCF/geomeTRIC outputs are kept untouched;")
        out.append("# this is purely additional.  Disable via cfg.molwatch_log =")
        out.append("# False at generation time if you don't want it.")
    out.append("import time as _mw_time")
    out.append("import numpy as _mw_np")
    out.append("")
    out.append("class _MolwatchEmitter:")
    out.append('    """Streams <JOB>.molwatch.log with one marker-delimited')
    out.append("    block per opt step.  See molbuilder spec for the format.")
    out.append('    """')
    out.append("    HARTREE_TO_EV          = 27.211386245988")
    out.append("    HARTREE_BOHR_TO_EV_ANG = 51.42208619")
    out.append("")
    out.append("    def __init__(self, path, job, mol):")
    out.append("        self.path = path")
    out.append("        self.job  = job")
    out.append("        self._scf_buf   = []   # per-cycle dicts; reset each new SCF")
    out.append("        self._step      = 0    # log block counter; step 0 reserved for preview")
    out.append("        with open(self.path, 'w') as fh:")
    out.append('            fh.write("# molwatch trajectory log v1\\n")')
    out.append('            fh.write("# generator: molbuilder/pyscf_input\\n")')
    out.append('            fh.write("# engine: pyscf\\n")')
    out.append('            fh.write(f"# job: {self.job}\\n")')
    out.append('            fh.write("# units: energy=eV, force=eV/Ang, coords=Ang\\n")')
    out.append('            fh.write(f"# created: {_mw_time.strftime(\'%Y-%m-%dT%H:%M:%S\')}\\n")')
    out.append('            fh.write("\\n")')
    out.append("        # Step 0: initial-state preview, written BEFORE any SCF runs.")
    out.append("        # Carries coordinates only; energy / forces / scf_history are")
    out.append("        # null because none have been computed yet.  This guarantees")
    out.append("        # molwatch can render the molecule the moment a user loads the")
    out.append("        # log -- they don't have to wait for the first SCF to finish.")
    out.append("        self._write_initial_preview(mol)")
    out.append("")
    out.append("    def _write_initial_preview(self, mol):")
    out.append("        coords_A = mol.atom_coords(unit='Ang')")
    out.append("        elements = [mol.atom_symbol(i) for i in range(mol.natm)]")
    out.append("        idx = self._step")
    out.append("        with open(self.path, 'a') as fh:")
    out.append('            fh.write(f"==== molwatch step {idx} begin ====\\n")')
    out.append('            fh.write(f"step_index: {idx}\\n")')
    out.append('            fh.write("kind: initial_preview\\n")')
    out.append('            fh.write(f"n_atoms: {mol.natm}\\n")')
    out.append('            fh.write("coordinates (Ang):\\n")')
    out.append("            for i, el in enumerate(elements):")
    out.append("                x, y, z = coords_A[i]")
    out.append('                fh.write(f"   {el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\\n")')
    out.append('            fh.write("energy (eV): None\\n")')
    out.append('            fh.write("forces (eV/Ang):\\n")')
    out.append('            fh.write("max_force (eV/Ang): None\\n")')
    out.append('            fh.write("scf_history begin\\n")')
    out.append('            fh.write("scf_history end\\n")')
    out.append('            fh.write(f"==== molwatch step {idx} end ====\\n")')
    out.append('            fh.write("\\n")')
    out.append("            fh.flush()")
    out.append("        self._step += 1")
    out.append("")
    out.append("    # ----- SCF cycle hook (wired to mf.callback) -----")
    out.append("    def scf_cycle_hook(self, envs):")
    out.append("        cycle = envs.get('cycle', None)        # 0-indexed in PySCF")
    out.append("        if cycle is None:")
    out.append("            return")
    out.append("        if cycle == 0:")
    out.append("            # New SCF run starts: clear cycle buffer")
    out.append("            self._scf_buf = []")
    out.append("        e_tot     = envs.get('e_tot', None)")
    out.append("        last_e    = envs.get('last_hf_e', None)")
    out.append("        norm_gorb = envs.get('norm_gorb', None)")
    out.append("        norm_ddm  = envs.get('norm_ddm', None)")
    out.append("        if e_tot is None:")
    out.append("            return")
    out.append("        e_eV    = float(e_tot)  * self.HARTREE_TO_EV")
    out.append("        dE_eV   = (float(e_tot) - float(last_e)) * self.HARTREE_TO_EV \\")
    out.append("                  if last_e is not None else 0.0")
    out.append("        g_eV_A  = (float(norm_gorb) * self.HARTREE_BOHR_TO_EV_ANG) \\")
    out.append("                  if norm_gorb is not None else None")
    out.append("        ddm     = float(norm_ddm) if norm_ddm is not None else None")
    out.append("        self._scf_buf.append({")
    out.append("            'cycle':   int(cycle) + 1,        # 1-indexed in our log")
    out.append("            'energy':  e_eV,")
    out.append("            'delta_E': dE_eV,")
    out.append("            'gnorm':   g_eV_A,")
    out.append("            'ddm':     ddm,")
    out.append("        })")
    out.append("")
    out.append("    # ----- opt step hook (wired to optimize(callback=...)) -----")
    out.append("    def opt_step_hook(self, envs):")
    out.append("        mol      = envs.get('mol')")
    out.append("        energy   = envs.get('energy')")
    out.append("        gradient = envs.get('gradients')")
    out.append("        if mol is None or energy is None or gradient is None:")
    out.append("            return")
    out.append("        coords_A = mol.atom_coords(unit='Ang')")
    out.append("        elements = [mol.atom_symbol(i) for i in range(mol.natm)]")
    out.append("        e_eV     = float(energy) * self.HARTREE_TO_EV")
    out.append("        F        = -_mw_np.asarray(gradient).reshape(-1, 3) \\")
    out.append("                      * self.HARTREE_BOHR_TO_EV_ANG  # eV/Ang")
    out.append("        f_mag    = _mw_np.sqrt((F * F).sum(axis=1))")
    out.append("        max_f    = float(f_mag.max()) if f_mag.size else 0.0")
    out.append("        scf      = list(self._scf_buf)")
    out.append("        idx      = self._step")
    out.append("        with open(self.path, 'a') as fh:")
    out.append('            fh.write(f"==== molwatch step {idx} begin ====\\n")')
    out.append('            fh.write(f"step_index: {idx}\\n")')
    out.append('            fh.write(f"n_atoms: {mol.natm}\\n")')
    out.append('            fh.write("coordinates (Ang):\\n")')
    out.append("            for i, el in enumerate(elements):")
    out.append("                x, y, z = coords_A[i]")
    out.append('                fh.write(f"   {el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\\n")')
    out.append('            fh.write(f"energy (eV): {e_eV:.8f}\\n")')
    out.append('            fh.write("forces (eV/Ang):\\n")')
    out.append("            for i, el in enumerate(elements):")
    out.append("                fx, fy, fz = F[i]")
    out.append('                fh.write(f"   {el:<2s}  {fx:14.8f}  {fy:14.8f}  {fz:14.8f}\\n")')
    out.append('            fh.write(f"max_force (eV/Ang): {max_f:.8f}\\n")')
    out.append('            fh.write("scf_history begin\\n")')
    out.append('            fh.write("#  cycle      energy(eV)         delta_E(eV)        gnorm(eV/Ang)            ddm\\n")')
    out.append("            for c in scf:")
    out.append("                g_str = (f\"{c['gnorm']:.8e}\" if c['gnorm'] is not None")
    out.append("                         else 'None')")
    out.append("                d_str = (f\"{c['ddm']:.8e}\" if c['ddm'] is not None")
    out.append("                         else 'None')")
    out.append("                fh.write(")
    out.append('                    f"   {c[\'cycle\']:5d}   {c[\'energy\']:18.8f}'
               '  {c[\'delta_E\']:18.8f}  {g_str:>20s}  {d_str:>16s}\\n"')
    out.append("                )")
    out.append('            fh.write("scf_history end\\n")')
    out.append('            fh.write(f"==== molwatch step {idx} end ====\\n")')
    out.append('            fh.write("\\n")')
    out.append("            fh.flush()")
    out.append("        self._step += 1")
    out.append("")
    out.append('_molwatch = _MolwatchEmitter(JOB + ".molwatch.log", JOB, mol)')
    out.append("mf.callback = _molwatch.scf_cycle_hook")
    out.append("")
    return out


def _emit_save_helper(v: bool) -> List[str]:
    """Inline XYZ writer that doesn't require ase / pyscf.tools."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  5. Save outputs")
    out.append("# ============================================================")
    if v:
        out.append("# Inline XYZ writer (Angstrom).  Avoids depending on ase or")
        out.append("# pyscf.tools.molden, both of which add startup cost.")
    out.append("def _save_xyz(mol_obj, path, comment='generated by molbuilder'):")
    out.append("    coords = mol_obj.atom_coords(unit='Ang')")
    out.append("    with open(path, 'w') as fh:")
    out.append('        fh.write(f"{mol_obj.natm}\\n{comment}\\n")')
    out.append("        for i in range(mol_obj.natm):")
    out.append("            sym = mol_obj.atom_symbol(i)")
    out.append("            x, y, z = coords[i]")
    out.append('            fh.write(f"{sym:<2s}  {x:14.8f}  '
               '{y:14.8f}  {z:14.8f}\\n")')
    out.append("    print(f'Wrote {path}')")
    out.append("")
    return out


def _emit_troubleshooting_block(cfg: PySCFConfig) -> List[str]:
    out: List[str] = []
    out.append("")
    out.append("# ============================================================")
    out.append("# TROUBLESHOOTING / TUNING HINTS")
    out.append("# ============================================================")
    out.append("#")
    out.append("# SCF won't converge:")
    out.append("#   * mf.level_shift = 0.2          (Hartree, lifts virtuals)")
    out.append("#   * mf.max_cycle = 300")
    out.append("#   * mf.init_guess = 'atom'        (more diffuse start)")
    out.append("#   * mf.diis_space = 12            (default 8)")
    out.append("#   * mf.damp = 0.3                 (start, then 0)")
    out.append("#")
    out.append("# Open-shell / radical:")
    out.append("#   * cfg.method='UKS', cfg.spin=N  (N = 2S, # unpaired electrons)")
    out.append("#   * after SCF: mf.stability_analysis()")
    out.append("#")
    out.append("# Forces look noisy / anisotropic:")
    out.append("#   * cfg.grid_level = 5            (denser DFT grid)")
    out.append("#   * cfg.scf_conv_tol = 1e-10      (tighter SCF)")
    out.append("#")
    out.append("# Job too slow:")
    out.append("#   * cfg.basis = 'def2-SVP' (already)")
    out.append("#   * cfg.density_fit = True (already)")
    out.append("#   * functional = 'PBE'            (pure GGA, 2-3x faster)")
    out.append("#   * raise OMP_NUM_THREADS / MKL_NUM_THREADS")
    out.append("#")
    out.append("# Geometry optimization oscillates:")
    out.append("#   * cfg.geom_max_steps += 100")
    out.append("#   * cfg.geom_conv_grms = 1e-3     (looser)")
    out.append("#   * Switch optimizer 'geometric' -> 'berny' for stiff systems")
    out.append("#")
    out.append("# Charged / open-shell anions need diffuse functions:")
    out.append("#   * cfg.basis = 'aug-cc-pVDZ' or 'def2-SVPD'")
    out.append("#   * cfg.scf_conv_tol = 1e-10")
    return out


# --------------------------------------------------------------------- #
#  File-level convenience wrapper                                       #
# --------------------------------------------------------------------- #


def convert(input_path: str,
            py_path: str,
            config: Optional[PySCFConfig] = None) -> dict:
    """Read an XYZ or PDB, write a runnable PySCF script.

    Returns a summary dict: ``{py, n_atoms, charge, label}``.
    """
    cfg = config or PySCFConfig()
    p = Path(input_path)
    ext = p.suffix.lower()
    if ext == ".pdb":
        struct = Structure.from_pdb(p)
    elif ext in (".xyz", ""):
        struct = Structure.from_xyz(p)
    else:
        raise ValueError(
            f"unsupported input extension {ext!r}; expected .xyz or .pdb"
        )
    text = render_script(struct, cfg)
    out_p = Path(py_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(text)
    return {
        "py":      str(out_p),
        "n_atoms": struct.n_atoms,
        "charge":  _resolve_charge(struct, cfg),
        "label":   cfg.job_name,
    }
