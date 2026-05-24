"""molbuilder.pyscf.input -- generate a runnable PySCF script for
molecule relaxation / single-point work.

Mirrors the molbuilder.siesta.input module:

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

Module name: this lives at ``molbuilder/pyscf/input.py`` so an
``import pyscf`` inside the generated user script is unambiguous (the
file name avoids any possibility that ``pyscf`` resolves to our local
module instead of the actual PySCF library).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..config.pyscf import PySCFConfig
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
#  Renderer                                                             #
# --------------------------------------------------------------------- #


def _atoms_block(struct: Structure, indent: str = "    ") -> str:
    """Format atoms as PySCF's multi-line `atom=` string (Angstrom)."""
    lines = []
    for el, (x, y, z) in zip(struct.elements, struct.positions):
        lines.append(f"{indent}{el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    return "\n".join(lines)


# Atomic-number lookup for the ECP heuristic.  Only need to identify
# heavy-atom presence (Z > 36 == above Kr); a partial table is enough.
# ase.data has a comprehensive table but we want molbuilder's pyscf
# generator to stay light on imports for the no-ase install path.
_ATOMIC_NUMBER = {
    "H":  1, "He":  2, "Li":  3, "Be":  4, "B":   5, "C":   6, "N":   7,
    "O":  8, "F":   9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P": 15, "S":  16, "Cl": 17, "Ar": 18, "K":  19, "Ca": 20, "Sc": 21,
    "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    "Kr": 36,
    # Z > 36 (need ECP for non-def2 bases):
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I":  53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Hf": 72, "Ta": 73, "W":  74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84,
}


def _resolve_ecp(struct: Structure, cfg: PySCFConfig):
    """Thin shim onto :func:`molbuilder.chemistry.resolve_pyscf_ecp`.

    The decision logic is shared with the spectra generator (the
    same rule -- def2 bundles its own ECP, lanl2dz auto-add for
    heavy atoms outside def2 -- applies regardless of WHICH script
    we're emitting; refactored to chemistry.py 2026-05-23 so the
    two generators can't drift).
    """
    from ..chemistry import resolve_pyscf_ecp
    return resolve_pyscf_ecp(struct, cfg.ecp, cfg.basis)


def _resolve_charge(struct: Structure, cfg: PySCFConfig) -> int:
    """Thin shim onto :func:`molbuilder.chemistry.resolve_net_charge`
    that bridges PySCFConfig's ``charge`` field to the shared rule
    (SiestaConfig uses ``net_charge`` -- different field name, same
    semantics).  Kept as a module-private helper so the rest of
    render_script doesn't have to know the chemistry import path.
    """
    from ..chemistry import resolve_net_charge
    return resolve_net_charge(struct, cfg.charge)


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

    # ---------- pre-emission validation ----------
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
    if cfg.optimize and cfg.write_molwatch_log and cfg.optimizer == "geometric":
        out.append(f"    {label}.molwatch.log     -- unified per-step log: marker-")
        out.append("                                  delimited blocks containing")
        out.append("                                  coords, energy (eV), forces")
        out.append("                                  (eV/Ang), and SCF cycle history.")
        out.append("                                  Single-file input for molwatch.")
    if cfg.compute_frequencies:
        out.append(f"    {label}.thermo.txt       -- post-relax harmonic frequencies")
        out.append("                                  (cm^-1) + RRHO thermochemistry")
        out.append("                                  (ZPE, U, H, G, S, Cv, Cp at the")
        out.append(f"                                  configured T = {cfg.temperature_K} K,")
        out.append(f"                                  P = {cfg.pressure_atm} atm).")
    out.append("")
    out.append("Dependencies:")
    out.append("    pip install pyscf")
    if cfg.optimize and cfg.optimizer == "geometric":
        out.append("    pip install geometric           # optimizer")
    if cfg.optimize and cfg.optimizer == "berny":
        out.append("    pip install pyberny             # optimizer")
    # PySCF 2.x ships D3/D3BJ natively (just set ``mf.disp = "d3bj"``).
    # Only D4 still requires the separate pyscf-dispersion package, so
    # we only advertise that install line when D4 is actually requested
    # (production or pre-opt stage) -- otherwise the hint is misleading.
    needs_dispersion_pkg = (
        (cfg.dispersion or "").lower() == "d4"
        or (cfg.preopt and (cfg.preopt_dispersion or "").lower() == "d4")
    )
    if needs_dispersion_pkg:
        out.append("    pip install pyscf-dispersion    # required for D4")
    out.append("")
    out.append("Or in one shot (full molbuilder runtime stack):")
    out.append("    pip install -r requirements-runtime.txt")
    out.append("    # or:  pip install 'molbuilder[runtime]'")
    out.append('"""')
    out.append("")

    # ---------------- Threading + runtime-info setup ---------------------
    # Shared with the spectra script -- defined in
    # molbuilder.runtime_info.  Pins BLAS to 1 thread per worker so
    # OMP * BLAS doesn't oversubscribe (a 20-physical / 40-logical host
    # otherwise sees load=40 from MKL/OpenBLAS spawning their own
    # thread pool on top of PySCF's OMP).  Auto-detects physical
    # cores at run time (cfg.threads=None) or honors the user's
    # explicit choice (cfg.threads=N).
    from ..runtime_info import (
        emit_threading_setup_lines,
        emit_runtime_info_capture_lines,
        emit_pyscf_post_import_lines,
    )
    out += emit_threading_setup_lines(cfg.threads)
    out += emit_runtime_info_capture_lines(
        use_gpu=bool(getattr(cfg, "use_gpu", False)),
        max_memory_mb=int(getattr(cfg, "max_memory_mb", 0)) or None,
    )
    out.append("import time")
    out.append("")
    # Import only what the script actually uses.  HF runs (method=RHF/UHF)
    # never touch the dft module; DFT runs (RKS/UKS, the default) need it
    # for both the production mf object and -- when preopt is enabled --
    # the always-DFT pre-opt warm-up.
    if is_dft or cfg.preopt:
        out.append("from pyscf import gto, scf, dft")
    else:
        out.append("from pyscf import gto, scf")
    # Size pyscf's thread pool AFTER import: env vars don't re-thread
    # an already-imported module (PySCF docs).
    out += emit_pyscf_post_import_lines()
    out.append("_RUNTIME_INFO['n_threads_pyscf'] = int(_pyscf_lib.num_threads())")
    out.append("print(f'molbuilder: pyscf.lib.num_threads() "
               "= {_RUNTIME_INFO[\"n_threads_pyscf\"]}')")
    out.append("")

    # GPU probe + _mb_to_gpu_if_enabled helper -- shared with the
    # spectra script via molbuilder.runtime_info so the cross-cutting
    # "detect, fall back, record" recipe lives in ONE place.  Caller
    # (this generator + spectra) decides WHERE to invoke the helper
    # on its mf object(s); the helper itself is identical.
    from ..runtime_info import emit_gpu_probe_lines
    out += emit_gpu_probe_lines(
        use_gpu=bool(getattr(cfg, "use_gpu", False)),
        min_compute_capability=7,
    )
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
        out.append("#  Build the molecule")
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
    # Effective Core Potential resolution (gap #8).  Heavy atoms
    # (Z > 36) on a non-def2 basis need an explicit ECP -- both for
    # cost (Pt has 78 electrons; treating the inner 60 as a pseudo-
    # potential is dramatic speedup) AND correctness (DFT without
    # scalar-relativistic ECPs gets Pt-Pt bond lengths and Au gaps
    # wrong by ~1 eV / ~0.1 A).  def2-* basis families bundle the
    # SBKJC / def2-ECP automatically, so we skip auto-emit there.
    ecp_chosen = _resolve_ecp(struct, cfg)
    if ecp_chosen and v:
        out += [
            "# Heavy atom(s) detected on a non-def2 basis -> using",
            f'# `ecp = "{ecp_chosen}"` for scalar-relativistic core',
            "# replacement.  Override via cfg.ecp = '<name>' or '' to disable.",
        ]
    out.append("mol = gto.M(")
    out.append("    atom = '''")
    out.append(_atoms_block(struct))
    out.append("    ''',")
    out.append(f'    basis      = "{cfg.basis}",')
    if ecp_chosen:
        # ECP can be either a string ("lanl2dz") or a per-element dict
        # ({"Pt": "lanl2dz", "Au": "stuttgart"}) -- both are valid PySCF
        # gto.M() inputs.  String -> emit as quoted literal; dict ->
        # emit as a Python dict-literal so PySCF sees it as a dict, not
        # a string-with-braces (which it would reject as an unknown name).
        if isinstance(ecp_chosen, dict):
            out.append(f'    ecp        = {ecp_chosen!r},')
        else:
            out.append(f'    ecp        = "{ecp_chosen}",')
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

    # ---------------- Unified molwatch log emitter (early, additive) ------
    # Defined and instantiated NOW -- before preopt -- so the log file
    # (header + initial-preview block) exists the moment the script
    # starts running.  Preopt can take hours on a real molecule; we
    # don't want the Watch tab staring at "no file to load" the whole
    # time.  SCF cycle hooks are wired per-stage at each mf object
    # below (mf1 for preopt, mf for production).  The opt-step hook
    # is wired at each `optimize(...)` call.
    if cfg.optimize and cfg.write_molwatch_log and cfg.optimizer == "geometric":
        out += _emit_molwatch_emitter(v, cfg)

    # ------------------------------------------------------------- preopt
    if cfg.preopt and cfg.optimize:
        out += _emit_preopt_block(cfg, charge, v)

    # ------------------------------------------------------------- main scf
    out.append("# ============================================================")
    if cfg.preopt and cfg.optimize:
        out.append("#  Main run -- production functional / basis")
    else:
        out.append("#  SCF setup")
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
    if cfg.dispersion and cfg.dispersion.lower() != "none" and is_dft:
        out.append(f'mf.disp = "{cfg.dispersion}"')
    if cfg.solvent:
        eps = _SOLVENTS.get(cfg.solvent.lower())
        if eps is None:
            raise ValueError(
                f"unknown solvent {cfg.solvent!r}; "
                f"valid: {sorted(_SOLVENTS)}"
            )
        out.append("# PCM solvation -- continuum model (cheaper than ddCOSMO).")
        # Use the SCF-method form ``mf.PCM()`` instead of the bare
        # ``pcm.PCM(mf)`` constructor (P1).  In PySCF 2.x ``mf.PCM()``
        # returns a PCM-decorated SCF object that exposes
        # ``.with_solvent``; ``pcm.PCM(mf)`` returns a bare solvent
        # object (no .with_solvent), so the next two lines used to crash.
        out.append("mf = mf.PCM()")
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
    else:
        # Show a commented level_shift template ONLY when an open-
        # shell metal is in the structure -- the most common reason
        # an SCF won't converge for Fe / Mn / Co / Ni / Cu systems.
        # For clean-organics the line would be noise.  Discoverable
        # without being prescriptive.
        from ..chemistry import detect_open_shell_metals
        _metals = detect_open_shell_metals(struct)
        if _metals:
            out.append("# Hard SCF (typical for open-shell metals like "
                       f"{', '.join(_metals)}):")
            out.append("# Uncomment to apply a virtual-orbital level "
                       "shift (Eh).  Typical 0.1-0.3 for hard cases;")
            out.append("# helps SCF converge when the HOMO-LUMO gap is "
                       "small / unphysical mixing occurs.")
            out.append("# mf.level_shift = 0.2")
    # Hard-SCF troubleshooting knobs (gap #10).  Only emit when
    # bumped from PySCF defaults so tutorial scripts stay clean
    # for the easy-converge path.
    if cfg.diis_space != 8:
        out.append(f"mf.diis_space = {cfg.diis_space}")
    if cfg.damp:
        out.append(f"mf.damp = {cfg.damp}")
    if cfg.chkfile:
        out.append('mf.chkfile = JOB + ".chk"')

    # GPU patch: promote the fully-assembled production mf to its
    # gpu4pyscf equivalent when the runtime probe at script-start
    # succeeded (_USING_GPU=True).  No-op on CPU nodes.  Called
    # AFTER density_fit / disp / PCM / chkfile / conv_tol so
    # .to_gpu() sees the complete CPU mf and the GPU mirror has
    # the same settings.
    out.append("mf = _mb_to_gpu_if_enabled(mf)")

    # Wire the production-mf SCF callback so per-cycle SCF history
    # is captured for production opt steps.  The emitter itself was
    # instantiated earlier (before preopt); we just attach the hook
    # here once mf is in its final form.
    if cfg.optimize and cfg.write_molwatch_log and cfg.optimizer == "geometric":
        out.append(_emit_molwatch_callback_wire("mf"))
    out.append("")

    # ------------------------------------------------------------- run
    if cfg.optimize:
        if v:
            out.append("# ============================================================")
            out.append("#  Geometry optimization")
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
        if cfg.write_molwatch_log and cfg.optimizer == "geometric":
            if v:
                out.append("    # callback fires once per accepted opt step;")
                out.append("    # _molwatch.opt_step_hook flushes one block to")
                out.append("    # <JOB>.molwatch.log with coords/energy/forces +")
                out.append("    # the SCF cycles captured since the previous step.")
            out.append("    callback              = _molwatch.opt_step_hook,")
        out.append(")")
        # Re-evaluate at the relaxed geometry so the printed energy
        # is unambiguously the converged SCF at mol_eq's coordinates
        # (gap #9).  optimize() leaves mf with the LAST line-search
        # SCF, which may not be at mol_eq exactly -- the difference
        # is small (mHa) but matters when comparing reaction energies
        # across runs.  One extra SCF, cheap relative to the opt.
        #
        # Three things going on here:
        #   1. ``mf.make_rdm1()`` snapshots the converged density at
        #      the previous (line-search) geometry from the cached MOs.
        #      This lets kernel() warm-start from that DM instead of
        #      cold-starting from MINAO -- saves ~10-30 SCF cycles
        #      since the post-opt geometry is a tiny perturbation of
        #      the line-search step (R1).
        #   2. ``mf.reset(mol_eq)`` updates self.mol AND invalidates
        #      cached integrals (_eri / with_df / _opt) bound to the
        #      previous mol.  Without reset(), kernel() would reuse
        #      stale integrals and the result would not be the SCF
        #      at mol_eq (P3).
        #   3. ``mf.kernel(dm0=dm_prev)`` rebuilds integrals at mol_eq
        #      and converges from the warm-started DM.
        if v:
            out.append("# Re-evaluate at the relaxed geometry: optimize() leaves")
            out.append("# mf bound to the last line-search SCF, not necessarily")
            out.append("# the SCF AT mol_eq.  reset() drops cached integrals so")
            out.append("# kernel() rebuilds them at mol_eq's coordinates;")
            out.append("# dm0=dm_prev warm-starts from the converged DM when")
            out.append("# the previous SCF state is intact.")
        # ``mf.make_rdm1()`` errors with ``'>' not supported between
        # 'NoneType' and 'int'`` when geomeTRIC's optimize() failed to
        # converge or otherwise left mf without a valid ``mo_occ``.
        # Guard the warm-start: if mf doesn't carry a usable density,
        # fall back to mf.init_guess (MINAO) for the post-opt SCF.
        out.append("dm_prev = (mf.make_rdm1() "
                   "if mf.mo_coeff is not None and mf.mo_occ is not None "
                   "else None)")
        out.append("mf.reset(mol_eq)")
        out.append("mf.kernel(dm0=dm_prev)")
        out.append('print(f"Final energy: {mf.e_tot:.8f} Hartree")')
    else:
        if v:
            out.append("# ============================================================")
            out.append("#  Single-point SCF (no optimization)")
            out.append("# ============================================================")
        out.append("e = mf.kernel()")
        out.append('print(f"Total energy: {e:.8f} Hartree")')
        out.append("mol_eq = mol")
    out.append("")

    # ------------------------------------------------------------- stability
    # Open-shell stability check (UKS / UHF only).
    #
    # Why: open-shell SCFs can converge to broken-symmetry SADDLE
    # points -- the energy looks converged but the wavefunction is
    # not the variational minimum.  PySCF's mf.stability() examines
    # internal (orbital rotation) and external (real -> complex /
    # restricted -> unrestricted) instabilities and prints a warning
    # if found, with new MO coefficients to restart from.
    #
    # We do NOT auto-rerun the SCF when an instability is reported --
    # we surface the warning to the user and let them decide whether
    # to take the suggested step.  Real-world advice: if stability
    # warns, restart with `mf.kernel(dm0=mf.make_rdm1(mo_coeff,
    # mf.mo_occ))` after replacing mo_coeff with the new vectors;
    # the cost is one extra SCF, the alternative is silently shipping
    # a non-variational answer.
    #
    # We don't emit this for RKS / RHF: closed-shell stability is
    # mostly a singlet -> triplet check that's rarely the user's
    # concern (and the call is no-op cheap but adds noise to a
    # tutorial script that's already dense).
    if method_class.startswith("U"):
        if v:
            out.append("# ============================================================")
            out.append("#  Open-shell stability check")
            out.append("# ============================================================")
            out.append("# Catches broken-symmetry saddles in UKS / UHF: a result")
            out.append("# that LOOKS converged but isn't the variational minimum.")
            out.append("# mf.stability() prints a warning + suggested MOs")
            out.append("# if an instability is found.  See PySCF's docs for")
            out.append("# rerunning from the suggested vectors.")
        out.append('print("\\n=== Stage: stability analysis ===")')
        out.append("mf.stability()")
        out.append("")

    # ------------------------------------------------------------- frequencies
    # Analytic Hessian + RRHO thermochemistry, opt-in.  Runs at mol_eq
    # using the already-converged mf, so it costs one Hessian
    # construction (no extra SCF).  Wrapped in try/except so a failure
    # here does NOT lose the converged energy + optimized geometry
    # that have already been printed / saved.  Imaginary modes are
    # reported (count + cm^-1) but the script does not auto-perturb;
    # the user decides whether to restart the optimization along the
    # imag coordinate.
    if cfg.compute_frequencies:
        out += _emit_frequencies_block(cfg, v)

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

    # Post-processing hook (gap #6).  Commented call templates for
    # the follow-ups users typically want after a relaxation.
    # Default-disabled so the script's behaviour is unchanged;
    # uncomment to enable.
    out.append("")
    out.append("# === Post-processing hook (commented templates) ===")
    if v:
        out += [
            "# Common follow-ups on the converged density at mol_eq.",
            "# All four use the already-built mf object; no extra SCF",
            "# (PySCF re-uses the converged density matrix) -- the cost",
            "# is one matrix multiply per analysis.  Enable any subset.",
        ]
    out += [
        "#",
        "# 1. Mulliken population (per-atom partial charges):",
        "# pop, chg = mf.mulliken_pop()",
        "# print('Mulliken charges:', chg)",
        "#",
        "# 2. Dipole moment (Debye):",
        "# dip = mf.dip_moment(unit='Debye')",
        "# print(f'Dipole moment: {dip}')",
        "#",
        "# 3. Full SCF analyze() report (energies, gaps, populations):",
        "# mf.analyze()",
        "#",
        "# 4. NPA / NBO charges (cleaner than Mulliken; needs nbo wrap):",
        "# from pyscf import lo",
        "# c_nao = lo.orth.lowdin(mol_eq.intor('int1e_ovlp'))",
        "# # ... see PySCF docs for full NPA / NBO recipe",
    ]

    # ------------------------------------------------------------- hints
    if v:
        out += _emit_troubleshooting_block(cfg)

    return "\n".join(out) + "\n"


def _emit_preopt_block(cfg: PySCFConfig, charge: int, v: bool) -> List[str]:
    """Stage 1: cheap geometry warm-up before the production functional."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Pre-optimization  (cheap warm-up: PBE/def2-SVP)")
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
    # Pre-opt always uses DFT (a cheap functional + small basis is the
    # whole point of the warm-up).  We mirror the RESTRICTED-vs-
    # UNRESTRICTED choice from the production method (RHF/RKS -> RKS,
    # UHF/UKS -> UKS) but force the dft module regardless: even when
    # the production run is plain HF, the pre-opt switches to a DFT
    # functional (cfg.preopt_functional, default PBE) to clean up the
    # geometry quickly.
    out.append(f'mf1 = dft.{cfg.method.upper().replace("HF", "KS")}(mol_pre)')
    out.append(f'mf1.xc = "{cfg.preopt_functional}"')
    if cfg.preopt_density_fit:
        out.append("mf1 = mf1.density_fit()")
    if cfg.preopt_dispersion and cfg.preopt_dispersion.lower() != "none":
        out.append(f'mf1.disp = "{cfg.preopt_dispersion}"')
    out.append(f"mf1.conv_tol  = {cfg.scf_conv_tol:.0e}")
    out.append(f"mf1.max_cycle = {cfg.scf_max_cycle}")
    # Mirror the production-stage SCF tuning knobs onto the pre-opt
    # mean-field (P2).  If the user needed level_shift / damp /
    # diis_space / a non-default init_guess to make production
    # converge, the pre-opt warm-up needs the same help -- otherwise
    # pre-opt diverges silently before production runs.
    out.append(f'mf1.init_guess = "{cfg.scf_init_guess}"')
    if cfg.level_shift:
        out.append(f"mf1.level_shift = {cfg.level_shift}")
    if cfg.diis_space != 8:
        out.append(f"mf1.diis_space = {cfg.diis_space}")
    if cfg.damp:
        out.append(f"mf1.damp = {cfg.damp}")
    # GPU patch: same shape as the production-mf line above -- promote
    # the preopt mf to gpu4pyscf when the runtime probe succeeded.
    # Done AFTER density_fit / disp / convergence settings so .to_gpu()
    # sees the complete CPU object.
    out.append("mf1 = _mb_to_gpu_if_enabled(mf1)")
    # Wire the preopt SCF callback so the molwatch log captures preopt
    # SCF history too -- otherwise the user only sees a single block per
    # preopt opt step with empty scf_history (and the "Watch tab can't
    # see anything until preopt finishes" UX bug returns at SCF granularity).
    if cfg.write_molwatch_log and cfg.optimizer == "geometric":
        out.append(_emit_molwatch_callback_wire("mf1"))
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
    if cfg.write_molwatch_log and cfg.optimizer == "geometric":
        if v:
            out.append("    # Stream preopt opt steps to <JOB>.molwatch.log so")
            out.append("    # the Watch tab shows progress from frame 1 onwards")
            out.append("    # rather than waiting for the (potentially long) preopt")
            out.append("    # to finish before the first step is logged.")
        out.append("    callback          = _molwatch.opt_step_hook,")
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


def _emit_frequencies_block(cfg: PySCFConfig, v: bool) -> List[str]:
    """Analytic Hessian + RRHO thermochemistry block.

    Inserted between the post-opt SCF (where ``mf`` is converged at
    ``mol_eq``) and the optimized-geometry save step.  The block is
    a single try/except so any failure (Hessian unavailable for the
    functional, OOM, etc.) prints a diagnostic but does NOT lose the
    converged energy + optimized geometry already on disk.
    """
    P_pa = cfg.pressure_atm * 101325.0      # PySCF's thermo() wants Pa
    out: List[str] = []
    out.append("")
    out.append("# ============================================================")
    out.append("#  Harmonic frequencies + RRHO thermochemistry")
    out.append("# ============================================================")
    if v:
        out.append("# Why: a relaxed geometry that's a true minimum has only")
        out.append("# REAL vibrational modes; imaginary modes mean the optimizer")
        out.append("# converged to a saddle point.  The RRHO thermochemistry")
        out.append("# (zero-point energy + finite-T corrections to U / H / G)")
        out.append("# is what you add to the electronic energy when reporting")
        out.append("# reaction or binding free energies.")
        out.append("#")
        out.append("# Cost: one analytic Hessian -- typically 5-15x a single SCF")
        out.append("# for a small molecule, more for larger systems.  No extra")
        out.append("# SCF is needed (we use the already-converged mf above).")
        out.append("#")
        out.append("# Caveats:")
        out.append("#   * RRHO assumes harmonic vibrations + rigid rotor +")
        out.append("#     ideal gas.  Low-frequency modes (< ~50 cm^-1) inflate")
        out.append("#     the entropy artifically; quasi-RRHO (Grimme) tames")
        out.append("#     this but isn't applied here.")
        out.append("#   * With mf.disp set, PySCF's Hessian module adds the")
        out.append("#     dispersion contribution in 2.6+.  For tight phonon")
        out.append("#     work cross-check against numerical frequencies.")
        out.append("#   * If the geometry is not a stationary point, expect")
        out.append("#     translation/rotation modes to leak into low-cm^-1")
        out.append("#     vibrations; finite-T corrections will be unreliable.")
    out.append('print("\\n=== Stage: harmonic frequencies + thermochemistry ===")')
    out.append("try:")
    out.append("    from pyscf.hessian import thermo as _mb_thermo")
    out.append("    _mb_hess = mf.Hessian().kernel()")
    out.append("    _mb_freq = _mb_thermo.harmonic_analysis(mf.mol, _mb_hess)")
    out.append("    _mb_wn = _mb_freq[\"freq_wavenumber\"]")
    out.append(f"    _mb_therm = _mb_thermo.thermo(mf, _mb_freq[\"freq_au\"], "
               f"{cfg.temperature_K!r}, {P_pa!r})")
    out.append("    _mb_imag = int(sum(1 for _w in _mb_wn if "
               "getattr(_w, 'imag', 0.0) != 0))")
    out.append("    with open(JOB + \".thermo.txt\", \"w\") as _mb_fh:")
    out.append("        _mb_fh.write(\"# molbuilder PySCF harmonic analysis "
               "+ RRHO thermochemistry\\n\")")
    out.append(f"        _mb_fh.write(\"# T = {cfg.temperature_K} K, "
               f"P = {cfg.pressure_atm} atm ({P_pa:.1f} Pa)\\n\")")
    out.append("        _mb_fh.write(f\"# Method: {mf.__class__.__name__}\"")
    out.append("                     f\"{(' ' + mf.xc) if hasattr(mf, 'xc') and mf.xc else ''}\"")
    out.append("                     f\" / {mf.mol.basis}\\n\")")
    out.append("        _mb_fh.write(f\"# Geometry: mol_eq "
               "({mf.mol.natm} atoms)\\n\")")
    out.append("        _mb_fh.write(\"\\n[frequencies] (cm^-1)\\n\")")
    out.append("        for _i, _w in enumerate(_mb_wn):")
    out.append("            _mb_v = float(getattr(_w, 'real', _w))")
    out.append("            _mb_im = getattr(_w, 'imag', 0.0)")
    out.append("            _mb_tag = '  (imag)' if _mb_im != 0 else ''")
    out.append("            _mb_fh.write(f\"  mode {_i+1:3d}  "
               "{_mb_v:12.3f}{_mb_tag}\\n\")")
    out.append("        _mb_fh.write(\"\\n[thermochemistry]\\n\")")
    # _mb_therm is a dict[str, (value, unit)] in PySCF 2.x.
    out.append("        for _k, _v_ in sorted(_mb_therm.items()):")
    out.append("            if isinstance(_v_, tuple) and len(_v_) == 2:")
    out.append("                _mb_fh.write(f\"  {_k:18s} = {_v_[0]:18.10f}  "
               "{_v_[1]}\\n\")")
    out.append("            else:")
    out.append("                _mb_fh.write(f\"  {_k:18s} = {_v_!r}\\n\")")
    out.append("    print(f\"Frequencies: {len(_mb_wn)} modes "
               "({_mb_imag} imaginary).  Thermo summary -> {JOB}.thermo.txt\")")
    if cfg.optimize:
        # Only worth the warn when we expected a minimum.
        out.append("    if _mb_imag > 0:")
        out.append("        print(f\"WARN: {_mb_imag} imaginary mode(s) at the "
                   "relaxed geometry -- this is a saddle, not a minimum.  "
                   "Perturb along the imag coord and re-optimize, or tighten "
                   "geom_conv_grms.\")")
    out.append("except Exception as _mb_exc:")
    out.append("    print(f\"Frequency analysis FAILED: {_mb_exc}\\n\"")
    out.append("          f\"Converged energy + optimized geometry are still "
               "on disk; rerun with --no-compute-frequencies to skip.\")")
    out.append("")
    return out


def _emit_molwatch_emitter(v: bool, cfg: "PySCFConfig") -> List[str]:
    """Inline streaming writer for ``<JOB>.molwatch.log``.

    The emitter is instantiated **early** -- BEFORE preopt -- so the
    log file (header + initial-preview block) exists from the moment
    the script starts running.  Preopt can take hours on a real
    molecule; without this ordering the Watch tab would have no file
    to load until preopt finishes, defeating the "live trajectory"
    promise.

    Hooks are wired separately at each stage:

      * ``mf1.callback = _molwatch.scf_cycle_hook``  (preopt SCF cycles)
      * ``optimize(mf1, ..., callback=_molwatch.opt_step_hook)`` (preopt steps)
      * ``mf.callback = _molwatch.scf_cycle_hook``   (production SCF cycles)
      * ``optimize(mf,  ..., callback=_molwatch.opt_step_hook)`` (production steps)

    Block layout, parser tolerance, and other contract details are
    documented on the source class
    :class:`molbuilder.trajectory_log.emitter.MolwatchEmitter`.
    The class source is inlined here verbatim via :func:`inspect.getsource`
    so the generated script stays self-contained (no molbuilder runtime
    dependency on the user's machine) while keeping a single source of
    truth that's directly testable as a real Python module.
    """
    import inspect

    from ..trajectory_log.emitter import MolwatchEmitter

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
        out.append("# Source of truth: molbuilder.trajectory_log.emitter -- the")
        out.append("# class below is inlined verbatim from there at script-")
        out.append("# generation time so this script stays self-contained.  Do")
        out.append("# NOT edit the inline copy; edit the module and regenerate.")
        out.append("#")
        out.append("# All standard PySCF/geomeTRIC outputs are kept untouched;")
        out.append("# this is purely additional.  Disable via cfg.write_molwatch_log =")
        out.append("# False at generation time if you don't want it.")
    out.append("import time as _mw_time")
    out.append("import numpy as _mw_np")
    out.append("")
    # Inline the class definition itself.  inspect.getsource includes
    # the leading `class MolwatchEmitter:` line and full body, properly
    # indented.  The script's globals supply `_mw_time` and `_mw_np`,
    # which the methods reference at call time.
    out.append(inspect.getsource(MolwatchEmitter).rstrip())
    out.append("")
    # Instantiate as early as possible (BEFORE preopt) so the log
    # file -- with header + initial-preview block -- exists the
    # moment the script starts running.  Otherwise a long preopt
    # (which can take hours on a real molecule) would mean the user
    # has nothing to load on the Watch tab until preopt finishes.
    # The SCF callback is wired separately at each stage's mf object
    # (see emit_scf_callback_wiring below).
    # Stage-aware molwatch-log filename (job-layout v1).  Compute
    # via the L1 ``molwatch_log_basename`` helper so the rule has
    # ONE source of truth across SIESTA + PySCF emitters; the
    # generated script gets the resolved suffix as a literal so it
    # stays self-contained at runtime (no molbuilder import).
    from ..trajectory_log.format import molwatch_log_basename
    _placeholder      = "_X_"
    _resolved_for_X   = molwatch_log_basename(_placeholder,
                                              getattr(cfg, "stage", None))
    # Strip the placeholder; what's left is the suffix the generator
    # appends to ``JOB`` at runtime.
    _suffix = _resolved_for_X[len(_placeholder):]
    out.append(f'_molwatch = MolwatchEmitter('
               f'JOB + {_suffix!r}, JOB, mol, runtime_info=_RUNTIME_INFO)')
    out.append("")
    # Run-state markers.  The watch UI reads these to render a binary
    # "Finished / Ongoing / Error" badge -- authoritative when present,
    # not a stall heuristic (long-iteration runs would false-positive).
    #
    # Strategy: install excepthook to capture uncaught exceptions, then
    # an atexit hook that always runs (clean exit OR exception OR Ctrl-C)
    # to write the conclusion line.  SIGKILL / power loss leaves the
    # file without markers, which correctly reads as "ongoing" -- the
    # process didn't have a chance to finalize.
    out.append("import atexit as _mw_atexit")
    out.append("import sys as _mw_sys")
    out.append("_molwatch_run = {'error': None}")
    out.append("def _molwatch_excepthook(exc_type, exc_value, exc_tb):")
    out.append("    _molwatch_run['error'] = f'{exc_type.__name__}: {exc_value}'")
    out.append("    _mw_sys.__excepthook__(exc_type, exc_value, exc_tb)")
    out.append("_mw_sys.excepthook = _molwatch_excepthook")
    out.append("def _molwatch_finalize():")
    out.append("    try:")
    out.append("        with open(_molwatch.path, 'a') as _fh:")
    out.append("            _ts = _mw_time.strftime('%Y-%m-%dT%H:%M:%S')")
    out.append("            if _molwatch_run['error']:")
    out.append("                _msg = _molwatch_run['error'].replace(chr(10), ' ')")
    out.append("                _fh.write(f'# error: {_msg}\\n')")
    out.append("            _fh.write(f'# concluded: {_ts}\\n')")
    out.append("    except Exception:")
    out.append("        pass    # don't break the user's exit on a logging issue")
    out.append("_mw_atexit.register(_molwatch_finalize)")
    out.append("")
    return out


def _emit_molwatch_callback_wire(mf_var: str) -> str:
    """One-line snippet that wires a per-cycle SCF callback to the
    given mean-field object.  Used at both stages so the molwatch
    log captures SCF iterations from preopt and production runs
    alike."""
    return f"{mf_var}.callback = _molwatch.scf_cycle_hook"


def _emit_save_helper(v: bool) -> List[str]:
    """Inline XYZ writer that doesn't require ase / pyscf.tools."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Helper: XYZ writer (defined early so initial-geom snapshot works)")
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
    out.append("#   * after SCF: mf.stability()")
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
