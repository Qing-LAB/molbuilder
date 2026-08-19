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

import math
from pathlib import Path
from typing import List, Optional

from ..config.pyscf import PySCFConfig
# § 4 rule 2's reading of `restart`, shared with SIESTA -- one
# field, one rule, one place that reads it.
from ..identity import continues
from ..structure import Structure
# ONE name for each of the two doors this writer opens, imported once at
# module scope -- as `siesta/input.py` already does.  Seven aliases for
# ``script_emit`` and three for ``layout`` stood here, one pair
# re-imported inside each function, so the same door had seven spellings
# and `spec_for`'s return annotation named one that no scope defined.
from .. import script_emit as _sc
from . import layout as _layout


#: How many times an open-shell SCF may be re-converged from the
#: orbitals ``mf.stability()`` suggests before the script gives up and
#: warns.  Three (user decision 2026-08-13): a genuine broken-symmetry
#: solution is usually repaired on the first restart, and a case that
#: survives three has a problem no amount of retrying will fix.
#: Exhausting them WARNS and continues -- a hint does not end the run.
_STABILITY_MAX_RESTARTS = 3

#: How much the energy must FALL for a stability restart to count as a
#: repair, in Hartree.  1e-8 Ha is far below chemical significance
#: (1 kcal/mol = 1.6e-3 Ha) and comfortably above SCF numerical noise.
#:
#: The energy is the criterion, not the orbital coefficients.  Comparing
#: coefficients looks obvious and is wrong: a degenerate shell -- O2's
#: pi pair, any symmetric radical -- can be rotated freely within its
#: degenerate space, so ``stability()`` returns numerically different
#: orbitals describing an identical state, for ever.  Measured on O2
#: triplet UHF/STO-3G: round 1 genuinely repairs (dE = -1.5e-03), then
#: rounds 2 and 3 return dE = +3.6e-10 and +1.6e-09 -- no improvement,
#: yet a coefficient test called all three unstable and the run ended
#: with a false warning.
_STABILITY_ENERGY_TOL = 1e-8


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


# Atomic-number lookup -- used by the ECP heuristic AND by the
# molbuilder.validation electron-count parity check.  Full Z=1-118
# table so the parity sum never reads a silent ``0`` for an actinide
# / transactinide (Pa/Th/U/Np/Pu/Am/Cm/Bk/Cf/Es/Fm/Md/No/Lr + the
# transactinides Rf-Og), and so the ECP heuristic correctly flags
# every heavy element instead of stopping at Po.  Pre-2026-05-26 the
# table stopped at Z=84 (Po), missing all actinides; caught by the
# 2026-05-26 third-pass review (silent ``KeyError->0`` would produce
# wrong electron counts for any structure containing those elements).
#
# Kept hand-rolled (not ``from ase.data import atomic_numbers``) so
# the generator stays light on imports for the no-ase install path
# the original docstring promised.
_ATOMIC_NUMBER = {
    # Row 1-4 (Z=1-36)
    "H":  1, "He":  2, "Li":  3, "Be":  4, "B":   5, "C":   6, "N":   7,
    "O":  8, "F":   9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P": 15, "S":  16, "Cl": 17, "Ar": 18, "K":  19, "Ca": 20, "Sc": 21,
    "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    "Kr": 36,
    # Row 5 (Z=37-54).  Z > 36: ECP needed for non-def2 bases.
    "Rb": 37, "Sr": 38, "Y":  39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I":  53, "Xe": 54,
    # Row 6 (Z=55-86): Cs through Rn, including lanthanides Ce-Lu.
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61,
    "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68,
    "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W":  74, "Re": 75,
    "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    # Row 7 (Z=87-118): Fr through Og, including actinides Th-Lr.
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U":  92, "Np": 93,
    "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106,
    "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
    "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
}


def _resolve_ecp(struct: Structure, cfg: PySCFConfig):
    """Thin shim onto :func:`molbuilder.chemistry.resolve_pyscf_ecp`.

    The rule is shared with the spectra generator regardless of WHICH
    script we're emitting (refactored to chemistry.py 2026-05-23 so the
    two generators can't drift).  Since 2026-08-13 the rule is simply
    *which declared elements are present*: no basis is consulted and no
    ECP is added that the user did not name.
    """
    from ..chemistry import resolve_pyscf_ecp
    return resolve_pyscf_ecp(struct, cfg.ecp, cfg.ecp_atoms)


def _resolve_charge(struct: Structure, cfg: PySCFConfig) -> int:
    """Thin shim onto :func:`molbuilder.chemistry.resolve_net_charge`
    that bridges PySCFConfig's ``charge`` field to the shared rule
    (SiestaConfig uses ``net_charge`` -- different field name, same
    semantics).  Kept as a module-private helper so the rest of
    render_script doesn't have to know the chemistry import path.
    """
    from ..chemistry import resolve_net_charge
    return resolve_net_charge(struct, cfg.charge)


def spec_for(struct: Structure,
                  config: Optional[PySCFConfig] = None,
                  *,
                  stage_token: Optional[str] = None) -> "_sc.RenderedDeck":
    """Format a Structure as a runnable PySCF script (Python text).

    The result is what you'd write by hand if you knew exactly what
    every PySCF knob does -- with verbose comments turned on by
    default so you can read the file as documentation of the choices.

    ``stage_token`` names **which rung of the ladder this deck is**, and it is
    used.  `stages.md` § 1.1a: a PySCF ladder is N decks and N jobs, so two rungs
    are two processes writing into the same calculation -- and anything they both
    name would collide.  So the token goes into every name the script itself
    chooses: PySCF's own log (``<JOB>_<NN>_<stage>.log``), geomeTRIC's trajectory
    and opt log (``<JOB>_geom_<NN>_<stage>_optim.xyz``, ``..._geom_<NN>_<stage>.log``)
    and the molwatch trajectory log (``<JOB>_<NN>_<stage>.molwatch.log``) -- the
    same three names SIESTA suffixes (§ 1.1a, consequence 1).

    **The ``JOB`` literal stays unsuffixed**, exactly as SIESTA's ``SystemLabel``
    does and for the same reason (§ 1.1a, consequence 2): the engine finds the
    previous rung's checkpoint and optimized geometry by that name, so a name
    that changed per rung would hide them.

    The DECK's filename still carries a token when `prep` gives it one: the
    caller builds that name, not this function.  Ignored here means *the
    script's internals are not re-suffixed*, not *the token is discarded*.
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

    # EVERYTHING THE PARAMETERS SUB-STEP WRITES, collected as it is written.
    # The CHECK gate reads the finished file back and asks whether each of
    # these survived into it -- the rule that closes the loop between what this
    # function intended and what PySCF will read (`script-preparation.md`
    # § 3.3).  Collected HERE because the script is assembled here; a set
    # assembled anywhere else is a second answer to "what did this deck write".
    def _science_a(struct, cfg) -> str:
        """One run of the deck's science, in the order PySCF reads it.

        The layout is walked in order, so what sits between two runs of
        settings sits between them in the deck too.
        """
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
        if cfg.optimize:
            # ONE rung's targets, because one deck IS one rung: a ladder is N
            # decks and N jobs (`stages.md` § 1.1a).  A summary of the whole
            # ladder would be this deck describing runs it cannot see.
            out.append(f"Optimizer : {cfg.optimizer}"
                       + (f"   (rung {stage_token})" if stage_token else ""))
            out.append(f"            maxsteps={cfg.geom_max_steps}, "
                       f"grms={cfg.geom_grms:.1e} Ha/Bohr, "
                       f"gmax={cfg.geom_gmax:.1e}, "
                       f"conv_tol={cfg.scf_conv_tol:.1e}")
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
            _rung = f"_{stage_token}" if stage_token else ""
            out.append(f"    {label}_geom{_rung}_optim.xyz   -- this rung's streaming")
            out.append("                                          trajectory (multi-frame")
            out.append("                                          XYZ).  A ladder is one")
            out.append("                                          job per rung, so each")
            out.append("                                          writes its own.")
            out.append(f"    {label}_geom{_rung}.log         -- geomeTRIC's opt log")
            out.append("                                          for this rung.")
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
        out.append("    Use the generated molbuilder run wrapper.")
        out.append("    Bootstrap managed environments once:")
        out.append("        bash scripts/install-env.sh bootstrap --yes")
        if cfg.optimize and cfg.optimizer == "berny":
            out.append("    Berny is optional and installed separately in molbuilder-pySCF:")
            out.append("        conda run -n molbuilder-pySCF pip install pyberny")
        out.append("\n")
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
            max_memory_mb=(int(cfg.max_memory_mb) if cfg.max_memory_mb else None),
        )
        out.append("import time")
        # ``_os`` is used by the warm-restart blocks (geometry override
        # at gto.M() time + chkfile SCF init-guess after mf is built).
        # Hoisted out of the chkfile-conditional emission so the geometry
        # warm-restart, which runs unconditionally before mol is built,
        # can call ``_os.path.exists`` without depending on a downstream
        # conditional import landing in the script.
        out.append("import os as _os")
        out.append("")
        # Import only what the script actually uses.  HF runs (method=RHF/UHF)
        # never touch the dft module; DFT runs (RKS/UKS, the default) need it.
        if is_dft:
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
        from ..runtime_info import (
            emit_gpu_probe_lines, GPU4PYSCF_MIN_COMPUTE_CAPABILITY,
        )
        out += emit_gpu_probe_lines(
            use_gpu=bool(getattr(cfg, "use_gpu", False)),
            min_compute_capability=GPU4PYSCF_MIN_COMPUTE_CAPABILITY,
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
            if opt_pkg == "pyberny":
                out.append('        "Install the optional Berny optimizer in molbuilder-pySCF:\\n"')
                out.append('        "conda run -n molbuilder-pySCF pip install pyberny\\n"')
            else:
                out.append('        "Bootstrap or repair the managed backend:\\n"')
                out.append('        "bash scripts/install-env.sh bootstrap --yes\\n"')
            out.append('        f"(import error: {_exc})"')
            out.append("    )")
        if cfg.solvent:
            out.append("from pyscf.solvent import pcm")
        out.append("")
        out.append("t0 = time.time()")
        out.append(f'JOB = "{label}"')
        out.append("")
        # ---- _mb_outfile helper: ALL output paths resolve relative to
        # the script directory, NOT the process cwd ---------------------
        # Why: PySCF / geomeTRIC may chdir() during optimisation
        # (geomeTRIC's optimize() builds scratch in a temp dir; PySCF's
        # mol.build() writes the .log relative to cwd at gto.M() time).
        # If a user invokes ``python myjob.py`` from a different
        # directory than where the script lives -- OR if a downstream
        # tool chdir's the process during the run -- output artefacts
        # would scatter across the filesystem.  The .run.sh wrapper
        # chdir's to the script directory before launching, but the
        # script must be robust when invoked directly too.  Resolving
        # everything via ``_mb_outfile(name)`` makes the script land
        # ALL its outputs next to itself, regardless of cwd.
        out.append("from pathlib import Path as _MB_Path")
        out.append("_MB_SCRIPT_DIR = _MB_Path(__file__).resolve().parent")
        out.append("def _mb_outfile(name):")
        out.append("    p = _MB_Path(name)")
        out.append("    return str(p if p.is_absolute() else _MB_SCRIPT_DIR / p)")
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
            # THE REASONS COME FROM THE CATALOGUE, even here.  These four are
            # ARGUMENTS of the ``gto.M(...)`` call rather than lines of their own,
            # so they are not walked by the parameters sub-step -- but a value is
            # still written together with the reason it holds
            # (`script-preparation.md` § 3.2, W2), and the reason is still read from
            # the declaration rather than typed beside it.
            for _item in ("spin", "charge", "symmetry", "max_memory_mb"):
                out += _sc.parameter(_item, "pyscf").note()
        # Effective Core Potential.  Why one is worth declaring: for an
        # element like Pt (78 electrons) an ECP replaces the inner 60 with a
        # pseudopotential -- a large speedup, and more importantly a
        # correctness matter, since DFT without scalar-relativistic ECPs
        # gets Pt-Pt bond lengths and Au gaps wrong by ~0.1 A / ~1 eV.
        #
        # WHICH atoms is the user's declaration (``ecp`` + ``ecp_atoms``),
        # not a rule this generator applies.  The deck states what was
        # asked for; ``validation`` is where a structure that looks like it
        # wants an ECP gets said out loud, for a person to confirm.
        ecp_chosen = _resolve_ecp(struct, cfg)
        if ecp_chosen and v:
            _named = ", ".join(sorted(ecp_chosen))
            out += [
                f"# ECP `{cfg.ecp}` applied to: {_named}",
                f"# (from ecp_atoms = {list(cfg.ecp_atoms)!r}).  Empty either",
                "# side = no ECP; nothing is added that you did not name.",
            ]
        # Geometry warm-restart hook (task #539).  The atom literal is
        # bound to ``_atom_block`` so the if-exists block below can
        # override it from a prior run's ``<JOB>_optimized.xyz``.  The
        # runwrap's ``--cold`` glob moves ``_optimized.xyz`` aside when
        # the user wants a fresh start; otherwise the script auto-resumes
        # from the relaxed geometry on the next ``--continue`` invocation
        # (analog to SIESTA's automatic ``.XV`` read, per script-execution.md
        # "Cross-engine equivalence table").
        #
        # Two guards: ``os.path.exists`` AND ``getsize > 0`` so a stale
        # 0-byte file from a crashed prior run doesn't trigger a parse on
        # an empty file.  A parse failure (malformed XYZ) falls through to
        # the literal -- we never silently feed garbage to gto.M().
        out.append("_atom_block = '''")
        out.append(_atoms_block(struct))
        out.append("'''")
        # THE READ IS GATED ON ``restart``, and on nothing else (`run-identity.md`
        # § 4 rule 2).  The geometry this reads is the PREVIOUS rung's, so no flag
        # of this deck's own -- not ``save_optimized_xyz``, which says whether this
        # run WRITES one, and not ``optimize``, which says whether this run relaxes
        # -- can answer whether to start from it.  Until ``restart`` existed the
        # write flags doubled as the read gate, which made *"write a checkpoint but
        # do not resume from one"* unsayable: § 4's "present but not honoured".
        if continues(cfg):
            out.append('_opt_path = _mb_outfile(JOB + "_optimized.xyz")')
            out.append("if _os.path.exists(_opt_path) "
                       "and _os.path.getsize(_opt_path) > 0:")
            out.append("    try:")
            out.append("        with open(_opt_path) as _mb_xyz_fh:")
            out.append("            _xyz_lines = _mb_xyz_fh.read().splitlines()")
            # XYZ format: line 0 = atom count, line 1 = comment, lines 2..N+1
            # = "ELEM  X  Y  Z" rows.  We rebuild _atom_block as PySCF expects
            # (4 cols, whitespace-separated, Ang) and re-prefix every row with
            # the same 4-space indent the literal uses so a downstream reader
            # sees a uniform block shape.
            out.append("        _n_xyz = int(_xyz_lines[0].strip())")
            out.append("        _rows = []")
            out.append("        for _row in _xyz_lines[2:2 + _n_xyz]:")
            out.append("            _parts = _row.split()")
            out.append("            if len(_parts) < 4:")
            out.append("                raise ValueError("
                       "f\"malformed XYZ row: {_row!r}\")")
            out.append("            _el = _parts[0]")
            out.append("            _x, _y, _z = (float(_parts[1]), "
                       "float(_parts[2]), float(_parts[3]))")
            out.append("            _rows.append("
                       "f\"    {_el:<2s}  {_x:14.8f}  {_y:14.8f}  {_z:14.8f}\")")
            out.append("        _atom_block = \"\\n\".join(_rows)")
            out.append('        print(f"[molbuilder] continuation: loaded '
                       'geometry from {_opt_path} ({_n_xyz} atoms)")')
            out.append("    except (OSError, ValueError, IndexError) as _mb_e:")
            out.append('        print(f"[molbuilder] warning: could not parse '
                       '{_opt_path} ({_mb_e}); using literal geometry from script")')
        out.append("mol = gto.M(")
        out.append("    atom       = _atom_block,")
        out.append(f'    basis      = "{cfg.basis}",')
        if ecp_chosen:
            # ONE shape: ``resolve_pyscf_ecp`` returns ``{element: name}`` or
            # None, so this is a Python dict-literal every time.  It must NOT
            # be quoted -- a string-with-braces is what PySCF rejects as an
            # unknown ECP name, and that was a real bug once.  The string
            # branch that stood beside this went with the ``str | dict``
            # field (2026-08-13).
            out.append(f'    ecp        = {ecp_chosen!r},')
        out.append(f"    charge     = {charge},")
        out.append(f"    spin       = {cfg.spin},")
        out.append(f"    symmetry   = {cfg.symmetry},")
        out.append(f"    verbose    = {cfg.verbose},")
        if cfg.log_file:
            # § 1.1a consequence 1: the token reaches the engine's log, exactly as
            # it reaches SIESTA's ``.out``.  Two rungs are two processes in one
            # folder, and an unsuffixed name would have the second overwrite the
            # first.  ``JOB`` itself stays unsuffixed (consequence 2) -- what the
            # engine finds the previous rung's state by must not move.
            _logname = f"_{stage_token}.log" if stage_token else ".log"
            out.append(f'    output     = _mb_outfile(JOB + {_logname!r}),')
        # UNSET means no cap (`template-unification-plan.md` § 5.1): the line is
        # OMITTED rather than emitted as None, so PySCF uses its own default --
        # which reads the machine -- instead of being handed a Python None.
        if cfg.max_memory_mb:
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
            out.append('_save_xyz(mol, _mb_outfile(JOB + "_initial.xyz"), '
                       '"Initial geometry (input)")')
        out.append("")

        # ---------------- Unified molwatch log emitter (early, additive) ------
        # Defined and instantiated NOW -- before ``optimize()`` -- so the log
        # file (header + initial-preview block) exists the moment the script
        # starts running.  A rung's optimize() can take hours on a real molecule;
        # we don't want the Watch tab staring at "no file to load" the whole
        # time.  SCF cycle hook is wired on the production mf below; the opt-step
        # hook is wired on the ``optimize(...)`` call.
        # NOTE: the molwatch emitter is NOT constructed here.  It writes its
        # whole header -- including every ``# runtime.<key>`` line -- inside
        # __init__, so it must be built only once _RUNTIME_INFO is complete;
        # see the construction site below the SCF setup for the full story.

        # ------------------------------------------------------------- main scf
        out.append("# ============================================================")
        out.append("#  SCF setup")
        out.append("# ============================================================")
        out.append(f"mf = {('dft' if is_dft else 'scf')}.{method_class}(mol)")
        # THE FUNCTIONAL, THE GRID, DENSITY FITTING AND DISPERSION go through the
        # one door.  Twelve hand-written sentences stood here describing exactly
        # these four items -- the catalogue describes them too, and only one of the
        # two was kept in step with the declarations.
        return "\n".join(out) if out else None

    def _science_b(struct, cfg) -> str:
        """One run of the deck's science, in the order PySCF reads it.

        The layout is walked in order, so what sits between two runs of
        settings sits between them in the deck too.
        """
        out: List[str] = []
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
        # THE SCF SETTINGS GO THROUGH THE ONE DOOR
        # (`execution/script-preparation.md` § 4.2).  ``layout.SCF_SECTION`` names
        # the catalogue items and ``layout.line`` says how PySCF spells each one;
        # the framework reads the declaration, writes the catalogue's note above
        # the value, and skips whatever ``line`` declines.  The four hand-written
        # sentences that stood here said what the catalogue already says about
        # these very items -- and said it in a place nothing kept in step with the
        # declarations.
        return "\n".join(out) if out else None

    def _science_c(struct, cfg) -> str:
        """One run of the deck's science, in the order PySCF reads it.

        The layout is walked in order, so what sits between two runs of
        settings sits between them in the deck too.
        """
        out: List[str] = []
        # SCF convergence is judged on TWO quantities -- the energy change
        # and the orbital-gradient norm -- and it is the gradient that
        # decides how clean the forces are.  PySCF derives it from the
        # energy tolerance when unset (verified in scf.hf.kernel: ``if
        # conv_tol_grad is None: conv_tol_grad = numpy.sqrt(conv_tol)``),
        # so tightening conv_tol alone moves the force criterion only as a
        # square root.  Either way the script states the effective number:
        # a derived value the user cannot see is the same problem as an
        # undocumented one.
        if cfg.scf_conv_tol_grad <= 0:
            _grad_tol = math.sqrt(cfg.scf_conv_tol)
            if v:
                out.append(f"# mf.conv_tol_grad left at PySCF's default: "
                           f"sqrt(conv_tol) = {_grad_tol:.2e}.")
                out.append("# Set cfg.scf_conv_tol_grad (e.g. 1e-6) to fix it "
                           "independently of the energy tolerance.")
        if not cfg.level_shift:
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
        if cfg.chkfile:
            out.append('mf.chkfile = _mb_outfile(JOB + ".chk")')
            # Continuation: a rung that says ``continue`` starts its SCF from the
            # density the rung before it converged, instead of MINAO / atom.  That
            # turns "full SCF from scratch" into "small refine on top of a
            # converged DM", and it is the density half of what a rung hands the
            # next one -- the geometry half is the ``_optimized.xyz`` read above
            # (`stages.md` § 1.1a; SIESTA carries the same pair as .DM and .XV).
            #
            # Nested inside ``cfg.chkfile`` because PySCF reads the checkpoint
            # THROUGH ``mf.chkfile``: ``init_guess = "chkfile"`` with no path set
            # has nothing to open.  That is an engine coupling, not the write flag
            # doubling as a read gate -- ``restart`` is what decides, and this only
            # says the mechanism has to be wired up for it to be able to.
            #
            # Gated at runtime on a non-empty file as well, so a stale 0-byte
            # checkpoint from a crashed run does not trigger.  ``_os`` is imported
            # at the top of the script.
            if continues(cfg):
                out.append("_chk_path = _mb_outfile(JOB + \".chk\")")
                out.append("if _os.path.exists(_chk_path) and "
                           "_os.path.getsize(_chk_path) > 0:")
                out.append('    mf.init_guess = "chkfile"')
                out.append('    print(f"[molbuilder] continuation: loading '
                           'SCF init guess from {_chk_path}")')

        # GPU patch: promote the fully-assembled production mf to its
        # gpu4pyscf equivalent when the runtime probe at script-start
        # succeeded (_USING_GPU=True).  No-op on CPU nodes.  Called
        # AFTER density_fit / disp / PCM / chkfile / conv_tol so
        # .to_gpu() sees the complete CPU mf and the GPU mirror has
        # the same settings.
        out.append("mf = _mb_to_gpu_if_enabled(mf)")

        # Second-order (Newton-Raphson) SCF -- the last rung of the
        # convergence escalation, after DIIS and level shift / damping.
        #
        # Emitted AFTER the GPU promotion on purpose.  ``.to_gpu()`` wants
        # the fully-assembled plain SCF object (see the comment above), and
        # gpu4pyscf's own SCF classes carry ``.newton()`` (checked against
        # gpu4pyscf 1.7.0), so wrapping last works on both paths and
        # wrapping first would hand ``to_gpu`` a class it need not know.
        if cfg.scf_soscf:
            if v:
                out.append("")
                out.append("# Second-order SCF: solve for the orbital rotation "
                           "directly (Newton-Raphson)")
                out.append("# instead of extrapolating Fock matrices.  Costs "
                           "more per iteration and more")
                out.append("# memory, but converges cases where DIIS oscillates "
                           "indefinitely -- open-shell")
                out.append("# metals and near-degenerate frontier orbitals are "
                           "the usual reasons.")
                out.append("#")
                out.append("# Two behaviours change and neither is an error:")
                out.append("#   * mf.max_cycle now counts MACRO iterations "
                           "(each runs many micro-")
                out.append("#     iterations), so the same number buys far more "
                           "work than under DIIS.")
                out.append("#   * mf.diis_space / mf.damp stop applying -- the "
                           "Newton solver does not")
                out.append("#     use them.  Its own damping knob is "
                           "mf.ah_level_shift (default 0).")
            out.append("mf = mf.newton()")
            out.append('print("[molbuilder] SCF solver: second-order (SOSCF, '
                       'mf.newton()); max_cycle counts macro iterations.")')

        # Record what the SCF will ACTUALLY converge to, read off the LIVE
        # object rather than restated from the config -- a reported value
        # that cannot drift from what the run did.  conv_tol_grad stays
        # None until kernel() derives it, so apply PySCF's own rule here
        # (scf.hf.kernel: sqrt(conv_tol)) to report a number rather than a
        # null.  Which of the two it is gets recorded alongside, because
        # "we chose 3.2e-5" and "PySCF derived 3.2e-5" are different facts.
        out.append("")
        out.append("_RUNTIME_INFO['scf_conv_tol'] = float(mf.conv_tol)")
        out.append("_RUNTIME_INFO['scf_conv_tol_grad'] = float("
                   "mf.conv_tol_grad if mf.conv_tol_grad is not None "
                   "else mf.conv_tol ** 0.5)")
        out.append("_RUNTIME_INFO['scf_conv_tol_grad_source'] = ("
                   "'explicit' if mf.conv_tol_grad is not None "
                   "else 'derived: sqrt(conv_tol)')")
        out.append(f"_RUNTIME_INFO['scf_soscf'] = {bool(cfg.scf_soscf)!r}")
        out.append("_RUNTIME_INFO['scf_solver_class'] = type(mf).__name__")
        out.append("print(f\"[molbuilder] SCF convergence: energy "
                   "{_RUNTIME_INFO['scf_conv_tol']:.1e} Hartree, orbital "
                   "gradient {_RUNTIME_INFO['scf_conv_tol_grad']:.1e} \"\n"
                   "      f\"({_RUNTIME_INFO['scf_conv_tol_grad_source']}); "
                   "solver {_RUNTIME_INFO['scf_solver_class']}.\")")

        out += _emit_effective_parameters(cfg, is_dft)

        # Construct the molwatch emitter HERE -- after the SCF setup, after
        # the _RUNTIME_INFO writes above -- and then wire the callback.
        #
        # It used to be built before the SCF setup so the Watch tab had a
        # file to load early (a stage's optimize() can run for hours, and
        # "no file to load" is a bad thing to stare at).  That intent is
        # preserved: everything between there and here is attribute
        # assignment on ``mf``, and the first expensive call -- the SCF in
        # the stability block -- is still below us.
        #
        # But MolwatchEmitter.__init__ writes the ENTIRE header, every
        # ``# runtime.<key>`` line included, and a log header cannot be
        # rewritten once the first data block follows it.  Building it
        # before the SCF setup therefore froze _RUNTIME_INFO at whatever it
        # held at that moment, and silently dropped every key written after
        # -- which is exactly what happened to the five scf_* facts added
        # above: present in the process, present on stdout, absent from the
        # artifact /results actually renders.  The GPU keys survived only
        # because the probe happens to run before the old construction site.
        #
        # So the rule is: _RUNTIME_INFO is populated FIRST, the emitter is
        # built AFTER.  Any future runtime fact belongs above this line.
        if cfg.optimize and cfg.write_molwatch_log and cfg.optimizer == "geometric":
            out += _emit_molwatch_emitter(v, cfg, stage_token)
            out.append(_emit_molwatch_callback_wire("mf"))
        out.append("")

        # ---------------------------------------------- SCF + stability
        # BEFORE any geometry work.  Optimizing on a broken-symmetry saddle
        # produces a wrong geometry and wrong frequencies, and finding out
        # afterwards helps nobody.  Open-shell only; a closed-shell check is
        # a singlet->triplet question the user rarely asked.
        _stability_checked = method_class.startswith("U")
        _derived["stability_checked"] = _stability_checked
        if _stability_checked:
            out += _emit_stability_block(cfg, v)

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
                out.append("# Per-tier convergence (Gaussian-OPT family):")
                out.append("#   screening    gmax 2.0e-3 Ha/Bohr  conv_tol 1e-7  max_steps 30")
                out.append("#   loose preopt gmax 2.0e-3 Ha/Bohr  conv_tol 1e-7  max_steps 50")
                out.append("#   publishable  gmax 4.5e-4 Ha/Bohr  conv_tol 1e-9  max_steps 200")
                out.append("#                (this is the Gaussian-OPT default and what reviewers expect)")
                out.append("#   tight        gmax 1.5e-5 Ha/Bohr  conv_tol 1e-10 max_steps 100")
                out.append("#                (vib / IR / NEB barriers)")
                out.append("# The ladder in task.json ships these as its default rungs.")
                out.append("# See")
                out.append("# docs/engines/tuning.md sect. 4 for the full preset")
                out.append("# table + SIESTA <-> PySCF crosswalk + citations.")
            # Frozen-atom constraints (three-stage contract carrier).  When
            # Structure.frozen_atoms is non-empty AND we're using the
            # geomeTRIC optimizer (only one with constraint support), write
            # a sibling <JOB>.constraints.txt at run time and pass it via
            # the ``constraints=`` kwarg.  Indices are 1-based per geomeTRIC.
            # See molbuilder/structure.py + spectra/pyscf_script.py for the
            # cross-engine carrier; the spectra path uses cfg.frozen_indices
            # while Build PySCF reads struct.frozen_atoms directly so /modify
            # sidecar flows through without an explicit form field.
            frozen = list(getattr(struct, "frozen_atoms", []) or [])
            emit_constraints = bool(frozen) and cfg.optimizer == "geometric"
            _derived["emit_constraints"] = emit_constraints
            if emit_constraints:
                if v:
                    out += [
                        '# Frozen atoms from /modify sidecar (or '
                        'Structure.frozen_atoms): hold these atom positions',
                        '# fixed during the optimisation.  Indices are 1-based '
                        'per geomeTRIC; molbuilder Structure',
                        '# is 0-based so we shift below.  Without this block '
                        'geomeTRIC moves every atom.',
                    ]
                from ..engine_atom_index import geometric_atom_index
                ids_1based = ",".join(str(geometric_atom_index(i)) for i in frozen)
                out.append(f'# Source: Structure.frozen_atoms = {frozen!r}  (0-based)')
                out.append(f'_FROZEN_CONSTRAINTS_PATH = _mb_outfile(JOB + ".constraints.txt")')
                out.append('with open(_FROZEN_CONSTRAINTS_PATH, "w") as _fh:')
                out.append('    _fh.write("$freeze\\n")')
                out.append(f'    _fh.write("xyz {ids_1based}\\n")')
            elif frozen and cfg.optimizer != "geometric":
                out += [
                    f'# WARNING: Structure.frozen_atoms = {frozen!r}  (0-based)',
                    f'#   but optimizer = {cfg.optimizer!r} -- only the geomeTRIC',
                    '#   optimizer supports frozen-atom constraints.  Switch to',
                    "#   ``cfg.optimizer = 'geometric'`` to honor the sidecar.",
                ]
        return "\n".join(out) if out else None

    def _science_d(struct, cfg) -> Optional[str]:
        """The run itself: the optimiser call, or the single point.

        Split from the block above it on 2026-08-19 so the six convergence
        targets could be a SECTION of the layout rather than a section
        rendered inside a block.  The comment that stood at the old call
        site said the section *cannot* be a top-level member because it is
        emitted inside the optimise branch; a branch is a reason to omit a
        member, not a reason to hide one -- `spec_for` holds the config and
        can answer which (`script-preparation.md` § 4.1).
        """
        out: List[str] = []
        if cfg.optimize:
            out += _emit_optimization(
                cfg, _derived["emit_constraints"], stage_token)
        else:
            if v:
                out.append("# ============================================================")
                out.append("#  Single-point SCF (no optimization)")
                out.append("# ============================================================")
            if not _derived["stability_checked"]:
                out.append("e = mf.kernel()")
            else:
                # Already converged AND stability-checked above; re-running
                # would repeat the work and discard the stabilised orbitals.
                out.append("# SCF converged + stability-checked above.")
            out.append('print(f"Total energy: {e:.8f} Hartree")')
            out.append("mol_eq = mol")
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
            out.append('_save_xyz(mol_eq, _mb_outfile(JOB + "_optimized.xyz"), '
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

        return "\n".join(out) if out else None


    # WHAT THIS DECK DERIVED -- the one channel between layout members, and
    # the same one `siesta/input.py` keeps.  A block that closes over its own
    # locals cannot be split, and splitting is what puts a section where the
    # framework can see it; `emit_constraints` and `stability_checked` are
    # worked out while one block renders and read by the next.
    _derived: dict = {}

    # ----- ONE DeckSpec, and the framework runs the step -----
    # The reader's section, the record blocks and the banner are the
    # framework's (`script-preparation.md` § 4.2a): this writer assembled them
    # itself until 2026-08-18, which made them two copies of one idea -- the
    # half of roadmap P4 that phase 1 did not close.
    spec = _sc.DeckSpec(
        engine="pyscf",
        layout=(_sc.Block("system, molecule and the mean field", _science_a),
                _layout.DFT_SECTION,
                _sc.Block("solvent and the SCF preamble", _science_b),
                _layout.SCF_SECTION,
                _sc.Block("what the run writes", _science_c),
                *((_layout.GEOMETRY_SECTION,) if cfg.optimize else ()),
                _sc.Block("the run itself", _science_d)),
        line=_layout.line(cfg, is_dft=is_dft),
        # section_title: the framework's default.  Both engines write a
        # heading as a `#` comment, so both restated the default verbatim
        # until 2026-08-19 -- two more copies of one string, and a slot
        # that LOOKED exercised.  It stays a slot because the comment
        # character is genuinely an engine's syntax; it is simply not one
        # these two differ on.
        provenance_defaults=lambda c: {
            "use_gpu":       str(bool(getattr(c, "use_gpu", False))).lower(),
            "density_fit":   str(bool(getattr(c, "density_fit", True))).lower(),
            "threads": ("auto" if getattr(c, "threads", None) is None
                        else str(c.threads)),
            "max_memory_mb": ("no cap" if not c.max_memory_mb
                              else str(int(c.max_memory_mb))),
        },
        created_by="molbuilder render_script",
        check_rules=_layout.check_rules,
    )
    return spec



def render_script(struct: Structure,
                  config: Optional[PySCFConfig] = None,
                  *, stage_token: Optional[str] = None) -> str:
    """Format a Structure as a runnable PySCF script (Python text).

    **A thin call over :func:`spec_for`.**  The engine describes its deck; the
    framework renders it.  This name survives because the test suite and the
    ``convert`` route point at it -- what moved is what it does, not what it is
    called (`archive/2026-08-18-preparation-backend-plan.md` § 3.1a).

    Prefer ``spec_for`` + ``script_emit.prepare_deck`` where a deck is being
    WRITTEN: that runs validate -> render -> write -> check in one place (§ 4.3).
    """
    spec = spec_for(struct, config, stage_token=stage_token)
    cfg = config or PySCFConfig()
    return _sc.render_deck(spec, struct, cfg,
                                  verbose=cfg.verbose_comments)

def _emit_effective_parameters(cfg: PySCFConfig, is_dft: bool) -> List[str]:
    """Record, at run time, **every** parameter this run is set to.

    Three columns, because three different questions get asked when a result
    looks wrong:

    * **catalogue** -- what this project recommends. Tells you whether a value
      was a deliberate choice or just what you got.
    * **this run** -- what the description resolved to. The request.
    * **engine** -- what ``mol`` / ``mf`` actually hold, read off the live
      objects after setup and before any work starts.

    **The third column is why this exists.** A value that silently failed to
    apply -- a solver that overrode it, a wrapper that changed what it counts --
    shows up as a *disagreement between columns two and three*. A record that
    only echoed our own intent could never show that.  ``-`` means we did not ask -- the
    engine has no such setting: a molbuilder-level flag like ``save_optimized_xyz``
    still decides what the run produces, so it is recorded, but there is
    nothing to read it back from.  ``(absent)`` means we asked and the object
    did not have it, which is itself worth seeing.

    **Full coverage, and generated.**  Every catalogue item this engine
    declares appears -- values a person changed, values left at the default,
    and values that never reach the engine.  The list comes from the catalogue
    and the read-back path from each item's own ``anchor``, so a new parameter
    joins the record with no edit here.  A hand-kept list would answer *what
    somebody remembered to add*.
    """

    v = cfg.verbose_comments
    out: List[str] = [""]
    if v:
        out.append("# Every parameter this run is set to, recorded before any")
        out.append("# work starts.  The last column is read off the live PySCF")
        out.append("# objects: if it disagrees with the one before it, the")
        out.append("# setting did not take.")
    out.append("def _mb_read(_fn):")
    out.append("    try:")
    out.append("        return _fn()")
    out.append("    except Exception:")
    out.append("        return '(absent)'")
    out.append("")
    out.append("_MB_PARAMS = {}")
    for name in _layout.recorded_items():
        param = _sc.parameter(name, "pyscf", config=cfg)
        expr = _layout.readback(param)
        if expr and not is_dft and name in ("functional", "grid_level",
                                            "dispersion"):
            expr = None       # no such attribute on a Hartree-Fock object
        third = f"_mb_read(lambda: {expr})" if expr else "'-'"
        out.append(f"_MB_PARAMS[{name!r}] = ({param.default!r}, "
                   f"{param.value!r}, {third})")
    out.append("")
    _fmt = '"#   %-26s %-18s %-18s %s"'
    out.append('print("' + _sc.begin_marker(_sc.BLOCK_PARAMETERS) + '")')
    out.append(f'print({_fmt} % ("parameter", "catalogue", "this run", '
               f'"engine"))')
    out.append("for _k, (_d, _r, _e) in _MB_PARAMS.items():")
    out.append(f"    print({_fmt} % (_k, _d, _r, _e))")
    out.append('print("' + _sc.end_marker(_sc.BLOCK_PARAMETERS) + '")')
    out.append("# the effective value where there is one, the request otherwise")
    out.append("_RUNTIME_INFO.update({_k: (_r if _e == '-' else _e)")
    out.append("                      for _k, (_d, _r, _e) in _MB_PARAMS.items()})")
    return out


def _emit_stability_block(cfg: PySCFConfig, v: bool) -> List[str]:
    """Initial SCF + open-shell stability loop, emitted BEFORE any
    geometry work.

    Why before.  An open-shell SCF can converge to a broken-symmetry
    SADDLE point: the energy stops changing, ``mf.converged`` is True,
    and the wavefunction is still not the variational minimum.  Until
    2026-08-13 this script called ``mf.stability()`` AFTER the whole
    optimization and only printed what it found -- so a run that landed
    on a saddle at the first geometry optimized every subsequent step
    and computed its frequencies on the wrong electronic state, then
    said so at the end.  The emitter's own comment named the remedy and
    declined to apply it.

    What it does now: converge, check, and if the check hands back
    better orbitals, rebuild the density matrix from them and converge
    again -- up to three times.  Persistent instability WARNS and
    continues (user decision 2026-08-13): a hint does not get to end the
    run.

    Measured on O2 triplet UHF/STO-3G, which is internally unstable:
    the first SCF reports -147.63404851 Ha as converged; one restart
    from the suggested orbitals gives -147.63555611 Ha, 1.5 mHa lower,
    and the next check then passes.

    Emitted only for UKS / UHF.  Closed-shell stability is a
    singlet->triplet question that is rarely the user's concern and
    costs a check nobody asked for.
    """
    out: List[str] = []
    if v:
        out += [
            "# ============================================================",
            "#  SCF + open-shell stability check",
            "# ============================================================",
            "# An open-shell SCF can settle on a broken-symmetry SADDLE",
            "# point: the energy stops moving and `mf.converged` is True,",
            "# but the wavefunction is NOT the lowest-energy one.  Every",
            "# geometry step and every frequency computed afterwards would",
            "# then describe the wrong electronic state.",
            "#",
            "# So we converge, ask `mf.stability()`, and if it hands back",
            "# better orbitals we rebuild the density matrix from them and",
            "# converge again -- before any geometry work starts.",
            "#",
            "# On O2 triplet (UHF/STO-3G) the first SCF reports",
            "# -147.63404851 Ha as converged; one restart gives",
            "# -147.63555611 Ha -- 1.5 mHa lower -- and is then stable.",
            "#",
            "# A run that is still unstable after the retries WARNS and",
            "# continues: this is advice, not a veto.  The log says which",
            "# of the three outcomes happened, so 'checked and stable' is",
            "# never confused with 'never checked'.",
        ]
    out += [
        'print("\\n=== Stage: SCF + stability ===")',
        "e = mf.kernel()",
        'print(f"[molbuilder] initial SCF: {e:.8f} Hartree '
        '(converged={mf.converged})")',
        "",
        f"_MB_STABILITY_MAX = {int(_STABILITY_MAX_RESTARTS)}",
        f"_MB_STABILITY_ETOL = {_STABILITY_ENERGY_TOL!r}",
        "_MB_STABILITY_ROUNDS = 0",
        "_MB_STABLE = None          # None = could not check",
        "import numpy as _mb_np",
        "try:",
        "    for _r in range(1, _MB_STABILITY_MAX + 1):",
        "        _internal = mf.stability()[0]",
        "        if _mb_np.allclose(_mb_np.asarray(_internal),",
        "                           _mb_np.asarray(mf.mo_coeff)):",
        "            _MB_STABLE = True          # nothing suggested at all",
        "            break",
        "        _e_prev = e",
        "        e = mf.kernel(mf.make_rdm1(_internal, mf.mo_occ))",
        "        if e < _e_prev - _MB_STABILITY_ETOL:",
        "            _MB_STABILITY_ROUNDS = _r",
        '            print(f"[molbuilder] stability round {_r}: '
        'instability repaired, "',
        '                  f"{_e_prev:.8f} -> {e:.8f} Hartree '
        '(dE={e - _e_prev:+.3e})")',
        "            continue",
        # The energy did not fall, so the orbitals stability() handed
        # back are not a better solution -- they are the SAME solution
        # expressed differently.  Degenerate shells (O2's pi pair, any
        # symmetric radical) are rotated freely within the degenerate
        # space, so a coefficient comparison flags them forever while
        # the physics is identical.  The energy is the criterion; the
        # coefficients are not.
        "        _MB_STABLE = True",
        '        print(f"[molbuilder] stability round {_r}: suggested '
        'orbitals gave no "',
        '              f"further improvement (dE={e - _e_prev:+.3e}); '
        'treating as stable. "',
        '              f"Common for degenerate shells, which are rotated '
        'freely within "',
        '              f"the degenerate space.")',
        "        break",
        "    else:",
        "        _MB_STABLE = False",
        "except (NotImplementedError, AttributeError) as _exc:",
        # Law A: a check that could not run says so.  Silence would read
        # as a clean bill of health.
        '    print(f"[molbuilder] stability: NOT CHECKED -- this method '
        'does not implement it ({_exc})")',
        "",
        "# One line, three distinguishable outcomes.",
        "if _MB_STABLE is None:",
        '    print("[molbuilder] stability: NOT CHECKED. The energy below '
        'has not been tested for a broken-symmetry solution.")',
        "elif _MB_STABLE and _MB_STABILITY_ROUNDS == 0:",
        '    print("[molbuilder] stability: CHECKED, stable on the first '
        'SCF (no restart needed).")',
        "elif _MB_STABLE:",
        '    print(f"[molbuilder] stability: CHECKED, reached a stable '
        'solution after {_MB_STABILITY_ROUNDS} restart(s).")',
        "else:",
        '    print(f"[molbuilder] stability: WARNING -- still internally '
        'unstable after {_MB_STABILITY_MAX} restarts. "',
        '          "The results below may describe a saddle point, not '
        'the ground state. "',
        '          "Continuing because this is advice, not a veto -- but '
        'treat the geometry and any "',
        '          "frequencies as provisional.")',
        "",
    ]
    return out


def _emit_optimization(cfg: PySCFConfig,
                       emit_constraints: bool,
                       stage_token: Optional[str] = None
                       ) -> List[str]:
    """The relaxation this deck runs — **one rung, one call**.

    `stages.md` § 1.1a: a PySCF ladder is N decks and N jobs exactly as SIESTA's
    is, so a deck carries one rung's targets and there is nothing here to loop
    over.  The reason is *a ladder exists so that somebody looks between the
    rungs*, and a single process running every rung ends once, at the end.

    The rung hands the next one exactly what SIESTA's does: the geometry
    (``<JOB>_optimized.xyz``) and the converged density (``<JOB>.chk``), both in
    PySCF's warm-file vocabulary, and both read back by a rung whose ``restart``
    says ``continue``.

    The six convergence targets come through the one door, so each arrives with
    the catalogue's explanation of what it does and what this project set it to.
    """
    v = cfg.verbose_comments

    # THE SIX TARGETS ARE NOT WRITTEN HERE.  ``GEOMETRY_SECTION`` is a member of
    # the layout, above this block, so the framework walks it and knows what it
    # said to write -- which is what lets the check gate ask whether each target
    # survived into the file.  This function rendered the section itself until
    # 2026-08-19, on the reasoning that *"the section cannot be a top-level
    # layout member because it is emitted inside the optimise branch"*.  A
    # branch is a reason to OMIT a member, not to hide one: `spec_for` holds the
    # config and answers which (`script-preparation.md` § 4.1).
    out: List[str] = [""]
    if v:
        out.append("# One optimize() call: this deck IS one rung of the ladder.")
        out.append("# The ladder lives in task.json and runs as separate jobs,")
        out.append("# so that a person can look at this geometry before")
        out.append("# spending anything on the next rung.")
    out.append("def _mb_run_optimization(_hard_fail):")
    out.append("    return optimize(")
    out.append("        mf,")
    out.extend(_layout.geom_kwargs())
    out.append("        assert_convergence    = _hard_fail,")
    if emit_constraints:
        out.append("        constraints           = _FROZEN_CONSTRAINTS_PATH,")
    if cfg.write_trajectory and cfg.optimizer == "geometric":
        # The rung's own trajectory name.  Two rungs are two processes writing
        # into one calculation, so the one name this script chooses for itself
        # has to say which rung it is (`stages.md` § 1.1a, consequence 1).
        _traj = (f"JOB + '_geom_{stage_token}'" if stage_token
                 else "JOB + '_geom'")
        out.append(f"        prefix                = _mb_outfile({_traj}),")
    if cfg.write_molwatch_log and cfg.optimizer == "geometric":
        out.append("        callback              = _molwatch.opt_step_hook,")
    out.append("    )")
    out.append("")
    policy = (cfg.on_nonconvergence or "halt").strip().lower()
    if v:
        out += _sc.parameter("on_nonconvergence", "pyscf").note()
        if policy == "continue":
            out += _sc.parameter("geom_continue_retries", "pyscf").note()
        out.append(f"# This rung: {policy!r}.")
    if policy == "proceed":
        if v:
            out.append("#   take whatever geomeTRIC produced when the step")
            out.append("#   budget ran out; the next rung starts from it.")
        out.append("mol_eq = _mb_run_optimization(_hard_fail=False)")
    elif policy == "continue":
        retries = int(cfg.geom_continue_retries or 0)
        if v:
            out.append(f"#   retry the same targets up to {retries} more time(s)")
            out.append("#   (total budget = max_steps x (1 + retries)), then raise.")
        out.append(f"_budget = 1 + {retries}")
        out.append("for _attempt in range(_budget):")
        out.append("    try:")
        out.append("        mol_eq = _mb_run_optimization(_hard_fail=True)")
        out.append("        break")
        out.append("    except RuntimeError as _e:")
        out.append("        if 'not converged' not in str(_e).lower():")
        out.append("            raise            # a genuinely different error")
        out.append("        if _attempt == _budget - 1:")
        out.append("            raise            # exhausted -> halt")
        out.append('        print(f"WARN: did not converge in "')
        out.append('              f"{_GEOM_MAX_STEPS} steps; retrying "')
        out.append('              f"({_budget - 1 - _attempt} left)")')
    else:
        if v:
            out.append("#   raise on non-convergence rather than hand on a")
            out.append("#   geometry nobody accepted.")
        out.append("mol_eq = _mb_run_optimization(_hard_fail=True)")
    out.append("")
    if v:
        out.append("# Re-converge SCF at the relaxed geometry: snapshot the")
        out.append("# converged density so we do not restart from MINAO, drop")
        out.append("# the stale integrals, then converge at mol_eq.  That is")
        out.append("# the state the answer below is read from.")
    out.append("dm_prev = (mf.make_rdm1()")
    out.append("           if mf.mo_coeff is not None and mf.mo_occ is not None")
    out.append("           else None)")
    out.append("mf.reset(mol_eq)")
    out.append("mf.kernel(dm0=dm_prev)")
    out.append('print(f"\\nFinal energy: {mf.e_tot:.8f} Hartree")')
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
        # WHY and COST are this parameter's own reasons, so they come from
        # its declaration -- which also puts them in front of whoever is
        # deciding on the form, where the hand-typed version never reached.
        # What stays here is guidance about the METHOD: the RRHO
        # approximation and where it misleads, which no single parameter owns.
        out += _sc.parameter("compute_frequencies", "pyscf").note()
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
    out.append("    with open(_mb_outfile(JOB + \".thermo.txt\"), \"w\") as _mb_fh:")
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
                   "Perturb along the imag coord and re-optimize, or "
                   "tighten geom_grms and run this rung again.\")")
    out.append("except Exception as _mb_exc:")
    out.append("    print(f\"Frequency analysis FAILED: {_mb_exc}\\n\"")
    out.append("          f\"Converged energy + optimized geometry are still "
               "on disk; rerun with --no-compute-frequencies to skip.\")")
    out.append("")
    return out


def _emit_molwatch_emitter(v: bool, cfg: "PySCFConfig",
                           stage_token: Optional[str] = None) -> List[str]:
    """Inline streaming writer for this rung's ``.molwatch.log``.

    ``stage_token`` names the rung, and it reaches two things: the log's own
    filename and the one entry of its convergence-target map.  § 1.1a
    consequence 1 -- a ladder is N decks and N jobs, so two rungs are two
    processes in one folder and an unsuffixed log would have the second
    overwrite the first.

    The emitter is instantiated **early** -- before ``optimize()`` -- so the
    log file (header + initial-preview block) exists from the moment the
    script starts running.  A rung can take hours on a real molecule; without
    this ordering the Watch tab would have no file to load until it finished,
    defeating the "live trajectory" promise.

    Hooks are wired once on the production mf:

      * ``mf.callback = _molwatch.scf_cycle_hook``   (every SCF cycle)
      * ``optimize(mf, ..., callback=_molwatch.opt_step_hook)`` (every
        accepted opt step)

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
    # Instantiate as early as possible (BEFORE ``optimize()``) so
    # the log file -- with header + initial-preview block -- exists
    # the moment the script starts running.  Otherwise a long rung
    # (which can take hours on a real molecule) would mean the
    # user has nothing to load on the Watch tab until that stage
    # finishes.  The SCF callback is wired on the production mf below.
    # This rung's log takes the basename of the deck that produced it, and
    # that is ONE rule rather than two (`stages.md` § 7) -- so the name comes
    # from ``molwatch_log_basename`` rather than being spelled again here.
    # The generated script gets the resolved suffix as a literal so it stays
    # self-contained at runtime (no molbuilder import on the user's machine).
    from ..trajectory_log.format import molwatch_log_basename
    _placeholder      = "_X_"
    _resolved_for_X   = molwatch_log_basename(_placeholder, stage_token)
    # Strip the placeholder; what's left is the suffix the generator
    # appends to ``JOB`` at runtime.
    _suffix = _resolved_for_X[len(_placeholder):]
    # Convergence targets for the molwatch header.  ONE entry, because one
    # deck is one rung (§ 1.1a): the values are this config's own, already
    # resolved by `prep` from the description ⊕ this stage's overrides.  The
    # molwatch parser keys them as
    # ``runtime_info["convergence_targets"][<stage>][<leaf>]`` and the JS
    # reader (web/static/lib/trajectory/core.js) flattens to the last stage
    # for the threshold-line render, which with one entry is that entry.
    #
    # The key is the stage TOKEN, the same string this log's own filename
    # carries, so a log and the targets inside it name the rung identically.
    #
    # All six geomeTRIC criteria go in, not a subset: the Results-tab
    # inspector draws a threshold line for any of the five convergence checks
    # (max-grad, RMS-grad, max-displ, RMS-displ, energy-step) plus the two
    # iteration caps, and a value that never reaches the log is a value the
    # user set and no plot shows.
    #
    # geomeTRIC's gmax is in Ha/Bohr; convert to eV/Å (the unit the
    # Results-tab force plot uses) so the threshold lines land on
    # the right y-value.  Conversion constant 51.42208619 = ASE /
    # NIST historical convention (matches MolwatchEmitter's
    # HARTREE_BOHR_TO_EV_ANG).  dmax / drms are already in Angstrom
    # (geomeTRIC's source, not its docs); etol is Hartree -> eV for
    # symmetry with the force plot.
    _ha_bohr_to_ev_ang = 51.42208619
    _max_force_eV = float(cfg.geom_gmax) * _ha_bohr_to_ev_ang
    _rms_force_eV = float(cfg.geom_grms) * _ha_bohr_to_ev_ang
    _energy_tol_eV = float(cfg.geom_etol) * 27.211386245988
    out.append("_CONVERGENCE_TARGETS = {")
    out.append(f"    {(stage_token or 'run')!r}: {{")
    out.append(f"        'max_force_tol_eV_per_A': {_max_force_eV!r},")
    out.append(f"        'rms_force_tol_eV_per_A': {_rms_force_eV!r},")
    out.append(f"        'max_displ_ang':          {float(cfg.geom_dmax)!r},")
    out.append(f"        'rms_displ_ang':          {float(cfg.geom_drms)!r},")
    out.append(f"        'energy_step_tol_eV':     {_energy_tol_eV!r},")
    out.append(f"        'scf_energy_tol':         {float(cfg.scf_conv_tol)!r},")
    out.append(f"        'max_scf_iter':           {int(cfg.scf_max_cycle)!r},")
    out.append(f"        'max_geom_iter':          {int(cfg.geom_max_steps)!r},")
    out.append("    },")
    out.append("}")
    out.append(f'_molwatch = MolwatchEmitter('
               f'_mb_outfile(JOB + {_suffix!r}), JOB, mol, '
               f'runtime_info=_RUNTIME_INFO, '
               f'convergence_targets=_CONVERGENCE_TARGETS)')
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
    given mean-field object.  Called once on the production mf so the
    molwatch log captures every SCF iteration this rung runs."""
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
    out.append("#   * Loosen this rung's gmax / grms and prep it again")
    out.append("#   * Add a looser warm-up rung ahead of it in task.json")
    out.append("#   * Raise this rung's geom_max_steps")
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
    # THROUGH THE CODEC -- the one reader of a structure AND its sidecar
    # (`model/structure.md` § 2.4).  This used the bare loader, so a structure
    # whose `.molstruct.json` names frozen atoms produced a script with NO
    # geomeTRIC `$freeze` block and relaxed every atom, silently.  `prep` and
    # the web route have always used the codec; the single-shot converters of
    # BOTH engines did not.  A bare `.xyz` with no sidecar reads as before.
    from ..workingcopy_structure import StructureCodec
    if ext in (".pdb", ".xyz", ""):
        struct = StructureCodec().load(p)
    else:
        raise ValueError(
            f"unsupported input extension {ext!r}; expected .xyz or .pdb"
        )
    out_p = Path(py_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # STEP 3, WHOLE, IN ONE CALL -- the same call `prep` makes.
    # THE CHECK GATE RUNS ON EVERY ROUTE THAT WRITES A DECK: `prep` is not the
    # only door, this one is the CLI's, and a script that does not parse is no
    # less broken for having been produced here.  Two routes writing the same
    # artifact and only one of them checking it is the shape of defect this
    # layer exists to end -- which is why the order is the framework's and not
    # restated per route (`script-preparation.md` § 4.3).
    _sc.prepare_deck(spec_for(struct, cfg), struct, cfg, out_p)
    return {
        "py":      str(out_p),
        "n_atoms": struct.n_atoms,
        "charge":  _resolve_charge(struct, cfg),
        "label":   cfg.job_name,
    }
