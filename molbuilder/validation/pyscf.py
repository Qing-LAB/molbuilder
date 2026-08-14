"""PySCF-specific validators + the PySCFConfig aggregator.

The aggregator ``_validate_pyscf`` is what gets registered against
``PySCFConfig`` in the engine-validator registry; its CALL ORDER is
the public contract (every test that counts issues by position
depends on it).  This module preserves that order verbatim from the
pre-2026-06-13 flat ``molbuilder/validation.py``.

Split per docs/science/validation.md  No logic
changes; relocation only.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..issues import Issue
from ..structure import Structure
from .chemistry import (_check_ecp_declared_for_the_atoms_that_usually_want_one,
                        _check_metal_basis_adequacy,
                        check_open_shell_metal,
                        _check_peptide_protonation)
from .sidecar import _check_frozen_atoms_consumed


# --------------------------------------------------------------------- #
#  Shared PySCF numerical-grid rule (the ONE body — was duplicated).    #
#                                                                       #
#  The grid-sensitive XC class is META-GGA (τ-dependent: SCAN/TPSS/     #
#  M06-L/…), NOT "hybrids" -- a hybrid's HF exchange is analytic (off   #
#  the DFT grid), so pure hybrid-GGAs are grid-robust.  Below grid      #
#  level 4 a meta-GGA's oscillatory integrand picks up grid noise that  #
#  dominates forces / frequencies.  The SAME gate matters at two call   #
#  sites: geometry-opt FORCES (_validate_pyscf) and spectra HESSIAN     #
#  frequencies (spectra/pyscf_engine.render_checks).  Both used to      #
#  carry a duplicated copy keyed WRONGLY on "hybrid" (V4 dedup; the     #
#  meta-GGA re-key + corrected rationale is the scientific-audit fix).  #
#  One detector-pair + one gate now; message context-selected.          #
# --------------------------------------------------------------------- #

GRID_FLOOR = 4

# Substring markers of a hybrid (fraction-of-HF-exchange) functional.
# Deny-list-by-substring because the functional namespace is sprawling
# (B3*, PBE0, M06*, ωB97*, CAM-B3LYP, TPSS0, MN15, HSE, …).  This gate
# only drives a benign advisory, so a false positive is harmless and a
# false negative just skips the hint; conservative = treat as hybrid.
_HYBRID_MARKERS = ("b3", "pbe0", "bhandh", "m06", "mn15", "cam-",
                   "wb97", "ωb97", "tpss0", "x3lyp", "b97", "hse")


def is_hybrid_functional(name: str) -> bool:
    """True if ``name`` names a hybrid functional (has HF exchange)."""
    n = (name or "").lower()
    return any(tag in n for tag in _HYBRID_MARKERS)


# Meta-GGA (and hybrid-meta) functionals depend on the kinetic-energy density
# τ (and sometimes ∇²ρ).  Their XC integrand is far more oscillatory than
# LDA/GGA, so the numerical (Becke) integration grid must be dense or the
# Hessian / forces pick up grid noise.  THIS is the grid-sensitive class --
# NOT "hybrids": a hybrid's HF-exchange is evaluated analytically from the ERIs
# (PySCF ``get_k``), never on the DFT grid, so pure hybrid-GGAs (B3LYP, PBE0)
# are comparatively grid-robust.  SCAN / r²SCAN / TPSS / M06-L / B97M-V are the
# functionals that genuinely need level ≥ 4 for smooth frequencies.
# (Mardirossian & Head-Gordon, Mol. Phys. 115, 2315 (2017).)
_META_GGA_MARKERS = ("scan", "tpss", "m06", "m08", "m11", "mn12", "mn15",
                     "b97m", "revtpss")


def is_meta_gga_functional(name: str) -> bool:
    """True if ``name`` names a meta-GGA / hybrid-meta functional (τ-dependent
    XC → grid-sensitive)."""
    n = (name or "").lower()
    return any(tag in n for tag in _META_GGA_MARKERS)


def check_dft_grid_level(cfg, *, context: str) -> List[Issue]:
    """Grid-density advisory for a DFT Hessian / geometry opt (or nothing).

    Fires for the GRID-SENSITIVE functional classes -- meta-GGAs (τ-dependent:
    SCAN/TPSS/M06-L/… -- the physically-correct target) and hybrids (kept as a
    conservative superset; harmless since grid ≥ 4 is never wrong).  Pure
    LDA/GGA (PBE, BLYP, BP86, revPBE, RPBE) is grid-robust and not flagged.

    ``context`` selects the rationale:
      * ``"optimisation"`` — geometry-opt forces (Build tab).
      * ``"spectra"``       — Hessian / harmonic frequencies (Spectra tab).
    """
    grid = getattr(cfg, "grid_level", None)
    functional = getattr(cfg, "functional", "") or ""
    meta = is_meta_gga_functional(functional)
    hybrid = is_hybrid_functional(functional)
    if grid is None or grid >= GRID_FLOOR or not (meta or hybrid):
        return []
    kind = ("meta-GGA (τ-dependent)" if meta
            else "hybrid") + f" functional ({functional})"
    if context == "spectra":
        msg = (f"Grid level {grid} is below the recommended minimum of "
               f"{GRID_FLOOR} for a {kind}.  Semi-local meta-GGA XC "
               f"(kinetic-energy-density dependent) is sensitive to the "
               f"numerical integration grid; below level {GRID_FLOOR} the "
               f"grid noise typically dominates the frequency error.  Raise "
               f"the grid level for publication-quality results.")
    else:
        msg = (f"grid_level = {grid} with a {kind}: the τ-dependent semi-local "
               f"XC is grid-sensitive, so forces are noisy at this grid "
               f"density (~1e-4 Ha/Bohr floor).  Bump to grid_level = "
               f"{GRID_FLOOR} for production geometry optimisation; level 3 is "
               f"fine for energies / screening only")
    return [Issue("warn", msg, "config.grid_level")]


# --------------------------------------------------------------------- #
#  PySCF aggregator                                                     #
#                                                                       #
#  CALL ORDER IS LOAD-BEARING.  Tests that count issues by position    #
#  depend on this exact sequence.  Do not reorder.                     #
# --------------------------------------------------------------------- #


def _check_periodic_structure_in_a_gas_phase_script(
        struct: Structure) -> List[Issue]:
    """THE EMITTER IS GAS-PHASE.  Say so when the structure is not.

    This renderer builds a molecular ``gto.M()``.  It has no lattice, no
    k-points, and no way to express one -- a periodic calculation is PySCF's
    ``pbc`` module, which is a different builder entirely.  So a structure with
    a repeating axis produces a script that quietly drops the cell and computes
    an ISOLATED CLUSTER instead: not a rough version of what was asked for, a
    different calculation.

    Nothing said so until 2026-08-03.  A three-axis-periodic NaCl cell with a
    5.6 Å lattice generated a two-atom gas-phase script, the lattice appeared
    nowhere in it, and no check mentioned the difference.

    WARN, NOT ERROR, and that is the project's rule rather than a hedge: an
    isolated-cluster calculation of a periodic input is legal and occasionally
    deliberate, and only the physically impossible refuses (``report()`` raises
    on error severity, so an error here would mean no script at all).  The user
    decides; the user is told first.

    Keyed on ``axis_kind``, which is the authoritative field and is never None
    after construction -- ``pbc`` is its derived view and collapses `transport`
    into the same True as `periodic`.  Both are wrong for a gas-phase script,
    and both are named here for what they are.
    """
    kinds = tuple(struct.axis_kind or ("isolated", "isolated", "isolated"))
    repeating = [("abc"[i], k) for i, k in enumerate(kinds) if k != "isolated"]
    if not repeating:
        return []
    where = ", ".join(f"{axis} ({kind})" for axis, kind in repeating)
    cell_text = "no explicit lattice"
    if struct.cell is not None:
        lengths = np.linalg.norm(np.asarray(struct.cell, dtype=float), axis=1)
        cell_text = ("lattice lengths "
                     + " × ".join(f"{v:.3g} Å" for v in lengths))
    return [Issue(
        "warn",
        f"This structure is periodic along {where}, but the PySCF script "
        f"generated here is GAS-PHASE: it builds a molecular gto.M() with no "
        f"lattice and no k-points, so your cell ({cell_text}) is NOT used and "
        f"the result is an isolated cluster of {struct.n_atoms} atoms. For a "
        f"periodic calculation use SIESTA, or set the axes to 'isolated' "
        f"(Modify → Cell tab) if an isolated cluster is what you want.",
        "cell.periodic_in_gas_phase",
    )]


def _validate_pyscf(struct: Structure, cfg,
                    cell: Optional[np.ndarray] = None, **_) -> List[Issue]:
    """PySCF-specific checks.

    ``cell`` is not used to BUILD anything -- this emitter is gas-phase -- but
    the structure's own periodicity is checked, because a periodic structure
    reaching a gas-phase emitter is a silent change of calculation
    (``_check_periodic_structure_in_a_gas_phase_script``).  The argument is
    accepted for signature uniformity with the engine-validator registry.
    """
    issues: List[Issue] = []

    # Periodicity vs what this emitter can express.  FIRST, because it changes
    # what every other finding is about: the rest describe a cluster
    # calculation, and this says whether you asked for one.
    issues += _check_periodic_structure_in_a_gas_phase_script(struct)

    # Open-shell metal + closed-shell SCF: shared rule with SIESTA.
    method_upper = (getattr(cfg, "method", "") or "").upper()
    issues += check_open_shell_metal(
        struct,
        is_closed_shell=(getattr(cfg, "spin", 0) == 0
                         and method_upper in ("RKS", "RHF")),
        engine_label=f"PySCF (spin=0, method={cfg.method})",
    )

    # Frozen-atom carrier (three-stage contract).  PySCF emits the
    # geomeTRIC constraints file only when ``cfg.optimize`` is True
    # AND ``cfg.optimizer == 'geometric'`` -- berny's API doesn't
    # accept a constraints file in pyscf.geomopt.berny_solver, and
    # single-point runs have nothing to constrain.
    _opt_geometric = (bool(getattr(cfg, "optimize", False))
                      and getattr(cfg, "optimizer", "") == "geometric")
    _drop_reason = ""
    if not bool(getattr(cfg, "optimize", False)):
        _drop_reason = "cfg.optimize = False (single-point energy; no relaxation)"
    elif getattr(cfg, "optimizer", "") != "geometric":
        _drop_reason = (
            f"cfg.optimizer = {cfg.optimizer!r}; only the geomeTRIC "
            f"optimizer accepts a constraints file in PySCF's geomopt API"
        )
    issues += _check_frozen_atoms_consumed(
        struct,
        engine="PySCF",
        honored=_opt_geometric,
        reason_when_dropped=_drop_reason,
    )
    # Basis adequacy for transition metals.
    issues += _check_metal_basis_adequacy(
        struct, basis=getattr(cfg, "basis", ""),
        engine_label=f"PySCF method={cfg.method}",
    )

    # The ECP hint -- directly after basis adequacy, because the two are the
    # same conversation: what the basis covers, and what the core potential
    # covers.  It ASKS.  Since 2026-08-13 nothing picks an ECP for the user,
    # so this is the only place a bare all-electron Pt gets mentioned.
    issues += _check_ecp_declared_for_the_atoms_that_usually_want_one(
        struct,
        ecp=getattr(cfg, "ecp", "") or "",
        ecp_atoms=getattr(cfg, "ecp_atoms", ()) or (),
        basis=getattr(cfg, "basis", ""),
        engine_label=f"PySCF method={cfg.method}",
    )

    # spin = 2S; must be a non-negative integer.  PySCFConfig exposes
    # spin as an int with default 0; a negative value is meaningless
    # (2S is the count of unpaired electrons, never negative).
    if getattr(cfg, "spin", 0) < 0:
        issues.append(Issue(
            "error",
            f"spin = {cfg.spin} is negative; spin counts unpaired "
            f"electrons (2S), must be 0 or positive",
            "config.spin",
        ))

    # spin > 0 with a closed-shell method (RKS/RHF) is silently wrong:
    # the user wanted open-shell.
    method = (getattr(cfg, "method", "") or "").upper()
    if cfg.spin > 0 and method.startswith("R") and method in ("RKS", "RHF"):
        issues.append(Issue(
            "warn",
            f"spin = {cfg.spin} is set but method = {method} is closed-shell; "
            f"use UKS / UHF for open-shell systems (otherwise SCF "
            f"either fails to converge or quietly returns wrong "
            f"electronic structure)",
            "config.method",
        ))

    # R3: silent open-shell miss when method=RKS/RHF (closed-shell) AND
    # spin=0 AND the system has an odd electron count -- a radical the
    # user didn't recognise (CH3., NO, charged-organic with odd e-).
    # ``spin=0`` on an odd-electron system is a contradiction: an
    # unpaired electron MUST exist somewhere, so RKS would either fail
    # SCF or quietly return the wrong electronic state.  Compute
    # parity from charge + atomic numbers.
    if method in ("RKS", "RHF") and cfg.spin == 0:
        # Lazy import: pyscf/input.py owns _ATOMIC_NUMBER (an L2 sibling
        # module).  Importing here, not at module top, to avoid a hard
        # cycle should pyscf/input.py ever start to import this file at
        # module load.
        from ..pyscf.input import _ATOMIC_NUMBER, _resolve_charge
        # ``.capitalize()`` on each element: defense in depth.  Structure.
        # from_pdb / from_xyz canonicalise to ``Fe`` / ``Cl`` / ``Na`` at
        # the parser boundary (108c7ff, 2026-05-26), so this rule should
        # only ever see canonical case in practice -- but a caller
        # constructing a Structure directly via ``Structure(elements=
        # ['FE', ...])`` would silently bypass the parser fix and
        # ``_ATOMIC_NUMBER.get('FE', 0)`` returns 0, mis-counting
        # electrons and producing wrong parity verdicts.
        n_e = (sum(_ATOMIC_NUMBER.get(el.capitalize(), 0)
                   for el in struct.elements)
               - _resolve_charge(struct, cfg))
        if n_e % 2 == 1:
            issues.append(Issue(
                "warn",
                f"odd electron count (n_e = {n_e}) with method = {method} "
                f"and spin = 0 -- the system is a radical and a closed-"
                f"shell SCF will either fail to converge or quietly "
                f"return the wrong electronic state.  Switch to UKS / "
                f"UHF and set spin = 1 (or higher for multi-radical "
                f"systems).  If the count is wrong because charge auto-"
                f"detection missed something, set cfg.charge explicitly",
                "config.method",
            ))

    # Stages ladder (task #534).  When cfg.optimize is True the
    # generator's _emit_stages_loop walks cfg.stages and emits one
    # optimize() call per enabled row.  ``validate_stages`` enforces
    # the structural invariants the generator can't recover from --
    # empty list, all-disabled, duplicate names, bogus name chars,
    # non-positive numeric knobs, non-int / non-positive max_steps.
    # WITHOUT this check, an empty-or-all-disabled stages list
    # silently emits ``STAGES = []`` + an empty for-loop, which
    # leaves ``mol_eq`` unbound and the downstream
    # ``_save_xyz(mol_eq, ...)`` / frequencies block raises
    # NameError at runtime.  Surface as ``"error"`` so the Build
    # tab blocks the Generate POST instead of shipping a broken
    # script.
    if bool(getattr(cfg, "optimize", False)):
        from ..config.pyscf import validate_stages
        for _msg in validate_stages(getattr(cfg, "stages", []) or []):
            issues.append(Issue("error", _msg, "config.stages"))

    # Peptide protonation: PeptideBuilder + AddHs builds the gas-phase
    # NEUTRAL form (Asp / Glu protonated, Lys / Arg neutral, etc.).
    # For sequences containing charged side chains, the physiological
    # charge differs.  Surface the gap so the user knows the script
    # is using neutral defaults; they can override with cfg.charge.
    issues += _check_peptide_protonation(struct, getattr(cfg, "charge", None))

    # Inverse case: UKS / UHF with spin = 0 is almost always a mistake.
    # The unrestricted formalism on a closed-shell system runs at ~2x
    # the SCF cost (separate alpha / beta blocks), is more numerically
    # fragile (broken-symmetry saddle points are reachable), and gives
    # the same answer as RKS / RHF unless the user specifically wanted
    # broken-symmetry (e.g. anti-ferromagnetic singlet).  Warn so the
    # default-of-RKS user who flipped to UKS to "be safe" is told it's
    # the wrong default-of-safe.
    if cfg.spin == 0 and method in ("UKS", "UHF"):
        issues.append(Issue(
            "warn",
            f"method = {method} (unrestricted) with spin = 0 (closed shell) "
            f"runs the unrestricted formalism at ~2x the SCF cost of the "
            f"corresponding R{method[1:]}; switch to R{method[1:]} unless you "
            f"specifically want a broken-symmetry singlet "
            f"(e.g. anti-ferromagnetic system)",
            "config.method",
        ))

    # Meta-GGA / hybrid with grid_level < 4: the τ-dependent semi-local XC is
    # grid-sensitive, so forces become noisy at the ~1e-4 Ha/Bohr scale the
    # optimizer cares about.  Warn but allow -- the user may be screening at
    # level 3 deliberately.  ONE shared gate/body (validation.pyscf).
    issues += check_dft_grid_level(cfg, context="optimisation")

    return issues
