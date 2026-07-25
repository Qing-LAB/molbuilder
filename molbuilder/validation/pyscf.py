"""PySCF-specific validators + the PySCFConfig aggregator.

The aggregator ``_validate_pyscf`` is what gets registered against
``PySCFConfig`` in the engine-validator registry; its CALL ORDER is
the public contract (every test that counts issues by position
depends on it).  This module preserves that order verbatim from the
pre-2026-06-13 flat ``molbuilder/validation.py``.

Split per docs/protocols/scientific-validation.md § 10.  No logic
changes; relocation only.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..issues import Issue
from ..structure import Structure
from .chemistry import (_check_metal_basis_adequacy,
                        check_open_shell_metal,
                        _check_peptide_protonation)
from .sidecar import _check_frozen_atoms_consumed


# --------------------------------------------------------------------- #
#  Shared PySCF numerical-grid rule (the ONE body — was duplicated).    #
#                                                                       #
#  Hybrid functionals have a sharper HF-exchange contribution that      #
#  needs a denser XC integration grid; below level 4 the grid noise     #
#  dominates.  The SAME gate (hybrid AND grid_level < 4) matters at     #
#  two call sites with different consequences: geometry-opt FORCES      #
#  (this module, _validate_pyscf) and spectra HESSIAN frequencies       #
#  (spectra/pyscf_engine.preflight).  Both used to carry their own      #
#  copy with a slightly different hybrid detector — the drift the       #
#  backend-architecture review (V4) flagged.  One detector + one gate   #
#  now; the message is context-selected so each site keeps its own      #
#  physically-correct rationale.                                        #
# --------------------------------------------------------------------- #

HYBRID_GRID_FLOOR = 4

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


def check_hybrid_grid_level(cfg, *, context: str) -> List[Issue]:
    """Emit the hybrid-functional grid-density advisory (or nothing).

    ``context`` selects the physically-correct rationale:
      * ``"optimisation"`` — geometry-opt forces (Build tab).
      * ``"spectra"``       — Hessian / harmonic frequencies (Spectra tab).
    """
    grid = getattr(cfg, "grid_level", None)
    functional = getattr(cfg, "functional", "") or ""
    if grid is None or grid >= HYBRID_GRID_FLOOR \
            or not is_hybrid_functional(functional):
        return []
    if context == "spectra":
        msg = (f"Grid level {grid} is below the recommended minimum of "
               f"{HYBRID_GRID_FLOOR} for a hybrid functional ({functional}).  "
               f"Hybrid functionals have a sharper exchange contribution that "
               f"needs a denser numerical grid; below level "
               f"{HYBRID_GRID_FLOOR} the grid-integration noise typically "
               f"dominates the frequency error.  Raise the grid level for "
               f"publication-quality results.")
    else:
        msg = (f"grid_level = {grid} with a hybrid functional "
               f"({functional.upper()}): hybrid-DFT forces are noisy at this "
               f"grid density (forces look ~1e-4 Ha/Bohr noise floor).  Bump "
               f"to grid_level = {HYBRID_GRID_FLOOR} for production geometry "
               f"optimisation; level 3 is fine for energies / screening only")
    return [Issue("warn", msg, "config.grid_level")]


# --------------------------------------------------------------------- #
#  PySCF aggregator                                                     #
#                                                                       #
#  CALL ORDER IS LOAD-BEARING.  Tests that count issues by position    #
#  depend on this exact sequence.  Do not reorder.                     #
# --------------------------------------------------------------------- #


def _validate_pyscf(struct: Structure, cfg,
                    cell: Optional[np.ndarray] = None, **_) -> List[Issue]:
    """PySCF-specific checks.  ``cell`` is unused (PySCF jobs are gas-
    phase or PCM-solvent here); accepted for signature uniformity
    with the engine-validator registry."""
    issues: List[Issue] = []

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

    # Hybrid functional with grid_level < 4: forces become noisy at the
    # ~1e-4 Ha/Bohr scale the optimizer cares about.  Warn but allow --
    # the user may be doing screening at level 3 deliberately.  ONE shared
    # gate/body (was duplicated in spectra/pyscf_engine; V4).
    issues += check_hybrid_grid_level(cfg, context="optimisation")

    return issues
