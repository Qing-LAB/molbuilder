"""PySCF-specific validators + the PySCFConfig aggregator.

The aggregator ``_validate_pyscf`` is what gets registered against
``PySCFConfig`` in the engine-validator registry.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..issues import Issue
from ..structure import Structure
from .chemistry import (check_species_labels,
                        _check_ecp_declared_for_the_atoms_that_usually_want_one,
                        _check_metal_basis_adequacy)
from .sidecar import _check_frozen_atoms_consumed


# --------------------------------------------------------------------- #
#  Shared PySCF numerical-grid rule (the ONE body).                     #
#                                                                       #
#  The grid-sensitive XC class is META-GGA (τ-dependent: SCAN/TPSS/     #
#  M06-L/…), NOT "hybrids" -- a hybrid's HF exchange is analytic (off   #
#  the DFT grid), so pure hybrid-GGAs are grid-robust.  Below grid      #
#  level 4 a meta-GGA's oscillatory integrand picks up grid noise that  #
#  dominates forces / frequencies.  The SAME gate serves two call       #
#  sites: geometry-opt FORCES (_validate_pyscf) and the vibration       #
#  kind's HESSIAN frequencies (validation/spectra.py).  One             #
#  detector-pair + one gate; message context-selected.                  #
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
      * ``"optimisation"`` — geometry-opt forces.
      * ``"spectra"``       — Hessian / harmonic frequencies (the vibration
        kind).
    """
    # Hartree-Fock integrates no exchange-correlation on a grid: the item
    # enters nothing there, whatever functional the template still names.
    if not cfg.is_dft:
        return []
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
                    cell: Optional[np.ndarray] = None, *,
                    calculation: str = "optimization", **_) -> List[Issue]:
    """PySCF-specific checks.

    ``cell`` is not used to BUILD anything -- this emitter is gas-phase -- but
    the structure's own periodicity is checked, because a periodic structure
    reaching a gas-phase emitter is a silent change of calculation
    (``_check_periodic_structure_in_a_gas_phase_script``).  The argument is
    accepted for signature uniformity with the engine-validator registry.

    ``calculation`` is the described kind (the double-fire dedup, ruled
    2026-08-21): on a VIBRATION deck the kind's science
    (`validation/spectra.py`, over the deck's own view) owns the grid and
    frozen-atoms verdicts, and the copies here DEFER -- each fired twice
    otherwise, once reasoned from optimization fields the vibration deck
    ignores (`cfg.optimize`, grid context="optimisation").  What stays
    unconditional is what the kind never checks: periodicity, basis
    adequacy and the ECP hint.  The charge and the spin are neither's: the
    electronic state's one family runs from `validate` for every kind.
    """
    vibration = calculation == "vibration"
    issues: List[Issue] = []

    # Periodicity vs what this emitter can express.  FIRST, because it changes
    # what every other finding is about: the rest describe a cluster
    # calculation, and this says whether you asked for one.  EVERY KIND, a
    # vibration among them (user: "just note that periodicity will not be
    # respected in pySCF").
    issues += _check_periodic_structure_in_a_gas_phase_script(struct)

    # The charge and the spin are NOT judged here: they are the electronic
    # state's, and `validate` asks its one family once for every engine and
    # kind (`validation.chemistry.check_electronic_state`,
    # `science/chemistry-correctness.md` § 2a).

    # Frozen-atom carrier (three-stage contract).  PySCF emits the
    # geomeTRIC constraints file only when ``cfg.optimize`` is True:
    # single-point runs have nothing to constrain.
    #
    # DEFERRED on the vibration kind: the vibration deck ignores both
    # fields this rationale is keyed on (it always relaxes, always via
    # geomeTRIC, and constrains the frozen set through its own $freeze
    # emission -- ruled 2026-08-21); the kind's science states the
    # frozen regime explicitly instead.
    if not vibration:
        _relaxes = bool(getattr(cfg, "optimize", False))
        _drop_reason = ("" if _relaxes else
                        "cfg.optimize = False (single-point energy; "
                        "no relaxation)")
        issues += _check_frozen_atoms_consumed(
            struct,
            engine="PySCF",
            honored=_relaxes,
            reason_when_dropped=_drop_reason,
        )
    # Pattern B: region labels this optimization run does not consume
    # are named.  The vibration kind runs its own copy over the deck's
    # view (with the same frozen-label exclusion), so this defers there.
    if not vibration:
        from .sidecar import check_unconsumed_region_labels
        issues += check_unconsumed_region_labels(
            struct, engine="PySCF", calculation=calculation)
        # ...and the FIRST validation of a junction, beside it because it is
        # the same question one step further: the labels say which atoms are
        # leads, and a lead must come through the relaxation unmoved.  Asked
        # HERE, where it is cheap -- the compose-time gate that refuses a
        # MOVED lead is correct and runs after the relaxation is paid for.
        from .sidecar import check_electrode_labels_are_frozen
        issues += check_electrode_labels_are_frozen(struct)

    # Solvation, engine-side: these are facts about the BUILD and the
    # vocabulary, not about the vibration -- both decks emit the same
    # solvent lines (scf_setup.emit_solvent_lines).
    _solv = str(getattr(cfg, "solvent", "") or "").strip().lower()
    if _solv:
        from ..pyscf.scf_setup import SOLVENTS
        if _solv not in SOLVENTS:
            issues.append(Issue(
                "error",
                f"unknown solvent {_solv!r}; this deck's PCM dielectric "
                f"table knows: {', '.join(sorted(SOLVENTS))}.  (The "
                f"emitter would refuse the same name with a stack trace "
                f"at prep -- this is that refusal, at preflight.)",
                "config.solvent",
            ))
        if "SMD" in str(getattr(cfg, "solvent_method", "") or "").upper():
            issues.append(Issue(
                "error",
                "solvent_method = SMD: this PySCF build was compiled "
                "without the SMD module (probe: RuntimeError 'compile "
                "with -DENABLE_SMD=ON', 2026-08-21).  Use a PCM variant "
                "(IEF-PCM / C-PCM / COSMO), or rebuild PySCF with SMD "
                "enabled.  SMD reference: Marenich, Cramer & Truhlar, "
                "J. Phys. Chem. B 113, 6378 (2009) [Marenich2009].",
                "config.solvent_method",
            ))

    # Basis adequacy for transition metals.
    issues += _check_metal_basis_adequacy(
        struct, basis=getattr(cfg, "basis", ""),
        engine_label=f"PySCF method={cfg.method}",
    )

    # The ECP hint -- directly after basis adequacy, because the two are the
    # same conversation: what the basis covers, and what the core potential
    # covers.  It ASKS.  Nothing picks an ECP for the user, so this is the
    # only place a bare all-electron Pt gets mentioned.
    issues += _check_ecp_declared_for_the_atoms_that_usually_want_one(
        struct,
        ecp=getattr(cfg, "ecp", "") or "",
        ecp_atoms=getattr(cfg, "ecp_atoms", ()) or (),
        basis=getattr(cfg, "basis", ""),
        engine_label=f"PySCF method={cfg.method}",
    )

    # NO LADDER CHECK HERE, and that is not a gap.  A ladder is declared in
    # task.json for both engines (`stages.md` § 1.1a), so its structural
    # invariants are the DESCRIPTION's and are checked where descriptions are:
    # ``task.Task.validate`` refuses an empty stage list, duplicate names and
    # an all-disabled ladder, and ``validation/task.py`` checks every
    # override against the ``range`` / ``choices`` the schema declares.  This
    # validator sees ONE
    # rung's resolved config and cannot see the ladder at all.

    # The species labels -- asked here because the gate reports a bad label
    # once; the peptide's charge advisory is the electronic state's.
    issues += check_species_labels(struct, engine_label="PySCF")

    # A SETTING THAT ENTERS NOTHING IS NOT LEFT SILENT (engines/pyscf.md
    # § 7a, engines/vibration.md § 3.1, § 4.10): Hartree-Fock has no
    # functional (`PySCFConfig.is_dft`), so one changed from its default
    # under HF is said to do nothing -- on both kinds, since the deck
    # sets no `mf.xc` for either.  The value is still RECORDED (the deck's
    # parameter record, the result's `config`), as every value is.  The
    # dispersion correction is NOT among them: HF takes it like any method.
    # Placed after every finding a DFT run can raise and before the grid
    # check, which says nothing under Hartree-Fock (it has no grid either)
    # -- so no count-by-position caller sees a shift.
    if not cfg.is_dft:
        from ..config.pyscf import PySCFConfig
        _default = PySCFConfig.__dataclass_fields__["functional"].default
        _v = getattr(cfg, "functional", None)
        if _v not in (None, _default):
            issues.append(Issue(
                "warn",
                f"functional = {_v!r} has no effect here: the method is "
                f"Hartree-Fock, which has no functional -- the deck sets "
                f"no `mf.xc`",
                "config.functional",
            ))

    # Meta-GGA / hybrid with grid_level < 4: the τ-dependent semi-local XC is
    # grid-sensitive, so forces become noisy at the ~1e-4 Ha/Bohr scale the
    # optimizer cares about.  Warn but allow -- the user may be screening at
    # level 3 deliberately.  ONE shared gate/body (validation.pyscf).
    if not vibration:      # the kind runs the same gate, context="spectra"
        issues += check_dft_grid_level(cfg, context="optimisation")

    return issues
