"""PySCF's own settings checks.

:func:`validate_pyscf` is the part PySCF's record in the engine table names
(``validator``), asked by the settings gate (`validation.validate`) with what
the gate works out once: the engine, the box, the electronic state.  A
Gaussian basis's coverage of a metal and the core potentials PySCF applies
are PySCF's chemistry, so they live here.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..issues import Issue
from ..structure import Structure
from .sidecar import _check_frozen_atoms_consumed


#: The row-5 boundary (Kr, Z=36).  It is NOT a definition of "heavy" and
#: decides nothing on its own -- it only bounds WHOM THE HINT BELOW NAMES,
#: and the hint prints the criterion so a reader can disagree with it.
#:
#: The user's ruling (2026-08-13): *"there is no point to limit matching
#: to heavy -- who defines heavy? there is no clear reasoning or standard
#: ... explicit is better than implicit."*  So it is the threshold of a
#: question, never of an action.
_ECP_HINT_Z = 36


def _check_ecp_declared_for_the_atoms_that_usually_want_one(
        struct: Structure, *, ecp: str, ecp_atoms, basis: str,
        engine_label: str) -> List[Issue]:
    """WARN when a structure carries post-Kr elements and no ECP covers them.

    **This hint asks; it never chooses.**  It speaks because of the other
    half of the user's ruling: *"you can still have the validation
    function to give hints -- that should be confirmed."*

    An element is covered by an ``ecp`` name whose ``ecp_atoms`` patterns
    select it -- and by nothing else.  **A def2 basis does NOT cover it by
    itself**: PySCF's def2 files
    carry the core potential (``def2-svp.dat``: *Au nelec 60*), but
    ``gto.M`` applies one only when ``ecp`` is given (PySCF 2.14
    ``gto/mole.py``, ``build``; ``check_sanity`` merely warns), and the deck
    gives one only when the person declares it.  So on a def2 basis the hint
    names that basis's own core potential -- the one it was built for.
    """
    from ..chemistry import atomic_number
    from ..pyscf.ecp import resolve_ecp

    covered = resolve_ecp(struct, ecp, ecp_atoms) or {}
    uncovered: List[str] = []
    for el in struct.elements:
        sym = str(el).strip()
        try:
            z = atomic_number(sym)
        except KeyError:
            # A label naming no element -- `check_species_labels` already
            # reports it as an error.  Skipping keeps THIS check about ECPs.
            continue
        if z > _ECP_HINT_Z and sym not in covered and sym not in uncovered:
            uncovered.append(sym)
    if not uncovered:
        return []

    named = ", ".join(uncovered)
    them = "them" if len(uncovered) > 1 else "it"
    if (basis or "").lower().replace("_", "").replace("-", "").startswith("def2"):
        # THE BASIS'S OWN CORE POTENTIAL, named: a def2 basis for these
        # elements is a valence basis built for it, so all-electron on it is
        # not a cheaper answer but a broken one.
        message = (
            f"No effective core potential covers {named}, so "
            f"{engine_label} will put every electron of {them} into basis "
            f"'{basis}' -- a valence basis, built for its own core potential "
            f"({'these elements' if len(uncovered) > 1 else 'this element'} "
            f"past Kr).  {engine_label} applies that core potential only when "
            f"the script names it, and nothing names it for you.  To use it: "
            f"ecp = '{basis}' with ecp_atoms = {uncovered!r}.")
    else:
        message = (
            f"No effective core potential covers {named}, so {engine_label} "
            f"will treat {them} ALL-ELECTRON on basis '{basis}'.  For "
            f"elements past Kr that is usually wrong twice over: the cost is "
            f"large (Pt alone carries 78 electrons), and without a "
            f"scalar-relativistic ECP the bond lengths and orbital energies "
            f"are off -- Pt-Pt by ~0.1 A, Au gaps by ~1 eV.  To use one, name "
            f"it and say which atoms get it: ecp = 'lanl2dz' with ecp_atoms = "
            f"{uncovered!r} (or ['*'] for every element present); a def2 "
            f"basis's own is named the same way, ecp = the basis.  If you "
            f"meant all-electron, this is the confirmation: nothing is added "
            f"for you.  (Named here because Z > {_ECP_HINT_Z}; that bound "
            f"decides what gets mentioned, nothing else.)")
    return [Issue("warn", message, "config.ecp")]


def _check_metal_basis_adequacy(struct: Structure, *,
                                  basis: str, engine_label: str
                                  ) -> List[Issue]:
    """PySCF's Gaussian basis sets: ones like STO-3G / 6-31G / 6-31G(d)
    have poor or no coverage of transition-metal d-orbitals.  Pair with
    Fe / Mn / Co / Ni / Cu / Mo etc. and the SCF converges to a
    distorted electronic structure with the wrong d-orbital ordering.

    Recommendations encoded here: def2-SVP is the production minimum;
    def2-TZVP is publication-quality.  Anything smaller for a
    transition-metal-containing structure -> WARN.
    """
    # ALL transition metals, not just open-shell: d-orbital basis coverage is
    # equally needed for closed-shell d10 metals (Zn/Cd/Hg/Pd/Pt) -- the
    # concern is orbital coverage, orthogonal to spin state.
    from ..chemistry import detect_transition_metals
    metals = detect_transition_metals(struct)
    if not metals:
        return []
    b = (basis or "").lower().strip()
    # Bases known to be inadequate for transition metals (no d set
    # for first-row TMs, or no functions at all for second/third row).
    INADEQUATE = ("sto-3g", "sto-6g", "3-21g", "6-31g",
                  "6-31g(d)", "6-31g*", "6-31g**", "6-31gd", "6-311g")
    if any(b == bad or b.startswith(bad + "/") for bad in INADEQUATE):
        return [Issue(
            "warn",
            (f"Basis '{basis}' has inadequate coverage of transition-"
             f"metal d-orbitals.  {engine_label} requested for "
             f"structure containing {', '.join(metals)}.  Bases like "
             f"STO-3G / 6-31G(d) lack the polarisation/diffuse "
             f"functions needed to describe metal-ligand bonds + spin "
             f"states; the SCF often converges to a distorted "
             f"electronic structure (wrong d-orbital occupations, "
             f"wrong spin-gap energies).  Recommended minimum for "
             f"transition metals: def2-SVP.  Publication quality: "
             f"def2-TZVP or cc-pVTZ-DK."),
            "config.basis",
        )]
    return []


# --------------------------------------------------------------------- #
#  Shared PySCF numerical-grid rule (the ONE body).                     #
#                                                                       #
#  The grid-sensitive XC class is META-GGA (τ-dependent: SCAN/TPSS/     #
#  M06-L/…), NOT "hybrids" -- a hybrid's HF exchange is analytic (off   #
#  the DFT grid), so pure hybrid-GGAs are grid-robust.  Below grid      #
#  level 4 a meta-GGA's oscillatory integrand picks up grid noise that  #
#  dominates forces / frequencies.  The SAME gate serves two call       #
#  sites: geometry-opt FORCES (validate_pyscf) and the vibration        #
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
        f"(the Molbuilder tab's Cell page) if an isolated cluster is what you want.",
        "cell.periodic_in_gas_phase",
    )]


def _gpu_library_limits(struct: Structure, cfg, state) -> List[Issue]:
    """What gpu4pyscf does not do that PySCF does, where this run asks for
    the GPU -- each a ``warn``, never an ``error``: choosing the GPU
    knowing it is the person's (`execution/gpu.md` G11,
    `engines/pyscf.md` § 7.3).  An unrestricted reference: gpu4pyscf has
    no stability analysis, so on the GPU the SCF keeps its first solution
    unchecked.  A solvent model: gpu4pyscf's has not been compared with
    PySCF's here."""
    out: List[Issue] = []
    if state is not None:
        if state.spin_treatment.value == "unrestricted":
            out.append(Issue(
                "warn",
                "use_gpu with an unrestricted reference: gpu4pyscf has no "
                "stability analysis, so on the GPU the SCF keeps its first "
                "solution unchecked -- the CPU path checks it, and may find "
                "a lower state (engines/pyscf.md § 7.3).  The run's log says "
                "`stability: NOT CHECKED`.",
                "config.use_gpu"))
    if str(getattr(cfg, "solvent", "") or "").strip():
        out.append(Issue(
            "warn",
            "use_gpu with a solvent model: gpu4pyscf's solvent has not been "
            "compared with PySCF's here -- the numbers are its own "
            "(engines/pyscf.md § 7.3).",
            "config.use_gpu"))
    return out


def validate_pyscf(struct: Structure, cfg,
                   cell: Optional[np.ndarray] = None, *,
                   calculation: str, engine: str, state, **_) -> List[Issue]:
    """PySCF's own checks -- the part its engine record names
    (``validator``).  ``state`` is the electronic state the gate worked out
    (``None`` where a species label names no element).

    ``cell`` is not used to BUILD anything -- this emitter is gas-phase -- but
    the structure's own periodicity is checked, because a periodic structure
    reaching a gas-phase emitter is a silent change of calculation
    (``_check_periodic_structure_in_a_gas_phase_script``).  The argument is
    taken as every engine's validator takes it.

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
    from ..engine_table import engine as _engine
    words = _engine(engine).words
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
            engine=words,
            honored=_relaxes,
            reason_when_dropped=_drop_reason,
        )
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

    # gpu4pyscf'S OWN LIMITS, SAID -- every kind (`execution/gpu.md` G11;
    # the user, 2026-10-21: "a hint and information is enough").
    if bool(getattr(cfg, "use_gpu", False)):
        issues += _gpu_library_limits(struct, cfg, state)

    # Basis adequacy for transition metals.
    issues += _check_metal_basis_adequacy(
        struct, basis=getattr(cfg, "basis", ""),
        engine_label=f"{words} method={cfg.method}",
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
        engine_label=f"{words} method={cfg.method}",
    )

    # NO LADDER CHECK HERE, and that is not a gap.  A ladder is declared in
    # task.json for both engines (`stages.md` § 1.1a), so its structural
    # invariants are the DESCRIPTION's and are checked where descriptions are:
    # ``task.Task.validate`` refuses an empty stage list and duplicate names,
    # and ``validation/task.py`` checks every
    # override against the ``range`` / ``choices`` the schema declares.  This
    # validator sees ONE
    # rung's resolved config and cannot see the ladder at all.

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
