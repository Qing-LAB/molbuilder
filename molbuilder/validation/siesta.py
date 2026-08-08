"""SIESTA-specific validators + the SiestaConfig aggregator.

The aggregator ``_validate_siesta`` is what gets registered against
``SiestaConfig`` in the engine-validator registry; its CALL ORDER is
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
from .chemistry import check_open_shell_metal, _check_peptide_protonation
from .sidecar import _check_frozen_atoms_consumed


def _check_siesta_pseudo_coverage(struct: Structure, cfg,
                                    *, dest_dir=None) -> List[Issue]:
    """Run molbuilder.pseudos.check_coverage on cfg.psml_lib so the
    SIESTA Build->Generate preflight catches:
      * missing .psml files (SIESTA's ``pseudo_read: ERROR: Pseudopotential
        file not found`` after 5 minutes of MPI init -- we surface
        at click-time instead);
      * XC family / authors mismatch (SIESTA SILENTLY uses the
        pseudo's XC even when XC.authors in the .fdf disagrees;
        bond lengths come out wrong with no error -- only molbuilder
        catches this).

    Without cfg.psml_lib set we emit a single WARN telling the
    user SIESTA will hard-fail at run time looking for them.
    Suggests projects/pseudopotential/ as the convention since
    that's where the new-project skeleton creates one.
    """
    psml_lib = getattr(cfg, "psml_lib", None)
    if not psml_lib:
        # No path configured -- SIESTA will refuse to start.  Don't
        # ERROR (user might know what they're doing + intend to fill
        # it in by hand); WARN with the actionable hint.
        return [Issue(
            "warn",
            ("cfg.psml_lib is not set -- SIESTA needs .psml files for "
             "every element (H, C, N, O, S, Fe, ...) and will refuse "
             "to start without them.  Download from "
             "http://www.pseudo-dojo.org (PBE-SR, standard, PSML "
             "format) and set cfg.psml_lib to that directory.  "
             "Convention: projects/pseudopotential/ next to your "
             "structure files.  Once set, this preflight will check "
             "coverage + XC-family match against your structure's "
             "elements automatically."),
            "config.psml_lib",
        )]
    from ..pseudos import resolve_psml_lib
    psml_dir = resolve_psml_lib(psml_lib, dest_dir=dest_dir)
    if not psml_dir.is_dir():
        # Relative paths try dest-relative first (if we have dest_dir),
        # then fall back to projects/-relative (see
        # pseudos.resolve_psml_lib).  When the user gives a relative
        # path that misses both anchors, tell them what we tried.
        from pathlib import Path as _P
        is_relative = not _P(psml_lib).expanduser().is_absolute()
        if is_relative:
            hint = (
                f"  Note: relative paths are resolved against the "
                f".fdf destination dir first (the portable form the "
                f"Save handler persists), then against ``projects/`` "
                f"(the documented convention).  Tried: {psml_dir}.  "
                f"Either create that directory, use an absolute path, "
                f"or pick the directory via the file-picker."
            )
        else:
            hint = ""
        # Severity rule:
        #   * ABSOLUTE path that doesn't exist -> always ERROR.  No
        #     amount of context can rescue an absolute path.
        #   * RELATIVE path + NO dest_dir context -> WARN.  The form
        #     might hold a dest-relative path (post-Save form rewrite);
        #     Save-time install-pseudos re-checks with dest_dir.
        #   * RELATIVE path + dest_dir given (Save preflight) -> ERROR.
        #     We tried both anchors; if neither hits the file is really
        #     missing and Save will fail downstream.
        if not is_relative:
            severity = "error"
        else:
            severity = "error" if dest_dir is not None else "warn"
        return [Issue(
            severity,
            f"cfg.psml_lib path does not exist or is not a directory: "
            f"{psml_lib}.  SIESTA will not find any pseudopotentials."
            + hint,
            "config.psml_lib",
        )]
    # Derive expected XC family from cfg.xc_authors (PBE/PBEsol/...
    # -> GGA; CA/PZ/PW -> LDA; DRSLL/LMKLL -> VDW).
    GGA = {"pbe", "pbesol", "blyp", "revpbe", "rpbe"}
    LDA = {"ca", "pz", "pw"}
    VDW = {"drsll", "lmkll"}
    xc_authors = (getattr(cfg, "xc_authors", "") or "").strip()
    a = xc_authors.lower()
    expected_family = ("GGA" if a in GGA
                       else "LDA" if a in LDA
                       else "VDW" if a in VDW
                       else None)
    from ..pseudos import check_coverage, ERROR_STATUSES
    out: List[Issue] = []
    for entry in check_coverage(
        struct.elements, psml_dir,
        expected_xc_family=expected_family,
        expected_xc_authors=xc_authors or None,
    ):
        if entry.status == "ok":
            continue
        # ERROR_STATUSES (missing / dead_projector / xc_family_mismatch) BLOCK:
        # the run cannot be correct.  The rest -- xc_mismatch (same-family author
        # diff) / relativistic_mismatch / generator_mismatch / parse_warning --
        # are advisory (warn).  The set is shared with the CLI (pseudos.py) so
        # the two surfaces can't drift.
        severity = "error" if entry.status in ERROR_STATUSES else "warn"
        out.append(Issue(severity, entry.message,
                          f"config.psml_lib.{entry.element}"))
    return out


def _check_siesta_mesh_cutoff(cfg) -> List[Issue]:
    """SIESTA-specific: mesh_cutoff below the production-defensible
    floor (150 Ry).

    Why 150 Ry as the warn threshold (vs. the slider's hard floor of
    100 Ry):

    SIESTA's real-space mesh cutoff controls the integration grid
    fineness.  Below ~150 Ry, organic / biomolecule systems show
    energy errors of tens of meV and force errors that visibly
    affect a relaxation -- the geometry converges to a slightly
    wrong minimum.  Production literature numbers cluster around
    200-300 Ry; tight basis (TZP) and vibrational work want 400+.

    100 Ry is allowed by the slider as a "I'm doing a 5-minute
    sanity check" floor; below 150 we add a soft nudge so the user
    sees the trade-off before they hit Save.  WARN severity (not
    ERROR) -- the user may genuinely want a screening calc.

    This rule was deferred from the 2026-05-27 holistic-math audit
    and landed 2026-05-28 alongside the cell-volume tightening.
    """
    mc = getattr(cfg, "mesh_cutoff", None)
    if mc is None:
        return []
    try:
        mc_val = float(mc)
    except (TypeError, ValueError):
        return []
    # Gated on >= 100 Ry: the dataclass metadata-range check (lower
    # bound 100) already warns for values below the slider floor with
    # the SAME ``where`` field (``config.mesh_cutoff``).  Without the
    # 100 floor here, a value of 5 Ry would produce TWO warnings on
    # the same field, and the existing tests counting issues by
    # ``where`` would over-count.  Honest semantics: metadata-range
    # owns the "below the slider floor" case; this rule owns the
    # "above the floor but below production-defensible" case.
    if 100.0 <= mc_val < 150.0:
        return [Issue(
            "warn",
            (f"mesh_cutoff = {mc_val:g} Ry is below the production "
             f"floor of ~150 Ry.  Forces / energies on organic and "
             f"biomolecule systems are noticeably wrong at this "
             f"cutoff (tens of meV; a relaxation converges to a "
             f"slightly different minimum).  Production-typical: "
             f"200-300 Ry; tight basis (TZP) or vibrational work: "
             f"400+ Ry.  Keep this value only for a quick screening "
             f"calc."),
            "config.mesh_cutoff",
        )]
    return []


def _check_siesta_charged_makov_payne_notice(struct: Structure,
                                              cfg) -> List[Issue]:
    """SIESTA-specific: charged system in a periodic supercell carries
    an image-charge artefact that padding alone does NOT remove.

    See Makov & Payne, Phys. Rev. B 51, 4014 (1995).  Leading term:

        E_bias ~ q^2 * alpha / (2 * L * eps_r)

    For q = +/-1 at typical molbuilder vacuum-cell sizes (15-25 A) the
    bias is 0.5-1.5 eV -- well above chemical accuracy (~0.04 eV).

    molbuilder does NOT auto-apply the Makov-Payne correction; this
    warn surfaces the issue so users computing redox / pKa /
    deprotonation energies know to apply it post-hoc.  Deferred for a
    future "Makov-Payne emission" capability (see design.md decisions
    log; task #165 retains the open item).

    Severity: WARN (not ERROR) -- the calculation still runs and a
    user doing a single-point screening calc may not care.  We're
    nudging, not blocking.

    Skip conditions:
      * net_charge unset or zero
      * The caller passed a non-vacuum cell explicitly AND that cell
        looks like a real crystal (no auto-padding bump signature) --
        we don't have enough info to know if the user wants to model
        a periodic crystal (in which case the artefact IS the
        physics) or a vacuum supercell.  Conservative: still warn,
        the user can dismiss.
    """
    # Resolve charge: explicit user override or auto-detected.
    from ..chemistry import resolve_net_charge
    from ..siesta.makov_payne import compute_correction
    try:
        q = resolve_net_charge(struct, getattr(cfg, "net_charge", None))
    except Exception:
        return []
    if q == 0:
        return []
    # Estimate the correction magnitude at a representative vacuum
    # cell size.  Real SIESTA cells vary; the message gives the user
    # the order-of-magnitude before they actually run.  Cell sizes
    # 15 / 20 / 25 Å bracket the typical molbuilder vacuum range.
    eps_ref = 1.0
    dE_15 = compute_correction(q=q, L_angstrom=15.0, epsilon_r=eps_ref)
    dE_20 = compute_correction(q=q, L_angstrom=20.0, epsilon_r=eps_ref)
    dE_25 = compute_correction(q=q, L_angstrom=25.0, epsilon_r=eps_ref)
    return [Issue(
        "warn",
        (f"Charged system (NetCharge = {q:+d}) in a finite supercell.  "
         f"SIESTA's periodic-cell setup adds a uniform compensating "
         f"background charge so the calculation runs, but the total "
         f"energy carries an image-charge bias from the molecule's "
         f"interaction with its periodic replicas.  Estimated "
         f"correction magnitude (vacuum, cubic-Madelung): "
         f"~{dE_15:.2f} eV at L=15 Å, ~{dE_20:.2f} eV at L=20 Å, "
         f"~{dE_25:.2f} eV at L=25 Å — well above chemical accuracy.  "
         f"molbuilder emits a companion ``makov_payne_correction.py`` "
         f"script alongside the FDF; after SIESTA finishes, run it "
         f"to get the corrected total for the cell SIESTA actually "
         f"used.  See Makov & Payne, PRB 51, 4014 (1995)."),
        "config.net_charge.makov_payne",
    )]


# Recommended per-side vacuum (Angstrom) for an isolated molecule so its
# periodic images don't interact: basis orbitals reach 4-7 A per atom with a DZP
# basis and the inter-image gap is 2*vacuum, so the neutral floor is set above
# the largest orbital radius; a charged system needs far more because the
# image-charge Coulomb bias decays only as 1/L (see the Makov-Payne notice).
_VACUUM_MIN_NEUTRAL = 8.0
_VACUUM_MIN_CHARGED = 25.0


def _check_siesta_vacuum_adequacy(struct: Structure, cfg) -> List[Issue]:
    """Too little vacuum on an ISOLATED axis lets the molecule interact with
    its own periodic images.

    Lives HERE, in the validator, rather than in the emitter: it used to be a
    Python ``warnings.warn`` inside ``render_fdf``, which reached the server's
    stderr and therefore never reached a web user at all -- a 2.5 A vacuum box
    went to SIESTA with nothing said, and the user learnt of it only from
    SIESTA's own "multiply-connected orbital pairs" message (2026-07-29).
    Clause R5 of the delivery contract (science/validation.md 4.1): a finding
    never travels as a warning.  As an Issue it reaches BOTH surfaces -- the
    web panel through the endpoint's ``issues[]``, and the CLI through
    ``render_fdf``'s own ``report(validate(...))``.

    Periodic / transport axes are skipped: a crystal or a device sets the box
    there, not the vacuum.  Never mutates -- the structure is the truth."""
    from ..chemistry import resolve_net_charge
    try:
        q = resolve_net_charge(struct, getattr(cfg, "net_charge", None))
    except Exception:                      # noqa: BLE001 -- charge is advisory here
        q = 0
    min_vac = _VACUUM_MIN_CHARGED if q else _VACUUM_MIN_NEUTRAL
    kinds = struct.axis_kind or ("isolated", "isolated", "isolated")

    # MANUAL REGIME: an explicit cell IS the box, and vacuum is reference-only
    # (structure-periodicity.md § 6.2).  Reading a vacuum here would report a
    # number that never reaches the calculation -- a molecule in a hand-typed
    # 30 A box would be told its vacuum is thin.  What matters on a typed box
    # is the gap actually ACHIEVED, and ``cell.image_distance`` measures that
    # directly, from the atoms and the box rather than from a setting.
    if struct.cell is not None:
        return []

    # The RESOLVED per-side gap, not the stored one: unset means "no vacuum
    # chosen", and an unset isolated axis is still given a default gap, which
    # is thin by this check's own standard and worth saying so.
    vac = struct.effective_vacuum()
    thin = [(i, vac[i]) for i, k in enumerate(kinds)
            if k == "isolated" and vac[i] < min_vac]
    if not thin:
        return []
    defaulted = set(struct.defaulted_vacuum_axes())
    where = ", ".join(
        f"axis {i} ({v:g} Å"
        + (" — the default, none set)" if i in defaulted else ")")
        for i, v in thin)
    return [Issue(
        "warn",
        (f"Thin vacuum on an isolated system: {where}. Recommended ≥ "
         f"{min_vac:g} Å per side ({'charged' if q else 'neutral'}) so the "
         f"molecule's periodic images don't interact — the gap between images "
         f"is 2×vacuum, and basis orbitals reach several Å per atom. Set "
         f"'vacuum' on the structure (Modify → Cell tab); the geometry is not "
         f"changed for you."),
        "cell.vacuum_thin",
    )]


def _check_siesta_spin_polarized_needs_spin_total(struct: Structure,
                                                    cfg) -> List[Issue]:
    """SIESTA-specific: spin_polarized=True + spin_total=None + open-
    shell metal -> ERROR.

    The propor: ERROR: IMAX = 0 failure mode (2026-05-24 hemeC-dithiol
    incident): SIESTA's initial-DM constructor tries to find a zero-
    net-spin proportional split for each atom's reference-config
    electrons.  For a closed-shell atom (H/C/N/O/S) this is trivial.
    For a transition metal with a semicore-rich pseudo (e.g. Fe with
    3p⁶3d⁶4s² in the valence) the constraint "exactly zero net spin
    on a d-shell, distributed over integer orbital indices" has no
    valid solution -- propor's loop variable IMAX stays at 0, SIESTA
    aborts before the SCF loop ever runs.

    Fix: force a non-zero spin_total.  The chemistry-aware suggestion
    + alternatives come from chemistry.suggest_spin_total() so the
    user gets actionable numbers instead of having to look up
    ligand-field rules for the metal in question.

    Why ERROR (not WARN): SIESTA WILL refuse to start.  Failing fast
    in molbuilder saves the user a 30-second SIESTA startup just to
    be told ``propor: ERROR: IMAX = 0``.
    """
    if not bool(getattr(cfg, "spin_polarized", False)):
        return []
    # The check ALSO fires when the user explicitly set spin_total=0.0:
    # that's the exact propor IMAX=0 trigger we're trying to catch
    # (zero net spin on a d/f shell has no valid proportional split).
    # Earlier ``is not None`` gate let this silently through (caught in
    # the 2026-05-25 review).  Now: fire when spin_total is None OR
    # numerically zero.
    _spin = getattr(cfg, "spin_total", None)
    if _spin is not None and float(_spin) != 0.0:
        return []
    from ..chemistry import detect_open_shell_metals, suggest_spin_total
    metals = detect_open_shell_metals(struct)
    if not metals:
        return []
    preferred, alternatives = suggest_spin_total(metals)
    lines = [
        f"Spin polarized is enabled but spin_total is not set, AND "
        f"the structure contains open-shell metal(s): "
        f"{', '.join(metals)}.  SIESTA's initial-DM constructor "
        f"(propor) cannot find a zero-net-spin split for these atoms "
        f"with semicore-rich pseudos and will abort with "
        f"``propor: ERROR: IMAX = 0`` before the SCF loop starts.",
        "",
        f"START HERE: set cfg.spin_total = {preferred}  "
        f"(2S, in μB; SIESTA emits this as ``Spin.Total``).  "
        f"This is the most common starting value for the metals "
        f"detected; adjust if SCF doesn't converge to the chemistry "
        f"you expect.",
    ]
    if alternatives:
        lines.append("")
        lines.append("Alternatives to sweep through if the starting "
                      "value doesn't match the chemistry (run with "
                      "each, pick lowest-energy SCF):")
        for value, desc in alternatives:
            lines.append(f"  spin_total = {value:>4g}  -- {desc}")
    return [Issue("error", "\n".join(lines), "config.spin_total")]


# --------------------------------------------------------------------- #
#  SIESTA aggregator                                                    #
#                                                                       #
#  CALL ORDER IS LOAD-BEARING.  Tests that count issues by position    #
#  depend on this exact sequence.  Do not reorder.                     #
# --------------------------------------------------------------------- #


def _validate_siesta(struct: Structure, cfg,
                     cell: Optional[np.ndarray],
                     *, dest_dir=None, **_) -> List[Issue]:
    """SIESTA-specific checks.

    Registered with the engine-validator dispatch at module bottom
    (the decorator is applied after the SiestaConfig type is
    importable -- avoids the import cycle between validation.py and
    siesta/input.py at definition time).

    ``dest_dir`` (keyword-only) is passed through to the pseudo-
    coverage check so dest-relative ``cfg.psml_lib`` paths resolve
    correctly post-Save (see pseudos.resolve_psml_lib).
    """
    issues: List[Issue] = []

    # Peptide protonation hint -- same as PySCF side; see
    # _check_peptide_protonation for the full rationale.
    issues += _check_peptide_protonation(struct, getattr(cfg, "net_charge", None))

    # Pseudopotential coverage (the actionable use of pseudos.py).
    # Wired into preflight + render: missing files become ERROR
    # Issues (SIESTA hard-fails without them); XC mismatches
    # become WARN (silent wrong bond lengths otherwise).
    issues += _check_siesta_pseudo_coverage(struct, cfg, dest_dir=dest_dir)

    # MeshCutoff floor: warn below 150 Ry (production-defensible
    # threshold).  The dataclass slider lower bound is 100 Ry; this
    # rule catches the 100-149 Ry window with a soft nudge.
    issues += _check_siesta_mesh_cutoff(cfg)

    # Makov-Payne notice: charged-supercell image-charge bias.
    # We DON'T auto-apply the correction (see function docstring +
    # design.md decisions log); we surface it so the user knows
    # what's missing.
    issues += _check_siesta_charged_makov_payne_notice(struct, cfg)

    # Vacuum adequacy on isolated axes (R5: was a warnings.warn in the
    # emitter, invisible to the web; now a finding on every surface).
    issues += _check_siesta_vacuum_adequacy(struct, cfg)

    # Open-shell metal + closed-shell SCF: shared rule with PySCF.
    issues += check_open_shell_metal(
        struct,
        is_closed_shell=not bool(getattr(cfg, "spin_polarized", False)),
        engine_label="SIESTA (spin_polarized = False)",
    )

    # Frozen-atom carrier (three-stage contract).  SIESTA honors
    # struct.frozen_atoms via %block Geometry.Constraints which is
    # only meaningful inside an MD/relax block.  When relax_type is
    # "none" the relaxer doesn't run, so the constraint is a no-op.
    relax = (getattr(cfg, "relax_type", "") or "").lower()
    issues += _check_frozen_atoms_consumed(
        struct,
        engine="SIESTA",
        honored=(relax not in ("none", "")),
        reason_when_dropped=(
            f"cfg.relax_type = {cfg.relax_type!r} (no MD/relax block "
            f"is emitted, so Geometry.Constraints would be a no-op)"
        ),
    )

    # NOTE (2026-08-07, P2 unit 2): this validator used to walk ``cfg.stages``
    # here and re-check every stage's relax knobs.  It does not any more, and
    # nothing replaced it in this function -- BY DESIGN, not by omission.
    #
    # A config has no stage list (engines/stages.md § 1.1), and § 4 R2 says a
    # stage is validated as a RESOLVED WHOLE, never as a diff: the caller
    # resolves each stage through ``effective_config`` and calls THIS
    # function on the result, once per stage.  So each stage's relax_type,
    # steps, force tol and displacement cap are checked by the ordinary
    # single-config rules above -- the same rules, not a parallel copy of
    # them, which is what made the old block drift from them.
    #
    # The two checks that were genuinely about the LADDER rather than about
    # any one stage moved to where the ladder is: an empty / all-disabled
    # list and duplicate names are refused by ``siesta/input.py``'s
    # ``_enabled_stages`` (and, for a description read from disk, by
    # ``task.py``).  Cross-stage findings -- a ladder that loosens -- are
    # P2 unit 6 and carry no stage label (§ 4).

    # SIESTA-specific: spin_polarized + no spin_total + open-shell metal
    # -> propor: ERROR: IMAX = 0 (initial-DM constructor abort).  See
    # the 2026-05-24 hemeC-dithiol incident for the failure mode
    # walk-through: SIESTA tries to find a zero-net-spin split for
    # the metal's d/f shell using its semicore-rich pseudo, can't
    # land on a valid IMAX, and exits before SCF starts.  Trigger
    # this proactively at preflight so the user fixes the .fdf in
    # the form (or sees the recipe) instead of paying a 30-second
    # SIESTA startup just to be told "IMAX = 0".
    issues += _check_siesta_spin_polarized_needs_spin_total(struct, cfg)

    # Electron-count parity (cross-engine).  SIESTA's "spin" is
    # expressed as spin_total (μ_B); when spin_polarized=False it's
    # implicitly 0.  We need an integer 2S to call the shared parity
    # helper, so derive: round(spin_total) -> 2S.  Skip when
    # net_charge is unset (auto-detect path handles it inside
    # render_fdf via resolve_net_charge).
    #
    # Severity rule (refined 2026-05-23 from the original always-ERROR):
    #   * ERROR only when the user EXPLICITLY set spin_total -- a real
    #     user-asserted contradiction with the electron count.
    #   * WARN when spin_total is None (dataclass default) -- the user
    #     didn't actually claim spin=0; the default did.  For odd-
    #     electron systems we nudge them toward spin_polarized=True
    #     without blocking the render.  Avoids surprising failures
    #     when callers pass net_charge=0 to a synthetic / fictitious
    #     state (e.g. test fixtures, charge-override sweeps).
    from ..chemistry import check_spin_charge_parity
    if getattr(cfg, "net_charge", None) is not None:
        spin_explicit = getattr(cfg, "spin_total", None) is not None
        spin_total = getattr(cfg, "spin_total", None) or 0.0
        spin_2s = int(round(spin_total))
        if not cfg.spin_polarized and spin_2s != 0:
            # User asked for non-zero spin without spin_polarized;
            # already handled by the warning above + the existing
            # spin_total-without-polarized warning.  Skip parity
            # (SIESTA will accept it but won't use it).
            pass
        else:
            err = check_spin_charge_parity(struct, cfg.net_charge, spin_2s)
            if err:
                severity = "error" if spin_explicit else "warn"
                issues.append(Issue(severity, err, "config.spin_total"))

    # Spin.Total set without spin polarised: SIESTA silently ignores it.
    if cfg.spin_total is not None and not cfg.spin_polarized:
        issues.append(Issue(
            "warn",
            f"spin_total = {cfg.spin_total} is set but spin_polarized "
            f"is False; SIESTA will silently ignore the total-spin pin",
            "config.spin_total",
        ))

    if cell is None:
        return issues

    # k-grid vs cell extent: distinguish three cases per axis:
    #   * vacuum direction (atoms span < 85% of axis) -> k=1 correct,
    #     k>1 wasted
    #   * periodic direction (atoms span > 85% of axis) -> k=1
    #     under-converged when other axes are sampled
    #   * indeterminate (no atoms or single-axis tiny molecule) ->
    #     fall back to the cell-extent heuristic
    #
    # Pre-fix the heuristic used cell-extent alone, which mis-flagged
    # vacuum-padded long axes (e.g. a 12-mer DNA in an 80 Å cell with
    # kgrid (4, 4, 1) along the molecular axis is correct vacuum, not
    # periodic).  Atoms spanning < 85% of an axis means there's
    # vacuum padding at the ends -> the user opted for vacuum on
    # that axis, k=1 is right.
    diag_lengths = [float(np.linalg.norm(cell[i])) for i in range(3)]
    if struct.n_atoms > 0:
        atom_extent = struct.positions.max(axis=0) - struct.positions.min(axis=0)
    else:
        atom_extent = np.zeros(3)
    # The AUTHORITATIVE per-axis periodicity is ``struct.axis_kind``
    # (structure-periodicity.md) -- "periodic" / "isolated" / "transport".
    # Trust it when present; the span-ratio geometry heuristic below is only
    # the fallback for structures that carry no axis_kind.  (The heuristic
    # alone mis-flagged real crystals whose atoms don't reach the cell edge:
    # a rocksalt cell spans ~50%, so its periodic axes read as "vacuum" and
    # got a spurious "k>1 wasted" warn, while a genuinely under-sampled
    # periodic axis with atoms below the 85% span was MISSED.)
    axis_kind = getattr(struct, "axis_kind", None)
    for axis, (k, length) in enumerate(zip(cfg.kgrid, diag_lengths)):
        kind = axis_kind[axis] if (axis_kind and axis < len(axis_kind)) else None
        if kind in ("isolated", "transport"):
            # A vacuum (isolated) or NEGF-open (transport) axis must NOT be
            # Brillouin-zone sampled; k>1 there is wasted (isolated) or wrong
            # (transport imposes a fake Bloch periodicity along the lead).
            if k != 1:
                issues.append(Issue(
                    "warn",
                    f"kgrid[{axis}] = {k} on a {kind} axis; a {kind} axis is "
                    f"not Brillouin-zone sampled (k must be 1) -- k>1 adds "
                    f"cost" + ("" if kind == "isolated"
                              else " and imposes a fake periodicity"),
                    "config.kgrid",
                ))
            continue
        if kind == "periodic":
            if k == 1 and any(kk > 1 for kk in cfg.kgrid):
                issues.append(Issue(
                    "warn",
                    f"kgrid[{axis}] = 1 on a periodic axis while another "
                    f"axis uses k > 1; likely under-converged sampling on "
                    f"this axis",
                    "config.kgrid",
                ))
            continue

        # --- fallback (axis_kind unknown): span-ratio geometry heuristic ---
        # Span ratio: how much of the cell axis the atoms cover.
        # Near 1.0 -> atoms reach edge -> periodic intent.
        # Near 0.0 -> atoms cluster, edges are vacuum -> vacuum intent.
        span_ratio = (atom_extent[axis] / length) if length > 0 else 0.0
        is_periodic_axis = span_ratio > 0.85

        if k != 1 and not is_periodic_axis and length >= 5.0:
            # User asked for k-points on a vacuum-padded axis; rare
            # and almost always wasted cost.  Don't warn for tiny
            # cells (length < 5 Å) where the heuristic is unreliable.
            issues.append(Issue(
                "warn",
                f"kgrid[{axis}] = {k} along an axis of {length:.1f} Å "
                f"where atoms span only {atom_extent[axis]:.1f} Å "
                f"({span_ratio*100:.0f}%); this looks like a vacuum-padded "
                f"axis -- k>1 there adds cost without improving accuracy",
                "config.kgrid",
            ))
        elif k == 1 and is_periodic_axis and any(kk > 1 for kk in cfg.kgrid):
            # An axis where atoms span the full extent (slab / wire /
            # crystal direction) with k=1 while another axis is
            # sampled -- almost always a forgotten k-grid value.
            issues.append(Issue(
                "warn",
                f"kgrid[{axis}] = 1 along an axis where atoms span "
                f"{atom_extent[axis]:.1f} of {length:.1f} Å "
                f"({span_ratio*100:.0f}%, looks periodic) while "
                f"another axis uses k > 1; likely under-converged "
                f"sampling on this axis",
                "config.kgrid",
            ))

    # Net dipole > 1 D in vacuum (no dipole correction).  Image-image
    # dipole interactions in PBC shift molecular energies by an amount
    # that scales with the dipole magnitude squared and as 1/L^3 with
    # the cell size (dipole-dipole ~ 1/r^3).  We use a heuristic EN-based
    # partial-charge
    # estimate (see chemistry.estimate_dipole_moment_debye) -- not a
    # research-grade dipole, but enough to flag "polar molecule in a
    # finite vacuum cell" and recommend a larger cell or an explicit
    # dipole correction.
    #
    # Triggered only when the cell looks like the auto-vacuum case:
    # all kgrid axes == 1 (Gamma-only sampling, no PBC physics
    # intended).  A genuine periodic crystal with k>1 is meant to
    # carry image-image interactions and shouldn't trip this warning.
    if all(k == 1 for k in cfg.kgrid) and len(struct.positions) > 0:
        try:
            from ..chemistry import (estimate_dipole_moment_debye,
                                    formal_charge_from_phosphates)
            net_charge = (cfg.net_charge if cfg.net_charge is not None
                          else formal_charge_from_phosphates(struct))
            dipole = estimate_dipole_moment_debye(struct,
                                                  total_charge=float(net_charge))
        except Exception:
            dipole = 0.0
        if dipole > 1.0:
            issues.append(Issue(
                "warn",
                f"estimated net dipole = {dipole:.1f} D in a 3-D vacuum cell "
                f"-- image-image dipole interactions shift energies (~1/L^3).  "
                f"For an isolated molecule the fix is a LARGER vacuum box "
                f"(dipole-dipole falls off fast).  (SIESTA's SlabDipoleCorrection "
                f"is for a 2-D SLAB with vacuum on one axis, NOT a 3-D "
                f"molecule.)  Estimate from EN-based partial charges; rough "
                f"+/- 50%.",
                "geometry.dipole",
            ))

    # Atoms outside [0, 1) fractional coords with wrap_into_cell=False
    # mean the visualiser will see the molecule in the neighbour cell.
    if not cfg.wrap_into_cell and len(struct.positions) > 0:
        try:
            inv  = np.linalg.inv(cell)
            frac = struct.positions @ inv
            outside = np.any((frac < 0) | (frac >= 1), axis=1)
            n_out = int(outside.sum())
            if n_out > 0:
                issues.append(Issue(
                    "warn",
                    f"{n_out} of {len(struct.positions)} atoms have "
                    f"fractional coords outside [0, 1) but wrap_into_cell "
                    f"= False; visualisations will show the molecule in "
                    f"the neighbour cell",
                    "config.wrap_into_cell",
                ))
        except np.linalg.LinAlgError:
            pass   # Singular cell -- already flagged by determinant check

    return issues
