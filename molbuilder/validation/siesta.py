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
from .chemistry import check_species_labels
from .sidecar import _check_frozen_atoms_consumed


def _check_siesta_pseudo_coverage(struct: Structure, cfg,
                                    *, dest_dir=None,
                                    relativistic: str = "scalar"
                                    ) -> List[Issue]:
    """Run molbuilder.pseudos.check_coverage on the pseudopotentials THIS RUN
    WILL OPEN, so the SIESTA Build->Generate preflight catches:
      * missing .psml files (SIESTA's ``pseudo_read: ERROR: Pseudopotential
        file not found`` after 5 minutes of MPI init -- we surface
        at click-time instead);
      * XC family / authors mismatch (SIESTA SILENTLY uses the
        pseudo's XC even when XC.authors in the .fdf disagrees;
        bond lengths come out wrong with no error -- only molbuilder
        catches this).

    WHICH FILES those are is prep's rule, asked through the one function that
    states it (`pseudos.psml_sources`): the calculation's own folder first,
    then the library ``cfg.psml_lib`` names, for what the folder lacks.  This
    gate read the library alone until 2026-09-25, and prep never told it the
    folder -- so every rung of a transport ladder, whose pseudopotentials come
    with the citation and never from a library, was told "cfg.psml_lib is not
    set".

    With no folder (the Build tab, before a save) only the library can
    answer, and an unset one is a WARN; with a folder, a species in neither
    place is an ERROR.  Suggests projects/pseudopotential/ as the convention
    since that's where the new-project skeleton creates one.

    ``relativistic`` is what the run needs of each file -- ``spin-orbit`` for
    a spin-orbit treatment, which needs fully-relativistic pseudopotentials
    (`science/chemistry-correctness.md` § 2a.3), ``scalar`` otherwise.  The
    caller reads it off the electronic state; until 2026-09-28 nothing passed
    it, so a spin-orbit run was screened as scalar.
    """
    from ..pseudos import (PsmlLibError, check_coverage, ERROR_STATUSES,
                           expected_xc_family, psml_sources, resolve_psml_lib)
    labels = list(dict.fromkeys(str(e).strip() for e in struct.elements))
    found = (psml_sources(labels, dest_dir=dest_dir) if dest_dir is not None
             else dict.fromkeys(labels))
    lacking = [el for el in labels if found[el] is None]
    read_from: dict = {}                  # directory -> the labels read there
    for el in labels:
        if found[el] is not None:
            read_from.setdefault(found[el], []).append(el)

    psml_lib = getattr(cfg, "psml_lib", None)
    if lacking and not psml_lib:
        # WHERE THE PSEUDOPOTENTIALS COME FROM IS STATED, OR THE FOLDER
        # ALREADY HAS THEM -- there is no third answer, and leaving it
        # unstated is the one implicit thing left on this path (user,
        # 2026-09-19: *"the pseudopotential file has to be explicit and it
        # has to be strictly checked at the configuration time"*).
        #
        # SEVERITY IS DECIDED THE WAY THE MISSING-DIRECTORY CASE BELOW
        # DECIDES IT, and for the same reason: with a calculation folder in
        # hand this is answerable, and without one it is not.
        #
        #   * folder known, and it lacks a species      -> ERROR.  Neither
        #     source exists; SIESTA cannot start, and saying so at Generate
        #     beats finding out after MPI init.
        #   * no folder yet (Build tab, before a save)   -> WARN.  Whether
        #     the folder will supply them is not knowable yet, and refusing
        #     on a guess is the thing this whole path is being cleared of.
        return [Issue(
            "error" if dest_dir is not None else "warn",
            ("cfg.psml_lib is not set -- SIESTA needs .psml files for "
             "every element (H, C, N, O, S, Fe, ...) and will refuse "
             "to start without them.  Download from "
             "http://www.pseudo-dojo.org (PBE-SR, standard, PSML "
             "format) and set cfg.psml_lib to that directory.  "
             "Convention: the bare name `pseudopotential`, which means "
             "the projects tree this calculation lives in "
             "(job-contracts.md 2.5a) -- do NOT write the projects/ "
             "prefix.  Once set, this preflight will check "
             "coverage + XC-family match against your structure's "
             "elements automatically."
             + ("  Or put the .psml files in the calculation folder itself "
                "-- pseudos already beside the calculation are used without "
                "this field." if dest_dir is not None else "")),
            "config.psml_lib",
        )]
    if lacking:
        try:
            psml_dir = resolve_psml_lib(psml_lib, dest_dir=dest_dir)
        except PsmlLibError as exc:
            # A spelling the rule cannot answer (outside the tree / dotted).
            # ERROR, and the message already teaches the rule (2.5a).
            return [Issue("error", str(exc), "config.psml_lib")]
        if not psml_dir.is_dir():
            # The spelling named ONE anchor (job-contracts.md 2.5a) and the
            # folder is not there.  Say which anchor, in the rule's own words
            # -- `pseudos.describe_psml_anchor` owns that sentence so this
            # surface and `prep`'s cannot describe the rule differently.
            from ..pseudos import describe_psml_anchor
            from pathlib import Path as _P
            is_relative = not _P(psml_lib).expanduser().is_absolute()
            # Severity, and the one thing that changes it:
            #   * ABSOLUTE miss -> ERROR.  Nothing about context can rescue it.
            #   * RELATIVE, calculation folder known -> ERROR.  The anchor the
            #     spelling named is available and the folder is not there.
            #   * RELATIVE, NO calculation folder -> WARN.  A dotted spelling
            #     means "from this calculation", and there is no calculation
            #     yet; this ran against the server's own tree instead, so a
            #     miss here does not prove a miss at prep time.
            severity = ("error" if (not is_relative or dest_dir is not None)
                        else "warn")
            return [Issue(
                severity,
                f"cfg.psml_lib path does not exist or is not a directory: "
                f"{psml_lib}.  SIESTA will not find any pseudopotentials.  "
                + describe_psml_anchor(psml_lib, dest_dir=dest_dir)
                + "  Create that directory, use an absolute path, or pick "
                  "the directory with the file-picker.",
                "config.psml_lib",
            )]
        read_from.setdefault(psml_dir, []).extend(lacking)
    # The expected XC family, from the ONE table (`pseudos.expected_xc_family`).
    # It was spelled out here and again in `cli.py`, and the two disagreed --
    # the CLI copy had no VDW arm.
    xc_authors = (getattr(cfg, "xc_authors", "") or "").strip()
    expected_family = expected_xc_family(xc_authors)
    out: List[Issue] = []
    for directory, els in read_from.items():
        for entry in check_coverage(
            els, directory,
            expected_xc_family=expected_family,
            expected_xc_authors=xc_authors or None,
            expected_relativistic=relativistic,
        ):
            if entry.status == "ok":
                continue
            # ERROR_STATUSES (missing / dead_projector / xc_family_mismatch)
            # BLOCK: the run cannot be correct.  The rest -- xc_mismatch
            # (same-family author diff) / relativistic_mismatch /
            # generator_mismatch / parse_warning -- are advisory (warn).  The
            # set is shared with the CLI (pseudos.py) so the two surfaces
            # can't drift.
            severity = "error" if entry.status in ERROR_STATUSES else "warn"
            message = entry.message
            if entry.status == "missing" and dest_dir is not None:
                message = (f"no .psml file for {entry.element} in the "
                           f"calculation folder, and {message}")
            out.append(Issue(severity, message,
                              f"config.psml_lib.{entry.element}"))
    return out


def _psml_files(struct, cfg, *, dest_dir=None) -> dict:
    """Each species' `.psml` as the run will open it: the calculation's own
    folder first, then the library (`pseudos.psml_sources`, prep's rule).  A
    species in neither place is absent -- the coverage check reports it."""
    from ..pseudos import psml_sources, resolve_psml_lib
    library = None
    psml_lib = getattr(cfg, "psml_lib", None)
    if psml_lib:
        try:
            library = resolve_psml_lib(psml_lib, dest_dir=dest_dir)
        except Exception:                                # noqa: BLE001
            library = None     # a spelling the rule refuses: coverage says so
    found = psml_sources(getattr(struct, "elements", []) or [],
                         dest_dir=dest_dir, library=library)
    return {el: d / f"{el}.psml" for el, d in found.items() if d is not None}


def _declared_cutoff_ry(struct, cfg, *, dest_dir=None):
    """The strictest mesh cutoff THE PSEUDOS THEMSELVES ask for, or ``None``.

    Layer 2 of `science/pseudopotentials.md` § 2a: a pseudopotential does not
    only have to be sound, it **states** what the calculation must give it.
    PseudoDojo writes `cutoff_hint_normal` (Ry) into the file; this reads it
    back for the elements actually in the structure.

    **The MAXIMUM wins, because there is one grid.** The mesh is a single
    global real-space grid and every species' density lives on it, so it must
    satisfy the most demanding element. An average, or a minimum, would answer
    *lower* the more species a system has — backwards, since adding a species
    can only make the grid's job harder.

    **Silence abstains.** Only the eleven elements v0.5 re-generated carry
    hints, so a real system states fewer numbers than it has atoms — BDT on
    gold states exactly one (S, 147 Ry). An element with no hint must not pull
    the requirement down; it simply does not raise it.

    Returns ``(ry, element)`` for the element that set the bar, or
    ``(None, None)``. Silent on every failure the coverage check already
    reports — a missing directory is an INTEGRITY finding, and layer 2 has
    nothing to add to it.

    Read from the files the run will open (`_psml_files`): until 2026-09-25
    this read the library alone, so a calculation whose pseudopotentials sat
    in its own folder -- every transport rung -- had its hints ignored.
    """
    try:
        from ..pseudos import parse_psml_header
        best_ry, best_el = None, None
        for el, f in sorted(_psml_files(struct, cfg, dest_dir=dest_dir).items()):
            ry = parse_psml_header(f).suggested_mesh_ry
            if ry is not None and (best_ry is None or ry > best_ry):
                best_ry, best_el = float(ry), el
        return (best_ry, best_el)
    except Exception:                                    # noqa: BLE001
        # A requirement we could not read is not a requirement we may invent.
        return (None, None)


def _check_siesta_mesh_cutoff(cfg, struct=None, *, dest_dir=None) -> List[Issue]:
    """Is the mesh cutoff adequate — asked of the FILES first, the
    literature only when they are silent.

    **One question, one check, two sources for the answer** *(§ 2a.1,
    2026-09-03)*. A pseudo that states its own recommended cutoff has measured
    this calculation; a 150 Ry literature floor has not. So a declared number
    outranks the floor, and the floor answers only where nothing is declared —
    the same rule the rank count follows (`running-a-job.md` § 3.1: read from
    a record, never guessed). Two separate checks would have put a guess and a
    measurement on one field and let the user pick.

    The threshold is the `normal` hint; `high` is named in the message,
    because it is what tight and vibrational work wants (`tuning.md` § 1) and
    that is a decision, not a bar everyone must clear.

    Below, the original floor, unchanged and now scoped to its real case:

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

    # THE FILES FIRST.  When any pseudo in this calculation states a
    # recommended cutoff, that number is about THIS calculation and the
    # literature floor is not -- so it answers, and the floor stays quiet.
    declared, el = (_declared_cutoff_ry(struct, cfg, dest_dir=dest_dir)
                    if struct is not None else (None, None))
    if declared is not None:
        if mc_val >= declared:
            return []
        from ..pseudos import parse_psml_header
        high = None
        try:
            f = _psml_files(struct, cfg, dest_dir=dest_dir)[el]
            high = parse_psml_header(f).cutoff_hints_ry.get("high")
        except Exception:                                # noqa: BLE001
            pass
        return [Issue(
            "warn",
            (f"mesh_cutoff = {mc_val:g} Ry is below {declared:g} Ry, which "
             f"is what {el}'s own pseudopotential asks for "
             f"(PseudoDojo's `normal` hint, read from {el}.psml).  The mesh "
             f"is ONE grid for the whole calculation, so the most demanding "
             f"element sets it"
             + (f"; {el} is that element here" if el else "")
             + ".  Below it that element's density is under-resolved, and "
               "the error sits on those atoms rather than averaging away."
             + (f"  For tight or vibrational work the same file suggests "
                f"{high:g} Ry." if high else "")),
            "config.mesh_cutoff",
        )]
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
                                              state) -> List[Issue]:
    """SIESTA-specific: charged system in a periodic supercell carries
    an image-charge artefact that padding alone does NOT remove.

    See Makov & Payne, Phys. Rev. B 51, 4014 (1995).  Leading term:

        E_bias ~ q^2 * alpha / (2 * L * eps_r)

    For q = +/-1 at typical molbuilder vacuum-cell sizes (15-25 A) the
    bias is 0.5-1.5 eV -- well above chemical accuracy (~0.04 eV).

    molbuilder does NOT auto-apply the Makov-Payne correction; this
    warn surfaces the issue so users computing redox / pKa /
    deprotonation energies know to apply it post-hoc.

    Severity: WARN (not ERROR) -- the calculation still runs and a
    user doing a single-point screening calc may not care.  We're
    nudging, not blocking.

    ``state`` is the electronic state the deck carries
    (`science/chemistry-correctness.md` § 2a): its charge -- stated, the
    run's, or the phosphate rule's; 0 on a transport rung, whose junction is
    neutral by rule (this fired on every rung of a charged citation until
    2026-09-28) -- and whether its system is finite.  Skipped at charge 0.

    KEYED ON THE SYSTEM (§ 2b, the M6 review): a charged MOLECULE gets the
    estimate, and the word that SIESTA applies the leading term itself in a
    cubic cell (``siesta: Emadel``, already in E_KS), which the companion
    script reads first.  A charged REPEATING cell -- a slab, a defect in a
    crystal -- gets no formula, because the point-charge correction is the
    wrong one there; it is told its energy needs a defect-specific
    treatment.  Both got the molecule's message until then.
    """
    from ..siesta.makov_payne import compute_correction
    q = int(state.net_charge.value)
    if q == 0:
        return []
    if not state.finite:
        return [Issue(
            "warn",
            (f"Charged repeating cell (NetCharge = {q:+d}).  SIESTA adds a "
             f"uniform compensating background charge, so the total energy "
             f"is not comparable with a neutral cell's, and SIESTA's own "
             f"monopole correction does not apply (it corrects a molecule "
             f"only).  A point-charge correction in a cubic box is the wrong "
             f"formula for a slab or a defect in a crystal, so molbuilder "
             f"writes none: the energy needs a defect-specific treatment "
             f"(science/chemistry-correctness.md § 2b)."),
            "config.net_charge.makov_payne",
        )]
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
         f"SIESTA applies the leading term itself when the cell is simple, "
         f"face- or body-centred cubic (``siesta: Emadel``, already in "
         f"E_KS), and not otherwise.  molbuilder emits a companion "
         f"``makov_payne_correction.py`` beside the FDF; after SIESTA "
         f"finishes, run it: it reads ``Emadel`` first and adds only what "
         f"SIESTA did not.  See Makov & Payne, PRB 51, 4014 (1995)."),
        "config.net_charge.makov_payne",
    )]


# Recommended per-side vacuum (Angstrom) for an isolated molecule so its
# periodic images don't interact: basis orbitals reach 4-7 A per atom with a DZP
# basis and the inter-image gap is 2*vacuum, so the neutral floor is set above
# the largest orbital radius; a charged system needs far more because the
# image-charge Coulomb bias decays only as 1/L (see the Makov-Payne notice).
_VACUUM_MIN_NEUTRAL = 8.0
_VACUUM_MIN_CHARGED = 25.0


def _check_siesta_vacuum_adequacy(struct: Structure,
                                  charge: int) -> List[Issue]:
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
    there, not the vacuum.  Never mutates -- the structure is the truth.

    ``charge`` is the electronic state's resolved charge: a charged system
    needs far more vacuum (its image bias decays only as 1/L)."""
    q = int(charge)
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


# --------------------------------------------------------------------- #
#  SIESTA aggregator                                                    #
#                                                                       #
#  CALL ORDER IS LOAD-BEARING.  Tests that count issues by position    #
#  depend on this exact sequence.  Do not reorder.                     #
# --------------------------------------------------------------------- #


def _validate_siesta(struct: Structure, cfg,
                     cell: Optional[np.ndarray],
                     *, dest_dir=None, calculation: str = "",
                     **_) -> List[Issue]:
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
    # ONE FACT, ONE FINDING (science/validation.md § 7): the vibration kind
    # names the unconsumed region labels and states the held atoms itself,
    # from the rank rule, so this validator defers both families on that
    # kind -- the same deferral the PySCF validator makes.  A second copy
    # here spoke of "relaxation" on a run that relaxes nothing.
    vibration = calculation == "vibration"

    # Pattern B, re-homed here from the deleted web endpoints (C-shared
    # 2026-08-21): region labels this run does not consume are named --
    # the frozen label is excluded (SIESTA consumes it as
    # Geometry.Constraints).
    from .sidecar import check_unconsumed_region_labels
    if not vibration:
        issues += check_unconsumed_region_labels(
            struct, engine="SIESTA", calculation=calculation)
    # ...and the FIRST validation of a junction, beside it because it is
    # the same question one step further: the labels say which atoms are
    # leads, and a lead must come through the relaxation unmoved.  Asked
    # HERE, where it is cheap -- the compose-time gate that refuses a
    # MOVED lead is correct and runs after the relaxation is paid for.
    from .sidecar import check_electrode_labels_are_frozen
    # The fact holds on every kind (an unheld lead is moved by whatever
    # moves atoms); the sentence names the run that would move it, so a
    # force-constant run is not told about a relaxation it does not do
    # (science/validation.md § 7).
    issues += check_electrode_labels_are_frozen(
        struct, run=("the force-constant run" if vibration
                     else "the relaxation"))

    # A species label must name an element -- SIESTA needs the Z beside
    # it in ChemicalSpeciesLabel.  Blocks here with a readable message
    # rather than raising from inside the emitter.
    issues += check_species_labels(struct, engine_label="SIESTA")

    # THE ELECTRONIC STATE THE DECK WILL CARRY (`science/chemistry-
    # correctness.md` § 2a) -- the charge and the treatment the checks
    # below are about.  Its own findings (parity, the open-shell guard, what
    # SIESTA cannot run) are one family asked from `validate` for every
    # engine; here it is only READ.  None when a species label names no
    # element: the state is an electron count, and the label check above
    # owns that finding.
    from ..chemistry import every_label_resolves
    from ..electronic_state import KINDS, electronic_state
    state = (electronic_state(struct, cfg, kind=calculation)
             if calculation in KINDS and every_label_resolves(struct)
             else None)

    # Pseudopotential coverage (the actionable use of pseudos.py).
    # Wired into preflight + render: missing files become ERROR
    # Issues (SIESTA hard-fails without them); XC mismatches
    # become WARN (silent wrong bond lengths otherwise).  A spin-orbit run
    # needs fully-relativistic files.
    issues += _check_siesta_pseudo_coverage(
        struct, cfg, dest_dir=dest_dir,
        relativistic=("spin-orbit" if state is not None
                      and state.spin_treatment.value == "spin-orbit"
                      else "scalar"))

    # MeshCutoff floor: warn below 150 Ry (production-defensible
    # threshold).  The dataclass slider lower bound is 100 Ry; this
    # rule catches the 100-149 Ry window with a soft nudge.
    issues += _check_siesta_mesh_cutoff(cfg, struct, dest_dir=dest_dir)

    # The checks that read the CHARGE the deck carries.  With no state -- a
    # label naming no element, the label check's finding above -- they stand
    # down rather than judge the structure as neutral: this took 0 there
    # until the M6 review, whatever charge the template stated.
    if state is not None:
        # Makov-Payne notice: charged-supercell image-charge bias.  We DON'T
        # auto-apply the correction (see function docstring + design.md
        # decisions log); we surface it so the user knows what's missing.
        issues += _check_siesta_charged_makov_payne_notice(struct, state)
        # Vacuum adequacy on isolated axes (R5: was a warnings.warn in the
        # emitter, invisible to the web; now a finding on every surface).
        issues += _check_siesta_vacuum_adequacy(struct,
                                                state.net_charge.value)

    # Frozen-atom carrier (three-stage contract).  SIESTA honors
    # struct.frozen_atoms via %block Geometry.Constraints which is
    # only meaningful inside an MD/relax block.  When relax_type is
    # "none" the relaxer doesn't run, so the constraint is a no-op.
    #
    # NOT ON A TRANSPORT RUNG, which writes no MD block at all: the junction
    # was relaxed upstream and every rung computes at that geometry
    # (engines/transport.md § 1), so the frozen set holds nothing there.
    # Reasoned from `relax_type`'s catalogue default, this said "held fixed
    # during SIESTA relaxation" on every junction rung (2026-09-25) -- the
    # vibration kind's failure (science/validation.md § 7) on a second kind.
    relax = (getattr(cfg, "relax_type", "") or "").lower()
    if not vibration and calculation != "transport":
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
    # list and duplicate names are refused by ``task.py``, where a
    # description's ladder is read (the render-time copy in
    # ``_enabled_stages`` died with its producers, step 6 u5).  Cross-stage findings -- a ladder that loosens -- are
    # P2 unit 6 and carry no stage label (§ 4).

    # A VALUE THAT CANNOT MATTER, said out loud.  The free-energy tolerance is
    # loaded by SIESTA either way and installed as a criterion only when
    # `SCF.FreeE.Converge` is on (read_options.F90 / siesta_forces.F90).  A user
    # who tightens the tolerance to chase a convergence problem, with the switch
    # off, changes nothing and has no way to discover that from the run.
    _tol_default = 1e-4
    if (not getattr(cfg, "scf_energy_converge", False)
            and cfg.dm_energy_tolerance is not None
            and abs(float(cfg.dm_energy_tolerance) - _tol_default) > 1e-12):
        issues.append(Issue(
            "warn",
            f"SCF free-energy tolerance (DM.EnergyTolerance) is set to "
            f"{cfg.dm_energy_tolerance:g} eV, away from its default, but "
            f"free-energy convergence is switched off -- so SIESTA reads the "
            f"value and never uses it. Turn on 'Also require the free energy "
            f"to settle' (SCF.FreeE.Converge) to make this tolerance apply, or "
            f"leave it at its default",
            "config.dm_energy_tolerance",
        ))

    if cell is None:
        return issues

    # k-grid vs the axes (user rule, 2026-08-20): ``k > 1`` is the USER'S
    # EXPLICIT statement -- "sample a supercell along this axis" -- so that is
    # the only place a consistency question exists.  ``k == 1`` states
    # nothing (correct for an isolated axis, and a legitimate Gamma-only
    # choice for a periodic one) and is validated NOT AT ALL.
    #
    # Where k > 1, two facts can contradict the statement:
    #   * the axis is declared ``isolated`` / ``transport`` -- sampling a
    #     direction the user said does not repeat (isolated) or must not be
    #     given fake Bloch periodicity (transport);
    #   * the axis is periodic on paper but its periodic images sit far
    #     apart -- the GEOMETRIC gap (cell extent minus atom span) is the
    #     real vacuum, whether or not the vacuum field was ever set.  A gap
    #     >= 5 A usually means the images are meant to interact weakly or
    #     not at all, so k > 1 earns a HINT, not a refusal: minor-image
    #     interaction can be a deliberate setup, and the user knows which.
    #
    # (This replaces two earlier rules the 2026-08-20 decision retired: a
    # span-ratio heuristic that judged intent geometrically even at k == 1,
    # and an "under-converged" warning on k == 1 periodic axes -- both were
    # validating an axis about which the user had stated nothing.)
    VACUUM_HINT_A = 5.0
    diag_lengths = [float(np.linalg.norm(cell[i])) for i in range(3)]
    if struct.n_atoms > 0:
        atom_extent = struct.positions.max(axis=0) - struct.positions.min(axis=0)
    else:
        atom_extent = np.zeros(3)
    axis_kind = getattr(struct, "axis_kind", None)
    for axis, (k, length) in enumerate(zip(cfg.kgrid, diag_lengths)):
        if k == 1:
            continue                       # nothing stated, nothing checked
        kind = axis_kind[axis] if (axis_kind and axis < len(axis_kind)) else None
        if kind in ("isolated", "transport"):
            issues.append(Issue(
                "warn",
                f"kgrid[{axis}] = {k} on a {kind} axis; a {kind} axis is "
                f"not Brillouin-zone sampled (k must be 1) -- k>1 adds "
                f"cost" + ("" if kind == "isolated"
                          else " and imposes a fake periodicity"),
                "config.kgrid",
            ))
            continue
        gap = max(0.0, length - float(atom_extent[axis]))
        if gap >= VACUUM_HINT_A:
            issues.append(Issue(
                "warn",
                f"kgrid[{axis}] = {k} samples a supercell along an axis "
                f"whose periodic images sit ~{gap:.1f} A apart; if the "
                f"images are meant not to interact, k = 1 is the usual "
                f"choice -- if a weak image interaction is deliberate, "
                f"carry on",
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
    if state is not None and all(k == 1 for k in cfg.kgrid) \
            and len(struct.positions) > 0:
        try:
            from ..chemistry import estimate_dipole_moment_debye
            # The state's charge -- this spelled the charge rule out inline
            # until 2026-09-28, a second copy of it.
            dipole = estimate_dipole_moment_debye(
                struct, total_charge=float(state.net_charge.value))
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

    return issues
