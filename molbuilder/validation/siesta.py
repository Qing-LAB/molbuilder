"""SIESTA-specific validators + the SiestaConfig aggregator.

The aggregator ``_validate_siesta`` is what gets registered against
``SiestaConfig`` in the engine-validator registry.
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
    WILL OPEN, so the SIESTA preflight catches:
      * missing .psml files (SIESTA's ``pseudo_read: ERROR: Pseudopotential
        file not found`` after 5 minutes of MPI init -- we surface
        at click-time instead);
      * XC family / authors mismatch (SIESTA SILENTLY uses the
        pseudo's XC even when XC.authors in the .fdf disagrees;
        bond lengths come out wrong with no error -- only molbuilder
        catches this).

    WHICH FILES those are is prep's rule, asked through the one function that
    states it (`pseudos.psml_sources`): the calculation's own folder first,
    then the library ``cfg.psml_lib`` names, for what the folder lacks.

    With no folder (the Build tab, before a save) only the library can
    answer, and an unset one is a WARN; with a folder, a species in neither
    place is an ERROR.  Suggests ``CONVENTIONAL_LIBRARY`` as the
    convention.

    ``relativistic`` is what the run needs of each file -- ``spin-orbit`` for
    a spin-orbit treatment, which needs fully-relativistic pseudopotentials
    (`science/chemistry-correctness.md` § 2a.3), ``scalar`` otherwise.  The
    caller reads it off the electronic state.
    """
    from ..pseudos import (CONVENTIONAL_LIBRARY, PsmlLibError, check_coverage,
                           ERROR_STATUSES, expected_xc_family, psml_sources,
                           resolve_psml_lib)
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
             "to start without them."
             + _the_convention_covers(lacking, cfg, dest_dir=dest_dir,
                                      relativistic=relativistic)
             + "  Download from "
             "http://www.pseudo-dojo.org (PBE-SR, standard, PSML "
             "format) and set cfg.psml_lib to that directory.  "
             f"Convention: the bare name `{CONVENTIONAL_LIBRARY}`, which "
             "means the projects tree this calculation lives in "
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


def _the_convention_covers(lacking, cfg, *, dest_dir=None,
                           relativistic: str = "scalar") -> str:
    """The sentence an unset directory earns when the tree's own
    ``pseudopotential`` folder would answer it (plan § 5w K20): named only
    when it holds every element the calculation lacks and the coverage check
    refuses none of them -- a suggestion, never a fill.  ``""`` otherwise."""
    from ..pseudos import (CONVENTIONAL_LIBRARY, ERROR_STATUSES, PsmlLibError,
                           check_coverage, expected_xc_family,
                           resolve_psml_lib)
    try:
        folder = resolve_psml_lib(CONVENTIONAL_LIBRARY, dest_dir=dest_dir)
    except PsmlLibError:
        return ""
    if not lacking or not folder.is_dir():
        return ""
    xc_authors = (getattr(cfg, "xc_authors", "") or "").strip()
    found = check_coverage(lacking, folder,
                           expected_xc_family=expected_xc_family(xc_authors),
                           expected_xc_authors=xc_authors or None,
                           expected_relativistic=relativistic)
    if any(e.status in ERROR_STATUSES for e in found):
        return ""
    return (f"  The tree's `{CONVENTIONAL_LIBRARY}` folder covers all "
            f"{len(lacking)} element(s) this needs ({', '.join(lacking)}): "
            f"set the field to `{CONVENTIONAL_LIBRARY}`.")


def _relativistic(state) -> str:
    """What the run needs of each file: fully relativistic for a spin-orbit
    treatment (`science/chemistry-correctness.md` § 2a.3), scalar otherwise."""
    return ("spin-orbit" if state is not None
            and state.spin_treatment.value == "spin-orbit" else "scalar")


def pseudopotential_findings(struct: Structure, cfg, *,
                             calculation: str = "optimization",
                             dest_dir=None) -> List[Issue]:
    """The pseudopotential check alone, as the settings gate asks it -- for a
    door that asks nothing else: the hand-over, before it writes a SIESTA
    calculation's folder, which it names as ``dest_dir`` so the files
    already there count as at `prep` (`web/handover-procedure.md` § 2.2,
    plan § 5w K20)."""
    from ..chemistry import every_label_resolves
    from ..electronic_state import KINDS, electronic_state
    state = (electronic_state(struct, cfg, kind=calculation)
             if calculation in KINDS and every_label_resolves(struct)
             else None)
    return _check_siesta_pseudo_coverage(struct, cfg, dest_dir=dest_dir,
                                         relativistic=_relativistic(state))


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

    Read from the files the run will open (`_psml_files`).
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

    Below, the literature floor, scoped to its real case:

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
    neutral by rule -- and whether its system is finite.  Skipped at charge 0.

    KEYED ON THE SYSTEM (§ 2b, the M6 review): a charged MOLECULE gets the
    estimate, and the word that SIESTA applies the leading term itself in a
    cubic cell (``siesta: Emadel``, already in E_KS), which the companion
    script reads first.  A charged REPEATING cell -- a slab, a defect in a
    crystal -- gets no formula, because the point-charge correction is the
    wrong one there; it is told its energy needs a defect-specific
    treatment.
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

    Lives HERE, in the validator, rather than in the emitter: a finding never
    travels as a warning (science/validation.md 4.1, clause R5).  As an Issue
    it reaches BOTH surfaces -- the web panel through the endpoint's
    ``issues[]``, and `jobset prep` through ``script_emit.render_deck``'s
    ``report(validate(...))``.

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
                     k_meshes=None, refused=frozenset(),
                     **_) -> List[Issue]:
    """SIESTA-specific checks.

    Registered in `validation/__init__.py`'s ``_ENGINE_VALIDATORS``.

    ``dest_dir`` (keyword-only) is passed through to the pseudo-
    coverage check so dest-relative ``cfg.psml_lib`` paths resolve
    correctly post-Save (see pseudos.resolve_psml_lib).

    ``k_meshes`` are the k-point meshes the deck writes, when its spec built
    them (`kmesh.mesh_for`); without them the one this configuration writes
    is derived through the same door.  ``refused`` names the items the one
    per-value door refused: a mesh built from one is not judged.
    """
    issues: List[Issue] = []
    # ONE FACT, ONE FINDING (science/validation.md § 7): the vibration kind
    # names the unconsumed region labels and states the held atoms itself,
    # from the rank rule, so this validator defers both families on that
    # kind -- the same deferral the PySCF validator makes.
    vibration = calculation == "vibration"

    # Pattern B: region labels this run does not consume are named --
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
    # ...and the room at its transport boundary -- one layer spacing of the
    # lead, I12 -- said here as a warning: a fused junction cell relaxes as
    # it stands and is refused only on a transport rung, whose kind gate
    # owns the rule there (`_validate_transport_kind`).
    if calculation != "transport":
        from .sidecar import check_junction_boundary
        issues += check_junction_boundary(struct, vacuum=False)

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
        struct, cfg, dest_dir=dest_dir, relativistic=_relativistic(state))

    # MeshCutoff floor: warn below 150 Ry (production-defensible
    # threshold).  The dataclass slider lower bound is 100 Ry; this
    # rule catches the 100-149 Ry window with a soft nudge.
    issues += _check_siesta_mesh_cutoff(cfg, struct, dest_dir=dest_dir)

    # The checks that read the CHARGE the deck carries.  With no state -- a
    # label naming no element, the label check's finding above -- they stand
    # down rather than judge the structure as neutral.
    if state is not None:
        # Makov-Payne notice: charged-supercell image-charge bias.  We DON'T
        # auto-apply the correction (see function docstring); we surface it
        # so the user knows what's missing.
        issues += _check_siesta_charged_makov_payne_notice(struct, state)
        # Vacuum adequacy on isolated axes (R5: a finding on every surface).
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

    # THE K-POINT MESH, judged as the deck writes it (`engines/siesta.md`
    # § 6.1): the spec hands the gate the mesh(es) it built -- a transport
    # rung's open or lead axis, a transmission's own grid -- and a caller
    # that hands none gets the one this configuration writes on its own.
    # ONE derivation, `kmesh.mesh_for`.
    from .. import kmesh as _kmesh
    meshes = tuple(m for m in (k_meshes if k_meshes is not None else (
        _kmesh.mesh_for(cfg, getattr(struct, "axis_kind", None),
                        kind=calculation or "optimization"),)) if m is not None)
    issues += _kmesh.check(meshes, struct, cell=cell, refused=refused)

    if cell is None:
        return issues

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
    # Triggered only for a box of vacuum on every axis -- an isolated
    # molecule: the axes SIESTA computes on (`cell.engine_axis_kinds`,
    # `model/structure-periodicity.md` § 2.1), never the k-point count.
    from ..cell import engine_axis_kinds
    from ..template import engine_name
    vacuum_box = all(k == "isolated" for k in
                     engine_axis_kinds(engine_name(type(cfg)), struct))
    if state is not None and vacuum_box and len(struct.positions) > 0:
        try:
            from ..chemistry import estimate_dipole_moment_debye
            # The state's charge.
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
