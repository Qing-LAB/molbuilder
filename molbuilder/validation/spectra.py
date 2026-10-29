"""The spectra render gate -- the science checks a vibration deck runs.

Called from ``validation/__init__.py``'s vibration arm.  Lives in
``validation/`` beside the other engines' gates.
"""

from __future__ import annotations

from typing import List, Optional

from ..issues import Issue
from ..structure import Structure, atom_words


def _sentence(text: str) -> str:
    """``text`` opening a sentence: its first letter capital, the rest as
    written (``str.capitalize`` lowercases the rest -- *Raman* included)."""
    return text[:1].upper() + text[1:]


def spectra_render_checks(struct: Structure, cfg, *,
                          inactive=frozenset()) -> List[Issue]:
    """PySCF SCIENTIFIC advisories — THE render gate for a vibration
    deck (grid / amplitude / the mode selection / the frozen set).  The
    charge and the spin are the electronic state's one family, asked from
    `validate` for every kind.

    Called from ``validation/__init__.py``'s vibration arm, over the
    deck's config view -- an ADAPTER over PySCFConfig, so ``cfg`` is
    duck-typed to the spectra vocabulary.  ``inactive`` names the settings
    that enter nothing here (`metadata._inactive`): the probe's advice
    stands aside while its displacement is one of them."""
    issues: List[Issue] = []

    # -- PySCF-specific scientific advisories ------------------

    # Grid level with a hybrid functional.  PySCF's
    # default grid level is 3 ("screening"); level 4 is the
    # production minimum for hybrids.  Below that the XC numerical
    # integration noise dominates the Hessian and gives garbage
    # frequencies (~5-20 cm⁻¹ wander).  ONE shared gate/body with the
    # PySCF validator -- the
    # "spectra" context selects the frequency-error rationale.
    from .pyscf import check_dft_grid_level
    issues.extend(check_dft_grid_level(cfg, context="spectra"))

    # THE PROBE'S ADVICE, while the probe runs: its displacement enters
    # nothing under `skip` (the item's `active_when`), and no advice is
    # given about a value that enters nothing (`engines/template.md` § 5).
    if "displacement_amplitude_ang" not in inactive:
        # Displacement amplitude.  The [0.02, 0.20] Å acceptance
        # range is empirical -- not derived from a single source.
        # 0.02 Å keeps
        # the probe inside the linear-response regime (ΔE_orbital
        # ∝ A), at the cost of needing a tight SCF tolerance to
        # resolve the smaller ΔE.  The script's default
        # ``scf_conv_tol = 1e-9`` is usually sufficient (FD noise on ΔE_HOMO
        # at 0.02 Å is ~1e-8 Ha << ΔE for a typical bond-stretch
        # mode).  Above 0.20 Å cubic anharmonicity in the
        # potential becomes significant (cf. Mills1972 §2.4).
        amp = cfg.displacement_amplitude_ang
        if amp < 0.02:
            issues.append(Issue(
                severity="warn",
                message=(f"Displacement amplitude {amp:g} Å is "
                         f"smaller than the accepted range "
                         f"(0.02-0.20 Å).  At this scale the "
                         f"finite-difference orbital-energy slope "
                         f"is at or below the noise floor of "
                         f"the default scf_conv_tol=1e-9 SCF; either tighten "
                         f"scf_conv_tol further or raise the "
                         f"displacement to at least 0.02 Å."),
                where="config.displacement_amplitude_ang",
            ))
        elif amp > 0.20:
            issues.append(Issue(
                severity="warn",
                message=(f"Displacement amplitude {amp:g} Å is "
                         f"larger than the accepted range "
                         f"(0.02-0.20 Å).  At large amplitudes the "
                         f"potential isn't linear in the displacement "
                         f"any more -- anharmonic terms contaminate "
                         f"the orbital-energy slope you want to "
                         f"measure.  Lower to 0.20 Å or less."),
                where="config.displacement_amplitude_ang",
            ))

        # ---- Amplitude x SCF-tolerance coupling (SCIENTIFIC-AUDIT, FN-4) ----
        # The finite-difference orbital-energy SIGNAL scales with the
        # displacement amplitude; it must sit well ABOVE the SCF noise floor
        # (~scf_conv_tol).  A SMALL amplitude paired with a LOOSE conv_tol
        # buries the signal -- the exact failure the amplitude window guards,
        # but the window alone never looked at the actual tolerance.  Warn
        # when both are near their risky ends (conservative; a soft advisory).
        conv = getattr(cfg, "scf_conv_tol", None)
        if (conv is not None and conv > 1e-7 and amp <= 0.05):
            issues.append(Issue(
                severity="warn",
                message=(f"Small displacement ({amp:g} Å) with a loose "
                         f"scf_conv_tol ({conv:g}): the finite-difference "
                         f"orbital-energy slope (∝ amplitude) may be at or "
                         f"below the SCF noise floor (~conv_tol), giving noisy "
                         f"electronic-structure shifts.  Tighten scf_conv_tol "
                         f"(≤1e-8) or raise the displacement amplitude."),
                where="config.scf_conv_tol",
            ))

    # The charge, the spin and the method are NOT judged here: they are the
    # electronic state's, and `validate` asks its one family once for every
    # engine and kind (`validation.chemistry.check_electronic_state`,
    # `science/chemistry-correctness.md` § 2a).

    # The mode selector (:func:`selector_issues`): an empty explicit list is
    # a run that computes no orbital-energy data though the person asked for
    # some.
    issues.extend(selector_issues(
        cfg, none_means="No per-mode orbital-energy data will be computed."))
    issues.extend(raman_issues(cfg, engine="pyscf"))

    # Large-system cost advisory.  The Hessian cost scales like
    # N_free² and the Raman finite-difference step adds 6·N_free
    # SCFs.  For ~30+ free atoms this can dominate the run; if
    # the user has metal-slab-or-similar anchors they should
    # consider freezing them.
    try:
        n_atoms = (struct.n_atoms
                   if hasattr(struct, "n_atoms")
                   else len(struct.elements))
    except Exception:
        n_atoms = None
    if n_atoms is not None:
        # FROZEN ATOMS ARE INDICES: the region store holds indices, and
        # that is what the deck writes into the constraints file.
        # A COST CLAIM IS A MEASUREMENT (science/normal-modes.md R8).  With
        # atoms held, the deck takes second derivatives for the free atoms
        # only (PySCF's atmlst reaches the coupled-perturbed solve) and runs
        # the Raman and finite-difference infrared loops over them only; the
        # held atoms still enter through the SCF, which stays a whole-system
        # SCF.  The measured numbers are in the design's § 10.
        n_free_estimate = max(0, n_atoms - len(cfg.frozen_indices))
        if n_free_estimate > 30 and not cfg.frozen_indices:
            issues.append(Issue(
                severity="warn",
                message=(
                    f"This structure has {n_atoms} atoms, none of them "
                    f"frozen.  Second derivatives and the Raman "
                    f"finite-difference loop (6 displaced SCFs per free "
                    f"atom) are taken for the free atoms only, so freezing "
                    f"a slab, surface or other anchor you do not need to "
                    f"vibrate scales both down in proportion; each SCF "
                    f"still covers the whole system.  Ignore this if the "
                    f"whole system needs to vibrate."
                ),
                where="structure.regions",
            ))
        # INFRARED WITH ATOMS HELD is the finite-difference dipole route --
        # the analytic one takes the whole Hessian -- so its cost is said
        # where Compute IR, on by default, is the only strength asked
        # (`engines/vibration.md` § 3.1, § 4.6; the deck's
        # `_WANT_ANALYTIC_IR`).  With Raman on it rides that sweep.
        if (cfg.frozen_indices and n_free_estimate
                and bool(getattr(cfg, "compute_ir", False))
                and not bool(getattr(cfg, "compute_raman", False))):
            issues.append(Issue(
                severity="info",
                message=(
                    f"With atoms held, the infrared is taken from the dipole "
                    f"at small pushes of each free atom: "
                    f"{6 * n_free_estimate} extra SCFs (6 per free atom), "
                    f"each covering the whole system, beside the Hessian.  "
                    f"Turn Compute IR off for the frequencies alone."),
                where="config.compute_ir",
            ))
        # WHAT RAMAN AND THE ORBITAL PROBE COST, said when they are asked
        # (`engines/vibration.md` § 3.3: "the cost of what was ticked") --
        # the deck's own loops, counted as it runs them.
        if n_free_estimate and bool(getattr(cfg, "compute_raman", False)):
            issues.append(Issue(
                severity="info",
                message=(
                    f"Raman: {6 * n_free_estimate} extra SCFs (6 per free "
                    f"atom), each covering the whole system, with a "
                    f"polarizability response calculation at each and no "
                    f"density fitting"
                    + ("; the infrared comes from the same sweep."
                       if bool(getattr(cfg, "compute_ir", False)) else ".")),
                where="config.compute_raman",
            ))
        _probe = str(getattr(cfg, "es_mode_selection", "skip"))
        if _probe in ("all", "explicit"):
            try:
                _picked = (len(cfg.explicit_modes) if _probe == "explicit"
                           else modes_of(struct, "pyscf")[2])
            except ValueError:
                _picked = None   # refused by the selector's own check
            if _picked:
                issues.append(Issue(
                    severity="info",
                    message=(
                        "Orbital energies along a mode: "
                        + (f"{2 * _picked} extra SCFs (2 per mode listed)"
                           if _probe == "explicit" else
                           f"up to {2 * _picked} extra SCFs (2 per mode"
                           + (" within the window" if (
                               getattr(cfg, "freq_min_cm1", None) is not None
                               or getattr(cfg, "freq_max_cm1", None)
                               is not None) else "")
                           + f", {_picked} modes in all)")
                        + ", each covering the whole system."),
                    where="config.es_mode_selection",
                ))

    # WHAT SURVIVES THE FREEZE, from the one derivation (science/normal-modes.md
    # R1, R7).  Holding atoms removes the whole-body motions that would move
    # them and leaves the rest -- a turn about the line through two held
    # atoms, three turns about a single held atom -- and those leftovers are
    # not vibrations.  The deck projects them out before diagonalising, so
    # nothing here is a warning: it is the count the person will see missing
    # from 3 N_free, said before the run is paid for.
    _frozen_idx = sorted(int(i) for i in (struct.frozen_atoms or []))
    if _frozen_idx and getattr(struct, "positions", None) is not None:
        try:
            # ON THE AXES THE DECK COMPUTES ON -- PySCF's, a cluster's
            # (`cell.engine_axis_kinds`, plan § 5w K8).
            _n_free, _n_rigid, _n_modes = modes_of(struct, "pyscf")
        except ValueError:
            _n_rigid = None      # an index out of range is reported below
        if _n_rigid:
            issues.append(Issue(
                severity="info",
                message=(
                    f"Holding {len(_frozen_idx)} atom(s) leaves {_n_rigid} "
                    f"whole-body motion(s) of the free atoms that cost no "
                    f"energy (a turn about the held atoms).  They are not "
                    f"vibrations: the deck removes them before diagonalising, "
                    f"so {_n_modes} modes will be reported, not "
                    f"{3 * _n_free}, and the count removed is recorded with "
                    f"the results."
                ),
                where="structure.regions",
            ))

    # A SETTING THAT ENTERS NOTHING IS NOT LEFT SILENT (engines/vibration.md
    # § 3.1; user, 2026-09-28).  The pressure enters only the free molecule's
    # gas-phase translation; with atoms held the thermochemistry is the
    # vibrational sums alone and records no pressure (§ 4.7).  A value the
    # person set is said to do nothing -- warned, not hidden, so the form's
    # shape does not follow the structure.  The default is the config's own.
    from ..config.pyscf import PySCFConfig
    _p_set = getattr(cfg, "pressure_atm", None)
    _p_default = PySCFConfig.__dataclass_fields__["pressure_atm"].default
    if _frozen_idx and _p_set is not None and float(_p_set) != float(_p_default):
        issues.append(Issue(
            severity="warn",
            message=(
                f"pressure_atm = {float(_p_set):g} atm has no effect here: "
                f"atoms are held, so the thermochemistry is the vibrational "
                f"contributions alone, and a pressure enters only a free "
                f"molecule's gas-phase translation.  The result records no "
                f"pressure (engines/vibration.md § 4.7)."
            ),
            where="config.pressure_atm",
        ))


    # A STRUCTURE THAT REPEATS IS COMPUTED AS A CLUSTER, AND SAID SO -- a note,
    # not a refusal (user, 2026-09-29: "why should it be a refusal? just note
    # that periodicity will not be respected in pySCF").  gto.M builds a
    # molecule in free space, and the script's harmonic analysis removes the
    # motions of that free cluster (it passes isolated kinds to
    # `vibrational_modes`), so the calculation agrees with itself; what it
    # does not do is respect the cell, and the engine's one check,
    # `cell.periodic_in_gas_phase` (`validation/pyscf.py`), says that for every
    # PySCF calculation, a vibration among them.

    # Frozen-atom sanity: every explicit index must be within
    # the structure's atom range.
    if cfg.frozen_indices:
        n = struct.n_atoms if hasattr(struct, "n_atoms") else None
        if n is None:
            # Duck-typed: fall back to len(elements).
            try:
                n = len(struct.elements)
            except Exception:
                n = None
        if n is not None:
            bad = [i for i in cfg.frozen_indices if not 0 <= int(i) < n]
            if bad:
                issues.append(Issue(
                    severity="error",
                    message=(f"the frozen atoms name {atom_words(bad)}, "
                             f"outside this structure's {n} atoms -- its "
                             f"indices are 0 … {n - 1}."),
                    where="structure.regions",
                ))

    # Boundary-condition guards (the three-stage contract): sidecar ->
    # form (cfg) -> script must be explicit, consistent, fully respected.
    # The script render itself emits cfg.frozen_indices verbatim (no
    # silent merge); the set travels with the structure (`web/spectra.md`
    # § 8) and the deck's view lifts it from there.

    # THE FROZEN SET, SAID OUT LOUD (user ruling 2026-08-21: which atoms
    # to freeze is the user's own call -- honored, never second-guessed
    # -- and what the freeze MEANS must be explicit).  INFO, not a warn:
    # nothing is wrong; the reader is told the regime before paying for
    # the run.  The Methods paragraph restates it beside the results.
    frozen_now = sorted(int(i) for i in (cfg.frozen_indices or []))
    if frozen_now:
        # NAMED IN THE USER ORDER -- a viewer's structure may be an engine's
        # (`model/overview.md` § 2.2a, `Structure.atoms_named`).
        named = (struct.atoms_named(frozen_now)
                 if hasattr(struct, "atoms_named") else atom_words(frozen_now))
        issues.append(Issue(
            severity="info",
            message=(
                f"{len(frozen_now)} atom(s) ({named}) are "
                f"frozen: the deck's relaxation, when it runs, holds them "
                f"fixed (geomeTRIC $freeze; under already_relaxed there is "
                f"no relaxation), and the Hessian is taken over the free "
                f"atoms only.  "
                f"Frequencies will be those of the free atoms moving in "
                f"the static field of the fixed ones; thermochemistry "
                f"is vibrational-only."
            ),
            where="structure.regions",
        ))

    # THE STRUCTURE'S OWN EVIDENCE beside the statement (vibration.md § 2.2):
    # the record a finished relaxation left on the pair, judged against
    # this calculation's `geom_gmax` -- in eV/Å, the record's unit -- and
    # its engine (no recorded contract exists for PySCF decks yet, so the
    # level of theory is compared by engine alone).
    from ..constants import HARTREE_BOHR_EV_ANGSTROM_ASE as _EV_ANG_PER_EH_BOHR
    from .sidecar import check_relaxation_record
    _gmax = getattr(cfg, "geom_gmax", None)
    issues.extend(check_relaxation_record(
        struct, engine="pyscf",
        already_relaxed=bool(getattr(cfg, "already_relaxed", False)),
        force_tolerance_ev_ang=(float(_gmax) * _EV_ANG_PER_EH_BOHR
                                if _gmax is not None else None)))


    # --- The solvation matrix (category 2; PROBED live against pyscf
    # 2.13 on 2026-08-21).  PySCF's PCM carries the analytic gradient and
    # the analytic Hessian (RKS and UKS, with and without density
    # fitting), which solves under equilibrium solvation and adds the
    # solvent's own term (`with_solvent.hess`).  The deck's OTHER routes
    # are built without it -- the held-atom Hessian (`hess_elec(atmlst=)`
    # + `hess_nuc`), the analytic IR block and Raman's polarizability loop
    # -- so PCM reaches one route, and the others
    # are refused until built and measured (`engines/vibration.md` § 4.6;
    # plan § 5w K17).  SMD is compiled out of this build; ddCOSMO has no
    # analytic Hessian.
    _solv = str(getattr(cfg, "solvent", "") or "").lower()
    _smethod = str(getattr(cfg, "solvent_method", "") or "").upper()
    if _solv:
        # (The SMD refusal is the ENGINE validator's: a compiled-out
        # module is a build fact, not a vibration fact, and the
        # optimization deck emits the same solvent lines.)
        if "DDCOSMO" in _smethod:
            # NOT bare "COSMO": that is a legal catalogue choice served
            # through pyscf.solvent.pcm (mf.PCM() + with_solvent.method
            # = "COSMO"), whose analytic Hessian this block itself
            # vouches for below; the ddCOSMO rationale is measured on a
            # DIFFERENT module (pyscf.solvent.ddcosmo -- the mf.ddCOSMO()
            # class this deck never constructs).
            issues.append(Issue(
                severity="error",
                message=(
                    "solvent_method = ddCOSMO: pyscf 2.13 provides its "
                    "gradient but NOT its analytic Hessian (probed "
                    "2026-08-21: Hessian raises AttributeError), and a "
                    "vibration IS a Hessian.  Use IEF-PCM / C-PCM, whose "
                    "Hessian is analytic here.  ddCOSMO reference: "
                    "Lipparini et al., J. Chem. Phys. 141, 184108 (2014) "
                    "[Lipparini2014]."),
                where="config.solvent_method",
            ))
        else:
            # Only for a name the dielectric table knows: the engine
            # validator refused unknown names, and a regime note about a
            # run that will not happen would be double-speak.
            from ..pyscf.scf_setup import SOLVENTS as _SOLV_TABLE
            _held = list(getattr(cfg, "frozen_indices", []) or [])
            # (route asked for, what turns it off) -- in the deck's order.
            _asked = (([(f"{len(_held)} held atom(s) -- the held-atom "
                         f"Hessian", "release the held atoms")]
                       if _held else [])
                      + ([("Raman -- its polarizability loop",
                           "turn Raman off")]
                         if bool(getattr(cfg, "compute_raman", False))
                         else [])
                      + ([("IR -- its dipole-derivative route",
                           "turn IR off")]
                         if bool(getattr(cfg, "compute_ir", False))
                         else []))
            _routes = [r for r, _off in _asked]
            if _routes and _solv in _SOLV_TABLE:
                # PCM REACHES ONE ROUTE (ruled 2026-09-29 --
                # `engines/vibration.md` § 4.6).  For a name the
                # dielectric table knows: an unknown one is the engine
                # validator's refusal already.
                issues.append(Issue(
                    severity="error",
                    message=(
                        f"PCM solvation ({_solv}) reaches one route: the "
                        f"frequencies of a structure with no atoms held, "
                        f"with IR and Raman off -- PySCF's full analytic "
                        f"Hessian carries the solvent's response "
                        f"(with_solvent.hess), and the routes this "
                        f"calculation asks for are built without it: "
                        f"{'; '.join(_routes)}.  "
                        f"{_sentence(', '.join(off for _r, off in _asked))}"
                        f" -- or remove the solvent "
                        f"(engines/vibration.md § 4.6)."),
                    where="config.solvent",
                ))
            elif _solv in _SOLV_TABLE:
                issues.append(Issue(
                    severity="info",
                    message=(
                        f"PCM solvation ({_solv}): the relaxation and the "
                        f"Hessian run under one solvated Hamiltonian -- the "
                        f"Hessian adds the solvent's own response "
                        f"(with_solvent.hess).  The numbers use the "
                        f"EQUILIBRIUM-solvation approximation: the continuum "
                        f"relaxes fully at every displaced geometry, so fast "
                        f"non-equilibrium solvent response is not in the "
                        f"line shapes.  Tomasi, Mennucci & Cammi, Chem. Rev. "
                        f"105, 2999 (2005) [Tomasi2005]; Cances, Mennucci & "
                        f"Tomasi, J. Chem. Phys. 107, 3032 (1997) "
                        f"[Cances1997]."),
                    where="config.solvent",
                ))

    elif _smethod and _smethod not in ("IEF-PCM",):
        # A method with no solvent methods nothing -- say so rather
        # than letting the field sit inert (the honesty gate's rule:
        # every shown knob speaks).
        issues.append(Issue(
            severity="warn",
            message=(
                f"solvent_method = {_smethod} is set but `solvent` is "
                f"empty, so no solvation model is applied and the "
                f"method choice does nothing.  Pick a solvent (the PCM "
                f"dielectric table: water, methanol, ethanol, acetone, "
                f"dmso, thf, chloroform, toluene, hexane) or clear the "
                f"method."),
            where="config.solvent_method",
        ))

    # --- symmetry (category 2; probed 2026-08-21) -------------------
    if bool(getattr(cfg, "symmetry", False)) and not bool(
            getattr(cfg, "already_relaxed", False)):
        issues.append(Issue(
            severity="error",
            message=(
                "symmetry = true with in-deck relaxation: a geomeTRIC "
                "step leaves the point group, and PySCF's "
                "re-symmetrization would reorient the frame under the "
                "optimizer (probed: C2v -> Cs on a 0.005 A "
                "displacement).  Symmetry is honored on the "
                "already_relaxed path, where the equilibrium SCF and "
                "Hessian run under the group (PCM included) and the "
                "displaced points force it off.  Set already_relaxed = "
                "true (relax first, without symmetry), or drop "
                "symmetry.  Wilson, Decius & Cross (1955) [Wilson1955] "
                "for the symmetry classification of modes -- irrep "
                "labels in the results are a planned follow-up."),
            where="config.symmetry",
        ))

    # on_nonconvergence policy (pyscf.md § 3): taking a relaxation that
    # did not converge is a legitimate survey-mode choice, but on a
    # VIBRATION run every downstream quantity -- frequencies, intensities,
    # thermochemistry -- is the curvature at the geometry it reached.
    _pol = str(getattr(cfg, "on_nonconvergence", "halt") or "halt").lower()
    if _pol == "proceed":
        issues.append(Issue(
            severity="warn",
            message=(
                "on_nonconvergence = 'proceed': a relaxation that does not "
                "meet geomeTRIC's criteria in geom_max_steps is kept where "
                "it stopped, and frequencies computed on a not-quite-relaxed "
                "geometry commonly show spurious imaginary modes and shifted "
                "band positions.  Legitimate for a survey; for publishable "
                "spectra use 'halt' (or 'continue' with "
                "geom_continue_retries).  When it fires, the result says so "
                "in the relaxation's warning, beside relaxation.converged -- "
                "the judged force against geom_gmax."),
            where="config.on_nonconvergence",
        ))

    return issues


def siesta_vibration_checks(struct: Structure, cfg, *,
                            design: Optional[Structure] = None
                            ) -> List[Issue]:
    """The vibration kind's science on SIESTA -- the force-constant run.

    What holds on both engines is said once here and in
    `spectra_render_checks` by the same words: how many whole-body
    motions survive the freeze, from the one derivation, and the
    structure's own relaxation record read against this calculation.  What
    is this route's own: at least one free atom to nudge, and the two
    states of the person's box -- unticked the run's relaxation phase runs
    first, ticked the force constants are taken at the geometry as given
    and the job's finish measures it (`engines/vibration.md` § 2.2, § 5.8,
    § 5.11).
    """
    issues: List[Issue] = []
    n = int(struct.n_atoms)
    held = sorted(int(i) for i in (struct.frozen_atoms or []))
    bad = [i for i in held if not 0 <= i < n]
    if bad:
        issues.append(Issue(
            severity="error",
            message=(f"the frozen set names {struct.atoms_named(bad)}, outside "
                     f"this structure's {n} atoms -- its indices are "
                     f"0 … {n - 1}."),
            where="structure.regions"))
        return issues
    n_free = n - len(held)
    if n_free == 0:
        issues.append(Issue(
            severity="error",
            message=("every atom is held: a force-constant run needs at "
                     "least one free atom to nudge."),
            where="structure.regions"))
        return issues
    if held:
        from ..template import engine_name
        _f, n_rigid, n_modes = modes_of(struct, engine_name(type(cfg)))
        if n_rigid:
            issues.append(Issue(
                severity="info",
                message=(
                    f"Holding {len(held)} atom(s) leaves {n_rigid} "
                    f"whole-body motion(s) of the free atoms that cost no "
                    f"energy (a turn about the held atoms).  They are not "
                    f"vibrations: they are removed before diagonalising, "
                    f"so {n_modes} modes will be reported, "
                    f"not {3 * n_free}, and the count removed is recorded "
                    f"with the results."),
                where="structure.regions"))
        issues.append(Issue(
            severity="info",
            message=(
                f"{len(held)} atom(s), {struct.atoms_named(held)}, are held: "
                f"they sit "
                f"in Geometry.Constraints outside the FC.First..FC.Last "
                f"range, so no force constant is taken with respect to "
                f"them and the run nudges the {n_free} free atom(s) only "
                f"(6 force evaluations per free atom, each a whole-system "
                f"SCF).  Frequencies are those of the free atoms moving in "
                f"the static field of the held ones; thermochemistry is "
                f"vibrational-only.  With Compute IR on, every atom's "
                f"Hirshfeld and Voronoi charge is recorded at every nudge, "
                f"so the spectrum viewer gives the free atoms' infrared; "
                f"with Compute Raman on, Raman comes from the "
                f"optical calls at the modes Raman for picks."),
            where="structure.regions"))
    # THE RELAXATION IS THE PERSON'S EXPLICIT CHOICE (`engines/vibration.md`
    # § 2.2, § 5.8), made with one box.  Unticked, the run relaxes first --
    # its relaxation call before the force constants, to this template's own
    # tolerance (§ 5.11) -- and the finding says so.  Ticked, nothing relaxes, and the
    # finding is a WARNING in plain words: off a stationary point the
    # frequencies will be off; the job's finish measures the reference-step
    # forces against the same tolerance and says whether the statement held.
    # Never a refusal: the statement is the person's to make.
    _tol = getattr(cfg, "relax_force_tol", None)
    _tol_text = (f"{float(_tol):g} eV/Å" if _tol is not None
                 else "the template's relax_force_tol")
    if not bool(getattr(cfg, "already_relaxed", False)):
        issues.append(Issue(
            severity="info",
            message=(
                f"The structure is not stated to be relaxed, so the run "
                f"relaxes it first: its relaxation phase runs before the "
                f"force constants, to this calculation's own force "
                f"tolerance ({_tol_text}, the relaxation settings on this "
                f"form), and the force constants are taken at the relaxed "
                f"geometry (engines/vibration.md 5.11)."),
            where="config.already_relaxed"))
    else:
        issues.append(Issue(
            severity="warn",
            message=(
                f"You stated the structure is already relaxed at this level "
                f"of theory, so nothing here relaxes it: the force constants "
                f"are taken at the geometry as given.  If it is not relaxed "
                f"at this level of theory -- another code, another basis, a "
                f"looser tolerance -- the frequencies will be off, the low "
                f"ones most.  The job measures the forces SIESTA "
                f"evaluates at that geometry (its FC step 0) against this "
                f"calculation's force tolerance ({_tol_text}) and reports "
                f"the verdict with the modes."),
            where="config.already_relaxed"))
    # THE STRUCTURE'S OWN EVIDENCE beside the statement (vibration.md § 2.2):
    # the record a finished relaxation left on the pair, judged against
    # THIS calculation's tolerance and level of theory.
    from ..chemistry import every_label_resolves
    from ..electronic_state import electronic_state
    from ..parse.contract import contract_fields_of
    from .sidecar import check_relaxation_record
    # The electronic state is part of the level of theory the record is
    # compared against, RESOLVED (ES7): a frequency at a geometry relaxed
    # for another charge or spin is usually a mistake.
    _state = (electronic_state(struct, cfg, kind="vibration")
              if every_label_resolves(struct) else None)
    # Against the FILE's coordinates (`design`), not the placed subject: a
    # placement is a rigid shift the record never saw (`validate`, ``design``).
    issues.extend(check_relaxation_record(
        design if design is not None else struct, engine="siesta",
        already_relaxed=bool(getattr(cfg, "already_relaxed", False)),
        force_tolerance_ev_ang=(float(_tol) if _tol is not None else None),
        level=contract_fields_of(cfg, state=_state,
                                 axis_kind=struct.axis_kind,
                                 kind="vibration")))
    # RAMAN FOR (§ 3.1): the range's ends, when one is asked.
    issues.extend(raman_issues(cfg, engine="siesta"))
    return issues


def selector_issues(cfg, *, none_means: str) -> List[Issue]:
    """THE PROBE'S MODE SELECTOR'S CHECKS, PySCF's (`engines/vibration.md`
    § 4.8).  The explicit list is read by its one reader (`selection.
    parse_index_list`, 1-based) when the selector is ``explicit`` -- the
    catalogue's "only read when selector = explicit": text it cannot read
    is refused here, before a deck is written from it; an empty list is
    said now, not after the wall time -- ``none_means`` what the run then
    does not compute.  A window outside ``all`` is the generic check's,
    from the items' ``active_when`` (`metadata._check_inactive`)."""
    issues: List[Issue] = []
    selector = str(getattr(cfg, "es_mode_selection", ""))
    if selector == "all":
        # THE WINDOW'S TWO ENDS, stated against each other: refused, as
        # Raman's range is (:func:`raman_issues`).
        lo = getattr(cfg, "freq_min_cm1", None)
        hi = getattr(cfg, "freq_max_cm1", None)
        if lo is not None and hi is not None and float(lo) > float(hi):
            issues.append(Issue(
                severity="error",
                message=(f"Probe modes from, {float(lo):g} cm⁻¹, is above "
                         f"Probe modes up to, {float(hi):g} cm⁻¹: the "
                         f"window's two ends conflict -- swap them, or "
                         f"clear one."),
                where="config.freq_min_cm1"))
        return issues
    if selector != "explicit":
        return issues
    from ..spectra.selection import parse_index_list
    text = str(getattr(cfg, "es_explicit_indices", "") or "")
    try:
        listed = parse_index_list(text, first=1)
    except ValueError as e:
        return issues + [Issue(
            severity="error",
            message=(
                f"The explicit mode list {text!r} can't be read: {e}.  "
                f"Write 1-based mode numbers separated by commas, with "
                f"ranges if you like: \"3, 7, 12\" or \"3-7, 12\"."),
            where="config.es_explicit_indices")]
    if not listed:
        issues.append(Issue(
            severity="warn",
            message=(
                f"Mode selection is set to \"explicit\" but no mode "
                f"indices were entered.  {none_means}  Either add at least "
                f"one mode index, or switch the selector to \"skip\" or "
                f"\"all\"."),
            where="config.es_explicit_indices"))
    return issues


def raman_issues(cfg, *, engine: str) -> List[Issue]:
    """RAMAN FOR's checks, on both engines (`engines/vibration.md` § 3.1):
    with Compute Raman on and a frequency range asked, a range with neither
    end is refused -- a range stated nowhere -- and so is one whose minimum
    is above its maximum, two ends that conflict.  On PySCF, whose sweep
    gives every mode at once, a range is said to narrow nothing."""
    if not bool(getattr(cfg, "compute_raman", False)):
        return []
    if str(getattr(cfg, "raman_modes", "all")) != "range":
        return []
    lo = getattr(cfg, "raman_freq_min_cm1", None)
    hi = getattr(cfg, "raman_freq_max_cm1", None)
    from ..template import catalogue, choice_words, one
    modes = one(catalogue(), "raman_modes")
    if lo is None and hi is None:
        return [Issue(
            severity="error",
            message=(f"Raman for is {choice_words(modes, 'range')} and "
                     f"neither Raman from nor Raman up to is given: give "
                     f"one or both ends of the range in cm⁻¹, or choose "
                     f"{choice_words(modes, 'all')} for every mode."),
            where="config.raman_modes")]
    if lo is not None and hi is not None and float(lo) > float(hi):
        return [Issue(
            severity="error",
            message=(f"Raman from, {float(lo):g} cm⁻¹, is above Raman up "
                     f"to, {float(hi):g} cm⁻¹: the range's two ends "
                     f"conflict -- swap them, or clear one."),
            where="config.raman_freq_min_cm1")]
    if engine == "pyscf":
        return [Issue(
            severity="info",
            message=("PySCF computes Raman for every mode at once, from its "
                     "polarizability sweep, so the frequency range narrows "
                     "nothing here; on SIESTA it would pick the modes "
                     "(engines/vibration.md § 3.1)."),
            where="config.raman_modes")]
    return []


def modes_of(struct: Structure, engine: str):
    """``(n_free, n_rigid, n_modes)`` of ``struct`` for ``engine``'s run --
    its held atoms the structure's one read of them
    (`Structure.frozen_atoms`), the whole-body motions on the axes the
    engine computes on (`cell.engine_axis_kinds`), the count the rank rule's
    (`normal_modes.mode_count`, `science/normal-modes.md` § 3.1): the one
    count prep's checks, its size line and the Methods text state.  An index
    outside the structure is a :class:`ValueError` (`rigid_motions`)."""
    from ..cell import engine_axis_kinds
    from ..spectra.normal_modes import mode_count
    return mode_count(struct.positions,
                      sorted(int(i) for i in (struct.frozen_atoms or [])),
                      engine_axis_kinds(engine, struct),
                      cell=getattr(struct, "cell", None))


# The render-time checks are the module's public door.
__all__ = ["spectra_render_checks", "siesta_vibration_checks",
           "selector_issues", "raman_issues", "modes_of"]
