"""The spectra render gate -- the science checks a vibration deck runs.

One module, one body, two callers (see ``spectra_render_checks``'s
docstring).  Lives in ``validation/`` beside the other engines' gates;
the retired ``spectra/pyscf_engine.py`` carried it as a classmethod.
"""

from __future__ import annotations

from typing import List

from ..issues import Issue
from ..structure import Structure


def spectra_render_checks(struct: Structure,
                          cfg) -> List[Issue]:
    """PySCF SCIENTIFIC advisories — THE render gate for a vibration
    deck (grid / amplitude / parity / method / open-shell).

    MOVED at P3 (2026-08-21) from the retired
    ``PySCFSpectraEngine.render_checks`` classmethod, unchanged in
    substance.  Two callers, one body: ``validation/__init__.py``'s
    vibration arm and ``pyscf/vibration_deck.py`` directly -- the
    deck's config view is an ADAPTER over PySCFConfig, so the
    type-keyed registry cannot see it (which is exactly how this gate
    silently skipped between P1 and P3; the direct call closes that).
    ``cfg`` is duck-typed to the spectra vocabulary for the same
    reason.  (The third caller this docstring used to name,
    ``_validate_spectra``, was the type-keyed SpectraConfig
    validator; it retired with the class on 2026-08-22.)  No selector-availability checks; those were
    preflight-only UX and retired with the preflight route."""
    issues: List[Issue] = []

    # -- PySCF-specific scientific advisories ------------------

    # Grid level with a hybrid functional (spec § 11.4).  PySCF's
    # default grid level is 3 ("screening"); level 4 is the
    # production minimum for hybrids.  Below that the XC numerical
    # integration noise dominates the Hessian and gives garbage
    # frequencies (~5-20 cm⁻¹ wander).  ONE shared gate/body with the
    # Build-tab PySCF validator (was a duplicated rule; V4) -- the
    # "spectra" context selects the frequency-error rationale.
    from .pyscf import check_dft_grid_level
    issues.extend(check_dft_grid_level(cfg, context="spectra"))

    # Displacement amplitude.  The [0.02, 0.20] Å acceptance
    # range is empirical -- not derived from a single source.
    # The lower bound was relaxed from 0.04 to 0.02 on 2026-
    # 05-19 to match the SpectraConfig default; 0.02 Å keeps
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

    # ---- Electron-count parity (THE standard pre-SCF check) ----
    # PySCF's ``spin`` = 2S = n_unpaired = n_alpha - n_beta.  Its
    # parity must match the total electron count
    # (Σ Z - charge).  Catching this at preflight gives a clearer
    # error than PySCF's runtime "Mol.nelectron is odd, but spin=0".
    from ..chemistry import (check_spin_charge_parity,
                              detect_open_shell_metals,
                              explain_metal_spin)
    # A LABEL NAMING NO ELEMENT DOES NOT ARRIVE HERE AS AN EXCEPTION.
    # This used to catch `KeyError` and report the raw symbol itself;
    # `check_spin_charge_parity` now stands down on an unresolvable label
    # and returns None (`chemistry.every_label_resolves`), because the
    # finding belongs to `check_species_labels`, which names the engine
    # and says what to do.  Two reporters for one fact is what that move
    # ended, so the catch went with it rather than sitting unreachable.
    parity_err = check_spin_charge_parity(struct, cfg.charge, cfg.spin)
    if parity_err:
        # Severity is the explicit-vs-guessed rule, shared with the
        # engine validator (G-1d) and mirroring _validate_siesta's: a
        # mismatch is an ERROR whenever either number is the user's own
        # claim -- a nonzero spin is always explicit (the default is 0),
        # and a set net_charge is explicit.  Only the BOTH-GUESSES case
        # (default spin 0, auto-detected charge) stays a WARN nudging
        # toward an explicit charge: the phosphate heuristic sees only
        # phosphates, so the guess may be what is wrong, not the
        # physics, and refusing on two guesses would block a legitimate
        # run.  A config with no `net_charge` field types its charge
        # directly -- explicit.
        _raw = getattr(cfg, "net_charge", "typed-directly")
        if _raw is None and int(getattr(cfg, "spin", 0) or 0) == 0:
            issues.append(Issue(
                severity="warn",
                message=(parity_err
                         + "  The charge here came from auto-detection "
                           "(phosphates only); if it missed something, "
                           "set net_charge explicitly."),
                where="config.charge",
            ))
        else:
            issues.append(Issue(
                severity="error",
                message=parity_err,
                where="config.charge",
            ))

    # ---- Open-shell metal sanity check ----
    # Delegated to the shared validator so the Spectra preflight,
    # the SIESTA/PySCF Build-tab preflights, and the form's
    # detection chip all read from the same source of truth
    # (``ChemistryAnalysis.suggested_treatment``).  The pre-2026-06-13
    # Au-BDT-Au incident was caused by a parallel ``metals``-only
    # check in this very block — see docs/web/ui-contract.md
    # Rule 1.  ``metals`` (the flat detection list) is still computed
    # so the supplemental ``explain_metal_spin`` info-line below
    # can echo (element, spin) → (likely oxidation state) for
    # non-spin=0 cases.
    from . import check_open_shell_metal
    method_upper = cfg.method.upper()
    is_closed_shell = (cfg.spin == 0
                       and method_upper in ("RKS", "RHF"))
    issues.extend(check_open_shell_metal(
        struct,
        is_closed_shell=is_closed_shell,
        engine_label=f"PySCF spectra ({cfg.method})",
    ))
    metals = detect_open_shell_metals(struct)
    if metals and not is_closed_shell:
        # Metal present + the user DID pick a non-default spin.
        # Echo back what their (element, spin) implies so they can
        # sanity-check the oxidation state.  Severity=info so it
        # doesn't add to the warn/error count; it just labels.
        for m in metals:
            hint = explain_metal_spin(m, cfg.spin)
            if hint:
                issues.append(Issue(
                    severity="info",
                    message=(
                        f"{m} + spin={cfg.spin}: {hint}.  "
                        f"Confirm against your experimental data "
                        f"(Mössbauer / UV-Vis / EPR) or the "
                        f"chemistry of the rest of the molecule "
                        f"(porphyrin protonation, axial ligands)."
                    ),
                    where="config.spin",
                ))

    # Method / spin / functional compatibility.
    method = cfg.method.upper()
    if method not in ("RKS", "UKS", "RHF", "UHF"):
        issues.append(Issue(
            severity="error",
            message=(f"SCF method '{cfg.method}' isn't supported "
                     f"here.  Pick one of RKS (closed-shell DFT, "
                     f"the usual default), UKS (open-shell DFT), "
                     f"RHF (closed-shell Hartree-Fock), or UHF "
                     f"(open-shell Hartree-Fock)."),
            where="config.method",
        ))
    # Selector-by-Raman scientific caveat.  top_n / threshold
    # rank vibrational modes by Raman activity, but Raman
    # brightness is NOT the same as electron-phonon coupling
    # strength.  Transport-critical modes can be Raman-weak and
    # would be silently dropped.  This is a SCIENTIFIC caveat
    # (the user picked a valid option) -- warn, don't error.
    if cfg.es_mode_selection in ("top_n", "threshold"):
        issues.append(Issue(
            severity="warn",
            message=(
                f"Mode selection \"{cfg.es_mode_selection}\" ranks "
                f"vibrational modes by Raman activity, but Raman "
                f"brightness is NOT the same as electron-phonon "
                f"coupling strength.  A mode that's important for "
                f"transport (or any IETS/inelastic application) "
                f"can be Raman-weak and would be silently skipped "
                f"by this selector.  For transport-preparation "
                f"runs, look at the spectrum first with mode "
                f"selection = \"skip\", then re-run with "
                f"\"explicit\" listing the modes you care about, "
                f"or use \"all\" if cost allows.  Cf. "
                f"[Galperin2007]."
            ),
            where="config.es_mode_selection",
        ))

    # Empty explicit list: the run will produce no orbital-energy
    # data even though the user asked for it.  Warn so they don't
    # waste wall time discovering this after the run.
    if (cfg.es_mode_selection == "explicit"
            and not cfg.es_explicit_indices):
        issues.append(Issue(
            severity="warn",
            message=(
                "Mode selection is set to \"explicit\" but no "
                "mode indices were entered.  No per-mode "
                "orbital-energy data will be computed.  Either "
                "add at least one mode index, or switch the "
                "selector to \"skip\", \"all\", \"top_n\", or "
                "\"threshold\"."
            ),
            where="config.es_explicit_indices",
        ))

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
        # FROZEN ATOMS ARE INDICES.  This subtracted an element-match
        # count as well, which was always zero: the config this receives
        # is the deck's view, and it supplies indices only.  The
        # element/residue vocabulary went with `SpectraConfig`
        # (2026-08-22) -- the region store holds indices, and that is
        # what the deck writes into the constraints file.
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

    # WHAT SURVIVES THE FREEZE, from the one derivation (science/normal-modes.md
    # R1, R7).  Holding atoms removes the whole-body motions that would move
    # them and leaves the rest -- a turn about the line through two held
    # atoms, three turns about a single held atom -- and those leftovers are
    # not vibrations.  The deck projects them out before diagonalising, so
    # nothing here is a warning: it is the count the person will see missing
    # from 3 N_free, said before the run is paid for.  A table of cases stood
    # here ("6 - 2 x frozen ... -ish") and was wrong for CO2 with both O held.
    _frozen_idx = sorted(int(i) for i in (cfg.frozen_indices or []))
    _positions = getattr(struct, "positions", None)
    if _frozen_idx and _positions is not None:
        from ..spectra.normal_modes import rigid_motions
        try:
            _n_rigid = len(rigid_motions(
                _positions, _frozen_idx,
                getattr(struct, "axis_kind", None) or ("isolated",) * 3,
                cell=getattr(struct, "cell", None)))
        except ValueError:
            _n_rigid = None      # an index out of range is reported below
        if _n_rigid:
            _n_free = int(len(_positions)) - len(_frozen_idx)
            issues.append(Issue(
                severity="info",
                message=(
                    f"Holding {len(_frozen_idx)} atom(s) leaves {_n_rigid} "
                    f"whole-body motion(s) of the free atoms that cost no "
                    f"energy (a turn about the held atoms).  They are not "
                    f"vibrations: the deck removes them before diagonalising, "
                    f"so {3 * _n_free - _n_rigid} modes will be reported, not "
                    f"{3 * _n_free}, and the count removed is recorded with "
                    f"the results."
                ),
                where="structure.regions",
            ))

    # THIS ENGINE COMPUTES A MOLECULE IN FREE SPACE.  gto.M has no lattice, so
    # a structure that repeats or continues along an axis would be computed
    # as a cluster in silence -- and the harmonic analysis would then remove
    # the six motions of a free molecule where the structure's own periodicity
    # says three (science/normal-modes.md 3.1a).  Refused, naming the axis and
    # the two honest ways out.
    _kinds = tuple(getattr(struct, "axis_kind", None) or ())
    _repeating = [f"{'xyz'[i]} ({k})" for i, k in enumerate(_kinds)
                  if k != "isolated"]
    if _repeating:
        issues.append(Issue(
            severity="error",
            message=(
                f"this structure states a repeating or continuing axis "
                f"({', '.join(_repeating)}), and a PySCF vibration computes a "
                f"molecule in free space: it would be run as a cluster while "
                f"the description says periodic.  If a cluster is what you "
                f"mean, make every axis isolated on the Cell page; a "
                f"vibration of the periodic system belongs to the SIESTA "
                f"engine (science/normal-modes.md 3.1a)."
            ),
            where="structure.axis_kind",
        ))

    # GPU advisory: if the user asked for GPU acceleration, check
    # (1) whether gpu4pyscf is importable on the molbuilder host
    # and (2) whether the host has a GPU that actually meets
    # gpu4pyscf's minimum compute capability (7.0 = Volta).  The
    # generated script STOPS on a missing gpu4pyscf (no CPU
    # fallback, 2026-08-17); checking here lets the user fix things
    # before the run rather than after.
    if cfg.use_gpu:
        issues.extend(_gpu_capability_advisories())

    # compute_ir advisory RETIRED 2026-08-21 -- it warned that IR
    # was "not implemented", which P1 falsified (the deck computes IR
    # via dipole derivatives, band-level validated against literature
    # water intensities; archive/2026-09-01-roadmap.md § 5 records the closure).  Found
    # by the honesty gate's render probe: a validator claiming a
    # capability is absent is the same drift as a diagram drawing a
    # file that is gone.

    # Method / spin consistency (ports the Build-tab guards from
    # validation.pyscf; the old "cfg doesn't carry spin yet" note here
    # was STALE -- SpectraConfig HAS a `spin` field, so RKS/RHF + spin>0
    # and open-shell RKS on an odd-electron system used to pass the render
    # gate unflagged).  A restricted method (RKS/RHF) forces nα=nβ ⇒ 2S=0,
    # so a non-zero spin with it is a contradiction.
    method_u = cfg.method.upper()
    # (The restricted-method-with-spin refusal and the radical arm are
    # the ENGINE validator's, unconditional for both kinds -- G-1c's
    # error-level finding and the parity block's folded advice.  Copies
    # of both stood here and double-fired on every vibration deck, one
    # of them at a softer severity than the refusal it duplicated;
    # deleted with the U6 close, 2026-08-22.)

    # Frozen-atom sanity: every explicit index must be within
    # the structure's atom range.  Element / residue rules are
    # checked at script-render time when we have the full
    # frozen mask.
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
                    message=(f"\"Freeze by atom index\" contains "
                             f"out-of-range numbers {bad}.  This "
                             f"structure has {n} atoms; valid "
                             f"indices are 0..{n - 1} (counting "
                             f"from zero)."),
                    where="structure.regions",
                ))

    # Boundary-condition guards (design.md "Sidecar-driven
    # boundary conditions — the three-stage contract"):
    #
    # The contract:  sidecar -> form (cfg) -> script must be
    # explicit, consistent, fully respected.  The script render
    # itself emits cfg.frozen_indices verbatim (no silent merge);
    # these two preflight checks make divergence + unconsumed
    # labels visible so nothing is silently absorbed.

    # (Pattern A -- the sidecar-vs-form divergence warn -- retired
    # 2026-08-21 with the frozen-atoms ruling.  Its premise was the old
    # form field: a SECOND copy of the frozen set that could disagree
    # with the structure's.  Since P2 the set travels with the structure
    # (`web/spectra.md` § 8) and the deck's view lifts it from there, so
    # the comparison had become the sidecar against itself -- a check
    # that could never fire.)

    # THE FROZEN SET, SAID OUT LOUD (user ruling 2026-08-21: which atoms
    # to freeze is the user's own call -- honored, never second-guessed
    # -- and what the freeze MEANS must be explicit).  INFO, not a warn:
    # nothing is wrong; the reader is told the regime before paying for
    # the run.  The Methods paragraph restates it beside the results.
    frozen_now = sorted(int(i) for i in (cfg.frozen_indices or []))
    if frozen_now:
        issues.append(Issue(
            severity="info",
            message=(
                f"{len(frozen_now)} atom(s) (indices {frozen_now}) are "
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

    # Pattern B -- THE one home (validation/sidecar.py, U5): region
    # labels this run does not consume are named; the reserved frozen
    # label is excluded (the relaxation constrains it, the Hessian mask
    # reads it).  A hand-written copy of the same rule stood here and
    # had already diverged once (E-M7.1's false alarm was fixed in only
    # one of the two).
    from .sidecar import check_unconsumed_region_labels
    issues.extend(check_unconsumed_region_labels(
        struct, engine="PySCF vibration"))


    # --- The solvation matrix (category 2; PROBED live against pyscf
    # 2.13 on 2026-08-21 -- every verdict below is a measured fact,
    # not a recalled one).  PCM carries the FULL derivative chain a
    # vibration needs: analytic gradient, analytic Hessian (RKS and
    # UKS, with and without density fitting), and the CPHF
    # polarizability WITH the solvent in the response.  SMD is
    # compiled out of this build; ddCOSMO has no analytic Hessian.
    _solv = str(getattr(cfg, "solvent", "") or "").lower()
    _smethod = str(getattr(cfg, "solvent_method", "") or "").upper()
    if _solv:
        # (The SMD refusal is the ENGINE validator's since the U6 close:
        # a compiled-out module is a build fact, not a vibration fact,
        # and the optimization deck emits the same solvent lines.)
        if "DDCOSMO" in _smethod:
            # NOT bare "COSMO": that is a legal catalogue choice served
            # through pyscf.solvent.pcm (mf.PCM() + with_solvent.method
            # = "COSMO"), whose analytic Hessian this block itself
            # vouches for below.  Matching it here refused a legal
            # dropdown value with a rationale measured on a DIFFERENT
            # module (pyscf.solvent.ddcosmo -- the mf.ddCOSMO() class
            # this deck never constructs).
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
            if _solv in _SOLV_TABLE:
                issues.append(Issue(
                    severity="info",
                    message=(
                        f"PCM solvation ({_solv}): relaxation, Hessian, IR "
                        f"and Raman all run under the SAME solvated "
                        f"Hamiltonian -- consistent by construction (the "
                        f"polarizability response includes the solvent; "
                        f"measured).  The numbers use the EQUILIBRIUM-"
                        f"solvation approximation: the continuum relaxes "
                        f"fully at every displaced geometry, so fast "
                        f"non-equilibrium solvent response is not in the "
                        f"line shapes.  Tomasi, Mennucci & Cammi, Chem. Rev. "
                        f"105, 2999 (2005) [Tomasi2005]; Cances, Mennucci & "
                        f"Tomasi, J. Chem. Phys. 107, 3032 (1997) "
                        f"[Cances1997]."),
                    where="config.solvent",
                ))
        if bool(getattr(cfg, "use_gpu", False)):
            issues.append(Issue(
                severity="error",
                message=(
                    "solvent + use_gpu: the solvated derivative chain has "
                    "been validated on the CPU path only (2026-08-21); "
                    "gpu4pyscf's PCM has not been probed with this deck's "
                    "to_gpu promotion.  Run this calculation on CPU, or "
                    "ask for the GPU+PCM validation to be scheduled."),
                where="config.use_gpu",
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

    # optimizer: geomeTRIC by REFUSAL, and the refusal is informed
    # (the user's 2026-08-21 ruling: an educated suggestion, with the
    # source).  Probed against the run environment on 2026-08-21:
    # pyberny is NOT installed in molbuilder-pySCF, and berny_solver's
    # optimize() offers no step callback -- the tracked relaxation
    # phase (n_steps / max-force ticking in the viewer) is fed by
    # geomeTRIC's callback, so berny would run blind even if present.
    # PySCF manual: pyscf.geomopt (geometric_solver vs berny_solver).
    _optzr = str(getattr(cfg, "optimizer", "geometric") or "geometric").lower()
    if _optzr != "geometric":
        issues.append(Issue(
            severity="error",
            message=(
                f"optimizer = {_optzr!r}: the vibration calculation "
                f"relaxes with geomeTRIC only.  pyberny is not installed "
                f"in the run environment (probed 2026-08-21), and its "
                f"solver exposes no per-step callback, so the tracked "
                f"relaxation phase the Results tab shows would run "
                f"blind.  Set optimizer = 'geometric' (the geom_* "
                f"criteria map to its convergence_* keywords; see the "
                f"PySCF manual, pyscf.geomopt)."),
            where="config.optimizer",
        ))

    # on_nonconvergence policy (pyscf.md § 7a's role table): warning
    # past a failed equilibrium SCF is a legitimate survey-mode choice,
    # but on a VIBRATION run every downstream quantity -- frequencies,
    # intensities, thermochemistry -- inherits the unconverged density.
    _pol = str(getattr(cfg, "on_nonconvergence", "halt") or "halt").lower()
    if _pol == "proceed":
        issues.append(Issue(
            severity="warn",
            message=(
                "on_nonconvergence = 'proceed': the relaxation's convergence "
                "will not be asserted, and frequencies computed on a "
                "not-quite-relaxed geometry commonly show spurious imaginary "
                "modes and shifted band positions.  Legitimate for a survey; "
                "for publishable spectra use 'halt' (or 'continue' with "
                "geom_continue_retries).  The artifact records "
                "relaxation.converged = null with a warning when this "
                "policy fires."),
            where="config.on_nonconvergence",
        ))

    return issues


def siesta_vibration_checks(struct: Structure, cfg) -> List[Issue]:
    """The vibration kind's science on SIESTA -- the force-constant run.

    What holds on both engines is said once here and in
    `spectra_render_checks` by the same words: how many whole-body
    motions survive the freeze, from the one derivation.  What is this
    route's own: at least one free atom to nudge, and that nothing is
    relaxed here -- the input geometry is the stationary point, so a
    person cites a relaxed structure or accepts § 4's consequence.
    """
    issues: List[Issue] = []
    n = int(struct.n_atoms)
    held = sorted(int(i) for i in (struct.frozen_atoms or []))
    bad = [i for i in held if not 0 <= i < n]
    if bad:
        issues.append(Issue(
            severity="error",
            message=(f"the frozen set names atom index(es) {bad}, outside "
                     f"this structure's {n} atoms (0-based)."),
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
        from ..spectra.normal_modes import rigid_motions
        n_rigid = len(rigid_motions(
            struct.positions, held,
            getattr(struct, "axis_kind", None) or ("isolated",) * 3,
            cell=getattr(struct, "cell", None)))
        if n_rigid:
            issues.append(Issue(
                severity="info",
                message=(
                    f"Holding {len(held)} atom(s) leaves {n_rigid} "
                    f"whole-body motion(s) of the free atoms that cost no "
                    f"energy (a turn about the held atoms).  They are not "
                    f"vibrations: they are removed before diagonalising, "
                    f"so {3 * n_free - n_rigid} modes will be reported, "
                    f"not {3 * n_free}, and the count removed is recorded "
                    f"with the results."),
                where="structure.regions"))
        issues.append(Issue(
            severity="info",
            message=(
                f"{len(held)} atom(s) (indices {held}) are held: they sit "
                f"in Geometry.Constraints outside the FC.First..FC.Last "
                f"range, so no force constant is taken with respect to "
                f"them and the run nudges the {n_free} free atom(s) only "
                f"(6 force evaluations per free atom, each a whole-system "
                f"SCF).  Frequencies are those of the free atoms moving in "
                f"the static field of the held ones; thermochemistry is "
                f"vibrational-only.  Intensities are not computed on this "
                f"route."),
            where="structure.regions"))
    # THE RELAXATION IS THE PERSON'S EXPLICIT CHOICE (`engines/vibration.md`
    # § 2.2, § 5.8), made with one box.  Unticked, the ladder relaxes first
    # -- a `relax` stage before the force constants, to this template's own
    # tolerance -- and the finding says so.  Ticked, nothing relaxes, and the
    # finding is a WARNING in plain words: off a stationary point the
    # frequencies will be off; the read-back measures the reference-step
    # forces against the same tolerance and says whether the statement held.
    # Never a refusal: the statement is the person's to make.
    _tol = getattr(cfg, "relax_force_tol", None)
    _tol_text = (f"{float(_tol):g} eV/Å" if _tol is not None
                 else "the template's relax_force_tol")
    if not bool(getattr(cfg, "already_relaxed", False)):
        issues.append(Issue(
            severity="info",
            message=(
                f"The structure is not stated to be relaxed, so the ladder "
                f"relaxes it first: a `relax` stage runs before the "
                f"force-constant stage, to this calculation's own force "
                f"tolerance ({_tol_text}, the relaxation settings on this "
                f"form), and the force constants are taken at the relaxed "
                f"geometry."),
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
                f"ones most.  The read-back measures the forces SIESTA "
                f"evaluates at that geometry (its FC step 0) against this "
                f"calculation's force tolerance ({_tol_text}) and reports "
                f"the verdict with the modes."),
            where="config.already_relaxed"))
    # THE STRUCTURE'S OWN EVIDENCE beside the statement (vibration.md § 2.2):
    # the record a finished relaxation left on the pair, judged against
    # THIS calculation's tolerance and level of theory.
    from ..parse.contract import contract_fields_of
    from .sidecar import check_relaxation_record
    issues.extend(check_relaxation_record(
        struct, engine="siesta",
        already_relaxed=bool(getattr(cfg, "already_relaxed", False)),
        force_tolerance_ev_ang=(float(_tol) if _tol is not None else None),
        level=contract_fields_of(cfg)))
    from .sidecar import check_unconsumed_region_labels
    issues.extend(check_unconsumed_region_labels(
        struct, engine="SIESTA vibration"))
    return issues


# The GPU advisory helper -- RECOVERED 2026-08-21.  The P3 move took
# render_checks' body but left this classmethod behind in the deleted
# engine file; the surviving `cls._gpu_capability_advisories()` call was
# a latent NameError guarded by use_gpu -- proven crashing by the probe
# now pinned in tests/test_vibration_render_gate.py.  The compute-
# capability minimum comes from its one home, runtime_info.
def _gpu_capability_advisories() -> List[Issue]:
    """Return [] if gpu4pyscf + a supported GPU are available,
    else one warn-severity Issue describing what's missing.
    Always WARN (not ERROR) -- the generated script falls back
    to CPU automatically, so an unusable GPU is annoying but
    not fatal.

    **That is no longer true** (user, 2026-08-17; `engines/overview.md`
    § 3a G-5): the script STOPS when the GPU is missing.  These stay
    ``warn`` rather than ``error`` for one reason -- this preflight runs
    on the SERVER, and the job may run somewhere else.  A missing GPU
    here is evidence, not a verdict.  What changed is the MESSAGE: it no
    longer promises a fallback that was removed.
    """
    from ..runtime_info import GPU4PYSCF_MIN_COMPUTE_CAPABILITY as _MIN_CC
    try:
        import gpu4pyscf  # noqa: F401
    except ImportError:
        return [Issue(
            severity="warn",
            message=(
                "GPU acceleration requested, but gpu4pyscf is "
                "not installed on this server.  There is NO CPU "
                "fallback: if the machine that runs this job "
                "also lacks it, the run will STOP.  To get the "
                "GPU speed-up: "
                "pip install gpu4pyscf-cuda12x  (or cuda11x for "
                "older drivers).  Requires an NVIDIA GPU."
            ),
            where="config.use_gpu",
        )]

    # gpu4pyscf is installed; probe the actual device via cupy.
    try:
        import cupy
    except ImportError:
        return [Issue(
            severity="warn",
            message=(
                "GPU acceleration requested -- gpu4pyscf is "
                "installed, but cupy (its required dependency) "
                "isn't.  Reinstall gpu4pyscf.  There is no CPU "
                "fallback -- the run stops if cupy is missing "
                "where it executes."
            ),
            where="config.use_gpu",
        )]

    # Count devices.  This will fail if the CUDA runtime isn't
    # accessible (driver mismatch, no GPU present, etc.).
    try:
        n_devs = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return [Issue(
            severity="warn",
            message=(
                f"GPU acceleration requested but the CUDA "
                f"runtime couldn't enumerate devices "
                f"({type(exc).__name__}: {exc}).  Check that "
                f"the NVIDIA driver is installed and the CUDA "
                f"toolkit version matches gpu4pyscf's build.  "
                f"There is no CPU fallback -- the run stops "
                f"where this cannot be resolved."
            ),
            where="config.use_gpu",
        )]
    if n_devs == 0:
        return [Issue(
            severity="warn",
            message=(
                "GPU acceleration requested but no NVIDIA GPU "
                "was detected on this host.  There is no CPU "
                "fallback: if the machine that runs this job "
                "has none either, the run STOPS.  Untick "
                "\"Use GPU\" to run on the CPU deliberately."
            ),
            where="config.use_gpu",
        )]

    # Inspect device 0's compute capability.
    try:
        props = cupy.cuda.runtime.getDeviceProperties(0)
    except Exception as exc:
        return [Issue(
            severity="warn",
            message=(
                f"GPU acceleration requested but the device "
                f"properties for GPU 0 couldn't be read "
                f"({type(exc).__name__}: {exc}).  There is no "
                f"CPU fallback -- the run stops."
            ),
            where="config.use_gpu",
        )]
    name = props.get("name", "(unknown GPU)")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    major = int(props.get("major", 0))
    minor = int(props.get("minor", 0))
    if major < _MIN_CC:
        return [Issue(
            severity="warn",
            message=(
                f"GPU acceleration requested, but the detected "
                f"GPU ({name}, compute capability {major}.{minor}) "
                f"is older than gpu4pyscf supports.  gpu4pyscf "
                f"requires compute capability "
                f"{_MIN_CC}.0 or "
                f"newer (Volta / Turing / Ampere / Hopper / "
                f"Blackwell -- typically RTX 20xx, V100, A100, "
                f"H100 or any consumer GPU from 2018 onward).  "
                f"Running on a {major}.{minor}-class card will "
                f"either fail with cryptic CUDA errors or "
                f"silently fall back to slow paths.  The "
                f"generated script detects this at runtime and "
                f"STOPS -- there is no CPU fallback.  Untick "
                f"\"Use GPU\" to run on the CPU deliberately."
            ),
            where="config.use_gpu",
        )]

    # Everything checks out -- gpu4pyscf is installed and the
    # detected GPU meets the minimum compute capability.  No
    # warning.
    return []

# (_is_hybrid_functional removed 2026-07, V4: the hybrid detector +
#  grid-floor gate now live once in validation.pyscf.is_hybrid_functional
#  / check_dft_grid_level, called by spectra_render_checks above.)


# The render-time checks are the module's public door (the retired
# PySCFSpectraEngine's name here made `import *` raise).
__all__ = ["spectra_render_checks"]
