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
    substance.  Two callers, one body: ``_validate_spectra`` (the
    registered SpectraConfig validator, for callers holding a real
    SpectraConfig) and ``pyscf/vibration_deck.py`` directly -- the
    deck's config view is an ADAPTER over PySCFConfig, so the
    type-keyed registry cannot see it (which is exactly how this gate
    silently skipped between P1 and P3; the direct call closes that).
    ``cfg`` is duck-typed to the spectra vocabulary for the same
    reason.  No selector-availability checks; those were
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
                              explain_metal_spin,
                              total_electrons)
    try:
        parity_err = check_spin_charge_parity(
            struct, cfg.charge, cfg.spin,
        )
    except KeyError as e:
        # Unknown element symbol (typo / bad PDB column fallback).
        issues.append(Issue(
            severity="error",
            message=str(e),
            where="structure",
        ))
        parity_err = None
    if parity_err:
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
        # Conservative n_free estimate: total - explicit-index
        # freezes - element-match freezes.  Doesn't account for
        # overlap; the over-estimate of frozen atoms is fine
        # here (we'd under-warn rather than spam).
        n_frz_idx = len(cfg.frozen_indices)
        try:
            n_frz_elem = sum(
                1 for el in struct.elements
                if el in cfg.frozen_elements
            )
        except Exception:
            n_frz_elem = 0
        n_free_estimate = max(0, n_atoms - n_frz_idx - n_frz_elem)
        if (n_free_estimate > 30
                and not cfg.frozen_elements
                and not cfg.frozen_indices):
            issues.append(Issue(
                severity="warn",
                message=(
                    f"This structure has {n_atoms} atoms, none "
                    f"of them frozen -- the Hessian cost grows "
                    f"~N² and the Raman finite-difference step "
                    f"adds 6·N more SCFs.  If part of the system "
                    f"is a metal slab, surface, or other anchor "
                    f"you don't actually need to vibrate, freeze "
                    f"it via \"Freeze by element\" or \"Freeze "
                    f"by atom index\" -- the cost saving is "
                    f"typically large.  Ignore this if the whole "
                    f"system needs to vibrate."
                ),
                where="config.frozen_indices",
            ))

    # Partial-Hessian projection advisory.  When SOME atoms are
    # frozen but FEWER THAN 3 (or 3+ but they're collinear), the
    # partial-Hessian path can't fully anchor the system in
    # space.  The result is 1-5 "spurious" near-zero modes that
    # correspond to rigid-body motion of the free fragment, not
    # real vibrations.  Three non-collinear anchor atoms remove
    # all six translation+rotation DOFs; fewer leaves a residue.
    # We can't resolve element / residue freezes into actual
    # atom counts without the Structure's atom list -- but
    # element / residue freezes typically pin many atoms (a
    # whole metal slab, a whole residue).  Only warn for the
    # genuinely-suspect case: frozen_indices has 1 or 2 entries
    # AND nothing else is being frozen.
    if (len(cfg.frozen_indices) in (1, 2)
            and not cfg.frozen_elements
            and not cfg.frozen_residue_names):
        issues.append(Issue(
            severity="warn",
            message=(
                f"You've frozen only {len(cfg.frozen_indices)} "
                f"atom(s).  That isn't enough to fully anchor "
                f"the free fragment in space (you need at "
                f"least 3 non-collinear frozen atoms to remove "
                f"all 6 translation+rotation degrees of "
                f"freedom).  The vibrational analysis will "
                f"include {6 - 2 * len(cfg.frozen_indices)}-ish "
                f"spurious near-zero modes corresponding to "
                f"rigid-body motion of the free atoms.  These "
                f"won't crash the run but you should ignore "
                f"them in your spectrum interpretation."
            ),
            where="config.frozen_indices",
        ))

    # GPU advisory: if the user asked for GPU acceleration, check
    # (1) whether gpu4pyscf is importable on the molbuilder host
    # and (2) whether the host has a GPU that actually meets
    # gpu4pyscf's minimum compute capability (7.0 = Volta).  The
    # generated script is robust to a missing gpu4pyscf (CPU
    # fallback), but checking here lets the user fix things
    # before the run rather than after.
    if cfg.use_gpu:
        issues.extend(_gpu_capability_advisories())

    # compute_ir advisory RETIRED 2026-08-21 -- it warned that IR
    # was "not implemented", which P1 falsified (the deck computes IR
    # via dipole derivatives, band-level validated against literature
    # water intensities; roadmap.md § 5 records the closure).  Found
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
    if cfg.spin > 0 and method_u in ("RKS", "RHF"):
        issues.append(Issue(
            severity="warn",
            message=(f"spin = {cfg.spin} is set but method = {method_u} "
                     f"is closed-shell (restricted); a restricted SCF "
                     f"forces all electrons paired (2S=0).  Use UKS / UHF "
                     f"for open-shell systems, or set spin = 0."),
            where="config.method",
        ))
    if method_u in ("RKS", "RHF") and cfg.spin == 0:
        # Odd electron count with a restricted closed-shell method + spin=0
        # is a radical the user didn't recognise: an unpaired electron MUST
        # exist, so RKS/RHF either fails SCF or returns the wrong state.
        try:
            parity_err2 = check_spin_charge_parity(struct, cfg.charge, 1)
            odd = parity_err2 is None   # spin=1 fits ⇒ odd electron count
        except Exception:
            odd = False
        if odd:
            issues.append(Issue(
                severity="warn",
                message=(f"odd electron count with method = {method_u} and "
                         f"spin = 0 -- the system is a radical and a "
                         f"closed-shell SCF will fail to converge or "
                         f"return the wrong electronic state.  Switch to "
                         f"UKS / UHF and set spin = 1 (or set cfg.charge "
                         f"if auto-detection miscounted)."),
                where="config.method",
            ))

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
                    where="config.frozen_indices",
                ))

    # Boundary-condition guards (design.md "Sidecar-driven
    # boundary conditions — the three-stage contract"):
    #
    # The contract:  sidecar -> form (cfg) -> script must be
    # explicit, consistent, fully respected.  The script render
    # itself emits cfg.frozen_indices verbatim (no silent merge);
    # these two preflight checks make divergence + unconsumed
    # labels visible so nothing is silently absorbed.

    # Pattern A: divergence warn.  If the structure's sidecar
    # ``frozen_atoms`` (populated via the /modify selection
    # panel) carries indices NOT in cfg.frozen_indices, the
    # user is about to generate a script that does NOT freeze
    # atoms the sidecar said were frozen.  Surface it so the
    # user can either (a) re-load the structure to pull the
    # sidecar values into the form, (b) include them
    # manually, (c) clear them in /modify, or (d) intentionally
    # override.  Either way: explicit, not silent.
    sidecar_frozen = getattr(struct, "frozen_atoms", None) or []
    if sidecar_frozen:
        in_cfg = set(int(i) for i in cfg.frozen_indices)
        missing = sorted(set(sidecar_frozen) - in_cfg)
        if missing:
            issues.append(Issue(
                severity="warn",
                message=(
                    f"The structure's sidecar has {len(sidecar_frozen)} "
                    f"frozen atom(s) (indices {sorted(sidecar_frozen)}) "
                    f"but the form's \"Freeze by atom index\" doesn't "
                    f"include {missing}.  The generated script will "
                    f"freeze only what the form lists.  Either "
                    f"include those indices in the form, clear them "
                    f"in /modify, or accept the override."
                ),
                where="config.frozen_indices",
            ))

    # Pattern B: unrecognized-label notice.  The selection panel
    # also writes ``regions`` (L-electrode, bridge, interface,
    # …) for transport-engine workflows.  The spectra engine
    # does NOT consume regions; the partial-Hessian path
    # operates on the frozen / free atom partition alone.
    # Surface this explicitly so the user knows their region
    # labels are NOT influencing the spectrum calculation --
    # they're carried forward in the sidecar for /transport
    # but inert here.  Pinned by the three-stage contract:
    # "every label NOT understood by the current engine MUST
    # be named explicitly in a preflight issue."
    regions = getattr(struct, "regions", None) or {}
    non_empty_regions = sorted(
        name for name, idxs in regions.items() if idxs
    )
    if non_empty_regions:
        issues.append(Issue(
            severity="warn",
            message=(
                f"This structure carries region label(s) "
                f"{non_empty_regions}, which the PySCF spectra "
                f"engine does NOT consume.  They will be ignored "
                f"for the Hessian / Raman calculation but stay "
                f"in the sidecar for /transport.  If you meant "
                f"these atoms to be frozen during the spectrum "
                f"run, mark them as such via /modify -> Assign "
                f"to \"frozen_atoms\"."
            ),
            where="structure.regions",
        ))


    # on_nonconvergence policy (pyscf.md § 7a's role table): warning
    # past a failed equilibrium SCF is a legitimate survey-mode choice,
    # but on a VIBRATION run every downstream quantity -- frequencies,
    # intensities, thermochemistry -- inherits the unconverged density.
    _pol = str(getattr(cfg, "on_nonconvergence", "halt") or "halt").lower()
    if _pol in ("warn", "continue"):
        issues.append(Issue(
            severity="warn",
            message=(
                "on_nonconvergence is set to warn-and-continue.  On a "
                "vibration run an unconverged equilibrium SCF poisons "
                "everything computed after it (Hessian columns, IR/Raman "
                "intensities, ZPE/G).  Keep 'halt' unless you are "
                "deliberately surveying hard cases and will inspect the "
                "convergence flags by hand."),
            where="config.on_nonconvergence",
        ))

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
#  / check_dft_grid_level, called by preflight() above.)


__all__ = ["PySCFSpectraEngine"]
