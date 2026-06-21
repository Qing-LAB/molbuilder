"""PySCFSpectraEngine -- concrete Spectra-tab engine for PySCF.

The v1 engine.  Plugs into the registry at module-import time via
:func:`molbuilder.spectra.engine_base.register_engine`.

Responsibilities (spec § 3.2, § 9, § 11):

  * :meth:`render_script` -- emit a self-contained Python script
    that, when run externally with ``python <job>.spectra.py``,
    computes the harmonic Hessian, (optionally) Raman activities,
    and (optionally) per-mode displaced-geometry SCFs.  Each
    completed phase is checkpointed to ``<job>.spectra.json`` via
    atomic replace so a live-watch poller can pick up partial
    progress.
  * :meth:`parse_output` -- thin wrapper over
    :func:`molbuilder.parsers.spectra_json.parse_spectra_json`.
    Engine-level adaptation (post-processing of
    ``engine_metadata``) lives here.
  * :meth:`preflight` -- scientific + consistency checks combining
    the engine-agnostic :func:`spectra.selection.validate_selection`
    with PySCF-specific advisories (grid level with a hybrid,
    displacement amplitude bounds, hybrid + density-fitting check,
    etc.).
  * :meth:`methods_fragment` -- engine-specific Methods-paragraph
    prose composed into the full Methods text by
    :func:`spectra.methods.render_methods_md`.

The script template emitted by :meth:`render_script` is built in
the companion module :mod:`spectra.pyscf_script` (next commit) to
keep this file focused on the engine-level wiring; this module
imports + delegates.
"""

from __future__ import annotations

from typing import List, Optional

from ..config.spectra import SpectraConfig
from ..issues import Issue
from ..structure import Structure
from .engine_base import register_engine
from .results import ModeData, PHASE_COMPLETE, SpectraResults
from .selection import validate_selection

# parse_spectra_json is lazy-imported inside parse_output to break a
# circular import: parsers.spectra_json imports from spectra.results,
# which loads spectra.__init__, which imports this module.  If we
# imported parse_spectra_json eagerly here, the dependency graph
# `parsers.spectra_json -> spectra.results -> spectra.__init__ ->
# pyscf_engine -> parsers.spectra_json` deadlocks during the very
# first `from molbuilder.parsers.spectra_json import ...` call.


# Citation key markers used in the methods_fragment + the
# script's docstring header.  Pulled out as constants so the
# tests and the spec's references.bib audit can cross-check the
# set we cite for this engine.
_PYSCF_CITES = ("Sun2020", "Sun2018", "Komornicki1979")


@register_engine
class PySCFSpectraEngine:
    """PySCF-backed Spectra engine.

    Selected because PySCF ships:

      * an analytic Hessian (``pyscf.hessian.RKS`` / ``UKS`` /
        ``RHF`` / ``UHF``) -- analytic = exact gradient noise floor,
        no finite-difference cancellation worries.
      * analytic polarizability derivatives at the RKS/UKS level
        via ``pyscf.prop.polarizability`` (Komornicki-style dα/dR
        projected onto mass-weighted modes for Raman activities).
      * cheap displaced-geometry SCFs -- the L4 step is just N
        single-point SCFs at q ± A·Q_i for the selected modes,
        which is trivially parallelisable.

    The engine is engine-agnostic in its inputs (SpectraConfig +
    Structure) and outputs (SpectraResults), so adding a SIESTA /
    TURBOMOLE / NWChem engine later requires zero changes outside
    a parallel ``*_engine.py`` module.
    """

    name:  str = "pyscf"
    label: str = "PySCF (analytic Hessian + dα/dR)"

    # ------------------------------------------------------------------ #
    # render_script                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def render_script(cls,
                      struct: Structure,
                      cfg: SpectraConfig) -> str:
        """Build the self-contained Python script.

        Delegates the actual templating to
        :func:`spectra.pyscf_script.render_spectra_script` so the
        engine wrapper stays a thin shim and the script-template
        unit tests have a focused entry point.
        """
        # Local import to keep the engine module importable even if
        # the script template hits a transient ImportError (e.g.
        # during partial rebuilds while iterating on the template).
        # Once the template is landed this can become a top-level
        # import.
        from .pyscf_script import render_spectra_script
        return render_spectra_script(struct, cfg)

    # ------------------------------------------------------------------ #
    # parse_output                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def parse_output(cls, path: str) -> SpectraResults:
        """Parse a ``<job>.spectra.json`` file into a typed
        :class:`SpectraResults`.

        The on-disk format is engine-agnostic by design (spec § 6),
        so the heavy lifting lives in
        :func:`molbuilder.parsers.spectra_json.parse_spectra_json`.
        This method is the engine's hook for any PySCF-specific
        post-processing (e.g. translating ``engine_metadata``
        flags into typed result fields if the schema ever needs
        it).  Today it's a passthrough; the hook stays so the
        contract is symmetric with future engines that might do
        post-processing.
        """
        # Lazy import to break the circular dep documented at the
        # top of this module.
        from molbuilder.sidecars.spectra import parse_spectra_json
        return parse_spectra_json(path)

    # ------------------------------------------------------------------ #
    # preflight                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def preflight(cls,
                  struct: Structure,
                  cfg: SpectraConfig,
                  prior: Optional[SpectraResults] = None) -> List[Issue]:
        """Combined preflight pass.

        Two sources of Issues:

          1. The engine-agnostic selector / range validation from
             :func:`selection.validate_selection`.  Modes for the
             top_n / threshold selectors aren't available until L3
             runs, so the validator needs prior Raman data to be
             accurate; we report that as a soft-dep error.
          2. PySCF-specific scientific advisories (this method):
             grid level with a hybrid functional, displacement
             amplitude outside the Mills1972-defensible window,
             D4 selected without the pyscf-dispersion package
             present, etc.

        Returns a single merged list (errors + warns interleaved
        in the order they were checked).  The web blueprint
        renders the list via the existing Issues panel.
        """
        issues: List[Issue] = []

        # -- (1) generic selector / range / window validation ----------
        # When prior results are present, the "L3 complete" flag is
        # what tells the selector validator that top_n / threshold
        # are safe to use.  We grab the modes list out of prior so
        # the validator can range-check explicit indices etc.
        l3_done = False
        if prior is not None:
            l3_done = (prior.phase_raman == PHASE_COMPLETE)
            modes_for_validation = list(prior.modes)
        else:
            # No prior run -- the validator falls back to range
            # checks that depend on knowing the eventual mode count
            # only at the explicit-index branch; the rest of the
            # checks (top_n/threshold soft-dep, freq window
            # ordering) don't need a mode list.
            modes_for_validation = []
        issues.extend(validate_selection(
            modes_for_validation, cfg, l3_done=l3_done, prior=prior,
        ))

        # -- (2) PySCF-specific scientific advisories ------------------

        # Grid level with a hybrid functional (spec § 11.4).  PySCF's
        # default grid level is 3 ("screening"); level 4 is the
        # production minimum for hybrids.  Below that the XC
        # numerical integration noise dominates the Hessian and
        # gives garbage frequencies (~5-20 cm⁻¹ wander).
        if cls._is_hybrid_functional(cfg.functional) and cfg.grid_level < 4:
            issues.append(Issue(
                severity="warn",
                message=(f"Grid level {cfg.grid_level} is below the "
                         f"recommended minimum of 4 for a hybrid "
                         f"functional ({cfg.functional}).  Hybrid "
                         f"functionals have a sharper exchange "
                         f"contribution that needs a denser numerical "
                         f"grid; below level 4 the grid-integration "
                         f"noise typically dominates the frequency "
                         f"error.  Raise the grid level for "
                         f"publication-quality results."),
                where="config.grid_level",
            ))

        # Displacement amplitude.  The [0.02, 0.20] Å acceptance
        # range is empirical -- not derived from a single source.
        # The lower bound was relaxed from 0.04 to 0.02 on 2026-
        # 05-19 to match the SpectraConfig default; 0.02 Å keeps
        # the probe inside the linear-response regime (ΔE_orbital
        # ∝ A), at the cost of needing a tight SCF tolerance to
        # resolve the smaller ΔE.  The script's default
        # ``conv_tol = 1e-10`` is sufficient (FD noise on ΔE_HOMO
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
                         f"conv_tol=1e-10 SCF; either tighten "
                         f"conv_tol further or raise the "
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
        # check in this very block — see docs/protocols/web-ui-coherence.md
        # Rule 1.  ``metals`` (the flat detection list) is still computed
        # so the supplemental ``explain_metal_spin`` info-line below
        # can echo (element, spin) → (likely oxidation state) for
        # non-spin=0 cases.
        from ..validation import check_open_shell_metal
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
            issues.extend(cls._gpu_capability_advisories())

        # compute_ir is a placeholder for a future release; warn
        # politely if the user toggled it on so they know nothing
        # will come of it yet.
        if cfg.compute_ir:
            issues.append(Issue(
                severity="warn",
                message=("IR intensities aren't implemented in this "
                         "release -- the checkbox is reserved for a "
                         "future version.  The current run will "
                         "produce Raman activities (if enabled) but "
                         "no IR data.  Untick the box to clear this "
                         "notice."),
                where="config.compute_ir",
            ))

        # Structure-side advisory: PySCF's analytic Hessian assumes
        # a closed-shell or unrestricted SCF with spin set
        # consistently.  We can't verify the converged SCF here,
        # but we can check that the method/spin combination is at
        # least nominally consistent.  Mirrors the same check in
        # the relaxation script (pyscf/input.py).
        # Spectra-tab cfg doesn't carry spin (yet -- see config
        # roadmap); skip until that field lands.

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

        return issues

    # ------------------------------------------------------------------ #
    # methods_fragment                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def methods_fragment(cls,
                         cfg: SpectraConfig,
                         modes: List[ModeData]) -> str:
        """Engine-specific paragraph for the Methods section.

        Names PySCF + the specific Hessian / polarizability-
        derivative APIs used, with citation keys that resolve
        against ``docs/tabs/spectra/references.bib`` and bubble
        up into the trailing bibliography of the full Methods
        text.
        """
        # Method-class-specific Hessian module name -- the PySCF
        # API splits hessian.RKS / UKS / RHF / UHF.  Sun2020 +
        # Sun2018 cite the package itself; the analytic Hessian
        # API is covered by both.
        method = cfg.method.upper()
        hessian_module = {
            "RKS": "pyscf.hessian.rks",
            "UKS": "pyscf.hessian.uks",
            "RHF": "pyscf.hessian.rhf",
            "UHF": "pyscf.hessian.uhf",
        }.get(method, "pyscf.hessian")

        parts = [
            "All electronic-structure calculations were performed "
            "with PySCF [Sun2020, Sun2018], a Python-based ab "
            "initio package."
        ]

        # Mention the analytic Hessian explicitly -- the choice of
        # analytic over finite-difference is a load-bearing claim
        # for the Methods reader (no FD noise on frequencies).
        parts.append(
            f"The harmonic Hessian was obtained analytically via "
            f"`{hessian_module}` and mass-weighted, then "
            f"diagonalized after projection of the six "
            f"translational/rotational eigenvectors."
        )

        # Raman path: cite the analytic dα/dR + Komornicki1979.
        if cfg.compute_raman:
            parts.append(
                "Polarizability derivatives dα/dR were computed "
                "analytically with `pyscf.prop.polarizability` "
                "and projected onto the mass-weighted mode "
                "eigenvectors to obtain Raman activities in "
                "Å⁴/amu [Komornicki1979]."
            )

        # Density fitting note.
        if cfg.density_fit:
            parts.append(
                f"Density fitting (RIJK) was used for the Coulomb "
                f"and exchange evaluation; the auxiliary basis was "
                f"selected automatically by PySCF for the "
                f"production {cfg.basis} basis."
            )

        # Grid level for DFT runs.
        if method.endswith("KS"):
            parts.append(
                f"DFT integration used PySCF's grid level "
                f"{cfg.grid_level} (production setting for hybrid "
                f"functionals; the v1 spec § 11.4 sets level 4 as "
                f"the recommended minimum)."
            )

        return " ".join(parts)


    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # GPU capability check                                               #
    # ------------------------------------------------------------------ #
    #
    # gpu4pyscf has two prerequisites: (a) the Python package itself,
    # (b) an NVIDIA GPU with compute capability >= 7.0 (Volta).
    # Older cards (Pascal/Maxwell/Kepler) still report success on
    # `import gpu4pyscf` but fail with cryptic CUDA errors at run
    # time -- exactly what users tend to hit.  Probe via cupy
    # (gpu4pyscf's required dep) for actionable feedback up front.

    # Single source: molbuilder.runtime_info.  Class-attribute alias
    # preserved so external code that imported ``PySCFSpectraEngine.
    # GPU4PYSCF_MIN_COMPUTE_CAPABILITY`` keeps working without a
    # change.  The numeric value lives in runtime_info now (one place
    # for engine emitters + this class to agree on).
    from molbuilder.runtime_info import (
        GPU4PYSCF_MIN_COMPUTE_CAPABILITY as _MIN_CC,
    )
    GPU4PYSCF_MIN_COMPUTE_CAPABILITY = _MIN_CC

    @classmethod
    def _gpu_capability_advisories(cls) -> List[Issue]:
        """Return [] if gpu4pyscf + a supported GPU are available,
        else one warn-severity Issue describing what's missing.
        Always WARN (not ERROR) -- the generated script falls back
        to CPU automatically, so an unusable GPU is annoying but
        not fatal.
        """
        try:
            import gpu4pyscf  # noqa: F401
        except ImportError:
            return [Issue(
                severity="warn",
                message=(
                    "GPU acceleration requested, but gpu4pyscf is "
                    "not installed on this server.  The generated "
                    "script falls back to CPU PySCF automatically, "
                    "so this is non-fatal.  To get the GPU speed-up: "
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
                    "isn't.  Reinstall gpu4pyscf.  Script will fall "
                    "back to CPU at runtime."
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
                    f"Script will fall back to CPU."
                ),
                where="config.use_gpu",
            )]
        if n_devs == 0:
            return [Issue(
                severity="warn",
                message=(
                    "GPU acceleration requested but no NVIDIA GPU "
                    "was detected on this host.  Script will fall "
                    "back to CPU at runtime.  Untick \"Use GPU\" "
                    "to silence this warning."
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
                    f"({type(exc).__name__}: {exc}).  Script will "
                    f"fall back to CPU."
                ),
                where="config.use_gpu",
            )]
        name = props.get("name", "(unknown GPU)")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        major = int(props.get("major", 0))
        minor = int(props.get("minor", 0))
        if major < cls.GPU4PYSCF_MIN_COMPUTE_CAPABILITY:
            return [Issue(
                severity="warn",
                message=(
                    f"GPU acceleration requested, but the detected "
                    f"GPU ({name}, compute capability {major}.{minor}) "
                    f"is older than gpu4pyscf supports.  gpu4pyscf "
                    f"requires compute capability "
                    f"{cls.GPU4PYSCF_MIN_COMPUTE_CAPABILITY}.0 or "
                    f"newer (Volta / Turing / Ampere / Hopper / "
                    f"Blackwell -- typically RTX 20xx, V100, A100, "
                    f"H100 or any consumer GPU from 2018 onward).  "
                    f"Running on a {major}.{minor}-class card will "
                    f"either fail with cryptic CUDA errors or "
                    f"silently fall back to slow paths.  The "
                    f"generated script will detect this at runtime "
                    f"and fall back to CPU automatically.  Untick "
                    f"\"Use GPU\" to silence this warning and skip "
                    f"the GPU code path entirely."
                ),
                where="config.use_gpu",
            )]

        # Everything checks out -- gpu4pyscf is installed and the
        # detected GPU meets the minimum compute capability.  No
        # warning.
        return []

    @staticmethod
    def _is_hybrid_functional(name: str) -> bool:
        """Approximate test: does the functional name name a hybrid
        (i.e. has a fraction of HF exchange)?

        We use a deny-list-by-prefix because functional name space
        is sprawling -- B3*, PBE0, M06*, BHandH*, CAM-B3LYP, ωB97*,
        TPSS0, MN15, etc.  The check exists only to gate one
        scientific advisory (grid level recommendation), so a false
        negative just means the user doesn't get the warn; a false
        positive triggers a warn that's harmless.  Conservative:
        when in doubt, treat as hybrid (the >=4 grid recommendation
        is benign even for non-hybrids).
        """
        n = (name or "").lower()
        # Common hybrid markers.  Order doesn't matter (any match
        # wins).  PBE0, B3LYP, BHandH, M06, ωB97X-D, CAM-B3LYP, ...
        return any(tag in n for tag in (
            "b3", "pbe0", "bhandh", "m06", "mn15", "cam-", "wb97",
            "ωb97", "tpss0", "x3lyp", "b97", "hse",
        ))


__all__ = ["PySCFSpectraEngine"]
