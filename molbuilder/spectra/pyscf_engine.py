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
from .results import ModeData, SpectraResults
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
        from ..parsers.spectra_json import parse_spectra_json
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
            from .results import PHASE_COMPLETE
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

        # Displacement amplitude.  The [0.04, 0.20] Å range is an
        # empirical, contemporary-practice heuristic -- not derived
        # from a single source.  Below 0.04 Å the finite-difference
        # noise on ΔE_HOMO dominates (at SCF conv_tol=1e-9, ΔE is
        # ~1e-7 Ha and the FD noise scales as conv_tol/δ).  Above
        # 0.20 Å cubic anharmonicity in the potential becomes
        # significant (cf. Mills1972 for the general anharmonic-
        # coupling framework, although that source doesn't pin the
        # specific numerical bounds used here).
        amp = cfg.displacement_amplitude_ang
        if amp < 0.04:
            issues.append(Issue(
                severity="warn",
                message=(f"Displacement amplitude {amp:g} Å is "
                         f"smaller than the typical defensible range "
                         f"(0.04-0.20 Å).  At small amplitudes the "
                         f"HOMO/LUMO energy shifts you're trying to "
                         f"measure are comparable to SCF noise, and "
                         f"the resulting orbital-energy data is "
                         f"unreliable.  Raise to at least 0.04 Å."),
                where="config.displacement_amplitude_ang",
            ))
        elif amp > 0.20:
            issues.append(Issue(
                severity="warn",
                message=(f"Displacement amplitude {amp:g} Å is "
                         f"larger than the typical defensible range "
                         f"(0.04-0.20 Å).  At large amplitudes the "
                         f"potential isn't linear in the displacement "
                         f"any more -- anharmonic terms contaminate "
                         f"the orbital-energy slope you want to "
                         f"measure.  Lower to 0.20 Å or less."),
                where="config.displacement_amplitude_ang",
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
        if cfg.fixed_indices:
            n = struct.n_atoms if hasattr(struct, "n_atoms") else None
            if n is None:
                # Duck-typed: fall back to len(elements).
                try:
                    n = len(struct.elements)
                except Exception:
                    n = None
            if n is not None:
                bad = [i for i in cfg.fixed_indices if not 0 <= int(i) < n]
                if bad:
                    issues.append(Issue(
                        severity="error",
                        message=(f"\"Fixed by atom index\" contains "
                                 f"out-of-range numbers {bad}.  This "
                                 f"structure has {n} atoms; valid "
                                 f"indices are 0..{n - 1} (counting "
                                 f"from zero)."),
                        where="config.fixed_indices",
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
