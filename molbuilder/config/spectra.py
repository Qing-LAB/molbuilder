"""SpectraConfig -- every parameter the Spectra-tab engine reads.

L1 dataclass.  Field metadata (``label`` / ``unit`` / ``range`` /
``choices`` / ``tier`` / ``help`` / optional ``id_suffix`` /
``null_label`` / ``pattern`` / ``validate``) drives:

  * the schema-driven Build / Spectra form (via
    ``web/blueprints/_shared.py::dataclass_to_form_schema``);
  * the validator pass (via ``validation.py``);
  * the click-based CLI (via the dataclass-to-click bridge in
    ``cli.py``); and
  * the Methods-text generator (which reads ``label`` + ``unit``
    + ``help`` to compose the manuscript-ready Methods paragraph).

The full design contract is in ``docs/tabs/spectra/spec.md`` § 4.

This is the **engine-agnostic** config -- field names describe
WHAT the user is asking for, not HOW the engine computes it.
Adding a future engine (SIESTA, ...) reuses the same fields; the
engine's ``render_script`` consumes them.

Default values follow current quantum-chemistry good practice for
a relaxed small/medium organic molecule at the Spectra workflow's
entry point:

    * B3LYP/def2-SVP with D3BJ dispersion -- the same defaults as
      the Build tab's PySCF section, on the assumption that the
      structure was relaxed at this level and the spectra are
      computed at the same level for consistency.
    * Density fitting on (essentially free accuracy for hybrid
      DFT in PySCF).
    * Grid level 4 (hybrid-functional safe; v1 spec § 11.4 warns
      when < 4 with a hybrid).
    * Displacement amplitude 0.02 Å -- chosen to stay well inside
      the linear-response regime where ΔE_orbital is proportional to
      the displacement.  At this amplitude the anharmonic-cubic
      contamination (Mills 1972 §2.4) is negligible and the
      orbital-shift slope is the physically meaningful number; the
      trade-off is that the absolute orbital-energy differences are
      small (~meV) and benefit from a tight SCF convergence.  See
      the ``displacement_amplitude_ang`` field below for the
      contemporary-practice context (0.02-0.15 Å).
    * Per-mode electronic structure off by default
      (``es_mode_selection = "skip"``) so a first-pass run is
      cheap; users opt in to ``top_n`` / ``explicit`` after they
      see the Raman spectrum and pick modes of interest.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List, Optional

# Shared with SiestaConfig + PySCFConfig -- one regex, one rule
# for the job-layout basename pattern (see job-layout.md and
# config/siesta.py for the spec).
from .siesta import _validate_basename


@dataclass
class SpectraConfig:
    # Explicit form-section order so the schema-driven UI renders
    # sections in workflow order, independent of field declaration
    # order in this file.
    _form_section_order = (
        "System",
        "Method",
        "Frozen atoms",
        "Spectrum",
        "Electronic structure",
        "SCF",
        "Runtime",
    )

    # One-paragraph description per form section.  Surfaced in the
    # UI directly below each section's legend so the user reads
    # "what is this group of knobs for?" before digging into the
    # individual fields.  Plain language; no internal-architecture
    # jargon.
    _form_section_descriptions = {
        "System": (
            "Engine selection and job-name identity.  The engine is "
            "the quantum-chemistry code that will run; the job name "
            "becomes the basename of every output file produced by "
            "the generated script."
        ),
        "Method": (
            "Level of theory for the SCF, the Hessian, and (when "
            "enabled) the polarizability derivatives.  Defaults "
            "(B3LYP / def2-SVP / D3BJ dispersion / density fitting "
            "on) are production-quality for typical organic "
            "molecules and metal-molecule junctions."
        ),
        "Frozen atoms": (
            "Atoms held in place during the vibrational analysis "
            "('frozen' = 'fixed' in this codebase; molbuilder uses "
            "'frozen' as the canonical term across UI, data, and "
            "engines, matching the quantum-chemistry literature).  "
            "Most common use: freeze a metal slab in a "
            "molecule-metal junction so the vibrational modes "
            "describe only the molecule.  Three rules are combined "
            "with OR: an atom is frozen if it matches by element, "
            "by residue name (PDB only), or by index."
        ),
        "Spectrum": (
            "Which spectrum-related quantities to compute.  "
            "Frequencies are always computed.  Raman activities "
            "are optional -- turning them off makes the run "
            "significantly faster but you get only the line "
            "positions, no intensities.  IR intensities are "
            "reserved for a future release."
        ),
        "Electronic structure": (
            "Per-mode orbital-energy probe: for each selected "
            "vibrational mode, the script computes how the HOMO, "
            "LUMO, and nearby orbitals shift when the molecule is "
            "displaced along the mode.  Used downstream for "
            "electron-phonon coupling analysis (transport, "
            "inelastic spectroscopy).  This is the most expensive "
            "step in a typical run; cost scales linearly with the "
            "number of selected modes."
        ),
        "SCF": (
            "Self-consistent-field convergence criteria and the "
            "DFT integration grid.  Tighter tolerances + denser "
            "grids give more accurate frequencies but cost more "
            "SCF cycles per geometry.  The defaults are calibrated "
            "for production vibrational analysis with a hybrid "
            "functional; loosen for cheap exploratory runs, "
            "tighten if you suspect frequency noise."
        ),
        "Runtime": (
            "How the run uses your hardware -- memory budget, "
            "CPU thread count, optional NVIDIA GPU acceleration "
            "via gpu4pyscf, and log verbosity.  These don't "
            "affect the science, only the wall time and the "
            "amount of diagnostic output."
        ),
    }

    # ----------------- System -----------------

    engine: str = field(default="pyscf", metadata={
        "section": "System",
        "workflow_group": "profile",
        "label":   "Engine",
        # Only PySCF in v1.  SIESTA is reserved (see
        # spec.md § 13.2); adding it later extends this tuple by
        # one entry and the form/CLI/validator pipelines pick up
        # the new choice automatically.
        "choices": ("pyscf",),
        "engine_key": '(molbuilder: backend selector)',
        "help":    "computational engine that runs the Hessian, "
                   "polarizability derivatives, and the per-mode "
                   "displaced-geometry SCFs",
    })
    job_name: str = field(default="spectra", metadata={
        "section":  "System",
        "workflow_group": "profile",
        "label":    "Job name",
        "id_suffix": "job-name",
        "pattern":  r"^[A-Za-z0-9_\-]+$",
        "engine_key": '(molbuilder: filename + log-name basename)',
        "help":     "filesystem-safe basename for emitted files "
                    "(spectra.py + spectra.json + thermo.txt).  "
                    "Same rule as SIESTA SystemLabel / PySCF job_name "
                    "-- see docs/protocols/job-layout.md.",
        "validate": _validate_basename("job_name"),
    })

    # ----------------- Method -----------------

    method: str = field(default="RKS", metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "SCF method",
        "choices": ("RKS", "UKS", "RHF", "UHF"),
        "engine_key": 'RKS / UKS / RHF / UHF  (PySCF class selection)',
        "help":    "RKS (restricted Kohn-Sham): closed-shell DFT, "
                   "all electrons paired, spin must be 0.  "
                   "UKS (unrestricted Kohn-Sham): open-shell DFT, "
                   "required for ANY non-zero spin (radicals, "
                   "transition metals, etc.).  RHF / UHF: Hartree-Fock "
                   "equivalents (rarely useful by themselves -- pick a "
                   "functional with RKS/UKS for production work).  "
                   "RULE: a structure containing Fe / Mn / Co / Ni / Cu "
                   "(open-shell transition metals) ALMOST ALWAYS needs "
                   "UKS with a non-zero spin -- see the spin field below.",
    })
    charge: int = field(default=0, metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Net charge",
        "range":   (-10, 10),
        "engine_key": 'gto.M(charge=...)',
        "help":    "Net molecular charge in units of e.  Default 0 = "
                   "neutral molecule.  Examples: deprotonated carboxylate "
                   "= -1; protonated amine = +1; deprotonated phosphate "
                   "= -1 or -2; bis-thiolate Fe(III) heme = -1 (porphyrin "
                   "-2, Fe +3, 2 thiolates -2: net -1, NOT neutral).  "
                   "Counts as a CHECK: total electrons = sum(Z) - charge; "
                   "must have parity matching spin (even when spin=0).",
    })
    spin: int = field(default=0, metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Spin (2S = # unpaired electrons)",
        "range":   (0, 10),
        "engine_key": 'gto.M(spin=...)  # 2S, # of unpaired electrons',
        "help":    "PySCF spin convention: 2S = n_alpha - n_beta = "
                   "number of unpaired electrons.  NOT the multiplicity "
                   "(2S+1).  Examples: closed-shell singlet = 0; "
                   "radical/doublet = 1; triplet = 2.  Fe(II) cases: "
                   "low-spin (CO/CN heme) = 0; intermediate-spin "
                   "(4-coord Fe-porphyrin) = 2; high-spin (deoxy-heme, "
                   "bis-thiolate) = 4.  Fe(III) cases: low-spin (bis-"
                   "imidazole) = 1; intermediate-spin = 3; high-spin "
                   "(met-myoglobin) = 5.  ANY non-zero value REQUIRES "
                   "method = UKS or ROKS.  Wrong value = unphysical SCF "
                   "with huge forces.",
    })
    functional: str = field(default="B3LYP", metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Functional",
        "engine_key": 'mf.xc = ...',
        "help":    "XC functional name (libxc string).  Recommendations: "
                   "B3LYP -- modern hybrid default for organic / "
                   "biomolecule chemistry.  PBE0 -- faster hybrid "
                   "(20% HF exchange vs B3LYP's 20% too, similar quality "
                   "for organics).  wB97X-D -- range-separated hybrid "
                   "with built-in dispersion; best for non-covalent / "
                   "long-range cases.  TPSSh -- meta-GGA hybrid, often "
                   "the best balance for transition-metal complexes "
                   "(reliable spin gaps).  PBE / BLYP -- pure GGAs, "
                   "2-3x faster but worse for HOMO-LUMO gaps.  "
                   "For Fe heme work, TPSSh or B3LYP+D3BJ are standard.",
    })
    basis: str = field(default="def2-SVP", metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Basis set",
        "engine_key": 'gto.M(basis=...)',
        "help":    "Gaussian basis set name.  Recommendations by tier: "
                   "MINIMAL: STO-3G (toy only).  SMALL: 6-31G(d) -- "
                   "fast but inadequate for transition metals.  "
                   "PRODUCTION MINIMUM: def2-SVP -- safe organic default; "
                   "for transition metals consider def2-SVP for Fe-row "
                   "(includes Stuttgart ECP automatically via PySCF).  "
                   "ACCURATE: def2-TZVP, cc-pVTZ (1.5-3x cost).  "
                   "DIFFUSE (for anions / weak interactions): aug-cc-pVDZ, "
                   "def2-SVPD.  For Fe heme: def2-SVP is the production "
                   "minimum; def2-TZVP for publication-quality.",
    })
    ecp: Optional[str] = field(default=None, metadata={
        "section":    "Method",
        "workflow_group": "profile",
        "label":      "Pseudopotential (ECP)",
        "null_label": "(auto)",
        "engine_key": 'gto.M(ecp=...)',
        "help":       "Effective core potential name -- PySCF replaces "
                      "the core electrons of heavy atoms (typically Z > 36) "
                      "with an analytic potential, so the SCF only treats "
                      "the chemically-active valence shells.  PySCF ships "
                      "the lookup data inside the package; no external "
                      "files needed (unlike SIESTA's downloaded "
                      "pseudopotentials).  Recommendations: blank (auto) "
                      "lets molbuilder pick 'lanl2dz' when Z > 36 atoms "
                      "are present AND the basis isn't def2-* (def2-* "
                      "bundles its own Stuttgart ECP, so no separate "
                      "ECP is needed -- works for Mo/W/Pt out of the "
                      "box with def2-SVP).  Override with 'lanl2dz' / "
                      "'stuttgart' / 'sbkjc' for explicit selection; "
                      "set to 'none' or '' to disable.  Non-def2 basis "
                      "+ heavy atoms WITHOUT this set silently does "
                      "all-electron SCF on the heavy atoms -- usually "
                      "wrong.",
    })
    dispersion: Optional[str] = field(default="d3bj", metadata={
        "section":    "Method",
        "workflow_group": "profile",
        "label":      "Dispersion",
        "choices":    ("d3", "d3bj", "d4", "none"),
        "null_label": "(none)",
        "engine_key": 'mf = mf.add_dispersion(...)',
        "help":       "dispersion correction; D3 with Becke-Johnson "
                      "damping is the modern default [Grimme2011]",
    })
    density_fit: bool = field(default=True, metadata={
        "section": "Method",
        "workflow_group": "profile",
        "label":   "Density fitting",
        "engine_key": 'mf = mf.density_fit()',
        "help":    "speeds up each SCF cycle 5-10x by approximating "
                   "the two-electron Coulomb / exchange integrals "
                   "with an auxiliary basis.  Accuracy loss is "
                   "negligible for typical organic molecules; turn "
                   "off only if you need every-last-digit reference "
                   "energies.  Note: the Raman polarizability path "
                   "automatically falls back to non-DF for the "
                   "displaced-geometry calculations (the analytic "
                   "polarizability code doesn't support DF yet).",
    })

    # ----------------- Frozen atoms -----------------
    #
    # Three filters combined with UNION semantics: an atom is frozen
    # if it matches ANY of element / residue-name / explicit-index.
    # ("Frozen" is the canonical term across molbuilder -- field
    # names, sidecar key, UI labels.  Some quantum-chemistry contexts
    # use "fixed atoms" for the same concept; we standardise on
    # "frozen" to match the spectroscopy literature.)
    # See spec.md § 7 for the algorithm.
    #
    # The three lists are stored as native Python types here; the
    # form sends comma-separated strings, which the server's
    # `_shared.coerce_to_field_type` splits / parses (existing
    # Sequence[str] handler covers frozen_elements + frozen_residue_names;
    # frozen_indices is parsed by a small helper at config-from-params
    # time -- see the engine's `preflight`).

    frozen_elements: List[str] = field(default_factory=list, metadata={
        "section": "Frozen atoms",
        "workflow_group": "profile",
        "label":   "Freeze by element",
        "engine_key": '(molbuilder: frozen-atom filter by element)',
        "help":    "comma-separated element symbols whose atoms are "
                   "held in place during the vibrational analysis "
                   "(e.g. \"Au\" to freeze a gold electrode in a "
                   "metal-molecule-metal junction).  Freezing atoms "
                   "removes their contribution to the Hessian and "
                   "reduces the mode count; it does NOT remove them "
                   "from the SCF or from the displaced-geometry "
                   "calculations.",
    })
    frozen_residue_names: List[str] = field(default_factory=list, metadata={
        "section": "Frozen atoms",
        "workflow_group": "profile",
        "label":   "Freeze by residue name",
        "engine_key": '(molbuilder: frozen-atom filter by PDB residue)',
        "help":    "comma-separated PDB residue names whose atoms "
                   "are held in place (e.g. \"ALA,GLY\" to freeze "
                   "specific peptide residues).  Requires the input "
                   "structure to carry residue labels -- works for "
                   "PDB-derived structures, not bare XYZ.",
    })
    frozen_indices: List[int] = field(default_factory=list, metadata={
        "section": "Frozen atoms",
        "workflow_group": "profile",
        "label":   "Freeze by atom index",
        "engine_key": '(molbuilder: frozen-atom filter by explicit index, sidecar-bridged)',
        "help":    "comma-separated 0-based atom indices, optionally "
                   "with ranges (e.g. \"0-35, 100, 150-200\").  Use "
                   "when you need finer control than element- or "
                   "residue-level freezing.",
    })

    # ----------------- Spectrum -----------------

    compute_raman: bool = field(default=True, metadata={
        "section": "Spectrum",
        "workflow_group": "profile",
        "label":   "Compute Raman activities",
        "engine_key": '(molbuilder: finite-diff polarizability path)',
        "help":    "compute Raman scattering intensity for every "
                   "mode.  Cost: roughly 6 × (number of free atoms) "
                   "extra SCF calculations (one per ±finite-difference "
                   "displacement in each Cartesian direction).  Turn "
                   "off if you only want vibrational frequencies "
                   "(faster, but no spectrum y-axis).",
    })
    compute_ir: bool = field(default=False, metadata={
        "section": "Spectrum",
        "workflow_group": "profile",
        "label":   "Compute IR intensities (scaffold; not yet validated)",
        "tier":    "advanced",
        "engine_key": '(molbuilder: finite-diff dipole-moment derivative path)',
        "help":    "Compute IR absorption intensities (km/mol) from "
                   "finite-difference dipole-moment derivatives.  "
                   "v1 constraint: this rides on the Raman FD loop, "
                   "so it requires compute_raman = True (otherwise "
                   "you'd pay for displaced SCFs twice).  "
                   "NOTE: the IR scaffold is wired end-to-end but the "
                   "absolute km/mol values have NOT been validated "
                   "against an external code (Gaussian/ORCA).  Use "
                   "for relative intensities and qualitative analysis "
                   "until the validation is complete; see docs/tabs/"
                   "spectra/spec.md §13.1 for status.",
    })
    displacement_amplitude_ang: float = field(default=0.02, metadata={
        "section": "Spectrum",
        "workflow_group": "stage",
        "label":   "Displacement amplitude",
        "unit":    "Å",
        "range":   (0.02, 0.30),
        "tier":    "advanced",
        "engine_key": '(molbuilder: finite-difference step amplitude)',
        "help":    "how far atoms are pushed along each mode "
                   "eigenvector when probing how the orbitals shift "
                   "(only used by the per-mode electronic-structure "
                   "step).  The default 0.02 Å sits inside the "
                   "linear-response regime where ΔE_orbital is "
                   "proportional to the displacement, so the slope "
                   "you extract is the physically meaningful number "
                   "rather than a finite-difference-amplitude "
                   "artefact.  Trade-off: orbital-energy differences "
                   "shrink to ~meV at small amplitude and need a "
                   "tight SCF tolerance (default ``conv_tol=1e-10`` "
                   "is sufficient).  Above ~0.10 Å anharmonic "
                   "mixing starts to contaminate the response (see "
                   "[Mills1972] §2.4); above ~0.20 Å the linear-"
                   "response assumption breaks outright.",
    })

    # ----------------- Electronic structure -----------------
    #
    # Model 2 selector (spec § 8) -- the user picks ONE of:
    #   none / all / top_n / threshold / explicit
    # The corresponding value field (es_top_n / es_threshold /
    # es_explicit_indices) only matters when its selector is chosen.
    # The compatibility engine in the JS form locks the inactive
    # value fields so the user can't set conflicting state.

    es_mode_selection: str = field(default="skip", metadata={
        "section": "Electronic structure",
        "workflow_group": "stage",
        "label":   "Mode selection",
        "id_suffix": "es-selection",
        "choices": ("skip", "all", "top_n", "threshold", "explicit"),
        "engine_key": '(molbuilder: per-mode electronic-structure selector)',
        "help":    "which vibrational modes get the displaced-"
                   "geometry orbital-energy probe.  Each chosen mode "
                   "costs two extra SCF calculations (one at +A, one "
                   "at -A along the mode), so this is the most "
                   "expensive part of the run and its cost scales "
                   "linearly with how many modes you pick.\n\n"
                   "    skip      -- don't run this step at all; "
                   "you get a spectrum but no per-mode HOMO/LUMO "
                   "data.  Use this for first-pass exploration.\n"
                   "    all       -- every mode (cost ≈ 2·N modes).\n"
                   "    top_n     -- the N modes with the strongest "
                   "Raman activity.\n"
                   "    threshold -- every mode whose Raman activity "
                   "exceeds your cutoff.\n"
                   "    explicit  -- you list specific mode numbers.\n\n"
                   "Caveat: top_n and threshold rank by Raman "
                   "brightness, which is NOT the same as "
                   "electron-phonon coupling strength -- a mode "
                   "that's transport-critical can be Raman-weak.  "
                   "See [Galperin2007].  When in doubt, use explicit "
                   "(after looking at the spectrum) or all.",
    })
    es_top_n: int = field(default=10, metadata={
        "section": "Electronic structure",
        "workflow_group": "stage",
        "label":   "Top-N modes",
        "range":   (1, 1000),
        "tier":    "advanced",
        "engine_key": '(molbuilder: per-mode selector parameter)',
        "help":    "(only used when selector = top_n) how many of "
                   "the brightest Raman-active modes to compute "
                   "orbital-energy data for.  Cost grows linearly: "
                   "N modes = 2·N SCF calculations.",
    })
    es_threshold: float = field(default=1.0, metadata={
        "section": "Electronic structure",
        "workflow_group": "stage",
        "label":   "Raman-activity threshold",
        "unit":    "Å⁴/amu",
        "range":   (0.0, 1000.0),
        "tier":    "advanced",
        "engine_key": '(molbuilder: per-mode selector parameter)',
        "help":    "(only used when selector = threshold) Raman "
                   "activity cutoff in Å⁴/amu; every mode brighter "
                   "than this gets orbital-energy data.  Final mode "
                   "count is unpredictable -- depends on how many "
                   "modes happen to be above your cutoff.",
    })
    es_explicit_indices: List[int] = field(default_factory=list, metadata={
        "section": "Electronic structure",
        "workflow_group": "stage",
        "label":   "Explicit modes",
        "tier":    "advanced",
        "engine_key": '(molbuilder: per-mode selector parameter)',
        "help":    "(only used when selector = explicit) comma-"
                   "separated list of 1-based mode numbers to "
                   "compute orbital-energy data for, e.g. "
                   "\"3, 7, 12\".  Ranges supported: "
                   "\"3-7, 12\".  Typical workflow: run with "
                   "selector = skip first, look at the spectrum, "
                   "then re-run with selector = explicit and the "
                   "modes you care about.",
    })
    # Frequency-range filter (spec § 2.5.4 + § 8.1).  Composes with
    # the selector above: restricts the selector's output to modes
    # within [freq_min, freq_max].  Either bound = None removes that
    # side; both = None means no filter.  Ignored when selector =
    # "explicit" (user named specific modes, the window doesn't
    # override).  Useful for targeting a chemically interesting
    # window (e.g., 2800-3200 cm⁻¹ for C-H stretches) -- pays off
    # at L4 only (L2 + L3 are fixed-cost).
    freq_min_cm1: Optional[float] = field(default=None, metadata={
        "section":    "Electronic structure",
        "workflow_group": "stage",
        "label":      "Min frequency",
        "unit":       "cm⁻¹",
        "null_label": "(no lower bound)",
        "tier":       "advanced",
        "engine_key": '(molbuilder: per-mode frequency filter)',
        "help":       "restrict orbital-energy data to modes at or "
                      "above this frequency.  Useful for skipping "
                      "low-frequency rocking / librational modes that "
                      "are often noisy and rarely matter for "
                      "transport.  Caveat: filtering may skip modes "
                      "whose strong electron-phonon coupling lies "
                      "outside your chosen window (see [Galperin2007]).  "
                      "Ignored when selector = explicit.",
    })
    freq_max_cm1: Optional[float] = field(default=None, metadata={
        "section":    "Electronic structure",
        "workflow_group": "stage",
        "label":      "Max frequency",
        "unit":       "cm⁻¹",
        "null_label": "(no upper bound)",
        "tier":       "advanced",
        "engine_key": '(molbuilder: per-mode frequency filter)',
        "help":       "restrict orbital-energy data to modes at or "
                      "below this frequency.  Combine with Min "
                      "frequency to target a specific spectral "
                      "window (e.g. 2800-3200 cm⁻¹ for C-H stretches).  "
                      "Ignored when selector = explicit.",
    })
    es_n_homo_below: int = field(default=5, metadata={
        "section": "Electronic structure",
        "workflow_group": "stage",
        "label":   "Orbitals below HOMO to save",
        "id_suffix": "es-n-homo-below",
        "range":   (0, 50),
        "tier":    "advanced",
        "engine_key": '(molbuilder: orbital-window record size)',
        "help":    "how many frontier orbitals BELOW the HOMO to "
                   "record at each displaced geometry.  Five is "
                   "enough to study HOMO/LUMO behaviour for "
                   "transport; raise it to see a richer slice of "
                   "the orbital landscape (e.g. for "
                   "density-of-states plots).  Doesn't change cost.",
    })
    es_n_lumo_above: int = field(default=5, metadata={
        "section": "Electronic structure",
        "workflow_group": "stage",
        "label":   "Orbitals above LUMO to save",
        "id_suffix": "es-n-lumo-above",
        "range":   (0, 50),
        "tier":    "advanced",
        "engine_key": '(molbuilder: orbital-window record size)',
        "help":    "how many frontier orbitals ABOVE the LUMO to "
                   "record at each displaced geometry.  Five matches "
                   "the HOMO setting for a symmetric window around "
                   "the gap.  Doesn't change cost.",
    })

    # ----------------- SCF -----------------

    scf_conv_tol: float = field(default=1e-9, metadata={
        "section": "SCF",
        "workflow_group": "stage",
        "label":   "SCF energy convergence (scf.conv_tol)",
        "unit":    "Hartree",
        "range":   (1e-12, 1e-4),
        "tier":    "advanced",
        "engine_key": 'mf.conv_tol',
        "help":    "tight stopping criterion for each self-consistent "
                   "field calculation.  1e-9 Ha is the standard "
                   "production setting for vibrational analysis (the "
                   "Hessian eigenvalues are sensitive to SCF noise; "
                   "looser tolerances let frequency error grow into "
                   "the cm⁻¹ range).  Tighten to 1e-10 if you suspect "
                   "frequency noise; loosen to 1e-7 for cheaper "
                   "exploratory runs.",
    })
    scf_max_cycle: int = field(default=100, metadata={
        "section": "SCF",
        "workflow_group": "budget",
        "label":   "Max SCF iterations (scf.max_cycle)",
        "range":   (10, 1000),
        "tier":    "advanced",
        "engine_key": 'mf.max_cycle',
        "help":    "cap on iterations PER self-consistent calculation.  "
                   "If the SCF still hasn't converged at this point "
                   "the run aborts with a clear error.  100 is "
                   "generous; if a particular geometry struggles to "
                   "converge it's usually a hint that the geometry "
                   "or charge is wrong, not that you need more "
                   "iterations.",
    })
    grid_level: int = field(default=4, metadata={
        "section": "SCF",
        "workflow_group": "stage",
        "label":   "DFT integration grid level",
        "range":   (0, 9),
        "tier":    "advanced",
        "engine_key": 'mf.grids.level',
        "help":    "DFT exchange-correlation integrals are evaluated "
                   "on a numerical grid; this is the radial/angular "
                   "density.  0 = coarsest (smoke tests only), 3 = "
                   "fast screening, 4 = production minimum for hybrid "
                   "functionals (B3LYP, PBE0, M06...), 5 = tight, "
                   "9 = ultra-tight reference quality.  Cost roughly "
                   "doubles per level; for vibrational analysis with "
                   "a hybrid functional level 4 is the recommended "
                   "lower bound -- below that, grid noise becomes "
                   "the dominant frequency error.",
    })

    # ----------------- Runtime -----------------

    # Runtime/performance knobs are NOT tier="advanced": they
    # directly govern the resource load of the run (a wrong-by-
    # default thread count can yield 2x oversubscription on a
    # hyperthreaded host -- see design.md threading notes).  Keep
    # them visually prominent on the form.
    max_memory_mb: int = field(default=4000, metadata={
        "section":  "Runtime",
        "workflow_group": "budget",
        "label":    "Max memory (max_memory)",
        "unit":     "MB",
        "id_suffix": "max-memory",
        "range":    (100, 1_000_000),
        "engine_key": 'mol.max_memory',
        "help":     "memory budget the SCF code is allowed to use "
                    "for intermediates (ERI tensors, density-fit "
                    "auxiliaries, etc.).  Larger values let bigger "
                    "molecules fit in memory; raise this if you see "
                    "out-of-memory errors with a 50+ atom system.",
    })
    threads: Optional[int] = field(default=None, metadata={
        "section":    "Runtime",
        "workflow_group": "budget",
        "label":      "CPU threads",
        "null_label": "(auto: physical cores)",
        "engine_key": "lib.num_threads(N) + os.environ['OMP_NUM_THREADS']",
        "help":       "how many CPU threads PySCF uses.  Default "
                      "(blank) auto-detects PHYSICAL cores (not "
                      "logical/HT) -- hyperthreading rarely helps "
                      "QC kernels and can hurt cache locality.  The "
                      "emitted script also pins BLAS to 1 thread "
                      "per worker (OPENBLAS_NUM_THREADS=1, "
                      "MKL_NUM_THREADS=1) so PySCF threads * BLAS "
                      "threads don't multiply -- the canonical "
                      "anti-oversubscription recipe.  Set explicitly "
                      "to bench, or to leave cores free for other "
                      "jobs on a shared node.",
    })
    use_gpu: bool = field(default=False, metadata={
        "section": "Runtime",
        "workflow_group": "budget",
        "label":   "Use GPU (NVIDIA, via gpu4pyscf)",
        "id_suffix": "use-gpu",
        "engine_key": 'gpu4pyscf: mf = mf.to_gpu()',
        "help":    "run the SCF and Hessian on an NVIDIA GPU via the "
                   "gpu4pyscf extension (\"pip install "
                   "gpu4pyscf-cuda12x\" on the machine that runs the "
                   "script).  Speed-up is typically 10-50× over a "
                   "16-core CPU for hybrid DFT.  The generated script "
                   "tries gpu4pyscf at runtime and falls back to CPU "
                   "PySCF automatically if it isn't installed, so "
                   "leaving this on is safe even when the script may "
                   "eventually run on a CPU-only node.  NVIDIA-only "
                   "(AMD/Intel GPUs are not currently supported).  "
                   "The Raman polarizability step stays on CPU "
                   "regardless because gpu4pyscf doesn't yet expose "
                   "analytic polarizability.",
    })
    verbose: int = field(default=4, metadata={
        "section": "Runtime",
        "workflow_group": "budget",
        "label":   "Log verbosity (verbose)",
        "range":   (0, 9),
        "tier":    "advanced",
        "engine_key": 'mol.verbose',
        "help":    "how much detail the SCF prints to stdout.  "
                   "0 = silent, 3 = warnings only, 4 = standard "
                   "(SCF cycle table + final energy), 5 = debug.  "
                   "Higher levels produce more diagnostic output "
                   "but slow stdout-bound runs slightly.",
    })
    verbose_comments: bool = field(default=True, metadata={
        "section": "Runtime",
        "workflow_group": "profile",
        "label":   "Verbose comments in generated script",
        "engine_key": '(molbuilder: emit explanatory comments into spectra.py)',
        "help":    "embed inline scientific explanations + the "
                   "Methods-section prose + literature citations "
                   "into the generated spectra.py.  Leaving this on "
                   "makes the script self-documenting: a colleague "
                   "reading it sees what the choices mean and why.  "
                   "Turn off only if you want a stripped-down "
                   "minimal-comment script.",
    })

    def __post_init__(self) -> None:
        # Reject any string-valued field whose value is outside its
        # declared `choices` tuple.  Without this, a typo (e.g. the
        # pre-rename `es_mode_selection="none"`) is accepted silently
        # and the downstream script's selector branches no-op, leaving
        # a phase marked "complete" with zero modes selected.
        for f in fields(self):
            choices = f.metadata.get("choices")
            if not choices:
                continue
            value = getattr(self, f.name)
            if value is None:  # Optional[str] fields legitimately allow None
                continue
            if value not in choices:
                raise ValueError(
                    f"{f.name}={value!r} is not one of "
                    f"{tuple(choices)}"
                )


__all__ = ["SpectraConfig"]
