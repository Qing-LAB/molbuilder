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
    * Displacement amplitude 0.10 Å -- a defensible production
      value (anharmonic-cubic mixing < 1 % per Mills 1972 §2.4;
      finite-difference noise on ΔE_HOMO suppressed).
    * Per-mode electronic structure off by default
      (``es_mode_selection = "skip"``) so a first-pass run is
      cheap; users opt in to ``top_n`` / ``explicit`` after they
      see the Raman spectrum and pick modes of interest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    # ----------------- System -----------------

    engine: str = field(default="pyscf", metadata={
        "section": "System",
        "label":   "Engine",
        # Only PySCF in v1.  SIESTA is reserved (see
        # spec.md § 13.2); adding it later extends this tuple by
        # one entry and the form/CLI/validator pipelines pick up
        # the new choice automatically.
        "choices": ("pyscf",),
        "help":    "computational engine that runs the Hessian, "
                   "polarizability derivatives, and the per-mode "
                   "displaced-geometry SCFs",
    })
    job_name: str = field(default="spectra", metadata={
        "section":  "System",
        "label":    "Job name",
        "id_suffix": "job-name",
        "pattern":  r"^[A-Za-z0-9_\-]+$",
        "help":     "filesystem-safe basename for emitted files "
                    "(spectra.py + spectra.json + thermo.txt).  "
                    "Same rule as SIESTA SystemLabel / PySCF job_name "
                    "-- see docs/protocols/job-layout.md.",
        "validate": _validate_basename("job_name"),
    })

    # ----------------- Method -----------------

    method: str = field(default="RKS", metadata={
        "section": "Method",
        "label":   "SCF method",
        "choices": ("RKS", "UKS", "RHF", "UHF"),
        "help":    "RKS / UKS / RHF / UHF",
    })
    functional: str = field(default="B3LYP", metadata={
        "section": "Method",
        "label":   "Functional",
        "help":    "XC functional name (libxc string); B3LYP is the "
                   "modern default for organic / biomolecule chemistry",
    })
    basis: str = field(default="def2-SVP", metadata={
        "section": "Method",
        "label":   "Basis set",
        "help":    "Gaussian basis set; def2-SVP is the production "
                   "minimum, def2-TZVP for accurate work",
    })
    dispersion: Optional[str] = field(default="d3bj", metadata={
        "section":    "Method",
        "label":      "Dispersion",
        "choices":    ("d3", "d3bj", "d4", "none"),
        "null_label": "(none)",
        "help":       "dispersion correction; D3 with Becke-Johnson "
                      "damping is the modern default [Grimme2011]",
    })
    density_fit: bool = field(default=True, metadata={
        "section": "Method",
        "label":   "Density fitting",
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
    # Three filters combined with UNION semantics: an atom is fixed
    # if it matches ANY of element / residue-name / explicit-index.
    # See spec.md § 7 for the algorithm.
    #
    # The three lists are stored as native Python types here; the
    # form sends comma-separated strings, which the server's
    # `_shared.coerce_to_field_type` splits / parses (existing
    # Sequence[str] handler covers fixed_elements + fixed_residue_names;
    # fixed_indices is parsed by a small helper at config-from-params
    # time -- see the engine's `preflight`).

    fixed_elements: List[str] = field(default_factory=list, metadata={
        "section": "Frozen atoms",
        "label":   "Fixed by element",
        "help":    "comma-separated element symbols whose atoms are "
                   "held in place during the vibrational analysis "
                   "(e.g. \"Au\" to freeze a gold electrode in a "
                   "metal-molecule-metal junction).  Freezing atoms "
                   "removes their contribution to the Hessian and "
                   "reduces the mode count; it does NOT remove them "
                   "from the SCF or from the displaced-geometry "
                   "calculations.",
    })
    fixed_residue_names: List[str] = field(default_factory=list, metadata={
        "section": "Frozen atoms",
        "label":   "Fixed by residue name",
        "help":    "comma-separated PDB residue names whose atoms "
                   "are held in place (e.g. \"ALA,GLY\" to freeze "
                   "specific peptide residues).  Requires the input "
                   "structure to carry residue labels -- works for "
                   "PDB-derived structures, not bare XYZ.",
    })
    fixed_indices: List[int] = field(default_factory=list, metadata={
        "section": "Frozen atoms",
        "label":   "Fixed by atom index",
        "help":    "comma-separated 0-based atom indices, optionally "
                   "with ranges (e.g. \"0-35, 100, 150-200\").  Use "
                   "when you need finer control than element- or "
                   "residue-level freezing.",
    })

    # ----------------- Spectrum -----------------

    compute_raman: bool = field(default=True, metadata={
        "section": "Spectrum",
        "label":   "Compute Raman activities",
        "help":    "compute Raman scattering intensity for every "
                   "mode.  Cost: roughly 6 × (number of free atoms) "
                   "extra SCF calculations (one per ±finite-difference "
                   "displacement in each Cartesian direction).  Turn "
                   "off if you only want vibrational frequencies "
                   "(faster, but no spectrum y-axis).",
    })
    compute_ir: bool = field(default=False, metadata={
        "section": "Spectrum",
        "label":   "Compute IR intensities (not yet implemented)",
        "tier":    "advanced",
        "help":    "Reserved for a future release that will add IR "
                   "absorption intensities (dipole-moment "
                   "derivatives).  This checkbox does nothing today; "
                   "leaving it off avoids surprises when it activates "
                   "in a later version.",
    })
    displacement_amplitude_ang: float = field(default=0.10, metadata={
        "section": "Spectrum",
        "label":   "Displacement amplitude",
        "unit":    "Å",
        "range":   (0.02, 0.30),
        "tier":    "advanced",
        "help":    "how far atoms are pushed along each mode "
                   "eigenvector when probing how the orbitals shift "
                   "(only used by the per-mode electronic-structure "
                   "step).  Larger amplitude = more sensitivity to "
                   "the mode but more anharmonic contamination of "
                   "the linear response; smaller amplitude = cleaner "
                   "but noisier orbital-energy differences.  0.05-"
                   "0.15 Å is the contemporary-practice range; above "
                   "~0.20 Å anharmonic mixing becomes non-negligible "
                   "(see [Mills1972] for the general framework); "
                   "below ~0.04 Å the orbital-energy noise from the "
                   "SCF tolerance tends to dominate the signal.",
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
        "label":   "Mode selection",
        "id_suffix": "es-selection",
        "choices": ("skip", "all", "top_n", "threshold", "explicit"),
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
        "label":   "Top-N modes",
        "range":   (1, 1000),
        "tier":    "advanced",
        "help":    "(only used when selector = top_n) how many of "
                   "the brightest Raman-active modes to compute "
                   "orbital-energy data for.  Cost grows linearly: "
                   "N modes = 2·N SCF calculations.",
    })
    es_threshold: float = field(default=1.0, metadata={
        "section": "Electronic structure",
        "label":   "Raman-activity threshold",
        "unit":    "Å⁴/amu",
        "range":   (0.0, 1000.0),
        "tier":    "advanced",
        "help":    "(only used when selector = threshold) Raman "
                   "activity cutoff in Å⁴/amu; every mode brighter "
                   "than this gets orbital-energy data.  Final mode "
                   "count is unpredictable -- depends on how many "
                   "modes happen to be above your cutoff.",
    })
    es_explicit_indices: List[int] = field(default_factory=list, metadata={
        "section": "Electronic structure",
        "label":   "Explicit modes",
        "tier":    "advanced",
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
        "label":      "Min frequency",
        "unit":       "cm⁻¹",
        "null_label": "(no lower bound)",
        "tier":       "advanced",
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
        "label":      "Max frequency",
        "unit":       "cm⁻¹",
        "null_label": "(no upper bound)",
        "tier":       "advanced",
        "help":       "restrict orbital-energy data to modes at or "
                      "below this frequency.  Combine with Min "
                      "frequency to target a specific spectral "
                      "window (e.g. 2800-3200 cm⁻¹ for C-H stretches).  "
                      "Ignored when selector = explicit.",
    })
    es_n_homo_below: int = field(default=5, metadata={
        "section": "Electronic structure",
        "label":   "Orbitals below HOMO to save",
        "id_suffix": "es-n-homo-below",
        "range":   (0, 50),
        "tier":    "advanced",
        "help":    "how many frontier orbitals BELOW the HOMO to "
                   "record at each displaced geometry.  Five is "
                   "enough to study HOMO/LUMO behaviour for "
                   "transport; raise it to see a richer slice of "
                   "the orbital landscape (e.g. for "
                   "density-of-states plots).  Doesn't change cost.",
    })
    es_n_lumo_above: int = field(default=5, metadata={
        "section": "Electronic structure",
        "label":   "Orbitals above LUMO to save",
        "id_suffix": "es-n-lumo-above",
        "range":   (0, 50),
        "tier":    "advanced",
        "help":    "how many frontier orbitals ABOVE the LUMO to "
                   "record at each displaced geometry.  Five matches "
                   "the HOMO setting for a symmetric window around "
                   "the gap.  Doesn't change cost.",
    })

    # ----------------- SCF -----------------

    scf_conv_tol: float = field(default=1e-9, metadata={
        "section": "SCF",
        "label":   "SCF energy convergence (scf.conv_tol)",
        "unit":    "Hartree",
        "range":   (1e-12, 1e-4),
        "tier":    "advanced",
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
        "label":   "Max SCF iterations (scf.max_cycle)",
        "range":   (10, 1000),
        "tier":    "advanced",
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
        "label":   "DFT integration grid level",
        "range":   (0, 9),
        "tier":    "advanced",
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

    max_memory_mb: int = field(default=4000, metadata={
        "section":  "Runtime",
        "label":    "Max memory (max_memory)",
        "unit":     "MB",
        "id_suffix": "max-memory",
        "range":    (100, 1_000_000),
        "tier":     "advanced",
        "help":     "memory budget the SCF code is allowed to use "
                    "for intermediates (ERI tensors, density-fit "
                    "auxiliaries, etc.).  Larger values let bigger "
                    "molecules fit in memory; raise this if you see "
                    "out-of-memory errors with a 50+ atom system.",
    })
    threads: Optional[int] = field(default=None, metadata={
        "section":    "Runtime",
        "label":      "CPU threads",
        "null_label": "(inherit from the environment)",
        "tier":       "advanced",
        "help":       "how many CPU threads to use.  Leave blank to "
                      "inherit from OMP_NUM_THREADS (or auto-detect).  "
                      "Set explicitly if your cluster scheduler "
                      "doesn't propagate the env, or to leave cores "
                      "free for other jobs.",
    })
    use_gpu: bool = field(default=False, metadata={
        "section": "Runtime",
        "label":   "Use GPU (NVIDIA, via gpu4pyscf)",
        "id_suffix": "use-gpu",
        "tier":    "advanced",
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
        "label":   "Log verbosity (verbose)",
        "range":   (0, 9),
        "tier":    "advanced",
        "help":    "how much detail the SCF prints to stdout.  "
                   "0 = silent, 3 = warnings only, 4 = standard "
                   "(SCF cycle table + final energy), 5 = debug.  "
                   "Higher levels produce more diagnostic output "
                   "but slow stdout-bound runs slightly.",
    })
    verbose_comments: bool = field(default=True, metadata={
        "section": "Runtime",
        "label":   "Verbose comments in generated script",
        "help":    "embed inline scientific explanations + the "
                   "Methods-section prose + literature citations "
                   "into the generated spectra.py.  Leaving this on "
                   "makes the script self-documenting: a colleague "
                   "reading it sees what the choices mean and why.  "
                   "Turn off only if you want a stripped-down "
                   "minimal-comment script.",
    })


__all__ = ["SpectraConfig"]
