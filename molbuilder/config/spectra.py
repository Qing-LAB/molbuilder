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
      (``es_mode_selection = "none"``) so a first-pass run is
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
        "help":    "use density fitting (RIJK) for the Coulomb / "
                   "exchange evaluation -- 5-10x SCF speedup with "
                   "negligible accuracy loss for organics",
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
        "help":    "comma-separated element symbols whose atoms stay "
                   "frozen during the Hessian (e.g. \"Au\" to fix a "
                   "metal slab in a metal-molecule-metal junction)",
    })
    fixed_residue_names: List[str] = field(default_factory=list, metadata={
        "section": "Frozen atoms",
        "label":   "Fixed by residue name",
        "help":    "comma-separated PDB residue names whose atoms stay "
                   "frozen (e.g. \"ALA,GLY\" to fix specific peptide "
                   "residues); requires a structure with residue info",
    })
    fixed_indices: List[int] = field(default_factory=list, metadata={
        "section": "Frozen atoms",
        "label":   "Fixed by atom index",
        "help":    "comma-separated 0-based atom indices, optionally "
                   "with ranges (e.g. \"0-35, 100, 150-200\")",
    })

    # ----------------- Spectrum -----------------

    compute_raman: bool = field(default=True, metadata={
        "section": "Spectrum",
        "label":   "Compute Raman activities",
        "help":    "compute analytic polarizability derivatives "
                   "[Komornicki1979] and project onto each mode to "
                   "get Raman activities; False runs only the Hessian "
                   "(diagnostic / wavenumber-only)",
    })
    compute_ir: bool = field(default=False, metadata={
        "section": "Spectrum",
        "label":   "Compute IR intensities (reserved; v1.2)",
        "tier":    "advanced",
        "help":    "RESERVED -- the IR add-on (dipole derivatives) is "
                   "scheduled for the 1c milestone; this field is in "
                   "the schema so its arrival doesn't change the wire "
                   "shape, but the engine ignores True in v1",
    })
    displacement_amplitude_ang: float = field(default=0.10, metadata={
        "section": "Spectrum",
        "label":   "Displacement amplitude",
        "unit":    "Å",
        "range":   (0.02, 0.30),
        "tier":    "advanced",
        "help":    "±A·Q_i along each mode's eigenvector for the "
                   "per-mode electronic-structure SCFs; 0.05-0.15 Å "
                   "is the contemporary-practice range (above ~0.20 Å "
                   "anharmonic mixing becomes non-negligible, see "
                   "[Mills1972] for the general framework; below "
                   "~0.04 Å the ΔE_HOMO noise from the SCF tolerance "
                   "tends to dominate)",
    })

    # ----------------- Electronic structure -----------------
    #
    # Model 2 selector (spec § 8) -- the user picks ONE of:
    #   none / all / top_n / threshold / explicit
    # The corresponding value field (es_top_n / es_threshold /
    # es_explicit_indices) only matters when its selector is chosen.
    # The compatibility engine in the JS form locks the inactive
    # value fields so the user can't set conflicting state.

    es_mode_selection: str = field(default="none", metadata={
        "section": "Electronic structure",
        "label":   "Mode selection",
        "id_suffix": "es-selection",
        "choices": ("none", "all", "top_n", "threshold", "explicit"),
        "help":    "which modes get per-mode electronic-structure data "
                   "(2 SCFs per selected mode): none = spectrum only; "
                   "all = every mode; top_n / threshold = prune by "
                   "Raman activity (caveat: misses modes with weak "
                   "Raman but strong electron-phonon coupling, see "
                   "[Galperin2007]); explicit = user picks indices",
    })
    es_top_n: int = field(default=10, metadata={
        "section": "Electronic structure",
        "label":   "Top-N modes",
        "range":   (1, 1000),
        "tier":    "advanced",
        "help":    "(when selector=top_n) number of highest-Raman-"
                   "activity modes that get ES data",
    })
    es_threshold: float = field(default=1.0, metadata={
        "section": "Electronic structure",
        "label":   "Raman-activity threshold",
        "unit":    "Å⁴/amu",
        "range":   (0.0, 1000.0),
        "tier":    "advanced",
        "help":    "(when selector=threshold) only modes with Raman "
                   "activity above this value get ES data",
    })
    es_explicit_indices: List[int] = field(default_factory=list, metadata={
        "section": "Electronic structure",
        "label":   "Explicit modes",
        "tier":    "advanced",
        "help":    "(when selector=explicit) comma-separated 1-based "
                   "mode indices; the natural two-stage workflow is to "
                   "run with selector=none first, see the spectrum, "
                   "then re-run with selector=explicit listing the "
                   "modes of interest",
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
        "help":       "constrain L4 mode selection to modes with "
                      "frequency >= this value; useful for skipping "
                      "low-frequency rocking / librational modes that "
                      "are often noisy and rarely transport-relevant. "
                      "Caveat: filtering may skip modes whose strong "
                      "electron-phonon coupling lies outside the "
                      "window [Galperin2007].  Ignored when "
                      "selector=explicit.",
    })
    freq_max_cm1: Optional[float] = field(default=None, metadata={
        "section":    "Electronic structure",
        "label":      "Max frequency",
        "unit":       "cm⁻¹",
        "null_label": "(no upper bound)",
        "tier":       "advanced",
        "help":       "constrain L4 mode selection to modes with "
                      "frequency <= this value; combine with "
                      "freq_min_cm1 to focus on a window.  Ignored "
                      "when selector=explicit.",
    })
    es_n_homo_below: int = field(default=5, metadata={
        "section": "Electronic structure",
        "label":   "HOMO-N (orbitals below HOMO)",
        "id_suffix": "es-n-homo-below",
        "range":   (0, 50),
        "tier":    "advanced",
        "help":    "number of orbitals BELOW HOMO recorded at each "
                   "displaced geometry; 5 is enough for transport "
                   "prep, raise for richer DOS visualisation",
    })
    es_n_lumo_above: int = field(default=5, metadata={
        "section": "Electronic structure",
        "label":   "LUMO+M (orbitals above LUMO)",
        "id_suffix": "es-n-lumo-above",
        "range":   (0, 50),
        "tier":    "advanced",
        "help":    "number of orbitals ABOVE LUMO recorded at each "
                   "displaced geometry",
    })

    # ----------------- SCF -----------------

    scf_conv_tol: float = field(default=1e-9, metadata={
        "section": "SCF",
        "label":   "scf.conv_tol",
        "unit":    "Hartree",
        "range":   (1e-12, 1e-4),
        "tier":    "advanced",
        "help":    "SCF convergence tolerance on the energy",
    })
    scf_max_cycle: int = field(default=100, metadata={
        "section": "SCF",
        "label":   "scf.max_cycle",
        "range":   (10, 1000),
        "tier":    "advanced",
        "help":    "max SCF cycles per single-point",
    })
    grid_level: int = field(default=4, metadata={
        "section": "SCF",
        "label":   "DFT grid level",
        "range":   (0, 9),
        "tier":    "advanced",
        "help":    "0=coarse, 3=screening, 4=default (hybrid-friendly), "
                   "5=tight, 9=ultra; v1 spec § 11.4 warns when < 4 "
                   "with a hybrid functional",
    })

    # ----------------- Runtime -----------------

    max_memory_mb: int = field(default=4000, metadata={
        "section":  "Runtime",
        "label":    "max_memory",
        "unit":     "MB",
        "id_suffix": "max-memory",
        "range":    (100, 1_000_000),
        "tier":     "advanced",
        "help":     "memory hint passed to PySCF",
    })
    threads: Optional[int] = field(default=None, metadata={
        "section":    "Runtime",
        "label":      "Threads",
        "null_label": "(inherit OMP_NUM_THREADS)",
        "tier":       "advanced",
        "help":       "OMP_NUM_THREADS pin; None inherits from the env",
    })
    verbose: int = field(default=4, metadata={
        "section": "Runtime",
        "label":   "PySCF verbose",
        "range":   (0, 9),
        "tier":    "advanced",
        "help":    "PySCF log level: 0 silent, 4 info (default), 5 debug",
    })
    verbose_comments: bool = field(default=True, metadata={
        "section": "Runtime",
        "label":   "Verbose comments in script",
        "help":    "emit inline tuning hints + Methods-paragraph "
                   "docstring + citation keys in the generated "
                   "spectra.py (publication-quality default; see "
                   "spec § 11)",
    })


__all__ = ["SpectraConfig"]
