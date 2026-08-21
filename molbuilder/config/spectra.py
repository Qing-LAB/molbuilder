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

The full design contract is in ``docs/web/spectra.md`` § 4.

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
      contemporary-practice context (0.02-0.20 Å).
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
        # Only PySCF in v1.  SIESTA is reserved (see
        # spec.md § 13.2); adding it later extends this tuple by
        # one entry and the form/CLI/validator pipelines pick up
        # the new choice automatically.
        "choices": ("pyscf",),
    })
    job_name: str = field(default="spectra", metadata={
        "pattern":  r"^[A-Za-z0-9_\-]+$",
        "validate": _validate_basename("job_name"),
    })

    # ----------------- Method -----------------

    method: str = field(default="RKS", metadata={
        "choices": ("RKS", "UKS", "RHF", "UHF"),
    })
    charge: int = field(default=0, metadata={
        "range":   (-10, 10),
    })
    spin: int = field(default=0, metadata={
        "range":   (0, 10),
    })
    functional: str = field(default="B3LYP", metadata={
    })
    basis: str = field(default="def2-SVP", metadata={
    })
    # ECP: the SAME two plain fields as PySCFConfig, rewritten together
    # 2026-08-13 so the siblings cannot drift on the one setting whose
    # old shape was the reason `strmap` existed.  Empty means empty; no
    # Z threshold and no basis family decides anything.
    ecp: str = field(default="", metadata={
    })
    ecp_atoms: List[str] = field(default_factory=list, metadata={
        "skip_cli":   True,
    })
    dispersion: Optional[str] = field(default="d3bj", metadata={
        "choices":    ("d3", "d3bj", "d4", "none"),
    })
    density_fit: bool = field(default=True, metadata={
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
    })
    frozen_residue_names: List[str] = field(default_factory=list, metadata={
    })
    frozen_indices: List[int] = field(default_factory=list, metadata={
    })

    # ----------------- Spectrum -----------------

    compute_raman: bool = field(default=True, metadata={
    })
    compute_ir: bool = field(default=False, metadata={
    })
    displacement_amplitude_ang: float = field(default=0.02, metadata={
        # Recommended range == the engine gate's validated window
        # (pyscf_engine.py warns <0.02 and >0.20): above ~0.20 Å the
        # linear-response assumption breaks (help text below).  Kept in
        # lockstep so the form's out-of-range auto-warn and the science
        # gate agree on the same boundary.
        "range":   (0.02, 0.20),
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
        "choices": ("skip", "all", "top_n", "threshold", "explicit"),
    })
    es_top_n: int = field(default=10, metadata={
        "range":   (1, 1000),
    })
    es_threshold: float = field(default=1.0, metadata={
        "range":   (0.0, 1000.0),
    })
    es_explicit_indices: List[int] = field(default_factory=list, metadata={
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
    })
    freq_max_cm1: Optional[float] = field(default=None, metadata={
    })
    es_n_homo_below: int = field(default=5, metadata={
        "range":   (0, 50),
    })
    es_n_lumo_above: int = field(default=5, metadata={
        "range":   (0, 50),
    })

    # ----------------- SCF -----------------

    scf_conv_tol: float = field(default=1e-9, metadata={
        "range":   (1e-12, 1e-4),
    })
    scf_max_cycle: int = field(default=100, metadata={
        "range":   (10, 1000),
    })
    grid_level: int = field(default=4, metadata={
        "range":   (0, 9),
    })

    # ----------------- Runtime -----------------

    # Runtime/performance knobs are NOT tier="advanced": they
    # directly govern the resource load of the run (a wrong-by-
    # default thread count can yield 2x oversubscription on a
    # hyperthreaded host -- see design.md threading notes).  Keep
    # them visually prominent on the form.
    max_memory_mb: int = field(default=4000, metadata={
        "range":    (100, 1_000_000),
    })
    threads: Optional[int] = field(default=None, metadata={
    })
    use_gpu: bool = field(default=False, metadata={
    })
    verbose: int = field(default=4, metadata={
        "range":   (0, 9),
    })
    verbose_comments: bool = field(default=True, metadata={
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
