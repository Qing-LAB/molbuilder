"""TransportConfig -- every parameter the Transport-tab engine reads.

Engine-agnostic, dataclass-of-fields with metadata, same architecture
as :class:`SpectraConfig`.  The metadata drives:

  * the schema-driven Transport form;
  * the validator pass;
  * the click-based CLI;
  * the Methods-text generator.

Engines (registry surface in :mod:`molbuilder.transport`; concrete
implementations land in Phase B.3):

  * **transiesta** -- TranSIESTA from the SIESTA suite, NEGF + LDA/GGA
    pseudopotentials.  Handles realistic electrode + bridge sizes
    (≤ ~few hundred atoms in the device region).
  * **pyscf-negf** -- PySCF + a custom NEGF self-energy driver,
    Gaussian-basis with hybrid functionals available.  Smaller
    systems (~ tens of atoms in the device region) but higher-level
    XC.

Both consume the SAME relaxed geometry + the SAME ``.molstruct.json``
sidecar (region labels assigned in /modify -- see
:mod:`molbuilder.parsers.molstruct_json`).  The two engines'
``render_script`` methods emit different inputs from the same
TransportConfig + Structure pair.

Future: inelastica integration (electron-phonon-resolved transmission,
IETS).  That's a SEPARATE engine that consumes a TransportConfig +
the .spectra.json from the Spectra tab; not in scope for B.1.

Defaults are chosen for a metal-molecule-metal junction at zero bias:

  * V = 0, transmission window ±2 eV around E_F at 401 points
    (5 meV resolution -- fine enough to resolve typical Au-thiol
    junction features in the ±1 V conductance window).
  * Electronic temperature 300 K (room-T Fermi smearing on the leads).
  * Transverse k-mesh (1, 1, 1) -- finite molecule between leads,
    no Brillouin-zone integration needed.  For 1D-periodic electrodes
    in the transport direction this lifts to (Nx, Ny, 1) where Nx/Ny
    are the lead periodicities.
  * NEGF complex contour: 32 imaginary-axis points + 8 along the
    real axis (per the standard semicircle prescription, see
    Brandbyge et al. 2002 §IV).

The contour-point counts are exposed but tier="advanced" because the
defaults are robust for typical organic-on-metal junctions; users
shouldn't have to touch them unless they see density-of-states
artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Shared with the other config modules -- one regex, one rule for the
# job-layout basename pattern.
from .siesta import _validate_basename


# --------------------------------------------------------------------- #
#  Region-label conventions (canonical strings used by both engines)    #
# --------------------------------------------------------------------- #
#
# These are the keys we expect in the .molstruct.json sidecar's
# ``regions`` map for a 2-terminal junction.  Other tools (modify-tab
# UI, transport engines) import these instead of hard-coding strings,
# so a future rename happens once.

REGION_LEFT_ELECTRODE  = "L-electrode"
REGION_RIGHT_ELECTRODE = "R-electrode"
REGION_BRIDGE          = "bridge"

# The full set of expected region labels for a 2-terminal calculation.
# A sidecar that's missing one of these or includes an unknown label
# is rejected at preflight by the transport engine.
EXPECTED_REGIONS_2T = (
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
    REGION_BRIDGE,
)


# --------------------------------------------------------------------- #
#  TransportConfig                                                      #
# --------------------------------------------------------------------- #


@dataclass
class TransportConfig:
    # Form-section render order.  Mirrors SpectraConfig's pattern.
    _form_section_order = (
        "System",
        "Geometry",
        "Electrodes",
        "Transmission",
        "NEGF",
        "Runtime",
    )

    _form_section_descriptions = {
        "System": (
            "Engine selection and job-name identity.  TranSIESTA "
            "handles larger device regions with pseudopotentials; "
            "PySCF-NEGF supports hybrid functionals on smaller "
            "molecules.  Both consume the same relaxed structure + "
            "region-labelled sidecar."
        ),
        "Geometry": (
            "Input geometry + region labels.  The XYZ is the relaxed "
            "structure from the Build/Modify workflow; the sidecar "
            "JSON carries the L-electrode / R-electrode / bridge "
            "labels assigned in /modify.  Both files must agree on "
            "atom count and ordering -- the transport script verifies "
            "this at script-generation time."
        ),
        "Electrodes": (
            "Bias voltage applied across the junction and the "
            "transverse k-point mesh.  Bias is V_left - V_right in "
            "volts (positive draws electrons left-to-right).  The "
            "k-mesh sums over Brillouin-zone directions perpendicular "
            "to transport; (1, 1, 1) is correct for a finite molecule "
            "between leads."
        ),
        "Transmission": (
            "Energy window over which T(E) is evaluated.  Center at "
            "E_F (relative=True) for cleanest interpretation; the "
            "absolute-energy mode is for cross-engine comparison.  "
            "Resolution ≈ (e_max - e_min) / (n_points - 1)."
        ),
        "NEGF": (
            "Self-energy + contour parameters that the NEGF density "
            "integration uses.  The defaults are robust for typical "
            "organic-on-metal junctions; touch these only if you "
            "see density-of-states artefacts or convergence issues."
        ),
        "Runtime": (
            "How the run uses your hardware -- memory budget, CPU "
            "thread count, log verbosity.  These don't affect the "
            "science, only wall time and diagnostic output."
        ),
    }

    # ----------------- System -----------------

    engine: str = field(default="transiesta", metadata={
        "section": "System",
        "label":   "Engine",
        # PySCF-NEGF requires the PySCF + scipy stack; TranSIESTA
        # requires a working SIESTA-MPI build with TranSIESTA enabled.
        # Both backends will register as TransportEngine via
        # :mod:`molbuilder.transport` in Phase B.3.
        "choices": ("transiesta", "pyscf-negf"),
        "help":    "NEGF transport engine.  TranSIESTA: larger systems "
                   "with pseudopotentials, GGA/LDA only.  PySCF-NEGF: "
                   "smaller systems, all-electron with hybrid XC.",
    })
    job_name: str = field(default="transport", metadata={
        "section":   "System",
        "label":     "Job name",
        "id_suffix": "job-name",
        "pattern":   r"^[A-Za-z0-9_\-]+$",
        "help":      "filesystem-safe basename for emitted files "
                     "(transport.py + transport.json + transmission.dat).  "
                     "Same rule as SIESTA SystemLabel / PySCF job_name "
                     "-- see docs/protocols/job-layout.md.",
        "validate":  _validate_basename("job_name"),
    })

    # ----------------- Geometry -----------------

    structure_xyz_path: str = field(default="", metadata={
        "section": "Geometry",
        "label":   "Relaxed structure (XYZ)",
        "help":    "path to the relaxed-geometry XYZ from /modify.  "
                   "Atom ordering must match the indices stored in "
                   "the .molstruct.json sidecar (the transport "
                   "preflight checks this via the structure_hash).",
    })
    molstruct_json_path: str = field(default="", metadata={
        "section": "Geometry",
        "label":   "Region-label sidecar (.molstruct.json)",
        "help":    "path to the sidecar carrying L-electrode / "
                   "R-electrode / bridge labels.  Generated in /modify "
                   "when the user marks regions; carries a hash of "
                   "the XYZ bytes to detect stale labels.",
    })

    # ----------------- Electrodes -----------------

    bias_voltages_v: List[float] = field(default_factory=lambda: [0.0],
                                          metadata={
        "section": "Electrodes",
        "label":   "Bias voltages (V)",
        "help":    "comma-separated list of bias values V_L - V_R at "
                   "which to compute transmission.  Default [0.0] is "
                   "the zero-bias linear-response calculation (Landauer "
                   "conductance).  Each non-zero bias requires its own "
                   "NEGF density iteration so cost is roughly linear "
                   "in the list length.",
    })
    k_mesh_transverse: List[int] = field(default_factory=lambda: [1, 1, 1],
                                          metadata={
        "section": "Electrodes",
        "label":   "Transverse k-mesh",
        "help":    "Monkhorst-Pack grid (Nx, Ny, Nz) summed over the "
                   "directions perpendicular to transport.  For a "
                   "finite molecule between leads use (1, 1, 1).  For "
                   "1D-periodic electrodes set Nx,Ny to the lead "
                   "periodicities; Nz is the transport direction and "
                   "stays at 1 (NEGF handles that one analytically).",
    })
    electronic_temperature_k: float = field(default=300.0, metadata={
        "section": "Electrodes",
        "label":   "Electronic temperature",
        "unit":    "K",
        "range":   (10.0, 2000.0),
        "tier":    "advanced",
        "help":    "Fermi-Dirac smearing on the lead occupations.  "
                   "Room-T (300 K) is the standard for experimentally-"
                   "comparable transmission curves; lower T sharpens "
                   "the Fermi step and surfaces fine features at the "
                   "cost of NEGF integration stability.",
    })

    # ----------------- Transmission -----------------

    transmission_emin_ev: float = field(default=-2.0, metadata={
        "section": "Transmission",
        "label":   "Energy window min",
        "unit":    "eV",
        "range":   (-10.0, 0.0),
        "help":    "lower edge of the T(E) energy window.  Relative "
                   "to E_F when ``transmission_relative_to_ef`` is on "
                   "(the default).  ±2 eV is the standard window for "
                   "low-bias junction characterisation; widen for high-"
                   "bias or for sigma-states / resonances further from "
                   "E_F.",
    })
    transmission_emax_ev: float = field(default=2.0, metadata={
        "section": "Transmission",
        "label":   "Energy window max",
        "unit":    "eV",
        "range":   (0.0, 10.0),
        "help":    "upper edge of the T(E) energy window.  See "
                   "``transmission_emin_ev`` for sign conventions.",
    })
    transmission_n_points: int = field(default=401, metadata={
        "section": "Transmission",
        "label":   "Energy grid points",
        "range":   (51, 4001),
        "help":    "number of evenly-spaced energies at which T(E) is "
                   "computed.  Resolution = (emax - emin) / (N - 1); "
                   "default 401 over a 4 eV window = 10 meV/point, "
                   "fine enough to resolve typical Au-thiol "
                   "transmission features in the ±1 V conductance "
                   "window.",
    })
    transmission_relative_to_ef: bool = field(default=True, metadata={
        "section": "Transmission",
        "label":   "Energy is relative to E_F",
        "help":    "if on (default), energies are written as E - E_F; "
                   "off writes absolute energies (Hartree-tree, eV).  "
                   "Relative is the clean way to compare junctions "
                   "with different absolute Fermi levels.",
    })

    # ----------------- NEGF -----------------

    contour_n_circle: int = field(default=32, metadata={
        "section": "NEGF",
        "label":   "Contour points (imaginary axis)",
        "range":   (8, 128),
        "tier":    "advanced",
        "help":    "number of Gauss-Legendre nodes on the imaginary-"
                   "axis semicircle that integrates the NEGF density "
                   "matrix below E_F.  Standard prescription "
                   "(Brandbyge et al. 2002 §IV) recommends 32-64 "
                   "for converged densities on typical junctions; "
                   "raise if you see DOS artefacts near the chemical "
                   "potential.",
    })
    contour_n_real: int = field(default=8, metadata={
        "section": "NEGF",
        "label":   "Contour points (real axis)",
        "range":   (4, 64),
        "tier":    "advanced",
        "help":    "number of energy points on the real-axis bias-"
                   "window segment of the NEGF contour.  Only "
                   "relevant at non-zero bias; at V = 0 this segment "
                   "vanishes and the count is ignored.",
    })
    contour_e_bottom_ev: float = field(default=-40.0, metadata={
        "section": "NEGF",
        "label":   "Contour bottom",
        "unit":    "eV",
        "range":   (-100.0, -10.0),
        "tier":    "advanced",
        # Help-text rewrite 2026-06-10 post-review: the prior
        # "Au 5d ~-7 eV is well above this" was confusing because
        # -7 is numerically greater than -40 but in transport
        # context users expect "shallow vs deep" framing.  Restated
        # in terms of binding depth below E_F.
        "help":    "deepest energy on the NEGF complex contour, "
                   "relative to E_F (E_F = 0).  Must be BELOW the "
                   "lowest occupied state in the device region.  "
                   "Default -40 eV is safe for first-row + Au / "
                   "second-row leads (Au 5d binds at ~7 eV below "
                   "E_F; -40 eV gives ~33 eV margin).  Lower to "
                   "-50 / -60 eV for heavy-element electrodes "
                   "(Pt 5p, Pd 4p go ~20-25 eV below E_F).  "
                   "Brandbyge et al. 2002 § IV.",
    })

    # ----------------- Engine-specific (Method) -----------------

    # PySCF-NEGF method.  Ignored when engine='transiesta' (TranSIESTA
    # picks the level of theory via its own pseudopotential + XC
    # selection in the .fdf the engine emits).  These are exposed at
    # tier='advanced' so the form de-emphasises them when the user
    # has picked TranSIESTA.

    pyscf_functional: str = field(default="B3LYP", metadata={
        "section": "NEGF",
        "label":   "PySCF: XC functional",
        "tier":    "advanced",
        "help":    "(engine=pyscf-negf only) XC functional name "
                   "(libxc string).  Use the SAME functional as the "
                   "geometry relaxation -- transmission is sensitive "
                   "to E_F alignment, which depends on the SCF "
                   "method.",
    })
    pyscf_basis: str = field(default="def2-SVP", metadata={
        "section": "NEGF",
        "label":   "PySCF: basis set",
        "tier":    "advanced",
        "help":    "(engine=pyscf-negf only) Gaussian basis set.  "
                   "def2-SVP is the production minimum; def2-TZVP "
                   "is recommended when transmission features are "
                   "sensitive to lead-DOS detail.",
    })

    # TranSIESTA-specific method knobs.  Default values follow the
    # SIESTA Method-tab defaults in the Build workflow.

    siesta_mesh_cutoff_ry: int = field(default=300, metadata={
        "section": "NEGF",
        "label":   "TranSIESTA: mesh cutoff",
        "unit":    "Ry",
        "range":   (100, 1000),
        "tier":    "advanced",
        # Default tightened from 200 -> 300 Ry on 2026-06-10 post-
        # review: transport calculations almost always involve a
        # transition-metal electrode (Au, Pt, Pd), and the Au 5d
        # band requires >= 250-300 Ry per published TranSIESTA
        # work (Stokbro 2003, Brandbyge 2002).  200 Ry was suitable
        # only for first/second-row-only systems, a niche case for
        # transport.  Loosen back to 200 Ry for screening on
        # organic-only test cases.
        "help":    "(engine=transiesta only) real-space mesh cutoff "
                   "in Ry.  Default 300 Ry covers transition-metal "
                   "electrodes (Au / Pt / Pd 5d bands need >= 250-300 Ry "
                   "per Stokbro 2003).  Loosen to 200 Ry for organic-"
                   "only screening; raise to 400 Ry for heavy actinide "
                   "leads.",
    })

    # ----------------- Runtime -----------------

    max_memory_mb: int = field(default=8000, metadata={
        "section": "Runtime",
        "label":   "Memory budget",
        "unit":    "MB",
        "range":   (1000, 256000),
        "help":    "soft memory ceiling for the engine.  PySCF "
                   "respects this via its own max_memory parameter; "
                   "SIESTA-MPI splits the budget across MPI ranks.",
    })
    num_threads: int = field(default=4, metadata={
        "section": "Runtime",
        "label":   "CPU threads",
        "range":   (1, 256),
        "help":    "OMP_NUM_THREADS for the engine process.  For "
                   "SIESTA-MPI use 1 here and provide MPI ranks via "
                   "the runwrap.py SIESTA-MPI branch (one thread per "
                   "rank avoids oversubscription -- see runwrap.py "
                   "thread-pinning note).",
    })
    log_level: str = field(default="info", metadata={
        "section": "Runtime",
        "label":   "Log verbosity",
        "choices": ("warning", "info", "debug"),
        "help":    "engine log verbosity.  debug emits per-iteration "
                   "NEGF residuals + density-matrix norms; useful "
                   "when investigating convergence problems.",
    })
