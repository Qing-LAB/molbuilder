"""TransportConfig -- every parameter the Transport-tab engine reads.

Engine-agnostic, dataclass-of-fields with metadata, same architecture
as the other engine configs.  The metadata drives:

  * the schema-driven Transport form;
  * the validator pass;
  * the click-based CLI;
  * the Methods-text generator.

Engines (registry surface in :mod:`molbuilder.transport`; transiesta is
the shipped backend):

  * **transiesta** -- TranSIESTA from the SIESTA suite, NEGF + LDA/GGA
    pseudopotentials.  Handles realistic electrode + bridge sizes
    (≤ ~few hundred atoms in the device region).
  * **pyscf-negf** -- PySCF + a custom NEGF self-energy driver,
    Gaussian-basis with hybrid functionals available.  Smaller
    systems (~ tens of atoms in the device region) but higher-level
    XC.

Both consume the SAME relaxed geometry + the SAME ``.molstruct.json``
sidecar (region labels assigned in /modify -- see
:mod:`molbuilder.sidecars.molstruct`).  The two engines'
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
#: Atoms excluded from the NEGF region entirely (``TS.Atoms.Buffer``):
#: padding at the OUTER ends of the device, beyond the electrode blocks.
#: Optional -- most 2-terminal junctions need none.  Named 2026-08-28 with
#: the transport composite design (archive/2026-09-01-transport-design.md § 4.1a); the
#: categorical sort places buffer atoms outermost, by transport coordinate.
REGION_BUFFER          = "buffer"

# The defaults that ship with the modify tab.  See
# docs/engines/transport.md for the convention + scientific
# meaning of each label.
EXPECTED_REGIONS_2T = (
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
    REGION_BRIDGE,
)

# 2026-06-18 convention expansion: any region whose label ENDS WITH
# this suffix (case-insensitive, optionally preceded by ``-`` or
# ``_``) is treated by the TranSIESTA emitter as an electrode region.
# ``L-electrode``, ``R-electrode`` (the defaults) fit naturally;
# users can add e.g. ``tip-electrode`` for STM-style asymmetric leads
# or ``gate-electrode`` for 3-terminal devices without changing the
# emitter.  See docs/engines/transport.md.
ELECTRODE_LABEL_SUFFIX = "electrode"


def is_electrode_label(label: str) -> bool:
    """True iff ``label`` is to be treated as a TranSIESTA electrode
    region per the *-electrode convention.

    Accepts ``<name>-electrode`` (canonical), ``<name>_electrode``,
    and the bare suffix ``electrode``.  Case-insensitive.  Used by
    the validator + emitter to discover electrode regions without
    a closed enum.
    """
    if not isinstance(label, str):
        return False
    lo = label.lower()
    return (
        lo == ELECTRODE_LABEL_SUFFIX
        or lo.endswith("-" + ELECTRODE_LABEL_SUFFIX)
        or lo.endswith("_" + ELECTRODE_LABEL_SUFFIX)
    )


# --------------------------------------------------------------------- #
#  TransportConfig                                                      #
# --------------------------------------------------------------------- #


@dataclass
class TransportConfig:
    # Form-section render order -- and it is a SCIENTIFIC order, not a
    # historical one (2026-09-15, `engines/transport.md` § 3.3).  It was
    # System / Electrodes / Transmission / NEGF / Runtime, with the
    # electronic contract sitting inside "NEGF" where it never belonged and
    # nothing at all for the transverse k-grid, the broadening or the
    # outputs.  The order below is the order a person decides in: what is
    # computed (the window, its k-sampling), how sharply (broadening), what
    # is written (outputs), then the machinery (the density contour, the
    # leads), then what the citation already fixed, then the shell.
    _form_section_order = (
        "System",
        "Electrodes",
        "Transmission",
        "Transmission k-sampling",
        "Spin channel",
        "Broadening",
        "Outputs",
        "NEGF density contour",
        "Leads",
        "Electronic contract",
        "Runtime",
        "Logging",
    )

    _form_section_descriptions = {
        "Transmission k-sampling": (
            "How densely the transverse Brillouin zone is sampled when "
            "T(E) is evaluated.  tbtrans INHERITS the SCF's grid unless "
            "told otherwise, and a grid converged for a total energy is "
            "routinely far too coarse for transmission — so the standard "
            "convergence study is T(E_F) against this grid with "
            "everything else fixed."),
        "Spin channel": (
            "Which spin's transmission tbtrans reports.  Only "
            "meaningful once the device itself is spin-polarised, "
            "which the chemistry analysis in step 2 flags."),
        "Broadening": (
            "The imaginary parts that set the T(E) lineshape.  Too large "
            "smears resonances into a featureless curve; too small turns "
            "them into numerical noise."),
        "Outputs": (
            "Which quantities tbtrans writes besides the transmission.  "
            "Every one defaults to OFF in the engine, so a run produces "
            "T(E) and nothing else unless asked — there is no DOS or "
            "eigenchannel data on disk to plot later."),
        "NEGF density contour": (
            "The complex-contour integration that builds the "
            "non-equilibrium density matrix.  Production defaults are "
            "the SIESTA manual's; touch these when the density shows "
            "artefacts near the chemical potential or the SCF will not "
            "converge under bias."),
        "Leads": (
            "How the semi-infinite electrodes are treated: whether their "
            "own bulk Hamiltonian is used in the lead region, and how "
            "the electrode cell tiles the device's cross-section."),
        "Electronic contract": (
            "Basis, exchange-correlation and mesh.  These are the "
            "CITATION's to say — electrode and device must share them "
            "or the lead self-energy cannot attach seamlessly — so they "
            "are sealed unless the citation carries no deck."),
        "System": (
            "Engine selection and job-name identity.  TranSIESTA "
            "handles larger device regions with pseudopotentials; "
            "PySCF-NEGF supports hybrid functionals on smaller "
            "molecules.  Both consume the same relaxed structure + "
            "region-labelled sidecar."
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
            "The energy window T(E) is evaluated on.  The three "
            "numbers below become the from / to / points lines of the "
            "TBT.Contour.window block, so the curve comes back spaced "
            "(max − min) / (points − 1).  Whether tbtrans reads the "
            "two energies as absolute or as relative to the device's "
            "Fermi level is unresolved against SIESTA 5.4.2, which is "
            "why this tab offers no reference switch — see the "
            "transport contract, § 3.3."
        ),
        "NEGF": (
            "Self-energy + contour parameters that the NEGF density "
            "integration uses.  The defaults are robust for typical "
            "organic-on-metal junctions; touch these only if you "
            "see density-of-states artefacts or convergence issues."
        ),
        "Runtime": (
            "How the run uses your hardware — memory budget and CPU "
            "thread count.  Neither affects the science, only wall "
            "time."
        ),
        "Logging": (
            "How much the engine says while it runs.  Diagnostic "
            "output only; it changes nothing about the result."
        ),
    }

    # ----------------- System -----------------

    engine: str = field(default="transiesta", metadata={
        "section": "System",
        "workflow_group": "profile",
        "label":   "Engine",
        # ONLY registered backends are offered: a choice that
        # `get_engine` would refuse (`UnknownEngineError`) is a trap,
        # not an option.  TranSIESTA is the one shipped backend; a
        # PySCF-NEGF backend that registers itself adds its choice
        # back here in the same commit (all-electron, hybrid XC,
        # smaller systems -- wrong when the junction needs
        # pseudopotential-scale atom counts).
        "choices": ("transiesta",),
        "engine_key": '(molbuilder: backend selector)',
        "help":    "NEGF transport engine.  TranSIESTA: periodic "
                   "junctions with pseudopotentials, GGA/LDA.",
    })
    job_name: str = field(default="transport", metadata={
        "section":   "System",
        "workflow_group": "profile",
        "label":     "Job name",
        "id_suffix": "job-name",
        "pattern":   r"^[A-Za-z0-9_\-]+$",
        "engine_key": '(transiesta) SystemLabel / (molbuilder) filename basename',
        "help":      "filesystem-safe basename for emitted files "
                     "(transport.py + transport.json + transmission.dat).  "
                     "Same rule as SIESTA SystemLabel / PySCF job_name "
                     "-- see docs/execution/job-contracts.md.",
        "validate":  _validate_basename("job_name"),
    })

    # ----------------- Geometry -----------------
    #
    # No geometry-PATH fields here: the structure + region-label sidecar
    # ride in on the Generate POST body from the concealed MolView (the
    # viewer is the single source of geometry + labels), not typed paths.
    # The retired ``structure_xyz_path`` / ``molstruct_json_path`` form
    # fields (dead residue of the old path-entry design) were removed
    # 2026-07-25 -- nothing read them.

    # ----------------- Electrodes -----------------

    bias_voltages_v: List[float] = field(default_factory=lambda: [0.0],
                                          metadata={
        "section": "Electrodes",
        "workflow_group": "profile",
        "label":   "Bias voltages (V)",
        "engine_key": 'TS.Voltage  (transiesta — single value per .fdf today)',
        "help":    "comma-separated list of bias values V_L - V_R at "
                   "which to compute transmission.  Default [0.0] is "
                   "the zero-bias linear-response calculation (Landauer "
                   "conductance).  Each non-zero bias requires its own "
                   "NEGF density iteration so cost is roughly linear "
                   "in the list length.  |V| > 2 V surfaces a "
                   "linear-response-regime WARN in preflight "
                   "(di Ventra 2008).",
    })
    k_mesh_transverse: Tuple[int, int, int] = field(default=(1, 1, 1),
                                          metadata={
        "section": "Electrodes",
        "workflow_group": "stage",
        "label":   "Transverse k-mesh",
        "id_suffix": "k",
        "triple_labels": ("x", "y", "z"),
        "engine_key": '%block kgrid_Monkhorst_Pack  (transiesta)',
        "help":    "Monkhorst-Pack grid (Nx, Ny, Nz) summed over the "
                   "directions perpendicular to transport.  For a "
                   "finite molecule between leads use (1, 1, 1).  For "
                   "1D-periodic electrodes set Nx,Ny to the lead "
                   "periodicities; Nz is the transport direction and "
                   "stays at 1 (NEGF handles that one analytically).",
    })
    electronic_temperature_k: float = field(default=300.0, metadata={
        "section": "Electrodes",
        "workflow_group": "profile",
        "label":   "Electronic temperature",
        "unit":    "K",
        "range":   (10.0, 2000.0),
        "tier":    "advanced",
        "engine_key": 'ElectronicTemperature  (transiesta)',
        "help":    "Fermi-Dirac smearing on the lead occupations.  "
                   "Room-T (300 K) is the standard for experimentally-"
                   "comparable transmission curves; lower T sharpens "
                   "the Fermi step and surfaces fine features at the "
                   "cost of NEGF integration stability.",
    })

    # ----------------- Transmission -----------------

    transmission_emin_ev: float = field(default=-2.0, metadata={
        "section": "Transmission",
        "workflow_group": "stage",
        "label":   "Energy window min",
        "unit":    "eV",
        "range":   (-10.0, 0.0),
        "engine_key": '%block TBT.Contour.window -> from  (transiesta)',
        "help":    "lower edge of the T(E) energy window.  Relative "
                   "to E_F when ``transmission_relative_to_ef`` is on "
                   "(the default).  ±2 eV is the standard window for "
                   "low-bias junction characterisation; widen for high-"
                   "bias or for sigma-states / resonances further from "
                   "E_F.",
    })
    transmission_emax_ev: float = field(default=2.0, metadata={
        "section": "Transmission",
        "workflow_group": "stage",
        "label":   "Energy window max",
        "unit":    "eV",
        "range":   (0.0, 10.0),
        "engine_key": '%block TBT.Contour.window -> to  (transiesta)',
        "help":    "upper edge of the T(E) energy window.  See "
                   "``transmission_emin_ev`` for sign conventions.",
    })
    transmission_n_points: int = field(default=401, metadata={
        "section": "Transmission",
        "workflow_group": "stage",
        "label":   "Energy grid points",
        "range":   (51, 4001),
        "engine_key": '%block TBT.Contour.window -> points  (transiesta)',
        "help":    "number of evenly-spaced energies at which T(E) is "
                   "computed.  Resolution = (emax - emin) / (N - 1); "
                   "default 401 over a 4 eV window = 10 meV/point, "
                   "fine enough to resolve typical Au-thiol "
                   "transmission features in the ±1 V conductance "
                   "window.",
    })
    #: NOT EMITTED, and not rendered -- `UNRESOLVED_FIELDS` below.
    #:
    #: It wrote `TS.TBT.Erange.RelToEF`, which the 5.4.2 tbtrans does not
    #: know (`RelToEF`: zero occurrences).  Its replacement is NOT a
    #: rename: whether a `%block TBT.Contour` line's `from`/`to` are read
    #: as absolute energies or relative to the device E_F could not be
    #: settled from the binary, and **inventing a mapping is exactly what
    #: put four dead keywords in the deck** (`plan.md` § 5o).  It keeps
    #: its place so the question has an address; it offers no control
    #: until a live tbtrans answers it.
    transmission_relative_to_ef: bool = field(default=True, metadata={
        "section": "Transmission",
        "workflow_group": "profile",
        "label":   "Energy is relative to E_F",
        "engine_key": '(unresolved -- see the note above the field)',
        "help":    "NOT WIRED.  The energy reference of a "
                   "`%block TBT.Contour` line is unresolved against "
                   "tbtrans 5.4.2, so this writes nothing rather than "
                   "guess.  The window you set is emitted exactly as "
                   "given; how tbtrans references it is the open "
                   "question (plan.md 5o).",
    })

    # ----------------- NEGF -----------------

    # ================= Transmission k-sampling =================

    tbt_k_grid: Tuple[int, int, int] = field(
        default=(0, 0, 0), metadata={
            "section": "Transmission k-sampling",
            "workflow_group": "stage",
            "label":   "Transverse k-grid for T(E)  (0 0 0 = inherit the SCF's)",
            "engine_key": 'TBT.k  (tbtrans)',
            "help":    "THE CONVERGENCE KNOB THIS TAB COULD NOT EXPRESS "
                       "until 2026-09-15.  tbtrans inherits the SCF's "
                       "kgrid_Monkhorst_Pack, and a grid converged for a "
                       "total ENERGY is routinely far too coarse for "
                       "transmission: T(E) is an integral over the "
                       "transverse Brillouin zone and its features sharpen "
                       "with k-density.  The standard study is T(E_F) "
                       "against this grid with everything else fixed -- so "
                       "it is usually DENSER than the SCF's, and the "
                       "transport direction stays 1.  0 0 0 inherits, "
                       "which is the old behaviour and rarely the right "
                       "answer.",
        })
    # ================= Spin channel =================
    #
    # ITS OWN SECTION, and not "Transmission k-sampling" where it sat
    # until 2026-09-15.  Two reasons, and the second is the visible one:
    # a spin selector is not a k-sampling knob, and this field is
    # `profile` (nothing about staging tightens it) while the k-grid is
    # `stage` -- so one section name straddled two workflow-group cards
    # and the form drew the legend "Transmission k-sampling" TWICE, once
    # in Run profile over a spin box and again in Convergence targets
    # over the grid.  The section is the inner legend and the group is
    # the outer card (`web/form-schema.md` 1.3); the two axes are
    # orthogonal, so a section that spans groups gets repeated, by
    # design.  The fix is a name that belongs to one of them.
    tbt_spin: int = field(default=0, metadata={
        "section": "Spin channel",
        "workflow_group": "profile",
        "label":   "Spin channel (0 = both)",
        "range":   (0, 2),
        "tier":    "advanced",
        "engine_key": 'TBT.Spin  (tbtrans)',
        "help":    "1 selects spin-up and 2 spin-down; 0 (the engine's "
                   "default) does both.  Only meaningful for a "
                   "spin-polarised device run, which the chemistry "
                   "analysis flags when it finds an open-shell metal.",
    })

    # ================= Broadening =================

    tbt_elecs_eta_ev: float = field(default=0.001, metadata={
        "section": "Broadening",
        "workflow_group": "stage",
        "label":   "Electrode self-energy broadening",
        "unit":    "eV",
        "range":   (0.0, 1.0),
        "tier":    "advanced",
        "engine_key": 'TBT.Elecs.Eta  (tbtrans)',
        "help":    "the imaginary part in the lead surface Green "
                   "function.  The manual's default is 1 meV.  It sets "
                   "the T(E) LINESHAPE: too large smears resonances "
                   "into a featureless curve, too small turns them into "
                   "numerical noise.",
    })
    tbt_contours_eta_ev: float = field(default=0.0, metadata={
        "section": "Broadening",
        "workflow_group": "stage",
        "label":   "Device contour broadening (0 = engine default)",
        "unit":    "eV",
        "range":   (0.0, 1.0),
        "tier":    "advanced",
        "engine_key": 'TBT.Contours.Eta  (tbtrans)',
        "help":    "broadening on tbtrans's own energy contour.  0 "
                   "leaves the engine's default, min(eta_electrode)/10 "
                   "-- a FORMULA that tracks the field above, so a "
                   "number here decouples them.",
    })

    # ================= Outputs =================
    #
    # EVERY ONE OF THESE DEFAULTS TO FALSE IN TBTRANS, so a run writes
    # transmission and nothing else -- which is why the Results-tab
    # transmission inspector (plan.md W10) has no DOS or eigenchannel data
    # to read even in principle.  Off is kept as the default here too: each
    # costs disk and time, and asking for them is a decision.

    tbt_dos_gf: bool = field(default=False, metadata={
        "section": "Outputs",
        "workflow_group": "stage",
        "label":   "Green-function DOS",
        "engine_key": 'TBT.DOS.Gf  (tbtrans)',
        "help":    "writes the device DOS from the full Green function, "
                   "bound states included (DOS / AVDOS files).",
    })
    tbt_dos_a: bool = field(default=False, metadata={
        "section": "Outputs",
        "workflow_group": "stage",
        "label":   "Spectral DOS per electrode",
        "engine_key": 'TBT.DOS.A  (tbtrans)',
        "help":    "writes the spectral-function DOS, i.e. the DOS each "
                   "lead injects (ADOS / AVADOS).  This is what a "
                   "per-lead PDOS plot reads.",
    })
    tbt_dos_elecs: bool = field(default=False, metadata={
        "section": "Outputs",
        "workflow_group": "stage",
        "label":   "Bulk electrode DOS",
        "engine_key": 'TBT.DOS.Elecs  (tbtrans)',
        "help":    "writes the pristine bulk lead DOS (BDOS / AVBDOS) -- "
                   "the reference a transmission feature is judged "
                   "against.",
    })
    tbt_t_eig: int = field(default=0, metadata={
        "section": "Outputs",
        "workflow_group": "stage",
        "label":   "Transmission eigenchannels",
        "range":   (0, 20),
        "engine_key": 'TBT.T.Eig  (tbtrans)',
        "help":    "how many transmission EIGENVALUES to write (TEIG / "
                   "AVTEIG).  0 is the engine's default.  The "
                   "eigenchannel decomposition is what says WHICH "
                   "orbital carries the current, and it is the usual "
                   "next question after T(E).",
    })
    tbt_t_bulk: bool = field(default=False, metadata={
        "section": "Outputs",
        "workflow_group": "stage",
        "label":   "Bulk transmission",
        "tier":    "advanced",
        "engine_key": 'TBT.T.Bulk  (tbtrans)',
        "help":    "writes each lead's own bulk transmission (BTRANS) -- "
                   "the ballistic ceiling the junction is compared to.",
    })
    tbt_t_all: bool = field(default=False, metadata={
        "section": "Outputs",
        "workflow_group": "stage",
        "label":   "All electrode pairs",
        "tier":    "advanced",
        "engine_key": 'TBT.T.All  (tbtrans)',
        "help":    "writes every ordered electrode pair rather than "
                   "assuming T_ij = T_ji.  For two terminals at "
                   "equilibrium the assumption holds; under bias, or "
                   "with three or more leads, it does not.",
    })

    # ================= NEGF density contour =================
    #
    # THE THREE FIELDS THAT WERE HERE NAMED KEYWORDS THAT DO NOT EXIST.
    # `TS.ComplexContour.NumCircle` / `NumLine` / `Emin` are SIESTA-3.x
    # spellings; 5.4.2 keeps only the unused legacy `ComplexContour.NPoles`
    # (`plan.md` § 5o).  They are NOT renamed to the modern keywords,
    # because the modern ones are not the same quantities: `NumCircle` was a
    # node COUNT, and `TS.Contours.Eq.Pole` is an ENERGY.  Retired, and the
    # real controls added in their place.
    #
    # EMISSION POLICY for everything below, stated once: where the manual's
    # default is a NUMBER it is this field's default and always emitted, so
    # the deck is self-documenting and behaviour is unchanged.  Where the
    # manual's default is a FORMULA (`min[eta_e]/10`, `5 kB T`) the field
    # defaults to 0 meaning *leave it to the engine*, and nothing is emitted
    # -- writing a number there would silently replace a formula.

    negf_eq_pole_ev: float = field(default=1.5, metadata={
        "section": "NEGF density contour",
        "workflow_group": "stage",
        "label":   "Equilibrium pole energy",
        "unit":    "eV",
        "range":   (0.1, 10.0),
        "tier":    "advanced",
        "engine_key": 'TS.Contours.Eq.Pole  (transiesta)',
        "help":    "the energy at which the equilibrium contour's poles "
                   "sit, which fixes how many Matsubara poles the "
                   "residue sum carries.  The manual's default, 1.5 eV, "
                   "is the production value; raising it costs poles and "
                   "buys accuracy in the occupied density.",
    })
    negf_neq_eta_ev: float = field(default=0.0, metadata={
        "section": "NEGF density contour",
        "workflow_group": "stage",
        "label":   "Non-equilibrium broadening (0 = engine default)",
        "unit":    "eV",
        "range":   (0.0, 1.0),
        "tier":    "advanced",
        "engine_key": 'TS.Contours.nEq.Eta  (transiesta)',
        "help":    "imaginary part on the non-equilibrium (real-axis) "
                   "contour.  0 leaves the engine's own default, which "
                   "is min[eta_electrode]/10 -- a FORMULA, so a number "
                   "here replaces it rather than restating it.  Only "
                   "relevant under bias.",
    })
    negf_neq_fermi_cutoff_ev: float = field(default=0.0, metadata={
        "section": "NEGF density contour",
        "workflow_group": "stage",
        "label":   "Non-equilibrium Fermi cutoff (0 = engine default)",
        "unit":    "eV",
        "range":   (0.0, 10.0),
        "tier":    "advanced",
        "engine_key": 'TS.Contours.nEq.Fermi.Cutoff  (transiesta)',
        "help":    "how far beyond the bias window the non-equilibrium "
                   "integration runs.  0 leaves the engine's default of "
                   "5 kB T, which scales with the electronic "
                   "temperature -- a number here fixes it instead.",
    })

    # ================= Leads =================

    elecs_bulk: bool = field(default=True, metadata={
        "section": "Leads",
        "workflow_group": "stage",
        "label":   "Use the electrode's own bulk Hamiltonian",
        "tier":    "advanced",
        "engine_key": 'TS.Elecs.Bulk  (transiesta)',
        "help":    "on (the manual's default), the lead region uses the "
                   "Hamiltonian from the ELECTRODE calculation; off, it "
                   "uses the scattering region's own elements there.  On "
                   "is right whenever the electrode really is bulk-like, "
                   "which is what the region labels assert.",
    })
    electrode_bloch: Tuple[int, int, int] = field(
        default=(1, 1, 1), metadata={
            "section": "Leads",
            "workflow_group": "stage",
            "label":   "Bloch expansion of the electrode cell",
            "tier":    "advanced",
            "engine_key": 'bloch  (inside %block TS.Elec.<name>)',
            "help":    "how many times the electrode's own unit cell is "
                       "repeated to tile the device's cross-section.  "
                       "1 1 1 means the electrode cell already matches; "
                       "expanding a SMALLER electrode cell is the "
                       "standard cost saving, and it must match the "
                       "device geometry exactly or the lead will not "
                       "attach.",
        })

    # THE TWO PySCF FIELDS LEFT HERE ON 2026-09-15.
    #
    # `TransportConfig` is **TranSIESTA's** (`engines/transport.md` § 3.2).
    # `pyscf_functional` and `pyscf_basis` sat in the NEGF section, beside
    # three `TS.ComplexContour.*` fields, for an engine `registered_engines()`
    # does not list and `engine`'s own `choices` excludes -- so they were the
    # only two fields in the form with an engine name in the LABEL, which is
    # why the tab read as a PySCF page for a workflow that is TranSIESTA's
    # entire subject (user, 2026-09-15).  Neither was sealed, so both
    # travelled into `task.json` and merged into a config where `engine` is
    # hardcoded `"transiesta"` and nothing read them.
    #
    # They come back with a `PyscfNegfTransportConfig` authored WITH that
    # backend and the panel that renders it, in one commit -- not before.
    # An engine's parameter set is not designable in the abstract.

    # TranSIESTA-specific method knobs.  Default values follow the
    # SIESTA Method-tab defaults in the Build workflow.

    # The ELECTRONIC CONTRACT the electrode and device decks share
    # (transport-design.md § 3: basis · XC · energy shift identical by
    # construction, because they read ONE object).  Until 2026-08-28
    # these were hard-coded inside the emitter (DZP / PBE / 0.01 Ry);
    # the transport composite fills them at prep from the cited
    # attempt's own .fdf — the deck that actually ran is the truth
    # about a result — so the fields had to exist to be filled.
    basis_size: str = field(default="DZP", metadata={
        "section": "Electronic contract",
        "workflow_group": "stage",
        "label":   "Basis size",
        "tier":    "advanced",
        "choices": ("SZ", "SZP", "DZ", "DZP", "TZP"),
        "engine_key": 'PAO.BasisSize  (transiesta)',
        "help":    "(engine=transiesta only) PAO basis for every stage "
                   "of the transport ladder — electrode, seed and device "
                   "share it by construction.  In the transport composite "
                   "this is filled from the cited junction's own .fdf.",
    })
    energy_shift_ry: float = field(default=0.01, metadata={
        "section": "Electronic contract",
        "workflow_group": "stage",
        "label":   "PAO energy shift",
        "unit":    "Ry",
        "range":   (0.0001, 0.1),
        "tier":    "advanced",
        "engine_key": 'PAO.EnergyShift  (transiesta)',
        "help":    "(engine=transiesta only) orbital confinement energy "
                   "shift; shared by every stage of the transport ladder.  "
                   "In the transport composite this is filled from the "
                   "cited junction's own .fdf.",
    })
    xc_functional: str = field(default="GGA", metadata={
        "section": "Electronic contract",
        "workflow_group": "stage",
        "label":   "XC family",
        "tier":    "advanced",
        "choices": ("LDA", "GGA", "VDW"),
        "engine_key": 'XC.functional  (transiesta)',
        "help":    "(engine=transiesta only) exchange-correlation family; "
                   "shared by every stage of the transport ladder.  In "
                   "the transport composite this is filled from the cited "
                   "junction's own .fdf.",
    })
    xc_authors: str = field(default="PBE", metadata={
        "section": "Electronic contract",
        "workflow_group": "stage",
        "label":   "XC authors",
        "tier":    "advanced",
        "engine_key": 'XC.authors  (transiesta)',
        "help":    "(engine=transiesta only) the flavour within the XC "
                   "family (PBE, revPBE, CA, ...); shared by every stage "
                   "of the transport ladder.  In the transport composite "
                   "this is filled from the cited junction's own .fdf.",
    })

    siesta_mesh_cutoff_ry: int = field(default=300, metadata={
        "section": "Electronic contract",
        "workflow_group": "stage",
        "label":   "TranSIESTA: mesh cutoff",
        "unit":    "Ry",
        "range":   (100, 1000),
        "tier":    "advanced",
        "engine_key": 'MeshCutoff  (transiesta)',
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
        "workflow_group": "budget",
        "label":   "Memory budget",
        "unit":    "MB",
        "range":   (1000, 256000),
        "engine_key": '(runner) ulimit -v',
        "help":    "soft memory ceiling for the engine.  The runner "
                   "applies it as `ulimit -v`, and SIESTA-MPI splits "
                   "the budget across MPI ranks.",
    })
    num_threads: int = field(default=4, metadata={
        "section": "Runtime",
        "workflow_group": "budget",
        "label":   "CPU threads",
        "range":   (1, 256),
        "engine_key": 'OMP_NUM_THREADS  (runner shell wrapper)',
        "help":    "OMP_NUM_THREADS for the engine process.  For "
                   "SIESTA-MPI use 1 here and provide MPI ranks via "
                   "the runwrap.py SIESTA-MPI branch (one thread per "
                   "rank avoids oversubscription -- see runwrap.py "
                   "thread-pinning note).",
    })
    # Logging is its own section for the reason spelled out above
    # `tbt_spin`: memory and threads are `budget`, verbosity is
    # `profile`, and while all three were called "Runtime" the form drew
    # that legend twice -- in Run profile and again in Compute & budget.
    log_level: str = field(default="info", metadata={
        "section": "Logging",
        "workflow_group": "profile",
        "label":   "Log verbosity",
        "choices": ("warning", "info", "debug"),
        "engine_key": 'TBT.Verbosity  (tbtrans)',
        "help":    "engine log verbosity.  debug emits per-iteration "
                   "NEGF residuals + density-matrix norms; useful "
                   "when investigating convergence problems.  Maps onto "
                   "TBT.Verbosity, an integer 0-10 defaulting to 5 "
                   "(TBtrans reference): warning=2, info=5, debug=8.  "
                   "It claimed `WriteVerbosity` until 2026-09-15, which "
                   "is zero occurrences in the 5.4.2 binary.",
    })
