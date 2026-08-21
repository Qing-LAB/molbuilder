"""L1 result types for the Spectra tab.

Pinned by docs/web/spectra.md  Three dataclasses:

  * :class:`ModeElectronicStructure` -- the per-mode displaced-SCF
    block: equilibrium / ±A·Q_i MO energies + SCF energies.  Populated
    only when the user selected a mode for electronic-structure analysis
    (see spec § 8 / Model 2 selectors).
  * :class:`ModeData` -- one vibrational mode: frequency, eigenvector
    (free atoms only), Raman activity, optional IR intensity (1c
    reserved), and the optional :class:`ModeElectronicStructure`.
  * :class:`SpectraResults` -- the complete result of a Spectra run:
    metadata, equilibrium reference, list of modes, methods text,
    bibliography keys, and the ``complete`` flag (False during a
    live-watched in-progress run, True after the final phase).

These are the **engine-agnostic** result shape -- the parser
populates them from a ``.spectra.json`` regardless of which engine
produced it.  Adding a future engine (SIESTA, ...) does not change
this surface.

All three carry ``to_dict()`` / ``from_dict()`` for JSON round-trip
because the on-disk format (``<job>.spectra.json``), the
``/api/spectra/*`` HTTP responses, and the in-memory typed shape
share one schema -- the dataclass is the canonical structure, the
dict is its wire encoding.  numpy arrays serialise as nested
Python lists (round-trip via :func:`numpy.asarray`).

Type discipline: every numpy field is coerced to ``dtype=float``
and shape-validated in ``__post_init__``.  Passing an int array,
a list-of-lists, or a wrong-shape array fails LOUDLY at
construction time, not silently 100 lines downstream when a
caller assumes the float type.

Equality: ``a == b`` raises ``TypeError`` on all three dataclasses.
Scientific comparison of two spectra is never a yes/no question --
the real questions are "what's the Δfrequency at each mode?",
"max ΔHOMO shift?", "is the spectrum converged within tolerance
X cm⁻¹?".  None of those are bool-valued.  A future
:func:`spectra.compare` (not yet implemented; build when a real
caller needs it) will return structured deltas instead.  Until
then, accidental ``==`` raises with a pointer at the right API.

Schema version: pinned at 1 on :data:`SpectraResults.schema_version`.
Bumping the version requires the parser to grow a per-version
branch; see spec § 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# SCHEMA_VERSION history (incremented when the on-disk JSON shape changes):
#
#   v1  -- initial release.  Single mode-eigenvector field
#          ``eigenvector_free`` (used ambiguously for both 3D animation
#          and Raman-projection in different code paths -- correctness
#          bug, see decisions-log entry 2026-05-15).
#   v2  -- two explicit eigenvector forms per mode:
#            * eigenvector_canonical  -- Cartesian normal mode in the
#                canonical mass-weighted convention sum_k m_k|L_k|^2 = 1
#                (use for Placzek Raman, IR, electron-phonon...).
#            * eigenvector_display    -- max(|L_k|) = 1 per mode
#                (use for 3D animation, fixed-amplitude ES probe).
#          Old ``eigenvector_free`` field is dropped from the wire
#          format; ``from_dict`` continues to accept v1 documents as a
#          best-effort fallback (both new arrays populated from the
#          single legacy field, since the canonical normalisation is
#          not recoverable from v1).
#  v3 (2026-05-21): rename ``fixed_atom_idxs`` -> ``frozen_atom_idxs``
#          (terminology unification, design.md 2026-05-21).  No
#          backward compatibility: v1/v2 documents fail to load
#          with a clear schema-version error rather than silently
#          coercing.  Re-run / re-export the spectra.json to land
#          on v3.
#  v4 (2026-05-22): add ``runtime_info`` dict carrying CPU/thread
#          and GPU facts the running script collected.  Lets the
#          /results page show "20 threads, BLAS=1, GPU ON (RTX 4090,
#          CC 8.9)" so a user can verify the run matched their
#          configuration intent.  No backward compatibility: v3
#          documents fail with a clear schema-version error.
SCHEMA_VERSION = 5   # 5 (2026-08-20, spectra-migration plan D4): + the
#                    OPTIONAL `phase_relaxation` + `relaxation` progress
#                    block (the in-deck relaxation is a TRACKED step) and
#                    the OPTIONAL `thermo` block (RRHO re-homed from the
#                    retiring thermo.txt path; D2).  ADDITIVE -- a v4 file
#                    lacks them and reads whole, which is why the reader
#                    accepts a SET (the molstruct sidecar's own rule).
READABLE_SCHEMA_VERSIONS = frozenset({4, 5})


# Phase status vocabulary -- per-layer flag carried on
# :class:`SpectraResults` and emitted in the on-disk JSON.  Pinned
# here as a module constant so the engine, the parser, and the UI
# all read from one source.  Spec § 5 + § 6.1 describe transitions.

PHASE_EMPTY    = "empty"     # not yet computed
PHASE_RUNNING  = "running"   # script is mid-way through this phase
                             # (live-watch / atomic-replace JSON updates)
PHASE_COMPLETE = "complete"  # phase done, data is final for this run

_VALID_PHASE_STATES = (PHASE_EMPTY, PHASE_RUNNING, PHASE_COMPLETE)


def _reject_complex_then_asarray(value, *, field: str) -> np.ndarray:
    """Coerce ``value`` to a 1-D-or-greater float ndarray, raising
    ``TypeError`` LOUDLY if the input is complex.

    Plain ``np.asarray(complex_input, dtype=float)`` silently drops
    the imaginary part with only a numpy ComplexWarning -- the
    `__post_init__` then sees a clean real array and we lose data
    without ever raising.  Our wire format is all real-valued, so
    a complex input is a programmer error (or a hand-edited file
    gone wrong); fail with a clear message rather than corrupt
    the result quietly.
    """
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise TypeError(
            f"{field}: complex values are not supported by the v1 "
            f"Spectra wire format (the imaginary part would be "
            f"silently discarded).  If you need complex polarizability "
            f"or coupling tensors, encode as paired (re, im) real "
            f"arrays.  Got dtype={arr.dtype}."
        )
    return np.asarray(arr, dtype=float)


def _no_equality(self, other):  # noqa: ARG001
    """Shared explicit ``__eq__`` that refuses bool comparison.

    Used by all three Spectra dataclasses so accidental ``a == b``
    fails loudly with a pointer at the right API instead of either
    (a) producing the numpy-ambiguous-truth-value TypeError on the
    embedded ndarrays, or (b) silently giving a misleading False
    when the user actually wanted "is the spectrum converged".
    """
    raise TypeError(
        f"{type(self).__name__} equality is intentionally undefined. "
        "Comparing two spectra is never a yes/no question; the "
        "scientific operations are Δfrequency / Δactivity / "
        "Δ(HOMO,LUMO) per mode.  A structured comparator will live "
        "at `molbuilder.spectra.compare(...)` when a concrete "
        "caller needs it.  Until then, compare per-field "
        "(np.testing.assert_allclose on arrays, pytest.approx on "
        "scalars) or call the comparator above."
    )


# --------------------------------------------------------------------- #
#  Per-mode electronic structure                                        #
# --------------------------------------------------------------------- #


@dataclass(eq=False)
class ModeElectronicStructure:
    """Displaced-geometry SCF results for a single mode.

    Three geometries are sampled: equilibrium and ±A along the mode's
    mass-weighted eigenvector.  Each MO-energy array spans the window
    [HOMO − ``cfg.es_n_homo_below``, LUMO + ``cfg.es_n_lumo_above``]
    AT THAT DISPLACEMENT -- the same orbital count, but the indexing
    is per-geometry (orbitals can swap order under displacement, so
    the i-th entry of ``minus`` is not necessarily the same orbital
    as the i-th entry of ``eq``).  The Spectra-tab UI handles the
    matching when computing electron-phonon coupling constants.

    :func:`from_dict` accepts the wire form (lists of floats) and
    rebuilds numpy arrays on the way in.
    """

    amplitude_ang:        float

    # Each array shape (n_window,) in Hartree -- the orbital energy
    # window around HOMO/LUMO.
    mo_energies_eq_eh:    np.ndarray
    mo_energies_minus_eh: np.ndarray
    mo_energies_plus_eh:  np.ndarray

    # Index (into the window arrays above) of the HOMO at the
    # equilibrium geometry.  The HOMO+1 / LUMO is implicit
    # (homo_index_in_window + 1).
    homo_index_in_window: int

    # Total SCF energies in Hartree at each of the three geometries.
    scf_energy_eq_eh:     float
    scf_energy_minus_eh:  float
    scf_energy_plus_eh:   float

    __eq__ = _no_equality

    def __post_init__(self):
        """Normalise + shape-validate the MO arrays.

        All three arrays must be 1-D, dtype=float, and have the
        same length (the orbital window size).  We coerce dtype +
        contiguity on the way in so a caller passing a list, an
        int array, or a non-contiguous view is normalised once and
        the typed surface stays predictable downstream.
        """
        self.mo_energies_eq_eh    = _reject_complex_then_asarray(
            self.mo_energies_eq_eh,    field="ModeElectronicStructure.mo_energies_eq_eh")
        self.mo_energies_minus_eh = _reject_complex_then_asarray(
            self.mo_energies_minus_eh, field="ModeElectronicStructure.mo_energies_minus_eh")
        self.mo_energies_plus_eh  = _reject_complex_then_asarray(
            self.mo_energies_plus_eh,  field="ModeElectronicStructure.mo_energies_plus_eh")
        if self.mo_energies_eq_eh.ndim != 1:
            raise ValueError(
                f"ModeElectronicStructure.mo_energies_eq_eh must be 1-D; "
                f"got shape {self.mo_energies_eq_eh.shape}"
            )
        n = self.mo_energies_eq_eh.size
        if self.mo_energies_minus_eh.shape != (n,) or self.mo_energies_plus_eh.shape != (n,):
            raise ValueError(
                f"ModeElectronicStructure: mo_energies_{{eq,minus,plus}}_eh "
                f"must share the same shape; got eq={self.mo_energies_eq_eh.shape}, "
                f"minus={self.mo_energies_minus_eh.shape}, "
                f"plus={self.mo_energies_plus_eh.shape}"
            )
        if not 0 <= self.homo_index_in_window < n:
            raise ValueError(
                f"ModeElectronicStructure.homo_index_in_window={self.homo_index_in_window} "
                f"out of range [0, {n})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict.  numpy arrays -> nested lists; floats
        stay floats; ints stay ints.  Round-trip via
        :func:`from_dict` is byte-equal modulo float formatting."""
        return {
            "amplitude_ang":        float(self.amplitude_ang),
            "mo_energies_eq_eh":    self.mo_energies_eq_eh.tolist(),
            "mo_energies_minus_eh": self.mo_energies_minus_eh.tolist(),
            "mo_energies_plus_eh":  self.mo_energies_plus_eh.tolist(),
            "homo_index_in_window": int(self.homo_index_in_window),
            "scf_energy_eq_eh":     float(self.scf_energy_eq_eh),
            "scf_energy_minus_eh":  float(self.scf_energy_minus_eh),
            "scf_energy_plus_eh":   float(self.scf_energy_plus_eh),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModeElectronicStructure":
        """Inverse of :meth:`to_dict`.  Coerces list-of-float arrays
        back to ``np.ndarray`` so the typed surface always carries
        numpy.  Extra keys in ``d`` are ignored (forward compat)."""
        return cls(
            amplitude_ang        = float(d["amplitude_ang"]),
            mo_energies_eq_eh    = np.asarray(d["mo_energies_eq_eh"],    dtype=float),
            mo_energies_minus_eh = np.asarray(d["mo_energies_minus_eh"], dtype=float),
            mo_energies_plus_eh  = np.asarray(d["mo_energies_plus_eh"],  dtype=float),
            homo_index_in_window = int(d["homo_index_in_window"]),
            scf_energy_eq_eh     = float(d["scf_energy_eq_eh"]),
            scf_energy_minus_eh  = float(d["scf_energy_minus_eh"]),
            scf_energy_plus_eh   = float(d["scf_energy_plus_eh"]),
        )


# --------------------------------------------------------------------- #
#  One vibrational mode                                                 #
# --------------------------------------------------------------------- #


@dataclass(eq=False)
class ModeData:
    """One vibrational mode.

    Each eigenvector is restricted to the *free* atoms (frozen atoms
    don't move), shape ``(n_free, 3)``; the global
    :attr:`SpectraResults.free_atom_idxs` maps free-atom rows back
    to global atom indices.

    Two normalisations of the same physical mode are provided so
    consumers can pick the one their downstream task requires; they
    differ only by a per-mode scaling factor and are interchangeable
    for direction-of-motion purposes.  See the field docs below.

    Imaginary modes are reported with a negative frequency and
    :attr:`has_imag` ``= True``.  Sign convention follows
    ``ω = sign(λ) * sqrt(|λ|)`` where λ is the mass-weighted Hessian
    eigenvalue, so a saddle's "imaginary" mode becomes a negative
    real number for plotting purposes.

    The optional :attr:`electronic_structure` is populated only for
    modes the user selected via the Model 2 selector (spec § 8).
    Unselected modes have ``electronic_structure = None`` -- the UI
    renders an empty cell + "—" in the mode-list ES columns.

    IR intensity is reserved for the future 1c (IR add-on) work;
    always ``None`` in v1 emitted scripts.
    """

    index_1based:         int
    frequency_cm1:        float

    # Activities / intensities are optional because:
    #  * raman_activity_a4_amu is None when cfg.compute_raman = False
    #    (diagnostic / Hessian-only run);
    #  * ir_intensity_km_mol is always None in v1.
    raman_activity_a4_amu: Optional[float]
    ir_intensity_km_mol:   Optional[float]

    # Cartesian normal mode in the canonical mass-weighted convention:
    #     Σ_k m_k |L_k|² = 1     (m_k in atomic units of mass)
    # Use for any physics that depends on the actual amplitude of
    # nuclear motion -- Raman activity (Placzek 45a²+7γ² lands in
    # Å⁴/amu directly), IR intensity, electron-phonon coupling
    # gradients, normal-mode-analysis projections.
    eigenvector_canonical: np.ndarray

    # Same mode, rescaled per mode so that max(|L_k|) = 1.  Dimensionless.
    # Use for 3D animation (each mode reaches the same peak amplitude
    # on screen regardless of mass distribution) and for the fixed-
    # amplitude electron-phonon "probe displacement" in Phase 4.
    # DO NOT feed this into physical-amplitude formulas -- the units
    # are wrong by a per-mode factor.
    eigenvector_display:   np.ndarray

    # Sign-of-eigenvalue marker: negative-ω modes are flagged here
    # so the UI / parser / methods generator don't have to
    # re-derive from the sign every time.
    has_imag:              bool

    electronic_structure:  Optional[ModeElectronicStructure] = None

    __eq__ = _no_equality

    def __post_init__(self):
        """Validate the eigenvector shapes.

        Both eigenvector arrays must be 2-D with the last axis = 3
        (Cartesian x/y/z per free atom) and the same first-axis
        length (cross-mode consistency -- same n_free across all
        modes in a SpectraResults -- is checked at the result level,
        not here).
        """
        self.eigenvector_canonical = _reject_complex_then_asarray(
            self.eigenvector_canonical,
            field="ModeData.eigenvector_canonical")
        self.eigenvector_display = _reject_complex_then_asarray(
            self.eigenvector_display,
            field="ModeData.eigenvector_display")
        for name, arr in (("canonical", self.eigenvector_canonical),
                          ("display",   self.eigenvector_display)):
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    f"ModeData.eigenvector_{name} must have shape "
                    f"(n_free, 3); got {arr.shape}"
                )
        if self.eigenvector_canonical.shape != self.eigenvector_display.shape:
            raise ValueError(
                f"ModeData: eigenvector_canonical and eigenvector_display "
                f"must have the same shape; got "
                f"{self.eigenvector_canonical.shape} vs "
                f"{self.eigenvector_display.shape}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_1based":          int(self.index_1based),
            "frequency_cm1":         float(self.frequency_cm1),
            "raman_activity_a4_amu": (None if self.raman_activity_a4_amu is None
                                      else float(self.raman_activity_a4_amu)),
            "ir_intensity_km_mol":   (None if self.ir_intensity_km_mol is None
                                      else float(self.ir_intensity_km_mol)),
            "eigenvector_canonical": self.eigenvector_canonical.tolist(),
            "eigenvector_display":   self.eigenvector_display.tolist(),
            "has_imag":              bool(self.has_imag),
            "electronic_structure":  (None if self.electronic_structure is None
                                      else self.electronic_structure.to_dict()),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModeData":
        es = d.get("electronic_structure")
        # Resolve the eigenvector pair, accepting both schema versions:
        #
        #   v2 (current)  -- two explicit fields, used as-is.
        #   v1 (legacy)   -- single ``eigenvector_free`` field; we treat
        #                    it as the display form and copy it into the
        #                    canonical slot too.  This is best-effort
        #                    only: v1 didn't track the canonical
        #                    normalisation separately, so a downstream
        #                    re-projection (e.g. recomputing Raman from
        #                    scratch) needs to re-run the harmonic
        #                    analysis to get the correct canonical
        #                    eigenvectors back.  The raman_activity /
        #                    ir_intensity values stored in the v1 JSON
        #                    themselves are unaffected; they were
        #                    computed at emit time with whatever
        #                    normalisation that v1 build had.
        if "eigenvector_canonical" in d:
            ev_canon = np.asarray(d["eigenvector_canonical"], dtype=float)
            ev_disp  = np.asarray(
                d.get("eigenvector_display", d.get("eigenvector_free")),
                dtype=float,
            )
        else:
            ev_free  = np.asarray(d["eigenvector_free"], dtype=float)
            ev_canon = ev_free
            ev_disp  = ev_free
        return cls(
            index_1based          = int(d["index_1based"]),
            frequency_cm1         = float(d["frequency_cm1"]),
            raman_activity_a4_amu = (None if d.get("raman_activity_a4_amu") is None
                                     else float(d["raman_activity_a4_amu"])),
            ir_intensity_km_mol   = (None if d.get("ir_intensity_km_mol") is None
                                     else float(d["ir_intensity_km_mol"])),
            eigenvector_canonical = ev_canon,
            eigenvector_display   = ev_disp,
            has_imag              = bool(d.get("has_imag", False)),
            electronic_structure  = (None if es is None
                                     else ModeElectronicStructure.from_dict(es)),
        )


# --------------------------------------------------------------------- #
#  Which atoms a mode belongs to                                        #
# --------------------------------------------------------------------- #


def motion_share_by_element(elements: List[str],
                            eigenvector: Any,
                            atom_idxs: Optional[List[int]] = None,
                            ) -> Dict[str, float]:
    """Each element's share of a mode's motion, summing to 1.

    WHY THIS EXISTS, in one example.  In mode 30 of the benzene-dithiol
    result, the hydrogens have the largest displacement vectors -- |L| =
    1.15 against carbon's 0.98 -- so "which atom moves furthest" answers
    *hydrogen*.  The mode is a ring stretch: carbon carries 91% of it.
    Distance alone is the wrong question, because a light atom travels
    further for the same energy, and hydrogen is the lightest thing in
    most molecules.  Asking which atoms the mode BELONGS to means
    weighting by mass.

    WHAT IT COMPUTES.  The kinetic-energy distribution: atom *i*'s share
    is ``mᵢ|Lᵢ|²`` over the sum across atoms.  For a harmonic mode the
    ratio is the same at every phase of the oscillation, so it is a
    property of the mode rather than of the instant you look at it --
    which is what makes it the standard way modes are assigned.

    EITHER EIGENVECTOR WORKS.  The two stored forms differ by one scalar
    per mode (§ SCHEMA_VERSION v2: canonical is normalised in the
    mass-weighted metric, display so the largest component is 1), and a
    scalar cancels in a ratio.  Pass whichever is in hand.

    ``atom_idxs`` maps eigenvector rows onto ``elements`` when the mode
    covers only the free atoms -- the free-atom list from the result.
    Omit it when there is one row per atom.  Shares are returned largest
    first, and a mode with no motion at all returns ``{}`` rather than
    dividing by zero.
    """
    from ..chemistry import atomic_mass

    rows = np.asarray(eigenvector, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(
            f"motion_share_by_element: eigenvector must be (n_atoms, 3), "
            f"got {rows.shape}"
        )
    if atom_idxs is None:
        idxs = list(range(rows.shape[0]))
    else:
        idxs = [int(i) for i in atom_idxs]
    if len(idxs) != rows.shape[0]:
        raise ValueError(
            f"motion_share_by_element: {rows.shape[0]} eigenvector rows but "
            f"{len(idxs)} atom indices -- the mode does not match the structure"
        )

    weight: Dict[str, float] = {}
    total = 0.0
    for row, at in zip(rows, idxs):
        if at < 0 or at >= len(elements):
            raise ValueError(
                f"motion_share_by_element: atom index {at} is outside the "
                f"structure ({len(elements)} atoms)"
            )
        el = str(elements[at])
        w = atomic_mass(el) * float(np.dot(row, row))
        weight[el] = weight.get(el, 0.0) + w
        total += w
    if total <= 0.0:
        return {}
    return {el: w / total
            for el, w in sorted(weight.items(), key=lambda kv: -kv[1])}


# --------------------------------------------------------------------- #
#  Complete results from a Spectra run                                  #
# --------------------------------------------------------------------- #


@dataclass(eq=False)
class SpectraResults:
    """Engine-agnostic result of a Spectra run.

    ``complete`` is the live-watch flag (spec § 6, Option B
    "phase-checkpoint with atomic file replace"):

      * ``False`` while the run is mid-way -- some modes may have
        ``electronic_structure = None`` not because the user
        de-selected them but because their displaced SCFs haven't
        run yet.  The ``selected_mode_idxs_1based`` field tells the
        UI which modes WILL get ES data so it can show progress
        ("3 of 10 modes done").
      * ``True`` after the final phase -- ``methods_text`` and
        ``bibliography_keys`` are populated; the UI renders the
        Methods-preview button.

    Run identity is captured by ``structure_hash`` (SHA-256 of the
    canonical XYZ of the input structure) so the parser can refuse
    to merge results from a different molecule; ``engine`` +
    ``engine_version`` + ``molbuilder_version`` give the software
    stack used; ``timestamp`` is ISO-8601 UTC.
    """

    # Provenance
    schema_version:       int                    # = 1 for v1; bumps need a parser branch
    engine:               str                    # "pyscf" today
    engine_version:       str
    molbuilder_version:   str
    timestamp:            str                    # ISO-8601 UTC

    structure_hash:       str                    # "sha256:..." of canonical XYZ
    n_atoms_total:        int
    free_atom_idxs:       List[int]              # 0-based, complement of frozen
    frozen_atom_idxs:      List[int]              # 0-based

    # Reference SCF + MO spectrum at the input (un-displaced) geometry.
    equilibrium_scf_eh:        float
    equilibrium_mo_energies_eh: np.ndarray       # ALL MOs (not the window subset)
    equilibrium_homo_idx:      int               # index into the array above

    modes:                     List[ModeData]    # sorted by frequency ascending

    # Which modes were selected for ES treatment (1-based indices).
    # When live-watching, modes in this list may have
    # electronic_structure = None until their SCFs complete.
    selected_mode_idxs_1based: List[int]

    # The originating config as JSON-safe dict (provenance + replay).
    config:                    Dict[str, Any]

    # Methods-section prose + bibliography keys actually cited.
    # Populated as layers complete (grows as the run progresses);
    # may be empty strings / lists during early phases.
    methods_text:              str
    bibliography_keys:         List[str]

    # Per-layer status flags (spec § 5 + § 6.1).  Replaces the
    # older single `complete: bool` because the four-layer linear-
    # chain model needs per-phase granularity for the stepper UI
    # and the live-watch state machine.
    #
    # Each one of PHASE_EMPTY / PHASE_RUNNING / PHASE_COMPLETE
    # (validated at __post_init__).  L1 (Setup) has no flag of its
    # own -- the presence of a valid SpectraResults IS the
    # Setup-complete signal.
    phase_frequencies:         str = PHASE_EMPTY
    phase_raman:               str = PHASE_EMPTY
    phase_es:                  str = PHASE_EMPTY
    #: v5: the in-deck relaxation is a tracked step (user, 2026-08-20 --
    #: the viewer tracks ALL the steps; a silent gap while geomeTRIC works
    #: would betray that).  Complete-by-assertion under already_relaxed,
    #: with the gradient-check number in `relaxation` beside it.
    phase_relaxation:          str = PHASE_EMPTY
    #: v5: relaxation progress/result -- {enabled, already_relaxed,
    #: n_steps, max_force_eh_a, converged, warning?}.  Written live so the
    #: chip can show "step 14, max force 0.0042" ticking down.
    relaxation:                Dict[str, Any] = field(default_factory=dict)
    #: v5: RRHO thermochemistry (D2's re-homing) -- headline numbers at
    #: (temperature_K, pressure_atm), the T-grid arrays the viewer's
    #: G/H/S curves draw, and `regime`: "rrho" for a free molecule,
    #: "vibrational-only" (stated, never refused) when atoms are frozen --
    #: an anchored molecule does not rotate.
    thermo:                    Dict[str, Any] = field(default_factory=dict)

    # Equilibrium geometry -- element symbols + Cartesian positions
    # in Å.  Optional in the wire format (older results from
    # SCHEMA_VERSION=1 won't have this) so reading an older JSON
    # still works.  When present, the UI animates modes directly
    # from the loaded results without needing the user to keep the
    # XYZ in the input form.
    equilibrium_elements:      Optional[List[str]]  = None
    equilibrium_positions_ang: Optional[np.ndarray] = None

    # Engine-specific noise (parsing diagnostics, version detail) --
    # kept here so the common schema doesn't bloat for engine-only
    # fields and the UI can ignore it.
    engine_metadata:           Dict[str, Any] = field(default_factory=dict)

    # Runtime facts captured by the emitted script when it ran.  The
    # canonical key list is molbuilder.runtime_info.RUNTIME_INFO_KEYS --
    # NOT restated here.  It used to be, and the copy silently lost
    # ``max_memory_mb``; a list repeated in prose is a list that drifts.
    # Engines may also record keys beyond the canonical set (the PySCF
    # script adds scf_conv_tol, scf_solver_class, ...), so treat this as
    # an open dict whose canonical members are documented there.  Lets the
    # /results page display "this run used 20 PySCF threads, BLAS=1,
    # GPU ON (RTX 4090, CC 8.9, CUDA 12.4)" so users can verify
    # the resources actually matched what they expected -- a 40-on-
    # 20-cores oversubscription leaves a clear trail.  Optional
    # (older v4 results may not have all keys; the UI no-ops on
    # missing keys).
    runtime_info:              Dict[str, Any] = field(default_factory=dict)

    __eq__ = _no_equality

    def __post_init__(self):
        """Normalise + cross-field shape-validate.

        At the SpectraResults level we can check that:
          * equilibrium_mo_energies_eh is a 1-D float array;
          * homo_idx is in range;
          * free_atom_idxs + frozen_atom_idxs partition [0, n_atoms_total);
          * every mode's two eigenvector arrays (canonical, display)
            have the same n_free (= len(free_atom_idxs));
          * every mode's electronic_structure (when present) has
            the same window size.

        Anything that fails here is a programmer / parser error
        (the dataclass should never have been constructed); raising
        at __post_init__ catches the bug at the construction site
        rather than when the UI hits the inconsistency rendering.
        """
        # Equilibrium MO array.
        self.equilibrium_mo_energies_eh = _reject_complex_then_asarray(
            self.equilibrium_mo_energies_eh,
            field="SpectraResults.equilibrium_mo_energies_eh",
        )
        if self.equilibrium_mo_energies_eh.ndim != 1:
            raise ValueError(
                f"SpectraResults.equilibrium_mo_energies_eh must be 1-D; "
                f"got shape {self.equilibrium_mo_energies_eh.shape}"
            )
        n_mos = self.equilibrium_mo_energies_eh.size
        if not 0 <= self.equilibrium_homo_idx < n_mos:
            raise ValueError(
                f"SpectraResults.equilibrium_homo_idx={self.equilibrium_homo_idx} "
                f"out of range [0, {n_mos})"
            )
        # Free + fixed atom partition.
        free_set   = set(int(i) for i in self.free_atom_idxs)
        frozen_set = set(int(i) for i in self.frozen_atom_idxs)
        if free_set & frozen_set:
            raise ValueError(
                f"SpectraResults: free_atom_idxs and frozen_atom_idxs overlap "
                f"at indices {sorted(free_set & frozen_set)}"
            )
        # True partition: the union must be EXACTLY range(n_atoms_total).  A
        # count-only check passed an out-of-range index (e.g. free=[0,1,5],
        # frozen=[], n=3) -- which would then silently drop that atom's
        # displacement in the frontend scatter (spec.md § 5.1 invariant 1).
        expected = set(range(self.n_atoms_total))
        union    = free_set | frozen_set
        if union != expected:
            missing = sorted(expected - union)
            extra   = sorted(union - expected)
            raise ValueError(
                f"SpectraResults: free_atom_idxs + frozen_atom_idxs must "
                f"partition range({self.n_atoms_total}) "
                f"(spec.md § 5.1 invariant 1); "
                f"missing={missing} out-of-range/extra={extra}"
            )
        # Phase status validation.
        for name, val in (("phase_frequencies", self.phase_frequencies),
                          ("phase_raman",       self.phase_raman),
                          ("phase_es",          self.phase_es),
                          ("phase_relaxation",  self.phase_relaxation)):
            if val not in _VALID_PHASE_STATES:
                raise ValueError(
                    f"SpectraResults.{name}={val!r} is not a valid "
                    f"phase status; expected one of {_VALID_PHASE_STATES}"
                )
        # Equilibrium geometry (optional).  When both elements and
        # positions are supplied, validate shape + count match.
        if self.equilibrium_elements is not None or self.equilibrium_positions_ang is not None:
            if self.equilibrium_elements is None or self.equilibrium_positions_ang is None:
                raise ValueError(
                    "SpectraResults: equilibrium_elements and "
                    "equilibrium_positions_ang must be supplied together "
                    "(both or neither)."
                )
            self.equilibrium_positions_ang = _reject_complex_then_asarray(
                self.equilibrium_positions_ang,
                field="SpectraResults.equilibrium_positions_ang",
            )
            if (self.equilibrium_positions_ang.ndim != 2
                    or self.equilibrium_positions_ang.shape[1] != 3):
                raise ValueError(
                    f"SpectraResults.equilibrium_positions_ang must "
                    f"have shape (n_atoms, 3); got "
                    f"{self.equilibrium_positions_ang.shape}"
                )
            n_geom = self.equilibrium_positions_ang.shape[0]
            if n_geom != len(self.equilibrium_elements):
                raise ValueError(
                    f"SpectraResults: equilibrium_elements has "
                    f"{len(self.equilibrium_elements)} symbols but "
                    f"equilibrium_positions_ang has {n_geom} rows"
                )
            if n_geom != self.n_atoms_total:
                raise ValueError(
                    f"SpectraResults: geometry has {n_geom} atoms but "
                    f"n_atoms_total = {self.n_atoms_total}"
                )

        # Cross-mode shape consistency.  Allow the empty-modes case
        # (in-progress write before phase 2 -- no harmonic analysis yet).
        if self.modes:
            n_free = len(free_set)
            expected_shape = (n_free, 3)
            for m in self.modes:
                # The dataclass post_init already pinned canonical.shape
                # == display.shape, so checking either is sufficient.
                if m.eigenvector_canonical.shape != expected_shape:
                    raise ValueError(
                        f"SpectraResults: mode {m.index_1based} has eigenvector "
                        f"shape {m.eigenvector_canonical.shape}, expected "
                        f"{expected_shape}"
                    )
            # Cross-mode ES window-size consistency.
            es_window = None
            for m in self.modes:
                if m.electronic_structure is None:
                    continue
                w = m.electronic_structure.mo_energies_eq_eh.size
                if es_window is None:
                    es_window = w
                elif w != es_window:
                    raise ValueError(
                        f"SpectraResults: mode {m.index_1based} ES window has "
                        f"size {w}; expected {es_window} to match earlier modes"
                    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version":       int(self.schema_version),
            "engine":               str(self.engine),
            "engine_version":       str(self.engine_version),
            "molbuilder_version":   str(self.molbuilder_version),
            "timestamp":            str(self.timestamp),

            "structure_hash":       str(self.structure_hash),
            "n_atoms_total":        int(self.n_atoms_total),
            "free_atom_idxs":       [int(i) for i in self.free_atom_idxs],
            "frozen_atom_idxs":      [int(i) for i in self.frozen_atom_idxs],

            "equilibrium": {
                "scf_energy_eh":     float(self.equilibrium_scf_eh),
                "mo_energies_eh":    self.equilibrium_mo_energies_eh.tolist(),
                "homo_idx":          int(self.equilibrium_homo_idx),
                # Optional geometry; emitted only when present so
                # older readers ignore the missing keys cleanly.
                **({"elements":      [str(e) for e in self.equilibrium_elements]}
                   if self.equilibrium_elements is not None else {}),
                **({"positions_ang": self.equilibrium_positions_ang.tolist()}
                   if self.equilibrium_positions_ang is not None else {}),
            },

            "modes":                [m.to_dict() for m in self.modes],
            "selected_mode_idxs_1based": [int(i) for i in self.selected_mode_idxs_1based],

            "config":               dict(self.config),

            "methods_text":         str(self.methods_text),
            "bibliography_keys":    [str(k) for k in self.bibliography_keys],

            "phase_frequencies":    str(self.phase_frequencies),
            "phase_relaxation":     str(self.phase_relaxation),
            "relaxation":           dict(self.relaxation),
            "thermo":               dict(self.thermo),
            "phase_raman":          str(self.phase_raman),
            "phase_es":             str(self.phase_es),
            "engine_metadata":      dict(self.engine_metadata),
            "runtime_info":         dict(self.runtime_info),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectraResults":
        # Strict version gate (2026-06-26): the decoder self-enforces
        # the schema version rather than trusting the outer sidecar
        # reader.  Missing or non-current versions raise instead of
        # being silently reconstituted at whatever version the payload
        # claims.
        sv = d.get("schema_version")
        if sv is None or int(sv) not in READABLE_SCHEMA_VERSIONS:
            raise ValueError(
                f"SpectraResults: schema_version {sv!r} is not "
                f"supported; this molbuilder build reads "
                f"{sorted(READABLE_SCHEMA_VERSIONS)} (v5 added only the "
                f"optional relaxation/thermo blocks, so a v4 file reads "
                f"whole; older versions do not)."
            )
        eq = d["equilibrium"]
        return cls(
            schema_version       = int(d["schema_version"]),
            engine               = str(d["engine"]),
            engine_version       = str(d["engine_version"]),
            molbuilder_version   = str(d["molbuilder_version"]),
            timestamp            = str(d["timestamp"]),

            structure_hash       = str(d["structure_hash"]),
            n_atoms_total        = int(d["n_atoms_total"]),
            free_atom_idxs       = [int(i) for i in d["free_atom_idxs"]],
            frozen_atom_idxs      = [int(i) for i in d["frozen_atom_idxs"]],

            equilibrium_scf_eh         = float(eq["scf_energy_eh"]),
            equilibrium_mo_energies_eh = np.asarray(eq["mo_energies_eh"], dtype=float),
            equilibrium_homo_idx       = int(eq["homo_idx"]),

            # Optional geometry (added late in the schema; older
            # JSON files don't have these keys, so .get() with None
            # falls through cleanly).
            equilibrium_elements       = (
                [str(e) for e in eq["elements"]]
                if "elements" in eq else None
            ),
            equilibrium_positions_ang  = (
                np.asarray(eq["positions_ang"], dtype=float)
                if "positions_ang" in eq else None
            ),

            modes                = [ModeData.from_dict(m) for m in d["modes"]],
            selected_mode_idxs_1based = [int(i) for i in
                                          d.get("selected_mode_idxs_1based", [])],

            config               = dict(d.get("config", {})),

            methods_text         = str(d.get("methods_text", "")),
            bibliography_keys    = [str(k) for k in d.get("bibliography_keys", [])],

            phase_frequencies    = str(d.get("phase_frequencies", PHASE_EMPTY)),
            phase_relaxation     = str(d.get("phase_relaxation", PHASE_EMPTY)),
            relaxation           = dict(d.get("relaxation") or {}),
            thermo               = dict(d.get("thermo") or {}),
            phase_raman          = str(d.get("phase_raman",       PHASE_EMPTY)),
            phase_es             = str(d.get("phase_es",          PHASE_EMPTY)),
            engine_metadata      = dict(d.get("engine_metadata", {})),
            runtime_info         = dict(d.get("runtime_info", {})),
        )


__all__ = [
    "SCHEMA_VERSION",
    "ModeElectronicStructure",
    "ModeData",
    "SpectraResults",
]
