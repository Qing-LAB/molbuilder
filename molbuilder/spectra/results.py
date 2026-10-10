"""L1 result types for the Spectra tab.

Pinned by docs/web/spectra.md  Three dataclasses:

  * :class:`ModeElectronicStructure` -- the per-mode displaced-SCF
    block: equilibrium / ±A·Q_i MO energies + SCF energies.  Populated
    only when the user selected a mode for electronic-structure analysis.
  * :class:`ModeData` -- one vibrational mode: frequency, eigenvector
    (free atoms only), Raman activity, optional IR intensity, and the optional :class:`ModeElectronicStructure`.
  * :class:`SpectraResults` -- the complete result of a Spectra run:
    metadata, equilibrium reference, list of modes, methods text,
    bibliography keys, and the per-phase status flags.

These are the **engine-agnostic** result shape -- the parser
populates them from a ``.spectra.json`` regardless of which engine
produced it.

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

Schema version: :data:`SCHEMA_VERSION`; the reader accepts
:data:`READABLE_SCHEMA_VERSIONS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# TWO WAYS, because this module travels beside the SIESTA finish
# (`runwrap.VIBRATION_COMPANIONS`).
try:                                        # inside molbuilder
    from .derived import DERIVED_MODE_KEYS, disagreements, with_derived
except ImportError:                         # beside a job, in a bundle
    from derived import DERIVED_MODE_KEYS, disagreements, with_derived


# SCHEMA_VERSION (incremented when the on-disk JSON shape changes):
#
#  v4: ``runtime_info`` dict carrying CPU/thread and GPU facts the running
#          script collected.  v3 documents fail with a clear
#          schema-version error.
#  v5: + the OPTIONAL `phase_relaxation` + `relaxation` progress block (the
#          in-deck relaxation is a TRACKED step) and the OPTIONAL `thermo`
#          block.  ADDITIVE -- a v4 file lacks them and reads whole, which is
#          why the reader accepts a SET (the molstruct sidecar's own rule).
#  v6: + the OPTIONAL `removed_motions` block -- how many whole-body motions
#          the harmonic analysis projected out before diagonalising, and
#          their Cartesian patterns over the free atoms (science/normal-modes.md
#          R7: what was removed is stated beside what was kept).  ADDITIVE.
SCHEMA_VERSION = 7   # 7: + `equilibrium.masses_amu`, every atom's mass as
#                    the deck stated it and the analysis weighted by it, beside
#                    the geometry (I3); and each mode's DERIVED values -- the
#                    activity classes, the zero-point amplitude and
#                    displacement -- written by every engine through
#                    `spectra.derived` and checked on read (engines/vibration.md
#                    § 6.6).  A v4-v6 file reads as before: no masses, its
#                    derived values computed on read.
READABLE_SCHEMA_VERSIONS = frozenset({4, 5, 6, 7})


# Phase status vocabulary -- per-layer flag carried on
# :class:`SpectraResults` and emitted in the on-disk JSON.  Pinned
# here as a module constant so the engine, the parser, and the UI
# all read from one source.

PHASE_EMPTY    = "empty"     # not yet computed
PHASE_RUNNING  = "running"   # script is mid-way through this phase
                             # (live-watch / atomic-replace JSON updates)
PHASE_COMPLETE = "complete"  # phase done, data is final for this run
#: A phase the description never asked for -- written from the first
#: write and never changed, so *nothing was asked* cannot be read as
#: *nothing yet* (empty) or *done with nothing behind it* (complete).
#: Terminal: the viewer counts it finished (`engines/vibration.md` § 4.9).
PHASE_NOT_REQUESTED = "not requested"

_VALID_PHASE_STATES = (PHASE_EMPTY, PHASE_RUNNING, PHASE_COMPLETE,
                       PHASE_NOT_REQUESTED)


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


#: THE KEYS EACH BLOCK MAY CARRY -- the rows of `engines/vibration.md` § 6.2-6.4, and
#: nothing else.  The reader refuses a key outside them BY NAME.
#: A key that starts being written is a row in § 9b first, then here.
_ES_KEYS = frozenset({
    "amplitude_ang", "mo_energies_eq_eh", "mo_energies_minus_eh",
    "mo_energies_plus_eh", "homo_index_in_window", "scf_energy_eq_eh",
    "scf_energy_minus_eh", "scf_energy_plus_eh",
})
_MODE_KEYS = frozenset({
    "index_1based", "frequency_cm1", "raman_activity_a4_amu",
    "ir_intensity_km_mol", "eigenvector_canonical", "eigenvector_display",
    "has_imag", "electronic_structure",
    "eigenvector_free",                 # schema v1's single vector, read as both
    # DERIVED (`spectra.derived`, `engines/vibration.md` § 6.6) -- stated by
    # every writer from v7, checked on read.
    *DERIVED_MODE_KEYS,
})
_EQUILIBRIUM_KEYS = frozenset({
    "scf_energy_eh", "mo_energies_eh", "homo_idx", "elements", "positions_ang",
    "masses_amu",
})
_RESULTS_KEYS = frozenset({
    "schema_version", "engine", "engine_version", "molbuilder_version",
    "timestamp", "structure_hash", "n_atoms_total", "free_atom_idxs",
    "frozen_atom_idxs", "equilibrium", "modes", "selected_mode_idxs_1based",
    "config", "methods_text", "bibliography_keys", "phase_frequencies",
    "phase_relaxation", "relaxation", "thermo", "removed_motions",
    "hessian_scope", "n_atoms_in_hessian", "hessian_density_fit", "ir_route",
    "ir_fd_step_ang", "raman_route", "raman_fd_step_ang", "phase_raman",
    "phase_ir", "phase_es", "engine_metadata", "runtime_info",
})


def _refuse_unknown_keys(d, known, where: str) -> None:
    unknown = sorted(str(k) for k in d if k not in known)
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown} -- every key of this file "
            f"has a row in engines/vibration.md 6.2-6.4, and a key this reader does "
            f"not know is a number it would throw away in silence")


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
        numpy.  An unknown key is refused by name, never ignored."""
        _refuse_unknown_keys(d, _ES_KEYS, "ModeElectronicStructure")
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
    modes the user selected via the Model 2 selector.
    Unselected modes have ``electronic_structure = None`` -- the UI
    renders an empty cell + "—" in the mode-list ES columns.

    ``ir_intensity_km_mol`` is ``None`` when IR was not requested, and a
    number (0.00 included -- a mode can be genuinely IR-inactive) when it
    was.  How that number was obtained is recorded once for the run in
    :attr:`SpectraResults.ir_route`, not per mode, because one route
    produces the whole tensor.
    """

    index_1based:         int
    frequency_cm1:        float

    # Activities / intensities are optional because either channel may
    # simply not have been asked for -- `compute_raman` off for a
    # frequencies-only run, `compute_ir` off for a Raman-only one.  None
    # is "not computed" and is NEVER the same statement as 0.0, which is
    # a measured absence; `spectra.activity` keeps the two apart and the
    # viewer colours them differently.
    raman_activity_a4_amu: Optional[float]
    ir_intensity_km_mol:   Optional[float]

    # Cartesian normal mode in the canonical mass-weighted convention:
    #     Σ_k m_k |L_k|² = 1     (m_k in amu, `equilibrium.masses_amu`)
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
        _refuse_unknown_keys(d, _MODE_KEYS, "ModeData")
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
            _disp = d.get("eigenvector_display", d.get("eigenvector_free"))
            if _disp is None:
                # DERIVED (`engines/vibration.md` § 6.4): the canonical vector
                # rescaled per mode so max|L| = 1 -- what the animation
                # draws.  An engine writes the science form; this is ours.
                _peak = float(np.max(np.abs(ev_canon))) if ev_canon.size else 0.0
                ev_disp = ev_canon / _peak if _peak > 0 else ev_canon.copy()
            else:
                ev_disp = np.asarray(_disp, dtype=float)
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


def mass_of(i: int, elements: Sequence[str],
            masses_amu: Optional[Sequence[float]] = None) -> float:
    """Atom ``i``'s mass a result is weighted by, in amu: as the result
    states it (``equilibrium.masses_amu``, schema 7), or -- for a result
    before it, which states none -- ``chemistry.atomic_mass``'s, the table
    every deck's masses are stated from (`engines/vibration.md` § 6.6)."""
    if masses_amu is not None:
        return float(masses_amu[i])
    from ..chemistry import atomic_mass
    return atomic_mass(str(elements[i]))


def motion_share_by_element(elements: List[str],
                            eigenvector: Any,
                            atom_idxs: Optional[List[int]] = None,
                            masses_amu: Optional[Sequence[float]] = None,
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
    Omit it when there is one row per atom.  ``masses_amu`` is every
    atom's mass as the result states it (``equilibrium.masses_amu``, schema
    7); a result before it states none, and the shares are then weighted by
    ``chemistry.atomic_mass``, the table every deck's masses are stated from
    (`engines/vibration.md` § 6.6).  Shares are returned largest first, and
    a mode with no motion at all returns ``{}`` rather than dividing by zero.
    """
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
        w = mass_of(at, elements, masses_amu) * float(np.dot(row, row))
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

    Progress is carried by the per-phase ``phase_*`` flags.  While the
    run is mid-way some modes may have ``electronic_structure = None``
    not because the user de-selected them but because their displaced
    SCFs haven't run yet.  The ``selected_mode_idxs_1based`` field tells
    the UI which modes WILL get ES data so it can show progress
    ("3 of 10 modes done").

    Run identity is captured by ``structure_hash`` (SHA-256 of the
    canonical XYZ of the input structure) so the parser can refuse
    to merge results from a different molecule; ``engine`` +
    ``engine_version`` + ``molbuilder_version`` give the software
    stack used; ``timestamp`` is ISO-8601 UTC.
    """

    # Provenance
    schema_version:       int
    engine:               str
    engine_version:       str
    molbuilder_version:   str
    timestamp:            str                    # ISO-8601 UTC

    structure_hash:       str                    # "sha256:..." of canonical XYZ
    n_atoms_total:        int
    free_atom_idxs:       List[int]              # 0-based, complement of frozen
    frozen_atom_idxs:      List[int]              # 0-based

    # Reference SCF + MO spectrum at the input (un-displaced) geometry.
    # OPTIONAL as a block (`web/spectra.md` § 9b.3): a molecular-orbital
    # spectrum is PySCF's; SIESTA's force-constant route has a total energy
    # it does not report here and no HOMO to name.  ``None`` is "this engine
    # has none", never a fabricated orbital 0.  The three travel together.
    equilibrium_scf_eh:        Optional[float]
    equilibrium_mo_energies_eh: Optional[np.ndarray]  # ALL MOs (not the window subset)
    equilibrium_homo_idx:      Optional[int]      # index into the array above

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

    # Per-layer status flags: the four-layer linear-chain model needs
    # per-phase granularity for the stepper UI and the live-watch state
    # machine.
    #
    # Each one of PHASE_EMPTY / PHASE_RUNNING / PHASE_COMPLETE
    # (validated at __post_init__).  L1 (Setup) has no flag of its
    # own -- the presence of a valid SpectraResults IS the
    # Setup-complete signal.
    #: How dmu/dR was obtained, for the run as a whole:
    #:   "analytic"          -- pyscf.prop.infrared, off the Hessian's own
    #:                          CPHF solution, no extra SCFs;
    #:   "finite-difference" -- the 6N-SCF dipole sweep, either because
    #:                          Raman's displacement loop was running
    #:                          anyway (so the dipole read was free) or
    #:                          because the analytic module is absent;
    #:   "none"              -- IR was not requested.
    #: Older sidecars predate the field and parse as "" -- absence of a
    #: record, which the viewer must not render as a claim either way.
    ir_route:                  str = ""
    #: The finite-difference step actually used for dmu/dR, in Angstrom,
    #: when (and only when) that route ran.  Carried because a Methods
    #: section quoting a finite-difference derivative has to state its
    #: step to be reproducible; ``None`` on the analytic route, which
    #: has no step to state.
    ir_fd_step_ang:            Optional[float] = None
    #: How dalpha/dR was obtained, for the run as a whole (engines/vibration.md § 4.6):
    #:   "finite-difference" -- central differences of the analytic CPHF
    #:                          polarizability over the free Cartesians,
    #:                          the one Raman route there is;
    #:   "none"              -- Raman was not requested, or the engine
    #:                          computes no intensities.
    #: Older sidecars parse as "" -- absence of a record.
    raman_route:               str = ""
    #: The step of that difference, in Angstrom, when the route ran.
    raman_fd_step_ang:         Optional[float] = None
    phase_frequencies:         str = PHASE_EMPTY
    phase_raman:               str = PHASE_EMPTY
    #: The infrared intensities' own flag (`engines/vibration.md` § 4.9):
    #: closed with the Raman sweep that carries the dipole, or by the
    #: dipole sweep alone.  Older sidecars parse as "" -- absence of a
    #: record, which a reader must not take for *empty* (asked, not yet
    #: started): a finished run written before the flag has none.
    phase_ir:                  str = ""
    phase_es:                  str = PHASE_EMPTY
    #: v5: the in-deck relaxation is a tracked step (user, 2026-08-20 --
    #: the viewer tracks ALL the steps; a silent gap while geomeTRIC works
    #: would betray that).  Complete-by-assertion under already_relaxed,
    #: with the gradient-check number in `relaxation` beside it.
    phase_relaxation:          str = PHASE_EMPTY
    #: v5: relaxation progress/result -- {enabled, already_relaxed,
    #: n_steps, max_force_eh_bohr, max_force_all_atoms_eh_bohr, converged, warning?}.  Written live so the
    #: chip can show "step 14, max force 0.0042" ticking down.
    relaxation:                Dict[str, Any] = field(default_factory=dict)
    #: v5: RRHO thermochemistry -- headline numbers at
    #: temperature_K (and pressure_atm for "rrho"; null for
    #: "vibrational-only", which no pressure enters), the T-grid arrays the
    #: viewer's curves draw, and `regime`: "rrho" for a free molecule,
    #: "vibrational-only" (stated, never refused) when atoms are held: there
    #: is no gas-phase partition function to add, and the whole-body motions
    #: that survived the hold were removed before diagonalising
    #: (`removed_motions`).  The headline and every grid point are ONE sum,
    #: and the headline temperature is on the grid (engines/vibration.md § 4.7).
    thermo:                    Dict[str, Any] = field(default_factory=dict)
    #: v6: what the harmonic analysis removed before diagonalising --
    #: {count, patterns: (count, n_free, 3) Cartesian, orthonormal over the
    #: free atoms}.  Every mode in `modes` is a vibration BECAUSE these were
    #: taken out first (science/normal-modes.md R3-R4); a reader that wants
    #: to see the difference between 3 N_free and len(modes) finds it here.
    #: Empty on a file written before the block existed.
    removed_motions:           Dict[str, Any] = field(default_factory=dict)
    #: v6: what the Hessian covered -- "free" (second derivatives for the
    #: free atoms only; atoms are held) or "all" (every atom; nothing held)
    #: -- and how many atoms that was.  A reader comparing two runs' costs
    #: or Methods paragraphs needs it stated, not inferred from the frozen
    #: list.  "" on a file written before the field existed.
    hessian_scope:             str = ""
    n_atoms_in_hessian:        Optional[int] = None
    #: v6: whether the Hessian itself was density fitted.  The SCF may be
    #: while the Hessian is not (the free-atom route), and a Methods
    #: paragraph has to say which.  None on a file written before the field.
    hessian_density_fit:       Optional[bool] = None

    # Equilibrium geometry -- element symbols + Cartesian positions
    # in Å.  Optional in the wire format.  When present, the UI animates modes directly
    # from the loaded results without needing the user to keep the
    # XYZ in the input form.
    equilibrium_elements:      Optional[List[str]]  = None
    equilibrium_positions_ang: Optional[np.ndarray] = None
    #: v7: every atom's mass in amu, in the input order -- the masses the
    #: analysis weighted by, as the deck stated them (`engines/vibration.md`
    #: § 6.2, I3).  Travels with the geometry: from v7 a file stating the
    #: geometry states them, and a file before v7 states none.
    equilibrium_masses_amu:    Optional[np.ndarray] = None

    # Engine-specific noise (parsing diagnostics, version detail) --
    # kept here so the common schema doesn't bloat for engine-only
    # fields and the UI can ignore it.
    engine_metadata:           Dict[str, Any] = field(default_factory=dict)

    # Runtime facts captured by the emitted script when it ran.  The
    # canonical key list is molbuilder.runtime_info.RUNTIME_INFO_KEYS --
    # NOT restated here: a list repeated in prose is a list that drifts.
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
        # Equilibrium block -- present whole, or absent whole.  A null in
        # one slot of a block the other slots fill is a broken file, not an
        # engine that has none: the energy is the reference every displaced
        # SCF is measured against, and a None there fails a hundred lines
        # later inside a subtraction.
        _eq = (self.equilibrium_scf_eh, self.equilibrium_mo_energies_eh,
               self.equilibrium_homo_idx)
        _absent = [v is None for v in _eq]
        if any(_absent) and not all(_absent):
            raise ValueError(
                "SpectraResults: the equilibrium block (scf_energy_eh, "
                "mo_energies_eh, homo_idx) travels whole or not at all; "
                f"absent = {dict(zip(('scf_energy_eh', 'mo_energies_eh', 'homo_idx'), _absent))}")
        if self.equilibrium_mo_energies_eh is not None:
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
            if (self.equilibrium_homo_idx is None
                    or not 0 <= self.equilibrium_homo_idx < n_mos):
                raise ValueError(
                    f"SpectraResults.equilibrium_homo_idx="
                    f"{self.equilibrium_homo_idx} out of range [0, {n_mos})"
                )
        # Free + fixed atom partition.
        free_set   = set(int(i) for i in self.free_atom_idxs)
        frozen_set = set(int(i) for i in self.frozen_atom_idxs)
        if free_set & frozen_set:
            raise ValueError(
                f"SpectraResults: free_atom_idxs and frozen_atom_idxs overlap "
                f"at indices {sorted(free_set & frozen_set)}"
            )
        # True partition: the union must be EXACTLY range(n_atoms_total) --
        # an out-of-range index would silently drop that atom's
        # displacement in the frontend scatter (`web/spectra.md` § 8).
        # Without materialising range(n_atoms_total): a file claiming 1e12
        # atoms must be refused, not answered with a MemoryError.  A set of n distinct indices all inside [0, n) IS
        # range(n); the listing of what is missing stops after twenty.
        n = int(self.n_atoms_total)
        union = free_set | frozen_set
        extra = sorted(i for i in union if not 0 <= i < n)
        if extra or len(union) != n:
            from itertools import islice
            missing = list(islice((i for i in range(max(n, 0))
                                   if i not in union), 20))
            more = "..." if len(union) + len(missing) < n else ""
            raise ValueError(
                f"SpectraResults: free_atom_idxs + frozen_atom_idxs must "
                f"partition range({n}) "
                f"(web/spectra.md § 8); "
                f"missing={missing}{more} out-of-range/extra={extra}"
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
        if self.phase_ir not in _VALID_PHASE_STATES + ("",):
            raise ValueError(
                f"SpectraResults.phase_ir={self.phase_ir!r} is not a valid "
                f"phase status; expected one of {_VALID_PHASE_STATES}, or "
                f"'' for a file written before the flag")
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
        # THE MASSES TRAVEL WITH THE GEOMETRY (v7, I3): one positive finite
        # number per atom, stated wherever the geometry is.
        if self.equilibrium_masses_amu is not None:
            if self.equilibrium_positions_ang is None:
                raise ValueError(
                    "SpectraResults: equilibrium masses without the geometry "
                    "they belong to")
            self.equilibrium_masses_amu = _reject_complex_then_asarray(
                self.equilibrium_masses_amu,
                field="SpectraResults.equilibrium_masses_amu")
            m = self.equilibrium_masses_amu
            if m.shape != (int(self.n_atoms_total),) or not (
                    np.isfinite(m).all() and (m > 0).all()):
                raise ValueError(
                    f"SpectraResults: equilibrium masses_amu must be one "
                    f"positive finite number per atom ({self.n_atoms_total}); "
                    f"got {m.tolist()}")
        elif (int(self.schema_version) >= 7
                and self.equilibrium_positions_ang is not None):
            raise ValueError(
                f"SpectraResults: a schema-{self.schema_version} result states "
                f"its geometry without its masses -- equilibrium.masses_amu "
                f"travels with it (engines/vibration.md 6.2)")

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

    def _modes_with_derived(self) -> List[Dict[str, Any]]:
        """Mode dicts, each with its DERIVED values stated beside the rest --
        the activity classes (a whole-run decision: a mode is active when
        it clears a fraction of the STRONGEST band in its own channel, and
        one mode does not know the others) and the zero-point amplitude and
        displacement -- through the one derivation every writer uses
        (`spectra.derived`, `engines/vibration.md` § 6.6)."""
        return with_derived([m.to_dict() for m in self.modes])

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
                "scf_energy_eh":     (None if self.equilibrium_scf_eh is None
                                      else float(self.equilibrium_scf_eh)),
                "mo_energies_eh":    (None if self.equilibrium_mo_energies_eh is None
                                      else self.equilibrium_mo_energies_eh.tolist()),
                "homo_idx":          (None if self.equilibrium_homo_idx is None
                                      else int(self.equilibrium_homo_idx)),
                # Optional geometry; emitted only when present so
                # older readers ignore the missing keys cleanly.
                **({"elements":      [str(e) for e in self.equilibrium_elements]}
                   if self.equilibrium_elements is not None else {}),
                **({"positions_ang": self.equilibrium_positions_ang.tolist()}
                   if self.equilibrium_positions_ang is not None else {}),
                **({"masses_amu":    self.equilibrium_masses_amu.tolist()}
                   if self.equilibrium_masses_amu is not None else {}),
            },

            # THE DERIVED VALUES ride WITH the modes, through the one
            # derivation (`spectra.derived`): the activity classes -- a
            # whole-run judgement, never re-derived as an epsilon in the
            # viewer -- and the zero-point amplitude and displacement.
            "modes":                self._modes_with_derived(),
            "selected_mode_idxs_1based": [int(i) for i in self.selected_mode_idxs_1based],

            "config":               dict(self.config),

            "methods_text":         str(self.methods_text),
            "bibliography_keys":    [str(k) for k in self.bibliography_keys],

            "phase_frequencies":    str(self.phase_frequencies),
            "phase_relaxation":     str(self.phase_relaxation),
            "relaxation":           dict(self.relaxation),
            "thermo":               dict(self.thermo),
            "removed_motions":      dict(self.removed_motions),
            "hessian_scope":        str(self.hessian_scope),
            "n_atoms_in_hessian":   (None if self.n_atoms_in_hessian is None
                                     else int(self.n_atoms_in_hessian)),
            "hessian_density_fit":  (None if self.hessian_density_fit is None
                                     else bool(self.hessian_density_fit)),
            "ir_route":             str(self.ir_route),
            "ir_fd_step_ang":       (None if self.ir_fd_step_ang is None
                                     else float(self.ir_fd_step_ang)),
            "raman_route":          str(self.raman_route),
            "raman_fd_step_ang":    (None if self.raman_fd_step_ang is None
                                     else float(self.raman_fd_step_ang)),
            "phase_raman":          str(self.phase_raman),
            "phase_ir":             str(self.phase_ir),
            "phase_es":             str(self.phase_es),
            "engine_metadata":      dict(self.engine_metadata),
            "runtime_info":         dict(self.runtime_info),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectraResults":
        # Strict version gate: the decoder self-enforces
        # the schema version rather than trusting the outer sidecar
        # reader.  Missing or unreadable versions raise instead of
        # being silently reconstituted at whatever version the payload
        # claims.
        sv = d.get("schema_version")
        if sv is None or int(sv) not in READABLE_SCHEMA_VERSIONS:
            raise ValueError(
                f"SpectraResults: schema_version {sv!r} is not "
                f"supported; this molbuilder build reads "
                f"{sorted(READABLE_SCHEMA_VERSIONS)} (v5 and v6 added only "
                f"optional blocks -- relaxation/thermo, then "
                f"removed_motions -- and v7 the masses and the stated "
                f"derived values, so a v4 file reads whole; older versions "
                f"do not)."
            )
        _refuse_unknown_keys(d, _RESULTS_KEYS, "SpectraResults")
        eq = d["equilibrium"]
        _refuse_unknown_keys(eq, _EQUILIBRIUM_KEYS, "SpectraResults.equilibrium")
        # THE DERIVED VALUES A v7 FILE STATES ARE ITS INPUTS' (§ 6.6): each
        # computed again by the one derivation, and a file stating another
        # is refused by name.  A file before v7 states them or not; they are
        # computed on read either way.
        if int(sv) >= 7:
            wrong = disagreements(d.get("modes") or [])
            if wrong:
                raise ValueError(
                    "SpectraResults: the derived values this file states are "
                    "not its own inputs' (engines/vibration.md 6.6) -- "
                    + "; ".join(wrong[:5])
                    + (f"; and {len(wrong) - 5} more" if len(wrong) > 5
                       else ""))
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

            equilibrium_scf_eh         = (None if eq.get("scf_energy_eh") is None
                                          else float(eq["scf_energy_eh"])),
            equilibrium_mo_energies_eh = (None if eq.get("mo_energies_eh") is None
                                          else np.asarray(eq["mo_energies_eh"], dtype=float)),
            equilibrium_homo_idx       = (None if eq.get("homo_idx") is None
                                          else int(eq["homo_idx"])),

            # Optional geometry: absent keys read as None.
            equilibrium_elements       = (
                [str(e) for e in eq["elements"]]
                if "elements" in eq else None
            ),
            equilibrium_positions_ang  = (
                np.asarray(eq["positions_ang"], dtype=float)
                if "positions_ang" in eq else None
            ),
            equilibrium_masses_amu     = (
                np.asarray(eq["masses_amu"], dtype=float)
                if "masses_amu" in eq else None
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
            removed_motions      = dict(d.get("removed_motions") or {}),
            hessian_scope        = str(d.get("hessian_scope", "")),
            n_atoms_in_hessian   = (None if d.get("n_atoms_in_hessian") is None
                                    else int(d["n_atoms_in_hessian"])),
            hessian_density_fit  = (None if d.get("hessian_density_fit") is None
                                    else bool(d["hessian_density_fit"])),
            ir_route             = str(d.get("ir_route", "")),
            ir_fd_step_ang       = (None if d.get("ir_fd_step_ang") is None
                                    else float(d["ir_fd_step_ang"])),
            raman_route          = str(d.get("raman_route", "")),
            raman_fd_step_ang    = (None if d.get("raman_fd_step_ang") is None
                                    else float(d["raman_fd_step_ang"])),
            phase_raman          = str(d.get("phase_raman",       PHASE_EMPTY)),
            phase_ir             = str(d.get("phase_ir",          "")),
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
