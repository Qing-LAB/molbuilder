"""L1 result types for the Spectra tab.

Pinned by docs/tabs/spectra/spec.md § 5.  Three dataclasses:

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


SCHEMA_VERSION = 1


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

    The eigenvector is restricted to the *free* atoms (fixed atoms
    don't move), so its shape is ``(n_free, 3)`` -- the global
    ``free_atom_idxs`` on :class:`SpectraResults` maps free-atom
    rows back to atom indices.

    Imaginary modes are reported with a negative frequency and
    :attr:`has_imag` ``= True``.  v1's PySCF engine follows the
    sign convention ``ω = sign(λ) * sqrt(|λ|)`` where λ is the
    mass-weighted Hessian eigenvalue, so a saddle's "imaginary"
    mode becomes a negative real number for plotting purposes.

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

    # Mass-weighted normal-mode eigenvector, shape (n_free, 3),
    # units Å * amu^(-1/2) per the harmonic-analysis convention.
    eigenvector_free:      np.ndarray

    # Sign-of-eigenvalue marker: negative-ω modes are flagged here
    # so the UI / parser / methods generator don't have to
    # re-derive from the sign every time.
    has_imag:              bool

    electronic_structure:  Optional[ModeElectronicStructure] = None

    __eq__ = _no_equality

    def __post_init__(self):
        """Normalise + shape-validate the eigenvector.

        Eigenvector must be 2-D with the last axis = 3 (Cartesian
        x/y/z displacement per free atom).  Number of free atoms
        is implicit from the first axis -- cross-mode consistency
        (every mode has the same n_free) is validated at the
        SpectraResults level, not here.
        """
        self.eigenvector_free = _reject_complex_then_asarray(
            self.eigenvector_free, field="ModeData.eigenvector_free")
        if self.eigenvector_free.ndim != 2 or self.eigenvector_free.shape[1] != 3:
            raise ValueError(
                f"ModeData.eigenvector_free must have shape (n_free, 3); "
                f"got {self.eigenvector_free.shape}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_1based":          int(self.index_1based),
            "frequency_cm1":         float(self.frequency_cm1),
            "raman_activity_a4_amu": (None if self.raman_activity_a4_amu is None
                                      else float(self.raman_activity_a4_amu)),
            "ir_intensity_km_mol":   (None if self.ir_intensity_km_mol is None
                                      else float(self.ir_intensity_km_mol)),
            "eigenvector_free":      self.eigenvector_free.tolist(),
            "has_imag":              bool(self.has_imag),
            "electronic_structure":  (None if self.electronic_structure is None
                                      else self.electronic_structure.to_dict()),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModeData":
        es = d.get("electronic_structure")
        return cls(
            index_1based          = int(d["index_1based"]),
            frequency_cm1         = float(d["frequency_cm1"]),
            raman_activity_a4_amu = (None if d.get("raman_activity_a4_amu") is None
                                     else float(d["raman_activity_a4_amu"])),
            ir_intensity_km_mol   = (None if d.get("ir_intensity_km_mol") is None
                                     else float(d["ir_intensity_km_mol"])),
            eigenvector_free      = np.asarray(d["eigenvector_free"], dtype=float),
            has_imag              = bool(d.get("has_imag", False)),
            electronic_structure  = (None if es is None
                                     else ModeElectronicStructure.from_dict(es)),
        )


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
    free_atom_idxs:       List[int]              # 0-based, complement of fixed
    fixed_atom_idxs:      List[int]              # 0-based

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

    __eq__ = _no_equality

    def __post_init__(self):
        """Normalise + cross-field shape-validate.

        At the SpectraResults level we can check that:
          * equilibrium_mo_energies_eh is a 1-D float array;
          * homo_idx is in range;
          * free_atom_idxs + fixed_atom_idxs partition [0, n_atoms_total);
          * every mode's eigenvector_free has the same n_free
            (= len(free_atom_idxs));
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
        free_set  = set(int(i) for i in self.free_atom_idxs)
        fixed_set = set(int(i) for i in self.fixed_atom_idxs)
        if free_set & fixed_set:
            raise ValueError(
                f"SpectraResults: free_atom_idxs and fixed_atom_idxs overlap "
                f"at indices {sorted(free_set & fixed_set)}"
            )
        if len(free_set) + len(fixed_set) != self.n_atoms_total:
            raise ValueError(
                f"SpectraResults: free ({len(free_set)}) + fixed "
                f"({len(fixed_set)}) atom counts != n_atoms_total "
                f"({self.n_atoms_total})"
            )
        # Phase status validation.
        for name, val in (("phase_frequencies", self.phase_frequencies),
                          ("phase_raman",       self.phase_raman),
                          ("phase_es",          self.phase_es)):
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
            for m in self.modes:
                if m.eigenvector_free.shape != (n_free, 3):
                    raise ValueError(
                        f"SpectraResults: mode {m.index_1based} has eigenvector "
                        f"shape {m.eigenvector_free.shape}, expected ({n_free}, 3)"
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
            "fixed_atom_idxs":      [int(i) for i in self.fixed_atom_idxs],

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
            "phase_raman":          str(self.phase_raman),
            "phase_es":             str(self.phase_es),
            "engine_metadata":      dict(self.engine_metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectraResults":
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
            fixed_atom_idxs      = [int(i) for i in d["fixed_atom_idxs"]],

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
            phase_raman          = str(d.get("phase_raman",       PHASE_EMPTY)),
            phase_es             = str(d.get("phase_es",          PHASE_EMPTY)),
            engine_metadata      = dict(d.get("engine_metadata", {})),
        )


__all__ = [
    "SCHEMA_VERSION",
    "ModeElectronicStructure",
    "ModeData",
    "SpectraResults",
]
