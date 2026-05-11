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

Schema version: pinned at 1 on :data:`SpectraResults.schema_version`.
Bumping the version requires the parser to grow a per-version
branch; see spec § 6.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


SCHEMA_VERSION = 1


# --------------------------------------------------------------------- #
#  Per-mode electronic structure                                        #
# --------------------------------------------------------------------- #


@dataclass
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


@dataclass
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


@dataclass
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
    # Populated when complete = True; may be empty strings / lists
    # during live watch.
    methods_text:              str
    bibliography_keys:         List[str]

    # Live-watch flag: True iff the run wrote its final phase.
    complete:                  bool

    # Engine-specific noise (parsing diagnostics, version detail) --
    # kept here so the common schema doesn't bloat for engine-only
    # fields and the UI can ignore it.
    engine_metadata:           Dict[str, Any] = field(default_factory=dict)

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
            },

            "modes":                [m.to_dict() for m in self.modes],
            "selected_mode_idxs_1based": [int(i) for i in self.selected_mode_idxs_1based],

            "config":               dict(self.config),

            "methods_text":         str(self.methods_text),
            "bibliography_keys":    [str(k) for k in self.bibliography_keys],

            "complete":             bool(self.complete),
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

            modes                = [ModeData.from_dict(m) for m in d["modes"]],
            selected_mode_idxs_1based = [int(i) for i in
                                          d.get("selected_mode_idxs_1based", [])],

            config               = dict(d.get("config", {})),

            methods_text         = str(d.get("methods_text", "")),
            bibliography_keys    = [str(k) for k in d.get("bibliography_keys", [])],

            complete             = bool(d.get("complete", False)),
            engine_metadata      = dict(d.get("engine_metadata", {})),
        )


__all__ = [
    "SCHEMA_VERSION",
    "ModeElectronicStructure",
    "ModeData",
    "SpectraResults",
]
