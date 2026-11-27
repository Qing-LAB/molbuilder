"""L1 result types of a vibration's result, ``<label>.spectra.json``.

Pinned by `engines/vibration.md` § 6 (the result file).  Three dataclasses:

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

Schema version: :data:`SCHEMA_VERSION`, the one the reader reads
(:data:`READABLE_SCHEMA_VERSIONS`, :func:`readable_version`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the SIESTA finish and the PySCF
# script (`runwrap.members`).
try:                                        # inside molbuilder
    from .derived import (THERMAL_KEY, disagreements, free_masses_of,
                          stated_temperatures, thermal_block, with_derived)
    from .normal_modes import mode_temperatures
    from .result_keys import (CHARGE_STEP_KEYS, EQUILIBRIUM_KEYS, ES_KEYS,
                              MODE_KEYS, OPTICAL_KEYS, OPTICAL_MODE_KEYS,
                              PARTIAL_CHARGES_KEYS, RESULT_KEYS)
    from ..atom_permutation import MAP_KEY
    from ..persist import LATER_WAY, check_keys
    from ..structure import Structure as _Structure
    from ..parse.engines.siesta_grammar import (CHARGE_PARTITIONS,
                                                FC_STEP_PATTERN,
                                                fc_step_count)
    from ..parse.engines._run_ending import FAILED, NOT_DONE, NOT_RUN
except ImportError:                         # beside a job, in a bundle
    from derived import (  # type: ignore[no-redef]
        THERMAL_KEY, disagreements, free_masses_of, stated_temperatures,
        thermal_block, with_derived)
    from normal_modes import mode_temperatures  # type: ignore[no-redef]
    from result_keys import (  # type: ignore[no-redef]
        CHARGE_STEP_KEYS, EQUILIBRIUM_KEYS, ES_KEYS, MODE_KEYS,
        OPTICAL_KEYS, OPTICAL_MODE_KEYS, PARTIAL_CHARGES_KEYS, RESULT_KEYS)
    from atom_permutation import MAP_KEY  # type: ignore[no-redef]
    from persist import LATER_WAY, check_keys  # type: ignore[no-redef]
    from structure import Structure as _Structure  # type: ignore[no-redef]
    from siesta_grammar import (  # type: ignore[no-redef]
        CHARGE_PARTITIONS, FC_STEP_PATTERN, fc_step_count)
    from _run_ending import (  # type: ignore[no-redef]
        FAILED, NOT_DONE, NOT_RUN)


#: THE ON-DISK JSON SHAPE'S VERSION, moved when the shape changes -- each
#: version's change is `engines/vibration.md` § 6.1's table.  15: each mode's
#: MECHANICS, stated -- `motion_share` (each free atom's share of the motion),
#: `largest_zero_point_move_ang`, `reduced_mass_amu` and
#: `force_constant_mdyn_per_ang` on every mode, and the block `modes_thermal`,
#: each mode's spread and mean quanta at the modes' own temperatures --
#: written by every engine through `spectra.derived` and checked on read
#: (§ 6.11 § 3).
SCHEMA_VERSION = 15
#: THE RELEASE THAT FIRST WROTE IT, and its date -- said to a person whose
#: result is older (:func:`readable_version`).
SCHEMA_WRITTEN_SINCE = "2.79.0 (2026-11-23)"
#: The versions the reader reads: the one this molbuilder writes -- a result
#: of an older one is refused and the vibration run again (the user's
#: decision, 2026-11-26).
READABLE_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})


def readable_version(actual: Any) -> int:
    """THE RESULT FILE'S VERSION, if this build reads it -- the one rule,
    asked by the sidecar reader and :meth:`SpectraResults.from_dict` alike:
    a whole number (never a bool, which ``True == 1`` would let past, nor a
    string or a float) of :data:`READABLE_SCHEMA_VERSIONS`; else
    :class:`ValueError` in a person's words, naming the version the file
    states, the one this molbuilder reads and the way on
    (`engines/vibration.md` § 6.1, § 6.7)."""
    reads = f"this molbuilder reads schema {SCHEMA_VERSION} alone"
    if actual is None:
        raise ValueError(f"the result states no schema version: {reads}")
    if not isinstance(actual, int) or isinstance(actual, bool):
        raise ValueError(f"the result's schema version {actual!r} is not a "
                         f"whole number: {reads}")
    if actual < SCHEMA_VERSION:
        raise ValueError(f"the result is schema {actual}, written by a "
                         f"molbuilder before {SCHEMA_WRITTEN_SINCE}: {reads} "
                         f"-- run the vibration again")
    if actual not in READABLE_SCHEMA_VERSIONS:
        raise ValueError(f"the result is schema {actual}, written by a later "
                         f"molbuilder than this one: {reads} -- {LATER_WAY}")
    return actual


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

#: HOW A RESULT'S STRENGTHS WERE OBTAINED, for the run as a whole
#: (`engines/vibration.md` § 4.6): each route word the engine's own road
#: writes, read by the Methods paragraph and the Raman view; checked at
#: ``__post_init__``.
ROUTE_ANALYTIC = "analytic"
ROUTE_FINITE_DIFFERENCE = "finite-difference"
ROUTE_CHARGES = "charges"
ROUTE_OPTICAL = "optical"
ROUTE_NONE = "none"
#: ...the infrared's, dmu/dR (:attr:`SpectraResults.ir_route`), and the
#: Raman's, dalpha/dR (:attr:`SpectraResults.raman_route`).
IR_ROUTES = (ROUTE_ANALYTIC, ROUTE_FINITE_DIFFERENCE, ROUTE_CHARGES,
             ROUTE_NONE)
RAMAN_ROUTES = (ROUTE_FINITE_DIFFERENCE, ROUTE_OPTICAL, ROUTE_NONE)

_VALID_PHASE_STATES = (PHASE_EMPTY, PHASE_RUNNING, PHASE_COMPLETE,
                       PHASE_NOT_REQUESTED)


def finite_or_none(values):
    """``values`` -- orbital energies -- as a list, each NaN or infinity as
    ``None``: what a result holds where one is not finite
    (`engines/vibration.md` § 6.4), since
    `sidecars.spectra.write_spectra_payload` refuses one."""
    return [float(x) if math.isfinite(x) else None
            for x in np.asarray(values, dtype=float)]


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


#: A FRAME'S STATE as the optical citation states it (`engines/vibration.md`
#: § 5.11 § 6, § 7): its point's own word (`runpoints.state_word`) -- but
#: *finished* for done, and *pending* for a call not yet ended (queued,
#: running), the file's own two words.
OPTICAL_FINISHED, OPTICAL_PENDING = "finished", "pending"
OPTICAL_STATES = (OPTICAL_FINISHED, NOT_DONE, FAILED, NOT_RUN,
                  OPTICAL_PENDING)

#: THE PARTITIONS a result's charges may state (`engines/vibration.md`
#: § 5.10 ②, `science/vibrational-optical-response.md` § 1a).
PARTITIONS = tuple(p.name for p in CHARGE_PARTITIONS)


def checked_partial_charges(block: Mapping[str, Any], n_atoms: int,
                            free_atom_idxs: Sequence[int]) -> Dict[str, Any]:
    """A result's ``partial_charges`` (v11), checked and in its one shape:
    ``{displacement_ang, steps, charges_e}`` -- the steps the force-constant
    run's own (the reference, then each free atom along x, y, z, side -1
    before +1, SIESTA's order, `siesta_grammar.FC_STEP_PATTERN`), every
    partition one
    of :data:`PARTITIONS` with one finite charge per atom at every step.
    Refused by name otherwise (`engines/vibration.md` § 6.7)."""
    where = "SpectraResults.partial_charges"
    check_keys(where, block, PARTIAL_CHARGES_KEYS, needed=())
    delta = float(block.get("displacement_ang", "nan"))
    if not (math.isfinite(delta) and delta > 0):
        raise ValueError(f"{where}: displacement_ang must be a positive "
                         f"number of Angstrom, not {block.get('displacement_ang')!r}")
    steps = [dict(s) for s in block.get("steps") or ()]
    free = sorted(int(i) for i in free_atom_idxs)
    expected = fc_step_count(len(free))
    if len(steps) != expected:
        raise ValueError(f"{where}: {len(steps)} steps; the force-constant "
                         f"run over {len(free)} free atom(s) takes "
                         f"{expected}")
    for s in steps:
        check_keys(f"{where}.steps", s, CHARGE_STEP_KEYS, needed=())
    if (steps[0].get("atom"), steps[0].get("axis"),
            steps[0].get("side")) != (None, None, 0):
        raise ValueError(f"{where}: step 0 is the reference geometry "
                         f"{{atom: null, axis: null, side: 0}}, not {steps[0]}")
    moved = []
    per_atom = len(FC_STEP_PATTERN)
    for k in range(len(free)):
        first = 1 + per_atom * k
        group = steps[first: first + per_atom]
        atoms = {s.get("atom") for s in group}
        pattern = tuple((s.get("axis"), s.get("side")) for s in group)
        if len(atoms) != 1 or pattern != FC_STEP_PATTERN:
            raise ValueError(f"{where}: steps {first}..{first + per_atom - 1} "
                             f"are not one atom along x, y, z, side -1 "
                             f"before +1: {group}")
        moved.append(int(group[0]["atom"]))
    if sorted(moved) != free:
        raise ValueError(f"{where}: the steps move atoms {sorted(moved)}; "
                         f"the free atoms are {free}")
    charges = dict(block.get("charges_e") or {})
    if not charges or any(p not in PARTITIONS for p in charges):
        raise ValueError(f"{where}: charges_e names {sorted(charges)}; "
                         f"a partition is one of {list(PARTITIONS)}")
    out = {}
    for name, rows in charges.items():
        arr = np.asarray(rows, dtype=float)
        if arr.shape != (len(steps), int(n_atoms)) or not np.all(
                np.isfinite(arr)):
            raise ValueError(f"{where}: charges_e.{name} must be one finite "
                             f"charge per atom ({n_atoms}) at every step "
                             f"({len(steps)}); it is {arr.shape}")
        out[name] = arr.tolist()
    return {"displacement_ang": delta,
            "steps": [{"atom": (None if s["atom"] is None else int(s["atom"])),
                       "axis": s["axis"], "side": int(s["side"])}
                      for s in steps],
            "charges_e": {p: out[p] for p in PARTITIONS if p in out}}


def checked_optical(block: Mapping[str, Any]) -> Dict[str, Any]:
    """A result's ``optical`` citation (v13), checked and in its one shape
    (`engines/vibration.md` § 6.2): ``data_file`` the optical calls' data
    file's name, ``data_sha256`` its bytes' (``None`` before it exists),
    ``direction`` the light's three components, ``displacement_ang`` ``δ``,
    ``selection`` as asked and as applied, ``reference`` frame 0's state,
    and ``modes`` -- one row per mode the selection applied, ``{mode,
    frequency_cm1, dq_amu12_ang, plus, minus}``, each side's state.  A key
    not here, or one missing -- in the block or a mode row -- a direction
    not three numbers, no ``δ`` or a state not one of
    :data:`OPTICAL_STATES` is refused by name."""
    where = "SpectraResults.optical"
    check_keys(where, block, sorted(OPTICAL_KEYS))
    for m in block.get("modes") or ():
        check_keys(f"{where}.modes", m, sorted(OPTICAL_MODE_KEYS))
    if block.get("displacement_ang") is None:
        raise ValueError(f"{where} states no displacement_ang -- the "
                         f"force constants' step the frames were moved by")
    direction = [float(x) for x in block.get("direction") or ()]
    if len(direction) != 3:
        raise ValueError(f"{where}.direction is three components, not "
                         f"{block.get('direction')!r}")
    states = [block.get("reference")] + [
        m.get(side) for m in block.get("modes") or ()
        for side in ("plus", "minus")]
    bad = sorted({str(s) for s in states if s not in OPTICAL_STATES})
    if bad:
        raise ValueError(f"{where}: frame state(s) {bad}; a frame is "
                         f"{', '.join(OPTICAL_STATES)}")
    return {"data_file": str(block["data_file"]),
            "data_sha256": (None if block.get("data_sha256") is None
                            else str(block["data_sha256"])),
            "direction": direction,
            "displacement_ang": float(block["displacement_ang"]),
            "selection": (None if block.get("selection") is None
                          else dict(block["selection"])),
            "reference": str(block["reference"]),
            "modes": [{"mode": int(m["mode"]),
                       "frequency_cm1": float(m["frequency_cm1"]),
                       "dq_amu12_ang": float(m["dq_amu12_ang"]),
                       "plus": str(m["plus"]), "minus": str(m["minus"])}
                      for m in block.get("modes") or ()]}


def _optical_read(d: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A file's ``optical`` as it states it: ``None`` where it states none;
    a stated block for :func:`checked_optical` to judge."""
    block = d["optical"]
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"SpectraResults.optical is a block, not {block!r}")
    return dict(block)


def _partial_charges_read(d: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A file's ``partial_charges`` as it states it: ``None`` where it states
    none; a stated block whatever it holds, for :func:`checked_partial_charges`
    to judge -- never read as absent (`engines/vibration.md` § 6.2)."""
    block = d["partial_charges"]
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"SpectraResults.partial_charges is a block "
                         f"{{displacement_ang, steps, charges_e}}, not "
                         f"{block!r}")
    return dict(block)


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
            # A DISPLACED WINDOW SHORTER THAN THE EQUILIBRIUM'S is padded with
            # NaN by the script: written null (`engines/vibration.md` § 6.4).
            "mo_energies_eq_eh":    finite_or_none(self.mo_energies_eq_eh),
            "mo_energies_minus_eh": finite_or_none(self.mo_energies_minus_eh),
            "mo_energies_plus_eh":  finite_or_none(self.mo_energies_plus_eh),
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
        check_keys("ModeElectronicStructure", d, ES_KEYS, needed=())
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

    # THE DIPOLE CHANGE ALONG THE MODE, whole (v12): dmu/dQ, x, y, z in
    # D/(A amu^1/2), as the engine projected it -- the vector
    # ``ir_intensity_km_mol`` is the orientation average of, and every light
    # direction's strength is computed from (engines/vibration.md § 4.6).
    # ``None`` where the engine kept none (SIESTA, whose infrared is its
    # charges'; PySCF where no infrared was asked).
    dipole_derivative_d_per_ang_amu12: Optional[Tuple[float, float, float]] = None

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
        if self.dipole_derivative_d_per_ang_amu12 is not None:
            v = np.asarray(self.dipole_derivative_d_per_ang_amu12, dtype=float)
            if v.shape != (3,) or not np.all(np.isfinite(v)):
                raise ValueError(
                    f"ModeData.dipole_derivative_d_per_ang_amu12 is three "
                    f"finite numbers, x, y, z; got "
                    f"{self.dipole_derivative_d_per_ang_amu12!r}")
            self.dipole_derivative_d_per_ang_amu12 = tuple(float(x) for x in v)

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
            "dipole_derivative_d_per_ang_amu12": (
                None if self.dipole_derivative_d_per_ang_amu12 is None
                else list(self.dipole_derivative_d_per_ang_amu12)),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModeData":
        check_keys("ModeData", d, MODE_KEYS, needed=())
        es = d.get("electronic_structure")
        return cls(
            index_1based          = int(d["index_1based"]),
            frequency_cm1         = float(d["frequency_cm1"]),
            raman_activity_a4_amu = (None if d.get("raman_activity_a4_amu") is None
                                     else float(d["raman_activity_a4_amu"])),
            ir_intensity_km_mol   = (None if d.get("ir_intensity_km_mol") is None
                                     else float(d["ir_intensity_km_mol"])),
            eigenvector_canonical = np.asarray(d["eigenvector_canonical"],
                                               dtype=float),
            eigenvector_display   = np.asarray(d["eigenvector_display"],
                                               dtype=float),
            has_imag              = bool(d.get("has_imag", False)),
            electronic_structure  = (None if es is None
                                     else ModeElectronicStructure.from_dict(es)),
            dipole_derivative_d_per_ang_amu12 = d[
                "dipole_derivative_d_per_ang_amu12"],
        )


# --------------------------------------------------------------------- #
#  The modes' temperatures a result states                              #
# --------------------------------------------------------------------- #


def _stated_mode_temperatures(d: Mapping[str, Any]) -> Optional[List[float]]:
    """The modes' temperatures a result file states (v15, `derived.
    stated_temperatures`, as `normal_modes.mode_temperatures` takes them)
    -- ``None`` where it states none."""
    stated = stated_temperatures(d.get(THERMAL_KEY))
    return None if stated is None else mode_temperatures(None, stated)


# --------------------------------------------------------------------- #
#  Complete results from a Spectra run                                  #
# --------------------------------------------------------------------- #


def equilibrium_block(scf_energy_eh, mo_energies_eh, homo_idx,
                      masses_amu=None) -> Dict[str, Any]:
    """The result's equilibrium block (`engines/vibration.md` § 6.2): its
    energy subgroup, which travels whole or not at all, and the masses the
    modes are weighted by, beside the structure -- stated where they are
    known."""
    return {"scf_energy_eh": scf_energy_eh, "mo_energies_eh": mo_energies_eh,
            "homo_idx": homo_idx,
            **({"masses_amu": masses_amu} if masses_amu is not None else {})}


def relaxation_block(*, enabled, already_relaxed, n_steps=0, record=None,
                     max_force_eh_bohr=None, max_force_all_atoms_eh_bohr=None,
                     converged=None, warning=None) -> Dict[str, Any]:
    """The result's relaxation block (`engines/vibration.md` § 6.2) -- ONE
    shape, both engines' writers': every key stated from the first write,
    ``None`` where not yet known, ``record`` the record of the call that
    relaxed, whole, where one did -- SIESTA's `relax` call of the run, read
    from its index (§ 5.11 § 3) -- else ``None``: PySCF relaxes inside its
    one deck."""
    return {"enabled": bool(enabled), "already_relaxed": bool(already_relaxed),
            "n_steps": int(n_steps), "record": record,
            "max_force_eh_bohr": max_force_eh_bohr,
            "max_force_all_atoms_eh_bohr": max_force_all_atoms_eh_bohr,
            "converged": converged, "warning": warning}


def removed_motions_of(patterns) -> Dict[str, Any]:
    """What the harmonic analysis removed before diagonalising, as the result
    states it (`engines/vibration.md` § 6.2): the count and the Cartesian
    patterns over the free atoms -- both engines' record."""
    rows = [np.asarray(p, dtype=float).tolist() for p in patterns]
    return {"count": len(rows), "patterns": rows}


@dataclass(eq=False)
class SpectraResults:
    """Engine-agnostic result of a Spectra run.

    Progress is carried by the per-phase ``phase_*`` flags.  While the
    run is mid-way some modes may have ``electronic_structure = None``
    not because the user de-selected them but because their displaced
    SCFs haven't run yet.  The ``selected_mode_idxs_1based`` field tells
    the UI which modes WILL get ES data so it can show progress
    ("3 of 10 modes done").

    ``structure`` states the geometry the result was taken from, whole
    (`engines/vibration.md` § 6.10); ``engine`` +
    ``engine_version`` + ``molbuilder_version`` give the software
    stack used; ``timestamp`` is ISO-8601 UTC.
    """

    # Provenance
    schema_version:       int
    engine:               str
    engine_version:       str
    molbuilder_version:   str
    timestamp:            str                    # ISO-8601 UTC

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

    #: How dmu/dR was obtained, for the run as a whole:
    #:   "analytic"          -- pyscf.prop.infrared, off the Hessian's own
    #:                          CPHF solution, no extra SCFs;
    #:   "finite-difference" -- the 6N-SCF dipole sweep, either because
    #:                          Raman's displacement loop was running
    #:                          anyway (so the dipole read was free) or
    #:                          because the analytic module is absent;
    #:   "charges"           -- SIESTA: the force-constant run's partial
    #:                          charges, stored; the infrared derived from
    #:                          them on request (`spectra.infrared`);
    #:   "none"              -- no route has produced it: not requested, or
    #:                          on PySCF, before its Hessian.
    ir_route:                  str = ROUTE_NONE
    #: The finite-difference step actually used for dmu/dR, in Angstrom,
    #: when (and only when) that route ran.  Carried because a Methods
    #: section quoting a finite-difference derivative has to state its
    #: step to be reproducible; ``None`` on the analytic route, which
    #: has no step to state.
    ir_fd_step_ang:            Optional[float] = None
    #: How dalpha/dR was obtained, for the run as a whole (engines/vibration.md § 4.6):
    #:   "finite-difference" -- PySCF: central differences of the analytic
    #:                          CPHF polarizability over the free Cartesians;
    #:   "optical"           -- SIESTA: its optical calls' eps2 at +dQ and
    #:                          -dQ along each picked mode, the strength
    #:                          derived on request at a laser
    #:                          (`spectra.raman`);
    #:   "none"              -- Raman was not requested, the engine computes
    #:                          no intensities, or on PySCF, before its sweep
    #:                          has ended.
    raman_route:               str = ROUTE_NONE
    #: The step of that difference, in Angstrom, when the route ran.
    raman_fd_step_ang:         Optional[float] = None
    # EACH PHASE'S STATE -- one of :data:`_VALID_PHASE_STATES`: empty,
    # running, complete, or not requested, which is terminal
    # (`engines/vibration.md` § 4.9); checked at ``__post_init__``.
    phase_frequencies:         str = PHASE_EMPTY
    phase_raman:               str = PHASE_EMPTY
    #: The infrared intensities' own flag (`engines/vibration.md` § 4.9):
    #: closed with the Raman sweep that carries the dipole, or by the
    #: dipole sweep alone; on SIESTA, complete once the run's charges are
    #: stored.
    phase_ir:                  str = PHASE_EMPTY
    phase_es:                  str = PHASE_EMPTY
    #: v5: the in-deck relaxation is a tracked step (user, 2026-08-20 --
    #: the viewer tracks ALL the steps; a silent gap while geomeTRIC works
    #: would betray that).  `not requested` from the first write under
    #: already_relaxed (`engines/vibration.md` § 4.9), the gradient-check
    #: number in `relaxation` beside it.
    phase_relaxation:          str = PHASE_EMPTY
    #: v5: relaxation progress/result, the one shape both writers write
    #: (:func:`relaxation_block`).  Written live so the chip can show
    #: "step 14, max force 0.0042" ticking down.
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
    #: Empty before the harmonic analysis has run (§ 6.5).
    removed_motions:           Dict[str, Any] = field(default_factory=dict)
    #: v6: what the Hessian covered -- "free" (second derivatives for the
    #: free atoms only; atoms are held) or "all" (every atom; nothing held)
    #: -- and how many atoms that was.  A reader comparing two runs' costs
    #: or Methods paragraphs needs it stated, not inferred from the frozen
    #: list.  "" before the Hessian is computed (§ 6.5).
    hessian_scope:             str = ""
    n_atoms_in_hessian:        Optional[int] = None
    #: v6: whether the Hessian itself was density fitted.  The SCF may be
    #: while the Hessian is not (the free-atom route), and a Methods
    #: paragraph has to say which.  None before the Hessian, and on SIESTA.
    hessian_density_fit:       Optional[bool] = None

    # Equilibrium geometry -- element symbols + Cartesian positions in Å:
    # the structure's, filled from it where the result states one
    # (``__post_init__``); handed in by a writer that builds the result in
    # memory (`spectra.vibrational_analysis`), checked against the
    # structure where both are stated.  ``None`` before any geometry
    # exists; never a key of the file (§ 6.10.2).
    equilibrium_elements:      Optional[List[str]]  = None
    equilibrium_positions_ang: Optional[np.ndarray] = None
    #: v7: every atom's mass in amu, in the user order -- the masses the
    #: analysis weighted by, as the deck stated them (`engines/vibration.md`
    #: § 6.2, I3).  Travels with the geometry: stated wherever the geometry
    #: is, and refused without it.
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
    # 20-cores oversubscription leaves a clear trail.  PySCF's: a SIESTA
    # result states none, and the UI no-ops on missing keys.
    runtime_info:              Dict[str, Any] = field(default_factory=dict)

    #: v9: THE STRUCTURE, WHOLE -- `Structure.to_dict(view=False)` of the
    #: equilibrium structure (engines/vibration.md § 6.10): the geometry the Hessian is
    #: taken at, its box, its labels, the held atoms as `frozen_atoms`.
    #: ``None`` only while a PySCF run has no geometry yet.
    structure:                 Optional[Dict[str, Any]] = None

    #: v10: THE ATOM INDEX MAP (model/overview.md § 2.2b) -- the result is
    #: listed by user index, and each atom's engine index is its stages'
    #: deck order (held atoms first for SIESTA's; PySCF's the user order).
    #: Every writer states it, from the run's first write; a result without
    #: one is refused (:meth:`atom_index_map_stated`).
    atom_index_map:            Optional[Dict[str, Any]] = None

    #: v11: THE FORCE-CONSTANT RUN'S CHARGES (engines/vibration.md § 5.10
    #: ②) -- :func:`checked_partial_charges`'s shape; ``None`` on PySCF and
    #: from a deck that asked for none.
    partial_charges:           Optional[Dict[str, Any]] = None

    #: v13: THE OPTICAL CALLS' CITATION (engines/vibration.md § 5.11 § 8,
    #: § 6.2) -- :func:`checked_optical`'s shape; ``None`` where no optical
    #: call was planned, and on PySCF.
    optical:                   Optional[Dict[str, Any]] = None

    #: v15: THE MODES' TEMPERATURES the file states its ``modes_thermal`` at
    #: (engines/vibration.md § 6.6) -- the file's own list, kept so a file
    #: read and written again states the same block; ``None`` on a result
    #: built in memory, whose list is today's at the record's temperature
    #: (`normal_modes.mode_temperatures`), and on a file before its modes
    #: exist, which states none (§ 6.5).
    mode_temperatures_k:       Optional[List[float]] = None

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
        # Free + fixed atom partition -- AND ITS ORDER: the free list is the
        # order of every mode's eigenvector rows, ascending (§ 6.10.2,
        # "stated and checked"); a file listing them otherwise would animate
        # the wrong atoms with no word.
        free_list = [int(i) for i in self.free_atom_idxs]
        if free_list != sorted(free_list):
            raise ValueError(
                "SpectraResults: free_atom_idxs must be ascending -- the "
                "order of every mode's eigenvector rows")
        free_set   = set(free_list)
        frozen_set = set(int(i) for i in self.frozen_atom_idxs)
        # THE SCOPE AGREES WITH THE HELD SET: `all` only where nothing is
        # held, `free` only where something is (§ 6.10.2).
        if self.hessian_scope == "all" and frozen_set:
            raise ValueError(
                "SpectraResults: hessian_scope 'all' with atoms held "
                f"({sorted(frozen_set)[:20]}) -- a held atom is outside the "
                "Hessian")
        if self.hessian_scope == "free" and not frozen_set:
            raise ValueError(
                "SpectraResults: hessian_scope 'free' with no atom held -- "
                "the Hessian is over every atom")
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
                f"partition range({n})"
                f"; "
                f"missing={missing}{more} out-of-range/extra={extra}"
            )
        # Phase status validation.
        for name, val in (("phase_frequencies", self.phase_frequencies),
                          ("phase_raman",       self.phase_raman),
                          ("phase_ir",          self.phase_ir),
                          ("phase_es",          self.phase_es),
                          ("phase_relaxation",  self.phase_relaxation)):
            if val not in _VALID_PHASE_STATES:
                raise ValueError(
                    f"SpectraResults.{name}={val!r} is not a valid "
                    f"phase status; expected one of {_VALID_PHASE_STATES}"
                )
        for name, val, routes in (("ir_route", self.ir_route, IR_ROUTES),
                                  ("raman_route", self.raman_route,
                                   RAMAN_ROUTES)):
            if val not in routes:
                raise ValueError(
                    f"SpectraResults.{name}={val!r} is not a route; "
                    f"expected one of {routes}")
        # THE STRUCTURE, WHOLE (§ 6.10): stated, it IS the geometry -- the
        # file's own count, held list and coordinates are checked against it
        # (§ 6.6).  A result built in memory from its geometry alone states
        # the structure that geometry makes (:meth:`structure_stated`).
        if self.structure is not None:
            s = _Structure.from_dict(self.structure)
            if s.n_atoms != int(self.n_atoms_total):
                raise ValueError(
                    f"SpectraResults: the structure holds {s.n_atoms} atoms "
                    f"and n_atoms_total says {self.n_atoms_total}")
            held = sorted(s.frozen_atoms)
            if held != sorted(frozen_set):
                raise ValueError(
                    f"SpectraResults: the structure holds {held} as "
                    f"frozen_atoms and the file states {sorted(frozen_set)}")
            pos = np.asarray(s.positions, dtype=float)
            if self.equilibrium_elements is None:
                self.equilibrium_elements = list(s.elements)
                self.equilibrium_positions_ang = pos
            elif (list(self.equilibrium_elements) != list(s.elements)
                  or not np.allclose(self.equilibrium_positions_ang, pos,
                                     rtol=0.0, atol=1e-9)):
                raise ValueError(
                    "SpectraResults: the structure and the equilibrium "
                    "geometry disagree")
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
        # THE MASSES TRAVEL WITH THE GEOMETRY (I3): one positive finite
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
        elif self.equilibrium_positions_ang is not None:
            raise ValueError(
                "SpectraResults: the result states its geometry without its "
                "masses -- equilibrium.masses_amu travels with it")

        # THE HESSIAN'S SCOPE, stated and checked (§ 6.6): `free` holds the
        # free atoms' count, `all` every atom's.
        if self.n_atoms_in_hessian is not None and self.hessian_scope:
            want = {"free": len(free_set), "all": n}.get(self.hessian_scope)
            if want is not None and int(self.n_atoms_in_hessian) != want:
                raise ValueError(
                    f"SpectraResults: hessian_scope {self.hessian_scope!r} "
                    f"holds {want} atoms, and n_atoms_in_hessian says "
                    f"{self.n_atoms_in_hessian}")

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

        # THE ATOM INDEX MAP, checked by its one reader over these atoms
        # (`model/overview.md` § 2.2b) -- after the atom count is.
        self.atom_index_map_stated()
        # THE CHARGES, checked against the free atoms they were taken over.
        if self.partial_charges is not None:
            self.partial_charges = checked_partial_charges(
                self.partial_charges, int(self.n_atoms_total),
                self.free_atom_idxs)
        # THE OPTICAL CITATION, checked in its one shape.
        if self.optical is not None:
            self.optical = checked_optical(self.optical)

    def _modes_with_derived(self) -> List[Dict[str, Any]]:
        """Mode dicts, each with its DERIVED values stated beside the rest --
        the activity classes (a whole-run decision: a mode is active when
        it clears a fraction of the STRONGEST band in its own channel, and
        one mode does not know the others), the zero-point amplitude and
        displacement, and the mode's mechanics -- through the one derivation
        every writer uses (`spectra.derived`, `engines/vibration.md` § 6.6,
        § 6.11 § 3), by the masses the modes were weighted by."""
        return with_derived([m.to_dict() for m in self.modes],
                            self._free_masses())

    def _free_masses(self) -> Optional[List[float]]:
        """The free atoms' masses the modes were weighted by, in the free
        list's order (:meth:`atoms_weighted`); ``None`` before any geometry
        exists."""
        weighted = self.atoms_weighted(self.free_atom_idxs)
        return None if weighted is None else weighted[1].tolist()

    def _mode_temperatures(self) -> List[float]:
        """The modes' temperatures (`normal_modes.mode_temperatures`): the
        file's own where it states them, else today's list at the record's
        temperature."""
        return mode_temperatures((self.thermo or {}).get("temperature_K"),
                                 self.mode_temperatures_k)

    def structure_stated(self) -> Optional[Dict[str, Any]]:
        """THE STRUCTURE THE RESULT STATES (§ 6.10): the one it was read or
        built with, or -- one built in memory from its geometry alone
        (`spectra.vibrational_analysis` handed no structure) -- the one its
        equilibrium geometry and held list make (`Structure.to_dict`: no
        box, no other label -- stated absent, never invented); ``None``
        before any geometry exists (§ 4.9)."""
        if self.structure is not None:
            return dict(self.structure)
        if self.equilibrium_elements is None:
            return None
        s = _Structure(elements=list(self.equilibrium_elements),
                       positions=np.asarray(self.equilibrium_positions_ang,
                                            dtype=float),
                       frozen_atoms=sorted(int(i)
                                           for i in self.frozen_atom_idxs))
        # A RESULT FILE'S STRUCTURE, as both writers state it: the view's
        # fields left out (`Structure.to_dict(view=False)`).
        return s.to_dict(view=False)

    def atoms_weighted(self, atoms: Optional[Sequence[int]] = None):
        """EACH ATOM AS THE MODES WEIGHT IT -- its element, its species
        label's name (`Structure.chemical_symbols`: prep has made every
        label's name an element, `engines/vibration.md` § 6.11 § 3), and its
        mass, the result's own (``equilibrium.masses_amu``, which travels
        with the geometry): ``(elements, masses_amu)`` for the ``atoms``
        asked, in the order asked -- every atom in the user's order where
        none are -- or ``None`` before any geometry exists.  The one reading
        the load door's element shares (its free atoms alone),
        :meth:`vibration` (every atom), the free atoms' infrared
        (`spectra.infrared`) and a displacement sweep's mode matching ask
        (§ 6.6)."""
        stated = self.structure_stated()
        if stated is None:
            return None
        structure = _Structure.from_dict(stated)
        symbols = structure.chemical_symbols()
        asked = (range(len(symbols)) if atoms is None
                 else [int(i) for i in atoms])
        masses = np.asarray(self.equilibrium_masses_amu, dtype=float)
        return (tuple(symbols[i] for i in asked),
                np.array([masses[i] for i in asked], dtype=float))

    def vibration(self, read_path: str = "", *,
                  result_sha256: Optional[str] = None):
        """THE RESULT'S MODES AS A PERSON COMPUTES WITH THEM
        (`vibration.Vibration`, `engines/vibration.md` § 6.11 § 6): each mode
        over every atom in the user's order, held atoms still, by the masses
        the result states -- each atom by its element (its species
        label's name), each atom's species label and the structure's region
        labels as written, at the geometry the result states, the temperature
        its thermochemistry states and the modes' own temperatures its
        ``modes_thermal`` is at (:meth:`_mode_temperatures`) -- each shared
        field the `Vibration`'s, its modes bound to it, with the structure the
        result states; ``read_path`` the file it was read from and
        ``result_sha256`` the result JSON's (`sidecars.spectra.
        result_sha256_of`, the read door's).  Refused by name before any
        geometry exists."""
        from ..vibration import Mode, Vibration
        weighted = self.atoms_weighted()
        if weighted is None:
            raise ValueError("the result states no geometry yet: its modes "
                             "come with its force constants")
        elements, masses = weighted
        structure = _Structure.from_dict(self.structure_stated())
        positions = np.asarray(structure.positions, dtype=float)
        free = tuple(int(i) for i in self.free_atom_idxs)
        t = (self.thermo or {}).get("temperature_K")
        temperature = None if t is None else float(t)
        labels = tuple(str(e) for e in structure.elements)
        regions = {str(k): tuple(int(i) for i in v)
                   for k, v in (structure.regions or {}).items()}
        temperatures = tuple(self._mode_temperatures())
        modes = tuple(
            Mode(index_1based=int(m.index_1based),
                 frequency_cm1=float(m.frequency_cm1),
                 has_imag=bool(m.has_imag),
                 eigenvector_canonical_per_amu12=np.asarray(
                     m.eigenvector_canonical, dtype=float))
            for m in self.modes)
        return Vibration(modes=modes, free_atom_idxs=free, masses_amu=masses,
                         element_symbols=elements, positions_ang=positions,
                         record_temperature_k=temperature,
                         species_labels=labels, regions=regions,
                         mode_temperatures_k=temperatures,
                         read_path=str(read_path), structure=structure,
                         result_sha256=result_sha256)

    def atom_index_map_stated(self):
        """THE ATOM INDEX MAP the result states, read by the map's one reader
        over the result's atoms (`model/overview.md` § 2.2b) -- refused by
        that reader where the result states none, or not one of its
        atoms."""
        try:
            from ..atom_permutation import AtomIndexMap
        except ImportError:                 # beside a job, in a bundle
            from atom_permutation import AtomIndexMap  # type: ignore
        return AtomIndexMap.from_dict(self.atom_index_map,
                                      n_atoms=int(self.n_atoms_total),
                                      source="the result's atom_index_map")

    def to_dict(self) -> Dict[str, Any]:
        structure = self.structure_stated()
        return {
            "schema_version":       int(self.schema_version),
            "engine":               str(self.engine),
            "engine_version":       str(self.engine_version),
            "molbuilder_version":   str(self.molbuilder_version),
            "timestamp":            str(self.timestamp),

            "n_atoms_total":        int(self.n_atoms_total),
            "free_atom_idxs":       [int(i) for i in self.free_atom_idxs],
            # THE HELD ATOMS, stated once: the structure's `frozen_atoms`
            # label where the structure is stated (v9); as their own list
            # in a file written before any geometry exists (§ 6.10.2) --
            # never derived from the count.
            **({"structure": structure} if structure is not None
               else {"frozen_atom_idxs":
                     [int(i) for i in self.frozen_atom_idxs]}),

            # THE MASSES THE MODES ARE WEIGHTED BY stay the analysis's own,
            # beside the structure (§ 6.10.1).
            "equilibrium": equilibrium_block(
                (None if self.equilibrium_scf_eh is None
                 else float(self.equilibrium_scf_eh)),
                (None if self.equilibrium_mo_energies_eh is None
                 else self.equilibrium_mo_energies_eh.tolist()),
                (None if self.equilibrium_homo_idx is None
                 else int(self.equilibrium_homo_idx)),
                (None if self.equilibrium_masses_amu is None
                 else self.equilibrium_masses_amu.tolist())),

            # THE DERIVED VALUES ride WITH the modes, through the one
            # derivation (`spectra.derived`): the activity classes -- a
            # whole-run judgement, never re-derived as an epsilon in the
            # viewer -- the zero-point amplitude and displacement, and each
            # mode's mechanics: each free atom's share of the motion, the
            # largest zero-point move, the reduced mass and force constant.
            "modes":                self._modes_with_derived(),
            # EACH MODE'S SPREAD AND MEAN QUANTA at the modes' own
            # temperatures (v15, § 6.11 § 3) -- null before the modes exist,
            # stated so from the run's first write (§ 6.5).
            THERMAL_KEY:            thermal_block(
                [m.to_dict() for m in self.modes], self._mode_temperatures()),
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
            # THE ATOM INDEX MAP, always stated (§ 2.2b, F1): the result's
            # own.
            MAP_KEY:                self.atom_index_map_stated().to_dict(),
            "partial_charges":      self.partial_charges,
            "optical":              self.optical,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectraResults":
        # THE ONE VERSION RULE, the reader's too (:func:`readable_version`):
        # a version missing or unreadable raises, never reconstituted at
        # whatever version the payload claims.
        readable_version(d.get("schema_version"))
        check_keys("SpectraResults", d, RESULT_KEYS, needed=())
        eq = d["equilibrium"]
        check_keys("SpectraResults.equilibrium", eq, EQUILIBRIUM_KEYS,
                   needed=())
        # THE HELD ATOMS: the file states them in its structure's
        # `frozen_atoms` label, or -- written before any geometry exists
        # (§ 6.10.2) -- as `frozen_atom_idxs`.  Stated nowhere, the file is
        # refused: the held set is never derived from the count.
        structure = d.get("structure")
        if "frozen_atom_idxs" in d:
            frozen = [int(i) for i in d["frozen_atom_idxs"]]
        elif isinstance(structure, dict):
            # THROUGH THE STRUCTURE'S OWN DOOR (`model/structure.md` § 2.1:
            # outside the class nobody picks a structure dict apart).
            frozen = sorted(int(i) for i in
                            (_Structure.from_dict(structure).frozen_atoms
                             or ()))
        else:
            raise ValueError(
                "SpectraResults: neither a structure nor frozen_atom_idxs "
                "states which atoms were held")
        # THE DERIVED VALUES THE FILE STATES ARE ITS INPUTS' (§ 6.6): each
        # computed again by the one derivation, by the masses the modes were
        # weighted by -- the file's own, refused by name where it has modes
        # and states none, as its writer refuses (`derived.free_masses_of`)
        # -- and a file stating another value is refused by name.
        wrong = disagreements(d, free_masses_of(d) if d["modes"] else None)
        if wrong:
            raise ValueError(
                "SpectraResults: the derived values this file states are "
                "not its own inputs' -- "
                + "; ".join(wrong[:5])
                + (f"; and {len(wrong) - 5} more" if len(wrong) > 5
                   else ""))
        return cls(
            schema_version       = int(d["schema_version"]),
            engine               = str(d["engine"]),
            engine_version       = str(d["engine_version"]),
            molbuilder_version   = str(d["molbuilder_version"]),
            timestamp            = str(d["timestamp"]),
            n_atoms_total        = int(d["n_atoms_total"]),
            free_atom_idxs       = [int(i) for i in d["free_atom_idxs"]],
            frozen_atom_idxs      = frozen,
            structure            = (dict(structure) if isinstance(structure, dict)
                                    else None),

            equilibrium_scf_eh         = (None if eq.get("scf_energy_eh") is None
                                          else float(eq["scf_energy_eh"])),
            equilibrium_mo_energies_eh = (None if eq.get("mo_energies_eh") is None
                                          else np.asarray(eq["mo_energies_eh"], dtype=float)),
            equilibrium_homo_idx       = (None if eq.get("homo_idx") is None
                                          else int(eq["homo_idx"])),

            # THE MASSES, stated with the geometry; none before it exists.
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
            relaxation           = dict(d["relaxation"]),
            thermo               = dict(d["thermo"]),
            removed_motions      = dict(d["removed_motions"]),
            hessian_scope        = str(d.get("hessian_scope", "")),
            n_atoms_in_hessian   = (None if d.get("n_atoms_in_hessian") is None
                                    else int(d["n_atoms_in_hessian"])),
            hessian_density_fit  = (None if d.get("hessian_density_fit") is None
                                    else bool(d["hessian_density_fit"])),
            ir_route             = str(d["ir_route"]),
            ir_fd_step_ang       = (None if d.get("ir_fd_step_ang") is None
                                    else float(d["ir_fd_step_ang"])),
            raman_route          = str(d["raman_route"]),
            raman_fd_step_ang    = (None if d.get("raman_fd_step_ang") is None
                                    else float(d["raman_fd_step_ang"])),
            phase_raman          = str(d.get("phase_raman",       PHASE_EMPTY)),
            phase_ir             = str(d["phase_ir"]),
            phase_es             = str(d.get("phase_es",          PHASE_EMPTY)),
            engine_metadata      = dict(d["engine_metadata"]),
            runtime_info         = dict(d.get("runtime_info", {})),
            atom_index_map       = dict(d[MAP_KEY]),
            partial_charges      = _partial_charges_read(d),
            optical              = _optical_read(d),
            mode_temperatures_k  = _stated_mode_temperatures(d),
        )


__all__ = [
    "SCHEMA_VERSION", "OPTICAL_STATES", "checked_optical",
    "ModeElectronicStructure",
    "ModeData",
    "SpectraResults",
]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.atom_permutation",
                "molbuilder.parse.engines.siesta_grammar",
                "molbuilder.persist",
                "molbuilder.parse.engines._run_ending",
                "molbuilder.spectra.derived",
                "molbuilder.spectra.normal_modes",
                "molbuilder.spectra.result_keys",
                "molbuilder.structure")
