"""THE INFRARED OF A SET OF ATOMS, from a SIESTA force-constant run's charges
(`science/vibrational-optical-response.md` § 2; `engines/vibration.md`
§ 5.10 ②, § 6.6).

MODULE  spectra.infrared (L2; numpy, the result and the structure)
ROLE    :func:`infrared_view` -- the ONE derivation of what a person asks of
        the charges a result states (``partial_charges``): for a set of
        atoms, light polarized along a direction, a partition, per mode the
        set's dipole change, its infrared strength along that direction and
        the charge the mode moves into or out of the set;
        :func:`labelled_atoms` -- a label of the result's structure as its
        atoms, the one way both doors name the set, and :func:`labels_of`
        the labels it may be named by; :func:`read_ask` -- the load door's
        reading of what a person asks (atoms, direction, partition), by the
        rules :func:`infrared_view` applies to a script's ask too;
        :func:`no_charges_reason` -- why a result states no charges, in
        its own facts
USED-BY the spectrum viewer's load door (`web/spectra.md` § 3.2) and a
        person's script (`molbuilder.postprocess.infrared_view`)

**Nothing chosen is stored.**  The result holds every atom's charge at every
force-constant step; the set, the direction and the partition are the
person's, asked here and derived on request -- never written into the file
(`engines/vibration.md` § 6.6).

**The direction is applied after the run.**  The field enters only through
``ê · ∂μ/∂Q``: the run gives the whole vector, and every direction is a dot
product of it (§ 2.3 of the method).
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..constants import DEBYE_E_ANGSTROM
from .activity import CHANNEL_PRESENT_FLOOR_IR_KM_MOL, ir_strength_km_mol
from ..structure import Structure
from .results import PARTITIONS, SpectraResults, mass_of

#: The directions a person names by an axis.
DIRECTIONS: Dict[str, Tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
#: The partition a view is taken by when the person names none -- Hirshfeld,
#: whose shares change smoothly as the atoms move (the method's § 1a).
DEFAULT_PARTITION = "hirshfeld"
#: The release that first keeps a force-constant run's charges in its result.
CHARGES_KEPT_SINCE = "2.36.0"

#: HOW A RESULT'S MOLECULE MEETS THE LIGHT, by whether the result holds atoms
#: (the method's § 2.3; the user, 2026-10-24: "if no fix atoms are listed,
#: that means gas phase, and you can keep conventional average, but when we
#: have fixed atoms that means we have rigidity and surface effects"):
#: held -- in one orientation, its strength along the light's direction;
#: free -- tumbling, the strength averaged over orientations, no direction.
ORIENTATION_HELD = "held"
ORIENTATION_FREE = "free"

#: WHERE A RESULT'S INFRARED COMES FROM (:func:`source_of`): SIESTA's
#: charges, for a set the person names; or the dipole change the engine
#: stored whole (PySCF, `engines/vibration.md` § 4.6), the whole system's.
SOURCE_CHARGES = "charges"
SOURCE_DIPOLE = "dipole"
#: The built-in set (`web/spectra.md` § 3.2): everything the vibration moved.
FREE_SET: Dict[str, str] = {"set": "free"}
FREE_SET_NAME = "the free atoms"
#: The set a stored dipole change is of.
WHOLE_SYSTEM_NAME = "the whole system"


class InfraredError(ValueError):
    """The view cannot be derived -- the message names what the result holds
    and what was asked, ready to surface verbatim."""


@dataclass(frozen=True)
class InfraredMode:
    """One mode's infrared, for the set and direction asked."""
    index_1based: int
    frequency_cm1: float
    has_imag: bool
    #: the set's dipole change along the mode, a vector, in e/amu^½
    #: (e·Å per amu^½·Å)
    dmu_dq_e_per_amu12: Tuple[float, float, float]
    #: ``A(ê)`` -- the integrated absorption for a field along ê, km/mol
    intensity_km_mol: float
    #: ``A(ê)`` over the strongest real mode's; ``None`` when the strongest
    #: is at or below the infrared channel's band floor
    #: (`activity.CHANNEL_PRESENT_FLOOR_IR_KM_MOL`) -- no band to rank against
    relative: Optional[float]
    #: the charge the mode moves into or out of the set, e per amu^½·Å --
    #: ``None`` for a stored dipole change, the whole system's, which
    #: exchanges charge with nothing
    charge_moved_e_per_amu12_ang: Optional[float]


@dataclass(frozen=True)
class InfraredView:
    """What :func:`infrared_view` answers: the set (its label and atoms by
    user index), how the molecule meets the light (:func:`orientation_of`)
    and, held, the direction, the partition, the step the derivatives were
    taken at, and every mode's infrared."""
    atoms_label: str
    atoms: Tuple[int, ...]
    #: :data:`SOURCE_CHARGES` or :data:`SOURCE_DIPOLE`
    source: str
    orientation: str
    #: the unit direction and its axis name -- ``None`` and ``""`` for a
    #: free molecule, whose strength is the orientation average
    direction: Optional[Tuple[float, float, float]]
    direction_name: str
    partition: str
    displacement_ang: float
    modes: Tuple[InfraredMode, ...]

    def to_wire(self) -> Dict[str, Any]:
        """As a page is answered: every field, each mode a dict."""
        return {"atoms_label": self.atoms_label, "atoms": list(self.atoms),
                "source": self.source, "orientation": self.orientation,
                "direction": (None if self.direction is None
                              else list(self.direction)),
                "direction_name": self.direction_name,
                "partition": self.partition,
                "displacement_ang": self.displacement_ang,
                "modes": [{"index_1based": m.index_1based,
                           "frequency_cm1": m.frequency_cm1,
                           "has_imag": m.has_imag,
                           "dmu_dq_e_per_amu12": list(m.dmu_dq_e_per_amu12),
                           "intensity_km_mol": m.intensity_km_mol,
                           "relative": m.relative,
                           "charge_moved_e_per_amu12_ang":
                               m.charge_moved_e_per_amu12_ang}
                          for m in self.modes]}


def _structure_of(result: SpectraResults) -> Structure:
    stated = result.structure_stated()
    if stated is None:
        raise InfraredError("the result states no geometry yet")
    return Structure.from_dict(stated)


def orientation_of(result: SpectraResults) -> str:
    """:data:`ORIENTATION_HELD` where the result holds atoms -- the molecule
    rigid in a junction or on a surface, meeting the light in one
    orientation -- else :data:`ORIENTATION_FREE`: nothing held, a free
    molecule, tumbling."""
    held = int(result.n_atoms_total) > len(result.free_atom_idxs)
    return ORIENTATION_HELD if held else ORIENTATION_FREE


def source_of(result: SpectraResults) -> Optional[str]:
    """Where ``result``'s infrared comes from: :data:`SOURCE_CHARGES` where
    it states the force-constant run's charges; :data:`SOURCE_DIPOLE` where
    the engine computed the infrared itself, from the whole system's dipole
    (PySCF); ``None`` where it has neither -- a SIESTA result that keeps no
    charges, a run that computed no infrared."""
    if result.partial_charges is not None:
        return SOURCE_CHARGES
    if str(result.engine).lower() != "siesta" and any(
            m.ir_intensity_km_mol is not None for m in result.modes):
        return SOURCE_DIPOLE
    return None


def sets_of(result: SpectraResults) -> List[Dict[str, Any]]:
    """The sets the infrared of a charges result may be asked of, each as
    the page shows it and as an ask names it: the built-in
    :data:`FREE_SET`, then every label of the structure that holds atoms."""
    return ([{"name": FREE_SET_NAME, "ask": dict(FREE_SET)}]
            + [{"name": label, "ask": label} for label in labels_of(result)])


def atoms_of(result: SpectraResults, atoms: Any) -> Tuple[List[int], str]:
    """The set an ask names, by user index, and its name: the built-in
    :data:`FREE_SET` -- the atoms the vibration moved -- or a label of the
    result's structure."""
    if atoms == FREE_SET:
        return sorted(int(k) for k in result.free_atom_idxs), FREE_SET_NAME
    if isinstance(atoms, str) and atoms:
        return labelled_atoms(result, atoms), atoms
    raise InfraredError(f"the set is a label of the result's structure or "
                        f"{FREE_SET}, not {atoms!r}")


def labels_of(result: SpectraResults) -> List[str]:
    """The labels of the result's structure that hold atoms -- what a set
    may be named by (`web/spectra.md` § 3.2)."""
    regions = _structure_of(result).regions or {}
    return sorted(k for k, v in regions.items() if v)


def labelled_atoms(result: SpectraResults, label: str) -> List[int]:
    """The atoms of ``label``, a label of the result's structure, by user
    index -- refused naming the labels it holds."""
    regions = _structure_of(result).regions or {}
    if label not in regions or not regions[label]:
        raise InfraredError(
            f"the result's structure holds no label {label!r}; its labels "
            f"are {labels_of(result) or 'none'}")
    return sorted(int(i) for i in regions[label])


def no_charges_reason(result: SpectraResults) -> str:
    """Why ``result`` states no charges, from its own facts -- its engine
    and the schema it was written at (`engines/vibration.md` § 5.10 ②):
    a PySCF result's infrared is its own; a SIESTA result written before
    :data:`CHARGES_KEPT_SINCE` keeps none, whatever its run printed; a
    later one whose force-constant run printed none."""
    if str(result.engine).lower() != "siesta":
        return (f"a {result.engine} result: its infrared is its own -- the "
                f"IR column, from its dipole derivatives")
    if int(result.schema_version) < 11:
        return (f"the result was written before molbuilder "
                f"{CHARGES_KEPT_SINCE}, which first keeps the force-constant "
                f"run's charges (a deck prepared before 2.35.2 printed none; "
                f"one prepared with 2.35.2 printed them, and its finish kept "
                f"none).  Prepare the force-constant stage with "
                f"{CHARGES_KEPT_SINCE} or later and run it again -- once "
                f"launched, from the state saved before its group's first "
                f"launch (execution/job-system.md, How the checkpoint "
                f"supports you)")
    return "the force-constant run printed no charges"


def _unit(direction: Union[str, Sequence[float]]) -> Tuple[Tuple[float, ...],
                                                           str]:
    if isinstance(direction, str):
        if direction not in DIRECTIONS:
            raise InfraredError(f"the direction is one of "
                                f"{sorted(DIRECTIONS)} or a vector, not "
                                f"{direction!r}")
        return DIRECTIONS[direction], direction
    numbers = (isinstance(direction, (list, tuple, np.ndarray))
               and len(direction) == 3
               and all(isinstance(x, Real) and not isinstance(x, bool)
                       for x in direction))
    v = np.asarray(direction, dtype=float) if numbers else np.zeros(0)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or not np.any(v):
        raise InfraredError(f"the direction is three finite numbers, not "
                            f"all zero -- not {direction!r}")
    # SCALED FIRST, so neither a huge nor a tiny vector over- or underflows
    # its norm.
    v = v / np.abs(v).max()
    v = v / np.linalg.norm(v)
    return tuple(float(x) for x in v), ""


def _partition_named(partition: Any) -> str:
    if not isinstance(partition, str) or partition not in PARTITIONS:
        raise InfraredError(f"the partition is one of {list(PARTITIONS)}, "
                            f"not {partition!r}")
    return partition


#: The keys an ask may carry.
ASK_KEYS = ("atoms", "direction", "partition")


def read_ask(raw: Any) -> Dict[str, Any]:
    """WHAT A PERSON ASKS of a result's infrared, as the load door reads it:
    ``{direction, atoms, partition}`` -- ``direction`` ``x``, ``y``, ``z``
    or three numbers not all zero, always; ``atoms`` a label of the result's
    structure or :data:`FREE_SET`, or ``None`` where no set is named (a
    stored dipole change is the whole system's); ``partition`` one of
    :data:`PARTITIONS`, :data:`DEFAULT_PARTITION` when unsaid -- by the
    rules :func:`infrared_view` applies to every ask.  Refused by name when
    it is not one -- another key, a value of another kind: a malformed ask,
    which no result could answer (the load door's 400, `web/web-api.md`)."""
    shape = (f"infrared is {{direction: x, y, z or three numbers, atoms: a "
             f"label of the result's structure or {FREE_SET}, partition: "
             f"{' or '.join(PARTITIONS)}}} -- not {raw!r}")
    if not isinstance(raw, dict) or raw.get("direction") is None \
            or any(k not in ASK_KEYS for k in raw):
        raise InfraredError(shape)
    atoms = raw.get("atoms")
    if atoms is not None and atoms != FREE_SET and not (
            isinstance(atoms, str) and atoms):
        raise InfraredError(shape)
    _unit(raw["direction"])
    partition = raw.get("partition")
    return {"atoms": (dict(atoms) if isinstance(atoms, dict) else atoms),
            "direction": raw["direction"],
            "partition": _partition_named(
                DEFAULT_PARTITION if partition is None else partition)}


def infrared_view(result: SpectraResults, *, atoms: Sequence[int],
                  direction: Union[str, Sequence[float]],
                  partition: str = DEFAULT_PARTITION,
                  atoms_label: str = "") -> InfraredView:
    """THE INFRARED OF THE SET ``atoms`` (user indices) for light whose
    field points along ``direction`` (``x``, ``y``, ``z`` or a vector), the
    charges by ``partition`` -- from the result's ``partial_charges``, as
    `science/vibrational-optical-response.md` § 2 states it:

    * the set's charge and dipole at every force-constant step, the dipole
      about the set's own centre of mass at equilibrium -- the masses the
      result states -- and at that step's positions --
      the reference geometry with the step's atom moved by its side times
      the step;
    * the central difference over each atom's two sides along each axis;
    * projected on each mode's canonical pattern: ``∂μ/∂Q`` and ``∂q/∂Q``;
    * where the result holds atoms, the molecule meets the light in one
      orientation: ``A(ê) = 3 · IR_KM_MOL_PER_D2_A2_AMU · (ê·∂μ/∂Q in
      D/Å/amu^½)²``; where it holds none, it tumbles, and the strength is
      the orientation average ``IR_KM_MOL_PER_D2_A2_AMU · |∂μ/∂Q|²``, no
      direction (:func:`orientation_of`) -- the direction asked is read and
      not used;
    * ranked against the strongest real mode -- no rank where that is at
      or below the infrared channel's band floor, the rule every strength
      goes by (`engines/vibration.md` § 6.6).

    Refused, by name: a result that states no charges, a partition it does
    not state, an empty set or an atom it does not hold, a direction that
    is no direction."""
    pc = result.partial_charges
    if pc is None:
        raise InfraredError(f"the result states no charges: "
                            f"{no_charges_reason(result)}")
    if _partition_named(partition) not in pc["charges_e"]:
        raise InfraredError(f"the result states {sorted(pc['charges_e'])} "
                            f"charges, not {partition!r}")
    n = int(result.n_atoms_total)
    if isinstance(atoms, (str, bytes)) or not all(
            isinstance(a, Integral) and not isinstance(a, bool)
            for a in atoms):
        raise InfraredError(f"the set is one or more of the result's {n} "
                            f"atoms, by user index -- not {atoms!r}")
    chosen = sorted({int(a) for a in atoms})
    if not chosen or any(not 0 <= a < n for a in chosen):
        raise InfraredError(f"the set is one or more of the result's {n} "
                            f"atoms, by user index -- not {list(atoms)}")
    e, name = _unit(direction)
    e = np.asarray(e, dtype=float)

    structure = _structure_of(result)
    positions = np.asarray(structure.positions, dtype=float)
    q = np.asarray(pc["charges_e"][partition], dtype=float)
    # THE CENTRE OF MASS of the set at equilibrium, a fixed point: where a
    # mode moves charge into or out of the set, the dipole change depends on
    # it (the method's § 2.1; the user, 2026-10-24: "center should be mass
    # center").
    masses = np.array([mass_of(a, structure.elements,
                               result.equilibrium_masses_amu)
                       for a in chosen])
    centre = (masses[:, None] * positions[chosen]).sum(axis=0) / masses.sum()
    delta = float(pc["displacement_ang"])
    steps = pc["steps"]

    def dipole_and_charge(i: int) -> Tuple[np.ndarray, float]:
        p = positions.copy()
        s = steps[i]
        if s["atom"] is not None:
            p[s["atom"], "xyz".index(s["axis"])] += s["side"] * delta
        qs = q[i, chosen]
        return ((p[chosen] - centre) * qs[:, None]).sum(axis=0), float(qs.sum())

    # THE DERIVATIVES PER ATOM AND AXIS -- steps come in -/+ pairs
    # (`results.FC_STEP_PATTERN`, checked when the result was built).
    dmu_dR: Dict[Tuple[int, int], np.ndarray] = {}
    dQ_dR: Dict[Tuple[int, int], float] = {}
    for i in range(1, len(steps), 2):
        (mu_m, Q_m), (mu_p, Q_p) = dipole_and_charge(i), dipole_and_charge(i + 1)
        key = (int(steps[i]["atom"]), "xyz".index(steps[i]["axis"]))
        dmu_dR[key] = (mu_p - mu_m) / (2.0 * delta)
        dQ_dR[key] = (Q_p - Q_m) / (2.0 * delta)

    free = [int(k) for k in result.free_atom_idxs]
    orientation = orientation_of(result)
    rows = []
    for m in result.modes:
        L = np.asarray(m.eigenvector_canonical, dtype=float)
        dmu = sum(dmu_dR[(k, b)] * L[r, b]
                  for r, k in enumerate(free) for b in range(3))
        dQ = sum(dQ_dR[(k, b)] * L[r, b]
                 for r, k in enumerate(free) for b in range(3))
        dmu = np.asarray(dmu, dtype=float)
        rows.append((m, dmu, _strength(orientation, e, dmu * DEBYE_E_ANGSTROM),
                     float(dQ)))
    held = orientation == ORIENTATION_HELD
    return InfraredView(atoms_label=str(atoms_label), atoms=tuple(chosen),
                        source=SOURCE_CHARGES, orientation=orientation,
                        direction=(tuple(float(x) for x in e) if held
                                   else None),
                        direction_name=name if held else "",
                        partition=partition, displacement_ang=delta,
                        modes=_modes_of(rows))


def _strength(orientation: str, e: np.ndarray, dmu_debye: np.ndarray) -> float:
    """A mode's strength from its dipole change in D/(Å amu^½): along the
    light's direction for a held molecule, the orientation average for a
    free one (:func:`orientation_of`) -- by the one formula
    (`activity.ir_strength_km_mol`)."""
    return ir_strength_km_mol(
        dmu_debye, e if orientation == ORIENTATION_HELD else None)


def _modes_of(rows) -> Tuple[InfraredMode, ...]:
    """Each mode's infrared from ``(mode, dmu in e/amu^½, strength, charge
    moved)``, ranked against the strongest real mode -- no rank at or below
    the infrared band floor (`engines/vibration.md` § 6.6)."""
    strongest = max((a for m, _, a, _ in rows if not m.has_imag), default=0.0)
    if strongest <= CHANNEL_PRESENT_FLOOR_IR_KM_MOL:
        strongest = 0.0     # no band: every computed value is residue
    return tuple(InfraredMode(
        index_1based=int(m.index_1based),
        frequency_cm1=float(m.frequency_cm1), has_imag=bool(m.has_imag),
        dmu_dq_e_per_amu12=tuple(float(x) for x in dmu),
        intensity_km_mol=float(a),
        relative=(float(a) / strongest if strongest > 0.0 else None),
        charge_moved_e_per_amu12_ang=dQ) for m, dmu, a, dQ in rows)


def dipole_absent_reason(result: SpectraResults) -> str:
    """Why a result whose infrared is the engine's own keeps no dipole
    change to read along a direction."""
    return (f"this {result.engine} result keeps no dipole change along its "
            f"modes (written before molbuilder 2.37.0, schema "
            f"{result.schema_version}), so the light's direction cannot be "
            f"applied to it: its column is its orientation average")


def dipole_view(result: SpectraResults, *,
                direction: Union[str, Sequence[float]]) -> InfraredView:
    """THE INFRARED FROM THE DIPOLE CHANGE THE ENGINE STORED WHOLE (PySCF,
    `engines/vibration.md` § 4.6): the whole system's, each mode's
    ``dipole_derivative_d_per_ang_amu12`` read along ``direction`` where the
    result holds atoms, its orientation average where it holds none
    (:func:`orientation_of`).  Refused, by name, where the result's infrared
    is not the engine's own, or where it keeps no vector."""
    if source_of(result) != SOURCE_DIPOLE:
        raise InfraredError(f"this {result.engine} result's infrared is not "
                            f"the engine's own dipole change")
    if any(m.dipole_derivative_d_per_ang_amu12 is None for m in result.modes):
        raise InfraredError(dipole_absent_reason(result))
    e, name = _unit(direction)
    e = np.asarray(e, dtype=float)
    orientation = orientation_of(result)
    rows = []
    for m in result.modes:
        v = np.asarray(m.dipole_derivative_d_per_ang_amu12, dtype=float)
        rows.append((m, v / DEBYE_E_ANGSTROM, _strength(orientation, e, v),
                     None))
    held = orientation == ORIENTATION_HELD
    return InfraredView(atoms_label=WHOLE_SYSTEM_NAME,
                        atoms=tuple(range(int(result.n_atoms_total))),
                        source=SOURCE_DIPOLE, orientation=orientation,
                        direction=(tuple(float(x) for x in e) if held
                                   else None),
                        direction_name=name if held else "",
                        partition="", displacement_ang=0.0,
                        modes=_modes_of(rows))


__all__ = ["ASK_KEYS", "CHARGES_KEPT_SINCE", "DEFAULT_PARTITION", "DIRECTIONS",
           "FREE_SET", "FREE_SET_NAME", "ORIENTATION_FREE", "ORIENTATION_HELD",
           "SOURCE_CHARGES", "SOURCE_DIPOLE", "WHOLE_SYSTEM_NAME", "atoms_of",
           "dipole_absent_reason", "dipole_view", "orientation_of", "sets_of",
           "source_of",
"InfraredError", "InfraredMode",
           "InfraredView", "infrared_view", "labelled_atoms", "labels_of",
           "no_charges_reason", "read_ask"]
