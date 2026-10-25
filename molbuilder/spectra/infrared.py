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
        atoms, the one way both doors name the set
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..constants import DEBYE_E_ANGSTROM, IR_KM_MOL_PER_D2_A2_AMU
from ..structure import Structure
from .results import PARTITIONS, SpectraResults

#: The directions a person names by an axis.
DIRECTIONS: Dict[str, Tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

#: km/mol per (D/Å)²/amu for a molecule held in one orientation, its field
#: along one direction: three times the tumbling average
#: (`constants.IR_KM_MOL_PER_D2_A2_AMU`; the method's § 2.3).
ORIENTED_KM_MOL_PER_D2_A2_AMU = 3.0 * IR_KM_MOL_PER_D2_A2_AMU


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
    #: ``A(ê)`` over the strongest real mode's, ``None`` when none absorbs
    relative: Optional[float]
    #: the charge the mode moves into or out of the set, e/amu^½
    charge_moved_e_per_amu12: float


@dataclass(frozen=True)
class InfraredView:
    """What :func:`infrared_view` answers: the set (its label and atoms by
    user index), the direction, the partition, the step the derivatives
    were taken at, and every mode's infrared."""
    atoms_label: str
    atoms: Tuple[int, ...]
    direction: Tuple[float, float, float]
    direction_name: str
    partition: str
    displacement_ang: float
    modes: Tuple[InfraredMode, ...]

    def to_wire(self) -> Dict[str, Any]:
        """As a page is answered: every field, each mode a dict."""
        return {"atoms_label": self.atoms_label, "atoms": list(self.atoms),
                "direction": list(self.direction),
                "direction_name": self.direction_name,
                "partition": self.partition,
                "displacement_ang": self.displacement_ang,
                "modes": [{"index_1based": m.index_1based,
                           "frequency_cm1": m.frequency_cm1,
                           "has_imag": m.has_imag,
                           "dmu_dq_e_per_amu12": list(m.dmu_dq_e_per_amu12),
                           "intensity_km_mol": m.intensity_km_mol,
                           "relative": m.relative,
                           "charge_moved_e_per_amu12":
                               m.charge_moved_e_per_amu12}
                          for m in self.modes]}


def _structure_of(result: SpectraResults) -> Structure:
    stated = result.structure_stated()
    if stated is None:
        raise InfraredError("the result states no geometry yet")
    return Structure.from_dict(stated)


def labelled_atoms(result: SpectraResults, label: str) -> List[int]:
    """The atoms of ``label``, a label of the result's structure, by user
    index -- refused naming the labels it holds."""
    regions = _structure_of(result).regions or {}
    if label not in regions or not regions[label]:
        held = sorted(k for k, v in regions.items() if v)
        raise InfraredError(
            f"the result's structure holds no label {label!r}; its labels "
            f"are {held or 'none'}")
    return sorted(int(i) for i in regions[label])


def _unit(direction: Union[str, Sequence[float]]) -> Tuple[Tuple[float, ...],
                                                           str]:
    if isinstance(direction, str):
        if direction not in DIRECTIONS:
            raise InfraredError(f"the direction is one of "
                                f"{sorted(DIRECTIONS)} or a vector, not "
                                f"{direction!r}")
        return DIRECTIONS[direction], direction
    v = np.asarray(direction, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or not np.any(v):
        raise InfraredError(f"the direction is three finite numbers, not "
                            f"all zero -- not {direction!r}")
    v = v / np.linalg.norm(v)
    return tuple(float(x) for x in v), ""


def infrared_view(result: SpectraResults, *, atoms: Sequence[int],
                  direction: Union[str, Sequence[float]],
                  partition: str = "hirshfeld",
                  atoms_label: str = "") -> InfraredView:
    """THE INFRARED OF THE SET ``atoms`` (user indices) for light whose
    field points along ``direction`` (``x``, ``y``, ``z`` or a vector), the
    charges by ``partition`` -- from the result's ``partial_charges``, as
    `science/vibrational-optical-response.md` § 2 states it:

    * the set's charge and dipole at every force-constant step, the dipole
      about the set's own equilibrium centre and at that step's positions --
      the reference geometry with the step's atom moved by its side times
      the step;
    * the central difference over each atom's two sides along each axis;
    * projected on each mode's canonical pattern: ``∂μ/∂Q`` and ``∂Q/∂Q``;
    * ``A(ê) = 3 · IR_KM_MOL_PER_D2_A2_AMU · (ê·∂μ/∂Q in D/Å/amu^½)²``,
      ranked against the strongest real mode.

    Refused, by name: a result that states no charges, a partition it does
    not state, an empty set or an atom it does not hold, a direction that
    is no direction."""
    pc = result.partial_charges
    if pc is None:
        raise InfraredError(
            "the result states no charges: its force-constant run printed "
            "none (a deck prepared before 2.35.2), or it is a PySCF result, "
            "whose infrared is its own")
    if partition not in PARTITIONS or partition not in pc["charges_e"]:
        raise InfraredError(f"the result states {sorted(pc['charges_e'])} "
                            f"charges, not {partition!r}")
    n = int(result.n_atoms_total)
    chosen = sorted({int(a) for a in atoms})
    if not chosen or any(not 0 <= a < n for a in chosen):
        raise InfraredError(f"the set is one or more of the result's {n} "
                            f"atoms, by user index -- not {list(atoms)}")
    e, name = _unit(direction)
    e = np.asarray(e, dtype=float)

    positions = np.asarray(_structure_of(result).positions, dtype=float)
    q = np.asarray(pc["charges_e"][partition], dtype=float)
    centre = positions[chosen].mean(axis=0)
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
    rows = []
    for m in result.modes:
        L = np.asarray(m.eigenvector_canonical, dtype=float)
        dmu = sum(dmu_dR[(k, b)] * L[r, b]
                  for r, k in enumerate(free) for b in range(3))
        dQ = sum(dQ_dR[(k, b)] * L[r, b]
                 for r, k in enumerate(free) for b in range(3))
        along = float(np.dot(e, dmu)) * DEBYE_E_ANGSTROM
        rows.append((m, np.asarray(dmu, dtype=float),
                     ORIENTED_KM_MOL_PER_D2_A2_AMU * along * along, float(dQ)))
    strongest = max((a for m, _, a, _ in rows if not m.has_imag), default=0.0)
    modes = tuple(InfraredMode(
        index_1based=int(m.index_1based),
        frequency_cm1=float(m.frequency_cm1), has_imag=bool(m.has_imag),
        dmu_dq_e_per_amu12=tuple(float(x) for x in dmu),
        intensity_km_mol=float(a),
        relative=(float(a) / strongest if strongest > 0.0 else None),
        charge_moved_e_per_amu12=dQ) for m, dmu, a, dQ in rows)
    return InfraredView(atoms_label=str(atoms_label), atoms=tuple(chosen),
                        direction=tuple(float(x) for x in e),
                        direction_name=name, partition=partition,
                        displacement_ang=delta, modes=modes)


__all__ = ["DIRECTIONS", "ORIENTED_KM_MOL_PER_D2_A2_AMU", "InfraredError",
           "InfraredMode", "InfraredView", "infrared_view", "labelled_atoms"]
