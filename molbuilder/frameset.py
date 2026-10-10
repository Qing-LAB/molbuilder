"""A MODE'S FRAME SET -- its definition and its rules, the one home in code
of `model/structure.md` § 2.2f.

MODULE  frameset (L1; the Structure, numpy and the standard library)
ROLE    the definition as data -- every row a mode's frame set states, where
        it lives, whether it is given, measured from the coordinates or
        derived, and its unit (:data:`ROWS`), the per-atom masses channel
        (:data:`MASS_CHANNEL`) and the ``info`` cluster
        (:data:`INFO_KEY`) -- and its one reader, :func:`read`, which checks
        § 2.2f's six rules and answers the set as a typed value, or ``None``
        for a structure that states none of it
USED-BY the transport citation door (`transport.compose`), the transport
        record (`transport.record`, `transport.average`), and the frame
        generator when it is built (`engines/vibration.md` § 5.10 ③)

**The coordinates are the truth; the rows and the channel describe them**
(user, 2026-10-10: "full data record with clear meaning/definition is
crucial"; "explicit is always better than implicit in data science").  Every
row that can be checked against the coordinates is, to within what their six
written decimals allow (`Structure.to_xyz`), and every derived row against
its formula.  A row's unit is in its name, as in the vibration result the
values come from; a row's ``unit`` field, when written, is the ASCII spelling
:data:`ROWS` states.  Nothing else of the structure is read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .structure import XYZ_DECIMALS, atom_words, frame_words

#: Each atom's mass, the value channel the mode's coordinate is weighted by
#: -- molbuilder's own (`model/structure-annotations.md`).
MASS_CHANNEL = "mass_amu"

#: Where the mode came from: ``info.vibration = {run, result_sha256}``.
INFO_KEY = "vibration"
INFO_FIELDS = ("run", "result_sha256")


@dataclass(frozen=True)
class Row:
    """One row of the definition: its ``name`` (its unit in it), ``where`` it
    lives (``structure`` -- the structure's rows -- or ``frame``, each
    frame's), its ``kind`` (``given``, ``measured`` from the coordinates, or
    ``derived`` from the others) and the ASCII spelling of its ``unit``
    (``None``: a number without one)."""
    name: str
    where: str
    kind: str
    unit: Optional[str]


#: THE DEFINITION, as data -- `model/structure.md` § 2.2f's table, read by
#: every rule below and by every door that writes or reads a set.
ROWS: Tuple[Row, ...] = (
    Row("mode_index_1based", "structure", "given", None),
    Row("frequency_cm1", "structure", "given", "cm-1"),
    Row("temperature_k", "structure", "given", "K"),
    Row("zero_point_amplitude_amu12_ang", "structure", "given", "amu^1/2 angstrom"),
    Row("sigma_amu12_ang", "structure", "derived", "amu^1/2 angstrom"),
    Row("displacement_ang", "frame", "measured", "angstrom"),
    Row("max_atom_displacement_ang", "frame", "measured", "angstrom"),
    Row("q_amu12_ang", "frame", "derived", "amu^1/2 angstrom"),
    Row("node_sigma", "frame", "derived", None),
    Row("weight", "frame", "given", None),
)
STRUCTURE_ROWS = tuple(r for r in ROWS if r.where == "structure")
FRAME_ROWS = tuple(r for r in ROWS if r.where == "frame")

#: How far the weights' sum may be from 1 (§ 2.2f rule 5) -- the record
#: states it beside the average.
WEIGHT_SUM_TOLERANCE = 1e-6
#: How far a derived row may be from its formula, relative (rule 4): each is
#: the formula of values stated at full precision.
FORMULA_RTOL = 1e-9
#: How far one written coordinate may be from the value it stands for: half
#: the last of the decimals `Structure.to_xyz` writes (`XYZ_DECIMALS`).  A
#: displacement between two frames is then within twice it per component,
#: `sqrt(3)` times that an atom.
COORDINATE_HALF_DECIMAL_ANG = 0.5 * 10.0 ** -XYZ_DECIMALS


class FrameSetError(ValueError):
    """A mode's frame set that breaks its definition -- the message names the
    frame and the row, ready to surface verbatim."""


@dataclass(frozen=True)
class FrameRows:
    """One frame's rows, as stated."""
    displacement_ang: float
    max_atom_displacement_ang: float
    q_amu12_ang: float
    node_sigma: float
    weight: float


@dataclass(frozen=True)
class ModeFrameSet:
    """A mode's frame set, checked: the structure's rows, each frame's, every
    atom's mass and where the mode came from -- every value as stated."""
    mode_index_1based: int
    frequency_cm1: float
    temperature_k: float
    zero_point_amplitude_amu12_ang: float
    sigma_amu12_ang: float
    frames: Tuple[FrameRows, ...]
    masses_amu: Tuple[float, ...]
    vibration: Mapping[str, Any]

    @property
    def weights(self) -> List[float]:
        """Each frame's weight, in frame order."""
        return [f.weight for f in self.frames]

    def structure_rows(self) -> Dict[str, Any]:
        """The structure's rows, ``{name: value}`` -- what the record states
        beside the average."""
        return {r.name: getattr(self, r.name) for r in STRUCTURE_ROWS}


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite number, or ``None`` -- true/false is not a
    number here, though Python counts it as one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _row(structure, name: str, frame: Optional[int]) -> Optional[Mapping]:
    """The row ``name`` of the structure's rows, or of frame ``frame``'s."""
    return next((r for r in structure.customized_rows(frame)
                 if r["name"] == name), None)


def read(structure) -> Optional[ModeFrameSet]:
    """The mode's frame set ``structure`` states, checked by
    `model/structure.md` § 2.2f's rules -- or ``None`` when it states none of
    the definition: a family of frames.  :class:`FrameSetError`, naming the
    frame and the row, for a set that breaks a rule."""
    n = structure.n_frames
    srows = {r.name: _row(structure, r.name, None) for r in STRUCTURE_ROWS}
    frows = [{r.name: _row(structure, r.name, f) for r in FRAME_ROWS}
             for f in range(n)]
    channel = (structure.annotations or {}).get(MASS_CHANNEL)
    info = (structure.info or {}).get(INFO_KEY)
    stated = (any(v is not None for v in srows.values())
              or any(v is not None for fr in frows for v in fr.values())
              or channel is not None or info is not None)
    if not stated:
        return None

    # RULE 1 -- ALL OR NONE.
    missing = [f"the structure's {k}" for k, v in srows.items() if v is None]
    for f, fr in enumerate(frows):
        gone = [k for k, v in fr.items() if v is None]
        if gone:
            missing.append(f"{frame_words(f, n)}'s {', '.join(gone)}")
    if channel is None:
        missing.append(f"the {MASS_CHANNEL} channel")
    if not isinstance(info, Mapping) or any(
            not isinstance(info.get(k), str) or not info.get(k)
            for k in INFO_FIELDS):
        missing.append(f"info.{INFO_KEY} with its "
                       f"{' and '.join(INFO_FIELDS)}")
    if missing:
        raise FrameSetError(
            f"a mode's frame set states every row of its definition, or none "
            f"of them -- this one does not state {'; '.join(missing)} "
            f"(model/structure.md 2.2f)")

    # THE VALUES: numbers, each in its row's own unit.
    bad: List[str] = []
    for r in ROWS:
        for f, row in ([(None, srows[r.name])] if r.where == "structure"
                       else [(f, frows[f][r.name]) for f in range(n)]):
            where = ("the structure's" if f is None
                     else f"{frame_words(f, n)}'s")
            if _number(row["value"]) is None:
                bad.append(f"{where} {r.name} is {row['value']!r}, not a "
                           f"number")
            unit = row.get("unit")
            # RULE 6 -- a row's unit, when written, is the unit its name states.
            if unit is not None and unit != r.unit:
                bad.append(f"{where} {r.name} states the unit {unit!r}"
                           + (f", not {r.unit!r}" if r.unit else
                              ", for a number without one"))
    mode = srows["mode_index_1based"]["value"]
    if (isinstance(mode, bool) or not isinstance(mode, int)
            or mode < 1):
        bad.append(f"the structure's mode_index_1based is {mode!r}, not a "
                   f"mode's number counted from 1")
    masses = _masses(channel, structure.n_atoms, bad)
    if bad:
        raise FrameSetError("; ".join(bad) + " (model/structure.md 2.2f)")

    def value(f, name):
        return float(frows[f][name]["value"])
    nu = float(srows["frequency_cm1"]["value"])
    temp = float(srows["temperature_k"]["value"])
    q_zp = float(srows["zero_point_amplitude_amu12_ang"]["value"])
    sigma = float(srows["sigma_amu12_ang"]["value"])
    # RULE 4's DOMAIN -- a real mode at a temperature: what the formulas
    # below are defined on, so each is checked, never divided by zero.
    undefined = [
        f"the structure's {name} is {v!r}, not {what}"
        for name, v, ok, what in (
            ("frequency_cm1", nu, nu > 0.0, "positive -- a real mode"),
            ("zero_point_amplitude_amu12_ang", q_zp, q_zp > 0.0, "positive"),
            ("sigma_amu12_ang", sigma, sigma > 0.0, "positive"),
            ("temperature_k", temp, temp >= 0.0, "0 K or above"))
        if not ok]
    if undefined:
        raise FrameSetError("; ".join(undefined)
                            + " (model/structure.md 2.2f)")

    # RULE 2 -- FRAME 0 IS THE EQUILIBRIUM.
    at_rest = [name for name in ("displacement_ang",
                                 "max_atom_displacement_ang", "q_amu12_ang",
                                 "node_sigma") if value(0, name) != 0.0]
    if at_rest:
        bad.append(f"{frame_words(0, n)} is the equilibrium the mode is "
                   f"taken at, so its {', '.join(at_rest)} "
                   f"{'is' if len(at_rest) == 1 else 'are'} 0, not "
                   + ", ".join(repr(value(0, k)) for k in at_rest))

    # RULE 3 -- THE MEASURED ROWS MATCH THE COORDINATES, AND THE FRAMES MOVE
    # ALONG ONE MODE AT THEIR STATED POSITIONS.
    coords = np.asarray(structure.frames if n > 1
                        else [structure.positions], dtype=float)
    dR = coords - coords[0]
    m = np.asarray(masses, dtype=float)
    per_atom = math.sqrt(3.0) * 2.0 * COORDINATE_HALF_DECIMAL_ANG
    tol_disp = per_atom * math.sqrt(structure.n_atoms)
    tol_q = per_atom * math.sqrt(float(m.sum()))
    u = np.sqrt(m)[None, :, None] * dR                     # amu^1/2.A
    q = np.array([value(f, "q_amu12_ang") for f in range(n)])
    g = int(np.argmax(np.abs(q)))
    norm_g = float(np.linalg.norm(u[g]))
    # No frame moved, so there is no direction to measure along: a frame
    # then sits at its q only when it neither moves nor states a position.
    e_hat = (np.sign(q[g]) * u[g] / norm_g if norm_g > 0.0 else None)
    for f in range(n):
        disp = float(np.linalg.norm(dR[f]))
        top = float(np.linalg.norm(dR[f], axis=1).max())
        for name, measured, tol in (
                ("displacement_ang", disp, tol_disp),
                ("max_atom_displacement_ang", top, per_atom)):
            if abs(value(f, name) - measured) > tol:
                bad.append(f"{frame_words(f, n)}'s {name} is "
                           f"{value(f, name)!r}; its coordinates give "
                           f"{measured!r}")
        off = (float(np.linalg.norm(u[f] - q[f] * e_hat))
               if e_hat is not None
               else math.hypot(float(np.linalg.norm(u[f])), q[f]))
        if off > 2.0 * tol_q:
            bad.append(
                f"{frame_words(f, n)} does not sit at q_amu12_ang "
                f"{float(q[f])!r} along the mode the set moves along: its "
                f"mass-weighted displacement is {off:.3g} amu^1/2 angstrom "
                f"from there (size {float(np.linalg.norm(u[f])):.6g})")

    # RULE 4 -- THE DERIVED ROWS MATCH THEIR FORMULAS.
    from .spectra.derived import thermal_spread_amu12_ang
    want = thermal_spread_amu12_ang(q_zp, nu, temp)
    if not math.isclose(sigma, want, rel_tol=FORMULA_RTOL, abs_tol=0.0):
        bad.append(f"the structure's sigma_amu12_ang is {sigma!r}; "
                   f"zero_point_amplitude_amu12_ang * sqrt(coth(h c nu / "
                   f"2 k_B T)) gives {want!r}")
    for f in range(n):
        node = value(f, "node_sigma")
        if not math.isclose(node, q[f] / sigma, rel_tol=FORMULA_RTOL,
                            abs_tol=0.0):
            bad.append(f"{frame_words(f, n)}'s node_sigma is {node!r}; "
                       f"q_amu12_ang / sigma_amu12_ang gives "
                       f"{float(q[f] / sigma)!r}")

    # RULE 5 -- THE WEIGHTS.
    weights = [value(f, "weight") for f in range(n)]
    outside = [f"{frame_words(f, n)}'s weight is {w!r}"
               for f, w in enumerate(weights) if not 0.0 < w <= 1.0]
    if outside:
        bad.append("; ".join(outside) + " -- a weight is a number in (0, 1]")
    total = math.fsum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        bad.append(f"the {n} frames' weights sum to {total!r}, not 1 "
                   f"within {WEIGHT_SUM_TOLERANCE:g}")
    if bad:
        raise FrameSetError("; ".join(bad) + " (model/structure.md 2.2f)")

    return ModeFrameSet(
        mode_index_1based=int(mode), frequency_cm1=nu, temperature_k=temp,
        zero_point_amplitude_amu12_ang=q_zp, sigma_amu12_ang=sigma,
        frames=tuple(FrameRows(**{r.name: value(f, r.name)
                                  for r in FRAME_ROWS}) for f in range(n)),
        masses_amu=tuple(masses), vibration=dict(info))


def without_definition(structure) -> None:
    """Take a mode's definition off ``structure``, in place -- § 2.2f's rows,
    the :data:`MASS_CHANNEL` channel and ``info.vibration``: what one frame
    taken out of a set becomes (`model/structure.md` § 2.2e), one frame being
    no sample of a mode's distribution.  The rows a person wrote, the labels
    and the rest of ``info`` stay."""
    for r in STRUCTURE_ROWS:
        structure.remove_customized(r.name)
    for f in range(structure.n_frames):
        for r in FRAME_ROWS:
            structure.remove_customized(r.name, frame=f)
    (structure.annotations or {}).pop(MASS_CHANNEL, None)
    structure.drop_info(INFO_KEY)


def _masses(channel, n_atoms: int, bad: List[str]) -> List[float]:
    """Every atom's mass from the :data:`MASS_CHANNEL` channel -- one
    positive finite number per atom, its gaps and bad values added to
    ``bad``."""
    if getattr(channel, "kind", None) != "value":
        bad.append(f"the {MASS_CHANNEL} channel is not a value channel")
        return []
    data = channel.data or {}
    out: List[float] = []
    for i in range(n_atoms):
        v = _number(data.get(i))
        if v is None or not v > 0.0:
            bad.append(f"{atom_words(i)}'s {MASS_CHANNEL} is "
                       f"{data.get(i)!r}, "
                       f"not a positive mass")
        out.append(v if v is not None else float("nan"))
    return out
