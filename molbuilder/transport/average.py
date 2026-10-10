"""THE MODE'S AVERAGE OVER A FRAME SET -- `engines/transport.md` § 2a.12;
the science is `science/vibrational-averaging.md` (§ 5 the rule, § 7 what it
assumes, § 8 what the record carries); plan § 5z Q17-e.

A frame set samples ONE normal mode.  The structure's `customized` rows
announce it -- the mode and the rule's order -- and each frame's own rows say
what that frame is: its displacement in units of the mode's spread and its
weight in the average (`model/structure.md` § 2.2d).  Transport reads those
four names -- the constants below, as `sort.py` owns the region names --
computes with them alone, and shows every other row whole, on the record's
points.

**The weights are checked, never assumed**: every frame states one, frame 0
included, each in (0, 1], and they sum to 1 within the tolerance the record
writes.  A set whose frames state none is a family with no average, and the
record says so.  **A point not done is a gap**: the average at a voltage
waits for every frame's point there, never computed from the frames that
happen to be done.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

#: THE ROWS TRANSPORT READS (`engines/transport.md` § 2a.9): the mode and the
#: rule's order on the structure's rows, each frame's node and weight on its
#: own.  Every other row is the writer's, carried and shown.
ROW_MODE = "mode"
ROW_ORDER = "order"
ROW_NODE = "node_sigma"
ROW_WEIGHT = "weight"

#: How far the stated weights' sum may be from 1 -- written in the record
#: (§ 2a.12; the frame generator writes every weight at full precision).
WEIGHT_SUM_TOLERANCE = 1e-6

#: What the average assumes, said beside its numbers
#: (`science/vibrational-averaging.md` § 7).
ASSUMPTIONS = (
    "A harmonic mode: the molecule's position along the mode follows its "
    "Gaussian thermal distribution, the same either side of equilibrium; a "
    "strongly anharmonic mode skews it, which a harmonic frame set does not "
    "carry.",
    "Static frames: each frame is a frozen geometry, the electron crossing "
    "fast against the vibration; the inelastic steps at eV = hbar*omega and "
    "a strongly coupled level's polaron shift need the electron-vibration "
    "coupling and are not here.",
    "The equilibrium distribution, one voltage at a time: the average is "
    "taken over the frames at each voltage separately, with the weights of "
    "the junction at equilibrium; a bias that heats the mode is not "
    "modelled.",
    "Each frame's electrons in their ground state for that frame: every "
    "frame runs its own self-consistent device.",
    "The level alignment is DFT's, frame by frame.",
    "The electrodes do not move: the held atoms are the leads, shared by "
    "every frame.",
)

#: How close two voltages, or a node and its mirror, must be to be one.
_SAME = 1e-9


class FrameSetError(ValueError):
    """A frame set's rows break the weight rule -- the message names the
    frames and what they state, ready to surface verbatim."""


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite number, or ``None`` -- true/false is not a
    number here, though Python counts it as one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def frame_weights(structure) -> Optional[List[float]]:
    """Each frame's stated weight, in frame order -- or ``None`` when no
    frame states one: a family of frames with no average (§ 2a.12).
    :class:`FrameSetError`, naming the frames, for a set where some frames
    state a weight and others do not, a weight that is not a number in
    (0, 1], or weights whose sum is not 1 within
    :data:`WEIGHT_SUM_TOLERANCE` -- with the sum found."""
    from .stages import frame_words
    n = structure.n_frames
    said = [structure.customized_value(ROW_WEIGHT, frame=f) for f in range(n)]
    if all(w is None for w in said):
        return None
    unstated = [f for f, w in enumerate(said) if w is None]
    if unstated:
        raise FrameSetError(
            f"{', '.join(frame_words(f, n) for f in unstated)} "
            f"state{'s' if len(unstated) == 1 else ''} no {ROW_WEIGHT} while "
            f"the others do -- every frame states its weight, the base "
            f"included, or none does (engines/transport.md 2a.12)")
    bad = [(f, w) for f, w in enumerate(said)
           if _number(w) is None or not 0.0 < _number(w) <= 1.0]
    if bad:
        raise FrameSetError(
            "; ".join(f"{frame_words(f, n)} states the {ROW_WEIGHT} {w!r}"
                      for f, w in bad)
            + " -- a weight is a number in (0, 1] (engines/transport.md "
              "2a.12)")
    weights = [_number(w) for w in said]
    total = math.fsum(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise FrameSetError(
            f"the {n} frames' weights sum to {total!r}, not 1 within "
            f"{WEIGHT_SUM_TOLERANCE:g} -- every frame states its weight, at "
            f"full precision, and the set's weights sum to 1 "
            f"(engines/transport.md 2a.12)")
    return weights


def mode_average(structure, weights: Optional[List[float]],
                 points: List[Dict], missing: List[Dict]) -> Dict:
    """The record's ``average`` block (§ 2a.12): the mode and the rule's
    order, each frame's weight and node as stated, the tolerance the weights
    were checked to, the assumptions -- and, at each voltage the points ran
    at, the mode's average over every frame at its stated weight
    (``⟨T(E)⟩ = Σ_f W_f T_f(E)``), its change from the base frame's
    (``ΔT(E) = ⟨T(E)⟩ − T₀(E)``), at E_F the averaged conductance and its
    change in per cent of the base frame's, and the curvature ``T″`` from
    the pair of frames nearest equilibrium.  ``points`` are the record's
    done points, ``missing`` its pending and failed ones -- each a dict with
    ``frame`` and ``bias_v``.  ``why`` says, in one sentence, why there is no
    average: nothing composed yet (``structure`` ``None``), one structure,
    or no weights stated."""
    n = structure.n_frames if structure is not None else None
    nodes = ([_number(structure.customized_value(ROW_NODE, frame=f))
              for f in range(n)] if structure is not None else None)
    block: Dict = {
        "frames": n,
        "mode": (structure.customized_value(ROW_MODE)
                 if structure is not None else None),
        "order": (structure.customized_value(ROW_ORDER)
                  if structure is not None else None),
        "weights": weights,
        "weight_sum": math.fsum(weights) if weights is not None else None,
        "tolerance": WEIGHT_SUM_TOLERANCE,
        "nodes_sigma": nodes,
        "assumptions": list(ASSUMPTIONS),
        "why": None,
        "at": [],
    }
    if structure is None:
        block["why"] = ("the junction is not composed yet -- the "
                        "calculation's first prep composes it")
        return block
    if n < 2:
        block["why"] = ("one structure, not a frame set: there are no frames "
                        "to average over")
        return block
    if weights is None:
        block["why"] = (f"no frame of the set states a {ROW_WEIGHT}: a family "
                        f"of frames with no average -- each frame's curve is "
                        f"its own point")
        return block
    voltages: List[float] = []
    for p in list(points) + list(missing):
        v = float(p["bias_v"])
        if not any(abs(v - u) < _SAME for u in voltages):
            voltages.append(v)
    block["at"] = [_at_voltage(v, n, weights, nodes, points, missing)
                   for v in voltages]
    return block


def _at_voltage(v: float, n: int, weights: List[float],
                nodes: List[Optional[float]], points: List[Dict],
                missing: List[Dict]) -> Dict:
    """One voltage's entry of :func:`mode_average`."""
    import numpy as np
    from .stages import frame_words
    here = {p["frame"]: p for p in points if abs(float(p["bias_v"]) - v) < _SAME}
    waits = [f for f in range(n) if f not in here]
    # EVERY KEY, EVERY ENTRY: ``None`` where this voltage has no value.
    entry: Dict = {"bias_v": v, "waits_for": waits, "why": None,
                   "energy_ev": None, "transmission": None,
                   "delta_transmission": None, "conductance_g0": None,
                   "base_conductance_g0": None, "delta_conductance_g0": None,
                   "conductance_change_percent": None,
                   "conductance_why": None, "curvature": None}
    if waits:
        # A GAP IN THE FAMILY: the average at this voltage waits for every
        # frame's point, never computed from the frames that are done.
        entry["why"] = (f"{', '.join(frame_words(f, n) for f in waits)} "
                        f"{'has' if len(waits) == 1 else 'have'} no "
                        f"transmission at {v:g} V yet: the average waits for "
                        f"every frame")
        return entry
    grid = here[0]["energy_ev"]
    for f in range(1, n):
        if here[f]["energy_ev"] != grid:
            raise FrameSetError(
                f"at {v:g} V the transmission of {frame_words(f, n)} is on "
                f"another energy grid than {frame_words(0, n)}'s -- every "
                f"frame's transmission deck is the one template's")
    curves = np.asarray([here[f]["transmission"] for f in range(n)],
                        dtype=float)
    avg = np.asarray(weights, dtype=float) @ curves
    entry.update(energy_ev=list(grid),
                 transmission=[float(x) for x in avg],
                 delta_transmission=[float(x) for x in avg - curves[0]])
    # AT E_F: the frames' own conductances, G = G0 · T(E_F), averaged at the
    # same weights -- interpolation is linear, so it is the averaged
    # curve's own.
    gs = [here[f].get("conductance_g0") for f in range(n)]
    entry["base_conductance_g0"] = gs[0]
    if any(g is None for g in gs):
        entry["conductance_why"] = ("the transmission window does not "
                                    "straddle E_F")
    else:
        g_avg = math.fsum(wf * g for wf, g in zip(weights, gs))
        entry.update(conductance_g0=g_avg, delta_conductance_g0=g_avg - gs[0])
        if gs[0]:
            entry["conductance_change_percent"] = (100.0 * (g_avg - gs[0])
                                                   / gs[0])
        else:
            entry["conductance_why"] = ("the base frame's conductance is "
                                        "zero: no change in per cent of it")
    entry["curvature"] = _curvature(n, nodes, gs)
    return entry


def _curvature(n: int, nodes: List[Optional[float]],
               gs: List[Optional[float]]) -> Dict:
    """``T″`` at E_F, ``(T₊ + T₋ − 2T₀)/h²`` (§ 2a.12): ``T₀`` the base
    frame's, ``T±`` the pair of frames nearest equilibrium -- the nodes
    ``±h`` closest to zero -- ``h`` in units of the mode's spread, so the
    value is per σ².  ``value`` ``None`` with ``why`` when the frames do not
    give it: a node not stated, the base not at node 0, no pair mirrored
    about it, or a conductance not read."""
    from .stages import frame_words
    out: Dict = {"value": None, "per": "sigma^2", "h_sigma": None,
                 "frames": None, "why": None}
    if any(x is None for x in nodes):
        unstated = [f for f, x in enumerate(nodes) if x is None]
        out["why"] = (f"{', '.join(frame_words(f, n) for f in unstated)} "
                      f"state{'s' if len(unstated) == 1 else ''} no "
                      f"{ROW_NODE}")
        return out
    if abs(nodes[0]) > _SAME:
        out["why"] = (f"the base, {frame_words(0, n)}, states {ROW_NODE} "
                      f"{nodes[0]!r}, not 0: the second difference is taken "
                      f"about the undisplaced frame")
        return out
    plus = [(x, f) for f, x in enumerate(nodes) if x > _SAME]
    minus = [(x, f) for f, x in enumerate(nodes) if x < -_SAME]
    if not plus or not minus:
        out["why"] = "no frame on one side of equilibrium"
        return out
    (hp, fp), (hm, fm) = min(plus), max(minus)
    if abs(hp + hm) > _SAME:
        out["why"] = (f"the frames nearest equilibrium, at {hp!r} and "
                      f"{hm!r}, are not mirrored about it")
        return out
    out.update(h_sigma=hp, frames=[fp, fm])
    if any(gs[f] is None for f in (0, fp, fm)):
        out["why"] = "the transmission window does not straddle E_F"
        return out
    out["value"] = (gs[fp] + gs[fm] - 2.0 * gs[0]) / (hp * hp)
    return out
