"""THE MODE'S AVERAGE OVER A FRAME SET -- `engines/transport.md` § 2a.12;
the science is `science/vibrational-averaging.md` (§ 5 the rule, § 7 what it
assumes, § 8 what the record carries); plan § 5z Q17-e.

A frame set samples ONE normal mode, and states what every frame is -- a
mode's frame set, checked at the citation door (`frameset.read`,
`model/structure.md` § 2.2f).  The average is computed with each frame's
stated weight alone; the mode's definition -- its structure's rows, the
weights' sum and the tolerance they were checked to -- is stated beside it.

**What the average is and is not**: the weighted sum of every frame's T(E)
and its difference from the base frame's -- what the frames give.  No
curvature is read off them: a level crossing E_F within the vibration makes
the transmission jump between frames, and a second difference of three is
then no curvature (`science/vibrational-averaging.md` § 5.3; the user,
2026-10-10).  **A point not done is a gap**: the average at a voltage waits
for every frame's point there, never computed from the frames that happen to
be done.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

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

#: How close two voltages must be to be one.
_SAME = 1e-9


class AverageError(ValueError):
    """The frames' transmissions cannot be averaged -- the message names the
    voltage and the frame, ready to surface verbatim."""


def mode_average(n_frames: Optional[int], frame_set,
                 points: List[Dict], missing: List[Dict]) -> Dict:
    """The record's ``average`` block (§ 2a.12): the mode's definition as the
    set states it (``frame_set``, `frameset.ModeFrameSet`, ``None`` for a
    structure that states none) -- its structure's rows, each frame's weight,
    their sum and the tolerance they were checked to -- the assumptions, and,
    at each voltage the points ran at, the mode's average over every frame at
    its stated weight (``⟨T(E)⟩ = Σ_f W_f T_f(E)``), its change from the base
    frame's (``ΔT(E) = ⟨T(E)⟩ − T₀(E)``), and at E_F the averaged conductance
    beside the base frame's, with the change in per cent of it.  ``n_frames``
    is the composed junction's count (``None`` before its first prep);
    ``points`` are the record's done points, ``missing`` its pending and
    failed ones -- each a dict with ``frame`` and ``bias_v``.  ``why`` says,
    in one sentence, why there is no average: nothing composed yet, one
    structure, or a family of frames stating no mode."""
    from ..frameset import STRUCTURE_ROWS, WEIGHT_SUM_TOLERANCE
    weights = frame_set.weights if frame_set is not None else None
    # EVERY KEY, EVERY RECORD: the definition's structure rows by their own
    # names, ``None`` where the set states none.
    rows = (frame_set.structure_rows() if frame_set is not None
            else {r.name: None for r in STRUCTURE_ROWS})
    block: Dict = {
        "frames": n_frames,
        **rows,
        "weights": weights,
        "weight_sum": math.fsum(weights) if weights is not None else None,
        "tolerance": WEIGHT_SUM_TOLERANCE,
        "assumptions": list(ASSUMPTIONS),
        "why": None,
        "at": [],
    }
    if n_frames is None:
        block["why"] = ("the junction is not composed yet -- the "
                        "calculation's first prep composes it")
        return block
    if n_frames < 2:
        block["why"] = ("one structure, not a frame set: there are no frames "
                        "to average over")
        return block
    if frame_set is None:
        block["why"] = ("the set states no mode's definition -- a family of "
                        "frames with no average; each frame's curve is its "
                        "own point (model/structure.md 2.2f)")
        return block
    voltages: List[float] = []
    for p in list(points) + list(missing):
        v = float(p["bias_v"])
        if not any(abs(v - u) < _SAME for u in voltages):
            voltages.append(v)
    block["at"] = [_at_voltage(v, n_frames, weights, points, missing)
                   for v in voltages]
    return block


def _at_voltage(v: float, n: int, weights: List[float],
                points: List[Dict], missing: List[Dict]) -> Dict:
    """One voltage's entry of :func:`mode_average`."""
    import numpy as np
    from ..structure import frame_words
    here = {p["frame"]: p for p in points if abs(float(p["bias_v"]) - v) < _SAME}
    waits = [f for f in range(n) if f not in here]
    # EVERY KEY, EVERY ENTRY: ``None`` where this voltage has no value.
    entry: Dict = {"bias_v": v, "waits_for": waits, "why": None,
                   "energy_ev": None, "transmission": None,
                   "delta_transmission": None, "conductance_g0": None,
                   "base_conductance_g0": None, "delta_conductance_g0": None,
                   "conductance_change_percent": None,
                   "conductance_why": None}
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
            raise AverageError(
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
    return entry
