"""THE FRAMES, POST-PROCESSED -- what a multi-frame transport calculation's
points are combined and read as, after the calculation has run
(`engines/transport.md` § 2a.12; `model/structure.md` § 2.2f).

MODULE  frame_average (L2; the frames' rows (`frameset`), the transport
        stages' voltage rule, its averaged conductance
        (`transport.conductance`) and its data file's groups, numpy)
ROLE    the post-process over a multi-frame transport calculation, kept out of
        its execution: the frames' weighted average at each voltage
        (:func:`average_block`) and its summary lines (:func:`summary_lines`),
        the frame set the record cites (:func:`citation_block`), the data
        file's frame part (:func:`data_file_variables`,
        :func:`fill_data_file`), what a page is answered of the frames
        (:func:`answer`), and a data file's frames averaged at weights a
        caller gives (:func:`averaged_at`) -- with what transport's average
        assumes (:data:`ASSUMPTIONS`)
USED-BY the transport record (`transport.record`), the data file
        (`transport.datafile`), the junction's composition
        (`transport.compose`), the describe door and the record's answer
        (`web/blueprints/transport.py`), Task setup's folder answer
        (`web/blueprints/build.py`), the post-processing layer
        (`postprocess.reaveraged`)

**Transport runs the frames and keeps every point; this combines them**
(the user, 2026-10-18: "how to interpret and present the data is entirely
result tab or some other's work -- meta data is providing the background and
some parameters (such as averaging when needed) for the calculation, it
should not be deeply integrated with transport framework itself").  The one
parameter the combination takes is each frame's ``weight``
(`frameset.read`); every other row is its maker's, carried as written and
never read here, and a page is answered each by its stored name, value and
unit, none interpreted (the user, 2026-11-26: "transport does transport
calculation and that's it").  Nothing here refuses a row.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .frameset import WEIGHT_ROW, finite_number
from .units import volts_words

#: WHAT TRANSPORT'S AVERAGE OF THE FRAMES ASSUMES, said beside its numbers
#: wherever it is taken (`science/vibrational-averaging.md` § 7, transport's
#: own) -- what holds for ANY frames; what holds for frames of a vibration
#: is the maker's ``assumptions`` row, carried with the set's rows as
#: written (the user, 2026-11-26: "transport does transport calculation and
#: that's it").
ASSUMPTIONS = (
    "Static frames: each frame is a fixed geometry, and the transmission is "
    "computed at it with its atoms held still.",
    "One set of weights at every voltage: the average is taken over the "
    "frames at each voltage separately, each frame's weight the same at "
    "every voltage.",
    "Each frame's electrons in their ground state for that frame: every "
    "frame runs its own self-consistent device.",
    "The level alignment is DFT's, frame by frame.",
    "The electrodes do not move: the held atoms are the leads, shared by "
    "every frame.",
)


class AverageError(ValueError):
    """The frames' transmissions cannot be averaged -- the message names the
    voltage and the frame, ready to surface verbatim."""


# ----------------------------------------------------------------------- #
#  The frames' weighted average -- the record's ``average`` block          #
# ----------------------------------------------------------------------- #

def average_block(n_frames: Optional[int], frame_set,
                  points: List[Dict], missing: List[Dict],
                  same_volts: Callable[[float, float], bool], *,
                  unread: Optional[str] = None) -> Dict:
    """The record's ``average`` block (`engines/transport.md` § 2a.12):
    each frame's weight as its rows state it (``frame_set``,
    `frameset.FrameRows`, ``None`` for a structure that states no rows),
    their sum and its tolerance, what is said of them, what the average
    assumes where one is taken, and, at each voltage the points ran at, the
    average over every frame at its stated weight (``⟨T(E)⟩ = Σ_f W_f
    T_f(E)``), its change from frame 0's (``ΔT(E) = ⟨T(E)⟩ − T₀(E)``), and
    at E_F the averaged conductance beside frame 0's, with the change in per
    cent of it.  ``n_frames`` is the composed junction's count (``None``
    before its first prep); ``points`` are the record's done points,
    ``missing`` its pending, not run and failed ones -- each a dict with
    ``frame`` and ``bias_v``; ``same_volts`` the stages' one voltage rule;
    ``unread``, why the composed junction that holds the frames does not
    read, where it does not.  ``why`` says, in one sentence, why there is
    no average: nothing composed yet, one structure, a junction that does
    not read, frames stating no rows -- a family -- or frames that do not
    each state a weight."""
    from .frameset import WEIGHT_SUM_TOLERANCE
    weights = (list(frame_set.weights) if frame_set is not None
               and frame_set.weights is not None else None)
    # EVERY KEY, EVERY RECORD.
    block: Dict = {
        "frames": n_frames,
        "weights": weights,
        "weight_sum": (frame_set.weight_sum if frame_set is not None
                       else None),
        "tolerance": WEIGHT_SUM_TOLERANCE,
        "warnings": list(frame_set.warnings) if frame_set is not None else [],
        "assumptions": [],
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
    if unread is not None:
        # FRAMES COUNTED, THEIR RECORD NOT READ: the composed junction that
        # holds them does not read (`transport.record`, `junction_unread`).
        block["why"] = ("the composed junction, which holds the frames and "
                        "their weights, does not read -- no average; each "
                        "frame's curve is its own point")
        return block
    if frame_set is None:
        # A FAMILY: frames stating no rows (`model/structure.md` § 2.2f).
        block["why"] = ("the frames state no rows -- a family: no average; "
                        "each frame's curve is its own point")
        return block
    if weights is None:
        block["why"] = (f"the frames do not each state a {WEIGHT_ROW} -- no "
                        f"average; each frame's curve is its own point")
        return block
    block["assumptions"] = list(ASSUMPTIONS)
    from .transport.stages import voltages_once
    voltages = voltages_once((p["bias_v"] for p in list(points) + list(missing)),
                             same_volts)
    block["at"] = [at_voltage(v, n_frames, weights, points, missing,
                              same_volts) for v in voltages]
    return block


def at_voltage(v: float, n: int, weights: List[float],
                points: List[Dict], missing: List[Dict],
                same_volts: Callable[[float, float], bool]) -> Dict:
    """One voltage's entry of :func:`average_block`."""
    import numpy as np
    from .structure import frame_words
    here = {p["frame"]: p for p in points if same_volts(p["bias_v"], v)}
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
                        f"transmission at {volts_words(v)} yet: the average waits for "
                        f"every frame")
        return entry
    from .transport.energies import same_grid
    grid = here[0]["energy_ev"]
    for f in range(1, n):
        # ONE GRID, energy for energy (`energies.same_grid`): a point that
        # shares energies holds the finished point's
        # (`engines/transport.md` § 2a.16).
        if not same_grid(here[f]["energy_ev"], grid):
            raise AverageError(
                f"at {volts_words(v)} the transmission of {frame_words(f, n)} is on "
                f"another energy grid than the first frame's -- every "
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
        from .transport.conductance import frames_average
        g_avg = frames_average(weights, gs)
        entry.update(conductance_g0=g_avg, delta_conductance_g0=g_avg - gs[0])
        if gs[0]:
            entry["conductance_change_percent"] = (100.0 * (g_avg - gs[0])
                                                   / gs[0])
        else:
            entry["conductance_why"] = (f"{frame_words(0, n)}'s "
                                        f"conductance is zero: no change "
                                        f"in per cent of it")
    return entry




def _as_written(value: Any) -> str:
    """A row's value as `summarize` prints it: as written -- a whole number
    whole, a number at its full precision, a name as it reads.  A float
    printed whole below 1e15, where every whole number is a float exactly
    (2^53 is 9.0e15)."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def _row_line(row: Dict[str, Any]) -> str:
    """One row in a line: its name, its value and its unit, as written -- no
    unit where it states none or the dimensionless one."""
    unit = row.get("unit")
    return (f"{row['name']} {_as_written(row.get('value'))}"
            + (f" {unit}" if unit not in (None, "", "1") else ""))


def summary_lines(avg: Dict, frame_set: Optional[Dict] = None) -> List[str]:
    """The frames' average, as `summarize` prints it: what is said of the
    weights, the weights, at each voltage the averaged conductance beside
    frame 0's -- the set's reference, its number from 1 in the line -- and
    its change in per cent, or why there is none; what it assumes, and the
    frame set's rows (``frame_set``, the record's block,
    :func:`citation_block`) each by its name, value and unit as written."""
    from .structure import frame_words
    if not avg.get("frames") or avg["frames"] < 2:
        return []
    lines = [f"  the frames' average: warning: {w}"
             for w in avg.get("warnings") or ()]
    if avg.get("why"):
        lines.append(f"  the frames' average: {avg['why']}")
    else:
        total, tol = avg.get("weight_sum"), avg.get("tolerance")
        within = (total is not None and tol is not None
                  and abs(total - 1.0) <= tol)
        lines.append(f"  the frames' average over {avg['frames']} frames, "
                     f"weights {avg['weights']} (sum {total!r}"
                     + (f", within {tol:g} of 1" if within else "") + "):")
        for e in avg.get("at", ()):
            if e.get("why"):
                lines.append(f"  {e['bias_v']:>8.3f} V  {e['why']}")
                continue
            g, d = e.get("conductance_g0"), e.get("conductance_change_percent")
            g0 = e.get("base_conductance_g0")
            lines.append(
                f"  {e['bias_v']:>8.3f} V  <G> = "
                + (f"{g:.6f} G0" if g is not None else "--")
                + f" against {frame_words(0, avg['frames'])}'s "
                + (f"{g0:.6f} G0" if g0 is not None else "--")
                + ", change "
                + (f"{d:+.4f} %" if d is not None
                   else f"-- ({e.get('conductance_why')})"))
        if avg.get("assumptions"):
            lines.append("  it assumes: " + " ".join(avg["assumptions"]))
    rows = (frame_set or {}).get("rows") or ()
    if rows:
        lines.append("  the frame set's rows, as written:")
        lines.extend(f"    {_row_line(r)}" for r in rows)
    return lines


# ----------------------------------------------------------------------- #
#  What a page is answered of the frames                                   #
# ----------------------------------------------------------------------- #

def answer(structure, said: Sequence[str] = ()) -> Dict[str, object]:
    """What a citation's frames are, as every page that shows them reads it
    (`engines/transport.md` § 3.8.1: the describe door, Task setup's echo,
    the Results tab): ``{n_frames, definition, said}`` -- the structure's
    frame count, its rows **as the set stores them**, ``None`` for a
    structure that states none, and ``said``, transport's words of the
    citation (a calculation's pair no longer as its record pinned it,
    `compose.pair_changed_said`).  The definition is ``{structure: [{name,
    value, unit}], columns: [{name, unit}], frames: [[a value per column,
    None where a frame states none] per frame], weights, weight_sum,
    warnings}`` -- the columns in the order the frames first state them,
    each its unit as the first frame stating one writes it.  No row is
    interpreted: what a row means is the set's own (its ``note``, carried
    in the record, not drawn)."""
    from .frameset import read
    said = [str(s) for s in said]
    if structure is None:
        return {"n_frames": 1, "definition": None, "said": said}
    rows = read(structure)
    if rows is None:
        return {"n_frames": structure.n_frames, "definition": None,
                "said": said}
    columns: Dict[str, Dict[str, Any]] = {}
    for frame in rows.frames:
        for r in frame:
            columns.setdefault(r["name"], {"name": r["name"],
                                           "unit": r.get("unit")})
    return {"n_frames": structure.n_frames, "said": said, "definition": {
        "structure": [{"name": r["name"], "value": r.get("value"),
                       "unit": r.get("unit")}
                      for r in rows.structure],
        "columns": list(columns.values()),
        "frames": [[rows.value(n, f) for n in columns]
                   for f in range(len(rows.frames))],
        "weights": list(rows.weights) if rows.weights is not None else None,
        "weight_sum": rows.weight_sum,
        "warnings": list(rows.warnings),
    }}


def average_said(n_frames: Optional[int], frame_set,
                 points: List[Dict], missing: List[Dict],
                 same_volts: Callable[[float, float], bool], *,
                 unread: Optional[str] = None) -> Dict:
    """:func:`average_block` as a page draws it (`engines/transport.md`
    § 3.8.1): the block, or -- the frames' transmissions on different
    energy grids (:class:`AverageError`) -- the block of no average, its
    ``why`` the refusal's words, so the Results tab's report still opens
    with every point and draws the refusal where the average would be."""
    try:
        return average_block(n_frames, frame_set, points, missing,
                             same_volts, unread=unread)
    except AverageError as exc:
        block = average_block(n_frames, None, [], [], same_volts,
                              unread=unread)
        block.update(why=str(exc), at=[])
        return block


def citation_block(structure, provenance: Optional[Dict]
                   ) -> Optional[Dict[str, Any]]:
    """The record's ``frame_set`` block (`engines/transport.md` § 2a.12): the
    frame set the calculation cites -- the pair and its files' sha256 as the
    citation pinned them (``provenance``, the slot's) and the structure's
    rows as written, each with its meaning (``note``).  ``None`` where the
    citation is no frame set -- one structure, or nothing composed yet."""
    from .frameset import read
    if structure is None or structure.n_frames < 2:
        return None
    prov = provenance or {}
    rows = read(structure)
    return {"citation": prov.get("citation"),
            "files": dict(prov.get("files") or {}),
            "rows": list(rows.structure) if rows is not None else []}


# ----------------------------------------------------------------------- #
#  The data file's frame part                                              #
# ----------------------------------------------------------------------- #

def _or_nan(value) -> float:
    """``value`` as a number for a variable, NaN -- the fill value -- where
    it is none or not a number."""
    v = finite_number(value)
    return float("nan") if v is None else v


def data_file_variables() -> Tuple[Any, ...]:
    """The data file's frame part, as its variables (`ncdata.Var`): each
    frame's ``weight``, a label beside ``frame``; the weights' sum and
    tolerance; and the frames' average at each voltage.  Every other row
    of the frame set is its maker's, carried in ``/structure``'s
    ``structure_json`` as the structure carries it (`engines/transport.md`
    § 2a.12)."""
    from .ncdata import Var
    return (
        Var(WEIGHT_ROW, ("frame",), "given", "1", "the frame's weight",
            "the frame's share of the frames' average, as the frame set's "
            "maker states it (model/structure.md 2.2f); the fill value where "
            "the frame states none", auxiliary=True),
        Var("weight_sum", (), "derived", "1", "the frames' weights, summed",
            "sum_f weight_f, checked against 1 within weight_tolerance; the "
            "fill value where the frames state no weights"),
        Var("weight_tolerance", (), "given", "1", "the weights' tolerance",
            "how far from 1 their sum may be; the fill value where the "
            "frames state no weights"),
        Var("average_transmission", ("bias_v", "energy_ev"), "derived", "1",
            "the frames' average transmission",
            "sum_f weight_f T_f(E) over every frame, at its stated weight",
            key="transmission"),
        Var("average_delta_transmission", ("bias_v", "energy_ev"), "derived",
            "1", "the average's change",
            "<T(E)> - T_0(E), from frame 0's, <T(E)> as "
            "average_transmission is taken", key="delta_transmission"),
        Var("average_conductance_g0", ("bias_v",), "derived", "G0",
            "the frames' averaged conductance",
            "sum_f weight_f G_f / G0, at the weights average_transmission "
            "is taken at", key="conductance_g0"),
        Var("average_conductance_change_percent", ("bias_v",), "derived",
            "%", "the averaged conductance's change",
            "100 (<G> - G_f0) / G_f0, G_f0 frame 0's conductance, <G> as "
            "average_conductance_g0 is taken",
            key="conductance_change_percent"),
        Var("average_why", ("bias_v",), "state", "",
            "why the frames' average is none",
            "why there is no average at this voltage: one structure, "
            "frames that state no rows (a family) or not every one a "
            "weight -- the same at every voltage -- or the frames whose "
            "point at this voltage is not done yet",
            dtype=str, key="why"),
        Var("average_conductance_why", ("bias_v",), "state", "",
            "why the averaged conductance or its change is none",
            "the window not straddling E_F, or frame 0's conductance zero",
            dtype=str, key="conductance_why"),
    )


#: WHAT EACH VOLTAGE'S ENTRY OF THE AVERAGE FILLS (:func:`at_voltage`): the
#: variables of :func:`data_file_variables` along the voltage axis, each
#: value at its record key (`ncdata.Var.key`).
AT_VOLTAGE = tuple(v.name for v in data_file_variables()
                   if "bias_v" in v.dims)

#: THE FRAME PART'S VARIABLES OF THE SUMMARY -- the frames' average at each
#: voltage and the weights' sum and tolerance -- across the points, where a
#: grouped data file lays it (`engines/transport.md` § 2a.12, *The
#: calculation as groups*); each frame's ``weight`` is the junction's, a
#: label beside ``frame`` wherever that axis runs.
SUMMARY_PART = AT_VOLTAGE + tuple(v.name for v in data_file_variables()
                                  if not v.dims)


#: THE FRAME PART'S ATTRIBUTES THAT HOLD RECORDS (:func:`fill_data_file`
#: below) -- written as JSON text and read back as the blocks they hold by
#: the data file's one list of them (`transport.datafile.RECORD_ATTRS`).
RECORD_ATTRS = ("frame_warnings", "average_assumptions")


def fill_data_file(values: Dict[str, Any], attrs: Dict[str, Any],
                   structure, average: Dict, biases: List[float],
                   index: Callable[[float, List[float]], int]) -> None:
    """Fill the data file's frame part, in place: each frame's ``weight``,
    the fill value where a frame states none; the frames' average at each
    voltage and why it is none there (``index`` placing a voltage on
    ``biases``); what is said of the weights, their sum and its tolerance,
    and what the average assumes."""
    from .frameset import read
    rows = read(structure) if structure is not None else None
    if rows is not None:
        for f in range(len(rows.frames)):
            values[WEIGHT_ROW][f] = _or_nan(rows.value(WEIGHT_ROW, f))
    # WHY AN AVERAGE IS NONE, at each voltage: the block's reason at every
    # one, or the voltage's own.
    if average.get("why"):
        values["average_why"][:] = str(average["why"])
    # EACH VOLTAGE'S AVERAGE, as the block states it (:data:`AT_VOLTAGE`,
    # `ncdata.copy_stated`).
    from .ncdata import copy_stated
    averaged = [v for v in data_file_variables() if v.name in AT_VOLTAGE]
    for e in average.get("at") or ():
        copy_stated(e, averaged, (index(e["bias_v"], biases),), values)
    if rows is not None:
        attrs["frame_warnings"] = list(rows.warnings)
        if rows.weights is not None:
            values["weight_sum"][...] = float(average.get("weight_sum"))
            values["weight_tolerance"][...] = float(average.get("tolerance"))
            attrs["average_assumptions"] = list(
                average.get("assumptions") or ())


def averaged_at(data_file, weights: Sequence[float]) -> Dict[str, Any]:
    """A TRANSPORT CALCULATION'S FRAMES AVERAGED AT WEIGHTS A CALLER GIVES,
    from its data file ``data_file`` (`transport.datafile.read_data_file`;
    `engines/transport.md` § 2a.12): ``weights`` one per frame, frame 0
    first; each voltage by the record's own rule (:func:`at_voltage`) -- a
    voltage whose points are not all done left out, its reason said, as
    the record's average is.  Answers ``{bias_v, energy_ev, transmission
    (voltage x energy), conductance_g0, conductance_change_percent (each
    per voltage, NaN where none), said, assumptions}`` -- transport's own
    assumptions beside the numbers, as the record's average states them
    (:data:`ASSUMPTIONS`); nothing is computed again and
    nothing is written.  Refused by name: weights not one per frame, or not
    numbers (:class:`AverageError`); a data file of another schema
    (`transport.datafile.DataFileError`)."""
    import numpy as np
    from .transport.datafile import (STRUCTURE_GROUP, SUMMARY_GROUP,
                                     read_data_file)
    try:
        given = np.asarray(weights, dtype=float).ravel()
    except (TypeError, ValueError):
        given = None
    groups = read_data_file(data_file)["groups"]
    # THE FRAMES ARE THE JUNCTION'S, the points' figures the summary's
    # (`engines/transport.md` § 2a.12, *The calculation as groups*).
    data = {**groups[STRUCTURE_GROUP]["variables"],
            **groups[SUMMARY_GROUP]["variables"]}

    def value(name):
        return np.asarray(data[name]["data"])
    n = int(value("frame").size)
    if given is None or given.size != n or not np.isfinite(given).all():
        raise AverageError(
            f"the data file holds {n} frames: the weights are one finite "
            f"number per frame, frame 0 first -- "
            + (f"a {type(weights).__name__} given" if given is None
               else f"{given.size} given" if given.size != n
               else "not every one a finite number"))
    given = [float(w) for w in given]
    bias = [float(v) for v in value("bias_v")]
    energy = [float(e) for e in value("energy_ev")]
    curves = np.asarray(value("transmission"), dtype=float)
    g = np.asarray(value("conductance_g0"), dtype=float)
    done = value("done") == 1
    # THE DONE POINTS AS THE RECORD HOLDS THEM, each on the file's one grid,
    # its conductance ``None`` where the file keeps none.
    points = [{"frame": f, "bias_v": v, "energy_ev": energy,
               "transmission": curves[f, b, :],
               "conductance_g0": (float(g[f, b]) if np.isfinite(g[f, b])
                                  else None)}
              for b, v in enumerate(bias) for f in range(n) if done[f, b]]
    said: List[str] = []
    transmission = np.full((len(bias), len(energy)), np.nan)
    conductance = np.full(len(bias), np.nan)
    change = np.full(len(bias), np.nan)
    from .transport.stages import same_volts
    for b, v in enumerate(bias):
        # EACH VOLTAGE BY THE RECORD'S RULE, with these weights: no value
        # where it has none, its reason said -- the file's own axis, so a
        # voltage is its own index.
        at = at_voltage(v, n, given, points, [], same_volts)
        for why in (at["why"], at["conductance_why"]):
            if why:
                said.append(f"at {volts_words(v)}: {why}")
        if at["transmission"] is not None:
            transmission[b] = at["transmission"]
        if at["conductance_g0"] is not None:
            conductance[b] = at["conductance_g0"]
        if at["conductance_change_percent"] is not None:
            change[b] = at["conductance_change_percent"]
    return {"bias_v": bias, "energy_ev": energy,
            "transmission": transmission, "conductance_g0": conductance,
            "conductance_change_percent": change, "said": said,
            "assumptions": list(ASSUMPTIONS)}


__all__ = ["ASSUMPTIONS", "AT_VOLTAGE", "AverageError", "RECORD_ATTRS",
           "SUMMARY_PART", "answer", "at_voltage", "average_block",
           "average_said", "averaged_at", "citation_block",
           "data_file_variables", "fill_data_file", "summary_lines"]
