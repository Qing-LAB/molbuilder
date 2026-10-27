"""THE FRAMES, POST-PROCESSED -- what a multi-frame calculation's points are
combined and read as, after the calculation has run (`engines/transport.md`
§ 2a.12; `model/structure.md` § 2.2f).

MODULE  frame_average (L2; the frames' rows (`frameset`), the transport
        stages' voltage rule, numpy)
ROLE    the post-process over a multi-frame transport calculation, kept out of
        its execution: the frames' weighted average at each voltage
        (:func:`average_block`) and its summary lines (:func:`summary_lines`),
        the frame set the record cites (:func:`citation_block`), the data
        file's frame part (:func:`data_file_variables`, :func:`with_meanings`,
        :func:`fill_data_file`), what a page is answered of the frames
        (:func:`answer`) and what is said of their rows
        (:func:`warnings_of`) -- with the interpretation a mode's frame set's
        maker writes its rows in: the vocabulary's page words and meanings
        (:data:`ROWS`) and what a harmonic mode's average assumes
        (:data:`ASSUMPTIONS`)
USED-BY the frame generator (`mode_frames`: the vocabulary it writes in),
        the transport record (`transport.record`), the data file
        (`transport.datafile`), the describe door and Task setup's folder
        answer (`web/blueprints/transport.py`, `web/blueprints/build.py`)

**Transport runs the frames and keeps every point; this combines and reads
them** (the user, 2026-10-18: "how to interpret and present the data is
entirely result tab or some other's work -- meta data is providing the
background and some parameters (such as averaging when needed) for the
calculation, it should not be deeply integrated with transport framework
itself").  The one parameter the combination takes is each frame's
``weight`` (`frameset.read`); every other row is carried as written, and a
row the vocabulary below knows is said in its words.  Nothing here refuses a
row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------- #
#  A mode's frame set's vocabulary -- how its maker's rows are said        #
# ----------------------------------------------------------------------- #

#: Each atom's mass, the value channel a mode's maker weights its coordinate
#: by (`model/structure-annotations.md`).
MASS_CHANNEL = "mass_amu"

#: Where a mode's maker says the mode came from: ``info.vibration = {run,
#: result_sha256}``.
INFO_KEY = "vibration"


@dataclass(frozen=True)
class Row:
    """One row of a mode's frame set's vocabulary: its ``name`` (its unit in
    it), ``where`` it lives (``structure`` or ``frame``), its ``kind``
    (``given``, ``measured`` from the coordinates, ``derived`` from the
    others), the ASCII spelling of its ``unit`` (``None``: a number without
    one), how a page says it -- its ``words`` and its ``page_unit`` -- and
    what it means, ``means``: the words its maker writes as the row's
    ``note``, so the set says what each number is wherever it is read
    (`model/structure.md` § 2.2f)."""
    name: str
    where: str
    kind: str
    unit: Optional[str]
    words: str
    page_unit: Optional[str]
    means: str

    @property
    def unit_spelled(self) -> str:
        """Its unit as a file spells it -- ``"1"``, the dimensionless unit,
        for a number without one: the data file's ``units``, and what a
        frame set's ``unit`` field may say (:func:`check_units`)."""
        return self.unit or "1"


#: THE VOCABULARY -- `model/structure.md` § 2.2f's table: the rows a mode's
#: frame set is written in, how a page says each, and what each means.
ROWS: Tuple[Row, ...] = (
    Row("mode_index_1based", "structure", "given", None,
        "Mode", None,
        "which mode of the vibration result, counted as the result counts "
        "them (index_1based)"),
    Row("frequency_cm1", "structure", "given", "cm-1",
        "Frequency", "cm⁻¹",
        "the mode's wavenumber, as the vibration result states it"),
    Row("temperature_k", "structure", "given", "K",
        "Temperature", "K",
        "the temperature whose thermal distribution the frames sample, "
        "where the set's maker states one -- the frame generator states "
        "none"),
    Row("zero_point_amplitude_amu12_ang", "structure", "given",
        "amu^1/2 angstrom", "Zero-point amplitude", "amu½·Å",
        "the mode's zero-point amplitude Q_zp = sqrt(hbar / 2 omega) in the "
        "mass-weighted coordinate q, as the vibration result states it"),
    Row("max_displacement_ang", "structure", "given", "angstrom",
        "Maximum displacement", "Å",
        "the largest single atom's move at the outermost frames -- the "
        "input that sets the spread, chosen by the set's maker"),
    Row("sigma_amu12_ang", "structure", "derived", "amu^1/2 angstrom",
        "Spread σ", "amu½·Å",
        "the spread sigma of the Gaussian in q the frames sample -- from "
        "the maximum displacement, sigma = q_out / (sqrt(2) x_out), q_out = "
        "max_displacement / max_A |L_A|, x_out the rule's outermost node; "
        "or, where the set states a temperature, the thermal spread "
        "Q_zp sqrt(coth(hbar omega / 2 k_B T))"),
    Row("sampling_rule", "structure", "given", None,
        "Sampling rule", None,
        "the rule, by name, that placed the frames along the mode and gave "
        "their weights.  gauss-hermite, the frame generator's: the frames "
        "sit at the nodes q_f of an n-point Gauss-Hermite rule for a "
        "Gaussian of spread sigma in q, density exp(-q^2 / 2 sigma^2) / "
        "(sigma sqrt(2 pi)), and sum_f weight_f X_f approximates that "
        "Gaussian's average of a property X -- exactly when X is a "
        "polynomial in q of degree below 2n; a response not smooth over the "
        "spread, a level crossing E_F, is averaged poorly"),
    Row("sampling_nodes", "structure", "given", None,
        "Nodes", None,
        "n, the rule's number of nodes, one frame at each -- three "
        "Gauss-Hermite nodes are q = 0, +-sqrt(3) sigma, weighted 2/3, "
        "1/6, 1/6"),
    Row("mass_source", "structure", "given", None,
        "Masses", None,
        "which masses the mass_amu channel holds, by name, and so which "
        "masses q is measured with and the mode vector L is normalised "
        "with, sum_A m_A |L_A|^2 = 1.  standard atomic weights: the "
        "isotope-averaged standard atomic weights (chemistry.atomic_mass) "
        "a molbuilder vibration's deck states, as its result states them"),
    Row("displacement_ang", "frame", "measured", "angstrom",
        "Displacement", "Å",
        "the frame's plain displacement from frame 0, sqrt(sum_A |R_A - "
        "R0_A|^2) over every atom -- no masses"),
    Row("max_atom_displacement_ang", "frame", "measured", "angstrom",
        "Largest atom move", "Å",
        "its largest single atom's displacement from frame 0, "
        "max_A |R_A - R0_A|"),
    Row("q_amu12_ang", "frame", "derived", "amu^1/2 angstrom",
        "Position along the mode, q", "amu½·Å",
        "its position along the mode, the mass-weighted normal coordinate: "
        "every atom displaced by q L_A, with sum_A m_A |L_A|^2 = 1 over the "
        "mass_amu masses; |q| = sqrt(sum_A m_A |R_A - R0_A|^2), its sign "
        "the side of frame 0 it sits on; the frame's node, sqrt(2) sigma "
        "x_f"),
    Row("node_sigma", "frame", "derived", None,
        "q in units of σ", None,
        "the same position in units of the spread, q / sigma"),
    Row("weight", "frame", "given", None,
        "Weight", None,
        "the frame's share of the mode's average: the sampling rule's "
        "weight at this frame's node, the weights summing to 1, so "
        "sum_f weight_f X_f is the rule's estimate of the Gaussian average "
        "of X"),
)
STRUCTURE_ROWS = tuple(r for r in ROWS if r.where == "structure")
FRAME_ROWS = tuple(r for r in ROWS if r.where == "frame")
_STRUCTURE = {r.name: r for r in STRUCTURE_ROWS}
_FRAME = {r.name: r for r in FRAME_ROWS}

#: What a harmonic mode's average assumes, said beside its numbers where the
#: frames state a mode (`science/vibrational-averaging.md` § 7) -- the
#: interpretation of a mode's frame set, never transport's.
ASSUMPTIONS = (
    "A harmonic mode: the molecule's position along the mode follows a "
    "Gaussian of the spread the set states, the same either side of "
    "equilibrium; a strongly anharmonic mode skews it, which a harmonic "
    "frame set does not carry.",
    "Static frames: each frame is a frozen geometry, the electron crossing "
    "fast against the vibration; the inelastic steps at eV = hbar*omega and "
    "a strongly coupled level's polaron shift need the electron-vibration "
    "coupling and are not here.",
    "One distribution, one voltage at a time: the average is taken over "
    "the frames at each voltage separately, with the same weights at every "
    "voltage; a bias that heats the mode is not modelled.",
    "Each frame's electrons in their ground state for that frame: every "
    "frame runs its own self-consistent device.",
    "The level alignment is DFT's, frame by frame.",
    "The electrodes do not move: the held atoms are the leads, shared by "
    "every frame.",
)


def warnings_of(frame_set) -> List[str]:
    """What is said of a frame set's rows, beside the frames on every reader
    -- the record, the data file, a page: what the weights' reader said
    (`frameset.read`), and the vocabulary's rows written without their
    meaning (their ``note``; `model/structure.md` § 2.2f), named in one
    line and read as written."""
    if frame_set is None:
        return []
    # BARE BY THE READERS' RULE (:func:`_note`): a frame row is read with
    # the first meaning any frame writes for it.
    framed = dict.fromkeys(r["name"] for frame in frame_set.frames
                           for r in frame)
    named = ([r["name"] for r in frame_set.structure
              if r["name"] in _STRUCTURE and not str(r.get("note") or "")]
             + [n for n in framed
                if n in _FRAME and _note(frame_set, n, "frame") is None])
    one = len(named) == 1
    return list(frame_set.warnings) + (
        [f"the frame set's {'row' if one else 'rows'} {', '.join(named)} "
         f"{'is' if one else 'are'} written without "
         f"{'its' if one else 'their'} meaning (the row's note): "
         f"{'it is' if one else 'each is'} read with the meaning "
         f"molbuilder's frame-set table gives it (model/structure.md "
         f"2.2f)"]
        if named else [])


def _note(frame_set, name: str, where: str) -> Optional[str]:
    """Row ``name``'s meaning as the set writes it -- the structure's row, or
    the first frame stating one -- or ``None`` where it writes none."""
    sets = ([frame_set.structure] if where == "structure"
            else list(frame_set.frames))
    return next((str(r["note"]) for rows in sets for r in rows
                 if r["name"] == name and str(r.get("note") or "")), None)


def _states_a_mode(frame_set) -> bool:
    """Do the frames state a mode -- the structure's ``mode_index_1based``?"""
    return frame_set is not None and \
        frame_set.value("mode_index_1based") is not None


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite number, or ``None`` -- true/false is no
    number here, though Python counts it as one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


# ----------------------------------------------------------------------- #
#  The frames' weighted average -- the record's ``average`` block          #
# ----------------------------------------------------------------------- #

class AverageError(ValueError):
    """The frames' transmissions cannot be averaged -- the message names the
    voltage and the frame, ready to surface verbatim."""


def average_block(n_frames: Optional[int], frame_set,
                  points: List[Dict], missing: List[Dict],
                  same_volts: Callable[[float, float], bool]) -> Dict:
    """The record's ``average`` block (`engines/transport.md` § 2a.12): the
    frames' rows as written (``frame_set``, `frameset.FrameRows`, ``None``
    for a structure that states none) -- the vocabulary's structure rows by
    their names, every row as written, each frame's weight, their sum, what
    is said of them -- what a mode's average assumes where the frames state
    a mode, and, at each voltage the points ran at, the average over every
    frame at its stated weight (``⟨T(E)⟩ = Σ_f W_f T_f(E)``), its change from
    the base frame's (``ΔT(E) = ⟨T(E)⟩ − T₀(E)``), and at E_F the averaged
    conductance beside the base frame's, with the change in per cent of it.
    ``n_frames`` is the composed junction's count (``None`` before its
    first prep); ``points`` are the record's done points, ``missing`` its
    pending and failed ones -- each a dict with ``frame`` and ``bias_v``;
    ``same_volts`` the stages' one voltage rule.  ``why`` says, in one
    sentence, why there is no average: nothing composed yet, one structure,
    or frames that do not each state a weight."""
    from .frameset import WEIGHT_ROW, WEIGHT_SUM_TOLERANCE
    if frame_set is not None:
        check_units(frame_set)
    weights = (list(frame_set.weights) if frame_set is not None
               and frame_set.weights is not None else None)
    # EVERY KEY, EVERY RECORD: the vocabulary's structure rows by their own
    # names, as written, ``None`` where the structure states none; and every
    # row as written beside them.
    named = {r.name: (frame_set.value(r.name) if frame_set is not None
                      else None) for r in STRUCTURE_ROWS}
    block: Dict = {
        "frames": n_frames,
        **named,
        "rows": list(frame_set.structure) if frame_set is not None else [],
        "weights": weights,
        "weight_sum": math.fsum(weights) if weights is not None else None,
        "tolerance": WEIGHT_SUM_TOLERANCE,
        "warnings": warnings_of(frame_set),
        "assumptions": (list(ASSUMPTIONS) if _states_a_mode(frame_set)
                        else []),
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
    if weights is None:
        block["why"] = (f"the frames do not each state a {WEIGHT_ROW} -- no "
                        f"average; each frame's curve is its own point "
                        f"(model/structure.md 2.2f)")
        return block
    voltages: List[float] = []
    for p in list(points) + list(missing):
        v = float(p["bias_v"])
        if not any(same_volts(v, u) for u in voltages):
            voltages.append(v)
    block["at"] = [_at_voltage(v, n_frames, weights, points, missing,
                               same_volts) for v in voltages]
    return block


def _at_voltage(v: float, n: int, weights: List[float],
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
                        f"transmission at {v:g} V yet: the average waits for "
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


def summary_lines(avg: Dict) -> List[str]:
    """The frames' average, as `summarize` prints it: what is said of the
    weights, the frames' rows -- a mode's set in its vocabulary's words --
    the weights, at each voltage the averaged conductance beside the base
    frame's and its change in per cent -- or why there is none."""
    if not avg.get("frames") or avg["frames"] < 2:
        return []
    said = [f"  the frames' average: warning: {w}"
            for w in avg.get("warnings") or ()]
    if avg.get("why"):
        return said + [f"  the frames' average: {avg['why']}"]
    # THE ROWS AS A PAGE SAYS THEM (:func:`_said`, the frame card's one
    # rule): the vocabulary's words and units in its order, any other row's
    # name and unit as written, each number to six figures as the card
    # shows it.
    rows = ", ".join(_row_words(r) for r in _in_order(avg.get("rows") or ()))
    head = (f"  the frames' average over {avg['frames']} frames"
            + (f" -- {rows}" if rows else ""))
    total, tol = avg.get("weight_sum"), avg.get("tolerance")
    within = (total is not None and tol is not None
              and abs(total - 1.0) <= tol)
    lines = said + [head + f", weights {avg['weights']} (sum {total!r}"
                    + (f", within {tol:g} of 1" if within else "") + "):"]
    for e in avg.get("at", ()):
        if e.get("why"):
            lines.append(f"  {e['bias_v']:>8.3f} V  {e['why']}")
            continue
        g, d = e.get("conductance_g0"), e.get("conductance_change_percent")
        g0 = e.get("base_conductance_g0")
        lines.append(
            f"  {e['bias_v']:>8.3f} V  <G> = "
            + (f"{g:.6f} G0" if g is not None else "--")
            + " against the base frame's "
            + (f"{g0:.6f} G0" if g0 is not None else "--")
            + ", change "
            + (f"{d:+.4f} %" if d is not None
               else f"-- ({e.get('conductance_why')})"))
    if avg.get("assumptions"):
        lines.append("  it assumes: " + " ".join(avg["assumptions"]))
    return lines


# ----------------------------------------------------------------------- #
#  What a page is answered of the frames                                   #
# ----------------------------------------------------------------------- #

def answer(structure) -> Dict[str, object]:
    """What a citation's frames are, as every page that shows them reads it
    (`engines/transport.md` § 3.8.1: the describe door, Task setup's echo):
    ``{n_frames, definition}`` -- the structure's frame count and its rows
    as a page shows them (:func:`_wire`), ``None`` for a structure that
    states none."""
    from .frameset import read
    if structure is None:
        return {"n_frames": 1, "definition": None}
    rows = read(structure)
    if rows is not None:
        check_units(rows)
    return {"n_frames": structure.n_frames,
            "definition": _wire(rows, structure) if rows is not None
            else None}


def answer_said(structure) -> Dict[str, object]:
    """:func:`answer` as every page draws it: the frames, or -- their rows
    refused (:func:`check_units`) -- the refusal in the form a citation that
    does not read takes, ``{n_frames, definition: None, why}``, its words
    the refusal's (`engines/transport.md` § 3.8.1, `web/task-setup.md`
    § 3): the page names it and still opens."""
    try:
        return answer(structure)
    except AverageError as exc:
        return {"n_frames": getattr(structure, "n_frames", None),
                "definition": None, "why": str(exc)}


def check_units(rows) -> None:
    """THE NAME CARRIES ITS UNIT (`model/structure.md` § 2.2f): a row of the
    vocabulary -- the structure's or a frame's -- written with another unit
    states a contradiction, refused by name (:class:`AverageError`) wherever
    the rows are read for a record (:func:`average_block`), a data file
    (:func:`fill_data_file`) or a page (:func:`answer`) -- never one unit on
    the page and another in the data file.  ``rows`` is
    `frameset.FrameRows`."""
    for vocabulary, written in ((_STRUCTURE, rows.structure),
                                *((_FRAME, frame) for frame in rows.frames)):
        for r in written:
            known = vocabulary.get(r["name"])
            unit = r.get("unit")
            if (known is not None and unit not in (None, "")
                    and str(unit) != known.unit_spelled):
                raise AverageError(
                    f"the frame set's row {r['name']!r} is written with the "
                    f"unit {unit!r}; the vocabulary's name means "
                    f"{known.unit_spelled!r} (model/structure.md 2.2f) -- "
                    f"write the value in that unit, or give the row another "
                    f"name")


def _said(name: str, unit: Any, note: Optional[str],
          vocabulary: Dict[str, Row], **value) -> Dict[str, Any]:
    """A row as a page says it: the vocabulary's words, kind and unit for a
    row it knows where the row lives (its unit checked first,
    :func:`check_units`), else its name and unit as written -- and what it
    means, ``means``: its ``note`` as the set writes it, else the
    vocabulary's (`model/structure.md` § 2.2f)."""
    known = vocabulary.get(name)
    if known is not None:
        return {"name": name, **value, "unit": known.unit,
                "kind": known.kind, "words": known.words,
                "page_unit": known.page_unit,
                "means": note or known.means}
    return {"name": name, **value, "unit": unit, "kind": "as written",
            "words": name, "page_unit": unit, "means": note or ""}


def _in_order(rows) -> List[Dict[str, Any]]:
    """The structure's rows in the vocabulary's order, any other row after
    them as written -- the order every surface says them in."""
    order = {r.name: i for i, r in enumerate(STRUCTURE_ROWS)}
    return sorted(rows, key=lambda r: order.get(r["name"], len(order)))


def _row_words(row: Dict[str, Any]) -> str:
    """One structure row in a line of text, as the frame card says it: its
    words, its value -- a number to six figures -- and its page unit."""
    said = _said(row["name"], row.get("unit"), None, _STRUCTURE,
                 value=row["value"])
    v = said["value"]
    text = f"{v:.6g}" if _number(v) is not None else str(v)
    return (f"{said['words']} {text}"
            + (f" {said['page_unit']}" if said.get("page_unit") else ""))


def _wire(rows, structure) -> Dict[str, Any]:
    """``{structure: [{name, value, unit, kind, words, page_unit, means}],
    columns: [{name, unit, kind, words, page_unit, means}], frames: [[a value
    per column, None where a frame states none] per frame], weights,
    weight_sum, warnings, mode}`` -- ``mode`` whether the frames state one -- and
    ``masses_amu`` and ``vibration`` where the structure carries them; the
    vocabulary's rows first, in its order."""
    srows = _in_order(rows.structure)
    stated: List[str] = []
    units: Dict[str, Any] = {}
    for frame in rows.frames:
        for r in frame:
            if r["name"] not in units:
                stated.append(r["name"])
                units[r["name"]] = r.get("unit")
    names = ([r.name for r in FRAME_ROWS if r.name in stated]
             + [n for n in stated if n not in _FRAME])
    out: Dict[str, Any] = {
        "structure": [_said(r["name"], r.get("unit"),
                            str(r.get("note") or "") or None, _STRUCTURE,
                            value=r["value"]) for r in srows],
        "columns": [_said(n, units.get(n), _note(rows, n, "frame"), _FRAME)
                    for n in names],
        "frames": [[rows.value(n, f) for n in names]
                   for f in range(len(rows.frames))],
        "weights": list(rows.weights) if rows.weights is not None else None,
        "weight_sum": rows.weight_sum,
        "warnings": warnings_of(rows),
        # WHETHER THE FRAMES STATE A MODE -- said here, so no page spells
        # the row that decides it (`transport.md` § 3.8.1: the card keeps
        # no list).
        "mode": _states_a_mode(rows),
    }
    masses = _masses(structure)
    if masses is not None:
        out["masses_amu"] = masses
    info = (structure.info or {}).get(INFO_KEY)
    if isinstance(info, dict):
        out["vibration"] = dict(info)
    return out


def citation_block(structure, provenance: Optional[Dict]
                   ) -> Optional[Dict[str, Any]]:
    """The record's ``frame_set`` block (`engines/transport.md` § 2a.12): the
    frame set the calculation cites, whole -- the pair and its files' sha256
    as the citation pinned them (``provenance``, the slot's), the
    structure's rows as written, each with its meaning (``note``), where the
    mode came from (:data:`INFO_KEY`) and every atom's mass
    (:data:`MASS_CHANNEL`) -- so the average beside it says, in the record
    itself, which rule weighted it and which masses its coordinate was
    measured with.  ``None`` where the citation is no frame set -- one
    structure, or nothing composed yet; inside the block, each key empty
    where the set states none."""
    from .frameset import read
    if structure is None or structure.n_frames < 2:
        return None
    prov = provenance or {}
    rows = read(structure)
    info = (structure.info or {}).get(INFO_KEY)
    return {"citation": prov.get("citation"),
            "files": dict(prov.get("files") or {}),
            "rows": list(rows.structure) if rows is not None else [],
            "vibration": dict(info) if isinstance(info, dict) else None,
            MASS_CHANNEL: _masses(structure)}


def _masses(structure) -> Optional[List[Any]]:
    """Every atom's value on the :data:`MASS_CHANNEL` channel, as written,
    where the structure carries it as a value channel -- else ``None``."""
    channel = (structure.annotations or {}).get(MASS_CHANNEL)
    data = getattr(channel, "data", None)
    if getattr(channel, "kind", None) != "value" or not isinstance(data, dict):
        return None
    return [data.get(i) for i in range(structure.n_atoms)]


# ----------------------------------------------------------------------- #
#  The data file's frame part                                              #
# ----------------------------------------------------------------------- #

def _or_nan(value) -> float:
    """``value`` as a number for a variable, NaN -- the fill value -- where
    it is none or not a number."""
    v = _number(value)
    return float("nan") if v is None else v


def data_file_variables(Var) -> Tuple[Any, ...]:
    """The data file's frame part, as its variables (``Var``, the data file's
    own record, `transport.datafile.Var`): along ``frame`` every per-frame
    row of the vocabulary -- its words the ``long_name``, its meaning the
    ``definition``, which the set's own meaning replaces where it writes one
    (:func:`with_meanings`) -- and every row each frame states as written
    (``frame_rows``); along ``atom`` the :data:`MASS_CHANNEL` channel; and
    the frames' average at each voltage, each naming the rule it was taken
    at."""
    return tuple(Var(r.name, ("frame",), r.kind, r.unit_spelled, r.words,
                     _defined(r.means, "model/structure.md 2.2f"),
                     auxiliary=True)
                 for r in FRAME_ROWS) + (
        Var("frame_rows", ("frame",), "given", "", "the frame's rows",
            "every row the frame states, as written: a JSON list of its rows "
            "(model/structure.md 2.2f)", dtype=str, auxiliary=True),
        Var(MASS_CHANNEL, ("atom",), "given", "amu", "the atom's mass",
            "the structure's mass_amu channel, as written -- which masses, "
            "the mass_source attribute (model/structure.md 2.2f; the fill "
            "value where it carries none)", auxiliary=True),
        Var("average_transmission", ("bias_v", "energy_ev"), "derived", "1",
            "the frames' average transmission",
            "sum_f weight_f T_f(E) over every frame, at its stated weight -- "
            "the frame set's sampling_rule at its sampling_nodes frames "
            "where it states them (the structure_rows attribute)"),
        Var("average_delta_transmission", ("bias_v", "energy_ev"), "derived",
            "1", "the average's change",
            "<T(E)> - T_0(E), from frame 0's, <T(E)> as "
            "average_transmission is taken"),
        Var("average_conductance_g0", ("bias_v",), "derived", "G0",
            "the frames' averaged conductance",
            "sum_f weight_f G_f / G0, at the weights average_transmission "
            "is taken at"),
        Var("average_conductance_change_percent", ("bias_v",), "derived",
            "%", "the averaged conductance's change",
            "100 (<G> - G_0) / G_0, in per cent of frame 0's, <G> as "
            "average_conductance_g0 is taken"),
        Var("average_why", ("bias_v",), "state", "",
            "why the frames' average is none",
            "at every voltage, why there is no average (one structure, the "
            "weights not stated); or the frames this voltage's waits for",
            dtype=str),
        Var("average_conductance_why", ("bias_v",), "state", "",
            "why the averaged conductance or its change is none",
            "the window not straddling E_F, or frame 0's conductance zero",
            dtype=str),
    )


def _defined(means: str, said_in: str) -> str:
    """A frame row's variable ``definition``: what the row means, and where
    that meaning is said -- the vocabulary's table, or the frame set."""
    return (f"{means} ({said_in}; the fill value where the frame states "
            f"none)")


def with_meanings(variables: Tuple[Any, ...], structure) -> Tuple[Any, ...]:
    """``variables`` -- the data file's table, :func:`data_file_variables`
    among them -- with each frame row's ``definition`` the meaning the
    structure's frames write for it (its ``note``, the first frame stating
    one), the vocabulary's where they write none (`model/datafiles.md` § 2:
    a cited value keeps what it means)."""
    from dataclasses import replace
    from .frameset import read
    rows = read(structure) if structure is not None else None
    if rows is None:
        return tuple(variables)
    notes = {r.name: _note(rows, r.name, "frame") for r in FRAME_ROWS}
    return tuple(replace(v, definition=_defined(notes[v.name],
                                                "as the frame set writes it"))
                 if notes.get(v.name) else v for v in variables)


def fill_data_file(values: Dict[str, Any], attrs: Dict[str, Any],
                   structure, average: Dict, biases: List[float],
                   index: Callable[[float, List[float]], int]) -> None:
    """Fill the data file's frame part, in place: each frame's rows -- the
    vocabulary's numbers in their own variables, the fill value where a
    frame states none, and every row as written in ``frame_rows`` -- the
    masses, the frames' average at each voltage and why it is none there
    (``index`` placing a voltage on ``biases``), and the attributes: the
    vocabulary's structure rows and ``info.vibration`` where stated, every
    row as written, what is said of the weights, their sum, and what the
    average assumes.  An attribute takes a number or a text: any other
    value is said as text."""
    from .frameset import read
    from .ncdata import json_attr
    rows = read(structure) if structure is not None else None
    if rows is not None:
        check_units(rows)
        masses = _masses(structure)
        if masses is not None:
            values[MASS_CHANNEL][:] = [_or_nan(m) for m in masses]
        for f, frame in enumerate(rows.frames):
            values["frame_rows"][f] = json_attr(list(frame))
            for r in FRAME_ROWS:
                values[r.name][f] = _or_nan(rows.value(r.name, f))
    # WHY AN AVERAGE IS NONE, at each voltage: the block's reason at every
    # one, or the voltage's own.
    if average.get("why"):
        values["average_why"][:] = str(average["why"])
    for e in average.get("at") or ():
        b = index(e["bias_v"], biases)
        for key, name in (("why", "average_why"),
                          ("conductance_why", "average_conductance_why")):
            if e.get(key):
                values[name][b] = str(e[key])
        for key, name in (("transmission", "average_transmission"),
                          ("delta_transmission",
                           "average_delta_transmission"),
                          ("conductance_g0", "average_conductance_g0"),
                          ("conductance_change_percent",
                           "average_conductance_change_percent")):
            if e.get(key) is not None:
                values[name][b] = e[key]
    if rows is not None:
        for r in STRUCTURE_ROWS:
            v = rows.value(r.name)
            if v is not None:
                attrs[r.name] = (v if _number(v) is not None
                                 or isinstance(v, str) else str(v))
        info = (structure.info or {}).get(INFO_KEY)
        if isinstance(info, dict):
            for k in ("run", "result_sha256"):
                attrs[f"vibration_{k}"] = str(info.get(k) or "")
        # EVERY ROW AS WRITTEN, and what is said of the weights.
        attrs["structure_rows"] = json_attr(list(rows.structure))
        attrs["frame_warnings"] = json_attr(warnings_of(rows))
        if rows.weights is not None:
            attrs["weight_sum"] = float(average.get("weight_sum"))
            attrs["weight_tolerance"] = float(average.get("tolerance"))
            attrs["average_assumptions"] = json_attr(
                list(average.get("assumptions") or ()))


__all__ = ["ASSUMPTIONS", "AverageError", "FRAME_ROWS", "INFO_KEY",
           "MASS_CHANNEL", "ROWS", "Row", "STRUCTURE_ROWS", "answer",
           "average_block", "citation_block", "data_file_variables",
           "fill_data_file", "summary_lines", "warnings_of",
           "with_meanings"]
