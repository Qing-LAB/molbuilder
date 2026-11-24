"""A VIBRATION'S FRAME SETS, ON THE HOST -- the rows a vibration's frame set
is written in and what a page says of them, and the weights a set's rows
give for a state (`engines/vibration-frames.md` § 1, § 5.1, § 6, § 7).

MODULE  vibration_frames (L2, the host's; it imports `vibration`, the
        structure and its codec, `frameset`, the units table (`units`) --
        never transport)
ROLE    the rows' vocabulary (:data:`ROWS`, § 5.1: each name, where it
        lives, its kind, unit, meaning and page words) and what a mode's own
        set assumes (:func:`assumptions_of_a_mode`); a cited pair's page
        words and provenance (:func:`said_of`, § 6); the weights for a set's
        rows at a state (:func:`weights_for_set`, § 7)
USED-BY the frame generator (`spectra.mode_frames`: the vocabulary it
        writes in, until R3d2b2b2 retires it), the web layer (the transport blueprint's describe door and
        record answer, the build blueprint's folder answer: ``frame_set_said``),
        the post-processing layer (`postprocess.weights_for_set`,
        `postprocess.reaveraged`)

**The vocabulary is the maker's, never the calculation's** (the user,
2026-10-18: "how to interpret and present the data is entirely result tab or
some other's work"): a transport calculation over the frames reads each
frame's ``weight`` and carries every other row as written
(`model/structure.md` § 2.2f); what a row means and how a page says it is
this module's, the pages above both asking it.  Nothing here refuses a
row.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .frameset import WEIGHT_ROW, finite_number

#: Each atom's mass, the value channel a vibration's set weights its
#: coordinates by (`model/structure-annotations.md`).
MASS_CHANNEL = "mass_amu"

#: Where a set written before 2.81.0 says the modes came from:
#: ``info.vibration = {run, result_sha256}`` -- read, never written (§ 5.1).
INFO_KEY = "vibration"


@dataclass(frozen=True)
class Row:
    """One row of the vocabulary (§ 5.1): its ``name`` -- which carries its
    unit where it has one (``unit``, read from the name) -- ``where`` it
    lives (``structure`` or ``frame``), its ``kind`` (``given``,
    ``measured`` from the coordinates, ``derived`` from the others), how a
    page says it -- its ``words`` -- and what it means, ``means``: the words
    the toolkit writes as the row's ``note``; ``prose``, a row whose value
    is sentences, which a page draws on a line of its own
    (`engines/transport.md` § 3.8.1)."""
    name: str
    where: str
    kind: str
    words: str
    means: str
    prose: bool = False

    @property
    def unit(self) -> Optional[str]:
        """Its unit, the one its name carries (`units.unit_of_name`), or
        ``None`` for a number of none or a text."""
        from .units import unit_of_name
        return unit_of_name(self.name)

    @property
    def unit_spelled(self) -> str:
        """Its unit as the units table spells it: ``"1"`` for a number of
        none."""
        return self.unit or "1"

    @property
    def page_unit(self) -> Optional[str]:
        """Its unit as a page shows it, the units table's
        (`units.UNIT_ON_A_PAGE`), or ``None`` for none."""
        from .units import UNIT_ON_A_PAGE
        return UNIT_ON_A_PAGE[self.unit_spelled] or None

    @property
    def unit_written(self) -> Optional[str]:
        """What its maker writes in the row's ``unit``: blank where the name
        carries the unit, ``1`` for a number of none
        (`model/structure.md` § 2.2d) -- a text's ``1`` not kept."""
        return None if self.unit else self.unit_spelled


#: THE VOCABULARY -- `engines/vibration-frames.md` § 5.1's table, in its
#: order: the rows a vibration's frame set is written in, how a page says
#: each and what each means.
ROWS: Tuple[Row, ...] = (
    Row("vibration_run", "structure", "given", "Vibration run",
        "the vibration run the modes were read from -- its folder in the "
        "projects tree"),
    Row("vibration_result_sha256", "structure", "given", "Result sha256",
        "the sha256 of that run's result JSON"),
    Row("mass_source", "structure", "given", "Masses",
        "which masses the mass_amu channel holds, by name, and so which "
        "masses the coordinate Q is measured with: standard atomic weights, "
        "a molbuilder vibration's -- the isotope-averaged standard atomic "
        "weights (chemistry.atomic_mass) the vibration's deck stated, as its "
        "result states them; every mode vector L is normalised with them, "
        "sum_A m_A |L_A|^2 = 1"),
    Row("frequency_scale", "structure", "given", "Frequency scale",
        "the factor every mode's frequency was scaled by, each pattern kept"),
    Row("assumptions", "structure", "given", "Assumptions",
        "the maker's assumptions, in words -- what the weights' average "
        "holds under (science/vibrational-averaging.md 7, items 1, 7, 8, 9)",
        prose=True),
    Row("sampling_rule", "structure", "given", "Sampling rule",
        "the recipe, by name, that placed the frames and gave their weights: "
        "gauss-hermite, a thermal state's Gauss rule -- the frames at the "
        "nodes of an m-point Gauss-Hermite rule for the Gaussian of spread "
        "sigma in Q, so sum_f weight_f X_f is exact for a response X "
        "polynomial in Q of degree below 2m; gauss, another state's own "
        "Gauss rule; monte-carlo, snapshots of every included mode drawn "
        "from its state, each of weight 1/N"),
    Row("sampling_nodes", "structure", "given", "Nodes",
        "a mode's own set: m, the rule's number of nodes, one frame at each "
        "(3: a thermal set's Q = 0, +-sqrt(3) sigma, weighted 2/3, 1/6, "
        "1/6)"),
    Row("samples", "structure", "given", "Snapshots",
        "a snapshot set: N, the snapshots drawn -- the frames after frame 0"),
    Row("seed", "structure", "given", "Seed",
        "a snapshot set: the seed, a whole number 0 or more, each mode's "
        "streams are keyed by (with the mode's number)"),
    Row("start", "structure", "given", "First snapshot",
        "a snapshot set: the first snapshot of the seed's sequence drawn -- "
        "0, or where a set continues another"),
    Row("pooled", "structure", "given", "Pooled",
        "a pooled set: 1; each part's seed, start and samples in "
        "info.sampling"),
    Row("rng", "structure", "given", "Random generator",
        "a snapshot set: the generator and numpy's version -- the same seed "
        "gives the same frames on the same numpy"),
    Row("antithetic", "structure", "given", "Mirrored pairs",
        "a snapshot set: 1 when each snapshot is followed by its mirror "
        "(frames 2j-1 and 2j a pair), 0 when every snapshot is drawn alone"),
    Row("modes", "structure", "given", "Modes",
        "a snapshot set: the modes included, by number"),
    Row("set_apart_modes", "structure", "given", "Modes set apart",
        "a snapshot set: each mode given a state of its own, in words -- its "
        "state, its frequency where it is not the result's, and each drive "
        "group with its modes and lags",
        prose=True),
    Row("quantum", "structure", "given", "Quantum",
        "1 when the set's thermal motion is quantum (zero-point motion "
        "included), 0 when classical"),
    Row("mode_index_1based", "structure", "given", "Mode",
        "a mode's own set: which mode, counted as the vibration result "
        "counts it (index_1based)"),
    Row("frequency_cm1", "structure", "given", "Frequency",
        "a mode's own set: the wavenumber its frames were placed for -- the "
        "result's own, or a scaled one, the result's then in "
        "result_frequency_cm1"),
    Row("result_frequency_cm1", "structure", "given", "Result's frequency",
        "a mode's own set at a scaled frequency: the mode's wavenumber as "
        "the result states it"),
    Row("zero_point_amplitude_amu12_ang", "structure", "given",
        "Zero-point amplitude",
        "a mode's own set: the zero-point amplitude Q_zp = sqrt(hbar / 2 "
        "omega) at frequency_cm1"),
    Row("temperature_k", "structure", "given", "Temperature",
        "the temperature of the motion the frames sample: a thermal "
        "state's -- for a mode pumped at random moments its effective "
        "temperature, the one whose thermal state holds the same mean "
        "number of quanta; a warm driven mode's, the temperature of the "
        "environment whose thermal width the swing carries; a snapshot "
        "set's thermal modes'"),
    Row("quanta", "structure", "given", "Quanta n",
        "a mode's own set of n quanta exactly: n"),
    Row("swing_quanta", "structure", "given", "Swing quanta",
        "a driven mode's set: the quanta the push adds, n_s; its swing "
        "Q_0 = 2 Q_zp sqrt(n_s)"),
    Row("mean_quanta", "structure", "derived", "Mean quanta n",
        "a mode's own set: the state's total mean number of quanta -- at a "
        "temperature n = 1 / (exp(hbar omega / k_B T) - 1); a driven "
        "state's adds its swing quanta"),
    Row("spread_amu12_ang", "structure", "derived", "Spread σ",
        "a mode's own set: the r.m.s. of the state the frames sample, "
        "sigma = Q_zp sqrt(2n + 1) for a quantum state"),
    Row("max_displacement_ang", "structure", "measured", "Outermost move",
        "a mode's own set: the outermost move -- the largest single atom's "
        "move at the outermost frames, max_A |R_A - R0_A| = |Q_out| max_A "
        "|L_A| -- the check against a bond length"),
    Row("displacement_ang", "frame", "measured", "Displacement",
        "the frame's plain displacement from frame 0, sqrt(sum_A |R_A - "
        "R0_A|^2) over every atom -- no masses"),
    Row("max_atom_displacement_ang", "frame", "measured", "Largest atom move",
        "its largest single atom's displacement from frame 0, max_A |R_A - "
        "R0_A|"),
    Row("q_amu12_ang", "frame", "derived", "Position along the mode, q",
        "a mode's own set: the frame's position along the mode, the "
        "mass-weighted normal coordinate -- every atom displaced by q L_A, "
        "sum_A m_A |L_A|^2 = 1 over the mass_amu masses; |q| = sqrt(sum_A "
        "m_A |R_A - R0_A|^2), its sign the side of frame 0 it sits on"),
    Row("node_sigma", "frame", "derived", "q in units of σ",
        "a mode's own set: the same position in units of the state's "
        "spread, q / sigma"),
    Row(WEIGHT_ROW, "frame", "given", "Weight",
        "the frame's share of the average: sum_f weight_f X_f is the "
        "recipe's estimate of the average of X over the motion the set "
        "samples -- a Gauss rule's weight at its node, 1/N for a snapshot, "
        "0 for a prepended frame 0; a share of the average, not the chance "
        "of finding the molecule at the frame"),
)

#: A NAME A SET WRITTEN BEFORE 2.81.0 GAVE A ROW, read as the row it is now
#: (§ 5.1).
_OLDER_NAMES = {"sigma_amu12_ang": "spread_amu12_ang"}

_BY_PLACE = {(r.where, r.name): r for r in ROWS}


def _row_of(name: str, where: str) -> Optional[Row]:
    """The vocabulary's row named ``name`` where it lives, an older set's
    name read as the row it is now -- or ``None``."""
    return _BY_PLACE.get((where, _OLDER_NAMES.get(name, name)))


def _speaks_for(name: str, unit: Any, where: str) -> Optional[Row]:
    """The vocabulary's row a set's row is said in: one it names where the
    row's unit is the vocabulary's **or it states none** -- the name is a
    claim about the unit (§ 5.1) -- else ``None``, the row said by its name
    and unit as written."""
    row = _row_of(name, where)
    if row is None or unit not in (None, "", row.unit_spelled):
        return None
    return row


def assumptions_of_a_mode(state_words: str) -> str:
    """The ``assumptions`` row of a mode's own set (§ 5; science § 7,
    items 1 and 7): the harmonic mode in ``state_words`` -- its state, as
    the set states it -- one mode alone, and its rule's degree as its
    only error."""
    return ("Harmonic modes: the molecule's position along the mode follows "
            f"its state's distribution -- {state_words} -- the same either "
            "side of equilibrium; a strongly anharmonic mode skews it, which "
            "a harmonic frame set does not carry.  One mode alone: every "
            "other mode stays at the relaxed geometry.  A mode's own set "
            "carries no statistical error, only its rule's degree: its "
            "weights average a response polynomial in the mode's coordinate "
            "exactly up to degree 2m - 1, m the rule's nodes.")


def assumptions_of_set(rows, info=None) -> Optional[str]:
    """What a set's average assumes, as its maker states it: its
    ``assumptions`` row; for a mode's own set written before 2.81.0 -- told
    by its own marks, the spread's older name or an ``info.vibration``
    (``info``, the structure's), a thermal set as every one the generator
    wrote then -- which states none, the harmonic assumptions in the words a
    set written now states, at its temperature (science § 7: "never
    dropped"); else ``None``, a person's set's state not known."""
    if rows is None:
        return None
    own = rows.value("assumptions")
    if own is not None:
        return str(own)
    older = (any(rows.value(name) is not None for name in _OLDER_NAMES)
             or isinstance((info or {}).get(INFO_KEY), dict))
    temperature = finite_number(rows.value("temperature_k"))
    if _kind_of(rows) == _MODE_SET and older and temperature is not None:
        return assumptions_of_a_mode(thermal_state_words(temperature))
    return None


def thermal_state_words(temperature_k: float) -> str:
    """A thermal state in the words a mode's own set's ``assumptions`` row
    says it."""
    return (f"the thermal state at {temperature_k:g} K, a Gaussian of spread "
            f"sigma = Q_zp sqrt(2n + 1)")


#: WHAT A SET IS, by the rows its maker wrote (:func:`_kind_of`).
_MODE_SET, _SNAPSHOT_SET = "mode", "snapshots"


def _kind_of(rows) -> Optional[str]:
    """What a set is, by its rows -- the one rule :func:`said_of` and
    :func:`weights_for_set` ask: a snapshot set (:data:`_SNAPSHOT_SET`) by
    its ``samples`` row or its ``monte-carlo`` rule; a mode's own set
    (:data:`_MODE_SET`) by its mode's number, or its frequency and zero-point
    amplitude; else ``None`` -- frames a person built."""
    if rows is None:
        return None
    if rows.value("samples") is not None or str(
            rows.value("sampling_rule") or "") == "monte-carlo":
        return _SNAPSHOT_SET
    if rows.value("mode_index_1based") is not None or (
            rows.value("frequency_cm1") is not None
            and rows.value("zero_point_amplitude_amu12_ang") is not None):
        return _MODE_SET
    return None


#: What a page says a set is, the words after its count (§ 6;
#: `engines/transport.md` § 3.8.1): frame 0 of a snapshot set is the relaxed
#: geometry, no snapshot.
_WHAT = {_MODE_SET: "frames of one mode",
         _SNAPSHOT_SET: "frames, the relaxed geometry and its snapshots"}


# ----------------------------------------------------------------------- #
#  What a page says of a cited set                                         #
# ----------------------------------------------------------------------- #

def _words(row: Row, note: Optional[str]) -> Dict[str, Any]:
    """How a page says a row the vocabulary speaks for: its words, page
    unit and kind, its meaning -- the set's own note, else the
    vocabulary's -- and whether it is prose."""
    return {"words": row.words, "page_unit": row.page_unit,
            "kind": row.kind, "means": note or row.means,
            "prose": row.prose}


def said_of(pair_path, recorded_files: Optional[Dict[str, str]] = None, *,
            structure=None) -> Dict[str, Any]:
    """WHAT A PAGE SAYS OF A CITED FRAME SET (§ 6; `engines/transport.md`
    § 3.8.1) -- of ``structure``, the frames the page draws (a calculation's
    composed junction, a citation as read), else of the pair at
    ``pair_path`` read now: beside the rows as a calculation carries them,
    the toolkit's words for each it speaks for
    -- ``{structure: {name: {words, page_unit, kind, means, prose}}, frames:
    {name: …}}``, a row the vocabulary names said in its words where its
    unit is the vocabulary's or it states none -- what the set is
    (``what``, the words after its count: ``frames of one mode``; ``frames,
    the relaxed geometry and its snapshots``; else ``None``), where a
    vibration's set came from (``provenance``: ``{run, result_sha256,
    stated_as, rows}`` -- its ``vibration_run`` and
    ``vibration_result_sha256`` rows, named in ``rows`` so a page says them
    once, or an older set's ``info.vibration`` -- else ``None``), what a
    mode's own set written before 2.81.0 assumes (``assumptions``:
    ``{words, value}``, the vocabulary's words for the row and
    :func:`assumptions_of_set`'s, for a set that states no row -- else
    ``None``), and what is said (``said``):
    with ``recorded_files`` -- the calculation's record's ``files``, each
    file's name and the sha256 it pinned -- a pair at ``pair_path`` either
    of whose files is not as pinned
    (`workingcopy_structure.pair_files_changed`), the frames shown being the
    ones the calculation ran; a pair that does not read, where it is read;
    the vocabulary's rows written without their meaning."""
    from .frameset import read
    from .workingcopy_structure import StructureCodec, pair_files_changed
    out: Dict[str, Any] = {"structure": {}, "frames": {}, "what": None,
                           "provenance": None, "assumptions": None,
                           "said": []}
    path = Path(os.fspath(pair_path))
    changed = (pair_files_changed(path, recorded_files)
               if recorded_files is not None else [])
    if changed:
        out["said"].append(
            f"the cited pair has changed since this calculation cited it -- "
            f"{', '.join(changed)} {'is' if len(changed) == 1 else 'are'} "
            f"not as its record pinned: the frames shown are the ones the "
            f"calculation ran; cite the pair as it is now in a new "
            f"calculation, or put it back")
    if structure is None:
        try:
            structure = StructureCodec().load(path, frames=True,
                                              said_out=[])
        except (ValueError, OSError) as exc:
            why = ("is not there" if isinstance(exc, FileNotFoundError)
                   else "does not read as a structure pair")
            out["said"].append(
                f"the cited pair {path.name} {why} now: its rows are shown "
                f"without their words or where they came from -- put it "
                f"back, or cite the pair as it is now in a new calculation")
            return out
    rows = read(structure)
    out["provenance"] = _provenance(rows, structure)
    if rows is None:
        return out
    if rows.value("assumptions") is None:
        # A MODE'S OWN SET WRITTEN BEFORE 2.81.0 states no assumptions row:
        # its harmonic picture is said beside it all the same, labelled as
        # the vocabulary labels the row.
        older = assumptions_of_set(rows, structure.info)
        if older is not None:
            row = _row_of("assumptions", "structure")
            out["assumptions"] = {"words": row.words, "value": older}
    bare: List[str] = []
    for where, written, of_frames in (("structure", [rows.structure], False),
                                      ("frames", list(rows.frames), True)):
        place = "frame" if of_frames else "structure"
        for frame in written:
            for r in frame:
                name = r["name"]
                if name in out[where]:
                    continue
                row = _speaks_for(name, r.get("unit"), place)
                if row is None:
                    continue
                note = rows.meaning(name, of_frames=of_frames)
                out[where][name] = _words(row, note)
                if note is None:
                    bare.append(row.words)
    if bare:
        one = len(bare) == 1
        out["said"].append(
            f"the set states no meaning for {', '.join(bare)}: "
            f"{'it is' if one else 'each is'} shown with the vibration "
            f"frame sets' own")
    out["what"] = _WHAT.get(_kind_of(rows))
    return out


def _provenance(rows, structure) -> Optional[Dict[str, Any]]:
    """Where a vibration's set came from: its two provenance rows, named in
    ``rows``, or an older set's ``info.vibration`` -- ``None`` where it
    states neither; ``rows`` ``None`` for a structure that states no
    rows."""
    stated = [n for n in ("vibration_run", "vibration_result_sha256")
              if rows is not None and rows.value(n) is not None]
    if stated:
        return {"run": rows.value("vibration_run"),
                "result_sha256": rows.value("vibration_result_sha256"),
                "stated_as": "its rows " + " and ".join(stated),
                "rows": stated}
    info = (structure.info or {}).get(INFO_KEY)
    if isinstance(info, dict) and (info.get("run") or info.get(
            "result_sha256")):
        return {"run": info.get("run"),
                "result_sha256": info.get("result_sha256"),
                "stated_as": f"its info.{INFO_KEY}, as an earlier "
                             f"molbuilder wrote it",
                "rows": []}
    return None


# ----------------------------------------------------------------------- #
#  The weights for a set's rows at a state                                 #
# ----------------------------------------------------------------------- #

def weights_for_set(rows, *, temperature_k: Optional[float] = None,
                    swing_quanta: Optional[float] = None,
                    quanta: Optional[int] = None,
                    mean_quanta: Optional[float] = None,
                    classical: bool = False):
    """THE WEIGHTS FOR A SET'S ROWS AT A STATE (§ 7): ``rows`` the frame
    set's rows (`frameset.FrameRows`, as `postprocess.frame_rows` reads them
    from a calculation's data file); the state named as § 2's table names it
    (`vibration.state`); each frame's coordinate ``q_amu12_ang`` and the
    mode's ``frequency_cm1`` and ``zero_point_amplitude_amu12_ang`` read
    from the rows, frame 0's included -- the weights by
    `vibration.State.weights_at` given the set's written weights, a
    `vibration.Weights`.  Said first, where it is the set's: a state that
    takes a temperature, given none, taken at the set's own
    ``temperature_k`` row, named; a swing named by its quanta alone on a set
    that states a temperature above 0 K, that temperature not used.  Each
    row it reads is taken in the unit its name carries
    (`model/structure.md` § 2.2d: a row is made so); a set written before
    2.81.1 whose row states another is read the same, unsaid.

    Refused by name (``ValueError``): what is no frame set's rows; no rows;
    a snapshot set -- its frames have no single mode's coordinate and its
    weights are equal: another motion is another set (science § 5.5); a
    set stating no mode's frequency and zero-point amplitude, or a frame
    stating no coordinate; a state that takes a temperature, given none, of
    a set that states none; and what `vibration.state` and ``weights_at``
    refuse."""
    from .frameset import FrameRows
    from .structure import frame_words
    from .vibration import (THE_SETS, _named, _quanta_words,
                            _temperature_of, state)
    if rows is not None and not isinstance(rows, FrameRows):
        raise ValueError(
            f"the weights are for a frame set's rows -- "
            f"postprocess.frame_rows(data_file) for a transport "
            f"calculation's, frameset.read(structure) for a structure's -- "
            f"not a {type(rows).__name__}")
    if rows is None:
        raise ValueError("no frame set's rows -- one structure, or frames "
                         "that state none (a family): there is no mode's "
                         "coordinate to weight them by")
    if _kind_of(rows) == _SNAPSHOT_SET:
        raise ValueError(
            "a snapshot set is not re-weighted for another motion: its "
            "frames have no single mode's coordinate and its weights are "
            "equal -- the weights another distribution would need are "
            "likelihood ratios, the frames needed growing by their mean "
            "square (science/vibrational-averaging.md 5.5); another motion "
            "is another set")
    nu = finite_number(rows.value("frequency_cm1"))
    q_zp = finite_number(rows.value("zero_point_amplitude_amu12_ang"))
    if nu is None or q_zp is None:
        raise ValueError(
            "the frames state no mode's frequency and zero-point amplitude "
            "(frequency_cm1, zero_point_amplitude_amu12_ang), which a mode's "
            "own set states: a state gives them no weights")
    n = len(rows.frames)
    q = [finite_number(rows.value("q_amu12_ang", f)) for f in range(n)]
    missing = [frame_words(f, n) for f, v in enumerate(q) if v is None]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} {'states' if len(missing) == 1 else 'state'}"
            f" no position along the mode (q_amu12_ang): every frame's "
            f"coordinate is what its weight is computed from")
    named = _named(temperature_k, mean_quanta, quanta, swing_quanta,
                   classical)
    stated = finite_number(rows.value("temperature_k"))
    t, source = _temperature_of(named, temperature_k, stated, THE_SETS)
    weights = state(q_zp, nu, temperature_k=t, mean_quanta=mean_quanta,
                    quanta=quanta, swing_quanta=swing_quanta,
                    classical=classical).weights_at(q, written=rows.weights)
    first: List[str] = []
    if source == THE_SETS:
        first.append(f"the {'classical' if classical else 'thermal'} state "
                     f"at {source} temperature, {t:g} K (its temperature_k "
                     f"row)")
    elif (stated and temperature_k is None and swing_quanta is not None):
        # SCIENCE § 6's TRAP: a warm set re-weighted to a swing named alone
        # reads the cold one -- said, and what a temperature would mean.
        first.append(
            f"{'the classical swing alone' if classical else 'the cold coherent state'}"
            f", {_quanta_words(float(swing_quanta))}: the set's temperature, "
            f"{stated:g} K, is not used -- temperature_k names the "
            f"environment whose thermal width the swing carries")
    return replace(weights, said=(*first, *weights.said)) if first else weights


__all__ = ["INFO_KEY", "MASS_CHANNEL", "ROWS", "Row",
           "assumptions_of_a_mode", "assumptions_of_set", "said_of",
           "thermal_state_words", "weights_for_set"]
