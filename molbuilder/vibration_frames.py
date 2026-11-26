"""A VIBRATION'S FRAME SETS, ON THE HOST -- a run's folder read as its
vibration, the one writer and reader of a frame set, the rows it is written
in and what a page says of them, and the weights a set's rows give for a
state (`engines/vibration-frames.md` § 1, § 4, § 5, § 5.1, § 6, § 7).

MODULE  vibration_frames (L2, the host's; it imports `vibration`, the
        structure and its codec, its sidecar's path (`sidecars.molstruct`),
        `frameset`, the units table (`units`), the masses' table's name
        (`chemistry`), the result's reader (`sidecars.spectra`) and its
        files' roles (`runfiles`), the one refusal of an unknown key
        (`persist`), the constants (`constants`), and for a run's folder the
        run door
        (`runs`, `projects`, `jobset.rung`, `identity`) and the SIESTA calls'
        names (`siesta.vibration_phases`) -- never transport)
ROLE    a vibration from either of its forms (:func:`vibration_of`) and a
        run's folder read as its vibration (:func:`vibration_of_run`, § 1);
        the one frame-set writer and its reader (:func:`write_frame_set`,
        :func:`read_frame_set`, § 5) and a mode's state's outermost move
        (:func:`outermost_move`, § 4); the doors' set -- the command line's
        and the page's -- a mode's own in any state asked every time, and
        what a state means for a mode before its set is written
        (:func:`frameset_from_mode`, :func:`readout`, § 6); the rows'
        vocabulary (:data:`ROWS`, § 5.1: each name, where it lives, its
        kind, unit, meaning and page words) and what a set assumes
        (:func:`assumptions_of_a_mode`, :func:`state_words`); a cited pair's
        page words and provenance (:func:`said_of`, § 6); the weights for a
        set's rows at a state (:func:`weights_for_set`, § 7)
USED-BY the command line (`molbuilder frameset from-mode`:
        :func:`frameset_from_mode`), the web layer (the spectrum tab's
        load door, its readout, and its write door: :func:`readout`,
        :func:`frameset_from_mode`; the transport blueprint's describe door
        and record answer, the build blueprint's folder answer:
        ``frame_set_said``), the post-processing
        layer (`postprocess.vibration`, `write_frame_set`,
        `read_frame_set`, `outermost_move`, `weights_for_set`,
        `reaveraged`)

**The vocabulary is the maker's, never the calculation's** (the user,
2026-10-18: "how to interpret and present the data is entirely result tab or
some other's work"): a transport calculation over the frames reads each
frame's ``weight`` and carries every other row as written
(`model/structure.md` § 2.2f); what a row means and how a page says it is
this module's, the pages above both asking it.  Nothing here refuses a
row it reads; the writer refuses by name what it is asked to write against
§ 5.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .frameset import WEIGHT_ROW, finite_number
from .runfiles import ROLE_SPECTRA, ROLE_SPECTRA_DATA
from .vibration import DEFAULT_NODES

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
        "holds under (science/vibrational-averaging.md 7, items 1 and 7)",
        prose=True),
    Row("sampling_rule", "structure", "given", "Sampling rule",
        "the recipe, by name, that placed the frames and gave their weights: "
        "gauss-hermite, a bell curve's Gauss rule, a thermal or classical "
        "state's -- the frames at the "
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
        "stream is keyed by (with the mode's number)"),
    Row("start", "structure", "given", "First snapshot",
        "a snapshot set: the first snapshot of the seed's sequence drawn -- "
        "0, or where a set continues another"),
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
        "state, and its frequency where it is not the result's",
        prose=True),
    Row("quantum", "structure", "given", "Quantum",
        "1 when the set's motion is quantum -- zero-point motion included "
        "-- 0 when classical: a mode's own set its state's; a snapshot set "
        "its background's, where a mode not set apart keeps it (absent "
        "where every mode is set apart), each set-apart mode's own in "
        "info.sampling"),
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
        "number of quanta; a snapshot set's background, where a mode not "
        "set apart keeps it"),
    Row("quanta", "structure", "given", "Quanta n",
        "a mode's own set of n quanta exactly: n"),
    Row("mean_quanta", "structure", "derived", "Mean quanta n",
        "a mode's own set: the state's mean number of quanta -- at a "
        "temperature n = 1 / (exp(hbar omega / k_B T) - 1)"),
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

#: THE RULES the toolkit's sets are placed by, as their ``sampling_rule``
#: row names them (§ 5.1): a bell curve's Gauss-Hermite rule -- a thermal
#: or classical state's -- another state's own Gauss rule, snapshots.
_RULE_GAUSS_HERMITE, _RULE_GAUSS, _RULE_MONTE_CARLO = (
    "gauss-hermite", "gauss", "monte-carlo")


def _kind_of(rows) -> Optional[str]:
    """What a set is, by its rows -- the one rule :func:`said_of` and
    :func:`weights_for_set` ask: a snapshot set (:data:`_SNAPSHOT_SET`) by
    its ``samples`` row or its ``monte-carlo`` rule; a mode's own set
    (:data:`_MODE_SET`) by its mode's number, or its frequency and zero-point
    amplitude; else ``None`` -- frames a person built."""
    if rows is None:
        return None
    if rows.value("samples") is not None or str(
            rows.value("sampling_rule") or "") == _RULE_MONTE_CARLO:
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
#  A run's folder, read as its vibration                                    #
# ----------------------------------------------------------------------- #

class FrameSetError(ValueError):
    """A vibration's frame set, or the vibration it is made from, cannot be
    had -- the message names what is missing and where it is read from
    (§ 1, § 5)."""


#: THE RESULT FILE'S TWO NAMES (`runfiles.WRITTEN`), the second form a
#: vibration is read from.
_RESULT_FORMS = (f"<label>{ROLE_SPECTRA}, or its data file "
                 f"<label>{ROLE_SPECTRA_DATA}")

#: THE TWO FORMS a vibration is read from (§ 1), said by every refusal of a
#: path that is neither where both are taken (:func:`vibration_of`); the run
#: door, which takes the run's folder alone, names that one.
TWO_FORMS = (f"a vibration is read from its run's folder (…/01_freq/run-0) "
             f"or its result file ({_RESULT_FORMS})")


def vibration_of(path):
    """A VIBRATION FROM EITHER OF ITS TWO FORMS (§ 1) -- the one door that
    tells them apart, which `postprocess.vibration` hands on: a run's folder
    through the run door (:func:`vibration_of_run`), a result file -- either
    of its two (`sidecars.spectra.result_role_of`) -- through the result's
    reader, ``result_sha256`` the JSON's either way, ``read_path`` the path
    as given.  A path that is neither is refused naming both
    (:data:`TWO_FORMS`): not there (`SpectraJsonNotFoundError`, a
    `FileNotFoundError`), a file of another name (`SpectraJsonError`), a
    folder that holds no run's result (:class:`FrameSetError`)."""
    from .sidecars.spectra import (SpectraJsonError, SpectraJsonNotFoundError,
                                   read_vibration_result, result_role_of,
                                   result_sha256_of)
    where = Path(os.fspath(path))
    if where.is_dir():
        return _vibration_of_run(path, both_forms=True)
    if not where.exists():
        raise SpectraJsonNotFoundError(
            f"no file or folder at {str(where)!r} -- {TWO_FORMS}")
    if result_role_of(where) is None:
        raise SpectraJsonError(f"{where.name}: not a vibration result -- "
                               f"{TWO_FORMS}")
    return read_vibration_result(path).vibration(
        read_path=os.fspath(path), result_sha256=result_sha256_of(path))


def vibration_of_run(run_folder):
    """A RUN'S FOLDER, READ AS ITS VIBRATION (§ 1) -- the one door from a
    vibration run's folder to its `vibration.Vibration`: its result read
    through the result's reader, with what only the run knows -- ``run``
    the folder, ``result_sha256`` the result JSON's, and frame 0's
    ``relaxation_calculation``, the record of the relaxation its geometry
    came from (:func:`_relaxation_record`), or ``relaxation_said``, why
    there is none.  ``read_path`` the folder, as given.

    Refused by name (:class:`FrameSetError`), naming the run's folder, the
    one form it reads -- the command line and the page ask it: no run of
    ours -- or a calculation whose description does not read, in its
    reader's words -- no result in it, and what the result's reader refuses
    (a result before schema 7 is read as that reader reads it, with the
    table's masses)."""
    return _vibration_of_run(run_folder, both_forms=False)


def _vibration_of_run(run_folder, *, both_forms: bool):
    """:func:`vibration_of_run`, its refusals' way on naming the run's
    folder alone or, where the door asking takes both (:func:`vibration_of`),
    the two forms."""
    from .runs import on_label_at, place_of, run_of
    from .sidecars.spectra import (SpectraJsonError, parse_spectra_json,
                                   result_sha256_of)
    from .siesta.vibration_phases import force_constants_call
    folder = Path(os.fspath(run_folder))
    place = place_of(folder)
    if not place.ours or place.problem:
        # WHY IT IS NO RUN, in the run door's words: a calculation whose
        # description does not read says so, naming its way out.
        raise FrameSetError(
            f"{run_folder}: its calculation's description does not read -- "
            f"{place.problem}" if place.problem else
            f"{run_folder}: no run of ours -- "
            + (TWO_FORMS if both_forms else
               "name the folder of the vibration run whose modes are read "
               "(…/01_freq/run-0)"))
    # THE RESULT, through the one door for a result on the label
    # (`runs.on_label_at`): a run's own folder, or a run of points' root,
    # where its finish writes it (`engines/vibration.md` § 5.11 § 2).
    result = on_label_at(folder, ROLE_SPECTRA)
    if result is None:
        raise FrameSetError(
            f"{run_folder}: no vibration result here "
            f"({place.task.label}{ROLE_SPECTRA}) -- a vibration run's "
            f"folder holds it once its finish has written it"
            + (f"; {TWO_FORMS}" if both_forms else ""))
    # THE RUN WHOSE DECK MEASURED THE FORCE CONSTANTS: the folder named --
    # PySCF's one run -- or, a run of points' root, its force-constant call
    # (`vibration_phases.force_constants_call`).
    run = run_of(folder) or run_of(force_constants_call(folder))
    if run is None:
        raise FrameSetError(
            f"{run_folder}: holds {result.name} but no run that measured it")
    try:
        res = parse_spectra_json(result)
    except SpectraJsonError as exc:
        raise FrameSetError(f"{result.name}: {exc}") from exc
    try:
        v = res.vibration(read_path=os.fspath(run_folder),
                          result_sha256=result_sha256_of(result))
    except ValueError as exc:
        raise FrameSetError(f"{result.name}: {exc}") from exc
    calculation, why = _relaxation_record(run, res)
    return replace(v, run=folder, relaxation_calculation=calculation,
                   relaxation_said=why)


def _relaxation_record(run, res) -> Tuple[Optional[Dict[str, Any]], str]:
    """THE RECORD OF THE RELAXATION a vibration run's geometry came from
    (§ 1, `v.relaxation_calculation`; `engines/vibration.md` § 5.10 ③) and,
    where there is none, why: the run's own `relax` call relaxed it at the
    one template's settings, so its own deck's record (`runs.view_of`); a
    structure stated already relaxed states its own -- the calculation's
    structure beside it, its ``info.calculation``.  WHETHER IT WAS STATED
    RELAXED is what the run's own result ``res`` states
    (``relaxation.already_relaxed``, written on either engine) -- never the
    description as it stands now, which may have changed since, nor a
    SIESTA deck's block, which a PySCF deck has none of."""
    from .identity import as_typed
    from .jobset.rung import structure_path
    from .runs import view_of
    from .workingcopy_structure import StructureCodec
    if not (res.relaxation or {}).get("already_relaxed"):
        own = (view_of(run).info or {}).get("calculation")
        return own, ("" if own is not None else
                     f"{run.label}'s deck records no settings (a PySCF deck "
                     f"states none), so the set states no info.calculation "
                     f"and is no transport citation")
    path = structure_path(run.place.task, run.place.root)
    stated = (StructureCodec().load(path).info or {}).get("calculation") \
        if path is not None else None
    if stated is None:
        return None, (
            f"the structure {run.label} measured was stated already relaxed "
            f"and states no record of the relaxation it came from, so the "
            f"set states no info.calculation and is no transport citation -- "
            f"save the structure from its relaxation run (the Results tab, or "
            f"`{as_typed('xv2xyz', '--from-run')}`) and run the vibration "
            f"from it")
    return stated, ""


# ----------------------------------------------------------------------- #
#  The one writer                                                          #
# ----------------------------------------------------------------------- #

@dataclass(frozen=True)
class Written:
    """What :func:`write_frame_set` wrote (§ 5): ``path`` the pair's
    ``.xyz``, ``n_frames`` its frames -- a prepended frame 0 among them --
    and ``said`` what the writer said."""
    path: Path
    n_frames: int
    said: Tuple[str, ...] = ()


#: THE WRITER'S OWN ROWS, which a person's rows may not name (§ 5): each
#: frame's weight and its measured moves; the set's provenance.
_WRITERS_FRAME_ROWS = (WEIGHT_ROW, "displacement_ang",
                       "max_atom_displacement_ang")
_WRITERS_ROWS = ("vibration_run", "vibration_result_sha256", "mass_source",
                 "frequency_scale")


def _nodes_of(frames) -> int:
    """The nodes ``m`` of a mode's own rule, from the degree it averages
    exactly, ``2m - 1`` (§ 4; the rule `vibration.State.frames` states) --
    the toolkit's turning of a degree into nodes, :func:`_degree_of` the
    reverse."""
    return int((frames.exact_to_degree + 1) // 2)


def state_words(s) -> str:
    """A mode's state (a `vibration.ModeState`) in the words a mode's own
    set's ``assumptions`` row says it (§ 5.1) -- one wording for every state
    of § 2, a thermal state's :func:`thermal_state_words`."""
    from .vibration import CLASSICAL, THERMAL, _quanta_words
    st = s.state
    t = s.temperature_k
    if st.kind == THERMAL:
        return (thermal_state_words(float(t)) if t is not None else
                f"a thermal state of "
                f"{_quanta_words(st.thermal_quanta, kind='mean ')}, a "
                f"Gaussian of spread sigma = Q_zp sqrt(2n + 1)")
    if st.kind == CLASSICAL:
        return (f"the classical bell curve at {t:g} K, a Gaussian of "
                f"spread sqrt(k_B T) / omega, no zero-point motion")
    return (f"{_quanta_words(st.quanta)} exactly, the oscillator's "
            f"standing wave |psi_n|^2")


def _keeps_background(motion) -> bool:
    """Whether an included mode keeps the motion's background -- thermal,
    or classical, at its temperature -- not set apart: the background said
    in the ``assumptions`` row, and the ``quantum`` and ``temperature_k``
    rows written (§ 5.1: "a snapshot set its background's, where a mode not
    set apart keeps it")."""
    return any(k not in motion.set_apart for k in motion.states)


def _assumptions_of_snapshots(snaps, imaginary: Sequence[int]) -> str:
    """A snapshot set's ``assumptions`` row (§ 5; science § 7, items 1 and
    7): harmonic modes, each mode's state -- the background where an
    included mode keeps it, each mode set apart in its own words -- their
    independence, every mode not included at the
    relaxed geometry, the ``imaginary`` ones (no harmonic state) named, and
    a finite set's statistical error."""
    motion = snaps.motion
    apart = motion.set_apart_modes
    if _keeps_background(motion):
        kept = (f"{'classical' if motion.classical else 'thermal'} at "
                f"{motion.temperature_k:g} K")
        states = (f"the modes not set apart {kept}; set apart: {apart}"
                  if apart else f"every included mode {kept}")
    else:
        states = f"each mode set apart: {apart}"
    every = sorted(int(k) for k in snaps.mode_index_1based) == sorted(
        m.index_1based for m in motion.vibration.modes)
    pairs = (", averaged in mirrored pairs" if snaps.antithetic else "")
    out = ("Harmonic modes: each mode's coordinate follows its state's "
           f"distribution -- {states} -- and the modes move independently"
           + ("" if every else
              "; every mode not included stays at the relaxed geometry"
              + (f" (imaginary, no harmonic state: "
                 f"{', '.join(str(k) for k in imaginary)})" if imaginary
                 else ""))
           + f".  A finite set of {snaps.samples} snapshots{pairs}: its "
           "average carries a statistical error, which the analysis of its "
           "results computes (Snapshots.standard_error).")
    return out


def _sampling_of(snaps) -> Dict[str, Any]:
    """A snapshot set's ``info.sampling`` (§ 5): everything that made its
    frames -- the modes, the background, each mode's state by its fields,
    how the modes were chosen, the modes left out, the motion's notes,
    whether its snapshots are mirrored pairs -- and each frame's coordinate
    along each mode, indexed as the set is written; read back by
    :func:`read_frame_set`."""
    from dataclasses import asdict
    motion = snaps.motion
    states = {}
    for k, s in motion.states.items():
        states[str(k)] = {
            **asdict(s.state),
            "temperature_k": (None if s.temperature_k is None
                              else float(s.temperature_k)),
            "temperature_from": s.temperature_from,
            "frequency_cm1": float(s.mode.frequency_cm1)}
    return {
        "modes": [int(k) for k in snaps.mode_index_1based],
        "temperature_k": float(motion.temperature_k),
        "temperature_from": motion.temperature_from,
        "classical": bool(motion.classical),
        "frequency_scale": float(motion.vibration.frequency_scale),
        "states": states,
        "set_apart": [int(k) for k in motion.set_apart],
        "mode_selection": list(motion.chosen_by),
        "left_out": [int(k) for k in motion.left_out],
        "notes": list(motion.notes),
        "antithetic": bool(snaps.antithetic),
        "q_amu12_ang": [[float(x) for x in row]
                        for row in snaps.q_amu12_ang]}


def _row(row: Row, value: Any) -> Dict[str, Any]:
    """A row the writer writes in the vocabulary (§ 5.1): its unit where
    the name carries none, its meaning as its note."""
    return {"name": row.name, "value": value, "unit": row.unit_written,
            "note": row.means}


def _measured(positions, f: int) -> Dict[str, float]:
    """Frame ``f``'s measured moves from frame 0, from the coordinates alone
    (§ 5.1): its plain displacement over every atom and its largest single
    atom's (`vibration._largest_of_geometries`, the one rule)."""
    import numpy as np
    from .vibration import _largest_of_geometries
    p = np.asarray(positions, dtype=float)
    moves = np.linalg.norm(p[f] - p[0], axis=1)
    return {"displacement_ang": float(np.sqrt(np.sum(moves ** 2))),
            "max_atom_displacement_ang": float(
                _largest_of_geometries(p[[0, f]], ())[0][1])}


def _checked_target(path) -> Path:
    """THE PAIR'S NAME, asked before anything is read or written (§ 5): a
    name the codec writes, and no pair already there -- a calculation
    citing it is pinned to it.  Refused by name (:class:`FrameSetError`)."""
    from .sidecars.molstruct import sidecar_path_for
    from .workingcopy_structure import StructureCodec
    out = Path(os.fspath(path))
    refused = StructureCodec.refused_name(out)
    if refused is not None:
        raise FrameSetError(refused)
    taken = [p for p in (out, sidecar_path_for(out)) if p.exists()]
    if taken:
        raise FrameSetError(
            f"{', '.join(p.name for p in taken)} already "
            f"{'exists' if len(taken) == 1 else 'exist'} -- a pair is never "
            f"written over, since a calculation citing it is pinned to it: "
            f"choose another name")
    return out


#: What a set read back without its vibration needs to be written again.
_GIVE_ITS_VIBRATION = (
    "a set read back without its vibration is written again once its "
    "vibration is given: read_frame_set(path, vibration=v)")


def write_frame_set(path, frames=None, weights=None, *, vibration=None,
                    rows=(), frame_rows=(), frame0_rows=None) -> Written:
    """THE ONE FRAME-SET WRITER (§ 5), in two forms:
    ``write_frame_set(path, frames)`` for the recipes -- ``frames`` a
    `vibration.Frames` (a mode's own set) or a `vibration.Snapshots`, which
    bring their vibration, their state and their rows -- and
    ``write_frame_set(path, positions_ang, weights, *, vibration, rows=(),
    frame_rows=(), frame0_rows=None)`` for a person's own frames: ``rows``
    and each of ``frame_rows`` (one list per frame given) rows ``{name,
    value, unit, note}`` (`model/structure.md` § 2.2d), ``frame0_rows`` the
    person's rows for a frame 0 the writer prepends.

    Frame 0 is the relaxed geometry -- a recipe's own (an odd rule's node,
    or prepended at weight 0), or a person's first frame where it stands
    within the structure's rule of two places (``SAME_PLACE_ANG``, a pair's
    six decimals), written as ``vibration.displaced({})`` exactly, else
    prepended at weight 0.  Every
    set gets each frame's measured rows and ``weight``, the provenance rows,
    the masses, frame 0's ``info.calculation``, an ``assumptions`` row (a
    recipe's; a person's own) and § 5.1's rows of what made it; a snapshot
    set its ``info.sampling``.  Through the structure's one writer; returns
    :class:`Written`.

    Refused by name (:class:`FrameSetError`): a pair already at ``path``
    (a calculation citing it is pinned to it), a name the codec does not
    write, frames of a bare state (no geometries), a set read back without
    its vibration, a person's set read back (a :class:`FrameSet`, written
    again as a person's frames), a person's frames without their weights or
    vibration, or with as many weights or frame rows as frames not, a
    person's row -- in ``rows``, ``frame_rows`` or ``frame0_rows`` -- that
    is no mapping or is named like one of the writer's own, a set of one
    frame, a recipe's frames given what it brings.
    Said, never refused: weights not summing to 1, a person's frames moving
    a held atom, a set with no record of the relaxation its geometry came
    from, a person's set stating no ``assumptions`` row, a person's frame
    row absent at a prepended frame 0."""
    import numpy as np
    from .chemistry import MASS_SOURCE
    from .frameset import read
    from .projects import address
    from .structure import (SAME_PLACE_ANG, AtomChannel, atom_words,
                            farthest_apart, frame_words)
    from .vibration import CLASSICAL, LEVEL, Frames, Snapshots, Vibration
    from .workingcopy_structure import StructureCodec

    out = _checked_target(path)
    if isinstance(frames, FrameSet):
        raise FrameSetError(
            "a person's set read back is written again as a person's "
            "frames: write_frame_set(path, fs.positions_ang, fs.weights, "
            "vibration=v, rows=..., frame_rows=...), leaving out the "
            "writer's own rows (each frame's weight and measured moves, the "
            "provenance)")
    said: List[str] = []
    stated: Dict[str, Any] = {}          # the structure's rows, by name
    per_frame: List[Dict[str, Any]] = []  # each frame's recipe rows
    sampling = None
    recipe = isinstance(frames, (Frames, Snapshots))
    if recipe:
        if weights is not None or vibration is not None or rows \
                or frame_rows or frame0_rows is not None:
            raise FrameSetError(
                "a recipe's frames bring their own vibration, weights and "
                "rows: write_frame_set(path, frames) -- a person's frames "
                "are write_frame_set(path, positions_ang, weights, *, "
                "vibration, ...)")
        if isinstance(frames, Frames):
            if frames.state is None:
                raise FrameSetError(
                    _GIVE_ITS_VIBRATION if frames.positions_ang is not None
                    else "frames of a bare state have no geometries -- make "
                         "them from a mode's state: "
                         "v.mode(7).state(...).frames(3)")
            s = frames.state
            v = s.mode._v
            positions = np.asarray(frames.positions_ang, dtype=float)
            w = np.asarray(frames.weights, dtype=float)
            q = np.asarray(frames.q_amu12_ang, dtype=float)[:, 0]
            sigma = float(s.spread_amu12_ang)
            per_frame = [{"q_amu12_ang": float(x), "node_sigma":
                          float(x) / sigma if sigma > 0.0 else 0.0}
                         for x in q]
            mode = s.mode
            st = s.state
            stated.update({
                "assumptions": assumptions_of_a_mode(state_words(s)),
                "sampling_rule": (_RULE_GAUSS if st.kind == LEVEL
                                  else _RULE_GAUSS_HERMITE),
                "sampling_nodes": _nodes_of(frames),
                "quantum": 0 if st.kind == CLASSICAL else 1,
                "mode_index_1based": int(mode.index_1based),
                "frequency_cm1": float(mode.frequency_cm1),
                "result_frequency_cm1": (
                    None if mode.result_frequency_cm1 is None
                    else float(mode.result_frequency_cm1)),
                "zero_point_amplitude_amu12_ang": float(
                    mode.zero_point_amplitude_amu12_ang),
                "temperature_k": (None if s.temperature_k is None
                                  else float(s.temperature_k)),
                "quanta": int(st.quanta) if st.kind == LEVEL else None,
                "mean_quanta": (None if s.mean_quanta is None
                                else float(s.mean_quanta)),
                "spread_amu12_ang": sigma,
                "max_displacement_ang": outermost_move(s, _nodes_of(frames))})
        else:
            snaps = frames
            motion = snaps.motion
            if motion is None:
                raise FrameSetError(_GIVE_ITS_VIBRATION)
            v = motion.vibration
            positions = np.asarray(snaps.positions_ang, dtype=float)
            w = np.asarray(snaps.weights, dtype=float)
            per_frame = [{} for _ in positions]
            sampling = _sampling_of(snaps)
            included = [int(k) for k in snaps.mode_index_1based]
            every_real = sorted(included) == sorted(
                m.index_1based for m in v.modes if not m.has_imag)
            imaginary = [m.index_1based for m in v.modes if m.has_imag
                         and m.index_1based not in included]
            stated.update({
                "assumptions": _assumptions_of_snapshots(snaps, imaginary),
                "sampling_rule": _RULE_MONTE_CARLO,
                "samples": int(snaps.samples),
                "seed": int(snaps.seed),
                "start": int(snaps.start),
                "rng": f"numpy PCG64, SeedSequence([seed, mode, 0]), "
                       f"numpy {np.__version__}",
                "antithetic": 1 if snaps.antithetic else 0,
                "modes": ("every real mode" if every_real
                          else ", ".join(str(k) for k in included)),
                "set_apart_modes": motion.set_apart_modes or None,
                "quantum": ((0 if motion.classical else 1)
                            if _keeps_background(motion) else None),
                "temperature_k": (float(motion.temperature_k)
                                  if _keeps_background(motion) else None)})
        person_rows: List[Dict[str, Any]] = []
        person_frames: List[List[Dict[str, Any]]] = [[] for _ in positions]
    else:
        if not isinstance(vibration, Vibration):
            raise FrameSetError(
                "a person's frames are written with the vibration they are "
                "of: write_frame_set(path, positions_ang, weights, "
                "vibration=v, ...)")
        if weights is None:
            raise FrameSetError("a person's frames are written with their "
                                "weights, one per frame")
        v = vibration
        given_positions = np.asarray(frames, dtype=float)
        if given_positions.ndim != 3 or given_positions.shape[1:] != (
                len(v.masses_amu), 3):
            raise FrameSetError(
                f"the frames are (frames, {len(v.masses_amu)}, 3) -- every "
                f"frame the vibration's {len(v.masses_amu)} atoms; got shape "
                f"{given_positions.shape}")
        given_w = np.asarray(weights, dtype=float).reshape(-1)
        if len(given_w) != len(given_positions):
            raise FrameSetError(
                f"{len(given_w)} weight(s) for {len(given_positions)} "
                f"frame(s): one weight per frame")
        person_frames = [list(r) for r in frame_rows] if frame_rows else [
            [] for _ in given_positions]
        if len(person_frames) != len(given_positions):
            raise FrameSetError(
                f"{len(person_frames)} frame row set(s) for "
                f"{len(given_positions)} frame(s): one set per frame")
        zero = list(frame0_rows or ())
        if not all(isinstance(r, dict) for r in [*rows, *zero]) or not all(
                isinstance(r, dict) for rs in person_frames for r in rs):
            raise FrameSetError(
                "a row is a mapping {name, value, unit?, note?} "
                "(model/structure.md 2.2d), in rows, in each frame's "
                "frame_rows and in frame0_rows")
        person_rows = [dict(r) for r in rows]
        named = sorted(
            ({r.get("name") for rs in [*person_frames, zero] for r in rs}
             & set(_WRITERS_FRAME_ROWS))
            | ({r.get("name") for r in person_rows} & set(_WRITERS_ROWS)))
        if named:
            raise FrameSetError(
                f"a row named {', '.join(named)} is the writer's own -- each "
                f"frame's weight is the weights given, its moves measured "
                f"from the coordinates, the provenance the vibration's: name "
                f"the row otherwise")
        # FRAME 0 AS THE PAIR HOLDS IT (§ 5; the user, 2026-11-24): a first
        # frame where the relaxed geometry stands, by the structure's one
        # rule of two places -- the rounding of a pair's six decimals --
        # IS frame 0, written exactly; one further off gets it prepended.
        relaxed = np.asarray(v.displaced({}), dtype=float)
        if farthest_apart(given_positions[0], relaxed)[1] <= SAME_PLACE_ANG:
            positions = np.concatenate([relaxed[None], given_positions[1:]])
            w = given_w
        else:
            positions = np.concatenate([relaxed[None], given_positions])
            w = np.concatenate([[0.0], given_w])
            zero = [dict(r) for r in zero]
            absent = sorted({r["name"] for rs in person_frames for r in rs}
                            - {r.get("name") for r in zero})
            if absent:
                said.append(
                    f"the prepended frame 0 states no "
                    f"{', '.join(absent)}: the person's frames state "
                    f"{'it' if len(absent) == 1 else 'them'} and frame 0, "
                    f"the relaxed geometry, does not -- give its values in "
                    f"frame0_rows")
            person_frames = [zero] + person_frames
        if not any(r.get("name") == "assumptions" for r in person_rows):
            said.append(
                "the set states no assumptions row: what its average holds "
                "under is not said beside it -- give one, rows=[{\"name\": "
                "\"assumptions\", \"value\": ...}]")
        per_frame = [{} for _ in positions]

    if len(positions) < 2:
        raise FrameSetError(
            "a frame set holds two frames or more (model/structure.md "
            "2.2f): one frame is the relaxed geometry alone -- the "
            "structure itself, which needs no set")
    base = v.structure
    if base is None:
        raise FrameSetError("the vibration states no structure to write the "
                            "frames of")

    # EACH FRAME'S ROWS, IN THE VOCABULARY'S ORDER (§ 5.1): the measured
    # moves, the recipe's coordinate, the weight -- the person's own first.
    sets: List[List[Dict[str, Any]]] = []
    for f in range(len(positions)):
        values = {**_measured(positions, f), **per_frame[f],
                  WEIGHT_ROW: float(w[f])}
        sets.append(list(person_frames[f]) + [
            _row(r, values[r.name]) for r in ROWS
            if r.where == "frame" and r.name in values])
    try:
        built = base.with_frames(positions, frame_rows=sets)
    except ValueError as exc:
        raise FrameSetError(str(exc)) from exc

    # WHERE THE MODES CAME FROM, AND WHAT MADE THE FRAMES (§ 5, § 5.1), in
    # the vocabulary's order, after the person's own rows.
    stated.update({
        "vibration_run": None if v.run is None else address(v.run),
        "vibration_result_sha256": v.result_sha256,
        "mass_source": MASS_SOURCE,
        "frequency_scale": (None if float(v.frequency_scale) == 1.0
                            else float(v.frequency_scale))})
    try:
        for r in person_rows:
            built.set_customized(r.get("name"), r.get("value"),
                                 unit=r.get("unit"), note=r.get("note"))
        for r in ROWS:
            if r.where == "structure" and stated.get(r.name) is not None:
                built.set_customized(**_row(r, stated[r.name]))
    except ValueError as exc:
        raise FrameSetError(str(exc)) from exc
    built.set_channel(MASS_CHANNEL, AtomChannel(
        "value", {i: float(m) for i, m in enumerate(v.masses_amu)}))
    if sampling is not None:
        built.set_info("sampling", sampling)

    # FRAME 0'S SETTINGS -- the record of the relaxation its geometry came
    # from, the one a transport citation takes its settings from (§ 1).
    if v.relaxation_calculation is not None:
        built.set_info("calculation", dict(v.relaxation_calculation))
    elif v.run is not None:
        if v.relaxation_said:
            said.append(v.relaxation_said)
    else:
        said.append(
            "the vibration was not read from its run's folder"
            + (f" (read from {Path(v.read_path).name})" if v.read_path
               else "")
            + ": the set states no vibration_run and no info.calculation, "
            "and is no transport citation -- read the run's folder, "
            "vibration(run_folder), to write one that is")

    # WHAT THE WRITER CHECKS IS WHAT IT OWNS (§ 5): the weights, in the
    # container's words, and the held atoms where frame 0 has them.
    written_rows = read(built)
    said.extend(written_rows.warnings if written_rows is not None else ())
    held = [i for i in range(len(v.masses_amu))
            if i not in set(v.free_atom_idxs)]
    if held:
        moving = [f for f in range(1, len(positions))
                  if farthest_apart(positions[f][held],
                                    positions[0][held])[1] > SAME_PLACE_ANG]
        if moving:
            said.append(
                f"{', '.join(frame_words(f, len(positions)) for f in moving)} "
                f"{'moves' if len(moving) == 1 else 'move'} a held atom "
                f"({atom_words(held, v.element_symbols, limit=4)} held): "
                f"written as given")
    try:
        StructureCodec().write(built, out)
    except ValueError as exc:
        raise FrameSetError(str(exc)) from exc
    return Written(path=out, n_frames=int(len(positions)), said=tuple(said))


def outermost_move(s, nodes: Optional[int] = None) -> float:
    """THE OUTERMOST MOVE of a mode's state ``s`` (a `vibration.ModeState`,
    any state of § 2) for its ``nodes``-point rule, in Angstrom (§ 4): the
    largest single atom's move at the outermost frames -- the check a
    person reads against a bond length, before a set is written and on its
    ``max_displacement_ang`` row (`vibration.outermost_move_ang`, the one
    rule, its default rule where ``nodes`` is unsaid)."""
    from .vibration import outermost_move_ang
    return outermost_move_ang(
        s.state, s.mode.eigenvector_canonical_over_atoms_per_amu12, nodes)


def _mode_of(v, mode):
    """Mode ``mode`` of the vibration ``v`` by its number -- a whole number
    -- or refused naming the modes there are (:class:`FrameSetError`)."""
    try:
        return v.mode(mode)
    except ValueError as exc:
        raise FrameSetError(str(exc)) from exc


def outermost_move_of(v, mode, *, nodes: Optional[int] = None,
                      **state) -> Optional[float]:
    """The outermost move of mode ``mode`` (its number) of the vibration
    ``v`` in the state § 2's arguments ``state`` name (:func:`outermost_move`)
    -- ``None`` for an imaginary mode, which has no state; a mode the
    vibration does not hold refused by name (:class:`FrameSetError`)."""
    m = _mode_of(v, mode)
    if m.has_imag:
        return None
    try:
        return outermost_move(m.state(**state), nodes)
    except FrameSetError:
        raise
    except ValueError as exc:
        raise FrameSetError(str(exc)) from exc


# ----------------------------------------------------------------------- #
#  The doors' set -- the command line's and the page's (§ 6)              #
# ----------------------------------------------------------------------- #

#: THE FRAME COUNTS THE DOORS OFFER -- `frameset from-mode --nodes`, the
#: page's *Frames* -- a state's own default first (`vibration.DEFAULT_NODES`,
#: three), five and seven the check on it
#: (`science/vibrational-averaging.md` § 5.3).  A script names any.
OFFERED_NODES = (DEFAULT_NODES, 5, 7)

#: The doors' words for the state they ask (§ 2: asked every time).
_ASK_THE_STATE = (
    "the state the frames sample is asked every time, never assumed: a "
    "temperature (the thermal state), a mean number of quanta, or a number "
    "of quanta")

#: THE STATE'S ARGUMENTS, as the doors take them (§ 6) -- the page's ask
#: (`frameset_state`) and the write door's keys, the command line's flags
#: spelled as options (``--temperature``, ``--quanta``, …).
STATE_ASK_KEYS = ("temperature_k", "quanta", "mean_quanta")

#: Each argument in the doors' words.
_ARGUMENT_WORDS = {"temperature_k": "a temperature",
                   "mean_quanta": "a mean number of quanta",
                   "quanta": "a number of quanta"}


class StateRefused(FrameSetError):
    """The state the doors were asked names none they write -- named
    nowhere, two at once, a value that is none, an ask of another shape or
    with a key the doors do not take (§ 2, § 6) -- refused before any run is
    read: at the command line a usage error."""


def _offered(nodes) -> int:
    """``nodes`` when it is one of :data:`OFFERED_NODES`, refused by name
    otherwise (:class:`FrameSetError`)."""
    if nodes not in OFFERED_NODES:
        raise FrameSetError(
            f"the frame set is written at "
            f"{', '.join(str(n) for n in OFFERED_NODES[:-1])} or "
            f"{OFFERED_NODES[-1]} nodes, not {nodes}")
    return int(nodes)


def doors_state(*, temperature_k=None, quanta=None,
                mean_quanta=None) -> Dict[str, Any]:
    """THE STATE THE DOORS ASK (§ 2's table: `frameset from-mode` and
    *Write frame set…* ask it every time; § 6) -- the arguments named, as
    `vibration.Mode.state` takes them, checked before any run is read: a
    set of the state table's (`vibration.STATES`; the doors name no
    classical one) -- each value by the table's own rule
    (`vibration.checked_state`), a temperature in the doors' words.
    Refused by name (:class:`StateRefused`): none named, a set the doors do
    not take, a value that is none."""
    from .vibration import STATES, checked_state, checked_temperature_k
    named = {k: v for k, v in zip(STATE_ASK_KEYS, (temperature_k, quanta,
                                                   mean_quanta))
             if v is not None}
    if not named:
        raise StateRefused(_ASK_THE_STATE)
    if frozenset(named) not in {frozenset(names) for _, names in STATES}:
        raise StateRefused(
            f"{' and '.join(_ARGUMENT_WORDS[k] for k in named)} name no "
            f"one state -- {_ASK_THE_STATE}")
    if "temperature_k" in named:
        try:
            named["temperature_k"] = checked_temperature_k(
                named["temperature_k"])
        except ValueError:
            raise StateRefused(
                f"the temperature is the one the frames sample -- the "
                f"junction's, or a pumped mode's effective one -- a number "
                f"of kelvin, 0 or more; {temperature_k!r} is not one"
            ) from None
    try:
        checked_state(**named)
    except ValueError as exc:
        raise StateRefused(str(exc)) from exc
    return named


def read_state_ask(raw: Any) -> Dict[str, Any]:
    """THE STATE A PAGE ASKS its readout at (`web/spectra.md` § 4.3, the
    load door's ``frameset_state``) -- ``{temperature_k?, quanta?,
    mean_quanta?}`` (:data:`STATE_ASK_KEYS`), read by the
    doors' rule (:func:`doors_state`).  Refused by name
    (:class:`StateRefused`): not that shape, a key it does not take -- with
    the key it is close to (`persist.unknown_keys`) -- or a state the doors
    do not take."""
    from .persist import unknown_keys
    if not isinstance(raw, dict):
        raise StateRefused(f"frameset_state is {{{', '.join(STATE_ASK_KEYS)}"
                           f"}}, the state's arguments; got {raw!r}")
    said = unknown_keys(raw, STATE_ASK_KEYS)
    if said:
        raise StateRefused(f"frameset_state: {said}")
    return doors_state(**raw)


def _state_of(m, state: Dict[str, Any]):
    """Mode ``m``'s state named by ``state`` (§ 2), the table's refusal
    said as :class:`FrameSetError`."""
    try:
        return m.state(**state)
    except ValueError as exc:
        raise FrameSetError(str(exc)) from exc


#: WHERE THE DOORS WARN THAT A MODE HOLDS MANY QUANTA, in eV -- the energy
#: its quanta put into it, ``<n> h c nu``: far up its ladder, where the
#: harmonic picture a set is drawn from may no longer hold (the user's
#: choice, 2026-11-24; science § 2a.2, § 7 item 1 -- a torsion can leave it
#: well below).  Said and, on the page, asked to confirm; never refused.
HARMONIC_LIMIT_EV = 1.0


def _beyond_harmonic(m, s) -> Optional[str]:
    """Words for mode ``m`` in state ``s`` where its quanta put more than
    :data:`HARMONIC_LIMIT_EV` into it, else ``None``."""
    from .constants import CM1_EV
    from .vibration import _quanta_words
    if s.mean_quanta is None:
        return None
    energy = float(s.mean_quanta) * float(m.frequency_cm1) * CM1_EV
    if energy <= HARMONIC_LIMIT_EV:
        return None
    return (f"mode {m.index_1based} holds "
            f"{_quanta_words(s.mean_quanta, spec='.4g')} in this "
            f"state, {energy:.3g} eV -- more than {HARMONIC_LIMIT_EV:g} eV, "
            f"far up its ladder, where the harmonic picture the frames are "
            f"drawn from may no longer hold")


def readout(m, state: Dict[str, Any],
            node_counts=OFFERED_NODES) -> Optional[Dict[str, Any]]:
    """WHAT A STATE MEANS FOR MODE ``m`` BEFORE ITS SET IS WRITTEN
    (`web/spectra.md` § 4.3) -- the page's readout under the dialog, the
    load door's ``frameset`` block: ``{mean_quanta, spread_amu12_ang,
    spread_over_zero_point, outermost_move_ang: {"3": …, …}, state,
    beyond_harmonic}`` for the state ``state`` names (:func:`doors_state`),
    the block's ``state`` it in the set's words (:func:`state_words`),
    the outermost move for each frame count of ``node_counts``;
    ``beyond_harmonic`` the words where its quanta put more than
    :data:`HARMONIC_LIMIT_EV` into the mode, else ``None``.  ``None`` for an
    imaginary mode, which has no state."""
    if m.has_imag:
        return None
    s = _state_of(m, doors_state(**state))
    q_zp = float(m.zero_point_amplitude_amu12_ang)
    spread = float(s.spread_amu12_ang)
    return {
        "mean_quanta": float(s.mean_quanta),
        "spread_amu12_ang": spread,
        "spread_over_zero_point": spread / q_zp,
        "outermost_move_ang": {str(n): outermost_move(s, n)
                               for n in node_counts},
        "state": state_words(s),
        "beyond_harmonic": _beyond_harmonic(m, s)}


@dataclass(frozen=True)
class ModeFrames:
    """What :func:`frameset_from_mode` wrote: the pair's ``.xyz``
    (``path``), the mode's state in the set's words (``state``), its mean
    number of quanta and its spread against its zero-point amplitude
    (``spread_over_zero_point``), each frame's coordinate in units of the
    spread, largest atom move and weight (``frames``), and what was said
    (``said``): the writer's, and where the quanta are many, the doors'
    warning (:data:`HARMONIC_LIMIT_EV`)."""
    path: Path
    state: str
    mean_quanta: float
    spread_over_zero_point: float
    frames: Tuple[Dict[str, float], ...]
    said: Tuple[str, ...]

    def to_wire(self, root=None) -> Dict[str, Any]:
        """As a page is answered: the pair's path from ``root`` (the tree),
        each frame, and what is said."""
        from .projects import address
        path = (address(self.path, root) if root is not None
                else str(self.path))
        return {"path": path, "frames": [dict(f) for f in self.frames],
                "said": list(self.said)}


def frameset_from_mode(run_folder, *, mode, out, nodes=OFFERED_NODES[0],
                       temperature_k=None, quanta=None,
                       mean_quanta=None) -> ModeFrames:
    """THE COMMAND LINE'S AND THE PAGE'S FRAME SET (§ 6; the user's decision,
    `engines/vibration.md` § 6.11 § 9 item 2): vibration run
    ``run_folder``'s mode ``mode`` -- counted as its result counts it -- in
    the state the arguments name, asked every time (:func:`doors_state`),
    its frames the ``nodes``-point rule of that state (one of
    :data:`OFFERED_NODES`), written as one pair at ``out`` (its ``.xyz``) by
    the one writer (:func:`write_frame_set`), its run read by the run door
    (:func:`vibration_of_run`).

    Refused by name (:class:`FrameSetError`), the frame count, the state
    and the pair's name before the run is read: another frame count, a
    state not named, two at once or a value that is none
    (:class:`StateRefused`, :func:`doors_state`), a pair already at
    ``out``; then what the run door and the writer refuse, a mode the result
    does not hold or an imaginary one."""
    nodes = _offered(nodes)
    state = doors_state(temperature_k=temperature_k, quanta=quanta,
                        mean_quanta=mean_quanta)
    _checked_target(out)
    v = vibration_of_run(run_folder)
    m = _mode_of(v, mode)
    if m.has_imag:
        raise FrameSetError(
            f"mode {mode} is imaginary ({m.frequency_cm1:g} cm-1): it has no "
            f"amplitude, and the geometry is no minimum along it")
    s = _state_of(m, state)
    written = write_frame_set(out, s.frames(nodes))
    # WHAT IT WROTE, as the set states it -- each frame's coordinate in
    # units of the spread, largest move and weight, its rows -- and what the
    # state means, the readout's (one place each).
    from .frameset import read
    from .workingcopy_structure import StructureCodec
    rows = read(StructureCodec().load(written.path, frames=True))
    frames = tuple({"q_sigma": float(rows.value("node_sigma", f)),
                    "max_atom_displacement_ang": float(
                        rows.value("max_atom_displacement_ang", f)),
                    "weight": float(w)}
                   for f, w in enumerate(rows.weights))
    said = readout(m, state, node_counts=(nodes,))
    return ModeFrames(
        path=written.path, state=state_words(s),
        mean_quanta=said["mean_quanta"],
        spread_over_zero_point=said["spread_over_zero_point"],
        frames=frames,
        said=((said["beyond_harmonic"],) if said["beyond_harmonic"] else ())
        + tuple(written.said))


# ----------------------------------------------------------------------- #
#  The one reader                                                          #
# ----------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class FrameSet:
    """A frame set a person built, read back as written (§ 5): item ``i``
    frame ``i``; ``positions_ang`` (frames x atoms x 3), ``weights``
    (frames,), ``rows`` its rows as written (`frameset.FrameRows`); in the
    shape every set read back shares, ``q_amu12_ang`` (frames x 0) and
    ``mode_index_1based`` (0,) -- no mode's coordinate -- each frame's
    largest single-atom move from frame 0 and the mode moving it (0, none
    known), and ``said``."""
    positions_ang: Any
    weights: Any
    rows: Any
    said: Tuple[str, ...] = ()

    @property
    def q_amu12_ang(self):
        import numpy as np
        return np.zeros((len(self.weights), 0))

    @property
    def mode_index_1based(self):
        import numpy as np
        return np.zeros(0, dtype=int)

    @property
    def max_atom_displacement_ang(self):
        from .vibration import _largest_of_geometries
        return _largest_of_geometries(self.positions_ang, ())[0]

    @property
    def max_atom_mode_index_1based(self):
        from .vibration import _largest_of_geometries
        return _largest_of_geometries(self.positions_ang, ())[1]


def _mode_set_of(rows) -> Tuple[float, float, Dict[str, Any]]:
    """A MODE'S OWN SET'S ROWS READ AS ITS MODE AND ITS STATE -- the one
    reader :func:`weights_for_set` and :func:`read_frame_set` ask: the
    mode's ``frequency_cm1`` and ``zero_point_amplitude_amu12_ang``, and the
    state the set was written for, as § 2's arguments (``temperature_k``,
    ``mean_quanta`` for a set named by them, ``quanta``, ``classical`` for a
    set whose ``quantum`` row is 0).  Refused by name
    (``ValueError``): a set stating no mode's frequency and zero-point
    amplitude."""
    nu = finite_number(rows.value("frequency_cm1"))
    q_zp = finite_number(rows.value("zero_point_amplitude_amu12_ang"))
    if nu is None or q_zp is None:
        raise ValueError(
            "the frames state no mode's frequency and zero-point amplitude "
            "(frequency_cm1, zero_point_amplitude_amu12_ang), which a mode's "
            "own set states: a state gives them no weights")
    own: Dict[str, Any] = {}
    quanta = finite_number(rows.value("quanta"))
    temperature = finite_number(rows.value("temperature_k"))
    if quanta is not None:
        own["quanta"] = int(quanta)
    elif temperature is not None:
        own["temperature_k"] = temperature
    elif finite_number(rows.value("mean_quanta")) is not None:
        own["mean_quanta"] = finite_number(rows.value("mean_quanta"))
    if finite_number(rows.value("quantum")) == 0.0:
        own["classical"] = True
    return nu, q_zp, own


def _mode_at(v, number: int, frequency_cm1: float):
    """Mode ``number`` of ``v`` at the frequency a set states for it -- the
    result's own, or another (a measured one) -- the one place a set's
    frequency is matched to its mode's.  Exactly: the set states the
    number its mode was at, and a row's number reads back as written."""
    mode = v.mode(number)
    if float(mode.frequency_cm1) != frequency_cm1:
        mode = mode.with_frequency(frequency_cm1)
    return mode


def _at_scale(v, scale: float):
    """The vibration ``v`` at the frequency scale a set states (1 where it
    states none) -- set at that total from the result's own frequencies
    (`vibration.Vibration._at_total`, the one place), never compounded on
    ``v``'s, so the frequencies are the very numbers the set was written
    at."""
    scale = float(scale)
    return v if float(v.frequency_scale) == scale else v._at_total(scale)


def _degree_of(nodes: float) -> int:
    """The degree an ``nodes``-point rule averages exactly, ``2m - 1`` --
    :func:`_nodes_of`'s reverse (`vibration.State.frames` states it)."""
    return int(2 * nodes - 1)


def _run_of_set(provenance) -> Optional[Path]:
    """The vibration run a set names (:func:`_provenance`'s ``run`` -- its
    ``vibration_run`` row, or an older set's ``info.vibration``), as a
    folder: its address read back through the projects tree
    (`projects.at`), or the full path it was written as where it lay
    outside the tree."""
    from .projects import OutsideRoot, at
    named = (provenance or {}).get("run")
    if not named:
        return None
    try:
        return at(str(named))
    except OutsideRoot:
        return Path(str(named))


#: THE RULES the toolkit's mode sets are written by -- a set stating one of
#: them, with its nodes, is the toolkit's own (§ 4, § 5.1).
_MODE_RULES = (_RULE_GAUSS_HERMITE, _RULE_GAUSS)


def read_frame_set(path, *, vibration=None):
    """A FRAME SET READ BACK AS WRITTEN (§ 5) -- every frame, frame 0
    included, item ``i`` frame ``i``: a `vibration.Snapshots` for the
    toolkit's snapshot set (its ``info.sampling``), a `vibration.Frames` for
    the toolkit's own mode set (its ``sampling_rule`` a mode's rule with its
    ``sampling_nodes``, every frame's coordinate stated; its state rebuilt
    from its rows, of its vibration), or a :class:`FrameSet` for any other
    -- a person's, said where it names a recipe's rows.  The vibration is
    the one given, or the run the set names reopened
    (:func:`vibration_of_run`), brought to the set's frequency scale; where
    neither is had, or the run's result is not the one the set names, the
    set is read without it -- its geometries, weights and analyses whole --
    and said.

    Refused by name (:class:`FrameSetError`): a pair that does not read, a
    structure that states no frame set's rows, a vibration given that is no
    vibration or not the set's (its result's sha256 not the set's)."""
    import numpy as np
    from .frameset import read
    from .vibration import THE_SETS, Frames, Snapshots, Vibration
    from .workingcopy_structure import StructureCodec
    name = Path(os.fspath(path)).name
    try:
        struct = StructureCodec().load(path, frames=True, said_out=[])
    except (ValueError, OSError) as exc:
        raise FrameSetError(f"{name}: {exc}") from exc
    rows = read(struct)
    if rows is None:
        raise FrameSetError(f"{name} states no frame set's rows -- one "
                            f"structure, or frames that state none")
    positions = (np.asarray(struct.frames, dtype=float)
                 if struct.frames is not None
                 else np.asarray(struct.positions, dtype=float)[None])
    weights = np.asarray(rows.weights if rows.weights is not None
                         else [np.nan] * struct.n_frames, dtype=float)
    said: List[str] = list(rows.warnings)
    provenance = _provenance(rows, struct)
    stated_sha = (provenance or {}).get("result_sha256")
    v = vibration
    if v is not None:
        if not isinstance(v, Vibration):
            raise FrameSetError(f"vibration is a vibration "
                                f"(postprocess.vibration(...)), not a "
                                f"{type(v).__name__}")
        if stated_sha and v.result_sha256 and v.result_sha256 != stated_sha:
            raise FrameSetError(
                f"the vibration given is not {name}'s: its result's sha256 "
                f"is {v.result_sha256[:12]}…, the set's "
                f"{str(stated_sha)[:12]}… -- give the vibration it was "
                f"written from, or none")

    # WHAT THE SET IS: the toolkit's own by what the toolkit writes.
    sampling = (struct.info or {}).get("sampling")
    kind = _kind_of(rows)
    number = finite_number(rows.value("mode_index_1based"))
    n = struct.n_frames
    q_rows = [finite_number(rows.value("q_amu12_ang", f)) for f in range(n)]
    if kind == _SNAPSHOT_SET and not isinstance(sampling, dict):
        said.append("the set states a snapshot set's rows but no "
                    "info.sampling, which the toolkit writes: it is read as "
                    "a person's set")
        kind = None
    if kind == _MODE_SET:
        try:
            nu, _q_zp, own = _mode_set_of(rows)
        except ValueError:
            nu = None
        nodes = finite_number(rows.value("sampling_nodes"))
        if (nu is None or number is None or nodes is None
                or rows.value("sampling_rule") not in _MODE_RULES
                or any(x is None for x in q_rows)):
            said.append("the set states a mode's rows but not as the "
                        "toolkit writes its own -- its rule, nodes, mode and "
                        "every frame's coordinate: it is read as a person's "
                        "set, as written")
            kind = None
    if kind is None:
        return FrameSet(positions, weights, rows, tuple(said))

    if v is None:
        run = _run_of_set(provenance)
        if run is None:
            said.append("the set names no vibration run: it is read without "
                        "its vibration -- give one, read_frame_set(path, "
                        "vibration=v)")
        else:
            try:
                v = vibration_of_run(run)
            except FrameSetError as exc:
                said.append(f"its vibration run does not read now ({exc}): "
                            f"the set is read without it -- give one, "
                            f"read_frame_set(path, vibration=v)")
            if v is not None and stated_sha and v.result_sha256 != stated_sha:
                said.append("its vibration run's result is not the one the "
                            "set was written from (its sha256 differs): the "
                            "set is read without it -- give the vibration it "
                            "was written from, read_frame_set(path, "
                            "vibration=v)")
                v = None

    if kind == _MODE_SET:
        q = np.array([[x] for x in q_rows], dtype=float)
        state = None
        if v is not None:
            scale = finite_number(rows.value("frequency_scale")) or 1.0
            try:
                state = _mode_at(_at_scale(v, scale), int(number),
                                 nu).state(**own)
            except ValueError as exc:
                said.append(f"its rows name no state the toolkit's table "
                            f"holds ({exc}): it is read without its state")
            if state is not None and "temperature_k" in own:
                state = replace(state, temperature_from=THE_SETS)
        return Frames(q, weights, _degree_of(nodes), state=state,
                      positions_ang=positions, read_said=tuple(said),
                      read_mode=int(number))

    modes = np.array(sampling.get("modes") or [], dtype=int)
    q = np.asarray(sampling.get("q_amu12_ang") or [], dtype=float).reshape(
        n, len(modes))
    motion = _motion_of(v, sampling) if v is not None else None
    return Snapshots(motion, modes, q, bool(sampling.get("antithetic")),
                     int(rows.value("seed")), int(rows.value("start")),
                     written_positions_ang=positions, read_said=tuple(said))


def _motion_of(v, sampling):
    """The motion a snapshot set's ``info.sampling`` records, rebuilt
    whole on its vibration ``v`` at the set's frequency scale -- each mode's
    state by its fields, the motion's notes -- so a set read back draws and
    says as the one written."""
    from dataclasses import fields
    from .vibration import ModeState, Motion, State
    v = _at_scale(v, sampling.get("frequency_scale") or 1.0)
    states = {}
    for k, s in (sampling.get("states") or {}).items():
        states[int(k)] = ModeState(
            state=State(**{f.name: s[f.name] for f in fields(State)}),
            mode=_mode_at(v, int(k), float(s["frequency_cm1"])),
            temperature_k=s.get("temperature_k"),
            temperature_from=s.get("temperature_from"))
    return Motion(v, states, float(sampling["temperature_k"]),
                  str(sampling.get("temperature_from") or ""),
                  bool(sampling.get("classical")),
                  tuple(str(n) for n in sampling.get("notes") or ()),
                  tuple(int(k) for k in sampling.get("set_apart") or ()),
                  tuple(int(k) for k in sampling.get("left_out") or ()),
                  tuple(str(s) for s in sampling.get("mode_selection") or ()))


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
    ``temperature_k`` row, named.  Each
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
    from .vibration import THE_SETS, _named, _temperature_of, state
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
    nu, q_zp, _own = _mode_set_of(rows)
    n = len(rows.frames)
    q = [finite_number(rows.value("q_amu12_ang", f)) for f in range(n)]
    missing = [frame_words(f, n) for f, v in enumerate(q) if v is None]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} {'states' if len(missing) == 1 else 'state'}"
            f" no position along the mode (q_amu12_ang): every frame's "
            f"coordinate is what its weight is computed from")
    named = _named(temperature_k, mean_quanta, quanta, classical)
    stated = finite_number(rows.value("temperature_k"))
    t, source = _temperature_of(named, temperature_k, stated, THE_SETS)
    weights = state(q_zp, nu, temperature_k=t, mean_quanta=mean_quanta,
                    quanta=quanta,
                    classical=classical).weights_at(q, written=rows.weights)
    if source != THE_SETS:
        return weights
    return replace(weights, said=(
        f"the {'classical' if classical else 'thermal'} state at {source} "
        f"temperature, {t:g} K (its temperature_k row)", *weights.said))


__all__ = ["FrameSetError", "TWO_FORMS", "vibration_of", "INFO_KEY", "MASS_CHANNEL", "ROWS", "Row",
           "assumptions_of_a_mode", "assumptions_of_set", "said_of",
           "state_words", "thermal_state_words", "vibration_of_run",
           "weights_for_set", "write_frame_set", "Written", "read_frame_set",
           "FrameSet", "outermost_move", "outermost_move_of",
           "OFFERED_NODES", "STATE_ASK_KEYS", "StateRefused", "doors_state",
           "read_state_ask", "readout", "HARMONIC_LIMIT_EV", "ModeFrames",
           "frameset_from_mode"]
