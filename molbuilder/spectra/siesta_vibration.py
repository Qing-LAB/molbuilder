"""A SIESTA vibration, finished by its own job: what the force-constant run left -> the result.

MODULE  spectra.siesta_vibration (L2; numpy, the standard library and its
        siblings -- nothing that needs the package installed)
ROLE    the SIESTA route's FINISH (`engines/vibration.md` § 5.5): read a
        finished force-constant run and the deck that ran it, each record
        through the reader that owns it, run the one analysis, and write
        ``<label>.spectra.json`` beside the run
USED-BY each call's run script of a SIESTA vibration's run in phases,
        which runs ``mb_vibration.pyz plan`` before SIESTA and ``record``
        after it (`engines/vibration.md` § 5.11 § 3,
        `runwrap.VIBRATION_BUNDLE`); :func:`finish` is the door for anything
        holding a finished force-constant call
TRAVELS in ``mb_vibration.pyz`` (`runwrap.VIBRATION_COMPANIONS`)

SIESTA takes no second derivative.  Its force-constant run nudges the free
atoms and writes ``<SystemLabel>.FC``; everything after that is the harmonic
analysis both engines share (`spectra.vibrational_analysis`).  A vibration
calculation ends with its result on both engines (user, 2026-09-28), so the
SIESTA job runs that analysis itself, in the attempt, with the job's own
python -- there is no step after the launch.

EVERY INPUT IS A RECORD THE ATTEMPT HOLDS, and each is read by its owner:

  * the deck (``SystemLabel``, the species, the FC range) -- the fdf
    reader, `parse.fdf`;
  * the deck's molbuilder blocks -- their one reader, `deck_record`:
    ``atom-metadata``'s order record (the deck's atoms in held-first order,
    the way back to the user order; none for a deck the sort left in the
    user order -- `model/overview.md` § 2.2a, I5), ``engine-offset`` (the axis
    kinds, R3, and the box the run had, `box_of_run`) and ``vibration`` (the stationarity criterion, the
    thermochemistry's temperature, the person's statement, the stage,
    Raman for, the molbuilder that prepared it);
  * the run's index -- its one reader, `vibration_phases.read_index`: how
    each call ended, and the `relax` call's relaxation record;
  * ``<label>.FC`` -- `parse.engines.siesta_fc`;
  * the run's output (FC step 0's geometry and forces, SIESTA's version) --
    SIESTA's one reading pass, `siesta_reader`;
  * the masses -- every atom's, as `prep` stated them in the ``vibration``
    block from `chemistry.atomic_mass` (I3): the analysis weights by them and
    the result states them.

WHICH ATOMS WERE NUDGED is the deck's ``FC.First``..``FC.Last`` -- the range
the run was told, which the ``.FC``'s own row count must agree with -- and
the held atoms are the rest: in the held-first copy (`engines/vibration.md`
§ 5.2) the free atoms are exactly that trailing range, and a deck that is
not so shaped is refused rather than read.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from ..atom_permutation import (Permutation, map_rows_to_user_order,
                                    map_to_user_index)
    from ..parse.engines.siesta_epsimg_reader import read_epsimg
    from ..constants import BOHR_ANGSTROM, HARTREE_BOHR_EV_ANGSTROM_ASE
    from ..parse.engines.siesta_fc import (ForceConstantFile,
                                           fc_block_asymmetry,
                                           hessian_from_fc, read_fc,
                                           reference_frame_of)
    from ..parse.engines.siesta_reader import SiestaReader
    from ..parse.engines._run_ending import (ending_of, not_done_why,
                                             verdict_of)
    from ..parse.fdf import (deck_atoms, fdf_key, read_deck, read_fdf,
                             system_label)
    from ..runfiles import compose
    from ..structure import Structure
    from ..parse.contract import relaxation_of_output
    from ..deck_record import (DeckRecordUnreadable, box_of_run,
                               extract_atom_metadata,
                               extract_engine_offset,
                               extract_vibration_record, order_record_of)
    from ..engine_atom_index import convert_from_engine_numbering, siesta_numbering
    from ..sidecars.spectra import dump_spectra_json
    from ..wrapper_log import (LOG_CLOCK, LOG_LINE, log_of_run,
                               read_wrapper_log)
    from .methods import (extract_citation_keys, siesta_methods_text,
                          with_siesta_raman_route)
    from .results import (FC_STEP_PATTERN, PHASE_COMPLETE,
                          PHASE_NOT_REQUESTED, PHASE_RUNNING, SpectraResults)
    from .vibrational_analysis import vibrational_analysis
    from .. import calcdirs
    from ..jobset.planned import Plan, bring
    from ..persist import sha256_of
    from .optical_data import write_optical_data
    from ..runfiles import (COORDINATES_FILE, FIRST_ATTEMPT, MONITOR_BUNDLE,
                            PSEUDO_DIR, ROLE_OPTICAL, ROLE_SPECTRA,
                            VIBRATION_BUNDLE,
                            RunNames, beside_the_deck, find, parse)
    from ..paths import frame_token, run_folder_of
    from ..runrecord import newest_run, read_continued_from, stamp
    from ..siesta.deck_lines import coordinates_text
    from ..siesta.vibration_phases import (FC, IR_CHARGES, OPTICAL, RELAX,
                                           Call,
                                           add_call, calls_of, entry_of,
                                           index_path, phase, products_of,
                                           read_index, summary, write_index)
    from .normal_modes import displaced, mode_over_atoms, q_for_largest_move
    from .selection import select_modes
except ImportError:                         # beside a job, in mb_vibration.pyz
    from siesta_epsimg_reader import read_epsimg  # type: ignore[no-redef]
    from optical_data import write_optical_data  # type: ignore[no-redef]
    from atom_permutation import (Permutation,  # type: ignore[no-redef]
                                  map_rows_to_user_order, map_to_user_index)
    from constants import BOHR_ANGSTROM, HARTREE_BOHR_EV_ANGSTROM_ASE
    from siesta_fc import (ForceConstantFile, fc_block_asymmetry,
                           hessian_from_fc, read_fc, reference_frame_of)
    from siesta_reader import SiestaReader
    from _run_ending import (ending_of,  # type: ignore[no-redef]
                             not_done_why, verdict_of)
    from fdf import (deck_atoms, fdf_key, read_deck,  # type: ignore[no-redef]
                     read_fdf, system_label)
    from runfiles import compose
    from structure import Structure
    from contract import relaxation_of_output  # type: ignore[no-redef]
    from deck_record import (DeckRecordUnreadable, box_of_run,
                             extract_atom_metadata,
                             extract_engine_offset, extract_vibration_record,
                             order_record_of)
    from engine_atom_index import convert_from_engine_numbering, siesta_numbering
    from spectra_sidecar import dump_spectra_json
    from wrapper_log import (LOG_CLOCK, LOG_LINE,  # type: ignore[no-redef]
                             log_of_run, read_wrapper_log)
    from methods import (extract_citation_keys,  # type: ignore[no-redef]
                         siesta_methods_text, with_siesta_raman_route)
    from results import (FC_STEP_PATTERN, PHASE_COMPLETE,  # type: ignore[no-redef]
                         PHASE_NOT_REQUESTED, PHASE_RUNNING, SpectraResults)
    from vibrational_analysis import vibrational_analysis
    import calcdirs
    from planned import Plan, bring
    from persist import sha256_of  # type: ignore[no-redef]
    from runfiles import (COORDINATES_FILE, FIRST_ATTEMPT,  # type: ignore[no-redef]
                          MONITOR_BUNDLE, PSEUDO_DIR, ROLE_OPTICAL,
                          ROLE_SPECTRA, VIBRATION_BUNDLE, RunNames, beside_the_deck, find,
                          parse)
    from paths import frame_token, run_folder_of  # type: ignore[no-redef]
    from runrecord import (newest_run,  # type: ignore[no-redef]
                           read_continued_from, stamp)
    from deck_lines import coordinates_text
    from vibration_phases import (FC, IR_CHARGES, OPTICAL,  # type: ignore[no-redef]
                                  RELAX, Call, add_call,
                                  calls_of, entry_of, index_path, phase,
                                  products_of, read_index, summary,
                                  write_index)
    from normal_modes import displaced, mode_over_atoms, q_for_largest_move
    from selection import select_modes

#: What the result says about how the geometry was reached: the run it
#: finishes relaxes nothing (`engines/vibration.md` § 5.5).
GEOMETRY_NOTE = ("the force-constant run relaxes nothing: its reference "
                 "geometry is taken as the stationary point")


def vibration_record(*, stage: str, force_criterion_ev_ang: Optional[float],
                     already_relaxed: bool,
                     force_criterion_from: str = "",
                     relaxation: Optional[Mapping[str, Any]],
                     temperature_K: float,
                     masses_amu: Sequence[float],
                     molbuilder_version: str,
                     raman: Optional[Mapping[str, Any]] = None,
                     relaxation_stage: Optional[str] = None,
                     title: str = "",
                     identity: Optional[Mapping[str, Any]] = None) -> dict:
    """The deck's ``vibration`` block, format v5 (`engines/vibration.md`
    § 5.3, § 5.11 § 4): what the finish reads back and no SIESTA keyword
    states.  Built by `prep`, which holds every fact -- the stage it
    prepares, the stage's resolved config (``relax_force_tol``,
    ``already_relaxed``), the thermochemistry's temperature, every atom's
    mass in the deck's order (``masses_amu``, `chemistry.atomic_mass`'s,
    I3), the molbuilder that renders the deck, **Raman for** as the optical
    phase applies it (``raman``: ``modes`` -- ``all`` or ``range`` -- and the
    range's two bounds; ``None`` with Compute Raman off -- the one fact of
    that phase no keyword states; the record step refuses a block without
    the key), and what no other block of the deck
    carries of the structure -- its ``title`` and its ``identity`` columns
    (`structure.IDENTITY_FIELDS`, in the deck's order), for the structure the
    result states (`engines/vibration.md` § 6.10.1) -- and placed in the
    deck as given (`script_emit.emit_vibration_record`).  Its keys are read
    by :func:`result_of`.  The relaxation record is the `relax` call's, read
    from the run's index at the finish, so ``relaxation`` is ``None`` from
    prep; a record given names the stage it is of."""
    if relaxation is not None and not relaxation_stage:
        # THE RECORD NAMES THE STAGE IT IS OF: the finish's remedy for a
        # reference geometry that is not stationary is chosen by it
        # (`vibrational_analysis.nonstationary_remedy`), and a record with no
        # stage would fall to the person's-statement remedy.
        raise ValueError("a relaxation record in the vibration block needs "
                         "the stage it is of (relaxation_stage)")
    return {"stage": str(stage),
            "force_criterion_ev_ang": (None if force_criterion_ev_ang is None
                                       else float(force_criterion_ev_ang)),
            # where the tolerance came from -- said, never read
            "force_criterion_from": str(force_criterion_from),
            "already_relaxed": bool(already_relaxed),
            "relaxation": (dict(relaxation) if relaxation is not None
                           else None),
            "relaxation_stage": (str(relaxation_stage)
                                 if relaxation is not None else None),
            "temperature_K": float(temperature_K),
            "masses_amu": [float(m) for m in masses_amu],
            "molbuilder_version": str(molbuilder_version),
            "raman": (dict(raman) if raman is not None else None),
            "title": str(title or ""),
            "identity": {k: (list(v) if v is not None else None)
                         for k, v in (identity or {}).items()}}


class FinishError(ValueError):
    """A force-constant run that cannot be finished -- the message names the
    file and what disagrees, ready for the session log."""


@dataclass(frozen=True)
class ForceConstantRun:
    """What a finished SIESTA force-constant run holds, every field read from
    the attempt through the reader that owns it.  Per-atom fields are in the
    deck's order -- the sorted copy's (`engines/vibration.md` § 5.2)."""
    deck: Path
    output: Path
    label: str
    #: the species label of each atom, and its atomic number
    elements: Tuple[str, ...]
    atomic_numbers: Tuple[int, ...]
    #: FC step 0 -- the undisplaced geometry and the forces there
    positions_ang: np.ndarray
    reference_forces_ev_ang: np.ndarray
    #: 0-based, in range order: ``FC.First - 1`` .. ``FC.Last - 1``
    displaced: Tuple[int, ...]
    cell: Optional[np.ndarray]
    axis_kind: Tuple[str, str, str]
    #: the deck's order record -- ``None`` for a deck in the user order
    permutation: Optional[Permutation]
    #: THE DECK'S STATEMENTS OF THE STRUCTURE but its atoms (`engines/
    #: vibration.md` § 6.10.1): its ATOM-METADATA payload (the labels, the
    #: channels, the named rows, the order record), its ENGINE-OFFSET
    #: payload (the box), and from its `vibration` block what no block
    #: carries -- the title and the identity columns.
    atom_metadata: Optional[dict]
    engine_offset: Optional[dict]
    title: str
    identity: dict
    fc: ForceConstantFile
    siesta_version: str
    #: the deck's ``vibration`` block (`script_emit.emit_vibration_record`)
    record: Mapping[str, Any]
    #: each step's partial charges as the run's output printed them -- one
    #: ``{partition: every atom's charge}`` per step, the deck's order,
    #: ``None`` for a step that printed none (`engines/vibration.md`
    #: § 5.10 ②)
    step_charges: Tuple[Optional[Mapping[str, Sequence[float]]], ...] = ()
    #: the partitions the deck asked for at every step -- its own
    #: ``Charge.*`` lines (`vibration_phases.IR_CHARGES`): none with
    #: Compute IR off.
    charges_asked: Tuple[str, ...] = ()

    @property
    def held(self) -> Tuple[int, ...]:
        """The atoms the run did not nudge -- the held set, in this order."""
        nudged = set(self.displaced)
        return tuple(i for i in range(len(self.elements)) if i not in nudged)


def read_force_constant_run(deck, output) -> ForceConstantRun:
    """Every input the finish needs, read from the attempt ``deck`` sits in
    and from ``output``, the run's own account.  Raises :class:`FinishError`
    naming the file when a record is missing or two of them disagree."""
    deck, output = Path(deck), Path(output)
    attempt = deck.parent
    # THE DECK WHOLE, its coordinates read from the file beside it
    # (`parse.fdf.read_deck`, `engines/vibration.md` § 5.11 § 4).
    text = read_deck(deck)
    label = system_label(text)
    if not label:
        raise FinishError(f"{deck.name} states no SystemLabel")
    scalars = read_fdf(text)[0]

    # THE ATOMS, as the deck wrote them -- the one reader (`parse.fdf`).
    try:
        atoms = deck_atoms(text)
    except ValueError as exc:
        raise FinishError(f"{deck.name}: {exc}") from None
    if not atoms.labels:
        raise FinishError(f"{deck.name} holds no atoms")
    atomic_numbers, elements = atoms.atomic_numbers, atoms.labels
    n = len(elements)

    # WHICH ATOMS WERE NUDGED: the range the run was told.
    try:
        first = int(scalars[fdf_key("FC.First")][0])
        last = int(scalars[fdf_key("FC.Last")][0])
    except (KeyError, IndexError, ValueError):
        raise FinishError(f"{deck.name} states no FC.First / FC.Last: it is "
                          f"not a force-constant deck") from None
    # SIESTA's atom numbers to the deck's 0-based positions, through the one
    # door for engine numbering (`model/overview.md` § 2); every check below
    # is on positions.
    lo, hi = convert_from_engine_numbering(first, "siesta"), convert_from_engine_numbering(last, "siesta")
    if not 0 <= lo <= hi < n:
        raise FinishError(f"{deck.name}: FC.First {first} .. FC.Last {last} "
                          f"is not a range over its {n} atoms")
    displaced = tuple(range(lo, hi + 1))
    if displaced[-1] != n - 1:
        raise FinishError(
            f"{deck.name}: the nudged atoms FC.First {first} .. FC.Last "
            f"{last} are not the trailing run of its {n} atoms, so the deck "
            f"was not written from the held-first copy "
            f"(engines/vibration.md 5.2)")

    try:
        offset = extract_engine_offset(text)
        record = extract_vibration_record(text)
        atom_metadata = extract_atom_metadata(text)
        order = order_record_of(atom_metadata)
    except DeckRecordUnreadable as exc:
        raise FinishError(f"{deck.name}: {exc}") from exc
    if not offset or len(offset.get("axis_kind") or ()) != 3:
        raise FinishError(
            f"{deck.name} carries no engine-offset block, so nothing says "
            f"which axes repeat: re-prep the stage")
    if record is None:
        raise FinishError(
            f"{deck.name} carries no vibration block -- the stationarity "
            f"criterion and the run's Raman for are not recorded: "
            f"re-prep the stage")
    # THE BOX THE RUN HAD -- the one rule the structure the result states is
    # built with (`deck_record.box_of_run`, the ENGINE-OFFSET block), never
    # a second read of the deck's lattice (2026-10-18).
    cell = (box_of_run(offset) or {}).get("cell")

    # THE RUN'S OWN ACCOUNT: FC step 0, and the version SIESTA states.
    reading = SiestaReader().feed_text(
        output.read_text(encoding="utf-8", errors="replace")).finish()
    step = reference_frame_of(reading, name=output.name)
    said = tuple(str(r[0]) for r in step["coords"])
    if said != elements:
        raise FinishError(
            f"{output.name} describes {len(said)} atoms in an order that is "
            f"not {deck.name}'s: the run and the deck disagree about the "
            f"atoms")
    build = reading.get("runtime_info", {}).get("siesta_build") or {}

    # THE ENGINE'S NAME ON THE LABEL, composed by the one grammar
    # (`runfiles.compose`), which travels beside the job too.
    fc_path = attempt / compose(label, ".FC")
    if not fc_path.is_file():
        raise FinishError(f"no {fc_path.name} in {attempt}: the "
                          f"force-constant run left no force constants")
    return ForceConstantRun(
        deck=deck, output=output, label=label,
        elements=elements, atomic_numbers=atomic_numbers,
        positions_ang=np.array([r[1:4] for r in step["coords"]], dtype=float),
        reference_forces_ev_ang=np.asarray(step["forces"], dtype=float),
        displaced=displaced,
        cell=(np.asarray(cell, dtype=float) if cell is not None else None),
        axis_kind=tuple(str(k) for k in offset["axis_kind"]),
        permutation=order,
        atom_metadata=atom_metadata, engine_offset=offset,
        title=str(record.get("title") or ""),
        identity=dict(record.get("identity") or {}),
        fc=read_fc(fc_path),
        siesta_version=str(build.get("version") or ""),
        record=record,
        step_charges=tuple(s.get("charges") for s in reading["steps"]),
        # WHAT THE DECK ASKED, from its own lines -- the one statement of
        # it (`engines/vibration.md` § 5.3).
        charges_asked=tuple(p for kw, value, p in IR_CHARGES
                            if (scalars.get(fdf_key(kw)) or [None])[0]
                            == value))


def equilibrium_structure(run: ForceConstantRun) -> dict:
    """THE STRUCTURE THE RESULT STATES, whole and in the user order
    (`engines/vibration.md` § 6.10.1): the force-constant reference geometry
    with the box the run had (`deck_record.box_of_run`, the run door's
    rule), the deck's labels, channels, rows and order record applied
    through the structure's one door (``Structure.apply_atom_metadata``),
    the title and the identity columns from the `vibration` block, then put
    back in the user order (``Structure.restored``).  **Checked**: the
    atoms the run did not nudge are the structure's `frozen_atoms` -- a deck
    whose two statements of the held set disagree is refused."""
    struct = Structure(
        elements=list(run.elements),
        positions=np.array(run.positions_ang, dtype=float),
        title=run.title,
        **{k: (list(v) if v is not None else None)
           for k, v in run.identity.items()})
    box = box_of_run(run.engine_offset) or {}
    struct.apply_metadata_dict({**struct.metadata_to_dict(), **box})
    struct.apply_atom_metadata(run.atom_metadata or {})
    held_label = sorted(int(i) for i in (struct.frozen_atoms or ()))
    if held_label != sorted(run.held):
        raise FinishError(
            f"{run.deck.name} states two held sets: its labels hold "
            f"{struct.atoms_named(held_label) or 'no atom'}, and its "
            f"FC.First .. FC.Last leaves {struct.atoms_named(run.held) or 'no atom'} "
            f"unnudged (engines/vibration.md 6.10.1)")
    return struct.restored().to_dict(view=False)


def partial_charges_of(run: ForceConstantRun) -> Optional[dict]:
    """THE FORCE-CONSTANT RUN'S CHARGES, as the result states them
    (`engines/vibration.md` § 5.10 ②, § 6.2): every atom's charge at every
    step, by partition, in the user order, beside each step's displaced
    atom (its user index), axis and side -- the steps SIESTA's own order
    (§ 5.4: the reference, then each nudged atom along x, y, z, side -1
    before +1).  ``None`` when the deck asked for none -- Compute IR off
    (``charges_asked``).  A partition asked that the output prints at no
    step, or at some steps and not at every one, is refused: the run and
    its charges disagree."""
    if not run.charges_asked:
        return None
    printed = {p for s in run.step_charges if s for p in s}
    missing = [p for p in run.charges_asked if p not in printed]
    if missing:
        raise FinishError(
            f"{run.deck.name} asks for the {' and '.join(missing)} charges at "
            f"every step (Compute IR) and {run.output.name} prints none: the "
            f"run and its deck disagree")
    n = len(run.elements)
    expected = 1 + len(FC_STEP_PATTERN) * len(run.displaced)
    if len(run.step_charges) != expected:
        raise FinishError(
            f"{run.output.name} reads {len(run.step_charges)} steps; the "
            f"force-constant run over {len(run.displaced)} free atom(s) "
            f"takes {expected}")
    every_atom = list(range(n))
    charges = {}
    for p in run.charges_asked:
        rows = []
        for i, s in enumerate(run.step_charges):
            row = (s or {}).get(p)
            if row is None or len(row) != n:
                raise FinishError(
                    f"{run.output.name}: step {i} prints no {p} charges for "
                    f"its {n} atoms, which other steps print -- the run and "
                    f"its charges disagree")
            rows.append([float(q) for q in map_rows_to_user_order(
                run.permutation, row, every_atom, n)[0]])
        charges[p] = rows
    steps = [{"atom": None, "axis": None, "side": 0}]
    for k in run.displaced:
        user = map_to_user_index(run.permutation, [k], n)[0]
        for axis, side in FC_STEP_PATTERN:
            steps.append({"atom": int(user), "axis": axis, "side": side})
    return {"displacement_ang": float(run.fc.displacement_ang),
            "steps": steps, "charges_e": charges}


def result_of(run: ForceConstantRun,
              relaxation: Optional[Mapping[str, Any]] = None
              ) -> SpectraResults:
    """The run's result: its block, masses and geometry through the one
    analysis (`spectra.vibrational_analysis`), with what this route says of
    itself -- the Methods paragraph, the force-constant facts, the stage.
    ``relaxation`` is the run's `relax` call's record, from the run's index
    (`engines/vibration.md` § 5.11 § 3), which the result's `relaxation`
    block carries whole; ``None`` when the structure was stated relaxed."""
    n = len(run.elements)
    if run.fc.n_atoms != n:
        raise FinishError(f"{Path(run.fc.path).name} describes "
                          f"{run.fc.n_atoms} atoms; {run.deck.name} has {n}")
    H = hessian_from_fc(run.fc, run.displaced)
    rec = dict(run.record)
    if relaxation is not None:
        rec["relaxation"] = dict(relaxation)
        # THE STAGE WHOSE RUN RELAXED -- this one, its `relax` call (§ 5.11
        # § 1): the remedy's launch line names it.
        rec["relaxation_stage"] = rec.get("stage")
    criterion = rec.get("force_criterion_ev_ang")
    if rec.get("relaxation") is not None and not rec.get("relaxation_stage"):
        raise FinishError(
            f"{run.deck.name}'s vibration block carries a relaxation record "
            f"but not the stage it is of -- a deck prepared before its "
            f"remedy named the stage (2026-09-29): prepare the stage again")
    if rec.get("temperature_K") is None:
        raise FinishError(
            f"{run.deck.name}'s vibration block states no temperature_K -- "
            f"a deck prepared before its finish summed at the template's "
            f"temperature (2026-09-28): prepare the stage again")
    # THE MASSES, as `prep` stated them (I3) -- one per atom of the deck.
    masses = rec.get("masses_amu")
    if masses is None:
        raise FinishError(
            f"{run.deck.name}'s vibration block states no masses_amu -- a "
            f"deck prepared before the masses were stated in it (2026-10-10, "
            f"format v2): prepare the stage again")
    if len(masses) != n:
        raise FinishError(
            f"{run.deck.name}'s vibration block states {len(masses)} masses "
            f"for its {n} atoms")
    res = vibrational_analysis(
        H, [float(m) for m in masses],
        run.positions_ang, run.elements, run.held,
        axis_kind=run.axis_kind, cell=run.cell, permutation=run.permutation,
        structure=equilibrium_structure(run),
        engine="siesta", engine_version=run.siesta_version,
        molbuilder_version=str(rec.get("molbuilder_version") or ""),
        geometry_note=GEOMETRY_NOTE,
        reference_forces_ev_ang=run.reference_forces_ev_ang,
        force_criterion_ev_ang=criterion,
        already_relaxed=bool(rec.get("already_relaxed")),
        ladder_relaxation=rec.get("relaxation"),
        relaxation_stage=rec.get("relaxation_stage"),
        temperature_K=float(rec["temperature_K"]),
        config={"engine": "siesta", "calculation": "vibration",
                "stage": rec.get("stage")},
        engine_metadata={
            "fc_file": Path(run.fc.path).name,
            "fc_displacement_ang": run.fc.displacement_ang,
            # SIESTA'S NUMBERING -- the deck's held-first order, 1-based:
            # an engine fact, not an atom's identity (`model/overview.md`
            # § 2.2a).
            "fc_range_engine_1based": [siesta_numbering(run.displaced[0]),
                                siesta_numbering(run.displaced[-1])],
            # The block's own numerics, before the symmetrisation hides them
            # (`engines/vibration.md` § 5.5).
            "fc_asymmetry_max_ev_ang2": fc_block_asymmetry(run.fc,
                                                           run.displaced),
            "reference_force_criterion_ev_ang": (
                None if criterion is None else float(criterion))})
    # THE METHODS PARAGRAPH, once the analysis has said how many motions it
    # removed -- its own count, not a second derivation of it (R1).
    charges = partial_charges_of(run)
    methods = siesta_methods_text(
        displacement_bohr=run.fc.displacement_ang / BOHR_ANGSTROM,
        n_free=len(run.displaced), n_held=len(run.held),
        n_rigid=int(res.removed_motions["count"]),
        siesta_version=run.siesta_version,
        charge_partitions=tuple(charges["charges_e"]) if charges else ())
    # THE INFRARED IS THE CHARGES', derived on request from them -- done
    # once they are stored (`engines/vibration.md` § 5.11 § 8).
    return replace(res, methods_text=methods,
                   bibliography_keys=extract_citation_keys(methods),
                   partial_charges=charges,
                   phase_ir=(PHASE_COMPLETE if charges
                             else PHASE_NOT_REQUESTED),
                   ir_route="charges" if charges else "none")


def _with_raman_phase(res, run, optical, phase: str):
    """``res`` with its Raman phase -- the optical calls' citation
    ``optical`` (``None`` where the run planned none), the phase ``phase``
    where it has calls, the route, the step ``delta`` and the Methods
    paragraph's sentence for that route -- one composition for the force
    constants' record step and the finish (`engines/vibration.md` § 5.11
    § 8).  With no call planned, the deck's `vibration` block says why:
    Compute Raman off (`raman` null) -- *not requested* -- or Raman asked
    over a range that kept no mode -- *complete*, nothing left to run, the
    Methods sentence naming the range."""
    raman = run.record.get("raman")
    route = "optical" if optical is not None else "none"
    methods = with_siesta_raman_route(res.methods_text, route, raman)
    return replace(
        res, optical=optical, raman_route=route,
        phase_raman=(phase if optical is not None
                     else PHASE_COMPLETE if raman is not None
                     else PHASE_NOT_REQUESTED),
        raman_fd_step_ang=(float(run.fc.displacement_ang)
                           if optical is not None else None),
        methods_text=methods,
        bibliography_keys=extract_citation_keys(methods))


def finish(deck, output) -> Path:
    """Finish the force-constant run ``deck`` ran (its output ``output``):
    write ``<label>.spectra.json`` beside it and return its path.  The same
    door for anything holding a finished force-constant call."""
    return _finished(deck, output)[2]


def _finished(deck, output, *, into=None, relaxation=None):
    """The force-constant call ``deck`` ran, read and analysed; its result
    pair written into ``into`` (the run's root, `engines/vibration.md`
    § 5.11 § 2) -- beside the deck with none -- with ``relaxation``, the
    run's `relax` call's record."""
    run = read_force_constant_run(deck, output)
    res = result_of(run, relaxation=relaxation)
    out = (Path(into) if into is not None else Path(deck).parent) / compose(
        run.label, ROLE_SPECTRA)
    dump_spectra_json(res, out)
    return run, res, out


# --------------------------------------------------------------------- #
#  A run in phases: the plan and record steps of each call               #
#  (`engines/vibration.md` § 5.11 § 3)                                   #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class _CallHere:
    """The call this step runs in: its folder, its deck, its run's folder
    and the calculation's root (`calcdirs.where`), the stage's names and
    the run's index path."""
    folder: Path
    deck: Path
    label: str
    names: "RunNames"
    run_dir: Path
    root: Path

    @property
    def index(self) -> Path:
        return index_path(self.run_dir, self.names, FIRST_ATTEMPT)

    @property
    def stage_dir(self) -> Path:
        return self.run_dir.parent


def _here(deck) -> _CallHere:
    """Where this step runs, read off the call's folder and its deck's
    name (`calcdirs.where`; `runfiles.parse`): the run's folder above it,
    the calculation's root, and the stage's names."""
    deck = Path(deck).resolve()
    folder = deck.parent
    text = read_deck(deck)
    label = system_label(text)
    if not label:
        raise FinishError(f"{deck.name} states no SystemLabel")
    rec = parse(deck.name, label)
    if rec is None or rec.stage is None:
        raise FinishError(f"{deck.name} is not named on {label!r} with a "
                          f"stage (runfiles.compose)")
    w = calcdirs.where(folder)
    if w.run_folder is None or w.root is None:
        raise FinishError(f"{folder} is no call of a run: its place record "
                          f"names no run above it (calcdir.json)")
    return _CallHere(folder=folder, deck=deck, label=label,
                     names=RunNames.of(label, rec.stage, "hierarchical"),
                     run_dir=Path(w.run_folder).resolve(),
                     root=Path(w.root).resolve())


def _species_index(deck_text: str) -> Mapping[str, int]:
    """Each species label's index in the deck's ``ChemicalSpeciesLabel`` --
    what a coordinates row names its species by."""
    rows = read_fdf(deck_text)[1].get(fdf_key("ChemicalSpeciesLabel")) or []
    return {str(r[2]): int(r[0]) for r in rows}


def _newest_output(folder: Path, names: "RunNames") -> Optional[Path]:
    """The engine output of the call in ``folder``'s NEWEST RUN -- the run
    door's (`runrecord.newest_run`: the highest index any file of it
    reached) -- or ``None`` where that run printed none: an engine that
    died before its first line is never read from the try before it."""
    n = newest_run(folder, names)
    got = [p for p, rf in find(folder, names.label, role=".out",
                               stage=names.stage) if rf.run == n]
    return got[0] if n is not None and got else None


def _reading(output: Path):
    """SIESTA's one reading pass over ``output`` (`SiestaReader`)."""
    return SiestaReader().feed_text(
        Path(output).read_text(encoding="utf-8", errors="replace")).finish()


def _call_of(here: _CallHere, phase_name: str) -> "Call":
    """The call this folder is: the phase's own for the relaxation and the
    force constants, the folder's name for an optical call."""
    return Call(phase_name, here.folder.name if phase_name == OPTICAL
                else phase_name)


def _notify(here: _CallHere, event: str, **words) -> None:
    """The run speaks once (§ 5.11 § 7a): ``mb_monitor.pyz notify``, the
    monitor's bundle beside this deck, with the run's identity -- the verb
    passes an event already sent for the run.  Its failure is logged, never
    the step's."""
    import subprocess
    bundle = here.folder / MONITOR_BUNDLE
    if not bundle.is_file():
        _say("WARN", f"vibration: {MONITOR_BUNDLE} is not beside the deck: "
                     f"the run's {event} is not sent")
        return
    cmd = [sys.executable, str(bundle), "notify", event,
           "--label", here.label, "--stage", here.names.stage,
           "--run", str(FIRST_ATTEMPT), "--dir", str(here.run_dir)]
    for k, v in words.items():
        if v is not None:
            cmd += [f"--{k}", str(v)]
    got = subprocess.run(cmd, capture_output=True, text=True)
    if got.returncode != 0:
        _say("WARN", f"vibration: notify {event} exited {got.returncode}: "
                     f"{(got.stderr or got.stdout).strip()}")


def plan_call(deck, phase_name: str) -> None:
    """THE PLAN STEP of one call, before its engine (§ 5.11 § 3): the call's
    coordinates where they are another call's result -- the force
    constants' from the `relax` call's output, written as its
    `coordinates.fdf` -- its entry in the run's index (its deck's and
    coordinates' sha256, where they came from, its seed), and the run's
    `start`, sent through the monitor's bundle when the run's record holds
    none."""
    here = _here(deck)
    rule = phase(phase_name)
    call = _call_of(here, phase_name)
    index = read_index(here.index)
    coords = here.folder / COORDINATES_FILE
    came_from = "the structure, as prep wrote it"
    # A RELAXATION CONTINUED from its own last run's warm files, handed
    # over by launch (§ 5.11 § 3, § 6): the run it continues, said.
    continued = (read_continued_from(here.folder, here.names, FIRST_ATTEMPT)
                 if phase_name == RELAX else None)
    if continued is not None:
        came_from = f"its own last geometry, continued from {continued}"
    if phase_name == FC and RELAX in {c.phase for c in calls_of(index)}:
        out = _newest_output(here.run_dir / RELAX, here.names)
        if out is None:
            raise FinishError(f"the relaxation left no output in "
                              f"{here.run_dir / RELAX}: the force constants "
                              f"have no geometry to take")
        reading = _reading(out)
        steps = reading.get("steps") or []
        last = next((s for s in reversed(steps) if s.get("coords")), None)
        if last is None:
            raise FinishError(f"{out.name} holds no geometry step")
        text = read_deck(here.deck)
        labels = [str(r[0]) for r in last["coords"]]
        positions = [[float(x) for x in r[1:4]] for r in last["coords"]]
        coords.write_text(coordinates_text(labels, positions,
                                           _species_index(text)),
                          encoding="utf-8")
        came_from = f"{RELAX}/{out.name}, its last geometry step"
    elif phase_name == OPTICAL:
        row = entry_of(index, call) or {}
        came_from = str(row.get("coordinates_from") or
                        "planned by the force constants' record step")
    if not coords.is_file():
        raise FinishError(f"{coords.name} is not beside {here.deck.name}: "
                          f"the call has no coordinates to compute at")
    seed = (here.folder / compose(here.label, rule.seed[1])
            if rule.seed is not None else None)
    seeded = seed is not None and seed.is_file()
    add_call(index, call,
             deck_sha256=sha256_of(here.deck),
             coordinates_sha256=sha256_of(coords),
             coordinates_from=came_from,
             seed=(seed.name if seeded else None),
             seed_from=(f"{rule.seed[0]}, copied by the walk" if seeded
                        else f"its own warm files, from {continued}"
                        if continued is not None else None),
             started=stamp())
    write_index(here.index, index)
    _say("INFO", f"vibration: {call.words()} planned -- coordinates from "
                 f"{came_from}")
    _notify(here, "start")


def _relaxation_record(output: Path) -> Optional[dict]:
    """The `relax` call's record -- the one record function
    (`parse.contract.relaxation_of_output`, `model/parse.md` § 5b.1) over
    SIESTA's one reading pass: each step a frame, its structure, its
    forces and the engine's own Max lines; ``None`` for a run that relaxed
    nothing."""
    from types import SimpleNamespace
    reading = _reading(output)
    frames = [
        SimpleNamespace(
            structure=Structure(
                elements=[str(r[0]) for r in s["coords"]],
                positions=np.array([r[1:4] for r in s["coords"]],
                                   dtype=float)),
            forces=s.get("forces"), max_force=s.get("max_force"),
            max_force_constrained=s.get("max_force_constrained"),
            step_index=i)
        for i, s in enumerate(reading.get("steps") or []) if s.get("coords")]
    return relaxation_of_output(
        output,
        SimpleNamespace(frames=frames,
                        runtime_info=reading.get("runtime_info") or {},
                        run_state=reading.get("run_state")),
        engine="siesta")


def _plan_optical(here: _CallHere, run: ForceConstantRun, res: SpectraResults,
                  index: dict) -> int:
    """THE OPTICAL CALLS PLANNED once the force constants are known
    (§ 5.11 § 3): the modes *Raman for* picks (`select_modes`, on the deck's
    `vibration` block's ``raman``), each picked mode's ``δQ = δ /
    max_k |L_k|`` -- ``δ`` the `.FC` header's displacement -- and for each
    frame a call folder below the run's `optical/`, opened through the
    opener's travelling core (`calcdirs.record`, `planned.bring` from the
    stage's `optical/`), its `coordinates.fdf` (``R₀ ± δQ·L``, the one
    displacement rule, `normal_modes.displaced`) and its index entry.
    Returns how many were planned -- 0 with Compute Raman off, the stage
    running no optical phase, or when a range keeps no mode."""
    if "raman" not in run.record:
        raise FinishError(
            f"{run.deck.name}'s vibration block states no raman -- a deck "
            f"prepared before Raman for (2.45.0): prepare the stage again")
    raman = run.record["raman"]
    if raman is None:
        return 0
    stage_optical = here.stage_dir / OPTICAL
    freqs = [float(m.frequency_cm1) for m in res.modes]
    picked = select_modes(freqs, str(raman.get("modes")),
                          freq_min_cm1=raman.get("freq_min_cm1"),
                          freq_max_cm1=raman.get("freq_max_cm1"))
    delta = float(run.fc.displacement_ang)
    # THE SELECTION, AS ASKED AND AS APPLIED (§ 7), whatever it picks --
    # none picked is an answer the run states.
    index["selection"] = {"asked": dict(raman),
                          "applied": [int(k) for k in picked],
                          "delta_ang": delta}
    if not picked:
        return 0
    if not stage_optical.is_dir():
        raise FinishError(
            f"Raman for picks {len(picked)} mode(s) and the stage holds "
            f"no {OPTICAL}/ folder for their calls -- a stage prepared "
            f"before its optical phase (2.42.0): prepare it again")
    n = len(run.elements)
    r0 = np.asarray(run.positions_ang, dtype=float)
    # THE MODE OVER EVERY ATOM, in the deck's order: the result lists it by
    # user index (`model/overview.md` § 2.2b), the deck by the engine's.
    order = (run.permutation.map_to_user_index(list(range(n)))
             if run.permutation is not None else list(range(n)))
    species = _species_index(read_deck(run.deck))
    frames = [(Call(OPTICAL, frame_token(0), mode=None, side=0), r0)]
    for k in picked:
        m = res.modes[k - 1]
        l_user = mode_over_atoms(m.eigenvector_canonical, res.free_atom_idxs, n)
        l_deck = np.asarray(l_user, dtype=float)[order]
        dq = q_for_largest_move(l_deck, delta)
        for side in (+1, -1):
            frames.append((Call(OPTICAL, frame_token(len(frames)), mode=int(k),
                                side=side, frequency_cm1=float(m.frequency_cm1),
                                dq_amu12_ang=float(dq)),
                           displaced(r0, l_deck, side * dq)))
    plan = Plan()
    container = here.run_dir / OPTICAL
    if not (container / calcdirs.FILENAME).is_file():
        plan.folder(container)
        plan.text(*calcdirs.record(container, role=calcdirs.CONTAINER,
                                   root=here.root))
    # WHAT A CALL'S FOLDER HOLDS -- the deck and the one list of what
    # travels beside it (`runfiles.beside_the_deck`, as a launch brings a
    # run: `materialize.bring_files`), from the stage's phase folder; and
    # the calculation's shared package -- the pseudopotentials under its
    # `pseudos/`, the engine seam's rule (`jobset.engines`): SIESTA opens
    # `<label>.psml` in the folder it runs from and no other.
    files = [here.deck.name,
             *beside_the_deck(here.deck.name, VIBRATION_BUNDLE, here.names)]
    shared = sorted(f"{PSEUDO_DIR}/{p.name}"
                    for p in (here.root / PSEUDO_DIR).glob("*.psml"))
    for call, positions in frames:
        folder = container / call.name
        plan.folder(folder)
        # ...then the call's own over the phase folder's: its place record
        # (a call of the run, below it) and its coordinates.
        bring(plan, files, into=folder, from_dir=stage_optical)
        bring(plan, shared, into=folder, from_dir=stage_optical,
              also=(here.root,))
        plan.text(*calcdirs.record(folder, role=calcdirs.RUN, root=here.root))
        plan.text(folder / COORDINATES_FILE,
                  coordinates_text(run.elements, positions, species))
    plan.carry_out()
    for call, _positions in frames:
        add_call(index, call,
                 coordinates_sha256=sha256_of(container / call.name
                                            / COORDINATES_FILE),
                 coordinates_from=("the force-constant run's reference "
                                   "geometry" if call.mode is None else
                                   f"the reference geometry displaced by "
                                   f"{call.side:+d} δQ along mode "
                                   f"{call.mode}"),
                 planned=stamp())
    return len(frames)


def _optical_settings(here: _CallHere, index: dict) -> dict:
    """``/settings``: every optical keyword as the stage's optical deck --
    the one each optical call copies -- writes it, the light's direction
    (its ``Optical.Vector``), the selection as asked and as applied, ``δ``,
    and the deck's whole text and sha256."""
    deck = here.stage_dir / OPTICAL / here.deck.name
    spelled: dict = {}
    params, blocks = read_fdf(read_deck(deck), spelled_out=spelled)
    keywords: dict = {}
    for key, value in params.items():
        if key.startswith(fdf_key("Optical")):
            keywords[spelled.get(key, key)] = " ".join(value)
    for key, rows in blocks.items():
        if key.startswith(fdf_key("Optical")):
            keywords[spelled.get(key, key)] = [" ".join(r) for r in rows]
    vector = blocks.get(fdf_key("Optical.Vector")) or []
    sel = index.get("selection") or {}
    return {"keywords": keywords,
            "direction": ([float(x) for x in vector[0][:3]] if vector
                          else None),
            "selection": {"asked": sel.get("asked"),
                          "applied": sel.get("applied")},
            "delta_ang": sel.get("delta_ang"),
            "deck_name": deck.name,
            "deck_text": deck.read_text(encoding="utf-8"),
            "deck_sha256": sha256_of(deck)}


def _call_facts(here: _CallHere, folder: Path) -> dict:
    """What a call's own files state of it (§ 5.11 § 8): from its newest
    output, through SIESTA's one reading pass, the geometry it computed at
    (the deck's order), its SCF's iterations, total energy and Fermi level,
    and the build; from its session log -- its run's -- the engine's wall
    time (`wrapper_log`, the one source of it)."""
    out = _newest_output(folder, here.names) if folder.is_dir() else None
    if out is None:
        return {}
    reading = _reading(out)
    facts: dict = {"siesta_version": ((reading.get("runtime_info") or {})
                                      .get("siesta_build") or {})
                   .get("version")}
    step = next((s for s in reversed(reading.get("steps") or [])
                 if s.get("coords")), None)
    if step is not None:
        scf = step.get("scf_history") or []
        facts.update(
            coords=np.asarray([r[1:4] for r in step["coords"]], dtype=float),
            scf_iterations=len(scf) if scf else None,
            total_energy_ev=step.get("energy"),
            fermi_level_ev=scf[-1].get("ef") if scf else None)
    named = parse(out.name, here.label)
    log = log_of_run(folder, here.label,
                     named.run if named is not None and named.run is not None
                     else FIRST_ATTEMPT, here.names.stage)
    if log is not None:
        sections = read_wrapper_log(log.read_text(encoding="utf-8",
                                                  errors="replace"))
        if sections:
            facts["engine_elapsed_s"] = sections[0].get("engine_elapsed_s")
    return facts


def _optical_payload(here: _CallHere, index: dict, res: SpectraResults
                     ) -> Optional[dict]:
    """THE OPTICAL CALLS AS THEIR DATA FILE STATES THEM (§ 5.11 § 8),
    composed from the run's own records -- the index (each call's mode,
    side, frequency and ``δQ``; its state by the run's one done rule), each
    call's own files (:func:`_call_facts`, its ``.EPSIMG`` through its one
    reader) and the result (the reference structure, its atom index map,
    the masses).  ``None`` where the run planned no optical call.  Every
    frame's geometry is put in the user order through the result's map; a
    frame whose spectrum is on another grid than the others' is refused by
    name -- the calls share one deck."""
    calls = [c for c in calls_of(index) if c.phase == OPTICAL]
    if not calls:
        return None
    states = {r["name"]: r["state"] for r in _rows(here, index)
              if r["phase"] == OPTICAL}
    atoms = res.atom_index_map_stated()
    run_rel = here.run_dir.relative_to(here.root).as_posix()
    grid = spins = version = None
    frames = []
    for c in calls:
        row = entry_of(index, c) or {}
        folder = here.run_dir / c.rel
        state = states.get(c.name, "pending")
        facts = _call_facts(here, folder) if state != "pending" else {}
        coords = facts.get("coords")
        version = version or facts.get("siesta_version")
        f = {"call": c.rel, "mode": c.mode,
             "frequency_cm1": row.get("frequency_cm1"),
             "side": int(c.side or 0),
             "dq_amu12_ang": row.get("dq_amu12_ang"),
             # THE USER ORDER, through the result's map (§ 2.2b): the
             # call's rows stand in the engine's.
             "position_ang": (coords[list(atoms.engine)].tolist()
                              if coords is not None and atoms.engine
                              is not None and len(coords) == atoms.n_atoms
                              else None),
             "state": state,
             # THE RUN THAT COMPUTED IT: this one, or for a call taken over
             # the run its call folder lies in (`paths.run_folder_of`).
             "computed_in": (run_folder_of(str(row["taken_from"])).as_posix()
                             if row.get("taken_from")
                             else run_rel if row.get("ended") is not None
                             else ""),
             **{k: facts.get(k) for k in ("scf_iterations", "total_energy_ev",
                                          "fermi_level_ev",
                                          "engine_elapsed_s")}}
        eps = folder / compose(here.label, ".EPSIMG")
        if state == "finished" and eps.is_file():
            e = read_epsimg(eps)
            if grid is None:
                grid, spins = e.energy_ev, e.spin
            elif (e.spin != spins or e.energy_ev.shape != grid.shape
                  or not np.array_equal(e.energy_ev, grid)):
                raise FinishError(
                    f"{c.rel}/{eps.name}: its photon energies or spin "
                    f"channels are not the other calls' -- the calls share "
                    f"one deck, so one grid")
            f.update(f_sum=list(e.f_sum), drude_wp2_ha2=list(e.drude_wp2_ha2),
                     eps2=e.eps2)
        frames.append(f)
    masses = res.equilibrium_masses_amu
    return {"label": here.label, "run": run_rel,
            "siesta_version": version or res.engine_version,
            "molbuilder_version": res.molbuilder_version,
            "settings": _optical_settings(here, index),
            "structure": res.structure,
            "atom_index_map": atoms.to_dict(),
            "masses_amu": ([float(m) for m in masses]
                           if masses is not None else None),
            "photon_energy_ev": grid.tolist() if grid is not None else [],
            "spin": list(spins or ()), "frames": frames}


def _write_optical(here: _CallHere, index: dict, res: SpectraResults
                   ) -> Optional[dict]:
    """THE OPTICAL CALLS' RECORD, rewritten (§ 5.11 § 8): ``<label>.
    optical.nc`` at the run's root (`spectra.optical_data`), and the
    result's citation of it (`spectra.results.checked_optical`) -- the file
    and its sha256, the direction, ``δ``, the selection, each frame's
    state -- or ``None`` where the run planned no optical call."""
    payload = _optical_payload(here, index, res)
    if payload is None:
        return None
    path = write_optical_data(
        payload, here.run_dir / compose(here.label, ROLE_OPTICAL))
    frames = payload["frames"]
    sides: dict = {}
    for f in frames:
        if f["mode"] is not None:
            sides.setdefault(f["mode"], {})[f["side"]] = f
    settings = payload["settings"]
    return {"data_file": path.name, "data_sha256": sha256_of(path),
            "direction": settings["direction"],
            "displacement_ang": settings["delta_ang"],
            "selection": settings["selection"],
            "reference": next(f["state"] for f in frames
                              if f["mode"] is None),
            "modes": [{"mode": int(k),
                       "frequency_cm1": by[+1]["frequency_cm1"],
                       "dq_amu12_ang": by[+1]["dq_amu12_ang"],
                       "plus": by[+1]["state"], "minus": by[-1]["state"]}
                      for k, by in sides.items()]}


def _force_constants(here: _CallHere, index: dict
                     ) -> Optional[Tuple[ForceConstantRun, SpectraResults]]:
    """THE RUN'S FORCE CONSTANTS, read and analysed from its `fc` call's own
    files -- the call this run computed, or took over from the run that did
    (§ 5.11 § 6), whose result lies at that run's root, never this one's --
    with the `relax` call's record from the index: ``(run, result)``, or
    ``None`` where the call left no output.  What the finish writes the
    result from and what each optical call's record is put in the user
    order by.  ``None`` where the force-constant call is not done by the
    run's one done rule (:func:`_rows`) -- a call that failed leaves its
    output and a partial `.FC`, and builds nothing (§ 5)."""
    if not any(r["done"] for r in _rows(here, index) if r["phase"] == FC):
        return None
    fc_out = _newest_output(here.run_dir / FC, here.names)
    if fc_out is None:
        return None
    relax_row = entry_of(index, Call(RELAX, RELAX))
    run = read_force_constant_run(here.run_dir / FC / here.deck.name, fc_out)
    return run, result_of(run,
                          relaxation=(relax_row or {}).get("relaxation"))


class _NothingToWrite(Exception):
    """The finish has no force constants to write a result from."""


def _rows(here: _CallHere, index: dict) -> List[dict]:
    """Each call the index lists, by the one done rule this machine's
    done door asks (§ 5.11 § 6, § 7; `continuation.done`): done when it
    ended with exit 0, its record step did not fail, its SCF converged
    where its phase's verdict counts (`PhaseRule.verdict_counts`, the
    verdict its record step read) and it holds its phase's products
    (`products_of`) -- the last two said in the rule's own words
    (`_run_ending.not_done_why`); each row with the call's words and the
    run it was taken over from, where it was (`summary`)."""
    rows = []
    for row in index.get("calls") or []:
        call = Call.from_dict(row)
        ended = row.get("ended") is not None
        said = row.get("verdict")
        why = "; ".join(w for w in (
            (f"exit {row.get('exit')}" if ended
             and row.get("exit") not in (0, "0") else None),
            row.get("record_error")) if w) or None
        if ended and why is None:
            why = not_done_why(
                (said["of"], bool(said["converged"])) if said else None,
                counts=phase(call.phase).verdict_counts,
                missing=[name for name in products_of(
                    here.names.stage, here.label, here.root, point=call)
                    if not (here.run_dir / call.rel / name).is_file()])
        ok = ended and why is None
        rows.append({"phase": row["phase"], "name": row.get("name"),
                     "words": call.words(), "done": ok, "ended": ended,
                     "started_from": row.get("taken_from"),
                     "state": ("finished" if ok else
                               "failed" if ended else "pending"),
                     "why": why})
    return rows


def _still_to_run(here: _CallHere, index: dict) -> List[dict]:
    """THE CALLS THE WALK MAY STILL RUN (§ 5.11 § 5): listed, not ended,
    and every phase each needs -- of those the run walks -- holding a call
    done (`PHASES`' ``needs``, the walk's).  A call whose need did not end
    done is not run, so the finish runs whatever ended without it."""
    rows = _rows(here, index)
    walked = set(index.get("phases") or ())
    done = {r["phase"] for r in rows if r["done"]}
    return [r for r in rows if not r["ended"]
            and all(n in done for n in phase(r["phase"]).needs
                    if n in walked)]


def _run_state(here: _CallHere, index: dict) -> Tuple[str, str]:
    """The run's state from its index, by the one done rule (:func:`_rows`):
    finished when every call is done, else failed naming the first call not
    done by its words, and why where its record step said."""
    rows = _rows(here, index)
    words = summary(rows, index.get("phases") or ())
    first = next((r for r in rows if not r["done"]), None)
    if first is None:
        return "finished", words
    return "failed", (f"{words}; {first['words']}: "
                      f"{first['why'] or first['state']}")


def _finish_run(here: _CallHere, index: dict) -> None:
    """THE FINISH, once no call the walk may still run is left
    (:func:`_still_to_run`, § 5.11 § 3, § 5): the result pair rewritten
    final from the force-constant call, with the `relax` call's record and
    the Raman phase's state -- nothing to write where the force constants
    never ran, the run's state saying why; the finish's own entry; the
    run's `finish` sent with its state."""
    started = stamp()
    try:
        got = _force_constants(here, index)
        if got is None:
            # NO FORCE CONSTANTS DONE -- the call failed, or what it builds
            # on did not end done (§ 5) -- so no result; the run's state
            # says why.
            _say("INFO", "vibration: the force-constant call is not done "
                         "-- no result to write")
            raise _NothingToWrite
        run, res = got
        out = here.run_dir / compose(here.label, ROLE_SPECTRA)
        # THE OPTICAL CALLS' RECORD, final, and its citation: the phase
        # complete once every call has ended (§ 5.11 § 8).
        optical = _write_optical(here, index, res)
        res = _with_raman_phase(res, run, optical, PHASE_COMPLETE)
        dump_spectra_json(res, out)
        code = 0
        _say("INFO", f"vibration: the finish wrote {out.name}")
    except _NothingToWrite:
        code = 0
    except Exception as exc:                            # noqa: BLE001
        import traceback
        _say("ERROR", f"vibration: the finish failed -- "
                      f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        code = 1
    index["finish"] = {"started": started, "ended": stamp(), "exit": code}
    write_index(here.index, index)
    state, detail = _run_state(here, index)
    if code != 0:
        state, detail = "failed", f"{detail}; the finish exited {code}"
    _notify(here, "finish", state=state, detail=detail)


def record_call(deck, phase_name: str, output, exit_code: int) -> None:
    """THE RECORD STEP of one call, after its engine, success or failure
    (§ 5.11 § 3): how it ended into the run's index; the `relax` call's
    relaxation record; the force constants' analysis -- the result pair at
    the run's root, the optical calls planned; and, once no call the walk
    may still run is left (:func:`_still_to_run`), the finish."""
    here = _here(deck)
    call = _call_of(here, phase_name)
    index = read_index(here.index)
    output = Path(output)
    # THE ENDING FIRST, success or failure (§ 3): what the analysis below
    # finds is the call's too, but never the reason its ending is missing.
    add_call(index, call, ended=stamp(), exit=int(exit_code))
    write_index(here.index, index)
    _say("INFO", f"vibration: {call.words()} ended, exit {exit_code}")
    # WHAT THE STEP ITSELF COULD NOT DO is the call's, in the index -- its
    # conclusion then reads failed, and the run's one done rule says so
    # too (:func:`_rows`) -- and the step's failure, said after the finish
    # has had its turn: a record that cannot be written never keeps the
    # run from ending.
    failed = None

    def _record_failed(what: str, exc: Exception) -> None:
        nonlocal failed
        _say("ERROR", f"vibration: {what} -- {type(exc).__name__}: {exc}")
        add_call(index, call, record_error=f"{what}: {exc}")
        write_index(here.index, index)
        failed = exc

    # WHAT IT SAYS IT CONVERGED, read from its output by the reader this
    # machine's status door reads it with (`_run_ending.verdict_of`) -- the
    # one done rule's verdict (:func:`_rows`).
    if output.is_file():
        try:
            said = verdict_of([ending_of(output)])
            add_call(index, call, verdict=(
                {"of": said[0], "converged": said[1]} if said else None))
            write_index(here.index, index)
        except Exception as exc:                        # noqa: BLE001
            _record_failed("its verdict was not read", exc)
    if phase_name == RELAX and output.is_file():
        try:
            add_call(index, call, relaxation=_relaxation_record(output))
            write_index(here.index, index)
        except Exception as exc:                        # noqa: BLE001
            _record_failed("the relaxation record was not read", exc)
    if phase_name == FC and int(exit_code) == 0:
        try:
            # THE ANALYSIS, read once by the run's one reading of its force
            # constants, and the result written once below -- never a
            # result on disk for an analysis whose planning then failed.
            got = _force_constants(here, index)
            if got is None:
                # NOT DONE by the one rule -- not converged, or no .FC --
                # so nothing to analyse; the call's row says why (§ 5).
                why = next(r["why"] for r in _rows(here, index)
                           if r["phase"] == FC)
                _say("INFO", f"vibration: {call.words()} is not done -- "
                             f"{why}: nothing to analyse")
            else:
                run, res = got
                out = here.run_dir / compose(here.label, ROLE_SPECTRA)
                planned = _plan_optical(here, run, res, index)
                # THE OPTICAL CALLS' RECORD, every frame pending, and the
                # result's citation of it: the Raman phase running (§ 8).
                optical = (_write_optical(here, index, res) if planned
                           else None)
                res = _with_raman_phase(res, run, optical, PHASE_RUNNING)
                dump_spectra_json(res, out)
                add_call(index, call, modes=len(res.modes))
                write_index(here.index, index)
                _say("INFO", f"vibration: {len(res.modes)} mode(s) from "
                             f"{Path(run.fc.path).name}; {planned} optical "
                             f"call(s) planned; wrote {out.name}")
        except Exception as exc:                        # noqa: BLE001
            # THE ANALYSIS FAILED after a clean engine exit: the run is
            # over -- nothing after the force constants can run -- so the
            # finish has its turn below and says so.
            _record_failed("the force-constant analysis failed", exc)
    # AN OPTICAL CALL'S RECORD: the data file rewritten with it (§ 8).
    if phase_name == OPTICAL:
        try:
            got = _force_constants(here, index)
            if got is None:
                raise FinishError(
                    f"the force-constant call in {here.run_dir / FC} is not "
                    f"done: the optical calls' record has no reference "
                    f"geometry to stand on")
            _write_optical(here, index, got[1])
        except Exception as exc:                        # noqa: BLE001
            _record_failed("the optical calls' record was not written", exc)
    if not _still_to_run(here, index):
        _finish_run(here, index)
    if failed is not None:
        raise failed


def _say(level: str, message: str) -> None:
    """One line in the wrapper's session log, in its own format."""
    sys.stderr.write(LOG_LINE % (time.strftime(LOG_CLOCK), level, message)
                     + "\n")
    sys.stderr.flush()


_USAGE = ("vibration: usage: mb_vibration.pyz finish <deck> <output> | "
          "plan <phase> <deck> | record <phase> <deck> <output> --exit <rc>")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The bundle's verbs (`engines/vibration.md` § 5.11 § 3): ``plan
    <phase> <deck>`` before a call's engine, ``record <phase> <deck>
    <output> --exit <rc>`` after it, and ``finish <deck> <output>`` -- the
    force-constant analysis alone, for anything holding a finished
    force-constant call.  0 when done, 1 when it could not be (the reason
    and its traceback in the session log), 2 when asked wrongly."""
    args = list(sys.argv[1:] if argv is None else argv)
    verb = args[0] if args else None
    try:
        if verb == "plan" and len(args) == 3:
            plan_call(args[2], args[1])
            return 0
        if (verb == "record" and len(args) == 6 and args[4] == "--exit"
                and args[5].lstrip("-").isdigit()):
            record_call(args[2], args[1], args[3], int(args[5]))
            return 0
        if verb != "finish" or len(args) != 3:
            _say("ERROR", _USAGE)
            return 2
        deck, output = args[1:]
        run, res, out = _finished(deck, output)
    except Exception as exc:                            # noqa: BLE001
        import traceback
        _say("ERROR", f"vibration: {verb or 'the step'} failed -- "
                      f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    _say("INFO", f"vibration: {len(res.modes)} mode(s) from "
                 f"{Path(run.fc.path).name}; "
                 f"{res.removed_motions['count']} whole-body motion(s) "
                 f"removed before diagonalising; the Hessian over "
                 f"{res.n_atoms_in_hessian} of {res.n_atoms_total} atoms")
    rx = res.relaxation or {}
    if rx.get("max_force_eh_bohr") is not None:
        f_ev = rx["max_force_eh_bohr"] * HARTREE_BOHR_EV_ANGSTROM_ASE
        crit = run.record.get("force_criterion_ev_ang")
        verdict = ("stationary" if rx.get("converged")
                   else "NOT stationary" if rx.get("converged") is False
                   else "no criterion")
        _say("INFO" if rx.get("converged") is not False else "WARN",
             f"vibration: the reference geometry's largest force on the "
             f"free atoms is {f_ev:.4f} eV/Å"
             + (f" against {float(crit):g} eV/Å" if crit is not None else "")
             + f" -> {verdict}")
        if rx.get("converged") is False:
            _say("WARN", f"vibration: {rx.get('warning')}")
    _say("INFO", f"vibration: wrote {out.name}")
    return 0
