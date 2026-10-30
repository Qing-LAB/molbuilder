"""A SIESTA vibration, finished by its own job: what the force-constant run left -> the result.

MODULE  spectra.siesta_vibration (L2; numpy, the standard library and its
        siblings -- nothing that needs the package installed)
ROLE    the SIESTA route's FINISH (`engines/vibration.md` § 5.5): read a
        finished force-constant run and the deck that ran it, each record
        through the reader that owns it, run the one analysis, and write
        ``<label>.spectra.json`` beside the run
USED-BY each call's report steps (`mb_monitor.pyz point`), which run the
        kind's own steps in their place -- ``plan <deck>`` before SIESTA,
        ``record <deck> <output> --exit <rc>`` after it, ``finish-run
        <deck>`` once no call is left (`execution/runs-of-points.md` § 1.2;
        `engines/vibration.md` § 5.11 § 3; `runwrap.VIBRATION_BUNDLE`); the
        bundle's ``finish`` verb is the door, by hand, for anything holding
        a finished force-constant call (§ 5.5)
TRAVELS in ``mb_vibration.pyz`` (`runwrap.VIBRATION_COMPANIONS`)

SIESTA takes no second derivative.  Its force-constant run nudges the free
atoms and writes ``<SystemLabel>.FC``; everything after that is the harmonic
analysis both engines share (`spectra.vibrational_analysis`).  A vibration
calculation ends with its result on both engines (user, 2026-09-28), so the
SIESTA job runs that analysis itself, in its run, with the job's own
python -- there is no step after the launch.

EVERY INPUT IS A RECORD THE CALL'S FOLDER HOLDS, and each is read by its
owner:

  * the deck (``SystemLabel``, the species, the FC range) -- the fdf
    reader, `parse.fdf`;
  * the deck's molbuilder blocks -- their one reader, `deck_record`:
    ``atom-metadata``'s order record (the deck's atoms in held-first order,
    the way back to the user order; none for a deck the sort left in the
    user order -- `model/overview.md` § 2.2a, I5), ``engine-offset`` (the axis
    kinds, R3, and the box the run had, `box_of_run`) and ``vibration`` (the stationarity criterion, the
    thermochemistry's temperature, the person's statement, the stage,
    Raman for, the molbuilder that prepared it);
  * the run's index -- its one reader, `runpoints.read_index`: how each
    call ended, and the `relax` call's relaxation record;
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
from typing import Any, Mapping, Optional, Sequence, Tuple

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
    from ..parse.fdf import (GEOMETRY_KEYS, deck_atoms, fdf_key, read_deck,
                             read_fdf, system_label)
    from ..runfiles import compose
    from ..structure import Structure
    from ..parse.contract import judged_force, relaxation_of_output
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
    from ..runfiles import (COORDINATES_FILE, FIRST_TRY,
                            PSEUDO_DIR, PSEUDO_FILE, ROLE_OPTICAL, ROLE_SPECTRA,
                            VIBRATION_BUNDLE,
                            RunNames, beside_the_deck, find_fixed,
                            fixed_name, parse)
    from ..paths import frame_token, run_folder_of
    from ..runrecord import newest_output, read_continued_from, stamp
    from .. import runpoints as _rp
    from ..siesta.deck_lines import coordinates_text
    from ..siesta.vibration_phases import (FC, IR_CHARGES, OPTICAL, RELAX,
                                           Call, calls_of,
                                           force_constants_call, phase,
                                           point_entry)
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
    from fdf import (GEOMETRY_KEYS, deck_atoms,  # type: ignore[no-redef]
                     fdf_key, read_deck, read_fdf, system_label)
    from runfiles import compose
    from structure import Structure
    from contract import judged_force, relaxation_of_output  # type: ignore[no-redef]
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
    from runfiles import (COORDINATES_FILE, FIRST_TRY,  # type: ignore[no-redef]
                          PSEUDO_DIR, PSEUDO_FILE, ROLE_OPTICAL,
                          ROLE_SPECTRA, VIBRATION_BUNDLE, RunNames, beside_the_deck,
                          find_fixed, fixed_name, parse)
    from paths import frame_token, run_folder_of  # type: ignore[no-redef]
    from runrecord import (newest_output,  # type: ignore[no-redef]
                           read_continued_from, stamp)
    import runpoints as _rp  # type: ignore[no-redef]
    from deck_lines import coordinates_text
    from vibration_phases import (FC, IR_CHARGES, OPTICAL,  # type: ignore[no-redef]
                                  RELAX, Call, calls_of,
                                  force_constants_call, phase,
                                  point_entry)
    from normal_modes import displaced, mode_over_atoms, q_for_largest_move
    from selection import select_modes

#: What the result says about how the geometry was reached: the run it
#: finishes relaxes nothing (`engines/vibration.md` § 5.5).
GEOMETRY_NOTE = ("the force-constant run relaxes nothing: its reference "
                 "geometry is taken as the stationary point")


def vibration_record(*, stage: str, force_criterion_ev_ang: Optional[float],
                     already_relaxed: bool,
                     force_criterion_from: str = "",
                     temperature_K: float,
                     masses_amu: Sequence[float],
                     molbuilder_version: str,
                     raman: Optional[Mapping[str, Any]] = None,
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
    from the run's index at the finish (§ 5.11 § 3), so the block states it
    ``null``, and its stage with it."""
    return {"stage": str(stage),
            "force_criterion_ev_ang": (None if force_criterion_ev_ang is None
                                       else float(force_criterion_ev_ang)),
            # where the tolerance came from -- said, never read
            "force_criterion_from": str(force_criterion_from),
            "already_relaxed": bool(already_relaxed),
            "relaxation": None,
            "relaxation_stage": None,
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
    the call's folder through the reader that owns it.  Per-atom fields are in the
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
    """Every input the finish needs, read from the folder ``deck`` sits in
    and from ``output``, the run's own account.  Raises :class:`FinishError`
    naming the file when a record is missing or two of them disagree."""
    deck, output = Path(deck), Path(output)
    folder = deck.parent
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
    fc_path = folder / compose(label, ".FC")
    if not fc_path.is_file():
        raise FinishError(f"no {fc_path.name} in {folder}: the "
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
                "stage": rec.get("stage"),
                # HOW IT WAS COMPUTED -- the functional, the basis, the mesh
                # -- as the force-constant deck states it (§ 6.2).
                "deck": _deck_cited(run.deck)},
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


def _finished(deck, output):
    """THE ``finish`` VERB, by hand (`engines/vibration.md` § 5.5): the
    force-constant call ``deck`` ran (its output ``output``), read and
    analysed, its result written beside the deck."""
    run = read_force_constant_run(deck, output)
    res = result_of(run)
    out = Path(deck).parent / compose(run.label, ROLE_SPECTRA)
    dump_spectra_json(res, out)
    return run, res, out


# --------------------------------------------------------------------- #
#  A run of points: the kind's own steps of each call, run by its report #
#  steps (`execution/runs-of-points.md` § 1.2; `engines/vibration.md`     #
#  § 5.11 § 3)                                                           #
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
    #: The call's folder below the run -- its entry's key -- as the folder
    #: door answers it (`calcdirs.where`).
    rel: str

    @property
    def index(self) -> Path:
        return _rp.index_path(self.run_dir, self.names)

    @property
    def place(self) -> "calcdirs.RunPlace":
        """The run's places, composed by the one door
        (`calcdirs.RunPlace`, `runs-of-points.md` § 2.4)."""
        return calcdirs.RunPlace.of(self.run_dir)


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
                     names=RunNames(label, rec.stage),
                     run_dir=Path(w.run_folder).resolve(),
                     root=Path(w.root).resolve(), rel=str(w.point))


def _species_index(deck_text: str) -> Mapping[str, int]:
    """Each species label's index in the deck's ``ChemicalSpeciesLabel`` --
    what a coordinates row names its species by."""
    rows = read_fdf(deck_text)[1].get(fdf_key("ChemicalSpeciesLabel")) or []
    return {str(r[2]): int(r[0]) for r in rows}


def _reading(output: Path):
    """SIESTA's one reading pass over ``output`` (`SiestaReader`)."""
    return SiestaReader().feed_text(
        Path(output).read_text(encoding="utf-8", errors="replace")).finish()


def _call_of(here: _CallHere, index: dict) -> "Call":
    """The call this folder is, as the run's index lists it -- refused by
    name where it lists none."""
    entry = _rp.entry_of(index, here.rel)
    if entry is None:
        raise FinishError(f"the run's index lists no call {here.rel!r}")
    return Call.from_entry(entry)


def _facts(index: dict, call: "Call") -> dict:
    """What the run's index keeps of ``call`` beyond what it is."""
    return (_rp.entry_of(index, call.rel) or {}).get("facts") or {}


def plan_call(deck) -> None:
    """THE PLAN STEP of one call, before its engine -- the vibration's own
    step ``before`` (`execution/runs-of-points.md` § 1.2; § 5.11 § 3): the
    call's coordinates where they are another call's result -- the force
    constants' from the `relax` call's output, written as its
    `coordinates.fdf` -- and what its index entry keeps of it: its deck's
    and coordinates' sha256, where they came from, its seed.  Its start
    and the run's `start` are its report step's."""
    here = _here(deck)
    index = _rp.read_index(here.index)
    call = _call_of(here, index)
    rule = phase(call.phase)
    coords = here.folder / COORDINATES_FILE
    came_from = "the structure, as prep wrote it"
    # A CALL CONTINUED from its own last run's warm files -- the hand-over
    # launch makes of a call whose relaunch is `continue` (§ 5.11 § 6): the
    # run it continues, said.
    continued = (read_continued_from(here.folder)
                 if rule.relaunch == "continue" else None)
    if continued is not None:
        came_from = f"its own last geometry, continued from {continued}"
    # ITS COORDINATES, where its phase's row says they come from
    # (`PhaseRule.coordinates`): planned by the step of the call that plans
    # it, or that call's last geometry step, where the run walks it.
    source = rule.coordinates
    if rule.planned:
        came_from = str(_facts(index, call).get("coordinates_from") or
                        f"planned by the {source} call's record step")
    elif source in {c.phase for c in calls_of(index)}:
        source_dir = here.place.point(Call(source, source).rel)
        out = newest_output(source_dir, here.names)
        if out is None:
            raise FinishError(f"{source_dir.name} left no output in "
                              f"{source_dir}: {call.words()} has no "
                              f"geometry to take")
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
        came_from = f"{source}/{out.name}, its last geometry step"
    if not coords.is_file():
        raise FinishError(f"{coords.name} is not beside {here.deck.name}: "
                          f"the call has no coordinates to compute at")
    seed = (here.folder / compose(here.label, rule.seed[1])
            if rule.seed is not None else None)
    seeded = seed is not None and seed.is_file()
    _rp.note(index, call.rel,
             deck_sha256=sha256_of(here.deck),
             coordinates_sha256=sha256_of(coords),
             coordinates_from=came_from,
             seed=(seed.name if seeded else None),
             seed_from=(f"{rule.seed[0]}, copied by the walk" if seeded
                        else f"its own warm files, from {continued}"
                        if continued is not None else None))
    _rp.write_index(here.index, index)
    _say("INFO", f"vibration: {call.words()} planned -- coordinates from "
                 f"{came_from}")


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


def _relaxation_outcome(record: Optional[Mapping[str, Any]],
                        retried: bool = False,
                        continued: Optional[str] = None) -> Optional[str]:
    """THE RELAXATION'S OUTCOME, as its milestone says it (`run-reports.md`
    § 2.9): whether it converged, in how many moves, and the largest force
    on the free atoms against its tolerance -- its record's own facts;
    ``None`` for a run that relaxed nothing.  The record is of the call's
    newest output, so a call the wrapper ``retried`` in place counts the
    moves of its last try, and a relaxation ``continued`` from an
    earlier run's geometry the moves since -- each said."""
    if not record:
        return None
    n = int(record["n_steps"])
    force = judged_force(record)
    said = "converged" if record["converged"] else "did not converge"
    since = ((" of its last try" if retried else "")
             + (f" continuing {continued}" if continued else ""))
    return (f"the relaxation {said} in {n} move{'' if n == 1 else 's'}"
            f"{since}: the largest force "
            f"on the free atoms {float(force):.4g} eV/Ang "
            f"(tol {float(record['force_tolerance_ev_ang']):g})")


def _force_constants_outcome(res: SpectraResults, planned: int) -> str:
    """THE FORCE CONSTANTS' OUTCOME, as their milestone says it
    (`run-reports.md` § 2.9): how many modes, and how many imaginary; the
    largest force on the free atoms at the reference geometry against the
    tolerance, and whether that makes it a stationary point -- the result's
    own verdict (its `relaxation` block, `engines/vibration.md` § 2.2); and
    the optical calls planned."""
    imag = sum(bool(m.has_imag) for m in res.modes)
    bits = [f"the force constants: {len(res.modes)} modes, "
            + (f"{imag} imaginary" if imag else "none imaginary")]
    stationary = res.relaxation or {}
    f_eh = stationary.get("max_force_eh_bohr")
    if f_eh is not None:
        tol = res.engine_metadata.get("reference_force_criterion_ev_ang")
        verdict = {True: "a stationary point",
                   False: "not a stationary point"}.get(
                       stationary.get("converged"))
        bits.append(
            f"at the reference geometry the largest force on the free "
            f"atoms {f_eh * HARTREE_BOHR_EV_ANGSTROM_ASE:.4g} eV/Ang"
            + (f" (tol {float(tol):g})" if tol is not None else "")
            + (f" -- {verdict} at this level of theory" if verdict else ""))
    if planned:
        bits.append(f"{planned} optical call{'' if planned == 1 else 's'} "
                    f"planned")
    return "; ".join(bits)


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
    stage_optical = here.place.prepared(Call(OPTICAL).prepared)
    freqs = [float(m.frequency_cm1) for m in res.modes]
    picked = select_modes(freqs, str(raman.get("modes")),
                          freq_min_cm1=raman.get("freq_min_cm1"),
                          freq_max_cm1=raman.get("freq_max_cm1"))
    delta = float(run.fc.displacement_ang)
    # THE SELECTION, AS ASKED AND AS APPLIED (§ 7), whatever it picks --
    # none picked is an answer the run states.
    index.setdefault("facts", {})["selection"] = {
        "asked": dict(raman), "applied": [int(k) for k in picked],
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
    # WHAT A CALL'S FOLDER HOLDS -- the deck and the one list of what
    # travels beside it (`runfiles.beside_the_deck`, as a launch brings a
    # run: `materialize.bring_files`), from the stage's phase folder; and
    # the calculation's shared package -- the pseudopotentials under its
    # `pseudos/`, the engine seam's rule (`jobset.engines`): SIESTA opens
    # `<label>.psml` in the folder it runs from and no other.
    files = [here.deck.name,
             *beside_the_deck(here.deck.name, VIBRATION_BUNDLE, here.names)]
    shared = sorted(f"{PSEUDO_DIR}/{p.name}" for p, _fields
                    in find_fixed(here.root / PSEUDO_DIR, PSEUDO_FILE))
    for call, positions in frames:
        folder = here.place.point(call.rel)
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
    # EACH CALL'S ENTRY in the run's index (`runpoints.add_point`): what it
    # is, what it needs -- the phases its row names -- and what plans it,
    # the force constants' call (`runs-of-points.md` § 2.3).
    planner = Call(phase(OPTICAL).planned_by, phase(OPTICAL).planned_by)
    for call, _positions in frames:
        e = point_entry(call, here.label)
        _rp.add_point(
            index, rel=e["rel"], words=e["words"], phase=e["phase"],
            round=e["round"], must_hold=e["must_hold"],
            verdict_counts=e["verdict_counts"],
            needs=[Call(n, n).rel for n in phase(OPTICAL).needs],
            planned_by=planner.rel,
            facts={**e["facts"],
                   "coordinates_sha256": sha256_of(
                       here.place.point(call.rel) / COORDINATES_FILE),
                   "coordinates_from": (
                       "the force-constant run's reference geometry"
                       if call.mode is None else
                       f"the reference geometry displaced by "
                       f"{call.side:+d} δQ along mode {call.mode}"),
                   "planned": stamp()})
    return len(frames)


def _deck_cited(deck: Path) -> dict:
    """``{name, sha256, keywords, blocks, pseudopotentials}`` -- the
    force-constant deck as the result cites it (`engines/vibration.md`
    § 6.2): every keyword and block it states, as it writes them, by their
    spelling, but its geometry (`parse.fdf.GEOMETRY_KEYS`), which the result
    states from its own source, the structure; and each species'
    pseudopotential -- the file SIESTA opened beside the deck,
    ``<label>.psml`` (`runfiles.PSEUDO_FILE`) -- by name and sha256.  The
    deck's sha256 is its file's alone."""
    spelled: dict = {}
    params, blocks = read_fdf(read_deck(deck), spelled_out=spelled)
    species = [str(row[2]) for row
               in blocks.get(fdf_key("ChemicalSpeciesLabel")) or ()]
    pseudos = [(label, fixed_name(PSEUDO_FILE, element=label))
               for label in species]
    return {"name": deck.name, "sha256": sha256_of(deck),
            "keywords": {spelled.get(key, key): " ".join(value)
                         for key, value in params.items()
                         if key not in GEOMETRY_KEYS},
            "blocks": {spelled.get(key, key): [" ".join(r) for r in rows]
                       for key, rows in blocks.items()
                       if key not in GEOMETRY_KEYS},
            "pseudopotentials": [{"species": label, "name": name,
                                  "sha256": sha256_of(deck.parent / name)}
                                 for label, name in pseudos]}


def _optical_settings(here: _CallHere, index: dict) -> dict:
    """``/settings``: every optical keyword as the stage's optical deck --
    the one each optical call copies -- writes it, the light's direction
    (its ``Optical.Vector``), the selection as asked and as applied, ``δ``,
    and the deck's whole text and sha256."""
    deck = here.place.prepared(Call(OPTICAL).prepared) / here.deck.name
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
    sel = (index.get("facts") or {}).get("selection") or {}
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
    and the build; from that try's session log, the engine's wall time
    (`wrapper_log`, the one source of it)."""
    out = newest_output(folder, here.names) if folder.is_dir() else None
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
    log = log_of_run(folder, here.label, parse(out.name, here.label).run,
                     here.names.stage)
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
    # EACH FRAME'S STATE by the run's one rule: finished where the call is
    # DONE -- its spectrum stands -- failed where it ended otherwise.
    states = {r["name"]: ("finished" if r["done"] else
                          "failed" if r["ended"] else "pending")
              for r in _rp.rows_of(index, here.run_dir)
              if r["phase"] == OPTICAL}
    atoms = res.atom_index_map_stated()
    run_rel = here.run_dir.relative_to(here.root).as_posix()
    grid = spins = version = None
    frames = []
    for c in calls:
        row = _rp.entry_of(index, c.rel) or {}
        known = row.get("facts") or {}
        folder = here.place.point(c.rel)
        state = states.get(c.name, "pending")
        facts = _call_facts(here, folder) if state != "pending" else {}
        coords = facts.get("coords")
        version = version or facts.get("siesta_version")
        f = {"call": c.rel, "mode": c.mode,
             "frequency_cm1": known.get("frequency_cm1"),
             "side": int(c.side or 0),
             "dq_amu12_ang": known.get("dq_amu12_ang"),
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
        eps = folder / compose(here.label, phase(OPTICAL).products[0])
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
    run's one done rule (`runpoints.rows_of`) -- a call that failed leaves its
    output and a partial `.FC`, and builds nothing (§ 5)."""
    if not any(r["done"] for r in _rp.rows_of(index, here.run_dir)
               if r["phase"] == FC):
        return None
    fc_dir = force_constants_call(here.run_dir)
    fc_out = newest_output(fc_dir, here.names)
    if fc_out is None:
        return None
    run = read_force_constant_run(fc_dir / here.deck.name, fc_out)
    return run, result_of(run, relaxation=_facts(
        index, Call(RELAX, RELAX)).get("relaxation"))


def finish_run(deck) -> None:
    """THE FINISH -- the vibration's own step once no call the walk may
    still run is left (`execution/runs-of-points.md` § 1.2, the kind's
    ``finish``; § 5.11 § 3, § 5): the result pair rewritten final from the
    force-constant call, with the `relax` call's record and the Raman
    phase complete, and the optical calls' data file final.  Nothing to
    write where the force constants are not done -- the run's state says
    why.  The index's closing, the finish's own entry and the run's
    `finish` are its report step's (`mb_monitor.pyz point ended`)."""
    here = _here(deck)
    index = _rp.read_index(here.index)
    got = _force_constants(here, index)
    if got is None:
        # NO FORCE CONSTANTS DONE -- the call failed, or what it builds on
        # did not run -- so no result (§ 5).
        _say("INFO", "vibration: the force-constant call is not done -- no "
                     "result to write")
        return
    run, res = got
    out = here.run_dir / compose(here.label, ROLE_SPECTRA)
    # THE OPTICAL CALLS' RECORD, final, and its citation: the phase complete
    # once every call has ended (§ 5.11 § 8).
    optical = _write_optical(here, index, res)
    dump_spectra_json(_with_raman_phase(res, run, optical, PHASE_COMPLETE),
                      out)
    _say("INFO", f"vibration: the finish wrote {out.name}")


def record_call(deck, output, exit_code: int) -> None:
    """THE RECORD STEP of one call -- the vibration's own step ``after``,
    run by its report step once its end, exit and verdict are stamped,
    success or failure (`execution/runs-of-points.md` § 1.2; § 5.11 § 3):
    the `relax` call's relaxation record and its outcome; the force
    constants' analysis -- the result pair at the run's root, the optical
    calls planned, its outcome; an optical call's data.  What it could not
    do is its call's, in the index (``step_error``), and its exit -- the
    report step says it; never the run's events, which are the report
    step's."""
    here = _here(deck)
    index = _rp.read_index(here.index)
    call = _call_of(here, index)
    output = Path(output)
    failed = None

    def _record_failed(what: str, exc: Exception) -> None:
        nonlocal failed
        _say("ERROR", f"vibration: {what} -- {type(exc).__name__}: {exc}")
        _rp.stamp_step_error(index, call.rel, what,
                             f"{type(exc).__name__}: {exc}")
        _rp.write_index(here.index, index)
        failed = failed or exc

    if call.phase == RELAX and output.is_file():
        try:
            relaxation = _relaxation_record(output)
            _rp.note(index, call.rel, relaxation=relaxation)
            _rp.write_index(here.index, index)
        except Exception as exc:                        # noqa: BLE001
            _record_failed("the relaxation record was not read", exc)
        else:
            # ITS OUTCOME IN WORDS: words that cannot be composed are
            # logged, never the call's failure.
            try:
                named = parse(output.name, here.label)
                continued = read_continued_from(here.folder)
                _rp.note(index, call.rel, outcome=_relaxation_outcome(
                    relaxation, retried=(named is not None and named.run
                                         not in (None, FIRST_TRY)),
                    continued=(str(continued) if continued is not None
                               else None)))
                _rp.write_index(here.index, index)
            except Exception as exc:                    # noqa: BLE001
                _say("WARN", f"vibration: the relaxation's outcome was not "
                             f"put in words -- {type(exc).__name__}: {exc}")
    if call.phase == FC and int(exit_code) == 0:
        try:
            # THE ANALYSIS, read once by the run's one reading of its force
            # constants, and the result written once below -- never a
            # result on disk for an analysis whose planning then failed.
            got = _force_constants(here, index)
            if got is None:
                # NOT DONE by the one rule -- not converged, or no .FC --
                # so nothing to analyse; the call's row says why (§ 5).
                why = next(r["why"] for r in _rp.rows_of(index, here.run_dir)
                           if r["rel"] == call.rel)
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
                dump_spectra_json(_with_raman_phase(res, run, optical,
                                                    PHASE_RUNNING), out)
                _rp.note(index, call.rel, modes=len(res.modes),
                         outcome=_force_constants_outcome(res, planned))
                _rp.write_index(here.index, index)
                _say("INFO", f"vibration: {len(res.modes)} mode(s) from "
                             f"{Path(run.fc.path).name}; {planned} optical "
                             f"call(s) planned; wrote {out.name}")
        except Exception as exc:                        # noqa: BLE001
            # THE ANALYSIS FAILED after a clean engine exit: nothing after
            # the force constants can run, and the run's finish says so.
            _record_failed("the force-constant analysis failed", exc)
    # AN OPTICAL CALL'S RECORD: the data file rewritten with it (§ 8).
    if call.phase == OPTICAL:
        try:
            got = _force_constants(here, index)
            if got is None:
                raise FinishError(
                    f"the force-constant call in "
                    f"{force_constants_call(here.run_dir)} is not "
                    f"done: the optical calls' record has no reference "
                    f"geometry to stand on")
            _write_optical(here, index, got[1])
        except Exception as exc:                        # noqa: BLE001
            _record_failed("the optical calls' record was not written", exc)
    if failed is not None:
        raise failed


def _say(level: str, message: str) -> None:
    """One line in the wrapper's session log, in its own format."""
    sys.stderr.write(LOG_LINE % (time.strftime(LOG_CLOCK), level, message)
                     + "\n")
    sys.stderr.flush()


_USAGE = ("vibration: usage: mb_vibration.pyz plan <deck> | record <deck> "
          "<output> --exit <rc> | finish-run <deck> | finish <deck> <output>")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The bundle's verbs: the vibration's own steps, which a call's report
    steps run in their place (`execution/runs-of-points.md` § 1.2; the
    kind table's ``steps``) -- ``plan <deck>`` before a call's engine,
    ``record <deck> <output> --exit <rc>`` after it, ``finish-run <deck>``
    once no call is left -- and ``finish <deck> <output>``, the
    force-constant analysis alone, by hand, for anything holding a finished
    force-constant call (§ 5.5).  0 when done, 1 when it could not be (the
    reason and its traceback in the session log), 2 when asked wrongly."""
    args = list(sys.argv[1:] if argv is None else argv)
    verb = args[0] if args else None
    try:
        if verb == "plan" and len(args) == 2:
            plan_call(args[1])
            return 0
        if (verb == "record" and len(args) == 5 and args[3] == "--exit"
                and args[4].lstrip("-").isdigit()):
            record_call(args[1], args[2], int(args[4]))
            return 0
        if verb == "finish-run" and len(args) == 2:
            finish_run(args[1])
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
