"""A SIESTA vibration, finished by its own job: what the force-constant run left -> the result.

MODULE  spectra.siesta_vibration (L2; numpy, ASE, the standard library and its
        siblings -- nothing that needs the package installed)
ROLE    the SIESTA route's FINISH (`engines/vibration.md` § 5.5): read a
        finished force-constant run and the deck that ran it, each record
        through the reader that owns it, run the one analysis, and write
        ``<label>.spectra.json`` beside the run
USED-BY the wrapper of a SIESTA force-constant stage, which runs
        ``python mb_vibration.pyz <deck> <output>`` after SIESTA exits
        cleanly (`runwrap.VIBRATION_BUNDLE`); :func:`finish` is the same door
        for anything holding a finished attempt
TRAVELS in ``mb_vibration.pyz`` (`runwrap.VIBRATION_COMPANIONS`)

SIESTA takes no second derivative.  Its force-constant run nudges the free
atoms and writes ``<SystemLabel>.FC``; everything after that is the harmonic
analysis both engines share (`spectra.vibrational_analysis`).  A vibration
calculation ends with its result on both engines (user, 2026-09-28), so the
SIESTA job runs that analysis itself, in the attempt, with the job's own
python -- there is no step after the launch.

EVERY INPUT IS A RECORD THE ATTEMPT HOLDS, and each is read by its owner:

  * the deck (``SystemLabel``, the species, the FC range, the lattice) --
    the fdf reader, `parse.fdf`;
  * the deck's molbuilder blocks -- their one reader, `deck_record`:
    ``engine-offset`` (the axis kinds, R3) and ``vibration`` (the
    stationarity criterion, the thermochemistry's temperature, the person's
    statement, the ladder's relaxation record, the stage, the molbuilder
    that prepped it);
  * ``<label>.FC`` -- `parse.engines.siesta_fc`;
  * the run's output (FC step 0's geometry and forces, SIESTA's version) --
    SIESTA's one reading pass, `siesta_reader`;
  * ``atom-permutation.json``, the copy in the attempt -- `atom_permutation`;
  * the masses -- ASE's standard weights by atomic number, the table
    `chemistry.atomic_mass` reads (I3).

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
# ASE AT LOAD, not where the masses are read: the wrapper asks whether the
# finish loads before the engine starts (`mb_vibration.pyz loads`), and a job
# env without ASE must fail that question, not the expensive run's end.
from ase.data import atomic_masses

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from ..atom_permutation import Permutation, read_permutation
    from ..constants import BOHR_ANGSTROM, HARTREE_BOHR_EV_ANGSTROM_ASE
    from ..parse.engines.siesta_fc import (ForceConstantFile,
                                           fc_block_asymmetry,
                                           hessian_from_fc, read_fc,
                                           reference_frame_of)
    from ..parse.engines.siesta_reader import SiestaReader
    from ..parse.fdf import _norm, _parse_fdf, parse_fdf_params, system_label
    from ..deck_record import extract_engine_offset, extract_vibration_record
    from ..engine_atom_index import from_engine_index, siesta_atom_index
    from ..sidecars.spectra import dump_spectra_json
    from ..wrapper_log import LOG_CLOCK, LOG_LINE
    from .methods import extract_citation_keys, siesta_methods_text
    from .results import SpectraResults
    from .vibrational_analysis import vibrational_analysis
except ImportError:                         # beside a job, in mb_vibration.pyz
    from atom_permutation import Permutation, read_permutation
    from constants import BOHR_ANGSTROM, HARTREE_BOHR_EV_ANGSTROM_ASE
    from siesta_fc import (ForceConstantFile, fc_block_asymmetry,
                           hessian_from_fc, read_fc, reference_frame_of)
    from siesta_reader import SiestaReader
    from fdf import _norm, _parse_fdf, parse_fdf_params, system_label
    from deck_record import extract_engine_offset, extract_vibration_record
    from engine_atom_index import from_engine_index, siesta_atom_index
    from spectra_sidecar import dump_spectra_json
    from wrapper_log import LOG_CLOCK, LOG_LINE
    from methods import extract_citation_keys, siesta_methods_text
    from results import SpectraResults
    from vibrational_analysis import vibrational_analysis

#: What the result says about how the geometry was reached: the run it
#: finishes relaxes nothing (`engines/vibration.md` § 5.5).
GEOMETRY_NOTE = ("the force-constant run relaxes nothing: its reference "
                 "geometry is taken as the stationary point")


def vibration_record(*, stage: str, force_criterion_ev_ang: Optional[float],
                     already_relaxed: bool,
                     relaxation: Optional[Mapping[str, Any]],
                     temperature_K: float,
                     molbuilder_version: str) -> dict:
    """The deck's ``vibration`` block (`engines/vibration.md` § 5.3): what
    the finish reads back and no SIESTA keyword states.  Built by `prep`,
    which holds every fact -- the stage it prepares, the stage's resolved
    config (``relax_force_tol``, ``already_relaxed``), the `relax` stage's
    relaxation record it read the coordinates from
    (`parse.contract.relaxation_of_output`, ``None`` without one), the
    thermochemistry's temperature, and the molbuilder that renders the
    deck -- and placed in the deck as given
    (`script_emit.emit_vibration_record`).  Its keys are read by
    :func:`result_of` -- and, before the deck is written, its
    ``relaxation`` by the stage's own checks (`validate`'s ``prior``,
    handed in by the SIESTA spec)."""
    return {"stage": str(stage),
            "force_criterion_ev_ang": (None if force_criterion_ev_ang is None
                                       else float(force_criterion_ev_ang)),
            "already_relaxed": bool(already_relaxed),
            "relaxation": (dict(relaxation) if relaxation is not None
                           else None),
            "temperature_K": float(temperature_K),
            "molbuilder_version": str(molbuilder_version)}


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
    permutation: Permutation
    fc: ForceConstantFile
    siesta_version: str
    #: the deck's ``vibration`` block (`script_emit.emit_vibration_record`)
    record: Mapping[str, Any]

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
    text = deck.read_text(encoding="utf-8", errors="replace")
    label = system_label(text)
    if not label:
        raise FinishError(f"{deck.name} states no SystemLabel")
    scalars, blocks = _parse_fdf(text)

    # THE ATOMS, as the deck wrote them: each row's species index names a
    # (Z, label) pair in ChemicalSpeciesLabel.
    species = {}
    for row in blocks.get(_norm("ChemicalSpeciesLabel")) or ():
        if len(row) >= 3:
            species[int(row[0])] = (int(row[1]), row[2])
    rows = blocks.get(_norm("AtomicCoordinatesAndAtomicSpecies")) or ()
    try:
        picked = [species[int(r[3])] for r in rows]
    except (KeyError, IndexError, ValueError):
        raise FinishError(
            f"{deck.name}: an atom names a species ChemicalSpeciesLabel "
            f"does not define") from None
    if not picked:
        raise FinishError(f"{deck.name} holds no atoms")
    atomic_numbers = tuple(z for z, _ in picked)
    elements = tuple(lab for _, lab in picked)
    n = len(elements)

    # WHICH ATOMS WERE NUDGED: the range the run was told.
    try:
        first = int(scalars[_norm("FC.First")][0])
        last = int(scalars[_norm("FC.Last")][0])
    except (KeyError, IndexError, ValueError):
        raise FinishError(f"{deck.name} states no FC.First / FC.Last: it is "
                          f"not a force-constant deck") from None
    # SIESTA's atom numbers to the deck's 0-based positions, through the one
    # door for engine numbering (`model/overview.md` § 2); every check below
    # is on positions.
    lo, hi = from_engine_index(first, "siesta"), from_engine_index(last, "siesta")
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

    offset = extract_engine_offset(text)
    if not offset or len(offset.get("axis_kind") or ()) != 3:
        raise FinishError(
            f"{deck.name} carries no engine-offset block, so nothing says "
            f"which axes repeat: re-prep the stage")
    record = extract_vibration_record(text)
    if record is None:
        raise FinishError(
            f"{deck.name} carries no vibration block -- the stationarity "
            f"criterion and the ladder's relaxation are not recorded: "
            f"re-prep the stage")
    cell = parse_fdf_params(text, source=deck.name).cell_ang

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

    fc_path = attempt / f"{label}.FC"
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
        permutation=read_permutation(attempt),
        fc=read_fc(fc_path),
        siesta_version=str(build.get("version") or ""),
        record=record)


def result_of(run: ForceConstantRun) -> SpectraResults:
    """The run's result: its block, masses and geometry through the one
    analysis (`spectra.vibrational_analysis`), with what this route says of
    itself -- the Methods paragraph, the force-constant facts, the stage."""
    n = len(run.elements)
    if run.fc.n_atoms != n:
        raise FinishError(f"{Path(run.fc.path).name} describes "
                          f"{run.fc.n_atoms} atoms; {run.deck.name} has {n}")
    H = hessian_from_fc(run.fc, run.displaced)
    rec = run.record
    criterion = rec.get("force_criterion_ev_ang")
    if rec.get("temperature_K") is None:
        raise FinishError(
            f"{run.deck.name}'s vibration block states no temperature_K -- "
            f"a deck prepared before its finish summed at the template's "
            f"temperature (2026-09-28): prepare the stage again")
    res = vibrational_analysis(
        H, [float(atomic_masses[z]) for z in run.atomic_numbers],
        run.positions_ang, run.elements, run.held,
        axis_kind=run.axis_kind, cell=run.cell, permutation=run.permutation,
        label=run.label, engine="siesta", engine_version=run.siesta_version,
        molbuilder_version=str(rec.get("molbuilder_version") or ""),
        geometry_note=GEOMETRY_NOTE,
        reference_forces_ev_ang=run.reference_forces_ev_ang,
        force_criterion_ev_ang=criterion,
        already_relaxed=bool(rec.get("already_relaxed")),
        ladder_relaxation=rec.get("relaxation"),
        temperature_K=float(rec["temperature_K"]),
        config={"engine": "siesta", "calculation": "vibration",
                "stage": rec.get("stage")},
        engine_metadata={
            "fc_file": Path(run.fc.path).name,
            "fc_displacement_ang": run.fc.displacement_ang,
            "fc_range_1based": [siesta_atom_index(run.displaced[0]),
                                siesta_atom_index(run.displaced[-1])],
            # The block's own numerics, before the symmetrisation hides them
            # (`engines/vibration.md` § 5.5).
            "fc_asymmetry_max_ev_ang2": fc_block_asymmetry(run.fc,
                                                           run.displaced),
            "reference_force_criterion_ev_ang": (
                None if criterion is None else float(criterion))})
    # THE METHODS PARAGRAPH, once the analysis has said how many motions it
    # removed -- its own count, not a second derivation of it (R1).
    methods = siesta_methods_text(
        displacement_bohr=run.fc.displacement_ang / BOHR_ANGSTROM,
        n_free=len(run.displaced), n_held=len(run.held),
        n_rigid=int(res.removed_motions["count"]),
        siesta_version=run.siesta_version)
    return replace(res, methods_text=methods,
                   bibliography_keys=extract_citation_keys(methods))


def finish(deck, output) -> Path:
    """Finish the force-constant run ``deck`` ran (its output ``output``):
    write ``<label>.spectra.json`` beside it and return its path.  The job's
    last step, and the same door for anything holding a finished attempt."""
    return _finished(deck, output)[2]


def _finished(deck, output):
    run = read_force_constant_run(deck, output)
    res = result_of(run)
    out = Path(deck).parent / f"{run.label}.spectra.json"
    dump_spectra_json(res, out)
    return run, res, out


def _say(level: str, message: str) -> None:
    """One line in the wrapper's session log, in its own format."""
    sys.stderr.write(LOG_LINE % (time.strftime(LOG_CLOCK), level, message)
                     + "\n")
    sys.stderr.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python mb_vibration.pyz <deck> <output>`` -- 0 when the result is
    written, 1 when it could not be (the reason and its traceback in the
    session log), 2 when it was asked wrongly."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        _say("ERROR", "vibration: usage: mb_vibration.pyz <deck> <output>")
        return 2
    deck, output = args
    try:
        run, res, out = _finished(deck, output)
    except Exception as exc:                            # noqa: BLE001
        import traceback
        _say("ERROR", f"vibration: the modes could not be derived from "
                      f"{Path(output).name} -- {type(exc).__name__}: {exc}")
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
