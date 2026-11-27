"""THE DATA FILE OF A CALCULATION'S RUNS AND THEIR STEPS -- every step its runs
finished, on one grid, each step's SCF cycles, and each run's own record, in
groups (`model/datafiles.md` § 9.5; every name it holds is there).

MODULE  stepsdata (L2; the run door's steps, the engine table's SCF columns,
        the data-file framework)
ROLE    :func:`data_file_bytes` -- the file composed now from what the runs
        hold, each read by the one reader of a run's steps (`runs.steps_at`),
        so every row stands on the person's atom as ``read_steps`` hands it
        back: written nowhere, an export hands it out -- an optimization's data
        file and a vibration's runs alike; :func:`read_data_file` -- the file
        back as data, refusing another schema by name
USED-BY optimization.export, spectra.export, postprocess (``read_steps_data``)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .ncdata import (Var, axis_values, coordinate, scf_history_values,
                     scf_history_vars)
from .persist import FileRefused
from .spectra.results import PARTITIONS

#: Which layout this is, at which version (`model/datafiles.md` § 2b).
DATA_SCHEMA = "molbuilder/run-steps@1"

#: The stress's components, in SIESTA's order: its ``Voigt[x,y,z,yz,xz,xy]``.
VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")

_OUTPUT = ("the run's own output, named by its output_file, and for "
           "SIESTA's coordinates and energies the .MD.nc history beside it, "
           "for the steps it holds")

#: The step axis and the run each step is of -- ``/steps``' and, so that
#: group read alone is whole, ``/steps/scf``'s.
_STEP = Var("step", ("step",), "coordinate", "1", "a step",
            "from 0, run by run: each geometry step a run finished, and the "
            "final geometry a SIESTA relaxation ends on", dtype="i4")
_RUN_OF_STEP = Var("run_of_step", ("step",), "given", "",
                   "the run the step is of",
                   "that run's folder, as /runs run names it", dtype=str,
                   auxiliary=True)

#: ``/runs`` -- each run's record.
RUN_VARS = (
    coordinate("atom"), coordinate("xyz"), coordinate("cell_vector"),
    Var("run", ("run",), "coordinate", "", "a run",
        "its folder below the calculation, stage by stage and each stage's "
        "starts in turn -- the ladder's order, not always the order they ran "
        "in (continued_from); a run with no step is here too, its said "
        "saying why", dtype=str),
    Var("stage_of_run", ("run",), "given", "", "the stage the run is of",
        "the stage's name", dtype=str, auxiliary=True),
    Var("continued_from", ("run",), "given", "", "the run it continued from",
        "that run's folder, as the run axis names it, by the run's own "
        "record; empty for a run that started from the structure", dtype=str),
    Var("output_file", ("run",), "state", "", "the file its steps are from",
        "the run's own output, read by molbuilder's reader of a run's steps",
        dtype=str),
    Var("final_total_energy_ev", ("run",), "derived", "eV",
        "the run's final total energy",
        "the last energy of a step it finished, as /steps total_energy_ev "
        "states it; NaN where none states one"),
    Var("force_tolerance_ev_per_ang", ("run",), "raw", "eV/angstrom",
        "the tolerance the run's relaxation was to reach",
        "as the run states it; NaN for a run that relaxed nothing",
        source=_OUTPUT),
    Var("force_within_tolerance", ("run",), "derived", "1",
        "whether the run's largest force was within its tolerance",
        "its largest absolute Cartesian force component on the atoms it "
        "moved, at the last step stating one, against its tolerance: "
        "SIESTA's own test; geomeTRIC tests each atom's force length beside "
        "other criteria, and a run still going moves on (run_state); not "
        "stated for a run that relaxed nothing", dtype="i1", flag=True),
    Var("scf_converged", ("run",), "state", "1",
        "whether its last SCF converged",
        "as the run says it; not stated where it cannot say", dtype="i1",
        flag=True, source=_OUTPUT),
    Var("scf_criteria", ("run",), "raw", "", "what its SCFs had to reach",
        "as the run states it: {phase: {residual: {tolerance, unit, "
        "required}}} (web/trajectory.md 3); empty where it states none",
        dtype=str, json=True, source=_OUTPUT),
    Var("run_state", ("run",), "state", "", "how the run ended, as its file "
        "says", "ended, stopped, out_of_memory, or running: the file shows no "
        "ending, which a slow run and one that died with no word both are",
        dtype=str),
    Var("said", ("run",), "state", "", "what the run says beside its numbers",
        "why a run holds no step here; a step still being computed or "
        "stopped inside, which is not among the steps; forces it does not "
        "compute", dtype=str),
    Var("held", ("run", "atom"), "raw", "1", "whether the run holds the atom",
        "as the run's own output says", dtype="i1", flag=True,
        source=_OUTPUT),
    Var("applied_offset_ang", ("run", "xyz"), "given", "angstrom",
        "what prep moved the structure it was handed by",
        "that structure plus this is where a first stage starts, and the deck "
        "already holds it; a later stage starts where the stage before ended; "
        "NaN where the deck records none"),
    Var("cell_ang", ("run", "cell_vector", "xyz"), "given", "angstrom",
        "the cell the run ended in",
        "a row per lattice vector: the output's own, else the box the run's "
        "deck placed the atoms in, which on an isolated axis (axis_kind) is "
        "no lattice; NaN for none"),
    Var("axis_kind", ("run", "cell_vector"), "given", "",
        "each cell vector's periodicity",
        "periodic, isolated or transport, as the run's structure states it "
        "(model/structure-periodicity.md)", dtype=str),
)

#: ``/steps`` -- every step, run after run.
STEP_VARS = (
    coordinate("atom"), coordinate("xyz"),
    Var("voigt", ("voigt",), "coordinate", "", "stress component",
        "xx, yy, zz, yz, xz or xy, SIESTA's order", dtype=str),
    _STEP,
    Var("element", ("atom",), "given", "", "the atom's element",
        "the chemical symbol, in the user order", dtype=str, auxiliary=True),
    _RUN_OF_STEP,
    Var("position_ang", ("step", "atom", "xyz"), "raw", "angstrom",
        "the atom's position",
        "the coordinates the engine ran, in the user order; they hold the "
        "move /runs applied_offset_ang states", source=_OUTPUT),
    Var("force_ev_per_ang", ("step", "atom", "xyz"), "raw", "eV/angstrom",
        "the force on the atom",
        "in the user order; NaN where the step states none", source=_OUTPUT),
    Var("total_energy_ev", ("step",), "raw", "eV", "the step's total energy",
        "SIESTA's E_KS, which its output prints to 0.0001 eV and its .MD.nc "
        "history holds in full for the steps between; PySCF's SCF total "
        "energy; NaN where the step states none, as a SIESTA relaxation's "
        "last row, the final geometry it wrote", source=_OUTPUT),
    Var("max_force_ev_per_ang", ("step",), "derived", "eV/angstrom",
        "the largest force, over every atom",
        "the largest absolute Cartesian component on any atom; NaN where the "
        "step states no force"),
    Var("max_force_free_ev_per_ang", ("step",), "derived", "eV/angstrom",
        "the largest force, over the atoms the run moved",
        "the largest absolute Cartesian component on an atom the run does "
        "not hold: what a relaxation's tolerance tests; NaN where the step "
        "states no force or the run holds every atom"),
    *(Var(f"{name}_charge_e", ("step", "atom"), "raw", "e",
          f"the atom's {name.capitalize()} charge",
          "its valence charge less the electrons the partition gives it -- "
          "positive where electrons left it -- in the user order; NaN where "
          "the run printed none, SIESTA printing them where its deck asks",
          source=_OUTPUT)
      for name in PARTITIONS),
    Var("stress_kbar", ("step", "voigt"), "raw", "kbar",
        "the step's stress tensor",
        "as the run printed it; NaN where it printed none", source=_OUTPUT),
)


def _scf_vars(columns) -> Tuple[Var, ...]:
    """``/steps/scf`` -- each step's SCF, the framework's one layout of an
    SCF history (`ncdata.scf_history_vars`), ``columns`` its engine's: the
    step axis and its run beside it, so the group read alone is whole."""
    return (_STEP, _RUN_OF_STEP,
            coordinate("scf_cycle", more="the step's own rows"),
            *scf_history_vars(columns, ("step",), whose="the step",
                              source="the run's own output, named by its "
                                     "output_file"))


class DataFileError(ValueError):
    """A calculation whose runs do not lay out on the file's one grid, or a
    file that is not this one -- the message names which."""


class DataFileUnreadable(DataFileError, FileRefused):
    """A file read back that is not this one at a version this molbuilder
    reads (:func:`read_data_file`): a file its reader cannot read, carrying
    the mark the two edges say (`persist.FileRefused`,
    `execution/job-contracts.md` § 6.1).  The writer's checks carry none."""


def _runs_read(calc) -> List[Dict[str, Any]]:
    """Each run of ``calc``: its folder below the calculation, its stage, what
    it continued from, its steps (`runs.steps_at`) -- ``None`` where it gives
    none, ``why`` then saying why, in the door's words."""
    from .jobset.runstatus import runs_of
    from .runrecord import read_continued_from
    from .runs import RunStepsError, steps_at
    base = Path(calc.base)
    read = []
    for stage, folder in runs_of(calc):
        continued = read_continued_from(folder)
        row = {"run": folder.relative_to(base).as_posix(), "stage": stage,
               "continued_from": "" if continued is None else str(continued),
               "steps": None, "why": ""}
        try:
            row["steps"] = steps_at(folder, named=row["run"])
        except RunStepsError as exc:
            row["why"] = str(exc)
        read.append(row)
    return read


def data_file_bytes(calc, name: str) -> Tuple[bytes, str]:
    """``(bytes, said)``: the data file ``name`` of the opened calculation
    ``calc``'s runs and their steps as they stand (`model/datafiles.md`
    § 9.5), written nowhere, its title naming the calculation by its kind --
    and which runs give no step and why, ``""`` when all do: such a run
    stands on the ``run`` axis at its fill, so a run that died is never read
    as one that was not asked.  :class:`DataFileError` names a calculation
    with no step at all, and runs that are not of one structure."""
    from . import __version__
    from .engine_table import engine as _engine_row, scf_columns
    from .ncdata import NC_LOCK, as_bytes, empty, flag, in_memory, state, write_group
    read = _runs_read(calc)
    gave = [row for row in read if row["steps"] is not None]
    if not gave:
        raise DataFileError(f"{Path(calc.base).name} holds no step a run has "
                            f"finished yet")
    first = gave[0]["steps"]
    atoms = first.atom_index_map
    for row in gave:
        steps = row["steps"]
        if (steps.elements != first.elements
                or steps.atom_index_map.user != atoms.user
                or steps.atom_index_map.engine != atoms.engine):
            raise DataFileError(f"{row['run']} is not of the atoms of "
                                f"{gave[0]['run']}: one calculation is one "
                                f"structure")
    n_steps = sum(len(row["steps"].energy_ev) for row in gave)
    n_cycles = max([1] + [len(r) for row in gave
                          for r in row["steps"].scf_rows])
    engine = str(calc.task.engine)
    columns = scf_columns(engine)
    scf_vars = _scf_vars(columns)
    run_dims = {"atom": atoms, "xyz": 3, "cell_vector": 3, "run": len(read)}
    step_dims = {"atom": atoms, "xyz": 3, "voigt": len(VOIGT),
                 "step": n_steps}
    scf_dims = {"step": n_steps, "scf_cycle": n_cycles}
    runs = {v.name: empty(v, run_dims) for v in RUN_VARS}
    steps_v = {v.name: empty(v, step_dims) for v in STEP_VARS}
    scf = {v.name: empty(v, scf_dims) for v in scf_vars}
    for values in (runs, steps_v):
        for axis in ("atom", "xyz"):
            values[axis][:] = axis_values(axis, atoms.n_atoms)
    runs["cell_vector"][:] = axis_values("cell_vector", 3)
    steps_v["voigt"][:] = list(VOIGT)
    runs["run"][:] = [row["run"] for row in read]
    runs["stage_of_run"][:] = [row["stage"] for row in read]
    runs["continued_from"][:] = [row["continued_from"] for row in read]
    runs["said"][:] = [row["why"] for row in read]
    steps_v["step"][:] = np.arange(n_steps)
    steps_v["element"][:] = list(first.elements)
    scf["step"][:] = np.arange(n_steps)
    scf["scf_cycle"][:] = np.arange(n_cycles)
    at = 0
    for r, row in enumerate(read):
        steps = row["steps"]
        if steps is None:
            continue
        k = len(steps.energy_ev)
        where = slice(at, at + k)
        steps_v["run_of_step"][where] = row["run"]
        scf["run_of_step"][where] = row["run"]
        steps_v["position_ang"][where] = steps.positions_ang
        steps_v["force_ev_per_ang"][where] = steps.forces_ev_per_ang
        steps_v["total_energy_ev"][where] = steps.energy_ev
        steps_v["max_force_ev_per_ang"][where] = steps.max_force_ev_per_ang
        steps_v["max_force_free_ev_per_ang"][where] = (
            steps.max_force_free_ev_per_ang)
        steps_v["stress_kbar"][where] = steps.stress_voigt_kbar
        for partition, rows in steps.charges_e.items():
            steps_v[f"{partition}_charge_e"][where] = rows
        # EACH STEP'S SCF ROWS, the framework's one layout; a step that
        # ran none -- a SIESTA relaxation's final geometry -- keeps the
        # count's fill, -1.
        for s, cycles in enumerate(steps.scf_rows):
            if cycles:
                scf_history_values(scf, columns, cycles, (at + s,))
        at += k
        runs["output_file"][r] = steps.output_file
        record = steps.relaxation or {}
        if steps.final_energy_ev is not None:
            runs["final_total_energy_ev"][r] = steps.final_energy_ev
        if record:
            runs["force_tolerance_ev_per_ang"][r] = record[
                "force_tolerance_ev_ang"]
            runs["force_within_tolerance"][r] = flag(record["converged"])
        runs["scf_converged"][r] = flag(steps.scf_converged)
        runs["scf_criteria"][r] = steps.scf_criteria
        runs["run_state"][r] = steps.run_state
        runs["said"][r] = "; ".join(steps.said)
        runs["held"][r] = [flag(i in steps.held_atoms)
                           for i in range(atoms.n_atoms)]
        if steps.applied_offset_ang is not None:
            runs["applied_offset_ang"][r] = steps.applied_offset_ang
        if steps.last.cell is not None:
            runs["cell_ang"][r] = np.asarray(steps.last.cell, dtype=float)
        if steps.last.axis_kind is not None:
            runs["axis_kind"][r] = list(steps.last.axis_kind)
    with NC_LOCK:
        ds = in_memory(name)
        state(ds, title=(f"the {calc.task.calculation} {calc.task.label}: "
                         f"every step its runs finished, each step's SCF "
                         f"cycles, and each run's own record"),
              schema=DATA_SCHEMA,
              molbuilder_version=__version__,
              derived_from=(f"{_engine_row(engine).words}'s output of each "
                            f"run, named by its output_file"))
        ds.setncatts({"label": str(calc.task.label), "engine": engine})
        write_group(ds.createGroup("runs"), run_dims, RUN_VARS, runs,
                    title="each run's own record: how it ended, what it "
                          "held, what its SCFs and its relaxation had to "
                          "reach")
        g = ds.createGroup("steps")
        write_group(g, step_dims, STEP_VARS, steps_v,
                    title="every step the runs finished, run after run: "
                          "each atom's position and force, the energy, the "
                          "stress, the largest force")
        write_group(g.createGroup("scf"), scf_dims, scf_vars, scf,
                    title="each step's SCF: how many rows it holds, and "
                          "each row by the columns its engine declares")
        data = as_bytes(ds)
    return data, "; ".join(f"{row['run']} gives no step: {row['why']}"
                           for row in read if row["steps"] is None)


def read_data_file(path) -> Dict[str, Any]:
    """The file as data: ``{"attrs": {...}, "groups": {path: {"attrs",
    "variables"}}}`` -- ``runs``, ``steps``, ``steps/scf`` -- each variable's
    values as numpy arrays, its fill as written (NaN for a real number, -1
    for a count -- ``scf_cycles`` where none ran), a flag as stored (1, 0,
    -1 not stated; xarray masks both as NaN), a record as the record
    (`model/datafiles.md` § 3a), with its units, long name, definition and
    source -- the framework's one read of a file (`ncdata.read_file`).
    Refuses another schema by name."""
    from .ncdata import read_file
    got = read_file(path, schemas=(DATA_SCHEMA,), error=DataFileUnreadable,
                    way="its export writes it again")
    return {"attrs": got["attrs"],
            "groups": {k: v for k, v in got["groups"].items() if k}}


__all__ = ["DATA_SCHEMA", "DataFileError", "RUN_VARS", "STEP_VARS",
           "VOIGT", "data_file_bytes", "read_data_file"]
