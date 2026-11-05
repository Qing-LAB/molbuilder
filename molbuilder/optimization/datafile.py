"""THE DATA FILE OF AN OPTIMIZATION -- ``<label>.optimization.nc``: every step
its runs finished, on one grid, and each run's own record
(`engines/optimization-postprocessing.md` § 6; `model/datafiles.md` § 1).

MODULE  optimization.datafile (L4)
ROLE    :func:`data_file_bytes` -- the file composed now from what the runs
        hold, each read by the one reader of a run's steps (`runs.steps_at`),
        so every row stands on the person's atom as ``read_steps`` hands it
        back: written nowhere, an export hands it out; :func:`read_data_file`
        -- the file back as data, refusing another schema by name
USED-BY optimization.export, postprocess
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..ncdata import Var
from ..spectra.results import PARTITIONS

#: What an optimization's data file is named on its label.
DATA_FILE = ".optimization.nc"

#: Which file this is, at which version (`model/datafiles.md` § 2b).
DATA_SCHEMA = "molbuilder/optimization-data@1"

#: The stress's components, in SIESTA's order: its ``Voigt[x,y,z,yz,xz,xy]``.
VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")

_OUTPUT = ("the run's own output, named by output_file, and for SIESTA's "
           "coordinates and energies the .MD.nc history beside it, for the "
           "steps it holds")

_VARS = (
    Var("atom", ("atom",), "coordinate", "1", "the atom",
        "its index by the file's atom index map, from 0", dtype="i4"),
    Var("xyz", ("xyz",), "coordinate", "", "Cartesian component", "x, y or z",
        dtype=str),
    Var("cell_vector", ("cell_vector",), "coordinate", "", "cell vector",
        "a, b or c", dtype=str),
    Var("voigt", ("voigt",), "coordinate", "", "stress component",
        "xx, yy, zz, yz, xz or xy, SIESTA's order", dtype=str),
    Var("run", ("run",), "coordinate", "", "a run",
        "its folder below the calculation, rung by rung and each rung's starts "
        "in turn -- the ladder's order, not always the order they ran in "
        "(continued_from); a run with no step is here too, said says why",
        dtype=str),
    Var("step", ("step",), "coordinate", "1", "a step",
        "from 0, run by run: each geometry step a run finished, and the final "
        "geometry a SIESTA relaxation ends on", dtype="i4"),
    Var("element", ("atom",), "given", "", "the atom's element",
        "the chemical symbol, in the user order", dtype=str, auxiliary=True),
    Var("run_of_step", ("step",), "given", "", "the run the step is of",
        "that run's folder, as the run axis names it", dtype=str,
        auxiliary=True),
    Var("rung_of_run", ("run",), "given", "", "the rung the run is of",
        "the stage's name", dtype=str, auxiliary=True),
    Var("continued_from", ("run",), "given", "", "the run it continued from",
        "that run's folder, as the run axis names it, by the run's own record; "
        "empty for a run that started from the structure", dtype=str),
    Var("output_file", ("run",), "state", "", "the file its steps are from",
        "the run's own output, read by molbuilder's reader of a run's steps",
        dtype=str),
    Var("position_ang", ("step", "atom", "xyz"), "raw", "angstrom",
        "the atom's position",
        "the coordinates the engine ran, in the user order; they hold the "
        "move applied_offset_ang states", source=_OUTPUT),
    Var("force_ev_per_ang", ("step", "atom", "xyz"), "raw", "eV/angstrom",
        "the force on the atom",
        "in the user order; NaN where the step states none", source=_OUTPUT),
    Var("energy_ev", ("step",), "raw", "eV", "the step's energy",
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
    Var("final_energy_ev", ("run",), "derived", "eV", "the run's final energy",
        "the last energy of a step it finished, as energy_ev states it; NaN "
        "where none states one"),
    Var("force_tolerance_ev_per_ang", ("run",), "raw", "eV/angstrom",
        "the tolerance the run's relaxation was to reach",
        "as the run states it; NaN for a run that relaxed nothing",
        source=_OUTPUT),
    Var("force_within_tolerance", ("run",), "derived", "1",
        "whether the run's largest force was within its tolerance",
        "its largest absolute Cartesian force component on the atoms it moved, "
        "at the last step stating one, against its tolerance: SIESTA's own "
        "test; geomeTRIC tests each atom's force length beside other criteria, "
        "and a run still going moves on (run_state); not stated for a run that "
        "relaxed nothing",
        dtype="i1", flag=True),
    Var("scf_converged", ("run",), "raw", "1", "whether its last SCF converged",
        "as the run says it; not stated where it cannot say", dtype="i1",
        flag=True, source=_OUTPUT),
    Var("run_state", ("run",), "state", "", "how the run ended, as its file says",
        "ended, stopped, out_of_memory, or running: the file shows no ending, "
        "which a slow run and one that died with no word both are", dtype=str),
    Var("said", ("run",), "state", "", "what the run says beside its numbers",
        "why a run holds no step here; a step still being computed or stopped "
        "inside, which is not among the steps; forces it does not compute",
        dtype=str),
    Var("held", ("run", "atom"), "raw", "1", "whether the run holds the atom",
        "as the run's own output says", dtype="i1", flag=True, source=_OUTPUT),
    Var("applied_offset_ang", ("run", "xyz"), "given", "angstrom",
        "what prep moved the structure it was handed by",
        "that structure plus this is where a first rung starts, and the deck "
        "already holds it; a later rung starts where the rung before ended; "
        "NaN where the deck records none"),
    Var("cell_ang", ("run", "cell_vector", "xyz"), "given", "angstrom",
        "the cell the run ended in",
        "a row per lattice vector: the output's own, else the box the run's "
        "deck placed the atoms in, which on an isolated axis (axis_kind) is no "
        "lattice; NaN for none"),
    Var("axis_kind", ("run", "xyz"), "given", "", "each axis's kind",
        "periodic, isolated or as the run's structure states it", dtype=str),
)


class DataFileError(ValueError):
    """An optimization that does not lay out on the file's one grid, or a file
    that is not this one -- the message names which."""


def data_file_name(calc) -> str:
    """``<label>.optimization.nc``."""
    return f"{calc.task.label}{DATA_FILE}"


def _runs_read(calc) -> List[Dict[str, Any]]:
    """Each run of ``calc``: its folder below the calculation, its rung, what it
    continued from, its steps (`runs.steps_at`) -- ``None`` where it gives none,
    ``why`` then saying why, in the door's words."""
    from ..jobset.runstatus import runs_of
    from ..runrecord import read_continued_from
    from ..runs import RunStepsError, steps_at
    base = Path(calc.base)
    read = []
    for rung, folder in runs_of(calc):
        continued = read_continued_from(folder)
        row = {"run": folder.relative_to(base).as_posix(), "rung": rung,
               "continued_from": "" if continued is None else str(continued),
               "steps": None, "why": ""}
        try:
            row["steps"] = steps_at(folder)
        except RunStepsError as exc:
            row["why"] = str(exc)
        read.append(row)
    return read


def data_file_bytes(calc) -> Tuple[str, bytes, str]:
    """``(name, bytes, said)``: ``<label>.optimization.nc`` of the opened
    calculation ``calc`` as it stands, written nowhere, and which runs give no
    step and why, ``""`` when all do: such a run stands on the file's ``run``
    axis at its fill, so a rung that died is never read as a rung that was not
    asked.  :class:`DataFileError` names a calculation with no step at all, and
    runs that are not of one structure."""
    from .. import __version__
    from ..ncdata import NC_LOCK, as_bytes, empty, flag, in_memory, state, write_group
    read = _runs_read(calc)
    name = data_file_name(calc)
    gave = [row for row in read if row["steps"] is not None]
    if not gave:
        raise DataFileError(f"{Path(calc.base).name} holds no step a run has "
                            f"finished yet: there is nothing to export")
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
    dims = {"atom": atoms, "xyz": 3, "cell_vector": 3, "voigt": len(VOIGT),
            "run": len(read), "step": n_steps}
    values = {v.name: empty(v, dims) for v in _VARS}
    values["atom"][:] = np.arange(atoms.n_atoms)
    values["xyz"][:] = ["x", "y", "z"]
    values["cell_vector"][:] = ["a", "b", "c"]
    values["voigt"][:] = list(VOIGT)
    values["run"][:] = [row["run"] for row in read]
    values["rung_of_run"][:] = [row["rung"] for row in read]
    values["continued_from"][:] = [row["continued_from"] for row in read]
    values["said"][:] = [row["why"] for row in read]
    values["step"][:] = np.arange(n_steps)
    values["element"][:] = list(first.elements)
    at = 0
    for r, row in enumerate(read):
        steps = row["steps"]
        if steps is None:
            continue
        k = len(steps.energy_ev)
        where = slice(at, at + k)
        values["run_of_step"][where] = row["run"]
        values["position_ang"][where] = steps.positions_ang
        values["force_ev_per_ang"][where] = steps.forces_ev_per_ang
        values["energy_ev"][where] = steps.energy_ev
        values["max_force_ev_per_ang"][where] = steps.max_force_ev_per_ang
        values["max_force_free_ev_per_ang"][where] = (
            steps.max_force_free_ev_per_ang)
        values["stress_kbar"][where] = steps.stress_voigt_kbar
        for partition, rows in steps.charges_e.items():
            values[f"{partition}_charge_e"][where] = rows
        at += k
        values["output_file"][r] = steps.output_file
        record = steps.relaxation or {}
        if steps.final_energy_ev is not None:
            values["final_energy_ev"][r] = steps.final_energy_ev
        if record:
            values["force_tolerance_ev_per_ang"][r] = record[
                "force_tolerance_ev_ang"]
            values["force_within_tolerance"][r] = flag(record["converged"])
        values["scf_converged"][r] = flag(steps.scf_converged)
        values["run_state"][r] = steps.run_state
        values["said"][r] = "; ".join(steps.said)
        values["held"][r] = [flag(i in steps.held_atoms)
                             for i in range(atoms.n_atoms)]
        if steps.applied_offset_ang is not None:
            values["applied_offset_ang"][r] = steps.applied_offset_ang
        if steps.last.cell is not None:
            values["cell_ang"][r] = np.asarray(steps.last.cell, dtype=float)
        if steps.last.axis_kind is not None:
            values["axis_kind"][r] = list(steps.last.axis_kind)
    with NC_LOCK:
        ds = in_memory(name)
        state(ds, title=(f"the optimization {calc.task.label}: every step its "
                         f"runs finished, each atom's position and force, and "
                         f"each run's own record"),
              schema=DATA_SCHEMA, molbuilder_version=__version__,
              source=(f"derived from {calc.task.engine}'s output of each run, "
                      f"named by output_file"))
        ds.setncatts({"label": str(calc.task.label),
                      "engine": str(calc.task.engine)})
        write_group(ds, dims, _VARS, values)
        data = as_bytes(ds)
    return name, data, "; ".join(f"{row['run']} gives no step: {row['why']}"
                                 for row in read if row["steps"] is None)


def read_data_file(path) -> Dict[str, Any]:
    """``<label>.optimization.nc`` as data: ``{"attrs": {...}, "variables":
    {name: {"dims", "data", "attrs"}}}`` -- each variable's values as numpy
    arrays, a number's fill NaN and a flag ``True``, ``False`` or ``None`` where
    not stated (`ncdata.unflag`), its units, long name, definition and source.
    Refuses another schema by name."""
    from ..ncdata import read_variables, reading, unflag
    with reading(path) as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        if attrs.get("schema") != DATA_SCHEMA:
            raise DataFileError(f"{Path(path).name} states the schema "
                                f"{attrs.get('schema')!r}, not {DATA_SCHEMA}")
        variables = read_variables(ds)
    for var in variables.values():
        if "flag_values" in var["attrs"]:
            var["data"] = np.vectorize(unflag, otypes=[object])(var["data"])
    return {"attrs": attrs, "variables": variables}


__all__ = ["DATA_FILE", "DATA_SCHEMA", "DataFileError", "VOIGT",
           "data_file_bytes", "data_file_name", "read_data_file"]
