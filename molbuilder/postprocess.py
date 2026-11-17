"""THE POST-PROCESSING LAYER -- one import for a person's script or notebook
(`model/parse.md` § 3, *One layer*; `model/overview.md` § 2.2b).

MODULE  postprocess (above the parse package and the run door; gathers what
        is already there, a second reader of nothing)
ROLE    ``parse`` -- an engine's file, a run's file → its result (the
        registry's), in the file's own order, the map on the answer
        (`atom_index.parsed`);
        ``read_steps`` -- a run's steps, in the user order; ``list_runs`` -- a
        calculation's runs;
        ``read_structure`` -- a structure pair, in the user
        order; ``atom_index_map`` -- which index a file's
        atoms are listed by, and both indices of each (`atom_index`);
        ``read_transport_data`` / ``read_spectra_data`` /
        ``read_optical_data`` -- the data files a calculation writes, each
        with its atom index map, and ``read_steps_data`` -- an exported
        calculation's runs and their steps -- ``read_raman_data`` -- an exported Raman view;
        ``read_chart_data`` -- what one chart drew, its own Export's file; ``infrared_view`` -- the free atoms' infrared from a
        SIESTA vibration result's charges (`spectra.infrared`);
        ``raman_view`` -- a SIESTA vibration's Raman at a laser, from its
        optical calls (`spectra.raman`); ``pdos`` -- a transport point's PDOS
        of a mixed selection from its data file and digests, summed as the
        report sums (`tbtnc.pdos_from_files`); ``reaveraged`` -- a frame
        set's transport calculation averaged over another motion
        (`frame_average.reaveraged`); ``outermost_move`` -- how far
        a mode's frame set at a temperature moves its atom that moves most
        (`spectra.mode_frames`); ``read_xv`` / ``read_xv_cell`` -- SIESTA's
        own ``.XV``, IN THE ENGINE'S ORDER, as its docstring says; and where
        a calculation is: ``projects_root``, ``project_dir``, ``topic_dir``,
        ``structure_dir``, ``list_projects``, ``list_topics``,
        ``list_structures`` (`projects`)
        -- each name handing its question to the doors that own it, no
        rule of the layer's own, a keyword unsaid left to its owner's
        default (`model/parse.md` § 3)
USED-BY a notebook's kernel, which imports the molbuilder its server runs
        (`web/jupyter.md` § 7), and the post-processing documents' examples

    from molbuilder.postprocess import parse, atom_index_map

It imports with the notebook env's packages alone -- numpy, netCDF4, sisl,
xarray -- and molbuilder's own doors ask the same functions at their own
floors.
"""
from __future__ import annotations

from typing import Any, Dict

from .atom_index import AtomIndexUnknown, atom_index_map
from .atom_permutation import MAP_KEY, AtomIndexMap
from .parse.coords import read_xv, read_xv_cell
from .projects import (list_projects, list_structures, list_topics,
                       project_dir, projects_root, structure_dir, topic_dir)
from .runs import RunStepsError


def parse(path, **kwargs):
    """The registry's ``parse`` (`model/parse.md` § 3), **in the file's own
    order**, with what the atom index knows that the registry does not
    (`atom_index.parsed`): whose order that is, ``result.atom_index_map``, on a reading of a structure or of
    a run's steps (a run's ``.out``, its ``.XV``, a trajectory log, a
    geometry), and the same on each ``Structure`` it holds, whose
    ``map_to_user_index`` and ``restored`` then answer for the run's order.  A
    file an engine wrote lists its atoms by engine index, which is not your
    atom's index where the run sorted them: for your atoms, ask
    :func:`read_steps`, or map the rows with ``atom_index_map``.

    A parser whose kept reading states its atoms' indices
    (``keeps_atom_index_map`` -- a TBtrans data file's) is handed the run's map
    (:func:`atom_index_map`), so the reading it keeps states both columns
    (§ 5e); a file no run of ours holds is read with none, its reading saying
    why."""
    from .atom_index import parsed
    return parsed(path, **kwargs)


def read_steps(path):
    """THE STEPS A RUN FINISHED, IN YOUR ATOMS' ORDER (`model/overview.md`
    § 2.2b; `runs.StepsTable`), whatever order its engine ran them in: ``path``
    a run's folder -- an optimization's, a transport seed's or device point's,
    a SIESTA vibration's ``relax`` or ``fc`` call -- or a file of it holding its
    steps.

    ::

        steps = read_steps("01_coarse/run-0")
        steps.atom_index_map.listed_by       # 'user_index'
        steps.elements[0]                    # your atom 0's chemical symbol
        steps.positions_ang[-1, 0]           # it, at the last step, in Å
        steps.forces_ev_per_ang[0, 0]        # the force on it at the first, eV/Å
        steps.final_energy_ev                # the last energy the run finished, eV
        steps.relaxation["converged"]        # its largest force within its tolerance
        steps.max_force_free_ev_per_ang      # what that tolerance tests, by step
        steps.held_atoms                     # the atoms it held, by your index
        steps.last                           # the last geometry: labels, box, your order

    NaN where a step states none.  An electrode rung's cell is a structure of
    its own, with no user order: its rows are its own engine index, and
    ``said`` says so.  Refused by name (:class:`RunStepsError`): a path that is
    not there or that no run of ours holds, a file the run does not write its
    steps into -- its ``.XV``, a pair saved beside it, read by :func:`parse` --
    a run whose result holds none -- a transmission point's, a vibration's --
    and a run that has finished no step yet."""
    from .runs import steps_at
    return steps_at(path)


def list_runs(calculation) -> list:
    """EVERY RUN OF A CALCULATION, as folders, stage by stage and each stage's
    starts in turn (`jobset.runstatus.runs_of`) -- a run of points each of its
    points' folders; what :func:`read_steps` and :func:`atom_index_map` are
    asked of.  This is the ladder's order, not the order they ran in: a stage
    can be launched again after a later one.  Refused by name: a folder that
    holds no calculation (`calculation.NotACalculation`)."""
    from .calculation import open as open_calculation
    from .jobset.runstatus import runs_of
    return [folder for _stage, folder in runs_of(open_calculation(calculation))]


def read_structure(path):
    """A STRUCTURE -- a ``.xyz`` with its ``.molstruct.json``, or a ``.pdb``
    alone -- through the codec's read (`workingcopy_structure.StructureCodec`),
    IN THE USER ORDER: an engine's record (a calculation's
    ``junction.xyz``) put back through its order record, every frame with
    it.  ``(structure, atoms, said)``: ``atoms`` the atom index map OF THE
    STRUCTURE RETURNED -- :func:`atom_index_map` of the path, an engine's
    record's relisted by user index as the structure now is
    (`AtomIndexMap.as_listed_by_user`; `model/overview.md` § 2.2b); and
    what the read says -- the order put back, a pair that states no atom
    order read as listed by user index (F4) -- a line each; a pair whose
    order nothing states is refused by name (I9)."""
    from .atom_permutation import LISTED_BY_ENGINE
    from .workingcopy_structure import StructureCodec
    said: list = []
    struct = StructureCodec().load(path, frames=True, in_user_order=True,
                                   said_out=said)
    atoms = atom_index_map(path)
    if atoms.listed_by == LISTED_BY_ENGINE:
        atoms = atoms.as_listed_by_user()
    return struct, atoms, [n.message for n in said]


def read_transport_data(path) -> Dict[str, Any]:
    """A transport calculation's data file, ``<label>.transport.nc``, as
    data (`transport.datafile.read_data_file`: its attributes and every
    group's variables with their words -- ``structure``, ``stages/<stage>``,
    ``points/<point>`` and its ``device``, ``summary``) and
    ``atom_index_map`` -- its atoms listed by
    user index, each atom's engine index beside (`engines/transport.md`
    § 2a.12)."""
    from .transport.datafile import read_data_file
    return {**read_data_file(path), MAP_KEY: atom_index_map(path)}


def read_spectra_data(path) -> Dict[str, Any]:
    """A vibration result's data file, ``<label>.spectra.nc``, back as the
    payload its JSON holds (`spectra.datafile.read_spectra_data`) and
    ``atom_index_map`` -- its atoms listed by user index
    (`engines/vibration.md` § 6.10.5)."""
    from .spectra.datafile import read_spectra_data as _read
    return {**_read(path), MAP_KEY: atom_index_map(path)}


def read_optical_data(path) -> Dict[str, Any]:
    """A SIESTA vibration's optical calls, ``<label>.optical.nc``, back as
    the payload it was written from (`spectra.optical_data.
    read_optical_data`): each frame's eps2 over the photon energies and
    spin, its state, mode and side, the optical deck's settings and text,
    the reference structure -- and ``atom_index_map``, its atoms listed by
    user index (`engines/vibration.md` § 5.11 § 8)."""
    from .spectra.optical_data import read_optical_data as _read
    return {**_read(path), MAP_KEY: atom_index_map(path)}


def read_steps_data(path) -> Dict[str, Any]:
    """A calculation's runs and their steps -- an optimization's data file,
    ``<label>.optimization.nc``, or a vibration's ``<label>.steps.nc``, what
    an export writes -- as data (`stepsdata.read_data_file`: its attributes
    and its groups ``runs``, ``steps``, ``steps/scf``, every variable with its
    words) and ``atom_index_map``, its atoms listed by user index
    (`model/datafiles.md` § 9.5; `engines/optimization-postprocessing.md`
    § 6)."""
    from .stepsdata import read_data_file
    return {**read_data_file(path), MAP_KEY: atom_index_map(path)}


def read_chart_data(path) -> Dict[str, Any]:
    """What one chart drew, ``<label>.<chart>.nc`` -- its own Export's file
    (`model/datafiles.md` § 9.7) -- as data (`chartdata.read_data_file`): its
    attributes, ``chosen`` the record of what was chosen, and every group by
    its path -- a group per kind of curve, a row per curve; the thresholds on
    the root, under ``""``."""
    from .chartdata import read_data_file
    return read_data_file(path)


def read_raman_data(path) -> Dict[str, Any]:
    """An exported Raman view, ``<label>.raman.nc``, back as data
    (`spectra.raman.read_raman_data`): the laser, the widths, the light's
    direction, ``δ``, the files it was derived from by name and sha256,
    frame 0's polarizability and each mode's row (`engines/vibration.md`
    § 5.11 § 10)."""
    from .spectra.raman import read_raman_data as _read
    return _read(path)


def raman_view(path, *, laser_nm: float, broaden_ev: "float | None" = None):
    """THE RAMAN of a SIESTA vibration result ``<label>.spectra.json`` at a
    laser, as the spectrum viewer asks it (`spectra.raman.raman_view_of`):
    ``laser_nm`` the laser's wavelength, ``broaden_ev`` the electronic
    broadening -- the run's own when unsaid, never below it; from the
    optical calls' data file beside the result, both files named with their
    sha256 in ``derived_from``."""
    from .spectra.raman import raman_view_of
    return raman_view_of(path, laser_nm=laser_nm, broaden_ev=broaden_ev)


def _said(**asks) -> Dict[str, Any]:
    """THE KEYWORDS A PERSON SAID -- each one unsaid (``None``) left to the
    door that owns the question, its default there and nowhere else."""
    return {k: v for k, v in asks.items() if v is not None}


def infrared_view(path, *, direction=None, partition=None):
    """THE INFRARED of a vibration result ``<label>.spectra.json``, as the
    spectrum viewer asks it (`spectra.infrared`): ``direction`` ``x``,
    ``y``, ``z`` or a vector (:data:`spectra.infrared.DEFAULT_DIRECTION`
    when unsaid); on a SIESTA result, the free atoms' -- the
    atoms the vibration moved -- from their charges, by ``partition``,
    ``hirshfeld`` or ``voronoi`` (:data:`spectra.infrared.DEFAULT_PARTITION`
    when unsaid); on a PySCF result, the whole system's, from the dipole
    change it stored, of no partition.  Held or free by the result's own
    held atoms (`engines/vibration.md` § 4.6; `science/vibrational-optical-
    response.md` § 2.3) -- the spectrum viewer's own door
    (`spectra.infrared.infrared_view_of`)."""
    from .sidecars.spectra import parse_spectra_json
    from .spectra.infrared import infrared_view_of
    return infrared_view_of(parse_spectra_json(path),
                            **_said(direction=direction, partition=partition))


def pdos(data_file, point, *, atoms=None, labels=None, regions=None,
         orbitals=None) -> Dict[str, Any]:
    """THE PDOS OF A MIXED SELECTION at one point of a transport
    calculation, from its data file ``<label>.transport.nc`` and the
    point's digests beside it (`engines/transport.md` § 2a.12, *A PDOS of
    the atoms you choose, from the files*; the user, 2026-11-16: "go with
    option 3, and allow mixed selections").  ``point`` is the point's group
    name (``"v0.2"``, a frame set's ``"f003/v0.2"``); the selection is the
    union of ``atoms`` (user indices), every atom carrying one of ``labels``
    as the junction writes it (matched as written), and every atom of
    ``regions`` (``L-electrode``, ``bridge``, ``R-electrode``) -- an atom
    named twice counted once -- narrowed to ``orbitals`` (``all``, unsaid,
    or a type of the file's ``orbital_type``).  Each digest is read as the reading the
    file was composed from (`tbtrans_nc.kept_at`: the data file's version it
    states, no ``.TBT.nc`` needed) and summed by the report's one sum
    (`tbtnc.summed_pdos`), a point that shares energies joined as the report
    joins it -- so the curve is the report's.  Answered by the transport
    package (`tbtnc.pdos_from_files`).

    Answers ``{point, energy_ev, pdos, units, selection, atoms,
    outside_device, orbitals, digests, said}``; :class:`tbtnc.TbtError`
    names what cannot answer -- a point the file does not hold, an atom,
    label or region that names none, an empty selection, a digest not kept
    when the file was written or not the reading it names."""
    from .transport.tbtnc import pdos_from_files
    return pdos_from_files(data_file, point,
                           **_said(atoms=atoms, labels=labels,
                                   regions=regions, orbitals=orbitals))


def reaveraged(path, *, temperature_k, swing_quanta=None) -> Dict[str, Any]:
    """A FRAME SET'S TRANSPORT CALCULATION AVERAGED OVER ANOTHER MOTION
    (`science/vibrational-averaging.md` § 5.4), from its data file
    ``<label>.transport.nc``: the bell curve at ``temperature_k`` -- the
    junction's, or a randomly pumped mode's effective one -- with a swing
    adding ``swing_quanta`` quanta on it, each frame weighted from its
    position along the mode alone (`frame_average.motion_weights`).  Nothing
    is computed again and nothing is written.  Answered by the frames'
    post-process (`frame_average.reaveraged`).

    Answers ``{motion, frames, weights, reach, bias_v, energy_ev,
    transmission, conductance_g0, conductance_change_percent, said}``: at
    each voltage the averaged T(E), G/G0 at E_F and its change in per cent
    of frame 0's -- by the record's own rule at each voltage
    (`frame_average.at_voltage`), with these weights: NaN where it has no
    value, its reason said, a voltage whose points are not all done among
    them -- and what is said beside them.  Refused by name
    (`frame_average.ReaverageError`): what `motion_weights` refuses, and a
    data file of another schema (`transport.datafile.DataFileError`)."""
    from .frame_average import reaveraged as _reaveraged
    return _reaveraged(path, temperature_k=temperature_k,
                       **_said(swing_quanta=swing_quanta))


def outermost_move(path, *, mode: int, temperature_k: float,
                   nodes=None):
    """THE OUTERMOST MOVE of mode ``mode`` (its number, as the result
    numbers its modes) of the vibration result ``<label>.spectra.json`` at
    ``temperature_k`` for ``nodes`` frames, in Angstrom -- the largest atom's
    move at the outermost frames of the set the frame generator writes
    there, read through the result's own reader (`spectra.mode_frames.outermost_move`);
    ``None`` for an imaginary mode, which has no spread.  ``nodes`` unsaid, the frame generator's
    first rule (`spectra.mode_frames.NODES`)."""
    from .spectra.mode_frames import outermost_move_of
    return outermost_move_of(path, mode=mode, temperature_k=temperature_k,
                             **_said(nodes=nodes))


__all__ = ["AtomIndexMap", "AtomIndexUnknown", "RunStepsError", "atom_index_map",
           "infrared_view", "list_runs", "parse", "raman_view", "read_optical_data",
           "read_steps_data", "read_raman_data", "read_spectra_data", "read_steps",
           "read_chart_data",
           "read_structure", "read_transport_data", "read_xv", "read_xv_cell", "reaveraged",
           "pdos",
           "outermost_move", "projects_root", "project_dir", "topic_dir",
           "structure_dir", "list_projects", "list_topics", "list_structures"]
