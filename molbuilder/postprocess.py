"""THE POST-PROCESSING LAYER -- one import for a person's script or notebook
(`model/parse.md` § 3, *One layer*; `model/overview.md` § 2.2b).

MODULE  postprocess (above the parse package and the run door; gathers what
        is already there, a second reader of nothing)
ROLE    ``parse`` -- an engine's file, a run's file → its result (the
        registry's), in the file's own order, the map on the answer;
        ``read_steps`` -- a run's steps, in the user order; ``list_runs`` -- a
        calculation's runs;
        ``read_structure`` -- a structure pair, in the user
        order; ``atom_index_map`` -- which index a file's
        atoms are listed by, and both indices of each (`atom_index`);
        ``read_transport_data`` / ``read_spectra_data`` /
        ``read_optical_data`` -- the data files a calculation writes, each
        with its atom index map, and ``read_optimization_data`` -- an exported
        optimization -- ``read_raman_data`` -- an exported Raman view; ``infrared_view`` -- the free atoms' infrared from a
        SIESTA vibration result's charges (`spectra.infrared`);
        ``raman_view`` -- a SIESTA vibration's Raman at a laser, from its
        optical calls (`spectra.raman`); ``pdos`` -- a transport point's PDOS
        of a mixed selection from its data file and digests, summed as the
        report sums (`tbtnc.summed_pdos`); ``reaveraged`` -- a frame set's
        transport calculation averaged over another motion
        (`frame_average.motion_weights`); ``outermost_move`` -- how far
        a mode's frame set at a temperature moves its atom that moves most
        (`spectra.mode_frames`); ``read_xv`` / ``read_xv_cell`` -- SIESTA's
        own ``.XV``, IN THE ENGINE'S ORDER, as its docstring says; and where
        a calculation is: ``projects_root``, ``project_dir``, ``topic_dir``,
        ``structure_dir``, ``list_projects``, ``list_topics``,
        ``list_structures`` (`projects`)
USED-BY a notebook's kernel, which imports the molbuilder its server runs
        (`web/jupyter.md` § 7), and the post-processing documents' examples

    from molbuilder.postprocess import parse, atom_index_map

It imports with the notebook env's packages alone -- numpy, netCDF4, sisl,
xarray -- and molbuilder's own doors ask the same functions at their own
floors.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from .atom_index import AtomIndexUnknown, atom_index_map
from .atom_permutation import MAP_KEY, AtomIndexMap
from .parse.coords import read_xv, read_xv_cell
from .projects import (list_projects, list_structures, list_topics,
                       project_dir, projects_root, structure_dir, topic_dir)
from .runs import RunStepsError


def parse(path, **kwargs):
    """The registry's ``parse`` (`model/parse.md` § 3), **in the file's own
    order**, with what this layer knows that the registry does not: whose
    order that is, ``result.atom_index_map``, on a reading of a structure or of
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
    from .atom_index import of_reading
    from .parse import detect
    parser = detect(path)
    if parser.keeps_atom_index_map and "atoms" not in kwargs:
        try:
            kwargs["atoms"] = atom_index_map(path)
        except AtomIndexUnknown:
            pass
    res = parser.parse(Path(path), **kwargs)
    if (res.result_kind == "trajectory"
            or (res.result_kind == "structure" and res.structure is not None)):
        atoms = of_reading(path, res)
        return replace(_on_the_order_record(res, path, atoms),
                       atom_index_map=atoms)
    return res


def _on_the_order_record(res, path, atoms):
    """``res`` with the order record of the file ``path`` on each structure it
    reads (`model/overview.md` § 2.2a, I5) -- a file listed by engine index, the
    record its run's deck states, of the atoms that record is over; a
    mid-flight placeholder is not.  Nothing else of the run is put on it: this
    is the file's reading.  A pair's reading carries its sidecar's record
    already: every reader of a `.xyz` with its sidecar reads it through the
    codec (`structure-pair`, `pyscf-geom`; `model/parse.md` § 4)."""
    from .atom_permutation import LISTED_BY_ENGINE
    from .runs import declared, run_of
    from .sidecars.molstruct import paired_sidecar
    if atoms.listed_by != LISTED_BY_ENGINE or atoms.user is None:
        return res
    p = Path(path)
    if paired_sidecar(p) is not None:
        return res
    run = run_of(p)
    order = declared(run).atom_order if run is not None else None
    if order is None or order.n_atoms != atoms.n_atoms:
        return res
    if res.result_kind == "structure":
        return replace(res, structure=res.structure.replace(atom_order=order))
    for fr in res.frames:
        if fr.structure.n_atoms == order.n_atoms:
            fr.structure = fr.structure.replace(atom_order=order)
    return res


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


def read_optimization_data(path) -> Dict[str, Any]:
    """An optimization's data file, ``<label>.optimization.nc`` -- what `jobset
    export` writes -- as data (`optimization.datafile.read_data_file`: its
    attributes and every variable with its words) and ``atom_index_map``, its
    atoms listed by user index (`engines/optimization-postprocessing.md` § 6)."""
    from .optimization.datafile import read_data_file
    return {**read_data_file(path), MAP_KEY: atom_index_map(path)}


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


def infrared_view(path, *, direction="z",
                  partition: "str | None" = None):
    """THE INFRARED of a vibration result ``<label>.spectra.json``, as the
    spectrum viewer asks it (`spectra.infrared`): ``direction`` ``x``,
    ``y``, ``z`` or a vector; on a SIESTA result, the free atoms' -- the
    atoms the vibration moved -- from their charges, by ``partition``,
    ``hirshfeld`` or ``voronoi`` (:data:`spectra.infrared.DEFAULT_PARTITION`
    when unsaid); on a PySCF result, the whole system's, from the dipole
    change it stored, of no partition.  Held or free by the result's own
    held atoms (`engines/vibration.md` § 4.6; `science/vibrational-optical-
    response.md` § 2.3) -- the spectrum viewer's own door
    (`spectra.infrared.infrared_view_of`)."""
    from .sidecars.spectra import parse_spectra_json
    from .spectra.infrared import DEFAULT_PARTITION, infrared_view_of
    return infrared_view_of(parse_spectra_json(path), direction=direction,
                            partition=partition or DEFAULT_PARTITION)


def pdos(data_file, point, *, atoms=(), labels=(), regions=(),
         orbitals="all") -> Dict[str, Any]:
    """THE PDOS OF A MIXED SELECTION at one point of a transport
    calculation, from its data file ``<label>.transport.nc`` and the
    point's digests beside it (`engines/transport.md` § 2a.12, *A PDOS of
    the atoms you choose, from the files*; the user, 2026-11-16: "go with
    option 3, and allow mixed selections").  ``point`` is the point's group
    name (``"v0.2"``, a frame set's ``"f003/v0.2"``); the selection is the
    union of ``atoms`` (user indices), every atom carrying one of ``labels``
    as the junction writes it (matched as written), and every atom of
    ``regions`` (``L-electrode``, ``bridge``, ``R-electrode``) -- an atom
    named twice counted once -- narrowed to ``orbitals`` (``all`` or a type
    of the file's ``orbital_type``).  Each digest is read as the reading the
    file was composed from (`tbtrans_nc.kept_at`: the data file's version it
    states, no ``.TBT.nc`` needed) and summed by the report's one sum
    (`tbtnc.summed_pdos`), a point that shares energies joined as the report
    joins it -- so the curve is the report's.

    Answers ``{point, energy_ev, pdos, units, selection, atoms,
    outside_device, orbitals, digests, said}``; :class:`tbtnc.TbtError`
    names what cannot answer -- a point the file does not hold, an atom,
    label or region that names none, an empty selection, a digest not kept
    when the file was written or not the reading it names."""
    from .atom_permutation import AtomIndexError, checked_indices
    from .parse.engines.tbtrans_nc import kept_at
    from .structure import Structure
    from .transport.energies import joined, restricted
    from .transport.tbtnc import READ_WAY, TbtError, per_atom, summed_pdos
    path = Path(data_file)
    here = path.parent
    data = read_transport_data(path)
    groups, amap = data["groups"], data[MAP_KEY]
    # A POINT IS A GROUP THAT SAYS WHETHER IT IS DONE -- not a frame's group
    # of points, nor a point's device run.
    held = sorted(k[len("points/"):] for k in groups
                  if k.startswith("points/")
                  and "done" in groups[k]["variables"])
    if point not in held:
        raise TbtError(f"{point} is not one of {path.name}'s points: "
                       f"{', '.join(held)}")
    J = {k: v["data"] for k, v in groups["structure"]["variables"].items()}
    P = {k: v["data"] for k, v in
         groups[f"points/{point}"]["variables"].items()}
    if int(P["done"]) != 1:
        why = str(P["why"]) or "no transmission yet"
        raise TbtError(f"{point} is not done: {why}")
    n = len(J["atom"])
    # THE SELECTION, a union -- each part naming at least one atom; an
    # index through the map's one bounds check (`checked_indices`).
    try:
        chosen = set(checked_indices(atoms, n, "user index"))
    except AtomIndexError as exc:
        raise TbtError(str(exc)) from exc
    # THE JUNCTION'S LABELS, the structure's own (`structure_json`, read
    # through its door): matched as written, never read for meaning.
    carried = Structure.from_dict(
        groups["structure"]["attrs"]["structure_json"]).regions or {}
    for label in labels:
        members = [int(a) for a in carried.get(label, ())]
        if not members:
            raise TbtError(f"no atom of the junction carries the label "
                           f"{label!r}")
        chosen.update(members)
    owned = sorted({str(r) for r in J["atom_region"] if str(r)})
    for region in regions:
        if region not in owned:
            raise TbtError(f"region {region!r}: one of {', '.join(owned)}")
        chosen.update(a for a in range(n) if str(J["atom_region"][a]) == region)
    if not chosen:
        raise TbtError("no atoms selected")
    # THE DIGESTS THE FILE NAMES -- only a point every reading of which was
    # kept and named when it was written (`datafile`): none named, its why.
    sources = P["dos_per_orbital_files"]
    if not sources:
        why = (str(P["dos_why"]) or str(P["dos_per_orbital_reading"])
               or "not read")
        raise TbtError(f"{point}: {path.name} names no reading of its DOS "
                       f"per orbital ({why}) -- {READ_WAY}, then summarize "
                       f"or export again")
    if orbitals != "all" and not len(J["orbital"]):
        why = groups["structure"]["attrs"].get("orbitals_why", "")
        raise TbtError(f"{path.name} holds no orbital table ({why}), so "
                       f"its orbitals' types are not known")
    # THE FILE'S ATOMS AS ITS DIGESTS LIST THEM, by engine index (F7), and
    # its orbital table as each atom's (l, m) -- the page's own shape.
    expected = amap.as_listed_by_engine()
    engine = dict(zip(range(n), amap.map_to_engine_index(list(range(n)))))
    basis = None if orbitals == "all" else per_atom(
        [(engine[int(J["orbital_atom"][o])], int(J["orbital_n"][o]),
          int(J["orbital_l"][o]), int(J["orbital_m"][o]),
          int(J["orbital_zeta"][o])) for o in range(len(J["orbital"]))])

    def read(folder):
        mine = [s for s in sources if (here / s["digest"]).parent == folder]
        readings = []
        for s in mine:
            got = kept_at(here / s["digest"], s["version"], need_user=True)
            if got is None:
                raise TbtError(
                    f"{s['digest']} is not the reading {path.name} was "
                    f"composed from -- not there, or read again from "
                    f"another run of its data file: summarize or export "
                    f"again")
            readings.append((Path(s["digest"]).name, got))
        return summed_pdos(readings, basis,
                           [engine[a] for a in sorted(chosen)], orbitals,
                           expected=expected,
                           basis_said=f"{path.name}'s orbital table")
    own = here / str(P["folder"])
    # JOINED AS THE REPORT JOINS IT, on the point's own grid -- the file's.
    got = restricted(joined(own, read, window_of=lambda _f: None),
                     [float(e) for e in P["energy_ev"]])
    return {"point": point, "energy_ev": got["energy_ev"],
            "pdos": got["pdos"], "units": "1/eV",
            "selection": {"atoms": sorted({int(a) for a in atoms}),
                          "labels": list(labels), "regions": list(regions)},
            "atoms": sorted(amap.map_to_user_index(got["atoms"])),
            "outside_device": sorted(
                amap.map_to_user_index(got["outside_device"])),
            "orbitals": orbitals,
            "digests": [s["digest"] for s in sources],
            "said": ("the DOS per orbital of the digests named, summed over "
                     "the orbitals of the atoms selected -- their union -- "
                     "that the point's DOS covers; a polarized point's two "
                     "channels summed.  The selection, and so the sum, is "
                     "the person's.")}


def reaveraged(path, *, temperature_k, swing_quanta=0.0) -> Dict[str, Any]:
    """A FRAME SET'S TRANSPORT CALCULATION AVERAGED OVER ANOTHER MOTION
    (`science/vibrational-averaging.md` § 5.4), from its data file
    ``<label>.transport.nc``: the bell curve at ``temperature_k`` -- the
    junction's, or a randomly pumped mode's effective one -- with a swing
    adding ``swing_quanta`` quanta on it, each frame weighted from its
    position along the mode alone (`frame_average.motion_weights`).  Nothing
    is computed again and nothing is written.

    Answers ``{motion, frames, weights, reach, bias_v, energy_ev,
    transmission, conductance_g0, conductance_change_percent, said}``: at
    each voltage the averaged T(E), G/G0 at E_F and its change in per cent
    of frame 0's -- by the record's own rule at each voltage
    (`frame_average.at_voltage`), with these weights: NaN where it has no
    value, its reason said, a voltage whose points are not all done among
    them -- and what is said beside them.  Refused by name
    (`frame_average.ReaverageError`): what `motion_weights` refuses, and a
    data file of another schema (`transport.datafile.DataFileError`)."""
    import numpy as np
    from .frame_average import at_voltage, motion_weights
    from .transport.datafile import read_data_file
    groups = read_data_file(path)["groups"]
    # THE FRAMES' ROWS ARE THE JUNCTION'S, the points' figures the
    # summary's (`engines/transport.md` § 2a.12, *The calculation as
    # groups*).
    data = {**groups["structure"]["variables"],
            **groups["summary"]["variables"]}

    def value(name):
        got = data.get(name)
        return None if got is None else np.asarray(got["data"])

    def scalar(name):
        got = value(name)
        return float(got) if got is not None and got.size == 1 else None
    q = value("q_amu12_ang")
    got = motion_weights(
        q if q is not None else [], temperature_k=temperature_k,
        swing_quanta=swing_quanta, frequency_cm1=scalar("frequency_cm1"),
        zero_point_amplitude_amu12_ang=scalar(
            "zero_point_amplitude_amu12_ang"))
    weights = list(got["weights"])
    n = len(weights)
    bias = [float(v) for v in value("bias_v")]
    energy = [float(e) for e in value("energy_ev")]
    curves = np.asarray(value("transmission"), dtype=float)
    g = np.asarray(value("conductance_g0"), dtype=float)
    done = np.asarray(value("done")) == 1
    # THE DONE POINTS AS THE RECORD HOLDS THEM, each on the file's one grid,
    # its conductance ``None`` where the file keeps none.
    points = [{"frame": f, "bias_v": v, "energy_ev": energy,
               "transmission": curves[f, b, :],
               "conductance_g0": (float(g[f, b]) if np.isfinite(g[f, b])
                                  else None)}
              for b, v in enumerate(bias) for f in range(n) if done[f, b]]
    said = list(got["said"])
    transmission = np.full((len(bias), len(energy)), np.nan)
    conductance = np.full(len(bias), np.nan)
    change = np.full(len(bias), np.nan)
    for b, v in enumerate(bias):
        # EACH VOLTAGE BY THE RECORD'S RULE, with these weights: no value
        # where it has none, its reason said (`engines/transport.md`
        # § 2a.12) -- the file's own axis, so a voltage is its own index.
        at = at_voltage(v, n, weights, points, [], lambda a, c: a == c)
        for why in (at["why"], at["conductance_why"]):
            if why:
                said.append(f"at {v:g} V: {why}")
        if at["transmission"] is not None:
            transmission[b] = at["transmission"]
        if at["conductance_g0"] is not None:
            conductance[b] = at["conductance_g0"]
        if at["conductance_change_percent"] is not None:
            change[b] = at["conductance_change_percent"]
    return {"motion": got["motion"],
            "frames": [{"frame": f, "q_amu12_ang": float(q[f]),
                        "weight": float(weights[f])} for f in range(n)],
            "weights": weights, "reach": got["reach"],
            "bias_v": bias,
            "energy_ev": energy,
            "transmission": transmission, "conductance_g0": conductance,
            "conductance_change_percent": change, "said": said}


def outermost_move(path, *, mode: int, temperature_k: float,
                   nodes: int = 3):
    """THE OUTERMOST MOVE of mode ``mode`` (its number, as the result
    numbers its modes) of the vibration result ``<label>.spectra.json`` at
    ``temperature_k`` for ``nodes`` frames, in Angstrom -- the largest atom's
    move at the outermost frames of the set the frame generator writes
    there, read through the result's own reader (`spectra.mode_frames.outermost_move`);
    ``None`` for an imaginary mode, which has no spread."""
    from .spectra.mode_frames import FrameSetError, outermost_move as _move
    from .sidecars.spectra import parse_spectra_json
    rows = parse_spectra_json(path).to_dict()["modes"]
    row = next((m for m in rows if m["index_1based"] == mode), None)
    if row is None:
        raise FrameSetError(f"{path} holds modes 1..{len(rows)}, not mode "
                            f"{mode!r}")
    return _move(row, temperature_k, nodes)


__all__ = ["AtomIndexMap", "AtomIndexUnknown", "RunStepsError", "atom_index_map",
           "infrared_view", "list_runs", "parse", "raman_view", "read_optical_data",
           "read_optimization_data", "read_raman_data", "read_spectra_data", "read_steps",
           "read_structure", "read_transport_data", "read_xv", "read_xv_cell", "reaveraged",
           "pdos",
           "outermost_move", "projects_root", "project_dir", "topic_dir",
           "structure_dir", "list_projects", "list_topics", "list_structures"]
