"""THE POST-PROCESSING LAYER -- one import for a person's script or notebook
(`model/parse.md` § 3, *One layer*; `model/overview.md` § 2.2b).

MODULE  postprocess (above the parse package and the run door; gathers what
        is already there, a second reader of nothing)
ROLE    ``parse`` -- an engine's file, a run's file → its result (the
        registry's); ``read_structure`` -- a structure pair, in the user
        order; ``atom_index_map`` -- which index a file's
        atoms are listed by, and both indices of each (`atom_index`);
        ``read_transport_data`` / ``read_spectra_data`` /
        ``read_optical_data`` -- the data files a calculation writes, each
        with its atom index map, and ``read_raman_data`` -- an exported
        Raman view; ``infrared_view`` -- the free atoms' infrared from a
        SIESTA vibration result's charges (`spectra.infrared`);
        ``raman_view`` -- a SIESTA vibration's Raman at a laser, from its
        optical calls (`spectra.raman`); and
        ``outermost_move`` -- how far a mode's frame set at a temperature
        moves its atom that moves most (`mode_frames`)
USED-BY a notebook's kernel, which imports the molbuilder its server runs
        (`web/jupyter.md` § 7), and the post-processing documents' examples

    from molbuilder.postprocess import parse, atom_index_map

It imports with the notebook env's packages alone -- numpy, netCDF4, sisl,
xarray -- and molbuilder's own doors ask the same functions at their own
floors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .atom_index import AtomIndexUnknown, atom_index_map
from .atom_permutation import MAP_KEY, AtomIndexMap


def parse(path, **kwargs):
    """The registry's ``parse`` (`model/parse.md` § 3), with what this layer
    knows that the registry does not: a parser whose kept reading states
    its atoms' indices (``keeps_atom_index_map`` -- a TBtrans data file's)
    is handed the run's atom index map (:func:`atom_index_map`), so the
    reading it keeps states both columns (§ 5e); a file no run of ours
    holds is read with none, its reading saying why."""
    from .parse import detect
    parser = detect(path)
    if parser.keeps_atom_index_map and "atoms" not in kwargs:
        try:
            kwargs["atoms"] = atom_index_map(path)
        except AtomIndexUnknown:
            pass
    return parser.parse(Path(path), **kwargs)


def read_structure(path):
    """A STRUCTURE PAIR -- ``.xyz`` or ``.pdb`` with its ``.molstruct.json``
    -- through the codec's read (`workingcopy_structure.StructureCodec`),
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
    variable with its words) and ``atom_index_map`` -- its atoms listed by
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
    change it stored (no partition named).  Held or free by the result's own
    held atoms (`engines/vibration.md` § 4.6; `science/vibrational-optical-
    response.md` § 2.3)."""
    from .sidecars.spectra import parse_spectra_json
    from .spectra.infrared import (DEFAULT_PARTITION, SOURCE_DIPOLE,
                                   InfraredError, dipole_view,
                                   infrared_view as _view, source_of)
    result = parse_spectra_json(path)
    if source_of(result) == SOURCE_DIPOLE:
        if partition is not None:
            raise InfraredError(f"{path}: a {result.engine} result's infrared "
                                f"is the whole system's dipole change -- it "
                                f"has no partition")
        return dipole_view(result, direction=direction)
    return _view(result, direction=direction,
                 partition=partition or DEFAULT_PARTITION)


def outermost_move(path, *, mode: int, temperature_k: float,
                   nodes: int = 3):
    """THE OUTERMOST MOVE of mode ``mode`` (its number, as the result
    numbers its modes) of the vibration result ``<label>.spectra.json`` at
    ``temperature_k`` for ``nodes`` frames, in Angstrom -- the largest atom's
    move at the outermost frames of the set the frame generator writes
    there, read through the result's own reader (`mode_frames.outermost_move`);
    ``None`` for an imaginary mode, which has no spread."""
    from .mode_frames import FrameSetError, outermost_move as _move
    from .sidecars.spectra import parse_spectra_json
    rows = parse_spectra_json(path).to_dict()["modes"]
    row = next((m for m in rows if m["index_1based"] == mode), None)
    if row is None:
        raise FrameSetError(f"{path} holds modes 1..{len(rows)}, not mode "
                            f"{mode!r}")
    return _move(row, temperature_k, nodes)


__all__ = ["AtomIndexMap", "AtomIndexUnknown", "atom_index_map",
           "infrared_view", "parse", "raman_view", "read_optical_data",
           "read_raman_data", "read_spectra_data", "read_structure",
           "read_transport_data", "outermost_move"]
