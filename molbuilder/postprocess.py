"""THE POST-PROCESSING LAYER -- one import for a person's script or notebook
(`model/parse.md` § 3, *One layer*; `model/overview.md` § 2.2b).

MODULE  postprocess (above the parse package and the run door; gathers what
        is already there, a second reader of nothing)
ROLE    ``parse`` -- an engine's file, a run's file → its result (the
        registry's); ``read_structure`` -- a structure pair, in the user
        order; ``atom_index_map`` -- which index a file's
        atoms are listed by, and both indices of each (`atom_index`);
        ``read_transport_data`` / ``read_spectra_data`` -- the two data files
        molbuilder writes, each with its atom index map
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


__all__ = ["AtomIndexMap", "AtomIndexUnknown", "atom_index_map", "parse",
           "read_spectra_data", "read_structure", "read_transport_data"]
