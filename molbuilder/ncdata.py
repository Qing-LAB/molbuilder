"""THE DATA-FILE FRAMEWORK -- the variable record, the one writer, the one door
and the tables every NetCDF file molbuilder writes or opens shares
(`model/datafiles.md`; each file's variables are its chapter's:
`engines/transport.md` § 2a.12, `engines/vibration.md` § 6.10.5,
`model/parse.md` § 5e).

MODULE  ncdata (L1; numpy, netCDF4 and persist)
ROLE    :class:`Var` -- one variable: its dimensions, its kind (given, raw,
        derived, state, coordinate), its units, its words; :data:`AXES` --
        every axis of every data file, one name for one concept, which
        :func:`check_axes` holds every group to (`model/datafiles.md` § 2a);
        :func:`empty` --
        the array a variable starts as, its fill stated; :func:`write_group`
        -- a group's dimensions and variables written from a table of
        :class:`Var`, each with its attributes and its CF ``coordinates``,
        the atom axis given as its atom index map; :func:`in_memory` /
        :func:`as_bytes` -- the file built in memory and handed to the one
        atomic writer (`persist.write_bytes`); :func:`reading` -- the one
        door every NetCDF file molbuilder opens goes through, one access and
        one handle at a time under :data:`NC_LOCK` (:class:`OpenedTwice` a
        file opened again inside its own access); :func:`read_variables` /
        :func:`read_atom_index_map` -- a group's variables and the map of its
        atom axis, read back; :func:`as_tables` -- a data file as long CSV
        tables and a README, every word the file's own (an export)
USED-BY the transport data file (`transport.datafile`: written and read) and
        its export (`transport.export`, the tables); the vibration result's
        data file (`spectra.datafile`), beside the job in both bundles
        (`runwrap.VIBRATION_COMPANIONS`, `runwrap.PYSCF_COMPANIONS`); a
        TBtrans data file and its digest (`parse.engines.tbtrans_nc`); a
        relaxation's ``.MD.nc`` (`parse.engines.siesta_mdnc`); the atom index
        map's reader of a data file (`atom_index`); a frame set's rows in the
        transport data file (`frame_average`, its JSON attributes)

**Encoding, so a round trip is exact**: a number's fill is NaN, a text's the
empty string, an integer's its stated fill; a boolean is an ``int8`` with
``flag_values 0 1`` (:func:`flag`); an empty list is a dimension of length
0; a nested dict is JSON text (:func:`json_attr`).  Imports netCDF4 at load:
a job whose environment lacks it stops here, before the expensive part
(`engines/vibration.md` § 6.10.6).
"""
from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import netCDF4
import numpy as np

# TWO WAYS, because this module travels beside the job.
try:                                        # inside molbuilder
    from .atom_permutation import (ATOM_INDEX_BASE, FIELD_BASE, FIELD_ENGINE,
                                   FIELD_LISTED_BY, FIELD_USER, FIELD_WHY,
                                   AtomIndexMap)
    from .persist import json_safe
except ImportError:                         # beside a job, in a bundle
    from atom_permutation import (  # type: ignore[no-redef]
        ATOM_INDEX_BASE, FIELD_BASE, FIELD_ENGINE, FIELD_LISTED_BY,
        FIELD_USER, FIELD_WHY, AtomIndexMap)
    from persist import json_safe  # type: ignore[no-redef]

#: THE ATOM AXIS's name -- a dimension of this name is given to the writer as
#: its atom index map, never a count (`model/overview.md` § 2.2b, F6).
ATOM_AXIS = "atom"


@dataclass(frozen=True)
class Var:
    """One variable of the file: its ``name``, its dimensions, its ``kind``
    (``coordinate``, ``given``, ``raw`` -- as an engine wrote it --,
    ``measured`` -- read off the structures, a frame set's rows
    (`model/structure.md` § 2.2f) --, ``derived``, ``state``), ``units``, ``long_name``, ``definition`` and,
    for a raw one, ``source`` -- the file it was read from.  A
    ``coordinate`` names a dimension's points: the dimension's own values
    when it is named as the dimension (a NetCDF coordinate variable), else
    a label of them.  An ``auxiliary`` variable labels the points of its one
    dimension -- each variable over that dimension names it in its CF
    ``coordinates`` attribute, so a NetCDF reader opens it as a
    coordinate.  ``fill`` is an integer variable's fill value, where a cell
    can be not stated (a flag of a run that did not run: neither 0 nor 1);
    a number's is NaN, a text's the empty string.  A boolean variable is
    an ``int8`` with ``flag=True``: ``flag_values 0 1`` beside it.  A
    variable whose VALUES are atom indices -- a subset of the atoms, the
    atom each orbital is on -- is ``atom_index=True``: they count by the
    file's atom index map, and its attribute says so
    (`model/overview.md` § 2.2b)."""
    name: str
    dims: Tuple[str, ...]
    kind: str
    units: str
    long_name: str
    definition: str
    source: str = ""
    dtype: Any = "f8"
    auxiliary: bool = False
    fill: Any = None
    flag: bool = False
    atom_index: bool = False


#: The map's two columns as variables on the atom axis -- labels of its
#: points, so a NetCDF reader opens them as coordinates.
_MAP_VARS = (
    Var(FIELD_USER, (ATOM_AXIS,), "coordinate", "1", "the atom's user index",
        "its index in the structure the user built or cited, from 0 -- the "
        "structure's own view numbers it user index + 1 "
        "(model/overview.md 2.2b)", dtype="i4", auxiliary=True),
    Var(FIELD_ENGINE, (ATOM_AXIS,), "coordinate", "1",
        "the atom's engine index",
        "its index in the order the engine ran -- its deck's, from 0 "
        "(model/overview.md 2.2b)", dtype="i4", auxiliary=True),
)


@dataclass(frozen=True)
class Axis:
    """One axis of the data files (`model/datafiles.md` § 2a): its name,
    what its points are, and its coordinate's units -- ``1`` an index, the
    empty string labels."""
    name: str
    points: str
    units: str


#: EVERY AXIS OF EVERY DATA FILE, one name for one concept
#: (`model/datafiles.md` § 2a, its rows in its order): the one writer
#: refuses a dimension that is not one of them, and one without its
#: coordinate.  A new axis is a row here and there, in the change that first
#: writes it.
AXES: Dict[str, Axis] = {a.name: a for a in (
    Axis(ATOM_AXIS, "the atoms", "1"),
    Axis("free_atom", "the atoms a vibration moves", "1"),
    Axis("device_atom", "the atoms of TranSIESTA's device region", "1"),
    Axis("atom_bound", "each atom's first orbital, and one past the last",
         "1"),
    Axis("device_orbital", "the device region's orbitals", "1"),
    Axis("xyz", "a Cartesian component", ""),
    Axis("cell_vector", "the cell's three vectors", ""),
    Axis("reciprocal_vector", "the reciprocal cell's three vectors", ""),
    Axis("kpoint", "the k-points a reading averaged over", "1"),
    Axis("frame", "a frame set's frames", "1"),
    Axis("bias_v", "the voltages computed", "V"),
    Axis("iv_bias_v", "the I-V's voltages", "V"),
    Axis("energy_ev", "energy relative to the leads' Fermi level", "eV"),
    Axis("spin", "spin channels", ""),
    Axis("lead", "the two leads", ""),
    Axis("region", "molbuilder's regions of the junction", ""),
    Axis("eigenchannel", "transmission eigenchannels", "1"),
    Axis("rung", "the rungs that run once", ""),
    Axis("scf_cycle", "a run's SCF rows", "1"),
    Axis("charge_part", "TranSIESTA's charge columns", ""),
    Axis("mode", "vibrational modes", "1"),
    Axis("removed", "the rigid motions removed", "1"),
    Axis("mo", "molecular orbitals", "1"),
    Axis("es", "the modes the electronic-structure probe ran on", "1"),
    Axis("window", "an orbital's place in the probe's window", "1"),
    Axis("temperature_k", "the thermochemistry's temperatures", "K"),
)}


def check_axes(dims: Mapping[str, Any], variables: Sequence[Var]) -> None:
    """EVERY AXIS A ROW OF :data:`AXES`, WITH ITS COORDINATE
    (`model/datafiles.md` § 2a): each dimension a group creates or its
    variables use is one of them, and has a coordinate of its own name among
    the group's variables, in its axis's units -- else refused, naming the
    section where a new axis is added."""
    used = list(dict.fromkeys([*dims, *(d for v in variables
                                         for d in v.dims)]))
    unknown = [d for d in used if d not in AXES]
    if unknown:
        raise ValueError(
            f"axis {', '.join(unknown)} is not an axis of the data files: "
            f"its row is added to model/datafiles.md § 2a and ncdata.AXES "
            f"first")
    coordinate = {v.name: v for v in variables
                  if v.kind == "coordinate" and v.dims == (v.name,)}
    for d in used:
        if d not in coordinate:
            raise ValueError(
                f"axis {d} has no coordinate of its own name: every axis "
                f"has one, so a reader selects by value "
                f"(model/datafiles.md § 2a, rule 3)")
        if coordinate[d].units != AXES[d].units:
            raise ValueError(
                f"axis {d}'s coordinate states units "
                f"{coordinate[d].units!r}; the axis is in "
                f"{AXES[d].units!r} (model/datafiles.md § 2a)")


#: The fill an integer flag takes where a cell is not stated.
FLAG_FILL = -1

#: ONE NETCDF ACCESS AT A TIME IN THIS PROCESS.  The netCDF-C library is not
#: thread-safe and netCDF4 releases the GIL around its calls, so two threads
#: of one server inside it at once can crash the process (netCDF4 1.7's own
#: warning; measured 2026-10-22: a TBtrans data file read by two request
#: threads while two more opened digests segfaulted in 2 of 3 runs).  Every
#: NetCDF open, read and write molbuilder makes holds it -- a TBtrans data
#: file's, through sisl, one k-point at a time (`parse.engines.tbtrans_nc`);
#: a relaxation's ``.MD.nc`` (`parse.engines.siesta_mdnc`); the transport
#: and vibration data files -- so a long reading lets the page's other
#: requests in between.  Reentrant: a reader asks another.
NC_LOCK = threading.RLock()


def size_of(dim) -> int:
    """A dimension's size as ``dims`` gives it: a count, or -- the atom
    axis -- its atom index map's atom count."""
    return dim.n_atoms if isinstance(dim, AtomIndexMap) else int(dim)


def empty(var: Var, dims: Mapping[str, Any]) -> np.ndarray:
    """The array ``var`` starts as, over ``dims``: every cell its fill."""
    shape = tuple(size_of(dims[d]) for d in var.dims)
    if var.dtype is str:
        return np.full(shape, "", dtype=object)
    if var.dtype in ("i1", "i2", "i4", "i8"):
        fill = (FLAG_FILL if var.flag and var.fill is None
                else 0 if var.fill is None else var.fill)
        return np.full(shape, fill, dtype=var.dtype)
    return np.full(shape, np.nan)


def flag(value: Optional[bool]) -> int:
    """A boolean as the file states it: 1, 0, or :data:`FLAG_FILL` for
    not stated."""
    return FLAG_FILL if value is None else int(bool(value))


def unflag(value) -> Optional[bool]:
    """:func:`flag` read back."""
    v = int(value)
    return None if v == FLAG_FILL else bool(v)


def number_attr(value) -> float:
    """A nullable number as an attribute: NaN where the record states
    none."""
    return float("nan") if value is None else float(value)


def unnumber_attr(value) -> Optional[float]:
    """:func:`number_attr` read back."""
    v = float(value)
    return None if np.isnan(v) else v


def json_attr(value) -> str:
    """A nested block as an attribute: its JSON text, a number JSON cannot
    hold the null every reader takes (`persist.json_safe`), every character
    as written."""
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False,
                      sort_keys=False)


def write_group(group, dims: Mapping[str, Any], variables: Sequence[Var],
                values: Mapping[str, np.ndarray]) -> None:
    """``dims`` created on ``group`` (a dataset or a group) and every
    variable of ``variables`` written from ``values[name]`` with its
    attributes -- ``kind``, ``units``, ``long_name``, ``definition``, its
    ``source`` where raw, ``flag_values`` where a flag, ``atom_index`` where
    its values are atom indices, and the CF ``coordinates`` naming each
    auxiliary coordinate along its dimensions.  Numbers are compressed; the
    fill value is said.

    **The atom axis is given as its atom index map**, never a count
    (`model/overview.md` § 2.2b, F6): its fields are stated from it --
    ``atoms_listed_by`` and ``atom_index_base`` as the group's attributes,
    each column it knows as a variable on the axis -- and an axis given as
    a bare number is refused, as is a map whose atom count is not the
    axis's.  **Every axis is a row of** :data:`AXES` **with its coordinate**
    (:func:`check_axes`)."""
    check_axes(dims, variables)
    atoms = dims.get(ATOM_AXIS)
    if atoms is not None and not isinstance(atoms, AtomIndexMap):
        raise ValueError(
            "the atom axis is given as its atom index map, never a count: a "
            "file says which index its atoms are listed by "
            "(model/overview.md 2.2b)")
    if atoms is not None:
        stated = atoms.to_dict()
        group.setncatts({FIELD_LISTED_BY: stated[FIELD_LISTED_BY],
                         FIELD_BASE: np.int32(ATOM_INDEX_BASE),
                         # WHY A COLUMN IS NOT WRITTEN, where one is not (F1).
                         **({FIELD_WHY: stated[FIELD_WHY]}
                            if FIELD_WHY in stated else {})})
        columns = [v for v in _MAP_VARS if v.name in stated]
        variables = tuple(variables) + tuple(columns)
        values = {**values, **{v.name: np.asarray(stated[v.name], dtype="i4")
                               for v in columns}}
    for name, size in dims.items():
        group.createDimension(name, size_of(size))
    auxiliary = [a for a in variables if a.auxiliary]
    for var in variables:
        # THE FILL VALUE, said: NaN for a number, the variable's own for an
        # integer that can be not stated; compressed unless text.
        if var.dtype is str:
            kw: Dict[str, Any] = {}
        else:
            kw = {"zlib": True, "complevel": 4}
            if var.dtype not in ("i1", "i2", "i4", "i8"):
                kw["fill_value"] = np.nan
            elif var.flag and var.fill is None:
                kw["fill_value"] = FLAG_FILL
            elif var.fill is not None:
                kw["fill_value"] = var.fill
        nc = group.createVariable(var.name, var.dtype, var.dims, **kw)
        # THE LABELS OF ITS POINTS, named for a NetCDF reader (CF
        # `coordinates`): every auxiliary coordinate along its dimensions.
        labels = ([] if var.kind == "coordinate" or var.auxiliary else
                  [a.name for a in auxiliary if a.dims[0] in var.dims])
        nc.setncatts({"kind": var.kind, "units": var.units,
                      "long_name": var.long_name,
                      "definition": var.definition,
                      **({"source": var.source} if var.source else {}),
                      **({"flag_values": np.array([0, 1], dtype="i1"),
                          "flag_meanings": "no yes"} if var.flag else {}),
                      # ITS VALUES ARE ATOM INDICES, counting by the file's
                      # atom index map (§ 2.2b).
                      **({"atom_index": np.int8(1)} if var.atom_index
                         else {}),
                      **({"coordinates": " ".join(labels)} if labels
                         else {})})
        data = values[var.name]
        if data.size:
            nc[...] = data


#: The files an access holds open now, in this process -- the door's own
#: check that no file is opened a second time inside one access.
_OPEN: "set" = set()


class OpenedTwice(Exception):
    """A file opened again inside its own access -- a mistake in the code,
    never a file that does not read, so no reader's refusal swallows it."""


@contextlib.contextmanager
def reading(path, *, sile: bool = False):
    """THE ONE DOOR FOR EVERY NETCDF FILE molbuilder opens
    (`model/datafiles.md` § 4): ``with reading(path) as ds:`` -- the process's
    one lock held (:data:`NC_LOCK`), the file opened read-only -- a
    ``netCDF4.Dataset``, or with ``sile`` a sisl sile, for an engine's own
    file sisl reads (a TBtrans ``.TBT.nc``) -- and closed before the lock is
    released.  **No handle outlives its access**, so molbuilder's process
    never holds two handles on one file: the netCDF-C/HDF5 libraries crash a
    process that opens a file again after a second handle on it read a text
    variable and closed (measured 2026-10-23, netCDF-C 4.10.1, HDF5 2.2.0),
    and a door that never makes a second handle cannot reach it.  A file
    opened again inside its own access is refused by name -- the one
    mistake the door can see.  Nothing is read but what the reader reads;
    a long reading is a series of accesses."""
    key = str(Path(path).resolve())
    with NC_LOCK:
        if key in _OPEN:
            raise OpenedTwice(
                f"{Path(path).name} is opened again inside its own access -- "
                f"one handle a file in this process (model/datafiles.md 4)")
        _OPEN.add(key)
        try:
            if sile:
                import sisl
                handle = sisl.get_sile(str(path))
            else:
                handle = netCDF4.Dataset(str(path), mode="r")
            try:
                yield handle
            finally:
                handle.close()
        finally:
            _OPEN.discard(key)


def in_memory(name: str) -> "netCDF4.Dataset":
    """A NetCDF-4 dataset built in memory, to be handed to the one atomic
    writer as bytes (:func:`as_bytes`)."""
    return netCDF4.Dataset(name, "w", memory=4096, format="NETCDF4")


def as_bytes(ds: "netCDF4.Dataset") -> bytes:
    """The in-memory dataset closed, as the file's bytes."""
    return bytes(ds.close())


def read_atom_index_map(group, *, source: str,
                        said: str = "") -> Optional[AtomIndexMap]:
    """The atom index map ``group`` states of its atom axis
    (`model/overview.md` § 2.2b) -- its attributes and columns, read by the
    map's one reader, checked -- or ``None`` where it states none: a file
    written before the rule, which its reader answers by F3."""
    if FIELD_LISTED_BY not in group.ncattrs():
        return None
    d: Dict[str, Any] = {FIELD_LISTED_BY: str(group.getncattr(FIELD_LISTED_BY))}
    if FIELD_BASE in group.ncattrs():
        d[FIELD_BASE] = int(group.getncattr(FIELD_BASE))
    if FIELD_WHY in group.ncattrs():
        d[FIELD_WHY] = str(group.getncattr(FIELD_WHY))
    for name in (FIELD_USER, FIELD_ENGINE):
        if name in group.variables:
            d[name] = [int(i) for i in np.asarray(group.variables[name][...])]
    return AtomIndexMap.from_dict(d, n_atoms=len(group.dimensions[ATOM_AXIS]),
                                  source=source, said=said)


def read_variables(group) -> Dict[str, Dict[str, Any]]:
    """Every variable of ``group`` as ``{name: {"dims", "data", "attrs"}}``
    -- its values as a numpy array (the fill value as written), its
    attributes but the fill value."""
    group.set_auto_mask(False)
    return {name: {"dims": tuple(v.dimensions),
                   "data": np.asarray(v[...]),
                   "attrs": {k: v.getncattr(k) for k in v.ncattrs()
                             if k != "_FillValue"}}
            for name, v in group.variables.items()}


def as_tables(data: bytes, name: str) -> Tuple[Dict[str, str], str]:
    """A DATA FILE AS TABLES -- the NetCDF file ``name`` whose bytes are
    ``data``, laid out for a spreadsheet or a script, every word the file's
    own: **one long table per set of axes** in each group, every variable
    over the same dimensions in it, named by them
    (``frame_bias_v_energy_ev.csv``; a group's under its path,
    ``modes/mode.csv``; ``scalars.csv`` for none); one row per point of those
    axes, each axis a column of its coordinate's values under its own name (a
    dimension with no coordinate, its index), each variable a column named
    with its units (``dos_total [1/eV]``), a value not stated -- its fill --
    an empty cell.  And the README: the file's attributes, then table by
    table each column's long name, units, kind, definition and source, and
    where the file's fill is an empty cell.  Returns ``({table's path: its
    CSV text}, the README's text)`` -- any data file molbuilder writes lays
    out the same way."""
    import csv
    import io
    import itertools

    def said(v) -> Dict[str, Any]:
        return {k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"}

    def text(values) -> List[str]:
        """A variable's values as cells, in its array's order: the fill and
        NaN empty, a number at full precision."""
        arr = np.ma.asarray(values)
        mask = np.ma.getmaskarray(arr).ravel().tolist()
        flat = np.ma.getdata(arr).ravel().tolist()
        out = []
        for x, m in zip(flat, mask):
            if m or x is None:
                out.append("")
            elif isinstance(x, bytes):
                out.append(x.decode("utf-8", "replace"))
            elif isinstance(x, float):
                out.append("" if x != x else repr(x))
            else:
                out.append(str(x))
        return out

    def attr(v) -> str:
        """An attribute as JSON where it is a number or a list -- a value not
        stated said so -- its text as written otherwise."""
        if isinstance(v, str):
            return v
        v = np.asarray(v).tolist()
        if isinstance(v, float) and v != v:
            return "not stated"
        return json_attr(v)

    def read(group, path: str, sizes: Dict[str, int],
             axes: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
        """Each group's attributes, sizes and variables, read into memory --
        a dimension and a coordinate seen from an enclosing group as the
        format resolves them, the nearest enclosing one's."""
        sizes = {**sizes, **{k: len(d) for k, d in group.dimensions.items()}}
        axes = dict(axes)
        held: Dict[str, Any] = {}
        for var, v in group.variables.items():
            v.set_auto_mask(True)
            fill = (v.getncattr("_FillValue") if "_FillValue" in v.ncattrs()
                    else None)
            entry = (tuple(v.dimensions), v[...], said(v), fill)
            if tuple(v.dimensions) == (var,):
                axes[var] = entry
            else:
                held[var] = entry
        out.append({"path": path, "sizes": sizes, "axes": axes, "held": held,
                    "attrs": {k: group.getncattr(k)
                              for k in group.ncattrs()}})
        for sub, g in group.groups.items():
            read(g, f"{path}/{sub}" if path else sub, sizes, axes, out)

    # THE FILE READ under the lock every NetCDF access holds; its tables
    # written out of it.
    read_groups: List[Dict[str, Any]] = []
    with NC_LOCK, netCDF4.Dataset("data", mode="r", memory=data) as ds:
        read(ds, "", {}, {}, read_groups)

    def header(var: str, info: Dict[str, Any]) -> str:
        units = str(info.get("units") or "")
        return f"{var} [{units}]" if units not in ("", "1") else var

    tables: Dict[str, str] = {}
    lines = [f"{name}, as tables", "",
             "Every table is long: one row per point of its axes, each axis "
             "a column, each variable a column named with its units; an "
             "empty cell is a value the file does not state.", ""]
    for g in read_groups:
        path, sizes, axes, held = g["path"], g["sizes"], g["axes"], g["held"]
        here = f"{path}/" if path else ""
        if g["attrs"]:
            lines.extend([f"THE ATTRIBUTES OF {path or 'THE FILE'}", ""])
            # A TEXT OF SEVERAL LINES -- a record's file -- indented under
            # its name.
            lines.extend(f"  {k}: " + attr(v).replace("\n", "\n    ")
                         for k, v in g["attrs"].items())
            lines.append("")
        by_dims: Dict[Tuple[str, ...], List[str]] = {}
        for var, (dims, *_rest) in held.items():
            by_dims.setdefault(dims, []).append(var)
        for dims, names in by_dims.items():
            table = here + ("_".join(dims) or "scalars") + ".csv"
            axis_cells = [text(axes[d][1]) if d in axes
                          else [str(i) for i in range(sizes[d])]
                          for d in dims]
            columns = [text(held[n][1]) for n in names]
            out = io.StringIO()
            w = csv.writer(out, lineterminator="\n")
            w.writerow(list(dims) + [header(n, held[n][2]) for n in names])
            for k, point in enumerate(itertools.product(*axis_cells)):
                w.writerow(list(point) + [c[k] for c in columns])
            tables[table] = out.getvalue()
            rows = int(np.prod([sizes[d] for d in dims])) if dims else 1
            lines.extend([f"{table} -- over {', '.join(dims) or 'nothing'}, "
                          f"{rows} row{'s' if rows != 1 else ''}"])
            for var in [*dims, *names]:
                _d, _a, info, fill = (axes.get(var) or held.get(var)
                                      or ((), None, {}, None))
                what = "; ".join(
                    f"{k} {info[k]}" for k in ("long_name", "units", "kind",
                                               "definition", "source")
                    if info.get(k) not in (None, ""))
                if fill is not None and not (isinstance(fill, float)
                                             and fill != fill):
                    what += (f"; its fill, {attr(fill)} in the file, an "
                             f"empty cell here")
                lines.append(f"  {header(var, info) if var in held else var}"
                             f": {what or 'an index'}")
            lines.append("")
    return tables, "\n".join(lines) + "\n"


__all__ = ["Var", "Axis", "AXES", "check_axes", "ATOM_AXIS", "FLAG_FILL", "NC_LOCK", "as_tables", "empty",
           "read_atom_index_map", "size_of", "flag", "unflag", "number_attr",
           "reading", "OpenedTwice",
           "unnumber_attr", "json_attr", "write_group", "in_memory",
           "as_bytes", "read_variables"]
