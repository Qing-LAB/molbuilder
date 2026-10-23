"""A DATA FILE'S VARIABLES, EACH STATING ITSELF -- the variable record and the
writing loop every NetCDF file molbuilder writes shares (`engines/transport.md`
§ 2a.12, *the data file*; `engines/vibration.md` § 6.10.5).

MODULE  ncdata (L1; numpy and netCDF4)
ROLE    :class:`Var` -- one variable: its dimensions, its kind (given, raw,
        derived, state, coordinate), its units, its words; :func:`empty` --
        the array a variable starts as, its fill stated; :func:`write_group`
        -- a group's dimensions and variables written from a table of
        :class:`Var`, each with its attributes and its CF ``coordinates``;
        :func:`in_memory` / :func:`as_bytes` -- the file built in memory and
        handed to the one atomic writer (`persist.write_bytes`);
        :data:`NC_LOCK` -- held by every NetCDF access the server reaches
USED-BY the transport data file (`transport.datafile`), the vibration
        result's data file (`spectra.datafile`), beside the job in both
        bundles (`runwrap.VIBRATION_COMPANIONS`, `runwrap.PYSCF_COMPANIONS`),
        and a TBtrans data file's digest (`parse.engines.tbtrans_nc`)

**Encoding, so a round trip is exact**: a number's fill is NaN, a text's the
empty string, an integer's its stated fill; a boolean is an ``int8`` with
``flag_values 0 1`` (:func:`flag`); an empty list is a dimension of length
0; a nested dict is JSON text (:func:`json_attr`).  Imports netCDF4 at load:
a job whose environment lacks it stops here, before the expensive part
(`engines/vibration.md` § 6.10.6).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import netCDF4
import numpy as np


@dataclass(frozen=True)
class Var:
    """One variable of the file: its ``name``, its dimensions, its ``kind``
    (``coordinate``, ``given``, ``raw`` -- as an engine wrote it --,
    ``derived``, ``state``), ``units``, ``long_name``, ``definition`` and,
    for a raw one, ``source`` -- the file it was read from.  A
    ``coordinate`` names a dimension's points: the dimension's own values
    when it is named as the dimension (a NetCDF coordinate variable), else
    a label of them.  An ``auxiliary`` variable labels the points of its one
    dimension -- each variable over that dimension names it in its CF
    ``coordinates`` attribute, so a NetCDF reader opens it as a
    coordinate.  ``fill`` is an integer variable's fill value, where a cell
    can be not stated (a flag of a run that did not run: neither 0 nor 1);
    a number's is NaN, a text's the empty string.  A boolean variable is
    an ``int8`` with ``flag=True``: ``flag_values 0 1`` beside it."""
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


def empty(var: Var, dims: Mapping[str, int]) -> np.ndarray:
    """The array ``var`` starts as, over ``dims``: every cell its fill."""
    shape = tuple(int(dims[d]) for d in var.dims)
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
    """A nested block as an attribute: its JSON text."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=False)


def write_group(group, dims: Mapping[str, int], variables: Sequence[Var],
                values: Mapping[str, np.ndarray]) -> None:
    """``dims`` created on ``group`` (a dataset or a group) and every
    variable of ``variables`` written from ``values[name]`` with its
    attributes -- ``kind``, ``units``, ``long_name``, ``definition``, its
    ``source`` where raw, ``flag_values`` where a flag, and the CF
    ``coordinates`` naming each auxiliary coordinate along its
    dimensions.  Numbers are compressed; the fill value is said."""
    for name, size in dims.items():
        group.createDimension(name, int(size))
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
                      **({"coordinates": " ".join(labels)} if labels
                         else {})})
        data = values[var.name]
        if data.size:
            nc[...] = data


def in_memory(name: str) -> "netCDF4.Dataset":
    """A NetCDF-4 dataset built in memory, to be handed to the one atomic
    writer as bytes (:func:`as_bytes`)."""
    return netCDF4.Dataset(name, "w", memory=4096, format="NETCDF4")


def as_bytes(ds: "netCDF4.Dataset") -> bytes:
    """The in-memory dataset closed, as the file's bytes."""
    return bytes(ds.close())


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


__all__ = ["Var", "FLAG_FILL", "NC_LOCK", "empty", "flag", "unflag", "number_attr",
           "unnumber_attr", "json_attr", "write_group", "in_memory",
           "as_bytes", "read_variables"]
