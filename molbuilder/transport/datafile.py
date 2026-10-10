"""THE TRANSPORT DATA FILE -- ``<label>.transport.nc``, the record's results
as data on their grid (`engines/transport.md` § 2a.12, *the data file*).

MODULE  transport.datafile (L2; numpy, netCDF4 and molbuilder's own doors)
ROLE    one NetCDF-4 file per transport calculation, frame x voltage x
        energy: every point's raw outputs beside what molbuilder derives from
        them, a mode's frame set's definition beside every frame's values,
        the mode's average -- each variable stating its units, its long name,
        its definition and, for a raw one, the file it was read from.  The
        variables are :data:`VARIABLES`, one table the writer fills and the
        reader reads; :func:`write_data_file` writes it from the composition
        `summarize task` writes the JSON record from; :func:`read_data_file`
        is its door
USED-BY `jobset summarize task`, beside `record.write_record`

**One shape for every transport calculation** (user, 2026-10-10: "doing a
derivative is trivial if the result are saved with these parameter and
calculation results consistently aligned and logically organized"): one
structure at one voltage is a 1 x 1 grid, never another layout.  A point not
done is the fill value, with ``done`` 0 -- a gap, never filled in.  The JSON
record stays the report the Results tab and `summarize` read; this file is
the same results for analysis -- an average, a slope along the mode's ``q``,
a comparison of modes is a line of xarray on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

#: The file's schema, stated in it.
DATA_SCHEMA = "molbuilder/transport-data@1"

#: The regions molbuilder owns in the device, in order (`transport.sort`).
_REGIONS = ("L-electrode", "bridge", "R-electrode")

#: How far a point's DOS energies, read from its ``.TBT.nc`` at full
#: precision, may be from its transmission's, which TBtrans prints to five
#: decimals of an eV (``f10.5``, ``m_tbt_save.F90`` ``save_DAT``): the same
#: points, within half that last decimal.
AVTRANS_ENERGY_HALF_DECIMAL_EV = 0.5e-5


class DataFileError(ValueError):
    """The record does not lay out on one grid -- the message names what
    disagrees, ready to surface verbatim."""


@dataclass(frozen=True)
class Var:
    """One variable of the file: its ``name``, its dimensions, its ``kind``
    (``coordinate``, ``given``, ``raw`` -- as an engine wrote it --,
    ``derived``, ``state``), ``units``, ``long_name``, ``definition`` and,
    for a raw one, ``source`` -- the file it was read from."""
    name: str
    dims: Tuple[str, ...]
    kind: str
    units: str
    long_name: str
    definition: str
    source: str = ""
    dtype: Any = "f8"


def _frame_set_vars() -> Tuple[Var, ...]:
    """Every per-frame row of a mode's frame set, along ``frame`` -- the one
    definition's own names, kinds and units (`frameset.FRAME_ROWS`)."""
    from ..frameset import FRAME_ROWS
    words = {
        "displacement_ang": ("the frame's plain displacement from frame 0",
                             "sqrt(sum_A |R_A - R0_A|^2) over every atom, no "
                             "masses"),
        "max_atom_displacement_ang": ("its largest single atom's displacement",
                                      "max_A |R_A - R0_A|"),
        "q_amu12_ang": ("the frame's position along the mode",
                        "the mode's normal coordinate: every atom displaced by "
                        "q * L_A; |q| = sqrt(sum_A m_A |R_A - R0_A|^2), its "
                        "sign the side of frame 0"),
        "node_sigma": ("the frame's position in units of the mode's spread",
                       "q_amu12_ang / sigma_amu12_ang"),
        "weight": ("the frame's share of the mode's average",
                   "as the frame set states it; the set's weights sum to 1"),
    }
    return tuple(Var(r.name, ("frame",), r.kind, r.unit or "1",
                     words[r.name][0], words[r.name][1]
                     + " (model/structure.md 2.2f; the fill value where the "
                       "set states none)")
                 for r in FRAME_ROWS)


def _variables() -> Tuple[Var, ...]:
    S = str
    return (
        Var("frame", ("frame",), "coordinate", "1", "frame index",
            "counted from 0, as MolView's API counts; a person counts from 1 "
            "(model/structure.md 2.2e)", dtype="i4"),
        Var("frame_token", ("frame",), "coordinate", "", "the frame's folder",
            "counted from 1: f001 is frame 0 (engines/transport.md 2a.11)",
            dtype=S),
        Var("bias_v", ("bias",), "coordinate", "V", "bias voltage",
            "the voltage the device and the transmission ran at"),
        Var("energy_ev", ("energy",), "coordinate", "eV", "energy",
            "E - E_F, relative to the leads' Fermi level: the transmission's "
            "grid, as .TBT.AVTRANS prints it, to 5 decimals; each DOS, read "
            "from .TBT.nc, is on the same points within half that decimal"),
        Var("spin", ("spin",), "coordinate", "", "spin channel",
            "unpolarized, or up and down", dtype=S),
        Var("iv_bias_v", ("iv_bias",), "coordinate", "V", "the I-V's voltage",
            "the description's voltage list"),
        Var("atom", ("atom",), "coordinate", "1", "atom index",
            "counted from 0 in the composed junction -- the sorted copy every "
            "deck is written in; atom-permutation.json maps back to the input",
            dtype="i4"),
        Var("xyz", ("xyz",), "coordinate", "", "Cartesian component",
            "x, y, z", dtype=S),
        Var("element", ("atom",), "given", "", "the atom's element",
            "as the composed junction states it", dtype=S),
        Var("mass_amu", ("atom",), "given", "amu", "the atom's mass",
            "the frame set's mass_amu channel -- what its q is weighted by "
            "(model/structure.md 2.2f; the fill value where the set states "
            "none)"),
        Var("region", ("region",), "coordinate", "", "device region",
            "the regions molbuilder owns: L-electrode, bridge, R-electrode",
            dtype=S),
        Var("lead", ("lead",), "coordinate", "", "lead",
            "TBtrans's electrode names", dtype=S),
        Var("eigenchannel", ("eigenchannel",), "coordinate", "1",
            "eigenchannel", "counted from 0, as TBtrans orders them",
            dtype="i4"),
        *_frame_set_vars(),
        Var("positions_ang", ("frame", "atom", "xyz"), "given", "A",
            "atom positions", "every frame's coordinates as composed"),
        Var("transmission_by_spin", ("frame", "bias", "spin", "energy"),
            "raw", "1", "transmission, by spin channel",
            "T(E), k-averaged, channel by channel",
            source="<label>.TBT.AVTRANS_<L>-<R> (TBT_UP / TBT_DN polarized)"),
        Var("transmission", ("frame", "bias", "energy"), "derived", "1",
            "transmission per spin channel",
            "the one channel, or (T_up + T_down) / 2 -- what G is computed "
            "from"),
        Var("conductance_g0", ("frame", "bias"), "derived", "G0",
            "conductance", "G / G0 = T at E - E_F = 0, linearly interpolated; "
            "G0 = 2 e^2 / h"),
        Var("current_a_printed", ("frame", "bias"), "raw", "A",
            "current, as TBtrans printed it", "one spin channel's",
            source="the transmission point's TBtrans output"),
        Var("current_a", ("frame", "bias"), "derived", "A",
            "the junction's total current",
            "twice the printed figure unpolarized, the channels' sum "
            "polarized"),
        Var("iv_current_a", ("frame", "iv_bias"), "derived", "A",
            "the I-V's current",
            "each point's own total (self-consistent), or the record's linear "
            "response from that frame's 0 V transmission (low-bias)"),
        Var("device_ef_ev", ("frame", "bias"), "raw", "eV",
            "the device point's Fermi level",
            "its NEGF phase's last cycle, in TranSIESTA's frame",
            source="the device point's SIESTA output"),
        Var("device_vha_ev", ("frame", "bias"), "raw", "eV",
            "the device point's boundary Hartree potential",
            "ts-Vha, the shift between the device's frame and the seed's",
            source="the device point's SIESTA output"),
        Var("dos_total", ("frame", "bias", "energy"), "raw", "1/eV",
            "the device's density of states", "its Green-function DOS, "
            "k-averaged", source="<label>.TBT.nc"),
        Var("dos_region", ("frame", "bias", "region", "energy"), "raw",
            "1/eV", "the device DOS by region",
            "summed over the region's atoms in the device",
            source="<label>.TBT.nc"),
        Var("lead_spectral_dos", ("frame", "bias", "lead", "energy"), "raw",
            "1/eV", "a lead's spectral DOS", "TBT.DOS.A, k-averaged",
            source="<label>.TBT.nc"),
        Var("lead_bulk_dos", ("frame", "bias", "lead", "energy"), "raw",
            "1/eV", "a lead's bulk DOS", "TBT.DOS.Elecs, k-averaged",
            source="<label>.TBT.nc"),
        Var("eigenchannel_transmission",
            ("frame", "bias", "eigenchannel", "energy"), "raw", "1",
            "eigenchannel transmission", "TBT.T.Eig, k-averaged",
            source="<label>.TBT.nc"),
        Var("done", ("frame", "bias"), "state", "1", "the point is done",
            "1 when the point has its transmission; 0 otherwise, its values "
            "the fill value -- a gap, never filled in", dtype="i1"),
        Var("point_folder", ("frame", "bias"), "state", "",
            "the transmission point's folder",
            "tree-relative: where its raw values were read", dtype=S),
        Var("average_transmission", ("bias", "energy"), "derived", "1",
            "the mode's average transmission",
            "sum_f weight_f T_f(E) over every frame, at its stated weight"),
        Var("average_delta_transmission", ("bias", "energy"), "derived", "1",
            "the average's change", "<T(E)> - T_0(E), from frame 0's"),
        Var("average_conductance_g0", ("bias",), "derived", "G0",
            "the mode's averaged conductance", "sum_f weight_f G_f / G0"),
        Var("average_conductance_change_percent", ("bias",), "derived", "%",
            "the averaged conductance's change",
            "100 (<G> - G_0) / G_0, in per cent of frame 0's"),
    )


#: THE FILE'S VARIABLES -- one table the writer fills and the reader reads.
VARIABLES: Tuple[Var, ...] = _variables()


def data_file_path(base_dir, label: str) -> Path:
    """The ONE spelling of the file's location, composed by the run-file
    catalogue (`runfiles.compose`), as `record.record_path` is."""
    from ..runfiles import compose
    return Path(base_dir) / compose(label, ".transport.nc")


def write_data_file(base_dir, task, record: Dict) -> Path:
    """Write ``<label>.transport.nc`` from ``record`` -- the composition
    `record.collect_record` answers -- and the calculation's composed
    junction; returns its path.  :class:`DataFileError` names what does not
    lay out on one grid."""
    import netCDF4
    from ..frameset import STRUCTURE_ROWS, read as read_frame_set
    from ..persist import write_bytes
    from .record import composed_junction
    from .stages import frame_token, frames_of, rung_points
    base = Path(base_dir)
    junction = composed_junction(base)
    frame_set = read_frame_set(junction) if junction is not None else None
    n_frames = junction.n_frames if junction is not None else frames_of(base)
    biases = sorted({float(pt.bias_v) for pt in rung_points(
        task, "transmission", frames=n_frames)} or {0.0})
    done_points = list(record.get("points") or ())
    grid = done_points[0]["energy_ev"] if done_points else []
    for p in done_points:
        if p["energy_ev"] != grid:
            raise DataFileError(
                f"{p.get('point') or 'a point'}'s transmission is on another "
                f"energy grid than the first point's -- every point's "
                f"transmission deck is the one template's")
    polarized = any(p.get("spin") == "polarized" for p in done_points)
    spins = ["up", "down"] if polarized else ["unpolarized"]
    leads = sorted({lead for p in done_points
                    for key in ("lead_spectral", "lead_bulk")
                    for lead in ((p.get("dos") or {}).get(key) or {})})
    n_eig = max((len((p.get("dos") or {}).get("eigenchannels") or ())
                 for p in done_points), default=0)
    iv = record.get("iv") or {}
    iv_biases: List[float] = []
    for v in iv.get("voltages_v") or ():
        if not any(abs(float(v) - u) < 1e-9 for u in iv_biases):
            iv_biases.append(float(v))
    n_atoms = junction.n_atoms if junction is not None else 0

    dims = {"frame": n_frames, "bias": len(biases), "energy": len(grid),
            "spin": len(spins), "iv_bias": len(iv_biases), "atom": n_atoms,
            "xyz": 3, "region": len(_REGIONS), "lead": len(leads),
            "eigenchannel": n_eig}
    values: Dict[str, Any] = {}
    nan = np.nan

    def full(name):
        var = next(v for v in VARIABLES if v.name == name)
        shape = tuple(dims[d] for d in var.dims)
        if var.dtype is str:
            return np.full(shape, "", dtype=object)
        if var.dtype in ("i1", "i4"):
            return np.zeros(shape, dtype=var.dtype)
        return np.full(shape, nan)
    for v in VARIABLES:
        values[v.name] = full(v.name)
    values["frame"][:] = np.arange(n_frames)
    for f in range(n_frames):
        values["frame_token"][f] = frame_token(f)
    values["bias_v"][:] = biases
    values["energy_ev"][:] = grid
    values["spin"][:] = spins
    values["iv_bias_v"][:] = iv_biases
    values["atom"][:] = np.arange(n_atoms)
    values["xyz"][:] = ["x", "y", "z"]
    values["region"][:] = list(_REGIONS)
    values["lead"][:] = leads
    values["eigenchannel"][:] = np.arange(n_eig)
    if junction is not None:
        values["element"][:] = [str(e) for e in junction.elements]
        values["positions_ang"][:] = np.asarray(
            junction.frames if n_frames > 1 else [junction.positions])
    if frame_set is not None:
        values["mass_amu"][:] = frame_set.masses_amu
        for f, rows in enumerate(frame_set.frames):
            for name in rows.__dataclass_fields__:
                values[name][f] = getattr(rows, name)

    def at(p) -> Tuple[int, int]:
        f = p.get("frame")
        b = next(k for k, u in enumerate(biases)
                 if abs(float(p["bias_v"]) - u) < 1e-9)
        return (0 if f is None else int(f)), b
    for p in done_points:
        f, b = at(p)
        values["done"][f, b] = 1
        values["point_folder"][f, b] = p.get("attempt") or ""
        values["transmission"][f, b] = p["transmission"]
        chans = p.get("channels") or {}
        for s, name in enumerate(spins):
            values["transmission_by_spin"][f, b, s] = (
                chans[name] if polarized else p["transmission"])
        for name in ("conductance_g0", "current_a", "current_a_printed"):
            if p.get(name) is not None:
                values[name][f, b] = p[name]
        dos = p.get("dos") or {}
        if dos.get("energy_ev") is not None and not (
                len(dos["energy_ev"]) == len(grid)
                and np.allclose(dos["energy_ev"], grid, rtol=0.0,
                                atol=AVTRANS_ENERGY_HALF_DECIMAL_EV)):
            raise DataFileError(
                f"{p.get('point') or 'a point'}'s DOS is on another energy "
                f"grid than its transmission -- not the same points within "
                f"the {AVTRANS_ENERGY_HALF_DECIMAL_EV:g} eV the transmission "
                f"file's five decimals allow")
        if dos.get("total") is not None:
            values["dos_total"][f, b] = dos["total"]
        for r, region in enumerate(_REGIONS):
            if region in (dos.get("by_label") or {}):
                values["dos_region"][f, b, r] = dos["by_label"][region]
        for key, name in (("lead_spectral", "lead_spectral_dos"),
                          ("lead_bulk", "lead_bulk_dos")):
            for k, lead in enumerate(leads):
                if lead in (dos.get(key) or {}):
                    values[name][f, b, k] = dos[key][lead]
        for k, curve in enumerate(dos.get("eigenchannels") or ()):
            values["eigenchannel_transmission"][f, b, k] = curve
    for p in list(record.get("pending") or ()) + list(record.get("failed")
                                                       or ()):
        f, b = at(p)
        values["point_folder"][f, b] = p.get("attempt") or ""
    for f, v, i in zip(iv.get("frame") or [None] * len(iv_biases),
                       iv.get("voltages_v") or (), iv.get("current_a") or ()):
        k = next(k for k, u in enumerate(iv_biases)
                 if abs(float(v) - u) < 1e-9)
        if i is not None:
            values["iv_current_a"][0 if f is None else int(f), k] = i
    _device_facts(record, biases, values)
    average = record.get("average") or {}
    for e in average.get("at") or ():
        b = next((k for k, u in enumerate(biases)
                  if abs(float(e["bias_v"]) - u) < 1e-9), None)
        if b is None:
            continue
        for key, name in (("transmission", "average_transmission"),
                          ("delta_transmission",
                           "average_delta_transmission"),
                          ("conductance_g0", "average_conductance_g0"),
                          ("conductance_change_percent",
                           "average_conductance_change_percent")):
            if e.get(key) is not None:
                values[name][b] = e[key]

    attrs: Dict[str, Any] = {
        "schema": DATA_SCHEMA,
        "label": str(record.get("label") or task.label),
        "treatment": str(record.get("treatment") or ""),
        "energies_relative_to_ef": 1,
        "caveat": str(record.get("caveat") or ""),
        "frames": int(n_frames),
    }
    slot = ((record.get("provenance") or {}).get("slot") or {})
    for key in ("citation", "kind"):
        if slot.get(key) is not None:
            attrs[f"citation_{key}" if key != "citation" else "citation"] = (
                str(slot[key]))
    if frame_set is not None:
        for r in STRUCTURE_ROWS:
            attrs[r.name] = getattr(frame_set, r.name)
        for k in ("run", "result_sha256"):
            attrs[f"vibration_{k}"] = str(frame_set.vibration.get(k) or "")
        attrs["weight_sum"] = float(average.get("weight_sum"))
        attrs["weight_tolerance"] = float(average.get("tolerance"))
        attrs["average_assumptions"] = json.dumps(
            list(average.get("assumptions") or ()))
    if average.get("why"):
        attrs["average_why"] = str(average["why"])

    ds = netCDF4.Dataset("transport.nc", "w", memory=4096, format="NETCDF4")
    for name, size in dims.items():
        ds.createDimension(name, size)
    for var in VARIABLES:
        kw = {} if var.dtype is str or var.dtype in ("i1", "i4") else {
            "fill_value": np.nan}
        nc = ds.createVariable(var.name, var.dtype, var.dims, **kw)
        nc.setncatts({"kind": var.kind, "units": var.units,
                      "long_name": var.long_name,
                      "definition": var.definition,
                      **({"source": var.source} if var.source else {})})
        data = values[var.name]
        if data.size:
            nc[...] = data
    ds.setncatts(attrs)
    out = data_file_path(base, attrs["label"])
    write_bytes(out, bytes(ds.close()))
    return out


def _device_facts(record: Dict, biases: Sequence[float],
                  values: Dict[str, Any]) -> None:
    """Each point's device facts -- its NEGF Fermi level and boundary
    potential -- from the record's device rung: a rung that runs per point,
    point by point; one that runs once, its one run's, at every point."""
    device = next((s for s in record.get("stages") or ()
                   if s.get("stage") == "device"), None) or {}
    if device.get("by_point"):
        for p in device["by_point"]:
            negf = p.get("negf") or {}
            f = 0 if p.get("frame") is None else int(p["frame"])
            b = next((k for k, u in enumerate(biases)
                      if abs(float(p["bias_v"]) - u) < 1e-9), None)
            if b is None:
                continue
            for key, name in (("ef", "device_ef_ev"),
                              ("vha_ev", "device_vha_ev")):
                if negf.get(key) is not None:
                    values[name][f, b] = negf[key]
        return
    negf = device.get("negf") or {}
    for key, name in (("ef", "device_ef_ev"), ("vha_ev", "device_vha_ev")):
        if negf.get(key) is not None:
            values[name][...] = negf[key]


def read_data_file(path) -> Dict[str, Any]:
    """``<label>.transport.nc`` as data: ``{"attrs": {...}, "variables":
    {name: {"dims", "data", "attrs"}}}`` -- each variable's values as numpy
    arrays (the fill value NaN), its units, long name, definition and source.
    The file's door (`runfiles.WRITTEN`); refuses another schema by name."""
    import netCDF4
    with netCDF4.Dataset(str(path), "r") as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        if attrs.get("schema") != DATA_SCHEMA:
            raise DataFileError(f"{Path(path).name} states the schema "
                                f"{attrs.get('schema')!r}, not {DATA_SCHEMA}")
        ds.set_auto_mask(False)
        variables = {
            name: {"dims": tuple(v.dimensions),
                   "data": np.asarray(v[...]),
                   "attrs": {k: v.getncattr(k) for k in v.ncattrs()
                             if k != "_FillValue"}}
            for name, v in ds.variables.items()}
    return {"attrs": attrs, "variables": variables}
