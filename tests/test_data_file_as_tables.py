"""A data file laid out as tables (`ncdata.as_tables`; `engines/transport.md`
§ 2a.12, *Export*): one long table per set of axes, each axis a column of its
coordinate's values, each variable a column named with its units, a value
the file does not state an empty cell, and a README in the variables' own
words.

PREVENTS: a CSV export whose values sit on the wrong axis point -- a table
read in another order than its axes are written -- or whose unstated values
read as numbers (NaN, a flag's -1) a spreadsheet would plot.  The file is
built by the one writer every data file molbuilder writes goes through
(`ncdata.write_group`), and read back with the standard `csv` module.
"""
from __future__ import annotations

import csv
import io

import numpy as np


def _a_data_file() -> bytes:
    from molbuilder.ncdata import (NC_LOCK, Var, as_bytes, empty, in_memory,
                                   state, write_group)
    variables = (
        Var("frame", ("frame",), "coordinate", "1", "frame", "its index",
            dtype="i4"),
        Var("energy_ev", ("energy_ev",), "coordinate", "eV", "energy",
            "relative to E_F"),
        Var("frame_token", ("frame",), "coordinate", "", "frame's folder",
            "its token", dtype=str, auxiliary=True),
        Var("transmission", ("frame", "energy_ev"), "raw", "1", "T(E)",
            "as the run states it", source="T.TBT.AVTRANS_L-R"),
        Var("current_a", ("frame",), "derived", "A", "the current",
            "the junction's total"),
        Var("done", ("frame",), "state", "1", "the point is done",
            "1 when done", dtype="i1", flag=True),
    )
    dims = {"frame": 2, "energy_ev": 3}
    values = {v.name: empty(v, dims) for v in variables}
    values["frame"][:] = [0, 1]
    values["energy_ev"][:] = [-0.5, 0.0, 0.5]
    values["frame_token"][:] = ["f000", "f001"]
    values["transmission"][0] = [0.1, 0.2, 0.3]
    values["transmission"][1] = [0.4, np.nan, 0.6]     # one value not stated
    values["current_a"][:] = [1e-6, np.nan]
    values["done"][:] = [1, -1]                         # not stated: its fill
    with NC_LOCK:
        ds = in_memory("x.nc")
        write_group(ds, dims, variables, values)
        state(ds, title="a transport calculation's transmission",
              schema="test@1", molbuilder_version="0", source="the test")
        ds.setncatts({"label": "T"})
        return as_bytes(ds)


def _rows(text: str):
    return list(csv.DictReader(io.StringIO(text)))


def test_each_value_sits_on_its_axes_point_and_unstated_is_empty():
    from molbuilder.ncdata import as_tables
    tables, readme = as_tables(_a_data_file(), "T.transport.nc")
    assert sorted(tables) == ["frame.csv", "frame_energy_ev.csv"]
    by_point = {(r["frame"], r["energy_ev"]): r["transmission"]
                for r in _rows(tables["frame_energy_ev.csv"])}
    assert by_point == {("0", "-0.5"): "0.1", ("0", "0.0"): "0.2",
                        ("0", "0.5"): "0.3", ("1", "-0.5"): "0.4",
                        ("1", "0.0"): "", ("1", "0.5"): "0.6"}
    frames = _rows(tables["frame.csv"])
    assert [(r["frame"], r["frame_token"], r["current_a [A]"], r["done"])
            for r in frames] == [("0", "f000", "1e-06", "1"),
                                 ("1", "f001", "", "")]
    # THE README, IN THE VARIABLES' OWN WORDS: the file's attributes, each
    # table and each column's long name, units, kind, definition, source.
    assert readme.startswith("T.transport.nc, as tables")
    assert "label: T" in readme
    # A FILL THE FILE STATES, said beside the column that holds it.
    assert "its fill, -1 in the file, an empty cell here" in readme
    assert ("transmission: long_name T(E); units 1; kind raw; definition as "
            "the run states it; source T.TBT.AVTRANS_L-R") in readme
    assert "current_a [A]: long_name the current; units A" in readme


def test_a_groups_variables_are_tables_under_its_name():
    """A data file in groups -- the vibration's (`spectra.datafile`) --
    lays out the same way, each group's tables under its own path, each
    axis by its coordinate's values."""
    from molbuilder.ncdata import (NC_LOCK, Var, as_bytes, empty, in_memory,
                                   state, write_group)
    from molbuilder.ncdata import as_tables
    mode = (Var("mode", ("mode",), "coordinate", "1", "the mode's index",
                "from 0", dtype="i4"),
            Var("xyz", ("xyz",), "coordinate", "", "Cartesian component",
                "x, y, z", dtype=str),
            Var("frequency_cm1", ("mode",), "derived", "cm-1",
                "a mode's frequency", "from the Hessian"),
            Var("displacement", ("mode", "xyz"), "derived", "1",
                "the mode's direction", "normalized"))
    with NC_LOCK:
        ds = in_memory("v.nc")
        g = ds.createGroup("modes")
        values = {v.name: empty(v, {"mode": 2, "xyz": 3}) for v in mode}
        values["mode"][:] = [0, 1]
        values["xyz"][:] = ["x", "y", "z"]
        values["frequency_cm1"][:] = [100.0, 250.0]
        values["displacement"][:] = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        write_group(g, {"mode": 2, "xyz": 3}, mode, values,
                    title="the normal modes")
        state(ds, title="a vibration result", schema="test@1",
              molbuilder_version="0", source="the test")
        data = as_bytes(ds)
    tables, readme = as_tables(data, "V.spectra.nc")
    assert sorted(tables) == ["modes/mode.csv", "modes/mode_xyz.csv"]
    assert [r["frequency_cm1 [cm-1]"] for r in
            _rows(tables["modes/mode.csv"])] == ["100.0", "250.0"]
    # EACH AXIS BY ITS COORDINATE'S VALUES, in the group's table.
    assert [(r["mode"], r["xyz"]) for r in _rows(tables["modes/mode_xyz.csv"])
            if r["displacement"] == "1.0"] == [("0", "x"), ("1", "z")]
    assert "modes/mode.csv -- over mode, 2 rows" in readme
