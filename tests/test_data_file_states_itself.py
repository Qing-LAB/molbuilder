"""A data file states itself (`model/datafiles.md` § 2, § 2b, § 2c): every
variable its long name and definition, a number in one spelling of the
units table, a text no unit; every group what it holds; the file its title,
schema, molbuilder version and source; no quantity kept as an attribute --
each refused by the one writer, naming what is missing; and the file the
meaning of every unit it uses.

PREVENTS: a file molbuilder writes that a reader without molbuilder cannot
read -- a value with no meaning, `A` read as ampere where it meant
angstrom, a number whose unit is only its name's suffix, a group nobody
can tell the purpose of.  The writer is the one door every data file goes
through, so its refusal is the rule's whole enforcement; the files' own
tests write through it.
"""
from __future__ import annotations

import json

import pytest

from molbuilder.ncdata import Var

_E = Var("energy_ev", ("energy_ev",), "coordinate", "eV", "energy",
         "relative to E_F")


def _write(variables, *, title="the group", stated=True, attrs=None,
           group=True):
    from molbuilder.ncdata import (NC_LOCK, as_bytes, empty, in_memory, state,
                                   write_group)
    dims = {"energy_ev": 2}
    values = {v.name: empty(v, dims) for v in variables}
    with NC_LOCK:
        ds = in_memory("s.nc")
        try:
            g = ds.createGroup("g") if group else ds
            write_group(g, dims, variables, values,
                        **({"title": title} if group else {}))
            if attrs:
                g.setncatts(attrs)
            if stated:
                state(ds, title="a test file", schema="test@1",
                      molbuilder_version="0", source="the test")
        except ValueError:
            ds.close()
            raise
        return as_bytes(ds)


@pytest.mark.parametrize("case", [
    {"variables": (_E, Var("t", ("energy_ev",), "raw", "1", "T(E)", "")),
     "says": ["variable t states no definition"]},
    {"variables": (_E, Var("t", ("energy_ev",), "raw", "1", "", "as run")),
     "says": ["variable t states no long_name"]},
    {"variables": (_E, Var("x", ("energy_ev",), "given", "Angstrom", "x",
                           "a length")),
     "says": ["variable x states the unit 'Angstrom'", "datafiles.md 2c"]},
    {"variables": (_E, Var("x", ("energy_ev",), "given", "", "x", "a number")),
     "says": ["variable x states the unit ''", "'1' for none"]},
    {"variables": (_E, Var("w", ("energy_ev",), "given", "1", "a word",
                           "a label", dtype=str)),
     "says": ["variable w is text and states the unit '1'"]},
    {"variables": (_E,), "title": "",
     "says": ["a group states what it holds"]},
    {"variables": (_E,), "stated": False,
     "says": ["the file states no title, schema, molbuilder_version, source"]},
    {"variables": (_E,), "attrs": {"delta_ang": 0.01},
     "says": ["/g/ keeps the number delta_ang as an attribute"]},
], ids=["no definition", "no long name", "another spelling", "a number with "
        "no unit", "a text with a unit", "a group with no title",
        "a file not stated", "a quantity as an attribute"])
def test_the_writer_refuses_a_file_that_does_not_state_itself(case):
    with pytest.raises(ValueError) as refused:
        _write(case["variables"], title=case.get("title", "the group"),
               stated=case.get("stated", True), attrs=case.get("attrs"))
    for words in case["says"]:
        assert words in str(refused.value), str(refused.value)


def test_a_file_that_states_itself_says_what_its_units_mean():
    from molbuilder.ncdata import UNITS, reading
    data = _write((_E, Var("dos", ("energy_ev",), "raw", "1/eV", "the DOS",
                           "summed over the device"),
                   Var("count", (), "given", "1", "a count", "of things"),
                   Var("word", ("energy_ev",), "given", "", "a word",
                       "a label", dtype=str)),
                  attrs={"n_frames": 3, "note": "a sentence"})
    with reading("s.nc", data=data) as ds:
        root = {k: ds.getncattr(k) for k in ds.ncattrs()}
        assert root["title"] == "a test file" and root["schema"] == "test@1"
        assert root["Conventions"] == "CF-1.8" and root["source"] == "the test"
        assert json.loads(root["unit_meanings"]) == {
            u: UNITS[u] for u in ("1", "1/eV", "eV")}
        g = ds["g"]
        assert g.getncattr("title") == "the group"
        assert "units" not in g["word"].ncattrs()
