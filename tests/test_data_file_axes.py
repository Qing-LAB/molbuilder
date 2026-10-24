"""Every axis of every data file is a row of the axis table, with its
coordinate (`model/datafiles.md` § 2a; `ncdata.AXES`, `ncdata.check_axes`):
the one writer refuses a dimension that is not a row, one with no coordinate
of its own name, and one whose coordinate states other units than its axis.

PREVENTS: one concept under two names in two files (the digest's `energy`
beside the transport data file's `energy_ev`), or an axis a reader cannot
select by value -- each found only by a person whose script breaks on one
file and not the other.  The writer is the one door every data file goes
through, so its refusal is the rule's whole enforcement; the files' own
tests write through it (`tests/spectra/test_datafile.py`, the transport e2e).
"""
from __future__ import annotations

import pytest

from molbuilder.ncdata import Var

_ENERGY = Var("energy_ev", ("energy_ev",), "coordinate", "eV", "energy",
              "relative to E_F")
_T = Var("transmission", ("energy_ev",), "raw", "", "T(E)",
         "as the run states it", source="T.TBT.AVTRANS_L-R")


@pytest.mark.parametrize("case", [
    # rule 4: a dimension that is not a row is refused, naming the section
    {"dims": {"energy": 3},
     "variables": (Var("energy", ("energy",), "coordinate", "eV", "energy",
                       "relative to E_F"),),
     "says": ["axis energy is not an axis of the data files",
              "model/datafiles.md § 2a"]},
    # rule 3: an axis with no coordinate of its own name
    {"dims": {"energy_ev": 3}, "variables": (_T,),
     "says": ["axis energy_ev has no coordinate of its own name"]},
    # the row's units: a coordinate stating others
    {"dims": {"energy_ev": 3},
     "variables": (Var("energy_ev", ("energy_ev",), "coordinate", "Ry",
                       "energy", "relative to E_F"), _T),
     "says": ["axis energy_ev's coordinate states units 'Ry'",
              "the axis is in 'eV'"]},
    # a dimension a variable uses, though the group does not create it
    {"dims": {"energy_ev": 3},
     "variables": (_ENERGY, Var("dos_by_column", ("energy_ev", "column"),
                                "raw", "1/eV", "DOS", "per column")),
     "says": ["axis column is not an axis of the data files"]},
], ids=["not a row", "no coordinate", "other units", "used, not created"])
def test_the_writer_refuses_an_axis_the_table_does_not_hold(case):
    from molbuilder.ncdata import NC_LOCK, empty, in_memory, write_group
    sizes = {**case["dims"], "column": 2}
    values = {v.name: empty(v, sizes) for v in case["variables"]}
    with NC_LOCK:
        ds = in_memory("axes.nc")
        with pytest.raises(ValueError) as refused:
            write_group(ds, case["dims"], case["variables"], values)
        ds.close()
    for words in case["says"]:
        assert words in str(refused.value), str(refused.value)

