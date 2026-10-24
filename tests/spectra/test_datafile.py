"""The vibration result's data file gives back the payload the JSON holds
(`engines/vibration.md` § 6.10.5: "Read back, the NetCDF gives the payload
the JSON holds, and that round trip is the file's test").

The one writer writes both files from one payload; what this catches is a
variable, attribute or encoding of the ``.nc`` that drops or bends a value
-- a null read back as a number, a flag as an int, a block's key lost --
which reading the writer's code does not show, because the loss is in the
pairing of the writer's table with the reader's.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from molbuilder.sidecars.spectra import dump_spectra_json
from molbuilder.spectra.datafile import read_spectra_data
from tests.spectra._helpers import _make_results


def _with_everything():
    """A complete result: a geometry, an imaginary mode, a probed mode."""
    r = _make_results()
    r.equilibrium_elements = ["O", "H"]
    r.equilibrium_positions_ang = np.array([[0.0, 0.0, 0.0],
                                            [0.96, 0.0, 0.0]])
    r.equilibrium_masses_amu = np.array([15.999, 1.008])
    r.modes[0].frequency_cm1 = -42.0
    r.modes[0].has_imag = True
    r.relaxation = {"enabled": True, "already_relaxed": False,
                    "n_steps": 3, "max_force_eh_bohr": 1.2e-4,
                    "max_force_all_atoms_eh_bohr": None, "converged": True,
                    "warning": "", "record": {"n_steps": 3, "note": "x"}}
    r.thermo = {"regime": "vibrational-only", "temperature_K": 298.15,
                "pressure_atm": None, "zpe_eh": 0.01, "h_eh": 0.011,
                "s_eh_k": 1e-5, "g_eh": 0.008, "n_modes": 3,
                "n_imag_excluded": 1, "n_rigid_removed": 0, "note": "n",
                "grid": {"temperatures_K": [100.0, 298.15],
                         "zpe_eh": [0.01, 0.01], "u_vib_eh": [0.0, 0.001],
                         "h_eh": [0.01, 0.011], "s_eh_k": [0.0, 1e-5],
                         "g_eh": [0.01, 0.008]}}
    r.removed_motions = {"count": 1, "patterns": [[[1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0]]]}
    r.__post_init__()
    return r


@pytest.mark.parametrize("make", [_make_results, _with_everything],
                         ids=["an in-progress result", "a complete one"])
def test_the_data_file_gives_back_the_payload_the_json_holds(tmp_path, make):
    """MUTATION THIS MUST FAIL AGAINST: a key of `_ROOT`, `_RELAXATION` or
    `_THERMO` removed, or the reader's null rule for the zero-point
    displacement."""
    p = tmp_path / "job.spectra.json"
    dump_spectra_json(make(), p)
    nc = tmp_path / "job.spectra.nc"
    assert nc.is_file(), "the data file is written beside the JSON"
    payload = json.loads(p.read_text(encoding="utf-8"))
    back = read_spectra_data(nc)
    assert back == payload


def test_the_data_file_names_the_json_it_was_written_with(tmp_path):
    p = tmp_path / "job.spectra.json"
    dump_spectra_json(_make_results(), p)
    import netCDF4
    with netCDF4.Dataset(str(tmp_path / "job.spectra.nc")) as ds:
        said = ds.getncattr("json_sha256")
    assert said == hashlib.sha256(p.read_bytes()).hexdigest()


def test_every_axis_is_a_coordinate_a_reader_selects_by(tmp_path):
    """`model/datafiles.md` § 2: every axis a coordinate of its own name, so
    a reader selects by value -- a temperature of the thermochemistry's grid,
    a mode, an atom.  THE SILENT FAILURE (the 2026-10-23 review): the
    file's axes carried their values under other names (`index_1based`,
    `temperatures_K`), so `.sel(temperature=298.15)` raised and xarray
    indexed no axis of any group.

    MUTATION THIS MUST FAIL AGAINST: an axis's values named other than the
    axis."""
    import xarray as xr
    p = tmp_path / "job.spectra.json"
    dump_spectra_json(_with_everything(), p)
    nc = tmp_path / "job.spectra.nc"
    for group in ("structure", "equilibrium", "modes",
                  "modes/electronic_structure", "thermo/grid"):
        ds = xr.load_dataset(nc, group=group)
        assert set(ds.sizes) <= set(ds.indexes), (group, dict(ds.sizes),
                                                  list(ds.indexes))
    grid = xr.load_dataset(nc, group="thermo/grid")
    assert float(grid.g_eh.sel(temperature=298.15)) == pytest.approx(0.008)
    modes = xr.load_dataset(nc, group="modes")
    assert float(modes.frequency_cm1.sel(mode=1)) == pytest.approx(-42.0)
