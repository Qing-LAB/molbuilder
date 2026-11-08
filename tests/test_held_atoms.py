"""The atoms a run holds, stated by its own output (`docs/model/parse.md`
§ 5.3), case by case.

THE CASES ARE DATA -- ``tests/data/held_atoms.toml`` -- and each runs down
the road a person runs (`support.road.run_road_case`): `molbuilder describe` of a
held H2, `prep`.  The files are the ones our code writes; a test lays none
(`process/testing.md` § 6).

PREVENTS: a PySCF run whose held atoms nothing states -- its reader looked
for a sidecar beside the output by names a run of ours never gives one, so
the Results tab's arrows and the relaxation record counted the held atom too
(plan B12, W56 4c).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "held_atoms.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_a_run_states_the_atoms_it_holds(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)


def test_a_data_file_of_another_schema_is_not_read_as_an_optimizations(tmp_path):
    """`model/datafiles.md` § 8: a data file's reader refuses another schema by
    name.  The silent failure: another kind's file, or another version's, read
    with this one's meanings.  The file is the one writer's own."""
    from molbuilder.ncdata import NC_LOCK, as_bytes, in_memory, state
    from molbuilder.postprocess import read_optimization_data
    from molbuilder.optimization.datafile import DataFileError
    other = tmp_path / "other.nc"
    with NC_LOCK:
        ds = in_memory(other.name)
        state(ds, title="a file of another kind", schema="molbuilder/other@1",
              molbuilder_version="0", source="this test")
        other.write_bytes(as_bytes(ds))
    with pytest.raises(DataFileError, match="molbuilder/other@1"):
        read_optimization_data(other)


def test_an_export_before_any_run_has_finished_a_step_is_refused(tmp_path,
                                                                 monkeypatch):
    """`engines/optimization-postprocessing.md` § 6: nothing to export is said,
    and nothing written.  The silent failure: an empty data file a person
    takes for a result."""
    from support.road import described, molbuilder, prep_groups, road_tree
    bundle = described(road_tree(tmp_path, monkeypatch, engine="pyscf"),
                       engine="pyscf")
    prep_groups(bundle, "1")
    to = bundle.parent / "too_soon"
    refused = molbuilder("jobset", "export", "--format", "netcdf", "--to", to,
                         "--bundle", bundle)
    assert refused.exit_code != 0, refused.output
    assert "holds no step a run has finished" in refused.output
    assert not to.exists()


def test_the_force_figures_are_over_every_atom_and_the_atoms_moved():
    """The two figures of one step: the largest component over every atom, and
    over the atoms not held; every atom held has no second figure, and a step
    whose rows are not all there has neither (the real output cut mid-block is
    `test_siesta_relax_run_e2e.py`'s)."""
    from molbuilder.frame import Frame
    from molbuilder.parse.contract import force_figures
    from molbuilder.structure import Structure

    def step(forces):
        return Frame(structure=Structure(elements=["H", "O", "F"],
                                         positions=np.zeros((3, 3))),
                     step_index=0, forces=np.array(forces))
    whole = step([[0.0, 0.0, 9.0], [0.5, -0.2, 0.0], [0.0, 0.1, 0.0]])
    torn = step([[0.0, 0.0, 9.0], [0.001, 0.0, 0.0]])
    assert force_figures(whole, [0]) == (9.0, 0.5)
    assert force_figures(whole, [0, 1, 2]) == (9.0, None)
    assert force_figures(torn, [0]) == (None, None)
