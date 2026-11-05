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
from types import SimpleNamespace

import numpy as np
import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "held_atoms.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_a_run_states_the_atoms_it_holds(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)


def test_a_force_block_still_being_written_states_no_figure():
    """`contract.force_figures`: the largest force of a step is over its atoms,
    the held ones left out of the second figure, and a block with rows for only
    some of them answers neither, so a relaxation's record stands on the step
    before it.  The silent failure: the largest of the rows written so far read
    as the step's, `converged` on a real run whose figure was 0.0185 eV/Å against
    a tolerance of 0.01."""
    from molbuilder.frame import Frame
    from molbuilder.parse.contract import force_figures, relaxation_of_output
    from molbuilder.structure import Structure
    from molbuilder.trajectory_log.keys import MAX_FORCE_TOL

    def step(index, forces):
        return Frame(structure=Structure(elements=["H", "O", "F"],
                                         positions=np.zeros((3, 3))),
                     step_index=index, forces=np.array(forces))
    whole = step(0, [[0.0, 0.0, 9.0], [0.5, -0.2, 0.0], [0.0, 0.1, 0.0]])
    torn = step(1, [[0.0, 0.0, 9.0], [0.001, 0.0, 0.0]])
    assert force_figures(whole, [0]) == (9.0, 0.5)
    assert force_figures(whole, [0, 1, 2]) == (9.0, None)
    assert force_figures(torn, [0]) == (None, None)
    record = relaxation_of_output("job.out", SimpleNamespace(
        frames=[whole, torn], run_state="running",
        runtime_info={"frozen_atoms": [0],
                      "convergence_targets": {MAX_FORCE_TOL: 0.01}}))
    assert record["max_force_free_ev_ang"] == 0.5
    assert record["converged"] is False
