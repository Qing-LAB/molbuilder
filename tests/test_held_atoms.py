"""The atoms a run holds, stated by its own output (`docs/model/parse.md`
§ 5.3), case by case.

THE CASES ARE DATA -- ``tests/data/held_atoms.toml`` -- and each runs down
the road a person runs (`support.road.run_road_case`): `jobset init` of a
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

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "held_atoms.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_a_run_states_the_atoms_it_holds(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
