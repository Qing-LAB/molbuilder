"""Every launch value is stated (`docs/execution/architecture.md` § 5.2), case
by case.

THE CASES ARE DATA -- ``tests/data/launch_values.toml`` -- and each runs down
the road a person runs, through the one runner every contract table shares
(`support.road.run_road_case`): stated, or refused at prep naming where to
state it; what the `.sbatch` header and the run script carry; what the run
script does here.

PREVENTS: a value nobody stated reaching a job -- the target's width as a
rank count, one thread, a rank per GPU, the queue's ceiling as a wall, a
config default as a memory, the menu's first row as the queue.  Each was a
fill somewhere in the framework until 2026-10-02 (user: "explicit job config
is the only way allowed").
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "launch_values.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_every_launch_value_is_stated(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
