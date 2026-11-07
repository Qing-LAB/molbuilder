"""The prep protocol (`docs/execution/job-system.md` § 5.0 and *The task*),
case by case: the groups a prep picks, each one job; a group prepared and
never launched prepared again whole; before it writes, prep saves the
folder's state; what a stage takes from another run, taken at its first
launch.

THE CASES ARE DATA -- ``tests/data/prep_protocol.toml`` -- and each runs down
the road a person runs, through the one runner every contract table shares
(`support.road.run_road_case`).  What a stage that has run answers stands on a
calculation that ran: `test_a_stage_that_has_run_e2e.py`.

PREVENTS: a prep that writes before the person could save what was there, a
group whose members are no longer one job launched as it was prepared, and a
first launch taking from a run that has not finished.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "prep_protocol.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_the_prep_protocol(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
