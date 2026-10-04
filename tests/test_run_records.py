"""A run's own records -- whether it was launched, how it ended, and the run
to build on (`docs/execution/architecture.md` § 3.2, `job-system.md` § 5.4),
case by case: one door says whether a run ended on its own and with what exit
code, and status builds its state on it; every hand-over builds by default
only on a run that ended on its own with exit code 0; a launch record that
does not read is an error naming the file.

THE CASES ARE DATA -- ``tests/data/run_records.toml`` -- and each runs
down the road a person runs, through the one runner every contract table
shares (`support.road.run_road_case`).

PREVENTS, each measured before 2026-10-03 (plan W38 F3, W55 B8): status
saying *finished* for a run every hand-over refused -- an output that ended
with no conclusion, SIESTA's own end mark where molbuilder's wrapper ran --
and *finished* beside an exit code of 1; a frequency stage measured at a
relaxation that failed; a launch record that did not read taken as
*launched, the details lost* while launch's gates asked only whether the file
was there.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "run_records.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_a_runs_own_records(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
