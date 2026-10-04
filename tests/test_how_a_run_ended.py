"""How a run ended, and the run to build on (`docs/execution/architecture.md`
§ 3.2, `job-system.md` § 5.4), case by case: one door says whether a run
ended on its own and with what exit code; status builds its state on it, and
every hand-over builds by default only on a run that ended on its own with
exit code 0.

THE CASES ARE DATA -- ``tests/data/how_a_run_ended.toml`` -- and each runs
down the road a person runs, through the one runner every contract table
shares (`support.road.run_road_case`).

PREVENTS, each measured before 2026-10-03 (plan W38 F3, W55 B8): status
saying *finished* for a run every hand-over refused -- an output that ended
with no conclusion, SIESTA's own end mark where molbuilder's wrapper ran --
and *finished* beside an exit code of 1; a frequency stage measured at a
relaxation that failed.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "how_a_run_ended.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_how_a_run_ended(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
