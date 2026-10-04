"""What a stage builds on, and whether a viewer follows a run
(`docs/execution/job-system.md` § 5.4, `docs/execution/architecture.md`
§ 3.2, `docs/web/results.md` § 4.1), case by case.

THE CASES ARE DATA -- ``tests/data/hand_overs.toml`` -- and each runs down
the road a person runs (`support.road.run_road_case`): `jobset init`, `prep`
and `launch`, our wrapper running the suite's stand-in engine, which ends as
the row says.  The runs' records are the ones our wrapper writes as each
concludes; a test lays none (`process/testing.md` § 6).

PREVENTS: a run that ended with an error built on, a stage continuing from
an older run than the newest, and a live run a viewer stops following --
each measured uncovered once the tests that laid their runs by hand were
retired (2026-10-03).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "hand_overs.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_what_a_stage_builds_on(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
