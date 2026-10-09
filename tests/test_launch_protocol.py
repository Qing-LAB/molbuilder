"""The launch protocol (`docs/execution/job-system.md` § 6.0), case by case.

THE CASES ARE DATA -- ``tests/data/launch_protocol.toml`` -- and each runs
down the road a person runs (`support.road.run_road_case`): `jobset init`,
`prep` and `launch`, each row stopping before an engine would run -- a
refusal, or a dry run.

PREVENTS: a dry run that wrote -- a ledger line, or an attempt opened before
the person said yes; a refusal raised before the plan that left no line; a
run here written down only once it had ended (W55 D14).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "launch_protocol.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_a_launch_is_planned_shown_asked_sent_and_recorded(case, tmp_path,
                                                           monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
