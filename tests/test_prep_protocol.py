"""The prep protocol (`docs/execution/job-system.md` § 5.0), case by case: a
prepped stage is not prepped again -- a redo is a rollback; before it writes,
prep offers to save the folder's state; a prep refused after it began
writing puts the plan back.

THE CASES ARE DATA -- ``tests/data/prep_protocol.toml`` -- and each runs down
the road a person runs, through the one runner every contract table shares
(`support.road.run_road_case`).

PREVENTS: a prepped stage re-rendered under a run (the "already under way"
question this replaced, 2026-10-02), a prep that writes before the person
could save what was there, and a refused prep leaving its stage counted
prepped -- which, with re-preps refused, would lock the stage.
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
