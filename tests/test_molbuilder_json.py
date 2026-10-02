"""What `molbuilder.json` may hold (`docs/configuration.md` § 4), case by case.

THE CASES ARE DATA -- ``tests/data/molbuilder_json.toml`` -- and each runs
down the road a person runs, through the one runner every contract table
shares (`support.road.run_road_case`): a config file written as the case
says, then `prep` and `launch --dry-run`, and what they accept or refuse.

PREVENTS: a key that is read by nothing and looks effective, and a retired
one met as "unknown" with no word of what replaced it -- the `scheduler`,
`script_generation` and `execution` sections left this file on 2026-10-02
(user: "make sure molbuilder.json are cleaned up with the obsolete and
conflicting parameters removed").
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "molbuilder_json.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_what_molbuilder_json_may_hold(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
