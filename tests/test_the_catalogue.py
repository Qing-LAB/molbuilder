"""What molbuilder writes, by name (`docs/execution/job-contracts.md` § 2.2,
`docs/execution/project-layout.md` § 5), case by case.

THE CASES ARE DATA -- ``tests/data/the_catalogue.toml`` -- and each runs down
the road a person runs (`support.road.run_road_case`): `jobset init`, `prep`
and `launch`, our wrapper running the suite's stand-in engine.

PREVENTS: the Task setup card naming a file no run writes -- the flat
spelling of an attempt's launch record for a hierarchical stage, PySCF's log
for a SIESTA run, a parse log only a switch turns on -- seven of seventeen
names for a hierarchical H2, measured 2026-10-04 (plan D22).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "the_catalogue.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_what_molbuilder_writes(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
