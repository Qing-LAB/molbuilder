"""The GPU contract (`docs/execution/gpu.md` § 1), case by case.

THE CASES ARE DATA -- ``tests/data/gpu_contract.toml`` -- and each runs down
the road a person runs (`support.road.run_road_case`, the one runner every
contract table shares): `jobset init`, the description and the target's
record written as the case says, `prep`, `launch --dry-run`, and on a
machine with no queue the run script's own dry run.  Three layers (user,
2026-10-01: "(1) query, allow or deny, (2) correctly produce slurm
command/header, (3) correctly execute locally if that's the target"), and a
case checks the ones it names; the runner's header says which key is which.

PREVENTS: a GPU rule held by tests of internal steps, each building its own
inputs and restating the same assertion, so that changing the rule meant
patching every restatement -- thirty tests in fifteen files for one rule on
2026-10-01.  A rule changes; its rows change.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "gpu_contract.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_the_gpu_contract(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
