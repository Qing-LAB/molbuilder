"""The run a folder speaks for (`docs/model/parse.md` § 5.1,
`docs/execution/architecture.md` § 3.2), case by case.

THE CASES ARE DATA -- ``tests/data/the_speaking_run.toml`` -- and each runs
down the road a person runs (`support.road.run_road_case`): `jobset init`,
`prep` and `launch`, our wrapper running the suite's stand-in engine, which
ends as the row says.  The runs' files are the ones our wrapper writes; a test
lays none (`process/testing.md` § 6).

PREVENTS: a folder speaking for whichever file was written last -- a copied
or restored folder reorders file times, never runs (plan B11, W56 3b.2).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "the_speaking_run.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_the_run_a_folder_speaks_for(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
