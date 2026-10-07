"""The record `jobset probe` writes (`docs/configuration.md` § 5, M-1 and
M-6), case by case.

THE CASES ARE DATA -- ``tests/data/machine_record.toml`` -- and each runs the
probe as a person runs it, through the one runner every contract table shares
(`support.road.run_road_case`): what it declares (`--set`, `--scheduler`,
`--name`), what it asks over an existing record and what each answer keeps,
and what it says about a file at the record's path that does not read.

PREVENTS: a weaker probe erasing a declared fact (a login node that sees no
GPUs probing `null` over a recorded `4`), a scripted probe changing a record
nobody agreed to change, and an unreadable record replaced as if it were
absent.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "machine_record.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_what_the_probe_records(case, tmp_path, monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
