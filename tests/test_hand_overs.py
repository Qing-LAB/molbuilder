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


def test_a_force_constant_stage_builds_on_the_measured_relaxation():
    """A vibration's `freq` builds on its `relax` run -- the newest, which
    ended on its own with exit code 0 -- and is offered every run of `relax`
    and no start from the structure (`engines/vibration.md` § 5.2a's table,
    W38 F9).

    API-LEVEL, ON A MEASURED FIXTURE: the default is taken only when the
    relaxation wrote a geometry, which the suite's stand-in engine does not
    -- so it is asked of a real SIESTA relaxation, where it was measured
    (`tests/fixtures/siesta_relax`, read in place).  The road's rows hold
    the refusals."""
    from molbuilder.jobset.continuation import (continuation_answer,
                                                continue_from_choices)
    from molbuilder.task import read_task
    base = Path(__file__).parent / "fixtures" / "siesta_relax"
    task = read_task(base / "task.json")

    got, refused = continuation_answer(base, task, "freq")
    assert refused is None, refused
    assert (got.stage, got.source, got.by_default, got.linked) == (
        "relax", "01_relax/run-0", True, True), got
    assert "(the relaxation it builds on; concluded rc=0" in got.line()

    named, refused = continuation_answer(base, task, "freq",
                                         from_attempt="01_relax/run-0")
    assert refused is None and not named.by_default and named.linked, named

    offered = continue_from_choices(base, task, "freq")
    assert offered["from_stage"] == "relax" and offered["cold"] is False
    assert [r["source"] for r in offered["runs"]] == ["01_relax/run-0"]
