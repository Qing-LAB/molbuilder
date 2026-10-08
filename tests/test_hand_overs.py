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


def test_a_transport_rung_takes_no_from_or_cold(tmp_path):
    """A transport rung's inputs are its kind's -- gathered from the rungs
    upstream -- and a rung prepared is not prepared again, so ``--from`` and
    ``--cold`` name nothing it can take: refused before anything is written
    (`job-system.md` § 5.4).  ``--from`` naming another rung's run was taken
    until 2026-10-05, and its carry and the gather wrote into one attempt.

    API-LEVEL: the suite's road describes no transport calculation -- one
    cites a junction run (`tests/test_transport_prep.py` builds one, below
    the entry) -- and this is the hand-over door's refusal, asked as prep
    asks it at checkpoint 4a, before the calculation's own files are read."""
    from molbuilder.jobset.continuation import continuation_answer
    from molbuilder.task import Stage, Task, derive_run
    rungs = ("seed", "electrode_L", "electrode_R", "device", "transmission")
    task = Task(engine="siesta", shape="hierarchical",
                run=derive_run("T", "J/x", stage_names=rungs),
                structure=None, calculation="transport",
                slots={"junction": "J/x"}, bias=(0.0,), varies=(),
                execution={}, stages=tuple(
                    Stage(name=n, overrides={}) for n in rungs))
    for asked in ({"from_attempt": "01_seed/run-0"}, {"cold": True}):
        got, refused = continuation_answer(tmp_path, task, "device", **asked)
        assert got is None and refused and (
            "takes its inputs from the rungs upstream" in refused), refused
