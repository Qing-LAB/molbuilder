"""What a stage that has run still takes, and what it refuses
(`execution/job-system.md` § 5.0, `execution/job-request.md` § 5,
`web/task-setup.md` § 5, § 8), on calculations a real SIESTA ran
(`support/real_runs.py`; `process/testing.md` § 0, tier 3, and § 6).

What a stage has run is what the launch left -- its record, its runs, the state
its first launch saved -- so a row stands on a launch that happened.  Each
row is a copy of a shared run of its own (`copy_of`), and goes down the road
through the runner every contract table shares; what a row can say is a row of
``tests/data/a_stage_that_has_run.toml``.

PREVENTS: a launched stage re-rendered to compute something else under its runs,
its way back naming a state that is not the one its first launch saved, a
group's members changed once they ran as one job, a stage that ran taken out of
the description, and a later stage taking over a relaxation its source never
finished.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from _road import conda_hook, env_available
from support.road import read_request, run_road_case, write_request

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

TABLE = tomllib.loads((Path(__file__).parent / "data"
                       / "a_stage_that_has_run.toml").read_text())


def _rows(on: str) -> dict:
    """The table's rows on the run named ``on``, for `parametrize`."""
    rows = [c for c in TABLE["case"] if c["on"] == on]
    return dict(argvalues=rows, ids=[c["name"] for c in rows])


@pytest.fixture(scope="module")
def short_of_its_relaxation(real_co2_vibration, tmp_path_factory):
    """A vibration with a run whose relaxation is not done, and one whose is:
    a copy of the shared run, its run card given a wall already passed
    (`execution.time`, `job-request.md`), prepared again for that job request,
    and `freq` launched again cold -- run-1, its `relax` call told what is left
    of the wall, so SIESTA stops at its first check (`running-a-job.md` § 3.6).
    Said by the one status door: run-1 is the newest and its relaxation is not
    done."""
    from molbuilder.siesta.vibration_phases import RELAX
    from support.real_runs import (_road_at, _taken, _typed, copy_of,
                                   stage_status)

    def walled(c):
        request = read_request(c.bundle)
        request["execution"] = {**request.get("execution", {}),
                                "time": "00:00:20"}
        write_request(c.bundle, request)
        with _road_at(c.tree.parent, tmp_path_factory):
            _taken(c, "prep freq for the wall", "prep", "task", "--group",
                   "1", "--target", "this")
            _typed(c, "launch freq cold", "launch", "task", "--stage", "freq",
                   "--mode", "direct", "--foreground", "--yes", "--cold")
        status = stage_status(c.bundle, "freq")
        assert status.attempt == c.attempt("freq", 1).name, status
        assert not any(p["done"] for p in status.points
                       if p["phase"] == RELAX), status.points
    return copy_of(real_co2_vibration, "short-of-its-relaxation", walled)


def _row(case, run, tmp_path, monkeypatch) -> None:
    """``case`` on a copy of ``run`` of its own, the one door pointed at the
    tree the run was made in."""
    from conftest import _point_the_one_door_at
    from support.real_runs import copy_of
    _point_the_one_door_at(run.tree.parent, monkeypatch.setenv)
    monkeypatch.chdir(run.tree.parent)
    copy = copy_of(run, re.sub(r"\W+", "-", case["name"])[:60], lambda _c: None)
    run_road_case(TABLE, case, tmp_path, monkeypatch, ran=copy.bundle)


@pytest.mark.parametrize("case", **_rows("layered"))
def test_on_stages_launched_each_alone(case, real_h2_layered, tmp_path,
                                       monkeypatch):
    _row(case, real_h2_layered, tmp_path, monkeypatch)


@pytest.mark.parametrize("case", **_rows("one_group"))
def test_on_a_group_launched_as_one_job(case, real_h2_one_group, tmp_path,
                                        monkeypatch):
    _row(case, real_h2_one_group, tmp_path, monkeypatch)


@pytest.mark.parametrize("case", **_rows("short_of_its_relaxation"))
def test_on_a_run_short_of_its_relaxation(case, short_of_its_relaxation,
                                          tmp_path, monkeypatch):
    _row(case, short_of_its_relaxation, tmp_path, monkeypatch)
