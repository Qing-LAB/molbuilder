"""``job-request.json`` -- how and where a calculation runs, and its one reader.

The contract is ``docs/execution/job-request.md`` § 3.  The request read is
one carrying every block § 3's worked request carries -- the scheduler's
ask, the run card, a stage's own card, a benchmark's grid, what the run
tells -- written beside the description `molbuilder describe` made, each
block as a person states it (`support.road.described`).

WHAT THIS FILE OWNS: the request's own shape, and its stages read against
the description beside it -- every stage of the description has its entry,
no other, and each entry's group keeps the ladder's order.  Its
`allocation` and `notify` blocks are `test_job_request_allocation.py`'s and
`test_job_request_notify.py`'s.
"""
from __future__ import annotations

import pytest

from molbuilder import job_request


@pytest.fixture
def calculation(tmp_path, monkeypatch):
    """The road's H2 on SIESTA's shipped ladder, `coarse` then `medium`, as
    `molbuilder describe` writes it -- its job request stating every block
    § 3's worked request states, `medium` running on a card of its own."""
    from support.road import described, read_request, road_tree, write_request
    calc = described(
        road_tree(tmp_path, monkeypatch),
        allocation={"domain": "htc", "time": "0-04:00:00", "mem": "128G"},
        execution={"mpi_np": 8, "omp_threads": 2,
                   "diag_algorithm": "ELPA-2STAGE"},
        bench={"mpi_np": [4, 8, 16], "omp_threads": [1, 2]},
        notify={"on_scf_converged": True, "every_hours": 6,
                "channels": ["slack"], "report": ["*"]})
    request = read_request(calc)
    request["stages"]["medium"]["execution"] = {"mpi_np": 16,
                                                "time": "1-00:00:00"}
    write_request(calc, request)
    return calc


@pytest.fixture
def example(calculation) -> dict:
    """The request, as the calculation's `job-request.json` holds it."""
    from support.road import read_request
    return read_request(calculation)


@pytest.fixture
def description(calculation):
    """The description the request is read beside."""
    from molbuilder.task import of as task_of
    return task_of(calculation)


def test_round_trip_is_byte_identical(example, description, tmp_path):
    """read -> write -> read, and the bytes settle: a request nobody
    edited does not change on Save."""
    first = job_request.write(
        tmp_path, job_request.from_dict(example, description)).read_bytes()
    again = job_request.write(
        tmp_path, job_request.of(tmp_path, description)).read_bytes()
    assert first == again
    assert job_request.of(tmp_path, description) \
        == job_request.from_dict(example, description)


@pytest.mark.parametrize("groups,said", [
    ((1, None), "no 'group'"),
    ((2, 2), "the first stage is group 1"),
    ((1, 3), "repeats the number above it or adds one"),
    ((1, 0), "whole number from 1"),
    ((1, True), "whole number from 1"),
], ids=["absent", "first-not-1", "skips-one", "zero", "a-switch"])
def test_a_group_outside_the_rule_is_refused_naming_it(example, description,
                                                       groups, said):
    """§ 3.3: every stage states its group -- the first 1, each later one
    the number above it or one more.  A skipped or unordered number would
    prepare and launch the jobs out of the ladder's order."""
    for name, g in zip(("coarse", "medium"), groups):
        if g is None:
            del example["stages"][name]["group"]
        else:
            example["stages"][name]["group"] = g
    with pytest.raises(ValueError) as e:
        job_request.from_dict(example, description)
    assert said in str(e.value)


@pytest.mark.parametrize("edit,said", [
    (lambda st: st.pop("medium"), "no entry for 'medium'"),
    (lambda st: st.update(polish={"group": 2}),
     "'polish' is no stage of task.json"),
    (lambda st: st.update(COARSE={"group": 1}), "are one stage"),
], ids=["a-stage-without-its-entry", "an-entry-for-no-stage",
        "two-entries-one-name"])
def test_the_stages_are_the_descriptions_and_no_other(example, description,
                                                      edit, said):
    """§ 3: ``stages`` holds every stage the description holds, by its
    name, and no other -- each difference refused naming both files.  One
    Save writes both, so a difference is an edit of one without the other,
    and reading past it would run a stage in no job, or a job for a stage
    nobody described."""
    edit(example["stages"])
    with pytest.raises(ValueError) as e:
        job_request.from_dict(example, description)
    assert said in str(e.value) and "job-request.json" in str(e.value)
