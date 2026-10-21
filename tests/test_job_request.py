"""``job-request.json`` -- how and where a calculation runs, and its one reader.

The contract is ``docs/execution/job-request.md`` § 3.  The round-trip
fixture is § 3's worked request, copied into
``tests/data/job_request_example.json`` and read beside § 6's worked
description (``tests/data/task_example.json``): the documents and the files
are kept in line by review, and the tests read the files.

WHAT THIS FILE OWNS: the request's own shape, and its stages read against
the description beside it -- every stage of the description has its entry,
no other, and each entry's group keeps the ladder's order.  Its
`allocation` and `notify` blocks are `test_job_request_allocation.py`'s and
`test_job_request_notify.py`'s.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder import job_request
from molbuilder.task import Task

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "tests" / "data" / "job_request_example.json"
DESCRIPTION = Task.from_dict(json.loads(
    (REPO / "tests" / "data" / "task_example.json").read_text(
        encoding="utf-8")))


@pytest.fixture
def example() -> dict:
    """§ 3's worked request, as a person's `job-request.json` holds it."""
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_round_trip_is_byte_identical(example, tmp_path):
    """read -> write -> read, and the bytes settle: a request nobody
    edited does not change on Save."""
    first = job_request.write(
        tmp_path, job_request.from_dict(example, DESCRIPTION)).read_bytes()
    again = job_request.write(
        tmp_path, job_request.of(tmp_path, DESCRIPTION)).read_bytes()
    assert first == again
    assert job_request.of(tmp_path, DESCRIPTION) \
        == job_request.from_dict(example, DESCRIPTION)


@pytest.mark.parametrize("groups,said", [
    ((1, None), "no 'group'"),
    ((2, 2), "the first stage is group 1"),
    ((1, 3), "repeats the number above it or adds one"),
    ((1, 0), "whole number from 1"),
    ((1, True), "whole number from 1"),
], ids=["absent", "first-not-1", "skips-one", "zero", "a-switch"])
def test_a_group_outside_the_rule_is_refused_naming_it(example, groups, said):
    """§ 3.3: every stage states its group -- the first 1, each later one
    the number above it or one more.  A skipped or unordered number would
    prepare and launch the jobs out of the ladder's order."""
    for name, g in zip(("coarse", "tight"), groups):
        if g is None:
            del example["stages"][name]["group"]
        else:
            example["stages"][name]["group"] = g
    with pytest.raises(ValueError) as e:
        job_request.from_dict(example, DESCRIPTION)
    assert said in str(e.value)


@pytest.mark.parametrize("edit,said", [
    (lambda st: st.pop("tight"), "no entry for 'tight'"),
    (lambda st: st.update(polish={"group": 2}),
     "'polish' is no stage of task.json"),
    (lambda st: st.update(COARSE={"group": 1}), "are one stage"),
], ids=["a-stage-without-its-entry", "an-entry-for-no-stage",
        "two-entries-one-name"])
def test_the_stages_are_the_descriptions_and_no_other(example, edit, said):
    """§ 3: ``stages`` holds every stage the description holds, by its
    name, and no other -- each difference refused naming both files.  One
    Save writes both, so a difference is an edit of one without the other,
    and reading past it would run a stage in no job, or a job for a stage
    nobody described."""
    edit(example["stages"])
    with pytest.raises(ValueError) as e:
        job_request.from_dict(example, DESCRIPTION)
    assert said in str(e.value) and "job-request.json" in str(e.value)
