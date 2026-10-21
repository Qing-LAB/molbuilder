"""A described calculation is read through one door (`architecture.md`
§ 3.2, rule A16): `calculation.open` answers the folder states a verb
meets, and each record's own door answers ``None`` for a record that is
not there -- never a probe of a path at a call site.

Silent failure otherwise: nine call sites once wrote out "read the
description, find and read the template, load the plan if there", each
wrapping the refusal its own way, and migrate's lead-names step re-read a
plan the records step had already decided to rewrite (2026-10-19).  The
calculation is described through the designed road (`molbuilder describe`,
`tests/support/road.py`), never hand-built."""
from __future__ import annotations

import pytest

from molbuilder import calcdirs
from molbuilder import calculation as C
from molbuilder import task as T
from molbuilder import template_file as TP
from molbuilder.jobset.model import JobSet
from support.road import describe_calculation


def test_a_described_calculation_opens_once_with_its_records(
        tmp_path, monkeypatch):
    from molbuilder import job_request as JR
    bundle = describe_calculation(tmp_path, monkeypatch)
    calc = C.open(bundle)
    assert calc.base == bundle.resolve()
    assert calc.label == calc.task.label
    assert calc.request == JR.of(bundle, calc.task)
    assert calc.template.path == TP.template_path(bundle, calc.label)
    assert calc.template_text == calc.template.path.read_text(encoding="utf-8")
    assert calc.jobset is None, "no prep has written a plan yet"


@pytest.mark.parametrize("state, words, way", [
    # (what the folder holds, what the refusal says, the way out it names)
    ("empty", "is not a described calculation", "molbuilder describe"),
    ("no template", "no template beside it", "molbuilder describe"),
    # `execution/job-request.md` § 7: the two records are written together
    ("no job request", "no job-request.json beside it", "Task setup's Save"),
])
def test_a_folder_that_is_not_a_calculation_is_refused_as_one_thing(
        tmp_path, monkeypatch, state, words, way):
    from molbuilder import job_request as JR
    bundle = describe_calculation(tmp_path, monkeypatch)
    folder = bundle
    if state == "empty":
        folder = tmp_path / "bare"
        folder.mkdir()
    elif state == "no template":
        TP.template_path(bundle, T.of(bundle).label).unlink()
    else:
        JR.path_of(bundle).unlink()
    with pytest.raises(C.NotACalculation) as err:
        C.open(folder)
    assert words in str(err.value), str(err.value)
    assert way in str(err.value), str(err.value)


def test_each_records_door_answers_none_for_a_record_that_is_not_there(
        tmp_path):
    folder = tmp_path / "bare"
    folder.mkdir()
    assert T.of(folder) is None
    assert not calcdirs.described(folder)
    assert JobSet.of(folder) is None
    assert TP.of(folder, "anything") is None
