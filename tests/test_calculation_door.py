"""A described calculation is read through one door (`architecture.md`
§ 3.2, rule A16): `calculation.open` answers the folder states a verb
meets, and each record's own door answers ``None`` for a record that is
not there -- never a probe of a path at a call site.

Silent failure otherwise: nine call sites once wrote out "read the
description, find and read the template, load the plan if there", each
wrapping the refusal its own way, and migrate's lead-names step re-read a
plan the records step had already decided to rewrite (2026-10-19).  The
calculation is described through the designed road (`jobset init`,
`tests/support/road.py`), never hand-built."""
from __future__ import annotations

import pytest

from molbuilder import calculation as C
from molbuilder import task as T
from molbuilder import template as TP
from molbuilder.jobset.model import JobSet
from support.road import describe_calculation


def test_a_described_calculation_opens_once_with_its_three_records(
        tmp_path, monkeypatch):
    bundle = describe_calculation(tmp_path, monkeypatch)
    calc = C.open(bundle)
    assert calc.base == bundle.resolve()
    assert calc.label == calc.task.label
    assert calc.template.path == TP.template_path(bundle, calc.label)
    assert calc.template_text == calc.template.path.read_text(encoding="utf-8")
    assert calc.jobset is None, "no prep has written a plan yet"


@pytest.mark.parametrize("state, words", [
    # (what the folder holds, what the refusal says)
    ("empty", "is not a described calculation"),
    ("no template", "no template beside it"),
])
def test_a_folder_that_is_not_a_calculation_is_refused_as_one_thing(
        tmp_path, monkeypatch, state, words):
    bundle = describe_calculation(tmp_path, monkeypatch)
    if state == "empty":
        folder = tmp_path / "bare"
        folder.mkdir()
    else:
        folder = bundle
        TP.template_path(bundle, T.of(bundle).label).unlink()
    with pytest.raises(C.NotACalculation) as err:
        C.open(folder)
    assert words in str(err.value), str(err.value)
    assert "jobset init" in str(err.value)


def test_each_records_door_answers_none_for_a_record_that_is_not_there(
        tmp_path):
    folder = tmp_path / "bare"
    folder.mkdir()
    assert T.of(folder) is None
    assert not T.described(folder)
    assert JobSet.of(folder) is None
    assert TP.of(folder, "anything") is None
