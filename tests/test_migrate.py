"""`jobset migrate` on a record an older molbuilder wrote (`plan.md` W57
decision 7: such a file is refused naming the command, which rewrites it,
every value kept, each change said).

API-level, on a MEASURED fixture (`process/testing.md` § 6): the one
description of ours written before W57 --
`tests/fixtures/siesta_flat_h2/task.json.pre-w57`, the flat H2 run's own, as
its `init` wrote it on 2026-10-04 (an optimization's stated no
`calculation`).  The verb refuses that folder for the template the fixture
leaves out, so its records step is driven directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.jobset.migrate import _migrate_records
from molbuilder.task import read_task

OLDER = Path(__file__).parent / "fixtures" / "siesta_flat_h2" \
    / "task.json.pre-w57"


def test_an_older_description_is_refused_then_rewritten_every_value_kept(
        tmp_path):
    calc = tmp_path / "H2flat"
    calc.mkdir()
    old = OLDER.read_text(encoding="utf-8")
    (calc / "task.json").write_text(old, encoding="utf-8")

    with pytest.raises(ValueError) as refused:
        read_task(calc / "task.json")
    said = str(refused.value)
    assert "molbuilder jobset migrate --bundle" in said and calc.name in said, \
        said

    said, writes, task = _migrate_records(calc)
    assert task.calculation == "optimization"
    assert len(said) == 1 and "calculation" in said[0], said
    [(path, text)] = writes
    assert path == calc / "task.json"
    new = json.loads(text)
    assert new.pop("calculation") == "optimization"
    assert new == json.loads(old), "a value the older file stated changed"
