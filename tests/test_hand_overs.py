"""What a stage builds on: the kinds whose rungs take their inputs from
their own kind -- never from a run a person names -- refuse ``--from`` and
``--cold`` before anything is written (`docs/execution/job-system.md`
§ 5.4).

PREVENTS: a transport rung's gather and a named run's carry written into one
attempt (taken until 2026-10-05); a vibration's one stage started from a
`freq` run's `.XV`, a displaced geometry, or "cold" at a first launch that
has nothing to start cold from (`engines/vibration.md` § 5.11 § 6).
"""
from __future__ import annotations

import pytest


def _transport():
    from molbuilder.task import Stage, Task, derive_run
    rungs = ("seed", "electrode_L", "electrode_R", "device", "transmission")
    return Task(engine="siesta", shape="hierarchical",
                run=derive_run("T", "J/x", stage_names=rungs),
                structure=None, calculation="transport",
                slots={"junction": "J/x"}, bias=(0.0,), varies=(),
                stages=tuple(Stage(name=n, overrides={}) for n in rungs))


def _vibration():
    from molbuilder.task import Stage, StructureRef, Task, derive_run
    return Task(engine="siesta", shape="hierarchical",
                run=derive_run("V", stage_names=("freq",)),
                structure=StructureRef(source="v.xyz"),
                calculation="vibration", varies=(),
                stages=(Stage(name="freq", overrides={}),))


@pytest.mark.parametrize("task, stage, words", [
    (_transport, "device", "takes its inputs from the rungs upstream"),
    (_vibration, "freq", "builds on the calculation's structure"),
], ids=["a transport rung", "a SIESTA vibration's one stage"])
def test_a_linked_stage_takes_no_from_or_cold(tmp_path, task, stage, words):
    """A linked kind's inputs are its kind's -- a transport rung's gathered
    from the rungs upstream, a vibration's one stage built on the
    calculation's structure, its phases joined inside its run -- so
    ``--from`` and ``--cold`` name nothing it can take: refused by the one
    hand-over door, naming why, before anything is written.

    API-LEVEL: the suite's road describes no transport calculation (plan
    § 5x B8 T2), and this is the hand-over door's refusal, asked as prep
    and launch ask it, before the calculation's own files are read."""
    from molbuilder.calculation import Calculation
    from molbuilder.jobset.continuation import continuation_answer
    calc = Calculation(base=tmp_path, task=task(), template=None, jobset=None)
    for asked in ({"from_attempt": "01_seed/run-0"}, {"cold": True}):
        got, refused = continuation_answer(calc, stage, **asked)
        assert got is None and refused and words in refused, refused
