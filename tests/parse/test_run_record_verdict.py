"""The run record's ``verdict.ended`` states how the run's output ended, and
only once the output states an ending (`model/parse.md` § 5d.6; § 2b, P-S1;
§ 5d.1a).

**Why this is needed.**  The Results tab's Run panel shows *Ended: <how>*
from the record's ``verdict.ended`` (`web/results.md` § 3a).  An output with
no ending yet is ``running`` -- not finished (§ 2b) -- and a record field is
present only when a file states it (§ 5d.1a), so a live run's record states
no ``ended``.  THE SILENT FAILURE: a record that copied the output's
``running`` there, and a panel saying *Ended: running* of a run still going.

**Category: API** -- the record's ``ending`` row (`parse.dirs.record.
CONTRIBUTORS`, § 5d.1b) read with what the run's state door answers: its one
scan, one `RunEnding` per output (`RunStatus.endings`), the output's
``run_state`` in P-S1's vocabulary.  Each case is a row of the table below,
its expectation from the contract.  How an output's text becomes its ending
is the engine's reading, tested end to end (`process/testing.md` § 6).

MUTATION THIS MUST FAIL AGAINST: ``verdict.ended`` set from every output's
``run_state``, a running one's among them.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.dirs.job import RunStatus
from molbuilder.parse.dirs.record import CONTRIBUTORS, RunFiles, _declared
from molbuilder.parse.engines._run_ending import RunEnding
from molbuilder.runfiles import RunNames

#: ``(engine, its stdout role and the field of `RunFiles` that holds it --
#: the one output slot, every engine's -- the output's run_state, the run's
#: state, verdict.ended -- None: not stated)``.
CASES = [
    ("siesta", ".out", "stdout", "running", "running", None),
    ("siesta", ".out", "stdout", "ended", "finished", "ended"),
    ("siesta", ".out", "stdout", "stopped", "failed", "stopped"),
    ("siesta", ".out", "stdout", "out_of_memory", "failed", "out_of_memory"),
    ("pyscf", ".pyscf.log", "stdout", "running", "running", None),
    ("pyscf", ".pyscf.log", "stdout", "ended", "finished", "ended"),
]


@pytest.mark.parametrize(
    "engine, role, field, run_state, state, ended", CASES,
    ids=[f"{c[0]}-{c[3]}" for c in CASES])
def test_the_verdict_says_ended_once_the_output_states_an_ending(
        engine, role, field, run_state, state, ended, tmp_path):
    names = RunNames("H2", "01_coarse")
    output = tmp_path / names.name(role)
    status = RunStatus(state=state, detail="", active_source=output.name,
                       endings={output.name: RunEnding(run_state)})
    files = RunFiles(directory=tmp_path, engine=engine, launched_in=tmp_path,
                     status=status, names=names, **{field: output})
    row = next(r for r in CONTRIBUTORS if r.name == "ending")
    # AS THE RECORD CARRIES IT: the row's reading cut to the fields it
    # declares, which is all the record and the Run panel ever see.
    verdict = _declared(row.read(files), row.fields)["verdict"]
    assert verdict["state"] == state, verdict
    assert verdict.get("ended") == ended, verdict
    assert ("ended" in verdict) == (ended is not None), verdict
