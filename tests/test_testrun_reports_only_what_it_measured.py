"""The measuring instrument may not report a number it cannot support.

On 2026-09-12 a second pytest pointed at ``.test-progress/none2e.jsonl``
truncated it while a full run was writing it.  What was left was the TAIL of a
complete run -- 4947 test records, no ``start``, no ``collected`` -- and
``testrun.py status`` printed ``done (exit 1) | 4947/None ran | pass 4942``.
That was read as "~4000 tests are not executing", which became the first item
of a migration plan.  The suite had in fact run to completion: 9082 tests.

So the rule these tests hold is not about pytest at all -- it is that every
state this reader can be in either supports a count or says it does not.
``tools/progress_plugin.py``'s half is that a file is truncated only when its
previous run reached ``done``; a new run appends a generation instead of
destroying evidence.
"""
from __future__ import annotations

import json
import os

import pytest

from tools import progress_plugin, testrun


def _write(path, records):
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _run(run_id, n_tests, *, collected=None, done=None, pid=None,
         start_t=100.0):
    """One generation's records."""
    recs = [{"event": "start", "run": run_id, "time": start_t}]
    if pid is not None:
        recs[0]["pid"] = pid
    if collected is not None:
        recs.append({"event": "collected", "run": run_id, "n": collected,
                     "time": start_t + 1})
    for i in range(n_tests):
        recs.append({"event": "test", "run": run_id, "nodeid": f"t.py::t{i}",
                     "outcome": "passed", "duration": 0.0, "reason": "",
                     "time": start_t + 2 + i})
    if done is not None:
        recs.append({"event": "done", "run": run_id, "exitstatus": done,
                     "time": start_t + 2 + n_tests})
    return recs


def test_a_complete_run_is_done_and_carries_its_counts(tmp_path):
    p = tmp_path / "b.jsonl"
    _write(p, _run("r1", 10, collected=10, done=0, pid=os.getpid()))
    s = testrun._summarise("b", str(p))
    assert s["state"] == "done"
    assert (s["ran"], s["collected"], s["passed"]) == (10, 10, 10)


def test_a_file_with_no_start_record_is_UNUSABLE(tmp_path):
    """The 2026-09-12 shape exactly: the tail of a run, its head wiped."""
    p = tmp_path / "b.jsonl"
    tail = [r for r in _run("r1", 4947, collected=9081, done=1)
            if r["event"] not in ("start", "collected")]
    _write(p, tail)
    s = testrun._summarise("b", str(p))
    assert s["state"] == "unusable", s
    assert "truncated" in s["why"]
    assert "ran" not in s, (
        "an unusable file must not offer a count at all; the defect was that "
        f"a number was available to print: {s}")


def test_a_run_that_stopped_early_is_PARTIAL_not_done(tmp_path):
    p = tmp_path / "b.jsonl"
    _write(p, _run("r1", 4947, collected=9081, done=1))
    s = testrun._summarise("b", str(p))
    assert s["state"] == "partial", s
    assert "4947 of 9081" in s["why"]


def test_records_from_two_runs_in_one_generation_are_INTERLEAVED(tmp_path):
    p = tmp_path / "b.jsonl"
    recs = _run("r1", 5, collected=10)
    recs += [{"event": "test", "run": "r2", "nodeid": "other.py::t",
              "outcome": "passed", "duration": 0.0, "reason": "", "time": 200.0}]
    recs += [{"event": "done", "run": "r1", "exitstatus": 0, "time": 201.0}]
    _write(p, recs)
    s = testrun._summarise("b", str(p))
    assert s["state"] == "interleaved", s


def test_only_the_LAST_generation_is_summarised(tmp_path):
    """A new run appends rather than destroying; the reader answers about the
    newest one, and the older records stay on disk as evidence."""
    p = tmp_path / "b.jsonl"
    _write(p, _run("old", 3, collected=3, done=0, start_t=10.0)
              + _run("new", 7, collected=7, done=0, start_t=100.0))
    s = testrun._summarise("b", str(p))
    assert s["state"] == "done"
    assert (s["ran"], s["collected"], s["run"]) == (7, 7, "new")


def test_a_run_whose_process_is_gone_is_ABANDONED_not_running(tmp_path):
    p = tmp_path / "b.jsonl"
    dead = 999_999_999  # no such pid
    _write(p, _run("r1", 12, collected=500, pid=dead))
    s = testrun._summarise("b", str(p))
    assert s["state"] == "abandoned", s
    assert "killed or crashed" in s["why"]


def test_a_live_run_with_no_done_record_is_still_running(tmp_path):
    p = tmp_path / "b.jsonl"
    _write(p, _run("r1", 12, collected=500, pid=os.getpid()))
    assert testrun._summarise("b", str(p))["state"] == "running"


@pytest.mark.parametrize("records,banned", [
    ([r for r in _run("r1", 40, collected=900, done=1)
      if r["event"] not in ("start", "collected")], "pass "),
    (_run("r1", 40, collected=900, done=1), "done ("),
])
def test_status_never_prints_a_result_line_for_a_file_that_has_none(
        tmp_path, monkeypatch, capsys, records, banned):
    """The output itself is the contract: what a person reads must not look
    like a verdict when it is not one."""
    monkeypatch.setattr(testrun, "PROGRESS_DIR", str(tmp_path))
    _write(tmp_path / "none2e.jsonl", records)

    class _Args:
        batch = "none2e"
        fails = False

    rc = testrun.cmd_status(_Args())
    out = capsys.readouterr().out
    assert banned not in out, f"{banned!r} appeared in:\n{out}"
    assert rc == 1, "status must report a non-zero code for an untrusted file"


def test_a_nonzero_exit_no_failure_explains_is_UNEXPLAINED_not_done(tmp_path):
    """The 2026-09-22 shape: every test passed and pytest still exited 1.

    A session-scoped fixture raising AFTER its yield -- which is what all
    three `the_suite_leaves_your_*_alone` canaries in `tests/conftest.py` do
    -- fails the run without producing a failed CALL report. `done (exit 1) |
    pass 9513  FAIL 0` was printed and read as green while the checkout
    canary was firing.  The exit code outranks the counts.
    """
    p = tmp_path / "b.jsonl"
    _write(p, _run("r1", 9527, collected=9527, done=1))
    s = testrun._summarise("b", str(p))
    assert s["state"] == "unexplained", s
    assert "exited 1" in s["why"] and "teardown" in s["why"]


def test_a_teardown_failure_is_a_failure_and_not_a_second_test(tmp_path):
    """It explains the exit code, so the run is `done` -- and it must not
    inflate `ran`, which would make `ran`/`collected` meaningless."""
    p = tmp_path / "b.jsonl"
    recs = _run("r1", 10, collected=10, done=1)
    recs.insert(-1, {"event": "test", "run": "r1",
                     "nodeid": "t.py::t9 [teardown]", "outcome": "failed",
                     "duration": 0.0, "reason": "AssertionError: canary",
                     "time": 200.0})
    _write(p, recs)
    s = testrun._summarise("b", str(p))
    assert s["state"] == "done", s
    assert (s["ran"], s["collected"]) == (10, 10), \
        "the teardown record was counted as an eleventh test"
    assert s["failed"] == 1 and s["failed_ids"][0][0].endswith("[teardown]")


def test_the_plugin_records_a_teardown_failure_at_all(tmp_path):
    """The writer's half of the same defect: the report never reached the
    file, so no reader could have shown it."""
    written = []

    class _Report:
        when = "teardown"
        outcome = "failed"
        nodeid = "t.py::t0"
        duration = 0.0
        longreprtext = "conftest.py:498: AssertionError: THE CANARY FIRED"

    progress_plugin._STATE["path"] = str(tmp_path / "b.jsonl")
    progress_plugin._STATE["run"] = "r1"
    try:
        progress_plugin.pytest_runtest_logreport(_Report())
        written = [json.loads(l) for l in
                   (tmp_path / "b.jsonl").read_text().splitlines() if l]
    finally:
        progress_plugin._STATE["path"] = None
    assert len(written) == 1, "the teardown failure was dropped"
    assert written[0]["outcome"] == "failed"
    assert written[0]["nodeid"] == "t.py::t0 [teardown]", (
        "without the suffix it collides with the same test's passing CALL "
        "record and the reader shows one test as both passed and failed")


# --------------------------------------------------------------------------- #
#  The writer's half: evidence is not destroyed                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("last_line,finished", [
    ('{"event": "done", "exitstatus": 0, "time": 1.0}', True),
    ('{"event": "test", "nodeid": "t.py::t", "outcome": "passed", '
     '"duration": 0.0, "reason": "", "time": 1.0}', False),
    ('{"event": "start", "time": 1.0}', False),
    ('{"event": "test", "nodei', False),          # a killed process mid-write
])
def test_only_a_file_whose_run_reached_done_may_be_truncated(
        tmp_path, last_line, finished):
    p = tmp_path / "b.jsonl"
    p.write_text(last_line + "\n")
    assert progress_plugin._previous_run_finished(str(p)) is finished


def test_an_absent_or_empty_file_may_be_truncated(tmp_path):
    assert progress_plugin._previous_run_finished(str(tmp_path / "nope")) is True
    (tmp_path / "empty.jsonl").write_text("")
    assert progress_plugin._previous_run_finished(
        str(tmp_path / "empty.jsonl")) is True


def test_configure_appends_a_generation_instead_of_wiping_a_live_run(tmp_path):
    """THE 2026-09-12 defect, at its source.

    A second pytest pointed at a file a running one is writing used to truncate
    it, and the first run then appended its remaining records into the empty
    file -- producing a fragment that read as a run which stopped half way.
    """
    p = tmp_path / "none2e.jsonl"
    _write(p, _run("live", 3, collected=9081, pid=os.getpid()))
    before = p.read_text()

    class _Config:
        def getoption(self, _name):
            return str(p)

    # Restored HERE, not by a fixture: `pytest_configure` points the plugin's
    # module state at this tmp file, and the plugin writes THIS test's own
    # `call` report before any fixture teardown runs.  Leaving the restore to
    # monkeypatch sent that record into the tmp file instead of the live
    # progress file -- 15 tests passed and 14 records were written, and the
    # partial-run rule above duly flagged a complete run.  A test that must
    # borrow a module's global has to give it back inside its own body.
    _saved = dict(progress_plugin._STATE)
    try:
        progress_plugin.pytest_configure(_Config())
    finally:
        progress_plugin._STATE.update(_saved)

    after = p.read_text()
    assert after.startswith(before), (
        "the live run's records were destroyed:\n" + after[:400])
    recs = [json.loads(l) for l in after.splitlines() if l.strip()]
    assert [r["event"] for r in recs].count("start") == 2
    assert recs[-1]["event"] == "start" and recs[-1]["pid"] == os.getpid()
    # And the reader then answers about the NEW generation, not the fragment.
    assert testrun._summarise("b", str(p))["run"] == recs[-1]["run"]
