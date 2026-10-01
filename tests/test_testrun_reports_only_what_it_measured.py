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
destroying evidence -- and that a run spread over workers still has one writer
and reads exactly as a one-process run does.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

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


def test_a_file_that_has_gone_quiet_is_not_running_whatever_the_pid(tmp_path):
    """`all.jsonl` claimed `running` on 2026-09-22 off a `start` written on
    2026-09-11 -- eleven days after its pytest died.

    The record predates the plugin writing a `pid`, and the rule was
    "unknown counts as alive", so no amount of time could retire it.  That
    is not cosmetic: the standing rule is not to edit the working tree while
    a run is in flight, so a phantom run suppresses real work.  The plugin
    writes and flushes a record per test, so silence is the signal.
    """
    p = tmp_path / "b.jsonl"
    _write(p, _run("r1", 3255, collected=9197))      # no pid, no done
    assert testrun._summarise("b", str(p))["state"] == "running", \
        "a file written just now is live -- the clock must not be too tight"
    old = time.time() - testrun._SILENCE_MEANS_DEAD - 1
    os.utime(p, (old, old))
    assert testrun._summarise("b", str(p))["state"] == "abandoned", \
        "a progress file nothing has written to in half an hour is not a run"


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

    # RESTORE WHAT WAS THERE, never `None`.  `_write` returns early on a
    # falsy path, so parking `None` here does not "disable the plugin for
    # this test" -- it disables it for the REST OF THE SESSION, silently.
    # This test did exactly that on 2026-09-22: the live `none2e` file
    # stopped at 7812 of 9530 records with no `done`, pytest exited 0 having
    # passed everything, and `status` called a complete run ABANDONED.  The
    # neighbour below carries the same warning for the same reason.
    saved = dict(progress_plugin._STATE)
    progress_plugin._STATE["path"] = str(tmp_path / "b.jsonl")
    progress_plugin._STATE["run"] = "r1"
    try:
        progress_plugin.pytest_runtest_logreport(_Report())
        written = [json.loads(l) for l in
                   (tmp_path / "b.jsonl").read_text().splitlines() if l]
    finally:
        progress_plugin._STATE.update(saved)
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


# --------------------------------------------------------------------------- #
#  A run spread over workers                                                  #
# --------------------------------------------------------------------------- #

#: Every test notes the worker it ran on.
_NOTE = ("import os, pathlib\n"
         "import pytest\n"
         "def note(name):\n"
         "    (pathlib.Path({seen!r}) / name).write_text(\n"
         "        os.environ['PYTEST_XDIST_WORKER'])\n")


def _spread_run(tmp_path, files):
    """``testrun.py run --workers 2`` over a suite whose ``conftest.py`` and
    ``pyproject.toml`` ARE the real suite's (links), so the rules a run is
    spread by are the suite's own.  Returns the summary, the suite and where
    each test ran."""
    suite, seen = tmp_path / "suite", tmp_path / "seen"
    suite.mkdir()
    seen.mkdir()
    repo = Path(testrun.REPO)
    (suite / "conftest.py").symlink_to(repo / "tests" / "conftest.py")
    (suite / "pyproject.toml").symlink_to(repo / "pyproject.toml")
    for name, body in files.items():
        (suite / name).write_text(_NOTE.format(seen=str(seen)) + body)
    assert testrun.main(["run", str(suite), "--workers", "2"]) == 1
    s = testrun._summarise("custom", testrun._progress_path("custom"))
    return s, suite, {f.name: f.read_text() for f in seen.iterdir()}


@pytest.mark.slow
def test_a_run_spread_over_workers_reads_as_one_run(tmp_path, monkeypatch):
    """GOAL: a run spread over xdist workers is reported exactly as a
    one-process run is, and spread the way `testing.md` § 6.1a says.

    CONTRACT: one progress file from one writer, ``done`` with every
    collected test counted; the failed ids are the plain ones pytest accepts
    back, in the progress file and in the last-failed cache that ``run lf``
    reads; a file that fails to collect is a failure of its own, named by its
    path; a file runs whole in one worker; every file holding an engine test,
    marked on its module or on a test, shares one worker.
    """
    monkeypatch.setattr(testrun, "PROGRESS_DIR", str(tmp_path / "progress"))
    s, suite, ran_on = _spread_run(tmp_path, {
        "test_alpha.py": ("def test_a1(): note('a1')\n"
                          "def test_a2(): note('a2')\n"
                          "def test_a3(): note('a3')\n"),
        "test_beta.py": ("def test_b1(): note('b1')\n"
                         "def test_b2(): note('b2'); assert False\n"
                         "def test_b3(): note('b3')\n"),
        # The engine files are the largest groups and the mixed file's
        # unmarked tests outnumber the rest, so that grouped any other way
        # xdist's first assignment -- largest group first, one per worker --
        # would part them.
        "test_engine_one.py": ("pytestmark = pytest.mark.engine\n"
                               "def test_e1(): note('e1')\n"
                               "def test_e2(): note('e2')\n"
                               "def test_e3(): note('e3')\n"
                               "def test_e4(): note('e4')\n"),
        "test_engine_two.py": ("@pytest.mark.engine\n"
                               "def test_e5(): note('e5')\n"
                               + "".join(f"def test_f{i}(): note('f{i}')\n"
                                         for i in range(1, 7))),
        "test_broken.py": "import no_such_module\n",
    })
    assert (s["state"], s["collected"], s["ran"]) == ("done", 17, 17), s
    events = [e["event"] for e in testrun._read_events(s["path"])]
    assert [e for e in events if e != "test"] == [
        "start", "collect", "collected", "done"]
    failed = {"test_beta.py::test_b2", "test_broken.py"}
    assert {nid for nid, _ in s["failed_ids"]} == failed, s
    assert "no_such_module" in dict(s["failed_ids"])["test_broken.py"], s
    lastfailed = suite / ".pytest_cache" / "v" / "cache" / "lastfailed"
    assert set(json.loads(lastfailed.read_text())) == failed
    assert len({ran_on[t] for t in ("a1", "a2", "a3")}) == 1, ran_on
    engine = [f"e{i}" for i in range(1, 6)] + [f"f{i}" for i in range(1, 7)]
    assert len({ran_on[t] for t in engine}) == 1, ran_on


@pytest.mark.slow
def test_a_worker_that_dies_ends_the_run_and_names_its_test(
        tmp_path, monkeypatch):
    """GOAL: a crash in a spread run is reported as what it is, and the run
    ends -- the tests its worker had not reached are not run, and not counted.

    CONTRACT (`testing.md` § 6.1a): the test whose worker died -- here in its
    teardown, after its call passed -- is recorded as failed with the crash
    as its reason, so ``status --fails`` names it, and is counted once; the
    next test of its file never runs, so ``ran`` stays short of
    ``collected`` and the run reads PARTIAL; no replacement worker re-runs
    it (a replacement could be handed a finished group and wait for ever).
    """
    monkeypatch.setattr(testrun, "PROGRESS_DIR", str(tmp_path / "progress"))
    s, _suite, ran_on = _spread_run(tmp_path, {
        "test_crash.py": ("@pytest.fixture\n"
                          "def dies_after():\n"
                          "    yield\n"
                          "    os._exit(3)\n"
                          "def test_x(dies_after): note('x')\n"
                          "def test_y(): note('y')\n")})
    assert (s["state"], s["collected"], s["ran"]) == ("partial", 2, 1), s
    [(nid, reason)] = s["failed_ids"]
    assert nid == "test_crash.py::test_x" and "crashed" in reason, s
    assert "y" not in ran_on, ran_on


def test_run_lf_with_nothing_to_rerun_runs_nothing(tmp_path, monkeypatch):
    """GOAL: ``run lf`` reruns the last failures, never the whole suite.

    CONTRACT (`testing.md` § 6.1a): a last-failed list naming no test file
    here -- xdist's own report of workers that disagreed (``gw1``), the id of
    a file since deleted -- is nothing to rerun, where pytest's
    ``--last-failed`` would run every test.  Driven through ``run lf``; no
    pytest starts, so no ``lf`` progress is written.
    """
    cache = tmp_path / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "lastfailed").write_text(json.dumps(
        {"gw1": True, "tests/test_deleted.py::test_x": True}))
    monkeypatch.setattr(testrun, "REPO", str(tmp_path))
    monkeypatch.setattr(testrun, "PROGRESS_DIR", str(tmp_path / "progress"))
    assert testrun.main(["run", "lf"]) == 0
    assert not (tmp_path / "progress" / "lf.jsonl").exists()
