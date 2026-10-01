"""Live test-progress pytest plugin.

Streams EACH test outcome to a JSONL file the instant it finishes (flushed
per-test), so progress is retrievable LIVE at any moment -- unlike piping
pytest's stdout to a file, which buffers until the process exits.

Enable it on any pytest run:

    python -m pytest <targets> -p tools.progress_plugin --progress-file=PATH

``tools/testrun.py`` wires this up for you (per-batch files under
``.test-progress/``) and reads the file back with ``status``.

JSONL event schema (one JSON object per line).  Every record carries ``run``,
the id of the run that wrote it, so a file holding records from two runs can be
SEEN to (``tools/testrun.py status`` refuses to count such a file rather than
reporting a number from whatever survived):
    {"event":"start",     "run":<id>, "pid":<int>, "time":<epoch>}
    {"event":"collected", "run":<id>, "n":<int>, "time":<epoch>}
    {"event":"test",  "run":<id>, "nodeid":<str>, "outcome":"passed|failed|skipped",
                      # nodeid gains a " [teardown]" suffix for a teardown failure
                      "duration":<sec>, "reason":<short str>, "time":<epoch>}
    {"event":"collect", "run":<id>, "nodeid":<str>, "outcome":"failed",
                      # a file (or xdist's own check) that failed to collect
                      "reason":<short str>, "time":<epoch>}
    {"event":"done",  "run":<id>, "exitstatus":<int>, "time":<epoch>}

**A file is truncated only when the previous run FINISHED.**  Truncating at
every session start cost a real measurement on 2026-09-12: a second pytest
pointed at a batch's progress file wiped the records of the run already writing
it, and what was left -- the tail of a complete run, with no ``start`` and no
``collected`` -- read as a run that stopped half way.  Roughly 4000 tests were
reported as "not executed" that had executed.  When the file does not end in a
``done`` record the new run APPENDS its own ``start`` generation instead, so
nothing is destroyed and the reader can tell the generations apart.

**A run spread over xdist workers has one writer** (`docs/process/testing.md`
§ 6.1a): the process the run was started in.  Every worker's results are
relayed to it, so the file reads exactly as a one-process run's does; the
workers load this plugin too (they receive the run's own arguments) and stay
silent.  A module global holds the path, because there is one writer.
"""
import json
import os
import time

import pytest

_STATE = {"path": None, "run": None, "collected": False}


def pytest_addoption(parser):
    parser.addoption(
        "--progress-file", action="store", default=".test-progress.jsonl",
        help="Live JSONL progress file (tools/progress_plugin).",
    )


def _write(rec):
    path = _STATE["path"]
    if not path:
        return
    rec["run"] = _STATE["run"]
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    except OSError:
        pass  # progress logging must never break a test run


def _previous_run_finished(path):
    """Did the file's last record come from a run that reached ``done``?

    Only then is truncating it safe.  A file whose last record is a ``start``
    or a ``test`` belongs to a run that is either still writing or died; in
    both cases its records are evidence, and wiping them is how a complete run
    came to look like a half one.
    """
    try:
        with open(path) as fh:
            last = None
            for line in fh:
                line = line.strip()
                if line:
                    last = line
    except OSError:
        return True  # absent or unreadable -- nothing to preserve
    if last is None:
        return True  # empty
    try:
        return json.loads(last).get("event") == "done"
    except json.JSONDecodeError:
        return False  # a half-written line means a run was interrupted


def pytest_configure(config):
    # A WORKER IS NOT A WRITER.  Each one would open its own generation in the
    # one file -- with eight workers, nine `start` records from nine
    # processes, the interleaving `testrun.py status` refuses to count.  Its results reach the writer
    # through xdist, which fires `pytest_runtest_logreport` there for each.
    if hasattr(config, "workerinput"):
        return
    path = config.getoption("--progress-file")
    _STATE["path"] = path
    _STATE["run"] = f"{os.getpid()}-{int(time.time())}"
    _STATE["collected"] = False
    # Truncate only a file whose previous run finished; otherwise append a new
    # generation beside records that are still someone's evidence.
    if _previous_run_finished(path):
        try:
            open(path, "w").close()
        except OSError:
            _STATE["path"] = None
            return
    _write({"event": "start", "pid": os.getpid(), "time": time.time()})


def _collected(n):
    """The run's total, written once."""
    if not _STATE["collected"]:
        _STATE["collected"] = True
        _write({"event": "collected", "n": n, "time": time.time()})


def pytest_collection_finish(session):
    _collected(len(session.items))


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    """The run's total, when it is spread over workers.

    The writer collects nothing then -- the workers do -- so
    `pytest_collection_finish` never fires in it.  Every worker collects the
    whole run, and xdist aborts a run whose workers disagree, so the first
    worker's count is the run's.  A replacement worker, where xdist starts
    one, collects again; that count is not written a second time.
    """
    _collected(len(ids))


def _reason(report):
    """The last line of a failure's text -- where the error names itself."""
    lines = [ln for ln in (report.longreprtext or "").splitlines() if ln.strip()]
    return lines[-1][:300] if lines else ""


def pytest_collectreport(report):
    # A FILE THAT FAILS TO COLLECT -- an import that broke -- is a failure of
    # its own, and no test record can carry it: its tests never exist.  One
    # process stops at it; a spread run goes on with the other files
    # (xdist's own loop replaces the one that stops), and before this record
    # `status` then called the run UNEXPLAINED and blamed the canaries.
    if report.failed:
        _write({"event": "collect", "nodeid": report.nodeid,
                "outcome": "failed", "reason": _reason(report),
                "time": time.time()})


def pytest_runtest_logreport(report):
    # Record the CALL phase for every test, PLUS setup-phase failures/skips
    # (a test that errors or is skipped in setup never reaches "call") PLUS
    # teardown failures.
    #
    # TEARDOWN IS NOT AN AFTERTHOUGHT -- IT IS WHERE THE CANARIES LIVE.
    # `tests/conftest.py` guards the developer's config directory, checkout
    # and conda envs with session-scoped fixtures that RAISE AFTER the yield,
    # so the one report that names them arrives with `when == "teardown"`
    # (against whichever test happened to be last).  Skipping that phase made
    # pytest exit 1 while this file recorded every test as passed, and
    # `testrun.py status` printed `FAIL 0` -- a canary firing read as green.
    # Observed 2026-09-22: the checkout canary caught a real edit-during-run
    # and the summary said the suite was clean.
    #
    # A WORKER THAT DIES, in a spread run, is reported by xdist as a failure
    # of the test it was running, under a phase that is none of the three.
    # Left out, the one test that names the crash had no record at all.
    is_call = report.when == "call"
    is_setup_terminal = report.when == "setup" and report.outcome in ("failed", "skipped")
    is_teardown_failure = report.when == "teardown" and report.outcome == "failed"
    is_crash = report.when not in ("setup", "call", "teardown")
    if not (is_call or is_setup_terminal or is_teardown_failure or is_crash):
        return
    reason = _reason(report) if report.outcome == "failed" else ""
    _write({
        "event": "test",
        # The phase is part of the identity: a teardown failure shares its
        # nodeid with the same test's passing CALL record, and without this
        # the reader sees one test both passed and failed.
        "nodeid": (f"{report.nodeid} [teardown]" if is_teardown_failure
                   else report.nodeid),
        "outcome": report.outcome,
        "duration": round(getattr(report, "duration", 0.0), 2),
        "reason": reason,
        "time": time.time(),
    })


def pytest_sessionfinish(session, exitstatus):
    _write({"event": "done", "exitstatus": int(exitstatus), "time": time.time()})
