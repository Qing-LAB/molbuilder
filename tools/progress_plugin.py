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
                      "duration":<sec>, "reason":<short str>, "time":<epoch>}
    {"event":"done",  "run":<id>, "exitstatus":<int>, "time":<epoch>}

**A file is truncated only when the previous run FINISHED.**  Truncating at
every session start cost a real measurement on 2026-09-12: a second pytest
pointed at a batch's progress file wiped the records of the run already writing
it, and what was left -- the tail of a complete run, with no ``start`` and no
``collected`` -- read as a run that stopped half way.  Roughly 4000 tests were
reported as "not executed" that had executed.  When the file does not end in a
``done`` record the new run APPENDS its own ``start`` generation instead, so
nothing is destroyed and the reader can tell the generations apart.

Single-process only (xdist is not installed here); a module global holds the
path.  If xdist is ever added, switch to writing per-worker files.
"""
import json
import os
import time

_STATE = {"path": None, "run": None}


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
    path = config.getoption("--progress-file")
    _STATE["path"] = path
    _STATE["run"] = f"{os.getpid()}-{int(time.time())}"
    # Truncate only a file whose previous run finished; otherwise append a new
    # generation beside records that are still someone's evidence.
    if _previous_run_finished(path):
        try:
            open(path, "w").close()
        except OSError:
            _STATE["path"] = None
            return
    _write({"event": "start", "pid": os.getpid(), "time": time.time()})


def pytest_collection_finish(session):
    _write({"event": "collected", "n": len(session.items), "time": time.time()})


def pytest_runtest_logreport(report):
    # Record the CALL phase for every test, PLUS setup-phase failures/skips
    # (a test that errors or is skipped in setup never reaches "call").
    is_call = report.when == "call"
    is_setup_terminal = report.when == "setup" and report.outcome in ("failed", "skipped")
    if not (is_call or is_setup_terminal):
        return
    reason = ""
    if report.outcome == "failed":
        txt = report.longreprtext or ""
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        reason = lines[-1][:300] if lines else ""
    _write({
        "event": "test",
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "duration": round(getattr(report, "duration", 0.0), 2),
        "reason": reason,
        "time": time.time(),
    })


def pytest_sessionfinish(session, exitstatus):
    _write({"event": "done", "exitstatus": int(exitstatus), "time": time.time()})
