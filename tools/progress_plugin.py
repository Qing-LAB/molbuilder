"""Live test-progress pytest plugin.

Streams EACH test outcome to a JSONL file the instant it finishes (flushed
per-test), so progress is retrievable LIVE at any moment -- unlike piping
pytest's stdout to a file, which buffers until the process exits.

Enable it on any pytest run:

    python -m pytest <targets> -p tools.progress_plugin --progress-file=PATH

``tools/testrun.py`` wires this up for you (per-batch files under
``.test-progress/``) and reads the file back with ``status``.

JSONL event schema (one JSON object per line):
    {"event":"start",     "time":<epoch>}
    {"event":"collected", "n":<int>, "time":<epoch>}
    {"event":"test",  "nodeid":<str>, "outcome":"passed|failed|skipped",
                      "duration":<sec>, "reason":<short str>, "time":<epoch>}
    {"event":"done",  "exitstatus":<int>, "time":<epoch>}

Single-process only (xdist is not installed here); a module global holds the
path.  If xdist is ever added, switch to writing per-worker files.
"""
import json
import time

_STATE = {"path": None}


def pytest_addoption(parser):
    parser.addoption(
        "--progress-file", action="store", default=".test-progress.jsonl",
        help="Live JSONL progress file (tools/progress_plugin).",
    )


def _write(rec):
    path = _STATE["path"]
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    except OSError:
        pass  # progress logging must never break a test run


def pytest_configure(config):
    _STATE["path"] = config.getoption("--progress-file")
    # Truncate at session start so a re-run starts clean.
    try:
        open(_STATE["path"], "w").close()
    except OSError:
        _STATE["path"] = None
        return
    _write({"event": "start", "time": time.time()})


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
