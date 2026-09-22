#!/usr/bin/env python
"""Batched test runner + LIVE progress reader.

Runs the pytest suite in two speed-separated batches -- ``none2e`` (fast unit /
JS / node tests) and ``e2e`` (slow Playwright) -- with ``tools.progress_plugin``
streaming every result to ``.test-progress/<batch>.jsonl`` as it happens, so
``status`` can report live at any moment (no buffered-until-exit blindness).

Usage
-----
    # launch a batch (run it in the BACKGROUND from the shell / harness):
    python tools/testrun.py run none2e         # all non-e2e tests
    python tools/testrun.py run e2e            # all *_e2e.py
    python tools/testrun.py run all            # everything, one file
    python tools/testrun.py run e2e tests/test_molbuilder_e2e.py   # explicit targets
    python tools/testrun.py run lf             # rerun ONLY last-run failures (any batch)

    # read progress LIVE, any time, from another shell:
    python tools/testrun.py status             # summarise every batch
    python tools/testrun.py status e2e         # one batch
    python tools/testrun.py status --fails     # also print each failed id + reason
    python tools/testrun.py failed e2e         # bare failed node-ids (feed back to pytest)

Design notes
------------
* Progress files live under ``<repo>/.test-progress/`` (git-ignored) so ANY
  session retrieves them at a stable path -- no job-specific tmp.
* ``failed`` prints node-ids you can pass straight back to pytest to rerun only
  the failures -- the fix-the-whole-batch-then-verify loop, no full reruns.
* Single pytest process per batch is single-core, so two batches run
  concurrently on a multi-core box without contention (xdist not required).
"""
import argparse
import json
import os
import fcntl
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRESS_DIR = os.path.join(REPO, ".test-progress")

# Batch -> pytest target selection.  ``none2e`` excludes e2e FILES (their test
# names don't reliably contain "e2e", so -k is wrong -- ignore by glob).
BATCHES = {
    "none2e": ["tests/", "--ignore-glob=*_e2e.py"],
    "e2e":    ["tests/", "-o", "python_files=*_e2e.py"],  # collect only *_e2e.py
    "all":    ["tests/"],
}


def _progress_path(batch):
    return os.path.join(PROGRESS_DIR, f"{batch}.jsonl")


def _lock_path(batch):
    return os.path.join(PROGRESS_DIR, f"{batch}.lock")


def _acquire_batch_lock(batch):
    """Take an exclusive lock for this batch, or return None if held.

    ``flock`` on a lock file, which is the whole mechanism: the kernel releases
    it when this process exits -- normally, on Ctrl-C, or on SIGKILL -- so there
    is no stale state to detect and nothing to clean up.  The returned handle
    must stay open for the run's lifetime (keep it in a local).

    Deliberately NOT a pid file: that needs liveness probing, /proc reading to
    survive pid reuse, and deleting files we inferred were stale -- inference
    plus deletion, to solve a problem the kernel already solves.  The lock file
    itself is left in place; it holds no state.

    The pid inside is advisory only, so a refusal can name who holds it.
    """
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    fh = open(_lock_path(batch), "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:                     # held by another run
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def _lock_holder(batch):
    """Best-effort pid of the run holding the lock, or None.  Informational
    only -- never used to decide anything."""
    try:
        text = open(_lock_path(batch)).read().strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def _read_events(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a half-written last line during a live read -- skip it
    return out


def _writer_is_alive(start_rec):
    """Is the process that wrote this `start` record still running?

    Without this a file whose run was killed reads as ``running`` for ever --
    `.test-progress/all.jsonl` sat that way for a day.  Unknown (a file from
    before `pid` was recorded) counts as alive, because claiming a live run is
    dead is the worse error.
    """
    pid = start_rec.get("pid")
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, owned by someone else
    except OSError:
        return True
    return True


def _summarise(batch, path):
    """What the progress file actually supports saying.

    **A count is only a result when the file is one complete run.**  This
    function used to count whatever records were present and label the batch
    ``done`` if a ``done`` record existed anywhere, which on 2026-09-12
    reported the TAIL of a finished run -- a file a second pytest had truncated
    under it -- as a run that stopped after 4947 of 9081 tests.  Nothing in the
    output said the file was unreadable, so the number was believed and ~4000
    tests were recorded as "not executing" when they had all run.

    Three states replace that, and none of them prints a pass count:

    ``unusable``     no ``start`` record (the file was truncated under a run
                     that was still writing), or no ``collected`` record, so
                     there is nothing to compare ``ran`` against.
    ``interleaved``  records from more than one run inside one generation.
    ``partial``      a ``done`` record, but fewer tests ran than were
                     collected -- a real early stop, which IS worth knowing
                     and is exactly what must not be called ``done``.

    Only records from the LAST ``start`` are considered: a new run appends a
    generation rather than destroying the old one, so the newest is the one
    being asked about.
    """
    ev = _read_events(path)
    if not ev:
        return {"batch": batch, "state": "no-data", "path": path}

    starts = [i for i, e in enumerate(ev) if e["event"] == "start"]
    if not starts:
        return {"batch": batch, "state": "unusable", "path": path,
                "why": "no `start` record -- the file was truncated under a "
                       "run that was still writing it, so what is left is a "
                       "fragment, not a run"}
    gen = ev[starts[-1]:]
    run_id = gen[0].get("run")
    strays = {e.get("run") for e in gen} - {run_id}
    start_t = gen[0]["time"]
    collected = next((e["n"] for e in gen if e["event"] == "collected"), None)
    done = next((e for e in gen if e["event"] == "done"), None)
    tests = [e for e in gen if e["event"] == "test"]
    last_t = max((e["time"] for e in gen), default=start_t)
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for e in tests:
        counts[e["outcome"]] = counts.get(e["outcome"], 0) + 1
    # A teardown failure is an extra record against a test that already has a
    # CALL record, so it must not count toward `ran` -- otherwise `ran`
    # overshoots `collected` and the ran/collected comparison stops meaning
    # anything.  It still counts as a failure, because it is one.
    ran = sum(1 for e in tests if not e["nodeid"].endswith(" [teardown]"))

    if strays:
        state = "interleaved"
        why = (f"records from {len(strays) + 1} runs share this generation "
               f"({ran} test records cannot be attributed)")
    elif collected is None:
        state = "unusable"
        why = ("no `collected` record -- collection never finished, so there "
               "is no total to compare against")
    elif done and ran < collected:
        state = "partial"
        why = (f"the run stopped after {ran} of {collected} collected tests "
               f"(exit {done['exitstatus']}); this is NOT a suite result")
    elif done and done["exitstatus"] != 0 and counts["failed"] == 0:
        # THE EXIT CODE IS EVIDENCE TOO, AND IT OUTRANKS THE COUNTS.
        # pytest exits non-zero for things that never become a test record:
        # a session-scoped fixture raising in teardown (the conftest canaries
        # that guard the config dir, the checkout and the conda envs), an
        # internal error, a plugin error.  Printing `FAIL 0` there reported a
        # fired canary as a clean suite on 2026-09-22.  `status` now refuses
        # to call it done -- the discrepancy IS the finding.
        state = "unexplained"
        why = (f"pytest exited {done['exitstatus']} but no test was recorded "
               f"as failed. Something outside a test's call failed -- most "
               f"likely a session-scoped fixture raising in teardown (the "
               f"`the_suite_leaves_your_*_alone` canaries in "
               f"`tests/conftest.py`), an internal error, or a plugin error. "
               f"This is NOT green: read the runner's own stdout.")
    elif done:
        state = "done"
        why = None
    elif _writer_is_alive(gen[0]):
        state = "running"
        why = None
    else:
        state = "abandoned"
        why = (f"no `done` record and the run's process is gone -- it was "
               f"killed or crashed after {ran} tests; the counts below are "
               f"a fragment")

    return {
        "batch": batch,
        "state": state,
        "why": why,
        "run": run_id,
        "exitstatus": done["exitstatus"] if done else None,
        "collected": collected,
        "ran": ran,
        "remaining": (collected - ran) if collected is not None else None,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "elapsed": round((last_t - start_t), 1),
        "failed_ids": [(e["nodeid"], e.get("reason", "")) for e in tests
                       if e["outcome"] == "failed"],
        "path": path,
    }


def cmd_run(args):
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    extra = args.targets
    if args.batch == "lf":
        sel = ["tests/", "--last-failed", "--last-failed-no-failures", "none"]
        batch_file = "lf"
    elif args.batch in BATCHES:
        sel = list(BATCHES[args.batch]) if not extra else list(extra)
        batch_file = args.batch
    else:
        # treat the batch token as an explicit target path
        sel = [args.batch] + list(extra)
        batch_file = "custom"
    prog = _progress_path(batch_file)

    # One run per batch.  Two runs of the same batch append to one progress
    # file, and the interleaved events make `status` nonsense -- more tests
    # "ran" than were collected, the two runs' failures mixed, and the first
    # run's "done" reported while the second is still going.  That cost real
    # debugging time on 2026-07-29.  (Different batches are fine and intended:
    # separate files, separate locks.)
    lock = _acquire_batch_lock(batch_file)
    if lock is None:
        holder = _lock_holder(batch_file)
        who = f" (pid {holder})" if holder else ""
        if not args.force:
            print(f"[testrun] REFUSING: batch {batch_file!r} is already "
                  f"running{who}.", file=sys.stderr)
            print(f"[testrun]   watch it: python tools/testrun.py status "
                  f"{batch_file}", file=sys.stderr)
            print(f"[testrun] Two runs of one batch share {prog} and make the "
                  f"counts meaningless.  Pass --force to run anyway.",
                  file=sys.stderr)
            return 2
        print(f"[testrun] --force: second {batch_file} run alongside the one "
              f"already going{who}; both write {prog}, so `status "
              f"{batch_file}` interleaves them until the older exits.",
              file=sys.stderr)

    cmd = [sys.executable, "-m", "pytest", *sel,
           "-p", "tools.progress_plugin", f"--progress-file={prog}",
           "-q", "-rf", "--tb=line"]
    print(f"[testrun] batch={batch_file}  progress={prog}", flush=True)
    print("[testrun] " + " ".join(cmd), flush=True)
    # cache provider ON (default) so `run lf` works.
    #
    # ``lock`` stays in scope until this returns -- that is what holds it for
    # the run's duration; the kernel drops it however we exit.  On the --force
    # path it is None (the first run still holds it), so a THIRD run sees the
    # batch as free and is allowed too.  That is the honest consequence of
    # --force: it means "I accept an interleaved progress file", and pretending
    # to serialise the forced runs against each other would be a half-guarantee
    # worse than none.
    return subprocess.call(cmd, cwd=REPO)


def cmd_status(args):
    """Print each batch's state.  Non-zero when any batch cannot be trusted.

    A number that reads like a result is printed ONLY for ``done``; everything
    else leads with what is wrong with the file, because the failure this guards
    against was a believable-looking count.
    """
    batches = [args.batch] if args.batch else _known_batches()
    if not batches:
        print("no progress files under .test-progress/ yet")
        return 0
    untrustworthy = False
    for b in batches:
        s = _summarise(b, _progress_path(b))
        if s["state"] == "no-data":
            print(f"[{b}] no data")
            continue
        if s["state"] in ("unusable", "interleaved", "abandoned"):
            untrustworthy = True
            print(f"[{b}] {s['state'].upper()} -- not a suite result")
            print(f"      {s['why']}")
            if s["state"] == "abandoned":
                print(f"      ran {s['ran']}/{s['collected']}  "
                      f"pass {s['passed']}  FAIL {s['failed']}  "
                      f"skip {s['skipped']}")
            print(f"      {s['path']}")
            if args.fails and s.get("failed_ids"):
                print(f"      ({len(s['failed_ids'])} failure records are "
                      f"still real; ids below)")
                for nid, reason in s["failed_ids"]:
                    print(f"    FAIL {nid}")
                    if reason:
                        print(f"         -> {reason}")
            continue
        loud = s["state"] in ("partial", "unexplained")
        head = (f"[{b}] {s['state'].upper() if loud else s['state']}"
                + (f" (exit {s['exitstatus']})" if s['exitstatus'] is not None else "")
                + f" | {s['ran']}/{s['collected']} ran"
                + f" | pass {s['passed']}  FAIL {s['failed']}  skip {s['skipped']}"
                + f" | {s['elapsed']}s")
        print(head)
        if loud:
            untrustworthy = True
            print(f"      {s['why']}")
        if args.fails and s["failed_ids"]:
            for nid, reason in s["failed_ids"]:
                print(f"    FAIL {nid}")
                if reason:
                    print(f"         -> {reason}")
    return 1 if untrustworthy else 0


def cmd_failed(args):
    b = args.batch or "e2e"
    s = _summarise(b, _progress_path(b))
    for nid, _reason in s.get("failed_ids", []):
        print(nid)
    return 0


def _known_batches():
    if not os.path.isdir(PROGRESS_DIR):
        return []
    return sorted(f[:-6] for f in os.listdir(PROGRESS_DIR) if f.endswith(".jsonl"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)

    pr = sub.add_parser("run", help="launch a batch with live progress")
    pr.add_argument("batch", help="none2e | e2e | all | lf | <target path>")
    pr.add_argument("targets", nargs="*", help="explicit pytest targets/args")
    pr.add_argument("--force", action="store_true",
                    help="run even if this batch is already running "
                         "(their progress files will interleave)")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("status", help="summarise live progress")
    ps.add_argument("batch", nargs="?", help="one batch, or all if omitted")
    ps.add_argument("--fails", action="store_true", help="list failed ids + reasons")
    ps.set_defaults(func=cmd_status)

    pf = sub.add_parser("failed", help="print bare failed node-ids for a batch")
    pf.add_argument("batch", nargs="?", help="batch name (default e2e)")
    pf.set_defaults(func=cmd_failed)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
