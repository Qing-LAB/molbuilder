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


def _summarise(batch, path):
    ev = _read_events(path)
    if not ev:
        return {"batch": batch, "state": "no-data", "path": path}
    start = next((e["time"] for e in ev if e["event"] == "start"), None)
    collected = next((e["n"] for e in ev if e["event"] == "collected"), None)
    done = next((e for e in ev if e["event"] == "done"), None)
    tests = [e for e in ev if e["event"] == "test"]
    last_t = max((e["time"] for e in ev), default=start)
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for e in tests:
        counts[e["outcome"]] = counts.get(e["outcome"], 0) + 1
    ran = len(tests)
    return {
        "batch": batch,
        "state": "done" if done else "running",
        "exitstatus": done["exitstatus"] if done else None,
        "collected": collected,
        "ran": ran,
        "remaining": (collected - ran) if collected is not None else None,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "elapsed": round((last_t - start), 1) if start else None,
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
    cmd = [sys.executable, "-m", "pytest", *sel,
           "-p", "tools.progress_plugin", f"--progress-file={prog}",
           "-q", "-rf", "--tb=line"]
    print(f"[testrun] batch={batch_file}  progress={prog}", flush=True)
    print("[testrun] " + " ".join(cmd), flush=True)
    # cache provider ON (default) so `run lf` works.
    return subprocess.call(cmd, cwd=REPO)


def cmd_status(args):
    batches = [args.batch] if args.batch else _known_batches()
    if not batches:
        print("no progress files under .test-progress/ yet")
        return 0
    for b in batches:
        s = _summarise(b, _progress_path(b))
        if s["state"] == "no-data":
            print(f"[{b}] no data")
            continue
        head = (f"[{b}] {s['state']}"
                + (f" (exit {s['exitstatus']})" if s['exitstatus'] is not None else "")
                + f" | {s['ran']}/{s['collected']} ran"
                + f" | pass {s['passed']}  FAIL {s['failed']}  skip {s['skipped']}"
                + f" | {s['elapsed']}s")
        print(head)
        if args.fails and s["failed_ids"]:
            for nid, reason in s["failed_ids"]:
                print(f"    FAIL {nid}")
                if reason:
                    print(f"         -> {reason}")
    return 0


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
