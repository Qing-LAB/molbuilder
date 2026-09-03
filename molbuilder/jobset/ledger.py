"""The bundle's DECISION LEDGER — ``jobset-decisions.log``, one JSON line
per decision.

**Why a file when every verb already prints** (user rule, 2026-08-12): a
decision that only reached the terminal is gone by the time a job
misbehaves on a cluster hours later.  What is still there is the bundle —
so the bundle carries the record: which config file supplied each setting,
how the mode resolved and from where, which trial was picked and why,
whether a verdict was offered and what the answer was.  Debugging a
machine difference is then *reading the ledger in order*, not re-running.

**The layering** mirrors the rest of the engine: library layers RETURN
decision data (a picked trial, a provenance table, a merged plan) and the
SURFACE that acted on it appends the line — policy stays at the verb, the
ledger only records.  Per-job launch provenance already has a home
(``run.json``, `job-contracts.md` § 6.1) and is not duplicated here; the
ledger is the bundle-level ORDER of decisions across verbs.

Append-only JSONL, never rotated by us, and never fatal: a run must not
fail because its logbook could not be written.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

#: Beside ``job-set.json`` in the bundle root — one ledger per calculation,
#: all verbs interleaved in the order they actually ran.
LEDGER_FILE = "jobset-decisions.log"


def record(base, verb: str, decision: str, **facts) -> None:
    """Append one decision line: ``{"at", "verb", "decision", ...facts}``.

    ``facts`` must be JSON-serialisable; anything that isn't is recorded
    as its ``repr`` rather than dropped, because a half-told decision is
    the thing this file exists to prevent.  Secrets never come this way by
    construction: callers pass RESOLVED decisions (mode, stage, counts,
    file names), not raw config sections — the same exclusion the
    provenance display applies (runtime_config.config_provenance).
    """
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "verb": verb, "decision": decision}
    entry.update(facts)
    try:
        line = json.dumps(entry, default=repr, sort_keys=False)
        with (Path(base) / LEDGER_FILE).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass                              # the logbook must never break a run
