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
ledger only records.  Where more than one surface records the SAME decision,
what that line consists of lives here as a named function (:func:`prepped`)
and the surfaces call it: appending stays theirs, and the recipe stops being
something each of them has to remember.  Per-job launch provenance already
has a home
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


def prepped(base, *, kind: str, stage, dirs):
    """Record one ``prep`` — and RETURN the provenance it recorded.

    **What a prep line consists of lives here, not in each surface.**  The
    module rule above puts the append at the surface, and that is right:
    policy stays at the verb.  What it cannot do by itself is keep two
    surfaces spelling the same four facts the same way — and it did not.
    `prep` from the CLI wrote this line; `prep` from the browser wrote
    nothing at all, while the Task Setup bundle card went on listing
    ``jobset-decisions.log`` as *"every decision prep made, one line each"*
    (measured 2026-09-19: `grep -rn ledger molbuilder/web/` returns one
    line, the import of the NAME).  A person who preps in the browser opens
    the card, reads that promise, and finds no file.

    So the recipe has one home and the surfaces have one call.  Adding a
    third surface is that call, not four lines free to disagree with these.

    The provenance comes back because the CLI also PRINTS it
    (`format_provenance`) — the same table, recorded once and displayed
    once, rather than gathered twice.
    """
    from ..runtime_config import config_provenance

    base = Path(base)
    prov = config_provenance(project_dir=base)
    record(base, "prep", "prepped", kind=kind, stage=stage,
           job_dirs=sorted(rel_to(base, d) for d in dirs), provenance=prov)
    return prov


def rel_to(base, d) -> str:
    """*d* as a path from the calculation, or unchanged if it is outside.

    The ledger records job directories this way and `prep` PRINTS them this
    way, and the two must agree: with the attempt layer every trial's
    directory ends in ``run-<n>``, so a list reading "run-0, run-0" names
    nobody (user, 2026-08-28).  One function, because a second spelling is
    how the printed list and the recorded one come to disagree about the
    same directories.
    """
    try:
        return str(Path(d).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(d)
