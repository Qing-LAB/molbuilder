"""The bundle's DECISION LEDGER — ``jobset-decisions.log``, one JSON line
per decision.

**Why a file when every verb already prints** (user rule): a
decision that only reached the terminal is gone by the time a job
misbehaves on a cluster hours later.  What is still there is the bundle —
so the bundle carries the record: which config file supplied each setting,
how the mode resolved and from where, which trial was picked and why,
whether a verdict was offered and what the answer was.  Debugging a
machine difference is then *reading the ledger in order*, not re-running.

**The layering** mirrors the rest of the engine: library layers RETURN
decision data (a picked trial, a provenance table, a merged plan) and the
VERB that acted on it appends the line — policy stays at the verb, the
ledger only records.  A verb with one door appends at that surface; a verb
whose doors share ONE ENTRY appends in the entry, once, whichever door
called — `prep`'s, `jobset/prep.prep_stage` (`job-system.md` § 5.3), writes
its lines (`preflight-report`, `saved`, `continues` or `starts-cold`,
`gathers`, `launch-agreement`, `prepped`, and `refused`), after its save,
for the command line and the Task setup tab alike.  `launch`'s, likewise
(`jobset/submit.py`: `plan_launch`, `ask_launch`, `send_launch`), writes
its refusals, the question it put and its answer (`question`, a *no* too),
what it asked a scheduler (`asked`) and each submission as it goes
(`launched`, a run here when it starts); its verb, the refusals it says
before it calls the entry.  A dry run writes nothing.
What a line consists of, where more than one caller records the same
decision, lives here as a named function (:func:`prepped`), so the recipe
is never something each caller has to remember.  Per-job launch provenance already
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
from ..runfiles import LEDGER_FILE  # noqa: E402,F401 -- the catalogue's name


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


def prepped(base, *, kind: str, stage, dirs, provenance):
    """Record one ``prep`` — and RETURN the provenance it recorded.

    **What a prep line consists of lives here, not in each caller.**  The
    module rule above puts the append at the verb -- for `prep`, its one
    entry, which both doors call.  What it cannot do by itself is keep two
    callers spelling the same four facts the same way.  A surface that
    hand-builds the line is free to spell it differently, or to omit it
    altogether — and then the Task Setup bundle card's promise that
    ``jobset-decisions.log`` holds *"every decision prep made, one line
    each"* is false for whoever prepped through that surface: they open the
    card, read the promise, and find no file.

    So the recipe has one home and the entry has one call.

    ``provenance`` is which config files answered, as the prep entry read
    them with the machine's record at its checkpoint 4 (`prep.prep_stage`)
    -- the table `STAGE-PLAN.md` and the pipeline log carry.  It comes back because the
    answer carries it: the command line PRINTS it (`format_provenance`) and
    the Task setup tab receives it -- the same table, read once, recorded
    and displayed from the answer.  *(It was read again here, after the
    write, until 2026-10-05: at a first prep the record that answered was
    another file than the one STAGE-PLAN.md named.)*
    """
    base = Path(base)
    record(base, "prep", "prepped", kind=kind, stage=stage,
           job_dirs=sorted(rel_to(base, d) for d in dirs),
           provenance=provenance)
    return provenance


def rel_to(base, d) -> str:
    """*d* as a path from the calculation, or unchanged if it is outside.

    The ledger records job directories this way and `prep` PRINTS them this
    way, and the two must agree: with the attempt layer every trial's
    directory ends in ``run-<n>``, so a bare basename list reads "run-0,
    run-0" and names nobody (user).  One function, because a second
    spelling is how the printed list and the recorded one come to disagree
    about the same directories.
    """
    try:
        return str(Path(d).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(d)
