"""A run's own records -- what launch wrote into a run folder, and what the
run itself left saying how it ended: ``run.json`` (`project-layout.md`
§ 1.6.3), the conclusion marker, ``.continued-from``, ``.gathered-from``.
Floor 1 (`execution/architecture.md` § 2.1): plain facts read off one
folder, which know no calculation -- so launch, status, the run record and
the directory readers under `parse/` ask them without reaching the layout.

A module of its own since 2026-10-03 (W55 B7): they lived in
`jobset/materialize.py`, floor 4, and `parse/` imported them from there.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


#: Written by ``launch`` into the attempt, AFTER the launch succeeds
#: (``project-layout.md`` § 1.6).  Its presence is the only honest answer to
#: *has this been launched?* -- a queued job has produced nothing yet, so
#: "no output" and "not started" are indistinguishable from the directory alone.
RUN_LAUNCH_SCHEMA = "molbuilder/run-launch@1"
RUN_LAUNCH_FILE = "run.json"


def was_launched(where: Path, basename: Optional[str] = None) -> bool:
    """Whether ``launch`` has launched this run — its launch record exists:
    an attempt's ``run.json``, or with ``basename`` a flat stage's own
    (:func:`launch_record_at` says which).

    This is the whole reason that file exists. Without it, preparing a stage
    twice could rewrite the setup underneath a job already sitting in a queue,
    because a queued job has written nothing and looks exactly like one that
    was never started (§ 1.6).  *(It read ``run.json`` alone until
    2026-10-01, so a flat stage -- whose record is ``<basename>.run.json`` --
    always read as never launched, and was launched again over a run still
    in the queue; W52.)*
    """
    return launch_record_path(where, basename).is_file()


def continued_from_marker(where: Path, basename: Optional[str] = None
                          ) -> Path:
    """Where `prep` leaves the run a stage continues from, for `launch` to
    write into the launch record (`project-layout.md` § 1.6.3): the
    attempt's ``.continued-from``, or a flat stage's own
    ``<basename>.continued-from`` beside its other files -- the flat layout
    records it too (user, 2026-10-01)."""
    from .runfiles import tail
    return Path(where) / (".continued-from" if basename is None
                          else basename + tail(".continued-from"))


def attempt_concluded(attempt_dir: Path, basename: str) -> Optional[str]:
    """The conclusion marker's content when this attempt's LAST process got
    to say goodbye, else ``None`` (`project-layout.md` § 1.6, *the other
    file*).

    *Launched* spans three states; ``run.json`` separates none of them.
    The wrapper writes ``<basename>-run<N>.concluded`` as its last act on
    the MAIN path -- an engine error still reaches it, a kill never does
    -- so the marker separates *ran to its own end* from *still running
    or force-stopped*, and those last two are indistinguishable from
    files alone, which is why the caller asks the user rather than
    deciding.

    A warm-retry chain execs fresh wrappers; only the final process
    concludes, at the final run index.  So the question is asked of the
    HIGHEST index any per-run artifact reached: an earlier index's marker
    beside a newer unconcluded ``.out`` is a previous re-run's goodbye,
    not this one's.  The index ranges over ``.out`` AND ``.concluded``
    together, because an engine that dies before printing a single line
    leaves a marker and NO ``.out`` -- a real conclusion (rc rides the
    marker) that an out-only rule read as silence (caught by this file's
    own error-path test, first run).  Nothing per-run at all reads as
    unconcluded: a launch killed before the engine is exactly a process
    that never said goodbye.
    """
    d = Path(attempt_dir)
    # THE INDEX IS ASKED FOR, not scanned for.  This globbed
    # `f"{basename}-run*.{suffix}"` over three suffixes and pulled N back out
    # with a regex of its own -- the `-run<N>` counter written twice here and
    # once more in `summarize`, for a grammar `runfiles` composes in one
    # place (`project-layout.md` § 4.5).  `latest_run` reads it through
    # `runfiles.parse`, so a name this module cannot read is not counted as
    # an attempt.
    #
    # ACROSS EVERY ROLE, which is the rule and not an implementation detail.
    # The wrapper's redirect is engine-specific -- SIESTA's `-runN.out`,
    # PySCF's `-runN.pyscf.log` -- and an engine that dies before printing
    # leaves a `.concluded` and no output at all.  Asking without a `role=`
    # ranges over all of them, so a NEWER killed attempt cannot hide behind
    # an OLDER one's goodbye.
    from .runfiles import latest_run, tail as _rf_tail
    newest = latest_run(d, basename)
    if newest is None:
        return None
    # AND THE NAME IS COMPOSED BY THE GRAMMAR TOO.  This spelled
    # `f"{basename}-run{newest}.concluded"` one line after asking `latest_run`
    # for the counter -- the door answered the SEARCH and the caller still
    # built the name (`project-layout.md` § 4.5, the compose half).
    #
    # `tail`, NOT `compose`, and the difference is load-bearing here:
    # ``basename`` is whatever the DECK is called, and for a cited transport
    # directory that is a person's own file -- `my.relaxation.fdf`.  `compose`
    # would refuse it, correctly (§ 2.1: a label carrying a dot cannot be read
    # back out of a filename), and this function's job is to answer None, not
    # to raise at a person who named a file with a dot in it.  `tail` cuts the
    # `-run<N>.concluded` end off a real `compose` result, so the GRAMMAR owns
    # the tail and the caller owns the stem it was handed.
    mark = d / (basename + _rf_tail(".concluded", run=newest))
    try:
        return mark.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def conclusion_line(attempt_dir: Path, basename: str) -> Optional[str]:
    """The conclusion marker's first line -- ``rc=0 at <date>`` -- or
    ``None`` when the run has not concluded (:func:`attempt_concluded`).
    An EMPTY marker is a conclusion, so it reads ``""``: two readers took
    the first line two ways until 2026-10-01, and one of them raised on an
    empty marker after the new attempt was already open (W52)."""
    mark = attempt_concluded(attempt_dir, basename)
    return None if mark is None else (mark.splitlines() or [""])[0].strip()


def launch_record_path(where: Path, basename: Optional[str] = None) -> Path:
    """Where a launch is recorded (`project-layout.md` § 1.6.3): an attempt's
    ``run.json``; with ``basename`` -- a flat stage's deck stem,
    ``<label>_<token>`` -- that stage's own ``<basename>.run.json``, beside
    every other file of it in the calculation's one directory."""
    from .runfiles import tail
    return Path(where) / (RUN_LAUNCH_FILE if basename is None
                          else basename + tail(".run.json"))


def write_run_launch(attempt_dir: Path, *, mode: str, command: List[str],
                     job_id: Optional[str] = None,
                     continued_from: Optional[str] = None,
                     launched_at: Optional[str] = None,
                     placed_on: Optional[dict] = None,
                     basename: Optional[str] = None) -> Path:
    """Record a launch into the attempt — ``molbuilder/run-launch@1`` -- or,
    given ``basename``, a flat stage's own record in its calculation's
    directory (:func:`launch_record_path`).

    Written **after** the launch succeeds, so a failed launch leaves the
    attempt exactly as prepare left it and is still safe to prepare again
    (§ 1.6). ``continued_from`` is the run's provenance — *this geometry came
    from ``01_coarse/run-0``* — which is worth recording whether or not
    anything reads it back.

    ``placed_on`` is WHERE IT WAS SENT: the domain name, its partition and
    qos (`scheduler.md` R12 -- the queue half; what it LANDED ON is the
    monitor's ``[MACHINE]`` line, written on the node).  The placement was
    already in this file, buried inside the ``sbatch`` argv as ``-p``/``-q``
    — so reading it back meant parsing a command line, which is the
    re-derivation A4 exists to remove.  *(It said "WHERE IT RAN" and
    carried a ``node_type`` until 2026-08-27 -- a queue's opinion of
    itself, which the probe never wrote, and the reason S3's check never
    fired.)*

    Absent when there was no placement to record — a direct run, or a machine
    with no queue at all. **Absent means the question cannot be answered**,
    which a reader must not mistake for *yes*.
    """
    from datetime import datetime, timezone
    p = launch_record_path(attempt_dir, basename)
    body = {
        "schema": RUN_LAUNCH_SCHEMA,
        "mode": mode,
        "command": list(command),
        "job_id": job_id,
        "launched_at": launched_at or datetime.now(timezone.utc)
                                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # ABSENT, not null, when this run started from the structure.
    # ``checkpointing.md`` S3 words its test that way -- *"names a directory
    # that exists or is absent"* -- and the two are different to a reader that
    # tests for the key rather than for its truthiness.
    if continued_from:
        body["continued_from"] = str(continued_from)
    # Same absent-not-null rule: a direct run has no placement, and a reader
    # testing for the key learns that rather than reading a null as "nowhere".
    if placed_on:
        body["placed_on"] = dict(placed_on)
    p.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return p


def read_run_launch(attempt_dir, basename: Optional[str] = None
                    ) -> Optional[dict]:
    """The launch record, or ``None`` when there is none -- the reader
    beside :func:`write_run_launch`.

    An attempt's ``run.json`` answers for the attempt.  A flat stage has no
    attempt: ``basename`` names its own record (:func:`launch_record_path`);
    without one, the newest stage record in the directory answers for it --
    a flat calculation was launched when any of its stages was.

    `project-layout.md` § 1.6: *"Has this been launched? has no honest answer
    from the directory alone"*, so this file is the answer.  Present but
    unreadable reads ``{}``: launched, the details lost.  *(It was a private
    `runstatus._launch_record` until 2026-09-26, when the run record became
    its second reader.)*
    """
    if attempt_dir is None:
        return None
    p = launch_record_path(attempt_dir)
    if not p.is_file():
        if basename is not None:
            p = launch_record_path(attempt_dir, basename)
        else:
            from .runfiles import find_by_role
            stages = sorted(find_by_role(attempt_dir, ".run.json"),
                            key=lambda f: f.stat().st_mtime)
            p = stages[-1] if stages else p
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return {}


#: What a gathered rung took from which upstream attempt, one ``<file> <-
#: <attempt>`` line each -- an attempt's own file, beside ``run.json``:
#: written by `prep`'s gather (:func:`write_gathered_from`) and read by the
#: run record's provenance (:func:`read_gathered_from`, `model/parse.md`
#: § 5d.4).  *(Both sat in `prep` until 2026-09-26, so the record imported
#: the conductor to read one file.)*
GATHERED_FROM_FILE = ".gathered-from"


def write_gathered_from(attempt_dir, gathered) -> None:
    """``gathered`` -- ``[(source attempt, filename), ...]`` in the order
    taken -- as the attempt's ``.gathered-from``."""
    (Path(attempt_dir) / GATHERED_FROM_FILE).write_text(
        "".join(f"{fn} <- {src}\n" for src, fn in gathered),
        encoding="utf-8")


def read_gathered_from(attempt_dir) -> List[dict]:
    """``[{"file", "from"}]`` -- what this attempt was gathered from, in the
    order it was taken; ``[]`` when it gathered nothing."""
    p = Path(attempt_dir) / GATHERED_FROM_FILE
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        name, sep, src = line.partition(" <- ")
        if sep and name.strip() and src.strip():
            out.append({"file": name.strip(), "from": src.strip()})
    return out


__all__ = ["RUN_LAUNCH_SCHEMA", "RUN_LAUNCH_FILE", "was_launched", "continued_from_marker", "attempt_concluded", "conclusion_line", "launch_record_path", "write_run_launch", "read_run_launch", "GATHERED_FROM_FILE", "write_gathered_from", "read_gathered_from"]
