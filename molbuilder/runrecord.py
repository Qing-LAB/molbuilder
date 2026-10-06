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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# IT TRAVELS BESIDE EVERY JOB (`runwrap.MONITOR_COMPANIONS`): the monitor
# reports how a run ended with `parse.dirs.job.run_status`, whose state is
# built on :func:`ending` -- so this module is imported two ways, from the
# package or from beside the job, as `wrapper_log` is.
try:                                        # inside molbuilder
    from . import runfiles as _rf
except ImportError:                         # beside a job, as the monitor's
    import runfiles as _rf


def _persist():
    """`persist`, the one JSON reader and writer, imported where a record is
    read or written -- never beside a job, where only :func:`ending` is
    asked, so it does not travel."""
    from . import persist
    return persist


#: Written by ``launch`` into the attempt, AFTER the launch succeeds
#: (``project-layout.md`` § 1.6).  Its presence is the only honest answer to
#: *has this been launched?* -- a queued job has produced nothing yet, so
#: "no output" and "not started" are indistinguishable from the directory alone.
RUN_LAUNCH_SCHEMA = "molbuilder/run-launch@1"
RUN_LAUNCH_FILE = _rf.LAUNCH_RECORD_FILE      # the catalogue's name


class LaunchRecordError(ValueError):
    """A launch record that is there and does not read -- neither JSON, nor
    a ``molbuilder/run-launch`` record.  Never *launched* or *not
    launched*: whether the run was launched cannot be told from it, so
    every reader stops and names the file (`execution/architecture.md`
    § 3.2)."""


def continued_from_marker(where: Path, basename: Optional[str] = None
                          ) -> Path:
    """Where `prep` leaves the run a stage continues from, for `launch` to
    write into the launch record (`project-layout.md` § 1.6.3): the
    attempt's ``.continued-from``, or a flat stage's own
    ``<basename>.continued-from`` beside its other files -- the flat layout
    records it too (user, 2026-10-01)."""
    return Path(where) / (_rf.CONTINUED_FROM_FILE if basename is None
                          else basename + _rf.tail(".continued-from"))


def write_continued_from(where, run, *, basename: Optional[str] = None,
                         plan=None) -> Path:
    """THE ONE WRITER of the run a stage continues from -- every hand-over is
    one `Continuation`, and this is its record beside the run
    (`job-system.md` § 5.4): ``run``, an attempt from the calculation folder
    -- or on the flat layout the run's own name -- as
    :func:`continued_from_marker`'s file, through `persist`, or into
    ``plan`` (`jobset.planned.Plan`).  *(Three writers wrote it by hand
    until 2026-10-05: the attempt's opener, prep's flat arm and launch's
    flat re-launch.)*"""
    marker = continued_from_marker(where, basename)
    data = (str(run) + "\n").encode("utf-8")
    if plan is not None:
        plan.bytes(marker, data)
        return marker
    _persist().write_bytes(marker, data)
    return marker


def read_continued_from(where, basename: Optional[str] = None
                        ) -> Optional[str]:
    """The run :func:`write_continued_from` recorded, or ``None``."""
    try:
        text = continued_from_marker(where, basename).read_text(
            encoding="utf-8")
    except OSError:
        return None
    return text.strip() or None


def read_concluded(text: Optional[str]) -> Optional[dict]:
    """The conclusion marker's first line, ``rc=<N> at <when>[; <note>]`` as
    the wrapper writes it (`runwrap.py`), as ``{"code": N, "at": when,
    "note": note}`` -- ``at`` and ``note`` only when stated -- or ``None``
    when the text is not one.  The note is what the wrapper adds when the job's finish
    failed or cannot run (`parse.dirs.job.FINISH_FAILED`,
    `FINISH_CANNOT_LOAD`).  THE one reader of that line -- beside the marker
    it reads since 2026-10-03; `parse/dirs/job.py` held it before.
    """
    head = text.splitlines()[0] if text else ""
    m = re.search(r"\brc=(-?\d+)(?:\s+at\s+(.*?))?(?:;\s*(.*?))?\s*$", head)
    if m is None:
        return None
    return {"code": int(m.group(1)),
            **({"at": m.group(2)} if m.group(2) else {}),
            **({"note": m.group(3)} if m.group(3) else {})}


@dataclass(frozen=True)
class Ending:
    """HOW A RUN ENDED (`execution/architecture.md` § 3.2): the line its
    conclusion said -- molbuilder's marker's first line, ``rc=0 at <date>``;
    ``rc=?`` for an empty one -- or ``None`` when it has not ended on its own (still running, or force-stopped: no file
    tells those two apart); and the exit code that line states."""
    line: Optional[str] = None
    code: Optional[int] = None

    @property
    def concluded(self) -> bool:
        """It ended on its own -- an engine error included, a kill never."""
        return self.line is not None

    @property
    def ok(self) -> bool:
        """It ended on its own with exit code 0 -- one of the facts the
        status door weighs (`parse.dirs.run_status`); a hand-over builds on
        that door's answer, never on this alone
        (`jobset.continuation.usable`)."""
        return self.code == 0


def ending(where, basename: str) -> Ending:
    """THE door for *did this run end on its own, and with what exit code?*
    (`execution/architecture.md` § 3.2; plan W38 F3) -- asked by status,
    whose state is built on it and every hand-over builds on
    (`jobset.continuation.usable`), by launch's re-launch question and the
    transport citation.

    ``where`` is the run's folder; ``basename`` its stem, ``<label>_<token>``
    -- the run's own, in either shape: an attempt's folder may hold several
    run indexes, and a flat calculation's every stage.

    **Molbuilder's own marker first** -- ``<basename>-run<N>.concluded``,
    the wrapper's last act on its main path: an engine error still reaches
    it, a kill never does, and it carries the exit code.  It counts at the
    newest run index any file of the run reached (`runfiles.at_latest_run`):
    an earlier index's marker beside a newer run is that run's goodbye.

    **Nothing else**: a run molbuilder's wrapper did not run is not a run of
    ours, so it never ended on its own here -- the engine's own end mark
    (SIESTA's ``0_NORMAL_EXIT``) says the ENGINE ended, while the job -- a
    finish after it, the wrapper's own end -- may not have.  *(It answered,
    in a folder no wrapper of ours ran in, until 2026-10-03, for a
    relaxation run by hand and cited -- input molbuilder does not take,
    user, 2026-10-03.  Three readers answered this until that day too -- the
    marker alone for prep's gates, marker-or-mark for the citation, the
    output's ending first for status -- so status said finished where prep
    refused.)*
    """
    d = Path(where)
    marks = [m for m in _rf.find_by_role(d, ".concluded")
             if _rf.parse(m.name, basename) is not None]
    latest = _rf.at_latest_run(d, marks, basename)
    if latest:
        try:
            text = latest[0].read_text(encoding="utf-8")
        except OSError:
            return Ending()
        line = (text.splitlines() or [""])[0].strip() or "rc=?"
        said = read_concluded(line)
        return Ending(line, None if said is None else said["code"])
    return Ending()


def launch_record_path(where: Path, basename: Optional[str] = None) -> Path:
    """Where a launch is recorded (`project-layout.md` § 1.6.3): an attempt's
    ``run.json``; with ``basename`` -- a flat stage's deck stem,
    ``<label>_<token>`` -- that stage's own ``<basename>.run.json``, beside
    every other file of it in the calculation's one directory."""
    return Path(where) / (RUN_LAUNCH_FILE if basename is None
                          else basename + _rf.tail(".run.json"))


def write_launch(attempt_dir: Path, *, mode: str, command: List[str],
                 job_id: Optional[str] = None,
                 continued_from: Optional[str] = None,
                 launched_at: Optional[str] = None,
                 placed_on: Optional[dict] = None,
                 basename: Optional[str] = None) -> Path:
    """Record a launch into the attempt — ``molbuilder/run-launch@1`` -- or,
    given ``basename``, a flat stage's own record in its calculation's
    directory (:func:`launch_record_path`) -- through `persist`, so it is
    whole or absent, never half a file (`execution/architecture.md` § 2.1:
    a run's own records are read and written through `persist`).

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
    return _persist().write_json(p, body)


def launch_record(where, basename: Optional[str] = None) -> Optional[dict]:
    """THE door for *was this run launched?* (`execution/architecture.md`
    § 3.2; plan W38 F2) -- the run's launch record, or ``None`` when there
    is none: asked by status, the Run panel, prep (an unlaunched attempt is
    reused), every launch gate and the transport citation.

    An attempt's ``run.json`` answers for the attempt.  A flat stage has no
    attempt: ``basename`` names its own record (:func:`launch_record_path`);
    without one, the newest stage record in the directory answers for it --
    a flat calculation was launched when any of its stages was.

    `project-layout.md` § 1.6: *"Has this been launched? has no honest answer
    from the directory alone"*, so this file is the answer -- and **one that
    does not read** is :class:`LaunchRecordError`, naming the file, never
    an answer: it read ``{}``, *launched, the details lost*, until
    2026-10-03, while the gates asked whether the file existed.  *(It was a
    private `runstatus._launch_record` until 2026-09-26, and
    ``read_run_launch`` beside a second answerer, ``was_launched``, until
    2026-10-03.)*
    """
    if where is None:
        return None
    p = launch_record_path(where)
    if not p.is_file():
        if basename is not None:
            p = launch_record_path(where, basename)
        else:
            stages = sorted(_rf.find_by_role(where, ".run.json"),
                            key=lambda f: f.stat().st_mtime)
            p = stages[-1] if stages else p
    if not p.is_file():
        return None
    try:
        body = _persist().read_json(p)
        if not isinstance(body, dict):
            raise ValueError("not a JSON object")
        _persist().check_schema(body.get("schema"), RUN_LAUNCH_SCHEMA,
                                label=p.name)
    except (OSError, ValueError) as e:
        raise LaunchRecordError(
            f"{p} does not read as a launch record ({e}) -- whether this run "
            f"was launched cannot be told from it.  It is written once, at "
            f"launch, and the decision ledger's `launched` line holds what it "
            f"said (project-layout.md 1.6.3).") from e
    return body


#: What a gathered rung took from which upstream attempt, one ``<file> <-
#: <attempt>`` line each -- an attempt's own file, beside ``run.json``:
#: written by `prep`'s gather (:func:`write_gathered_from`) and read by the
#: run record's provenance (:func:`read_gathered_from`, `model/parse.md`
#: § 5d.4).  *(Both sat in `prep` until 2026-09-26, so the record imported
#: the conductor to read one file.)*
GATHERED_FROM_FILE = _rf.GATHERED_FROM_FILE   # the catalogue's name


def write_gathered_from(attempt_dir, gathered, *, plan=None) -> None:
    """``gathered`` -- ``[(source attempt, filename), ...]`` in the order
    taken -- as the attempt's ``.gathered-from``, through `persist`, or into
    ``plan`` (`jobset.planned.Plan`)."""
    data = "".join(f"{fn} <- {src}\n" for src, fn in gathered).encode("utf-8")
    if plan is not None:
        plan.bytes(Path(attempt_dir) / GATHERED_FROM_FILE, data)
        return
    _persist().write_bytes(Path(attempt_dir) / GATHERED_FROM_FILE, data)


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


__all__ = ["RUN_LAUNCH_SCHEMA", "RUN_LAUNCH_FILE", "LaunchRecordError", "continued_from_marker", "write_continued_from", "read_continued_from", "read_concluded", "Ending", "ending", "launch_record_path", "write_launch", "launch_record", "GATHERED_FROM_FILE", "write_gathered_from", "read_gathered_from"]
