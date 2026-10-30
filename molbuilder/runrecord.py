"""A run's own records -- what launch wrote into a run folder, and what the
run itself left saying how it ended: ``run.json`` (`project-layout.md`
§ 1.6.3), the conclusion marker, ``.continued-from``, ``.gathered-from``.
Floor 1 (`execution/architecture.md` § 2.1): plain facts read off one
folder, which know no calculation -- so launch, status, the run record and
the directory readers under `parse/` ask them without reaching the layout.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

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
    read or written -- from the package, or beside a job, where the
    monitor's `notify` verb writes the run's notification record
    (:func:`note_notified`), so it travels too
    (`runwrap.MONITOR_COMPANIONS`)."""
    try:                                    # inside molbuilder
        from . import persist
    except ImportError:                     # beside a job, as the monitor's
        import persist
    return persist


#: Written by ``launch`` into the attempt, AFTER the launch succeeds
#: (``project-layout.md`` § 1.6).  Its presence is the only honest answer to
#: *has this been launched?* -- a queued job has produced nothing yet, so
#: "no output" and "not started" are indistinguishable from the directory alone.
RUN_LAUNCH_SCHEMA = "molbuilder/run-launch@1"


class LaunchRecordError(ValueError):
    """A run's record of its launch that is there and does not read -- its
    launch record, neither JSON nor a ``molbuilder/run-launch`` record, or
    a run of points' index (`runpoints.read_index`).  Never *launched* or
    *not launched*: what the launch did cannot be told from it, so every
    reader stops and names the file (`execution/architecture.md` § 3.2)."""


@dataclass(frozen=True)
class RunRef:
    """A RUN OF THIS CALCULATION, AS A RECORD NAMES IT -- the one spelling
    `.continued-from`, `.gathered-from` and `run.json`'s ``continued_from``
    hold (`job-system.md` § 5.4): the run's ``folder`` from the
    calculation's (``01_coarse/run-0``; a point's ``04_device/run-0/v0.2``).
    Written and read one way: ``str(ref)``, and :meth:`read`."""
    folder: PurePosixPath

    def __str__(self) -> str:
        return self.folder.as_posix()

    @classmethod
    def at(cls, base, folder) -> "RunRef":
        """The run in ``folder``, named from the calculation's ``base``."""
        return cls(PurePosixPath(Path(folder).relative_to(base).as_posix()))

    @classmethod
    def read(cls, text: str) -> "RunRef":
        """``text`` as a record holds it."""
        return cls(PurePosixPath(text.strip()))

    def path(self, base) -> Path:
        """The folder its files lie in."""
        return Path(base) / self.folder

    def within(self, other: "RunRef") -> bool:
        """It lies inside ``other``'s folder -- a point of that run."""
        return other.folder in self.folder.parents

    def run_folder(self) -> "RunRef":
        """The run it lies in -- itself, or for a point of a run of points
        the run folder above it (`paths.run_folder_of`, the one reading of
        which run a point belongs to)."""
        from .paths import run_folder_of      # inside molbuilder only
        got = run_folder_of(self.folder)
        return self if got is None else RunRef(PurePosixPath(got.as_posix()))


def continued_from_marker(where) -> Path:
    """Where the run a run continues from is recorded, in the run's folder
    (`project-layout.md` § 1.6.3): its ``.continued-from``."""
    return Path(where) / _rf.CONTINUED_FROM_FILE


def write_continued_from(where, source: RunRef, *, plan=None) -> Path:
    """THE ONE WRITER of the run a run continues from -- a hand-over's
    record beside the run (`materialize.hand_over`, `job-system.md` § 5.4):
    ``source`` as the run's :func:`continued_from_marker`, through
    `persist`, or into ``plan`` (`jobset.planned.Plan`)."""
    marker = continued_from_marker(where)
    data = (str(source) + "\n").encode("utf-8")
    if plan is not None:
        plan.bytes(marker, data)
        return marker
    _persist().write_bytes(marker, data)
    return marker


def read_continued_from(where) -> Optional[RunRef]:
    """The run :func:`write_continued_from` recorded for the run in
    ``where``, or ``None`` when there is no marker -- the run started from
    the structure, or from what it gathered.  A marker that is there and
    does not read is a :class:`LaunchRecordError` naming it."""
    marker = continued_from_marker(where)
    try:
        text = marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise LaunchRecordError(
            f"{marker} does not read ({exc}) -- what this run continues "
            f"from cannot be told.") from None
    return RunRef.read(text) if text.strip() else None


def conclude_line(code, note: str = "", *args: str) -> str:
    """The run script's command that writes its conclusion marker
    (``$_concluded``) -- ``rc=<N> at <date>[; <note>]``, the line
    :func:`read_concluded` reads: ``code`` the exit code, a number or a
    shell word (``"$_siesta_exit"``); ``note`` what the job adds, its own
    ``%s`` filled by ``args``, shell words.  ONE spelling for every writer
    of the line, beside its one reader."""
    fmt = "rc=%s at %s" + (f"; {note}" if note else "") + "\\n"
    return " ".join(["printf", f'"{fmt}"', str(code), '"$(date)"',
                     *args]) + ' > "$_concluded"'


def read_concluded(text: Optional[str]) -> Optional[dict]:
    """The conclusion marker's first line, ``rc=<N> at <when>[; <note>]`` as
    the wrapper writes it (`runwrap.py`), as ``{"code": N, "at": when,
    "note": note}`` -- ``at`` and ``note`` only when stated -- or ``None``
    when the text is not one.  The note is what the wrapper adds when the
    job had no python of its env (`parse.dirs.job.NO_ENV_PYTHON`), or its
    finish -- or a run in phases' plan or record step -- failed or cannot
    run (`FINISH_FAILED`, `FINISH_CANNOT_LOAD`).  THE one reader of that
    line.
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


def ending(where, basename: str, run: Optional[int] = None) -> Ending:
    """THE door for *did this run end on its own, and with what exit code?*
    (`execution/architecture.md` § 3.2; plan W38 F3) -- asked by status,
    whose state is built on it and every hand-over builds on
    (`jobset.continuation.usable`), by launch's re-launch question and the
    transport citation.

    ``where`` is the run's folder; ``basename`` its stem, ``<label>_<token>``
    -- a run's folder may hold several tries.  ``run`` asks about that
    try; by default the newest (:func:`newest_run`'s rule,
    `runfiles.latest_run`, which `runfiles.at_run` asks).

    **Molbuilder's own marker first** -- ``<basename>-run<N>.concluded``,
    the wrapper's last act on its main path: an engine error still reaches
    it, a kill never does, and it carries the exit code.  It counts at the
    newest try any file of the run reached (`runfiles.at_run`): an earlier
    try's marker beside a newer try's is that earlier try's goodbye.

    **Nothing else**: a run molbuilder's wrapper did not run is not a run of
    ours, so it never ended on its own here -- the engine's own end mark
    (SIESTA's ``0_NORMAL_EXIT``) says the ENGINE ended, while the job -- a
    finish after it, the wrapper's own end -- may not have.
    """
    d = Path(where)
    marks = [m for m in _rf.find_by_role(d, ".concluded")
             if _rf.parse(m.name, basename) is not None]
    latest = _rf.at_run(d, marks, basename, run)
    if latest:
        try:
            text = latest[0].read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # A MARKER THAT IS THERE AND DOES NOT READ: the run concluded --
            # the wrapper wrote it -- with an exit code no one can tell, as
            # an empty one; said, never read as a run still going.
            return Ending(f"rc=? ({latest[0].name} does not read: {exc})")
        line = (text.splitlines() or [""])[0].strip() or "rc=?"
        said = read_concluded(line)
        return Ending(line, None if said is None else said["code"])
    return Ending()


def launch_record_path(where) -> Path:
    """Where the run in ``where`` records its launch (`project-layout.md`
    § 1.6.3): the run's folder's ``run.json`` -- an attempt's, a trial's --
    which answers for every try in it."""
    return Path(where) / _rf.LAUNCH_RECORD_FILE


#: The run's notification record's schema (`runfiles.NOTIFIED_RECORD_FILE`).
NOTIFIED_SCHEMA = "molbuilder/run-notified@1"


def stamp() -> str:
    """THE CLOCK every record of a run is stamped with -- UTC, to the
    second, ``Z``: the launch record's `launched_at`, the ending's
    `asked_at`, a run of points' index (each point's start and end, the
    finish's; `runpoints`).  One spelling, so two records of
    one run compare -- and the job's own address (`joblist`) and a sequence
    of groups' record (`jobset.sequence`), which a run's is compared with
    (:func:`stamp_time`)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_time(text) -> Optional[datetime]:
    """A stamp READ BACK -- :func:`stamp`'s spelling, or a record written
    with its offset (``+00:00``) -- as a time two records compare by;
    ``None`` for text that is no time."""
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def notified_path(where) -> Path:
    """Where the run in ``where`` keeps its notification record: beside its
    launch record (`runfiles.NOTIFIED_RECORD_FILE`)."""
    return Path(where) / _rf.NOTIFIED_RECORD_FILE


def read_notified(where) -> List[dict]:
    """Every event sent for the run, in order -- ``[{event, at}]``, ``at``
    the epoch second, a failed part's ``reason`` beside it -- or ``[]``
    while none was (`run-reports.md` § 2.5).
    A record that does not read is :class:`LaunchRecordError`, naming it."""
    p = notified_path(where)
    if not p.is_file():
        return []
    return read_notified_file(p)


def read_notified_file(path) -> List[dict]:
    """The notification record at ``path``, read -- the one reader
    (:func:`read_notified` by the run's folder, the registry's parser by
    the file; `run-reports.md` § 2.5); refused as :class:`LaunchRecordError`
    when it does not read as one."""
    p = Path(path)
    try:
        body = _persist().read_json(p)
        if not isinstance(body, dict):
            raise ValueError("not a JSON object")
        _persist().check_schema(body.get("schema"), NOTIFIED_SCHEMA,
                                label=p.name)
        return [dict(e) for e in body.get("sent") or []]
    except (OSError, ValueError) as e:
        raise LaunchRecordError(
            f"{p} does not read as the run's notification record ({e}): "
            f"what the run has said cannot be told from it.") from e


def note_notified(where, event: str, at: float,
                  reason: Optional[str] = None) -> Path:
    """``event`` sent for the run at ``at`` (the epoch second), appended to
    its notification record -- with ``reason``, a failed part's why, which
    the run does not report twice (`run-reports.md` § 2.9) -- written whole
    through `persist`, so it is whole or absent
    (`execution/architecture.md` § 2.1).  THE RECORD HAS TWO WRITERS -- a
    call's member monitor and its record step's `notify` verb, which can
    speak at the same moment -- so the read and the write are one step,
    under a lock on the folder: an event appended by the one is never lost
    to the other.  Both run in the call's job, on one node, which is all
    the lock must cover; a filesystem that refuses locks gets the event
    written unlocked, as one writer would."""
    import fcntl
    import os
    p = notified_path(where)
    fd = os.open(str(p.parent), os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            pass
        sent = read_notified(where)
        entry = {"event": str(event), "at": float(at)}
        if reason is not None:
            entry["reason"] = str(reason)
        sent.append(entry)
        _persist().write_bytes(
            p, _persist().json_text({"schema": NOTIFIED_SCHEMA,
                                     "sent": sent}).encode("utf-8"))
    finally:
        os.close(fd)
    return p


def newest_run(where, names: "_rf.RunNames") -> Optional[int]:
    """THE RUN'S NEWEST TRY IN ``where`` -- the one rule every reader of a
    run asks (`model/parse.md` § 5.1, `project-layout.md` § 1.6.1): the
    highest try any file named on the run's stem reached
    (`runfiles.latest_run`); the folder's one launch record answers for
    every try in it."""
    return _rf.latest_run(Path(where), names.stem)


def newest_output(where, names: "_rf.RunNames") -> Optional[Path]:
    """THE ENGINE OUTPUT OF THE RUN'S NEWEST TRY in ``where``
    (:func:`newest_run`) -- ``None`` where that try printed none: an engine
    that died before its first line is never read from the try before it.
    Asked of a point's folder by a run of points' own steps (a vibration
    call's coordinates and facts, a transport round's NEGF count)."""
    n = newest_run(where, names)
    if n is None:
        return None
    out = Path(where) / names.name(".out", run=n)
    return out if out.is_file() else None


# --------------------------------------------------------------------- #
#  A run of points' index -- what every kind's index shares              #
#  (`engines/vibration.md` § 5.11 § 7, `engines/transport.md` § 2a.11)   #
# --------------------------------------------------------------------- #

def write_launch(where: Path, *, mode: str, command: List[str],
                 job_id: Optional[str] = None,
                 continued_from: Optional[RunRef] = None,
                 launched_at: Optional[str] = None,
                 placed_on: Optional[dict] = None,
                 scheduler: Optional[dict] = None,
                 sequence: Optional[str] = None) -> Path:
    """Record the launch of the run in ``where``
    (:func:`launch_record_path`) — ``molbuilder/run-launch@1`` -- through
    `persist`, so it is whole or absent, never half a file
    (`execution/architecture.md` § 2.1: a run's own records are read and
    written through `persist`).

    Written **after** the send succeeds: a refused or failed send records
    nothing, the run stays opened and unlaunched, emptied by whatever fills
    it next (`project-layout.md` § 1.6.3). ``continued_from`` is the run's provenance — *this geometry came
    from ``01_coarse/run-0``* — which is worth recording whether or not
    anything reads it back.

    ``placed_on`` is WHERE IT WAS SENT: the domain name, its partition and
    qos (`scheduler.md` R12 -- the queue half; what it LANDED ON is the
    monitor's ``[MACHINE]`` line, written on the node) -- and the wall and
    memory it was sent with: prep's, the wall the launch's ``--time`` when
    it restates it (`job-system.md` § 6.0).

    ``null`` for a run here, which no queue placed -- written, never left
    out: a reader tells "no queue" from "not recorded" by the key.

    ``scheduler`` is the scheduler the job was sent to and where it writes
    the job's error file, from the run's own folder -- ``{kind, stderr}``,
    ``stderr`` null where the job's id was not read -- the file the run
    door reads for how the scheduler ended the job (`scheduler.md` R16);
    ``null`` for a run here.

    ``sequence`` is the name of the sequence of groups whose job sent the
    run (`job-system.md` § 6.0, *A sequence of groups*) -- its other keys
    then the job's, as the sequence's launch record states them; ``null``
    for a run its launch sent itself.
    """
    p = launch_record_path(where)
    body = {
        "schema": RUN_LAUNCH_SCHEMA,
        "mode": mode,
        "command": list(command),
        "job_id": job_id,
        "launched_at": launched_at or stamp(),
    }
    # EVERY KEY, EVERY TIME, its null stated (`checkpointing.md` S3,
    # `job-contracts.md` § 6.1): ``continued_from`` null -- it started from
    # the structure; ``placed_on`` null -- a run here, which no queue
    # placed; ``scheduler`` null -- a run here, sent to no scheduler.
    body["continued_from"] = (str(continued_from) if continued_from
                              else None)
    body["placed_on"] = dict(placed_on) if placed_on else None
    body["scheduler"] = dict(scheduler) if scheduler else None
    body["sequence"] = sequence
    # ``ended`` null -- the scheduler has not been found to have forgotten
    # the job; :func:`record_ended` fills it, once.
    body["ended"] = None
    return _persist().write_json(p, body)


def record_ended(where, *, asked_on: str, said: str) -> Path:
    """THE SCHEDULER'S ANSWER, WRITTEN ONCE (`running-a-job.md` § 4.2,
    *Asked once, written once*; `project-layout.md` § 1.6.3): the launch
    record of the run in ``where`` gets
    ``ended = {asked_on, asked_at, listed: False, said}``,
    ``said`` the scheduler's own words for a job it lists ended, empty for
    one it has forgotten (`scheduler.queue.listed`).  Written through
    `persist`, whole; a run with no record cannot be concluded and is
    refused."""
    body = launch_record(where)
    if body is None:
        raise LaunchRecordError(
            f"{launch_record_path(where)} is not there: a run with no launch "
            f"record cannot be concluded by its scheduler's answer.")
    body["ended"] = {
        "asked_on": asked_on,
        "asked_at": stamp(),
        "listed": False,
        "said": said or "",
    }
    return _persist().write_json(launch_record_path(where), body)


def launch_record(where) -> Optional[dict]:
    """THE door for *was this run launched?* (`execution/architecture.md`
    § 3.2; plan W38 F2) -- the launch record of the run in ``where``, or
    ``None`` when there is none: asked by status, the Run panel, prep (an
    unlaunched attempt is reused), every launch gate and the transport
    citation.  A run's folder -- an attempt, a trial's -- has one record,
    which answers for every try in it.

    `project-layout.md` § 1.6: *"Has this been launched? has no honest answer
    from the directory alone"*, so this file is the answer -- and **one that
    does not read** is :class:`LaunchRecordError`, naming the file, never
    an answer.
    """
    if where is None:
        return None
    p = launch_record_path(where)
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


def sent_wall_s(record: Optional[dict]) -> Optional[int]:
    """The wall a launch record says its job was sent with, in seconds --
    ``placed_on.time``, in the record dialect (`scheduler.quantities.
    slurm_time` wrote it), read by that dialect's one reader,
    `scheduler.quantities.parse_walltime`, which travels beside the job with
    this module (`runwrap.MONITOR_COMPANIONS`) -- or ``None`` where it
    states none: a run here, which no queue placed.  A wall that is there
    and does not read is a :class:`LaunchRecordError`, never read as no
    wall."""
    try:                                    # inside molbuilder
        from .scheduler.quantities import parse_walltime
    except ImportError:                     # beside a job, as the monitor's
        from quantities import parse_walltime
    placed = (record or {}).get("placed_on") or {}
    wall = placed.get("time")
    if wall in (None, ""):
        return None
    try:
        return parse_walltime(wall)
    except ValueError:
        raise LaunchRecordError(
            f"the launch record's wall, {wall!r}, is no wall time -- "
            f"D-HH:MM:SS is the spelling its writer uses -- so the time the "
            f"job has left cannot be counted from it.") from None


# What a run took from which upstream run, one ``<file> <- <run>`` line
# each -- a run's own file, or a point's: written by launch's one
# hand-over (`materialize.hand_over`, :func:`write_gathered_from`) and read
# by status, the transport record and the run record's provenance
# (:func:`read_gathered_from`, `model/parse.md` § 5d.4).


def write_gathered_from(attempt_dir, gathered: "List[Tuple[RunRef, str]]",
                        *, plan=None) -> None:
    """THE ONE WRITER of what a run gathered -- a hand-over's record beside
    the run (`materialize.hand_over`): ``gathered``, ``[(the run it came
    from, filename), ...]`` in the order taken, as the run's
    ``.gathered-from``, through `persist`, or into ``plan``
    (`jobset.planned.Plan`)."""
    data = "".join(f"{fn} <- {src}\n" for src, fn in gathered).encode("utf-8")
    if plan is not None:
        plan.bytes(Path(attempt_dir) / _rf.GATHERED_FROM_FILE, data)
        return
    _persist().write_bytes(Path(attempt_dir) / _rf.GATHERED_FROM_FILE, data)


def read_gathered_from(attempt_dir) -> "List[Tuple[RunRef, str]]":
    """``[(the run it came from, filename), ...]`` -- what this run was
    gathered from, in the order it was taken, as :func:`write_gathered_from`
    wrote it; ``[]`` when it gathered nothing.  A record that is there and
    does not read is a :class:`LaunchRecordError` naming it."""
    p = Path(attempt_dir) / _rf.GATHERED_FROM_FILE
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        raise LaunchRecordError(
            f"{p} does not read ({exc}) -- what this run gathered cannot be "
            f"told.") from None
    out = []
    for line in text.splitlines():
        name, sep, src = line.partition(" <- ")
        if sep and name.strip() and src.strip():
            out.append((RunRef.read(src), name.strip()))
    return out


def carried_gather(base, source) -> "List[Tuple[RunRef, str]]":
    """WHAT A RUN CARRIES -- ``[(the run it came from, filename), ...]``:
    ``source``'s own `.gathered-from` (:func:`read_gathered_from`), then, for
    each input it kept from the run before it (warm, so its record does not
    list it), what that run gathered, one hop at a time through
    `.continued-from` (:func:`read_continued_from`), each run named from
    ``base``.  A device launched warm still carries the seed's density its
    first launch gathered (`job-system.md` § 5.4).  A point of a run of
    points carries its own record alone.  The one answer the launch takes
    again (`submit._carry_the_gather`) and a reuse compares a finished run
    by (`jobset.reuse.takes_over`)."""
    from .paths import attempt_index
    base = Path(base)
    took = list(read_gathered_from(source))
    have = {fn for _src, fn in took}
    seen = {Path(source)}
    here = Path(source)
    while True:
        if attempt_index(here.name) is None:
            break
        prev = read_continued_from(here)
        if prev is None:
            break
        here = prev.path(base)
        if here in seen:
            break
        seen.add(here)
        for src, fn in read_gathered_from(here):
            if fn not in have:
                took.append((src, fn))
                have.add(fn)
    return took


# The finished calculation's run a run is a copy of, tree-relative -- one
# line, written by the prep that copies a group in from the finished
# calculation a transport calculation reuses (`jobset.reuse.copy_in`,
# :func:`write_copied_from`) and read by the readers that say *copied from*
# and by the way back (:func:`read_copied_from`; `engines/transport.md`
# § 2a.16).


def write_copied_from(run_dir, source: str, *, plan) -> None:
    """THE ONE WRITER of what a run is a copy of: ``source``, the finished
    calculation's run, tree-relative, as the run's ``.copied-from``, into
    ``plan`` (`jobset.planned.Plan`)."""
    plan.bytes(Path(run_dir) / _rf.COPIED_FROM_FILE,
               (source + "\n").encode("utf-8"))


def read_copied_from(run_dir) -> Optional[str]:
    """The finished calculation's run ``run_dir`` is a copy of, tree-relative,
    as :func:`write_copied_from` wrote it -- ``None`` for a run made here.
    A record that is there and does not read is a :class:`LaunchRecordError`
    naming it."""
    p = Path(run_dir) / _rf.COPIED_FROM_FILE
    try:
        said = p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise LaunchRecordError(
            f"{p} does not read ({exc}) -- which run this one is a copy of "
            f"cannot be told.") from None
    if not said:
        raise LaunchRecordError(
            f"{p} names no run -- which run this one is a copy of cannot be "
            f"told.")
    return said


def copied_of(base, folder) -> Optional[str]:
    """WHAT ``folder`` IS A COPY OF -- a run, or a point of a run of points,
    of the calculation in ``base`` -- tree-relative: its own
    ``.copied-from`` (a point taken over from the finished calculation its
    calculation reuses), or its run's with the point's place in it (a run
    copied in whole), one hop; ``None`` for one computed in this
    calculation (`engines/transport.md` § 2a.16).  The one answer every
    reader that says *copied from* asks: `status`, the Results tab's ladder
    and the transport record."""
    from .paths import run_folder_of
    base, folder = Path(base).resolve(), Path(folder).resolve()
    said = read_copied_from(folder)
    if said:
        return said
    rel = folder.relative_to(base)
    run = run_folder_of(rel)
    if run is not None and run != rel:
        said = read_copied_from(base / run)
        if said:
            return f"{said}/{rel.relative_to(run).as_posix()}"
    return None


def copied_in(stage_dir) -> Optional[str]:
    """What a stage's FIRST run is a copy of, tree-relative -- its
    ``.copied-from`` (:func:`read_copied_from`) -- or ``None`` for a stage
    whose runs were made here: a stage its calculation's prep COPIED IN from
    the finished calculation it reuses (`engines/transport.md` § 2a.16),
    asked by the way back -- the state saved before that prep."""
    from .paths import attempt_dir, attempts_in
    if stage_dir is None:
        return None
    runs = attempts_in(stage_dir)
    return read_copied_from(attempt_dir(stage_dir, runs[0])) if runs else None


__all__ = ["RUN_LAUNCH_SCHEMA", "LaunchRecordError", "RunRef", "newest_run",
           "newest_output",
           "continued_from_marker", "write_continued_from", "read_continued_from", "read_concluded", "Ending", "ending", "launch_record_path", "write_launch", "launch_record", "sent_wall_s", "write_gathered_from", "read_gathered_from", "carried_gather", "write_copied_from", "read_copied_from", "copied_of", "copied_in"]
