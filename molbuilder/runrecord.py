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
    read or written -- from the package, or beside a job, where a flat run's
    warm retry writes its own launch record (:func:`record_retry`), so it
    travels too (`runwrap.MONITOR_COMPANIONS`)."""
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
    """A launch record that is there and does not read -- neither JSON, nor
    a ``molbuilder/run-launch`` record.  Never *launched* or *not
    launched*: whether the run was launched cannot be told from it, so
    every reader stops and names the file (`execution/architecture.md`
    § 3.2)."""


@dataclass(frozen=True)
class RunRef:
    """A RUN OF THIS CALCULATION, AS A RECORD NAMES IT -- the one spelling
    `.continued-from`, `.gathered-from` and `run.json`'s ``continued_from``
    hold (`job-system.md` § 5.4): the run's ``folder`` from the
    calculation's (``01_coarse/run-0``; a point's ``04_device/run-0/v0.2``)
    -- or, where a stage's runs share the calculation's folder, the run's
    own ``name``, which every file of it carries (``H2_01_coarse-run1``),
    its folder the calculation's own (``.``).  Written and read one way:
    ``str(ref)``, and :meth:`read` with the layout the record lies in."""
    folder: PurePosixPath = PurePosixPath(".")
    name: Optional[str] = None

    def __str__(self) -> str:
        return self.name if self.name is not None else self.folder.as_posix()

    @classmethod
    def at(cls, base, folder) -> "RunRef":
        """The run in ``folder``, named from the calculation's ``base``."""
        return cls(PurePosixPath(Path(folder).relative_to(base).as_posix()))

    @classmethod
    def named(cls, name: str) -> "RunRef":
        """A run of a stage whose runs share the calculation's folder, by
        its own name (`runfiles.RunNames.run_name`)."""
        return cls(name=name)

    @classmethod
    def read(cls, text: str, *, shared: bool) -> "RunRef":
        """``text`` as a record holds it -- a run's name where the record's
        runs share the folder (``shared``), else a folder."""
        text = text.strip()
        return cls(name=text) if shared else cls(PurePosixPath(text))

    def path(self, base) -> Path:
        """The folder its files lie in."""
        return Path(base) / self.folder

    def within(self, other: "RunRef") -> bool:
        """It lies inside ``other``'s folder -- a point of that run."""
        return (self.name is None and other.name is None
                and other.folder in self.folder.parents)

    def run_folder(self) -> "RunRef":
        """The run it lies in -- itself, or for a point of a run of points
        the run folder above it (`paths.run_folder_of`, the one reading of
        which run a point belongs to)."""
        if self.name is not None:
            return self
        from .paths import run_folder_of      # inside molbuilder only
        got = run_folder_of(self.folder)
        return self if got is None else RunRef(PurePosixPath(got.as_posix()))


def continued_from_marker(where: Path, names: "_rf.RunNames",
                          run: int) -> Path:
    """Where the run a run continues from is recorded (`project-layout.md`
    § 1.6.3) -- named by the stage's names (`runfiles.RunNames`): an
    attempt's ``.continued-from``, or, where a stage's runs share a folder,
    run ``run``'s own ``<stem>-run<N>.continued-from``, named for the run
    that continues (user, 2026-10-01: the flat layout records it too; each
    run its own, plan W57 decision 2)."""
    return Path(where) / names.name(".continued-from", run)


def write_continued_from(where, source: RunRef, *, names: "_rf.RunNames",
                         run: int, plan=None) -> Path:
    """THE ONE WRITER of the run a run continues from -- a hand-over's
    record beside the run (`materialize.hand_over`, `job-system.md` § 5.4):
    ``source`` as run ``run``'s :func:`continued_from_marker`, through
    `persist`, or into ``plan`` (`jobset.planned.Plan`)."""
    marker = continued_from_marker(where, names, run)
    data = (str(source) + "\n").encode("utf-8")
    if plan is not None:
        plan.bytes(marker, data)
        return marker
    _persist().write_bytes(marker, data)
    return marker


def read_continued_from(where, names: "_rf.RunNames",
                        run: int) -> Optional[RunRef]:
    """The run :func:`write_continued_from` recorded for run ``run``, or
    ``None`` when there is no marker -- the run started from the structure,
    or from what it gathered.  A marker that is there and does not read is
    a :class:`LaunchRecordError` naming it."""
    marker = continued_from_marker(where, names, run)
    try:
        text = marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise LaunchRecordError(
            f"{marker} does not read ({exc}) -- what this run continues "
            f"from cannot be told.") from None
    return RunRef.read(text, shared=names.shared) if text.strip() else None


def read_concluded(text: Optional[str]) -> Optional[dict]:
    """The conclusion marker's first line, ``rc=<N> at <when>[; <note>]`` as
    the wrapper writes it (`runwrap.py`), as ``{"code": N, "at": when,
    "note": note}`` -- ``at`` and ``note`` only when stated -- or ``None``
    when the text is not one.  The note is what the wrapper adds when the job's finish
    failed or cannot run (`parse.dirs.job.FINISH_FAILED`,
    `FINISH_CANNOT_LOAD`).  THE one reader of that line.
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
    -- the run's own, in either shape: an attempt's folder may hold several
    run indexes, and a flat calculation's every stage.  ``run`` asks about
    that run index; by default the newest (`runfiles.at_run`).

    **Molbuilder's own marker first** -- ``<basename>-run<N>.concluded``,
    the wrapper's last act on its main path: an engine error still reaches
    it, a kill never does, and it carries the exit code.  It counts at the
    newest run index any file of the run reached (`runfiles.at_run`):
    an earlier index's marker beside a newer run is that run's goodbye.

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


def launch_record_path(where: Path, names: "_rf.RunNames",
                       run: int) -> Path:
    """Where run ``run``'s launch is recorded (`project-layout.md` § 1.6.3),
    named by its stage's names (`runfiles.RunNames`): the run's folder's
    ``run.json`` -- an attempt's, a trial's -- or, where a stage's runs
    share a folder, the run's own ``<stem>-run<N>.run.json`` beside every
    other file of it (plan W57 decisions 2 and 6)."""
    return Path(where) / names.name(".run.json", run)


def _launched_runs(where: Path, names: "_rf.RunNames") -> List[int]:
    """The numbers of a shared folder's launched runs of this stage -- one
    per launch record, read back through the grammar (`runfiles.parse`,
    with the stage's stem), lowest first."""
    out = []
    for f in _rf.find_by_role(Path(where), ".run.json"):
        rec = _rf.parse(f.name, names.stem)
        if rec is not None and rec.stage is None and rec.run is not None:
            out.append(rec.run)
    return sorted(out)


def newest_run(where, names: "_rf.RunNames") -> Optional[int]:
    """THE STAGE'S NEWEST RUN IN ``where`` -- the one rule every reader of
    a run asks (`model/parse.md` § 5.1, `project-layout.md` § 1.6.1): where
    a stage's runs share the folder, its newest LAUNCHED run, by its launch
    records -- a warm retry's included -- ``None`` before its first launch,
    since prep seeds its first run's progress log before it runs; in a
    folder of the run's own, its newest TRY, the highest index any file of
    it reached (`runfiles.latest_run`) -- the folder's one launch record
    answers for every try in it."""
    if names.shared:
        runs = _launched_runs(Path(where), names)
        return runs[-1] if runs else None
    return _rf.latest_run(Path(where), names.label, stage=names.stage)


def last_run(where, label: str) -> Optional["RunRef"]:
    """THE LAST RUN IN A FOLDER A CALCULATION'S STAGES SHARE -- the run whose
    restart files and results lie there, since every run of every stage
    writes the same ones (`job-system.md`, *What molbuilder does for you*,
    2): what ``<label>.last-run`` names, the record its run script wrote the
    moment its engine started (user, 2026-10-12: *"there should be a record
    manifest of the last run"*) -- never worked out from when the launch
    records were written.  ``None`` while no run has started there.  A
    record that does not read, or names no run of ``label``, is a
    :class:`LaunchRecordError`, never passed over."""
    p = Path(where) / _rf.compose(label, ".last-run")
    try:
        text = p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise LaunchRecordError(
            f"{p} does not read ({exc}) -- which run last wrote this folder "
            f"cannot be told.") from None
    if _rf.parse_run_name(text, label) is None:
        raise LaunchRecordError(
            f"{p} names {text!r}, which is no run of {label!r} -- which run "
            f"last wrote this folder cannot be told.")
    return RunRef.named(text)


def next_run(where, names: "_rf.RunNames") -> int:
    """THE NUMBER A NEW RUN TAKES (`project-layout.md` § 1.6.1) -- decided
    by `launch`, which hands it to the run script (`--run N`): where a
    stage's runs share a folder, one past its newest run (:func:`newest_run`,
    counted on the launch records), else 0; in a folder of the run's own,
    0 -- it is launched once."""
    if not names.shared:
        return _rf.FIRST_ATTEMPT
    newest = newest_run(where, names)
    return newest + 1 if newest is not None else _rf.FIRST_ATTEMPT


def write_launch(where: Path, *, names: "_rf.RunNames", run: int,
                 mode: str, command: List[str],
                 job_id: Optional[str] = None,
                 continued_from: Optional[RunRef] = None,
                 launched_at: Optional[str] = None,
                 placed_on: Optional[dict] = None,
                 retry_of: Optional[int] = None,
                 scheduler: Optional[dict] = None) -> Path:
    """Record run ``run``'s launch where its stage's names put it
    (:func:`launch_record_path`) — ``molbuilder/run-launch@1`` -- through
    `persist`, so it is whole or absent, never half a file
    (`execution/architecture.md` § 2.1: a run's own records are read and
    written through `persist`).

    Written **after** the launch succeeds, so a failed launch leaves the
    attempt exactly as prepare left it (§ 1.6). ``continued_from`` is the run's provenance — *this geometry came
    from ``01_coarse/run-0``* — which is worth recording whether or not
    anything reads it back.

    ``placed_on`` is WHERE IT WAS SENT: the domain name, its partition and
    qos (`scheduler.md` R12 -- the queue half; what it LANDED ON is the
    monitor's ``[MACHINE]`` line, written on the node) -- and the wall and
    memory it was sent with: prep's, the wall the launch's ``--time`` when
    it restates it (`job-system.md` § 6.0).

    ``null`` for a run here, which no queue placed -- written, never left
    out: a reader tells "no queue" from "not recorded" by the key.

    ``retry_of`` is the run a flat run's warm retry retries
    (:func:`record_retry`), ``null`` for a launch (plan W57 decision 6).

    ``scheduler`` is the scheduler the job was sent to and where it writes
    the job's error file, from the run's own folder -- ``{kind, stderr}``,
    ``stderr`` null where the job's id was not read -- the file the run
    door reads for how the scheduler ended the job (`scheduler.md` R16);
    ``null`` for a run here.
    """
    from datetime import datetime, timezone
    p = launch_record_path(where, names, run)
    body = {
        "schema": RUN_LAUNCH_SCHEMA,
        "mode": mode,
        "command": list(command),
        "job_id": job_id,
        "launched_at": launched_at or datetime.now(timezone.utc)
                                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # EVERY KEY, EVERY TIME, its null stated (`checkpointing.md` S3,
    # `job-contracts.md` § 6.1): ``continued_from`` null -- it started from
    # the structure; ``placed_on`` null -- a run here, which no queue
    # placed; ``retry_of`` null -- a launch; ``scheduler`` null -- a run
    # here, sent to no scheduler.
    body["continued_from"] = (str(continued_from) if continued_from
                              else None)
    body["placed_on"] = dict(placed_on) if placed_on else None
    body["retry_of"] = retry_of
    body["scheduler"] = dict(scheduler) if scheduler else None
    return _persist().write_json(p, body)


def record_retry(where, names: "_rf.RunNames", retried: int) -> Path:
    """A WARM RETRY, RECORDED where its stage's runs share a folder -- its
    own launch record, written by the run script as it re-runs itself,
    through the monitor's bundle (`mb_monitor.pyz retried`;
    `running-a-job.md` § 3.5): the run after ``retried``, the same job --
    its launch's mode, command, job id and queue -- with its own start,
    ``continued_from`` the run it retries, and ``retry_of`` that run's
    number (plan W57 decision 6).  In a folder of the run's own the
    folder's record answers for every run in it, and nothing is asked of
    this.  The run retried must have its record: a run with none is one no
    reader could place."""
    if not names.numbered(".run.json"):
        raise LaunchRecordError(
            f"{names.template('.run.json')} answers for every run in its "
            f"folder; a retry there records nothing of its own.")
    said = launch_record(where, names, retried)
    if said is None:
        raise LaunchRecordError(
            f"{launch_record_path(where, names, retried)} is not there: run "
            f"{retried} has no launch record, so its retry cannot be "
            f"recorded as the same job.")
    return write_launch(
        Path(where), names=names, run=retried + 1, mode=said["mode"],
        command=said["command"], job_id=said["job_id"],
        continued_from=RunRef.named(names.run_name(retried)),
        placed_on=said["placed_on"],
        retry_of=retried, scheduler=said.get("scheduler"))


def retried_to(where, names: "_rf.RunNames", run: int) -> int:
    """The run a launch of run ``run`` has reached through its warm
    retries, where a stage's runs share a folder -- each retry its own run,
    its record's ``retry_of`` the run before (:func:`record_retry`) -- or
    ``run`` itself, retried never or in a folder of its own, where the
    folder's newest try is the run's (`running-a-job.md` § 3.5)."""
    if not names.numbered(".run.json"):
        return run
    while True:
        try:
            nxt = launch_record(where, names, run + 1)
        except LaunchRecordError:
            return run
        if nxt is None or nxt.get("retry_of") != run:
            return run
        run += 1


def _refuse_unnumbered(where: Path, names: "_rf.RunNames") -> None:
    """A shared folder holding a stage's run files with no run's number --
    the names every row the catalogue numbers there (``attempt="shared"``)
    had before each run carried its own (plan W57 decision 2) -- was written
    by an older molbuilder: refused, naming the verb that numbers them,
    never read as another run's, or as none."""
    old = [f.name for f in (Path(where) / names.unnumbered(role)
                            for role in _rf.shared_numbered_roles())
           if f.is_file()]
    if old:
        # A LAYER BELOW `jobset` prints its line by the composer's rules
        # (`job-system.md` § 5.3): the folder quoted as a shell needs it.
        import shlex
        raise LaunchRecordError(
            f"{', '.join(old)}: a flat stage's run files carry their run's "
            f"number (`<stem>-run<N>`), and these carry none -- which run "
            f"each is cannot be told from the name.  Number them for the "
            f"stage's newest run with:\n"
            f"    molbuilder jobset migrate --bundle "
            f"{shlex.quote(str(Path(where)))}")


def launch_record(where, names: "_rf.RunNames",
                  run: Optional[int] = None) -> Optional[dict]:
    """THE door for *was this run launched?* (`execution/architecture.md`
    § 3.2; plan W38 F2) -- run ``run``'s launch record, or ``None`` when
    there is none: asked by status, the Run panel, prep (an unlaunched
    attempt is reused), every launch gate and the transport citation.

    Named by the stage's names (`runfiles.RunNames`): a folder of the run's
    own -- an attempt, a trial -- has one record, which answers for every
    run in it; where a stage's runs share a folder each run has its own,
    and ``run`` None asks for the stage's newest launched run's.  A shared folder's run files written before each carried
    its run's number are refused, naming `jobset migrate`.

    `project-layout.md` § 1.6: *"Has this been launched? has no honest answer
    from the directory alone"*, so this file is the answer -- and **one that
    does not read** is :class:`LaunchRecordError`, naming the file, never
    an answer.
    """
    if where is None:
        return None
    where = Path(where)
    if names.numbered(".run.json"):
        _refuse_unnumbered(where, names)
        if run is None:
            run = newest_run(where, names)
            if run is None:
                return None
    p = launch_record_path(where, names, run)
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
    ``placed_on.time``, in the record's one spelling ``D-HH:MM:SS``
    (`scheduler.quantities.slurm_time`, which wrote it) -- or ``None``
    where it states none: a run here, which no queue placed.  A wall that
    is there and does not read in that spelling is a
    :class:`LaunchRecordError`, never read as no wall."""
    placed = (record or {}).get("placed_on") or {}
    wall = placed.get("time")
    if wall in (None, ""):
        return None
    days, _dash, clock = str(wall).partition("-")
    parts = clock.split(":")
    if not _dash or len(parts) != 3 or not all(
            x.isdigit() for x in [days] + parts):
        raise LaunchRecordError(
            f"the launch record's wall, {wall!r}, is not D-HH:MM:SS -- the "
            f"spelling its writer uses -- so the time the job has left "
            f"cannot be counted from it.")
    h, m, s = (int(x) for x in parts)
    return int(days) * 86400 + h * 3600 + m * 60 + s


def retried_main(argv) -> int:
    """``retried --label L --stage S [--shared] --run N [--dir D]`` -- the
    command line of :func:`record_retry`, which the run script asks through
    the monitor's bundle before a warm retry re-runs it as run N+1
    (`running-a-job.md` § 3.5): the stage's names as the script was rendered
    with them (`runfiles.RunNames`), and the run retried.  Exit 0 when the
    record is written; 1, the reason on stderr, when it is not -- and the
    script then concludes un-retried."""
    import argparse
    import sys
    p = argparse.ArgumentParser(
        prog="mb_monitor retried",
        description="record a warm retry's own launch record "
                    "(running-a-job.md 3.5)")
    p.add_argument("--label", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--shared", action="store_true",
                   help="the stage's runs share this folder")
    p.add_argument("--run", type=int, required=True,
                   help="the run being retried")
    p.add_argument("--dir", default=".", dest="directory")
    a = p.parse_args(argv)
    try:
        record_retry(a.directory, _rf.RunNames(a.label, a.stage, a.shared),
                     a.run)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"retried: {exc}\n")
        return 1
    return 0


#: What a run took from which upstream run, one ``<file> <- <run>`` line
#: each -- a run's own file, or a point's: written by launch's one
#: hand-over (`materialize.hand_over`, :func:`write_gathered_from`) and read
#: by status, the transport record and the run record's provenance
#: (:func:`read_gathered_from`, `model/parse.md` § 5d.4).
GATHERED_FROM_FILE = _rf.GATHERED_FROM_FILE   # the catalogue's name


def write_gathered_from(attempt_dir, gathered: "List[Tuple[RunRef, str]]",
                        *, plan=None) -> None:
    """THE ONE WRITER of what a run gathered -- a hand-over's record beside
    the run (`materialize.hand_over`): ``gathered``, ``[(the run it came
    from, filename), ...]`` in the order taken, as the run's
    ``.gathered-from``, through `persist`, or into ``plan``
    (`jobset.planned.Plan`)."""
    data = "".join(f"{fn} <- {src}\n" for src, fn in gathered).encode("utf-8")
    if plan is not None:
        plan.bytes(Path(attempt_dir) / GATHERED_FROM_FILE, data)
        return
    _persist().write_bytes(Path(attempt_dir) / GATHERED_FROM_FILE, data)


def read_gathered_from(attempt_dir) -> "List[Tuple[RunRef, str]]":
    """``[(the run it came from, filename), ...]`` -- what this run was
    gathered from, in the order it was taken, as :func:`write_gathered_from`
    wrote it; ``[]`` when it gathered nothing.  A record that is there and
    does not read is a :class:`LaunchRecordError` naming it.  A gather is
    made on the hierarchical layout alone, so each run is named by its
    folder."""
    p = Path(attempt_dir) / GATHERED_FROM_FILE
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
            out.append((RunRef.read(src, shared=False), name.strip()))
    return out


__all__ = ["RUN_LAUNCH_SCHEMA", "LaunchRecordError", "RunRef", "newest_run", "last_run", "continued_from_marker", "write_continued_from", "read_continued_from", "read_concluded", "Ending", "ending", "launch_record_path", "next_run", "write_launch", "record_retry", "retried_to", "retried_main", "launch_record", "sent_wall_s", "GATHERED_FROM_FILE", "write_gathered_from", "read_gathered_from"]
