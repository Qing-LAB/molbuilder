"""THE JOBS LAUNCHED FROM THIS MACHINE -- each launch's address, kept the way
a server keeps its pidfile (`execution/job-system.md` § 6.0a): one file per
submission in molbuilder's state folder, written by the one sender the
moment it goes (`jobset.submit._go`), read by `molbuilder jobs` -- each read
by what it says, never by its file's name, so two clusters sharing a home
keep their jobs of one number apart (:func:`_file_of`, :func:`find`).

**An address, never a state.**  What a job is doing is read fresh each time
-- its process, the scheduler, its runs' own files -- so nothing here can go
stale against them.  Floor 1 (`execution/architecture.md` § 2.1): plain
facts of this machine -- and whether a job is going, its process or its
scheduler's list, the scheduler named by its calculation's copy of its
machine's record (:func:`is_going`), the one answer `molbuilder jobs` and
`jobset scratch` ask.
"""
from __future__ import annotations

import os
import signal
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .runrecord import stamp

#: Every job's file, at one major (`persist.check_schema`).
JOB_SCHEMA = "molbuilder/job@1"

#: Where a process's start is read: Linux's own account of its processes.
_PROC = Path("/proc")

#: THE JOB A READER RUNS INSIDE -- a group's launch inside a sequence of
#: groups' job (`job-system.md` § 6.0): ``(the job's address, the
#: sequence's name)``.  A run of points that job sent -- an earlier group's,
#: or its own group's after its walk returned -- is read by its own files,
#: never by asking that job, which is the reader's own and still going.
#: Set by :func:`inside`, scoped.
_INSIDE: "ContextVar[Optional[Tuple[str, str]]]" = ContextVar(
    "molbuilder_inside_job", default=None)


@contextmanager
def inside(job_id: Optional[str], sequence: Optional[str] = None):
    """Read, within this block, as a reader running inside job ``job_id``,
    the sequence ``sequence``'s (``None``: inside none) -- a group's launch
    inside a sequence's job, from its plan to what it prints last."""
    token = _INSIDE.set((str(job_id), str(sequence))
                        if job_id is not None else None)
    try:
        yield
    finally:
        _INSIDE.reset(token)


def inside_job() -> Optional[Tuple[str, str]]:
    """``(the job's address, the sequence's name)`` the reader runs inside
    (:func:`inside`), or ``None``."""
    return _INSIDE.get()


class JobFileError(ValueError):
    """A job's file that is there and does not read -- said with its path,
    and never taken for a job that has ended."""


class NotGoing(ValueError):
    """A run here that is no longer the process launched -- it ended, or
    its pid is another process's now -- so nothing is signalled."""


@dataclass(frozen=True)
class Job:
    """One submission's address (`job-system.md` § 6.0a, the table)."""
    #: What `molbuilder jobs stop` names: ``<host>-<pid>`` for a run here, a
    #: queue job's own id.
    id: str
    #: ``here`` or ``queue``.
    how: str
    #: The node it was launched on: a pid means something only there.
    host: str
    #: The calculation's folder.
    calculation: str
    #: Each member's stage -> ``{"run", "launch_record"}``: its run as a
    #: record names it (`runrecord.RunRef`), and its launch record's path
    #: from the calculation's folder -- what its state is asked through.
    members: Dict[str, Dict[str, str]]
    launched_at: str
    #: A run here: left in the background, or waited on.
    background: bool = False
    #: A run here: its process, and when the kernel started it
    #: (:func:`started_of`).
    pid: Optional[int] = None
    started: Optional[str] = None
    #: A queue job: the scheduler's id, and the cluster it was sent to as
    #: the scheduler names it -- ``None`` when it could not be read.
    job_id: Optional[str] = None
    cluster: Optional[str] = None
    #: A SEQUENCE OF GROUPS (`job-system.md` § 6.0): its name and each group
    #: it launches, in order -- ``{"name", "groups": [{"number",
    #: "stages"}]}`` -- its ``members`` added by each group's launch inside
    #: it as it sends its runs; ``None`` for every other job.
    sequence: Optional[dict] = None
    #: The file it was read from -- not written.
    path: Optional[Path] = field(default=None, compare=False)

    def to_dict(self) -> dict:
        return {"schema": JOB_SCHEMA, "id": self.id, "how": self.how,
                "host": self.host, "calculation": self.calculation,
                "members": {k: dict(v) for k, v in self.members.items()},
                "sequence": (dict(self.sequence) if self.sequence is not None
                             else None),
                "launched_at": self.launched_at,
                **({"background": self.background, "pid": self.pid,
                    "started": self.started} if self.how == "here" else
                   {"job_id": self.job_id, "cluster": self.cluster})}


def this_host() -> str:
    """This node's name, as a job's file records where it was launched."""
    return socket.gethostname()


def started_of(pid: int) -> Optional[str]:
    """When the kernel started process ``pid``, with the boot it belongs to
    -- ``"<boot id>:<clock ticks since the boot>"`` -- or ``None`` when no
    such process is going: what tells a run from a later process given the
    same pid (``/proc/<pid>/stat``, its 22nd field; the command name before
    it may hold spaces, so the split is at its last ``)``)."""
    try:
        boot = (_PROC / "sys/kernel/random/boot_id").read_text().strip()
        raw = (_PROC / str(pid) / "stat").read_text(errors="replace")
    except OSError:
        return None
    fields = raw[raw.rfind(")") + 2:].split()
    # AN ENDED PROCESS ITS PARENT HAS NOT YET REAPED (state ``Z``) keeps its
    # start: it is no process going.
    if len(fields) <= 19 or fields[0] == "Z":
        return None
    return f"{boot}:{fields[19]}"


def going_here(job: Job) -> Optional[bool]:
    """Whether a run here is still going -- the same process, by its pid and
    its start -- or ``None`` where this machine cannot tell: a queue job, a
    run launched on another host, or a machine with no ``/proc``."""
    if (job.how != "here" or job.host != this_host()
            or not (_PROC / "self" / "stat").is_file()):
        return None
    # A START NOT READ when the file was written, ``/proc`` there, is a
    # process gone by then -- never one that matches every later absence.
    return (job.pid is not None and job.started is not None
            and started_of(job.pid) == job.started)


def signal_here(job: Job) -> None:
    """SIGTERM to the run's process group -- a run left in the background
    leads its own, so the signal reaches its engine (`job-system.md`
    § 6.0) -- once it is checked, now, to be the process launched;
    :class:`NotGoing` when it is not."""
    if not going_here(job):
        raise NotGoing(f"{job.id} has ended")
    try:
        os.killpg(job.pid, signal.SIGTERM)
    except ProcessLookupError:
        raise NotGoing(f"{job.id} has ended") from None


def _dir() -> Path:
    from .config_dir import jobs_dir
    return jobs_dir()


def _file_of(job: "Job") -> Path:
    """Where a job's address is written first: a run here by its id
    (``<host>-<pid>``); a queue job by the cluster it was sent to and its
    number -- a home folder two clusters share holds both clusters' jobs of
    one number (the user, 2026-10-22) -- by its number alone where its
    cluster was not read.  Read back by what it says (:func:`find`), and
    written again where it was read from."""
    name = (f"{job.cluster}-{job.job_id}" if job.how == "queue"
            and job.cluster else job.id)
    return _dir() / f"{name}.json"


#: THE CLUSTER EACH SCHEDULER HERE ANSWERS FOR, once it has said: a
#: scheduler's name for its cluster does not change while a process runs.
_CLUSTERS: Dict[Any, str] = {}


class Asked:
    """The cluster a scheduler here answers for, as it names it -- asked
    once a command for each scheduler in the process, and only when a queue
    job is asked of; an answer it could not give is asked again by the next
    look."""
    def __init__(self):
        self._asked = {}

    def cluster(self, scheduler):
        if scheduler not in self._asked:
            got = _CLUSTERS.get(scheduler)
            if got is None:
                from .scheduler.queue import cluster
                got = cluster(scheduler)
                if got is not None:
                    _CLUSTERS[scheduler] = got
            self._asked[scheduler] = got
        return self._asked[scheduler]


def scheduler_of(job):
    """The scheduler a queue job was sent to: the one its calculation's copy
    of its machine's record names (`configuration.md` M-3) -- ``None`` when
    that copy does not read here (`record.calculation_environment`)."""
    from .scheduler.record import UnknownTarget, calculation_environment
    try:
        rec = calculation_environment(job.calculation)
    except UnknownTarget:
        return None
    return rec.scheduler if rec is not None else None


def address_of(job_id, calculation) -> Optional[Job]:
    """THE ADDRESS of queue job ``job_id`` this machine sent for the
    calculation at ``calculation`` -- its file on the jobs list, read by
    what it says -- or ``None``: none on file (sent from another machine, or
    its end said and its address removed), or another calculation's job of
    that number.  Two clusters' jobs of one number for one calculation: the
    one this machine's scheduler answers for (:func:`asked_here`)."""
    calc = Path(calculation).resolve()
    hits = [j for j in every() if isinstance(j, Job) and j.how == "queue"
            and j.job_id == str(job_id) and Path(j.calculation) == calc]
    if len(hits) < 2:
        return hits[0] if hits else None
    here = Asked()
    return next((j for j in hits if asked_here(j, here)[0] is not None),
                hits[0])


def asked_here(job: "Job", here: "Asked") -> "Tuple[Optional[Any], str]":
    """``(scheduler, "")`` when queue job ``job`` can be asked about on this
    machine -- its calculation's copy of its machine's record names the
    scheduler, and that scheduler answers for the cluster the job was sent
    to, as the scheduler names it -- else ``(None, why not)``: THE ONE RULE
    `molbuilder jobs` and the run door ask by (`job-system.md` § 6.0a;
    `running-a-job.md` § 4.2)."""
    scheduler = scheduler_of(job)
    if scheduler is None:
        from .scheduler.record import UnknownTarget, calculation_record
        if calculation_record(job.calculation).is_file():
            # THERE AND NOT READ, said as the record's own refusal says it
            # -- an older molbuilder's copy naming `jobset migrate`.
            return None, " ".join(
                str(UnknownTarget.unreadable_copy(job.calculation)).split())
        return None, ("its calculation's record of its machine is not here, "
                      "so its scheduler cannot be asked")
    if job.cluster is None:
        # WHERE IT WENT IS NOT KNOWN: another cluster sharing this home
        # would answer for an id that is not its own.
        return None, ("the cluster it was sent to was not read when it "
                      "went, so its scheduler is not asked")
    answers = here.cluster(scheduler)
    if answers != job.cluster:
        return None, (f"sent to cluster {job.cluster}: ask there"
                      if answers is not None else
                      "the scheduler cannot be asked here")
    return scheduler, ""


#: The scheduler's answers of the last ten seconds, by query and job -- one
#: look at a page, a run of points or a job's members asks for one job many
#: times (`running-a-job.md` § 4.2).
_ASKED_LATELY: Dict[Tuple[str, Optional[str], str], Tuple[float, Any]] = {}
_ASKED_TTL_S = 10.0


def asked_lately(job: "Job", what: str) -> Optional[Any]:
    """The scheduler's answer to query ``what`` -- ``listed`` or
    ``standing`` (`scheduler.queue`) -- about queue job ``job``, where this
    machine can ask it (:func:`asked_here`), asked at most once per
    :data:`_ASKED_TTL_S` per job in this process: `molbuilder jobs`, the
    run door and a queued run's standing hear one answer.  ``None`` where
    it cannot be asked."""
    import time
    from .scheduler import queue
    key = (what, job.cluster, str(job.job_id))
    now = time.monotonic()
    kept = _ASKED_LATELY.get(key)
    if kept is not None and now - kept[0] < _ASKED_TTL_S:
        return kept[1]
    scheduler, _why = asked_here(job, Asked())
    got = (getattr(queue, what)(scheduler, str(job.job_id))
           if scheduler is not None else None)
    _ASKED_LATELY[key] = (now, got)
    return got


def is_going(job: "Job", here: "Asked") -> "Tuple[Optional[bool], str]":
    """``(going, said)`` -- whether the job is still going, its process or
    the scheduler's list, ``None`` where this machine cannot ask
    (:func:`asked_here`) -- and that in words: THE ONE ANSWER, `molbuilder
    jobs` asks it of each job, `jobset scratch` of the jobs that hold its
    runs."""
    if job.how == "here":
        going = going_here(job)
        return going, {True: "its process is going",
                       False: "its process has ended",
                       None: (f"on {job.host}: ask there"
                              if job.host != this_host() else
                              "this machine cannot tell")}[going]
    scheduler, why = asked_here(job, here)
    if scheduler is None:
        return None, why
    # THE SCHEDULER'S OWN WORDS for a job it holds -- its state and reason,
    # so a job it took and will never start says why (`job-system.md`
    # § 6.0a; `scheduler.md` R6) -- the answer the run door hears too.
    answer = asked_lately(job, "listed")
    if answer is None:
        return None, "the scheduler cannot be asked here"
    going, said = answer
    if going:
        return True, "the scheduler lists it" + (f": {said}" if said else "")
    # ENDED, in its words while it still lists the job -- ``FAILED
    # (NonZeroExitCode)`` -- and only then forgotten.
    return False, (f"the scheduler lists it ended: {said}" if said
                   else "the scheduler no longer lists it")


#: A RUN HERE'S ID -- its node and its process: the one composition, for
#: its address (:func:`run_here`) and for a walk that types it where its
#: shell knows the two (`jobset.submit`, a sequence of groups' ``--in-job``;
#: `job-system.md` § 6.0).
HERE_ID = "{host}-{pid}"


def here_id(host, pid) -> str:
    """A run here's id (:data:`HERE_ID`) -- ``host`` and ``pid`` as values,
    or as the shell's own words for them (``$(hostname)``, ``$$``)."""
    return HERE_ID.format(host=host, pid=pid)


def run_here(*, pid: int, background: bool, calculation,
             members: Dict[str, Dict[str, str]],
             sequence: Optional[dict] = None) -> Job:
    """The address of a run here whose process exists."""
    host = this_host()
    return Job(id=here_id(host, pid), how="here", host=host,
               calculation=str(Path(calculation).resolve()),
               members=members, launched_at=stamp(),
               background=background, pid=pid, started=started_of(pid),
               sequence=sequence)


def queue_job(*, job_id: str, cluster: Optional[str], calculation,
              members: Dict[str, Dict[str, str]],
              sequence: Optional[dict] = None) -> Job:
    """The address of a job the scheduler took, by its id."""
    return Job(id=str(job_id), how="queue", host=this_host(),
               calculation=str(Path(calculation).resolve()),
               members=members, launched_at=stamp(),
               job_id=str(job_id), cluster=cluster, sequence=sequence)


def add_members(job_id: str, members: Dict[str, Dict[str, str]]) -> Job:
    """The runs a group's launch inside a sequence sends, added to the
    address of the job it rides (`job-system.md` § 6.0, *A sequence of
    groups*) -- read, its ``members`` grown, written again by the one
    writer where it was read from; :class:`JobFileError` when no file holds
    that job, or the number is two clusters' (:func:`find`)."""
    import dataclasses
    job = find(job_id)
    if job is None:
        raise JobFileError(f"{_dir()}: no job {job_id} is on file")
    grown = dataclasses.replace(job, members={**job.members, **members})
    write(grown)
    return grown


def write(job: Job) -> Path:
    """The job's file, written through the one writer into the folder made
    private (`config_dir.ensure_private_dir`): it names a calculation and a
    process of this account -- where it was read from, else
    :func:`_file_of`."""
    from .config_dir import ensure_private_dir
    from .persist import write_json
    ensure_private_dir(_dir())
    return write_json(job.path or _file_of(job), job.to_dict())


def read(path: Path) -> Job:
    """One job's file, read; :class:`JobFileError` when it does not."""
    from .persist import check_schema, read_json
    try:
        d = read_json(path)
        check_schema(d.get("schema", ""), JOB_SCHEMA, label=str(path))
        here = d["how"] == "here"
        return Job(id=d["id"], how=d["how"], host=d["host"],
                   calculation=d["calculation"],
                   members={k: {"run": v["run"],
                                "launch_record": v["launch_record"]}
                            for k, v in d["members"].items()},
                   launched_at=d["launched_at"],
                   background=bool(d.get("background")) if here else False,
                   pid=int(d["pid"]) if here else None,
                   started=d.get("started") if here else None,
                   job_id=None if here else str(d["job_id"]),
                   cluster=None if here else d.get("cluster"),
                   sequence=_sequence_of(d.get("sequence")), path=path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise JobFileError(f"{path}: {exc}") from None


def _sequence_of(raw) -> Optional[dict]:
    """A job's ``sequence``, read: ``None`` -- not a sequence; else its name
    and each group, its number and stages.  A file written before
    sequences were recorded states none, and was no sequence."""
    if raw is None:
        return None
    return {"name": str(raw["name"]),
            "groups": [{"number": int(g["number"]),
                        "stages": [str(s) for s in g["stages"]]}
                       for g in raw["groups"]]}


def every() -> List[Union[Job, JobFileError]]:
    """Every job's file, in the order the jobs went -- each the job, or why
    its file does not read."""
    d = _dir()
    got: List[Union[Job, JobFileError]] = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            got.append(read(f))
        except JobFileError as exc:
            # A FILE GONE SINCE IT WAS LISTED is a job whose end was said
            # meanwhile -- its own launch's, or another `jobs status`.
            if f.exists():
                got.append(exc)
    return sorted(got, key=lambda j: (getattr(j, "launched_at", ""),
                                      getattr(j, "id", "")))


def find(job_id: str) -> Optional[Job]:
    """The job ``job_id`` names -- a run here by its id, a queue job by its
    number -- read by what each file on the list says, whatever its file's
    name; ``None`` when none is on file.  Two clusters' jobs of that number:
    the one this machine's scheduler answers for (:func:`asked_here`), and
    :class:`JobFileError` naming both when neither or both are."""
    hits = [j for j in every() if isinstance(j, Job) and j.id == str(job_id)]
    if len(hits) < 2:
        return hits[0] if hits else None
    here = Asked()
    mine = [j for j in hits if asked_here(j, here)[0] is not None]
    if len(mine) == 1:
        return mine[0]
    raise JobFileError(
        f"job {job_id} is on file for clusters "
        f"{', '.join(sorted(str(j.cluster) for j in hits))} -- ask on the "
        f"cluster that ran it")


def forget(job: Job) -> None:
    """A job's address removed -- once its ending was said; its runs and
    the calculation's ledger keep its history."""
    (job.path or _file_of(job)).unlink(missing_ok=True)


__all__ = ["JOB_SCHEMA", "Job", "JobFileError", "NotGoing", "add_members",
           "address_of", "asked_here", "asked_lately", "every", "find",
           "inside",
           "inside_job",
           "forget", "going_here", "queue_job", "read", "run_here",
           "signal_here", "started_of", "this_host", "write"]
