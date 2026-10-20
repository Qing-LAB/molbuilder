"""THE JOBS LAUNCHED FROM THIS MACHINE -- each launch's address, kept the way
a server keeps its pidfile (`execution/job-system.md` § 6.0a): one file per
submission in molbuilder's state folder, written by the one sender the
moment it goes (`jobset.submit._go`), read by `molbuilder jobs`.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

#: Every job's file, at one major (`persist.check_schema`).
JOB_SCHEMA = "molbuilder/job@1"

#: Where a process's start is read: Linux's own account of its processes.
_PROC = Path("/proc")


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
    #: The file it was read from -- not written.
    path: Optional[Path] = field(default=None, compare=False)

    def to_dict(self) -> dict:
        return {"schema": JOB_SCHEMA, "id": self.id, "how": self.how,
                "host": self.host, "calculation": self.calculation,
                "members": {k: dict(v) for k, v in self.members.items()},
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dir() -> Path:
    from .config_dir import jobs_dir
    return jobs_dir()


def _file_of(job_id: str) -> Path:
    return _dir() / f"{job_id}.json"


class Asked:
    """The cluster a scheduler here answers for, as it names it -- asked
    once a command for each scheduler, and only when a queue job is asked
    of."""
    def __init__(self):
        self._asked = {}

    def cluster(self, scheduler):
        if scheduler not in self._asked:
            from .scheduler.queue import cluster
            self._asked[scheduler] = cluster(scheduler)
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


def is_going(job: "Job", here: "Asked") -> "Tuple[Optional[bool], str]":
    """``(going, said)`` -- whether the job is still going, its process or
    the scheduler's list, ``None`` where this machine cannot ask -- and
    that in words: THE ONE ANSWER, `molbuilder jobs` asks it of each job,
    `jobset scratch` of the jobs that hold its runs."""
    if job.how == "here":
        going = going_here(job)
        return going, {True: "its process is going",
                       False: "its process has ended",
                       None: (f"on {job.host}: ask there"
                              if job.host != this_host() else
                              "this machine cannot tell")}[going]
    from .scheduler.queue import listed
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
    answers = here.cluster(scheduler)
    if job.cluster is not None and answers != job.cluster:
        return None, (f"sent to cluster {job.cluster}: ask there"
                      if answers is not None else
                      "the scheduler cannot be asked here")
    # THE SCHEDULER'S OWN WORDS for a job it holds -- its state and reason,
    # so a job it took and will never start says why (`job-system.md`
    # § 6.0a; `scheduler.md` R6).
    answer = listed(scheduler, job.job_id)
    if answer is None:
        return None, "the scheduler cannot be asked here"
    going, said = answer
    if going:
        return True, "the scheduler lists it" + (f": {said}" if said else "")
    # ENDED, in its words while it still lists the job -- ``FAILED
    # (NonZeroExitCode)`` -- and only then forgotten.
    return False, (f"the scheduler lists it ended: {said}" if said
                   else "the scheduler no longer lists it")


def run_here(*, pid: int, background: bool, calculation,
             members: Dict[str, Dict[str, str]]) -> Job:
    """The address of a run here whose process exists."""
    host = this_host()
    return Job(id=f"{host}-{pid}", how="here", host=host,
               calculation=str(Path(calculation).resolve()),
               members=members, launched_at=_now(),
               background=background, pid=pid, started=started_of(pid))


def queue_job(*, job_id: str, cluster: Optional[str], calculation,
              members: Dict[str, Dict[str, str]]) -> Job:
    """The address of a job the scheduler took, by its id."""
    return Job(id=str(job_id), how="queue", host=this_host(),
               calculation=str(Path(calculation).resolve()),
               members=members, launched_at=_now(),
               job_id=str(job_id), cluster=cluster)


def write(job: Job) -> Path:
    """The job's file, written through the one writer into the folder made
    private (`config_dir.ensure_private_dir`): it names a calculation and a
    process of this account."""
    from .config_dir import ensure_private_dir
    from .persist import write_json
    ensure_private_dir(_dir())
    return write_json(_file_of(job.id), job.to_dict())


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
                   cluster=None if here else d.get("cluster"), path=path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise JobFileError(f"{path}: {exc}") from None


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
    """The job ``job_id`` names, or ``None`` when no file holds it;
    :class:`JobFileError` when its file does not read."""
    f = _file_of(job_id)
    return read(f) if f.is_file() else None


def forget(job: Job) -> None:
    """A job's address removed -- once its ending was said; its runs and
    the calculation's ledger keep its history."""
    _file_of(job.id).unlink(missing_ok=True)


__all__ = ["JOB_SCHEMA", "Job", "JobFileError", "NotGoing", "every", "find",
           "forget", "going_here", "queue_job", "read", "run_here",
           "signal_here", "started_of", "this_host", "write"]
