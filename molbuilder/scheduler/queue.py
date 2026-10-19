"""A job in the scheduler's queue -- the cluster it is sent to, whether the
scheduler still lists it and in what state, and its cancel (`execution/job-system.md`
§ 6.0a): asked of the machine record's scheduler -- its kind's words
(`kinds.toml`, `query`), run from where the record says its commands are
(`record.Scheduler`) -- by the one sender and by `molbuilder jobs`; and how
the scheduler ended a job, read from the job's own error file by the run
door (`scheduler.md` R16).  Each answer read by the kind's reader
(:data:`_READERS`, `scheduler.md` § 6a).
Floor 1: facts of a machine.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple


# --------------------------------------------------------------------- #
#  SLURM's answers -- its reader: what it prints, which no format states   #
# --------------------------------------------------------------------- #

#: The states SLURM lists a job in for a while after it has ended
#: (`MinJobAge`), when asked for it by its id.
_SLURM_ENDED = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"})


def _slurm_cluster(stdout: str) -> Optional[str]:
    """``scontrol show config``'s ``ClusterName``."""
    for ln in stdout.splitlines():
        key, _eq, value = ln.partition("=")
        if key.strip() == "ClusterName" and value.strip():
            return value.strip()
    return None


def _slurm_listed(returncode: int, stdout: str,
                  stderr: str) -> Optional[Tuple[bool, str]]:
    """``squeue``'s answer for one id, ``STATE|REASON`` -- ``(listed, said)``:
    listed while its state has not ended, and ``said`` the state with
    SLURM's reason for it, ``PENDING (QOSMaxWallDurationPerJobLimit)``, its
    own words (`scheduler.md` R6).  An id it has forgotten is answered with
    an error rather than an empty list, since SLURM keeps an ended job only
    `MinJobAge` seconds -- not listed; ``None`` for an answer that says
    neither."""
    if returncode != 0:
        return (False, "") if "Invalid job id" in stderr else None
    rows = [ln.partition("|") for ln in stdout.split("\n") if ln.strip()]
    if not rows:
        return False, ""
    state, _bar, reason = (x.strip() for x in rows[0])
    said = state if reason in ("", "None") else f"{state} ({reason})"
    return any(s.strip() not in _SLURM_ENDED for s, _b, _r in rows), said


def _slurm_standing(returncode: int, stdout: str,
                    stderr: str) -> Optional[Dict[str, Any]]:
    """``squeue``'s standing for one id, ``STATE|REASON|START|PRIORITY`` --
    ``{listed, state, reason, start, priority}``: ``start`` the time it
    expects to start, ``None`` while it states none (``N/A``); ``listed``
    False, the rest absent, for an id it has forgotten; ``None`` for an
    answer that says neither."""
    if returncode != 0:
        return {"listed": False} if "Invalid job id" in stderr else None
    rows = [ln.split("|") for ln in stdout.split("\n") if ln.strip()]
    if not rows or len(rows[0]) < 4:
        return {"listed": False} if not rows else None
    state, reason, start, priority = (x.strip() for x in rows[0][:4])
    return {"listed": True, "state": state,
            "reason": None if reason in ("", "None") else reason,
            "start": None if start in ("", "N/A") else start,
            "priority": priority or None}


def _slurm_ended(text: str, job_id: str) -> Optional[str]:
    """SLURM's line in a job's error file when it ends the job --
    ``*** JOB <id> ON <node> <its words> ***`` -- for job ``job_id``: the
    words inside the frame as SLURM wrote them, ``CANCELLED AT
    2026-10-18T01:19:15 DUE TO TIME LIMIT``; ``None`` where the file holds
    no such line for the job (`scheduler.md` R16).  Nothing past the frame
    is assumed: a cancel, a preemption or a node's loss reads in its own
    words."""
    framed = re.compile(r"\*\*\* JOB " + re.escape(job_id)
                        + r" ON \S+ (.+?) \*\*\*")
    for ln in text.splitlines():
        m = framed.search(ln)
        if m:
            return m.group(1).strip()
    return None


class _Readers(NamedTuple):
    """A kind's readers of what its scheduler says, which no format
    states (`scheduler.md` § 6a)."""
    #: Which cluster it serves (its ``cluster`` query's answer).
    cluster: Callable[[str], Optional[str]]
    #: Whether it still lists a job, and in what state (``listed``).
    listed: Callable[[int, str, str], Optional[Tuple[bool, str]]]
    #: A queued job's standing (``standing``): state, reason, expected
    #: start, priority.
    standing: Callable[[int, str, str], Optional[Dict[str, Any]]]
    #: How it says it ended a job, in the job's error file (R16).
    ended: Callable[[str, str], Optional[str]]


#: Each kind with a queue -> the readers of its answers.
_READERS = {"slurm": _Readers(_slurm_cluster, _slurm_listed,
                              _slurm_standing, _slurm_ended)}


def _asked(scheduler, what: str, **values):
    """``(kind, the command line)`` for query ``what``, or ``(kind, None)``
    where the record names no command for it."""
    from .kinds import kind
    k = kind(scheduler.kind)
    if not k.queue:
        return k, None
    return k, k.asked(what, scheduler.commands(), **values)


def cluster(scheduler) -> Optional[str]:
    """The cluster ``scheduler`` serves, as it names it (its kind's
    ``cluster`` query, read by its reader) -- what tells one cluster's job
    ids from another's where a home folder is shared -- or ``None`` when it
    cannot be read."""
    k, line = _asked(scheduler, "cluster")
    if line is None:
        return None
    try:
        r = subprocess.run(line, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return _READERS[k.name].cluster(r.stdout) if r.returncode == 0 else None


def listed(scheduler, job_id: str) -> Optional[Tuple[bool, str]]:
    """``(listed, said)`` -- whether ``scheduler`` still lists job
    ``job_id`` as not yet ended (queued, running, ending), and the state it
    lists it in with its reason, in the scheduler's words (`job-system.md`
    § 6.0a); ``None`` where it cannot be asked: no command for it on the
    record, one that does not run here, or an answer that says neither."""
    k, line = _asked(scheduler, "listed", id=job_id)
    if line is None:
        return None
    try:
        r = subprocess.run(line, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return _READERS[k.name].listed(r.returncode, r.stdout, r.stderr)


def standing(scheduler, job_id: str) -> Optional[Dict[str, Any]]:
    """A queued job's standing as ``scheduler`` states it -- ``{listed,
    state, reason, start, priority}`` (its kind's ``standing`` query, read
    by its reader; `scheduler.md` § 6a) -- or ``None`` where it cannot be
    asked here: no command for it on the record, one that does not run on
    this machine, or an answer that says neither."""
    k, line = _asked(scheduler, "standing", id=job_id)
    if line is None:
        return None
    try:
        r = subprocess.run(line, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return _READERS[k.name].standing(r.returncode, r.stdout, r.stderr)


def cancel(scheduler, job_id: str) -> str:
    """Stop job ``job_id`` (its kind's ``cancel``); returns the line run.
    :class:`RuntimeError` in the scheduler's own words when it refuses."""
    _k, line = _asked(scheduler, "cancel", id=job_id)
    if line is None:
        raise RuntimeError("the machine's record names no cancel command")
    said = " ".join(line)
    try:
        r = subprocess.run(line, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"{said} could not be run ({exc})") from None
    if r.returncode != 0:
        raise RuntimeError(f"{said} refused: "
                           f"{(r.stderr or r.stdout).strip()}")
    return said


def ended(kind_name: str, stderr, job_id: str) -> Optional[str]:
    """How the scheduler of kind ``kind_name`` says it ended job ``job_id``,
    in its own words -- read from the job's error file ``stderr`` by the
    kind's reader (`scheduler.md` R16) -- or ``None``: the file is not there
    or does not read, it holds no such line, or the kind keeps no queue."""
    readers = _READERS.get(kind_name)
    if readers is None:
        return None
    try:
        text = Path(stderr).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return readers.ended(text, str(job_id))


__all__ = ["cancel", "cluster", "ended", "listed"]
