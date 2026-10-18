"""A job in the scheduler's queue -- the cluster it is sent to, whether the
scheduler still lists it and in what state, and its cancel (`execution/job-system.md`
§ 6.0a): asked of the machine record's scheduler -- its kind's words
(`kinds.toml`, `query`), run from where the record says its commands are
(`record.Scheduler`) -- by the one sender and by `molbuilder jobs`; each
answer read by the kind's reader (:data:`_READERS`, `scheduler.md` § 6a).
Floor 1: facts of a machine.
"""
from __future__ import annotations

import subprocess
from typing import Optional, Tuple


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


#: Each kind with a queue -> the readers of its answers: which cluster it
#: serves, and whether it still lists a job and in what state.
_READERS = {"slurm": (_slurm_cluster, _slurm_listed)}


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
    return _READERS[k.name][0](r.stdout) if r.returncode == 0 else None


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
    return _READERS[k.name][1](r.returncode, r.stdout, r.stderr)


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


__all__ = ["cancel", "cluster", "listed"]
