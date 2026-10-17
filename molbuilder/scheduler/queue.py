"""A job in the scheduler's queue -- the cluster it is sent to, whether the
scheduler still lists it, and its cancel (`execution/job-system.md`
§ 6.0a): SLURM's own commands, run here, asked by the one sender and by
`molbuilder jobs`.  Floor 1: facts of this machine.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional


def cluster() -> Optional[str]:
    """The cluster this machine's scheduler serves, as it names it
    (``scontrol show config``'s ``ClusterName``) -- what tells one cluster's
    job ids from another's where a home folder is shared -- or ``None``
    when it cannot be read."""
    if shutil.which("scontrol") is None:
        return None
    try:
        r = subprocess.run(["scontrol", "show", "config"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        key, _eq, value = line.partition("=")
        if key.strip() == "ClusterName" and value.strip():
            return value.strip()
    return None


#: The states SLURM lists a job in for a while after it has ended
#: (`MinJobAge`), when asked for it by its id.
ENDED_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"})


def lists(job_id: str) -> Optional[bool]:
    """Whether the scheduler here still lists job ``job_id`` as not yet
    ended -- queued, running, ending -- ``False`` once it does not, and
    ``None`` where it cannot be asked: no ``squeue`` on this machine, or an
    answer that says neither."""
    if shutil.which("squeue") is None:
        return None
    try:
        r = subprocess.run(["squeue", "--noheader", f"--jobs={job_id}",
                            "--format=%T"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 0:
        return any(s not in ENDED_STATES for s in r.stdout.split())
    # AN ID IT HAS FORGOTTEN is answered with an error, not an empty list:
    # SLURM keeps an ended job only `MinJobAge` seconds.
    return False if "Invalid job id" in r.stderr else None


def cancel(job_id: str) -> None:
    """``scancel <job id>``; :class:`RuntimeError` in the scheduler's own
    words when it refuses."""
    try:
        r = subprocess.run(["scancel", str(job_id)], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"scancel {job_id} could not be run ({exc})") \
            from None
    if r.returncode != 0:
        raise RuntimeError(f"scancel {job_id} refused: "
                           f"{(r.stderr or r.stdout).strip()}")


__all__ = ["ENDED_STATES", "cancel", "cluster", "lists"]
