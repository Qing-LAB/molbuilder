"""Materialize engine — turn a :class:`JobSet` into on-disk per-job
directories (docs/protocols/staged-execution.md § 2.1).

Filesystem ONLY: it knows nothing about schedulers or engines.  For each
job it creates ``point-<name>/`` and lays relative symlinks for

  * the static ``shared`` package + the job's own ``script`` (identical
    bytes for the job — pseudos, geometry, monitor, the per-job input),
  * each ``carry`` file from the producing job's directory.

This is the generalization of the benchmark's ``_mb_point`` helper.

Carry symlinks point at concrete filenames in the producer's dir (e.g.
``../point-stage1/job.XV``).  At materialize time (prep) the producer has
not run yet, so these are intentionally **dangling** symlinks — they
resolve once the producer writes the file, and the submit engine's
dependency ordering guarantees the consumer starts only after that.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .model import JobSet


def relink(link_dir: Path, target: str, link_name: str) -> None:
    """Create ``link_dir/link_name`` -> ``target`` (a relative path),
    replacing any existing entry (``ln -sfn`` semantics).  Dangling
    targets are allowed (carry-forward before the producer runs, and the
    rendered wrappers the prep step links in afterwards)."""
    link_path = link_dir / link_name
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    os.symlink(target, link_path)


def job_dir_name(job_name: str) -> str:
    """The on-disk directory for a job — mirrors the benchmark's
    ``point-<...>`` convention so both job-set kinds read alike."""
    return f"point-{job_name}"


def materialize(jobset: JobSet, base_dir) -> List[Path]:
    """Create each job's directory under ``base_dir`` with its symlinks.

    Returns the list of created job directories (in JobSet order).  Idempotent:
    re-running refreshes the symlinks without duplicating anything.  Raises
    ``ValueError`` if the JobSet is structurally invalid (so a bad carry /
    duplicate name can't produce a broken tree).
    """
    errors = jobset.validate()
    if errors:
        raise ValueError(
            "cannot materialize an invalid JobSet:\n  - "
            + "\n  - ".join(errors))
    base = Path(base_dir)
    created: List[Path] = []
    for job in jobset.jobs:
        d = base / job_dir_name(job.name)
        d.mkdir(parents=True, exist_ok=True)
        # static package + this job's own input script: same bytes, one
        # level up in the bundle root.
        for fname in list(jobset.shared) + [job.script]:
            relink(d, os.path.join("..", fname), os.path.basename(fname))
        # runtime-produced carry-forward from the producing job's dir.
        for c in job.carry:
            target = os.path.join("..", job_dir_name(c.from_job), c.pattern)
            relink(d, target, os.path.basename(c.pattern))
        created.append(d)
    return created


__all__ = ["materialize", "job_dir_name", "relink"]
