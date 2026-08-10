"""Materialize engine — turn a :class:`JobSet` into on-disk per-job
directories (docs/execution/job-system.md).

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
from typing import Dict, List

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
    """The on-disk directory for a **sweep** job — the benchmark's
    ``point-<...>`` convention.

    A ladder's stage directory is NOT this: see :func:`job_dir_names`, which is
    what every caller should use, because the answer depends on the job SET
    (its kind, and the deck each job carries) rather than on a name alone.
    """
    return f"point-{job_name}"


def job_dir_names(jobset: JobSet) -> Dict[str, str]:
    """``{job name: directory name}`` for a whole JobSet — the naming authority.

    Two kinds, two conventions, and `project-layout.md` § 4.1 is explicit about
    which is which:

    | kind | directory |
    |---|---|
    | ``sweep`` (the benchmark) | ``point-<name>`` |
    | ``ladder`` (a stage ladder) | ``<seq>_<name>`` — ``01_coarse``, ``02_tight`` |

    Until 2026-08-10 every kind got ``point-<name>``, so a staged run's
    directories came out ``point-coarse/`` (`worked-example.md` gap 6). The
    contract's fix is *"branch on ``JobSet.kind``"*, and this is that branch.

    **The seq is read back off the deck, not counted here.** ``job.script`` is
    ``<label>_<NN>_<name>.fdf`` (decision 27), so the token the directory is
    named for is the one the deck already carries — which is what makes
    ``<NN>_<name>/<label>_<NN>_<name>.fdf`` a self-check rather than a
    repetition (§ 4.1). Counting positions here instead would reintroduce
    exactly what `engines/stages.md` R5 forbids: a number that shifts when the
    ladder changes, silently handing one stage's directory to another.

    A ladder job whose script carries no token falls back to ``point-<name>``.
    That is not a staged-producer JobSet — hand-written, or older than decision
    27 — and inventing a seq for it would be guessing at the one number that
    must never be guessed.
    """
    if jobset.kind != "ladder":
        return {j.name: job_dir_name(j.name) for j in jobset.jobs}
    from ..identity import parse_stage_token, stage_token
    out: Dict[str, str] = {}
    for j in jobset.jobs:
        parsed = parse_stage_token(os.path.basename(j.script), jobset.name)
        out[j.name] = (stage_token(*parsed) if parsed
                       else job_dir_name(j.name))
    return out


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
    dirs = job_dir_names(jobset)
    for job in jobset.jobs:
        d = base / dirs[job.name]
        d.mkdir(parents=True, exist_ok=True)
        # static package + this job's own input script: same bytes, one
        # level up in the bundle root.
        for fname in list(jobset.shared) + [job.script]:
            relink(d, os.path.join("..", fname), os.path.basename(fname))
        # runtime-produced carry-forward from the producing job's dir.
        for c in job.carry:
            target = os.path.join("..", dirs[c.from_job], c.pattern)
            relink(d, target, os.path.basename(c.pattern))
        created.append(d)
    return created


__all__ = ["materialize", "job_dir_name", "job_dir_names", "relink"]
