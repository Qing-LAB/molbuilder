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

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .model import JobSet

#: One attempt at running a stage.  ``project-layout.md`` § 1.5: immutable once
#: it has run, so a re-run is a NEW directory rather than an overwrite.
ATTEMPT_RE = re.compile(r"^run-(\d+)$")

#: Written by ``submit`` into the attempt, AFTER the launch succeeds
#: (``project-layout.md`` § 1.6).  Its presence is the only honest answer to
#: *has this been launched?* -- a queued job has produced nothing yet, so
#: "no output" and "not started" are indistinguishable from the directory alone.
RUN_LAUNCH_SCHEMA = "molbuilder/run-launch@1"
RUN_LAUNCH_FILE = "run.json"


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
    refs = stage_refs(jobset)
    return {j.name: (refs[j.name].token if j.name in refs
                     else job_dir_name(j.name))
            for j in jobset.jobs}


def stage_refs(jobset: JobSet) -> Dict[str, "object"]:
    """``{job name: StageRef}`` for a ladder — **the ordinal, read back**.

    This is the after-produce half of the resolver (§ 8f). ``seq`` is recovered
    from each deck's own token, which is where `project-layout.md` § 4.1 says it
    lives: *"read off the directory name and stored nowhere else"*. Nothing here
    counts positions, so a disabled stage leaves a gap rather than renumbering.

    A job whose script carries no token is **absent from the mapping** rather
    than given an invented ``seq`` — § 4.2's number is assigned once and never
    guessed. Callers fall back to the sweep convention.
    """
    from ..identity import StageRef, parse_stage_token
    out: Dict[str, object] = {}
    for j in jobset.jobs:
        parsed = parse_stage_token(os.path.basename(j.script), jobset.name)
        if parsed:
            out[j.name] = StageRef(*parsed)
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


# --------------------------------------------------------------------- #
#  Attempts — one directory per try at a stage (project-layout.md § 1.6)  #
# --------------------------------------------------------------------- #


def attempts(stage_dir: Path) -> List[int]:
    """The attempt numbers present under ``stage_dir``, ascending."""
    if not Path(stage_dir).is_dir():
        return []
    out = []
    for d in Path(stage_dir).iterdir():
        m = ATTEMPT_RE.match(d.name)
        if m and d.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def was_launched(attempt_dir: Path) -> bool:
    """Whether ``submit`` has launched this attempt — i.e. ``run.json`` exists.

    This is the whole reason that file exists. Without it, preparing a stage
    twice could rewrite the setup underneath a job already sitting in a queue,
    because a queued job has written nothing and looks exactly like one that
    was never started (§ 1.6).
    """
    return (Path(attempt_dir) / RUN_LAUNCH_FILE).is_file()


def resolve_attempt(stage_dir: Path) -> Tuple[Path, bool]:
    """The attempt directory to prepare into, and whether it is a fresh one.

    § 1.6: *"Preparing again is safe until the run has been launched.
    Otherwise splitting the two steps leaks directories — prepare, change your
    mind, prepare again, and an empty ``run-3`` sits there forever."*

    So the last attempt is REUSED when it has not been launched, and a new one
    is opened only when the last has. That also makes the numbering mean
    something: every ``run-<n>`` on disk was actually started.
    """
    existing = attempts(stage_dir)
    if existing:
        last = Path(stage_dir) / f"run-{existing[-1]}"
        if not was_launched(last):
            return last, False
        return Path(stage_dir) / f"run-{existing[-1] + 1}", True
    return Path(stage_dir) / "run-0", True


def prepare_attempt(jobset: JobSet, base_dir, stage_name: str, *,
                    continue_from: Optional[str] = None,
                    cold: bool = False,
                    carry: Optional[List[str]] = None) -> Dict[str, object]:
    """Set ONE stage up to run, and report what was done.

    The five steps § 1.6 names: **resolve** the next ``run-<n>``, **create**
    it, **link** the deck / monitor / shared package in, **copy** whatever this
    run continues from, and **report** — the report being the point, since
    preparing is still design and the split from starting is what gives you
    somewhere to look before committing cluster time.

    ``continue_from`` is a bundle-relative attempt directory —
    ``"01_coarse/run-0"``. **Which run you continue from is something you say,
    not something molbuilder guesses** (§ 1.6): continuing from ``run-0`` and
    from ``run-2`` are different scientific choices. ``cold=True`` means start
    clean, and with a directory per attempt that is simply *skip the copy* —
    there is nothing to move aside, because a fresh attempt is empty unless
    something is put in it.

    ``carry`` names the files to copy; it defaults to the engine's warm set for
    this job's label. They are **copied, never linked** — the engine writes to
    those very filenames, and writing through a link would destroy the result
    you started from.
    """
    base = Path(base_dir)
    dir_of = job_dir_names(jobset)
    job = next((j for j in jobset.jobs if j.name == stage_name), None)
    if job is None:
        known = ", ".join(j.name for j in jobset.jobs)
        raise ValueError(
            f"no stage named {stage_name!r} in this job-set; it has: {known}")

    stage_dir = base / dir_of[stage_name]
    stage_dir.mkdir(parents=True, exist_ok=True)
    attempt, is_new = resolve_attempt(stage_dir)
    attempt.mkdir(parents=True, exist_ok=True)

    # Inputs: the deck and the shared package, linked UP to the stage dir and
    # the bundle root.  Identical bytes for every attempt, so a link is right
    # here -- it is the warm state below that must be a copy.
    linked: List[str] = []
    for fname in [job.script] + list(jobset.shared):
        relink(attempt, os.path.join("..", "..", fname),
               os.path.basename(fname))
        linked.append(os.path.basename(fname))
    for extra in ("mb_monitor.py",):
        if (base / extra).exists():
            relink(attempt, os.path.join("..", "..", extra), extra)
            linked.append(extra)
    stem = Path(job.script).stem
    for wrapper in (f"{stem}.run.sh", f"{stem}.sbatch"):
        if (base / wrapper).exists():
            relink(attempt, os.path.join("..", "..", wrapper), wrapper)
            linked.append(wrapper)

    copied: List[str] = []
    if continue_from and not cold:
        src = base / continue_from
        if not src.is_dir():
            raise ValueError(
                f"--continue-from {continue_from!r}: no such attempt under "
                f"{base}. Name an attempt directory that has already run, "
                f"e.g. '01_coarse/run-0'.")
        names = carry if carry is not None else _warm_names(jobset, job)
        for name in names:
            f = src / name
            if f.is_file():
                shutil.copy2(f, attempt / name)
                copied.append(name)
        if not copied:
            raise ValueError(
                f"--continue-from {continue_from!r}: that attempt holds none "
                f"of the files this stage would continue from "
                f"({', '.join(names)}). Did it run?")

    # Leave the provenance where ``submit`` can find it: prep is what knows
    # which attempt this one continues from, and submit writes run.json.  A
    # marker file beats threading the value through a launch argument that
    # every caller would have to remember to pass.
    marker = attempt / ".continued-from"
    if copied:
        marker.write_text(str(continue_from) + "\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()

    return {
        "stage": stage_name,
        "dir": attempt,
        "fresh": is_new,
        "linked": linked,
        "copied": copied,
        "continued_from": (None if (cold or not continue_from)
                           else str(continue_from)),
        "cold": bool(cold),
    }


def _warm_names(jobset: JobSet, job) -> List[str]:
    """The engine's warm-restart files for this job, by name.

    Taken from the job's own ``carry`` patterns when it has them, so a
    hand-built chained JobSet keeps working; otherwise SIESTA's standard three,
    keyed on the label, which is what the staged producer stops declaring once
    stages no longer chain.
    """
    if job.carry:
        return [os.path.basename(c.pattern) for c in job.carry]
    return [f"{jobset.name}.XV", f"{jobset.name}.DM", f"{jobset.name}.CG"]


def write_run_launch(attempt_dir: Path, *, mode: str, command: List[str],
                     job_id: Optional[str] = None,
                     continued_from: Optional[str] = None,
                     launched_at: Optional[str] = None) -> Path:
    """Record a launch into the attempt — ``molbuilder/run-launch@1``.

    Written **after** the launch succeeds, so a failed submission leaves the
    attempt exactly as prepare left it and is still safe to prepare again
    (§ 1.6). The last field is the run's provenance — *this geometry came from
    ``01_coarse/run-0``* — which is worth recording whether or not anything
    reads it back.
    """
    from datetime import datetime, timezone
    p = Path(attempt_dir) / RUN_LAUNCH_FILE
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
    p.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return p


__all__ = ["materialize", "job_dir_name", "job_dir_names", "stage_refs",
           "relink",
           "attempts", "was_launched", "resolve_attempt", "prepare_attempt",
           "write_run_launch", "RUN_LAUNCH_SCHEMA", "RUN_LAUNCH_FILE"]
