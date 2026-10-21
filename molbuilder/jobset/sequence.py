"""A SEQUENCE OF GROUPS -- one job that launches each group named, in the
description's order, each when the one before it has finished
(`execution/job-system.md` § 6.0, *A sequence of groups*).

Floor 1 (`execution/architecture.md` § 2.1): the sequence's own launch
record -- written with the launch's plan before its job goes, stamped by the
sender as it goes (:func:`sent`), and read by each group's launch inside the
job, which records its runs as that job's.  Whether a sequence still going
holds a group is the ready door's (`ready.held_by_sequence`, floor 4); the
sequence's request, `group.sequence_request`'s (floor 3); its plan,
`submit._plan_sequence`'s.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

#: Every sequence's launch record, at one major (`persist.check_schema`).
SEQUENCE_SCHEMA = "molbuilder/sequence-launch@1"


class SequenceError(ValueError):
    """A sequence's launch record that is not there, or does not read --
    said with its path."""


@dataclass(frozen=True)
class Record:
    """A sequence's launch record (`job-contracts.md`, *Sequence launch
    record*): what each run a group's launch inside its job sends records of
    that job."""
    #: Each group, in order: ``{"number", "stages"}``.
    groups: List[Dict]
    #: ``submit`` -- a queue job -- or ``direct`` -- a walk run here.
    mode: str
    #: The line that sends the job.
    command: List[str]
    #: The queue, wall and memory it is sent with; ``None`` run here.
    placed_on: Optional[Dict]
    #: ``{"kind", "outputs_in"}`` -- the scheduler's kind and the folder its
    #: error file goes to, from the calculation's folder; ``None`` run here.
    scheduler: Optional[Dict]
    #: When it went -- written by the sender the moment it goes; ``None``
    #: in the plan, which decides what the job is, not when.
    launched_at: Optional[str] = None
    #: The scheduler's id, once it gives it; ``None`` until then, and run
    #: here.
    job_id: Optional[str] = None

    def to_dict(self) -> Dict:
        # EVERY KEY, EVERY TIME, its null an answer (`job-contracts.md`).
        return {"schema": SEQUENCE_SCHEMA,
                "groups": [{"number": int(g["number"]),
                            "stages": list(g["stages"])} for g in self.groups],
                "mode": self.mode, "command": list(self.command),
                "placed_on": (dict(self.placed_on) if self.placed_on
                              else None),
                "scheduler": (dict(self.scheduler) if self.scheduler
                              else None),
                "launched_at": self.launched_at, "job_id": self.job_id}

    def run_said(self, *, job_id: Optional[str], run_dir, base,
                 kind) -> Dict:
        """What a run a group's launch inside this sequence sends records
        of the job it rides (`runrecord.write_launch`'s keys): the job's
        mode, line, id, placement -- and where its scheduler writes the
        job's error file, from the run's own folder (``kind``, the
        scheduler's, names that file for ``job_id``)."""
        from ..runfiles import SCHEDULER_STDERR
        scheduler = None
        if self.scheduler is not None:
            stderr = None
            if job_id is not None:
                err = (Path(base) / self.scheduler["outputs_in"]
                       / kind.output_file(SCHEDULER_STDERR, job_id=job_id))
                stderr = os.path.relpath(err, run_dir)
            scheduler = {"kind": self.scheduler["kind"], "stderr": stderr}
        return {"mode": self.mode, "command": list(self.command),
                "job_id": job_id if self.mode == "submit" else None,
                "placed_on": self.placed_on, "scheduler": scheduler}


def record_path(base, names) -> Path:
    """Where a sequence's launch record lies: its name's, in the
    calculation's `launch/` (`runfiles.GroupNames.of_sequence`)."""
    from ..runfiles import LAUNCH_DIR
    return Path(base) / LAUNCH_DIR / names.name(".launch.json")


def record_text(rec: Record) -> str:
    """The record as written -- for a plan's writes, which carry text
    (`jobset.planned.Plan`); the same bytes :func:`write_record` writes
    (`persist.json_text`)."""
    from ..persist import json_text
    return json_text(rec.to_dict())


def write_record(path, rec: Record) -> Path:
    """The record, whole or absent (`persist`)."""
    from ..persist import write_json
    return write_json(Path(path), rec.to_dict())


def read_record(path) -> Record:
    """The record at ``path``; :class:`SequenceError` naming it when it is
    not there or does not read."""
    from ..persist import check_schema, read_json
    path = Path(path)
    if not path.is_file():
        raise SequenceError(f"{path}: no sequence's launch record is there "
                            f"(job-system.md § 6.0, *A sequence of groups*)")
    try:
        d = read_json(path)
        check_schema(d.get("schema", ""), SEQUENCE_SCHEMA, label=str(path))
        return Record(
            groups=[{"number": int(g["number"]),
                     "stages": [str(s) for s in g["stages"]]}
                    for g in d["groups"]],
            mode=str(d["mode"]), command=[str(c) for c in d["command"]],
            placed_on=d["placed_on"], scheduler=d["scheduler"],
            launched_at=d["launched_at"], job_id=d["job_id"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SequenceError(f"{path}: {exc}") from None


def sent(path, *, job_id: Optional[str]) -> Path:
    """The record of a job that went -- when, and the id the scheduler gave
    it (``None`` run here, or an id not read): written by the one sender the
    moment it goes, as a run's launch record is."""
    import dataclasses
    from datetime import datetime, timezone
    return write_record(path, dataclasses.replace(
        read_record(path),
        launched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        job_id=str(job_id) if job_id is not None else None))


__all__ = ["Record", "SEQUENCE_SCHEMA", "SequenceError", "read_record",
           "record_path", "record_text", "sent", "write_record"]
