"""Run-status reader — the "inform" layer
(docs/execution/job-system.md, § 14).

The resume contract is: **the modeling software resumes; molbuilder informs
and the user decides** (never auto-recovers — redoing a long run unknowingly
is a heavy penalty).  This module is the *inform* half: for a prepped/running
JobSet it answers, per stage, *did it finish? is it running? did it fail? are
the warm-restart files there?* and *which is the first incomplete stage* (the
one to resume from) — so the manual continue is a one-glance decision.

REUSE, not reinvention: per-stage run state comes from
``parse.dirs.job.decode_run_dir`` (the directory decoder behind the Results
tab + JobMonitor, ``job-decoder.md``); this module only adds the cross-stage
view (warm-file inventory + first-incomplete pointer).  It is **read-only**:
it inspects the tree, changes nothing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .materialize import job_dir_names
from .model import JobSet

# Engine-native warm-restart files keyed by the project id (system label),
# from script-execution.md's inventory.  Presence of these in a stage dir is
# what lets that stage (or the next) warm-start.
_WARM_FILES = {
    "siesta": (".XV", ".DM", ".CG"),
    "pyscf":  (".chk",),
}

# decode_run_dir states we treat as "this stage is done".
_DONE = "finished"


@dataclass(frozen=True)
class StageStatus:
    """One stage's status (read-only snapshot)."""
    name:       str
    dir:        str
    state:      str               # not-started/pending/running/stale/failed/finished
    detail:     str
    warm_files: List[str] = field(default_factory=list)  # restart files present

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class JobSetStatus:
    """The whole job-set's status + the resume pointer."""
    name:            str
    engine:          str
    stages:          List[StageStatus]
    first_incomplete: Optional[str]   # name of the first non-finished stage (resume here)
    complete:        bool             # every stage finished

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "engine": self.engine,
            "first_incomplete": self.first_incomplete,
            "complete": self.complete,
            "stages": [s.to_dict() for s in self.stages],
        }


def _warm_present(stage_dir: Path, label: str, engine: str) -> List[str]:
    """Which engine warm-restart files actually exist (real files, not
    dangling carry symlinks) in this stage's dir."""
    out: List[str] = []
    for ext in _WARM_FILES.get(engine, ()):
        f = stage_dir / f"{label}{ext}"
        if f.is_file():                      # follows symlinks; dangling -> False
            out.append(f.name)
    return out


def _stage_state(stage_dir: Path) -> tuple:
    """(state, detail) for one stage dir, reusing the directory decoder.
    Maps the decoder's "running/no .out yet" onto the more honest
    'not-started' / 'pending' the inform layer needs."""
    if not stage_dir.is_dir():
        return ("not-started", "no directory yet (not prepped)")
    # no engine .out at all -> prepped but not launched (decoder calls this
    # "running", which is misleading before anything has run).
    if not any(stage_dir.glob("*.out")) and not any(stage_dir.glob("*.log")):
        return ("pending", "prepped, not launched (no .out)")
    try:
        from ..parse.dirs.job import decode_run_dir
        res = decode_run_dir(stage_dir)
    except Exception as e:                    # decoder is fail-soft; stay informative
        return ("unknown", f"could not decode: {e}")
    st = res.status or {}
    return (st.get("state", "unknown"), st.get("detail", ""))


def jobset_status(jobset: JobSet, base_dir) -> JobSetStatus:
    """Read the on-disk status of every stage of ``jobset`` under
    ``base_dir`` (read-only).  ``first_incomplete`` is the first stage that
    is not ``finished`` — the stage to resume from; ``None`` (and
    ``complete=True``) when every stage finished."""
    base = Path(base_dir)
    label = jobset.name
    stages: List[StageStatus] = []
    first_incomplete: Optional[str] = None
    dirs = job_dir_names(jobset)
    for job in jobset.jobs:
        d = base / dirs[job.name]
        state, detail = _stage_state(d)
        stages.append(StageStatus(
            name=job.name, dir=d.name, state=state, detail=detail,
            warm_files=_warm_present(d, label, jobset.engine),
        ))
        if first_incomplete is None and state != _DONE:
            first_incomplete = job.name
    return JobSetStatus(
        name=jobset.name, engine=jobset.engine, stages=stages,
        first_incomplete=first_incomplete,
        complete=(first_incomplete is None),
    )


def render_status(status: JobSetStatus) -> str:
    """Human-readable status table + the resume pointer (the inform
    surface for the CLI / plan / UI)."""
    lines: List[str] = [
        f"JOB-SET STATUS -- {status.name} ({status.engine})",
        "",
    ]
    hdr = ("#", "stage", "state", "warm files", "detail")
    rows = [(str(i), s.name, s.state,
             ", ".join(s.warm_files) or "-", s.detail)
            for i, s in enumerate(status.stages)]
    w = [max(len(r[k]) for r in rows + [hdr]) for k in range(5)]
    def fmt(r):
        return "  ".join(s.ljust(w[k]) for k, s in enumerate(r))
    lines.append("  " + fmt(hdr))
    lines.append("  " + "  ".join("-" * w[k] for k in range(5)))
    lines += ["  " + fmt(r) for r in rows]
    lines.append("")
    if status.complete:
        lines.append("All stages finished. Nothing to resume.")
    else:
        lines.append(
            f"First incomplete stage: {status.first_incomplete}.  "
            "molbuilder does NOT auto-resume -- you decide: re-submit that "
            "stage (the engine warm-starts from its own restart files) or "
            "switch parameters (staged-relaxation-guide.md).")
    return "\n".join(lines)


__all__ = ["StageStatus", "JobSetStatus", "jobset_status", "render_status"]
