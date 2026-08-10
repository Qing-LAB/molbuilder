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
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..identity import StageRef, seq_text
from .materialize import (RUN_LAUNCH_FILE, attempts, job_dir_names,
                          latest_attempt, shape_of, stage_refs)
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
    #: not-started/pending/queued/running/stale/failed/finished.  ``queued`` is
    #: the contract's own word (project-layout.md § 1.6, *"queued as job
    #: 481923"*) for launched-but-no-output-yet, which is exactly the state an
    #: empty directory cannot be read for.
    state:      str
    detail:     str
    warm_files: List[str] = field(default_factory=list)  # restart files present
    #: The stage's assigned ordinal, read back off its deck -- NOT its row.
    #: ``None`` for a sweep point, which has no seq (project-layout.md § 4.1).
    seq: Optional[int] = None
    #: Which attempt this status was read from (``run-0``), or ``None`` for a
    #: flat run, which happens in the container itself (§ 1.5).
    attempt: Optional[str] = None
    #: Every attempt present, ascending -- the stage's history.  A re-run makes
    #: a new directory and leaves the old one exactly as it was (§ 1.5), so
    #: this is the list of tries, not a counter that can be off.
    attempts: List[int] = field(default_factory=list)
    #: The attempt's ``run.json``, whole, or ``None`` if it has not been
    #: launched.  Carried rather than picked apart so the per-stage view has
    #: the record without a SECOND reader of the same file -- the schema is
    #: versioned (``molbuilder/run-launch@1``) and may grow fields.
    launch: Optional[Dict[str, Any]] = None

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


def _launch_record(attempt: Optional[Path]) -> Optional[Dict[str, Any]]:
    """``run.json`` from an attempt, or ``None`` — fail-soft on a bad file.

    `project-layout.md` § 1.6: *"Has this been launched? has no honest answer
    from the directory alone"*, so this file is the answer and status must read
    it rather than infer from an absence of output.
    """
    if attempt is None:
        return None
    p = attempt / RUN_LAUNCH_FILE
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}                # present but unreadable: launched, details lost


def _stage_state(observed: Path, launch: Optional[Dict[str, Any]],
                 out_glob: str = "*") -> tuple:
    """(state, detail) for the directory a stage's run actually happened in.

    ``observed`` is the latest attempt where there is one, and the stage
    container for a flat run (`project-layout.md` § 1.5) — the caller resolves
    that, because *where a run happens* is a layout question and this layer
    only reads.

    ``launch`` is the attempt's ``run.json``. It is what separates *queued* from
    *never started*, which no amount of looking at an empty directory can do:
    § 1.6's *"a queued cluster job has produced nothing yet, so 'no output' and
    'not started' look identical"*, and its promise that status can then say
    *"queued as job 481923"* rather than guessing from an absence.
    """
    if not observed.is_dir():
        return ("not-started", "no directory yet (not prepped)")
    # ``out_glob`` is `Shape.stage_glob` -- "*" in the hierarchy, where the
    # directory has already selected the stage, and ``<label>_<NN>_<name>*`` in
    # flat, where ONE directory holds every stage and the filename is what
    # selects.  Without it a flat stage that has never run reports whatever its
    # sibling's `.out` says, because the glob matched the sibling's file.
    has_output = (any(observed.glob(out_glob + ".out"))
                  or any(observed.glob(out_glob + ".log")))
    if not has_output:
        if launch is None:
            return ("pending", "prepped, not launched (no run.json)")
        jid = launch.get("job_id")
        mode = launch.get("mode") or "?"
        return ("queued", (f"queued as job {jid}" if jid
                           else f"launched ({mode}), no output yet"))
    try:
        from ..parse.dirs.job import decode_run_dir
        res = decode_run_dir(observed)
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
    sh = shape_of(jobset, base_dir)
    dirs = job_dir_names(jobset, sh)
    refs = stage_refs(jobset)
    for job in jobset.jobs:
        d = base / dirs[job.name]
        # WHERE the run happened, asked of the layer that decides layout --
        # the latest attempt where there is one, the container for a flat run
        # (project-layout.md § 1.5).  Globbing `d` regardless was blind to the
        # whole attempt layer: a finished hierarchical stage read as "prepped,
        # not launched" because its .out is one level down.
        attempt = latest_attempt(d)
        observed = attempt or d
        launch = _launch_record(attempt)
        # WHICH FILES are this stage's, asked of the layout (§ 9's `Shape`).
        # In the hierarchy the directory already answered; in flat every stage
        # shares one, and the deck's token in each filename is the answer.
        token = refs[job.name].token
        out_glob = (sh.stage_glob(token, label)
                    if (sh is not None and token) else "*")
        state, detail = _stage_state(observed, launch, out_glob)
        stages.append(StageStatus(
            name=job.name, dir=d.name, state=state, detail=detail,
            seq=refs[job.name].seq,
            attempt=(attempt.name if attempt else None),
            attempts=attempts(d),
            launch=launch,
            warm_files=_warm_present(observed, label, jobset.engine),
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
    # The stage's SEQ, never its row.  This printed `enumerate()` until
    # 2026-08-10 -- a position where a reader reads an ordinal, which is the
    # number `engines/stages.md` R5 forbids as an identifier.  A sweep point
    # has no order, so it prints `-` from the one rule the plan table uses.
    # `attempt` is which run-<n> the row was READ FROM.  Without it the table
    # says "finished" without saying finished *when* -- and after a re-run the
    # difference between run-0 and run-2 is the whole question.  `-` for a flat
    # run, which happens in the container itself (project-layout.md § 1.5).
    hdr = ("seq", "stage", "attempt", "state", "warm files", "detail")
    rows = [(seq_text(s.seq), s.name, s.attempt or "-", s.state,
             ", ".join(s.warm_files) or "-", s.detail)
            for s in status.stages]
    # Widths and the rule are both driven off `hdr`, never off a literal count.
    # They were two hand-written numbers until 2026-08-10, and adding the
    # `attempt` column desynchronised them immediately: six headings over a
    # five-segment rule.
    w = [max(len(r[k]) for r in rows + [hdr]) for k in range(len(hdr))]
    def fmt(r):
        return "  ".join(s.ljust(w[k]) for k, s in enumerate(r))
    lines.append("  " + fmt(hdr))
    lines.append("  " + "  ".join("-" * n for n in w))
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


def render_stage_status(status: JobSetStatus, stage_name: str) -> str:
    """One stage, in full — the per-stage form `job-system.md` § 5.3 reserves.

    The table answers *where is this calculation up to*; this answers *what
    happened to this stage*, which is a different question and the one you ask
    before deciding whether to run it again. It is only answerable at all
    because of the attempt layer: the tries are directories, and the launch is
    a record rather than an inference from an empty folder (§ 1.5, § 1.6).

    Everything printed comes off the :class:`StageStatus` the reader already
    built. Nothing here opens a file — a second reader of ``run.json`` would be
    a second answer to *was this launched?*
    """
    s = next(x for x in status.stages if x.name == stage_name)
    rows: List[tuple] = []

    tries = ", ".join(f"run-{n}" for n in s.attempts) or "-"
    rows.append(("attempt", f"{s.attempt or '-'}"
                            + (f"   (of {tries})" if len(s.attempts) > 1
                               else "")))
    if s.launch:
        jid = s.launch.get("job_id")
        rows.append(("launched", f"{s.launch.get('mode', '?')}"
                                 + (f" as job {jid}" if jid else "")
                                 + f" at {s.launch.get('launched_at', '?')}"))
        cmd = s.launch.get("command")
        if cmd:
            rows.append(("command", " ".join(cmd)))
        # ABSENT means started from the structure (checkpointing.md S3), which
        # is a different claim from "continued from nothing" -- so it is only
        # printed when there is something to print.
        if s.launch.get("continued_from"):
            rows.append(("continued from", s.launch["continued_from"]))
    else:
        rows.append(("launched", "no  (no run.json -- prepared, not started)"))
    rows.append(("warm files", ", ".join(s.warm_files) or "-"))
    rows.append(("detail", s.detail or "-"))

    # The pad comes off the longest label, never a literal.  Hand-written and
    # it was 14 -- exactly the width of "continued from", so the one row that
    # had something to say ran its value straight into its own name.
    w = max(len(k) for k, _ in rows) + 2
    lines = [f"STAGE {StageRef(s.seq, s.name).label} -- {s.state}", ""]
    lines += [f"  {k.ljust(w)}{v}" for k, v in rows]
    lines.append("")
    lines.append(
        f"Directory: {s.dir}" + (f"/{s.attempt}" if s.attempt else ""))
    return "\n".join(lines)


__all__ = ["StageStatus", "JobSetStatus", "jobset_status", "render_status",
           "render_stage_status"]
