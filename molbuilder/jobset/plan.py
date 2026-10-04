"""Plan engine — render a human-readable table from a :class:`JobSet`
(docs/execution/job-system.md § 5.3: ``STAGE-PLAN.md``, and `status <stage>`).

Pure formatting: it knows nothing but the data model.  The one basis for
``STAGE-PLAN.md`` — per-job resources, visible before anything is
submitted.  There are no dependency or carry-forward columns, because
nothing chains stages: a carry is a COPY ``prep --from`` makes, not an
edge a plan could draw.  And there is no second plan file for a sweep — a
sweep's plan IS a ``STAGE-PLAN.md``, in its ``bench/`` container.
"""

from __future__ import annotations


#: The plan's filename.  A constant so the name has ONE home: spelled at
#: each call site it is free to drift, which is the same shape the run-file
#: grammar exists to end, one level up.  Not named on the label, and still
#: the catalogue's (`runfiles.WRITTEN`, `job-contracts.md` § 2.2, 2026-10-04):
#: every file molbuilder writes is a row there.
from ..runfiles import PLAN_FILE as FILENAME  # noqa: E402

from typing import List

from .materialize import stage_refs
from .model import JobSet


def resources_text(r) -> str:
    """A job's resources in one line -- ``n=32, c=2, gpu:1`` -- read by the
    written plan (`STAGE-PLAN.md`) and by `status <stage>`."""
    bits: List[str] = []
    if r.domain:
        bits.append(f"domain={r.domain}")
    # the per-job rank/core counts ARE the variation for a sweep -- show them.
    if r.mpi_np:
        bits.append(f"n={r.mpi_np}")
    if r.cpus_per_task:
        bits.append(f"c={r.cpus_per_task}")
    if r.gres:
        bits.append(r.gres)
    if r.time:
        bits.append(f"t={r.time}")
    if r.exclusive:
        bits.append("exclusive")
    if r.mem and not r.exclusive:
        bits.append(f"mem={r.mem}")
    return ", ".join(bits) if bits else "(none stated)"


def render_plan(jobset: JobSet) -> str:
    """Render the plan: one row per job — its seq, input deck, warm files
    and resources.  Reads only the JobSet -- no IO.

    Nothing here orders anything, so no column claims an order: a carry is
    a COPY ``prep --from`` makes, read from the attempt's marker, not a
    relation between rows."""
    js = jobset
    lines: List[str] = [
        f"JOB-SET PLAN -- {js.name} ({js.engine}, {js.kind})",
        f"Shared package (copied into every job dir): "
        f"{', '.join(js.shared) or '(none)'}",
        "",
    ]
    hdr = ("seq", "job", "input", "warm files", "resources")
    rows = []
    # The column is the stage's SEQ, never its row.  A row index is the
    # stage's POSITION, which `engines/stages.md` R5 forbids as an identifier
    # -- and printing it in the column a reader takes for the ordinal states
    # a falsehood: disable a stage and the two differ.  A job with no ordinal
    # prints `-` rather than falling back to the row, which would be the same
    # mistake wearing the column's name.
    refs = stage_refs(js)
    for j in js.jobs:
        # WHAT this job would take from a run it is continued from -- never
        # WHICH run.  A column naming another job asserts an edge, and there
        # are none; a list of file names is the strongest claim this plan can
        # make and still be true.
        warm = ", ".join(w.name for w in j.warm) or "-"
        rows.append((refs[j.name].seq_text, j.name, j.script, warm,
                     resources_text(j.resources)))
    # Off `hdr`, not off a literal count -- a hand-written column count stops
    # matching the header the moment a column is added.
    w = [max(len(r[k]) for r in rows + [hdr]) for k in range(len(hdr))]
    def fmt(r):
        return "  ".join(s.ljust(w[k]) for k, s in enumerate(r))
    lines.append("  " + fmt(hdr))
    lines.append("  " + "  ".join("-" * n for n in w))
    lines += ["  " + fmt(r) for r in rows]

    # How these are launched.  NEVER "submit in parallel", which is the one
    # thing that never happens: a scheduler is handed ONE job per invocation
    # (job-system.md § 5.3, user rule).  A plan that ends by recommending the
    # refused thing is worse than one that ends without advice.
    lines.append("")
    if js.kind == "ladder":
        # A staged ladder declares no edges (P7 unit 2) and that is the design,
        # not an omission -- so say what to do rather than leaving a reader to
        # infer that unrelated jobs may all be started at once.
        lines.append(f"Order: {len(js.jobs)} stage(s), run ONE AT A TIME -- "
                     "prep a stage, launch it, look at it, then the next "
                     "(project-layout.md § 1.6); each continues from the "
                     "stage before it by default (job-system.md § 5.4), and "
                     "`molbuilder jobset status` names the next command.")
    else:
        lines.append(f"Order: {len(js.jobs)} independent job(s) -- no ordering "
                     "between them.  Submit one at a time; a scheduler is "
                     "never handed several at once (job-system.md § 5.3).")
    return "\n".join(lines)


__all__ = ["render_plan"]
