"""Plan engine — render a human-readable table from a :class:`JobSet`
(docs/execution/job-system.md, § 8 D3).

Pure formatting: it knows nothing but the data model.  This is the unified
basis for ``STAGE-PLAN.md`` (and, once the bench migrates onto the
framework, ``BENCH-PLAN.md``) so the per-job resources + the dependency
graph + the carry-forward are visible before anything is submitted.
"""

from __future__ import annotations

from typing import List

from .materialize import stage_refs
from .model import JobSet


def _res_str(r) -> str:
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
    return ", ".join(bits) if bits else "(inherit defaults)"


def render_plan(jobset: JobSet) -> str:
    """Render the plan: one row per job (resources, dependency, carry) plus
    the dependency graph.  Reads only the JobSet -- no IO."""
    js = jobset
    lines: List[str] = [
        f"JOB-SET PLAN -- {js.name} ({js.engine}, {js.kind})",
        f"Shared package (symlinked into every job dir): "
        f"{', '.join(js.shared) or '(none)'}",
        "",
    ]
    hdr = ("seq", "job", "input", "warm files", "resources")
    rows = []
    # The column is the stage's SEQ, never its row.  It printed `enumerate()`
    # until 2026-08-10 -- the stage's POSITION, which `engines/stages.md` R5
    # forbids as an identifier, in the column a reader takes for the ordinal.
    # Disable a stage and the two differ.  A job with no ordinal prints `-`
    # rather than falling back to the row, which would be the same mistake
    # wearing the new column's name.
    refs = stage_refs(js)
    for j in js.jobs:
        # WHAT this job would take from a run it is continued from -- never
        # WHICH run.  The two columns here used to be `depends_on (dep_kind)`
        # and the carry patterns; both named another job, and both went with
        # the edges on 2026-08-10.
        warm = ", ".join(w.name for w in j.warm) or "-"
        rows.append((refs[j.name].seq_text, j.name, j.script, warm,
                     _res_str(j.resources)))
    # Off `hdr`, not off a literal count -- the same two hand-written numbers
    # in `runstatus` disagreed the moment a column was added.
    w = [max(len(r[k]) for r in rows + [hdr]) for k in range(len(hdr))]
    def fmt(r):
        return "  ".join(s.ljust(w[k]) for k, s in enumerate(r))
    lines.append("  " + fmt(hdr))
    lines.append("  " + "  ".join("-" * n for n in w))
    lines += ["  " + fmt(r) for r in rows]

    # How these are launched.  NOT "submit in parallel", which this said until
    # 2026-08-10 and which is now the one thing that never happens: a scheduler
    # is handed ONE job per invocation (job-system.md § 5.3, user rule).  A
    # plan that ends by recommending the refused thing is worse than one that
    # ends without advice.
    lines.append("")
    if js.kind == "ladder":
        # A staged ladder declares no edges (P7 unit 2) and that is the design,
        # not an omission -- so say what to do rather than leaving a reader to
        # infer that unrelated jobs may all be started at once.
        lines.append(f"Order: {len(js.jobs)} stage(s), run ONE AT A TIME -- "
                     "look at each before setting up the next "
                     "(project-layout.md § 1.6):")
        lines.append("       molbuilder jobset prep   run <stage> "
                     "--from <attempt>")
        lines.append("       molbuilder jobset submit run <stage> --mode ...")
    else:
        lines.append(f"Order: {len(js.jobs)} independent job(s) -- no ordering "
                     "between them.  Submit one at a time; a scheduler is "
                     "never handed several at once (job-system.md § 5.3).")
    return "\n".join(lines)


__all__ = ["render_plan"]
