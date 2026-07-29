"""Plan engine — render a human-readable table from a :class:`JobSet`
(docs/execution/job-system.md, § 8 D3).

Pure formatting: it knows nothing but the data model.  This is the unified
basis for ``STAGE-PLAN.md`` (and, once the bench migrates onto the
framework, ``BENCH-PLAN.md``) so the per-job resources + the dependency
graph + the carry-forward are visible before anything is submitted.
"""

from __future__ import annotations

from typing import List

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
    hdr = ("#", "job", "input", "depends on", "carries", "resources")
    rows = []
    for i, j in enumerate(js.jobs):
        dep = (f"{j.depends_on} ({j.dep_kind})" if j.depends_on else "-")
        carries = ", ".join(c.pattern for c in j.carry) or "-"
        rows.append((str(i), j.name, j.script, dep, carries, _res_str(j.resources)))
    w = [max(len(r[k]) for r in rows + [hdr]) for k in range(6)]
    def fmt(r):
        return "  ".join(s.ljust(w[k]) for k, s in enumerate(r))
    lines.append("  " + fmt(hdr))
    lines.append("  " + "  ".join("-" * w[k] for k in range(6)))
    lines += ["  " + fmt(r) for r in rows]

    # dependency graph (the chain, or "independent" for a sweep).
    lines.append("")
    if any(j.depends_on for j in js.jobs):
        chain = " -> ".join(j.name for j in js.jobs)
        lines.append(f"Order: {chain}  (each waits for the prior per its "
                     "dependency kind)")
    else:
        lines.append(f"Order: {len(js.jobs)} independent job(s) (submit in "
                     "parallel)")
    return "\n".join(lines)


__all__ = ["render_plan"]
