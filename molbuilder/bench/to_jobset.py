"""Benchmark sweep -> :class:`JobSet` producer (job-system.md § 4.2).

The benchmark GPU sweep is, structurally, a ``sweep`` JobSet: N independent
jobs that share one package (the ``job-gpu.fdf`` + pseudos), each isolated in
its own ``point-G<g>K<k>C<c>/`` dir, each with its own scheduler resources
(``-n = K*G``, ``-c = c``, ``--gres = gpu:<type>:G``).  This is the SECOND
producer of the engine-agnostic ``jobset`` framework (the SIESTA stage ladder
is the first) -- it lets a bench bundle be described, planned, and run through
the same materialize/prep/submit engines, discharging the parallel-isolation
debt (§ 12).

The grid itself is NOT recomputed here: it comes from
``adapters.sweep_grid`` -- the SAME enumeration the bash emitter
(``format_bench``) iterates, so the JobSet and the legacy sweep script can
never disagree about which points exist.
"""

from __future__ import annotations

from typing import List, Optional

from ..jobset.model import Job, JobSet, Resources
from .adapters import (SchedulerAdapter, _FALLBACK_KS, _check_gpu_type,
                       sweep_grid)


def sweep_to_jobset(adapter: SchedulerAdapter, env, *,
                    ks: Optional[List[int]] = None,
                    cs: Optional[List[int]] = None,
                    script_base: str = "job-gpu",
                    shared: Optional[List[str]] = None) -> JobSet:
    """Build the GPU-sweep :class:`JobSet` for ``env``.

    ``adapter`` supplies ``sweep_K`` (the default ranks-per-GPU set), so the
    grid matches ``adapter.format_bench`` exactly.  ``ks`` / ``cs`` override
    the swept ranks-per-GPU / cores-per-rank (§ 8.12).  Each point becomes a
    ``Job`` named ``G<g>K<k>C<c>`` -> dir ``point-G<g>K<k>C<c>/`` (the
    convention ``summarize`` reads), input ``<script_base>.fdf`` (shared), with
    per-point ``Resources`` the submit engine applies as sbatch CLI flags.

    Note: this is the GPU sweep only.  The CPU baseline is a single root-level
    run (``job-cpu``), not a per-point job, and stays outside the JobSet."""
    topo = env.topology
    gpn = topo.gpus_per_node or 1
    cps = topo.cores_per_socket
    gtype = _check_gpu_type(topo.gpu_type)
    ks_ = [int(k) for k in ks] if ks else (adapter.sweep_K(topo)
                                           or list(_FALLBACK_KS))
    cs_ = [int(c) for c in cs] if cs else None

    jobs: List[Job] = []
    for (g, k, c) in sweep_grid(gpn, cps, ks_, cs_):
        jobs.append(Job(
            name=f"G{g}K{k}C{c}",
            script=f"{script_base}.fdf",
            resources=Resources(
                mpi_np=k * g,
                cpus_per_task=c,
                gres=f"gpu:{gtype}:{g}",
            ),
        ))
    return JobSet(name=script_base, engine="siesta", kind="sweep",
                  shared=list(shared or []), jobs=jobs)


__all__ = ["sweep_to_jobset"]
