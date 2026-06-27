"""Scheduler adapters — turn an abstract job into concrete scripts for
one scheduler family.

The second half of the benchmark workflow's pluggable seam
(docs/protocols/benchmark-workflow.md § 4.3, § 4.5).  An adapter is the
*only* thing that differs between a workstation and a supercomputer; the
engine, monitor, estimator, timing, and data formats are shared.  One
adapter serves BOTH the benchmark scripts and the production run (reuse).

Adding a scheduler (PBS/LSF/cloud) = implementing one adapter and
registering it in :data:`ADAPTERS`; nothing else changes (§ 4.7).

**Stdlib-only** (ships to the target with the rest of the prep layer).
This increment lands the stable interface, adapter selection, and
``sweep_K`` (the topology-derived rank counts).  ``format_bench`` /
``format_run`` are declared on the interface and left to the next
increment (they compose the existing render layer), so callers can depend
on the shape now.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .environment import Environment, Topology

# Fallback rank-counts when cores-per-socket is unknown (so the sweep is
# still useful; the adapter notes the missing topology).
_FALLBACK_KS = (1, 2, 4, 8)


def divisors(n: int) -> List[int]:
    """Sorted positive divisors of ``n`` (``[]`` for non-positive)."""
    if n is None or n < 1:
        return []
    ds = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
        i += 1
    return sorted(ds)


class SchedulerAdapter:
    """Interface (§ 4.3).  Subclasses set ``name`` and implement
    ``matches`` + the formatters; ``sweep_K`` has a shared default."""

    name: str = "base"

    def matches(self, env: Environment) -> bool:
        raise NotImplementedError

    def sweep_K(self, topo: Topology) -> List[int]:
        """The GPU ranks-per-GPU values to sweep.  Default: the divisors
        of cores-per-socket, so every point fully uses the socket
        (``K*c = cores``, ``c = cores // K``); a non-divisor K would leave
        cores idle (§ 8).  Empty when cores-per-socket is unknown -- the
        caller then falls back to a declared default."""
        return divisors(topo.cores_per_socket) if topo.cores_per_socket \
            else []

    # ----- formatting (§ 4.3) --------------------------------------- #

    def gpu_launch_line(self, g: int, k: int, c: Optional[int],
                        gpu_type: str, script_base: str = "job-gpu") -> str:
        """One runnable launch command for a GPU (G, K) point against
        ``<script_base>``.  The ONLY scheduler-specific line; the grid
        logic (bench) and the single-point logic (run) are shared.
        Subclasses implement it."""
        raise NotImplementedError

    def cpu_launch_line(self, np: int, script_base: str) -> str:
        """One runnable launch command for a CPU point (``np`` ranks)
        against ``<script_base>``.  Subclasses implement it."""
        raise NotImplementedError

    def format_bench(self, env: Environment, *,
                     gpus_per_node: Optional[int] = None
                     ) -> Dict[str, str]:
        """Render the environment-tailored ``job-gpu-sweep.sh`` for this
        scheduler: the valid (G, K) grid as runnable lines (invalid ones
        as comments).  Returns ``{filename: content}`` so it can grow to
        emit more formatted files later.

        Running the produced script does the right thing per scheduler by
        construction: under SLURM each line is an ``sbatch`` (all points
        **queue in parallel**); on a workstation each line launches
        in-place (points run **sequentially**)."""
        topo = env.topology
        gpn = gpus_per_node or topo.gpus_per_node or 1
        cps = topo.cores_per_socket
        gtype = topo.gpu_type or "gpu"
        ks = self.sweep_K(topo) or list(_FALLBACK_KS)

        head = [
            "#!/usr/bin/env bash",
            f"# job-gpu-sweep.sh -- generated for scheduler '{self.name}'",
            f"#   topology: gpus/node={gpn} cores/socket="
            f"{cps if cps else '?'} gpu_type={gtype}",
            "#   knobs: G=GPUs, K=MPI ranks/GPU (-n=K*G), "
            "c=cores/rank (=cores_per_socket/K, full-socket).",
            "#   K values = " + ",".join(str(k) for k in ks)
            + (" (cores/socket unknown -> fallback set)" if not cps else ""),
        ]
        if self.name == "slurm":
            head.append("#   each line is an sbatch -> points QUEUE IN "
                        "PARALLEL.")
        else:
            head.append("#   each line launches in-place -> points run "
                        "SEQUENTIALLY (one box).")
        head.append("set -u")

        body: List[str] = []
        for g in range(1, gpn + 1):
            for k in ks:
                c = (cps // k) if cps else None
                invalid = (cps is not None and (k > cps or c is None or c < 1))
                if invalid:
                    body.append(f"# INVALID G={g} K={k}: K exceeds "
                                f"cores/socket={cps}")
                    continue
                line = self.gpu_launch_line(g, k, c, gtype)
                if g >= 2:
                    line += ("   # multi-GPU: no NCCL -- MEASURE, don't "
                             "assume; do NOT add --gpu-bind")
                body.append(line)

        content = "\n".join(head) + "\n\n" + "\n".join(body) + "\n"
        return {"job-gpu-sweep.sh": content}

    def format_run(self, choice: Dict, env: Environment, *,
                   script_base: str = "job") -> Dict[str, str]:
        """Apply the portable benchmark ``choice`` to the production job,
        **re-resolving the machine-specific knobs from this Environment**
        (benchmark-workflow.md § 5.4): the *mechanism* transfers (engine,
        ranks-per-GPU K), but the concrete per-rank cores ``c`` and GPU
        count ``G`` are recomputed for the local topology, never copied
        from the machine the benchmark ran on.

        ``choice`` is the ``choice`` block of a ``bench-result`` document
        (§ 5.3): ``{"engine": "gpu"|"cpu", "knobs": {...}}``.  Returns
        ``{"run-production.sh": <script>}`` -- a tiny launcher that records
        the translation and runs/submits the production scripts (assumed
        already engine-correct, the caller's job).
        """
        engine = (choice or {}).get("engine")
        knobs = (choice or {}).get("knobs") or {}
        topo = env.topology
        notes: List[str] = []

        if engine == "gpu":
            k = int(knobs.get("ranks_per_gpu") or 1)
            g_req = int(knobs.get("gpus") or 1)
            g = min(g_req, topo.gpus_per_node) if topo.gpus_per_node else g_req
            if topo.gpus_per_node and g < g_req:
                notes.append(f"G clamped {g_req}->{g} (this machine has "
                             f"{topo.gpus_per_node} GPU(s))")
            if topo.cores_per_socket:
                c = max(1, topo.cores_per_socket // k)
                if knobs.get("cores_per_rank") not in (None, c):
                    notes.append(f"c re-resolved to {c} = cores/socket"
                                 f"({topo.cores_per_socket})//K({k}) "
                                 f"[bench had {knobs.get('cores_per_rank')}]")
            else:
                c = knobs.get("cores_per_rank")
                notes.append("cores/socket unknown -> kept bench c; verify")
            cmd = self.gpu_launch_line(g, k, c, topo.gpu_type or "gpu",
                                       script_base)
        elif engine == "cpu":
            np = int(knobs.get("ranks") or 1)
            total = ((topo.sockets or 1) * topo.cores_per_socket
                     if topo.cores_per_socket else None)
            if total and np > total:
                notes.append(f"np clamped {np}->{total} (this machine has "
                             f"{total} cores)")
                np = total
            cmd = self.cpu_launch_line(np, script_base)
        else:
            raise ValueError(
                f"choice.engine must be 'gpu' or 'cpu'; got {engine!r}")

        head = [
            "#!/usr/bin/env bash",
            f"# run-production.sh -- generated for scheduler '{self.name}'",
            f"#   from the benchmark winner: engine={engine} knobs={knobs}",
            "#   the MECHANISM transfers across machines; the concrete -n/"
            "-c/-G are re-resolved here for THIS machine (§ 5.4):",
        ]
        head += [f"#     - {n}" for n in (notes or ["(no re-resolution "
                                                    "needed)"])]
        head.append("set -u")
        content = "\n".join(head) + "\n\n" + cmd + "\n"
        return {"run-production.sh": content}


class SlurmAdapter(SchedulerAdapter):
    """Shared cluster: jobs are submitted to a queue (one job per bench
    point, run in parallel).  Uses the thin sbatch header ->
    ``bash .run.sh`` model (slurm-integration.md § 3)."""
    name = "slurm"

    def matches(self, env: Environment) -> bool:
        return env.scheduler == "slurm"

    def gpu_launch_line(self, g, k, c, gpu_type, script_base="job-gpu"):
        # Override the sbatch header's defaults; the launcher reads
        # SLURM_NTASKS/_CPUS_PER_TASK so -n/-c and mpirun agree (§ 7.3).
        # Omit -c when cores/socket is unknown (let the header default it).
        cflag = "" if c is None else f"-c {c} "
        ctag = "" if c is None else f" c={c}"
        return (f"sbatch --gres=gpu:{gpu_type}:{g} -n {k * g} {cflag}"
                f"{script_base}.sbatch  # G={g} K={k}{ctag}")

    def cpu_launch_line(self, np, script_base):
        return f"sbatch -n {np} {script_base}.sbatch  # CPU np={np}"


class WorkstationAdapter(SchedulerAdapter):
    """Single machine: no scheduler; points run sequentially via a direct
    launch.  ``sweep_K`` is additionally bounded by the box -- with only a
    few GPUs there is no point in a wide multi-GPU sweep, and K is still
    capped at cores-per-socket by the divisor rule."""
    name = "workstation"

    def matches(self, env: Environment) -> bool:
        return env.scheduler == "workstation"

    def gpu_launch_line(self, g, k, c, gpu_type, script_base="job-gpu"):
        # No scheduler: pick the GPUs with CUDA_VISIBLE_DEVICES and drive
        # the launcher's GPU-mode overrides directly.  Runs in-place
        # (blocking) -> sequential sweep.
        cvd = ",".join(str(i) for i in range(g))
        omp = "" if c is None else f"MOLBUILDER_OMP_NUM_THREADS={c} "
        return (f"CUDA_VISIBLE_DEVICES={cvd} MOLBUILDER_MPI_NP={k * g} "
                f"{omp}./{script_base}.run.sh  # G={g} K={k}"
                + (f" c={c}" if c is not None else ""))

    def cpu_launch_line(self, np, script_base):
        return f"MB_NP={np} ./{script_base}.run.sh  # CPU np={np}"


# Registry — resolution order.  A new scheduler appends one entry here.
ADAPTERS: List[SchedulerAdapter] = [SlurmAdapter(), WorkstationAdapter()]


def get_adapter(env: Environment) -> SchedulerAdapter:
    """First registered adapter whose ``matches(env)`` is true (§ 4.4)."""
    for a in ADAPTERS:
        if a.matches(env):
            return a
    raise ValueError(
        f"no scheduler adapter matches environment {env.scheduler!r}; "
        f"registered: {[a.name for a in ADAPTERS]}")


__all__ = [
    "divisors", "SchedulerAdapter", "SlurmAdapter", "WorkstationAdapter",
    "ADAPTERS", "get_adapter",
]
