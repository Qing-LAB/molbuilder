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

import json
import re
from typing import Dict, List, Optional

from .environment import Environment, Topology


def _int_or(v, default):
    """Coerce ``v`` to int; return ``default`` on anything non-integer.
    Used to sanitize knob values from a foreign bench-result before they
    are interpolated into generated shell."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

# Fallback rank-counts when cores-per-socket is unknown (so the sweep is
# still useful; the adapter notes the missing topology).
_FALLBACK_KS = (1, 2, 4, 8)

# A script basename is interpolated UNQUOTED into generated shell, so it
# must not carry shell metacharacters / spaces (production base is
# user-derived from an fdf name).
_SAFE_BASE = re.compile(r"^[A-Za-z0-9._-]+$")


def _check_base(base: str) -> str:
    if not isinstance(base, str) or not _SAFE_BASE.fullmatch(base):
        raise ValueError(
            f"unsafe script_base {base!r}: only letters, digits, '.', "
            f"'_', '-' are allowed (it is interpolated into shell).")
    return base


# GPU type goes into ``--gres=gpu:<type>:N`` -- keep it a bare token.
_SAFE_GPU_TYPE = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_gpu_type(gtype: Optional[str]) -> str:
    """Sanitize the detected GPU type for shell interpolation; fall back
    to the generic ``gpu`` if absent or odd."""
    if gtype and _SAFE_GPU_TYPE.fullmatch(gtype):
        return gtype
    return "gpu"


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
                     gpus_per_node: Optional[int] = None,
                     ks: Optional[List[int]] = None
                     ) -> Dict[str, str]:
        """Render the environment-tailored ``job-gpu-sweep.sh`` for this
        scheduler: the valid (G, K) grid as runnable lines (invalid ones
        as comments).  Returns ``{filename: content}`` so it can grow to
        emit more formatted files later.

        ``ks`` overrides the swept ranks-per-GPU values (default: the
        cores-per-socket divisors -- full-socket points).  Pass an explicit
        list (e.g. ``[8, 16]``) to probe specific K; a K that does not
        divide cores/socket is still emitted but flagged as leaving some
        cores idle.

        Running the produced script does the right thing per scheduler by
        construction: under SLURM each point is an ``sbatch`` (all points
        **queue in parallel**); on a workstation each runs in-place
        (points run **sequentially**).

        **Output isolation:** every (G, K) point runs in its own
        ``point-G<g>K<k>/`` subdirectory (the shared fdf / run.sh / sbatch
        / monitor / pseudopotentials are symlinked in), so points never
        clobber the shared ``job-gpu`` basename and summarize can map each
        directory back to its (G, K) label."""
        topo = env.topology
        gpn = gpus_per_node or topo.gpus_per_node or 1
        cps = topo.cores_per_socket
        gtype = _check_gpu_type(topo.gpu_type)
        ks = [int(k) for k in ks] if ks else (
            self.sweep_K(topo) or list(_FALLBACK_KS))

        head = [
            "#!/usr/bin/env bash",
            f"# job-gpu-sweep.sh -- generated for scheduler '{self.name}'",
            f"#   topology: gpus/node={gpn} cores/socket="
            f"{cps if cps else '?'} gpu_type={gtype}",
            "#   knobs: G=GPUs, K=MPI ranks/GPU (-n=K*G), "
            "c=cores/rank (=cores_per_socket/K, full-socket).",
            "#   K values = " + ",".join(str(k) for k in ks)
            + (" (cores/socket unknown -> fallback set)" if not cps else ""),
            "#   each point runs in its own point-G<g>K<k>/ dir (isolated "
            "outputs).",
        ]
        if self.name == "slurm":
            head.append("#   sbatch per point -> points QUEUE IN PARALLEL.")
        else:
            head.append("#   in-place per point -> points run SEQUENTIALLY "
                        "(one box).")
        head += [
            "set -u",
            "",
            "# Isolate a sweep point: its own dir with the shared artifacts",
            "# symlinked in, so outputs don't collide on the job-gpu base.",
            "_mb_point() {",
            '    d="$1"; mkdir -p "$d"',
            "    for f in job-gpu.fdf job-gpu.run.sh job-gpu.sbatch "
            "mb_monitor.py; do",
            '        [ -e "$f" ] && ln -sfn "../$f" "$d/$f"',
            "    done",
            "    for p in *.psml *.psf *.vps; do",
            '        [ -e "$p" ] && ln -sfn "../$p" "$d/$p"',
            "    done",
            "}",
        ]

        body: List[str] = []
        for g in range(1, gpn + 1):
            for k in ks:
                c = (cps // k) if cps else None
                invalid = (cps is not None and (k > cps or c is None or c < 1))
                if invalid:
                    body.append(f"# INVALID G={g} K={k}: K exceeds "
                                f"cores/socket={cps}")
                    continue
                d = f"point-G{g}K{k}"
                ctag = "" if c is None else f" c={c}"
                idle = (cps - k * c) if (cps and c is not None) else 0
                under = (f"  ({idle} idle cores: K does not divide "
                         f"cores/socket)" if idle > 0 else "")
                caveat = ("  (multi-GPU: no NCCL -- MEASURE; do NOT add "
                          "--gpu-bind)" if g >= 2 else "")
                launch = self.gpu_launch_line(g, k, c, gtype)
                body.append(f"# G={g} K={k}{ctag}{under}{caveat}")
                body.append(f"_mb_point {d}")
                body.append(f"( cd {d} && {launch} )")

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
        _check_base(script_base)
        engine = (choice or {}).get("engine")
        knobs = (choice or {}).get("knobs") or {}
        topo = env.topology
        notes: List[str] = []

        # EVERY knob that reaches the shell MUST be integer-coerced first
        # -- the values come straight from a (possibly hand-edited or
        # foreign) bench-result JSON.  A raw string would otherwise be
        # interpolated unquoted and could inject (audit: format_run
        # cores_per_rank).  Non-numeric -> 1 (counts) or None (optional c).
        if engine == "gpu":
            k = _int_or(knobs.get("ranks_per_gpu"), 1)
            g_req = _int_or(knobs.get("gpus"), 1)
            bench_c = _int_or(knobs.get("cores_per_rank"), None)
            g = min(g_req, topo.gpus_per_node) if topo.gpus_per_node else g_req
            if topo.gpus_per_node and g < g_req:
                notes.append(f"G clamped {g_req}->{g} (this machine has "
                             f"{topo.gpus_per_node} GPU(s))")
            if topo.cores_per_socket:
                c = max(1, topo.cores_per_socket // k)
                if bench_c not in (None, c):
                    notes.append(f"c re-resolved to {c} = cores/socket"
                                 f"({topo.cores_per_socket})//K({k}) "
                                 f"[bench had {bench_c}]")
            else:
                c = bench_c
                notes.append("cores/socket unknown -> kept bench c; verify")
            cmd = self.gpu_launch_line(g, k, c, _check_gpu_type(topo.gpu_type),
                                       script_base)
        elif engine == "cpu":
            np = _int_or(knobs.get("ranks"), 1)
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
            # json.dumps keeps the knobs on ONE escaped line: a newline /
            # metachar in a value can't break out of this '#' comment.
            f"#   from the benchmark winner: engine={engine} "
            f"knobs={json.dumps(knobs)}",
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
        # BARE command (no trailing comment): format_bench wraps it in
        # ``( cd <dir> && ... )`` where an inline '#' would comment out ')'.
        cflag = "" if c is None else f"-c {c} "
        return (f"sbatch --gres=gpu:{gpu_type}:{g} -n {k * g} {cflag}"
                f"{script_base}.sbatch")

    def cpu_launch_line(self, np, script_base):
        return f"sbatch -n {np} {script_base}.sbatch"


class WorkstationAdapter(SchedulerAdapter):
    """Single machine: no scheduler; points run sequentially via a direct
    launch.  Uses the shared ``sweep_K`` (cores-per-socket divisors); the
    multi-GPU width of the sweep is bounded by ``gpus_per_node`` in
    :meth:`format_bench`, so a 1-GPU box yields only G=1 points."""
    name = "workstation"

    def matches(self, env: Environment) -> bool:
        return env.scheduler == "workstation"

    def gpu_launch_line(self, g, k, c, gpu_type, script_base="job-gpu"):
        # No scheduler: pick the GPUs with CUDA_VISIBLE_DEVICES and drive
        # the launcher's GPU-mode overrides directly.  Runs in-place
        # (blocking) -> sequential sweep.  BARE command (see SlurmAdapter).
        cvd = ",".join(str(i) for i in range(g))
        omp = "" if c is None else f"MOLBUILDER_OMP_NUM_THREADS={c} "
        return (f"CUDA_VISIBLE_DEVICES={cvd} MOLBUILDER_MPI_NP={k * g} "
                f"{omp}./{script_base}.run.sh")

    def cpu_launch_line(self, np, script_base):
        return f"MB_NP={np} ./{script_base}.run.sh"


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
