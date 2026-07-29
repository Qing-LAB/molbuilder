"""Scheduler adapters — turn an abstract job into concrete scripts for
one scheduler family.

The second half of the benchmark workflow's pluggable seam
(docs/execution/job-system.md, § 4.5).  An adapter is the
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
from typing import Dict, List, Optional, Tuple

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
_FALLBACK_CS = (1, 2, 4)        # cores/rank when cores-per-socket is unknown


def _bracket_cs(cps: Optional[int], k: int) -> List[int]:
    """Default cores-per-rank set for a given K: the *bracket*
    ``{1, cores//K, 2*cores//K}`` -- starved / one-socket / cross-socket --
    so each K row probes minimal, the conventional full-socket footprint, AND
    a deliberately cross-socket one (job-execution.md § 8.12).  Deduped, >=1.
    Falls back to ``_FALLBACK_CS`` when cores/socket is unknown."""
    if not cps:
        return list(_FALLBACK_CS)
    return sorted({1, max(1, cps // k), max(1, (2 * cps) // k)})


def sweep_grid(gpn, cps, ks, cs_explicit):
    """The canonical ``(G, K, c)`` enumeration -- the SINGLE source of truth
    for the sweep grid, iterated by BOTH the bash emitter (``format_bench``)
    AND the JobSet producer (``bench.to_jobset.sweep_to_jobset``), so the two
    can never define the grid differently.  Order: G outer, then K, then c
    (the per-K bracket ``{1, cores//K, 2*cores//K}`` when ``cs_explicit`` is
    None).  Yields ``(g, k, c)`` tuples."""
    for g in range(1, gpn + 1):
        for k in ks:
            for c in (cs_explicit if cs_explicit else _bracket_cs(cps, k)):
                yield (g, k, c)

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
                     ks: Optional[List[int]] = None,
                     cs: Optional[List[int]] = None
                     ) -> Dict[str, str]:
        """Render the environment-tailored ``job-gpu-sweep.sh`` for this
        scheduler: the (G, K, c) grid as runnable lines.  Returns
        ``{filename: content}`` so it can grow to emit more files later.

        ``ks`` overrides the swept ranks-per-GPU values (default: cores-per-
        socket divisors).  ``cs`` overrides the swept cores-per-rank (=OMP
        threads/rank) values (default per K: the *bracket*
        ``{1, cores//K, 2*cores//K}`` -- starved / one-socket / cross-socket).

        c is an INDEPENDENT axis, deliberately NOT capped at one socket: on a
        system where the scheduler does not co-locate the GPU with its ranks'
        socket (e.g. Sol), the "optimal" full-socket footprint is wishful and
        the real question -- *does more CPU working with the GPU beat the
        cross-socket traffic?* -- is exactly what the sweep must MEASURE rather
        than pre-decide (job-execution.md § 8.12).  Per-rank GPU<->NUMA binding
        stays best-effort in the wrapper (adapts to what SLURM granted, logs
        cross-socket); it no longer constrains the generated conditions.

        Running the produced script does the right thing per scheduler by
        construction: under SLURM each point is an ``sbatch`` (queue in
        parallel); on a workstation each runs in-place (sequential).

        **Output isolation:** every (G, K, c) point runs in its own
        ``point-G<g>K<k>C<c>/`` subdirectory (shared artifacts symlinked in),
        so points never clobber the ``job-gpu`` basename and summarize maps
        each directory back to its (G, K, c) label."""
        topo = env.topology
        gpn = gpus_per_node or topo.gpus_per_node or 1
        cps = topo.cores_per_socket
        gtype = _check_gpu_type(topo.gpu_type)
        ks = [int(k) for k in ks] if ks else (
            self.sweep_K(topo) or list(_FALLBACK_KS))
        cs_explicit = [int(c) for c in cs] if cs else None

        head = [
            "#!/usr/bin/env bash",
            f"# job-gpu-sweep.sh -- generated for scheduler '{self.name}'",
            f"#   topology: gpus/node={gpn} cores/socket="
            f"{cps if cps else '?'} gpu_type={gtype}",
            "#   knobs: G=GPUs, K=MPI ranks/GPU (-n=K*G), c=cores(OMP)/rank "
            "(-c=c).  c is an independent axis, NOT capped at one socket.",
            "#   K values = " + ",".join(str(k) for k in ks)
            + (" (cores/socket unknown -> fallback set)" if not cps else ""),
            "#   c values = " + (",".join(str(c) for c in cs_explicit)
                                 if cs_explicit
                                 else "{1, cores//K, 2*cores//K} per K "
                                      "(starved / 1-socket / cross-socket)"),
            "#   each point runs in its own point-G<g>K<k>C<c>/ dir (isolated "
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
        # K = MPI ranks SHARING the GPU (via MPS).  c = cores (OMP threads)
        # per rank -- an INDEPENDENT axis, not capped at one socket (§ 8.12).
        # Total CPU for this point = K*c*G.  The grid enumeration is the
        # shared ``sweep_grid`` (single source of truth with the JobSet
        # producer); this loop only renders each point's bash.
        for (g, k, c) in sweep_grid(gpn, cps, ks, cs_explicit):
            d = f"point-G{g}K{k}C{c}"
            total = k * c
            notes = []
            if cps and k > cps:
                notes.append(f"K={k} > cores/socket={cps}: ranks "
                             f"oversubscribe cores")
            if cps and total > 2 * cps:
                notes.append(f"{total} cores > node ({2*cps}): SLURM "
                             f"may REJECT -- a 'did not fit' data point")
            elif cps and total > cps:
                notes.append(f"{total} cores > 1 socket ({cps}): "
                             f"CROSS-SOCKET -- measures traffic vs "
                             f"throughput")
            if g >= 2:
                notes.append("multi-GPU: no NCCL -- MEASURE; do NOT "
                             "add --gpu-bind")
            note = ("  (" + "; ".join(notes) + ")") if notes else ""
            # Per-point job name (squeue differentiation, § 4.4) +
            # the explicitly-selected domain's -p/-q via $MB_GPU_PQ
            # (set by run-bench; empty -> the .sbatch header default).
            launch = self.gpu_launch_line(
                g, k, c, gtype, job_name=f"job-gpu-G{g}K{k}C{c}",
                pq="${MB_GPU_PQ:-}")
            body.append(f"# G={g} K={k} c={c} (total {total} cores){note}")
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

    def gpu_launch_line(self, g, k, c, gpu_type, script_base="job-gpu",
                        *, job_name=None, pq=None):
        # Override the sbatch header's defaults; the launcher reads
        # SLURM_NTASKS/_CPUS_PER_TASK so -n/-c and mpirun agree (§ 7.3).
        # Omit -c when cores/socket is unknown (let the header default it).
        # ``pq`` injects the explicitly-selected routing domain's -p/-q
        # (a shell expansion like ``$MB_GPU_PQ``; § 4.3); ``job_name`` makes
        # squeue rows distinct per point (§ 4.4).  BARE command (no trailing
        # comment): format_bench wraps it in ``( cd <dir> && ... )`` where an
        # inline '#' would comment out ')'.
        cflag = "" if c is None else f"-c {c} "
        pqf = f"{pq} " if pq else ""
        jf = f"-J {job_name} " if job_name else ""
        return (f"sbatch {pqf}{jf}--gres=gpu:{gpu_type}:{g} -n {k * g} {cflag}"
                f"{script_base}.sbatch")

    def cpu_launch_line(self, np, script_base, *, job_name=None, pq=None):
        pqf = f"{pq} " if pq else ""
        jf = f"-J {job_name} " if job_name else ""
        return f"sbatch {pqf}{jf}-n {np} {script_base}.sbatch"


class WorkstationAdapter(SchedulerAdapter):
    """Single machine: no scheduler; points run sequentially via a direct
    launch.  Uses the shared ``sweep_K`` (cores-per-socket divisors); the
    multi-GPU width of the sweep is bounded by ``gpus_per_node`` in
    :meth:`format_bench`, so a 1-GPU box yields only G=1 points."""
    name = "workstation"

    def matches(self, env: Environment) -> bool:
        return env.scheduler == "workstation"

    def gpu_launch_line(self, g, k, c, gpu_type, script_base="job-gpu",
                        *, job_name=None, pq=None):
        # Direct launch: no scheduler -> ``pq``/``job_name`` (sbatch-only
        # concepts) are ignored.
        # No scheduler: pick the GPUs with CUDA_VISIBLE_DEVICES and set the
        # per-point ranks/threads.  Use MB_NP / OMP_NUM_THREADS -- the vars
        # the wrapper actually honors for the LAUNCH (`_mpi_np` reads
        # MB_NP/SLURM_NTASKS; `_omp_threads` reads OMP_NUM_THREADS).  NOT
        # MOLBUILDER_MPI_NP/MOLBUILDER_OMP_NUM_THREADS: those only set the
        # GPU auto-default, which the wrapper's baked explicit mpi_np
        # SHADOWS -- so the sweep would not vary K (bug found 2026-06-28).
        # Same MB_NP convention as cpu_launch_line.
        cvd = ",".join(str(i) for i in range(g))
        omp = "" if c is None else f"OMP_NUM_THREADS={c} "
        return (f"CUDA_VISIBLE_DEVICES={cvd} MB_NP={k * g} "
                f"{omp}./{script_base}.run.sh")

    def cpu_launch_line(self, np, script_base, *, job_name=None, pq=None):
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


def parse_walltime(s) -> int:
    """SLURM walltime string -> seconds.  Accepts the forms SLURM accepts:
    ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``
    (slurm-integration.md § 4.3).  Empty -> 0.  Raises ValueError on garbage
    so a malformed config max_time fails loudly, not silently as 0."""
    s = str(s).strip()
    if not s:
        return 0
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        days = int(d)
        parts = [int(x) for x in s.split(":")] if s else [0]
        while len(parts) < 3:
            parts.append(0)
        h, m, sec = parts[0], parts[1], parts[2]
    else:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 1:
            h, m, sec = 0, parts[0], 0          # bare = minutes (SLURM rule)
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]   # MM:SS
        else:
            h, m, sec = parts[0], parts[1], parts[2]
    return ((days * 24 + h) * 60 + m) * 60 + sec


def domain_fits(domain: Dict, job_secs: int,
                job_mem_gb: Optional[float]) -> bool:
    """Does this routing ``domain`` fit a job of ``job_secs`` walltime and
    ``job_mem_gb`` memory (slurm-integration.md § 4.3)?  A domain with a
    ``max_mem_gb`` cap does NOT fit a job whose memory is unknown
    (``None``) -- we can't prove it fits, so we don't claim it (no silent
    over-ask, § 12)."""
    if job_secs > parse_walltime(domain["max_time"]):
        return False
    cap = domain.get("max_mem_gb")
    if cap is not None:
        if job_mem_gb is None or job_mem_gb > cap:
            return False
    return True


def fitting_domains(routing: List[Dict], job_secs: int,
                    job_mem_gb: Optional[float]) -> List[Dict]:
    """The domains (in list order) that fit the job -- § 4.3."""
    return [d for d in routing if domain_fits(d, job_secs, job_mem_gb)]


def recommend_domain(routing: List[Dict], job_secs: int,
                     job_mem_gb: Optional[float]) -> Optional[str]:
    """The recommended domain NAME: the FIRST (cheapest, by list order)
    that fits -- § 4.3.  ``None`` when none fits (the caller surfaces a
    refuse-to-emit)."""
    fits = fitting_domains(routing, job_secs, job_mem_gb)
    return fits[0]["name"] if fits else None


def resolve_mode(env: Environment, mode: Optional[str] = None) -> str:
    """Resolve the launch MODE (job-execution.md § 8.13): an explicit
    ``execution.mode`` ("direct" | "submit") wins; otherwise derive from the
    DETECTED scheduler (slurm -> submit, else -> direct).  Always returns a
    concrete "direct" or "submit"."""
    if mode in ("direct", "submit"):
        return mode
    return "submit" if env.scheduler == "slurm" else "direct"


def resolve_launch_adapter(env: Environment, *, mode: Optional[str] = None,
                           submit_via: str = "slurm"
                           ) -> Tuple[SchedulerAdapter, str]:
    """Select the adapter that LAUNCHES benchmark points, honoring the
    run-vs-submit policy independently of the detected scheduler
    (job-execution.md § 8.13).  Returns ``(adapter, resolved_mode)``.

      * ``submit`` -> the ``submit_via`` adapter (picked BY NAME, bypassing
        ``matches`` so "submit from an interactive shell" works);
      * ``direct`` -> the workstation adapter (direct bash, sequential).

    Topology still comes from ``env`` (detection); only the launch mechanism
    is chosen here -- detection and launch are decoupled."""
    rmode = resolve_mode(env, mode)
    if rmode == "submit":
        for a in ADAPTERS:
            if a.name == submit_via:
                return a, rmode
        raise ValueError(
            f"execution.submit_via={submit_via!r} has no registered adapter; "
            f"available: {[a.name for a in ADAPTERS]}")
    return WorkstationAdapter(), rmode


__all__ = [
    "divisors", "SchedulerAdapter", "SlurmAdapter", "WorkstationAdapter",
    "ADAPTERS", "get_adapter", "resolve_mode", "resolve_launch_adapter",
    "parse_walltime", "domain_fits", "fitting_domains", "recommend_domain",
]
