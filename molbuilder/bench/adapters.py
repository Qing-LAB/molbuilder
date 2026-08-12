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

from ..environment import Environment, Topology


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
    for the sweep grid, iterated by every consumer of the grid
    (today: `jobset prep bench`'s `_bench_inputs`), so no two consumers
    can define it differently.  Order: G outer, then K, then c
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

# --------------------------------------------------------------------- #
#  The bash-sweep EMITTERS (`format_bench` / `format_run`, the per-       #
#  scheduler launch lines) and the launch-policy pair (`resolve_mode` /   #
#  `resolve_launch_adapter`) were DELETED 2026-08-12 (step 6 u5) with     #
#  the shipped-bundle lifecycle: trials are rendered by `jobset prep      #
#  bench` and launched ONE per invocation by `jobset submit`, whose mode  #
#  comes from --mode / execution.mode and is never derived (running-a-    #
#  job.md § 5.4).  The domain-fit helpers (`domain_fits` /                #
#  `fitting_domains` / `recommend_domain`) went too -- their one caller   #
#  was the deleted `bench prep`; a future submit-side recommendation      #
#  rebuilds them against `scheduler.routing` where it is read.            #
#  What remains is the GRID (the single source the fold kept) and the     #
#  topology/adapter halves `_bench_inputs` consumes.                      #
# --------------------------------------------------------------------- #

class SlurmAdapter(SchedulerAdapter):
    """Shared cluster: jobs are submitted to a queue (one job per bench
    point, run in parallel).  Uses the thin sbatch header ->
    ``bash .run.sh`` model (job-system.md § 6)."""
    name = "slurm"

    def matches(self, env: Environment) -> bool:
        return env.scheduler == "slurm"

class WorkstationAdapter(SchedulerAdapter):
    """Single machine: no scheduler; points run sequentially via a direct
    launch.  Uses the shared ``sweep_K`` (cores-per-socket divisors); the
    multi-GPU width of the sweep is bounded by ``gpus_per_node`` in
    :meth:`format_bench`, so a 1-GPU box yields only G=1 points."""
    name = "workstation"

    def matches(self, env: Environment) -> bool:
        return env.scheduler == "workstation"

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
    (running-a-job.md § 5.3).  Empty -> 0.  Raises ValueError on garbage
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



__all__ = [
    "divisors", "SchedulerAdapter", "SlurmAdapter",
    "WorkstationAdapter", "ADAPTERS", "get_adapter",
    "parse_walltime", "sweep_grid",
]
