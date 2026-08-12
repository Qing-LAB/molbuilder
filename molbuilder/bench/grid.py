"""The benchmark GRID -- topology in, (G, K, c) points out.

What lives here after the 2026-08-12 fold (step 6 u5's tombstone below
is the authority; this header used to describe the pre-fold design --
script emitters, "ships to the target", "the next increment" -- none of
which survived): ``sweep_grid`` (the one G x K x C enumeration
`_bench_inputs` consumes), ``sweep_K`` (topology-derived rank counts),
Trials are rendered by `jobset prep bench` and launched by
`jobset submit` -- no script is formatted here, and nothing ships.
(The adapter classes this module was named for folded away 2026-08-12;
see the tombstone below.)
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


# --------------------------------------------------------------------- #
#  The scheduler-adapter CLASSES (SchedulerAdapter + Slurm/Workstation   #
#  + ADAPTERS + get_adapter) folded away 2026-08-12 (follow-up to the    #
#  U-program; this module renamed adapters.py -> grid.py with them).     #
#  Post-fold the subclasses defined ONLY `matches` -- name dispatch to   #
#  one shared sweep_K -- because everything scheduler-specific (script   #
#  formatting, launch policy) had already moved to jobset/runwrap.  The  #
#  K sweep is a fact about the TOPOLOGY, not the scheduler, so the       #
#  dispatch ceremony answered a question nobody was asking; and the old  #
#  unknown-scheduler refusal guarded script formatting that no longer    #
#  happens here.                                                         #
# --------------------------------------------------------------------- #

def sweep_K(topo: Topology) -> List[int]:
    """The GPU ranks-per-GPU values to sweep.  The divisors of
    cores-per-socket, so every point fully uses the socket (``K*c =
    cores``, ``c = cores // K``); a non-divisor K would leave cores idle
    (§ 8).  Empty when cores-per-socket is unknown -- the caller then
    falls back to a declared default (``_FALLBACK_KS``)."""
    return divisors(topo.cores_per_socket) if topo.cores_per_socket \
        else []


# ``parse_walltime`` moved DOWN to ``scheduler_probe.py`` (floor 1) with
# the redistribution: a SLURM time-string parser is machine-fact
# vocabulary, and its only consumer is the probe.  Living here it made
# floor 1 import upward -- the A-rules import checker is what caught it.


__all__ = [
    "divisors", "SchedulerAdapter", "SlurmAdapter",
    "WorkstationAdapter", "ADAPTERS", "get_adapter",
    "parse_walltime", "sweep_grid",
]
