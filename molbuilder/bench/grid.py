"""The benchmark GRID -- topology in, (G, K, c) points out.

What lives here after the 2026-08-12 fold (step 6 u5's tombstone below
is the authority; this header used to describe the pre-fold design --
script emitters, "ships to the target", "the next increment" -- none of
which survived): ``sweep_grid`` (the one G x K x C enumeration
`_bench_inputs` consumes), ``sweep_K`` (topology-derived rank counts),
Trials are rendered by `jobset prep bench` and launched by
`jobset launch` -- no script is formatted here, and nothing ships.
(The adapter classes this module was named for folded away 2026-08-12;
see the tombstone below.)
"""

from __future__ import annotations

from typing import List, Optional

from ..scheduler import Topology


# Fallback rank-counts when cores-per-socket is unknown (so the sweep is
# still useful; the adapter notes the missing topology).
_FALLBACK_KS = (1, 2, 4, 8)
_FALLBACK_CS = (1, 2, 4)        # cores/rank when cores-per-socket is unknown


def _bracket_cs(cps: Optional[int], k: int) -> List[int]:
    """Default cores-per-rank set for a given K: the *bracket*
    ``{1, cores//K, 2*cores//K}`` -- starved / one-socket / cross-socket --
    so each K row probes minimal, the conventional full-socket footprint, AND
    a deliberately cross-socket one (archived job-execution.md § 8.12, the
    design record).  Deduped, >=1.
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

# TWO GUARD-SHAPED CONSTANTS STOOD HERE AND GUARDED NOTHING, deleted
# 2026-08-23 with two definitions and zero uses between them.  Said out loud
# because a bare deletion of something named ``_SAFE_*`` invites a future
# reader to put it back:
#
#   * the basename guard is REAL and lives in `runwrap._SAFE_WRAPPER_NAME_RE`,
#     enforced at three call sites.  This was a second copy of it that no
#     caller ever reached;
#   * the GPU-type guard protected against a token reaching a shell.  It
#     cannot: `sbatch` is invoked with an argv LIST and molbuilder uses
#     ``shell=True`` nowhere, so a crafted type is a malformed ``--gres``
#     value SLURM rejects, not an injection.  Re-add a check here only if
#     that stops being true.



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


# ``parse_walltime`` moved DOWN to ``scheduler/probe.py`` (floor 1) with
# the redistribution: a SLURM time-string parser is machine-fact
# vocabulary, and its only consumer is the probe.  Living here it made
# floor 1 import upward -- the A-rules import checker is what caught it.


# (R8, 2026-08-12: this list still exported six names the fold deleted
# or moved -- the adapter classes, get_adapter, parse_walltime -- and
# omitted the two live ones; `import *` raised AttributeError.)
__all__ = ["divisors", "sweep_grid", "sweep_K", "_FALLBACK_KS"]
