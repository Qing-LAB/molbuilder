"""The benchmark GRID -- topology in, (G, K, c) points out.

``sweep_grid`` (the one G x K x C enumeration
`jobset/prep_inputs.bench_inputs` consumes), ``sweep_K`` (topology-derived rank counts).
Trials are rendered by `jobset prep bench` and launched by
`jobset launch` -- no script is formatted here, and nothing ships.
"""

from __future__ import annotations

from typing import List, Optional

from ..scheduler import Topology




def _bracket_cs(cps: Optional[int], k: int) -> List[int]:
    """Default cores-per-rank set for a given K: the *bracket*
    ``{1, cores//K, 2*cores//K}`` -- starved / one-socket / cross-socket --
    so each K row probes minimal, the conventional full-socket footprint, AND
    a deliberately cross-socket one.  Deduped, >=1.
    The machine's cores per socket are its record's; a grid is never
    proposed without them (`prep_inputs.bench_inputs` refuses first)."""
    return sorted({1, max(1, cps // k), max(1, (2 * cps) // k)})


def sweep_grid(gpn, cps, ks, cs_explicit):
    """The canonical ``(G, K, c)`` enumeration -- the SINGLE source of truth
    for the sweep grid, iterated by every consumer of the grid
    (today: `jobset prep bench`'s `prep_inputs.bench_inputs`), so no two consumers
    can define it differently.  Order: G outer, then K, then c
    (the per-K bracket ``{1, cores//K, 2*cores//K}`` when ``cs_explicit`` is
    None).  Yields ``(g, k, c)`` tuples."""
    for g in range(1, gpn + 1):
        for k in ks:
            for c in (cs_explicit if cs_explicit else _bracket_cs(cps, k)):
                yield (g, k, c)



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


def sweep_K(topo: Topology) -> List[int]:
    """The GPU ranks-per-GPU values to sweep.  The divisors of
    cores-per-socket, so every point fully uses the socket (``K*c =
    cores``, ``c = cores // K``); a non-divisor K would leave cores idle
    (§ 8).  Empty when cores-per-socket is unknown -- the caller then
    proposes no grid."""
    return divisors(topo.cores_per_socket) if topo.cores_per_socket \
        else []


__all__ = ["divisors", "sweep_grid", "sweep_K"]
