"""Placement — which queue a request goes in, and why not, when it cannot.

The contract is ``docs/execution/scheduler.md``; this module is § 5's decision
graph, in one walk:

    any queues at all?              no  -> None: run directly, no header
    did the user name one?          yes -> admit THAT one, or refuse
    which serve this KIND of work?  none -> refuse
    of those, which ADMIT it?       none -> refuse, with every reason
                                    some -> the cheapest ceiling that fits

Written as one function because it was two.  Until 2026-08-23 the CPU side
walked the menu for a row that fits while the GPU side looked only at the
first gpu-capable row -- so a grouped benchmark needing 38 minutes was routed
into ASU Sol's 15-minute ``debug`` queue, and when that did not fit, routing
returned "no preference" and let the rendered header's directives stand.  The
header named the same row.  Two branches, written separately, disagreeing.

**This module takes the menu; it does not fetch it.**  Reading configuration
belongs to a higher layer, and the package is stdlib-only so a record can be
read on the target inside a backend env with no molbuilder installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .admit import Request, admits, domain_serves_gpu


@dataclass(frozen=True)
class Placement:
    """A request bound to a domain — the ONE decision (R1).

    The header and the command line are two renderings of this, never two
    decisions, which is what stops them naming different queues.
    """
    domain: "object"
    partition: str
    qos: str

    @property
    def name(self) -> str:
        return getattr(self.domain, "name", "")


class Unplaceable(Exception):
    """No queue on this machine can take this request.

    Carries every reason rather than the first: a user who fixes the wall only
    to meet the core cap has been sent round twice (R4, R10).
    """

    def __init__(self, reasons: Sequence[str], *, gpu_side: bool):
        self.reasons = list(reasons)
        self.gpu_side = gpu_side
        super().__init__("; ".join(self.reasons) or "no domain admits it")


def candidates(routing, *, prefer_gpu: bool) -> List:
    """The queues that serve this KIND of work — § 5's third branch.

    A GPU request needs devices.  A CPU request PREFERS a cpu-only queue even
    on a gpu-capable cluster, because idle devices cost -- but the preference
    is expressed by ordering, not by exclusion: a cluster whose every queue
    has GPUs must still be able to run CPU work.
    """
    if prefer_gpu:
        return [d for d in routing if domain_serves_gpu(d)]
    cpu_only = [d for d in routing if not domain_serves_gpu(d)]
    return cpu_only or list(routing)



def _excess(row, request: Request) -> tuple:
    """How much bigger than the ask this queue's ceilings are.  Lower is
    tighter, and a tighter fit is the *cheaper* queue to get.

    **"The cheapest ceiling that fits" was never implemented on any axis but
    time.**  The menu is ordered by walltime alone, and `place` took the first
    row that admitted — so a 38-minute job needing 128 GB could land on a
    partition built for 2 TB work, wait behind it, and pay a scheduling
    penalty for memory nobody chose (`submission.md` § 3).

    The key is ``(unknown_ceilings, sum_of_ratios)``:

      * a dimension is compared only when **both** the ask states it and the
        row declares it — R3 again, an unstated ceiling is not a tight one;
      * rows whose ceilings we can actually measure sort **before** rows we
        would be guessing about.  A queue whose fit is known is a better
        choice than one that merely has not said no, and saying so in the sort
        key is what stops an unmeasured row winning by silence.

    It orders; it never admits.  A row that does not fit was already refused
    by :func:`admits` before this is consulted.
    """
    from .admit import domain_ceiling_s
    unknown = 0
    total = 0.0
    pairs = (
        (request.walltime_s, domain_ceiling_s(row)),
        (request.ranks, row.max_cores),
        (request.mem_gb, row.max_mem_gb),
    )
    for ask, ceiling in pairs:
        try:
            ask_f = float(ask) if ask else 0.0
            ceil_f = float(ceiling) if ceiling else 0.0
        except (TypeError, ValueError):
            ask_f = ceil_f = 0.0
        if ask_f > 0 and ceil_f > 0:
            total += ceil_f / ask_f
        elif ceil_f <= 0:
            unknown += 1
    return (unknown, total)


def place(routing, request: Request, *, prefer_gpu: bool,
          named: Optional[str] = None) -> Optional[Placement]:
    """Walk § 5's graph.  ``None`` means *this machine has no menu* — nothing
    was promised, so the rendered header's directives stand (R6).

    Raises :class:`Unplaceable` when there IS a menu and nothing on it can
    take the request.  That distinction is the whole of R6: refusing where we
    hold the record that says the scheduler will refuse, and proceeding where
    we hold no such record.
    """
    rows = list(routing or [])
    if not rows:
        return None

    if named:
        for d in rows:
            if d.name == named:
                why = admits(d, request)
                if why:
                    raise Unplaceable(why, gpu_side=prefer_gpu)
                return _bind(d, prefer_gpu)
        raise Unplaceable(
            [f"no domain named {named!r}; this machine offers "
             f"{', '.join(d.name for d in rows)}"], gpu_side=prefer_gpu)

    pool = candidates(rows, prefer_gpu=prefer_gpu)
    if not pool:
        raise Unplaceable(
            ["this machine has no gpu-capable queue"
             if prefer_gpu else "this machine has no queue for cpu work"],
            gpu_side=prefer_gpu)

    reasons: List[str] = []
    fits = []
    for d in pool:
        why = admits(d, request)
        if not why:
            fits.append(d)
        else:
            reasons.extend(why)
    if fits:
        # THE CHEAPEST CEILING THAT FITS -- across every dimension the request
        # states, not merely the shortest wall.  This took the FIRST admitting
        # row until 2026-08-23, and the menu is ordered by walltime, so the
        # choice was "the shortest queue that says yes" rather than "the queue
        # this job actually needs".
        #
        # `min` is stable, so among equally tight rows the menu's own order
        # still decides -- the recommendation R7 speaks of survives as the
        # tie-break rather than being overruled.
        return _bind(min(fits, key=lambda d: _excess(d, request)), prefer_gpu)
    raise Unplaceable(reasons, gpu_side=prefer_gpu)


def _bind(domain, prefer_gpu: bool) -> Placement:
    """A domain plus the partition this KIND of work actually goes to.

    ``gpu_partition`` is where a GPU job goes when that differs from the
    domain's ordinary partition — a declared field since phase 3, and read
    here rather than by each caller.
    """
    part = (domain.gpu_partition or domain.partition) if prefer_gpu \
        else domain.partition
    return Placement(domain=domain, partition=part, qos=domain.qos)
