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
    for d in pool:
        why = admits(d, request)
        if not why:
            return _bind(d, prefer_gpu)     # cheapest ceiling that fits
        reasons.extend(why)
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
