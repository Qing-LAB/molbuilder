"""Placement — the queue a job NAMES, bound and admitted, or why not.

The contract is ``docs/execution/scheduler.md``; this module is § 5's decision
graph, in one walk:

    any queues at all, and none named?  -> None: nothing was promised
    the job names one                   -> admit THAT one, or refuse
    the job names none on a menu        -> refuse: a job names its own queue

**Nothing here chooses a queue** *(user, 2026-10-02: "explicit job config is
the only way allowed"; `architecture.md` § 5.2)*.

**This module takes the menu; it does not fetch it.**  Reading configuration
belongs to a higher layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from .admit import Refusal
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

    def __init__(self, reasons: "Sequence[Refusal]", *, gpu_side: bool):
        # FINDINGS, not prose.  Each carries `where` -- `admit.cores`,
        # `admit.gpus` -- so a caller can say WHICH limit bit without
        # parsing the sentence.
        self.reasons = list(reasons)
        self.gpu_side = gpu_side
        super().__init__("; ".join(i.message for i in self.reasons)
                         or "no domain admits it")

    @property
    def limits(self) -> "tuple[str, ...]":
        """Which limits refused it, e.g. ``("admit.cores", "admit.mem")``."""
        return tuple(i.where for i in self.reasons)


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
          named: Optional[str]) -> Optional[Placement]:
    """Bind ``request`` to the queue the job NAMES, admitted on its row.

    ``None`` means *this machine has no menu and the job names no queue* --
    nothing was promised, so there is nothing to bind (R6).  Raises
    :class:`Unplaceable` when the job names no queue on a machine that has
    them (a job names its own -- `architecture.md` § 5.2), when the named
    queue is not on the menu, or when it cannot take the request: we hold the
    record that says the scheduler would refuse, so we say so first.  **A
    queue named on a machine with no menu is refused too**: the name is a
    request this machine cannot honour (W52).
    """
    rows = list(routing or [])
    if not named:
        if not rows:
            return None
        raise Unplaceable(
            [Refusal("no_domain", "this job",
                     note="it names no queue -- a job names its own "
                          "(allocation.domain, --domain); this machine "
                          f"offers {', '.join(d.name for d in rows)}")],
            gpu_side=prefer_gpu)
    if not rows:
        raise Unplaceable(
            [Refusal("no_domain", named,
                     note="this machine's record lists no queues to "
                          "choose from")],
            gpu_side=prefer_gpu)
    for d in rows:
        if d.name == named:
            why = admits(d, request)
            if why:
                raise Unplaceable(why, gpu_side=prefer_gpu)
            # The queue's own partition, GPU work included (`scheduler.md` § 4).
            return Placement(domain=d, partition=d.partition, qos=d.qos)
    # A finding like every other refusal (`admit.Refusal`), so
    # `Unplaceable` carries ONE shape rather than two.
    raise Unplaceable(
        [Refusal("no_domain", named,
                 note=f"this machine offers "
                      f"{', '.join(d.name for d in rows)}")],
        gpu_side=prefer_gpu)

