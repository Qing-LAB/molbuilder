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
from typing import Optional, Sequence

from .admit import Request, admits


@dataclass(frozen=True)
class Placement:
    """A request bound to a domain — the ONE decision (R1), made at prep and
    rendered once, in the header.
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

    def __init__(self, reasons: "Sequence[Refusal]"):
        # FINDINGS, not prose.  Each carries `where` -- `admit.cores`,
        # `admit.gpus` -- so a caller can say WHICH limit bit without
        # parsing the sentence.
        self.reasons = list(reasons)
        super().__init__("; ".join(i.message for i in self.reasons)
                         or "no domain admits it")


def place(routing, request: Request, *,
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
                          f"offers {', '.join(d.name for d in rows)}")])
    if not rows:
        raise Unplaceable(
            [Refusal("no_domain", named,
                     note="this machine's record lists no queues to "
                          "choose from")])
    for d in rows:
        if d.name == named:
            why = admits(d, request)
            if why:
                raise Unplaceable(why)
            # The queue's own partition, GPU work included (`scheduler.md` § 4).
            return Placement(domain=d, partition=d.partition, qos=d.qos)
    # A finding like every other refusal (`admit.Refusal`), so
    # `Unplaceable` carries ONE shape rather than two.
    raise Unplaceable(
        [Refusal("no_domain", named,
                 note=f"this machine offers "
                      f"{', '.join(d.name for d in rows)}")])

