"""Placement — the queue a job NAMES, bound, and its fit.

The contract is ``docs/execution/scheduler.md``; this module is § 5's decision
graph, in one walk:

    no queues on the record, and none named -> None: nothing was promised
    the job names one the record lists      -> bind it, with its fit (R6)
    the job names none on a record with queues, one it does not list, or one
    on a record with none                   -> refuse

**Nothing here chooses a queue** *(user, 2026-10-02: "explicit job config is
the only way allowed"; `architecture.md` § 5.2)*, and nothing here refuses a
request for what the record says of the queue it names: the fit is said, and
the scheduler decides (R6).

**This module takes the menu; it does not fetch it.**  Reading configuration
belongs to a higher layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .fit import Fit, Request, fit


@dataclass(frozen=True)
class Placement:
    """A request bound to the domain it names, and its fit — the ONE
    decision (R1), made at prep and rendered once, in the header."""
    domain: "object"
    partition: str
    qos: str
    fit: Fit

    @property
    def name(self) -> str:
        return getattr(self.domain, "name", "")


class Unplaceable(Exception):
    """The job names no queue on a machine that has them, names one its
    record does not list, or names one on a machine whose record lists none
    -- a statement missing or contradicting the record, said with the
    record's queues (R6)."""


def place(routing, request: Request, *,
          named: Optional[str]) -> Optional[Placement]:
    """Bind ``request`` to the queue the job NAMES, with its fit.

    ``None`` means *this machine has no menu and the job names no queue* --
    nothing was promised, so there is nothing to bind.  Raises
    :class:`Unplaceable` when the job names no queue on a machine that has
    them (a job names its own -- `architecture.md` § 5.2), when the named
    queue is not on the menu, or when a queue is named on a machine with no
    menu (W52)."""
    rows = list(routing or [])
    offered = ", ".join(d.name for d in rows)
    if not named:
        if not rows:
            return None
        raise Unplaceable(
            f"this job names no queue -- a job names its own "
            f"(allocation.domain, --domain); this machine offers {offered}")
    if not rows:
        raise Unplaceable(f"{named}: this machine's record lists no queues "
                          f"to choose from")
    for d in rows:
        if d.name == named:
            # The queue's own partition, GPU work included (`scheduler.md` § 4).
            return Placement(domain=d, partition=d.partition, qos=d.qos,
                             fit=fit(d, request))
    raise Unplaceable(f"{named}: this machine offers {offered}")
