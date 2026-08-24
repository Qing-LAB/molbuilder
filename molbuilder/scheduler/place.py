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



#: The axes placement may be ordered by, and the order it uses when nobody
#: says otherwise.  **A closed set**: an unknown name is refused rather than
#: dropped, because a priority silently ignored is a preference that looks
#: honoured and is not.
PRIORITY_AXES = ("cores", "memory", "walltime")

#: **GPU is not in that list, and is not missing from it.**  Whether a run
#: wants a device is settled BEFORE any ordering: `candidates` splits the menu
#: by kind, so a GPU request only ever sees gpu-capable queues and CPU work
#: prefers cpu-only ones.  It is structurally first, which is stronger than
#: being first in a sort key -- a tie-break can be outweighed, a filter cannot.
PRIORITY_DEFAULT = ("cores", "memory", "walltime")


def _excess(row, request: Request, priority=None) -> tuple:
    """Which admitting queue to prefer.  Lower sorts first.

    **Cores are the requirement; memory is the chooser.**  A calculation's
    wall-clock depends most critically on how many cores it gets, so the core
    count is not something to trade away — and `admits` has already guaranteed
    it before this is consulted.  Memory is different: these jobs are core
    bound and do not press against memory ceilings, so **the queue offering
    LESS memory is the one that is easier to allocate**, and asking for a
    2 TB partition you do not need buys a longer wait and nothing else.
    *(User, 2026-08-23; `submission.md` § 3.)*

    So the key is lexicographic, not a sum, and **the order is the person's**
    (`PRIORITY_DEFAULT`, overridable per site):

        (unknown ceilings, *ratios in the declared priority order*)

    The default is cores, then memory, then walltime.  A site whose jobs press
    on memory rather than on cores says so and gets a different order, which
    is a preference and lives in the config -- never in the machine record
    (M-1).

    **A sum was wrong and measurably so.**  The first version added the three
    ratios equally, and on a Sol-shaped menu the walltime ratios span 6x-76x
    while memory spans 2x-16x — so the sum was a walltime sort in disguise and
    the memory axis could not decide anything.  Averaging three quantities
    that differ by an order of magnitude in spread is a way of choosing by the
    loudest one.

    Unknown ceilings lead, so a row whose fit we can measure is preferred to
    one that has merely not said no (R3) — an unmeasured queue must not win by
    silence.  Each ratio is ``ceiling / ask``, computed only where **both**
    sides are known; where either is missing the axis contributes ``0.0`` and
    cannot decide.
    """
    from .admit import domain_ceiling_s
    pairs = {
        "cores":    (request.ranks, row.max_cores),
        "memory":   (request.mem_gb, row.max_mem_gb),
        "walltime": (request.walltime_s, domain_ceiling_s(row)),
    }
    order = tuple(priority) if priority else PRIORITY_DEFAULT
    unknown = 0
    ratios = []
    for axis in order:
        ask, ceiling = pairs[axis]
        try:
            ask_f = float(ask) if ask else 0.0
            ceil_f = float(ceiling) if ceiling else 0.0
        except (TypeError, ValueError):
            ask_f = ceil_f = 0.0
        if ceil_f <= 0:
            unknown += 1
        ratios.append(ceil_f / ask_f if (ask_f > 0 and ceil_f > 0) else 0.0)
    return (unknown, *ratios)


def check_priority(order) -> tuple:
    """Validate a declared priority order, or raise.

    An unknown axis is REFUSED, not dropped: a preference that is silently
    ignored looks honoured and is not, which is the failure this whole
    document exists to remove.  A partial order is legal -- naming only
    ``["memory"]`` means *memory decides and the rest may fall where they
    fall* -- because that is a real thing to want and refusing it would make
    the person write out axes they do not care about.
    """
    order = tuple(order or ())
    bad = [a for a in order if a not in PRIORITY_AXES]
    if bad:
        raise ValueError(
            f"placement priority names {', '.join(map(repr, bad))}, which "
            f"{'is' if len(bad) == 1 else 'are'} not an axis placement can "
            f"order by.  Choose from {', '.join(PRIORITY_AXES)}.  (Whether a "
            f"run wants a GPU is settled before any ordering -- the menu is "
            f"split by kind first -- so it is not one of these.)")
    dupes = [a for a in order if order.count(a) > 1]
    if dupes:
        raise ValueError(
            f"placement priority repeats {sorted(set(dupes))!r}; each axis "
            f"decides once or the order after it can never be reached.")
    return order


def place(routing, request: Request, *, prefer_gpu: bool,
          named: Optional[str] = None,
          priority: Optional[Sequence[str]] = None) -> Optional[Placement]:
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
        return _bind(min(fits, key=lambda d: _excess(d, request, priority)),
                     prefer_gpu)
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
