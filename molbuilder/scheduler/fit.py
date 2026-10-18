"""The fit -- how a request sits on a queue's machines.

The contract is ``docs/execution/scheduler.md`` § 3 (R2-R4, R6, R7, R10, R13)
and § 4 (*Fit*).  It refuses nothing (R6): prep prepares what is asked, prints
the fit, and the scheduler decides.

One reading (:func:`fit`) and one rendering (:attr:`Fit.line`), asked by prep's
placement, the benchmark grid and the queue table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

from .record import UNSET


def domain_serves_gpu(row) -> bool:
    """Whether a queue's record (a `record.Domain`) lists GPUs -- the probe
    writes each partition's gres types onto its row (`scheduler.md`
    R2a)."""
    return bool(row.gpu)


def domain_ceiling_s(row) -> Optional[int]:
    """This domain's stated wall in seconds, or ``None`` when it states none
    -- the one place ``max_time`` is parsed for a caller that needs the
    NUMBER: the fit, the machine listing and the queue table."""
    from .quantities import parse_walltime
    if not row or not row.max_time:
        return None
    try:
        return parse_walltime(str(row.max_time))
    except ValueError:
        return None


#: A unit's name for ONE of it.
_ONE = {"cores": "core", "GPUs": "GPU", "ranks": "rank",
        "cores per rank": "core per rank"}


def _counted(n, unit: str) -> str:
    """``n`` with its unit, one of it named as one: ``1 GPU``, ``4 GPUs``."""
    if not unit:
        return str(n)
    return f"{n} {_ONE.get(unit, unit) if str(n) == '1' else unit}"


@dataclass(frozen=True)
class Request:
    """What one job asks of a queue, in the units a domain states.

    Fields the caller does not know are ``None``, and ``None`` is never
    counted (R7).  ``mem_gb`` is a NUMBER because the record's is one; build
    it with :func:`parse_mem_gb` from whatever SLURM text the caller holds.
    """
    ranks:      Optional[int] = None
    cpus_per_task: Optional[int] = None
    #: How many GPUs -- a count, and no card (`scheduler.md` R2a).
    gpus:       Optional[int] = None
    mem_gb:     Optional[float] = None
    walltime_s: Optional[int] = None

    @property
    def cores(self) -> Optional[int]:
        """Cores this ask occupies on a node -- ranks x cpus-per-task."""
        if self.ranks is None:
            return None
        return self.ranks * max(self.cpus_per_task or 1, 1)

    @property
    def said(self) -> str:
        """The request as a person reads it -- ``64 cores · 900 GB · 1 GPU
        · 4h``, what it states and nothing else."""
        from .quantities import human_wall
        bits = []
        if self.cores is not None:
            bits.append(_counted(self.cores, "cores"))
        if self.mem_gb is not None:
            bits.append(f"{self.mem_gb:g} GB")
        if self.gpus:
            bits.append(_counted(self.gpus, "GPUs"))
        if self.walltime_s is not None:
            bits.append(human_wall(self.walltime_s))
        return " · ".join(bits) or "this request"


@dataclass(frozen=True)
class Over:
    """One limit a request is over -- what it asks, what the queue's
    machines or rules allow, with the numbers (R4).

    The numbers are the answer; the sentence is a rendering of them.
    """
    limit:   str          # "walltime" | "cores" | "mem" | "gpus"
                          # | "cpus_per_job" | "cpus_per_node" | "machines"
                          # | a benchmark cell's: "ask_ranks" | "ask_cpus"
                          # | "ask_gpus" | "box_cores" | "box_gpus"
    domain:  str
    asked:   Any = None
    allowed: Any = None
    unit:    str = ""     # "cores", "GPUs", "GB" -- when asked/allowed are bare
    note:    str = ""     # only what the numbers cannot say

    @property
    def message(self) -> str:
        if self.asked is None and self.allowed is None:
            return self.note or self.domain
        what = {"walltime": "wall", "cores": "largest machine",
                "mem": "largest machine", "gpus": "most GPUs on a node",
                "cpus_per_job": "cap per job", "cpus_per_node": "cap per node",
                "ask_ranks": "--np", "ask_cpus": "--cpus-per-task",
                "ask_gpus": "--gpus", "box_cores": "cores",
                "box_gpus": "GPUs"}.get(self.limit, "limit")
        tail = f"; {self.note}" if self.note else ""
        return (f"{_counted(self.asked, self.unit)} is over {self.domain}'s "
                f"{what} ({_counted(self.allowed, self.unit)}){tail}")


@dataclass(frozen=True)
class Fit:
    """How a request sits on one domain's machines (`scheduler.md` § 4)."""
    domain:  str
    asked:   str
    #: Nodes that hold the whole request, and how many the queue has --
    #: ``None`` when its record lists no machines (it is read by its
    #: ``max_*`` then, and nodes cannot be counted).
    nodes:   Optional[int]
    of:      Optional[int]
    #: The machine sizes that hold it, as a person reads them -- one per
    #: size, as the queue table groups them (`scheduler.md` R0).
    holding: Tuple[str, ...]
    #: Each limit the request is over, with its numbers.
    over:    Tuple[Over, ...]

    @property
    def holds(self) -> bool:
        """Some node holds the whole request, and it is over no rule."""
        return not self.over and (self.nodes is None or self.nodes > 0)

    @property
    def line(self) -> str:
        """The one rendering: ``64 cores · 900 GB fits 3 of 212 nodes of htc
        (1%): 96 cores · 1007.4 GB; 128 cores · 1511.2 GB``, or what it is
        over."""
        if self.nodes is None:
            head = f"{self.asked} fits {self.domain}" if not self.over \
                else f"{self.asked} on {self.domain}"
        elif self.nodes == 0:
            head = f"{self.asked} fits no node of {self.domain}"
        elif self.nodes == self.of:
            head = (f"{self.asked} fits the one node of {self.domain}"
                    if self.of == 1 else
                    f"{self.asked} fits all {self.of} nodes of {self.domain}")
        else:
            # A SHARE NEVER ROUNDED TO NONE OR ALL: 1 node of 212 is not 0%,
            # 211 of 212 not 100%.
            pct = 100 * self.nodes / self.of if self.of else 0
            share = ("<1%" if pct < 1 else ">99%" if pct > 99
                     else f"{pct:.0f}%")
            head = (f"{self.asked} fits {self.nodes} of {self.of} nodes of "
                    f"{self.domain} ({share}): " + "; ".join(self.holding))
        return head + "".join(f" -- {o.message}" for o in self.over)


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _gpus_on(machine: Mapping[str, Any]) -> int:
    """The GPUs one node of this machine type carries, whatever the card
    (R2a): every card it lists, counted together, as the scheduler counts a
    GPU ask that names none -- the probe writes one machine type per
    `sinfo` row, its cards each node's; 0 when it lists none."""
    counts = [_int(c) for c in (machine.get("gpu") or {}).values()]
    return sum(c for c in counts if c is not None)


def _sizes(machines, *, cards: bool) -> Tuple[str, ...]:
    """Machines as a person reads them, ONE PER SIZE as the queue table
    groups them (`scheduler.md` R0): ``128 cores · 503.2-1511.2 GB``, the
    cards at that size after it when the request asks GPUs."""
    from .quantities import mem_range
    by_size: dict = {}
    for m in machines:
        g = by_size.setdefault(_int(m.get("cores")), {"mem": [], "gpu": set()})
        g["mem"].append(_float(m.get("mem_gb")))
        g["gpu"].update(m.get("gpu") or {})
    out = []
    for cores in sorted(by_size):
        g = by_size[cores]
        bits = [f"{cores} cores"] + [b for b in [mem_range(g["mem"])] if b]
        if cards:
            bits.extend(sorted(g["gpu"]))
        out.append(" · ".join(bits))
    return tuple(out)


def _lacks(machine: Mapping[str, Any], request: Request, *,
           gpus_said: bool) -> List[Tuple[str, float]]:
    """Everything one node of this machine type lacks of the request, and
    what it has of each -- ``[("cores" | "mem" | "gpus", its figure)]``;
    empty when it holds the whole of it.  A field the record does not state
    is not counted (R3); ``gpus_said`` is whether the record says which
    machines carry GPUs at all."""
    cores, mem = _int(machine.get("cores")), _float(machine.get("mem_gb"))
    out: List[Tuple[str, float]] = []
    if request.cores is not None and cores is not None \
            and cores < request.cores:
        out.append(("cores", cores))
    if request.mem_gb is not None and mem is not None \
            and mem < request.mem_gb:
        out.append(("mem", mem))
    if request.gpus and gpus_said and _gpus_on(machine) < request.gpus:
        out.append(("gpus", _gpus_on(machine)))
    return out


def fit(domain, request: Request) -> Fit:
    """How ``request`` sits on ``domain``'s machines -- the nodes that hold
    the whole of it, of how many, and each limit it is over (R2-R4, R10,
    R13).  Refuses nothing (R6)."""
    over: List[Over] = []
    name = domain.name
    # THE QUEUE'S OWN RULES -- over them on every machine.
    ceiling = domain_ceiling_s(domain)
    if request.walltime_s is not None and ceiling is not None \
            and ceiling < request.walltime_s:
        from .quantities import human_wall
        over.append(Over("walltime", name, asked=human_wall(request.walltime_s),
                         allowed=domain.max_time))
    if request.cores is not None:
        for field, limit in (("max_cpus_per_job", "cpus_per_job"),
                             ("max_cpus_per_node", "cpus_per_node")):
            cap = getattr(domain, field, None)
            cap = None if cap is UNSET else _int(cap)
            if cap is not None and cap < request.cores:
                over.append(Over(limit, name, asked=request.cores,
                                 allowed=cap, unit="cores"))
    # A QUEUE WHOSE RECORD LISTS NO GPUs has no node that holds a GPU
    # request (R2a).
    no_gpu = bool(request.gpus) and not domain_serves_gpu(domain)
    if no_gpu:
        over.append(Over("gpus", name, asked=request.gpus, allowed=0,
                         unit="GPUs", note="its record lists no GPUs"))

    machines = [m for m in (getattr(domain, "node_types", None) or [])
                if isinstance(m, dict) and _int(m.get("cores"))]
    if not machines:
        # A RECORD THAT LISTS NO MACHINES is read by its largest figures.
        for limit, asked, cap, unit in (
                ("cores", request.cores, _int(domain.max_cores), "cores"),
                ("mem", request.mem_gb, _float(domain.max_mem_gb), "GB")):
            if asked is not None and cap is not None and cap < asked:
                over.append(Over(limit, name, asked=f"{asked:g}"
                                 if isinstance(asked, float) else asked,
                                 allowed=f"{cap:g}" if isinstance(cap, float)
                                 else cap, unit=unit))
        if request.gpus and domain_serves_gpu(domain):
            most = max([d.per_node for d in domain.devices
                        if d.per_node is not None], default=None)
            if most is not None and most < request.gpus:
                over.append(Over("gpus", name, asked=request.gpus,
                                 allowed=most, unit="GPUs"))
        return Fit(name, request.said, None, None, (), tuple(over))

    gpus_said = any(m.get("gpu") for m in machines)
    total = sum(_int(m.get("nodes")) or 0 for m in machines)
    held = [] if no_gpu else [
        m for m in machines if not _lacks(m, request, gpus_said=gpus_said)]
    nodes = sum(_int(m.get("nodes")) or 0 for m in held)
    if not held:
        # EACH LIMIT OVER EVERY MACHINE, its cores and memory too where the
        # queue lists no GPU (R4); else what each kind of machine lacks.
        every = _over_every_machine(name, machines, request,
                                    gpus_said=gpus_said and not no_gpu)
        over.extend(every)
        if not every and not no_gpu:
            over.append(_what_each_lacks(name, machines, request,
                                         gpus_said=gpus_said))
    return Fit(name, request.said, nodes, total,
               _sizes(held, cards=bool(request.gpus)), tuple(over))


def _over_every_machine(name: str, machines, request: Request, *,
                        gpus_said: bool) -> List[Over]:
    """Each limit the request is over on every machine, with the largest
    there is (R4)."""
    over: List[Over] = []
    for limit, asked, of, unit in (
            ("cores", request.cores,
             lambda m: _int(m.get("cores")), "cores"),
            ("mem", request.mem_gb,
             lambda m: _float(m.get("mem_gb")), "GB"),
            ("gpus", request.gpus if gpus_said else None, _gpus_on, "GPUs")):
        if not asked:
            continue
        largest = max([v for v in (of(m) for m in machines) if v is not None],
                      default=None)
        if largest is not None and largest < asked:
            over.append(Over(limit, name, asked=f"{asked:g}",
                             allowed=f"{largest:g}", unit=unit))
    return over


def _what_each_lacks(name: str, machines, request: Request, *,
                     gpus_said: bool) -> Over:
    """No one machine holds the request though each part alone fits some:
    what each kind of machine lacks, once per kind, with the range those
    machines have (R10).  Machines are named by size, grouped as the queue
    table groups them (R0); a size whose machines do not all lack a thing is
    named by the ones that do, with their memory and cards, so every clause
    is true of every machine it names."""
    from .quantities import mem_range
    by_size: dict = {}
    for m in machines:
        by_size.setdefault(_int(m.get("cores")), []).append(m)
    lacking: dict = {}
    for m in machines:
        for kind, has in _lacks(m, request, gpus_said=gpus_said):
            lacking.setdefault(kind, []).append((m, has))

    def named(entries) -> str:
        whole, some = [], []
        for size in sorted({_int(m.get("cores")) for m, _ in entries}):
            these = [m for m, _ in entries if _int(m.get("cores")) == size]
            if len(these) == len(by_size[size]):
                whole.append(str(size))
            else:
                some += [f"{_size_named(m)}" for m in these]
        parts = ([f"the {'/'.join(whole)}-core machines"] if whole else []) \
            + [f"the {s} machines" for s in dict.fromkeys(some)]
        return " and ".join(parts)

    def what(kind, values) -> str:
        if kind == "mem":
            return mem_range(values)
        if kind == "gpus" and not any(values):
            return "no GPU"
        lo, hi = min(values), max(values)
        unit = "cores" if kind == "cores" else "GPUs"
        return _counted(lo, unit) if lo == hi else f"{lo}-{hi} {unit}"
    clauses = [f"{named(entries)} have {what(kind, [h for _, h in entries])}"
               for kind, entries in sorted(
                   lacking.items(),
                   key=lambda kv: min(_int(m.get("cores")) for m, _ in kv[1]))]
    return Over("machines", name,
                note=f"no one machine of {name} holds all of it: "
                     + "; ".join(clauses))


def _size_named(machine: Mapping[str, Any]) -> str:
    """One machine type by its size, memory and cards --
    ``128-core · 503.2 GB`` -- where its size alone names others too."""
    bits = [f"{_int(machine.get('cores'))}-core"]
    if _float(machine.get("mem_gb")):
        bits.append(f"{_float(machine.get('mem_gb')):g} GB")
    bits.extend(sorted(machine.get("gpu") or {}))
    return " · ".join(bits)


__all__ = ["Fit", "Over", "Request", "domain_ceiling_s", "domain_serves_gpu",
           "fit"]
